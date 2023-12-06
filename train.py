import functools
import torch as th
import torch.distributed as dist
import dist_utils
import blobfile as bf
import utils as tools
import logger
import os

from torch.optim import AdamW
from resampler import LossAwareSampler, UniformSampler
from torch.nn.parallel.distributed import DistributedDataParallel as DDP


class TrainLoop:
    def init(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        save_interval,
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        """

        Args:
            model (_type_): _description_
            diffusion (_type_): _description_
            data (loader):
            batch_size (int): each rank's batch
            microbatch (int): if RAM is not enough, split batch_size to microbatch
            lr (float): learning rate
            save_interval (int): when to write in log
            schedule_sampler (_type_, optional): _description_. Defaults to None.
            weight_decay (float, optional): _description_. Defaults to 0.0.
            resume_checkpoint: (str): checkpoint path such as path/to/modelNNNNNN.pt
        """
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

        self._load_and_sync_parameters()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)

        if self.resume_step:
            self._load_optimizer_state()

        if th.cuda.is_available():
            self.use_ddp = True
            # broadcast_buffers: True is better for synchronize params in BN layer and so on. But there is False
            # bucket_cap_mb: limit each gradient size(MB)
            #
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_utils.dev()],
                output_device=dist_utils.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                self.use_ddp = False
                self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        """synchronize model parameters"""
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                self.model.load_state_dict(
                    dist_utils.load_sresume_steptate_dict(
                        resume_checkpoint, map_location=dist_utils.dev()
                    )
                )
        dist_utils.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            state_dict = dist_utils.load_state_dict(
                opt_checkpoint, map_location=dist_utils.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.optimize_normal()

    def forward_backward(self, batch, cond):
        tools.zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_utils.dev())
            micro_cond = {
                k: v[i : i + micro].to(dist_utils.dev()) for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_utils.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    # except last_batch
                    losses = compute_losses()
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_lossed(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            loss.backward()

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()

    def _log_grad_norm(self):
        """record sqrt sum of square grad"""
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad**2).sum().item()
        # logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def save(self):
        def save_checkpoint(params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(self.master_params)
        # save optim
        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # due to additional operations for rank0, other rank are blocked here.
        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())
