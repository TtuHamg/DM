import torch as th
import torch.distributed as dist
import dist_utils
from torch.optim import AdamW
from resampler import LossAwareSampler, UniformSampler


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
        resume_checkpoint,
        schedule_sampler=None,
        weight_decay=0.0,
    ):
        """

        Args:
            model (_type_): _description_
            diffusion (_type_): _description_
            data (_type_): _description_
            batch_size (_type_): _description_
            microbatch (_type_): _description_
            lr (_type_): _description_
            schedule_sampler (_type_, optional): _description_. Defaults to None.
            weight_decay (float, optional): _description_. Defaults to 0.0.
            resume_checkpoint: (str): checkpoint path such as path/to/modelNNNNNN.pt
        """
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch
        self.lr = lr
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                self.model.load_state_dict(
                    dist_utils.load_state_dict(
                        resume_checkpoint, map_location=dist_utils.dev()
                    )
                )


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
