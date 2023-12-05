from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


class ScheduleSampler(ABC):
    @abstractmethod
    def weights(self):
        """
        get a numpy array of weights
        the weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)  # get each timestep normalization weight
        indices_np = np.random.choice(
            len(p), size=(batch_size,), p=p
        )  # sample the timestep from the distribution of p
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_time_step])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_lossed(self, local_ts, local_losses):
        """_summary_

        Args:
            local_ts (_type_): _description_
            local_losses (_type_): _description_
        """
        batch_sizes = [
            th.tensor((0), dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        # gather the length of local_ts in each process and filled in batch_sizes
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)  # get the max length of batch from ranks

        # To avoid varying length, we choose the max length and others are padded zero.
        # but it don't affect to get timesteps.
        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        # timesteps and lossed are list(not nested list), which means it integrated timesteps and losses from different ranks in a list
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.items() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses()

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.
        #FIXME: how to keep it synchronizing

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_time_step, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_time_step], dtype=np.int)

    def weights(self):
        """
        #FIXME: still confuse why need add self.uniform_prob / len(weights)
        I can only think of one reason to explain this expression: 1. no matter (1-self.uniform_prob)
        or self.uniform_prob, both of it are coef. 2. divide len(weights) means uniformresample.
        3. it use mean square operation to calcuate weights, so it is called SecondMoment, which is referred to the
        optimization algorithms such as AdaGrad or Adam. In AdaGrad, it also "square" grad.
        """
        if not self._warmed_up():
            # if the number of history losses not raech history_per_term, use UniformResampler
            return np.ones([self.diffusion.num_time_step], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2), axis=-1)
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # pop out the oldest loss term
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """check if there are already history_per_term losses"""
        return (self._loss_counts == self.history_per_term).all()
