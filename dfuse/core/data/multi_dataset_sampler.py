"""Distributed-aware batch sampler for ConcatDataset with per-dataset ratios."""
from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class MultiDatasetBatchSampler(Sampler):
    """Per-rank batches mixing N concatenated sub-datasets at fixed ratios.

    Each yielded batch is a list[int] of length sum(batch_sizes).  Indices
    are offset for use against a torch.utils.data.ConcatDataset built from
    the same sub-datasets (in the same order).

    The sampler is itself distributed: each rank only emits its own slice
    of indices.  Do NOT pass the resulting DataLoader to
    accelerator.prepare() - that would re-shard via BatchSamplerShard.
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        batch_sizes: List[int],
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if len(dataset_sizes) != len(batch_sizes):
            raise ValueError(
                f"len(dataset_sizes)={len(dataset_sizes)} but "
                f"len(batch_sizes)={len(batch_sizes)}"
            )
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"Invalid num_replicas={num_replicas} or rank={rank}"
            )
        if any(b <= 0 for b in batch_sizes):
            raise ValueError(f"All batch_sizes must be > 0, got {batch_sizes}")

        self.dataset_sizes = list(dataset_sizes)
        self.batch_sizes = list(batch_sizes)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0


        self.cumulative_offsets = [0]
        for s in self.dataset_sizes[:-1]:
            self.cumulative_offsets.append(self.cumulative_offsets[-1] + s)


        self.per_rank_sizes = [s // num_replicas for s in self.dataset_sizes]


        for i, (per_rank, b) in enumerate(zip(self.per_rank_sizes, self.batch_sizes)):
            if per_rank < b:
                raise ValueError(
                    f"Sub-dataset {i}: per-rank size {per_rank} "
                    f"(= {self.dataset_sizes[i]} // {num_replicas}) "
                    f"is smaller than batch_size {b}; cannot produce a batch."
                )

        self.steps_per_epoch = max(
            per_rank // b
            for per_rank, b in zip(self.per_rank_sizes, self.batch_sizes)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _cycle_iter(self, dataset_idx: int) -> Iterator[List[int]]:
        """Endlessly yield batches of `batch_sizes[dataset_idx]` indices,
        offset to address the ConcatDataset.  Reshuffles each cycle."""
        size = self.dataset_sizes[dataset_idx]
        per_rank = self.per_rank_sizes[dataset_idx]
        batch_size = self.batch_sizes[dataset_idx]
        offset = self.cumulative_offsets[dataset_idx]
        cycle = 0
        while True:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(
                    self.seed
                    + self.epoch * 1_000_003
                    + dataset_idx * 1_009
                    + cycle
                )
                perm = torch.randperm(size, generator=g).tolist()
            else:
                perm = list(range(size))

            usable = per_rank * self.num_replicas
            perm = perm[:usable]
            rank_indices = perm[self.rank * per_rank : (self.rank + 1) * per_rank]

            n_batches = per_rank // batch_size
            for b in range(n_batches):
                start = b * batch_size
                yield [offset + idx for idx in rank_indices[start : start + batch_size]]

            cycle += 1

    def __iter__(self) -> Iterator[List[int]]:
        cycle_iters = [self._cycle_iter(i) for i in range(len(self.dataset_sizes))]
        for _ in range(self.steps_per_epoch):
            batch: List[int] = []
            for it in cycle_iters:
                batch.extend(next(it))
            yield batch

    def __len__(self) -> int:
        return self.steps_per_epoch
