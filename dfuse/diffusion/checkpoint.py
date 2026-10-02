"""D-FUSE training and sampling utilities."""

from __future__ import annotations

import os
from typing import Dict, Iterator, Tuple, Any

import torch
import torch.nn as nn


def get_local_data(param: nn.Parameter) -> torch.Tensor:
    """Return the local shard tensor for a parameter, regardless of backend.

    - DeepSpeed ZeRO-3: ``param.ds_tensor`` (always available, 1-D flat).
    - FSDP2 (DTensor): ``param._local_tensor``.
    - FSDP1 / DDP / plain: ``param.data`` (already the local flat shard or
      full tensor).
    """
    if hasattr(param, "ds_tensor"):
        return param.ds_tensor
    if hasattr(param, "_local_tensor"):
        return param._local_tensor
    return param.data


class CheckpointHandler:
    """Unified checkpoint I/O with streaming iteration and shard extraction."""

    def __init__(self, checkpoint_dir: str, world_size: int):
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size


    @staticmethod
    def get_partition_info() -> Tuple[int, int]:
        """Get data-parallel partition rank and world size for checkpoint sharding.

        Returns (partition_rank, world_size) or (0, 1) if distributed is not
        initialized.  Uses global DP rank — NOT the ZeRO++ HPZ subgroup rank —
        because both model parameters and optimizer state are primarily
        partitioned across all data-parallel ranks.
        """
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
            return 0, 1
        except Exception:
            return 0, 1

    @staticmethod
    def get_hpz_partition_size(accelerator) -> int:
        """Return the HPZ partition size for ZeRO++ checkpoint deduplication.

        With HPZ enabled, each subgroup of ``hpz_partition_size`` ranks holds
        a full copy of every parameter.  Only one subgroup's shards are unique,
        so we only need to save/load that many files.

        Returns ``world_size`` when HPZ is not active (standard ZeRO-3: every
        rank's shard is unique).
        """
        ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is None:
            return 1
        ds_config = getattr(ds_plugin, "deepspeed_config", None) or {}
        if not isinstance(ds_config, dict):
            return 1
        hpz = int(ds_config.get("zero_optimization", {}).get("zero_hpz_partition_size", 0))
        if hpz > 0:
            return hpz
        import torch.distributed as dist
        return dist.get_world_size() if dist.is_initialized() else 1

    @staticmethod
    def is_zero3_active(accelerator) -> bool:
        """Check if DeepSpeed ZeRO stage 3 is active."""
        ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        return ds_plugin is not None and getattr(ds_plugin, "zero_stage", 0) == 3

    @staticmethod
    def extract_rank_shard(
        full_tensor: torch.Tensor,
        param: nn.Parameter,
        partition_rank: int,
    ) -> torch.Tensor:
        """Extract rank shard."""
        if param is None or not hasattr(param, "ds_tensor"):
            return full_tensor.flatten().clone()

        shard_numel = param.ds_tensor.ds_numel
        full_flat = full_tensor.flatten()


        total = getattr(param, "ds_numel", full_flat.numel())
        full_param = full_flat[:total]


        n_partitions = -(-total // shard_numel)
        padded_numel = n_partitions * shard_numel
        if padded_numel > total:
            full_param = torch.cat([
                full_param,
                torch.zeros(padded_numel - total, dtype=full_param.dtype),
            ])


        position = partition_rank % n_partitions
        return full_param[position * shard_numel : (position + 1) * shard_numel]


    def detect_format(self) -> str:
        """Detect format."""
        d = self.checkpoint_dir
        if os.path.isfile(os.path.join(d, "optimizer_state.pt")) or\
           os.path.isdir(os.path.join(d, "optimizer_shards")):
            return "new"

        if os.path.isdir(d):
            for entry in os.listdir(d):
                full = os.path.join(d, entry)
                if os.path.isdir(full) and (entry.startswith("global_step") or entry.startswith("MODEL")):
                    return "legacy_zero3"

        if os.path.isfile(os.path.join(d, "pytorch_model.bin")) or\
           os.path.isfile(os.path.join(d, "model.safetensors")):
            return "legacy_ddp"

        return "unknown"

    def exists(self, name: str) -> bool:
        """Check if a checkpoint component exists (file or directory).

        Checks for: {name}.safetensors, {name}/ directory, {name}.pt
        """
        d = self.checkpoint_dir
        return (
            os.path.isfile(os.path.join(d, f"{name}.safetensors"))
            or os.path.isdir(os.path.join(d, name))
            or os.path.isfile(os.path.join(d, f"{name}.pt"))
        )


    def read_metadata(self, name: str) -> dict:
        """Read metadata from safetensors header or .pt file without loading tensors."""
        d = self.checkpoint_dir


        st_path = os.path.join(d, f"{name}.safetensors")
        if os.path.isfile(st_path):
            from safetensors import safe_open
            fh = safe_open(st_path, framework="pt", device="cpu")
            raw = fh.metadata() or {}
            return {k: self._parse_meta_val(v) for k, v in raw.items()}


        folder = os.path.join(d, name)
        rank0_st = os.path.join(folder, "rank_0.safetensors")
        if os.path.isdir(folder) and os.path.isfile(rank0_st):
            from safetensors import safe_open
            fh = safe_open(rank0_st, framework="pt", device="cpu")
            raw = fh.metadata() or {}
            return {k: self._parse_meta_val(v) for k, v in raw.items()}


        pt_path = os.path.join(d, f"{name}.pt")
        if os.path.isfile(pt_path):
            state = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
            meta = {}
            for k, v in state.items():
                if k not in ("shadows", "state", "param_groups"):
                    meta[k] = v
            return meta


        rank0_pt = os.path.join(folder, "rank_0.pt")
        if os.path.isdir(folder) and os.path.isfile(rank0_pt):
            state = torch.load(rank0_pt, map_location="cpu", mmap=True, weights_only=True)
            meta = {}
            for k, v in state.items():
                if k not in ("shadows", "state", "param_groups"):
                    meta[k] = v
            return meta

        return {}


    def iter_tensors(self, name: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iter tensors."""
        d = self.checkpoint_dir


        st_path = os.path.join(d, f"{name}.safetensors")
        if os.path.isfile(st_path):
            yield from self._iter_single_safetensors(st_path)
            return


        folder = os.path.join(d, name)
        if os.path.isdir(folder):
            rank0_st = os.path.join(folder, "rank_0.safetensors")
            rank0_pt = os.path.join(folder, "rank_0.pt")
            if os.path.isfile(rank0_st):
                yield from self._iter_sharded_safetensors(folder)
                return
            if os.path.isfile(rank0_pt):
                yield from self._iter_sharded_pt(folder)
                return


        pt_path = os.path.join(d, f"{name}.pt")
        if os.path.isfile(pt_path):
            yield from self._iter_single_pt(pt_path)
            return


        if name == "model":
            bin_path = os.path.join(d, "pytorch_model.bin")
            if os.path.isfile(bin_path):
                yield from self._iter_single_pt_raw(bin_path)
                return


        if name == "model":
            yield from self._iter_legacy_zero3()
            return

    def _iter_single_safetensors(self, path: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream tensors from a single safetensors file."""
        from safetensors import safe_open
        fh = safe_open(path, framework="pt", device="cpu")
        for key in fh.keys():
            yield key, fh.get_tensor(key)

    def _iter_sharded_safetensors(self, folder: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Assemble tensors from per-rank safetensors shards."""
        from safetensors import safe_open

        rank0_path = os.path.join(folder, "rank_0.safetensors")
        f0 = safe_open(rank0_path, framework="pt", device="cpu")
        meta = f0.metadata() or {}
        saved_subgroup_size = int(meta.get("subgroup_size", str(self.world_size)))

        handles = [(int(meta.get("partition_rank", "0")), f0)]
        for r in range(1, saved_subgroup_size):
            p = os.path.join(folder, f"rank_{r}.safetensors")
            if not os.path.isfile(p):
                raise RuntimeError(
                    f"Missing shard file {p}: expected {saved_subgroup_size} shards "
                    f"(from metadata subgroup_size) but rank_{r}.safetensors not found."
                )
            fh = safe_open(p, framework="pt", device="cpu")
            m = fh.metadata() or {}
            handles.append((int(m.get("partition_rank", str(r))), fh))
        handles.sort(key=lambda x: x[0])
        shard_handles = [h for _, h in handles]

        for key in shard_handles[0].keys():
            parts = [h.get_tensor(key) for h in shard_handles]
            yield key, self._assemble_tensor(parts)

    def _iter_sharded_pt(self, folder: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Assemble tensors from per-rank .pt shard files."""
        rank0_path = os.path.join(folder, "rank_0.pt")
        rank0_state = torch.load(rank0_path, map_location="cpu", mmap=True, weights_only=True)
        saved_subgroup_size = rank0_state.get("subgroup_size", self.world_size)

        states = [rank0_state]
        for r in range(1, saved_subgroup_size):
            p = os.path.join(folder, f"rank_{r}.pt")
            if not os.path.isfile(p):
                raise RuntimeError(
                    f"Missing shard file {p}: expected {saved_subgroup_size} shards "
                    f"but rank_{r}.pt not found."
                )
            states.append(torch.load(p, map_location="cpu", mmap=True, weights_only=True))
        states.sort(key=lambda s: s.get("partition_rank", 0))


        shard_dicts = [s.get("shadows", s) for s in states]
        for key in shard_dicts[0]:
            if isinstance(shard_dicts[0][key], torch.Tensor):
                parts = [sd[key] for sd in shard_dicts]
                yield key, self._assemble_tensor(parts)

    def _iter_single_pt(self, path: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iterate tensors from a single .pt file (shadows dict)."""
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        shadows = state.get("shadows", state)
        for key, val in shadows.items():
            if isinstance(val, torch.Tensor):
                yield key, val

    def _iter_single_pt_raw(self, path: str) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iter single pt raw."""
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                yield key, val

    def _iter_legacy_zero3(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Iter legacy zero3."""
        d = self.checkpoint_dir
        has_global_step = any(
            entry.startswith("global_step") or entry.startswith("MODEL")
            for entry in os.listdir(d)
            if os.path.isdir(os.path.join(d, entry))
        )
        if not has_global_step:
            return

        try:
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            state_dict = get_fp32_state_dict_from_zero_checkpoint(d)
            for key, val in state_dict.items():
                yield key, val
        except Exception as e:
            raise RuntimeError(
                f"[CheckpointHandler] Failed to extract legacy ZeRO-3 weights: {e}"
            ) from e

    @staticmethod
    def _assemble_tensor(parts: list) -> torch.Tensor:
        """Assemble a full tensor from shard parts."""
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=0)

    @staticmethod
    def _parse_meta_val(v: str):
        """Parse safetensors metadata string back to Python value."""
        try:
            if "." in v:
                return float(v)
            return int(v)
        except (ValueError, TypeError):
            return v


    def iter_optimizer_state(self, name: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iter optimizer state."""
        d = self.checkpoint_dir


        pt_path = os.path.join(d, f"{name}.pt")
        if os.path.isfile(pt_path):
            state = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
            for canon_name, param_state in state.get("state", {}).items():
                yield canon_name, param_state
            return


        folder = os.path.join(d, name)
        if os.path.isdir(folder):
            rank0_pt = os.path.join(folder, "rank_0.pt")
            if os.path.isfile(rank0_pt):
                yield from self._iter_optimizer_sharded(folder)
                return


        if name in ("optimizer_state", "optimizer_shards"):
            bin_path = os.path.join(d, "optimizer.bin")
            if os.path.isfile(bin_path):
                yield from self._iter_optimizer_legacy_bin(bin_path)
                return

    def _iter_optimizer_sharded(self, folder: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Assemble optimizer state from per-rank .pt shards."""
        rank0_path = os.path.join(folder, "rank_0.pt")
        rank0_state = torch.load(rank0_path, map_location="cpu", mmap=True, weights_only=True)
        saved_subgroup_size = rank0_state.get("subgroup_size", self.world_size)

        states = [rank0_state]
        for r in range(1, saved_subgroup_size):
            p = os.path.join(folder, f"rank_{r}.pt")
            if not os.path.isfile(p):
                raise RuntimeError(
                    f"Missing optimizer shard {p}: expected {saved_subgroup_size} "
                    f"shards but rank_{r}.pt not found."
                )
            states.append(torch.load(p, map_location="cpu", mmap=True, weights_only=True))
        states.sort(key=lambda s: s.get("partition_rank", 0))

        first_state = states[0].get("state", {})
        for canon_name in first_state:
            assembled = {}
            for k in first_state[canon_name]:
                parts = [s["state"][canon_name][k] for s in states]
                if isinstance(parts[0], torch.Tensor) and parts[0].dim() > 0:
                    assembled[k] = self._assemble_tensor(parts)
                else:
                    assembled[k] = parts[0]
            yield canon_name, assembled

    def _iter_optimizer_legacy_bin(self, path: str) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Iter optimizer legacy bin."""
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for param_idx, param_state in state.get("state", {}).items():
            yield str(param_idx), param_state


    def save_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        name: str,
        rank: int,
        is_sharded: bool,
        metadata: dict = None,
    ) -> None:
        """Save tensors as safetensors.

        DDP (is_sharded=False): rank 0 saves {name}.safetensors
        ZeRO-3 (is_sharded=True): each rank saves {name}/rank_{rank}.safetensors
        """
        from safetensors.torch import save_file

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        str_metadata = {k: str(v) for k, v in (metadata or {}).items()}

        if is_sharded:
            folder = os.path.join(self.checkpoint_dir, name)
            os.makedirs(folder, exist_ok=True)
            save_file(tensors, os.path.join(folder, f"rank_{rank}.safetensors"),
                      metadata=str_metadata)
        else:
            if rank == 0:
                save_file(tensors, os.path.join(self.checkpoint_dir, f"{name}.safetensors"),
                          metadata=str_metadata)

    def save_state(
        self,
        state: dict,
        name: str,
        rank: int,
        is_sharded: bool,
    ) -> None:
        """Save state dict as .pt file.

        DDP (is_sharded=False): rank 0 saves {name}.pt
        ZeRO-3 (is_sharded=True): each rank saves {name}/rank_{rank}.pt
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if is_sharded:
            folder = os.path.join(self.checkpoint_dir, name)
            os.makedirs(folder, exist_ok=True)
            torch.save(state, os.path.join(folder, f"rank_{rank}.pt"))
        else:
            if rank == 0:
                torch.save(state, os.path.join(self.checkpoint_dir, f"{name}.pt"))

    def save_single(self, state: dict, filename: str) -> None:
        """Save a single file unconditionally. Caller decides rank-gating."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save(state, os.path.join(self.checkpoint_dir, filename))
