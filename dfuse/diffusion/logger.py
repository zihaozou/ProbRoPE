import json
import os
import re
import random
import shutil

import numpy as np
import torch
from accelerate import Accelerator

from .ema import DistributedEMA
from .checkpoint import CheckpointHandler, get_local_data


class ModelLogger:
    def __init__(
        self,
        output_path,
        state_dict_converter=lambda x: x,
        keep_n_checkpoints=None,
        use_ema=False,
        ema_decay=0.9999,
    ):
        self.output_path = output_path
        self.state_dict_converter = state_dict_converter
        self.keep_n_checkpoints = keep_n_checkpoints
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema: DistributedEMA | None = None
        self.num_steps = 0
        self.saved_checkpoints_paths = []
        self._pending_latest_cleanup: str | None = None


    def _manage_checkpoints(self):
        if self.keep_n_checkpoints is not None:
            while len(self.saved_checkpoints_paths) > self.keep_n_checkpoints:
                path_to_remove = self.saved_checkpoints_paths.pop(0)
                if os.path.exists(path_to_remove):
                    shutil.rmtree(path_to_remove)

    def _parse_epoch_from_path(self, checkpoint_path: str) -> int:
        basename = os.path.basename(os.path.normpath(checkpoint_path))
        m = re.match(r"epoch-(\d+)", basename)
        return int(m.group(1)) if m else 0

    def _scan_existing_checkpoints(self):
        if not os.path.isdir(self.output_path):
            return
        dirs = sorted(
            [
                os.path.join(self.output_path, d)
                for d in os.listdir(self.output_path)
                if os.path.isdir(os.path.join(self.output_path, d))
                and re.match(r"epoch-\d+$", d)
            ],
            key=lambda p: self._parse_epoch_from_path(p),
        )
        self.saved_checkpoints_paths = dirs


    def init_ema(self, model: torch.nn.Module, accelerator: Accelerator):
        if not self.use_ema:
            return
        self.ema = DistributedEMA(
            model, decay=self.ema_decay, trainable_only=True,
            accelerator=accelerator,
        )
        accelerator.print(
            f"[EMA] Initialized with decay={self.ema_decay}, "
            f"tracking {len(self.ema.shadows)} parameter shards."
        )

    def apply_ema(self, model: torch.nn.Module) -> None:
        """Apply EMA shadows to model parameters.

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        if self.ema is not None:
            self.ema.apply_to_model(model)

    def restore_from_ema(self, model: torch.nn.Module) -> None:
        """Restore model parameters from EMA backup.

        Args:
            model: The **unwrapped** model (canonical parameter names).
        """
        if self.ema is not None:
            self.ema.restore_model(model)


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module):
        self.num_steps += 1
        if self.ema is not None:
            self.ema.update(accelerator.unwrap_model(model))

    def on_epoch_end(
        self, accelerator, model, epoch_id,
        save_epochs=None, optimizer=None, scheduler=None, metadata=None,
    ):
        if save_epochs is None or (epoch_id + 1) % save_epochs == 0:
            self.save_checkpoint(
                accelerator, model, f"epoch-{epoch_id}",
                optimizer=optimizer, scheduler=scheduler, metadata=metadata,
            )

    def on_training_end(
        self, accelerator, model,
        save_epochs=None, num_epochs=None,
        optimizer=None, scheduler=None, metadata=None,
    ):
        self.save_checkpoint(
            accelerator, model, f"epoch-{num_epochs - 1}",
            optimizer=optimizer, scheduler=scheduler, metadata=metadata,
        )


    @staticmethod
    def _build_canon_param_map(model, unwrapped):
        """Build {canonical_name: wrapped_param} mapping."""
        canonical_names = {id(p): n for n, p in unwrapped.named_parameters()}
        result = {}
        for wrapped_name, param in model.named_parameters():
            canon = canonical_names.get(id(param), wrapped_name)
            result[canon] = param
        return result


    def save_checkpoint(
        self, accelerator: Accelerator, model: torch.nn.Module, folder_name,
        optimizer=None, scheduler=None, metadata=None,
    ):
        accelerator.wait_for_everyone()
        checkpoint_dir = os.path.join(self.output_path, folder_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        handler = CheckpointHandler(checkpoint_dir, accelerator.num_processes)
        is_sharded = handler.is_zero3_active(accelerator)
        rank = accelerator.process_index


        self._save_model_weights(accelerator, model, handler)


        if optimizer is not None:
            self._save_optimizer(accelerator, model, optimizer, handler, rank, is_sharded)


        if scheduler is not None and rank == 0:
            handler.save_single(scheduler.state_dict(), "scheduler.pt")


        self._save_rng(handler, rank)


        self._save_ema_shadows(handler, rank, is_sharded)


        if self.ema is not None:
            self._save_ema_model_export(accelerator, model, handler)


        if metadata is not None and rank == 0:
            meta_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and folder_name != "latest":
            self.saved_checkpoints_paths.append(checkpoint_dir)
            self._manage_checkpoints()

            if self._pending_latest_cleanup is not None:
                pending = self._pending_latest_cleanup
                if os.path.exists(pending):
                    shutil.rmtree(pending)
                    print(f"[Resume] Deleted preemption checkpoint: {pending}")
                self._pending_latest_cleanup = None

    def _save_model_weights(self, accelerator, model, handler):
        """Save model weights.

        DDP: gather on rank 0 and save a single model.safetensors.
        ZeRO-3: each rank saves its local shard as model/rank_N.safetensors.
        """
        is_sharded = handler.is_zero3_active(accelerator)
        rank = accelerator.process_index

        if is_sharded:
            unwrapped = accelerator.unwrap_model(model)
            trainable_names = unwrapped.trainable_param_names()
            state_dict = {
                name: get_local_data(param).cpu().contiguous()
                for name, param in unwrapped.named_parameters()
                if name in trainable_names
            }
            state_dict = unwrapped.export_trainable_state_dict(state_dict)
            state_dict = self.state_dict_converter(state_dict)
            pr, sg = CheckpointHandler.get_partition_info()
            metadata = {"partition_rank": str(pr), "subgroup_size": str(sg)}
            handler.save_tensors(state_dict, "model", rank, is_sharded=True, metadata=metadata)
        else:
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(
                    state_dict
                )
                state_dict = self.state_dict_converter(state_dict)
                handler.save_tensors(
                    state_dict, "model", rank=0, is_sharded=False,
                )

    def _save_optimizer(self, accelerator, model, optimizer, handler, rank, is_sharded):
        """Save optimizer state with canonical parameter names."""
        unwrapped = accelerator.unwrap_model(model)
        param_name_map = {
            id(p): n for n, p in unwrapped.named_parameters()
        }

        for wrapped_name, param in model.named_parameters():
            if id(param) not in param_name_map:
                param_name_map[id(param)] = wrapped_name

        if is_sharded:
            self._save_optimizer_zero3(model, optimizer, handler, rank, param_name_map, accelerator)
        else:
            self._save_optimizer_ddp(optimizer, handler, rank, param_name_map)

    def _save_optimizer_ddp(self, optimizer, handler, rank, param_name_map):
        """Save DDP optimizer state with canonical param names on rank 0."""
        raw_state = optimizer.state_dict()

        param_id_to_name = {}
        idx = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                param_id_to_name[idx] = param_name_map.get(id(param), str(idx))
                idx += 1

        canonical_state = {}
        for param_idx, state_vals in raw_state["state"].items():
            canon_name = param_id_to_name.get(param_idx, str(param_idx))
            canonical_state[canon_name] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in state_vals.items()
            }

        canonical_groups = []
        idx = 0
        for group in raw_state["param_groups"]:
            group_copy = {k: v for k, v in group.items() if k != "params"}
            group_copy["params"] = []
            for _ in group["params"]:
                group_copy["params"].append(param_id_to_name.get(idx, str(idx)))
                idx += 1
            canonical_groups.append(group_copy)

        payload = {"state": canonical_state, "param_groups": canonical_groups}
        handler.save_state(payload, "optimizer_state", rank, is_sharded=False)

    def _save_optimizer_zero3(self, model, optimizer, handler, rank, param_name_map, accelerator):
        """Save ZeRO-3 optimizer state per-rank with canonical names."""
        partition_rank, subgroup_size = CheckpointHandler.get_partition_info()
        ds_optimizer = model.optimizer
        base_optimizer = ds_optimizer.optimizer

        canonical_state = {}
        canonical_groups = []
        skipped_params = []

        for group_id, fp16_params in enumerate(ds_optimizer.fp16_groups):
            group_param_names = []
            fp32_flat = ds_optimizer.fp32_partitioned_groups_flat[group_id]
            flat_state = base_optimizer.state.get(fp32_flat, {})

            for fp16_param in fp16_params:
                canon_name = param_name_map.get(id(fp16_param), f"unknown_{id(fp16_param)}")
                group_param_names.append(canon_name)

                param_id = ds_optimizer.get_param_id(fp16_param)
                if param_id not in ds_optimizer.grad_position:
                    skipped_params.append(canon_name)
                    continue
                grp_idx, offset, numel = ds_optimizer.grad_position[param_id]
                if hasattr(fp16_param, "ds_tensor"):
                    assert numel == fp16_param.ds_tensor.ds_numel, (
                        f"grad_position.numel ({numel}) != ds_tensor.ds_numel "
                        f"({fp16_param.ds_tensor.ds_numel}) for '{canon_name}'"
                    )
                if grp_idx != group_id:
                    continue

                param_state = {}
                for key, val in flat_state.items():
                    if isinstance(val, torch.Tensor) and val.numel() >= offset + numel:
                        param_state[key] = val.narrow(0, offset, numel).cpu()
                    else:
                        param_state[key] = val
                if param_state:
                    canonical_state[canon_name] = param_state

            orig_group_id = ds_optimizer.sub_group_to_group_id.get(group_id, group_id)
            if orig_group_id < len(base_optimizer.param_groups):
                group_info = {
                    k: v for k, v in base_optimizer.param_groups[orig_group_id].items()
                    if k != "params"
                }
            else:
                group_info = {}
            group_info["params"] = group_param_names
            canonical_groups.append(group_info)

        if skipped_params:
            accelerator.print(
                f"[Optimizer] Warning: Skipped {len(skipped_params)} params without "
                f"grad_position: {skipped_params[:5]}{'...' if len(skipped_params) > 5 else ''}"
            )

        payload = {
            "state": canonical_state,
            "param_groups": canonical_groups,
            "partition_rank": partition_rank,
            "subgroup_size": subgroup_size,
        }
        handler.save_state(payload, "optimizer_shards", rank, is_sharded=True)

    def _save_rng(self, handler, rank):
        """Save per-rank RNG states."""
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state()
        handler.save_single(rng_state, f"rng_state_{rank}.pt")

    def _save_ema_shadows(self, handler, rank, is_sharded):
        """Save main EMA shadows."""
        if self.ema is None:
            return
        shadows = {
            k: v.cpu().contiguous() for k, v in self.ema.shadows.items()
        }
        metadata = {
            "decay": str(self.ema.decay),
            "num_updates": str(self.ema.num_updates),
        }
        if is_sharded:
            pr, sg = CheckpointHandler.get_partition_info()
            metadata["partition_rank"] = str(pr)
            metadata["subgroup_size"] = str(sg)
        handler.save_tensors(shadows, "ema_shadows", rank, is_sharded, metadata)

    def _save_ema_model_export(self, accelerator, model, handler):
        """Save EMA model weights as safetensors.

        DDP: gather on rank 0 and save a single ema_model.safetensors.
        ZeRO-3: each rank saves its local shard as ema_model/rank_N.safetensors.
        """
        is_sharded = handler.is_zero3_active(accelerator)
        rank = accelerator.process_index
        unwrapped = accelerator.unwrap_model(model)

        accelerator.wait_for_everyone()
        if self.ema is not None:
            self.ema.apply_to_model(unwrapped)

        if is_sharded:
            trainable_names = unwrapped.trainable_param_names()
            state_dict = {
                name: get_local_data(param).cpu().contiguous()
                for name, param in unwrapped.named_parameters()
                if name in trainable_names
            }
            state_dict = unwrapped.export_trainable_state_dict(state_dict)
            state_dict = self.state_dict_converter(state_dict)
            if self.ema is not None:
                self.ema.restore_model(unwrapped)
            pr, sg = CheckpointHandler.get_partition_info()
            metadata = {"partition_rank": str(pr), "subgroup_size": str(sg)}
            handler.save_tensors(state_dict, "ema_model", rank, is_sharded=True, metadata=metadata)
        else:
            state_dict = accelerator.get_state_dict(model)
            if self.ema is not None:
                self.ema.restore_model(unwrapped)
            if accelerator.is_main_process:
                state_dict = unwrapped.export_trainable_state_dict(state_dict)
                state_dict = self.state_dict_converter(state_dict)
                handler.save_tensors(
                    state_dict, "ema_model", rank=0, is_sharded=False,
                )


    def _load_checkpoint_metadata(self, checkpoint_path):
        """Load checkpoint_metadata.json if it exists. Returns dict or None."""
        meta_path = os.path.join(checkpoint_path, "checkpoint_metadata.json")
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                return json.load(f)
        return None


    def resume(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        checkpoint_path: str,
        only_resume_weight: bool = False,
        steps_per_epoch: int = 0,
        optimizer=None,
        scheduler=None,
    ) -> int:
        """Resume."""
        if accelerator.is_main_process:
            self._scan_existing_checkpoints()

        handler = CheckpointHandler(checkpoint_path, accelerator.num_processes)
        is_sharded = handler.is_zero3_active(accelerator)
        rank = accelerator.process_index
        partition_rank, _ = (
            handler.get_partition_info() if is_sharded else (0, 1)
        )

        if only_resume_weight:
            self._load_model(handler, model, accelerator, is_sharded, partition_rank)
            self._sync_fp32_masters(model, accelerator)


            if self.ema is not None:
                if handler.exists("ema_shadows"):
                    self._load_ema_shadows(handler, model, accelerator, is_sharded, partition_rank)
                else:
                    unwrapped = accelerator.unwrap_model(model)
                    self.ema.copy_from_model(unwrapped)
                    accelerator.print(
                        "[Resume] No EMA shadows in checkpoint; "
                        "initialized EMA from resumed model weights."
                    )
            return 0

        fmt = handler.detect_format()
        accelerator.print(f"Detected checkpoint format: {fmt} at {checkpoint_path}")


        self._load_model(handler, model, accelerator, is_sharded, partition_rank, fmt)


        self._sync_fp32_masters(model, accelerator)


        if optimizer is not None:
            self._load_optimizer(
                handler, model, optimizer, accelerator,
                is_sharded, partition_rank, fmt,
            )


        self._load_scheduler(handler, scheduler, checkpoint_path, fmt)


        self._load_rng(handler, rank, checkpoint_path)


        if self.ema is not None:
            if handler.exists("ema_shadows"):
                self._load_ema_shadows(handler, model, accelerator, is_sharded, partition_rank)
            else:
                unwrapped = accelerator.unwrap_model(model)
                self.ema.copy_from_model(unwrapped)
                accelerator.print(
                    "[Resume] No EMA shadows in checkpoint; "
                    "initialized EMA from resumed model weights."
                )


        ckpt_metadata = self._load_checkpoint_metadata(checkpoint_path)
        is_latest = os.path.basename(os.path.normpath(checkpoint_path)) == "latest"

        if ckpt_metadata is not None:
            resumed_epoch = ckpt_metadata["epoch"]
            start_epoch = resumed_epoch + 1
            self.num_steps = start_epoch * steps_per_epoch

            accelerator.print(
                f"Resumed from {checkpoint_path} (via metadata JSON): "
                f"epoch={resumed_epoch}, global_step={ckpt_metadata.get('global_step', 'N/A')}, "
                f"start_epoch={start_epoch}, num_steps={self.num_steps}"
            )


            saved_world_size = ckpt_metadata.get("world_size")
            if saved_world_size is not None and saved_world_size != accelerator.num_processes:
                accelerator.print(
                    f"[Resume] Warning: checkpoint was saved with world_size={saved_world_size}, "
                    f"current world_size={accelerator.num_processes}"
                )
        elif is_latest:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' is named 'latest' but has no "
                f"checkpoint_metadata.json. Cannot determine epoch to resume from."
            )
        else:

            resumed_epoch = self._parse_epoch_from_path(checkpoint_path)
            start_epoch = resumed_epoch + 1
            self.num_steps = start_epoch * steps_per_epoch
            accelerator.print(
                f"Resumed from {checkpoint_path}: resumed_epoch={resumed_epoch}, "
                f"start_epoch={start_epoch}, num_steps={self.num_steps}"
            )


        if is_latest:
            self._pending_latest_cleanup = checkpoint_path

        return start_epoch


    def _load_model(self, handler, model, accelerator, is_sharded, partition_rank, fmt="new"):
        """Load model weights via streaming iteration."""
        unwrapped = accelerator.unwrap_model(model)
        canon_to_param = self._build_canon_param_map(model, unwrapped)


        if fmt == "legacy_zero3" and is_sharded:
            try:
                accelerator.print("[Resume] Legacy ZeRO-3 -> ZeRO-3: trying accelerator.load_state()...")
                accelerator.load_state(handler.checkpoint_dir)
                return
            except Exception as e:
                accelerator.print(f"[Resume] accelerator.load_state failed: {e}, falling back to streaming.")

        loaded = 0
        for name, full_tensor in handler.iter_tensors("model"):
            param = canon_to_param.get(name)
            if param is None:
                continue
            local = get_local_data(param)
            if is_sharded:
                shard = CheckpointHandler.extract_rank_shard(full_tensor, param, partition_rank)
            else:
                shard = full_tensor.flatten()[:local.numel()]
            if shard.shape != local.shape and shard.numel() == local.numel():
                shard = shard.reshape(local.shape)
            elif shard.numel() != local.numel():
                accelerator.print(
                    f"[Resume] Size mismatch for '{name}': "
                    f"checkpoint {shard.numel()} vs model {local.numel()}, skipping."
                )
                continue
            local.copy_(shard.to(local.device, local.dtype))
            loaded += 1

        accelerator.print(f"[Resume] Loaded {loaded} model parameters.")


        if is_sharded:
            unwrapped = accelerator.unwrap_model(model)
            invalidated = 0
            for param in unwrapped.parameters():
                if getattr(param, "ds_secondary_tensor", None) is not None:
                    param.ds_secondary_tensor = None
                    invalidated += 1
            if invalidated > 0:
                accelerator.print(
                    f"[Resume] Invalidated {invalidated} HPZ secondary tensors."
                )

    def _load_optimizer(self, handler, model, optimizer, accelerator, is_sharded, partition_rank, fmt):
        """Load optimizer state via streaming iteration."""

        opt_name = None
        if handler.exists("optimizer_shards"):
            opt_name = "optimizer_shards"
        elif handler.exists("optimizer_state"):
            opt_name = "optimizer_state"
        elif fmt in ("legacy_ddp",) and os.path.isfile(
            os.path.join(handler.checkpoint_dir, "optimizer.bin")
        ):
            opt_name = "optimizer_state"
        else:
            if fmt == "legacy_zero3":
                accelerator.print(
                    "[Resume] Legacy ZeRO-3 optimizer state cannot be loaded cross-mode. "
                    "Optimizer will start fresh."
                )
            else:
                accelerator.print("[Resume] No optimizer checkpoint found, skipping.")
            return

        unwrapped = accelerator.unwrap_model(model)
        canon_to_param = self._build_canon_param_map(model, unwrapped)

        if is_sharded:
            self._apply_optimizer_zero3(handler, opt_name, model, canon_to_param, partition_rank, accelerator)
        else:
            self._apply_optimizer_ddp(handler, opt_name, optimizer, canon_to_param, accelerator)

    def _apply_optimizer_ddp(self, handler, opt_name, optimizer, canon_to_param, accelerator):
        """Apply optimizer state from handler to DDP optimizer."""
        param_to_idx = {}
        idx_to_param = {}
        idx = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                param_to_idx[id(param)] = idx
                idx_to_param[str(idx)] = param
                idx += 1

        new_state = {}
        for canon_name, state_dict in handler.iter_optimizer_state(opt_name):
            param = canon_to_param.get(canon_name)
            if param is None:
                param = idx_to_param.get(canon_name)
            if param is None:
                continue
            param_idx = param_to_idx.get(id(param))
            if param_idx is None:
                continue
            restored = {}
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:


                    trimmed = v.flatten()[:param.numel()]
                    restored[k] = trimmed.reshape(param.shape).to(device=param.device)
                elif isinstance(v, torch.Tensor):
                    restored[k] = v.to(device=param.device)
                else:
                    restored[k] = v
            new_state[param_idx] = restored

        if new_state:
            current_sd = optimizer.state_dict()
            current_sd["state"] = new_state
            optimizer.load_state_dict(current_sd)
            accelerator.print(f"[Resume] Loaded optimizer state for {len(new_state)} parameters.")
        else:
            accelerator.print("[Resume] No matching optimizer state found.")

    def _apply_optimizer_zero3(self, handler, opt_name, model, canon_to_param, partition_rank, accelerator):
        """Apply optimizer state from handler to ZeRO-3 optimizer."""
        ds_optimizer = model.optimizer
        base_optimizer = ds_optimizer.optimizer
        loaded = 0
        skipped = []

        for canon_name, state_dict in handler.iter_optimizer_state(opt_name):
            if canon_name not in canon_to_param:
                continue
            fp16_param = canon_to_param[canon_name]
            param_id = ds_optimizer.get_param_id(fp16_param)
            if param_id not in ds_optimizer.grad_position:
                skipped.append(canon_name)
                continue

            grp_idx, offset, numel = ds_optimizer.grad_position[param_id]
            if hasattr(fp16_param, "ds_tensor"):
                assert numel == fp16_param.ds_tensor.ds_numel, (
                    f"grad_position.numel ({numel}) != ds_tensor.ds_numel "
                    f"({fp16_param.ds_tensor.ds_numel}) for '{canon_name}'"
                )
            fp32_flat = ds_optimizer.fp32_partitioned_groups_flat[grp_idx]
            flat_state = base_optimizer.state.get(fp32_flat)
            if flat_state is None:


                flat_state = {}
                for key, val in state_dict.items():
                    if isinstance(val, torch.Tensor) and val.dim() > 0:
                        flat_state[key] = torch.zeros(
                            fp32_flat.shape,
                            dtype=fp32_flat.dtype,
                            device=fp32_flat.device,
                        )
                    else:
                        flat_state[key] = val
                base_optimizer.state[fp32_flat] = flat_state
            if not flat_state:
                continue

            for key, full_val in state_dict.items():
                target = flat_state.get(key)
                if target is None:


                    if isinstance(full_val, torch.Tensor) and full_val.dim() > 0:
                        flat_state[key] = torch.zeros(
                            fp32_flat.shape,
                            dtype=fp32_flat.dtype,
                            device=fp32_flat.device,
                        )
                        target = flat_state[key]
                    else:
                        flat_state[key] = full_val
                        continue
                if not isinstance(full_val, torch.Tensor) or full_val.dim() == 0:
                    continue
                shard = CheckpointHandler.extract_rank_shard(full_val, fp16_param, partition_rank)
                if target.numel() >= offset + numel:
                    target[offset:offset + numel] = shard[:numel].to(
                        target.device, target.dtype
                    )
            loaded += 1

        if skipped:
            accelerator.print(
                f"[Resume] Warning: Skipped optimizer state for {len(skipped)} params "
                f"without grad_position: {skipped[:5]}{'...' if len(skipped) > 5 else ''}"
            )
        accelerator.print(f"[Resume] Loaded ZeRO-3 optimizer state for {loaded} parameters.")

    def _load_scheduler(self, handler, scheduler, checkpoint_path, fmt):
        """Load scheduler state."""
        if scheduler is None:
            return

        sched_path = os.path.join(handler.checkpoint_dir, "scheduler.pt")
        if os.path.isfile(sched_path):
            state = torch.load(sched_path, map_location="cpu", weights_only=True)
            scheduler.load_state_dict(state)
            return

        sched_bin = os.path.join(handler.checkpoint_dir, "scheduler.bin")
        if os.path.isfile(sched_bin):
            state = torch.load(sched_bin, map_location="cpu", weights_only=True)
            scheduler.load_state_dict(state)

    def _load_rng(self, handler, rank, checkpoint_path):
        """Load rng."""
        d = handler.checkpoint_dir
        for ext in (".pt", ".pkl"):
            path = os.path.join(d, f"rng_state_{rank}{ext}")
            if os.path.isfile(path):
                rng_state = torch.load(
                    path, map_location="cpu",
                    weights_only=False,
                )
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.random.set_rng_state(rng_state["cpu"])
                if "cuda" in rng_state and torch.cuda.is_available():
                    torch.cuda.set_rng_state(rng_state["cuda"])
                return
        print(f"[Resume] Warning: No RNG state found for rank {rank}, using current random state.")

    def _load_ema_shadows(self, handler, model, accelerator, is_sharded, partition_rank):
        """Load main EMA shadows via streaming iteration."""
        if self.ema is None:
            return
        if not handler.exists("ema_shadows"):
            accelerator.print("[Resume] Warning: ema_shadows checkpoint not found, skipping.")
            return

        meta = handler.read_metadata("ema_shadows")
        self.ema.decay = meta.get("decay", self.ema.decay)
        self.ema.num_updates = meta.get("num_updates", self.ema.num_updates)

        unwrapped = accelerator.unwrap_model(model)
        param_dict = dict(unwrapped.named_parameters())
        loaded = 0
        for canon_name, full_tensor in handler.iter_tensors("ema_shadows"):
            if canon_name not in self.ema.shadows:
                continue
            param = param_dict.get(canon_name)
            current = self.ema.shadows[canon_name]
            if is_sharded and param is not None:
                shard = CheckpointHandler.extract_rank_shard(full_tensor, param, partition_rank)
            elif is_sharded:
                accelerator.print(
                    f"[EMA] Warning: param '{canon_name}' not found in model, "
                    f"cannot extract shard in ZeRO-3 mode. Skipping."
                )
                continue
            else:
                shard = full_tensor.flatten()[:current.numel()]
            if shard.shape != current.shape and shard.numel() == current.numel():
                shard = shard.reshape(current.shape)
            elif shard.numel() != current.numel():
                accelerator.print(
                    f"[EMA] Size mismatch for '{canon_name}': "
                    f"checkpoint {shard.numel()} vs current {current.numel()}, skipping."
                )
                continue
            self.ema.shadows[canon_name] = shard.to(current.device, current.dtype)
            loaded += 1


    def _sync_fp32_masters(self, model, accelerator):
        """Sync FP32 master copies from loaded FP16 ds_tensor values.

        In ZeRO-3, every optimizer.step() casts fp32_partitioned_groups_flat
        back to FP16 and writes into ds_tensor.  After loading new checkpoint
        weights into ds_tensor, the stale FP32 masters must be updated —
        otherwise the first optimizer.step() overwrites loaded weights.
        """
        if not CheckpointHandler.is_zero3_active(accelerator):
            return

        ds_optimizer = model.optimizer
        synced = 0

        with torch.no_grad():
            for group_id, fp16_params in enumerate(ds_optimizer.fp16_groups):
                fp32_flat = ds_optimizer.fp32_partitioned_groups_flat[group_id]

                for fp16_param in fp16_params:
                    param_id = ds_optimizer.get_param_id(fp16_param)
                    if param_id not in ds_optimizer.grad_position:
                        continue
                    grp_idx, offset, numel = ds_optimizer.grad_position[param_id]
                    if grp_idx != group_id:
                        continue

                    local_data = get_local_data(fp16_param)
                    fp32_flat[offset:offset + numel].copy_(
                        local_data.float()[:numel]
                    )
                    synced += 1

        accelerator.print(
            f"[Resume] Synced {synced} FP32 master parameters from loaded FP16 weights."
        )
