import gc
import os, torch
import signal
from datetime import datetime, timezone
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_scheduler
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from .profiler import NullProfiler, StepProfiler
from .validation import merge_validation_results
import numpy as np
import psutil

_terminate_requested = False


def _sigterm_handler(signum, frame):
    global _terminate_requested
    _terminate_requested = True


def _gather_and_log_validation_results(
    accelerator, local_result, global_step, prefix, metric_fn=None, args=None
):
    """Gather per-rank validation results and log to W&B.

    All ranks participate in the gather. Only rank 0 merges results,
    runs ``metric_fn`` to compute metrics from saved files, and logs
    to W&B.

    Args:
        accelerator: HF Accelerator with wandb tracker initialized.
        local_result: This rank's dict from ``validation_fn``.
        global_step: Training step to attach to the W&B row.
        prefix: Metric key prefix, e.g. ``"val"`` or ``"val_ema"``.
        metric_fn: Optional callable ``(merged_result, device, args) -> dict``
            that computes metrics on rank 0 from saved frames on disk.
        args: Training args passed through to ``metric_fn``.
    """
    from accelerate.utils import gather_object


    per_rank = gather_object([local_result])

    if not accelerator.is_main_process:
        return

    merged = merge_validation_results(per_rank)


    scalars = {}
    if metric_fn is not None and merged["generated_videos"]:
        scalars = metric_fn(merged, accelerator.device, args)

    try:
        import wandb
    except ImportError:
        return

    try:
        wandb_run = accelerator.get_tracker("wandb", unwrap=True)
    except ValueError:
        return

    log_dict = {}
    for k, v in scalars.items():
        log_dict[f"{prefix}/{k}"] = v
    for k, v in merged["counters"].items():
        log_dict[f"{prefix}/{k}"] = v
    if merged["vis_video_paths"]:
        log_dict[f"{prefix}/cases"] = [
            wandb.Video(p, format="mp4") for p in merged["vis_video_paths"]
        ]

    if log_dict:
        wandb_run.log(log_dict, step=global_step)


def _zero_nan_grads(model) -> int:
    """Replace NaN/Inf values in all parameter gradients with 0.

    Works with DDP, FSDP (v1 & v2), and DeepSpeed (ZeRO 0/1/2/3).
    For FSDP1 sharded grads and DeepSpeed ZeRO-3 partitioned params,
    we guard with numel() > 0 to skip empty shards.

    Returns the total number of non-finite gradient elements found
    (on this rank only, no cross-rank reduction).
    """
    nan_count = 0
    for p in model.parameters():
        if p.grad is not None and p.grad.numel() > 0:
            bad = ~torch.isfinite(p.grad)
            if bad.any():
                nan_count += bad.sum().item()
    if nan_count > 0:
        for p in model.parameters():
            if p.grad is not None and p.grad.numel() > 0:
                p.grad.data.zero_()
    return nan_count


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_epochs: int = None,
    num_epochs: int = 1,
    validation_fn=None,
    metric_fn=None,
    val_every_n_epochs: int = None,
    worker_init_fn=None,
    collate_fn=None,
    train_logger=None,
    batch_sampler=None,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_epochs = args.save_epochs
        num_epochs = args.num_epochs
        val_every_n_epochs = getattr(args, "val_every_n_epochs", None)
        if train_logger is None:
            train_logger = "wandb" if getattr(args, "use_wandb", False) else None


        if hasattr(args, "seed") and args.seed is not None:
            set_seed(args.seed, device_specific=True)
            print(f"Random seed set to {args.seed} for reproducibility.")


    wandb_log_every_n_steps = (
        getattr(args, "wandb_log_every_n_steps", 1) if args is not None else 1
    )


    warmup_steps = getattr(args, "warmup_steps", 0) if args is not None else 0
    scheduler_type = (
        getattr(args, "scheduler_type", "constant_with_warmup")
        if args is not None
        else "constant_with_warmup"
    )

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay
    )
    if args.limit_train_batches is not None:
        dataset = torch.utils.data.Subset(
            dataset,
            np.linspace(
                0, len(dataset) - 1, args.limit_train_batches, dtype=int
            ).tolist(),
        )
    batch_size = getattr(args, "batch_size", 1) if args is not None else 1
    pin_memory = getattr(args, "pin_memory", False) if args is not None else False
    if collate_fn is None:
        collate_fn = lambda x: x[0]


    _user_worker_init_fn = worker_init_fn

    def _worker_init_fn_wrapper(worker_id):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if _user_worker_init_fn is not None:
            _user_worker_init_fn(worker_id)

    if batch_sampler is not None:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin_memory,
            worker_init_fn=_worker_init_fn_wrapper,
            prefetch_factor=1 if num_workers > 0 else None,
        )

        batch_size = sum(batch_sampler.batch_sizes)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=pin_memory,
            worker_init_fn=_worker_init_fn_wrapper,
            drop_last=True if len(dataset) >= batch_size else False,
            prefetch_factor=1 if num_workers > 0 else None,
        )


    num_training_steps = num_epochs * len(dataloader)
    warmup_steps_scaled = warmup_steps * accelerator.num_processes
    accelerator.print(
        f"Scheduler: {scheduler_type}, warmup_steps={warmup_steps} "
        f"(scaled to {warmup_steps_scaled} for {accelerator.num_processes} processes), "
        f"num_training_steps={num_training_steps}"
    )
    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps_scaled,
        num_training_steps=num_training_steps,
    )

    if batch_sampler is not None:


        if accelerator.state.deepspeed_plugin is not None:
            accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = batch_size
        model, optimizer, scheduler = accelerator.prepare(
            model, optimizer, scheduler
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )


    model_logger.init_ema(model, accelerator)


    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "before_train_start"):
        unwrapped_model.before_train_start(accelerator, model_logger)


    start_epoch = 0
    if args is not None and getattr(args, "resume_checkpoint", None) is not None:
        start_epoch = model_logger.resume(
            accelerator,
            model,
            checkpoint_path=args.resume_checkpoint,
            only_resume_weight=getattr(args, "only_resume_weight", False),
            steps_per_epoch=len(dataloader),
            optimizer=optimizer,
            scheduler=scheduler,
        )
        accelerator.wait_for_everyone()


    if validation_fn is not None and getattr(args, "val_before_training", False):
        accelerator.print("Running validation before training...")
        unwrapped_model = accelerator.unwrap_model(model)
        pipe = unwrapped_model.pipe

        use_ema_for_val = model_logger.ema is not None and getattr(
            args, "val_use_ema", True
        )
        if use_ema_for_val:
            model_logger.apply_ema(unwrapped_model)

        local_result = validation_fn(
            pipe=pipe,
            accelerator=accelerator,
            output_path=model_logger.output_path,
            epoch_id=-1,
            args=args,
        )
        accelerator.wait_for_everyone()

        if use_ema_for_val:
            model_logger.restore_from_ema(unwrapped_model)
            prefix = "val_ema"
        else:
            prefix = "val"

        if train_logger is not None:
            _gather_and_log_validation_results(
                accelerator,
                local_result,
                0,
                prefix=prefix,
                metric_fn=metric_fn,
                args=args,
            )


    global _terminate_requested
    _terminate_requested = False
    signal.signal(signal.SIGTERM, _sigterm_handler)


    profile_timing = (
        getattr(args, "profile_timing", False) if args is not None else False
    )
    if profile_timing:
        profiler = StepProfiler(device=accelerator.device)
        accelerator.print("Per-step timing profiler: ENABLED")
    else:
        profiler = NullProfiler()
    accelerator.unwrap_model(model).profiler = profiler

    def _build_metadata(epoch_id, is_preempted=False):
        return {
            "epoch": epoch_id,
            "global_step": model_logger.num_steps,
            "num_epochs": num_epochs,
            "world_size": accelerator.num_processes,
            "batch_size": batch_size,
            "learning_rate": scheduler.get_last_lr()[0],
            "is_preempted": is_preempted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    for epoch_id in range(start_epoch, num_epochs):
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch_id)
        nan_count = 0
        for data in (
            pbar := tqdm(
                dataloader,
                desc=f"Training Epoch {epoch_id}",
                disable=not accelerator.is_main_process,
            )
        ):
            profiler.mark_step_start()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                with profiler.stage("forward_total"):
                    if hasattr(dataset, "load_from_cache") and dataset.load_from_cache:
                        loss_result = model({}, inputs=data)
                    else:
                        loss_result = model(data)


                if isinstance(loss_result, dict):
                    loss = loss_result["loss"]
                    sub_losses = {k: v for k, v in loss_result.items() if k != "loss"}
                else:
                    loss = loss_result
                    sub_losses = {}

                if not torch.isfinite(loss).any():
                    accelerator.print(
                        f"Warning: Non-finite loss {loss.item()} detected at epoch {epoch_id}.\n",
                        "Batch information:",
                        "Sequence ID(s):",
                        data.get("sequence_id", "N/A"),
                        "Start frame(s):",
                        data.get("start_idx", "N/A"),
                        "End frame(s):",
                        data.get("end_idx", "N/A"),
                        "Target camera(s):",
                        data.get("target_cam_id", "N/A"),
                        "Reference camera(s):",
                        data.get("reference_cam_id", "N/A"),
                    )
                with profiler.stage("backward"):
                    accelerator.backward(loss)
                if accelerator.sync_gradients:
                    nan_count = _zero_nan_grads(model)
                    if nan_count > 0:
                        accelerator.print(
                            f"Warning: Found {nan_count} non-finite gradient elements in epoch {epoch_id}. All grads have been zeroed."
                        )
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                if accelerator.sync_gradients:
                    model_logger.on_step_end(accelerator, model)
            torch.cuda.empty_cache()
            profiler.end_step()
            loss_val = loss.item()
            sub_loss_vals = {k: v.item() for k, v in sub_losses.items()}
            current_lr = scheduler.get_last_lr()[0]
            global_step = model_logger.num_steps
            perf_ms = profiler.readout_ms()

            postfix = {
                "loss": f"{loss_val:.4f}",
                **{k: f"{v:.4f}" for k, v in sub_loss_vals.items()},
                "nan_grad": f"{nan_count}",
                "lr": f"{current_lr:.2e}",
            }
            if perf_ms:


                tqdm_labels = {
                    "dataload": "data",
                    "pipe_prepare": "prep",
                    "dit_forward": "dit",
                    "backward": "bwd",
                }
                for full, short in tqdm_labels.items():
                    if full in perf_ms:
                        postfix[short] = f"{perf_ms[full]:.0f}ms"
            pbar.set_postfix(postfix)


            if train_logger is not None and global_step % wandb_log_every_n_steps == 0:
                log_dict = {
                    "train/loss": loss_val,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch_id,
                    "train/global_step": global_step,
                    "train/nan_grad_count": nan_count,
                }
                for k, v in sub_loss_vals.items():
                    log_dict[f"train/{k}"] = v
                for k, v in perf_ms.items():
                    log_dict[f"perf/{k}_ms"] = v
                accelerator.log(log_dict, step=global_step)


            if _terminate_requested:
                accelerator.print("SIGTERM received, saving checkpoint and exiting...")
                break

        if _terminate_requested:
            break

        model_logger.on_epoch_end(
            accelerator,
            model,
            epoch_id,
            save_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=_build_metadata(epoch_id),
        )


        if validation_fn is not None and val_every_n_epochs is not None:
            if (epoch_id + 1) % val_every_n_epochs == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                pipe = unwrapped_model.pipe

                use_ema_for_val = model_logger.ema is not None and getattr(
                    args, "val_use_ema", True
                )
                if use_ema_for_val:
                    model_logger.apply_ema(unwrapped_model)

                local_result = validation_fn(
                    pipe=pipe,
                    accelerator=accelerator,
                    output_path=model_logger.output_path,
                    epoch_id=epoch_id,
                    args=args,
                )
                accelerator.wait_for_everyone()

                if use_ema_for_val:
                    model_logger.restore_from_ema(unwrapped_model)
                    prefix = "val_ema"
                else:
                    prefix = "val"

                if train_logger is not None:
                    _gather_and_log_validation_results(
                        accelerator,
                        local_result,
                        model_logger.num_steps + 1,
                        prefix=prefix,
                        metric_fn=metric_fn,
                        args=args,
                    )

    if _terminate_requested:

        metadata = _build_metadata(epoch_id, is_preempted=True)
        model_logger.save_checkpoint(
            accelerator, model, "latest",
            optimizer=optimizer, scheduler=scheduler, metadata=metadata,
        )
    else:
        metadata = _build_metadata(num_epochs - 1)
        model_logger.on_training_end(
            accelerator,
            model,
            save_epochs,
            num_epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            metadata=metadata,
        )


    if train_logger is not None:
        accelerator.end_training()


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(
                    model_logger.output_path, str(accelerator.process_index)
                )
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(
                    model_logger.output_path,
                    str(accelerator.process_index),
                    f"{data_id}.pth",
                )
                data = model(data)
                torch.save(data, save_path)
