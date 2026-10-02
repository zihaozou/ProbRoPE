"""Train D-FUSE with ProbRoPE on paired RGB and event videos."""

import os
import json
import torch
import argparse
import accelerate
import warnings
from functools import partial

from dfuse.core import ModelConfig
from dfuse.core.data.video_dataset import VideoDataset, VideoDatasetConfig
from dfuse.core.data.processors import (
    BaseProcessor,
    MultiViewPairProcessor,
    MultiViewPairProcessorConfig,
    StaticProcessor,
    StaticProcessorConfig,
)
from dfuse.utils.metadata_db import MetadataDB
from dfuse.core.data.multi_dataset_sampler import MultiDatasetBatchSampler
from dfuse.pipelines.dfuse import (
    DFusePipeline,
)
from dfuse.diffusion.validation import (
    DFuseValDataset,
    run_validation as _shared_run_validation,
    compute_dfuse_metrics,
)
from dfuse.diffusion import (
    DiffusionTrainingModule,
    FlowMatchSFTLoss,
    add_general_config,
    add_video_size_config,
    launch_data_process_task,
    launch_training_task,
    ModelLogger,
)
from dfuse.diffusion.profiler import NullProfiler
import torch.nn.functional as F
from datetime import timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_resize_scales(s: str):
    """Parse semicolon-separated comma-separated floats.

    Example: "1.0,0.5;0.25" -> [[1.0, 0.5], [0.25]]
    """
    return [
        [float(x) for x in part.split(",") if x.strip()]
        for part in s.split(";")
        if part.strip()
    ]


def _build_processor(
    dataset_type: str,
    args,
    enable_frr,
    resize_scales,
) -> BaseProcessor:
    """Build the data processor for a given dataset_type.

    Called once per --dataset_base_path after peeking at the path's
    metadata.db.  Returns a BaseProcessor instance configured from CLI
    args.  Raises ValueError for unsupported dataset_types.
    """
    if dataset_type == "multiview":
        return MultiViewPairProcessor(
            MultiViewPairProcessorConfig(
                num_frames_list=args.num_frames_list,
                resize_scales=resize_scales,
                resolution_alignment=args.resolution_alignment,
                enable_frame_rate_reduction=enable_frr,
                fps=args.fps,
                scene_scale=args.scene_scale,
                target_modalities=args.target_modalities,
                source_modalities=args.source_modalities,
                fixed_cameras_only=args.fixed_cameras_only,
                sampling_mode=args.sampling_mode,
                normalize_to_target_frame=args.normalize_to_target_frame,
            )
        )
    if dataset_type == "static":
        return StaticProcessor(
            StaticProcessorConfig(
                resize_scales=resize_scales,
                resolution_alignment=args.resolution_alignment,
                fps=args.fps,
                scene_scale=args.scene_scale,
                normalize_to_target_frame=args.normalize_to_target_frame,
            )
        )
    raise ValueError(
        f"Unsupported dataset_type for training: {dataset_type!r}"
    )


def run_validation(pipe, accelerator, output_path, epoch_id, args=None):
    """Runner-facing wrapper that builds the per-epoch output dir."""
    output_dir = os.path.join(output_path, "validation", f"epoch_{epoch_id:04d}")
    return _shared_run_validation(
        pipe=pipe, accelerator=accelerator, output_dir=output_dir, args=args,
    )


class DFuseTrainingModule(DiffusionTrainingModule):
    """
    Training module for the Wan Video DFuse pipeline.

    Trains cam_encoder, projector, and self_attn in each DiTBlock.
    All other parameters are frozen.
    """

    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_activation_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        torch_dtype=torch.bfloat16,
        deepspeed_enabled=False,
        source_noise_timestep_range=[200, 500],
        enable_source_noise=False,
        noise_level_rgb=None,
        noise_level_event=None,
        fps=30.0,
        source_dropout=0.0,
        modulation_type=None,
        enable_position_phase=False,
        cam_embed_mode="plucker",
        num_modalities=2,
        p_text_dropout=0.0,
        p_modality_drop=None,
        source_modalities=None,
        shard_models=None,


    ):
        super().__init__()

        if p_modality_drop is None:
            p_modality_drop = [0.0] * num_modalities


        if not use_gradient_checkpointing and not use_activation_offload:
            warnings.warn(
                "Neither gradient checkpointing nor activation offload is enabled. "
                "Enabling gradient checkpointing to prevent OOM."
            )
            use_gradient_checkpointing = True


        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )


        tokenizer_config = (
            ModelConfig(
                model_id="Wan-AI/Wan2.1-T2V-1.3B",
                origin_file_pattern="google/umt5-xxl/",
            )
            if tokenizer_path is None
            else ModelConfig(tokenizer_path)
        )


        from dfuse.configs import MODEL_CONFIGS

        for cfg in MODEL_CONFIGS:
            if cfg.get("model_name") == "dfuse":
                cfg["extra_kwargs"]["num_modalities"] = num_modalities
                cfg["extra_kwargs"]["modulation_type"] = modulation_type
                cfg["extra_kwargs"]["enable_position_phase"] = enable_position_phase


        self.pipe = DFusePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            strict_dit=False,
        )


        if hasattr(self.pipe.dit, "base_fps"):
            self.pipe.dit.base_fps = fps


        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model
        )


        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
            deepspeed_enabled=deepspeed_enabled,
        )


        if deepspeed_enabled:
            self.cache_trainable_param_names()


        if shard_models is not None:
            from dfuse.diffusion.sharding import ModuleShardingManager

            self.sharding_manager = ModuleShardingManager(self.pipe, shard_models)
            self.sharding_manager.detach()


        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_activation_offload = use_activation_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.source_noise_timestep_range = source_noise_timestep_range
        self.enable_source_noise = enable_source_noise
        self.noise_level_rgb = noise_level_rgb
        self.noise_level_event = noise_level_event
        self.source_dropout = source_dropout
        self.p_text_dropout = p_text_dropout
        self.p_modality_drop = p_modality_drop



        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: (
                FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi)
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: (
                FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi)
            ),
        }


        total_params = sum(p.numel() for p in self.pipe.dit.parameters())
        trainable_params_list = [
            (name, p) for name, p in self.pipe.dit.named_parameters() if p.requires_grad
        ]
        trainable_params = sum(p.numel() for _, p in trainable_params_list)
        print(
            f"DiT parameters: {total_params:,} total, {trainable_params:,} trainable "
            f"({100 * trainable_params / total_params:.1f}%)"
        )
        if trainable_params_list:
            max_name_len = max(len(name) for name, _ in trainable_params_list)
            col_w = min(max_name_len, 80)
            header = f"  {'Parameter Name':<{col_w}}  {'Shape':<28}  {'Count':>14}"
            sep = "  " + "-" * col_w + "  " + "-" * 28 + "  " + "-" * 14
            print(f"\n  Trainable Parameters ({len(trainable_params_list)}):")
            print(sep)
            print(header)
            print(sep)
            for name, p in trainable_params_list:
                shape_str = str(list(p.shape))
                count_str = f"{p.numel():>14,}"
                if len(name) > col_w:

                    print(f"  {name}")
                    print(f"  {'':<{col_w}}  {shape_str:<28}  {count_str}")
                else:
                    print(f"  {name:<{col_w}}  {shape_str:<28}  {count_str}")
            print(sep)

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        """Parse additional inputs from data dict."""
        for extra_input in extra_inputs:
            if extra_input in data:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        """Convert VideoDataset sample(s) to pipeline inputs for training.

        Handles both single-sample and batched data. Videos arrive in packed
        (P, 3) format and are unpacked to per-sample (T, 3, H, W) tensors.

        Returns:
            Tuple of (inputs_shared, inputs_posi, inputs_nega)
        """
        is_batched = "_batch_size" in data and data["_batch_size"] > 1

        if is_batched:
            batch_size = data["_batch_size"]
            num_frames_list = data["_num_frames_list"]
            heights = data["_target_heights"]
            widths = data["_target_widths"]
            view_num_frames_list = data["_source_view_num_frames"]


            tgt_pixel_counts = data["_target_pixel_counts"]
            tgt_chunks = list(
                torch.split(data["target_video"], tgt_pixel_counts, dim=0)
            )
            target_video_list = [
                chunk.reshape(T, H, W, 3).permute(0, 3, 1, 2)
                for chunk, T, H, W in zip(tgt_chunks, num_frames_list, heights, widths)
            ]


            src_pixel_counts = data["_source_pixel_counts"]
            src_chunks = list(
                torch.split(data["source_videos"], src_pixel_counts, dim=0)
            )
            view_pixel_counts_list = data["_source_view_pixel_counts"]
            source_video_list = []
            for sample_chunk, view_pcs, vfs, H, W in zip(
                src_chunks,
                view_pixel_counts_list,
                view_num_frames_list,
                heights,
                widths,
            ):
                view_chunks = list(torch.split(sample_chunk, view_pcs, dim=0))
                source_video_list.append(
                    [
                        vc.reshape(nf, H, W, 3).permute(0, 3, 1, 2)
                        for vc, nf in zip(view_chunks, vfs)
                    ]
                )

            prompt = data.get("prompt", [""] * batch_size)
            negative_prompt = None
            num_frames = num_frames_list

            target_fps = data.get("target_fps")
            source_view_fps = data.get("source_view_fps")

            target_modality = data.get("target_modality", [0] * batch_size)
            source_modality_list = data.get(
                "source_modalities",
                [[0] * len(vf) for vf in view_num_frames_list],
            )


            tgt_int_cat = data.get("target_intrinsics")
            tgt_ext_cat = data.get("target_extrinsics")
            if tgt_int_cat is not None:
                target_intrinsics = list(
                    torch.split(tgt_int_cat, num_frames_list, dim=0)
                )
                target_extrinsics = list(
                    torch.split(tgt_ext_cat, num_frames_list, dim=0)
                )
            else:
                target_intrinsics = None
                target_extrinsics = None

            src_frame_sizes = [sum(vf) for vf in view_num_frames_list]
            src_int_cat = data.get("source_intrinsics")
            src_ext_cat = data.get("source_extrinsics")
            if src_int_cat is not None:
                source_intrinsics = list(
                    torch.split(src_int_cat, src_frame_sizes, dim=0)
                )
                source_extrinsics = list(
                    torch.split(src_ext_cat, src_frame_sizes, dim=0)
                )
            else:
                source_intrinsics = None
                source_extrinsics = None

            source_view_T_video = view_num_frames_list

        else:

            H = data["target_height"]
            W = data["target_width"]
            T = data["num_frames"]
            num_frames = [T]
            heights = [H]
            widths = [W]
            view_num_frames = data["source_view_num_frames"]


            target_video_list = [
                data["target_video"].reshape(T, H, W, 3).permute(0, 3, 1, 2)
            ]


            view_pcs = data["source_view_pixel_counts"]
            src_view_chunks = list(torch.split(data["source_videos"], view_pcs, dim=0))
            source_video_list = [
                [
                    vc.reshape(nf, H, W, 3).permute(0, 3, 1, 2)
                    for vc, nf in zip(src_view_chunks, view_num_frames)
                ]
            ]

            prompt = [data.get("prompt", "")]
            negative_prompt = None

            fps_val = data.get("target_fps")
            target_fps = [fps_val] if fps_val is not None else None
            svfps = data.get("source_view_fps")
            source_view_fps = [svfps] if svfps is not None else None

            target_modality = [data.get("target_modality", 0)]
            source_modality_list = [
                data.get("source_modalities", [0] * data.get("num_source_views", 1))
            ]

            tgt_int = data.get("target_intrinsics")
            target_intrinsics = [tgt_int] if tgt_int is not None else None
            tgt_ext = data.get("target_extrinsics")
            target_extrinsics = [tgt_ext] if tgt_ext is not None else None
            src_int = data.get("source_intrinsics")
            source_intrinsics = [src_int] if src_int is not None else None
            src_ext = data.get("source_extrinsics")
            source_extrinsics = [src_ext] if src_ext is not None else None
            source_view_T_video = [view_num_frames]

        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": negative_prompt}

        inputs_shared = {
            "input_video": target_video_list,
            "source_video": source_video_list,
            "heights": heights,
            "widths": widths,
            "num_frames": num_frames,
            "cfg_scale": 1,
            "tiled": False,
            "tile_size": None,
            "tile_stride": None,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_activation_offload": self.use_activation_offload,
            "source_noise_timestep_range": self.source_noise_timestep_range
            if self.enable_source_noise
            else None,
            "noise_level_rgb": self.noise_level_rgb,
            "noise_level_event": self.noise_level_event,
            "source_dropout": self.source_dropout,
            "target_fps": target_fps,
            "source_view_fps": source_view_fps,
            "target_modality": target_modality,
            "source_modality_list": source_modality_list,
            "p_text_dropout": self.p_text_dropout,
            "p_modality_drop": self.p_modality_drop,
            "target_intrinsics": target_intrinsics,
            "target_extrinsics": target_extrinsics,
            "source_intrinsics": source_intrinsics,
            "source_extrinsics": source_extrinsics,
            "source_view_T_video": source_view_T_video,
        }

        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)

        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        """Forward pass for training."""
        profiler = getattr(self, "profiler", None) or NullProfiler()

        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )

        with profiler.stage("pipe_prepare"):
            for unit in self.pipe.units:
                with profiler.stage(f"pipe_prepare/{unit.__class__.__name__}"):
                    inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        inputs_shared, inputs_posi, inputs_nega = inputs

        with profiler.stage("dit_forward"):
            loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def dfuse_parser():
    """Argument parser for DFuse training script."""
    parser = argparse.ArgumentParser(
        description="Training script for Wan DFuse pipeline with multi-view c2w camera conditioning."
    )
    parser.add_argument("--config", help="JSON training configuration; CLI options override its values.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer. If None, uses default Wan tokenizer.",
    )
    parser.add_argument(
        "--initialize_model_on_cpu",
        default=False,
        action="store_true",
        help="Whether to initialize models on CPU (for low VRAM).",
    )

    parser.add_argument(
        "--num_frames_list",
        type=int,
        nargs="+",
        default=[17, 33],
        help="List of target subsequence lengths for multiview pair sampling.",
    )
    parser.add_argument(
        "--resize_scales",
        type=str,
        default="1.0",
        help="Semicolon-separated list of comma-separated resize-scale lists, "
        "one set per --dataset_base_path.  Example: '1.0,0.5;0.25'.",
    )
    parser.add_argument(
        "--batch_size_per_dataset",
        type=int,
        nargs="+",
        default=None,
        help="Per-dataset per-rank batch sizes.  When set, --batch_size is "
        "ignored and each step pulls B_i samples from dataset i.  Length "
        "must equal the number of --dataset_base_path entries.",
    )
    parser.add_argument(
        "--resolution_alignment",
        type=int,
        default=16,
        help="Round scaled H, W to this multiple (VAE requirement). Default: 16.",
    )
    parser.add_argument(
        "--enable_source_noise",
        default=False,
        action="store_true",
        help="Whether to add noise to source condition latents during training (for robustness).",
    )
    parser.add_argument(
        "--source_noise_timestep_range",
        type=int,
        nargs=2,
        default=[200, 500],
        help="The range of timesteps to sample from when adding noise to source condition latents during training (e.g., 200 500).",
    )
    parser.add_argument(
        "--noise_level_rgb",
        type=float,
        default=None,
        help="Fixed noise level for RGB source views during inference/validation.",
    )
    parser.add_argument(
        "--noise_level_event",
        type=float,
        default=None,
        help="Fixed noise level for event source views during inference/validation.",
    )

    parser.add_argument(
        "--enable_frame_rate_reduction",
        type=str,
        default="true,false",
        help="Comma-separated list of true/false per source modality (matching --source_modalities order). "
        "Controls which modalities get frame-rate reduction. Default: 'true,false' (reduce RGB, keep events full-rate).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fixed fps used for FPS-aware RoPE.",
    )
    parser.add_argument(
        "--scene_scale",
        type=float,
        default=1.0,
        help="Scale factor for extrinsics translation. Amplifies camera position signal in PRoPE.",
    )
    parser.add_argument(
        "--normalize_to_target_frame",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Normalize extrinsics so target camera 0 is at the origin with identity "
        "orientation. All cameras expressed relative to this frame. "
        "Use --no-normalize_to_target_frame to disable.",
    )
    parser.add_argument(
        "--source_dropout",
        type=float,
        default=0.0,
        help="Probability of dropping each source token before the transformer blocks (default: 0.0, no dropout). Target tokens are never dropped.",
    )

    parser.add_argument(
        "--modulation_type",
        type=str,
        default=None,
        choices=["sinc", "gaussian", "learned"],
        help="Modulation type for self-attention: 'sinc' (uniform PDF), 'gaussian', or 'learned' (SVD-based). None disables modulation.",
    )
    parser.add_argument(
        "--enable_position_phase",
        default=False,
        action="store_true",
        help="Enable phase for sinc/gaussian modulation (learned always has phase).",
    )
    parser.add_argument(
        "--cam_embed_mode",
        type=str,
        default="plucker",
        choices=["plucker"],
        help="D-FUSE uses CausalConv3D Plücker-ray camera conditioning.",
    )

    parser.add_argument(
        "--val_num_samples",
        type=int,
        default=-1,
        help="Number of samples to generate during each validation run. -1 means unlimited.",
    )
    parser.add_argument(
        "--val_num_frames",
        type=int,
        default=None,
        help="Use the first N frames for validation. If None, uses the full sequence length from metadata.",
    )

    parser.add_argument(
        "--num_modalities",
        type=int,
        default=None,
        help="Number of modalities for D-FUSE. "
        "Defaults to len(source_modalities) if not set.",
    )
    parser.add_argument(
        "--source_modalities",
        type=str,
        nargs="+",
        default=["rgb", "events"],
        help="Modality names for source views (e.g. rgb events). "
        "Position in this list determines the modality index for the DiT.",
    )
    parser.add_argument(
        "--target_modalities",
        type=str,
        nargs="+",
        default=["rgb"],
        help="Modality names for the target view. "
        "Must be a subset of --source_modalities.",
    )

    parser.add_argument(
        "--p_text_dropout",
        type=float,
        default=0.0,
        help="Probability of dropping text prompt per sample during training for CFG (default: 0.0).",
    )
    parser.add_argument(
        "--p_modality_drop",
        type=str,
        default=None,
        help="Comma-separated list of per-modality dropout probabilities (matching --source_modalities order). "
        "E.g. '0.1,0.5' means 10%% drop for first modality, 50%% for second. "
        "Default: all zeros (no dropout).",
    )

    parser.add_argument(
        "--val_cfg_drop_source",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Drop source video conditioning in the negative CFG branch during validation (default: True). "
        "Use --no-val_cfg_drop_source to keep source in the negative branch.",
    )
    parser.add_argument(
        "--fixed_cameras_only",
        default=False,
        action="store_true",
        help="Only sample from fixed cameras and skip blocked frames. "
        "Requires camera_types and fixed_camera_valid_frames tables in metadata.db "
        "(created by classify_cameras_and_mark_blocked_frames.py).",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default="single_view",
        choices=["single_view", "multi_view"],
        help="Camera sampling mode. 'single_view': same camera for target and all sources. "
        "'multi_view': target and each source view sampled independently (with replacement).",
    )


    parser.add_argument(
        "--profile-timing",
        dest="profile_timing",
        action="store_true",
        help="Enable per-step timing profiler (tqdm postfix + perf/ W&B keys).",
    )
    return parser


if __name__ == "__main__":
    parser = dfuse_parser()
    from training_config import parse_training_args
    args = parse_training_args(parser)


    if args.num_modalities is None:
        args.num_modalities = len(args.source_modalities)


    deepspeed_plugin = None
    if args.use_deepspeed:
        from accelerate.utils import DeepSpeedPlugin
        from dfuse.core.uneven_tensor import patch_deepspeed_for_uneven_tensor

        patch_deepspeed_for_uneven_tensor()


        import deepspeed.runtime.zero.partition_parameters as _ds_pp

        _orig_sec = _ds_pp.Init._partition_param_sec

        def _fixed_partition_param_sec(
            self, param, buffer=None, has_been_updated=False
        ):


            from deepspeed.runtime.zero.partition_parameters import (
                ZeroParamStatus,
                PartitionedParamStatus,
            )

            assert param.ds_status is not ZeroParamStatus.INFLIGHT
            if param.ds_status is not ZeroParamStatus.AVAILABLE:
                return
            if param.ds_secondary_tensor is not None and not has_been_updated:
                return
            tensor_size = self._aligned_size(param)
            secondary_partition_size = int(tensor_size // self.num_ranks_in_param_group)
            if param.ds_secondary_tensor is None:
                import torch as _torch

                t = _torch.empty(
                    secondary_partition_size,
                    dtype=param.dtype,
                    device=self.remote_device,
                )
                if self.pin_memory:
                    t = t.pin_memory()
                if not param.requires_grad and self.quantized_nontrainable_weights:
                    t, t.ds_quant_scale = self.quantizer_module.quantize(t)
                t.requires_grad = False
                param.ds_secondary_tensor = t
                param.ds_secondary_tensor.ds_numel = secondary_partition_size
                param.ds_secondary_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_secondary_tensor.final_location = None
            secondary_start = secondary_partition_size * self.rank_in_group
            sec_numel = max(
                0, min(param.ds_numel - secondary_start, secondary_partition_size)
            )
            if sec_numel > 0:
                import torch as _torch

                one_dim = param.contiguous().view(-1)
                with _torch.no_grad():
                    param.ds_secondary_tensor.narrow(0, 0, sec_numel).copy_(
                        one_dim.narrow(0, secondary_start, sec_numel)
                    )
            from deepspeed.utils import get_accelerator

            if not get_accelerator().resolves_data_dependency():
                get_accelerator().current_stream().synchronize()

        _ds_pp.Init._partition_param_sec = _fixed_partition_param_sec

        if args.deepspeed_config_file is not None:
            deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config_file)
        else:
            deepspeed_plugin = DeepSpeedPlugin(
                zero_stage=args.deepspeed_stage,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                offload_optimizer_device="cpu"
                if args.deepspeed_offload_optimizer
                else None,
                offload_param_device="cpu" if args.deepspeed_offload_params else None,
            )

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.use_wandb else None,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=args.find_unused_parameters,
                static_graph=args.static_graph,
            ),
            accelerate.InitProcessGroupKwargs(timeout=timedelta(minutes=60)),
        ],
    )
    num_cpu = len(os.sched_getaffinity(0))
    print(
        f"Rank {accelerator.process_index}: Accelerator initialized. Using {num_cpu} CPU cores for data loading."
    )
    accelerator.wait_for_everyone()


    if args.use_wandb:
        os.makedirs(args.output_path, exist_ok=True)
        init_kwargs = {"dir": args.output_path}
        if args.wandb_run_name is not None:
            init_kwargs["name"] = args.wandb_run_name
        if args.wandb_entity is not None:
            init_kwargs["entity"] = args.wandb_entity
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": init_kwargs},
        )

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]


    dataset_base_paths = [
        p.strip() for p in args.dataset_base_path.split(",") if p.strip()
    ]


    dataset_repeats = [
        int(v.strip()) for v in args.dataset_repeat.split(",") if v.strip()
    ]
    if len(dataset_repeats) == 1:
        dataset_repeats = dataset_repeats * len(dataset_base_paths)
    elif len(dataset_repeats) != len(dataset_base_paths):
        raise ValueError(
            f"--dataset_repeat has {len(dataset_repeats)} values but "
            f"--dataset_base_path has {len(dataset_base_paths)} paths. "
            f"Provide one value per path or a single value to broadcast."
        )


    enable_frr = [
        v.strip().lower() == "true" for v in args.enable_frame_rate_reduction.split(",")
    ]


    if args.p_modality_drop is not None:
        p_modality_drop = [float(v.strip()) for v in args.p_modality_drop.split(",")]
    else:
        p_modality_drop = [0.0] * args.num_modalities


    shard_models = None
    if args.use_deepspeed and args.trainable_models is not None:
        from dfuse.diffusion.sharding import ModuleShardingManager

        if args.shard_models is not None:
            shard_models = [s.strip() for s in args.shard_models.split(";")]
        else:
            shard_models = ModuleShardingManager.derive_shard_models(
                args.trainable_models
            )

    resize_scales_per_dataset = _parse_resize_scales(args.resize_scales)
    if len(resize_scales_per_dataset) != len(dataset_base_paths):
        raise ValueError(
            f"--resize_scales has {len(resize_scales_per_dataset)} sets but "
            f"--dataset_base_path has {len(dataset_base_paths)} paths."
        )


    sub_datasets = []
    for base_path, repeat, scales in zip(
        dataset_base_paths, dataset_repeats, resize_scales_per_dataset,
    ):
        metadata_path = args.dataset_metadata_path
        if metadata_path is None:
            metadata_path = os.path.join(base_path, "metadata.db")

        with MetadataDB(metadata_path, readonly=True) as _peek:
            type_counts = _peek.count_sequences_by_type()
        if len(type_counts) != 1:
            raise ValueError(
                f"{metadata_path}: expected exactly one dataset_type, "
                f"got {type_counts}"
            )
        dataset_type = next(iter(type_counts))

        processor = _build_processor(dataset_type, args, enable_frr, scales)
        ds_config = VideoDatasetConfig(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
        )
        sub_datasets.append(VideoDataset(ds_config, processor))

    if len(sub_datasets) == 1:
        dataset = sub_datasets[0]
    else:
        dataset = torch.utils.data.ConcatDataset(sub_datasets)

    if accelerator.is_main_process:
        print(
            f"Training dataset loaded with {len(dataset)} samples from {len(dataset_base_paths)} path(s)"
        )
        for bp, ds in zip(dataset_base_paths, sub_datasets):
            print(f"  - {bp} (processor={ds.processor.__class__.__name__})")
        print(f"  - num_frames_list: {args.num_frames_list}")


    val_dataset = None
    if args.val_dataset_base_path is not None and args.val_every_n_epochs is not None:
        val_dataset = True
        if accelerator.is_main_process:
            print(f"Validation enabled from: {args.val_dataset_base_path}")


    model = DFuseTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_activation_offload=args.use_activation_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        torch_dtype=torch_dtype,
        deepspeed_enabled=args.use_deepspeed,
        source_noise_timestep_range=args.source_noise_timestep_range,
        enable_source_noise=args.enable_source_noise,
        noise_level_rgb=args.noise_level_rgb,
        noise_level_event=args.noise_level_event,
        fps=args.fps,
        source_dropout=args.source_dropout,
        modulation_type=args.modulation_type,
        enable_position_phase=args.enable_position_phase,
        cam_embed_mode=args.cam_embed_mode,
        num_modalities=args.num_modalities,
        p_text_dropout=args.p_text_dropout,
        p_modality_drop=p_modality_drop,
        source_modalities=args.source_modalities,
        shard_models=shard_models,
    )


    model_logger = ModelLogger(
        args.output_path,
        keep_n_checkpoints=args.keep_n_checkpoints,
        use_ema=getattr(args, "use_ema", False),
        ema_decay=getattr(args, "ema_decay", 0.9999),
    )


    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }

    batch_sampler = None
    if args.batch_size_per_dataset is not None:
        if len(args.batch_size_per_dataset) != len(sub_datasets):
            raise ValueError(
                f"--batch_size_per_dataset has {len(args.batch_size_per_dataset)} "
                f"values but {len(sub_datasets)} datasets are configured."
            )
        batch_sampler = MultiDatasetBatchSampler(
            dataset_sizes=[len(d) for d in sub_datasets],
            batch_sizes=args.batch_size_per_dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=getattr(args, "seed", None) or 0,
        )
        if accelerator.is_main_process:
            print(
                f"MultiDatasetBatchSampler: batch_sizes={args.batch_size_per_dataset}, "
                f"per-rank batch={sum(args.batch_size_per_dataset)}, "
                f"steps_per_epoch={len(batch_sampler)}"
            )

    if args.task in ["sft", "sft:train"]:
        launcher_map[args.task](
            accelerator,
            dataset,
            model,
            model_logger,
            validation_fn=run_validation
            if val_dataset is not None
            else None,
            metric_fn=(
                partial(compute_dfuse_metrics, metrics=["psnr", "fid", "fvd"])
                if val_dataset is not None
                else None
            ),
            worker_init_fn=VideoDataset.worker_init_fn,
            collate_fn=VideoDataset.collate_fn,
            train_logger="wandb" if args.use_wandb else None,
            batch_sampler=batch_sampler,
            args=args,
        )
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
