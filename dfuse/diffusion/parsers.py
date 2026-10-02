import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_base_path",
        type=str,
        default="",
        required=True,
        help="Base path of the dataset.",
    )
    parser.add_argument(
        "--dataset_metadata_path",
        type=str,
        default=None,
        help="Path to the metadata file of the dataset.",
    )
    parser.add_argument(
        "--dataset_repeat",
        type=str,
        default="1",
        help="Comma-separated repeat counts, one per dataset_base_path. "
        "A single value is broadcast to all paths. E.g. '8,2' repeats "
        "the first dataset 8x and the second 2x.",
    )
    parser.add_argument(
        "--dataset_num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading.",
    )
    parser.add_argument(
        "--data_file_keys",
        type=str,
        default="image,video",
        help="Data file keys in the metadata. Comma-separated.",
    )
    parser.add_argument(
        "--pin_memory",
        default=False,
        action="store_true",
        help="Use pinned memory for DataLoader. Can speed up host-to-device transfers.",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="Limit the number of training batches per epoch. If None, use all batches.",
    )

    parser.add_argument(
        "--val_dataset_base_path",
        type=str,
        default=None,
        help="Base path of the validation dataset. If None, validation is disabled.",
    )
    parser.add_argument(
        "--val_every_n_epochs",
        type=int,
        default=None,
        help="Run validation sampling every N epochs. If None, validation is disabled.",
    )
    parser.add_argument(
        "--val_num_inference_steps",
        type=int,
        default=30,
        help="Number of inference steps for validation sampling.",
    )
    parser.add_argument(
        "--val_cfg_scale",
        type=float,
        default=5.0,
        help="CFG scale for validation sampling.",
    )
    parser.add_argument(
        "--val_before_training",
        default=False,
        action="store_true",
        help="Run a validation epoch before training starts.",
    )
    parser.add_argument(
        "--val_use_ema",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use EMA weights for validation sampling when EMA is enabled. "
             "Use --no-val_use_ema to validate with the online (non-EMA) weights. "
             "Default: True.",
    )
    return parser


def add_image_size_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Maximum number of pixels per frame, used for dynamic resolution.",
    )
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of images. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of images. Leave `height` and `width` empty to enable dynamic resolution.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=1024 * 1024,
        help="Maximum number of pixels per frame, used for dynamic resolution.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames per video. Frames are sampled from the video prefix.",
    )
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model_paths",
        type=str,
        default=None,
        help="Paths to load models. In JSON format.",
    )
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=None,
        help="Model ID with origin paths, e.g., Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors. Comma-separated.",
    )
    parser.add_argument(
        "--extra_inputs", default=None, help="Additional model inputs, comma-separated."
    )
    parser.add_argument(
        "--fp8_models", default=None, help="Models with FP8 precision, comma-separated."
    )
    parser.add_argument(
        "--offload_models",
        default=None,
        help="Models with offload, comma-separated. Only used in splited training.",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype: float32, float16, or bfloat16 (default: bfloat16).",
    )
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training. If provided, training will be resumed from this checkpoint.",
    )
    parser.add_argument(
        "--only_resume_weight",
        default=False,
        action="store_true",
        help="When enabled, only load model weights from checkpoint (at epoch-xx/model.safetensors), without restoring optimizer/scheduler state.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps for linear warmup. Default 0 means no warmup.",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="constant_with_warmup",
        choices=["constant", "constant_with_warmup"],
        help="LR scheduler type: 'constant' for no warmup, 'constant_with_warmup' for linear warmup then constant.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size per device. Requires all samples in a batch to have the same resolution and num_frames.",
    )
    parser.add_argument(
        "--trainable_models",
        type=str,
        default=None,
        help="Semicolon-separated models/patterns to train. "
        "Supports glob wildcards and {a,b} brace expansion. "
        "Examples: 'dit' or 'dit.blocks.*.self_attn.{q,k,v}'",
    )
    parser.add_argument(
        "--find_unused_parameters",
        default=False,
        action="store_true",
        help="Whether to find unused parameters in DDP.",
    )
    parser.add_argument(
        "--static_graph",
        default=False,
        action="store_true",
        help="Whether to use static graph.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay."
    )
    parser.add_argument(
        "--task", type=str, default="sft", required=False, help="Task type."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. If provided, seeds Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode: 'no' for float32, 'fp16' for float16, 'bf16' for bfloat16. "
             "If provided, overrides the accelerate launch config. If None, uses accelerate launch config.",
    )
    parser.add_argument(
        "--use_deepspeed",
        default=False,
        action="store_true",
        help="Enable DeepSpeed for distributed training.",
    )
    parser.add_argument(
        "--deepspeed_stage",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="DeepSpeed ZeRO stage: 1, 2, or 3 (default: 2).",
    )
    parser.add_argument(
        "--deepspeed_offload_optimizer",
        default=False,
        action="store_true",
        help="Offload optimizer states to CPU (for ZeRO stage 2/3).",
    )
    parser.add_argument(
        "--deepspeed_offload_params",
        default=False,
        action="store_true",
        help="Offload parameters to CPU (for ZeRO stage 3).",
    )
    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Path to custom DeepSpeed config JSON file. Overrides other deepspeed options.",
    )
    parser.add_argument(
        "--shard_models",
        type=str,
        default=None,
        help="Semicolon-separated top-level model names to shard with DeepSpeed/FSDP "
             "(e.g. 'dit' or 'dit;text_encoder'). Others stay replicated on each rank. "
             "Default: auto-derived from --trainable_models.",
    )
    parser.add_argument(
        "--use_ema",
        default=False,
        action="store_true",
        help="Enable Exponential Moving Average (EMA) of trainable parameters.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (default: 0.9999). Higher = slower update.",
    )
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--output_path", type=str, default="./models", help="Output save path."
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=None,
        help="Save a checkpoint every N epochs. If None, checkpoints will be saved every epoch.",
    )
    parser.add_argument(
        "--keep_n_checkpoints",
        type=int,
        default=None,
        help="Maximum number of checkpoints to keep. If None, keep all checkpoints.",
    )
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--lora_base_model",
        type=str,
        default=None,
        help="Which model LoRA is added to.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q;k;v;o;ffn.0;ffn.2",
        help="Semicolon-separated glob patterns for LoRA target modules. "
        "Supports * (wildcard) and {a,b} (alternatives). "
        "Examples: 'q;k;v;o' or 'dit.blocks.*.self_attn.{q,k,v}'",
    )
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default=None,
        help="Path to the LoRA checkpoint. If provided, LoRA will be loaded from this checkpoint.",
    )
    parser.add_argument(
        "--preset_lora_path",
        type=str,
        default=None,
        help="Path to the preset LoRA checkpoint. If provided, this LoRA will be fused to the base model.",
    )
    parser.add_argument(
        "--preset_lora_model",
        type=str,
        default=None,
        help="Which model the preset LoRA is fused to.",
    )
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable activation checkpointing (recomputation) to save GPU memory at the cost of extra compute.",
    )
    parser.add_argument(
        "--use_activation_offload",
        default=False,
        action="store_true",
        help="Offload activations to CPU memory. Can be used with or without --use_gradient_checkpointing. "
             "When used alone (without checkpointing), activations are saved to CPU instead of being recomputed, "
             "trading CPU-GPU transfer for compute — useful for attention where compute is O(n^2) but I/O is O(n).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    return parser


def add_wandb_config(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_wandb",
        default=False,
        action="store_true",
        help="Enable Weights & Biases (wandb) logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dfuse-training",
        help="W&B project name (default: dfuse-training).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name. If None, wandb generates one automatically.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team or username). If None, uses the default entity.",
    )
    parser.add_argument(
        "--wandb_log_every_n_steps",
        type=int,
        default=1,
        help="Log training metrics to W&B every N steps (default: 1).",
    )
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_wandb_config(parser)
    return parser
