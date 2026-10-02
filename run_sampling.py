"""Generate a novel-view video with the shared D-FUSE pipeline."""
import argparse
from pathlib import Path

import torch

from dfuse.pipelines.dfuse import DFusePipeline
from dfuse.utils.data import load_sampling_inputs, save_video


def sampling_parser():
    """Define checkpoint, input, and sampling options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, help='Directory containing cameras.json, prompt.txt, and source MP4s.')
    parser.add_argument('--dit_path', required=True, help='Complete exported D-FUSE safetensors checkpoint.')
    parser.add_argument('--vae_path', required=True)
    parser.add_argument('--t5_path', required=True)
    parser.add_argument('--tokenizer_dir', required=True)
    parser.add_argument('--output_mp4', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--torch_dtype', default='bfloat16', choices=['float32', 'float16', 'bfloat16'])
    parser.add_argument('--num_inference_steps', type=int, default=20)
    parser.add_argument('--sigma_shift', type=float, default=5.0)
    parser.add_argument('--cfg_scale', type=float, default=1.0)
    parser.add_argument('--negative_prompt', default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--modulation_type', default='learned', choices=['sinc', 'gaussian', 'learned'])
    parser.add_argument('--enable_position_phase', action='store_true')
    parser.add_argument('--num_modalities', type=int, default=2, choices=[2])
    parser.add_argument('--cpu_offload', action=argparse.BooleanOptionalAction, default=True,
                        help='Keep inactive models on CPU between pipeline stages.')
    parser.add_argument('--tiled', action='store_true', help='Use tiled VAE encoding and decoding.')
    return parser


def main(argv=None):
    args = sampling_parser().parse_args(argv)
    if args.num_inference_steps < 1 or args.sigma_shift <= 0:
        raise ValueError('Inference steps and sigma shift must be positive.')
    inputs = load_sampling_inputs(args.data_dir)
    pipeline = DFusePipeline.from_checkpoint_files(
        dit_path=args.dit_path, vae_path=args.vae_path,
        text_encoder_path=args.t5_path, tokenizer_path=args.tokenizer_dir,
        modulation_type=args.modulation_type, num_modalities=args.num_modalities,
        enable_position_phase=args.enable_position_phase,
        device=args.device, torch_dtype=getattr(torch, args.torch_dtype),
        cpu_offload=args.cpu_offload,
    )
    videos = pipeline(
        **inputs, negative_prompt=args.negative_prompt,
        seed=args.seed, rand_device='cpu', cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift, tiled=args.tiled,
    )
    output = Path(args.output_mp4)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_video(videos[0], str(output), fps=inputs['target_fps'])
    print(f'Wrote {output} ({len(videos[0])} frames at {inputs["target_fps"]} fps)')


if __name__ == '__main__':
    main()
