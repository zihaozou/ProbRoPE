# ProbRoPE

**Asynchronous Multimodal View Synthesis with Probabilistic Temporal Encoding**  
SIGGRAPH Asia 2026

[Project Website](https://zihaozou.github.io/ProbRoPE/) · [Paper](https://zihaozou.github.io/ProbRoPE/assets/siggraph-camauthor.pdf) · [Model](https://huggingface.co/zihaozou/D-FUSE) · [Dataset](https://huggingface.co/datasets/zihaozou/ProbRoPE) · [Fusion Studio](fusion-studio)

D-FUSE synthesizes novel views from asynchronous RGB and event videos using ProbRoPE, a probabilistic temporal encoding.

## Installation

Requires Python 3.10–3.13, a CUDA-capable GPU, and FFmpeg with shared libraries.

```bash
git clone https://github.com/zihaozou/ProbRoPE.git
cd ProbRoPE
pip install .
```

For distributed training and experiment logging:

```bash
pip install '.[distributed,logging]'
```

## Checkpoint

The pretrained D-FUSE model is available on [Hugging Face](https://huggingface.co/zihaozou/D-FUSE).

[Download dfuse.safetensors](https://huggingface.co/zihaozou/D-FUSE/resolve/main/dfuse.safetensors)

## Sampling

The pipeline automatically downloads and loads D-FUSE, the Wan VAE, UMT5 text encoder, and tokenizer.

```python
import torch
from dfuse import DFusePipeline
from dfuse.utils.data import load_sampling_inputs, save_video

pipe = DFusePipeline.from_pretrained(
    "zihaozou/D-FUSE",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

inputs = load_sampling_inputs("path/to/input")
videos = pipe(
    **inputs,
    seed=0,
    num_inference_steps=20,
    cfg_scale=1.0,
)
save_video(videos[0], "output.mp4", fps=inputs["target_fps"])
```

Prepare each input sample as:

```text
input/
├── rgb_source.mp4
├── event_source.mp4
├── cameras.json
└── prompt.txt
```

CPU offloading is enabled by default. Set `cpu_offload=False` when loading the pipeline to keep all models on the GPU.

## Dataset

The dataset is available on [Hugging Face](https://huggingface.co/datasets/zihaozou/ProbRoPE).

## Training

Set the model and dataset paths in `train_config.json`, then run:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    run_training.py --config train_config.json
```

For multi-GPU training with DeepSpeed ZeRO-3:

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
    run_training.py --config train_config.json \
    --use_deepspeed --deepspeed_config_file deepspeed_config.json
```

Resume training:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 \
    run_training.py --config train_config.json \
    --resume_checkpoint outputs/dfuse/epoch-40
```

## Acknowledgements

Our implementation builds on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and [Wan2.1](https://github.com/Wan-Video/Wan2.1). Third-party license notices are included in `LICENSE`.
