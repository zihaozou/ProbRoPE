---
library_name: dfuse
tags:
  - novel-view-synthesis
  - video
  - multimodal
  - probrope
---

# D-FUSE

D-FUSE synthesizes novel views from asynchronous RGB and event videos using ProbRoPE.

[Project website](https://zihaozou.github.io/ProbRoPE/) · [Code](https://github.com/zihaozou/ProbRoPE)

## Installation

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/zihaozou/D-FUSE
cd D-FUSE
pip install .
```

## Generate a video

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
videos = pipe(**inputs, seed=0, num_inference_steps=20, cfg_scale=1.0)
save_video(videos[0], "output.mp4", fps=inputs["target_fps"])
```

This loads the complete video-generation pipeline: D-FUSE with learned ProbRoPE
from the epoch-40 checkpoint, the Wan VAE, UMT5 text encoder, and tokenizer.
Required weights are downloaded and cached automatically. CPU offloading is
enabled by default.

## Input

```text
input/
├── rgb_source.mp4
├── event_source.mp4
├── cameras.json
└── prompt.txt
```
