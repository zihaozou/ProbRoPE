"""Peak Signal-to-Noise Ratio (PSNR) computation."""
import numpy as np
from PIL import Image


def compute_psnr(img_gen: Image.Image, img_gt: Image.Image) -> float:
    """Compute PSNR between two PIL images.

    If `img_gen` and `img_gt` differ in size, `img_gen` is bilinearly resized
    to match `img_gt` before the comparison. PSNR is evaluated at the GT
    resolution because GT is the reference; downstream metrics (FID/FVD/CLIP)
    do their own resizing so they don't hit this case.

    Args:
        img_gen: Generated image.
        img_gt: Ground truth image.

    Returns:
        PSNR in dB. Returns float('inf') if images are identical.
    """
    if img_gen.size != img_gt.size:
        img_gen = img_gen.resize(img_gt.size, Image.BILINEAR)
    gen = np.array(img_gen, dtype=np.float64)
    gt = np.array(img_gt, dtype=np.float64)
    mse = np.mean((gen - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def compute_psnr_from_videos(
    generated_video: str,
    reference_video: str,
    num_frames: int,
) -> float:
    """Compute mean PSNR between two videos, frame-by-frame.

    Decodes frames from both mp4s via torchcodec and computes per-frame
    PSNR, returning the average.

    Args:
        generated_video: Path to generated mp4 file.
        reference_video: Path to reference/GT mp4 file.
        num_frames: Number of frames to compare from the start.

    Returns:
        Mean PSNR in dB across all frames.
    """
    from torchcodec.decoders import VideoDecoder

    gen_dec = VideoDecoder(generated_video)
    ref_dec = VideoDecoder(reference_video)

    psnr_values = []
    for fi in range(num_frames):
        gen_frame = gen_dec[fi]
        ref_frame = ref_dec[fi]
        gen_pil = Image.fromarray(gen_frame.permute(1, 2, 0).numpy(), mode="RGB")
        ref_pil = Image.fromarray(ref_frame.permute(1, 2, 0).numpy(), mode="RGB")
        psnr_values.append(compute_psnr(gen_pil, ref_pil))

    del gen_dec, ref_dec
    if not psnr_values:
        return 0.0
    return float(np.mean(psnr_values))
