"""Measure the effect of output resolution and frame rate on Kandinsky S2V Lite."""

import argparse
import csv
import os
import time
from itertools import product
from pathlib import Path

VIDEO_SECONDS = 5
RESOLUTIONS = (512, 448, 384)
FRAME_RATES = (25, 21, 17)
CSV_FIELDS = (
    "gpus", "width", "height", "fps", "frames", "steps", "generation_seconds",
    "seconds_per_video_second", "peak_allocated_gib_per_gpu", "video",
)


def video_configuration(base, size, fps, temporal_stride):
    frames = VIDEO_SECONDS * fps
    latent_frames = (frames - 1) // temporal_stride + 1
    assert (latent_frames - 1) * temporal_stride + 1 == frames

    # Preserve the near history and shorten only the oldest, most compressed group.
    recent, middle, oldest = base.dit.frame_buckets
    oldest_stride = base.dit.patch_proj4x_coef[0]
    max_oldest = (latent_frames - recent - middle) // oldest_stride * oldest_stride

    data = base.model_dump()
    data["generation"].update(
        fps=fps, latent_frames=latent_frames, allowed_resolutions=[(size, size)]
    )
    data["dit"]["frame_buckets"] = (recent, middle, min(oldest, max_oldest))
    return type(base).model_validate(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("config", "image", "audio", "out-dir"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--attention-engine", choices=("flash_attention_3", "sdpa"), default="flash_attention_3")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    import torch
    import torch.distributed as dist
    from kandinsky_s2v.pipeline.config import load_config
    from kandinsky_s2v.pipeline.factory import build_pipeline
    from kandinsky_s2v.pipeline.pipeline import TEMPORAL_COMPRESSION_RATIO

    rank = int(os.environ.get("RANK", "0"))
    gpus = int(os.environ.get("WORLD_SIZE", "1"))
    base = load_config(args.config)
    summary = args.out_dir / "summary.csv"

    def synchronize():
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

    try:
        started = time.perf_counter()
        pipeline = build_pipeline(base, device="cuda", attention_engine=args.attention_engine)
        synchronize()
        if rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with summary.open("x", encoding="utf-8", newline="") as file:
                csv.writer(file).writerow(CSV_FIELDS)
            print(f"Model setup: {time.perf_counter() - started:.2f}s on {gpus} GPU(s)", flush=True)

        inputs = dict(audio=args.audio, image=args.image, chunks=1, seed=42, num_steps=args.steps)
        for size, fps in product(RESOLUTIONS, FRAME_RATES):
            config = video_configuration(base, size, fps, TEMPORAL_COMPRESSION_RATIO)
            pipeline.cfg = config
            pipeline.dit.frame_buckets = config.dit.frame_buckets
            if rank == 0:
                print(f"Warm-up: {size}x{size} at {fps} FPS", flush=True)
            pipeline(**inputs)
            synchronize()
            torch.cuda.reset_peak_memory_stats()

            path = args.out_dir / f"{size}px-{fps}fps-{args.steps}steps.mp4"
            started = time.perf_counter()
            result = pipeline(**inputs, save_path=path, show_progress=True)
            synchronize()
            elapsed = time.perf_counter() - started

            peak_bytes = torch.tensor(torch.cuda.max_memory_allocated(), device=pipeline.device, dtype=torch.int64)
            if dist.is_initialized():
                dist.reduce(peak_bytes, dst=0, op=dist.ReduceOp.MAX)
            if rank == 0:
                frames = VIDEO_SECONDS * fps
                if result.path is None or not path.is_file() or result.frames.shape[1:] != (frames, size, size):
                    raise RuntimeError(f"Expected {frames} frames at {size}x{size} in {path}")
                rate = elapsed / VIDEO_SECONDS
                peak_gib = peak_bytes.item() / 1024**3
                with summary.open("a", encoding="utf-8", newline="") as file:
                    csv.writer(file).writerow((
                        gpus, size, size, fps, frames, args.steps, f"{elapsed:.3f}",
                        f"{rate:.3f}", f"{peak_gib:.2f}", path.name,
                    ))
                print(f"{size}x{size} {fps} FPS: {elapsed:.2f}s, {rate:.3f} s/s, {peak_gib:.1f} GiB peak GPU", flush=True)
            del result
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
