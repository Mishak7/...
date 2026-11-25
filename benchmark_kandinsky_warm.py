"""Benchmark repeated Kandinsky S2V renders in one torchrun process group."""

import argparse
import os
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[16, 8])
    parser.add_argument("--attention-engine", choices=("sdpa", "flash_attention_3"), default="sdpa")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--compile", action="store_true", help="Compile visual blocks during the warm-up")
    args = parser.parse_args()

    if args.repeats < 1 or any(steps < 1 for steps in args.steps):
        parser.error("Steps and repeats must be positive")
    for path in (args.config, args.image, args.audio):
        if not path.is_file():
            parser.error(f"Missing input: {path}")

    import torch
    import torch.distributed as dist
    from kandinsky_s2v.pipeline.config import load_config
    from kandinsky_s2v.pipeline.factory import build_pipeline

    rank = int(os.environ.get("RANK", "0"))
    gpus = int(os.environ.get("WORLD_SIZE", "1"))
    config = load_config(args.config)
    if args.compile:
        compile_config = config.compile.model_copy(update={"strategy": "torch"})
        config = config.model_copy(update={"compile": compile_config})

    def synchronize():
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])

    try:
        started = time.perf_counter()
        pipe = build_pipeline(config, device="cuda", attention_engine=args.attention_engine)
        synchronize()
        if rank == 0:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            print(f"Model setup: {time.perf_counter() - started:.2f}s on {gpus} GPU(s)", flush=True)

        def render(steps, name):
            synchronize()
            started = time.perf_counter()
            path = args.out_dir / f"{name}.mp4"
            result = pipe(audio=str(args.audio), image=str(args.image), chunks=1, seed=42,
                          num_steps=steps, save_path=str(path), show_progress=True)
            synchronize()
            elapsed = time.perf_counter() - started
            if rank == 0:
                if result.path is None or not path.is_file():
                    raise RuntimeError(f"No output video written to {path}")
                print(f"{name}: {elapsed:.2f}s -> {path}", flush=True)
            return elapsed

        render(config.generation.num_steps, "warmup")
        for steps in args.steps:
            durations = [render(steps, f"steps-{steps}-run-{index + 1}") for index in range(args.repeats)]
            if rank == 0:
                print(f"{steps} steps: median {statistics.median(durations):.2f}s, "
                      f"best {min(durations):.2f}s ({len(durations)} measured runs)", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
