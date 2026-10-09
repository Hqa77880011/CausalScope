import argparse
import os
import subprocess
import sys
from pathlib import Path

from causalscope.overhead import collection_cost


def main():
    parser = argparse.ArgumentParser(
        description="Measure paired healthy jobs with and without telemetry sampling"
    )
    parser.add_argument(
        "--workload",
        choices=["resnet50", "mobilenet_v3_large", "vit_b_16", "bert_base"],
        required=True,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--interface")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.ranks, args.batch_size, args.steps, args.repeats) < 1:
        raise ValueError("Ranks, batch size, steps, and repeats must be positive")
    output = Path(args.output)
    for seed in range(args.repeats):
        run_id = f"{args.workload}-r{args.ranks}-seed{seed}"
        modes = (
            ("reference", "monitored") if seed % 2 == 0 else ("monitored", "reference")
        )
        for mode in modes:
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.ranks}",
                "-m",
                "causalscope.cli",
                "collect",
                "--workload",
                args.workload,
                "--dataset",
                args.dataset,
                "--output",
                str(output / mode),
                "--run-id",
                run_id,
                "--split",
                "test",
                "--seed",
                str(seed),
                "--batch-size",
                str(args.batch_size),
                "--steps",
                str(args.steps),
                "--workers",
                str(args.workers),
                "--segment-seconds",
                "0",
            ]
            if mode == "reference":
                command.append("--no-monitor")
            if args.interface:
                command.extend(["--interface", args.interface])
            if args.cpu:
                command.append("--cpu")
            print(f"pair={seed} mode={mode}", flush=True)
            subprocess.run(
                command, check=True, env={**os.environ, "OMP_NUM_THREADS": "1"}
            )
    collection_cost(
        output / "reference", output / "monitored", output / "collection_cost.json"
    )


if __name__ == "__main__":
    main()
