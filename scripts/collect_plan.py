import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Collect selected entries from a CausalScope execution plan"
    )
    parser.add_argument("--plan", required=True)
    parser.add_argument("--vision", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--interface")
    parser.add_argument("--cgroup")
    parser.add_argument("--block-device")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=30)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    stop = len(plan) if args.stop is None else args.stop
    if not 0 <= args.start < stop <= len(plan):
        raise ValueError("Select a valid nonempty plan interval")
    for index in range(args.start, stop):
        recipe = plan[index]
        if recipe["world_size"] % args.nnodes:
            raise ValueError(
                f"Execution {index} cannot divide {recipe['world_size']} ranks over {args.nnodes} nodes"
            )
        metadata = Path(args.output) / recipe["id"] / "metadata-rank0.json"
        if args.node_rank == 0 and metadata.exists():
            raise FileExistsError(
                f"Execution {recipe['id']} already exists; select a later --start"
            )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={args.nnodes}",
            f"--node_rank={args.node_rank}",
            f"--nproc_per_node={recipe['world_size'] // args.nnodes}",
            f"--master_addr={args.master_addr}",
            f"--master_port={args.master_port}",
            "-m",
            "causalscope.cli",
            "collect",
            "--plan",
            args.plan,
            "--index",
            str(index),
            "--dataset",
            args.text if recipe["workload"] == "bert_base" else args.vision,
            "--output",
            args.output,
            "--workers",
            str(args.workers),
            "--segment-seconds",
            str(args.segment_seconds),
        ]
        for name in ("interface", "cgroup", "block_device"):
            value = getattr(args, name)
            if value is not None:
                command.extend([f"--{name.replace('_', '-')}", value])
        print(f"[{index + 1}/{stop}] {recipe['id']}", flush=True)
        subprocess.run(command, check=True, env={**os.environ, "OMP_NUM_THREADS": "1"})


if __name__ == "__main__":
    main()
