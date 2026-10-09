import itertools
from pathlib import Path

import pandas as pd

from .faults import INTENSITIES
from .schema import read_json, write_json


WORKLOADS = ("resnet50", "mobilenet_v3_large", "vit_b_16", "bert_base")
PAIRS = (
    ("cpu", "network"),
    ("dataloader", "gpu_compute"),
    ("storage", "rank_compute"),
    ("network", "gpu_memory"),
    ("cpu", "storage"),
    ("gpu_compute", "rank_compute"),
)


def make_plan(output):
    plan = []
    batches = {"resnet50": 32, "mobilenet_v3_large": 32, "vit_b_16": 16, "bert_base": 8}
    for workload, ranks in itertools.product(WORKLOADS, (1, 2, 4)):
        common = {
            "workload": workload,
            "world_size": ranks,
            "batch_size": batches[workload],
            "fault_start_step": 60,
            "fault_steps": 120,
            "steps": 200,
        }
        for repetition in range(40):
            split = (
                "calibration"
                if repetition < 20
                else "train"
                if repetition < 32
                else "validation"
                if repetition < 36
                else "test"
            )
            plan.append(
                {
                    **common,
                    "id": f"{workload}-r{ranks}-healthy-{repetition}",
                    "split": split,
                    "faults": {},
                    "seed": repetition,
                }
            )
        for component, intensities in INTENSITIES.items():
            for level, intensity in enumerate(intensities):
                for repetition in range(8):
                    split = (
                        "train"
                        if repetition < 5
                        else "validation"
                        if repetition == 5
                        else "test"
                    )
                    plan.append(
                        {
                            **common,
                            "id": f"{workload}-r{ranks}-{component}-l{level}-{repetition}",
                            "split": split,
                            "faults": {component: intensity},
                            "intensity_level": level,
                            "seed": repetition,
                        }
                    )
        for a, b in PAIRS:
            for level in range(3):
                for repetition in range(5):
                    plan.append(
                        {
                            **common,
                            "id": f"{workload}-r{ranks}-{a}+{b}-l{level}-{repetition}",
                            "split": "test",
                            "faults": {
                                a: INTENSITIES[a][level],
                                b: INTENSITIES[b][level],
                            },
                            "intensity_level": level,
                            "seed": repetition,
                        }
                    )
    write_json(output, plan)
    return {
        "healthy": 480,
        "single_fault": 2016,
        "dual_fault": 1080,
        "total": len(plan),
    }


def merge_recordings(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(directory.glob("*/metadata-rank0.json")):
        metadata = read_json(path)
        if not metadata.get("completed", False):
            raise ValueError(f"Execution did not complete: {path.parent.name}")
        run_dir = path.parent
        ranks = list(run_dir.glob("metadata-rank*.json"))
        if len(ranks) != metadata["world_size"]:
            raise ValueError(f"Incomplete rank metadata: {run_dir.name}")
        if any(not read_json(rank).get("completed", False) for rank in ranks):
            raise ValueError(f"An execution rank did not complete: {run_dir.name}")
        frames = [
            pd.read_csv(file) for file in sorted(run_dir.glob("telemetry-rank*.csv"))
        ]
        if len(frames) != metadata["world_size"]:
            raise ValueError(f"Incomplete rank telemetry: {run_dir.name}")
        frame = pd.concat(frames, ignore_index=True)
        frame["bin"] = (frame.timestamp / 0.2).round().astype(int)
        minimum = {"gpu_clock", "loader_queue", "rank_progress", "disk_read"}
        rows = []
        for (time_bin, metric), group in frame.groupby(["bin", "metric"], sort=True):
            value = group.value.min() if metric in minimum else group.value.max()
            rows.append({"timestamp": time_bin * 0.2, "metric": metric, "value": value})
        telemetry = output / f"{run_dir.name}-telemetry.csv"
        pd.DataFrame(rows).to_csv(telemetry, index=False)
        step_files = sorted(run_dir.glob("steps-rank*.csv"))
        if len(step_files) != metadata["world_size"]:
            raise ValueError(f"Incomplete rank step timings: {run_dir.name}")
        steps = pd.concat(
            [pd.read_csv(file) for file in step_files],
            ignore_index=True,
        )
        steps = steps.groupby("step", as_index=False).agg(
            timestamp=("timestamp", "max"), step_ms=("step_ms", "max")
        )
        steps_path = output / f"{run_dir.name}-steps.csv"
        steps.to_csv(steps_path, index=False)
        roots = list(metadata["faults"])
        fault_metadata = read_json(
            run_dir / f"metadata-rank{metadata['fault_rank']}.json"
        )
        metadata["fault_start"] = fault_metadata["fault_start"]
        metadata["fault_end"] = fault_metadata["fault_end"]
        if roots and (metadata["fault_start"] is None or metadata["fault_end"] is None):
            raise ValueError(f"Incomplete fault interval: {run_dir.name}")
        manifest.append(
            {
                "id": run_dir.name,
                "workload": metadata["workload"],
                "batch_size": metadata["batch_size"],
                "world_size": metadata["world_size"],
                "split": metadata["split"],
                "roots": roots,
                "faults": metadata["faults"],
                "intensity_level": metadata.get("intensity_level"),
                "execution_seed": metadata["seed"],
                "fault_start": metadata["fault_start"],
                "fault_end": metadata["fault_end"],
                "telemetry": telemetry.name,
                "steps": steps_path.name,
            }
        )
    if not manifest:
        raise ValueError("No complete recordings found")
    write_json(output / "manifest.json", manifest)
    return len(manifest)
