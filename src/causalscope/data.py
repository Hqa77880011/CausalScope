from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import domain_key, read_json, write_json


def read_manifest(path, schema):
    path = Path(path).resolve()
    runs = read_json(path)
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must be a nonempty JSON list")
    ids = set()
    recordings = set()
    for source in runs:
        if source["id"] in ids:
            raise ValueError(f"Duplicate execution id: {source['id']}")
        ids.add(source["id"])
        if source["split"] not in ("calibration", "train", "validation", "test"):
            raise ValueError(f"Invalid execution split: {source['split']}")
        if source["world_size"] < 1 or source["batch_size"] < 1:
            raise ValueError("Batch size and rank count must be positive")
        if not set(source["roots"]).issubset(schema["components"]):
            raise ValueError(f"Unknown roots in {source['id']}")
        if len(source["roots"]) > 2 or len(set(source["roots"])) != len(
            source["roots"]
        ):
            raise ValueError("An execution must have zero, one, or two distinct roots")
        if source["split"] == "calibration" and source["roots"]:
            raise ValueError("Healthy calibration executions cannot contain faults")
        if source["roots"] and not 0 <= source["fault_start"] < source["fault_end"]:
            raise ValueError("Fault executions need valid start and end times")
        for field in ("telemetry", "steps"):
            source[field] = str((path.parent / source[field]).resolve())
        recording = (source["telemetry"], source["steps"])
        if recording in recordings:
            raise ValueError(
                "An execution recording cannot be reused under another id or partition"
            )
        recordings.add(recording)
    return runs


def align_run(run, schema, config):
    frame = pd.read_csv(run["telemetry"])
    steps = pd.read_csv(run["steps"])
    if not {"timestamp", "metric", "value"}.issubset(frame.columns):
        raise ValueError("Telemetry needs timestamp, metric, value columns")
    if not {"timestamp", schema["target"]}.issubset(steps.columns):
        raise ValueError("Steps need timestamp and the schema target")
    if frame.empty or steps.empty:
        raise ValueError(f"Empty recording: {run['id']}")
    frame = frame.sort_values("timestamp", kind="stable")
    steps = steps.sort_values("timestamp", kind="stable")
    if not np.isfinite(frame.timestamp).all() or not np.isfinite(steps.timestamp).all():
        raise ValueError(
            "Timestamps must be finite seconds relative to execution start"
        )
    dt = config["sample_seconds"]
    end = min(float(frame.timestamp.max()), float(steps.timestamp.max()))
    grid = np.arange(0, end + dt / 2, dt)
    names = [metric["name"] for metric in schema["metrics"]]
    values = np.full((len(grid), len(names)), np.nan, dtype=np.float32)
    missing = np.ones_like(values)
    age = np.full_like(values, config["window_seconds"])
    for i, name in enumerate(names):
        observed = frame.loc[(frame.metric == name) & np.isfinite(frame.value)]
        if observed.empty:
            continue
        times = observed.timestamp.to_numpy()
        indices = np.searchsorted(times, grid, side="right") - 1
        valid = indices >= 0
        selected = indices[valid]
        values[valid, i] = observed.value.to_numpy()[selected]
        age[valid, i] = grid[valid] - times[selected]
        missing[valid, i] = (
            age[valid, i] > config["max_forward_fill"] * dt + 1e-6
        ).astype(float)
    times = steps.timestamp.to_numpy()
    indices = np.searchsorted(times, grid, side="right") - 1
    target = np.full(len(grid), np.nan, dtype=np.float32)
    valid = indices >= 0
    target[valid] = steps[schema["target"]].to_numpy()[indices[valid]]
    return {
        "time": grid,
        "values": values,
        "missing": missing,
        "age": age,
        "target": target,
    }


class WindowDataset(Dataset):
    def __init__(self, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as arrays:
            self.arrays = {key: arrays[key] for key in arrays.files}
        self.metadata = read_json(path.with_suffix(".json"))

    def __len__(self):
        return len(self.arrays["x"])

    def __getitem__(self, index):
        return {
            key: torch.as_tensor(self.arrays[key][index])
            for key in ("x", "target", "roots", "fault", "domain")
        }


def prepare(manifest, schema, config, output, held_out=None):
    from .calibration import HealthyCalibration
    from .graph import build_graph

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs = read_manifest(manifest, schema)
    if held_out and held_out not in {run["workload"] for run in runs}:
        raise ValueError(f"Held-out workload absent: {held_out}")
    aligned = {run["id"]: align_run(run, schema, config["data"]) for run in runs}
    calibration_runs = [run for run in runs if run["split"] == "calibration"]
    calibration = HealthyCalibration.fit(
        calibration_runs, aligned, schema, config["data"]
    )
    calibration.save(output / "calibration.json")
    domains = sorted(
        {
            domain_key(run)
            for run in runs
            if run["split"] == "train" and run["workload"] != held_out
        }
    )
    if not domains:
        raise ValueError("No training domains")
    write_json(
        output / "metadata.json",
        {"schema": schema, "config": config, "domains": domains, "held_out": held_out},
    )
    healthy_windows = []
    buckets = {name: [] for name in ("train", "validation", "test")}
    length = round(config["data"]["window_seconds"] / config["data"]["sample_seconds"])
    stride = round(config["data"]["stride_seconds"] / config["data"]["sample_seconds"])
    if length < 4 or stride < 1:
        raise ValueError("Window needs at least four samples and stride at least one")
    for run in runs:
        record = aligned[run["id"]]
        key = domain_key(run)
        x = calibration.transform(key, record)
        split = run["split"]
        if held_out and run["workload"] == held_out and split != "calibration":
            split = "test"
        elif held_out and len(run["roots"]) > 1:
            split = "test"
        for end in range(length, len(x) + 1, stride):
            start = end - length
            window = x[start:end].transpose(1, 0, 2)
            if split == "calibration":
                if run["workload"] != held_out:
                    healthy_windows.append(window)
                continue
            roots = np.zeros(len(schema["components"]), dtype=np.float32)
            window_start = float(record["time"][start])
            window_end = float(record["time"][end - 1])
            if run["roots"]:
                if window_end >= run["fault_end"]:
                    continue
                if window_end >= run["fault_start"]:
                    for root in run["roots"]:
                        roots[schema["components"].index(root)] = 1
            valid_target = record["target"][start:end]
            valid_target = valid_target[np.isfinite(valid_target)]
            if not len(valid_target):
                continue
            target = float(calibration.normalize_target(key, np.mean(valid_target)))
            buckets[split].append(
                {
                    "x": window,
                    "target": target,
                    "roots": roots,
                    "fault": float(roots.any()),
                    "domain": domains.index(key) if key in domains else -1,
                    "meta": {
                        "run_id": run["id"],
                        "workload": run["workload"],
                        "domain": key,
                        "start": window_start,
                        "end": window_end,
                        "fault_start": run.get("fault_start"),
                        "fault_end": run.get("fault_end"),
                        "roots": run["roots"] if roots.any() else [],
                    },
                }
            )
    if not healthy_windows:
        raise ValueError("No visible healthy calibration window is long enough")
    graph = build_graph(np.stack(healthy_windows), schema, config["graph"])
    write_json(output / "graph.json", graph)
    for split, rows in buckets.items():
        if not rows:
            raise ValueError(
                f"No {split} windows; check durations and execution splits"
            )
        if split != "test" and any(row["domain"] < 0 for row in rows):
            raise ValueError("Every validation domain must also appear in training")
        arrays = {
            key: np.asarray(
                [row[key] for row in rows],
                dtype=np.float32 if key != "domain" else np.int64,
            )
            for key in ("x", "target", "roots", "fault", "domain")
        }
        np.savez_compressed(output / f"{split}.npz", **arrays)
        write_json(output / f"{split}.json", [row["meta"] for row in rows])
    return {split: len(rows) for split, rows in buckets.items()}
