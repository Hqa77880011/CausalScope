import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from .calibration import HealthyCalibration
from .data import WindowDataset
from .model import LocalizationModel
from .schema import write_json
from .scoring import rank_components
from .training import model_from_checkpoint


def benchmark(
    prepared,
    checkpoint_path,
    output,
    sizes,
    repeats=100,
    warmup=10,
    device="cpu",
    pairs=None,
):
    if repeats < 1 or warmup < 0:
        raise ValueError("Repeats must be positive and warmup nonnegative")
    dataset = WindowDataset(Path(prepared) / "test.npz")
    base_model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    del base_model
    if checkpoint["method"] != "causalscope":
        raise ValueError("Node-scaling benchmark requires a CausalScope checkpoint")
    base_count = len(checkpoint["schema"]["metrics"])
    base_x = dataset.arrays["x"][:1]
    key = dataset.metadata[0]["domain"]
    measurements = []
    process = psutil.Process()
    for size in sizes:
        if size < base_count:
            raise ValueError(
                f"Node count must be at least the checkpoint size ({base_count})"
            )
        schema = copy.deepcopy(checkpoint["schema"])
        graph = copy.deepcopy(checkpoint["graph"])
        calibration = copy.deepcopy(checkpoint["calibration"])
        for index in range(base_count, size):
            source = index % base_count
            metric = copy.deepcopy(schema["metrics"][source])
            metric["name"] = f"{metric['name']}_scale_{index}"
            schema["metrics"].append(metric)
            graph["nodes"].append(metric["name"])
            for edge in checkpoint["graph"]["edges"]:
                if edge["destination"] == source:
                    new_edge = dict(edge)
                    new_edge["destination"] = index
                    graph["edges"].append(new_edge)
            for baseline in calibration["domains"].values():
                baseline["median"].append(baseline["median"][source])
                baseline["scale"].append(baseline["scale"][source])
                baseline["regressions"].append(
                    copy.deepcopy(baseline["regressions"][source])
                )
        calibration["schema"] = schema
        model = LocalizationModel(
            schema,
            graph,
            len(checkpoint["domains"]),
            checkpoint["config"]["model"],
            "causalscope",
        ).to(device)
        weights = {
            name: value
            for name, value in checkpoint["state_dict"].items()
            if name != "kind_ids"
        }
        model.load_state_dict(weights, strict=False)
        model.eval()
        tensor = torch.from_numpy(base_x[:, np.arange(size) % base_count]).to(device)
        times = []
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for iteration in range(warmup + repeats):
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = perf_counter()
                rank_components(
                    model,
                    tensor,
                    [key],
                    HealthyCalibration(calibration),
                    checkpoint["config"]["ranking"],
                    checkpoint["temperature"],
                )
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = (perf_counter() - start) * 1000
                if iteration >= warmup:
                    times.append(elapsed)
        measurements.append(
            {
                "nodes": size,
                "latency_ms_mean": float(np.mean(times)),
                "latency_ms_p50": float(np.quantile(times, 0.5)),
                "latency_ms_p95": float(np.quantile(times, 0.95)),
                "rss_mb": process.memory_info().rss / 1e6,
                "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1e6
                if str(device).startswith("cuda")
                else None,
            }
        )
        del model, tensor
    report = {
        "device": device,
        "repeats": repeats,
        "warmup": warmup,
        "inference": measurements,
        "node_scaling": "Additional nodes duplicate measured channels; this measures computational cost.",
    }
    if pairs:
        import pandas as pd

        table = pd.read_csv(pairs)
        rows = []
        for row in table.to_dict("records"):
            if row["without_monitor_steps_per_second"] <= 0:
                raise ValueError("Reference throughput must be positive")
            rows.append(
                {
                    **row,
                    "throughput_loss_percent": 100
                    * (
                        1
                        - row["with_monitor_steps_per_second"]
                        / row["without_monitor_steps_per_second"]
                    ),
                }
            )
        report["collection"] = rows
    write_json(output, report)
    return report
