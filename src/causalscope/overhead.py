import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from .calibration import HealthyCalibration
from .data import WindowDataset
from .model import LocalizationModel
from .schema import read_json, write_json
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


def collection_cost(reference_directory, monitored_directory, output):
    import pandas as pd

    reference_directory, monitored_directory = (
        Path(reference_directory),
        Path(monitored_directory),
    )
    reference_runs = {
        path.parent.name: path.parent
        for path in reference_directory.glob("*/metadata-rank0.json")
    }
    monitored_runs = {
        path.parent.name: path.parent
        for path in monitored_directory.glob("*/metadata-rank0.json")
    }
    if not reference_runs or set(reference_runs) != set(monitored_runs):
        raise ValueError(
            "Reference and monitored directories need the same complete execution IDs"
        )
    pairs = []
    for run_id in sorted(reference_runs):
        reference = [
            read_json(path)
            for path in sorted(reference_runs[run_id].glob("metadata-rank*.json"))
        ]
        monitored = [
            read_json(path)
            for path in sorted(monitored_runs[run_id].glob("metadata-rank*.json"))
        ]
        ranks = reference[0]["world_size"]
        if (
            len(reference) != ranks
            or len(monitored) != ranks
            or any(not row.get("completed") for row in [*reference, *monitored])
        ):
            raise ValueError(f"Incomplete paired execution: {run_id}")
        expected = set(range(ranks))
        if {row["rank"] for row in reference} != expected or {
            row["rank"] for row in monitored
        } != expected:
            raise ValueError(f"Incomplete rank IDs in {run_id}")
        for rank in range(ranks):
            before = next(row for row in reference if row["rank"] == rank)
            after = next(row for row in monitored if row["rank"] == rank)
            if (
                before["monitored"]
                or not after["monitored"]
                or before["faults"]
                or after["faults"]
            ):
                raise ValueError(
                    "Collection-cost pairs must be healthy and differ in monitoring mode"
                )
            for field in (
                "workload",
                "world_size",
                "batch_size",
                "seed",
                "steps",
                "actual_steps",
                "num_workers",
                "segment_seconds",
                "host",
                "phase_mode",
                "dataset",
                "torch",
                "cuda",
            ):
                if before[field] != after[field]:
                    raise ValueError(
                        f"Paired field {field} differs for {run_id}, rank {rank}"
                    )
        throughput_before = min(row["steps_per_second"] for row in reference)
        throughput_after = min(row["steps_per_second"] for row in monitored)
        if throughput_before <= 0:
            raise ValueError("Reference step throughput must be positive")
        hosts = sorted({row["host"] for row in monitored})
        per_host = []
        for host in hosts:
            host_rows = [row for row in monitored if row["host"] == host]
            capacity = host_rows[0]["logical_cpus"]
            if not capacity or any(
                row["logical_cpus"] != capacity for row in host_rows
            ):
                raise ValueError(f"Invalid host CPU capacity for {host}")
            cpu_one_core = sum(
                row["overhead"]["monitor_cpu_percent_one_core"] for row in host_rows
            )
            per_host.append(
                {
                    "host": host,
                    "logical_cpus": capacity,
                    "monitor_cpu_percent_one_core": cpu_one_core,
                    "monitor_cpu_percent_host_capacity": cpu_one_core / capacity,
                }
            )
        rss_before = sum(row["process_rss_mb_end"] for row in reference)
        rss_after = sum(row["process_rss_mb_end"] for row in monitored)
        pairs.append(
            {
                "pair_id": run_id,
                "workload": reference[0]["workload"],
                "world_size": ranks,
                "seed": reference[0]["seed"],
                "without_monitor_steps_per_second": throughput_before,
                "with_monitor_steps_per_second": throughput_after,
                "throughput_loss_percent": 100
                * (1 - throughput_after / throughput_before),
                "monitor_cpu_percent_one_core": sum(
                    row["monitor_cpu_percent_one_core"] for row in per_host
                ),
                "monitor_cpu_percent_host_capacity_mean": float(
                    np.mean(
                        [row["monitor_cpu_percent_host_capacity"] for row in per_host]
                    )
                ),
                "reference_process_rss_mb_end": rss_before,
                "monitored_process_rss_mb_end": rss_after,
                "monitor_rss_delta_mb": rss_after - rss_before,
                "per_host": per_host,
            }
        )
    report = {
        "pairs": pairs,
        "pair_count": len(pairs),
        "summary": {
            field: float(np.mean([row[field] for row in pairs]))
            for field in (
                "throughput_loss_percent",
                "monitor_cpu_percent_one_core",
                "monitor_cpu_percent_host_capacity_mean",
                "monitored_process_rss_mb_end",
                "monitor_rss_delta_mb",
            )
        },
        "memory_definition": "Sum of worker-process RSS at training completion; delta is monitored minus reference RSS.",
        "throughput_definition": "Minimum rank-local synchronized steps per second",
    }
    write_json(output, report)
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "per_host"}
            for row in pairs
        ]
    ).to_csv(Path(output).with_suffix(".csv"), index=False)
    return report
