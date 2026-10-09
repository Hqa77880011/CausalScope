from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import WindowDataset
from .schema import read_json


def plot_results(inputs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    breakdowns = []
    for source in inputs:
        directory = Path(source)
        report = read_json(directory / "metrics.json")
        records.append(
            {"method": report["method"], "seed": report["seed"], **report["metrics"]}
        )
        breakdown = pd.read_csv(directory / "breakdown.csv")
        breakdown["method"] = report["method"]
        breakdowns.append(breakdown)
    table = pd.DataFrame(records)
    table.to_csv(output / "comparison.csv", index=False)
    fields = [
        field
        for field in (
            "top1",
            "top3",
            "macro_f1",
            "auroc",
            "ndcg_at_2",
            "exact_top2",
            "cover_at_1",
        )
        if field in table and table[field].notna().any()
    ]
    if fields:
        grouped = table.groupby("method")[fields].mean()
        grouped.plot.bar(
            figsize=(max(8, len(grouped) * 1.2), 5),
            color=[
                "#4678a5",
                "#76a4b8",
                "#5d918b",
                "#a6bbc7",
                "#8ba79c",
                "#627f91",
                "#94aeb3",
            ][: len(fields)],
        )
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output / "accuracy.png", dpi=180)
        plt.close()
    breakdown = pd.concat(breakdowns, ignore_index=True)
    for group in ("workload", "root"):
        rows = breakdown[breakdown.group == group]
        if "top1" not in rows or not rows.top1.notna().any():
            continue
        pivot = rows.pivot_table(
            index="label", columns="method", values="top1", aggfunc="mean"
        )
        pivot.plot.bar(figsize=(9, 5), colormap="GnBu")
        plt.ylabel("Top-1 accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(output / f"{group}.png", dpi=180)
        plt.close()
    delay = table[
        [
            field
            for field in (
                "method",
                "seed",
                "detection_rate",
                "detection_delay_seconds_mean",
                "undetected_executions",
            )
            if field in table
        ]
    ]
    delay.to_csv(output / "detection.csv", index=False)


def plot_timeline(prepared, predictions, run_id, output, window_end=None):
    dataset = WindowDataset(Path(prepared) / "test.npz")
    schema = read_json(Path(prepared) / "metadata.json")["schema"]
    rows = read_json(predictions)
    selected = [
        (i, row) for i, row in enumerate(dataset.metadata) if row["run_id"] == run_id
    ]
    if not selected:
        raise ValueError(f"Execution {run_id} absent from test data")
    if window_end is not None:
        selected = [min(selected, key=lambda pair: abs(pair[1]["end"] - window_end))]
    names = [metric["name"] for metric in schema["metrics"]]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    samples = {}
    dt = read_json(Path(prepared) / "metadata.json")["config"]["data"]["sample_seconds"]
    for index, row in selected:
        window = dataset.arrays["x"][index]
        for position in range(window.shape[1]):
            timestamp = round(row["start"] + position * dt, 8)
            samples[timestamp] = (
                window[:, position, 0],
                window[:, position, 1],
                dataset.arrays["step_trace"][index, position]
                if "step_trace" in dataset.arrays
                else np.nan,
            )
    times = sorted(samples)
    for name in ("net_tx", "allreduce_ms", "barrier_ms"):
        if name in names:
            node = names.index(name)
            values = [
                samples[time][0][node] if samples[time][1][node] == 0 else np.nan
                for time in times
            ]
            axes[0].plot(times, values, label=name)
    if "step_trace" in dataset.arrays:
        axes[0].plot(times, [samples[time][2] for time in times], label="step_ms")
    else:
        axes[0].plot(
            [row["end"] for _, row in selected],
            [dataset.arrays["target"][index] for index, _ in selected],
            label="step_ms_window_mean",
        )
    axes[0].set_ylabel("Healthy-normalized value")
    axes[0].legend()
    prediction_rows = sorted(
        [row for row in rows if row["run_id"] == run_id], key=lambda row: row["end"]
    )
    prediction_rows = [
        row
        for row in prediction_rows
        if times[0] - 1e-6 <= row["end"] <= times[-1] + 1e-6
    ]
    for i, component in enumerate(schema["components"]):
        axes[1].plot(
            [row["end"] for row in prediction_rows],
            [row["scores"][i] for row in prediction_rows],
            label=component,
        )
    axes[1].set_ylabel("Ranking score")
    axes[1].set_xlabel("Seconds from execution start")
    axes[1].legend(ncol=3, fontsize=8)
    fault_start = selected[0][1]["fault_start"]
    if fault_start is not None:
        for axis in axes:
            axis.axvline(fault_start, color="#667a88", linestyle="--")
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_paper_tables(directory):
    directory = Path(directory)
    main = pd.read_csv(directory / "main_results.csv")
    if not main.empty and {"top1", "top3"}.issubset(main.columns):
        main.set_index("method")[["top1", "top3"]].plot.bar(
            figsize=(10, 5), color=["#4678a5", "#76a4b8"]
        )
        plt.ylabel("Localization accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(directory / "main_accuracy.png", dpi=180)
        plt.close()
    folds = pd.read_csv(directory / "cross_workload.csv")
    if not folds.empty and "top1" in folds:
        workload_order = [
            name
            for name in ("resnet50", "mobilenet_v3_large", "vit_b_16", "bert_base")
            if name in set(folds.held_out)
        ]
        workload_order += sorted(set(folds.held_out) - set(workload_order))
        figure, axis = plt.subplots(figsize=(9, 5))
        for method, color in (
            ("tcn", "#a6bbc7"),
            ("gnn", "#5d918b"),
            ("no_invariance", "#76a4b8"),
            ("causalscope", "#4678a5"),
        ):
            rows = folds[folds.method == method].set_index("held_out")
            if not rows.empty:
                axis.plot(
                    workload_order,
                    rows.top1.reindex(workload_order),
                    marker="o",
                    label=method,
                    color=color,
                )
        axis.set_ylabel("Held-out Top-1 accuracy")
        axis.set_ylim(0, 1)
        axis.legend()
        figure.tight_layout()
        figure.savefig(directory / "cross_workload.png", dpi=180)
        plt.close(figure)
    for filename, metrics in (
        ("ablation", ["top1", "macro_f1"]),
        ("multifault", ["ndcg_at_2", "exact_top2", "cover_at_1"]),
    ):
        table = pd.read_csv(directory / f"{filename}.csv")
        available = [field for field in metrics if field in table]
        if table.empty or not available or not table[available].notna().any().any():
            continue
        table.set_index("method")[available].plot.bar(figsize=(10, 5), colormap="GnBu")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(directory / f"{filename}.png", dpi=180)
        plt.close()
    fault = pd.read_csv(directory / "fault_breakdown.csv")
    if not fault.empty and "top1" in fault and fault.top1.notna().any():
        fault.pivot(index="label", columns="method", values="top1").plot.bar(
            figsize=(10, 5), colormap="GnBu"
        )
        plt.ylabel("Per-fault Top-1 accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(directory / "fault_breakdown.png", dpi=180)
        plt.close()


def plot_overhead(report, output):
    report = read_json(report)
    rows = report["inference"]
    table = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(table.nodes, table.latency_ms_p50, marker="o", color="#4678a5")
    axes[0].set_ylabel("Median inference latency (ms)")
    axes[1].plot(table.nodes, table.rss_mb, marker="o", color="#5d918b")
    axes[1].set_ylabel("Process RSS (MB)")
    for axis in axes:
        axis.set_xlabel("Telemetry nodes")
    figure.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
