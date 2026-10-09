from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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


def plot_timeline(prepared, predictions, run_id, output):
    dataset = WindowDataset(Path(prepared) / "test.npz")
    schema = read_json(Path(prepared) / "metadata.json")["schema"]
    rows = read_json(predictions)
    selected = [
        (i, row) for i, row in enumerate(dataset.metadata) if row["run_id"] == run_id
    ]
    if not selected:
        raise ValueError(f"Execution {run_id} absent from test data")
    names = [metric["name"] for metric in schema["metrics"]]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    times = [row["end"] for _, row in selected]
    for name in ("net_tx", "allreduce_ms", "barrier_ms"):
        if name in names:
            values = [
                dataset.arrays["x"][i, names.index(name), -1, 0] for i, _ in selected
            ]
            axes[0].plot(times, values, label=name)
    axes[0].set_ylabel("Healthy-normalized value")
    axes[0].legend()
    prediction_rows = sorted(
        [row for row in rows if row["run_id"] == run_id], key=lambda row: row["end"]
    )
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
