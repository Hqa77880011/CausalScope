from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score

from .baselines import anomaly_scores
from .calibration import HealthyCalibration
from .data import WindowDataset
from .schema import read_json, write_json
from .scoring import rank_components
from .training import model_from_checkpoint


def metric_table(rows, components):
    single = [row for row in rows if len(row["roots"]) == 1]
    multi = [row for row in rows if len(row["roots"]) == 2]
    metrics = {
        "windows": len(rows),
        "single_fault_windows": len(single),
        "dual_fault_windows": len(multi),
    }
    if single:
        truth = np.array([components.index(row["roots"][0]) for row in single])
        predicted = np.array(
            [
                components.index(row["order"][0]) if not row["abstain"] else -1
                for row in single
            ]
        )
        metrics["top1"] = float(np.mean(truth == predicted))
        metrics["top3"] = float(
            np.mean(
                [
                    not row["abstain"]
                    and row["roots"][0] in row.get("initial_order", row["order"])[:3]
                    for row in single
                ]
            )
        )
        metrics["macro_f1"] = float(
            f1_score(
                truth,
                predicted,
                labels=np.arange(len(components)),
                average="macro",
                zero_division=0,
            )
        )
        aucs = []
        scores = np.array([row["scores"] for row in single])
        for i in range(len(components)):
            labels = truth == i
            if labels.any() and (~labels).any():
                aucs.append(roc_auc_score(labels, scores[:, i]))
        metrics["auroc"] = float(np.mean(aucs)) if aucs else None
        metrics["auroc_valid_classes"] = len(aucs)
        metrics["recommendation_coverage"] = float(
            np.mean([not row["abstain"] for row in single])
        )
    if multi:
        ndcg, exact, cover = [], [], []
        for row in multi:
            relevant = set(row["roots"])
            first = row["order"][:2] if not row["abstain"] else []
            dcg = sum(
                float(component in relevant) / np.log2(i + 2)
                for i, component in enumerate(first)
            )
            ndcg.append(dcg / (1 + 1 / np.log2(3)))
            exact.append(set(first) == relevant)
            cover.append(bool(first) and first[0] in relevant)
        metrics.update(
            {
                "ndcg_at_2": float(np.mean(ndcg)),
                "exact_top2": float(np.mean(exact)),
                "cover_at_1": float(np.mean(cover)),
            }
        )
    metrics["inference_latency_ms_mean"] = (
        float(np.mean([row["latency_ms"] for row in rows])) if rows else None
    )
    return metrics


def detection_delay(rows, stable_windows):
    executions = []
    for run_id in sorted({row["run_id"] for row in rows}):
        sequence = sorted(
            [row for row in rows if row["run_id"] == run_id], key=lambda row: row["end"]
        )
        faulty = [row for row in sequence if row["roots"]]
        if not faulty:
            continue
        streak, previous, first = 0, None, None
        for row in faulty:
            top = row["order"][0]
            if row["abstain"] or top not in row["roots"]:
                streak, previous = 0, None
            else:
                streak = streak + 1 if previous == top else 1
                previous = top
                if streak >= stable_windows:
                    first = row["end"] - row["fault_start"]
                    break
        executions.append(
            {
                "run_id": run_id,
                "detected": first is not None,
                "delay_seconds": first,
                "observation_seconds": faulty[-1]["end"] - faulty[0]["fault_start"],
            }
        )
    detected = [row["delay_seconds"] for row in executions if row["detected"]]
    summary = {
        "detection_rate": float(np.mean([row["detected"] for row in executions]))
        if executions
        else None,
        "detection_delay_seconds_mean": float(np.mean(detected)) if detected else None,
        "detection_delay_seconds_median": float(np.median(detected))
        if detected
        else None,
        "faulty_executions": len(executions),
        "undetected_executions": sum(not row["detected"] for row in executions),
    }
    return summary, executions


def evaluate(prepared, checkpoint_path, output, device="cpu", split="test"):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = WindowDataset(prepared / f"{split}.npz")
    prepared_metadata = read_json(prepared / "metadata.json")
    schema = prepared_metadata["schema"]
    components = schema["components"]
    neural = Path(checkpoint_path).suffix == ".pt"
    if neural:
        model, checkpoint = model_from_checkpoint(checkpoint_path, device)
        calibration = HealthyCalibration(checkpoint["calibration"])
        if calibration.state != read_json(prepared / "calibration.json"):
            raise ValueError(
                "Prepared data and checkpoint have different healthy calibration"
            )
        ranking_config = dict(checkpoint["config"]["ranking"])
        if checkpoint["method"] != "causalscope":
            ranking_config["counterfactual"] = False
            ranking_config["iterative"] = False
    else:
        checkpoint = joblib.load(checkpoint_path)
        ranking_config = checkpoint["config"]["ranking"]
    if schema != checkpoint["schema"]:
        raise ValueError("Prepared schema does not match checkpoint schema")
    rows = []
    for i, metadata in enumerate(dataset.metadata):
        x = dataset.arrays["x"][i : i + 1]
        if neural:
            tensor = torch.from_numpy(x).to(device)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            start = perf_counter()
            with torch.inference_mode():
                result = rank_components(
                    model,
                    tensor,
                    [metadata["domain"]],
                    calibration,
                    ranking_config,
                    checkpoint["temperature"],
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency = (perf_counter() - start) * 1000
            scores = result["scores"][0].cpu().tolist()
            order = [components[index] for index in result["order"][0].cpu().tolist()]
            initial_order = [
                components[index] for index in result["initial_order"][0].cpu().tolist()
            ]
            abstain = bool(result["abstain"][0])
            details = {
                "probabilities": result["probabilities"][0].cpu().tolist(),
                "gains": result["gains"][0].cpu().tolist(),
                "predicted_step": float(result["step"][0]),
                "fault_probability": float(result["fault_probability"][0]),
                "missing_fraction": float(result["missing_fraction"][0]),
            }
        else:
            start = perf_counter()
            values = anomaly_scores(checkpoint, x, device)[0]
            latency = (perf_counter() - start) * 1000
            scores = values.tolist()
            order = [components[index] for index in np.argsort(-values, kind="stable")]
            initial_order = order
            abstain = False
            details = {}
        strongest = {}
        for component in order[:2]:
            owned = [
                j
                for j, metric in enumerate(schema["metrics"])
                if metric["owner"] == component
            ]
            strength = np.abs(
                np.where(x[0, owned, :, 1] == 0, x[0, owned, :, 0], 0)
            ).mean(-1)
            strongest[component] = [
                schema["metrics"][owned[j]]["name"] for j in np.argsort(-strength)[:3]
            ]
        rows.append(
            {
                **metadata,
                **details,
                "components": components,
                "scores": scores,
                "order": order,
                "initial_order": initial_order,
                "abstain": abstain,
                "latency_ms": latency,
                "strongest_metrics": strongest,
            }
        )
    metrics = metric_table(rows, components)
    delay, incidents = detection_delay(rows, ranking_config["stable_windows"])
    metrics.update(delay)
    write_json(output / "predictions.json", rows)
    write_json(
        output / "metrics.json",
        {
            "method": checkpoint["method"],
            "seed": checkpoint["seed"],
            "held_out": prepared_metadata["held_out"],
            "split": split,
            "test_workloads": sorted({row["workload"] for row in rows}),
            "decision_thresholds": {
                key: ranking_config[key]
                for key in (
                    "probability_threshold",
                    "gain_threshold",
                    "missing_threshold",
                )
            },
            "metrics": metrics,
        },
    )
    write_json(output / "incidents.json", incidents)
    groups = []

    def group_label(row, field):
        if field == "workload":
            return row["workload"]
        if field == "root":
            return row["roots"][0] if len(row["roots"]) == 1 else None
        if field == "rank_scale":
            return str(row["world_size"]) if row.get("world_size") is not None else None
        if field == "intensity":
            return (
                str(row["intensity_level"])
                if row["roots"] and row.get("intensity_level") is not None
                else None
            )
        return "+".join(sorted(row["roots"])) if len(row["roots"]) == 2 else None

    for field in ("workload", "root", "rank_scale", "intensity", "fault_pair"):
        labels = sorted(
            {label for row in rows if (label := group_label(row, field)) is not None}
        )
        for label in labels:
            selected = [row for row in rows if group_label(row, field) == label]
            if selected:
                groups.append(
                    {
                        "group": field,
                        "label": label,
                        **metric_table(selected, components),
                    }
                )
    pd.DataFrame(groups).to_csv(output / "breakdown.csv", index=False)
    return metrics
