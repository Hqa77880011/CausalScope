from pathlib import Path

import pandas as pd

from .schema import read_json, write_json


SINGLE_METRICS = ("top1", "top3", "macro_f1", "auroc")
MULTI_METRICS = ("ndcg_at_2", "exact_top2", "cover_at_1")
ABLATIONS = (
    "causalscope",
    "no_prior",
    "prior_only",
    "data_only",
    "no_invariance",
    "no_counterfactual",
    "lstm_encoder",
)


def evaluation_directories(inputs):
    directories = set()
    for source in inputs:
        source = Path(source)
        if source.is_file() and source.name == "metrics.json":
            directories.add(source.parent.resolve())
        elif (source / "metrics.json").is_file():
            directories.add(source.resolve())
        else:
            directories.update(
                path.parent.resolve() for path in source.rglob("metrics.json")
            )
    if not directories:
        raise ValueError("No evaluated metrics.json reports found")
    return sorted(directories)


def mean_table(frame, groups, metrics):
    fields = [field for field in metrics if field in frame.columns]
    if frame.empty or not fields:
        return pd.DataFrame(columns=[*groups, *metrics, "seed_count"])
    grouped = frame.groupby(groups, dropna=False)
    result = grouped[fields].mean().reset_index()
    counts = grouped.seed.nunique().reset_index(name="seed_count")
    return result.merge(counts, on=groups)


def paper_report(inputs, output):
    from .visualization import plot_paper_tables

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records, breakdowns = [], []
    for directory in evaluation_directories(inputs):
        report = read_json(directory / "metrics.json")
        if report.get("split", "test") != "test":
            continue
        held_out = report.get("held_out")
        if held_out and report.get("test_workloads") != [held_out]:
            raise ValueError(
                f"Fold {directory} includes workloads other than {held_out}; prepare and evaluate it again"
            )
        record = {
            "method": report["method"],
            "seed": report["seed"],
            "held_out": held_out,
            "source": str(directory),
            **report["metrics"],
        }
        identity = (record["method"], record["seed"], held_out)
        if any(
            (row["method"], row["seed"], row["held_out"]) == identity for row in records
        ):
            raise ValueError(f"Duplicate method/seed/fold report: {identity}")
        records.append(record)
        table = pd.read_csv(directory / "breakdown.csv")
        table["method"], table["seed"], table["held_out"] = (
            report["method"],
            report["seed"],
            held_out,
        )
        breakdowns.append(table)
    if not records:
        raise ValueError("No test-split metric reports found")
    frame = pd.DataFrame(records)
    frame.to_csv(output / "seed_results.csv", index=False)
    main = frame[frame.held_out.isna()]
    baseline_methods = (
        "threshold",
        "pca",
        "isolation_forest",
        "autoencoder",
        "lstm",
        "tcn",
        "gnn",
        "causalscope",
    )
    mean_table(
        main[main.method.isin(baseline_methods)], ["method"], SINGLE_METRICS
    ).to_csv(output / "main_results.csv", index=False)
    mean_table(
        main[main.method.isin(ABLATIONS)], ["method"], ("top1", "macro_f1")
    ).to_csv(output / "ablation.csv", index=False)
    mean_table(
        main[main.method.isin(("tcn", "gnn", "no_iterative", "causalscope"))],
        ["method"],
        MULTI_METRICS,
    ).to_csv(output / "multifault.csv", index=False)
    folds = mean_table(
        frame[frame.held_out.notna()], ["method", "held_out"], SINGLE_METRICS
    )
    folds.to_csv(output / "cross_workload.csv", index=False)
    if folds.empty:
        transfer = pd.DataFrame(columns=["method", *SINGLE_METRICS, "fold_count"])
    else:
        fields = [field for field in SINGLE_METRICS if field in folds]
        transfer = folds.groupby("method")[fields].mean().reset_index()
        transfer["fold_count"] = transfer.method.map(
            folds.groupby("method").held_out.nunique()
        )
    transfer.to_csv(output / "cross_workload_mean.csv", index=False)
    breakdown = pd.concat(breakdowns, ignore_index=True)
    for group, filename, metrics in (
        ("root", "fault_breakdown", ("top1",)),
        ("rank_scale", "rank_scale", (*SINGLE_METRICS, *MULTI_METRICS)),
        ("intensity", "intensity", (*SINGLE_METRICS, *MULTI_METRICS)),
        ("fault_pair", "fault_pairs", MULTI_METRICS),
    ):
        selected = breakdown[(breakdown.group == group) & breakdown.held_out.isna()]
        mean_table(selected, ["method", "label"], metrics).to_csv(
            output / f"{filename}.csv", index=False
        )
    selection = []
    for record in records:
        path = Path(record["source"]).parent / "selection.json"
        if path.exists():
            selected = read_json(path)
            selection.append(
                {
                    "method": record["method"],
                    "seed": record["seed"],
                    "held_out": record["held_out"],
                    "parameter_count": selected["parameter_count"],
                    "temperature": selected["temperature"],
                    "best_validation_macro_f1": selected["best_validation_macro_f1"],
                }
            )
    pd.DataFrame(
        selection,
        columns=[
            "method",
            "seed",
            "held_out",
            "parameter_count",
            "temperature",
            "best_validation_macro_f1",
        ],
    ).to_csv(output / "model_selection.csv", index=False)
    summary = {
        "reports": len(records),
        "methods": sorted(frame.method.unique()),
        "held_out_workloads": sorted(frame.held_out.dropna().unique()),
        "seeds": sorted(int(seed) for seed in frame.seed.unique()),
        "cross_workload_average": "Mean over seeds within each fold, then equal-weight mean over available folds",
    }
    write_json(output / "summary.json", summary)
    plot_paper_tables(output)
    return summary
