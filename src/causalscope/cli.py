import argparse
import copy
import json
from pathlib import Path

from .schema import load_schema, read_json, read_yaml, write_json


def configuration(args):
    config = read_yaml(args.config)
    if getattr(args, "epochs", None) is not None:
        config["training"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None and args.command == "train":
        config["training"]["batch_size"] = args.batch_size
    return config


def train_method(prepared, output, config, method, seed, device):
    from .baselines import ANOMALY_METHODS, train_anomaly
    from .training import train_neural

    if method in ANOMALY_METHODS:
        return train_anomaly(prepared, output, config, method, seed, device)
    return train_neural(prepared, output, config, method, seed, device)


def run_suite(args):
    from .data import prepare
    from .evaluation import evaluate
    from .visualization import plot_results

    base = configuration(args)
    schema = load_schema(args.schema)
    outputs = []
    variants = {method: (method, copy.deepcopy(base)) for method in args.methods}
    if args.ablations:
        for name in (
            "no_prior",
            "prior_only",
            "data_only",
            "no_invariance",
            "no_counterfactual",
            "lstm_encoder",
            "no_iterative",
        ):
            variants[name] = ("causalscope", copy.deepcopy(base))
        variants["no_prior"][1]["graph"]["mode"] = "no_prior"
        variants["prior_only"][1]["graph"]["mode"] = "prior"
        variants["data_only"][1]["graph"]["mode"] = "data"
        variants["no_invariance"][1]["model"]["invariant"] = False
        variants["no_counterfactual"][1]["ranking"]["counterfactual"] = False
        variants["no_counterfactual"][1]["ranking"]["iterative"] = False
        variants["lstm_encoder"][1]["model"]["temporal"] = "lstm"
        variants["no_iterative"][1]["ranking"]["iterative"] = False
    graph_directories = {}
    for name, (method, config) in variants.items():
        mode = config["graph"]["mode"]
        if mode not in graph_directories:
            prepared = Path(args.output) / "prepared" / mode
            prepare(args.manifest, schema, config, prepared, args.held_out)
            graph_directories[mode] = prepared
        for seed in (
            args.seeds if args.seeds is not None else config["training"]["seeds"]
        ):
            directory = Path(args.output) / name / f"seed{seed}"
            checkpoint = train_method(
                graph_directories[mode], directory, config, method, seed, args.device
            )
            evaluate(
                graph_directories[mode],
                checkpoint,
                directory / "evaluation",
                args.device,
            )
            report_path = directory / "evaluation" / "metrics.json"
            report = read_json(report_path)
            report["method"] = name
            write_json(report_path, report)
            outputs.append(directory / "evaluation")
    plot_results(outputs, Path(args.output) / "plots")


def localize(args):
    import numpy as np
    import torch

    from .calibration import HealthyCalibration
    from .data import align_run
    from .online import replay_decisions
    from .schema import domain_key
    from .scoring import rank_components
    from .training import model_from_checkpoint

    model, checkpoint = model_from_checkpoint(args.checkpoint, args.device)
    if checkpoint["method"] != "causalscope":
        raise ValueError("Raw localization requires a CausalScope checkpoint")
    run = {
        "id": "input",
        "telemetry": args.telemetry,
        "steps": args.steps,
        "workload": args.workload,
        "batch_size": args.batch_size,
        "world_size": args.world_size,
    }
    key = domain_key(run)
    calibration = HealthyCalibration(checkpoint["calibration"])
    config = checkpoint["config"]
    record = align_run(run, checkpoint["schema"], config["data"])
    inputs = calibration.transform(key, record)
    length = round(config["data"]["window_seconds"] / config["data"]["sample_seconds"])
    stride = round(config["data"]["stride_seconds"] / config["data"]["sample_seconds"])
    components = checkpoint["schema"]["components"]
    rows = []
    with torch.inference_mode():
        for end in range(length, len(inputs) + 1, stride):
            x = inputs[end - length : end].transpose(1, 0, 2)[None]
            result = rank_components(
                model,
                torch.from_numpy(x).to(args.device),
                [key],
                calibration,
                config["ranking"],
                checkpoint["temperature"],
            )
            order = [components[i] for i in result["order"][0].cpu().tolist()]
            strongest = {}
            for component in order[:2]:
                owned = [
                    i
                    for i, metric in enumerate(checkpoint["schema"]["metrics"])
                    if metric["owner"] == component
                ]
                strength = np.abs(
                    np.where(x[0, owned, :, 1] == 0, x[0, owned, :, 0], 0)
                ).mean(-1)
                strongest[component] = [
                    checkpoint["schema"]["metrics"][owned[i]]["name"]
                    for i in np.argsort(-strength)[:3]
                ]
            rows.append(
                {
                    "run_id": "input",
                    "end": float(record["time"][end - 1]),
                    "components": components,
                    "order": order,
                    "scores": result["scores"][0].cpu().tolist(),
                    "probabilities": result["probabilities"][0].cpu().tolist(),
                    "gains": result["gains"][0].cpu().tolist(),
                    "abstain": bool(result["abstain"][0]),
                    "strongest_metrics": strongest,
                }
            )
    if not rows:
        raise ValueError("Recording is shorter than one monitoring window")
    write_json(Path(args.output) / "rankings.json", rows)
    write_json(
        Path(args.output) / "recommendations.json",
        replay_decisions(rows, config["ranking"]["stable_windows"]),
    )
    return {"windows": len(rows), "output": args.output}


def parser():
    root = argparse.ArgumentParser(
        prog="causalscope", description="CausalScope implementation"
    )
    commands = root.add_subparsers(dest="command", required=True)

    def configured(command):
        command.add_argument("--config", default="configs/default.yaml")
        command.add_argument("--schema", default="configs/schema.yaml")

    plan = commands.add_parser(
        "plan", help="Write the paper-sized execution matrix without running it"
    )
    plan.add_argument("--output", required=True)
    vision = commands.add_parser(
        "prepare-vision", help="Link an explicitly selected ImageNet-100 subset"
    )
    vision.add_argument("--source", required=True)
    vision.add_argument("--classes", required=True)
    vision.add_argument("--output", required=True)
    text = commands.add_parser(
        "prepare-text", help="Download WikiText-103 and write BERT token blocks"
    )
    text.add_argument("--output", required=True)
    text.add_argument("--block-size", type=int, default=128)
    collect = commands.add_parser(
        "collect",
        help="Train one workload while recording telemetry and optional faults",
    )
    collect.add_argument("--plan")
    collect.add_argument("--index", type=int, default=0)
    collect.add_argument("--run-id", default="healthy-run")
    collect.add_argument(
        "--workload",
        choices=["resnet50", "mobilenet_v3_large", "vit_b_16", "bert_base"],
        default="resnet50",
    )
    collect.add_argument("--dataset", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--batch-size", type=int, default=32)
    collect.add_argument("--workers", type=int, default=4)
    collect.add_argument("--steps", type=int, default=200)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument(
        "--split",
        choices=["calibration", "train", "validation", "test"],
        default="calibration",
    )
    collect.add_argument(
        "--faults",
        default="{}",
        help="JSON mapping from component names to table intensities",
    )
    collect.add_argument("--fault-rank", type=int, default=0)
    collect.add_argument("--fault-start-step", type=int, default=60)
    collect.add_argument("--fault-steps", type=int, default=120)
    collect.add_argument(
        "--segment-seconds",
        type=float,
        default=30,
        help="Minimum healthy, faulty, and post-fault segment length; use zero for exact step counts",
    )
    collect.add_argument("--interface")
    collect.add_argument("--cgroup")
    collect.add_argument("--block-device", help="Linux block device MAJOR:MINOR")
    collect.add_argument("--cpu", action="store_true")
    collect.add_argument("--no-monitor", action="store_true")
    merge = commands.add_parser(
        "merge", help="Merge complete rank recordings into a telemetry manifest"
    )
    merge.add_argument("--recordings", required=True)
    merge.add_argument("--output", required=True)
    prepare_command = commands.add_parser(
        "prepare",
        help="Align telemetry, fit healthy references, construct graph, and split windows",
    )
    configured(prepare_command)
    prepare_command.add_argument("--manifest", required=True)
    prepare_command.add_argument("--output", required=True)
    prepare_command.add_argument(
        "--held-out", help="Workload name reserved for cross-workload evaluation"
    )
    train = commands.add_parser(
        "train", help="Fit a localization method or anomaly baseline"
    )
    train.add_argument("--config", default="configs/default.yaml")
    train.add_argument("--prepared", required=True)
    train.add_argument("--output", required=True)
    train.add_argument(
        "--method",
        choices=[
            "causalscope",
            "threshold",
            "pca",
            "isolation_forest",
            "autoencoder",
            "lstm",
            "tcn",
            "gnn",
        ],
        default="causalscope",
    )
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch-size", type=int)
    evaluate_command = commands.add_parser(
        "evaluate", help="Compute localization metrics from measured windows"
    )
    evaluate_command.add_argument("--prepared", required=True)
    evaluate_command.add_argument("--checkpoint", required=True)
    evaluate_command.add_argument("--output", required=True)
    evaluate_command.add_argument("--device", default="cpu")
    evaluate_command.add_argument(
        "--split", choices=["validation", "test"], default="test"
    )
    suite = commands.add_parser(
        "suite",
        help="Run selected methods, seeds, and ablations on one execution split",
    )
    configured(suite)
    suite.add_argument("--manifest", required=True)
    suite.add_argument("--output", required=True)
    suite.add_argument("--held-out")
    suite.add_argument(
        "--methods",
        nargs="+",
        default=[
            "threshold",
            "pca",
            "isolation_forest",
            "autoencoder",
            "lstm",
            "tcn",
            "gnn",
            "causalscope",
        ],
        choices=[
            "threshold",
            "pca",
            "isolation_forest",
            "autoencoder",
            "lstm",
            "tcn",
            "gnn",
            "causalscope",
        ],
    )
    suite.add_argument("--seeds", nargs="+", type=int)
    suite.add_argument("--epochs", type=int)
    suite.add_argument("--ablations", action="store_true")
    suite.add_argument("--device", default="cpu")
    plot = commands.add_parser("plot", help="Plot real metric reports")
    plot.add_argument("--inputs", nargs="+", required=True)
    plot.add_argument("--output", required=True)
    timeline = commands.add_parser(
        "timeline", help="Plot aligned incident telemetry and ranking scores"
    )
    timeline.add_argument("--prepared", required=True)
    timeline.add_argument("--predictions", required=True)
    timeline.add_argument("--run-id", required=True)
    timeline.add_argument("--output", required=True)
    replay = commands.add_parser(
        "replay", help="Emit recommendations after stable consecutive rankings"
    )
    replay.add_argument("--predictions", required=True)
    replay.add_argument("--stable-windows", type=int, default=3)
    replay.add_argument("--output", required=True)
    raw = commands.add_parser(
        "localize", help="Rank an unlabeled aligned recording with a trained checkpoint"
    )
    raw.add_argument("--telemetry", required=True)
    raw.add_argument("--steps", required=True)
    raw.add_argument("--checkpoint", required=True)
    raw.add_argument("--workload", required=True)
    raw.add_argument("--batch-size", type=int, required=True)
    raw.add_argument("--world-size", type=int, required=True)
    raw.add_argument("--device", default="cpu")
    raw.add_argument("--output", required=True)
    overhead = commands.add_parser(
        "overhead",
        help="Measure model inference cost and optionally summarize paired throughput",
    )
    overhead.add_argument("--prepared", required=True)
    overhead.add_argument("--checkpoint", required=True)
    overhead.add_argument("--sizes", nargs="+", type=int, default=[19, 24, 32, 48])
    overhead.add_argument("--repeats", type=int, default=100)
    overhead.add_argument("--warmup", type=int, default=10)
    overhead.add_argument(
        "--pairs",
        help="CSV with measured with/without-monitor throughput and collection cost",
    )
    overhead.add_argument("--device", default="cpu")
    overhead.add_argument("--output", required=True)
    overhead_plot = commands.add_parser(
        "plot-overhead", help="Plot a measured node-scaling report"
    )
    overhead_plot.add_argument("--report", required=True)
    overhead_plot.add_argument("--output", required=True)
    return root


def main():
    args = parser().parse_args()
    result = None
    if args.command == "plan":
        from .experiments import make_plan

        result = make_plan(args.output)
    elif args.command == "prepare-vision":
        from .workloads import prepare_vision

        prepare_vision(args.source, args.classes, args.output)
    elif args.command == "prepare-text":
        from .workloads import prepare_text

        prepare_text(args.output, args.block_size)
    elif args.command == "collect":
        from .workloads import run_workload

        result = run_workload(args)
    elif args.command == "merge":
        from .experiments import merge_recordings

        result = {"executions": merge_recordings(args.recordings, args.output)}
    elif args.command == "prepare":
        from .data import prepare

        result = prepare(
            args.manifest,
            load_schema(args.schema),
            configuration(args),
            args.output,
            args.held_out,
        )
    elif args.command == "train":
        result = {
            "checkpoint": train_method(
                args.prepared,
                args.output,
                configuration(args),
                args.method,
                args.seed,
                args.device,
            )
        }
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(
            args.prepared, args.checkpoint, args.output, args.device, args.split
        )
    elif args.command == "suite":
        run_suite(args)
    elif args.command == "plot":
        from .visualization import plot_results

        plot_results(args.inputs, args.output)
    elif args.command == "timeline":
        from .visualization import plot_timeline

        plot_timeline(args.prepared, args.predictions, args.run_id, args.output)
    elif args.command == "replay":
        from .online import replay_decisions

        write_json(
            args.output,
            replay_decisions(read_json(args.predictions), args.stable_windows),
        )
    elif args.command == "localize":
        result = localize(args)
    elif args.command == "overhead":
        from .overhead import benchmark

        result = benchmark(
            args.prepared,
            args.checkpoint,
            args.output,
            args.sizes,
            args.repeats,
            args.warmup,
            args.device,
            args.pairs,
        )
    elif args.command == "plot-overhead":
        from .visualization import plot_overhead

        plot_overhead(args.report, args.output)
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
