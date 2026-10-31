"""Exercise the telemetry-to-localization path on generated functional inputs."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from causalscope.data import WindowDataset, prepare
from causalscope.evaluation import detection_delay, evaluate, metric_table
from causalscope.model import LocalizationModel
from causalscope.schema import read_json
from causalscope.training import model_from_checkpoint, train_neural
from causalscope.visualization import plot_results


def main(output):
    """Validate actual file I/O, learning, serialization and metric definitions."""
    torch.set_num_threads(1)
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    schema = yaml.safe_load((repo / "configs/schema.yaml").read_text())
    config = yaml.safe_load((repo / "configs/default.yaml").read_text())
    config["data"].update(window_seconds=6.4, stride_seconds=3.2)
    config["model"].update(hidden=8, dropout=0.0)
    config["graph"].update(max_lag=2)
    config["training"].update(epochs=2, batch_size=16, patience=2)
    config["ranking"].update(probability_grid=[0.25], gain_grid=[0.0], missing_grid=[0.2])
    runs = []
    for workload in ("functional_a", "functional_b"):
        for split in ("calibration", "train", "validation", "test"):
            roots_list = [[]] if split == "calibration" else [[], *[[c] for c in schema["components"]]]
            if split == "test":
                roots_list.append(schema["components"][:2])
            for number, roots in enumerate(roots_list):
                run_id = f"{workload}-{split}-{number}"
                times = np.arange(64) * 0.2
                rows = []
                for metric in schema["metrics"]:
                    values = 10 + np.sin(times * 0.7) + rng.normal(0, 0.3, len(times))
                    if metric["owner"] in roots:
                        values[times >= 3.2] += 3
                    rows.extend(zip(times, [metric["name"]] * len(times), values))
                telemetry, steps = f"{run_id}-telemetry.csv", f"{run_id}-steps.csv"
                pd.DataFrame(rows, columns=["timestamp", "metric", "value"]).to_csv(output / telemetry, index=False)
                pd.DataFrame({"timestamp": times, "step_ms": 20 + np.sin(times) + (times >= 3.2) * len(roots)}).to_csv(output / steps, index=False)
                runs.append(dict(id=run_id, workload=workload, split=split, roots=roots, world_size=1, batch_size=2,
                                 telemetry=telemetry, steps=steps, fault_start=3.2, fault_end=20.0))
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(runs), encoding="utf-8")
    prepared = output / "prepared"
    counts = prepare(manifest, schema, config, prepared)
    dataset = WindowDataset(prepared / "train.npz")
    metadata, graph = read_json(prepared / "metadata.json"), read_json(prepared / "graph.json")
    model = LocalizationModel(schema, graph, len(metadata["domains"]), config["model"])
    x = torch.from_numpy(dataset.arrays["x"][:16])
    prediction = model(x)
    loss = (F.cross_entropy(prediction["root_logits"], torch.arange(len(x)) % len(schema["components"]))
            + F.cross_entropy(prediction["domain_logits"], torch.arange(len(x)) % 2)
            + F.huber_loss(prediction["step"], torch.ones(len(x)))
            + F.binary_cross_entropy_with_logits(prediction["fault_logit"], torch.ones(len(x))))
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss.backward()
    for name in ("temporal.branches.0.0.conv.weight", "layers.0.prior_weight.weight", "root_head.weight", "domain_head.2.weight", "step_head.2.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0, name
    optimizer.step()
    assert not torch.equal(before["root_head.weight"], model.root_head.weight)
    model.eval()
    with torch.inference_mode():
        expected = model(x)
    torch.save(model.state_dict(), output / "roundtrip.pt")
    restored = LocalizationModel(schema, graph, len(metadata["domains"]), config["model"])
    restored.load_state_dict(torch.load(output / "roundtrip.pt", weights_only=True))
    restored.eval()
    with torch.inference_mode():
        actual = restored(x)
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=1e-6, atol=1e-6)
    checkpoint = train_neural(prepared, output / "model", config, "causalscope", 7, "cpu")
    assert np.isfinite(pd.read_csv(output / "model/training.csv").loss).all()
    loaded, _ = model_from_checkpoint(checkpoint)
    assert not loaded.training
    metrics = evaluate(prepared, checkpoint, output / "evaluation")
    plot_results([output / "evaluation"], output / "plots")
    components = ["a", "b", "c"]
    rows = [dict(roots=["a"], order=["a", "b", "c"], scores=[0.9, 0.1, 0.0], abstain=False, latency_ms=1),
            dict(roots=["b"], order=["a", "b", "c"], scores=[0.6, 0.4, 0.0], abstain=False, latency_ms=1),
            dict(roots=["a", "b"], order=["a", "c", "b"], scores=[0.7, 0.1, 0.2], abstain=False, latency_ms=1)]
    hand = metric_table(rows, components)
    assert hand["top1"] == 0.5 and hand["top3"] == 1.0
    assert np.isclose(hand["macro_f1"], 2 / 9) and hand["auroc"] == 1.0
    assert np.isclose(hand["ndcg_at_2"], 1 / (1 + 1 / np.log2(3)))
    assert hand["exact_top2"] == 0 and hand["cover_at_1"] == 1
    delay, _ = detection_delay([dict(run_id="r", roots=["a"], order=["a"], abstain=False, end=i, fault_start=0) for i in (1, 2, 3)], 3)
    assert delay["detection_delay_seconds_mean"] == 3
    result = dict(functional_inputs_only=True, windows=counts, finite_loss=float(loss.detach()),
                  checkpoint_roundtrip="rtol=atol=1e-6", gradient_update=True,
                  metrics_checked=["top1", "top3", "macro_f1", "auroc", "ndcg_at_2", "exact_top2", "cover_at_1", "detection_delay"],
                  evaluation_windows=metrics["windows"])
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/functional-check"))
    main(parser.parse_args().output)
