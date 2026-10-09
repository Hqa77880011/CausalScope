import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .calibration import HealthyCalibration
from .data import WindowDataset
from .model import LocalizationModel
from .schema import read_json, write_json
from .scoring import score_components


NEURAL_METHODS = ("causalscope", "lstm", "tcn", "gnn")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_from_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LocalizationModel(
        checkpoint["schema"],
        checkpoint["graph"],
        len(checkpoint["domains"]),
        checkpoint["config"]["model"],
        checkpoint["method"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def validation_f1(model, dataset, device, batch_size, component_count):
    logits = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size):
            logits.append(model(batch["x"].to(device))["root_logits"].cpu())
    logits = torch.cat(logits)
    roots = dataset.arrays["roots"]
    single = roots.sum(axis=1) == 1
    if not single.any():
        raise ValueError("Validation needs single-fault windows for model selection")
    workloads = np.array([row["workload"] for row in dataset.metadata])
    scores = []
    for workload in sorted(set(workloads[single])):
        selected = single & (workloads == workload)
        scores.append(
            f1_score(
                roots[selected].argmax(axis=1),
                logits[selected].argmax(dim=1).numpy(),
                labels=np.arange(component_count),
                average="macro",
                zero_division=0,
            )
        )
    return float(np.mean(scores)), logits


def calibrate_temperature(logits, roots):
    selected = roots.sum(axis=1) == 1
    logits = logits[selected].detach()
    target = torch.as_tensor(roots[selected].argmax(axis=1), dtype=torch.long)
    if not len(target):
        raise ValueError("Temperature calibration needs validation single faults")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature], max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20)
        loss = F.cross_entropy(logits / temperature, target)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20))


def calibrate_decisions(model, dataset, checkpoint, device):
    ranking = checkpoint["config"]["ranking"]
    effective = dict(ranking)
    if checkpoint["method"] != "causalscope":
        effective["counterfactual"] = False
    calibration = HealthyCalibration(checkpoint["calibration"])
    arrays = []
    batch_size = checkpoint["config"]["training"]["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            x = torch.from_numpy(dataset.arrays["x"][start : start + batch_size]).to(
                device
            )
            keys = [
                row["domain"] for row in dataset.metadata[start : start + batch_size]
            ]
            result = score_components(
                model, x, keys, calibration, effective, checkpoint["temperature"]
            )
            arrays.append(
                np.stack(
                    [
                        result["scores"].argmax(-1).cpu().numpy(),
                        result["probabilities"].max(-1).values.cpu().numpy(),
                        result["gains"].max(-1).values.cpu().numpy(),
                        result["missing_fraction"].cpu().numpy(),
                    ],
                    axis=1,
                )
            )
    evidence = np.concatenate(arrays)
    roots = dataset.arrays["roots"]
    eligible = roots.sum(-1) <= 1
    truth = np.where(roots.sum(-1) == 1, roots.argmax(-1), -1)
    workloads = np.array([row["workload"] for row in dataset.metadata])
    workload_names = sorted(set(workloads[eligible]))
    if not np.any(truth[eligible] >= 0):
        raise ValueError("Decision calibration needs single-fault validation windows")
    defaults = (
        ranking["probability_threshold"],
        ranking["gain_threshold"],
        ranking["missing_threshold"],
    )
    probability_grid = ranking.get("probability_grid", [defaults[0]])
    gain_grid = (
        ranking.get("gain_grid", [defaults[1]])
        if effective["counterfactual"]
        else [defaults[1]]
    )
    missing_grid = ranking.get("missing_grid", [defaults[2]])
    if not probability_grid or not gain_grid or not missing_grid:
        raise ValueError("Validation threshold grids must be nonempty")
    best, best_distance, chosen = -1.0, float("inf"), None
    for probability, gain, missing in itertools.product(
        probability_grid, gain_grid, missing_grid
    ):
        if not 0 <= probability <= 1 or not 0 <= missing <= 1 or gain < 0:
            raise ValueError("Invalid validation threshold grid")
        abstain = (evidence[:, 1] < probability) | (evidence[:, 3] > missing)
        if effective["counterfactual"]:
            abstain |= evidence[:, 2] < gain
        predicted = np.where(abstain, -1, evidence[:, 0].astype(int))
        scores = []
        for workload in workload_names:
            selected = eligible & (workloads == workload)
            scores.append(
                f1_score(
                    truth[selected],
                    predicted[selected],
                    labels=np.arange(len(checkpoint["schema"]["components"])),
                    average="macro",
                    zero_division=0,
                )
            )
        score = float(np.mean(scores))
        distance = sum(
            abs(value - default)
            for value, default in zip((probability, gain, missing), defaults)
        )
        if score > best or (score == best and distance < best_distance):
            best, best_distance = score, distance
            chosen = {
                "probability_threshold": float(probability),
                "gain_threshold": float(gain),
                "missing_threshold": float(missing),
            }
    ranking.update(chosen)
    checkpoint["decision_calibration"] = {
        "split": "validation",
        "workloads": workload_names,
        "windows": int(eligible.sum()),
        "mean_macro_f1": best,
        "thresholds": chosen,
        "healthy_windows": int((truth[eligible] == -1).sum()),
    }
    return checkpoint["decision_calibration"]


def calibrate_checkpoint(prepared, checkpoint_path, output, device="cpu"):
    prepared = Path(prepared)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    if checkpoint["schema"] != read_json(prepared / "metadata.json")[
        "schema"
    ] or checkpoint["calibration"] != read_json(prepared / "calibration.json"):
        raise ValueError(
            "Checkpoint schema and calibration must match prepared validation data"
        )
    dataset = WindowDataset(prepared / "validation.npz")
    _, logits = validation_f1(
        model,
        dataset,
        device,
        checkpoint["config"]["training"]["batch_size"],
        len(checkpoint["schema"]["components"]),
    )
    checkpoint["temperature"] = calibrate_temperature(logits, dataset.arrays["roots"])
    summary = calibrate_decisions(model, dataset, checkpoint, device)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return {"temperature": checkpoint["temperature"], **summary}


def train_neural(prepared, output, config, method, seed, device):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = read_json(prepared / "metadata.json")
    graph = read_json(prepared / "graph.json")
    schema = metadata["schema"]
    if config["graph"] != graph["config"]:
        raise ValueError(
            "Graph configuration changed; prepare data again with this config"
        )
    set_seed(seed)
    train_data = WindowDataset(prepared / "train.npz")
    validation_data = WindowDataset(prepared / "validation.npz")
    model = LocalizationModel(
        schema, graph, len(metadata["domains"]), config["model"], method
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    calibration = HealthyCalibration.load(prepared / "calibration.json")
    batch_size = config["training"]["batch_size"]
    best, stale = -1.0, 0
    history = []
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(config["training"]["epochs"]):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            prediction = model(batch["x"])
            fault = batch["fault"] > 0
            loss = (
                F.binary_cross_entropy_with_logits(
                    prediction["fault_logit"], batch["fault"]
                )
                * config["training"]["fault_weight"]
            )
            if fault.any():
                labels = batch["roots"][fault]
                labels = labels / labels.sum(-1, keepdim=True)
                loss = (
                    loss
                    - (labels * prediction["root_logits"][fault].log_softmax(-1))
                    .sum(-1)
                    .mean()
                )
            loss = loss + config["training"]["step_weight"] * F.huber_loss(
                prediction["step"], batch["target"]
            )
            if model.invariant:
                loss = loss + config["training"]["domain_weight"] * F.cross_entropy(
                    prediction["domain_logits"], batch["domain"]
                )
            if (
                method == "causalscope"
                and config["ranking"]["counterfactual"]
                and config["training"]["score_mass_weight"]
            ):
                keys = [
                    metadata["domains"][int(index)]
                    for index in batch["domain"].detach().cpu()
                ]
                ranked = score_components(
                    model,
                    batch["x"],
                    keys,
                    calibration,
                    config["ranking"],
                    full_candidates=True,
                )
                loss = (
                    loss
                    + config["training"]["score_mass_weight"]
                    * ranked["scores"].abs().sum(-1).mean()
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(batch["x"])
            count += len(batch["x"])
        score, _ = validation_f1(
            model, validation_data, device, batch_size, len(schema["components"])
        )
        history.append(
            {"epoch": epoch + 1, "loss": total / count, "validation_macro_f1": score}
        )
        pd.DataFrame(history).to_csv(output / "training.csv", index=False)
        print(
            f"epoch={epoch + 1} loss={total / count:.5f} validation_macro_f1={score:.5f}",
            flush=True,
        )
        if score > best:
            best, stale = score, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "schema": schema,
                    "graph": graph,
                    "domains": metadata["domains"],
                    "config": config,
                    "method": method,
                    "seed": seed,
                    "temperature": 1.0,
                    "calibration": calibration.state,
                    "validation_macro_f1": best,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                output / "model.pt",
            )
        else:
            stale += 1
        if stale >= config["training"]["patience"]:
            break
    model, checkpoint = model_from_checkpoint(output / "model.pt", device)
    _, logits = validation_f1(
        model, validation_data, device, batch_size, len(schema["components"])
    )
    checkpoint["temperature"] = calibrate_temperature(
        logits, validation_data.arrays["roots"]
    )
    if config["ranking"].get("tune_thresholds", True):
        calibrate_decisions(model, validation_data, checkpoint, device)
    torch.save(checkpoint, output / "model.pt")
    write_json(
        output / "selection.json",
        {
            "seed": seed,
            "best_validation_macro_f1": best,
            "temperature": checkpoint["temperature"],
            "parameter_count": checkpoint["parameter_count"],
            "decision_calibration": checkpoint.get("decision_calibration"),
        },
    )
    return str(output / "model.pt")
