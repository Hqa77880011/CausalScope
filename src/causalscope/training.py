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
    torch.save(checkpoint, output / "model.pt")
    write_json(
        output / "selection.json",
        {
            "seed": seed,
            "best_validation_macro_f1": best,
            "temperature": checkpoint["temperature"],
            "parameter_count": checkpoint["parameter_count"],
        },
    )
    return str(output / "model.pt")
