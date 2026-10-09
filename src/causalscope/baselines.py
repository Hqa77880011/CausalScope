from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowDataset
from .schema import read_json
from .training import set_seed


ANOMALY_METHODS = ("threshold", "pca", "isolation_forest", "autoencoder")


def summaries(x):
    observed = np.where(x[..., 1] == 0, x[..., 0], 0)
    return np.stack(
        [
            observed.mean(-1),
            observed.std(-1),
            np.quantile(observed, 0.05, axis=-1),
            np.quantile(observed, 0.95, axis=-1),
            observed[..., -1],
            x[..., 1].mean(-1),
        ],
        axis=-1,
    ).astype(np.float32)


class Autoencoder(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, inputs),
        )

    def forward(self, x):
        return self.layers(x)


def train_anomaly(prepared, output, config, method, seed, device):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = WindowDataset(prepared / "train.npz")
    validation = WindowDataset(prepared / "validation.npz")
    schema = read_json(prepared / "metadata.json")["schema"]
    healthy = train.arrays["fault"] == 0
    if healthy.sum() < 2:
        raise ValueError("Anomaly baselines require healthy training windows")
    features = summaries(train.arrays["x"][healthy])
    flat = features.reshape(len(features), -1)
    set_seed(seed)
    artifact = {"method": method, "schema": schema, "config": config, "seed": seed}
    if method == "pca":
        artifact["estimator"] = PCA(n_components=0.95, svd_solver="full").fit(flat)
    elif method == "isolation_forest":
        artifact["estimators"] = [
            IsolationForest(
                n_estimators=100, contamination="auto", random_state=seed
            ).fit(features[:, i])
            for i in range(features.shape[1])
        ]
    elif method == "autoencoder":
        model = Autoencoder(flat.shape[1], config["model"]["hidden"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        val_healthy = validation.arrays["fault"] == 0
        if not val_healthy.any():
            raise ValueError("Autoencoder needs healthy validation windows")
        val = summaries(validation.arrays["x"][val_healthy]).reshape(
            val_healthy.sum(), -1
        )
        val = torch.from_numpy(val).to(device)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(flat)),
            batch_size=config["training"]["batch_size"],
            shuffle=True,
        )
        best, stale = float("inf"), 0
        history = []
        for epoch in range(config["training"]["epochs"]):
            model.train()
            for (batch,) in loader:
                batch = batch.to(device)
                loss = F.mse_loss(model(batch), batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.inference_mode():
                value = float(F.mse_loss(model(val), val))
            history.append({"epoch": epoch + 1, "validation_mse": value})
            if value < best:
                best, stale = value, 0
                artifact["state_dict"] = {
                    key: tensor.detach().cpu().clone()
                    for key, tensor in model.state_dict().items()
                }
            else:
                stale += 1
            if stale >= config["training"]["patience"]:
                break
        pd.DataFrame(history).to_csv(output / "training.csv", index=False)
        artifact["inputs"] = flat.shape[1]
    joblib.dump(artifact, output / "baseline.joblib")
    return str(output / "baseline.joblib")


def anomaly_scores(artifact, x, device="cpu"):
    features = summaries(x)
    method = artifact["method"]
    if method == "threshold":
        nodes = np.maximum(np.abs(np.where(x[..., 1] == 0, x[..., 0], 0)) - 3, 0).mean(
            -1
        )
    elif method == "isolation_forest":
        nodes = np.stack(
            [
                -forest.score_samples(features[:, i])
                for i, forest in enumerate(artifact["estimators"])
            ],
            axis=1,
        )
    else:
        flat = features.reshape(len(x), -1)
        if method == "pca":
            estimator = artifact["estimator"]
            reconstructed = estimator.inverse_transform(estimator.transform(flat))
        else:
            model = artifact.get("_model")
            if model is None:
                model = Autoencoder(
                    artifact["inputs"], artifact["config"]["model"]["hidden"]
                ).to(device)
                model.load_state_dict(artifact["state_dict"])
                model.eval()
                artifact["_model"] = model
            with torch.inference_mode():
                reconstructed = model(torch.from_numpy(flat).to(device)).cpu().numpy()
        nodes = ((flat - reconstructed) ** 2).reshape(features.shape).mean(-1)
    component_scores = np.stack(
        [
            nodes[
                :,
                [
                    i
                    for i, metric in enumerate(artifact["schema"]["metrics"])
                    if metric["owner"] == component
                ],
            ].mean(-1)
            for component in artifact["schema"]["components"]
        ],
        axis=1,
    )
    return component_scores
