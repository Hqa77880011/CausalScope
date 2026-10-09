import numpy as np
from sklearn.linear_model import Ridge

from .schema import domain_key, read_json, write_json


class HealthyCalibration:
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, runs, aligned, schema, config):
        state = {"schema": schema, "config": config, "domains": {}}
        names = [metric["name"] for metric in schema["metrics"]]
        for key in sorted({domain_key(run) for run in runs}):
            records = [aligned[run["id"]] for run in runs if domain_key(run) == key]
            values = np.concatenate([record["values"] for record in records])
            missing = np.concatenate([record["missing"] for record in records])
            median = np.zeros(len(names))
            scale = np.ones(len(names))
            for i in range(len(names)):
                observed = values[(missing[:, i] == 0) & np.isfinite(values[:, i]), i]
                if len(observed):
                    median[i] = np.median(observed)
                    scale[i] = (
                        1.4826 * np.median(np.abs(observed - median[i]))
                        + config["epsilon"]
                    )
            target = np.concatenate([record["target"] for record in records])
            target = target[np.isfinite(target)]
            if len(target) < 4:
                raise ValueError(f"Insufficient healthy step samples for {key}")
            target_median = float(np.median(target))
            target_scale = float(
                1.4826 * np.median(np.abs(target - target_median)) + config["epsilon"]
            )
            z = np.nan_to_num((values - median) / scale, nan=0.0)
            regressions = []
            for i, metric in enumerate(schema["metrics"]):
                parents = [
                    names.index(source)
                    for source, dest in schema["prior_edges"]
                    if dest == metric["name"]
                    and schema["metrics"][names.index(source)]["owner"]
                    != metric["owner"]
                ]
                valid = (missing[:, i] == 0) & np.isfinite(values[:, i])
                if parents:
                    valid &= np.all(missing[:, parents] == 0, axis=1)
                if parents and valid.sum() >= len(parents) + 2:
                    regression = Ridge(alpha=1.0).fit(z[valid][:, parents], z[valid, i])
                    regressions.append(
                        {
                            "parents": parents,
                            "coef": regression.coef_.tolist(),
                            "intercept": float(regression.intercept_),
                        }
                    )
                else:
                    regressions.append(
                        {
                            "parents": [],
                            "coef": [],
                            "intercept": float(np.mean(z[valid, i]))
                            if valid.any()
                            else 0.0,
                        }
                    )
            state["domains"][key] = {
                "median": median.tolist(),
                "scale": scale.tolist(),
                "target_median": target_median,
                "target_scale": target_scale,
                "target_q75": float(np.quantile(target, 0.75)),
                "regressions": regressions,
            }
        if not state["domains"]:
            raise ValueError(
                "Provide independent healthy calibration executions for every domain"
            )
        return cls(state)

    def transform(self, key, record):
        if key not in self.state["domains"]:
            raise ValueError(f"No workload-matched healthy reference for {key}")
        baseline = self.state["domains"][key]
        z = (record["values"] - np.array(baseline["median"])) / np.array(
            baseline["scale"]
        )
        z = np.nan_to_num(z, nan=0.0)
        age = np.minimum(record["age"] / self.state["config"]["window_seconds"], 1.0)
        return np.stack([z, record["missing"], age], axis=-1).astype(np.float32)

    def normalize_target(self, key, value):
        baseline = self.state["domains"][key]
        return (value - baseline["target_median"]) / baseline["target_scale"]

    def replacement(self, x, keys, component):
        import torch

        result = x.clone()
        schema = self.state["schema"]
        for batch, key in enumerate(keys):
            models = self.state["domains"][key]["regressions"]
            for i, metric in enumerate(schema["metrics"]):
                if metric["owner"] != component:
                    continue
                model = models[i]
                value = torch.full_like(x[batch, i, :, 0], model["intercept"])
                if model["parents"]:
                    coef = x.new_tensor(model["coef"])
                    value = value + (
                        x[batch, model["parents"], :, 0].transpose(0, 1) * coef
                    ).sum(-1)
                result[batch, i, :, 0] = value
                result[batch, i, :, 1:] = 0
        return result

    def save(self, path):
        write_json(path, self.state)

    @classmethod
    def load(cls, path):
        return cls(read_json(path))
