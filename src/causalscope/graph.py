import numpy as np


def build_graph(windows, schema, config):
    names = [metric["name"] for metric in schema["metrics"]]
    n = len(names)
    if config["mode"] not in ("hybrid", "prior", "data", "no_prior"):
        raise ValueError("Graph mode must be hybrid, prior, data, or no_prior")
    edges = []
    if config["mode"] not in ("data", "no_prior"):
        edges.extend(
            {
                "source": names.index(source),
                "destination": names.index(destination),
                "type": "prior",
                "lag": 0,
                "weight": 1.0,
            }
            for source, destination in schema["prior_edges"]
        )
    if config["mode"] != "prior":
        counts = np.zeros((n, n), dtype=int)
        scores = np.zeros((n, n))
        lag_votes = np.zeros((n, n, config["max_lag"] + 1), dtype=int)
        for window in windows:
            best = np.zeros((n, n))
            lags = np.zeros((n, n), dtype=int)
            for source in range(n):
                for destination in range(n):
                    if source == destination:
                        continue
                    for lag in range(
                        1, min(config["max_lag"], window.shape[1] - 3) + 1
                    ):
                        valid = (window[source, :-lag, 1] == 0) & (
                            window[destination, lag:, 1] == 0
                        )
                        a = window[source, :-lag, 0][valid]
                        b = window[destination, lag:, 0][valid]
                        if len(a) < 4 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
                            continue
                        correlation = float(abs(np.corrcoef(a, b)[0, 1]))
                        if correlation > best[source, destination]:
                            best[source, destination] = correlation
                            lags[source, destination] = lag
            for destination in range(n):
                candidates = np.argsort(-best[:, destination], kind="stable")[
                    : config["incoming_cap"]
                ]
                for source in candidates:
                    if best[source, destination] >= config["min_correlation"]:
                        counts[source, destination] += 1
                        scores[source, destination] += best[source, destination]
                        lag_votes[source, destination, lags[source, destination]] += 1
        for destination in range(n):
            stable = np.flatnonzero(
                counts[:, destination] / len(windows) >= config["stability"]
            )
            means = scores[:, destination] / np.maximum(counts[:, destination], 1)
            selected = stable[
                np.argsort(-means[stable], kind="stable")[: config["incoming_cap"]]
            ]
            for source in selected:
                edges.append(
                    {
                        "source": int(source),
                        "destination": destination,
                        "type": "lagged",
                        "lag": int(lag_votes[source, destination].argmax()),
                        "weight": float(means[source]),
                    }
                )
    for relation in ("prior", "lagged"):
        relation_edges = [edge for edge in edges if edge["type"] == relation]
        mixture = (
            1.0
            if config["mode"] not in ("hybrid", "no_prior")
            else (
                config["prior_weight"]
                if relation == "prior"
                else 1 - config["prior_weight"]
            )
        )
        for destination in range(n):
            incoming = [
                edge for edge in relation_edges if edge["destination"] == destination
            ]
            total = sum(edge["weight"] for edge in incoming)
            for edge in incoming:
                edge["alpha"] = mixture * edge["weight"] / total
    return {"nodes": names, "edges": edges, "config": config}
