import json
from pathlib import Path

import yaml


def read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_schema(path):
    schema = read_yaml(path)
    names = [metric["name"] for metric in schema["metrics"]]
    components = schema["components"]
    if len(names) != len(set(names)) or len(components) != len(set(components)):
        raise ValueError("Schema names must be unique")
    if schema["target"] in names:
        raise ValueError("The step-time target cannot also be an input metric")
    for component in components:
        if not any(metric["owner"] == component for metric in schema["metrics"]):
            raise ValueError(f"Component {component} has no owned metrics")
    for metric in schema["metrics"]:
        if metric["owner"] is not None and metric["owner"] not in components:
            raise ValueError(f"Unknown owner: {metric['owner']}")
        if metric["kind"] not in ("component", "phase", "rank"):
            raise ValueError(f"Unknown node kind: {metric['kind']}")
    for source, destination in schema["prior_edges"]:
        if source not in names or destination not in names or source == destination:
            raise ValueError(f"Invalid prior edge: {source} -> {destination}")
    return schema


def domain_key(run):
    return f"{run['workload']}|batch={run['batch_size']}|ranks={run['world_size']}"
