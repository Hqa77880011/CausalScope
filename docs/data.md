# Telemetry input

The default schema is `configs/schema.yaml`. Metrics are scalar nodes with a component owner or an unowned phase role. `step_ms` is a separate regression target. Timestamps are seconds relative to the start of an execution, aligned across collectors and ranks.

## Manifest

A manifest is a nonempty JSON list. Paths are relative to the manifest directory. Each execution has a unique ID and one partition. The following is a format example; it contains no measured results:

```json
[
  {
    "id": "resnet50-reference-0",
    "workload": "resnet50",
    "batch_size": 32,
    "world_size": 2,
    "split": "calibration",
    "roots": [],
    "fault_start": null,
    "fault_end": null,
    "telemetry": "reference-telemetry.csv",
    "steps": "reference-steps.csv"
  },
  {
    "id": "resnet50-network-0",
    "workload": "resnet50",
    "batch_size": 32,
    "world_size": 2,
    "split": "test",
    "roots": ["network"],
    "fault_start": 30.0,
    "fault_end": 90.0,
    "telemetry": "network-telemetry.csv",
    "steps": "network-steps.csv"
  }
]
```

All four partitions must be present to train and evaluate: `calibration`, `train`, `validation`, and `test`. Healthy training and validation executions are needed by anomaly baselines and the fault-probability head. A calibration execution must be verified healthy and must not be reused as a training, validation, or test execution. Use a distinct ID and recording for every execution; changing the ID does not turn a reused recording into an independent sample.

`roots` is empty for healthy recordings and contains one or two component names for fault recordings. `fault_start` and `fault_end` are the actual applied interval, in seconds. The injected component is the label. No label is inferred from anomaly magnitude. Collector-generated manifests also preserve `faults` (component-to-intensity mapping), `intensity_level` (zero-based level 0–2), and `execution_seed`. Imported manifests can supply the same fields; unspecified intensities are omitted from intensity-specific tables.

## CSV files

Telemetry columns are:

```text
timestamp,metric,value
```

Each row carries one raw scalar observation in the units specified by the schema. Optional collector columns such as `sequence` and `rank` are allowed. An omitted row or a nonfinite value means a missing observation. The `merge` command aggregates rank recordings; directly imported CSVs should already have the desired node names and aggregation.

Steps columns are:

```text
timestamp,step_ms
```

A timestamp is the completion time of a step. The collector also records the integer `step` index. Distributed merging takes the maximum step duration and completion time across ranks, rather than summing overlapping rank durations.

For a per-rank graph, name metrics explicitly, such as `rank0.gpu_clock` and `rank1.gpu_clock`, and define their ownership and prior edges in a custom schema. Use the same schema, node ordering, units, and rank capacity across the recordings in that model. Pass it to `prepare --schema PATH` or `suite --schema PATH`.

## Preparation outputs

| Path | Meaning |
|---|---|
| `metadata.json` | Exact schema, configuration, visible domains, held-out workload |
| `calibration.json` | Healthy location/scale statistics, target statistics, conditional replacement coefficients |
| `graph.json` | Directed edges with type, positive lag, raw weight, and normalized message coefficient |
| `train.npz`, `validation.npz`, `test.npz` | Input tensors, regression targets, per-sample normalized step traces, component labels, fault labels, and visible-domain indices |
| `train.json`, `validation.json`, `test.json` | Execution identity, domain, window interval, and fault interval for each tensor |

Input tensors have shape `[windows, nodes, samples, 3]`. The features are robust-normalized value, missingness bit, and age divided by window duration, capped at one. Carry-forward observations become missing after two grid positions; their last value remains with a missingness bit. Values before the first observation are initialized to the healthy median. Completely unavailable healthy counters use zero location and unit scale while remaining missing.

The step target is the mean aligned observed step duration over the window, normalized against its matched healthy step statistics. Before-fault windows are healthy. Windows ending during an applied fault interval are labeled with the injected roots, including transition windows. Windows ending after fault removal are excluded from the localization datasets. Separate incident recordings can still be passed to `localize` for complete replay.

Overlapping windows remain within their execution's partition. A held-out fold moves every non-calibration execution of the target workload to test, excludes visible-workload test executions from that fold, and excludes visible-workload dual faults from training. Its test split therefore contains only the held-out workload. Its healthy references are used for normalization and replacement, while lagged-edge discovery uses only visible-workload calibration windows.

Retain raw recordings alongside their applied-fault metadata to interpret observations. Recordings, prepared arrays, and calibration artifacts belong under `data/` and are ignored by Git.
