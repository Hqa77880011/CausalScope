# Experiment protocols

These commands operate on measured recordings. They write results only when executed; the repository contains no populated experimental tables.

| Paper experiment | Command | Primary output |
|---|---|---|
| Main localization comparison, Table II and main-accuracy figure | `suite --ablations`, then `paper-report` | `main_results.csv`, `main_accuracy.png` |
| Graph, invariance, scoring, and temporal-encoder ablations, Table III | `suite --ablations`, then `paper-report` | `ablation.csv`, `ablation.png` |
| Simultaneous faults and iterative intervention, Table IV | Same suite, with dual-fault test recordings | `multifault.csv`, `multifault.png` |
| Per-fault accuracy, Table V | `evaluate`, then `paper-report` | `breakdown.csv`, `fault_breakdown.csv`, `fault_breakdown.png` |
| Four held-out workloads, cross-workload figure | `lowo` | `cross_workload.csv`, `cross_workload_mean.csv`, `cross_workload.png` |
| Network-to-all-reduce-to-barrier propagation figure | `timeline` on a measured network incident | 5 Hz normalized network, communication, barrier, and step traces |
| Inference latency and memory scaling | `overhead --sizes 24 36 48 60`, then `plot-overhead` | `overhead.json`, latency/memory figure |
| Monitor CPU, memory, and throughput loss | `scripts/measure_collection.py` or `collection-cost` on paired recordings | `collection_cost.json`, `collection_cost.csv` |
| Stable localization delay | `evaluate` | `incidents.json`, detection rate and delay in `metrics.json` |

The collection matrix covers four workloads, one/two/four ranks, seven single-root classes, three intensities, eight single-fault repetitions, and six dual-root pairs with three intensity combinations and five repetitions. The repository's pair list and same-level combinations are implementation choices because the paper does not enumerate the pairs. Healthy calibration executions remain independent of training, validation, and test.

The combined report also separates rank count, intensity level, and fault pair in `rank_scale.csv`, `intensity.csv`, and `fault_pairs.csv`. These inspect the dimensions of the paper's execution matrix. They do not add a new performance claim.

## Selection and aggregation

Every neural method first selects its epoch from visible-workload validation macro-F1, then fits validation temperature scaling, then selects its abstention thresholds using the same visible validation partition. Threshold selection includes healthy windows so that accepting a spurious root contributes false-positive component predictions. No held-out faulty recording participates in selection. The test checkpoint includes the chosen thresholds and the healthy calibration state.

Main comparison, ablation, and dual-fault tables average evaluated seed reports within each method. The cross-workload table first averages seeds within each held-out workload. Its final mean gives every available workload fold equal weight. `summary.json` lists the actual folds and seeds supplied; an incomplete set is not expanded with invented values.

Detection delay uses the recommendation time at the third consecutive accepted correct leading rank. Undetected executions remain in the incident table and detection-rate denominator. The reported mean/median delay uses detected executions. Classification scores aggregate windows; overlapping windows do not become independent executions.

CPU cost is reported both relative to one logical core and to each host's logical CPU capacity. Collection memory is the sum of rank-process RSS at workload completion, together with its signed paired difference. Inference memory is process RSS at each graph size, plus CUDA peak allocated memory when CUDA is used. Paired collection measurements keep phase instrumentation constant and isolate the extra sampling cost.

The paper does not report score-only performance, a fusion-weight sweep, execution-level confidence intervals, calibration error, or measured recovery after a physical repair. Those are outside this experiment protocol. Automatic baseline refresh and deployment drift evaluation also remain outside the offline experiments; the model uses independently verified healthy calibration.
