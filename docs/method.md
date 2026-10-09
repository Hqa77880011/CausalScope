# Method and implementation conventions

## Core computation

For a metric `i`, healthy calibration uses

```text
z_i = (x_i - median(H_i)) / (1.4826 * MAD(H_i) + 1e-6).
```

Calibration is fitted per workload, batch size, and rank count. Missingness and sample age accompany each value. Calibration executions are independent of model training and validation.

The default temporal branches use causal convolutions with dilations `(1, 2, 4)` and `(1, 4, 16)`, each producing 32 channels. Their concatenation receives layer normalization. Node-type embeddings distinguish component, phase, and rank nodes. Two graph layers use separate self, prior, and lagged transforms. A lagged edge shifts source representations into the destination's time axis by its stored positive lag; no future sample is read. Message coefficients normalize each relation's incoming weights and apply the prior/data mixture.

The final causal time position supplies each node state. Attention pools owned nodes into component representations. A second attention pool combines component and phase representations. Component logits share a linear root head; graph representations feed the step-time, domain, and fault heads.

The root objective is cross-entropy on faulty windows. If a training execution contains multiple roots, their normalized multi-hot vector defines a soft categorical target. Healthy windows do not receive a fabricated root component. They train Huber step regression, domain classification, and a binary fault head. The fault head supplies the fault probability used for healthy-state inspection; recommendation abstention follows the root-probability, replacement-gain, and missing-node rules.

The encoder receives a reversed domain gradient while the domain classifier minimizes its loss. AdamW minimizes root loss plus `0.5 * step_loss`, `0.2 * domain_loss` through gradient reversal, `1.0 * fault_loss`, and `1e-4 * score_mass`. Domain loss is disabled for the non-invariant variants. The score-mass term is the L1 mass of the fused nonnegative score vector; it is nearly constant when replacement gains are positive and is not treated as a sparsity objective.

For each owned metric, a ridge regression with coefficient penalty 1 estimates its healthy normalized value from incoming prior parents outside the component being replaced. Metrics without enough observed parent/child samples use their healthy normalized mean. This excludes internal group parents so that a group's replacement never conditions on its own faulty nodes. Replacement resets the group's value, missingness, and age features; every outside node retains its observation.

Inference evaluates the five largest root-probability candidates. Their nonnegative step-prediction decreases are fused as

```text
R_c = 0.55 * p_c + 0.45 * s_c / (sum(s) + 1e-6).
```

When all gains are zero, ordering follows root probabilities. For two-fault ranking, replace the first component, recompute the scores, exclude the first component from further selection, and take the next component. This is an input intervention, with unchanged downstream measurements.

Abstention uses the paper's defaults: maximum root probability below 0.45, maximum replacement gain below 0.05 normalized step units, or more than 20 percent missing nodes. A node is missing for this rule when more than half its window samples are marked missing. These thresholds are configuration values frozen before test; temperature scaling uses only validation single faults. Recommendation replay additionally requires three consecutive accepted leading ranks.

## Choices affecting experimental interpretation

The paper does not provide source code, telemetry traces, selected ImageNet classes, BERT objective, conditional baseline architecture, lag-correlation admission threshold, exact baseline architectures, component-pair list, or execution partitions. The repository makes these choices explicit in code and configuration.

* The default graph has 19 scalar metric nodes. Rank merging takes the minimum clock, loader queue, progress, and read bandwidth and the maximum of other metrics. This represents component symptoms across ranks in one fixed schema. Imported per-rank graphs can preserve separate rank nodes. The default schema therefore does not identify the faulty rank within a component class.
* Current step time is a regression target and is excluded from the encoder and its graph. Including it as input would make its own replacement-sensitive prediction uninformative. Forward, backward, data, optimizer, all-reduce, and barrier phases remain inputs and join graph pooling.
* The collection loop serializes forward, backward, gradient communication, optimizer update, and barrier timing. DDP's automatic gradient synchronization is suppressed for a step, and explicit averaged collectives follow backward. This gives disjoint phase measurements and differs from overlapped production DDP. `readiness_skew` is barrier residence time; `loader_queue` is the bounded outer prefetch queue rather than private DataLoader internals. NVML GPU utilization is the utilization counter available from NVML, not an SM-active profiler trace.
* Relative collector clocks start after a distributed barrier. Multi-host collection assumes synchronized launch and sufficiently aligned hosts for a 200 ms grid. Import externally synchronized timestamps if tighter alignment is needed. The collector records a completed phase until its next completed update; no unfinished duration is invented.
* The default minimum segment duration is 30 seconds. The original 60/120-step schedule can be selected with `--segment-seconds 0`; fast jobs may otherwise finish before enough model windows exist.
* The localization TCN, LSTM, and graph variants share the 64-channel width and stopping rule. Parameter counts are recorded rather than assumed equal. The LSTM uses a single unidirectional layer. The GNN baseline keeps temporal and graph encoding while disabling domain reversal and replacement ranking.
* PCA, Isolation Forest, and Autoencoder consume per-node window summaries: mean, standard deviation, 5th/95th percentiles, last value, and missing fraction. PCA retains 95 percent explained variance; Isolation Forest fits one 100-tree model per metric; the Autoencoder uses a 64-channel hidden layer and a 32-channel bottleneck. They learn from healthy training windows. Reconstruction/metric anomaly scores average within component ownership groups. Static Threshold averages the excess absolute normalized value above three.
* Calibration is fixed during training, evaluation, and replay. The implementation exposes the healthy 75th-percentile step statistic and fault probability, but does not automatically update baselines from suspected healthy deployment windows. A new configuration requires new independent calibration recordings and a newly prepared model artifact.
* Main metrics aggregate fault-labeled windows, and detection delay aggregates executions. Single-fault Top-3 uses the initial fused ranking; dual-fault metrics use the iterative ranking. Five seeds quantify initialization variation; seed averages and correlated window counts do not provide execution-level confidence intervals. Per-execution predictions are retained for additional analysis. Fault detection AUROC and component localization AUROC are separate concepts; the primary AUROC reported here is macro one-vs-rest component localization AUROC.
* The overhead runner uses duplicated channels for node-count scaling. Process RSS includes Python, PyTorch, and model memory. Collector CPU measures the sampling thread; collection RSS includes the workload. Paired throughput loss keeps the serialized phase instrumentation constant and measures the extra sampling overhead. Report these definitions alongside measurements.

## Ablations

| Suite name | Change |
|---|---|
| `no_prior` | Remove prior messages, retaining the original 0.3 lag-message coefficient |
| `prior_only` | Prior graph only, with incoming prior weights normalized to one |
| `data_only` | Lagged graph only, with incoming data weights normalized to one |
| `no_invariance` | Disable gradient reversal and domain loss |
| `no_counterfactual` | Rank by calibrated root probability and omit the gain abstention gate |
| `lstm_encoder` | Replace both TCN branches with a 64-channel unidirectional LSTM |
| `no_iterative` | Use the initial ranking for both component selections |

The two prior-removal rows in the paper overlap in their descriptions. Here they differ only in whether the original lag mixture coefficient is retained or renormalized. All other settings stay fixed. The no-counterfactual ablation retains the step regression head for a controlled training comparison.
