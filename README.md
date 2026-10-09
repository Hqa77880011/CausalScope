# CausalScope

CausalScope is an implementation of **CausalScope: Workload-Invariant Root-Cause Localization for Distributed AI Training via Telemetry Graphs**. It ranks CPU, DataLoader, GPU compute, network, storage, rank compute, and GPU memory faults from training telemetry.

The pipeline uses workload-matched healthy normalization, a two-branch temporal encoder, typed graph propagation, domain-adversarial learning, and component-level healthy replacement scores. It includes data collection, localization training, seven comparison methods, module ablations, cross-workload evaluation, incident replay, and result plots.

## 1. Install

Use Python 3.10 or newer. Localization training and inference support CPU and CUDA. Collecting the complete fault matrix requires Linux, NVIDIA GPUs, and a dedicated training environment with the required device and cgroup permissions.

```bash
git clone https://github.com/Hqa77880011/CausalScope.git
cd CausalScope
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
causalscope --help
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. Install the appropriate [PyTorch build](https://pytorch.org/get-started/locally/) before the package if you need a particular CUDA runtime. For workload collection and dataset preparation, also install:

```bash
python -m pip install -e '.[workloads]'
```

Torchvision must match the installed PyTorch version. The collector uses [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and `torchrun` for distributed jobs. It does not download pretrained model weights.

## 2. Prepare data

Localization consumes measured telemetry and independent healthy calibration executions. Either import your recordings using the [data format](docs/data.md), or collect them with the supplied workloads. The paper's telemetry recordings are not included in this repository.

Vision workloads use ImageNet-100. Obtain ImageNet through the [ImageNet download portal](https://www.image-net.org/download.php), arrange it as `train/<class>/...` and `val/<class>/...`, and provide a text file containing exactly 100 class directory names, one per line. The class list is an explicit dataset choice because the paper does not specify it. This command creates directory links, preserving the selected classes:

```bash
causalscope prepare-vision --source /datasets/imagenet --classes /datasets/imagenet100-classes.txt --output data/imagenet100
```

BERT uses [Salesforce WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext). The following command downloads the text and tokenizer, then writes fixed-length token blocks for masked language modeling:

```bash
causalscope prepare-text --output data/wikitext103 --block-size 128
```

The collection workloads are `resnet50`, `mobilenet_v3_large`, `vit_b_16`, and `bert_base`. Vision models train with 100 output classes; BERT-base uses masked language modeling with randomly initialized parameters.

## 3. Collect executions

For a single verified healthy execution:

```bash
torchrun --standalone --nproc_per_node=1 -m causalscope.cli collect \
  --workload resnet50 --dataset data/imagenet100 \
  --output data/recordings --run-id resnet50-calibration-0 \
  --split calibration --batch-size 32 --steps 200
```

Create additional healthy executions with distinct IDs for `train`, `validation`, and `test`. Calibration recordings must remain separate from these partitions. A fault example is:

```bash
torchrun --standalone --nproc_per_node=2 -m causalscope.cli collect \
  --workload resnet50 --dataset data/imagenet100 \
  --output data/recordings --run-id resnet50-loader-0 \
  --split train --batch-size 32 --faults '{"dataloader":50}'
```

`--fault-rank` selects the affected rank and defaults to zero. Faults start after at least 60 healthy steps and persist for at least 120 steps. By default, healthy, faulty, and recovery segments also last at least 30 seconds so that fast workloads provide complete 24-second windows. Set `--segment-seconds 0` to use the paper's exact step-count schedule. Each recording stores the actual interval, applied fault operations, package versions, and Git revision.

| Component | Mechanism | Accepted intensities | Extra requirements |
|---|---|---|---|
| `cpu` | cgroup v2 CPU quota | 75, 50, 25 percent of one core | `--cgroup` |
| `dataloader` | sleep once per worker batch collation | 20, 50, 100 ms | none |
| `gpu_compute` | NVML GPU clock lock | 1800, 1500, 1200 MHz | CUDA and clock-control permission |
| `network` | `tc` token-bucket bandwidth cap | 10, 5, 2.5 Gb/s | `--interface` and network-admin permission |
| `storage` | cgroup v2 read bandwidth limit | 800, 400, 200 MB/s | `--cgroup`, `--block-device MAJOR:MINOR` |
| `rank_compute` | sleep before the forward pass | 20, 50, 100 ms | none |
| `gpu_memory` | reserve GPU memory | 20, 35, 50 percent | sufficient free GPU memory |

CPU and storage faults require an existing delegated cgroup with `cpu.max`, `io.max`, and a writable `cgroup.procs`. The affected process joins this group before DataLoader workers start. Use a dedicated interface for bandwidth experiments. Fault controls restore modified quotas, read limits, and bandwidth caps when the job exits normally; GPU clock locks reset to the device default. Hardware support is checked when the corresponding control is applied. Network faults need inter-rank traffic to expose a training effect; a single-rank execution does not measure distributed communication degradation.

To construct the complete execution matrix without running it:

```bash
causalscope plan --output data/plan.json
```

The plan contains 480 healthy executions, 2,016 single-fault executions, and 1,080 dual-fault executions. Batch sizes, six component pairs, partitions, and seeds are stored in the plan and can be edited before collection. For a fully configured single-host environment, the runner executes selected plan entries sequentially:

```bash
python scripts/collect_plan.py --plan data/plan.json \
  --vision data/imagenet100 --text data/wikitext103 \
  --output data/recordings --interface eno1 \
  --cgroup /sys/fs/cgroup/causalscope --block-device 259:0
```

Replace interface and device identifiers with your experiment devices. Use `--start` and `--stop` to select an interval. On two hosts, select entries with two or four ranks, run the same interval on both hosts with `--nnodes 2`, distinct `--node-rank` values, and the same reachable `--master-addr`. Gather their rank files into one recording directory before merging. Existing execution IDs are never overwritten.

```bash
causalscope merge --recordings data/recordings --output data/telemetry
```

This writes `manifest.json`, long-format telemetry CSVs, and per-step CSVs. Check that fault intervals and missing counters match your recording conditions before preparing a dataset.

## 4. Prepare localization windows

```bash
causalscope prepare --manifest data/telemetry/manifest.json --output data/prepared
```

Preparation aligns samples to 200 ms, preserves missingness and sample age, fits healthy medians and conditional baselines, constructs the telemetry graph using healthy calibration windows, and writes execution-separated datasets. A configuration domain is the workload, per-rank batch size, and rank count. Every domain, including a held-out workload, needs its own healthy calibration recordings.

The default graph uses a 0.7 prior weight, positive lags up to 10 samples, a 0.5 correlation admission threshold, 70 percent window stability, and at most five incoming data-derived edges. The encoder uses 24-second windows with a two-second stride. Change these parameters in `configs/default.yaml` before preparation. `configs/schema.yaml` defines metric ownership, units, sources, and prior edges; schema and calibration are embedded in the model checkpoint.

## 5. Train and localize

```bash
causalscope train --prepared data/prepared --output runs/causalscope --device cuda:0
causalscope evaluate --prepared data/prepared --checkpoint runs/causalscope/model.pt --output results/causalscope --device cuda:0
```

The defaults are 64 hidden channels, dropout 0.2, AdamW at `1e-3`, weight decay `1e-4`, batch size 64, and at most 100 epochs. Early stopping uses mean validation macro-F1 across visible workloads with ten epochs of patience. The root probabilities receive validation-only temperature calibration. `--epochs`, `--batch-size`, and `--seed` override the corresponding training choices. Use `--device cpu` when CUDA is unavailable.

For an unlabeled incident recording in the same schema and a calibrated domain:

```bash
causalscope localize --telemetry data/incident-telemetry.csv --steps data/incident-steps.csv \
  --workload resnet50 --batch-size 32 --world-size 2 \
  --checkpoint runs/causalscope/model.pt --output results/incident
```

`rankings.json` contains component probabilities, replacement gains, rankings, and abstentions. `recommendations.json` emits the leading component after three consecutive accepted windows, together with its strongest owned metrics. A replacement gain is a decrease in predicted normalized step time; it is not a measured repair benefit.

## 6. Compare methods and run ablations

```bash
causalscope suite --manifest data/telemetry/manifest.json --output runs/comparison --device cuda:0
causalscope suite --manifest data/telemetry/manifest.json --output runs/ablations --methods causalscope --ablations --device cuda:0
```

The suite trains the requested methods using seeds 0–4, evaluates each checkpoint, and creates comparison plots. Override seeds with `--seeds 0 1`. Methods are Static Threshold (`threshold`), PCA (`pca`), Isolation Forest (`isolation_forest`), Autoencoder (`autoencoder`), LSTM (`lstm`), TCN (`tcn`), Telemetry GNN (`gnn`), and CausalScope (`causalscope`). Individual baselines also use `train --method NAME`; anomaly baselines save `baseline.joblib` instead of `model.pt`.

Ablations remove prior edges, use prior-only or data-only graphs, disable invariant learning, disable counterfactual ranking, replace the TCN with LSTM, or disable the second intervention pass. The settings and their precise meanings are documented in [implementation conventions](docs/method.md).

To evaluate a held-out workload:

```bash
causalscope suite --manifest data/telemetry/manifest.json --held-out vit_b_16 \
  --output runs/heldout-vit --methods causalscope --ablations --device cuda:0
```

Run a fold for each of the four workload names. Target faulty executions never train the model or graph. The target's independent healthy recordings provide normalization and replacement calibration. Held-out accuracy therefore measures localization with healthy target calibration.

## 7. Inspect metrics and plots

`evaluate` writes:

| File | Contents |
|---|---|
| `metrics.json` | Top-1, Top-3, macro-F1, macro component AUROC; NDCG@2, exact Top-2, Cover@1; detection delay and inference latency |
| `predictions.json` | Per-window labels, ordered components, scores, abstention, probabilities, gains, and metric evidence |
| `incidents.json` | Per-execution stable-detection delay and undetected incidents |
| `breakdown.csv` | Per-workload and per-fault results |

Scores are fractions in `[0, 1]`; multiply by 100 for percentages. Single-fault windows supply classification metrics. Dual-fault windows supply ranking metrics. An abstention scores zero for the corresponding ranking endpoints. Detection delay ends when the same correct leading component appears in three accepted consecutive windows. Its mean includes detected executions; the detection rate and undetected count report the remaining incidents. AUROC averages one-vs-rest component scores over classes with both positive and negative labels. Overlapping windows are not independent experimental replicates.

```bash
causalscope plot --inputs results/causalscope --output results/plots
causalscope timeline --prepared data/prepared --predictions results/causalscope/predictions.json \
  --run-id resnet50-loader-0 --output results/plots/timeline.png
```

Plots include method comparisons, workload transfer, per-fault accuracy, and observed fault timelines. Multiple input directories are combined into `comparison.csv`; figures average the supplied seed reports.

## 8. Measure online cost

```bash
causalscope overhead --prepared data/prepared --checkpoint runs/causalscope/model.pt \
  --sizes 19 24 32 48 --device cpu --output results/overhead.json
causalscope plot-overhead --report results/overhead.json --output results/plots/overhead.png
```

The benchmark measures full ranking latency, including screened interventions and iterative ranking, after warmup. It reports median, mean, and 95th-percentile latency, process RSS, and CUDA peak allocation when applicable. Additional graph nodes duplicate measured channels to measure computational scaling; they do not represent additional physical GPUs or new localization experiments.

Collection reports include monitor thread CPU as a percentage of one core and mean/peak workload-process RSS. For throughput loss, collect matched healthy jobs with identical parameters, once with monitoring and once with `--no-monitor`, using separate recording directories. Keep the phase instrumentation constant. Supply their measured values in a CSV:

```text
pair_id,without_monitor_steps_per_second,with_monitor_steps_per_second,monitor_cpu_percent_one_core,process_rss_mb_mean
```

Append measurement rows, then pass the file as `overhead --pairs PATH`. Throughput loss is `100 * (1 - monitored / reference)`. No throughput or overhead results are populated in advance.

## Implementation details

[docs/method.md](docs/method.md) centralizes choices that affect input semantics, training, and result interpretation. In particular, step time is a regression target rather than an input, default telemetry nodes aggregate ranks by named metric, phase timing serializes computation and communication, and healthy calibration is frozen during offline evaluation. Custom per-rank schemas are supported for imported telemetry.

The CI workflow performs source lint, syntax compilation, package installation, CLI loading, and wheel creation. It does not run training, localization experiments, or experiment tests. Datasets, model weights, runtime outputs, and the manuscript are excluded from version control.

MIT license.
