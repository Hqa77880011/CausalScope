# CausalScope

CausalScope is an implementation of **CausalScope: Workload-Invariant Root-Cause Localization for Distributed AI Training via Telemetry Graphs**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data collection and preparation

Workload collection uses the optional dependency group. Obtain [ImageNet](https://image-net.org/download.php) under its access terms with `train/<class>` and `val/<class>` folders. Create `data/imagenet100-classes.txt` with 100 class folder names, one per line; keep that list fixed. Text preparation downloads WikiText-103. Collection requires Linux resource controls; set interface/cgroup/block device to the actual test host before executing the collection plan. See [data.md](docs/data.md) for the telemetry/manifest schema.

```bash
python -m pip install -e ".[workloads]"
causalscope prepare-vision --source data/raw/imagenet --classes data/imagenet100-classes.txt --output data/imagenet100
causalscope prepare-text --output data/wikitext103 --block-size 128
causalscope plan --output data/plan.json
python scripts/collect_plan.py --plan data/plan.json --vision data/imagenet100 --text data/wikitext103 --output data/recordings --interface eth0 --cgroup /sys/fs/cgroup/causalscope --block-device 259:0
causalscope merge --recordings data/recordings --output data/telemetry
causalscope prepare --manifest data/telemetry/manifest.json --output data/prepared
```

## Training and localization

Use `configs/default.yaml` and `configs/schema.yaml` consistently. Collection seeds are retained in recordings; the localization model seed is explicit.

```bash
causalscope train --prepared data/prepared --config configs/default.yaml --seed 0 --device cpu --output runs/causalscope
causalscope localize --telemetry data/telemetry/resnet50-r2-dataloader-l0-0-telemetry.csv --steps data/telemetry/resnet50-r2-dataloader-l0-0-steps.csv --workload resnet50 --batch-size 32 --world-size 2 --checkpoint runs/causalscope/model.pt --output results/incident
```

For a new incident, replace the telemetry/step CSVs and workload, batch size and rank count with its actual collection metadata. The [input schema](docs/data.md) specifies required columns.

## Evaluation

```bash
causalscope evaluate --prepared data/prepared --checkpoint runs/causalscope/model.pt --device cpu --output results/causalscope
causalscope plot --inputs results/causalscope --output results/plots
```

## Comparisons and ablations

The suite fits and evaluates each requested method/ablation on the shared manifest; its seeds are explicit. Cross-workload and collection-overhead commands are in [experiments.md](docs/experiments.md).

```bash
causalscope suite --manifest data/telemetry/manifest.json --seeds 0 1 2 --ablations --device cpu --output runs/comparison
causalscope paper-report --inputs runs/comparison --output results/comparison
```

## Outputs

Collection produces rank telemetry/step CSVs and metadata. Merging writes a manifest with independent execution groups; preparation writes split NPZs, calibration and graph files. Training writes `model.pt` and `training.csv`; evaluation writes `predictions.json`, `metrics.json`, incident and breakdown files. Reports/plots read these computed records. Method conventions remain in [method.md](docs/method.md).

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0, PyYAML 6.0.2. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
