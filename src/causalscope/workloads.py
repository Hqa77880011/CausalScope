import csv
import json
import multiprocessing
import os
import queue
import subprocess
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler, default_collate

from .collection import Monitor
from .faults import FaultController
from .schema import read_json, write_json
from .training import set_seed


def physical_gpu_id(local_rank):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if visible == [""]:
        return local_rank
    selected = visible[local_rank]
    return selected if selected.startswith(("GPU-", "MIG-")) else int(selected)


class TokenBlocks(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self.length = len(np.load(self.path, mmap_mode="r", allow_pickle=False))
        self.array = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        return torch.tensor(np.array(self.array[index]), dtype=torch.long)


class DelayedCollator:
    def __init__(self, delay, bert=False, vocab_size=30522):
        self.delay = delay
        self.bert = bert
        self.vocab_size = vocab_size

    def __call__(self, samples):
        if self.delay.value:
            time.sleep(self.delay.value)
        batch = default_collate(samples)
        if not self.bert:
            return batch
        original = batch.clone()
        eligible = (batch != 0) & (batch != 101) & (batch != 102)
        masked = (torch.rand(batch.shape) < 0.15) & eligible
        if not eligible.any():
            raise ValueError("A text batch has no maskable tokens")
        if not masked.any():
            position = eligible.nonzero()[0]
            masked[position[0], position[1]] = True
        labels = original.masked_fill(~masked, -100)
        draws = torch.rand(batch.shape)
        batch[masked & (draws < 0.8)] = 103
        random_positions = masked & (draws >= 0.8) & (draws < 0.9)
        batch[random_positions] = torch.randint(
            self.vocab_size, (int(random_positions.sum()),)
        )
        return {
            "input_ids": batch,
            "attention_mask": (original != 0).long(),
            "labels": labels,
        }


class PrefetchBatches:
    def __init__(self, loader, capacity=2):
        self.queue = queue.Queue(capacity)
        self.stop = threading.Event()
        self.loader = loader
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            for batch in self.loader:
                while not self.stop.is_set():
                    try:
                        self.queue.put((True, batch), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
            self.queue.put((True, None))
        except (OSError, RuntimeError, ValueError) as error:
            self.queue.put((False, error))

    def next(self):
        valid, batch = self.queue.get()
        if not valid:
            raise batch
        return batch

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)


def prepare_vision(source, classes_path, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    classes = [
        line.strip()
        for line in Path(classes_path).read_text().splitlines()
        if line.strip()
    ]
    if len(classes) != 100 or len(set(classes)) != 100:
        raise ValueError(
            "ImageNet-100 class file must contain 100 distinct class directory names"
        )
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output / split).mkdir(exist_ok=True)
        for name in classes:
            if name != Path(name).name or name in (".", ".."):
                raise ValueError("Class names must be single directory names")
            target = source / split / name
            if not target.is_dir():
                raise ValueError(f"Class folder missing: {target}")
            link = output / split / name
            if link.exists():
                raise FileExistsError(link)
            link.symlink_to(target, target_is_directory=True)
    (output / "classes.txt").write_text("\n".join(classes) + "\n")


def prepare_text(output, block_size=128):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if block_size < 4 or block_size > 512:
        raise ValueError("BERT block size must be between 4 and 512")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    corpus = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    for split in ("train", "validation", "test"):
        raw_path = output / f"{split}.bin"
        pending = []
        count = 0
        try:
            with raw_path.open("wb") as stream:
                for batch in corpus[split].iter(batch_size=1000):
                    encoded = tokenizer(
                        batch["text"], add_special_tokens=False, truncation=False
                    )["input_ids"]
                    for tokens in encoded:
                        pending.extend(tokens)
                        while len(pending) >= block_size - 2:
                            block = [
                                tokenizer.cls_token_id,
                                *pending[: block_size - 2],
                                tokenizer.sep_token_id,
                            ]
                            stream.write(np.asarray(block, dtype=np.int32).tobytes())
                            count += 1
                            del pending[: block_size - 2]
            if not count:
                raise ValueError(f"No text blocks in {split}")
            source = np.memmap(
                raw_path, mode="r", dtype=np.int32, shape=(count, block_size)
            )
            target = np.lib.format.open_memmap(
                output / f"{split}.npy",
                mode="w+",
                dtype=np.int32,
                shape=(count, block_size),
            )
            for start in range(0, count, 65536):
                target[start : start + 65536] = source[start : start + 65536]
            target.flush()
            del source, target
        finally:
            raw_path.unlink(missing_ok=True)
    write_json(
        output / "dataset.json",
        {
            "source": "Salesforce/wikitext",
            "configuration": "wikitext-103-raw-v1",
            "tokenizer": "google-bert/bert-base-uncased",
            "block_size": block_size,
        },
    )


def run_workload(args):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    recipe = (
        read_json(args.plan)[args.index]
        if args.plan
        else {
            "id": args.run_id,
            "workload": args.workload,
            "batch_size": args.batch_size,
            "world_size": world_size,
            "faults": json.loads(args.faults),
            "split": args.split,
            "seed": args.seed,
            "steps": args.steps,
            "fault_start_step": args.fault_start_step,
            "fault_steps": args.fault_steps,
        }
    )
    if recipe["id"] != Path(recipe["id"]).name or recipe["id"] in (".", ".."):
        raise ValueError("Execution id must be a single directory name")
    if args.workers < 0 or args.segment_seconds < 0 or recipe["steps"] < 1:
        raise ValueError("Invalid worker count, segment duration, or step count")
    if recipe["world_size"] != world_size:
        raise ValueError("torchrun rank count does not match the selected experiment")
    if (
        recipe["steps"] <= recipe["fault_start_step"] + recipe["fault_steps"]
        and recipe["faults"]
    ):
        raise ValueError("Record at least one step after fault removal")
    if not 0 <= args.fault_rank < world_size:
        raise ValueError("Fault rank is outside this job")
    set_seed(recipe["seed"])
    cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    output = Path(args.output) / recipe["id"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / f"metadata-rank{rank}.json").exists():
        raise FileExistsError(f"Execution already recorded: {output}")
    original_group = None
    selected_faults = recipe["faults"] if rank == args.fault_rank else {}
    context = multiprocessing.get_context("spawn")
    delay = context.Value("d", 0.0)
    faults = FaultController(
        selected_faults, delay, device, args.cgroup, args.block_device, args.interface
    )
    bert = recipe["workload"] == "bert_base"
    if bert:
        from transformers import BertConfig, BertForMaskedLM

        dataset = TokenBlocks(Path(args.dataset) / "train.npy")
        model = BertForMaskedLM(BertConfig()).to(device)
    else:
        from torchvision import datasets, models, transforms

        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        dataset = datasets.ImageFolder(
            Path(args.dataset) / "train", transform=transform
        )
        if len(dataset.classes) != 100:
            raise ValueError("Vision workloads require an ImageNet-100 subset")
        model = getattr(models, recipe["workload"])(weights=None, num_classes=100).to(
            device
        )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=recipe["seed"]
    )
    loader_args = {
        "batch_size": recipe["batch_size"],
        "sampler": sampler,
        "num_workers": args.workers,
        "collate_fn": DelayedCollator(delay, bert),
        "drop_last": True,
    }
    if args.workers:
        loader_args.update(multiprocessing_context=context, persistent_workers=True)
    loader = DataLoader(dataset, **loader_args)
    if len(loader) < 1:
        raise ValueError("Dataset is smaller than one batch per rank")
    prefetch = None
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank] if cuda else None
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if world_size > 1:
        dist.barrier()
    start = time.perf_counter()
    monitor = (
        Monitor(
            output / f"telemetry-rank{rank}.csv",
            start,
            rank,
            interface=args.interface,
            gpu=local_rank if cuda else None,
        )
        if not args.no_monitor
        else None
    )
    metadata = {
        **recipe,
        "rank": rank,
        "fault_rank": args.fault_rank,
        "fault_start": None,
        "fault_end": None,
        "torch": torch.__version__,
        "python_packages": {
            name: version(name) for name in ("numpy", "torch", "psutil")
        },
        "cuda": torch.version.cuda,
        "host": os.uname().nodename
        if hasattr(os, "uname")
        else os.environ.get("COMPUTERNAME"),
        "monitored": monitor is not None,
        "phase_mode": "serialized_gradient_communication",
        "injection_events": [],
        "completed": False,
        "num_workers": args.workers,
        "dataset": str(Path(args.dataset).resolve()),
        "logical_cpus": psutil.cpu_count(),
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True)
    metadata["git_revision"] = git.stdout.strip() if git.returncode == 0 else None
    if cuda:
        import pynvml

        pynvml.nvmlInit()
        metadata["driver"] = pynvml.nvmlSystemGetDriverVersion()
        metadata["nccl"] = torch.cuda.nccl.version() if world_size > 1 else None
        pynvml.nvmlShutdown()
    phase_values = {}
    step_rows = []
    epoch = 0

    def synchronize():
        if cuda:
            torch.cuda.synchronize()

    def timed(action):
        synchronize()
        before = time.perf_counter()
        value = action()
        synchronize()
        return value, (time.perf_counter() - before) * 1000

    fault_start_step = None
    step = 0
    try:
        if selected_faults.keys() & {"cpu", "storage"}:
            if not args.cgroup:
                raise ValueError("CPU/storage faults require --cgroup")
            group = Path(args.cgroup)
            original_group = Path("/sys/fs/cgroup") / Path(
                "/proc/self/cgroup"
            ).read_text().strip().split("::", 1)[1].lstrip("/")
            (group / "cgroup.procs").write_text(str(os.getpid()))
        prefetch = PrefetchBatches(loader)
        if monitor is not None:
            monitor.__enter__()
        while True:
            if monitor is not None and monitor.error is not None:
                raise RuntimeError("Telemetry collection failed") from monitor.error
            elapsed = time.perf_counter() - start
            begin = bool(
                recipe["faults"]
                and metadata["fault_start"] is None
                and step >= recipe["fault_start_step"]
                and elapsed >= args.segment_seconds
            )
            finish = bool(
                recipe["faults"]
                and metadata["fault_start"] is not None
                and metadata["fault_end"] is None
                and step >= fault_start_step + recipe["fault_steps"]
                and elapsed - metadata["fault_start"] >= args.segment_seconds
            )
            done = step >= recipe["steps"] and (
                elapsed >= args.segment_seconds
                if not recipe["faults"]
                else metadata["fault_end"] is not None
                and elapsed - metadata["fault_end"] >= args.segment_seconds
            )
            control = torch.tensor(
                [begin, finish, done], dtype=torch.int32, device=device
            )
            if world_size > 1:
                dist.broadcast(control, src=0)
            begin, finish, done = [bool(value) for value in control.tolist()]
            if done:
                break
            if begin:
                metadata["fault_start"] = time.perf_counter() - start
                fault_start_step = step
                metadata["actual_fault_start_step"] = step
                faults.start()
                metadata["injection_events"] = list(faults.events)
            if finish:
                faults.stop()
                metadata["fault_end"] = time.perf_counter() - start
                metadata["actual_fault_end_step"] = step
            step_start = time.perf_counter()
            batch, phase_values["data_ms"] = timed(prefetch.next)
            if batch is None:
                prefetch.close()
                epoch += 1
                sampler.set_epoch(epoch)
                prefetch = PrefetchBatches(loader)
                batch, phase_values["data_ms"] = timed(prefetch.next)
            queue_depth = prefetch.queue.qsize()
            transfer_start = time.perf_counter()
            if bert:
                batch = {key: value.to(device) for key, value in batch.items()}
            else:
                batch = [value.to(device) for value in batch]
            synchronize()
            phase_values["data_ms"] += (time.perf_counter() - transfer_start) * 1000
            optimizer.zero_grad(set_to_none=True)
            sync_context = model.no_sync() if world_size > 1 else None
            if sync_context is not None:
                sync_context.__enter__()
            try:

                def forward():
                    if faults.rank_sleep:
                        time.sleep(faults.rank_sleep)
                    if bert:
                        return model(**batch).loss
                    return torch.nn.functional.cross_entropy(model(batch[0]), batch[1])

                loss, phase_values["forward_ms"] = timed(forward)
                _, phase_values["backward_ms"] = timed(loss.backward)
            finally:
                if sync_context is not None:
                    sync_context.__exit__(None, None, None)

            def communication():
                if world_size > 1:
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad)
                            parameter.grad.div_(world_size)

            _, phase_values["allreduce_ms"] = timed(communication)
            _, phase_values["optimizer_ms"] = timed(optimizer.step)
            _, phase_values["barrier_ms"] = timed(
                dist.barrier if world_size > 1 else lambda: None
            )
            step_time = (time.perf_counter() - step_start) * 1000
            relative = time.perf_counter() - start
            phase_values.update(
                {
                    "rank_progress": 1000 / max(step_time, 1e-9),
                    "readiness_skew": phase_values["barrier_ms"],
                    "loader_queue": queue_depth,
                }
            )
            if monitor is not None:
                monitor.update(phase_values)
            step_rows.append(
                {"step": step, "timestamp": relative, "step_ms": step_time}
            )
            step += 1
        metadata["steps_per_second"] = len(step_rows) / (time.perf_counter() - start)
        metadata["actual_steps"] = len(step_rows)
        metadata["segment_seconds"] = args.segment_seconds
        metadata["last_training_loss"] = float(loss.detach())
        metadata["process_rss_mb_end"] = psutil.Process().memory_info().rss / 1e6
        metadata["completed"] = True
    finally:
        faults.stop()
        if prefetch is not None:
            prefetch.close()
        if monitor is not None:
            monitor.__exit__(None, None, None)
            metadata["overhead"] = monitor.overhead()
        if original_group is not None:
            (original_group / "cgroup.procs").write_text(str(os.getpid()))
        with (output / f"steps-rank{rank}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "timestamp", "step_ms"])
            writer.writeheader()
            writer.writerows(step_rows)
        write_json(output / f"metadata-rank{rank}.json", metadata)
        if world_size > 1:
            dist.destroy_process_group()
    return metadata
