import json
import subprocess
from pathlib import Path


INTENSITIES = {
    "cpu": [75, 50, 25],
    "dataloader": [20, 50, 100],
    "gpu_compute": [1800, 1500, 1200],
    "network": [10, 5, 2.5],
    "storage": [800, 400, 200],
    "rank_compute": [20, 50, 100],
    "gpu_memory": [20, 35, 50],
}


class FaultController:
    def __init__(
        self,
        faults,
        shared_delay,
        device,
        cgroup=None,
        block_device=None,
        interface=None,
    ):
        if not isinstance(faults, dict) or not set(faults).issubset(INTENSITIES):
            raise ValueError("Faults must map known component names to intensities")
        self.faults = faults
        self.shared_delay = shared_delay
        self.device = device
        self.cgroup = Path(cgroup) if cgroup else None
        self.block_device = block_device
        self.interface = interface
        self.cleanup = []
        self.events = []
        self.reserved = None
        self.rank_sleep = 0
        self.active = False

    def _write(self, filename, value):
        if self.cgroup is None:
            raise ValueError(
                "CPU/storage faults require a delegated cgroup v2 directory"
            )
        path = self.cgroup / filename
        old = path.read_text().strip()
        path.write_text(value)
        self.cleanup.append(("file", path, old))
        self.events.append({"path": str(path), "value": value})

    def start(self):
        import torch

        if self.active:
            raise RuntimeError("Faults already active")
        self.active = True
        try:
            for component, intensity in self.faults.items():
                if intensity not in INTENSITIES[component]:
                    raise ValueError(
                        f"Unsupported intensity for {component}: {intensity}"
                    )
                if component == "cpu":
                    self._write("cpu.max", f"{int(100000 * intensity / 100)} 100000")
                elif component == "storage":
                    if not self.block_device:
                        raise ValueError(
                            "Storage throttle requires --block-device MAJOR:MINOR"
                        )
                    self._write(
                        "io.max", f"{self.block_device} rbps={int(intensity * 1e6)}"
                    )
                elif component == "dataloader":
                    self.shared_delay.value = intensity / 1000
                    self.events.append(
                        {
                            "action": "worker_collation_sleep",
                            "seconds": intensity / 1000,
                        }
                    )
                elif component == "rank_compute":
                    self.rank_sleep = intensity / 1000
                    self.events.append(
                        {"action": "rank_sleep", "seconds": intensity / 1000}
                    )
                elif component == "gpu_memory":
                    if self.device.type != "cuda":
                        raise ValueError("GPU memory fault requires CUDA")
                    total = torch.cuda.get_device_properties(self.device).total_memory
                    self.reserved = torch.empty(
                        int(total * intensity / 100),
                        dtype=torch.uint8,
                        device=self.device,
                    )
                    self.events.append(
                        {"action": "reserve_gpu_bytes", "bytes": self.reserved.numel()}
                    )
                elif component == "gpu_compute":
                    if self.device.type != "cuda":
                        raise ValueError("GPU clock fault requires CUDA")
                    import pynvml

                    pynvml.nvmlInit()
                    from .workloads import physical_gpu_id

                    gpu_id = physical_gpu_id(self.device.index)
                    handle = (
                        pynvml.nvmlDeviceGetHandleByUUID(gpu_id)
                        if isinstance(gpu_id, str)
                        else pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    )
                    try:
                        pynvml.nvmlDeviceSetGpuLockedClocks(
                            handle, intensity, intensity
                        )
                    except pynvml.NVMLError:
                        pynvml.nvmlShutdown()
                        raise
                    self.cleanup.append(("clock", pynvml, handle))
                    self.events.append(
                        {
                            "action": "nvmlDeviceSetGpuLockedClocks",
                            "gpu": gpu_id,
                            "min_mhz": intensity,
                            "max_mhz": intensity,
                        }
                    )
                elif component == "network":
                    if not self.interface:
                        raise ValueError("Network throttle requires --interface")
                    qdiscs = json.loads(
                        subprocess.check_output(
                            ["tc", "-j", "qdisc", "show", "dev", self.interface],
                            text=True,
                        )
                    )
                    if any(
                        entry.get("root")
                        and entry["kind"]
                        not in ("noqueue", "mq", "fq_codel", "pfifo_fast")
                        for entry in qdiscs
                    ):
                        raise ValueError(
                            "Interface has a custom root qdisc; use a dedicated experiment interface"
                        )
                    command = [
                        "tc",
                        "qdisc",
                        "add",
                        "dev",
                        self.interface,
                        "root",
                        "tbf",
                        "rate",
                        f"{intensity}gbit",
                        "burst",
                        "1mb",
                        "latency",
                        "50ms",
                    ]
                    subprocess.run(command, check=True)
                    self.cleanup.append(("network", self.interface, None))
                    self.events.append({"command": command})
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError):
            self.stop()
            raise

    def stop(self):
        self.shared_delay.value = 0
        self.rank_sleep = 0
        self.reserved = None
        for kind, target, previous in reversed(self.cleanup):
            if kind == "file":
                if target.name == "io.max":
                    restore = next(
                        (
                            line
                            for line in previous.splitlines()
                            if line.startswith(f"{self.block_device} ")
                        ),
                        f"{self.block_device} rbps=max wbps=max riops=max wiops=max",
                    )
                    target.write_text(restore)
                else:
                    target.write_text(previous)
            elif kind == "clock":
                target.nvmlDeviceResetGpuLockedClocks(previous)
                target.nvmlShutdown()
            elif kind == "network":
                subprocess.run(
                    ["tc", "qdisc", "del", "dev", target, "root"], check=True
                )
        self.cleanup.clear()
        self.active = False
