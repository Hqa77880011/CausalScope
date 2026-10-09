import csv
import os
import threading
import time
from pathlib import Path

import psutil


def tcp_retransmissions():
    path = Path("/proc/net/snmp")
    if not path.exists():
        return None
    lines = path.read_text().splitlines()
    for i in range(len(lines) - 1):
        if lines[i].startswith("Tcp:") and "RetransSegs" in lines[i]:
            header = lines[i].split()[1:]
            values = lines[i + 1].split()[1:]
            return int(values[header.index("RetransSegs")])
    return None


class Monitor:
    def __init__(self, path, start, rank, interval=0.2, interface=None, gpu=None):
        self.start = start
        self.rank = rank
        self.interval = interval
        self.interface = interface
        self.gpu = gpu
        self.latest = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.error = None
        self.wall_start = time.perf_counter()
        self.wall_end = None
        self.cpu_seconds = 0.0
        self.rss = []
        self.stream = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.stream)
        self.writer.writerow(["timestamp", "sequence", "rank", "metric", "value"])
        self.nvml = None
        self.handle = None
        if gpu is not None:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if visible != [""]:
                selected = visible[gpu]
                self.handle = (
                    pynvml.nvmlDeviceGetHandleByUUID(selected)
                    if selected.startswith(("GPU-", "MIG-"))
                    else pynvml.nvmlDeviceGetHandleByIndex(int(selected))
                )
            else:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            self._loop()
        except (OSError, ValueError, RuntimeError, psutil.Error) as error:
            self.error = error
            self.stop_event.set()

    def update(self, values):
        with self.lock:
            self.latest.update(values)

    def _network(self):
        if self.interface:
            counters = psutil.net_io_counters(pernic=True)
            if self.interface not in counters:
                raise ValueError(f"Network interface not found: {self.interface}")
            return counters[self.interface]
        return psutil.net_io_counters()

    def _loop(self):
        previous_time = time.perf_counter()
        previous_net = self._network()
        previous_disk = psutil.disk_io_counters()
        previous_context = psutil.cpu_stats().ctx_switches
        previous_retransmissions = tcp_retransmissions()
        previous_cpu = psutil.cpu_times()
        process = psutil.Process()
        sequence = 0
        while not self.stop_event.wait(self.interval):
            cpu_start = time.thread_time()
            now = time.perf_counter()
            elapsed = now - previous_time
            network = self._network()
            disk = psutil.disk_io_counters()
            context = psutil.cpu_stats().ctx_switches
            retransmissions = tcp_retransmissions()
            cpu = psutil.cpu_times()
            differences = [a - b for a, b in zip(cpu, previous_cpu)]
            total_cpu = sum(differences)
            idle_cpu = cpu.idle - previous_cpu.idle
            values = {
                "cpu_util": 100 * (1 - idle_cpu / max(total_cpu, 1e-9)),
                "context_switches": (context - previous_context) / elapsed,
                "net_tx": (network.bytes_sent - previous_net.bytes_sent)
                / elapsed
                / 1e6,
                "net_rx": (network.bytes_recv - previous_net.bytes_recv)
                / elapsed
                / 1e6,
            }
            if disk is not None and previous_disk is not None:
                values["disk_read"] = (
                    (disk.read_bytes - previous_disk.read_bytes) / elapsed / 1e6
                )
            if hasattr(cpu, "iowait"):
                values["io_wait"] = (
                    100 * (cpu.iowait - previous_cpu.iowait) / max(total_cpu, 1e-9)
                )
            if retransmissions is not None and previous_retransmissions is not None:
                values["retransmissions"] = (
                    retransmissions - previous_retransmissions
                ) / elapsed
            if self.handle is not None:
                nvml = self.nvml
                try:
                    utilization = nvml.nvmlDeviceGetUtilizationRates(self.handle)
                    memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                    values.update(
                        {
                            "gpu_util": utilization.gpu,
                            "gpu_clock": nvml.nvmlDeviceGetClockInfo(
                                self.handle, nvml.NVML_CLOCK_SM
                            ),
                            "gpu_memory": 100 * memory.used / memory.total,
                        }
                    )
                except nvml.NVMLError as error:
                    raise RuntimeError(f"GPU telemetry failed: {error}") from error
            with self.lock:
                values.update(self.latest)
            for metric, value in values.items():
                self.writer.writerow(
                    [now - self.start, sequence, self.rank, metric, value]
                )
            self.stream.flush()
            self.rss.append(process.memory_info().rss / 1e6)
            self.cpu_seconds += time.thread_time() - cpu_start
            previous_time, previous_net, previous_disk = now, network, disk
            previous_context, previous_retransmissions, previous_cpu = (
                context,
                retransmissions,
                cpu,
            )
            sequence += 1

    def __enter__(self):
        self._network()
        self.wall_start = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join()
        self.wall_end = time.perf_counter()
        self.stream.close()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()
        if self.error is not None and exc_type is None:
            raise RuntimeError("Telemetry collection failed") from self.error

    def overhead(self):
        elapsed = (
            self.wall_end if self.wall_end is not None else time.perf_counter()
        ) - self.wall_start
        return {
            "monitor_cpu_percent_one_core": 100 * self.cpu_seconds / max(elapsed, 1e-9),
            "process_rss_mb_mean": sum(self.rss) / len(self.rss) if self.rss else None,
            "process_rss_mb_peak": max(self.rss) if self.rss else None,
        }
