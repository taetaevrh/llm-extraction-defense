"""Measured driver-memory guard for one-model diagnostic subprocesses."""

import subprocess
import threading
import time


def driver_memory():
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True,
    ).strip().splitlines()[0]
    used, free = (float(value.strip()) for value in raw.split(","))
    return {"used_mib": used, "free_mib": free}


class GuardedPhase:
    def __init__(self, name, records, minimum_free_mib=1024):
        self.name, self.records, self.minimum = name, records, minimum_free_mib
        self.samples = []
        self.stop = threading.Event()
        self.sampling_error = None

    def check(self):
        if self.sampling_error:
            raise RuntimeError(f"Driver memory monitoring failed: {self.sampling_error}")
        if self.samples and min(s["free_mib"] for s in self.samples) < self.minimum:
            raise MemoryError("Measured memory headroom fell below 1 GiB; stop diagnostics")

    def sample(self):
        while not self.stop.is_set():
            try:
                self.samples.append(driver_memory())
            except Exception as error:
                self.sampling_error = str(error)
                return
            self.stop.wait(0.2)

    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self.samples.append(driver_memory())
        self.check()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        import torch
        try:
            torch.cuda.synchronize()
        finally:
            self.stop.set()
            self.thread.join()
            self.samples.append(driver_memory())
            self.records[self.name] = {
                "seconds": time.perf_counter() - self.started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                "sampled_device_peak_used_mib": max(s["used_mib"] for s in self.samples),
                "minimum_measured_driver_free_mib": min(s["free_mib"] for s in self.samples),
                "samples": len(self.samples), "sampling_wait_seconds": 0.2,
                "sampling_note": "Driver CLI execution adds latency; samples may miss short spikes",
                "sampling_error": self.sampling_error,
                "status": "failed" if error else "completed",
            }
        if not error:
            self.check()
