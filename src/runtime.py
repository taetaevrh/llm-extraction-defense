"""Deterministic runtime and measured resource records."""

import gc
import json
from pathlib import Path
import random
import subprocess
import threading
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)


class Phase:
    def __init__(self, name, records):
        self.name, self.records = name, records
        self._stop = threading.Event()
        self.samples = []

    def _sample(self):
        while not self._stop.is_set():
            # CUDA mem_get_info and physical device readings can differ under
            # Windows WDDM. Use the driver CLI for total-device observations.
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True,
            )
            self.samples.append(float(output.strip().splitlines()[0]))
            self._stop.wait(0.2)

    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, error_type, error, traceback):
        import torch
        torch.cuda.synchronize()
        self._stop.set()
        self._thread.join()
        self.records[self.name] = {
            "seconds": time.perf_counter() - self.started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "sampled_device_peak_used_mib": max(self.samples) if self.samples else None,
            "sampling_interval_seconds": 0.2, "sample_count": len(self.samples),
            "device_sampling_source": "nvidia-smi; 0.2s wait plus command execution per sample",
            "status": "failed" if error else "completed",
        }


def release_gpu():
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
