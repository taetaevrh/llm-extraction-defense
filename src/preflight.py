"""Repository-local paths and actual CUDA verification; no model computation."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from src.runtime import write_json

ROOT = Path(__file__).resolve().parents[1]


def configure_paths():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from the repository root")
    expected = ROOT / ".conda/envs/llm-extraction-defense"
    if Path(sys.prefix).resolve() != expected.resolve():
        raise RuntimeError("Use the existing project Conda environment")
    for key, relative in {
        "HF_HOME": ".cache/huggingface", "HF_DATASETS_CACHE": ".cache/huggingface/datasets",
        "TORCH_HOME": ".cache/torch", "XDG_CACHE_HOME": ".cache",
    }.items():
        os.environ[key] = str(ROOT / relative)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if os.environ.get("PYTHONHASHSEED") != "17":
        raise RuntimeError("Set PYTHONHASHSEED=17 before starting Python")


def verify_cuda():
    from src.runtime import seed_everything
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured BF16 is unsupported")
    seed_everything(17)
    free_before, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    x = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
    actual = x @ x.T
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    cpu = torch.arange(256, dtype=torch.float32).reshape(16, 16)
    if not torch.equal(actual.cpu(), cpu @ cpu.T):
        raise RuntimeError("GPU matrix computation differed from CPU reference")
    observation = {
        "status": "passed", "python": sys.version, "pytorch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_is_available": True, "bf16_supported": True,
        "gpu_name": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "compiled_architectures": torch.cuda.get_arch_list(),
        "free_vram_mib_before_computation": free_before / 2**20,
        "total_vram_mib": total / 2**20,
        "tiny_matmul_seconds": seconds, "tiny_matmul_sum": float(actual.sum()),
        "tiny_matmul_matches_cpu_exactly": True,
        "tiny_matmul_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap", "--format=csv"], text=True).strip(),
        "deterministic_algorithms": True, "tf32": False,
    }
    fields = observation["nvidia_smi"].splitlines()[1].split(",")
    observation["nvidia_free_vram_mib_after_computation"] = float(fields[2].strip().split()[0])
    observation["memory_reading_note"] = "CUDA free memory and NVIDIA-SMI physical-device readings can differ under Windows WDDM; both recorded."
    del x, actual, cpu
    torch.cuda.empty_cache()
    return observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configure_paths()
    try:
        result = verify_cuda()
    except Exception as error:
        write_json(args.output, {"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
