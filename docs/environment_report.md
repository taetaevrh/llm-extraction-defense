# Environment compatibility report

Inspection date: **2026-10-29, Asia/Bangkok**. Initial readings around 01:56
local time. RAM, free disk, and free VRAM are snapshots, not reserved capacity.
All sizes below use binary GiB/MiB. Work directory:
`D:\MU\Y3S3-Y4S1\Senior Project\me-al`.

## Detected host

| Item | Observed result | Evidence |
| --- | --- | --- |
| OS | Microsoft Windows 11 Home, 64-bit, version 10.0.26200, build 26200 | Win32_OperatingSystem |
| CPU | Intel Core i5-13500, 20 logical processors | Win32_Processor / Win32_ComputerSystem |
| RAM | 63.77 GiB usable physical total; 34.05 GiB free at inspection | 68,474,347,520 total bytes; 35,706,172 free KiB |
| Repository drive D: | 1,863.02 GiB total; 1,127.96 GiB free | Win32_LogicalDisk |
| System drive C: | 930.43 GiB total; 233.77 GiB free | Win32_LogicalDisk |
| Other fixed drive G: | 400.00 GiB total; 69.22 GiB free | Win32_LogicalDisk |
| Git | 2.54.0.windows.1 | git --version |
| Conda | 25.11.1; win-64; libmamba solver | conda --version / info |
| Initial active environment | base, Python 3.13.9 (Anaconda), 64-bit | python sys.version/executable |
| Initial PyTorch | Not installed in active base (module discovery returned false) | importlib.util.find_spec('torch') |

Base Python is at `C:\Users\varic\anaconda3\python.exe`. Other Conda
environments exist; their packages were not imported or changed. PyTorch
absence here refers to the inspected base and project environments, not every
Python installation on the machine. No package operation targeted base.

## GPU and CUDA

| Item | Observed result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Total dedicated VRAM reported | 8,188 MiB (approximately 8 GiB class) |
| Free VRAM at CSV probe | 6,069 MiB (5.93 GiB) |
| Driver | 617.42 |
| Driver model / display | Windows WDDM; display active |
| Compute capability | 8.9 (nvidia-smi query) |
| NVIDIA-SMI advertised CUDA UMD compatibility | 13.4 |
| Installed CUDA compiler on PATH | Toolkit release 13.4; nvcc V13.4.92 |
| Project PyTorch / torch.version.cuda | Not installed / not applicable |
| torch.cuda.is_available() | Not callable until PyTorch is installed; no runtime validation claimed |

`nvidia-smi` and the CSV query were sequential; free memory changed between
them. The CSV value is used above. Desktop applications already occupy part
of the nominal 8 GB; total VRAM is not the available training budget.

The CUDA number advertised by the driver, installed `nvcc` toolkit, and CUDA
runtime bundled with a future PyTorch build are separate version records.
NVIDIA documents backward compatibility with older CUDA applications on newer
drivers. Thus driver 617.42 advertising 13.4 is a favorable compatibility
indicator for an older supported runtime, **not proof of a particular PyTorch
wheel working**. See [NVIDIA compatibility FAQ](https://docs.nvidia.com/deploy/cuda-compatibility/frequently-asked-questions.html).

NVIDIA lists this GPU at compute capability 8.9, consistent with the local
probe. See [NVIDIA GPU table](https://developer.nvidia.com/cuda/gpus).
PyTorch build selection must also check Windows/Python support and architecture
compatibility. Consult the [official release matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
and [installation selector](https://pytorch.org/get-started/locally/).
No exact GPU package is selected or installed in Task 01. Do not infer that
the newest toolkit requires the same runtime version in PyTorch.

Before any later GPU dependency installation:

1. Refresh driver and free-VRAM readings; select a pinned Windows build for
   Python 3.11 with a compatible bundled runtime and RTX 4060 Ti support.
2. Prefer a maintained Conda package when suitable; if the needed build is
   supplied only as a wheel, use pip exclusively inside the activated project
   Conda environment and record exact versions/index/dependencies in YAML.
3. After installation, record `torch.__version__`, `torch.version.cuda`,
   `torch.backends.cudnn.version()`, `torch.cuda.is_available()`, device
   capability/name, architecture list, and a tiny synchronized GPU tensor
   operation. Passing this would validate the framework, not model feasibility.
4. Check model/quantization libraries separately before any model download.

## WSL availability

`wsl.exe` is installed and reports WSL **2.7.8.0**, kernel **6.18.33.1-1**,
and default WSL version **2**. The only registered distribution is
`docker-desktop`, version 2, stopped; it is also the default distribution.
No general-purpose research Linux distribution was detected. WSL GPU access,
Linux Python, and Linux Conda were not tested. This project environment uses
native Windows. No WSL distribution was started, installed, or modified.

## Project Conda environment

Declared name: `llm-extraction-defense`. Local prefix:
`.conda/envs/llm-extraction-defense`. Package cache: `.conda/pkgs`.
The source specification is `environment.yml`, using conda-forge only.
Created successfully with Conda; all 44 resolved packages are from conda-forge.
No pip-installed packages or GPU dependencies were added.

| Dependency | Installed version |
| --- | --- |
| Python | 3.11.18, 64-bit, conda-forge |
| NumPy | 2.2.6 |
| PyYAML | 6.0.3 |
| pytest | 8.4.2 |
| pip (available, not used for installation) | 26.2.2 |
| PyTorch / bundled CUDA runtime | Not installed |

Direct versions are pinned in `environment.yml`. Full package versions,
builds, and channels are saved in `dependency_versions.json`; a platform-specific
explicit package list with MD5 package hashes is in `conda-win-64.explicit.txt`.
`environment.resolved.yml` records all resolved dependencies. The resolved
export removes the absolute prefix and normalizes the auto-appended defaults
channel to nodefaults, matching the actual all-conda-forge installation.

Conda warns that this prefix contains spaces because the existing repository
parent is `Senior Project`. Do not move or rename the user's directory.
Activation/import checks cover the initial toolchain; future CUDA extensions
or build tools may still have quoting/path issues.

Named activation was tested by setting the session-local `CONDA_ENVS_PATH`
to `.conda/envs`, loading the existing Conda PowerShell hook, and running
`conda activate llm-extraction-defense`. `CONDA_DEFAULT_ENV` matched the requested
name and `sys.executable` pointed inside the repository-local prefix.

`python -m pytest -q` in that activated environment: **4 passed in 0.10s**.
Checks cover unresolved design/model choices, no-defense/budget/metric settings,
unmeasured template values, and ignored artifact/credential paths. The first
run had 3 passes and 1 failure from Windows CRLF conversion in a Git stdin
probe; the test now uses binary NUL-delimited input/output and passed on rerun.
These checks do not validate an experiment runner or GPU execution.

The base environment's Conda history file retained its inspected size
(24,203 bytes) and modification timestamp (2026-01-13 11:05:52 local time).
No install/update targeted base; this check is additional evidence rather
than a complete filesystem integrity audit.

## Technical risks specific to this 8 GB GPU

1. **Insufficient headroom:** only about 5.93 GiB was free during inspection.
   WDDM/display use and other applications vary. Check before every run and
   leave a measured safety margin; do not assume all 8 GB are available.
2. **Weight size is only a lower bound:** a 1-billion-parameter model needs
   about 1.86 GiB for 16-bit weights alone (2 bytes per parameter), excluding
   activations, gradients, optimizer states, KV cache, and allocator overhead.
   Conventional full fine-tuning can exceed 8 GB even when inference fits.
3. **Two-model residency:** simultaneously loading teacher and student can
   exceed the budget. Collect/cache teacher responses, unload it, then adapt
   the student. Treat CPU offload as a measured configuration choice.
4. **Context and batch growth:** long prompts, generation KV cache, attention
   buffers, and training activations can cause OOM. Start with short contexts,
   batch 1, and bounded output; record any gradient accumulation/checkpointing.
5. **Quantization trade-offs:** fewer weight bits do not remove activation and
   optimizer costs. Windows kernel/package support must be tested. Quantizing
   a teacher may change the target behavior; freeze it across all conditions.
6. **Resource noise and throughput:** CPU offload, small batches, and sharing
   a display GPU can increase runtime variance. Record per-phase memory and
   wall time, failed runs, and device-wide samples in addition to Torch peaks.
7. **Reproducibility constraints:** precision/kernel choices and deterministic
   algorithms can change memory/speed or fail. Freeze settings and record
   unsupported operations; seeds alone cannot guarantee identical outcomes.

These are feasibility risks, not measurements of any selected model. No claim
is made that a particular model or adaptation method fits this GPU.

## Verification boundary

Host OS/RAM/disk, Conda presence, NVIDIA device/driver/compiler, and WSL
registration were inspected directly. Initial base Python and PyTorch absence
were inspected directly. No CUDA kernel, model inference, dataset ingestion,
extraction, defense, or training has been executed in Task 01.

Reproducibility conventions follow the limits described in
[PyTorch reproducibility notes](https://docs.pytorch.org/docs/2.14/notes/randomness.html).
Documentation consulted on 2026-10-29; release selection must be refreshed when
GPU dependencies are actually added.
