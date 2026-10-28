$ErrorActionPreference = 'Stop'
# Execute from repository root. No global Conda settings are changed.
(& conda shell.powershell hook) | Out-String | Invoke-Expression
$env:CONDA_ENVS_PATH = Join-Path (Get-Location) '.conda/envs'
$env:CONDA_PKGS_DIRS = Join-Path (Get-Location) '.conda/pkgs'
conda activate llm-extraction-defense
if ($env:CONDA_DEFAULT_ENV -ne 'llm-extraction-defense') { throw 'Wrong environment' }
$env:PYTHONHASHSEED = '17'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -u -m src.run_baseline --config configs/task02_banking77.yaml
if ($LASTEXITCODE -ne 0) { throw "Baseline stopped with exit code $LASTEXITCODE. Inspect its failure record." }
