$ErrorActionPreference = 'Stop'
(& conda shell.powershell hook) | Out-String | Invoke-Expression
$env:CONDA_ENVS_PATH = Join-Path (Get-Location) '.conda/envs'
conda activate llm-extraction-defense
if ($env:CONDA_DEFAULT_ENV -ne 'llm-extraction-defense') { throw 'Wrong Conda environment' }
$env:PYTHONHASHSEED = '17'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -u -m src.diagnostics.run --config configs/task03_diagnostics.yaml
if ($LASTEXITCODE -ne 0) { throw 'Task03 stopped; preserve and inspect its failure report.' }
