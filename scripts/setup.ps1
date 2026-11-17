param([string]$StorageRoot = 'D:\icss\acc2')
$ErrorActionPreference = 'Stop'
$env:UV_CACHE_DIR = Join-Path $StorageRoot '.cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $StorageRoot '.python'
$env:TEMP = Join-Path $StorageRoot '.tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:UV_CACHE_DIR,$env:UV_PYTHON_INSTALL_DIR,$env:TEMP | Out-Null
Push-Location (Join-Path $PSScriptRoot '..')
try {
    uv python install 3.11
    if ($LASTEXITCODE -ne 0) { throw 'Python install failed' }
    uv sync --python 3.11 --extra dev
    if ($LASTEXITCODE -ne 0) { throw 'Environment sync failed' }
} finally { Pop-Location }
