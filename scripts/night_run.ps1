# Ночная очередь на GPU: ждёт окончания прогона VLM, затем бейзлайны на VLM-полях,
# три варианта дообучения (абляция данных и hard negatives) и оценка моделей.
# Запуск:  powershell -File scripts\night_run.ps1   (лог: data\logs\night_run.log)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".venv\Scripts\python.exe"
$log = "data\logs\night_run.log"

function Step($title, [string[]]$pyArgs) {
    Add-Content $log "`n===== $(Get-Date -Format 'HH:mm:ss') $title"
    & $py @pyArgs 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]|^Batches' } | Add-Content $log
    Add-Content $log "===== exit $LASTEXITCODE"
}

$vlmPid = Get-Content "data\logs\vlm_captions.pid" -ErrorAction SilentlyContinue
Add-Content $log "$(Get-Date -Format 'HH:mm:ss') жду завершения VLM (PID $vlmPid)"
while ($vlmPid -and (Get-Process -Id $vlmPid -ErrorAction SilentlyContinue)) { Start-Sleep 60 }

# Освобождаем видеопамять: выгружаем VLM из Ollama
try { Invoke-RestMethod http://localhost:11434/api/generate -Method Post -ContentType 'application/json' `
      -Body '{"model":"hf.co/Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M","keep_alive":0}' | Out-Null } catch {}
Start-Sleep 10

foreach ($field in "vlm", "human+vlm") {
    Step "baselines field=$field" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "bm25", "deepvk/USER-bge-m3", "--field", $field)
    Step "rrf field=$field" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", "USER-bge-m3", "--field", $field)
}

$runs = @(
    @("--queries", "synthetic", "--hard-negatives", "0"),
    @("--queries", "synthetic", "vlm", "--hard-negatives", "0"),
    @("--queries", "synthetic", "vlm", "--hard-negatives", "1")
)
foreach ($r in $runs) {
    Step "train $($r -join ' ')" (@("-m", "memebot.training.train_embedder") + $r)
}

foreach ($m in Get-ChildItem models -Directory | Where-Object { $_.Name -ne "checkpoints" }) {
    Step "eval $($m.Name)" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "models/$($m.Name)")
    Step "rrf $($m.Name)" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", $m.Name)
}
Add-Content $log "`n$(Get-Date -Format 'HH:mm:ss') ГОТОВО"
