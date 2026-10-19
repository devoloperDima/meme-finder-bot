# Весь конвейер одним скриптом. Каждый шаг можно перезапускать: прогоны продолжаются
# с места остановки, готовые модели не переобучаются. Сбор каналов идёт параллельно
# (scripts\scrape_channels.ps1), конвейер ждёт его там, где нужен корпус целиком.
# Лог: data\logs\pipeline.log

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".venv\Scripts\python.exe"
$log = "data\logs\pipeline.log"

function Log($msg) { Add-Content $log -Encoding UTF8 "$(Get-Date -Format 'dd.MM HH:mm:ss') $msg" }
function Running($pattern) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern }) }
function WaitFor($pattern) { while (Running $pattern) { Start-Sleep 60 } }

function Py([string]$title, [string[]]$pyArgs) {
    Log $title
    # Вывод — во временный файл и потом в лог: так лог не блокируется на время шага
    $tmp = "data\logs\step.tmp"
    & $py @pyArgs *> $tmp
    Get-Content $tmp -Encoding UTF8 | Where-Object { $_ -notmatch 'it/s\]|s/it\]|^Batches|Warning|warn\(' } | Add-Content $log -Encoding UTF8
    Log "$title`: exit $LASTEXITCODE"
}

function RestartBot {
    $botPid = Get-Content "data\logs\bot.pid" -ErrorAction SilentlyContinue
    if ($botPid) { Stop-Process -Id $botPid -Force -ErrorAction SilentlyContinue }
    Start-Sleep 3
    $env:SEARCH_DEVICE = "cpu"  # GPU занят VLM, обучением или судьями
    $bot = Start-Process -FilePath $py -ArgumentList '-m', 'memebot.bot.main' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "data\logs\bot.out" -RedirectStandardError "data\logs\bot.log"
    $bot.Id | Out-File "data\logs\bot.pid" -Encoding ascii
    Log "бот перезапущен (PID $($bot.Id))"
}

function UnloadOllama {
    foreach ($m in "hf.co/Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M", "hf.co/ggml-org/gemma-3-4b-it-GGUF:Q4_K_M") {
        try { Invoke-RestMethod http://localhost:11434/api/generate -Method Post -ContentType 'application/json' `
              -Body (@{ model = $m; keep_alive = 0 } | ConvertTo-Json) | Out-Null } catch {}
    }
    Start-Sleep 10
}

Log "=== старт конвейера"

# 1. VLM-описания исходного корпуса (нужны для обучения на «ситуациях»)
WaitFor '*memebot.indexing.vlm_captions*'
Py "VLM: исходный корпус" @("-m", "memebot.indexing.vlm_captions", "--scope", "eval")
UnloadOllama

# 2. Бейзлайны на текстах VLM (silver)
foreach ($field in "vlm", "human+vlm") {
    Py "silver baselines $field" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "bm25", "deepvk/USER-bge-m3", "--field", $field)
    Py "silver rrf $field" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", "USER-bge-m3", "--field", $field)
}
Py "silver SigLIP2" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "google/siglip2-base-patch16-256")
Py "silver rrf text+image" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", "USER-bge-m3", "siglip2-base-patch16-256")

# 3. Дообучение: абляция данных и hard negatives (готовые модели пропускаем)
$runs = @(
    @{ name = "USER-bge-m3-synthetic-hn0"; args = @("--queries", "synthetic", "--hard-negatives", "0") },
    @{ name = "USER-bge-m3-synthetic-vlm-hn0"; args = @("--queries", "synthetic", "vlm", "--hard-negatives", "0") },
    @{ name = "USER-bge-m3-synthetic-vlm-hn1"; args = @("--queries", "synthetic", "vlm", "--hard-negatives", "1") }
)
foreach ($r in $runs) {
    if (Test-Path "models\$($r.name)\model.safetensors") { Log "модель $($r.name) уже есть"; continue }
    Py "train $($r.name)" (@("-m", "memebot.training.train_embedder", "--name", $r.name) + $r.args)
}
$tuned = @(Get-ChildItem models -Directory | Where-Object { $_.Name -ne "checkpoints" -and (Test-Path "$($_.FullName)\model.safetensors") })
foreach ($m in $tuned) {
    Py "silver $($m.Name)" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "models/$($m.Name)")
    Py "silver rrf $($m.Name)" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", $m.Name)
}

# 4. Gold без человека: пул (TREC) с дообученными моделями, два независимых судьи, консенсус
$systems = @("bm25", "intfloat/multilingual-e5-base", "deepvk/USER-bge-m3", "google/siglip2-base-patch16-256") + @($tuned | ForEach-Object { "models/$($_.Name)" })
Py "пул gold" (@("-m", "memebot.evaluation.gold", "pool", "--depth", "10", "--systems") + $systems)
Py "судья qwen" @("-m", "memebot.evaluation.llm_judge", "run", "--judge", "qwen")
UnloadOllama
Py "судья gemma" @("-m", "memebot.evaluation.llm_judge", "run", "--judge", "gemma")
UnloadOllama
Py "консенсус судей" @("-m", "memebot.evaluation.llm_judge", "consensus")
Py "согласие судей" @("-m", "memebot.evaluation.llm_judge", "agreement")

foreach ($set in "gold-consensus", "gold-judge", "gold-gemma") {
    Py "${set}: системы" (@("-m", "memebot.evaluation.run_baselines", "--set", $set, "--retrievers") + $systems)
    Py "${set}: rrf bm25+USER-bge-m3" @("-m", "memebot.evaluation.run_baselines", "--set", $set, "--fuse", "bm25", "USER-bge-m3")
    foreach ($m in $tuned) {
        Py "${set}: rrf bm25+$($m.Name)" @("-m", "memebot.evaluation.run_baselines", "--set", $set, "--fuse", "bm25", $m.Name)
    }
}

# 5. Полный корпус: ждём сбор каналов, пересобираем, VLM для новых мемов (без описаний и свежие — первыми)
WaitFor '*scrape_channels.ps1*'
Py "сборка датасета" @("-m", "memebot.data.build_dataset")
RestartBot
Py "VLM: новые мемы" @("-m", "memebot.indexing.vlm_captions")
UnloadOllama
RestartBot
Log "=== ГОТОВО"
