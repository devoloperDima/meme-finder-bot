# Весь конвейер одним скриптом. Каждый шаг можно перезапускать: прогоны продолжаются
# с места остановки, готовые модели не переобучаются. Сбор каналов идёт параллельно
# (scripts\scrape_channels.ps1), конвейер ждёт его там, где нужен корпус целиком.
# Лог: data\logs\pipeline.log

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".venv\Scripts\python.exe"
$log = "data\logs\pipeline.log"
# В системных настройках Windows стоит прокси VPN, и без исключения запросы к localhost (Ollama)
# уходят в него и получают 503. Отключаем прокси для .NET-запросов скрипта и исключаем localhost для Python.
[System.Net.WebRequest]::DefaultWebProxy = $null
$env:NO_PROXY = "localhost,127.0.0.1,::1"

function Log($msg) { Add-Content $log -Encoding UTF8 "$(Get-Date -Format 'dd.MM HH:mm:ss') $msg" }
function Running($pattern) { [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern }) }
function WaitFor($pattern) { while (Running $pattern) { Start-Sleep 60 } }

function Py([string]$title, [string[]]$pyArgs) {
    Log $title
    # Вывод — во временный файл и потом в лог: так лог не блокируется на время шага.
    # Через cmd: перенаправление PowerShell 5.1 пишет UTF-16 и оборачивает stderr в ошибки PowerShell.
    $tmp = "data\logs\step.tmp"
    cmd /c "$py $($pyArgs -join ' ') > $tmp 2>&1"
    $script:code = $LASTEXITCODE
    Get-Content $tmp -Encoding UTF8 | ForEach-Object { ($_ -split "`r")[-1] } |
        Where-Object { $_ -notmatch 'it/s\]|s/it\]|^Batches|Warning|warn\(|^\s*$' } | Add-Content $log -Encoding UTF8
    Log "$title`: exit $script:code"
}

function PyRetry([string]$title, [string[]]$pyArgs, [int]$tries = 3) {
    # Шаги с Ollama (VLM, судьи) при массовых ошибках выходят с кодом 1 — повторяем, они продолжают с места
    for ($i = 1; $i -le $tries; $i++) {
        Py "$title (попытка $i)" $pyArgs
        if ($script:code -eq 0) { return }
        Start-Sleep 60
    }
}

function RestartBot {
    # Бот живёт в отдельном задании Планировщика (scripts\register_tasks.ps1), чтобы пережить конвейер.
    # Остановка задания не убивает дочерний python — завершаем его явно, иначе два бота на одном токене.
    Stop-ScheduledTask -TaskPath "\MemeBot\" -TaskName "Bot" -ErrorAction SilentlyContinue
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*memebot.bot.main*' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep 5
    Start-ScheduledTask -TaskPath "\MemeBot\" -TaskName "Bot"
    Log "бот перезапущен"
}

function UnloadOllama {
    foreach ($m in "hf.co/Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M", "hf.co/ggml-org/gemma-3-4b-it-GGUF:Q4_K_M") {
        try { Invoke-RestMethod http://localhost:11434/api/generate -Method Post -ContentType 'application/json' `
              -Body (@{ model = $m; keep_alive = 0 } | ConvertTo-Json) | Out-Null } catch {}
    }
    Start-Sleep 10
}

function QueueSize([string]$flags) {
    $n = cmd /c "$py -m memebot.indexing.vlm_captions --count $flags 2>nul" | Select-Object -Last 1
    return [int]$n
}

Log "=== старт конвейера"
WaitFor '*memebot.indexing.vlm_captions*'

# 1. Сначала свежие мемы для бота. У мемов из каналов нет ручных описаний — без VLM бот их не видит.
#    Пока идёт сбор каналов: пересборка датасета → VLM порцией (свежие первыми) → перезапуск бота.
do {
    $scraping = Running '*scrape_channels.ps1*'
    Py "сборка датасета" @("-m", "memebot.data.build_dataset")
    $left = QueueSize "--only-undescribed"
    if ($left -gt 0) {
        PyRetry "VLM: мемы без описаний, осталось $left" @("-m", "memebot.indexing.vlm_captions", "--only-undescribed", "--limit", "300")
        RestartBot
    } elseif ($scraping) {
        Start-Sleep 600
    }
} while ($scraping -or $left -gt 0)

# 2. VLM-описания исходного корпуса (нужны для обучения на «ситуациях»)
PyRetry "VLM: исходный корпус" @("-m", "memebot.indexing.vlm_captions", "--scope", "eval")
UnloadOllama

# 3. Бейзлайны на текстах VLM (silver)
foreach ($field in "vlm", "human+vlm") {
    Py "silver baselines $field" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "bm25", "deepvk/USER-bge-m3", "--field", $field)
    Py "silver rrf $field" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", "USER-bge-m3", "--field", $field)
}
Py "silver SigLIP2" @("-m", "memebot.evaluation.run_baselines", "--retrievers", "google/siglip2-base-patch16-256")
Py "silver rrf text+image" @("-m", "memebot.evaluation.run_baselines", "--fuse", "bm25", "USER-bge-m3", "siglip2-base-patch16-256")

# 4. Дообучение: абляция данных и hard negatives (готовые модели пропускаем)
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

# 5. Gold без человека: пул (TREC) с дообученными моделями, два независимых судьи, консенсус
$systems = @("bm25", "intfloat/multilingual-e5-base", "deepvk/USER-bge-m3", "google/siglip2-base-patch16-256") + @($tuned | ForEach-Object { "models/$($_.Name)" })
Py "пул gold" (@("-m", "memebot.evaluation.gold", "pool", "--depth", "10", "--systems") + $systems)
PyRetry "судья qwen" @("-m", "memebot.evaluation.llm_judge", "run", "--judge", "qwen")
UnloadOllama
PyRetry "судья gemma" @("-m", "memebot.evaluation.llm_judge", "run", "--judge", "gemma")
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

Py "сводка результатов" @("-m", "memebot.evaluation.report")

# 6. Остальное: «ситуации» от VLM для мемов, у которых уже есть ручные описания (свежие @text_meme, SoulQrat)
PyRetry "VLM: остальные мемы" @("-m", "memebot.indexing.vlm_captions")
UnloadOllama
RestartBot
Log "=== ГОТОВО"
