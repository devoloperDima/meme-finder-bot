# Дирижёр: сборка датасета и перезапуск бота по мере готовности данных,
# затем VLM-описания новых мемов, когда GPU освободится после обучения.
# Лог: data\logs\pipeline.log

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".venv\Scripts\python.exe"
$log = "data\logs\pipeline.log"

function Log($msg) { Add-Content $log "$(Get-Date -Format 'dd.MM HH:mm:ss') $msg" }

function Running($pattern) {
    [bool](Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pattern })
}

function WaitFor($pattern) { while (Running $pattern) { Start-Sleep 60 } }

function Build {
    Log "сборка датасета"
    & $py -m memebot.data.build_dataset 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]' } | Add-Content $log
    Log "сборка: exit $LASTEXITCODE"
}

function RestartBot {
    $botPid = Get-Content "data\logs\bot.pid" -ErrorAction SilentlyContinue
    if ($botPid) { Stop-Process -Id $botPid -Force -ErrorAction SilentlyContinue }
    Start-Sleep 3
    $env:SEARCH_DEVICE = "cpu"  # GPU занят VLM или обучением
    $bot = Start-Process -FilePath $py -ArgumentList '-m', 'memebot.bot.main' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "data\logs\bot.out" -RedirectStandardError "data\logs\bot.log"
    $bot.Id | Out-File "data\logs\bot.pid" -Encoding ascii
    Log "бот перезапущен (PID $($bot.Id))"
}

Log "старт: жду сбор @text_meme"
WaitFor '*scrape_telegram*text_meme*'
Build
RestartBot

Log "жду сбор каналов"
WaitFor '*scrape_channels.ps1*'
Build
RestartBot

Log "жду освобождения GPU (VLM старых мемов и обучение)"
WaitFor '*memebot.indexing.vlm_captions*'
WaitFor '*night_run.ps1*'

function Py([string]$title, [string[]]$pyArgs) {
    Log $title
    & $py @pyArgs 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]|^Batches' } | Add-Content $log
    Log "$title`: exit $LASTEXITCODE"
}

# Gold через LLM-судью. Сначала пул дополняется топом дообученных моделей (TREC-пулинг):
# иначе мемы, найденные только ими, остались бы неоценёнными и засчитывались как промахи.
$tuned = @(Get-ChildItem models -Directory | Where-Object { $_.Name -ne "checkpoints" } | ForEach-Object { "models/$($_.Name)" })
if ($tuned.Count) { Py "пул gold + дообученные" (@("-m", "memebot.evaluation.gold", "pool", "--depth", "10", "--systems") + $tuned) }
Py "LLM-судья" @("-m", "memebot.evaluation.llm_judge", "run")
Py "gold-judge: бейзлайны" @("-m", "memebot.evaluation.run_baselines", "--set", "gold-judge", "--retrievers",
    "bm25", "intfloat/multilingual-e5-base", "deepvk/USER-bge-m3")
Py "gold-judge: RRF" @("-m", "memebot.evaluation.run_baselines", "--set", "gold-judge", "--fuse", "bm25", "USER-bge-m3")
foreach ($m in $tuned) {
    $name = Split-Path $m -Leaf
    Py "gold-judge: $name" @("-m", "memebot.evaluation.run_baselines", "--set", "gold-judge", "--retrievers", $m)
    Py "gold-judge: RRF $name" @("-m", "memebot.evaluation.run_baselines", "--set", "gold-judge", "--fuse", "bm25", $name)
}

Log "VLM для новых мемов"
& $py -m memebot.indexing.vlm_captions 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]' } | Add-Content $log
Log "VLM: exit $LASTEXITCODE"
RestartBot
Log "ГОТОВО"
