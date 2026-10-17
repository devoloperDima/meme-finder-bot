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
Log "VLM для новых мемов"
& $py -m memebot.indexing.vlm_captions 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]' } | Add-Content $log
Log "VLM: exit $LASTEXITCODE"
RestartBot
Log "ГОТОВО"
