# Запуск бота для задания Планировщика \MemeBot\Bot. Логи: data\logs\bot.log
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$env:SEARCH_DEVICE = "cpu"  # GPU оставляем для VLM, обучения и судей
& ".venv\Scripts\python.exe" -m memebot.bot.main *>> "data\logs\bot.log"
