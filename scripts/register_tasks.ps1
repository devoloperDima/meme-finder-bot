# Регистрирует задания Планировщика Windows в папке \MemeBot\ (от текущего пользователя, без прав администратора):
#   Pipeline — конвейер (scripts\pipeline.ps1), запуск вручную
#   Scrape   — сбор мем-каналов (scripts\scrape_channels.ps1), запуск вручную
#   Bot      — бот: при входе в Windows и с перезапуском при падении
# Процессы заданий не зависят от окна Claude: прерывание команды или закрытие сессии их не останавливает.
# Удалить всё:  Get-ScheduledTask -TaskPath "\MemeBot\" | Unregister-ScheduledTask -Confirm:$false

$root = Split-Path $PSScriptRoot -Parent
$user = "$env:USERDOMAIN\$env:USERNAME"
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited

function Action($script) {
    New-ScheduledTaskAction -Execute "powershell.exe" -WorkingDirectory $root `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$root\scripts\$script`""
}

$long = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -StartWhenAvailable
$bot = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskPath "\MemeBot\" -TaskName "Pipeline" -Action (Action "pipeline.ps1") `
    -Principal $principal -Settings $long -Description "meme-finder-bot: VLM, обучение, судьи, оценка" -Force | Out-Null
Register-ScheduledTask -TaskPath "\MemeBot\" -TaskName "Scrape" -Action (Action "scrape_channels.ps1") `
    -Principal $principal -Settings $long -Description "meme-finder-bot: сбор мем-каналов" -Force | Out-Null
Register-ScheduledTask -TaskPath "\MemeBot\" -TaskName "Bot" -Action (Action "run_bot.ps1") `
    -Trigger (New-ScheduledTaskTrigger -AtLogOn -User $user) `
    -Principal $principal -Settings $bot -Description "meme-finder-bot: Telegram-бот" -Force | Out-Null

Get-ScheduledTask -TaskPath "\MemeBot\" | Select-Object TaskName, State
