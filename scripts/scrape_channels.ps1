# Сбор мемов из публичных каналов (тренды, свежие мемы-картинки). Ждёт, пока закончится
# текущий сбор @text_meme, чтобы не нагружать t.me параллельными запросами.
# Лог: data\logs\scrape_channels.log

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$py = ".venv\Scripts\python.exe"
$log = "data\logs\scrape_channels.log"

while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
       Where-Object { $_.CommandLine -like '*scrape_telegram*text_meme*' }) { Start-Sleep 30 }

function Scrape([string[]]$scrapeArgs) {
    Add-Content $log -Encoding UTF8 "`n===== $(Get-Date -Format 'dd.MM HH:mm:ss') $($scrapeArgs -join ' ')"
    # Через cmd: перенаправление PowerShell 5.1 пишет UTF-16 и оборачивает stderr в ошибки PowerShell
    cmd /c "$py -m memebot.data.scrape_telegram $($scrapeArgs -join ' ') > data\logs\scrape_step.tmp 2>&1"
    Get-Content "data\logs\scrape_step.tmp" -Encoding UTF8 | ForEach-Object { ($_ -split "`r")[-1] } |
        Where-Object { $_ -notmatch 'it/s\]|s/it\]|^\s*$' } | Add-Content $log -Encoding UTF8
}

# Разборы трендов (в т.ч. TikTok) — за два года: тренды живут дольше ленты.
Scrape @("--channels", "memepedia_Ru", "--since", "2024-10-17", "--max-posts", "1500")
# Свежие мемы-картинки — за последний год.
Scrape @("--channels", "memachh", "memdoze", "thedankestmemes", "ebanutie_otkritki", "--since", "2025-10-17", "--max-posts", "1200")
Scrape @("--channels", "holy_meoww", "memyprokotow", "--since", "2025-10-17", "--max-posts", "500")
Add-Content $log -Encoding UTF8 "`n$(Get-Date -Format 'dd.MM HH:mm:ss') ГОТОВО"
