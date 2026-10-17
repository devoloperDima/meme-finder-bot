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
    Add-Content $log "`n===== $(Get-Date -Format 'HH:mm:ss') $($scrapeArgs -join ' ')"
    & $py -m memebot.data.scrape_telegram @scrapeArgs 2>&1 | Where-Object { $_ -notmatch 'it/s\]|s/it\]' } | Add-Content $log
}

# Разборы трендов (в т.ч. TikTok) — за два года: тренды живут дольше ленты.
Scrape @("--channels", "memepedia_Ru", "--since", "2024-10-17", "--max-posts", "1500")
# Свежие мемы-картинки — за последний год.
Scrape @("--channels", "memachh", "memdoze", "thedankestmemes", "ebanutie_otkritki", "--since", "2025-10-17", "--max-posts", "1200")
Scrape @("--channels", "holy_meoww", "memyprokotow", "--since", "2025-10-17", "--max-posts", "500")
Add-Content $log "`n$(Get-Date -Format 'HH:mm:ss') ГОТОВО"
