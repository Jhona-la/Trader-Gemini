# MONITOR DEMO — revisa salud del motor cada 5 minutos
# Uso: powershell -File scripts\monitor_demo.ps1
$log = "logs\demo_soak.log"
$report = "logs\demo_monitor.log"

while ($true) {
    $now = Get-Date -Format "HH:mm:ss"
    $lines = (Get-Content $log -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    $opens = (Select-String -Path $log -Pattern "OPEN TRACE" -AllMatches | Measure-Object).Count
    $closes = (Select-String -Path $log -Pattern "CLOSE TRACE" -AllMatches | Measure-Object).Count
    $errors = (Select-String -Path $log -Pattern "panic|FATAL|KILL_SWITCH" -AllMatches | Measure-Object).Count
    
    # Último ml_prob de BTC
    $btcLine = Select-String -Path $log -Pattern "btcusdt.*ml=" | Select-Object -Last 1
    $btcMl = if ($btcLine) { ($btcLine.Line -replace '.*ml=([0-9.]+).*','$1') } else { "?" }
    
    # Último ensamble
    $ensLine = Select-String -Path $log -Pattern "btcusdt.*ensamble" | Select-Object -Last 1
    $ens = if ($ensLine) { ($ensLine.Line -replace '.*ensamble\[(.+)\].*','$1') } else { "?" }
    
    $status = "[$now] lines=$lines opens=$opens closes=$closes errors=$errors btc_ml=$btcMl ensemble=[$ens]"
    Write-Host $status
    Add-Content $report $status
    
    if ($errors -gt 0) {
        Write-Host "🚨 ERRORES DETECTADOS — revisar $log" -ForegroundColor Red
    }
    if ($closes -gt 0 -and ($opens -  $closes) -eq 0) {
        Write-Host "✅ Todas las posiciones cerradas" -ForegroundColor Green
    }
    
    Start-Sleep -Seconds 300
}
