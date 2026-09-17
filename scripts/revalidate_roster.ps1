# B3.12 — RE-VALIDACION MENSUAL DEL ROSTER (decaimiento del edge)
#
# POR QUE: los 10 modelos promovidos fueron validados jul->ago 2026. El edge
# de microestructura DECAE; sin re-validacion periodica el roster staleda en
# silencio. Este script repite el protocolo completo (aggTrades reales del
# mes previo -> gate cross-month) y reporta la decaimiento por simbolo:
# un modelo que ya no supera el margen anti-empate (0.001) debe RETIRARSE
# (borrar models/{SYM}_SCALP.json; el motor cae al fallback BTCUSDT_SCALP
# y hot-swap lo recoge en <=10s).
#
# USO (a partir de septiembre cerrado):
#   powershell -File scripts/revalidate_roster.ps1 -TrainMonth 2026-08 -ValMonth 2026-09
# Sin args usa el par de meses mas reciente cerrado (editar abajo).

param(
    [string]$TrainMonth = "2026-08",
    [string]$ValMonth   = "2026-09"
)

# Roster promovido 2026-09-15 (ver memoria B3):
$ROSTER = @("BTCUSDT","BNBUSDT","XRPUSDT","SOLUSDT","ADAUSDT","XLMUSDT","ATOMUSDT","NEARUSDT","ICPUSDT","XMRUSDT")

Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "========================================================"
Write-Host "RE-VALIDACION ROSTER: $TrainMonth -> $ValMonth"
Write-Host "========================================================"

foreach ($S in $ROSTER) {
    foreach ($M in @($TrainMonth, $ValMonth)) {
        $bin = "data/${S}_${M}_REAL.bin"
        if (-Not (Test-Path $bin)) {
            Write-Host "[descarga] $S $M"
            & .\target\release\binance_vision_sync.exe --aggtrades $S $M
            if (Test-Path "data/${S}_ticks_REAL.bin") {
                Move-Item "data/${S}_ticks_REAL.bin" $bin -Force
            }
        }
    }
    if (-Not (Test-Path "data/${S}_${TrainMonth}_REAL.bin") -or -Not (Test-Path "data/${S}_${ValMonth}_REAL.bin")) {
        Write-Host "[SKIP] $S sin datos completos"
        continue
    }
    Write-Host "`n======= $S ======="
    & .\target\release\train_forest.exe $S --in "data/${S}_${TrainMonth}_REAL.bin" --val-in "data/${S}_${ValMonth}_REAL.bin" 2>&1 |
        Select-String -Pattern "VEREDICTO|val logloss|GATE|💾|❌"
}

Write-Host "`n========================================================"
Write-Host "ACCION: los que muestren 'GATE: sin evidencia real de edge'"
Write-Host "deben RETIRARSE:  del .\models\{SYM}_SCALP.json"
Write-Host "El motor cae al fallback y hot-swap lo recoge en <=10s."
Write-Host "NO promover a ciegas: solo --promote si delta >= 0.001."
Write-Host "========================================================"
