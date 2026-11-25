# Supervisor de la captura en vivo v2 (40 simbolos) de Trader Gemini. Integrador, 15 nov 2026.
# Datos PUBLICOS de Binance futuros USD-M: sin llaves, sin cuenta y sin ordenes.
# Base: Desktop\PC-Turbo\captura_l2_supervisor.ps1 (v1). Cambios (spec Qoder 2.8): sin fecha de fin,
# guarda de disco a 40 GB (REGLAS_DE_TRABAJO.md regla 4), corre oculto (tarea programada con
# conhost --headless), una sola instancia (mutex), no arranca antes de -Inicio, prioridad AboveNormal
# del colector (como captura_prioridad.ps1) y espera DISCO-BAJO en lugar de terminar.
# Parar: crear PARAR_CAPTURA2.txt en la carpeta -Raiz (el colector sella la hora y sale con 0).
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(Mandatory = $true)][string]$Raiz,
    [Parameter(Mandatory = $true)][string]$Roster,
    [string]$Venue = 'mainnet',
    [string]$Inicio = '',
    [string]$Mensajes = ''
)
$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path $Raiz | Out-Null
$log = Join-Path $Raiz ("supervisor_v2_{0}.log" -f $Venue)
$parar = Join-Path $Raiz 'PARAR_CAPTURA2.txt'
# Umbral de REGLAS_DE_TRABAJO.md regla 4, en GiB como el supervisor v1 (/1GB).
$minLibreGB = 40
# Supervisor v1 linea 39-40: cinco salidas con error seguidas, cada una en menos de un minuto.
$maxRapidas = 5
$rapidaSeg = 60
# Pausa entre relanzamientos (v1 linea 41) y entre comprobaciones con disco bajo (vuelta del hilo de archivo x 5).
$pausaSeg = 5
$pausaDiscoSeg = 300
function Anota([string]$t) {
    $l = "{0} {1}" -f (Get-Date -Format o), $t
    for ($i = 0; $i -lt 5; $i++) {
        try { [IO.File]::AppendAllText($log, $l + "`r`n", [Text.UTF8Encoding]::new($false)); break } catch { Start-Sleep -Milliseconds 200 }
    }
}
function Aviso([string]$titulo, [string]$texto) {
    if (-not $Mensajes) { return }
    try {
        $f = Join-Path $Mensajes ("{0}-integrador-{1}.md" -f (Get-Date -Format 'yyyyMMdd-HHmm'), $titulo)
        if (-not (Test-Path -LiteralPath $f)) { [IO.File]::WriteAllText($f, $texto + "`r`n", [Text.UTF8Encoding]::new($false)) }
    } catch { Anota ("no pude escribir el aviso {0}: {1}" -f $titulo, $_.Exception.Message) }
}
$creado = $false
$mutex = New-Object System.Threading.Mutex($true, 'Global\TraderGemini_captura_v2_supervisor', [ref]$creado)
if (-not $creado) { Anota ("OTRA INSTANCIA del supervisor ya corre; esta (PID {0}) sale" -f $PID); exit 0 }
try {
    Set-Content -LiteralPath (Join-Path $Raiz 'supervisor_v2.pid') -Value $PID -Encoding ASCII
    if ($Inicio) {
        $ini = [DateTimeOffset]::Parse($Inicio)
        Anota ("SUPERVISOR v2 arranca (PID {0}); espera hasta {1}" -f $PID, $ini.ToString('o'))
        while ([DateTimeOffset]::Now -lt $ini) {
            if (Test-Path -LiteralPath $parar) { Anota 'PARADA antes de empezar: existe PARAR_CAPTURA2.txt'; exit 0 }
            Start-Sleep -Seconds ([Math]::Max(1, [Math]::Min(30, [int]($ini - [DateTimeOffset]::Now).TotalSeconds)))
        }
    }
    Anota ("SUPERVISOR v2 activo (PID {0}); venue {1}; raiz {2}; exe {3}" -f $PID, $Venue, $Raiz, $Exe)
    $rapidas = 0
    $avisoDisco = $false
    while ($true) {
        if (Test-Path -LiteralPath $parar) { Anota 'PARADA: existe PARAR_CAPTURA2.txt'; break }
        $libre = -1
        try { $libre = [int]((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB) } catch { }
        if ($libre -ge 0 -and $libre -lt $minLibreGB) {
            Anota ("DISCO-BAJO: {0} GB libres en C: (< {1}); no lanzo; reviso en {2} s" -f $libre, $minLibreGB, $pausaDiscoSeg)
            if (-not $avisoDisco) {
                Aviso 'DISCO-BAJO' ("DISCO-BAJO: la captura v2 esta en pausa: C: tiene {0} GB libres (umbral {1}, REGLAS_DE_TRABAJO.md regla 4). Relanza sola al volver a {1} GB. Registro: {2}" -f $libre, $minLibreGB, $log)
                $avisoDisco = $true
            }
            Start-Sleep -Seconds $pausaDiscoSeg
            continue
        }
        $avisoDisco = $false
        $sello = Get-Date -Format 'yyyyMMdd-HHmmss'
        $out = Join-Path $Raiz ("colector_{0}.stdout.log" -f $sello)
        $err = Join-Path $Raiz ("colector_{0}.stderr.log" -f $sello)
        $t0 = Get-Date
        $p = $null
        try {
            $p = Start-Process -FilePath $Exe -ArgumentList @('capture', $Venue, "`"$Raiz`"", "`"$Roster`"") -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
        } catch { Anota ("NO PUDE LANZAR el colector: {0}" -f $_.Exception.Message) }
        if ($p) {
            $null = $p.Handle  # sin esto Windows PowerShell pierde ExitCode al terminar el proceso
            Set-Content -LiteralPath (Join-Path $Raiz 'colector_v2.pid') -Value $p.Id -Encoding ASCII
            Anota ("LANZO el colector PID {0} (C: libre {1} GB); stdout {2}" -f $p.Id, $libre, $out)
            Start-Sleep -Seconds 2
            try { if (-not $p.HasExited) { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::AboveNormal } } catch { Anota ("no pude subir la prioridad: {0}" -f $_.Exception.Message) }
            $p.WaitForExit()
            $cod = $p.ExitCode
        } else { $cod = -1 }
        $dur = [int]((Get-Date) - $t0).TotalSeconds
        Anota ("SALIO el colector con codigo {0} tras {1} s" -f $cod, $dur)
        if ($cod -eq 0) {
            if (Test-Path -LiteralPath $parar) { Anota 'FIN: salida limpia por PARAR_CAPTURA2.txt'; break }
            Anota 'salida limpia sin PARAR_CAPTURA2.txt (Ctrl+C o cierre del sistema): relanzo'
        }
        if ($cod -eq 3) {
            $mensajeDisco = ("Captura v2 en pausa por disco bajo. Ver {0}. No se borran horas no vencidas." -f $log)
            Anota $mensajeDisco
            Aviso 'DISCO-BAJO' $mensajeDisco
            Start-Sleep -Seconds $pausaDiscoSeg
            continue
        }
        if ($dur -lt $rapidaSeg) { $rapidas++ } else { $rapidas = 0 }
        if ($rapidas -ge $maxRapidas) {
            Anota 'PARADA: cinco salidas con error seguidas, cada una en menos de un minuto (cuidado de IP: no reconectar en bucle)'
            Aviso 'CAPTURA-V2-PARADA' ("La captura v2 se detuvo tras {0} salidas rapidas seguidas. Ver {1} y los colector_*.stderr.log de {2}. Se relanza en el proximo inicio de sesion o con la tarea programada TraderGemini_captura_v2." -f $maxRapidas, $log, $Raiz)
            break
        }
        Start-Sleep -Seconds $pausaSeg
    }
} finally {
    Anota 'SUPERVISOR v2 termina'
    $mutex.ReleaseMutex()
}
