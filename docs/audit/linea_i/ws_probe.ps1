param(
    [Parameter(Mandatory = $true)][string]$Url,
    [int]$Seconds = 40
)
# Sonda pública de solo lectura: cuenta eventos por tipo ("e") y mide
# recepción − E con el reloj local corregido por el offset del servidor.
$ErrorActionPreference = 'Stop'

function Get-ServerOffset {
    $best = $null
    for ($i = 0; $i -lt 5; $i++) {
        $t0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $r = Invoke-RestMethod -Uri 'https://fapi.binance.com/fapi/v1/time' -TimeoutSec 5
        $t1 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
        $rtt = $t1 - $t0
        $off = [double]$r.serverTime - ($t0 + $t1) / 2.0
        if ($null -eq $best -or $rtt -lt $best.rtt) { $best = [pscustomobject]@{ rtt = $rtt; off = $off } }
    }
    return $best
}

$o = Get-ServerOffset
"REST RTT minimo = {0} ms, offset servidor-local = {1:N1} ms" -f $o.rtt, $o.off

$ws = [System.Net.WebSockets.ClientWebSocket]::new()
$cts = [System.Threading.CancellationTokenSource]::new()
$cts.CancelAfter([TimeSpan]::FromSeconds($Seconds + 15))
$tc0 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
$ws.ConnectAsync([Uri]$Url, $cts.Token).GetAwaiter().GetResult()
$tc1 = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
"Conectado a $Url en $($tc1 - $tc0) ms (TCP+TLS+upgrade)"

$buf = New-Object byte[] 262144
$seg = [ArraySegment[byte]]::new($buf)
$counts = @{}
$lat = @{}
$sw = [Diagnostics.Stopwatch]::StartNew()
$ms = New-Object System.IO.MemoryStream
$re = [regex]'"e":"([A-Za-z_]+)".*?"E":(\d+)'
try {
    while ($sw.Elapsed.TotalSeconds -lt $Seconds -and $ws.State -eq 'Open') {
        $ms.SetLength(0)
        do {
            $res = $ws.ReceiveAsync($seg, $cts.Token).GetAwaiter().GetResult()
            $ms.Write($buf, 0, $res.Count)
        } while (-not $res.EndOfMessage)
        $recv = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() + $o.off
        $txt = [Text.Encoding]::UTF8.GetString($ms.ToArray())
        $m = $re.Match($txt)
        if ($m.Success) {
            $e = $m.Groups[1].Value
            $counts[$e] = 1 + [int]$counts[$e]
            if (-not $lat.ContainsKey($e)) { $lat[$e] = New-Object System.Collections.Generic.List[double] }
            $lat[$e].Add($recv - [double]$m.Groups[2].Value)
        } else {
            $counts['(sin e/E)'] = 1 + [int]$counts['(sin e/E)']
        }
    }
} catch {
    "Fin de lectura: $($_.Exception.GetBaseException().Message)"
}
"Duracion efectiva: {0:N1} s" -f $sw.Elapsed.TotalSeconds
foreach ($k in ($counts.Keys | Sort-Object)) {
    $line = "  {0,-22} n={1,6}  tasa={2,8:N2}/s" -f $k, $counts[$k], ($counts[$k] / [Math]::Max(1, $sw.Elapsed.TotalSeconds))
    if ($lat.ContainsKey($k) -and $lat[$k].Count -gt 0) {
        $s = $lat[$k].ToArray(); [Array]::Sort($s)
        $p = { param($q) $s[[Math]::Min($s.Length - 1, [int]([Math]::Floor($q * ($s.Length - 1))))] }
        $line += "  recv-E ms: min={0:N0} p50={1:N0} p90={2:N0} p99={3:N0} max={4:N0}" -f $s[0], (& $p 0.5), (& $p 0.9), (& $p 0.99), $s[-1]
    }
    $line
}
try { $ws.Abort() } catch {}
