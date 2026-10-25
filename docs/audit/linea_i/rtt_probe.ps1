# Uso: rtt_probe.ps1 <salida.csv> [-n 40]. RTT y desfase de reloj frente a GET /fapi/v1/time.
param([Parameter(Mandatory)][string]$csv, [int]$n = 40)
Add-Type -AssemblyName System.Net.Http
$h = New-Object System.Net.Http.HttpClientHandler
$c = New-Object System.Net.Http.HttpClient($h)
$c.Timeout = [TimeSpan]::FromSeconds(5)
$out = @()
foreach ($base in @('https://fapi.binance.com','https://testnet.binancefuture.com')) {
  $sw = [System.Diagnostics.Stopwatch]::new()
  for ($i = 0; $i -lt $n; $i++) {
    $t0w = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    $sw.Restart()
    try {
      $r = $c.GetStringAsync("$base/fapi/v1/time").GetAwaiter().GetResult()
      $sw.Stop()
      $t1w = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
      $st = ([regex]::Match($r,'"serverTime":(\d+)')).Groups[1].Value
      $rtt = $sw.Elapsed.TotalMilliseconds
      $off = [int64]$st - ($t0w + ($t1w - $t0w)/2)
      $out += [pscustomobject]@{ base=$base; i=$i; rtt=[math]::Round($rtt,1); offset=[math]::Round($off,1) }
    } catch { $out += [pscustomobject]@{ base=$base; i=$i; rtt=-1; offset=$null } }
    Start-Sleep -Milliseconds 700
  }
}
$out | Export-Csv -NoTypeInformation -Path $csv
foreach ($g in ($out | Group-Object base)) {
  $ok = $g.Group | Where-Object { $_.rtt -ge 0 } | Select-Object -Skip 1
  $r = $ok.rtt | Sort-Object
  $k = $r.Count
  "{0}: n={1} min={2} p50={3} p90={4} max={5} frac<=350={6:N2} offset_p50={7}" -f $g.Name, $k, $r[0], $r[[int]($k*0.5)], $r[[int]($k*0.9)], $r[$k-1], (($r | Where-Object { $_ -le 350 }).Count / [math]::Max($k,1)), (($ok.offset | Sort-Object)[[int]($k*0.5)])
}
