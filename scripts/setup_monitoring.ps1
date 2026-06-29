$ErrorActionPreference = "Stop"
$toolsDir = "C:\Users\jhona\Documents\Proyectos\Trader Gemini\monitoring_tools"

if (!(Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
}

Write-Host "📥 Descargando Prometheus..."
$promUrl = "https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.windows-amd64.zip"
$promZip = "$toolsDir\prometheus.zip"
if (!(Test-Path $promZip)) {
    Invoke-WebRequest -Uri $promUrl -OutFile $promZip
}
Write-Host "📦 Extrayendo Prometheus..."
Expand-Archive -Path $promZip -DestinationPath $toolsDir -Force

Write-Host "📥 Descargando Grafana..."
$grafanaUrl = "https://dl.grafana.com/enterprise/release/grafana-enterprise-11.1.0.windows-amd64.zip"
$grafanaZip = "$toolsDir\grafana.zip"
if (!(Test-Path $grafanaZip)) {
    Invoke-WebRequest -Uri $grafanaUrl -OutFile $grafanaZip
}
Write-Host "📦 Extrayendo Grafana..."
Expand-Archive -Path $grafanaZip -DestinationPath $toolsDir -Force

# Configurar Prometheus
$promDir = Get-ChildItem $toolsDir -Filter "prometheus-*" -Directory | Select-Object -First 1
$promYaml = "$($promDir.FullName)\prometheus.yml"
$yamlContent = @"
global:
  scrape_interval: 1s
  evaluation_interval: 1s

scrape_configs:
  - job_name: 'trader_gemini'
    static_configs:
      - targets: ['localhost:9091']
"@
Set-Content -Path $promYaml -Value $yamlContent

Write-Host "✅ Instalación Completada."
