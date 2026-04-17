# 📘 TRADER GEMINI - MANUAL DE OPERACIONES V3 (GÉNESIS)

## 🚀 Comandos Principales

### 1. Iniciar Sistema (Producción)
```powershell
docker compose up -d --build
```
*Inicia todos los contenedores: Bot, Redis, Prometheus, Grafana.*

### 2. Ver Logs (Tiempo Real)
```powershell
docker compose logs -f trader-gemini
```

### 3. Detener Sistema
```powershell
docker compose down
```

### 4. Health Check (Manual)
```powershell
python utils/health_check.py
```

## 📊 Monitoreo

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Métricas**:
    - `omega_cpu_usage`: Uso de CPU.
    - `omega_portfolio_equity`: Balance total.
    - `omega_events_processed`: EPS (Eventos por segundo).

## 🆘 Solución de Problemas

### El bot no conecta a Binance
1. Verificar internet: `ping google.com`
2. Verificar hora: `w32tm /resync`
3. Revisar `.env`: Asegurar `BINANCE_API_KEY` correcta.

### Latencia Alta (> 100ms)
1. Reiniciar Docker: `docker compose restart`
2. Verificar carga de CPU: `docker stats`

### "Order Rejected"
1. Verificar saldo en Binance (USDT).
2. Revisar `min_qty` del par en `config.py`.

---
**Certificado: OMEGA GENESIS - 2026**
