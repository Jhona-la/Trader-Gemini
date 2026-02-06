# 🏦 TRADER GEMINI: INSTITUTIONAL HFT SYSTEM
**Version**: 2.1.0 (Institutional Candidate) | **Architecture**: Event-Driven Hybrid ML/Quant

Trader Gemini es un sistema de trading algorítmico de Grado Institucional diseñado para operar cestas de activos (26 símbolos) en Binance Futures con latencia mínima y validación estadística robusta.

---

## 🚀 QUICKSTART

### 1. Requisitos
- Python 3.10+
- Cuenta Binance Futures (Testnet o Real)
- Claves API en `.env`

### 2. Instalación
```bash
pip install -r requirements.txt
```

### 3. Ejecución (Modo Futures - Recomendado)
```bash
python main.py --mode futures
```
Esto iniciará:
- **Engine**: Motor de eventos (Trade Loop).
- **Dashboard**: Interfaz Web en `http://localhost:8501`.
- **Health Supervisor**: Monitor de integridad de hilos.

---

## 🧠 ARQUITECTURA DEL SISTEMA

### 1. Core (El Cerebro)
- `engine.py`: Event Loop lock-free de baja latencia.
- `portfolio.py`: Ledger atómico con soporte de `math_stats` (Hurst, Beta).
- `world_awareness.py`: Inyección de contexto global (Sesiones Londres/NY).

### 2. Strategies (La Lógica)
- `MLStrategyHybridUltimate`: Ensemble (RF + XGB + GBM) con Kelly Size Dinámico.
- `StatisticalStrategy`: Arbitraje Estadístico con Regresión Robusta (RANSAC) y Half-Life.

### 3. Safety (El Escudo)
- `risk_manager.py`: Kill Switch, Max Drawdown, Filtros de Correlación.
- `DatabaseHandler`: Persistencia WAL (Write-Ahead Logging) para concurrencia real.

---

## 📊 GESTIÓN DE ACTIVOS (26 SÍMBOLOS)

El sistema opera una **Cesta Institucional** definida en `config.py`.
> Para modificar activos, ver [docs/SYMBOLS.md](docs/SYMBOLS.md).

---

## 🛡️ PROTOCOLOS DE SEGURIDAD

1. **Kill Switch Matemático**: Si la Expectativa Matemática ($E$) de las últimas 20 operaciones es negativa, el sistema bloquea nuevas entradas (`utils/analytics.py`).
2. **Crash Recovery**: El estado se guarda atómicamente en `live_status.json` y SQLite. Si el proceso muere, se restaura la posición exacta al reiniciar.
3. **Cross-Pollination**: Las estrategias comparten inteligencia (Hurst Exponent) para evitar operar contra el régimen de mercado.

---

**Desarrollado por**: Equipo de Quant Development
**Estado**: FASE 7 COMPLETADA (System Hardening)
