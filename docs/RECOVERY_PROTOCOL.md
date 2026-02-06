# 🚑 Trader Gemini Disaster Recovery Protocol
> **Institutional Grade Recovery Procedures**
> *Version 1.0 | Last Updated: 2026-02-05*

## 1. System Failure Scenarios

### 🚨 Scenario A: Database Corruption (`trades.db`)
**Symptom**: `DatabaseError: database disk image is malformed` or WAL file locks.
**Auto-Recovery**:
1. System detects SQL error via `utils/data_manager.py`.
2. Switches to `trades_backup.db` automatically (Soft Failover).
3. Logs Critical Alert.
**Manual Recovery**:
1. Stop Bot: `Ctrl+C` or `kill <pid>`.
2. Rename `dashboard/data/scalping/trades.db` to `trades.db.corrupt`.
3. Restart Bot. System rebuilds DB from `trades.csv` (CSV is Master Truth).

### 🚨 Scenario B: API Connection Lost (Binance)
**Symptom**: `NetworkError` or `Timeout` loop.
**Defense**:
1. **Circuit Breaker** (`utils/circuit_breaker.py`) trips after 5 failures.
2. Bot enters `COOLDOWN` state (60s).
3. `LiquidityGuardian` blocks new orders.
**Recovery**:
1. Check Internet/VPN.
2. Verify Binance Status (https://www.binance.com/en/status).
3. Bot auto-retries with exponential backoff.

### 🚨 Scenario C: Execution Hang / Frozen Loop
**Symptom**: Heartbeat stops flowing in logs > 60s.
**Defense**:
1. **Health Supervisor** (`utils/health_supervisor.py`) monitors `last_heartbeat`.
2. If lag > 120s, Supervisor kills `Engine` thread and restarts it (Self-Healing).
**Manual Recovery**:
1. Check `logs/health_log.json`.
2. If persistent, restart process.

### 🚨 Scenario D: Flash Crash / Extreme Volatility
**Symptom**: Drawdown spikes > 5% in seconds.
**Defense**:
1. **Kill Switch** (`risk/kill_switch.py`) activates (Hard Stop).
2. `RiskManager` blocks all ENTRY signals.
3. Positions managed by Exchange-Side Stop Loss (Layer 3 Safety).
**Recovery**:
1. Wait for Market Stabilization.
2. Manual Reset: Call `risk_manager.kill_switch.reset_daily_losses()`.

## 2. Rollback Procedures

### 🔙 Codebase Rollback
If a new update breaks the bot:
1. `git checkout main` (or previous tag).
2. `pip install -r requirements.txt`.
3. Restart.

### 🔙 State Rollback
If `portfolio` state desyncs:
1. Edit `dashboard/data/scalping/status.csv` (last valid row).
2. Delete `dashboard/data/scalping/trades.db-shm` and `trades.db-wal`.
3. Restart. **Auto-Reconciliation** (Phase 13) will sync balance/positions from Binance.

## 3. Emergency Contacts
- **DevOps**: Jhonatan (Local)
- **Binance Support**: https://www.binance.com/en/chat
- **Critical Logs**: `logs/bot_YYYYMMDD.json`
