# 🎮 Captain's Manual: Trader Gemini Command Reference

This document is the **Single Source of Truth** for all executable commands, scripts, and utilities in the Trader Gemini ecosystem.

---

## 🚀 1. Core Execution (Flight Deck)
Primarily used to launch the trading engine.

### 🔥 Futures Mode (Live Trading)
**Preferred startup method.** Activates environment, forces correct directory, and prevents auto-close on error.
```powershell
.\START_FUTURES.bat
```
*Configuration Default:* Scalping Mode, 30% Position, 3 Concurrent.

### 🪙 Spot Mode (Accumulation)
For long-term accumulation or testing without leverage.
```powershell
.\START_SPOT.bat
```

### 🧠 Advanced Manual Launch
If you need to override defaults specifically (Capital, Symbols, Mode).
```powershell
# Custom Capital Override (e.g. $100)
python main.py --mode futures --capital 100.0

# Specific Symbol Focus
python main.py --mode futures --symbols "BTC/USDT,ETH/USDT"

# Scalping Optimized Mode (Force overrides)
python main.py --mode scalping --capital 15.0
```

---

## 📊 2. Dashboard & Monitoring (Control Tower)

### 📈 Futures Dashboard
Launches Streamlit interface for Futures data (Port 8501).
```powershell
.\DASHBOARD_FUTURES.bat
```
*URL:* `http://localhost:8501`

### 🔮 Oracle Live View (Brain Scan)
Shows the real-time probability and confidence for all active symbols.
**Safe to run while bot is active.**
```powershell
python check_oracle.py
```

### 📉 Spot Dashboard (Legacy)
```powershell
.\DASHBOARD_SPOT.bat
```

---

## 🛡️ 3. Diagnostics & Pre-Flight (Safety)

### 🩺 Daily Health Check (MORNING ROUTINE)
Verifies API latency, Capital Sync, and Env variables. **Run this every morning.**
```powershell
python health_check.py
```

### ✈️ Flight Diagnostic
Deep scan of configuration integrity and file system permissions.
```powershell
python flight_diagnostic.py
```

### 💧 Liquidity & Spread Check
Analyzes current market conditions without trading. Good for checking spread costs.
```powershell
python live_liquidity_check.py
```

### 🔎 Database Inspector
View raw contents of the local SQLite/JSON database to debug position states.
```powershell
python inspect_db.py
```

---

## �️ 4. Maintenance & Reset (Hangar)

### 🧨 Production Reset (EMERGENCY ONLY)
**WARNING:** Wipes local state (PnL, Positions history). Does **NOT** close positions on Binance. Use only after a crash or for a fresh start.
```powershell
python production_reset.py
```

### 🩹 Model Fixer
Repairs corrupted ML model files (`.joblib`) if the bot fails to load them.
```powershell
python fix_corrupted_models.py
```

### 🧹 Clean PyCache
Removes compiled python files to ensure clean execution logic.
```powershell
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse
```

---

## 🧠 5. Optimization & AI (Research Lab)

### 👻 Shadow Audit (Parameter Tuning)
Simulates recent market data to find the optimal Z-Score and RSI thresholds.
**Schedule:** Every Sunday.
```powershell
python run_shadow_audit.py
```

### 💥 Resilience Simulator (Flash Crash)
Tests if the Risk Manager correctly handles a -15% candle event.
```powershell
python simulate_flash_crash.py
```

### 📊 Adaptive Report
Generates a PDF report on how the bot adapted to market regimes.
```powershell
python report_sim_adaptive.py
```

---

## 🧪 6. Unit Tests & Verification

### 🕵️ Validate Capital Logic
Simulates various portfolio scenarios to ensure sizing logic is safe.
```powershell
python validate_capital.py
```

### 🔗 Verify CCXT Connection
Debug tool to test raw connectivity to Binance Futures API.
```powershell
python debug_ccxt_dapi.py
```

### 🔁 Hot Reload Test
Verifies that changing `config.py` updates the bot without restart.
```powershell
python test_hot_reload.py
```

---

## 👨‍🏫 Professor's Cheat Sheet

| Intent | Command |
| :--- | :--- |
| **Start Trading** | `.\START_FUTURES.bat` |
| **Check Dashboard** | `.\DASHBOARD_FUTURES.bat` |
| **Ask the Oracle** | `python check_oracle.py` |
| **I'm scared/Panic** | `Ctrl+C` in bot terminal window |
| **Morning Check** | `python health_check.py` |
| **Bot Crashed?** | `Get-Content logs/bot_*.json -Tail 20` |
| **Reset Everything** | `python production_reset.py` (Be careful) |

---
> **Doc. Version:** 2.0 (Full Arsenal)
> **Updated:** 2026-02-04
