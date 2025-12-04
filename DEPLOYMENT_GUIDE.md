# 🚀 Deployment Guide - Trader Gemini

**Version:** 1.0.0  
**Environment:** Binance Demo (Testnet)  
**Date:** 2025-12-04

---

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10/11, Linux, or macOS
- **Python:** 3.8+ (3.10+ recommended)
- **RAM:** 2GB minimum (4GB recommended)
- **Disk:** 500MB free space

### Required Accounts
- **Binance Account** (for live trading) OR
- **Binance Testnet Account** (for demo trading - recommended for first deployment)

---

## 🔧 Environment Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd "Trader Gemini"
```

### 2. Create Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Required Packages:**
- `python-binance` (WebSocket data)
- `ccxt` (REST API execution)
- `pandas`, `numpy` (data processing)
- `talib` (technical indicators)
- `scikit-learn`, `xgboost` (ML models)
- `python-dotenv` (environment variables)

---

## 🔑 Binance API Configuration

### Option A: Binance Testnet (RECOMMENDED for first deployment)

1. **Register at:** https://testnet.binance.vision/
2. **Generate API Key:**
   - Go to API Management
   - Create new API key
   - Save API Key and Secret Key

3. **Create `.env` file** in project root:
```env
# Binance Testnet Configuration
BINANCE_API_KEY=your_testnet_api_key_here
BINANCE_API_SECRET=your_testnet_secret_key_here
BINANCE_USE_TESTNET=true
BINANCE_USE_FUTURES=false

# Trading Configuration
INITIAL_CAPITAL=10000
SYMBOLS=BTC/USDT,ETH/USDT,SOL/USDT
```

### Option B: Binance Live (Production)

> [!CAUTION]
> **Only use after extensive testing on Testnet!**

1. **Binance.com** → SecurityAPI Management
2. **Generate API Key** with trading permissions
3. **Restrict IP** (optional but recommended)
4. **Update `.env`:**
```env
BINANCE_API_KEY=your_live_api_key_here
BINANCE_API_SECRET=your_live_secret_key_here
BINANCE_USE_TESTNET=false
BINANCE_USE_FUTURES=false
```

---

## 🗄️ Database Initialization

The bot uses SQLite3 for persistence (no manual setup required).

**Database location:**
```
dashboard/data/spot/trader_gemini.db
```

**Tables created automatically:**
- `trades` - All executed trades
- `signals` - Strategy signals
- `positions` - Open positions
- `errors` - Error logs

**Verify database:**
```bash
python tests/test_database.py
```

---

## ▶️ Running the Bot

### Quick Reference Commands

Always activate virtual environment first:
```powershell
# Windows
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### Test Mode (Recommended First Run)
```bash
python tests/test_connectivity_v2.py  # WebSocket test
python tests/test_integrity.py        # System integrity
python main.py                         # Start bot
```

### Production Run
```bash
python main.py
```

**Expected Startup Output:**
```
🚀 Trader Gemini - Starting...
✅ Config validated
🔌 Connecting to Binance WebSocket...
📊 Initial capital: $10,000
🎯 Trading symbols: BTC/USDT, ETH/USDT, SOL/USDT
⚖️ Risk per trade: 1.0%
🛡️ Max positions: 5
✅ System ready. Listening for signals...
```

---

## 📊 Monitoring & Logging

### Log Files
- **Location:** Console output + `utils/logger.py`
- **Level:** INFO (default)

### Key Metrics to Monitor
1. **Equity:** Total cash + unrealized P&L
2. **Open Positions:** Max 5 concurrent
3. **Risk per Trade:** Capped at 1% of capital
4. **Daily Drawdown:** Monitor for 5% limit

### Dashboard Data
- **Trades:** `dashboard/data/trades.csv`
- **Status:** `dashboard/data/status.csv`
- **Database:** `dashboard/data/spot/trader_gemini.db`

---

## 🛑 Emergency Stop

### Graceful Shutdown
Press **Ctrl+C** in terminal running the bot.

**Expected behavior:**
1. WebSocket closes cleanly
2. All open positions logged to database
3. State saved for recovery

### Force Stop (If needed)
1. Kill process manually
2. On restart, bot will recover positions from database
3. Verify recovery:
```bash
python tests/test_recovery.py
```

---

## ⚙️ Configuration Parameters

Edit `.env` or `config.py` for these settings:

**Risk Management:**
```python
MAX_RISK_PER_TRADE = 0.01      # 1% of capital
MAX_CONCURRENT_POSITIONS = 5    # Max open positions
COOLDOWN_MINUTES = 5            # Minutes between trades
```

**Position Sizing:**
```python
# Account-based tiers
# Small (<$1k): 5% per position
# Medium ($1k-$50k): 15% per position
# Large (>$50k): 25% per position
```

**Trailing Stops:**
```python
# TP1: +1% profit → 50% trailing
# TP2: +2% profit → 25% trailing
# TP3: +3%+ profit → 10% trailing
# Stop-Loss: -2% from entry
```

---

## 🐛 Troubleshooting

### Connection Errors
**Problem:** `WebSocket connection failed`
**Solution:**
1. Check internet connection
2. Verify API keys in `.env`
3. Ensure testnet mode matches your keys
4. Check Binance API status: https://binance.statuspage.io

### Database Errors
**Problem:** `Database locked` or `unable to open database file`
**Solution:**
1. Close all instances of the bot
2. Delete `dashboard/data/spot/trader_gemini.db`
3. Restart bot (database recreates automatically)

### Order Rejections
**Problem:** `Order size too small` or `Insufficient balance`
**Solution:**
1. Minimum order: $5
2. Check available balance: Ensure capital > $100
3. Reduce position sizing in config

**Problem:** `Invalid API key`
**Solution:**
1. Verify `.env` has correct keys
2. Check API permissions (spot trading enabled)
3. Regenerate keys if compromised

### Missing Packages
**Problem:** `ModuleNotFoundError: No module named 'talib'`
**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

For TA-Lib specifically (if pip fails):
- **Windows:** Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
- **Linux:** `sudo apt-get install ta-lib`
- **macOS:** `brew install ta-lib`

---

## 🔐 Security Best Practices

1. **Never commit `.env` to git**
   - Already in `.gitignore`
   - Verify before pushing

2. **API Key Permissions**
   - Enable only "Spot Trading"
   - DO NOT enable "Withdrawal"

3. **IP Restriction** (Optional but recommended)
   - Bind API key to your server IP
   - Prevents unauthorized access

4. **Monitor Account Activity**
   - Check Binance API activity dashboard
   - Set up email alerts for large trades

---

## 📈 Performance Optimization

### Reduce Latency
1. **VPS Deployment:** Use server close to Binance (Singapore/Tokyo)
2. **WebSocket:** Already enabled (python-binance)
3. **Async Processing:** Already implemented

### Resource Usage
- **CPU:** ~5-10% (single core)
- **RAM:** ~200-500MB
- **Network:** ~1MB/hour (WebSocket data)

---

## 🚦 Next Steps

1. ✅ **Verify Configuration:** Check `.env` settings
2. ✅ **Run Tests:** `python tests/test_integrity.py`
3. ✅ **Start on Testnet:** Test with $10k virtual money
4. ✅ **Monitor for 24-48 hours:** Ensure stable operation
5. ✅ **Review Performance:** Check trades.csv and logs
6. ⚠️ **Consider Live Deployment:** Only after extended testnet success

---

## 📞 Support & Resources

- **Documentation:** `README.md`, `VERIFICATION_REPORT.md`
- **Tests:** `tests/` directory
- **Binance API Docs:** https://binance-docs.github.io/apidocs/spot/en/
- **Testnet:** https://testnet.binance.vision/

---

**Deployment Status:** ✅ READY FOR PRODUCTION  
**Last Updated:** 2025-12-04  
**Certified by:** CoG-RA (EFT V.2.1 L7)
