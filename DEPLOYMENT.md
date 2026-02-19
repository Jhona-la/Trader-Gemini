# 🚀 TRADER GEMINI: DEPLOYMENT GUIDE

## 📋 Prerequisites
- **Python**: 3.10 or higher.
- **System**: Windows (Preferred for HFT/Event Loop) or Linux (Docker).
- **RAM**: Minimum 8GB (Recommended 16GB).
- **Network**: Low latency connection to Binance (Tokyo/Singapore for Futures).

---

## 🔧 Environment Configuration
1. **Copy Example**:
   ```bash
   copy .env.example .env
   ```
2. **Critical Variables**:
   Edit `.env` and ensure the following are set:
   ```ini
   # BINANCE CREDENTIALS
   BINANCE_API_KEY=your_api_key
   BINANCE_SECRET_KEY=your_secret_key
   
   # NETWORK
   BINANCE_USE_TESTNET=False  # Set True for testing
   BINANCE_USE_FUTURES=True   # Set False for Spot
   
   # TELEGRAM (Optional but Recommended)
   TELEGRAM_ENABLED=True
   TELEGRAM_BOT_TOKEN=your_token
   TELEGRAM_CHAT_ID=your_chat_id
   
   # RISK MANAGEMENT
   MAX_POSITION_SIZE=0.1      # % of Equity per trade
   GLOBAL_STOP_LOSS=0.02      # 2% Daily Checkpoint
   ```

---

## 🐳 Docker Deployment (Linux/Cloud)
1. **Build Image**:
   ```bash
   docker-compose build
   ```
2. **Run in Background**:
   ```bash
   docker-compose up -d
   ```
3. **View Logs**:
   ```bash
   docker-compose logs -f trader
   ```

---

## 🖥️ Manual Deployment (Windows/Local)
Recommended for lowest latency (Direct Kernel Access).

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure C++ Build Tools are installed for TA-Lib.*

2. **Launch God Mode** (High Priority Process):
   ```bash
   .\LAUNCH_GOD_MODE.bat
   ```
   *This script sets process priority to REALTIME/HIGH and disables GC for critical loops.*

---

## 📊 Monitoring & Dashboard
The system includes a Streamlit Pro Dashboard.

1. **Start Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
2. **Access**:
   Open browser at `http://localhost:8501`.

### Metrics to Watch
- **Latency**: Should be < 50ms in `Engine`.
- **Drift**: NTP skew should be < 500ms.
- **Heartbeat**: Ensure `API` and `Engine` components are GREEN.

---

## 🚨 Disaster Recovery
The system uses SQLite in **WAL Mode** for atomic persistence.
- **Crash Recovery**: On restart, the bot automatically reloads positions from `data.db`.
- **Manual Override**: Use the **KILL SWITCH** in the Dashboard to instantly stop all execution.

---

**Developed by**: Protocolo Metal-Core Omega Team
**Audit Status**: CERTIFIED (Fuerza Delta Level VI)
