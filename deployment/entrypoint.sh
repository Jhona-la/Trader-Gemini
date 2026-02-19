#!/bin/bash
set -e

# OMEGA PROTOCOL ENTRYPOINT
# -------------------------

echo "🚀 Starting Trader Gemini Container..."

# 1. Environment Check
if [ -z "$BINANCE_API_KEY" ]; then
    echo "⚠️  WARNING: BINANCE_API_KEY is not set. Bot may fail to connect."
fi

# 2. Database Init/Check (Optional)
# If we had a migration script, we'd run it here.
# python scripts/migrate_db.py

# 3. Mode Selection
if [ "$BOT_MODE" == "backtest" ]; then
    echo "🧪 Running in BACKTEST mode..."
    exec python main.py --backtest
else
    echo "⚡ Running in LIVE mode..."
    exec python main.py
fi
