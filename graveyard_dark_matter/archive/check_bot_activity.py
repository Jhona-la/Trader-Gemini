"""
Diagnostic script to verify multi-timeframe calculations
Run this while the bot is running to see what's happening
"""
import pandas as pd
import os
from datetime import datetime

def check_futures_activity():
    print("=" * 60)
    print("🔍 FUTURES BOT DIAGNOSTIC")
    print("=" * 60)
    
    status_file = "dashboard/data/futures/status.csv"
    trades_file = "dashboard/data/futures/trades.csv"
    
    # Check if files exist
    if not os.path.exists(status_file):
        print("❌ Status file not found. Bot may not have started yet.")
        return
    
    # Read status.csv
    try:
        df = pd.read_csv(status_file)
        if len(df) > 0:
            latest = df.iloc[-1]
            print(f"\n📊 Latest Status:")
            print(f"   Timestamp: {latest['timestamp']}")
            print(f"   Equity: ${latest['equity']:.2f}")
            print(f"   Cash: ${latest['cash']:.2f}")
            print(f"   Open Positions: {latest['positions']}")
            print(f"\n   Total rows: {len(df)}")
        else:
            print("⚠️  Status file is empty")
    except Exception as e:
        print(f"❌ Error reading status: {e}")
    
    # Check trades
    try:
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
            print(f"\n💼 Trades: {len(trades_df)} total")
            if len(trades_df) > 0:
                print("\n   Last 3 trades:")
                print(trades_df.tail(3)[['timestamp', 'symbol', 'side', 'quantity', 'price']])
        else:
            print("\n💼 No trades.csv file yet")
    except Exception as e:
        print(f"❌ Error reading trades: {e}")

def check_spot_activity():
    print("\n" + "=" * 60)
    print("🔍 SPOT BOT DIAGNOSTIC")
    print("=" * 60)
    
    status_file = "dashboard/data/spot/status.csv"
    trades_file = "dashboard/data/spot/trades.csv"
    
    # Check if files exist
    if not os.path.exists(status_file):
        print("❌ Status file not found. Spot bot may not have started yet.")
        return
    
    # Read status.csv
    try:
        df = pd.read_csv(status_file)
        if len(df) > 0:
            latest = df.iloc[-1]
            print(f"\n📊 Latest Status:")
            print(f"   Timestamp: {latest['timestamp']}")
            print(f"   Equity: ${latest['equity']:.2f}")
            print(f"   Cash: ${latest['cash']:.2f}")
            print(f"   Open Positions: {latest['positions']}")
            print(f"\n   Total rows: {len(df)}")
        else:
            print("⚠️  Status file is empty")
    except Exception as e:
        print(f"❌ Error reading status: {e}")
    
    # Check trades
    try:
        if os.path.exists(trades_file):
            trades_df = pd.read_csv(trades_file)
            print(f"\n💼 Trades: {len(trades_df)} total")
            if len(trades_df) > 0:
                print("\n   Last 3 trades:")
                print(trades_df.tail(3)[['timestamp', 'symbol', 'side', 'quantity', 'price']])
        else:
            print("\n💼 No trades.csv file yet")
    except Exception as e:
        print(f"❌ Error reading trades: {e}")

if __name__ == "__main__":
    check_futures_activity()
    check_spot_activity()
    
    print("\n" + "=" * 60)
    print("✅ Diagnostic Complete")
    print("=" * 60)
    print("\nIf you see 'Status file not found', the bots haven't logged yet.")
    print("If you see empty status files, the bots are initializing.")
    print("If you see 0 trades, the bots haven't found signals yet.")
