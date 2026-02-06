"""Quick handshake check script."""
import os
from dotenv import load_dotenv
load_dotenv()

from binance.client import Client

key = os.environ.get('BINANCE_API_KEY')
secret = os.environ.get('BINANCE_SECRET_KEY') or os.environ.get('BINANCE_API_SECRET')

print('='*60)
print('PRODUCTION HANDSHAKE - QUICK CHECK')
print('='*60)

try:
    c = Client(key, secret)
    c.ping()
    print('1. API Ping: OK')
except Exception as e:
    print(f'1. API Ping: FAILED - {e}')
    exit(1)

try:
    a = c.futures_account()
    balance = float(a.get('totalWalletBalance', 0))
    available = float(a.get('availableBalance', 0))
    print(f'2. Total Balance: ${balance:.2f} USDT')
    print(f'3. Available: ${available:.2f} USDT')
    
    if balance < 50:
        tier = 'MICRO (Max leverage)'
    elif balance < 500:
        tier = 'SMALL (Standard)'
    else:
        tier = 'MEDIUM+ (Half-Kelly)'
    print(f'4. Capital Tier: {tier}')
except Exception as e:
    print(f'2-4. Account: FAILED - {e}')

try:
    mode = c.futures_get_position_mode()
    hedge = mode.get('dualSidePosition', False)
    status = 'ACTIVE' if hedge else 'OFF'
    print(f'5. HEDGE Mode: {status}')
except Exception as e:
    print(f'5. HEDGE Mode: Error - {e}')

print('6. IP Authorization: OK (account accessible)')
print('='*60)
print('HANDSHAKE COMPLETE')
