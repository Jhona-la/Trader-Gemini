"""
Capital Validation Script - Admin Mode
Validates wallet balance and notional limits for production trading.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from binance.client import Client

key = os.environ.get('BINANCE_API_KEY')
secret = os.environ.get('BINANCE_SECRET_KEY') or os.environ.get('BINANCE_API_SECRET')

print('='*70)
print('💰 CAPITAL VALIDATION - PRODUCTION MODE')
print('='*70)

c = Client(key, secret)

# Get futures account
a = c.futures_account()

# Extract balances
total_margin = float(a['totalMarginBalance'])
total_wallet = float(a['totalWalletBalance'])
available = float(a['availableBalance'])
unrealized_pnl = float(a['totalUnrealizedProfit'])

print(f'\n📊 BALANCE REPORT:')
print(f'   Total Margin Balance:  ${total_margin:.2f} USDT')
print(f'   Total Wallet Balance:  ${total_wallet:.2f} USDT')
print(f'   Available Balance:     ${available:.2f} USDT')
print(f'   Unrealized PnL:        ${unrealized_pnl:.2f} USDT')

# Notional limit check
MIN_NOTIONAL = 5.0  # Binance minimum ~5 USDT
leverage = 10
min_capital_needed = MIN_NOTIONAL / leverage

print(f'\n📋 NOTIONAL LIMIT VALIDATION:')
print(f'   Binance Min Notional:  ${MIN_NOTIONAL:.2f} USDT')
print(f'   With 10x Leverage:     ${min_capital_needed:.2f} required per trade')

if available >= min_capital_needed:
    max_positions = int(available / min_capital_needed)
    print(f'   ✅ SUFFICIENT: Can open up to {max_positions} minimum positions')
else:
    print(f'   ❌ INSUFFICIENT: Need at least ${min_capital_needed:.2f} USDT')

# Position sizing recommendation
position_pct = 0.30  # 30% per position
recommended_size = available * position_pct
print(f'\n⚡ POSITION SIZING (30% rule):')
print(f'   Recommended per trade: ${recommended_size:.2f} USDT')
print(f'   With 10x leverage:     ${recommended_size * 10:.2f} notional')

# Capital tier
print(f'\n🏷️ CAPITAL TIER:')
if total_wallet < 50:
    tier = 'MICRO'
    strategy = 'Max leverage (10x), minimum positions, aggressive growth'
elif total_wallet < 500:
    tier = 'SMALL'
    strategy = 'Standard leverage (5x), balanced positions'
else:
    tier = 'MEDIUM+'
    strategy = 'Conservative (3x), Half-Kelly criterion'

print(f'   Tier: {tier}')
print(f'   Strategy: {strategy}')

# Sentinel-24 base configuration
print(f'\n🛡️ SENTINEL-24 BASE EQUITY:')
print(f'   Base 0 (Starting Equity): ${total_wallet:.2f} USDT')
print(f'   This will be used as reference for equity drift detection')

print('='*70)
print('✅ VALIDATION COMPLETE')
print('='*70)
