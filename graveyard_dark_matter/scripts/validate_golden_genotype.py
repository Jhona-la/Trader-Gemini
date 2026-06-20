#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════
🧬 GOLDEN GENOTYPE VALIDATION — MULTI-SYMBOL SCALE TEST
═══════════════════════════════════════════════════════════════════════
QUÉ: Backtest multi-símbolo (10 coins, 3 días) para validar que el
     Golden Genotype del Hyper-Evolver-V2 mantiene 80%+ WR a escala.
POR QUÉ: El evolver optimizó con 1 símbolo (BTC). Necesitamos confirmar
     que los parámetros escalan a 10+ monedas sin degradación.
PARA QUÉ: Validar que la meta de cientos de trades/día con 80% WR
     es alcanzable antes de ir a producción.
CÓMO: Replica EXACTAMENTE el loop de quick_diagnostic_bt.py (intra-bar,
     SL/TP dual-pass, SimpleExecutor) pero con 10 símbolos × 3 días.
CUÁNDO: Post-inyección de Config.Mutations.
DÓNDE: scripts/validate_golden_genotype.py
QUIÉN: Quant Developer + QA Engineer
"""
import os, sys, time, io, contextlib, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRADER_GEMINI_BACKTEST'] = 'true'

from config import Config
from core.backtest_infra import fetch_binance_data, BacktestDataProvider, COMMISSION_PCT, COMMISSION_MAKER
from core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from core.enums import EventType, SignalType, OrderSide, OrderType
from core.portfolio import Portfolio
from risk.risk_manager import RiskManager
from queue import Queue
from utils.cooldown_manager import cooldown_manager
from datetime import datetime, timezone
import numpy as np
import pandas as pd

Config.TELEGRAM_ENABLED = False
Config.EMAIL_ENABLED = False
Config.DISCORD_ENABLED = False
if hasattr(Config, 'Observability'):
    Config.Observability.TELEGRAM_ENABLED = False
    Config.Observability.DISCORD_ENABLED = False
    Config.Observability.EMAIL_ENABLED = False

random.seed(42)
np.random.seed(42)

lock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'STOP_TRADING.LOCK')
if os.path.exists(lock_file):
    os.remove(lock_file)

from core.backtest_infra import fetch_multi_symbol_data

# ═══════════════════════════════════════════════════════════
print("═" * 70)
print("🧬 GOLDEN GENOTYPE VALIDATION — 10 SYMBOLS × 3 DAYS")
print("═" * 70)

symbols = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT'
]
DAYS = 3

print(f"Downloading {DAYS} days for {len(symbols)} symbols...")
all_data = fetch_multi_symbol_data(symbols, days=DAYS)

valid_symbols = [s for s in symbols if all_data.get(s) is not None and len(all_data[s]) > 0]
print(f"✅ Got data for {len(valid_symbols)}/{len(symbols)} symbols")

events_queue = Queue()
dp = BacktestDataProvider(events_queue, valid_symbols, all_data)

bt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'data', 'golden_test')
os.makedirs(bt_dir, exist_ok=True)

portfolio = Portfolio(
    initial_capital=Config.INITIAL_CAPITAL,
    csv_path=os.path.join(bt_dir, 'golden_trades.csv'),
    status_path=os.path.join(bt_dir, 'golden_status.csv'),
    auto_save=False
)
portfolio.data_provider = dp
rm = RiskManager(max_concurrent_positions=10, portfolio=portfolio)

from strategies.technical import HybridScalpingStrategy
tech = HybridScalpingStrategy(dp, events_queue, horizon='SCALPING')

# ═══ SimpleExecutor (mirrored from quick_diagnostic_bt.py) ═══
class SimpleExecutor:
    def __init__(self, seed=42):
        self._rng = random.Random(seed)
        self.fills = 0
        self.current_bar_time = datetime.now(timezone.utc)

    def execute(self, order, price):
        if getattr(order, 'strategy_id', '') in ('PREDICTIVE_TP', 'PLACE_TP_LIMIT', 'PLACE_SL_LIMIT'):
            return None
        slip = self._rng.uniform(0.0, 0.0002)
        if order.direction == OrderSide.BUY:
            fp = price * (1 + slip)
        else:
            fp = price * (1 - slip)
        qty = order.quantity
        fc = fp * qty
        comm = fc * COMMISSION_MAKER
        self.fills += 1

        meta = order.metadata.copy() if order.metadata else {}
        meta['actual_order_type'] = 'limit'
        meta['is_close'] = getattr(order, 'is_close', False)
        meta['is_exit'] = getattr(order, 'is_exit', False)

        return FillEvent(
            timeindex=self.current_bar_time,
            symbol=order.symbol,
            exchange='BT_GOLDEN',
            quantity=qty,
            direction=order.direction,
            fill_cost=fc,
            commission=comm,
            strategy_id=order.strategy_id,
            fill_price=fp,
            order_id=f'GOLD_{self.fills}',
            sl_pct=order.sl_pct,
            tp_pct=order.tp_pct,
            horizon=order.horizon,
            leverage=order.leverage,
            metadata=meta,
        )

executor = SimpleExecutor()

# ═══ MAIN LOOP ═══
total = dp.total_epochs
warmup = min(100, total // 20)
signals_total = 0
fills_total = 0
rejected_total = 0
rejection_reasons = {}

cooldown_manager.reset()
if hasattr(cooldown_manager, 'custom_cooldowns'):
    cooldown_manager.custom_cooldowns.clear()

t_start = time.time()
print(f"Total epochs: {total} | Warmup: {warmup}")
print(f"Starting simulation ({len(valid_symbols)} symbols × {DAYS} days)...")

def _process_exit_signals_immediately(stop_sigs, price):
    global fills_total
    if not stop_sigs:
        return
    for sig in stop_sigs:
        order = rm.generate_order(sig, price)
        if order:
            fill = executor.execute(order, price)
            if fill:
                portfolio.update_fill(fill)
                fills_total += 1

for epoch in range(total):
    dp.update_bars()

    market_events = []
    while not events_queue.empty():
        evt = events_queue.get()
        if evt.type == EventType.MARKET:
            market_events.append(evt)

    bar_time = pd.to_datetime(dp.current_time_ms, unit='ms', utc=True)

    for evt in market_events:
        portfolio.update_market_price(evt.symbol, evt.close_price)

    if bar_time:
        executor.current_bar_time = bar_time
        cooldown_manager.set_virtual_time(bar_time)

    if epoch < warmup:
        continue

    # ═══ Intra-bar SL/TP (dual-pass) ═══
    try:
        for evt in market_events:
            bar = dp.get_latest_bars(evt.symbol, n=1)
            if bar is not None and len(bar) > 0:
                # Pass 1: Adverse price (SL)
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos['quantity']
                    if abs(qty) < 1e-8:
                        continue
                    pos_sym = v_key.split('_SCALPING')[0].split('_SWING')[0]
                    if pos_sym != evt.symbol:
                        continue
                    if qty > 0:
                        portfolio.update_market_price(evt.symbol, float(bar['low'][-1]))
                    else:
                        portfolio.update_market_price(evt.symbol, float(bar['high'][-1]))
                
                stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
                _process_exit_signals_immediately(stop_sigs, float(bar['close'][-1]))
                
                # Pass 2: Favorable price (TP)
                for v_key, vpos in list(portfolio.virtual_ledger.items()):
                    qty = vpos['quantity']
                    if abs(qty) < 1e-8:
                        continue
                    pos_sym = v_key.split('_SCALPING')[0].split('_SWING')[0]
                    if pos_sym != evt.symbol:
                        continue
                    if qty > 0:
                        portfolio.update_market_price(evt.symbol, float(bar['high'][-1]))
                    else:
                        portfolio.update_market_price(evt.symbol, float(bar['low'][-1]))
                
                stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
                _process_exit_signals_immediately(stop_sigs, float(bar['close'][-1]))
                
                portfolio.update_market_price(evt.symbol, evt.close_price)

            stop_sigs = rm.check_stops(portfolio, dp, symbol_filter=evt.symbol, now=bar_time)
            _process_exit_signals_immediately(stop_sigs, evt.close_price)

        if bar_time:
            global_stops = rm.check_stops(portfolio, dp, now=bar_time)
            if global_stops:
                for sig in global_stops:
                    price = dp.get_latest_price(sig.symbol) or 0
                    if price:
                        _process_exit_signals_immediately([sig], price)
    except Exception as e:
        from utils.error_handler import SystemIntegrityError
        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    # Run strategy
    try:
        class DummyEvent:
            pass
        dummy = DummyEvent()
        if bar_time:
            dummy.timestamp = bar_time
        tech.generate_signals(event=dummy)
    except:
        from utils.error_handler import SystemIntegrityError
        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    # Process signals
    while not events_queue.empty():
        evt = events_queue.get()
        etype = evt.type.name if hasattr(evt.type, 'name') else str(evt.type)

        if etype == 'SIGNAL':
            signals_total += 1
            price = dp.get_latest_price(evt.symbol)
            if not price:
                continue

            capture = io.StringIO()
            try:
                with contextlib.redirect_stdout(capture):
                    order = rm.generate_order(evt, price)
            except:
                order = None

            if order is None:
                rejected_total += 1
                captured = capture.getvalue()
                reason = 'UNKNOWN'
                for line in captured.strip().split('\n'):
                    if '[RISK] Rejected' in line:
                        reason = line.strip()
                        break
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue

            fill = executor.execute(order, price)
            if fill:
                portfolio.update_fill(fill)
                fills_total += 1

    if epoch % 500 == 0:
        elapsed = time.time() - t_start
        eq = portfolio.get_total_equity()
        open_pos = sum(1 for v in portfolio.virtual_ledger.values() if v['quantity'] != 0)
        print(f"  [{epoch}/{total}] Equity: ${eq:.2f} | Sig: {signals_total} | Fill: {fills_total} | Rej: {rejected_total} | Open: {open_pos} | {elapsed:.1f}s")

# ═══ FORCE CLOSE ═══
for v_key, vpos in list(portfolio.virtual_ledger.items()):
    qty = vpos['quantity']
    if qty == 0:
        continue
    horizon = vpos['horizon']
    parts = v_key.rsplit(f'_{horizon}', 1)
    symbol = parts[0] if len(parts) > 1 else v_key
    price = dp.get_latest_price(symbol)
    if not price:
        continue
    direction = OrderSide.SELL if qty > 0 else OrderSide.BUY
    close_fill = FillEvent(
        timeindex=datetime.now(timezone.utc),
        symbol=symbol,
        exchange='BT_GOLDEN',
        quantity=abs(qty),
        direction=direction,
        fill_cost=abs(qty) * price,
        commission=abs(qty) * price * COMMISSION_MAKER,
        strategy_id='FORCE_CLOSE',
        fill_price=price,
        horizon=horizon,
        metadata={'is_close': True}
    )
    portfolio.update_fill(close_fill)
    fills_total += 1

# ═══ RESULTS ═══
elapsed = time.time() - t_start
eq = portfolio.get_total_equity()
ret = (eq - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL * 100

print(f"\n{'═'*70}")
print(f"🏆 GOLDEN GENOTYPE VALIDATION RESULTS")
print(f"{'═'*70}")
print(f"  Duration:        {elapsed:.1f}s")
print(f"  Symbols:         {len(valid_symbols)}")
print(f"  Days:            {DAYS}")
print(f"  Epochs:          {total}")
print(f"  Signals:         {signals_total}")
print(f"  Fills:           {fills_total}")
print(f"  Rejected:        {rejected_total}")
print(f"  Final Equity:    ${eq:.4f}")
print(f"  Return:          {ret:+.2f}%")

if rejection_reasons:
    print(f"\n  REJECTION BREAKDOWN:")
    for reason, count in sorted(rejection_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {reason}: {count}")

if portfolio.trade_history:
    wins = sum(1 for t in portfolio.trade_history if t['net_pnl'] > 0)
    losses = sum(1 for t in portfolio.trade_history if t['net_pnl'] <= 0)
    total_trades = len(portfolio.trade_history)
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t['net_pnl'] for t in portfolio.trade_history)
    total_fees = sum(t['fees_paid'] for t in portfolio.trade_history)

    print(f"\n  TRADE BREAKDOWN:")
    print(f"    Total: {total_trades} | Wins: {wins} | Losses: {losses} | WR: {wr:.1f}%")
    print(f"    Net PnL: ${total_pnl:.4f} | Fees: ${total_fees:.4f}")

    # Per-symbol
    from collections import defaultdict
    sym_stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0.0})
    for t in portfolio.trade_history:
        s = t['symbol']
        sym_stats[s]['trades'] += 1
        if t['net_pnl'] > 0:
            sym_stats[s]['wins'] += 1
        sym_stats[s]['pnl'] += t['net_pnl']

    print(f"\n  PER-SYMBOL BREAKDOWN:")
    print(f"    {'Symbol':<12} {'Trades':>6} {'Wins':>5} {'WR':>6} {'PnL':>10}")
    print(f"    {'─'*45}")
    for s in sorted(sym_stats.keys()):
        st = sym_stats[s]
        swr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
        print(f"    {s:<12} {st['trades']:>6} {st['wins']:>5} {swr:>5.1f}% ${st['pnl']:>+8.4f}")

    # Exit reasons
    exit_counts = {}
    for t in portfolio.trade_history:
        reason = t['exit_reason']
        exit_counts[reason] = exit_counts.get(reason, 0) + 1
    print(f"\n  EXIT REASONS:")
    for reason, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    {reason}: {count}")

    # Top losers
    losing_trades = sorted([t for t in portfolio.trade_history if t['net_pnl'] < 0], key=lambda x: x['net_pnl'])
    if losing_trades:
        print(f"\n  TOP 5 WORST TRADES:")
        for idx, t in enumerate(losing_trades[:5]):
            print(f"    [{idx+1}] {t['direction']} {t['symbol']} | {t['exit_reason']} | Net: ${t['net_pnl']:.4f} | {t['duration_seconds']}s")

# Config verification
print(f"\n🔧 CONFIG MUTATIONS ACTIVE:")
for k, v in Config.Mutations.items():
    print(f"  {k}: {v}")
print(f"{'═'*70}")
