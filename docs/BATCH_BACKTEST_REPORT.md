# 🚀 REPORTE DE BACKTESTS MASIVOS - GOD MODE (BTC/USDT)

Reporte de viabilidad en múltiples horizontes (1, 7, 15 y 30 días) con $13 de capital inicial.

## 📊 Horizonte Evaluado: 1 Días
- **Capital Final Est.** : $N/A
- **Total Operaciones**  : N/A
- **Win Rate Est.**      : N/A%
- **Max Drawdown**       : N/A%

### Últimos Logs del Motor (Contexto Estratégico)
```text
====
🚀 GOD MODE BACKTEST v2.0 — PRODUCTION PARITY ENGINE
======================================================================
   📋 Config Source: config.py (Single Source of Truth)
   💰 Capital:  $13.00
   ⚡ Leverage: 10x
   💸 Fee:      0.0375% per side (0.0750% round-trip)
   🎯 Symbols:  ['BTC/USDT']
   📅 Days:     1
   🔄 Horizon:  SCALPING
   🛡️ Breakeven: 80% of TP distance
======================================================================

📡 Downloading 1d of 1m data for BTC/USDT...
📡 Descargando 1 días de datos para BTC/USDT...
✅ Descargados 1440 velas (1.0 días)
  ✅ 1,440 bars loaded (1.0 days)

======================================================================
🚀 GOD MODE BACKTEST — BTC/USDT [SCALPING]
   Capital: $13.0 | Leverage: 10x | Fee: 0.0375%/side
======================================================================
  📦 Initializing ALL God Mode production strategies...
    ✅ ML SCALPING ready (min_train=300)
    ✅ ML SWING ready (min_train=300)
    ✅ Sniper ready
    ✅ Technical ready

  📊 Processing 1,440 bars (~1.0 days)...

    [ 10%] Bar     144/1,440 | Equity: $13.00 | Trades: 0 | Speed: 4143 bars/s | ETA: 0s
    [ 20%] Bar     288/1,440 | Equity: $13.00 | Trades: 0 | Speed: 158 bars/s | ETA: 7s
    [ 30%] Bar     432/1,440 | Equity: $13.00 | Trades: 0 | Speed: 3 bars/s | ETA: 315s
    [ 40%] Bar     576/1,440 | Equity: $13.00 | Trades: 0 | Speed: 2 bars/s | ETA: 423s
    [ 50%] Bar     720/1,440 | Equity: $13.00 | Trades: 0 | Speed: 2 bars/s | ETA: 430s
    [ 60%] Bar     864/1,440 | Equity: $13.00 | Trades: 0 | Speed: 1 bars/s | ETA: 386s
    🧠 ML SCALPING Initial Training OK (bars: 1000)
    🧠 ML SWING Initial Training OK (bars: 1000)
    [ 70%] Bar   1,008/1,440 | Equity: $13.00 | Trades: 0 | Speed: 1 bars/s | ETA: 316s
    [ 80%] Bar   1,152/1,440 | Equity: $13.00 | Trades: 0 | Speed: 1 bars/s | ETA: 222s
    [ 90%] Bar   1,296/1,440 | Equity: $13.00 | Trades: 0 | Speed: 1 bars/s | ETA: 117s
    [100%] Bar   1,440/1,440 | Equity: $13.00 | Trades: 0 | Speed: 1 bars/s | ETA: 0s
💾 Persistence: Saved 1 Brains.

  ❌ ZERO trades for BTC/USDT [SCALPING]
     This means the strategies generated NO entry signals.
     ML Training attempts: 8
     Analysis of failure points:
      - NO_STRATEGY_SIGNALS: 1 bars

======================================================================
🏆 CONSOLIDATED REPORT — GOD MODE v2.0
======================================================================
Symbol       Hz            PNL$     PNL%     WR%     DD%  Sharpe Trades
----------------------------------------------------------------------
BTC/USDT     SCALPING    +0.000   +0.00%    0.0%   0.00%    0.00      0
----------------------------------------------------------------------
❌ TOTAL                   +0.000                                       0
======================================================================

📁 Results saved to: god_mode_backtest_results.json

```

---
