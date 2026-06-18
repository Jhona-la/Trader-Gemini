"""
🔬 FORENSIC DEEP AUDIT v35 — Trade-Level Forensic Analyzer
═══════════════════════════════════════════════════════════
QUÉ: Ejecuta un backtest completo con monkey-patch para capturar
     CADA trade individual con metadatos forenses completos.
POR QUÉ: Necesitamos saber exactamente qué diferencia ganadores de perdedores.
PARA QUÉ: Identificar patrones explotables y eliminar debilidades.
"""
import sys, os, json, time, argparse, warnings, uuid
import numpy as np
import pandas as pd

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from datetime import datetime, timezone, timedelta
from config import Config

# ── Global trade capture buffer ──
ALL_TRADES = []

def run_forensic_backtest():
    """Run backtest with forensic instrumentation."""
    from core.portfolio import Portfolio
    from core.backtest_infra import fetch_multi_symbol_data
    from scripts.run_god_mode_backtest import run_global_backtest
    
    # ── MONKEY-PATCH: Capture every closed trade ──
    original_record = Portfolio._record_closed_trade
    
    def patched_record(self, *args, **kwargs):
        result = original_record(self, *args, **kwargs)
        closed = getattr(self, '_last_closed_trade_data', None)
        if closed:
            ALL_TRADES.append(closed.copy())
        return result
    
    Portfolio._record_closed_trade = patched_record
    
    # ── Download data ──
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    days = 7
    
    print(f"\n🎯 Symbols to backtest: {symbols}")
    all_data = fetch_multi_symbol_data(symbols, days=days, max_workers=4)
    
    if not all_data:
        print("❌ No data downloaded. Aborting.")
        return None
    
    valid_symbols = list(all_data.keys())
    print(f"✅ Valid symbols: {len(valid_symbols)}/{len(symbols)}")
    
    # ── Run backtest ──
    print("=" * 70)
    print("🔬 FORENSIC DEEP AUDIT v35 — Starting Full Capture Backtest")
    print("=" * 70)
    
    results = run_global_backtest(
        all_data=all_data,
        symbols=valid_symbols,
        days=days,
        verbose=False,  # Quiet mode for cleaner output
    )
    
    return results


def analyze_trades():
    """Deep forensic analysis of captured trades."""
    if not ALL_TRADES:
        print("❌ No trades captured. Check backtest execution.")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 FORENSIC DEEP AUDIT — {len(ALL_TRADES)} TRADES CAPTURED")
    print(f"{'='*70}")
    
    # ── Separate Winners and Losers ──
    winners = [t for t in ALL_TRADES if t.get('net_pnl', 0) > 0]
    losers = [t for t in ALL_TRADES if t.get('net_pnl', 0) <= 0]
    
    total_pnl = sum(t.get('net_pnl', 0) for t in ALL_TRADES)
    total_gross = sum(t.get('gross_pnl', 0) for t in ALL_TRADES)
    total_fees = sum(t.get('fees_paid', 0) for t in ALL_TRADES)
    
    print(f"\n🏆 WINNERS: {len(winners)} | 🔴 LOSERS: {len(losers)}")
    print(f"   Win Rate: {len(winners)/len(ALL_TRADES)*100:.1f}%")
    print(f"   Total Net PnL: ${total_pnl:.4f}")
    print(f"   Total Gross PnL: ${total_gross:.4f}")
    print(f"   Total Fees Paid: ${total_fees:.4f}")
    if total_gross != 0:
        fee_drag = (total_fees / abs(total_gross)) * 100
        print(f"   Fee Drag (fees/gross): {fee_drag:.1f}%")
    
    # ── Per-Trade Detail ──
    print(f"\n{'='*70}")
    print("📋 DETALLE DE CADA TRADE")
    print(f"{'='*70}")
    
    for i, t in enumerate(ALL_TRADES, 1):
        net = t.get('net_pnl', 0)
        icon = "🟢" if net > 0 else ("🔴" if net < 0 else "⚪")
        tid = t.get('trade_id', 'NO-ID')
        sym = t.get('symbol', '???')
        direction = t.get('direction', '???')
        horizon = t.get('horizon', '???')
        strategy = t.get('strategy_id', '???')
        exit_reason = t.get('exit_reason', '???')
        entry_p = t.get('entry_price', 0)
        exit_p = t.get('exit_price', 0)
        dur = t.get('duration_seconds', 0)
        gross = t.get('gross_pnl', 0)
        fees = t.get('fees_paid', 0)
        
        dur_str = f"{dur:.0f}s" if dur < 60 else (f"{dur/60:.1f}m" if dur < 3600 else f"{dur/3600:.1f}h")
        
        print(f"\n{icon} Trade #{i} | ID: {tid}")
        print(f"   {sym} | {direction} | {horizon} | Strategy: {strategy}")
        print(f"   Entry: ${entry_p:,.4f} → Exit: ${exit_p:,.4f}")
        if entry_p > 0:
            move_pct = ((exit_p - entry_p) / entry_p) * 100
            print(f"   Price Move: {move_pct:+.4f}%")
        print(f"   Duration: {dur_str}")
        print(f"   Gross PnL: ${gross:,.4f} | Fees: ${fees:,.4f} | Net PnL: ${net:,.4f}")
        print(f"   Exit Reason: {exit_reason}")
        
        # Extra metadata
        meta = t.get('metadata', {}) or {}
        if meta:
            mfe = meta.get('mfe_pct') or t.get('mfe_pct')
            mae = meta.get('mae_pct') or t.get('mae_pct')
            if mfe is not None or mae is not None:
                print(f"   MFE: +{mfe:.3f}% | MAE: -{mae:.3f}%")
    
    # ── Comparative Analysis ──
    print(f"\n{'='*70}")
    print("🔬 ANÁLISIS COMPARATIVO: ¿QUÉ DIFERENCIA A LOS GANADORES?")
    print(f"{'='*70}")
    
    def analyze_group(trades, label):
        if not trades:
            print(f"\n   {label}: Sin trades.")
            return {}
        
        durations = [t.get('duration_seconds', 0) for t in trades]
        net_pnls = [t.get('net_pnl', 0) for t in trades]
        gross_pnls = [t.get('gross_pnl', 0) for t in trades]
        fees_list = [t.get('fees_paid', 0) for t in trades]
        
        strat_count = {}
        for t in trades:
            s = t.get('strategy_id', 'Unknown')
            strat_count[s] = strat_count.get(s, 0) + 1
        
        exit_count = {}
        for t in trades:
            e = t.get('exit_reason', 'Unknown')
            exit_count[e] = exit_count.get(e, 0) + 1
        
        hz_count = {}
        for t in trades:
            h = t.get('horizon', 'Unknown')
            hz_count[h] = hz_count.get(h, 0) + 1
        
        dir_count = {}
        for t in trades:
            d = t.get('direction', 'Unknown')
            dir_count[d] = dir_count.get(d, 0) + 1
        
        sym_count = {}
        for t in trades:
            s = t.get('symbol', 'Unknown')
            sym_count[s] = sym_count.get(s, 0) + 1
        
        stats = {
            'count': len(trades),
            'avg_duration_s': np.mean(durations) if durations else 0,
            'median_duration_s': np.median(durations) if durations else 0,
            'min_duration_s': min(durations) if durations else 0,
            'max_duration_s': max(durations) if durations else 0,
            'avg_net_pnl': np.mean(net_pnls) if net_pnls else 0,
            'avg_gross_pnl': np.mean(gross_pnls) if gross_pnls else 0,
            'avg_fees': np.mean(fees_list) if fees_list else 0,
            'total_net_pnl': sum(net_pnls),
            'strategies': strat_count,
            'exit_reasons': exit_count,
            'horizons': hz_count,
            'directions': dir_count,
            'symbols': sym_count,
        }
        
        print(f"\n{'─'*50}")
        print(f"   {label} ({len(trades)} trades)")
        print(f"{'─'*50}")
        print(f"   ⏱️ Duración Promedio:  {stats['avg_duration_s']:.0f}s ({stats['avg_duration_s']/60:.1f}m)")
        print(f"   ⏱️ Duración Mediana:   {stats['median_duration_s']:.0f}s ({stats['median_duration_s']/60:.1f}m)")
        print(f"   ⏱️ Duración Min/Max:   {stats['min_duration_s']:.0f}s / {stats['max_duration_s']:.0f}s")
        print(f"   💰 PnL Neto Promedio:  ${stats['avg_net_pnl']:.4f}")
        print(f"   💰 PnL Bruto Promedio: ${stats['avg_gross_pnl']:.4f}")
        print(f"   💸 Fees Promedio:      ${stats['avg_fees']:.4f}")
        print(f"   📊 Total Neto:         ${stats['total_net_pnl']:.4f}")
        
        print(f"\n   📊 Por Estrategia:")
        for s, c in sorted(strat_count.items(), key=lambda x: -x[1]):
            pct = c/len(trades)*100
            print(f"      {s}: {c} ({pct:.0f}%)")
        
        print(f"\n   📊 Por Razón de Cierre:")
        for e, c in sorted(exit_count.items(), key=lambda x: -x[1]):
            pct = c/len(trades)*100
            print(f"      {e}: {c} ({pct:.0f}%)")
        
        print(f"\n   📊 Por Horizonte:")
        for h, c in sorted(hz_count.items(), key=lambda x: -x[1]):
            pct = c/len(trades)*100
            print(f"      {h}: {c} ({pct:.0f}%)")
        
        print(f"\n   📊 Por Dirección:")
        for d, c in sorted(dir_count.items(), key=lambda x: -x[1]):
            pct = c/len(trades)*100
            print(f"      {d}: {c} ({pct:.0f}%)")
        
        print(f"\n   📊 Por Moneda:")
        for s, c in sorted(sym_count.items(), key=lambda x: -x[1]):
            pct = c/len(trades)*100
            print(f"      {s}: {c} ({pct:.0f}%)")
        
        return stats
    
    w_stats = analyze_group(winners, "🟢 GANADORES")
    l_stats = analyze_group(losers, "🔴 PERDEDORES")
    
    # ── Key Differentiators ──
    print(f"\n{'='*70}")
    print("⚡ FACTORES DIFERENCIADORES CLAVE")
    print(f"{'='*70}")
    
    if w_stats and l_stats:
        w_dur = w_stats.get('avg_duration_s', 0)
        l_dur = l_stats.get('avg_duration_s', 0)
        dur_diff = w_dur - l_dur
        print(f"\n   ⏱️ DURACIÓN:")
        print(f"      Ganadores: {w_dur:.0f}s | Perdedores: {l_dur:.0f}s")
        if dur_diff > 0:
            print(f"      → Ganadores duran {dur_diff:.0f}s MÁS. Paciencia = Profit.")
        else:
            print(f"      → Ganadores duran {abs(dur_diff):.0f}s MENOS. Velocidad = Profit.")
        
        print(f"\n   💸 IMPACTO DE FEES:")
        print(f"      Ganadores: Fee ${w_stats['avg_fees']:.4f} | Gross ${w_stats['avg_gross_pnl']:.4f} → Net ${w_stats['avg_net_pnl']:.4f}")
        print(f"      Perdedores: Fee ${l_stats['avg_fees']:.4f} | Gross ${l_stats['avg_gross_pnl']:.4f} → Net ${l_stats['avg_net_pnl']:.4f}")
        
        # Fee-killed trades
        fee_killed = [t for t in losers if t.get('gross_pnl', 0) > 0 and t.get('net_pnl', 0) <= 0]
        if fee_killed:
            print(f"\n   🧟 TRADES MATADOS POR FEES: {len(fee_killed)}")
            for t in fee_killed:
                print(f"      {t.get('symbol','?')} {t.get('direction','?')} | Gross: ${t.get('gross_pnl',0):.4f} - Fee: ${t.get('fees_paid',0):.4f} = Net: ${t.get('net_pnl',0):.4f}")
        
        # Strategy WR per group
        print(f"\n   🧠 WIN RATE POR ESTRATEGIA:")
        all_strats = set(list(w_stats.get('strategies', {}).keys()) + list(l_stats.get('strategies', {}).keys()))
        for s in sorted(all_strats):
            w = w_stats.get('strategies', {}).get(s, 0)
            l = l_stats.get('strategies', {}).get(s, 0)
            total = w + l
            wr = w/total*100 if total > 0 else 0
            icon = "✅" if wr > 60 else ("⚠️" if wr > 40 else "❌")
            print(f"      {icon} {s}: {w}W/{l}L (WR: {wr:.0f}%)")
        
        # Exit reason analysis
        print(f"\n   🚪 WIN RATE POR RAZÓN DE CIERRE:")
        all_exits = set(list(w_stats.get('exit_reasons', {}).keys()) + list(l_stats.get('exit_reasons', {}).keys()))
        for e in sorted(all_exits):
            w = w_stats.get('exit_reasons', {}).get(e, 0)
            l = l_stats.get('exit_reasons', {}).get(e, 0)
            total = w + l
            wr = w/total*100 if total > 0 else 0
            icon = "✅" if wr > 60 else ("⚠️" if wr > 40 else "❌")
            print(f"      {icon} {e}: {w}W/{l}L (WR: {wr:.0f}%)")
    
    # ── Trade ID Audit ──
    print(f"\n{'='*70}")
    print("🏷️ AUDITORÍA DE TRADE IDs")
    print(f"{'='*70}")
    
    trades_with_id = [t for t in ALL_TRADES if t.get('trade_id') and t.get('trade_id') not in ('UNKNOWN', 'NO-ID', None, '')]
    trades_without_id = [t for t in ALL_TRADES if not t.get('trade_id') or t.get('trade_id') in ('UNKNOWN', 'NO-ID', None, '')]
    
    print(f"   ✅ Trades CON trade_id: {len(trades_with_id)}/{len(ALL_TRADES)}")
    print(f"   ❌ Trades SIN trade_id: {len(trades_without_id)}/{len(ALL_TRADES)}")
    
    if trades_without_id:
        print(f"\n   Trades sin ID:")
        for t in trades_without_id:
            print(f"      {t.get('symbol','?')} {t.get('direction','?')} {t.get('horizon','?')} | Strategy: {t.get('strategy_id','?')} | Exit: {t.get('exit_reason','?')}")
    
    # ── Save to JSON ──
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'total_trades': len(ALL_TRADES),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners)/len(ALL_TRADES)*100 if ALL_TRADES else 0,
        'total_net_pnl': total_pnl,
        'total_gross_pnl': total_gross,
        'total_fees': total_fees,
        'trades_with_id': len(trades_with_id),
        'trades_without_id': len(trades_without_id),
        'trades': []
    }
    
    for t in ALL_TRADES:
        clean = {}
        for k, v in t.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                clean[k] = v
            elif isinstance(v, datetime):
                clean[k] = v.isoformat()
            elif isinstance(v, pd.Timestamp):
                clean[k] = v.isoformat()
            else:
                clean[k] = str(v)
        output['trades'].append(clean)
    
    out_path = os.path.join(_project_root, 'forensic_deep_audit.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n💾 Resultados guardados en {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    results = run_forensic_backtest()
    analyze_trades()
