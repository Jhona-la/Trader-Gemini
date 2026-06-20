"""
🔬 TRADE LIFECYCLE AUTOPSY — Full history of every trade
Traces: WHY opened → WHAT happened → WHY closed → WHO closed → WHO SHOULD HAVE closed

This is a forensic reconstruction of every trade's complete life story, parsing
the JSON details blob from bt_trades.csv to extract MFE, MAE, ML confidence, and more.
"""
import pandas as pd
import json
import os
import sys
import argparse
from collections import defaultdict

# Ensure project root is in path for config imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

def parse_args():
    parser = argparse.ArgumentParser(description="Trade Lifecycle Autopsy")
    parser.add_argument("--file", type=str, default="dashboard/data/backtest_temp/bt_trades.csv", help="Path to bt_trades.csv")
    return parser.parse_args()

def safe_json_load(s):
    """
    Parse the 'details' column from bt_trades.csv.
    Handles TWO formats:
      1. JSON (current production):  {"margin": 0.65, "setup": "MOMENTUM", "is_close": true, ...}
      2. Legacy string (old backtests): "Exchange: BINANCE_BACKTEST | Margin: 0.65 | Setup: None"
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    s = s.strip()
    # Attempt 1: Standard JSON
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        from utils.error_handler import SystemIntegrityError
        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
    # Attempt 2: Legacy pipe-delimited string → dict
    # Format: "Key1: Value1 | Key2: Value2 | ..."
    try:
        result = {}
        for pair in s.split('|'):
            pair = pair.strip()
            if ':' in pair:
                key, val = pair.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                val = val.strip()
                # Coerce types
                if val.lower() == 'none':
                    result[key] = None
                elif val.lower() in ('true', 'false'):
                    result[key] = val.lower() == 'true'
                else:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
        return result
    except Exception:
        return {}

def main():
    args = parse_args()
    
    if not os.path.exists(args.file):
        print(f"❌ Error: Could not find {args.file}")
        return

    print("=" * 80)
    print("  🔬 TRADE LIFECYCLE AUTOPSY — Complete History of Every Trade")
    print(f"     Source: {args.file}")
    print("=" * 80)

    try:
        df = pd.read_csv(args.file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Parse JSON details
    df['parsed_details'] = df['details'].apply(safe_json_load)
    
    # Extract columns from parsed_details
    df['is_close'] = df['parsed_details'].apply(lambda x: x['is_close'])
    df['pnl'] = df['parsed_details'].apply(lambda x: x['pnl'])
    df['fees'] = df['parsed_details'].apply(lambda x: x['fees'])
    df['mfe_pct'] = df['parsed_details'].apply(lambda x: x['mfe_pct'])
    df['mae_pct'] = df['parsed_details'].apply(lambda x: x['mae_pct'])
    df['duration_s'] = df['parsed_details'].apply(lambda x: x['duration_s'])
    df['exit_reason'] = df['parsed_details'].apply(lambda x: x['exit_reason'])
    df['ml_confidence'] = df['parsed_details'].apply(lambda x: x['ml_confidence'])
    
    # Separate opens and closes
    opens = df[df['type'] == 'FILL_OPEN'].copy()
    closes = df[df['type'] == 'FILL_CLOSE'].copy()
    
    # PHASE 1: EXIT STRATEGY CENSUS
    print("\n" + "=" * 80)
    print("  PHASE 1: EXIT STRATEGY CENSUS — Who kills our trades?")
    print("=" * 80)
    
    strat_stats = closes.groupby('exit_reason').agg(
        cnt=('pnl', 'count'),
        wins=('pnl', lambda x: (x > 0).sum()),
        losses=('pnl', lambda x: (x < 0).sum()),
        total_pnl=('pnl', 'sum'),
        avg_pnl=('pnl', 'mean'),
        total_fees=('fees', 'sum')
    ).reset_index()
    
    strat_stats = strat_stats.sort_values('total_pnl', ascending=True)
    
    print(f"\n  {'Exit Strategy':<45} {'#':<5} {'W':<4} {'L':<4} {'WR%':<7} {'Σ PnL':<12} {'Avg PnL':<10} {'Fees':<10}")
    print(f"  {'-'*100}")
    
    for _, r in strat_stats.iterrows():
        strat = str(r['exit_reason'])
        cnt, w, l = int(r['cnt']), int(r['wins']), int(r['losses'])
        wr = w / cnt * 100 if cnt > 0 else 0
        marker = "🔴" if r['total_pnl'] < 0 else "🟢"
        print(f"  {marker} {strat:<43} {cnt:<5} {w:<4} {l:<4} {wr:<6.1f}% {r['total_pnl']:<12.6f} {r['avg_pnl']:<10.6f} {r['total_fees']:<10.6f}")

    # PHASE 2: ENTRY ↔ EXIT PAIRING
    print("\n" + "=" * 80)
    print("  PHASE 2: ENTRY ↔ EXIT PAIRING — Reconstructing Trade Lifecycles")
    print("=" * 80)

    # We match by trade_id if available, otherwise by symbol + side + time heuristic
    trade_pairs = []
    
    # First, try to merge on trade_id if they exist and are valid UUIDs/not None
    # For robust matching, we iterate through closes and find the most recent matching open
    open_dict = defaultdict(list)
    for _, row in opens.iterrows():
        key = row['symbol']
        open_dict[key].append(row.to_dict())
        
    paired_count = 0
    for _, close_row in closes.iterrows():
        sym = close_row['symbol']
        close_dir = str(close_row['direction'])
        
        # A close order direction is opposite to the position
        pos_dir = 'BUY' if close_dir == 'SELL' else 'SELL'
        
        matching_open = None
        # Try to find the closest open in time before the close
        candidates = [o for o in open_dict[sym] if o['direction'] == pos_dir and o['datetime'] <= close_row['datetime']]
        if candidates:
            # Get the most recent one
            matching_open = sorted(candidates, key=lambda x: x['datetime'], reverse=True)[0]
            # Remove it so we don't double count
            open_dict[sym].remove(matching_open)
            
        trade_pairs.append({
            'entry': matching_open,
            'exit': close_row.to_dict(),
            'symbol': sym,
            'pnl': close_row['pnl'],
            'exit_strategy': close_row['exit_reason'],
            'duration_s': close_row['duration_s'],
            'mfe_pct': close_row['mfe_pct'],
            'mae_pct': close_row['mae_pct'],
            'ml_confidence': close_row['ml_confidence'] if matching_open is None else matching_open['ml_confidence']
        })
        if matching_open:
            paired_count += 1
            
    print(f"\n  Total trade exits found: {len(trade_pairs)}")
    print(f"  Successfully paired with entries: {paired_count}")
    print(f"  Wins: {sum(1 for tp in trade_pairs if tp['pnl'] > 0)}")
    print(f"  Losses: {sum(1 for tp in trade_pairs if tp['pnl'] < 0)}")

    # PHASE 3: AUTOPSY OF LOSERS
    print("\n" + "=" * 80)
    print("  PHASE 3: 🔴 AUTOPSY OF LOSING TRADES — Complete Story")
    print("=" * 80)

    losers = [tp for tp in trade_pairs if tp['pnl'] < 0]
    losers.sort(key=lambda x: x['pnl'])  # Worst first
    
    print(f"\n  Analyzing {len(losers)} losing trades...\n")
    losers_by_exit = defaultdict(list)
    for l in losers:
        losers_by_exit[l['exit_strategy']].append(l)

    print(f"  {'Exit Strategy':<45} {'Losses':<7} {'Σ PnL Lost':<12} {'Avg Loss':<10}")
    print(f"  {'-'*80}")
    for strat, trades in sorted(losers_by_exit.items(), key=lambda x: sum(t['pnl'] for t in x[1])):
        total = sum(t['pnl'] for t in trades)
        avg = total / len(trades)
        print(f"  🔴 {strat:<43} {len(trades):<7} {total:<12.6f} {avg:<10.6f}")

    # PHASE 4: INDIVIDUAL TRADE STORIES
    print("\n" + "=" * 80)
    print("  PHASE 4: 📖 INDIVIDUAL STORIES — Top 15 Worst Losers")
    print("=" * 80)

    for i, loser in enumerate(losers[:15]):
        entry = loser['entry']
        exit_t = loser['exit']
        
        print(f"\n  {'─'*70}")
        print(f"  📖 STORY #{i+1} | {loser['symbol']} | PnL: ${loser['pnl']:.6f}")
        print(f"  {'─'*70}")
        
        if entry:
            print(f"  📥 ENTRY:")
            print(f"     Time:       {entry['datetime']}")
            print(f"     Strategy:   {entry['strategy_id']}")
            print(f"     Side:       {entry['direction']}")
            print(f"     Price:      ${entry['price']}")
            print(f"     ML Conf:    {loser['ml_confidence']}")
        
        mfe_pct = loser['mfe_pct']
        mae_pct = loser['mae_pct']
        dur = loser['duration_s']
        
        print(f"\n  ⏳ DURING THE TRADE:")
        if dur:
            print(f"     Duration:   {dur:.0f}s ({dur/60:.1f} min)")
        print(f"     MFE:        {mfe_pct*100:.4f}% (best it got in our favor)")
        print(f"     MAE:        {mae_pct*100:.4f}% (worst against us)")
        
        if mfe_pct > 0.001:
            print(f"     ⚠️ INSIGHT: Price moved {mfe_pct*100:.4f}% IN OUR FAVOR before reversing!")
        
        print(f"\n  📤 EXIT:")
        print(f"     Time:       {exit_t['datetime']}")
        print(f"     Reason:     {loser['exit_strategy']}")
        print(f"     Price:      ${exit_t['price']}")
        print(f"     PnL:        ${loser['pnl']:.6f}")
        print(f"     Commission: ${exit_t['fees']:.6f}")

    # PHASE 5: MFE WASTE ANALYSIS
    print("\n" + "=" * 80)
    print("  PHASE 5: 💸 MFE WASTE — Profit Left on the Table")
    print("=" * 80)

    wasted_profit = 0
    trades_with_wasted_mfe = 0
    for tp in trade_pairs:
        mfe = tp['mfe_pct']
        pnl = tp['pnl']
        if mfe > 0.001 and pnl < 0:
            trades_with_wasted_mfe += 1
            # Estimate wasted notional (assume ~$6.50 per trade average)
            wasted_est = mfe * 6.50
            wasted_profit += wasted_est

    print(f"\n  Losing trades that WERE profitable at some point: {trades_with_wasted_mfe}")
    print(f"  Estimated total wasted profit (MFE not captured):  ${wasted_profit:.4f}")
    
    # PHASE 6: SYSTEMIC VERDICT (CSV-ONLY)
    print("\n" + "=" * 80)
    print("  PHASE 6: ⚖️ SYSTEMIC VERDICT (From CSV)")
    print("=" * 80)
    
    total_winners = sum(1 for tp in trade_pairs if tp['pnl'] > 0)
    total_losers = sum(1 for tp in trade_pairs if tp['pnl'] < 0)
    total_pnl = sum(tp['pnl'] for tp in trade_pairs)
    total_wr = total_winners / len(trade_pairs) * 100 if trade_pairs else 0
    
    print(f"\n  Total Trades:    {len(trade_pairs)}")
    print(f"  Win Rate:        {total_wr:.1f}%")
    print(f"  Total PnL:       ${total_pnl:.6f}")
    print(f"  Wasted MFE:      ${wasted_profit:.4f}")
    
    if losers_by_exit:
        worst_killer = min(losers_by_exit.items(), key=lambda x: sum(t['pnl'] for t in x[1]))
        worst_killer_pnl = sum(t['pnl'] for t in worst_killer[1])
        print(f"\n  🏴‍☠️ #1 CAPITAL KILLER: {worst_killer[0]}")
        print(f"     Responsible for: {len(worst_killer[1])} losses = ${worst_killer_pnl:.6f}")

    # ═══════════════════════════════════════════════════════════════
    # PHASES 7-9: SQLITE DEEP FORENSICS (CTOS Phase 4)
    # QUÉ: Cruza datos del CSV con la base de datos SQLite para
    #   obtener predicciones, votos de exit strategies, y chronicle.
    # POR QUÉ: El CSV tiene métricas básicas, pero SQLite tiene la
    #   historia completa de cada decisión tomada por el sistema.
    # PARA QUÉ: Diagnosticar exactamente qué predijo mal la IA,
    #   qué estrategia debió cerrar y no lo hizo, y cuál era el
    #   punto óptimo de cierre que nunca se tomó.
    # ═══════════════════════════════════════════════════════════════
    import sqlite3
    from config import Config
    
    db_path = os.path.join(getattr(Config, 'DATA_DIR', 'data'), 'trader_gemini.db')
    if not os.path.exists(db_path):
        print(f"\n  ⚠️ SQLite DB not found at {db_path}. Skipping Phases 7-9.")
    else:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # ═══ PHASE 7: PREDICTION ACCURACY AUDIT ═══
            print("\n" + "=" * 80)
            print("  PHASE 7: 🎯 PREDICTION ACCURACY AUDIT — ¿Qué predijo la IA vs realidad?")
            print("=" * 80)

            cursor.execute("""
                SELECT strategy_id, 
                       COUNT(*) as total,
                       SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                       AVG(predicted_magnitude_pct) as avg_pred_mag,
                       AVG(actual_magnitude_pct) as avg_actual_mag,
                       AVG(missed_profit_pct) as avg_missed,
                       AVG(confidence) as avg_conf
                FROM prediction_audit
                WHERE resolution_time IS NOT NULL
                GROUP BY strategy_id
                ORDER BY total DESC
            """)
            rows = cursor.fetchall()
            
            if rows:
                print(f"\n  {'Strategy':<30} {'#':<5} {'Correct':<8} {'Accuracy':<9} {'Pred%':<8} {'Real%':<8} {'Missed%':<9} {'Avg Conf':<8}")
                print(f"  {'-'*95}")
                for r in rows:
                    acc = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
                    marker = "🟢" if acc >= 50 else "🔴"
                    pred_mag = r['avg_pred_mag'] or 0
                    actual_mag = r['avg_actual_mag'] or 0
                    missed = r['avg_missed'] or 0
                    conf = r['avg_conf'] or 0
                    print(f"  {marker} {r['strategy_id']:<28} {r['total']:<5} {r['correct']:<8} {acc:<8.1f}% {pred_mag:<8.3f} {actual_mag:<8.3f} {missed:<9.3f} {conf:<8.3f}")
            else:
                print("\n  ℹ️ No prediction audit records found. Run trades to populate.")

            # Direction accuracy
            cursor.execute("""
                SELECT direction,
                       COUNT(*) as total,
                       SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                       AVG(actual_magnitude_pct) as avg_mag
                FROM prediction_audit
                WHERE resolution_time IS NOT NULL
                GROUP BY direction
            """)
            dir_rows = cursor.fetchall()
            if dir_rows:
                print(f"\n  📊 Accuracy por Dirección:")
                for r in dir_rows:
                    acc = (r['correct'] / r['total'] * 100) if r['total'] > 0 else 0
                    marker = "🟢" if acc >= 50 else "🔴"
                    print(f"     {marker} {r['direction']}: {acc:.1f}% ({r['correct']}/{r['total']}) | Avg Magnitude: {r['avg_mag'] or 0:.3f}%")

            # ═══ PHASE 8: EXIT STRATEGY vs STRATEGY ═══
            print("\n" + "=" * 80)
            print("  PHASE 8: 🔄 OPENER vs CLOSER — ¿Quién abrió vs quién cerró?")
            print("=" * 80)

            cursor.execute("""
                SELECT strategy_id as closer, action, reason,
                       COUNT(*) as cnt,
                       AVG(unrealized_pnl) as avg_pnl_at_exit,
                       AVG(price_at_decision) as avg_price
                FROM exit_strategy_log
                WHERE action = 'EXIT'
                GROUP BY strategy_id, action
                ORDER BY cnt DESC
            """)
            exit_rows = cursor.fetchall()
            
            if exit_rows:
                print(f"\n  {'Closer Strategy':<30} {'Action':<8} {'Count':<7} {'Avg PnL at Exit':<16} {'Reason Sample':<30}")
                print(f"  {'-'*95}")
                for r in exit_rows:
                    reason = (r['reason'] or '')[:30]
                    print(f"  {r['closer']:<30} {r['action']:<8} {r['cnt']:<7} ${r['avg_pnl_at_exit'] or 0:<14.6f} {reason:<30}")
            else:
                print("\n  ℹ️ No exit strategy log records found.")

            # Cross-reference: Who opened vs who closed in prediction_audit
            cursor.execute("""
                SELECT pa.strategy_id as opener,
                       COUNT(*) as total,
                       SUM(CASE WHEN pa.was_correct = 1 THEN 1 ELSE 0 END) as wins,
                       AVG(pa.actual_magnitude_pct) as avg_result,
                       AVG(pa.missed_profit_pct) as avg_missed
                FROM prediction_audit pa
                WHERE pa.resolution_time IS NOT NULL
                GROUP BY pa.strategy_id
                ORDER BY avg_result DESC
            """)
            opener_rows = cursor.fetchall()
            if opener_rows:
                print(f"\n  📊 Performance por Estrategia Abridora:")
                print(f"  {'Opener':<30} {'Trades':<8} {'WR%':<8} {'Avg Result%':<12} {'Avg Missed%':<12}")
                print(f"  {'-'*75}")
                for r in opener_rows:
                    wr = (r['wins'] / r['total'] * 100) if r['total'] > 0 else 0
                    marker = "🟢" if wr >= 50 else "🔴"
                    print(f"  {marker} {r['opener']:<28} {r['total']:<8} {wr:<7.1f}% {r['avg_result'] or 0:<12.3f} {r['avg_missed'] or 0:<12.3f}")

            # ═══ PHASE 9: TRADE CHRONICLE — OPTIMAL EXIT DETECTION ═══
            print("\n" + "=" * 80)
            print("  PHASE 9: 📜 CHRONICLE — Punto Óptimo de Cierre")
            print("=" * 80)

            # Check which columns exist in trade_chronicle
            cursor.execute("PRAGMA table_info(trade_chronicle)")
            tc_cols = {col[1] for col in cursor.fetchall()}
            has_direction = 'direction' in tc_cols
            has_oracle = 'oracle_prediction_magnitude' in tc_cols
            has_size = 'entry_size_usd' in tc_cols
            
            # Build adaptive query based on available columns
            select_fields = "trade_id, symbol, horizon"
            if has_direction: select_fields += ", direction"
            select_fields += ", MAX(mfe_so_far) as peak_mfe, MIN(mae_so_far) as worst_mae"
            select_fields += ", COUNT(*) as ticks_recorded, MAX(tick_number) as last_tick"
            select_fields += ", AVG(unrealized_pnl_pct) as avg_pnl"
            if has_oracle: select_fields += ", oracle_prediction_magnitude"
            if has_size: select_fields += ", entry_size_usd"
            
            cursor.execute(f"""
                SELECT {select_fields}
                FROM trade_chronicle
                GROUP BY trade_id
                HAVING ticks_recorded >= 3
                ORDER BY peak_mfe DESC
                LIMIT 20
            """)
            chronicle_rows = cursor.fetchall()
            
            if chronicle_rows:
                print(f"\n  Top 20 trades con más historia registrada:")
                print(f"  {'Trade ID':<38} {'Symbol':<12} {'Hz':<10} {'Ticks':<7} {'Peak MFE%':<10} {'Worst MAE%':<11} {'Avg PnL%':<9}")
                print(f"  {'-'*100}")
                for r in chronicle_rows:
                    tid = (r['trade_id'] or '')[:36]
                    sym = (r['symbol'] or '')[:10]
                    hz = (r['horizon'] or '')[:8]
                    peak_mfe = r['peak_mfe'] or 0
                    worst_mae = r['worst_mae'] or 0
                    avg_pnl = r['avg_pnl'] or 0
                    marker = "🟢" if avg_pnl > 0 else "🔴"
                    print(f"  {marker} {tid:<36} {sym:<12} {hz:<10} {r['ticks_recorded']:<7} {peak_mfe:<10.4f} {worst_mae:<11.4f} {avg_pnl:<9.4f}")
                    
                    # Missed opportunity detection
                    if peak_mfe > 0.1 and avg_pnl < 0:
                        print(f"     ⚠️ MISSED OPPORTUNITY: Peak MFE was +{peak_mfe:.4f}% but trade ended negative!")
            else:
                print("\n  ℹ️ No trade chronicle records found. Trades need to run to populate.")

            # Missed opportunities summary
            cursor.execute("""
                SELECT COUNT(*) as total_missed,
                       AVG(peak_mfe) as avg_peak_mfe,
                       SUM(peak_mfe) as total_missed_pct
                FROM (
                    SELECT trade_id,
                           MAX(mfe_so_far) as peak_mfe,
                           AVG(unrealized_pnl_pct) as avg_pnl
                    FROM trade_chronicle
                    GROUP BY trade_id
                    HAVING peak_mfe > 0.1 AND avg_pnl < 0
                )
            """)
            missed_row = cursor.fetchone()
            if missed_row and missed_row['total_missed']:
                print(f"\n  💸 MISSED OPPORTUNITIES SUMMARY:")
                print(f"     Trades that were profitable but closed in loss: {missed_row['total_missed']}")
                print(f"     Average peak MFE of missed trades: +{missed_row['avg_peak_mfe']:.4f}%")
                print(f"     Total MFE left on the table: +{missed_row['total_missed_pct']:.4f}%")

            conn.close()
            
        except Exception as e:
            print(f"\n  ❌ SQLite forensic analysis failed: {e}")
            import traceback
            traceback.print_exc()

    # ═══ PHASE 10: FINAL RECOMMENDATIONS ═══
    print("\n" + "=" * 80)
    print("  PHASE 10: 💡 RECOMENDACIONES SISTÉMICAS")
    print("=" * 80)
    
    print(f"\n  Total Trades Analyzed: {len(trade_pairs)}")
    print(f"  Win Rate: {total_wr:.1f}%")
    print(f"  Net PnL: ${total_pnl:.6f}")
    
    if total_wr < 55:
        print(f"\n  🔴 ALERTA: WR ({total_wr:.1f}%) por debajo del objetivo (55%+)")
        print(f"     → Revisar calidad de señales de entrada")
        print(f"     → Evaluar si los SL son demasiado ajustados para scalping")
    
    if wasted_profit > abs(total_pnl) * 0.3:
        print(f"\n  🔴 ALERTA: Ganancia desperdiciada (${wasted_profit:.4f}) > 30% del PnL total")
        print(f"     → Los exit strategies están cerrando demasiado temprano")
        print(f"     → Considerar trailing stop más permisivo")
    
    if losers_by_exit:
        worst = min(losers_by_exit.items(), key=lambda x: sum(t['pnl'] for t in x[1]))
        print(f"\n  🎯 ACCIÓN PRIORITARIA: Optimizar '{worst[0]}' (capital killer #1)")
        print(f"     → Responsable de {len(worst[1])} pérdidas = ${sum(t['pnl'] for t in worst[1]):.6f}")

    print("\n" + "=" * 80)
    print("  🔬 AUTOPSY COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

