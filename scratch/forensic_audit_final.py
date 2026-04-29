"""
AUDITORÍA FORENSE DEFINITIVA v2 - Trader Gemini God Mode Backtest
Reconcilia los datos del JSONL con lo que Telegram muestra al usuario.
Detecta la FUGA DE CAPITAL entre trades cerrados y balance del portfolio.
"""
import json
import re
from collections import Counter, defaultdict

def parse_backtick_value(line):
    """Extrae el valor entre backticks de una línea."""
    parts = line.split('`')
    if len(parts) >= 2:
        return parts[1]
    return None

def to_float(s):
    """Convierte string como '$22.35' o '-32.27%' o '+$0.0041' a float."""
    if not s:
        return None
    s = s.replace('$', '').replace('%', '').replace(',', '').replace('+', '').strip()
    try:
        return float(s)
    except:
        return None

def to_float_signed(s):
    """Convierte conservando signo negativo."""
    if not s:
        return None
    s = s.replace('$', '').replace('%', '').replace(',', '').strip()
    if s.startswith('+'):
        s = s[1:]
    try:
        return float(s)
    except:
        return None

def analyze():
    with open('dashboard/data/backtest_telemetry_spam.jsonl', 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    closed = []
    opened = []
    alerts = []
    
    for x in all_lines:
        try:
            data = json.loads(x)
            msg = data.get('message', '')
            if 'TRADE CERRADO' in msg:
                closed.append(msg)
            elif 'NUEVO TRADE INICIADO' in msg:
                opened.append(msg)
            elif 'ALERTA' in msg or 'CRITICAL' in msg or 'Kill Switch' in msg:
                alerts.append(msg)
        except:
            pass

    trades = []
    for m in closed:
        t = {'win': m.startswith('\U0001f7e2')}  # 🟢
        
        for line in m.split('\n'):
            if 'Par:' in line and '`' in line:
                t['pair'] = parse_backtick_value(line)
            if 'Dirección:' in line:
                t['side'] = 'LONG' if 'LONG' in line else 'SHORT'
            if 'Estrategia:' in line and 'Resumen' not in line:
                t['strategy'] = line.split('Estrategia:')[1].strip()
            if 'Razón de Cierre:' in line and '`' in line:
                t['reason'] = parse_backtick_value(line)
            if 'Duración:' in line and '`' in line:
                dur_raw = parse_backtick_value(line)
                if dur_raw:
                    t['duration_s'] = to_float(dur_raw.replace('s', ''))
            if 'Nocional:' in line and '`' in line:
                t['notional'] = to_float(parse_backtick_value(line))
            if 'PnL Bruto:' in line and '`' in line:
                t['gross_pnl'] = to_float_signed(parse_backtick_value(line))
            if 'Fees:' in line and '`' in line:
                raw = parse_backtick_value(line)
                if raw:
                    t['fee'] = abs(to_float_signed(raw) or 0)
            if '*PnL Neto:' in line and '`' in line:
                t['net_pnl'] = to_float_signed(parse_backtick_value(line))
            if 'Antes de trade:' in line and '`' in line:
                t['balance_before'] = to_float(parse_backtick_value(line))
            if 'Balance Total Ahora:' in line and '`' in line:
                t['balance_after'] = to_float(parse_backtick_value(line))
            if 'Crecimiento Acumulado:' in line and '`' in line:
                t['growth_pct'] = to_float_signed(parse_backtick_value(line))
            if 'WR Global:' in line and '`' in line:
                t['wr_global'] = to_float(parse_backtick_value(line))
            if 'Movimiento:' in line and '`' in line:
                t['movement_pct'] = to_float_signed(parse_backtick_value(line))
                
        trades.append(t)

    total = len(trades)
    wins = sum(1 for t in trades if t.get('win'))
    losses = total - wins
    
    net_pnls = [t['net_pnl'] for t in trades if t.get('net_pnl') is not None]
    gross_pnls = [t['gross_pnl'] for t in trades if t.get('gross_pnl') is not None]
    fees_list = [t['fee'] for t in trades if t.get('fee') is not None]
    
    total_net = sum(net_pnls)
    total_gross = sum(gross_pnls)
    total_fees = sum(fees_list)
    
    balance_befores = [t['balance_before'] for t in trades if t.get('balance_before') is not None]
    balance_afters = [t['balance_after'] for t in trades if t.get('balance_after') is not None]
    growths = [t['growth_pct'] for t in trades if t.get('growth_pct') is not None]
    
    durations = [t['duration_s'] for t in trades if t.get('duration_s') is not None]
    avg_dur_min = (sum(durations) / len(durations) / 60.0) if durations else 0
    
    by_pair = defaultdict(lambda: {'wins': 0, 'losses': 0, 'net': 0.0, 'count': 0, 'gross': 0.0, 'fees': 0.0})
    for t in trades:
        p = t.get('pair', 'UNKNOWN')
        by_pair[p]['count'] += 1
        by_pair[p]['net'] += t.get('net_pnl', 0) or 0
        by_pair[p]['gross'] += t.get('gross_pnl', 0) or 0
        by_pair[p]['fees'] += t.get('fee', 0) or 0
        if t.get('win'): by_pair[p]['wins'] += 1
        else: by_pair[p]['losses'] += 1
    
    by_reason = defaultdict(lambda: {'wins': 0, 'losses': 0, 'net': 0.0, 'count': 0})
    for t in trades:
        r = t.get('reason', 'UNKNOWN')
        by_reason[r]['count'] += 1
        by_reason[r]['net'] += t.get('net_pnl', 0) or 0
        if t.get('win'): by_reason[r]['wins'] += 1
        else: by_reason[r]['losses'] += 1
    
    by_side = defaultdict(lambda: {'wins': 0, 'losses': 0, 'net': 0.0, 'count': 0})
    for t in trades:
        s = t.get('side', 'UNKNOWN')
        by_side[s]['count'] += 1
        by_side[s]['net'] += t.get('net_pnl', 0) or 0
        if t.get('win'): by_side[s]['wins'] += 1
        else: by_side[s]['losses'] += 1
    
    fee_death = [t for t in trades if (t.get('gross_pnl') or 0) > 0 and (t.get('net_pnl') or 0) <= 0]
    
    sorted_by_net = sorted([t for t in trades if t.get('net_pnl') is not None], key=lambda x: x['net_pnl'])
    
    # ============================================================
    # REPORTE
    # ============================================================
    print("=" * 80)
    print("  📊 AUDITORÍA FORENSE DEFINITIVA: TRADER GEMINI (GOD MODE BACKTEST)")
    print("=" * 80)
    
    print("\n## 📈 MÉTRICAS GLOBALES (SEGÚN PORTFOLIO REAL - LO QUE TELEGRAM MUESTRA)")
    print(f"  Total Trades Cerrados:          {total}")
    print(f"  Trades Abiertos al final:       {len(opened) - len(closed)}")
    print(f"  Win Rate (🟢/🔴):               {wins}/{losses} = {(wins/total*100):.2f}%")
    
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  💰 BALANCE REAL DEL PORTFOLIO (COMO EN TELEGRAM)")
    print(f"  ═══════════════════════════════════════════════")
    if balance_befores:
        print(f"  'Antes de trade' PRIMERO:       ${balance_befores[0]:.2f}")
        print(f"  'Antes de trade' ÚLTIMO:        ${balance_befores[-1]:.2f}")
        print(f"  'Antes de trade' MAX:           ${max(balance_befores):.2f}")
    if balance_afters:
        print(f"  'Balance Total Ahora' PRIMERO:  ${balance_afters[0]:.2f}")
        print(f"  'Balance Total Ahora' ÚLTIMO:   ${balance_afters[-1]:.2f}")
        print(f"  'Balance Total Ahora' MAX:      ${max(balance_afters):.2f}")
        print(f"  'Balance Total Ahora' MIN:      ${min(balance_afters):.2f}")
        
    if growths:
        print(f"\n  📉 CRECIMIENTO ACUMULADO (lo que Telegram dice):")
        print(f"  Growth PRIMERO:                 {growths[0]:+.2f}%")
        print(f"  Growth ÚLTIMO (FINAL):          {growths[-1]:+.2f}%")
        print(f"  Growth MÁXIMO:                  {max(growths):+.2f}%")
        print(f"  Growth MÍNIMO:                  {min(growths):+.2f}%")
    
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  💵 PnL ACUMULADO (SUMA DE TRADES INDIVIDUALES)")
    print(f"  ═══════════════════════════════════════════════")
    print(f"  PnL Bruto acumulado:            ${total_gross:.4f}")
    print(f"  Comisiones totales:             -${total_fees:.4f}")
    print(f"  PnL Neto acumulado:             ${total_net:.4f}")
    print(f"  Ratio Fees/|GrossPnL|:          {(total_fees/abs(total_gross)*100) if total_gross else 0:.2f}%")
    
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  🚨🚨🚨 DISCREPANCIA CRÍTICA (LA FUGA) 🚨🚨🚨")
    print(f"  ═══════════════════════════════════════════════")
    if balance_afters and balance_befores:
        portfolio_change = balance_afters[-1] - balance_afters[0]
        unrealized_leak = balance_befores[-1] - balance_afters[-1]
        
        print(f"  Suma Net PnL de {len(net_pnls)} trades:   ${total_net:.4f}")
        print(f"  Cambio real Balance (after):     ${portfolio_change:.4f}")
        print(f"  FUGA = Net PnL - Cambio Balance: ${total_net - portfolio_change:.4f}")
        print(f"")
        print(f"  'Antes de trade' ÚLTIMO:         ${balance_befores[-1]:.2f}")
        print(f"  'Balance Total' ÚLTIMO:          ${balance_afters[-1]:.2f}")
        print(f"  UNREALIZED PnL (LEAK):           ${unrealized_leak:.2f}")
        print(f"")
        print(f"  📐 EXPLICACIÓN MATEMÁTICA:")
        print(f"  balance_before = initial_capital + realized_pnl_acumulado")
        print(f"  balance_after  = equity = cash + unrealized_pnl_todas_posiciones")
        print(f"  La DIFERENCIA = posiciones ABIERTAS con pérdidas no realizadas")
        print(f"  Cuando un trade se cierra, su PnL pasa a realized.")
        print(f"  Pero las posiciones AÚN ABIERTAS arrastran el equity hacia abajo.")
        print(f"  ➡️ Hay ${unrealized_leak:.2f} en posiciones abiertas PERDIENDO.")
    
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  ⏱️ DURACIÓN")
    print(f"  ═══════════════════════════════════════════════")
    print(f"  Duración promedio:              {avg_dur_min:.1f} minutos")
    if durations:
        print(f"  Duración mínima:                {min(durations)/60:.1f} min")
        print(f"  Duración máxima:                {max(durations)/60:.1f} min")
    
    print(f"\n## 🚪 DISTRIBUCIÓN POR RAZÓN DE CIERRE")
    for r, d in sorted(by_reason.items(), key=lambda x: x[1]['count'], reverse=True):
        wr = (d['wins']/d['count']*100) if d['count'] else 0
        print(f"  {r}: {d['count']} trades | WR: {wr:.1f}% | Net: ${d['net']:.4f}")
    
    print(f"\n## 💱 DISTRIBUCIÓN POR PAR")
    for p, d in sorted(by_pair.items(), key=lambda x: x[1]['net']):
        wr = (d['wins']/d['count']*100) if d['count'] else 0
        print(f"  {p}: {d['count']} trades | WR: {wr:.1f}% | Net: ${d['net']:.4f} | Fees: ${d['fees']:.4f}")
    
    print(f"\n## 📈📉 DISTRIBUCIÓN POR DIRECCIÓN (LONG vs SHORT)")
    for s, d in by_side.items():
        wr = (d['wins']/d['count']*100) if d['count'] else 0
        print(f"  {s}: {d['count']} trades | WR: {wr:.1f}% | Net: ${d['net']:.4f}")
    
    print(f"\n## 💀 FEE DEATH SPIRAL")
    print(f"  Trades con Bruto > 0 pero Neto <= 0: {len(fee_death)} de {total} ({len(fee_death)/total*100:.1f}%)")
    
    print(f"\n## 🏆 TOP 5 MEJORES TRADES")
    for t in sorted_by_net[-5:]:
        print(f"  {t.get('pair','?')} {t.get('side','?')} | Net: ${t.get('net_pnl',0):.4f} | Dur: {(t.get('duration_s',0) or 0)/60:.0f}m | {t.get('reason','?')}")
    
    print(f"\n## 💀 TOP 5 PEORES TRADES")
    for t in sorted_by_net[:5]:
        print(f"  {t.get('pair','?')} {t.get('side','?')} | Net: ${t.get('net_pnl',0):.4f} | Dur: {(t.get('duration_s',0) or 0)/60:.0f}m | {t.get('reason','?')}")
    
    # WR from engine
    wr_globals = [t['wr_global'] for t in trades if t.get('wr_global') is not None]
    if wr_globals:
        print(f"\n## 📊 WR GLOBAL (PORTFOLIO vs NUESTRO CÁLCULO)")
        print(f"  WR reportado engine (último):   {wr_globals[-1]:.1f}%")
        print(f"  WR calculado (emoji 🟢/🔴):      {(wins/total*100):.1f}%")
    
    # Equity curve
    if balance_afters:
        print(f"\n## 📉 EQUITY CURVE (cada 50 trades)")
        for i in range(0, len(balance_afters), 50):
            g = growths[i] if i < len(growths) else 0
            bb = balance_befores[i] if i < len(balance_befores) else 0
            ba = balance_afters[i]
            unreal = bb - ba if bb and ba else 0
            print(f"  #{i+1:4d}: Equity ${ba:.2f} | Realized(before) ${bb:.2f} | Unrealized ${-unreal:+.2f} | Growth: {g:+.2f}%")
        i = len(balance_afters) - 1
        g = growths[i] if i < len(growths) else 0
        bb = balance_befores[i] if i < len(balance_befores) else 0
        ba = balance_afters[i]
        unreal = bb - ba if bb and ba else 0
        print(f"  #{i+1:4d}: Equity ${ba:.2f} | Realized(before) ${bb:.2f} | Unrealized ${-unreal:+.2f} | Growth: {g:+.2f}%")

    print(f"\n## 🚨 ALERTAS CRÍTICAS EMITIDAS: {len(alerts)}")

if __name__ == "__main__":
    analyze()
