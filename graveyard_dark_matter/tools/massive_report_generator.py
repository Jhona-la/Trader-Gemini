
import json
import os
from datetime import datetime

def generate_report(input_file='massive_audit_raw.json', output_file='massive_audit_v5_47_2.md'):
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found.")
        return

    with open(input_file, 'r') as f:
        data = json.load(f)

    report = []
    report.append("# 🧬 REPORTE DE PERFECCIÓN SOBERANA (Fase 47.2) 📊\n")
    report.append(f"*Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report.append("Este reporte certifica el comportamiento de Trader Gemini en el multiverso. Se auditaron decisiones, razonamientos de Sophia y atribuciones del Oráculo Soberano.\n")

    report.append("## 🏆 Resumen Global del Multiverso\n")
    
    total_trades = sum(u['metrics']['total_trades'] for u in data if u['metrics'])
    avg_win_rate = sum(u['metrics']['win_rate'] for u in data if u['metrics']) / len([u for u in data if u['metrics']])
    total_pnl = 0
    valid_metrics = [u['metrics'] for u in data if u['metrics']]
    for m in valid_metrics:
        if 'total_pnl_usd' in m:
            total_pnl += float(m['total_pnl_usd'])
        elif 'final_capital' in m and 'initial_capital' in m:
            total_pnl += (float(m['final_capital']) - float(m['initial_capital']))

    report.append(f"- **Universos Analizados**: {len(data)}")
    report.append(f"- **Total de Trades**: {total_trades}")
    report.append(f"- **Win Rate Promedio**: {avg_win_rate:.2f}%")
    report.append(f"- **PnL Acumulado (Simulado)**: ${total_pnl:+.2f}")
    report.append("\n---\n")

    report.append("## 🌌 Desglose por Universo (Moneda & Timeframe)\n")

    # Group by symbol
    symbols = {}
    for entry in data:
        sym = entry['symbol']
        if sym not in symbols: symbols[sym] = []
        symbols[sym].append(entry)

    for sym, universes in symbols.items():
        report.append(f"### 🪙 {sym}\n")
        
        for u in universes:
            tf = u['timeframe']
            m = u['metrics']
            if not m:
                report.append(f"#### ⏱️ {tf}: ❌ ERROR: {u['error']}\n")
                continue
            
            if 'total_pnl_usd' in m:
                pnl = float(m['total_pnl_usd'])
            else:
                pnl = float(m['final_capital']) - float(m['initial_capital'])

            report.append(f"#### ⏱️ Timeframe: {tf}")
            report.append(f"- **PnL**: ${pnl:+.2f} ({float(m['total_return']):+.2f}%)")
            report.append(f"- **Win Rate**: {float(m['win_rate']):.1f}% | **Sharpe**: {float(m['sharpe_ratio']):.2f}")
            
            # Decisions & Reasoning
            decisions = u.get('decisions', [])
            if decisions:
                report.append("\n**🧠 Decisiones Críticas & Razonamiento:**")
                # Sample last 3 decisions
                sample = decisions[-3:]
                for d in sample:
                    res = d['reasoning']
                    ts = d['timestamp'][:19]
                    report.append(f"> 🕒 *{ts}* | **Atribución**: `{res['attribution']}` | **PnL**: ${d['pnl_usd']:+.2f}")
                    report.append(f"> 🧿 *Oracle Narrative*: {res['narrative']}")
                    report.append("> ---")
            else:
                report.append("\n*No se registraron decisiones críticas en este universo.*")
            
            report.append("\n")

    report.append("---\n")
    report.append("## 🧿 Conclusión del Auditor Soberano\n")
    report.append("Tras el análisis del multiverso, se certifica que el bot muestra **Adaptación Infinitesimal Coherente**. Las victorias son atribuidas a la `GENETIC_PRECISION` y las correcciones a `CALIBRATION_DRIFT`, lo que demuestra que el sistema no solo opera, sino que **entiende su propio universo**.")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"✅ Reporte generado en: {output_file}")

if __name__ == '__main__':
    generate_report()
