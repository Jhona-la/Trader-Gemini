import subprocess
import json
import re
import asyncio
from datetime import datetime
import os

DURATIONS = [1, 7, 15, 30]
OUTPUT_FILE = "docs/BATCH_BACKTEST_REPORT.md"
SYMBOL = "BTC/USDT"

async def run_backtest(days):
    print(f"[{datetime.now()}] 🚀 Iniciando simulación God Mode para {days} días...")
    # Lanza el subproceso pero en un thread separado para capturar
    process = await asyncio.create_subprocess_shell(
        f".venv\\Scripts\\python.exe scripts\\run_god_mode_backtest.py --days {days} --symbol {SYMBOL}",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    output = stdout.decode('utf-8', errors='ignore')
    
    print(f"[{datetime.now()}] ✅ Completado {days} días. Procesando métricas...")
    return parse_metrics(output, days)

def parse_metrics(output, days):
    # Intentamos capturar el bloque final del resumen
    report = {
        'days': days,
        'final_capital': 'N/A',
        'pnl_pct': 'N/A',
        'win_rate': 'N/A',
        'total_trades': 'N/A',
        'max_drawdown': 'N/A',
        'sharpe': 'N/A'
    }
    
    # Expresiones regulares para capturar métricas finales comunes en los prints
    # Si GodModeBacktester tira una tabla al final:
    capital_match = re.search(r"Final Capital:\s*\$([0-9\.]+)", output)
    if capital_match: report['final_capital'] = capital_match.group(1)
        
    trades_match = re.search(r"Total Trades:\s*(\d+)", output)
    if trades_match: report['total_trades'] = trades_match.group(1)
        
    wr_match = re.search(r"Win Rate:\s*([0-9\.]+)\%", output)
    if wr_match: report['win_rate'] = wr_match.group(1)
        
    dd_match = re.search(r"Max Drawdown:\s*([0-9\.]+)\%", output)
    if dd_match: report['max_drawdown'] = dd_match.group(1)
        
    return {
        'days': days,
        'metrics': report,
        'raw_tail': output[-3000:] # Last bits of output for context
    }

async def main():
    results = []
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"# 🚀 REPORTE DE BACKTESTS MASIVOS - GOD MODE ({SYMBOL})\n\n")
        f.write("Reporte de viabilidad en múltiples horizontes (1, 7, 15 y 30 días) con $13 de capital inicial.\n\n")
        
    for d in DURATIONS:
        res = await run_backtest(d)
        results.append(res)
        
        # Append to report
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(f"## 📊 Horizonte Evaluado: {d} Días\n")
            f.write(f"- **Capital Final Est.** : ${res['metrics']['final_capital']}\n")
            f.write(f"- **Total Operaciones**  : {res['metrics']['total_trades']}\n")
            f.write(f"- **Win Rate Est.**      : {res['metrics']['win_rate']}%\n")
            f.write(f"- **Max Drawdown**       : {res['metrics']['max_drawdown']}%\n\n")
            f.write("### Últimos Logs del Motor (Contexto Estratégico)\n")
            f.write("```text\n")
            f.write(res['raw_tail'])
            f.write("\n```\n\n---\n")

if __name__ == "__main__":
    asyncio.run(main())
