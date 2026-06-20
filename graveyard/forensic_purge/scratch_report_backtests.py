import os
import json
import datetime
import re

results_dir = "results"
backtests_dir = os.path.join(results_dir, "backtests")
now = datetime.datetime.now()
three_days_ago = now - datetime.timedelta(days=3)

files_to_check = []

def get_recent_files(directory):
    if not os.path.exists(directory):
        return
    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime >= three_days_ago:
                if fname.endswith(".json") or fname.endswith(".txt"):
                    files_to_check.append(fpath)

get_recent_files(results_dir)
get_recent_files(backtests_dir)
get_recent_files(".")

report = []

for fpath in files_to_check:
    fname = os.path.basename(fpath)
    if fname.endswith(".json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "metrics" in data:
                    metrics = data["metrics"]
                    report.append({
                        "file": fname,
                        "type": "json",
                        "capital": metrics["final_capital"],
                        "total_return_pct": metrics["total_return_pct"],
                        "trades": metrics["total_trades"],
                        "win_rate": metrics["win_rate"],
                        "max_drawdown": metrics["max_drawdown_pct"]
                    })
                elif "final_capital" in data:
                    report.append({
                        "file": fname,
                        "type": "json",
                        "capital": data["final_capital"],
                        "total_return_pct": data["total_return_pct"],
                        "trades": data["total_trades"],
                        "win_rate": data["win_rate"],
                        "max_drawdown": data["max_drawdown_pct"]
                    })
        except:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
    elif fname.endswith(".txt"):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                capital_match = re.search(r"Final Capital:\s*\$?([\d\.]+)", content)
                return_match = re.search(r"Total Return:\s*([\d\.\-]+)%", content)
                trades_match = re.search(r"Total Trades:\s*(\d+)", content)
                wr_match = re.search(r"Win Rate:\s*([\d\.\-]+)%", content)
                dd_match = re.search(r"Max Drawdown:\s*([\d\.\-]+)%", content)
                
                if capital_match and trades_match:
                    report.append({
                        "file": fname,
                        "type": "txt",
                        "capital": float(capital_match.group(1)),
                        "total_return_pct": float(return_match.group(1)) if return_match else 0,
                        "trades": int(trades_match.group(1)),
                        "win_rate": float(wr_match.group(1)) if wr_match else 0,
                        "max_drawdown": float(dd_match.group(1)) if dd_match else 0
                    })
        except:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

# sort by capital
report.sort(key=lambda x: x["capital"], reverse=True)

print(f"Encontrados {len(report)} resultados (JSON/TXT) en los últimos 3 días.")
for r in report[:20]: # Show top 20
    print(f"File: {r['file']}, Capital: {r['capital']:.2f}, Return: {r['total_return_pct']}%, WR: {r['win_rate']}%, Trades: {r['trades']}, DD: {r['max_drawdown']}%")
