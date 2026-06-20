import json
import numpy as np
import argparse

def summary(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error opening {input_file}: {e}")
        return
        
    print(f"## Backtest Execution Results: {input_file}\n")
    
    # Check if the format is the standard multi-horizon results or the new God Mode unified output
    if 'metrics' in data and 'version' in data:
        metrics = data['metrics']
        print(f"Version: {data.get('version')} | Run ID: {data.get('run_id')}")
        print(f"Total Return: {metrics.get('total_return_pct')} % | Sharpe: {metrics.get('sharpe_ratio')} | Max DD: {metrics.get('max_drawdown_pct')} %")
        print(f"Win Rate: {metrics.get('win_rate')} % | Total Trades: {metrics.get('total_trades')}")
        
    else:
        print("| Horizon | Strategy | Total PNL | Avg PNL % | Avg Win Rate | Avg Max DD | Avg Sharpe |")
        print("|---|---|---|---|---|---|---|")
        
        for hz, symbols in data.items():
            if not isinstance(symbols, dict): continue
            strats = ["Technical", "Sophia", "ML_XGBoost", "Orchestrator"]
            for s in strats:
                pnls, pnl_pcts, wrs, dds, sharpes = [], [], [], [], []
                for sym, results in symbols.items():
                    if isinstance(results, dict) and s in results:
                        r = results[s]
                        pnls.append(r['pnl_usd'])
                        pnl_pcts.append(r['pnl_pct'])
                        wrs.append(r['win_rate'])
                        dds.append(r['max_drawdown'])
                        sharpes.append(r['sharpe'])
                
                if pnls:
                    tot_pnl = sum(pnls)
                    avg_pct = np.mean(pnl_pcts)
                    avg_wr = np.mean(wrs)
                    avg_dd = np.mean(dds)
                    avg_s = np.mean(sharpes)
                    print(f"| {hz} | {s} | ${tot_pnl:.3f} | {avg_pct:.2f}% | {avg_wr:.1f}% | {avg_dd:.2f}% | {avg_s:.2f} |")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize backtest results')
    parser.add_argument('--input', type=str, required=True, help='Path to the results JSON file')
    args = parser.parse_args()
    summary(args.input)
