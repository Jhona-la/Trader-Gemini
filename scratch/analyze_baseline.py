import json

baseline_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests\god_mode_4b059492_7d.json"
new_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\results\backtests\god_mode_6ef56e43_7d.json"

with open(baseline_path, 'r') as f:
    baseline = json.load(f)

with open(new_path, 'r') as f:
    new_data = json.load(f)

def print_summary(name, data):
    print(f"=== {name} ===")
    metrics = data.get('metrics', {})
    print(f"Final Capital: ${metrics.get('final_capital', 0):.4f} (Return: {metrics.get('total_return_pct', 0):.2f}%)")
    print(f"Total Trades: {metrics.get('total_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
    print(f"Total Fees: ${metrics.get('fees_paid', 0):.4f}")
    
    # Strategy attribution
    print("\nStrategy attribution detail:")
    for k, v in data.get('strategy_attribution', {}).items():
        print(f"  {k}: trades={v.get('trades')}, win_rate={v.get('win_rate', 0)*100:.1f}%, pnl=${v.get('pnl', 0):.4f}")

    print("\nRejection reasons:")
    print(json.dumps(data.get('rejection_reasons', {}), indent=2))
    print("\n")

print_summary("BASELINE (4b059492)", baseline)
print_summary("NEW RUN (6ef56e43)", new_data)
