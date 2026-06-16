import json

def analyze():
    try:
        with open('dashboard/data/backtest_telemetry_spam.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Log file not found.")
        return

    closed_messages = []
    for x in lines:
        if 'TRADE CERRADO' in x:
            try:
                data = json.loads(x)
                if 'message' in data:
                    closed_messages.append(data['message'])
            except:
                pass

    net_pnls = []
    for m in closed_messages:
        for line in m.split('\n'):
            if '*PnL Neto:' in line:
                # Example line: *PnL Neto: `+$0.0041`* (+0.06%)
                parts = line.split('`')
                if len(parts) >= 2:
                    val_str = parts[1].replace('$', '').replace('+', '')
                    try:
                        net_pnls.append(float(val_str))
                    except:
                        pass
                break

    wins = sum(1 for p in net_pnls if p > 0)
    losses = sum(1 for p in net_pnls if p <= 0)
    total_net = sum(net_pnls)
    
    print(f"Total Closed Trades: {len(closed_messages)}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {(wins/len(net_pnls)*100) if net_pnls else 0:.2f}%")
    print(f"Net PnL sum: ${total_net:.4f}")

if __name__ == "__main__":
    analyze()
