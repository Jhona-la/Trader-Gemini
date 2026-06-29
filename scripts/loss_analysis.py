import pandas as pd
import os

def analyze_losses():
    paths = [
        r'c:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\data\futures\trades.csv',
        r'c:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\data\trades.csv',
        r'c:\Users\jhona\Documents\Proyectos\Trader Gemini\dashboard\data\trades_archive.csv'
    ]
    
    dfs = []
    for p in paths:
        if os.path.exists(p):
            try:
                dfs.append(pd.read_csv(p))
            except Exception as e:
                print(f"Error reading {p}: {e}")
                
    if not dfs:
        print("No trade data found.")
        return
        
    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        print("Data is empty.")
        return
        
    if 'net_pnl' in df.columns:
        pnl_col = 'net_pnl'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    else:
        print("No PnL column found.")
        return
        
    # Standardize
    df[pnl_col] = pd.to_numeric(df[pnl_col], errors='coerce')
    
    # Worst 10 trades
    worst = df.nsmallest(15, pnl_col)
    
    # Print the relevant columns
    cols = ['datetime', 'symbol', 'direction', 'quantity', 'entry_price', 'exit_price', pnl_col]
    if 'commission' in df.columns: cols.append('commission')
    if 'details' in df.columns: cols.append('details')
    
    existing_cols = [c for c in cols if c in worst.columns]
    
    print("\n" + "="*50)
    print("🚨 THE 15 WORST TRADES IDENTIFIED 🚨")
    print("="*50)
    print(worst[existing_cols].to_string(index=False))
    
    # General metrics
    total_fees = df['commission'].sum() if 'commission' in df.columns else 0
    total_loss_amount = worst[pnl_col].sum()
    
    print("\n" + "="*50)
    print("📊 LOSS METRICS")
    print(f"Total fees paid across all data: ${total_fees:.4f}")
    print(f"Total value of the 15 worst trades: ${total_loss_amount:.4f}")

if __name__ == "__main__":
    analyze_losses()
