import sys
sys.path.append('.') # To allow imports
from config import Config

def simulate_sizing():
    current_cap = 15.62
    risk_pct = 0.05
    risk_usd = current_cap * risk_pct
    
    # Event mock
    class EventMock:
        pass
    event = EventMock()
    # If the event doesn't have sl_pct, getattr returns 1.5
    raw_sl_pct = getattr(event, 'sl_pct', 1.5)
    
    # Wait, in HybridScalpingStrategy, event generated DOES have sl_pct?
    print(f"Risk USD: {risk_usd}")
    print(f"Raw SL: {raw_sl_pct}")
    
    sl_decimal = raw_sl_pct / 100.0 if raw_sl_pct > 0.1 else raw_sl_pct
    print(f"SL Decimal: {sl_decimal}")
    
    size_usd = (risk_usd / sl_decimal) if sl_decimal > 0 else (current_cap * 0.1)
    print(f"Size USD Initial: {size_usd}")
    
    metadata = {}
    tp_mult = metadata.get('tp_mult', 1.0)
    sl_mult = metadata.get('sl_mult', 1.0)
    
    sl_decimal = sl_decimal * sl_mult
    # ...
    
    size_usd = min(size_usd, current_cap * 10)
    print(f"Size USD after max_cap cap * 10 -> {size_usd}")
    
    # Then wait...
    # Institutional min:
    if size_usd < 5.0:
        size_usd = 5.0
        
    print(f"Final Size USD: {size_usd}")

if __name__ == '__main__':
    simulate_sizing()
