import logging
import random
from decimal import Decimal, getcontext
import time

# Set Decimal precision high enough
getcontext().prec = 50

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("PrecisionTest")

def run_precision_test():
    """
    Simulates 1,000,000 algorithmic trades to measure IEEE 754 floating point drift.
    Compares native float (fast) vs Decimal (precise).
    """
    print("="*60)
    print("🔢 PHASE 47: NUMERIC PRECISION STRESS TEST")
    print("="*60)
    
    # Initial Capital
    start_capital = 10000.00
    
    # 1. FLOAT SYSTEM (Current Trader Gemini)
    float_balance = start_capital
    
    # 2. DECIMAL SYSTEM (Control Group)
    decimal_balance = Decimal("10000.00")
    
    # Simulation Parameters
    iterations = 1_000_000
    print(f"🚀 Simulating {iterations:,} trades...")
    
    start_time = time.time()
    
    # Random see for reproducibility
    random.seed(42)
    
    # Pre-generate scenarios to avoid random overhead differences
    # We want to measure arithmetic drift, not random gen speed
    # But for simplicity, we do it in loop since we test drift not speed here
    
    drift_milestones = []
    
    for i in range(1, iterations + 1):
        # Scenario: Buying 0.001 BTC at ~50k
        qty = 0.001
        
        # Random price movement (+/- 0.5%)
        entry_price = 50000.00 + (random.random() * 100)
        exit_price = entry_price * (1.0 + (random.uniform(-0.005, 0.006))) # Slightly positive expectancy
        
        # FLOAT Calculation
        pnl_float = (exit_price - entry_price) * qty
        fee_float = (entry_price * qty * 0.0004) + (exit_price * qty * 0.0004) # Taker fees
        float_balance += (pnl_float - fee_float)
        
        # DECIMAL Calculation
        q_d = Decimal(str(qty))
        en_d = Decimal(f"{entry_price:.8f}") # Simulate price tick precision
        ex_d = Decimal(f"{exit_price:.8f}")
        
        pnl_d = (ex_d - en_d) * q_d
        fee_d = (en_d * q_d * Decimal("0.0004")) + (ex_d * q_d * Decimal("0.0004"))
        decimal_balance += (pnl_d - fee_d)
        
        # Check drift
        if i % 100_000 == 0:
            diff = abs(float(decimal_balance) - float_balance)
            print(f"  Step {i:,}: Float=${float_balance:,.6f} | Dec=${decimal_balance:,.6f} | Diff=${diff:.10f}")
            drift_milestones.append(diff)

    end_time = time.time()
    duration = end_time - start_time
    
    # Final Results
    final_diff = abs(float(decimal_balance) - float_balance)
    
    print("\n" + "="*60)
    print("📊 PRECISION REPORT")
    print("="*60)
    print(f"Trades Executed: {iterations:,}")
    print(f"Time Taken:      {duration:.2f}s")
    print(f"Final Float:     ${float_balance:,.12f}")
    print(f"Final Decimal:   ${decimal_balance:,.12f}")
    print("-" * 30)
    print(f"TOTAL DRIFT:     ${final_diff:.12f}")
    print("-" * 30)
    
    # Evaluation
    # Institutional Tolerance: $0.0001 per million volume is usually acceptable for HFT
    # If drift > $0.01, we have a problem.
    
    if final_diff < 0.01:
        print("✅ PASS: Floating point drift is negligible (< $0.01).")
        print("   Python 'float' (Inexact 53-bit) is safe for this strategy scale.")
    else:
        print("❌ FAIL: Significant precision loss detected.")
        print("   RECOMMENDATION: Migrate core accounting to 'decimal.Decimal'.")
    
    print("="*60)

if __name__ == "__main__":
    run_precision_test()
