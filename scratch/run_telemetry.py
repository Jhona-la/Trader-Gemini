import sys
import os
import traceback

sys.path.insert(0, '.')

def patch_and_run():
    print("Reading scripts/run_god_mode_backtest.py...")
    with open('scripts/run_god_mode_backtest.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Let's replace the except block to print the traceback
    target = "logger.debug(f\"Strategy {g_strat.strategy_id} error on {me.symbol}: {e}\")"
    replacement = "traceback.print_exc(); logger.debug(f\"Strategy {g_strat.strategy_id} error on {me.symbol}: {e}\")"
    
    if target not in code:
        target = "logger.debug(f'Strategy {g_strat.strategy_id} error on {me.symbol}: {e}')"
        replacement = "traceback.print_exc(); logger.debug(f'Strategy {g_strat.strategy_id} error on {me.symbol}: {e}')"
    
    if target not in code:
        print("Target not found! Code structure might have changed. Let's look for calculate_signals except:")
        # Let's find where calculate_signals is called and print surroundings
        idx = code.find("g_strat.calculate_signals(me)")
        if idx != -1:
            print("Found calculate_signals at index:", idx)
            print(code[idx:idx+300])
        return

    print("Patching target...")
    code_mod = code.replace(target, replacement)
    
    # We want to run it for a very short duration.
    # Let's see if we can limit the number of epochs or set duration_days to 0.01 or similar.
    # In run_god_mode_backtest.py, we have main() which parses arguments.
    # Let's write the modified code to a temporary file
    temp_script = "scratch/temp_backtest_run.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(code_mod)
    print(f"Patched script written to {temp_script}!")

if __name__ == "__main__":
    patch_and_run()
