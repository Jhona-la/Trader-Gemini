import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\risk\risk_manager.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace "[AEGIS]" prints with logger.warning("[RISK VETO]") so they get captured
content = content.replace('print(f"💀 [AEGIS] Global Veto: {symbol} is BLACKLISTED (Toxic Asset)")', 'logger.warning(f"🛑 [RISK VETO] AEGIS Toxic Asset: {symbol}")')
content = content.replace('print(\n                f"💀 [AEGIS] Global Veto: Kill Switch Active ({self.kill_switch.activation_reason})"\n            )', 'logger.warning(f"🛑 [RISK VETO] AEGIS Kill Switch Active: {self.kill_switch.activation_reason}")')
content = content.replace('print(f"🛑 [AEGIS] Frequency Limit Breached for {symbol} ({_sig_name}).")', 'logger.warning(f"🛑 [RISK VETO] AEGIS Frequency Limit: {symbol} {_sig_name}")')
content = content.replace('print(\n                f"❄️ [AEGIS] Cooldown active for {symbol} under {strategy_id}. Reason: {can_trade_res[1]}"\n            )', 'logger.warning(f"🛑 [RISK VETO] AEGIS Cooldown: {symbol} {strategy_id} {can_trade_res[1]}")')
content = content.replace('print(\n                f"🛡️ [AEGIS] Vetoing {strategy_id} for {symbol} in VOLATILE regime (Risk of whipsaw)."\n            )', 'logger.warning(f"🛑 [RISK VETO] AEGIS Volatile Regime Veto: {symbol} {strategy_id}")')
content = content.replace('print(f"🛡️ [AEGIS] Vetoing Mean Reversion in TRENDING regime.")', 'logger.warning(f"🛑 [RISK VETO] AEGIS Trending Regime Veto: {symbol} {strategy_id}")')

# Also fix the draw_down logging to have [RISK VETO]
content = content.replace('logger.critical(f"🛑 [DAILY DD HALT]', 'logger.critical(f"🛑 [RISK VETO] DAILY DD HALT')
content = content.replace('logger.warning(f"🛑 [LATENCY VETO]', 'logger.warning(f"🛑 [RISK VETO] LATENCY VETO')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("risk_manager.py patched successfully!")
