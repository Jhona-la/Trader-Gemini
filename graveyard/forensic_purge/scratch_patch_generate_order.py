import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\risk\risk_manager.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

patch_code = """
        # ================================================================
        # 1.0. PREDICTIVE TP LIMIT BYPASS
        # QUÉ: Genera una orden LIMIT en el exchange para el TP exacto
        # ================================================================
        if getattr(signal_event, "strategy_id", "") == "PLACE_TP_LIMIT":
"""

new_code = """
        # ================================================================
        # 1.1 NORMAL EXIT BYPASS
        # QUÉ: Exits directos deben saltar lógica de sizing/gates
        # ================================================================
        if signal_event.signal_type == SignalType.EXIT and getattr(signal_event, "strategy_id", "") != "PLACE_TP_LIMIT":
            return self._generate_exit_order(signal_event, current_price)

        # ================================================================
        # 1.0. PREDICTIVE TP LIMIT BYPASS
        # QUÉ: Genera una orden LIMIT en el exchange para el TP exacto
        # ================================================================
        if getattr(signal_event, "strategy_id", "") == "PLACE_TP_LIMIT":
"""

if patch_code in content and "1.1 NORMAL EXIT BYPASS" not in content:
    content_new = content.replace(patch_code, new_code)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_new)
    print("Fix de generate_order aplicado.")
else:
    print("No se pudo aplicar el fix (ya aplicado o no encontrado).")
