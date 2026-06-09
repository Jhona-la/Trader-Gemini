import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\risk\risk_manager.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Buscamos todas las llamadas a SignalEvent dentro de check_stops que usan strategy_id="ALGO"
# y las reemplazamos por strategy_id=pos.get("opener_strategy_id", "Unknown")
# Además guardamos el string original en "exit_reason" de metadata si no está.

# Ejemplo: strategy_id="TIME_STOP_ZOMBIE" -> strategy_id=pos.get('opener_strategy_id', 'Unknown')
# pero dejemos que re lo arregle todo donde sea literal string.

def replacer(match):
    full = match.group(0)
    strat_val = match.group(1)
    if 'pos.get' in strat_val or 'original_strategy' in strat_val or 'opener_id' in strat_val:
        return full
    # Si es literal
    return full.replace(strat_val, 'pos.get("opener_strategy_id", "Unknown")')

# Cuidado con hacer un regex demasiado amplio.
content_new = re.sub(r'strategy_id=("[^"]+"|trail_name|oracle_decision|original_strategy)', replacer, content)

with open(path, "w", encoding="utf-8") as f:
    f.write(content_new)

print("Fix aplicado a risk_manager.py")
