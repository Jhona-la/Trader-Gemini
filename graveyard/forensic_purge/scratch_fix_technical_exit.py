import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\strategies\technical.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

def replacer(match):
    full = match.group(0)
    strat_val = match.group(1)
    if 'position.get' in strat_val or 'pos.get' in strat_val or 'self.strategy_id' in strat_val:
        return full
    return full.replace(strat_val, 'position.get("opener_strategy_id", position.get("strategy_id", "Unknown"))')

# Nos enfocamos en check_exit que va desde la def check_exit hasta el final.
idx = content.find('def check_exit')
if idx != -1:
    part1 = content[:idx]
    part2 = content[idx:]
    part2_new = re.sub(r'strategy_id=("[^"]+")', replacer, part2)
    content_new = part1 + part2_new
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_new)
    print("Fix aplicado a technical.py")
else:
    print("No se encontro check_exit")
