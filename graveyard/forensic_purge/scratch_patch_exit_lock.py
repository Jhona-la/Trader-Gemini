import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\risk\risk_manager.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace all occurrences of: pos['exit_pending_time'] = time.time()
# With: pos['exit_pending_time'] = now.timestamp() if now else time.time()

# Also handle pos["exit_pending_time"] = time.time()

content_new = re.sub(r'pos\[\'exit_pending_time\'\]\s*=\s*time\.time\(\)', 'pos[\'exit_pending_time\'] = now.timestamp() if now else time.time()', content)
content_new = re.sub(r'pos\[\"exit_pending_time\"\]\s*=\s*time\.time\(\)', 'pos[\'exit_pending_time\'] = now.timestamp() if now else time.time()', content_new)

if content != content_new:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_new)
    print("Fix de exit_pending_time aplicado a risk_manager.py.")
else:
    print("No se realizaron cambios (ya estaban aplicados o no encontrados).")
