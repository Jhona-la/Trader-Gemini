import os

ignore_dirs = {'.venv', '.git', '__pycache__', 'wandb', '.pytest_cache', '.ruff_cache', '.vscode', '.models', '.models_backtest', 'build', '.xai', '.sixth', 'artifacts', 'logs', 'logs_test_verify', 'H1_1m', 'results', '.agent', '.agents', '.antigravitycli', '.qodo', 'grafana', 'aits_research', 'archive', 'antigravity_export'}
startpath = r'c:\Users\jhona\Documents\Proyectos\Trader Gemini'
lines = []
lines.append('# Estructura de Archivos del Proyecto Trader Gemini\n')
lines.append('A continuación se muestra la lista de archivos organizados por sus respectivas carpetas (excluyendo carpetas de entorno virtual, caché y logs para mayor claridad):\n')
lines.append('```markdown')
for root, dirs, files in os.walk(startpath):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    level = root.replace(startpath, '').count(os.sep)
    indent = '    ' * level
    folder_name = os.path.basename(root)
    if root == startpath:
        lines.append(f'- 📁 **Trader Gemini** (Raíz)')
    else:
        lines.append(f'{indent}- 📁 **{folder_name}/**')
    subindent = '    ' * (level + 1)
    for f in files:
        if f.endswith('.log') or f.endswith('.db') or f.endswith('.pyd') or f.endswith('.bak_audit') or f.endswith('.json'):
            continue
        lines.append(f'{subindent}- 📄 {f}')
lines.append('```\n')

with open(r'C:\Users\jhona\.gemini\antigravity\brain\1228be73-fcdc-406f-97d1-68e75b6245ee\lista_archivos.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
