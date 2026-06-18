import re

file_path = r'C:\Users\jhona\Documents\Proyectos\Trader Gemini\data\feature_store.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('df.empty', '(len(df) == 0)')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch applied to feature_store.py")
