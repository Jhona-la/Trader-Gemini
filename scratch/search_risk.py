import os
import sys

def search_files(directory, query):
    results = []
    for root, dirs, files in os.walk(directory):
        if ".git" in root or ".venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            results.append(path)
                except Exception:
                    pass
    return results

print("Searching for 'omniscient'...")
paths = search_files(".", "omniscient")
for p in paths:
    print(f"Match: {p}")
