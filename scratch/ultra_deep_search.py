import os

target = b"a996a033"
root_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"

for root, dirs, files in os.walk(root_dir):
    if any(d in root for d in [".git", ".venv", "__pycache__"]):
        continue
    for file in files:
        path = os.path.join(root, file)
        try:
            with open(path, "rb") as f:
                if target in f.read():
                    print(f"FOUND IN: {path}")
        except:
            pass
