import os

target = "a996a033"
root_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"

for root, dirs, files in os.walk(root_dir):
    # Skip some heavy directories
    if any(d in root for d in [".git", ".venv", "__pycache__", "wandb"]):
        continue
        
    for file in files:
        if file.endswith((".json", ".log", ".txt", ".csv")):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    if target in f.read():
                        print(f"FOUND TARGET in: {path}")
            except:
                pass
