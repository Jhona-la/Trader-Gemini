import os

search_terms = ["should_reject_signal", "get_strategy_metrics", "get_execution_params"]
root_dir = "c:/Users/jhona/Documents/Proyectos/Trader Gemini"
subdirs = ["core", "risk", "strategies", "ml", "execution", "data"]

for sd in subdirs:
    dir_path = os.path.join(root_dir, sd)
    if not os.path.exists(dir_path):
        continue
    for dirpath, _, filenames in os.walk(dir_path):
        if any(ignored in dirpath for ignored in [".git", ".models", "scratch", "cache", "historical"]):
            continue
        for fname in filenames:
            if fname.endswith(".py"):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    for term in search_terms:
                        if term in content:
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if term in line:
                                    print(f"{sd}/{fname}:{i+1}: {line.strip()}")
                except Exception as e:
                    pass
