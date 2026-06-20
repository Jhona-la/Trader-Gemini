import os
import re
import json

root_dir = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini"
target_dirs = ["core", "strategies", "ml", "risk"]
target_exts = {".py", ".pyx", ".c", ".cpp", ".rs"}

def scan_features():
    results = {
        "features": [],
        "tensors": [],
        "singletons": []
    }
    
    # Very basic regex heuristics
    feature_pattern = re.compile(r'(?:self\.|)(\w+_[fF]eature|\w+_[iI]ndicator|\w+_calc\w*)\s*=')
    tensor_pattern = re.compile(r'torch\.frombuffer|np\.ndarray\(buffer')
    singleton_pattern = re.compile(r'class \w+\(.*?Singleton.*?\):|global \w+')
    
    for d in target_dirs:
        dir_path = os.path.join(root_dir, d)
        if not os.path.exists(dir_path): continue
        for root, _, files in os.walk(dir_path):
            if any(ignored in root for ignored in ['.git', '__pycache__', 'build']):
                continue
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in target_exts:
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        for i, line in enumerate(lines):
                            if feature_pattern.search(line):
                                results["features"].append({"file": filepath, "line": i+1, "content": line.strip()})
                            if tensor_pattern.search(line):
                                results["tensors"].append({"file": filepath, "line": i+1, "content": line.strip()})
                            if singleton_pattern.search(line):
                                results["singletons"].append({"file": filepath, "line": i+1, "content": line.strip()})
                    except:
                        pass
                        
    with open(os.path.join(root_dir, "scratch", "feature_audit.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    scan_features()
