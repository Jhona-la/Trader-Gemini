import os
import re
import json
from pathlib import Path

def scan_features(root_dir):
    results = {
        "features": {},
        "memory_breaks": [],
        "float64_conversions": [],
        "ffi_crossings": []
    }
    
    # Feature assignment pattern (e.g., self.features['VPIN'] = ..., state['macd'] = ...)
    feature_pattern = re.compile(r"['\"](\w+)['\"]\s*\]?\s*=\s*(.+)")
    # Memory break pattern
    mem_break_pattern = re.compile(r"\.(copy|astype|fillna|reshape|transpose)\(")
    # FFI boundary pattern
    ffi_pattern = re.compile(r"(frombuffer|MemoryView|cdef|unsafe|ffi|PyObject)")
    
    exclude_dirs = {'.venv', 'venv', 'node_modules', '.git', '__pycache__', 'build', 'dist'}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in ['.py', '.pyx', '.c', '.cpp', '.rs']:
                filepath = Path(root) / file
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                        for i, line in enumerate(lines):
                            line_num = i + 1
                            
                            # Extract Features (heuristically)
                            if 'feature' in line.lower() or 'state' in line.lower():
                                m = feature_pattern.search(line)
                                if m:
                                    feat_name = m.group(1)
                                    if feat_name not in results["features"]:
                                        results["features"][feat_name] = []
                                    results["features"][feat_name].append({
                                        "file": str(filepath.relative_to(root_dir)),
                                        "line": line_num,
                                        "context": line.strip()
                                    })
                                    
                            # Check memory breaks
                            if mem_break_pattern.search(line):
                                results["memory_breaks"].append({
                                    "file": str(filepath.relative_to(root_dir)),
                                    "line": line_num,
                                    "type": "memory_contiguity_break",
                                    "context": line.strip()
                                })
                                
                            # Check float64 -> float32 implicit conversions
                            if 'float32' in line or 'f32' in line or '.astype(np.float32)' in line:
                                results["float64_conversions"].append({
                                    "file": str(filepath.relative_to(root_dir)),
                                    "line": line_num,
                                    "context": line.strip()
                                })
                                
                            # Check FFI
                            if ffi_pattern.search(line):
                                results["ffi_crossings"].append({
                                    "file": str(filepath.relative_to(root_dir)),
                                    "line": line_num,
                                    "context": line.strip()
                                })
                                
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    
    return results

if __name__ == "__main__":
    project_root = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini")
    print("Scanning for features and boundaries...")
    results = scan_features(project_root)
    
    output_file = project_root / "scratch" / "feature_findings.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_file}")
    print("\n--- GENEALOGIST SUMMARY ---")
    print(f"FEATURES FOUND: {len(results['features'])} unique features")
    print(f"MEMORY BREAKS: {len(results['memory_breaks'])}")
    print(f"FLOAT32 DOWNSCALES: {len(results['float64_conversions'])}")
    print(f"FFI CROSSINGS: {len(results['ffi_crossings'])}")
