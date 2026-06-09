import os

def search_ml_file(file_path):
    print(f"=== Searching {file_path} ===")
    if not os.path.exists(file_path):
        print("File does not exist.")
        return
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Let's count how many times "143" or "features" or "dim" is used.
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if '143' in line or 'feature_cols' in line or 'input_dim' in line or 'len(features)' in line:
            print(f"Line {i+1}: {line}")
            if len(line) < 150:
                # print a context of 3 lines before and after
                start = max(0, i-2)
                end = min(len(lines), i+3)
                print("\n".join(lines[start:end]))
                print("-" * 30)

ml_strategy_path = r"c:\Users\jhona\Documents\Proyectos\Trader Gemini\strategies\ml_strategy.py"
search_ml_file(ml_strategy_path)
