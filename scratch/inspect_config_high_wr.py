import os
import json

def inspect_file(filename):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "backtests")
    file_path = os.path.join(results_dir, filename)
    if not os.path.exists(file_path):
        print(f"File {filename} not found.")
        return
        
    print(f"\n========================================================")
    print(f"🔍 INSPECTING FILE: {filename}")
    print(f"========================================================")
    
    with open(file_path, "r") as f:
        data = json.load(f)
        
    print("📋 METRICS:")
    print(json.dumps(data.get("metrics"), indent=2))
    
    print("\n⚙️ CONFIGURATION:")
    # Clean config print (only key strategy params)
    config = data.get("config", {})
    # Since config might be large or structured, let's print selectively or print the whole thing if it's small.
    # Usually config is quite readable:
    print(json.dumps(config, indent=2))

if __name__ == "__main__":
    # Inspect two interesting files
    inspect_file("god_mode_5bdc0859_2d.json")
    inspect_file("god_mode_45cae928_7d.json")
