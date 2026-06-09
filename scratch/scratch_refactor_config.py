import os

def refactor_config_references(root_dir):
    replacements = {
        "Config.Horizons.Scalping": "Config.Horizons.Scalping",
        "Config.Horizons.Swing": "Config.Horizons.Swing",
        "Config.Horizons.GlobalThresholds": "Config.Horizons.GlobalThresholds",
        "Config.Horizons.Mutations": "Config.Horizons.Mutations"
    }
    
    count = 0
    for subdir, _, files in os.walk(root_dir):
        if '.venv' in subdir or '.git' in subdir or '__pycache__' in subdir:
            continue
            
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(subdir, file)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                new_content = content
                for old, new in replacements.items():
                    new_content = new_content.replace(old, new)
                    
                if new_content != content:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    count += 1
                    print(f"Refactored: {filepath}")
                    
    print(f"Total files refactored: {count}")

if __name__ == "__main__":
    refactor_config_references(r"c:\Users\jhona\Documents\Proyectos\Trader Gemini")
