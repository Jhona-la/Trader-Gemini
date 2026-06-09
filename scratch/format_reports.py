import os
import re

base_dir = r"C:\Users\jhona\.gemini\antigravity\brain\15de8b1e-38e4-4339-9be3-089a1e414d63\scratch"

for name in ["report_signal_flow", "report_position_lifecycle"]:
    raw_path = os.path.join(base_dir, f"{name}_1.md")
    out_path = os.path.join(base_dir, f"{name}.md")
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        continue
        
    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
    # Remove leading and trailing quotes if the string is wrapped in them
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1]
        
    # Unescape common sequences
    content = content.replace(r"\n", "\n")
    content = content.replace(r"\t", "\t")
    content = content.replace(r"\"", '"')
    content = content.replace(r"\'", "'")
    content = content.replace(r"\\", "\\")
    
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(content)
        
    print(f"Formatted and saved to: {out_path} ({len(content)} characters)")
