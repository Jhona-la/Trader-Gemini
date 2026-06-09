import os
import json

log_path = "C:/Users/jhona/.gemini/antigravity/brain/15de8b1e-38e4-4339-9be3-089a1e414d63/.system_generated/logs/transcript.jsonl"

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        try:
            step = json.loads(line)
            content = step.get("content", "")
            # Check for mentions of parameters that differ between config and another file
            if "config" in content.lower() and "risk_manager" in content.lower() and ("diferente" in content.lower() or "distinto" in content.lower() or "mismatch" in content.lower() or "divergen" in content.lower()):
                print(f"=== Step {step.get('step_index')} ===")
                print(content[:600])
                print("-" * 50)
        except Exception:
            pass
