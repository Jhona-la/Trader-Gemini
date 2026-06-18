import re
import os

files_to_patch = [
    'risk/risk_manager.py',
    'strategies/ml_strategy.py',
    'strategies/technical.py',
    'core/engine.py',
    'execution/binance_executor.py',
    'data/binance_loader.py'
]

for filepath in files_to_patch:
    print(f"Patching {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Replace except Exception as e: pass
    content = re.sub(
        r'(\n[ \t]*)except\s+Exception\s+as\s+e:\s*\n([ \t]+)pass', 
        r'\1except Exception as e:\n\2import logging\n\2logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)', 
        content
    )
    
    # 2. Replace except Exception: pass
    content = re.sub(
        r'(\n[ \t]*)except\s+Exception:\s*\n([ \t]+)pass', 
        r'\1except Exception as e:\n\2import logging\n\2logging.getLogger(__name__).error(f"Silent exception caught: {e}", exc_info=True)', 
        content
    )

    # 3. Replace logger.debug/warning/info that are actually catching exceptions and not logging them properly
    content = re.sub(
        r'logger\.debug\((f".*?(?:error|exception|failed).*?")\)', 
        r'logger.error(\1, exc_info=True)', 
        content,
        flags=re.IGNORECASE
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

print("Patching complete.")
