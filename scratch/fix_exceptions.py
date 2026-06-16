import os
import re
import glob

def fix_swallowed_exceptions():
    root_dir = r"C:\\Users\\jhona\\Documents\\Proyectos\\Trader Gemini"
    
    # Files identified by the scanner
    files_to_fix = [
        "core\\quantum_engine.py",
        "dashboard\\app.py",
        "data\\database.py",
        "core\\signal_scorer.py",
        "core\\sovereign_memory.py",
        "core\\state_manager.py",
        "data\\binance_loader.py"
    ]
    
    for rel_path in files_to_fix:
        filepath = os.path.join(root_dir, rel_path)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Replace bare "except:" with "except Exception as e:"
        # Only if it's on its own line and ends with colon
        # Be careful to preserve indentation
        content = re.sub(
            r'^(\s+)except:$',
            r'\1except Exception as e:\n\1    logger.exception(f"Bare except ghost bug: {e}")',
            content,
            flags=re.MULTILINE
        )
        
        # 2. Replace "except Exception:" with "except Exception as e:"
        content = re.sub(
            r'^(\s+)except Exception:$',
            r'\1except Exception as e:\n\1    logger.exception(f"Swallowed exception ghost bug: {e}")',
            content,
            flags=re.MULTILINE
        )
        
        # 3. Look for "except Exception as e:" followed immediately by "pass"
        # We will replace the pass with a logger.exception
        # Regex explanation:
        # (\s+)except Exception as e:\s*\n(\s+)pass\b
        # Replacement:
        # \1except Exception as e:\n\2logger.error(f"Ghost bug exception passed: {e}")
        content = re.sub(
            r'^(\s+)except Exception as e:\s*\n(\s+)pass\b',
            r'\1except Exception as e:\n\2logger.exception(f"Ghost bug passed: {e}")',
            content,
            flags=re.MULTILINE
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Fixed swallowed exceptions in {rel_path}")

if __name__ == '__main__':
    fix_swallowed_exceptions()
