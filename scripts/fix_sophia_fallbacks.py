import re
import os

file_path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\sophia\intelligence.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Pattern to find except: followed by a return statement
pattern = re.compile(r'except:\s*\n\s*return [^\n]+')

def replacement(match):
    return """except Exception as e:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError(f"Holographic Audit: Sophia module failure blocked. Details: {e}")"""

new_content = pattern.sub(replacement, content)

# There are also `except Exception as e:\n return 0.5` ? No, they were `except:`.

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Sophia coin-flip fallbacks fixed!")
