import os
import re
import ast
import shutil
from pathlib import Path

def mutate_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    mutated = False
    new_lines = []
    
    # regex for .get('key', default)
    # This is a basic heuristic for .get('str', val) or .get("str", val)
    # Handling nested parenthesis is hard with regex, so we'll do a simple match for the most common ones
    # pattern: variable.get('key', default)
    get_pattern = re.compile(r"(\w+(?:\[.*?\])?(?:[.]\w+)*)\.get\(\s*(['\"].*?['\"])\s*,\s*[^)]+\)")
    
    # regex for except Exception: pass or except: pass
    except_pass_pattern = re.compile(r"^(\s*)pass\s*$")
    in_except = False
    except_indent = ""
    
    # regex for return 0.0
    return_zero_pattern = re.compile(r"^(\s*)return\s+(0\.0|0)\s*$")
    
    for i, line in enumerate(lines):
        original_line = line
        
        # 1. dict.get('key', default) -> dict['key']
        if ".get(" in line:
            new_line, count = get_pattern.subn(r"\1[\2]", line)
            if count > 0:
                line = new_line
                mutated = True
        
        # Track except blocks
        if re.match(r"^\s*except.*:", line):
            in_except = True
            except_indent = re.match(r"^(\s*)", line).group(1)
            # If the except block doesn't capture the exception, we might need to modify it
            # But regex modification of the except clause is risky, so we just track state
        elif line.strip() != "" and not line.strip().startswith("#"):
            if in_except:
                # 2. except pass -> raise SystemIntegrityError
                m = except_pass_pattern.match(line)
                if m:
                    indent = m.group(1)
                    line = f"{indent}from utils.error_handler import SystemIntegrityError\n{indent}raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')\n"
                    mutated = True
                    in_except = False # Block handled
                
                # 3. return 0.0 -> raise SystemIntegrityError
                m = return_zero_pattern.match(line)
                if m:
                    indent = m.group(1)
                    line = f"{indent}from utils.error_handler import SystemIntegrityError\n{indent}raise SystemIntegrityError('Return 0.0 fallback blocked by Holographic Audit')\n"
                    mutated = True
                    in_except = False # Block handled
            
            if in_except and not (line.strip().startswith("except") or line.strip() == ""):
                # If we encounter any actual code other than pass or return 0.0, we exit the except state
                # Wait, what if there are multiple lines? We just exit except state for simplicity.
                in_except = False

        new_lines.append(line)

    if mutated:
        new_content = "".join(new_lines)
        # Validate AST
        try:
            ast.parse(new_content)
        except SyntaxError as e:
            print(f"Skipping {filepath} due to AST syntax error post-mutation: {e}")
            return False

        # Backup
        backup_dir = Path("scratch/backup_pre_mutacion")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{filepath.name}.bak"
        shutil.copy2(filepath, backup_path)
        
        # Write
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
            
        print(f"Mutated successfully: {filepath}")
        return True
    return False

def main():
    project_root = Path("C:/Users/jhona/Documents/Proyectos/Trader Gemini")
    exclude_dirs = {'.venv', 'venv', 'node_modules', '.git', '__pycache__', 'build', 'dist', 'tests'}
    
    count = 0
    for root, dirs, files in os.walk(project_root):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file.endswith('.py') and "scratch" not in root:
                filepath = Path(root) / file
                if mutate_file(filepath):
                    count += 1
                    
    print(f"\nTotal files mutated and verified: {count}")

if __name__ == "__main__":
    main()
