import os
import ast
import sys

def check_strict_gets(directory):
    violations = []
    if not os.path.exists(directory):
        return violations
        
    for root, _, files in os.walk(directory):
        for f in files:
            if not f.endswith('.py'): continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                try:
                    tree = ast.parse(file.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            if node.func.attr == 'get':
                                # Si tiene 2 argumentos (tiene default) o está encadenado
                                if len(node.args) == 2 or (isinstance(node.func.value, ast.Call) and getattr(node.func.value.func, 'attr', '') == 'get'):
                                    violations.append(f"{path}: L{node.lineno} -> Uso de .get() laxo o encadenado.")
                except SyntaxError:
                    from utils.error_handler import SystemIntegrityError
                    raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
    return violations

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Linter Anti-Falso-Positivo para detectar .get() con default.")
    parser.add_argument('--dirs', nargs='+', default=['core', 'strategies'], help='Directorios a escanear')
    args = parser.parse_args()
    
    all_violations = []
    for d in args.dirs:
        all_violations.extend(check_strict_gets(d))
        
    if all_violations:
        print(f"❌ {len(all_violations)} VIOLACIONES DE INTEGRIDAD DETECTADAS (Uso de .get con default):")
        for err in all_violations[:50]:
            print(err)
        print("...")
        sys.exit(1)
        
    print("✅ INTEGRIDAD DE CONTRATOS VALIDADA.")
    sys.exit(0)
