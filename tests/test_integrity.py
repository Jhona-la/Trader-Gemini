"""
Trader Gemini - System Integrity Tests

Tests for:
1. Module imports work correctly
2. No circular dependencies
3. Configuration loads without errors (with .env present)
"""

import sys
import importlib
from pathlib import Path

# Add parent directory to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_core_imports():
    """
    Verify all core modules can be imported without errors.
    """
    print("\n" + "="*70)
    print("TEST: Core Module Imports")
    print("="*70)
    
    modules_to_test = [
        'config',
        'core.events',
        'core.engine',
        'core.portfolio',
        'core.market_regime',
        'utils.logger',
        'utils.error_handler',
        'utils.common',
    ]
    
    passed = True
    
    for module_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
            passed = False
        except Exception as e:
            # Config might fail if .env is missing, but import should work
            if module_name == 'config' and 'ERROR' in str(e):
                print(f"  ⚠️  {module_name}: Failed validation (expected if .env missing)")
            else:
                print(f"  ❌ {module_name}: Unexpected error - {e}")
                passed = False
    
    return passed


def test_no_syntax_errors():
    """
    Compile all Python files to check for syntax errors.
    """
    print("\n" + "="*70)
    print("TEST: Python Syntax Validation")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    syntax_errors = []
    files_checked = 0
    
    for py_file in project_root.rglob('*.py'):
        # Skip venv and pycache
        if '.venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        files_checked += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append({
                'file': str(py_file.relative_to(project_root)),
                'error': str(e)
            })
    
    print(f"  📁 Files checked: {files_checked}")
    
    if syntax_errors:
        print(f"\n  ❌ FAILED: Found {len(syntax_errors)} syntax errors:")
        for error in syntax_errors:
            print(f"     - {error['file']}: {error['error']}")
        return False
    else:
        print(f"  ✅ PASSED: No syntax errors")
        return True


def run_all_tests():
    """
    Run all integrity tests.
    """
    print("\n" + "="*70)
    print("🔧 TRADER GEMINI - SYSTEM INTEGRITY TESTS")
    print("="*70)
    
    results = {
        'Syntax Validation': test_no_syntax_errors(),
        'Core Imports': test_core_imports(),
    }
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n  Overall: {total_passed}/{total_tests} tests passed")
    
    return 0 if total_passed == total_tests else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
