"""
Trader Gemini - Security & Configuration Validation Tests

This test suite validates:
1. No hardcoded API keys in source code
2. Environment variables are properly loaded
3. Configuration validation works correctly
4. System fails fast on missing credentials
"""

import os
import sys
import re
import glob
from pathlib import Path


def test_no_hardcoded_secrets():
    """
    Scan all Python files for hardcoded API keys.
    CRITICAL: This test ensures no secrets are committed to source control.
    """
    print("\n" + "="*70)
    print("TEST 1: Scanning for Hardcoded API Keys")
    print("="*70)
    
    # Pattern to detect potential API keys (long alphanumeric strings)
    # Binance API keys are typically 64 characters
    api_key_pattern = re.compile(r'["\']([A-Za-z0-9]{50,})["\']')
    
    issues_found = []
    files_scanned = 0
    
    # Get project root (parent of tests directory)
    project_root = Path(__file__).parent.parent
    
    # Scan all .py files except this test file
    for py_file in project_root.rglob('*.py'):
        # Skip test files, venv, and __pycache__
        if 'test' in str(py_file) or '.venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
            
        files_scanned += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Look for suspicious patterns
                matches = api_key_pattern.findall(content)
                
                for match in matches:
                    # Filter out common false positives
                    if 'your_' in match.lower() or 'example' in match.lower():
                        continue  # Template/example key
                    if 'test' in match.lower():
                        continue  # Test key
                    
                    # Found potential hardcoded key
                    issues_found.append({
                        'file': str(py_file.relative_to(project_root)),
                        'key_preview': match[:20] + '...' + match[-10:]
                    })
        except Exception as e:
            print(f"  Warning: Could not scan {py_file.name}: {e}")
    
    print(f"  📁 Files scanned: {files_scanned}")
    
    if issues_found:
        print(f"\n  ❌ FAILED: Found {len(issues_found)} potential hardcoded keys:")
        for issue in issues_found:
            print(f"     - {issue['file']}: {issue['key_preview']}")
        return False
    else:
        print(f"  ✅ PASSED: No hardcoded API keys detected")
        return True


def test_env_file_exists():
    """
    Verify .env file exists and is not in git.
    """
    print("\n" + "="*70)
    print("TEST 2: Environment File Configuration")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    env_example = project_root / '.env.example'
    gitignore = project_root / '.gitignore'
    
    passed = True
    
    # Check .env.example exists
    if env_example.exists():
        print("  ✅ .env.example exists (template available)")
    else:
        print("  ❌ .env.example NOT found")
        passed = False
    
    # Check .gitignore protects .env
    if gitignore.exists():
        with open(gitignore, 'r', encoding='utf-8', errors='ignore') as f:
            gitignore_content = f.read()
            if '.env' in gitignore_content:
                print("  ✅ .env is in .gitignore (protected from git)")
            else:
                print("  ❌ .env is NOT in .gitignore (SECURITY RISK!)")
                passed = False
    else:
        print("  ⚠️  .gitignore not found")
    
    # Check if .env exists (optional - user must create it)
    if env_file.exists():
        print("  ✅ .env file exists")
        
        # Verify it has required keys
        with open(env_file, 'r', encoding='utf-8', errors='ignore') as f:
            env_content = f.read()
            required_keys = [
                'BINANCE_DEMO_API_KEY',
                'BINANCE_DEMO_SECRET_KEY',
                'BINANCE_TESTNET_API_KEY',
                'BINANCE_TESTNET_SECRET_KEY'
            ]
            
            for key in required_keys:
                if key in env_content:
                    # Check if it has a value (not empty)
                    pattern = rf'{key}=(.+)'
                    match = re.search(pattern, env_content)
                    if match and match.group(1).strip():
                        print(f"  ✅ {key} is set")
                    else:
                        print(f"  ⚠️  {key} is present but empty")
                else:
                    print(f"  ⚠️  {key} not found in .env")
    else:
        print("  ⚠️  .env file does NOT exist (create from .env.example)")
        print("     Run: copy .env.example .env")
    
    return passed


def test_config_validation():
    """
    Test that config.py validates correctly and fails fast on missing keys.
    """
    print("\n" + "="*70)
    print("TEST 3: Configuration Validation Logic")
    print("="*70)
    
    # We can't actually run config.py here because it will load the real .env
    # Instead, we verify the validation function exists
    
    project_root = Path(__file__).parent.parent
    config_file = project_root / 'config.py'
    
    if not config_file.exists():
        print("  ❌ config.py not found!")
        return False
    
    with open(config_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    passed = True
    
    # Check for dotenv import
    if 'from dotenv import load_dotenv' in content:
        print("  ✅ dotenv imported")
    else:
        print("  ❌ dotenv NOT imported")
        passed = False
    
    # Check for load_dotenv() call
    if 'load_dotenv()' in content:
        print("  ✅ load_dotenv() called")
    else:
        print("  ❌ load_dotenv() NOT called")
        passed = False
    
    # Check for validation function
    if 'def validate_config' in content:
        print("  ✅ validate_config() function exists")
    else:
        print("  ❌ validate_config() function NOT found")
        passed = False
    
    # Check for os.getenv usage
    getenv_count = content.count('os.getenv')
    if getenv_count >= 4:  # At least 4 API keys
        print(f"  ✅ os.getenv() used {getenv_count} times")
    else:
        print(f"  ⚠️  os.getenv() only used {getenv_count} times (expected ≥4)")
    
    # Check for sys.exit on validation failure
    if 'sys.exit(1)' in content:
        print("  ✅ Fail-fast implemented (sys.exit on error)")
    else:
        print("  ❌ Fail-fast NOT implemented")
        passed = False
    
    return passed


def test_exception_handling():
    """
    Verify specific exception handling is used instead of generic Exception.
    """
    print("\n" + "="*70)
    print("TEST 4: Exception Handling Quality")
    print("="*70)
    
    project_root = Path(__file__).parent.parent
    
    # Files to check for improved exception handling
    critical_files = [
        'execution/binance_executor.py',
        'main.py',
        'config.py'
    ]
    
    total_generic = 0
    total_specific = 0
    
    for file_path in critical_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Count generic exceptions
        generic = len(re.findall(r'except Exception as', content))
        
        # Count specific exceptions
        specific_patterns = [
            'ccxt.NetworkError',
            'ccxt.ExchangeError',
            'ccxt.AuthenticationError',
            'ccxt.InsufficientFunds',
            'ccxt.InvalidOrder',
            'FileNotFoundError',
            'PermissionError',
            'JSONDecodeError',
            'KeyError'
        ]
        
        specific = sum(content.count(pattern) for pattern in specific_patterns)
        
        total_generic += generic
        total_specific += specific
        
        print(f"\n  {file_path}:")
        print(f"    Generic exceptions: {generic}")
        print(f"    Specific exceptions: {specific}")
    
    print(f"\n  📊 Summary:")
    print(f"    Total generic: {total_generic}")
    print(f"    Total specific: {total_specific}")
    
    # Pass if specific > generic
    if total_specific > total_generic:
        print(f"  ✅ PASSED: Specific exception handling dominates")
        return True
    else:
        print(f"  ⚠️  IMPROVEMENT NEEDED: More generic than specific exceptions")
        return False


def run_all_tests():
    """
    Run all validation tests and report results.
    """
    print("\n" + "="*70)
    print("🔒 TRADER GEMINI - SECURITY & VALIDATION TEST SUITE")
    print("="*70)
    
    results = {
        'Hardcoded Secrets': test_no_hardcoded_secrets(),
        'Environment Files': test_env_file_exists(),
        'Config Validation': test_config_validation(),
        'Exception Handling': test_exception_handling()
    }
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    pass_rate = (total_passed / total_tests) * 100
    
    print(f"\n  Overall: {total_passed}/{total_tests} tests passed ({pass_rate:.0f}%)")
    
    if total_passed == total_tests:
        print("\n  🎉 ALL TESTS PASSED - System validation successful!")
        return 0
    else:
        print("\n  ⚠️  SOME TESTS FAILED - Review issues above")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
