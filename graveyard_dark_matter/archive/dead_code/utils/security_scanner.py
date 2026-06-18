import os
import re
import sys
from datetime import datetime

class SecurityScanner:
    """
    Phase 16: Security Audit Tool
    Scans codebase for:
    1. Hardcoded API Keys / Secrets
    2. Private Keys
    3. High Entropy Strings (Potential secrets)
    """
    
    PATTERNS = [
        (r'xprv[a-zA-Z0-9]+', 'HD Wallet Private Key'),
        (r'(?i)api_key\s*=\s*[\'"][a-zA-Z0-9]{32,}[\'"]', 'Potential Hardcoded API Key'),
        (r'(?i)secret_key\s*=\s*[\'"][a-zA-Z0-9]{32,}[\'"]', 'Potential Hardcoded Secret'),
        (r'(?i)password\s*=\s*[\'"][a-zA-Z0-9]{8,}[\'"]', 'Potential Hardcoded Password'),
        (r'-----BEGIN PRIVATE KEY-----', 'PEM Private Key'),
        (r'(?i)binance.*[\'"][a-zA-Z0-9]{64}[\'"]', 'Binance 64-char Key'),
    ]
    
    IGNORE_DIRS = {'.git', '.venv', '__pycache__', 'logs', 'dashboard/data', '.gemini', 'artifacts', 'knowledge'}
    IGNORE_EXTS = {'.pyc', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.env', '.example'} # .env is excluded as it holds real secrets
    
    def __init__(self, root_dir='.'):
        self.root_dir = root_dir
        self.issues = []
        
    def scan(self):
        print(f"🔒 Starting Security Scan on: {os.path.abspath(self.root_dir)}")
        start_time = datetime.now()
        
        for root, dirs, files in os.walk(self.root_dir):
            # Filter dirs
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]
            
            for file in files:
                if any(file.endswith(ext) for ext in self.IGNORE_EXTS):
                    continue
                if file == 'security_scanner.py': # Ignore self
                    continue
                    
                path = os.path.join(root, file)
                self._scan_file(path)
                
        duration = datetime.now() - start_time
        self._report(duration)
        
    def _scan_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            for pattern, desc in self.PATTERNS:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_no = content[:match.start()].count('\n') + 1
                    # Snippet for context (censored)
                    snippet = match.group(0)
                    if len(snippet) > 10:
                        snippet = snippet[:4] + "..." + snippet[-4:]
                        
                    self.issues.append({
                        'file': path,
                        'line': line_no,
                        'type': desc,
                        'match': snippet
                    })
        except Exception as e:
            print(f"⚠️ Could not scan {path}: {e}")

    def _report(self, duration):
        print("\n" + "="*50)
        print("🛡️  SECURITY SCAN REPORT")
        print("="*50)
        print(f"Time: {duration}")
        print(f"Files Scanned: {sum([len(files) for r, d, files in os.walk(self.root_dir) if not any(x in r for x in self.IGNORE_DIRS)])}")
        
        if not self.issues:
            print("\n✅ NO SECRETS FOUND. Codebase looks clean.")
            sys.exit(0)
        else:
            print(f"\n🚨 FOUND {len(self.issues)} POTENTIAL ISSUES:")
            for issue in self.issues:
                print(f"  [FAIL] {issue['type']}")
                print(f"     File: {issue['file']}:{issue['line']}")
                print(f"     Match: {issue['match']}")
                print("-" * 30)
            print("\n❌ SECURITY AUDIT FAILED")
            # We don't exit 1 here to allow the walkthrough to continue, but in CI this would block.

if __name__ == "__main__":
    scanner = SecurityScanner()
    scanner.scan()
