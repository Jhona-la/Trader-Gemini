import os
import shutil
import glob

# DIRECTORIES TO PURGE (Phase 16-18)
DIRS_TO_CLEAN = [
    'models_genesis',
    'models',
    'data/parquet', # If exists
    'logs',
    'historical'
]

FILES_TO_DELETE = [
    'data/status.json',
    'data/trades.csv',
    'data/ledger.db', # If exists
    'dashboard/data/trades.csv',
    'dashboard/data/status.csv'
]

def purge_legacy():
    print("🧹 [PURGE] Starting Institutional Clean-up...")
    
    # 1. Delete Models & Data
    for d in DIRS_TO_CLEAN:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                print(f"✅ Deleted Directory: {d}")
            except Exception as e:
                print(f"❌ Failed to delete {d}: {e}")
                
    # 2. Delete Files
    for f in FILES_TO_DELETE:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"✅ Deleted File: {f}")
            except Exception as e:
                print(f"❌ Failed to delete {f}: {e}")
                
    # 3. Recursive __pycache__
    print("🧹 Cleaning __pycache__...")
    for root, dirs, files in os.walk("."):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                
    print("✨ SYSTEM TABULA RASA CONFIRMED.")

if __name__ == "__main__":
    purge_legacy()
