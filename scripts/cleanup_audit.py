import os
import shutil
import glob
from pathlib import Path

def cleanup_audit_files():
    """
    Moves temporary audit files to 'archive/' directory.
    Deletes truly temporary binaries or logs.
    """
    root_dir = Path(__file__).parent.parent
    archive_dir = root_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Patterns to match and move
    patterns_to_archive = [
        "audit_*.txt",
        "benchmark_*.txt",
        "test_results_*.txt",
        "verify_*.py",
        "check_*.py",
        "debug_*.py",
        "reproduce_*.py",
        "simulate_*.py",
        "bot_*.log"
    ]
    
    moved_count = 0
    
    print(f"🧹 Starting cleanup in: {root_dir}")
    print(f"📂 Archive target: {archive_dir}")
    
    for pattern in patterns_to_archive:
        for file_path in root_dir.glob(pattern):
            if file_path.name == "debug_tracer.py": # Exception: Keep utility
                continue
                
            try:
                # Move file
                target = archive_dir / file_path.name
                if target.exists():
                    os.remove(target) # Overwrite existing in archive
                shutil.move(str(file_path), str(target))
                print(f"📦 Archived: {file_path.name}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Failed to archive {file_path.name}: {e}")
                
    # Specific DELETE list (Temp CSVs)
    patterns_to_delete = [
        "tests/temp_*.csv"
    ]
    
    deleted_count = 0
    for pattern in patterns_to_delete:
        # glob supports ** for recursive? standard glob module does. Path.glob does too.
        # But "tests/temp_*.csv" is relative path. Path.glob takes pattern.
        # If pattern has slash, use rglob or simple glob on subdir.
        
        # Let's use simple string glob for deletion safety
        full_pattern = str(root_dir / pattern)
        for file_path_str in glob.glob(full_pattern):
            try:
                os.remove(file_path_str)
                print(f"🗑️ Deleted: {os.path.basename(file_path_str)}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {os.path.basename(file_path_str)}: {e}")

    print(f"✨ Cleanup Complete. Archived: {moved_count}, Deleted: {deleted_count}")

if __name__ == "__main__":
    cleanup_audit_files()
