"""
DATA CLEANUP UTILITY
====================
Archives large CSV files and cleans up dashboard data.
Run this once to archive the 1.2GB status.csv file.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


def archive_large_csv(file_path: str, max_size_mb: int = 100) -> bool:
    """
    Archive a CSV file if it exceeds max_size_mb.
    
    Args:
        file_path: Path to the CSV file
        max_size_mb: Maximum size in MB before archiving
    
    Returns:
        True if archived, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return False
    
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    
    print(f"📊 {file_path}: {size_mb:.1f} MB")
    
    if size_mb > max_size_mb:
        # Create archive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        archive_name = f"archive_{base_name.replace('.csv', '')}_{timestamp}.csv"
        archive_path = os.path.join(dir_name, archive_name)
        
        # Move to archive
        shutil.move(file_path, archive_path)
        print(f"✅ Archived: {archive_path}")
        
        # Create empty file with header
        with open(file_path, 'w') as f:
            f.write("timestamp,total_equity,cash,realized_pnl,unrealized_pnl,positions\n")
        print(f"✅ Created fresh: {file_path}")
        
        return True
    else:
        print(f"ℹ️  Size OK ({size_mb:.1f} MB < {max_size_mb} MB), no action needed")
        return False


def cleanup_dashboard_data(data_dir: str = "dashboard/data"):
    """
    Clean up all dashboard data directories.
    """
    print("=" * 60)
    print("🧹 DASHBOARD DATA CLEANUP UTILITY")
    print("=" * 60)
    
    # Directories to clean
    dirs_to_check = [
        data_dir,
        os.path.join(data_dir, "futures"),
        os.path.join(data_dir, "spot"),
    ]
    
    for dir_path in dirs_to_check:
        status_csv = os.path.join(dir_path, "status.csv")
        if os.path.exists(status_csv):
            print(f"\n📁 Checking: {dir_path}")
            archive_large_csv(status_csv, max_size_mb=100)
    
    print("\n" + "=" * 60)
    print("✅ Cleanup complete!")
    print("=" * 60)


def get_csv_line_count(file_path: str) -> int:
    """Get approximate line count without loading entire file."""
    if not os.path.exists(file_path):
        return 0
    
    count = 0
    with open(file_path, 'rb') as f:
        for _ in f:
            count += 1
    return count


def tail_csv(file_path: str, n_lines: int = 1000) -> list:
    """
    Read last N lines of a CSV efficiently.
    Uses reverse reading to avoid loading entire file.
    """
    if not os.path.exists(file_path):
        return []
    
    lines = []
    
    with open(file_path, 'rb') as f:
        # Go to end of file
        f.seek(0, 2)
        file_size = f.tell()
        
        if file_size == 0:
            return []
        
        # Read in chunks from the end
        chunk_size = 8192
        remaining = n_lines + 1  # +1 for header
        position = file_size
        buffer = b''
        
        while remaining > 0 and position > 0:
            # Calculate chunk to read
            read_size = min(chunk_size, position)
            position -= read_size
            f.seek(position)
            
            # Read and prepend to buffer
            buffer = f.read(read_size) + buffer
            
            # Count lines in buffer
            line_count = buffer.count(b'\n')
            if line_count >= remaining:
                break
        
        # Decode and split
        try:
            text = buffer.decode('utf-8')
            all_lines = text.strip().split('\n')
            
            # Return header + last n_lines
            if len(all_lines) <= n_lines:
                lines = all_lines
            else:
                # Get header (first line) and last n_lines
                header = all_lines[0]
                tail = all_lines[-n_lines:]
                lines = [header] + tail
                
        except UnicodeDecodeError:
            # Fallback: just return empty
            return []
    
    return lines


if __name__ == "__main__":
    cleanup_dashboard_data()
