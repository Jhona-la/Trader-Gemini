"""
🔬 PHASE 42: MEMORY LEAK HUNT
Uses tracemalloc to identify potential leaks in OrderBook, Buffers, and Queues.
"""
import tracemalloc
import time
import asyncio
import os
import sys

# Ensure root directory is in path
sys.path.insert(0, os.getcwd())

from main import main as run_bot
from utils.logger import logger

async def run_memory_audit(duration_sec=60):
    logger.info("🕵️ Starting Memory Leak Audit...")
    tracemalloc.start()
    
    # Snapshot 1
    snapshot1 = tracemalloc.take_snapshot()
    
    # Run bot for X seconds
    try:
        # We wrap the main bot in a task with a timeout
        task = asyncio.create_task(run_bot())
        logger.info(f"⏳ Running system for {duration_sec} seconds...")
        await asyncio.sleep(duration_sec)
        
        # Stop bot (if possible gracefully)
        # Note: run_bot usually has its own loop, we may need to signal it to stop
        # For audit purposes, we can just take the snapshot now
    except Exception as e:
        logger.warning(f"Audit run interrupted: {e}")
    
    # Snapshot 2
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print("\n" + "="*50)
    print("🧠 MEMORY AUDIT: TOP 10 INCREASES")
    print("="*50)
    for stat in top_stats[:10]:
        print(stat)
    print("="*50 + "\n")
    
    # Check specifically for OrderBook and Buffers
    logger.info("Checking specific modules...")
    for stat in top_stats:
        if 'core/orderbook' in str(stat) or 'numba' in str(stat):
            print(f"🚩 Potential concern: {stat}")

if __name__ == "__main__":
    asyncio.run(run_memory_audit())
