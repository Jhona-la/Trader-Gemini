
import cProfile
import pstats
import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import main
import asyncio
from utils.logger import setup_logger

logger = setup_logger("Profiler")

async def run_profiler(duration=60):
    """
    🔬 PHASE 41: SYSTEM PROFILING
    Runs the bot for 'duration' seconds under cProfile.
    Generates 'gemini_profile.prof' for analysis (SnakeViz/Tuna).
    """
    logger.info(f"🔬 STARTING PROFILER (Duration: {duration}s)...")
    
    # We need to wrap main() to stop it after duration
    # This is tricky because main() has a while loop.
    # We'll use a timeout on the main task.
    
    try:
        await asyncio.wait_for(main(), timeout=duration)
    except asyncio.TimeoutError:
        logger.info("⏱️ Profiling Finished (Timeout Reached)")
    except Exception as e:
        logger.error(f"Profiler Error: {e}")

def profile_entry_point():
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        asyncio.run(run_profiler(30)) # Run for 30 seconds
    except KeyboardInterrupt:
        pass
    finally:
        profiler.disable()
        
        # Save Stats
        output_file = "gemini_profile.prof"
        profiler.dump_stats(output_file)
        
        # Print Summary
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        logger.info("="*80)
        logger.info("📊 PROFILING RESULTS (Top 20 Time-Consumers)")
        logger.info("="*80)
        stats.print_stats(20)
        logger.info(f"💾 Full profile saved to {output_file}")
        logger.info("💡 Run 'snakeviz gemini_profile.prof' to visualize.")

if __name__ == "__main__":
    profile_entry_point()
