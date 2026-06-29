import asyncio
import numpy as np
import time
import sys
import os
import threading
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.engine import Engine, AsyncBoundedQueue
from core.events import MarketEvent
from utils.rl_buffer import NumbaExperienceBuffer
from core.online_learning import OnlineLearner
from core.gc_tuner import GCTuner
from core.rate_limiter import PredictiveRateLimiter

class MockStrategy:
    def __init__(self, symbol):
        self.symbol = symbol
        self.calculate_signals_count = 0
        self.market_context = None

    def calculate_signals(self, event):
        self.calculate_signals_count += 1
        
    def stop(self):
        pass

async def run_extreme_stress_test():
    print("\n🔥 Starting Extreme Load Stress Test (Metal-Core Omega)...")
    
    # 1. Setup Engine & Strategies
    # Increase queue to avoid dropping events during the burst
    engine = Engine(events_queue=AsyncBoundedQueue(maxsize=100000))
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
    strategies = [MockStrategy(s) for s in symbols]
    for s in strategies:
        engine.register_strategy(s)
    
    # 2. Setup Shared Components
    buffer_capacity = 50000
    state_dim = 25
    rl_buffer = NumbaExperienceBuffer(buffer_capacity, state_dim)
    learner = OnlineLearner(learning_rate=0.01)
    weights = np.zeros(state_dim * 4, dtype=np.float32)
    
    # 3. Burst Configuration
    n_events = 2000
    n_learning_threads = 2
    learning_ops_per_thread = 1000
    
    print(f"   > Scenario: {n_events} Market Events + {n_learning_threads * learning_ops_per_thread} Concurrent Learner Ops")
    
    # 4. Define Concurrent Learning (Threaded, using Spinlocks in Numba)
    def learning_worker():
        for i in range(learning_ops_per_thread):
            s = np.random.random(state_dim).astype(np.float32)
            a = 1.0
            r = 1.0
            sn = np.random.random(state_dim).astype(np.float32)
            
            # Push to buffer (Spin-locked)
            rl_buffer.push(s, a, r, sn)
            
            # Perform single-step learning if buffer has enough data
            if i % 2 == 0:
                learner.learn_single(weights, s, a, r, sn)
            
            # PHASE 49: Prevent total CPU starvation by yielding occasionally
            if i % 100 == 0:
                time.sleep(0.001)

    
    print(f"   > Scenario: {n_events} Market Events + {n_learning_threads * learning_ops_per_thread} Concurrent Learner Ops")
    
    # 4. Define Concurrent Learning (Threaded, using Spinlocks in Numba)
    def learning_worker():
        for _ in range(learning_ops_per_thread):
            s = np.random.random(state_dim).astype(np.float32)
            a = 1.0
            r = 1.0
            sn = np.random.random(state_dim).astype(np.float32)
            
            # Push to buffer (Spin-locked)
            rl_buffer.push(s, a, r, sn)
            
            # Perform single-step learning if buffer has enough data
            if len(rl_buffer) > 10:
                learner.learn_single(weights, s, a, r, sn)

    # 5. Start Engine Loop
    engine_task = asyncio.create_task(engine.start())
    
    # 6. Start Learning Threads
    learn_threads = [threading.Thread(target=learning_worker) for _ in range(n_learning_threads)]
    
    start_time = time.perf_counter()
    
    for t in learn_threads:
        t.start()
        
    # 7. Rapid Event Injection (Non-blocking)
    print("   > Injecting Burst...")
    for i in range(n_events):
        event = MarketEvent(
            symbol=symbols[i % len(symbols)],
            close_price=50000.0 + i,
            timestamp_ns=time.time_ns()
        )
        await engine.events.put(event)
        
    # 8. Async Wait for Threads (DO NOT BLOCK EVENT LOOP)
    print("   > Waiting for Learning Threads...")
    while any(t.is_alive() for t in learn_threads):
        await asyncio.sleep(0.1)
        
    # 9. Wait for Engine to drain queue
    print("   > Draining Queue...")
    while not engine.events.empty():
        await asyncio.sleep(0.1)
        
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # 10. Shutdown
    engine.stop()
    await engine_task
    
    # 11. Results
    total_ops = n_events + (n_learning_threads * learning_ops_per_thread)
    throughput = total_ops / duration
    
    print("\n📊 STRESS TEST RESULTS:")
    print(f"   > Total Duration: {duration:.2f}s")
    print(f"   > Total Operations: {total_ops}")
    print(f"   > Throughput: {throughput:.2f} ops/sec")
    print(f"   > Processed Events: {engine.metrics['processed_events']}")
    print(f"   > RL Buffer Final Size: {len(rl_buffer)}")
    
    # Assertions
    if engine.metrics['processed_events'] < n_events:
        print(f"❌ WARNING: {n_events - engine.metrics['processed_events']} events dropped!")
    else:
        print("✅ SUCCESS: All events processed.")
        
    if len(rl_buffer) != (n_learning_threads * learning_ops_per_thread):
        # Could be slightly different if capacity exceeded (it wraps)
        if (n_learning_threads * learning_ops_per_thread) <= buffer_capacity:
             print(f"❌ WARNING: RL Buffer Mismatch! Expected {n_learning_threads * learning_ops_per_thread}, Got {len(rl_buffer)}")
        else:
             print(f"✅ RL Buffer wrapped as expected (Capacity {buffer_capacity}).")
    else:
        print("✅ RL Buffer size consistent.")

    print("🏁 Stress Test Complete.")

if __name__ == "__main__":
    asyncio.run(run_extreme_stress_test())
