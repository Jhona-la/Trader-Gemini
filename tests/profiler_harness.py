"""
🔍 OMEGA-VOID §4.2: CPU Bottleneck Profiler Harness

QUÉ: Wrapper de profiling de bajo nivel para identificar cuellos de botella.
POR QUÉ: Sin datos de profiling, optimizar es ciego. Necesitamos saber
     EXACTAMENTE qué función consume más tiempo en el Ryzen 7 5700U.
PARA QUÉ: Identificar top-10 funciones más lentas, medir wait-states
     de threads, y generar reporte actionable de optimización.
CÓMO: Combina cProfile (call graph), time.perf_counter_ns (nano-precision),
     y threading snapshots (thread wait detection).
CUÁNDO: Después de cada optimización para validar impacto.
DÓNDE: tests/profiler_harness.py
QUIÉN: ProfilerHarness → usa engine.py, technical.py como targets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cProfile
import pstats
import io
import time
import threading
import traceback
import json
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class TimingResult:
    """Result from a single timed operation."""
    name: str
    calls: int
    total_ns: int
    min_ns: int
    max_ns: int
    avg_ns: float
    
    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000
    
    @property
    def avg_ms(self) -> float:
        return self.avg_ns / 1_000_000
    
    @property
    def min_ms(self) -> float:
        return self.min_ns / 1_000_000
    
    @property
    def max_ms(self) -> float:
        return self.max_ns / 1_000_000


class NanoTimer:
    """
    ⚡ OMEGA-VOID: Nanosecond-precision timer for hot path profiling.
    
    QUÉ: Context manager que mide tiempo con precisión de nanosegundos.
    POR QUÉ: time.time() tiene resolución de ~15ms en Windows.
         time.perf_counter_ns() tiene resolución de ~100ns.
         Para medir funciones de <1ms, necesitamos nano-precision.
    CÓMO: time.perf_counter_ns() con warmup para evitar cold-cache effects.
    
    Usage:
        timer = NanoTimer("function_name")
        for _ in range(1000):
            with timer:
                my_function()
        print(timer.result())
    """
    
    def __init__(self, name: str):
        self.name = name
        self._times: List[int] = []
        self._start: int = 0
    
    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter_ns() - self._start
        self._times.append(elapsed)
    
    def result(self) -> TimingResult:
        """Returns aggregated timing results."""
        if not self._times:
            return TimingResult(self.name, 0, 0, 0, 0, 0.0)
        
        return TimingResult(
            name=self.name,
            calls=len(self._times),
            total_ns=sum(self._times),
            min_ns=min(self._times),
            max_ns=max(self._times),
            avg_ns=sum(self._times) / len(self._times),
        )
    
    def reset(self):
        self._times = []


class ThreadSnapshotProfiler:
    """
    ⚡ OMEGA-VOID: Thread Wait-State Detector.
    
    QUÉ: Toma snapshots de todos los threads para detectar wait-states.
    POR QUÉ: Si un thread está bloqueado (en lock, I/O, sleep),
         no usa CPU pero impide que otros procesen. En un sistema con
         8 cores (Ryzen 7), un thread bloqueado es 12.5% de throughput perdido.
    PARA QUÉ: Identificar qué threads están esperando y por qué.
    CÓMO: threading.enumerate() + sys._current_frames() para stack traces.
    """
    
    def __init__(self, interval_ms: int = 100, n_samples: int = 50):
        self.interval_seconds = interval_ms / 1000
        self.n_samples = n_samples
        self.snapshots: List[Dict] = []
    
    def capture(self) -> List[Dict]:
        """
        Takes N snapshots of all thread states.
        
        Returns:
            List of snapshot dicts with thread info.
        """
        self.snapshots = []
        
        for sample_idx in range(self.n_samples):
            frames = sys._current_frames()
            threads = threading.enumerate()
            
            snapshot = {
                'sample': sample_idx,
                'timestamp': time.perf_counter_ns(),
                'n_threads': len(threads),
                'threads': [],
            }
            
            for t in threads:
                frame = frames.get(t.ident)
                stack = []
                if frame:
                    stack = traceback.format_stack(frame)[-3:]  # Last 3 frames
                
                thread_info = {
                    'name': t.name,
                    'id': t.ident,
                    'daemon': t.daemon,
                    'alive': t.is_alive(),
                    'stack_top': stack[-1].strip() if stack else 'N/A',
                }
                
                # Detect wait-states from stack
                if stack:
                    top_frame = ''.join(stack).lower()
                    if 'lock.acquire' in top_frame or 'threading.py' in top_frame:
                        thread_info['state'] = 'WAITING_LOCK'
                    elif 'socket.recv' in top_frame or 'select.select' in top_frame:
                        thread_info['state'] = 'WAITING_IO'
                    elif 'time.sleep' in top_frame:
                        thread_info['state'] = 'SLEEPING'
                    elif 'queue.get' in top_frame:
                        thread_info['state'] = 'WAITING_QUEUE'
                    else:
                        thread_info['state'] = 'RUNNING'
                else:
                    thread_info['state'] = 'UNKNOWN'
                
                snapshot['threads'].append(thread_info)
            
            self.snapshots.append(snapshot)
            time.sleep(self.interval_seconds)
        
        return self.snapshots
    
    def analyze(self) -> Dict:
        """
        Analyzes captured snapshots for bottlenecks.
        
        Returns wait-state statistics per thread.
        """
        thread_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for snap in self.snapshots:
            for t in snap['threads']:
                thread_stats[t['name']][t['state']] += 1
        
        # Calculate percentages
        results = {}
        for thread_name, states in thread_stats.items():
            total = sum(states.values())
            results[thread_name] = {
                state: {
                    'count': count,
                    'pct': round(count / total * 100, 1) if total > 0 else 0,
                }
                for state, count in states.items()
            }
        
        # Find most blocked threads
        blocked_threads = []
        for name, states in results.items():
            running_pct = states.get('RUNNING', {}).get('pct', 0)
            if running_pct < 50:  # Less than 50% time running = bottleneck
                blocked_threads.append({
                    'thread': name,
                    'running_pct': running_pct,
                    'dominant_state': max(states, key=lambda s: states[s]['count']),
                })
        
        return {
            'total_samples': len(self.snapshots),
            'thread_states': results,
            'blocked_threads': sorted(blocked_threads, key=lambda x: x['running_pct']),
            'n_bottlenecks': len(blocked_threads),
        }


class ProfilerHarness:
    """
    ⚡ OMEGA-VOID: Master Profiling Harness.
    
    QUÉ: Combina cProfile + NanoTimer + ThreadSnapshot en un solo runner.
    POR QUÉ: Cada herramienta captura un aspecto diferente del rendimiento.
         cProfile: call graph (qué función se llama más).
         NanoTimer: latencia exacta del hot path.
         ThreadSnapshot: thread wait-states (dónde se bloquea).
    PARA QUÉ: Generar reporte completo con top-10 bottlenecks.
    CÓMO: Ejecuta target function con instrumentación y genera JSON report.
    CUÁNDO: Post-optimización para validar mejoras.
    DÓNDE: tests/profiler_harness.py → ProfilerHarness
    QUIÉN: Dev/QA.
    """
    
    def __init__(self):
        self.cprofile_stats: Optional[pstats.Stats] = None
        self.nano_timers: Dict[str, NanoTimer] = {}
        self.thread_profiler = ThreadSnapshotProfiler()
    
    def profile_function(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        n_runs: int = 100,
        label: str = "",
    ) -> Dict:
        """
        Profiles a function with cProfile + NanoTimer.
        
        Args:
            func: Function to profile
            args, kwargs: Arguments to pass
            n_runs: Number of repetitions for nano timer
            label: Human-readable name
            
        Returns:
            Profile results dict.
        """
        if kwargs is None:
            kwargs = {}
        
        label = label or func.__name__
        
        # 1. cProfile (full call graph)
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Extract stats
        stream = io.StringIO()
        ps = pstats.Stats(profiler, stream=stream)
        ps.sort_stats('cumulative')
        ps.print_stats(20)  # Top 20
        cprofile_output = stream.getvalue()
        
        # Get top-10 functions by cumulative time
        top_functions = []
        for key, value in sorted(
            ps.stats.items(), 
            key=lambda x: x[1][3],  # cumulative time
            reverse=True
        )[:10]:
            filename, line, name = key
            ncalls, totcalls, tottime, cumtime, callers = value
            top_functions.append({
                'function': name,
                'file': os.path.basename(filename),
                'line': line,
                'ncalls': totcalls,
                'tottime_ms': round(tottime * 1000, 3),
                'cumtime_ms': round(cumtime * 1000, 3),
            })
        
        # 2. NanoTimer (precise timing)
        timer = NanoTimer(label)
        
        # Warmup (2 runs)
        for _ in range(min(2, n_runs)):
            func(*args, **kwargs)
        
        # Timed runs
        for _ in range(n_runs):
            with timer:
                func(*args, **kwargs)
        
        timing = timer.result()
        
        return {
            'label': label,
            'cprofile_top10': top_functions,
            'cprofile_full': cprofile_output[:2000],  # Truncated
            'nano_timing': {
                'calls': timing.calls,
                'avg_ms': round(timing.avg_ms, 4),
                'min_ms': round(timing.min_ms, 4),
                'max_ms': round(timing.max_ms, 4),
                'total_ms': round(timing.total_ms, 4),
            },
        }
    
    def profile_threads(self, duration_ms: int = 5000) -> Dict:
        """
        Captures thread states for duration.
        
        Args:
            duration_ms: How long to sample (default 5 seconds)
        """
        n_samples = max(10, duration_ms // 100)
        self.thread_profiler = ThreadSnapshotProfiler(
            interval_ms=100, 
            n_samples=n_samples
        )
        self.thread_profiler.capture()
        return self.thread_profiler.analyze()
    
    def generate_report(
        self,
        function_profiles: List[Dict],
        thread_profile: Optional[Dict] = None,
    ) -> Dict:
        """Generates comprehensive profiling report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': {
                'python_version': sys.version,
                'platform': sys.platform,
            },
            'function_profiles': function_profiles,
            'thread_analysis': thread_profile,
        }
        
        # Overall bottleneck summary
        all_top = []
        for fp in function_profiles:
            for fn in fp.get('cprofile_top10', [])[:3]:
                all_top.append({
                    'context': fp['label'],
                    **fn,
                })
        
        report['overall_top10'] = sorted(
            all_top, 
            key=lambda x: x.get('cumtime_ms', 0),
            reverse=True,
        )[:10]
        
        return report


def run_profiler():
    """Standalone profiler run with example targets."""
    print("=" * 60)
    print("🔍 OMEGA-VOID §4.2: CPU Bottleneck Profiler")
    print("=" * 60)
    
    harness = ProfilerHarness()
    profiles = []
    
    # Profile 1: NumPy operations (simulate indicator calculations)
    print("\n   📊 Profiling: NumPy indicator calculations...")
    
    def simulate_indicator_calc():
        """Simulates calculate_indicators from technical.py"""
        n = 500
        prices = np.random.randn(n).cumsum() + 50000
        
        # EMA
        ema = np.zeros(n)
        alpha = 2.0 / 15
        ema[0] = prices[0]
        for i in range(1, n):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        # RSI
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Bollinger
        sma = np.convolve(prices, np.ones(20)/20, mode='valid')
        
        return ema, sma
    
    import numpy as np
    
    result = harness.profile_function(
        simulate_indicator_calc,
        n_runs=200,
        label="Indicator Calculations (500 bars)",
    )
    profiles.append(result)
    
    print(f"      Avg: {result['nano_timing']['avg_ms']:.4f}ms")
    print(f"      Min: {result['nano_timing']['min_ms']:.4f}ms")
    print(f"      Max: {result['nano_timing']['max_ms']:.4f}ms")
    
    # Profile 2: JSON serialization (simulate state saves)
    print("\n   📊 Profiling: State serialization...")
    
    def simulate_state_save():
        """Simulates AtomicStateManager.save_json_atomic"""
        data = {
            'balance': 13.0,
            'positions': {f'COIN{i}/USDT': {'qty': 0.001, 'pnl': 0.01} for i in range(20)},
            'metrics': {'sharpe': 1.5, 'drawdown': 0.01},
        }
        json_str = json.dumps(data, default=str)
        return json_str
    
    result = harness.profile_function(
        simulate_state_save,
        n_runs=500,
        label="State Serialization (JSON)",
    )
    profiles.append(result)
    
    print(f"      Avg: {result['nano_timing']['avg_ms']:.4f}ms")
    
    # Profile 3: Thread snapshot
    print("\n   📊 Profiling: Thread states (2s capture)...")
    thread_result = harness.profile_threads(duration_ms=2000)
    
    print(f"      Active threads: {len(thread_result.get('thread_states', {}))}")
    print(f"      Bottlenecks: {thread_result.get('n_bottlenecks', 0)}")
    
    for bt in thread_result.get('blocked_threads', [])[:5]:
        print(f"      ⚠️ {bt['thread']}: {bt['running_pct']:.0f}% running "
              f"(mostly {bt['dominant_state']})")
    
    # Generate report
    report = harness.generate_report(profiles, thread_result)
    
    print(f"\n   🔝 Overall Top-5 Hottest Functions:")
    for i, fn in enumerate(report.get('overall_top10', [])[:5]):
        print(f"      {i+1}. {fn['function']} ({fn['file']}:{fn['line']}) "
              f"— {fn['cumtime_ms']:.3f}ms × {fn['ncalls']} calls")
    
    # Save
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'profiler_report.json'
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n   📄 Report saved: {report_path}")
    
    print("\n✅ Profiling complete")
    return report


if __name__ == '__main__':
    run_profiler()
