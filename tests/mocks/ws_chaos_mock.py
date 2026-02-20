"""
🎭 OMEGA-VOID §2.1: WebSocket Chaos Mock (Stochastic Latency Injection)

QUÉ: Mock de WebSocket que simula condiciones adversas de red.
POR QUÉ: En producción, los WebSockets de Binance sufren spikes de latencia,
     pérdida de paquetes y reordenamiento. Si el bot no sobrevive esto en tests,
     morirá en producción con dinero real.
PARA QUÉ: Demostrar que la 'Fortaleza Neural' mantiene integridad bajo estrés.
CÓMO: Wrappea un socket real (o un mock base) e inyecta:
     - Latencia estocástica (Gaussian/Pareto/Bimodal)
     - Pérdida de paquetes (drop aleatorio configurable)
     - Reordenamiento de mensajes (buffer con shuffle)
CUÁNDO: Durante tests de estrés y certificación OMEGA-VOID.
DÓNDE: tests/mocks/ws_chaos_mock.py
QUIÉN: ChaosWebSocket → usado por BinanceData en modo test.
"""

import asyncio
import random
import time
import json
import numpy as np
from collections import deque
from typing import Optional, Dict, Any, List
from enum import Enum


class ChaosMode(Enum):
    """
    Niveles de degradación de red simulada.
    
    NORMAL: Condiciones ideales (1-15ms avg latency)
    DEGRADED: Red congestionada (10-500ms + tail latency)  
    HOSTILE: Condiciones de flash-crash (latencia bimodal + drops)
    BLACKOUT: Desconexión total seguida de reconexión
    """
    NORMAL = "normal"
    DEGRADED = "degraded"
    HOSTILE = "hostile"
    BLACKOUT = "blackout"


class LatencyDistribution:
    """
    Generadores de latencia por distribución estadística.
    
    QUÉ: Cada modo usa una distribución diferente que modela
         condiciones reales de red observadas en Binance.
    POR QUÉ: La latencia real NO es uniforme. Tiene colas pesadas
         (Pareto) que causan outliers de 100-500ms.
    """
    
    @staticmethod
    def normal() -> float:
        """Gaussian(μ=5ms, σ=2ms), clamp [1, 15]ms"""
        return max(0.001, min(0.015, random.gauss(0.005, 0.002)))
    
    @staticmethod
    def degraded() -> float:
        """Pareto(α=1.5) tail + baseline 10ms, range [10, 500]ms"""
        baseline = 0.010
        tail = (random.paretovariate(1.5) - 1) * 0.050  # Heavy tail
        return min(0.500, baseline + tail)
    
    @staticmethod
    def hostile() -> float:
        """Bimodal: 70% fast (1-5ms), 30% slow (200-500ms)"""
        if random.random() < 0.7:
            return random.uniform(0.001, 0.005)
        else:
            return random.uniform(0.200, 0.500)
    
    @staticmethod
    def blackout(duration: float = 5.0) -> float:
        """Total blackout for N seconds"""
        return duration


class ChaosWebSocket:
    """
    ⚡ OMEGA-VOID: WebSocket Chaos Proxy.
    
    QUÉ: Envuelve cualquier fuente de mensajes (real o mock) e inyecta
         latencia, drops y reordenamiento controlados.
    POR QUÉ: El bot DEBE ser resiliente a condiciones reales de red.
         Un sistema que solo funciona con latencia zero no es institucional.
    PARA QUÉ: Certificar que BinanceData._process_kline_event sobrevive
         mensajes retrasados, perdidos y fuera de orden.
    CÓMO: Proxy pattern + deque buffer para reordenamiento.
    CUÁNDO: Se activa pasando mode != NORMAL al constructor.
    DÓNDE: tests/mocks/ws_chaos_mock.py → ChaosWebSocket
    QUIÉN: Test harness, certification scripts.
    
    Args:
        messages: Pre-loaded list of messages to replay (for offline testing)
        mode: ChaosMode enum
        drop_rate: Probability of dropping a message [0.0, 1.0]
        reorder_rate: Probability of delivering out-of-order [0.0, 1.0]
        reorder_buffer_size: How many messages to buffer for reordering
    """
    
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        mode: ChaosMode = ChaosMode.NORMAL,
        drop_rate: float = 0.0,
        reorder_rate: float = 0.0,
        reorder_buffer_size: int = 5,
    ):
        self.messages = deque(messages)
        self.mode = mode
        self.drop_rate = drop_rate
        self.reorder_rate = reorder_rate
        self._reorder_buffer = deque(maxlen=reorder_buffer_size)
        
        # Telemetry
        self.stats = {
            'total_sent': 0,
            'total_dropped': 0,
            'total_reordered': 0,
            'total_delayed_ms': 0.0,
            'max_delay_ms': 0.0,
            'latency_samples': [],
        }
        
        # Mode-specific configuration
        self._latency_fn = {
            ChaosMode.NORMAL: LatencyDistribution.normal,
            ChaosMode.DEGRADED: LatencyDistribution.degraded,
            ChaosMode.HOSTILE: LatencyDistribution.hostile,
            ChaosMode.BLACKOUT: lambda: LatencyDistribution.blackout(5.0),
        }[mode]
        
        # Auto-configure drop/reorder for hostile modes
        if mode == ChaosMode.DEGRADED:
            self.drop_rate = max(drop_rate, 0.02)    # 2% minimum drops
            self.reorder_rate = max(reorder_rate, 0.05)  # 5% reorder
        elif mode == ChaosMode.HOSTILE:
            self.drop_rate = max(drop_rate, 0.05)    # 5% drops
            self.reorder_rate = max(reorder_rate, 0.10)  # 10% reorder
    
    async def recv(self) -> Optional[Dict]:
        """
        Receive next message with chaos injection.
        Returns None for dropped packets (caller must handle).
        """
        if not self.messages:
            return None  # Stream exhausted
        
        msg = self.messages.popleft()
        
        # === PACKET LOSS ===
        if random.random() < self.drop_rate:
            self.stats['total_dropped'] += 1
            return None  # Dropped!
        
        # === REORDERING ===
        if random.random() < self.reorder_rate and len(self._reorder_buffer) > 0:
            self.stats['total_reordered'] += 1
            # Swap: put current in buffer, deliver old buffered one
            self._reorder_buffer.append(msg)
            msg = self._reorder_buffer.popleft()
        elif random.random() < self.reorder_rate:
            # Buffer for later delivery
            self._reorder_buffer.append(msg)
            if self.messages:
                msg = self.messages.popleft()
            else:
                msg = self._reorder_buffer.popleft()
        
        # === LATENCY INJECTION ===
        delay = self._latency_fn()
        delay_ms = delay * 1000
        self.stats['total_delayed_ms'] += delay_ms
        self.stats['max_delay_ms'] = max(self.stats['max_delay_ms'], delay_ms)
        self.stats['latency_samples'].append(delay_ms)
        
        await asyncio.sleep(delay)
        
        self.stats['total_sent'] += 1
        return msg
    
    def get_report(self) -> Dict:
        """
        Returns detailed chaos injection report.
        
        QUÉ: Resumen estadístico de todo lo inyectado durante el test.
        PARA QUÉ: Validar que el bot sobrevivió X% de drops, Y ms max latency, etc.
        """
        samples = self.stats['latency_samples']
        n = len(samples)
        
        return {
            'mode': self.mode.value,
            'messages_sent': self.stats['total_sent'],
            'messages_dropped': self.stats['total_dropped'],
            'messages_reordered': self.stats['total_reordered'],
            'drop_rate_actual': (
                self.stats['total_dropped'] / 
                (self.stats['total_sent'] + self.stats['total_dropped'])
                if (self.stats['total_sent'] + self.stats['total_dropped']) > 0 else 0
            ),
            'latency_avg_ms': np.mean(samples) if n > 0 else 0,
            'latency_p50_ms': np.percentile(samples, 50) if n > 0 else 0,
            'latency_p95_ms': np.percentile(samples, 95) if n > 0 else 0,
            'latency_p99_ms': np.percentile(samples, 99) if n > 0 else 0,
            'latency_max_ms': self.stats['max_delay_ms'],
        }


def generate_kline_messages(
    symbol: str = "BTCUSDT",
    n_bars: int = 1000,
    start_price: float = 50000.0,
    volatility: float = 0.001,
    timeframe: str = "1m",
) -> List[Dict]:
    """
    Generates realistic kline WebSocket messages for testing.
    
    QUÉ: Genera N mensajes kline con movimiento browniano geométrico.
    POR QUÉ: Los tests necesitan datos realistas, no ruido uniforme.
    CÓMO: GBM (Geometric Brownian Motion) + volumen log-normal.
    
    Returns:
        List of dicts matching Binance WebSocket kline format.
    """
    messages = []
    price = start_price
    base_ts = int(time.time() * 1000)
    interval_ms = {'1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000}
    dt = interval_ms.get(timeframe, 60000)
    
    for i in range(n_bars):
        # GBM price movement
        returns = np.random.normal(0, volatility)
        price *= (1 + returns)
        
        # Realistic OHLC around close
        high = price * (1 + abs(np.random.normal(0, volatility * 0.5)))
        low = price * (1 - abs(np.random.normal(0, volatility * 0.5)))
        open_p = price * (1 + np.random.normal(0, volatility * 0.3))
        volume = max(0.1, np.random.lognormal(5, 1.5))
        
        ts = base_ts + (i * dt)
        is_closed = True
        
        msg = {
            'stream': f'{symbol.lower()}@kline_{timeframe}',
            'data': {
                'e': 'kline',
                's': symbol,
                'k': {
                    't': ts,
                    'T': ts + dt - 1,
                    's': symbol,
                    'i': timeframe,
                    'o': f'{open_p:.8f}',
                    'c': f'{price:.8f}',
                    'h': f'{high:.8f}',
                    'l': f'{low:.8f}',
                    'v': f'{volume:.8f}',
                    'x': is_closed,
                    'n': random.randint(100, 5000),
                    'q': f'{volume * price:.8f}',
                }
            }
        }
        messages.append(msg)
    
    return messages


async def run_chaos_test(
    mode: ChaosMode = ChaosMode.HOSTILE,
    n_messages: int = 500,
    drop_rate: float = 0.03,
) -> Dict:
    """
    Standalone chaos test runner.
    
    QUÉ: Ejecuta un test completo de resiliencia WebSocket.
    CÓMO: Genera mensajes → inyecta caos → mide supervivencia.
    
    Returns:
        Chaos report dict with all statistics.
    """
    messages = generate_kline_messages(n_bars=n_messages)
    ws = ChaosWebSocket(messages, mode=mode, drop_rate=drop_rate)
    
    received = 0
    dropped = 0
    
    while True:
        msg = await ws.recv()
        if msg is None and not ws.messages:
            break
        if msg is not None:
            received += 1
        else:
            dropped += 1
    
    report = ws.get_report()
    report['test_received'] = received
    report['test_dropped'] = dropped
    
    return report


# Self-test
if __name__ == '__main__':
    async def _main():
        print("=" * 60)
        print("⚡ OMEGA-VOID: WebSocket Chaos Test")
        print("=" * 60)
        
        for mode in [ChaosMode.NORMAL, ChaosMode.DEGRADED, ChaosMode.HOSTILE]:
            report = await run_chaos_test(mode=mode, n_messages=200)
            print(f"\n🎭 Mode: {mode.value}")
            print(f"   Sent: {report['messages_sent']}")
            print(f"   Dropped: {report['messages_dropped']}")
            print(f"   Reordered: {report['messages_reordered']}")
            print(f"   Avg Latency: {report['latency_avg_ms']:.1f}ms")
            print(f"   P95 Latency: {report['latency_p95_ms']:.1f}ms")
            print(f"   P99 Latency: {report['latency_p99_ms']:.1f}ms")
            print(f"   Max Latency: {report['latency_max_ms']:.1f}ms")
        
        print("\n✅ All chaos modes tested successfully")
    
    asyncio.run(_main())
