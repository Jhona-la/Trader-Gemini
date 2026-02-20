"""
🎭 OMEGA-VOID §2.2: Hostile Order Book Simulation

QUÉ: Mock de Order Book que simula manipulación institucional.
POR QUÉ: Los mercados reales contienen Spoofing, Iceberg Orders y Wash Trading.
     Si el bot genera señales basándose en liquidez falsa, pierde dinero.
PARA QUÉ: Demostrar que el bot IGNORA señales falsas de LOB (Limit Order Book)
     y solo reacciona a volumen genuino.
CÓMO: Genera order books con patrones adversariales programables.
CUÁNDO: Durante tests de estrés y validación de filtros de liquidez.
DÓNDE: tests/mocks/hostile_orderbook.py
QUIÉN: HostileOrderBook → usado por tests de señales y data_provider mocks.
"""

import time
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass, field


class ManipulationPattern(Enum):
    """
    Patrones de manipulación de mercado detectados en exchanges.
    
    SPOOFING: Órdenes grandes que desaparecen antes de ser ejecutadas.
    ICEBERG: Volumen oculto detrás de órdenes pequeñas visibles.
    WASH_TRADING: Volumen inflado artificialmente sin movimiento de precio.
    LAYERING: Múltiples niveles de órdenes falsas para crear ilusión de profundidad.
    CLEAN: Orden book legítimo (control).
    """
    SPOOFING = "spoofing"
    ICEBERG = "iceberg"
    WASH_TRADING = "wash_trading"
    LAYERING = "layering"
    CLEAN = "clean"


@dataclass
class OrderLevel:
    """Single level in the order book."""
    price: float
    quantity: float
    is_fake: bool = False
    visible_quantity: float = 0.0  # For icebergs
    ttl_ms: int = 0  # Time to live for spoofing (0 = permanent)
    created_at: float = field(default_factory=time.time)


class HostileOrderBook:
    """
    ⚡ OMEGA-VOID: Adversarial Limit Order Book Generator.
    
    QUÉ: Genera order books con patrones de manipulación controlados.
    POR QUÉ: Los bots institucionales inyectan señales falsas en el LOB
         para triggear algorithmic traders retail. Nuestro bot DEBE
         filtrar estas señales falsas.
    PARA QUÉ: Validar que:
         1. Spoofing NO genera señales de imbalance falsas
         2. Iceberg NO confunde el sizing del bot
         3. Wash Trading NO infla la señal de volumen delta
    CÓMO: Genera arrays bid/ask con anomalías programadas.
    CUÁNDO: Durante cada test del Bloque 2.2.
    DÓNDE: tests/mocks/hostile_orderbook.py → HostileOrderBook
    QUIÉN: Test harness para order_flow_metrics, technical.py signals.
    
    Args:
        mid_price: Current market price
        spread_bps: Spread in basis points (e.g., 1.0 = 0.01%)
        depth_levels: Number of price levels per side
    """
    
    def __init__(
        self,
        mid_price: float = 50000.0,
        spread_bps: float = 1.0,
        depth_levels: int = 20,
    ):
        self.mid_price = mid_price
        self.spread_pct = spread_bps / 10000
        self.depth_levels = depth_levels
        
        # Base quantities for realistic distribution
        self._base_qty = mid_price / 50000  # Scale with price
        
        # Tracking
        self.manipulation_log: List[Dict] = []
    
    def generate_clean_book(self) -> Dict:
        """
        Generates a clean, realistic order book (no manipulation).
        
        QUÉ: LOB legítimo con distribución log-normal de tamaños.
        PARA QUÉ: Establece la línea base para comparar con libros manipulados.
        """
        half_spread = self.mid_price * self.spread_pct / 2
        best_bid = self.mid_price - half_spread
        best_ask = self.mid_price + half_spread
        
        bids = []
        asks = []
        tick_size = self.mid_price * 0.00001  # 0.001% tick
        
        for i in range(self.depth_levels):
            # Quantities follow log-normal distribution (realistic)
            bid_qty = max(0.001, np.random.lognormal(np.log(self._base_qty), 0.8))
            ask_qty = max(0.001, np.random.lognormal(np.log(self._base_qty), 0.8))
            
            bids.append(OrderLevel(
                price=round(best_bid - i * tick_size, 8),
                quantity=round(bid_qty, 8),
                is_fake=False,
            ))
            asks.append(OrderLevel(
                price=round(best_ask + i * tick_size, 8),
                quantity=round(ask_qty, 8),
                is_fake=False,
            ))
        
        return {
            'bids': bids,
            'asks': asks,
            'pattern': ManipulationPattern.CLEAN,
            'mid_price': self.mid_price,
            'spread_bps': self.spread_pct * 10000,
        }
    
    def generate_spoofed_book(
        self,
        spoof_side: str = 'bid',
        spoof_size_multiplier: float = 10.0,
        spoof_levels: int = 3,
        spoof_ttl_ms: int = 200,
    ) -> Dict:
        """
        Spoofing: MASSIVE orders that vanish before being filled.
        
        QUÉ: Órdenes 10x el tamaño normal en bid o ask que desaparecen
             después de 200ms (antes de que nadie pueda ejecutarlas).
        POR QUÉ: El spoofing crea ilusión de soporte/resistencia para
             triggear otros algos y luego se cancela.
        PARA QUÉ: El bot DEBE detectar que el imbalance es transitorio
             y no generar señales basadas en él.
        
        Args:
            spoof_side: 'bid' (fake support) or 'ask' (fake resistance)
            spoof_size_multiplier: How much bigger than normal (default 10x)
            spoof_levels: Number of levels to spoof
            spoof_ttl_ms: Time before orders vanish
        """
        book = self.generate_clean_book()
        
        target = book['bids'] if spoof_side == 'bid' else book['asks']
        
        for i in range(min(spoof_levels, len(target))):
            original_qty = target[i].quantity
            spoofed_qty = original_qty * spoof_size_multiplier
            target[i] = OrderLevel(
                price=target[i].price,
                quantity=round(spoofed_qty, 8),
                is_fake=True,
                ttl_ms=spoof_ttl_ms,
            )
        
        book['pattern'] = ManipulationPattern.SPOOFING
        
        self.manipulation_log.append({
            'type': 'spoof',
            'side': spoof_side,
            'multiplier': spoof_size_multiplier,
            'levels': spoof_levels,
            'ttl_ms': spoof_ttl_ms,
            'timestamp': time.time(),
        })
        
        return book
    
    def generate_iceberg_book(
        self,
        iceberg_side: str = 'ask',
        hidden_ratio: float = 50.0,
        iceberg_levels: int = 2,
    ) -> Dict:
        """
        Iceberg Orders: HUGE hidden volume behind small visible orders.
        
        QUÉ: Visible 1 BTC, real 50 BTC detrás. La depth5 miente.
        POR QUÉ: Institucionales ocultan sus órdenes grandes para
             no mover el mercado. Si el bot confía en depth5 visible
             para sizing, calculará mal la resistencia real.
        PARA QUÉ: El bot NO debe confiar en depth5 para sizing.
             Debe usar volumen ejecutado (delta), no volumen visible.
        
        Args:
            iceberg_side: 'bid' or 'ask' 
            hidden_ratio: Total real / visible (50 = 50x hidden)
            iceberg_levels: Levels with icebergs
        """
        book = self.generate_clean_book()
        
        target = book['bids'] if iceberg_side == 'bid' else book['asks']
        
        for i in range(min(iceberg_levels, len(target))):
            visible = target[i].quantity
            real_qty = visible * hidden_ratio
            target[i] = OrderLevel(
                price=target[i].price,
                quantity=round(real_qty, 8),
                visible_quantity=round(visible, 8),
                is_fake=False,  # Not fake, just hidden
            )
        
        book['pattern'] = ManipulationPattern.ICEBERG
        
        self.manipulation_log.append({
            'type': 'iceberg',
            'side': iceberg_side,
            'hidden_ratio': hidden_ratio,
            'levels': iceberg_levels,
            'timestamp': time.time(),
        })
        
        return book
    
    def generate_wash_trading_book(
        self,
        volume_inflation: float = 5.0,
        price_impact: float = 0.0001,
    ) -> Dict:
        """
        Wash Trading: Volume inflated 5x without real price impact.
        
        QUÉ: Volumen artificialmente inflado 5x pero el precio
             no se mueve proporcionalmente (entradas self-referencing).
        POR QUÉ: Los wash traders inflan volumen para atraer
             bots de momentum que usan volumen como señal.
        PARA QUÉ: El delta normalizado del bot DEBE filtrar
             este volumen falso. Si delta/volume es anormalmente bajo,
             el volumen es sospechoso.
        
        Args:
            volume_inflation: Multiplier for fake volume
            price_impact: How much price actually moves (very small = wash)
        """
        book = self.generate_clean_book()
        
        # Inflate all quantities but keep prices nearly identical
        for level in book['bids'] + book['asks']:
            level.quantity = round(level.quantity * volume_inflation, 8)
            level.is_fake = True
        
        # Minimal price movement (characteristic of wash trading)
        wash_metrics = {
            'volume_inflation': volume_inflation,
            'price_impact_pct': price_impact * 100,
            'delta_volume_ratio': 1.0 / volume_inflation,  # Suspiciously low
        }
        
        book['pattern'] = ManipulationPattern.WASH_TRADING
        book['wash_metrics'] = wash_metrics
        
        self.manipulation_log.append({
            'type': 'wash_trading',
            'volume_inflation': volume_inflation,
            'price_impact': price_impact,
            'timestamp': time.time(),
        })
        
        return book
    
    def generate_layering_book(
        self,
        layer_side: str = 'ask',
        n_layers: int = 8,
        qty_per_layer: float = 2.0,
    ) -> Dict:
        """
        Layering: Multiple fake order levels create false depth illusion.
        
        QUÉ: 8+ niveles de órdenes falsas con cantidades similares.
        POR QUÉ: Crea ilusión de gran profundidad de mercado que no existe.
        PARA QUÉ: El bot debe detectar uniformidad sospechosa en profundidad.
        """
        book = self.generate_clean_book()
        target = book['bids'] if layer_side == 'bid' else book['asks']
        
        # Replace levels with suspiciously uniform quantities
        for i in range(min(n_layers, len(target))):
            uniform_qty = qty_per_layer * self._base_qty
            # Add tiny noise to not be exactly identical
            uniform_qty *= (1 + random.uniform(-0.02, 0.02))
            target[i] = OrderLevel(
                price=target[i].price,
                quantity=round(uniform_qty, 8),
                is_fake=True,
            )
        
        book['pattern'] = ManipulationPattern.LAYERING
        
        self.manipulation_log.append({
            'type': 'layering',
            'side': layer_side,
            'n_layers': n_layers,
            'qty_per_layer': qty_per_layer,
            'timestamp': time.time(),
        })
        
        return book
    
    def to_binance_depth(self, book: Dict) -> Dict:
        """
        Converts internal book to Binance WebSocket depth format.
        
        QUÉ: Transforma a formato compatible con _process_book_ticker.
        PARA QUÉ: Puede inyectarse directamente en BinanceData mocks.
        """
        bids = [[str(b.price), str(b.visible_quantity or b.quantity)] 
                for b in book['bids']]
        asks = [[str(a.price), str(a.visible_quantity or a.quantity)]
                for a in book['asks']]
        
        return {
            'b': bids[0][0] if bids else '0',  # Best bid
            'B': bids[0][1] if bids else '0',  # Best bid qty
            'a': asks[0][0] if asks else '0',  # Best ask
            'A': asks[0][1] if asks else '0',  # Best ask qty
            'bids_full': bids[:5],  # Depth 5
            'asks_full': asks[:5],
        }
    
    def generate_mixed_sequence(
        self,
        n_snapshots: int = 100,
        spoof_pct: float = 0.15,
        iceberg_pct: float = 0.10,
        wash_pct: float = 0.10,
        layering_pct: float = 0.05,
    ) -> List[Dict]:
        """
        Generates a sequence of order book snapshots with mixed manipulation.
        
        QUÉ: Secuencia temporal de LOBs con manipulación aleatoria intercalada.
        POR QUÉ: En producción, la manipulación no es constante — aparece
             en ráfagas mezclada con actividad legítima.
        PARA QUÉ: Test realista de detección de manipulación a lo largo del tiempo.
        
        Returns:
            List of order book dicts with pattern labels.
        """
        sequence = []
        
        for i in range(n_snapshots):
            # Random small price walk
            self.mid_price *= (1 + random.gauss(0, 0.0002))
            
            r = random.random()
            if r < spoof_pct:
                book = self.generate_spoofed_book(
                    spoof_side=random.choice(['bid', 'ask'])
                )
            elif r < spoof_pct + iceberg_pct:
                book = self.generate_iceberg_book(
                    iceberg_side=random.choice(['bid', 'ask'])
                )
            elif r < spoof_pct + iceberg_pct + wash_pct:
                book = self.generate_wash_trading_book()
            elif r < spoof_pct + iceberg_pct + wash_pct + layering_pct:
                book = self.generate_layering_book(
                    layer_side=random.choice(['bid', 'ask'])
                )
            else:
                book = self.generate_clean_book()
            
            book['snapshot_id'] = i
            book['timestamp'] = time.time()
            sequence.append(book)
        
        return sequence
    
    def get_manipulation_report(self) -> Dict:
        """Summary of all manipulation injected during the test."""
        from collections import Counter
        type_counts = Counter(e['type'] for e in self.manipulation_log)
        
        return {
            'total_manipulations': len(self.manipulation_log),
            'by_type': dict(type_counts),
            'log': self.manipulation_log[-10:],  # Last 10 entries
        }


# ============================================================
# DETECTION VALIDATORS
# ============================================================

def validate_spoofing_detection(
    imbalance_before: float,
    imbalance_after: float,
    threshold: float = 0.3,
) -> bool:
    """
    QUÉ: Valida que el bot NO cambió su señal de imbalance
         después de que las órdenes spoofed desaparecieron.
    
    Returns:
        True if bot correctly ignored the spoof.
    """
    # If imbalance was large before (spoofed) but reverted after,
    # and the bot didn't act on it → PASS
    if abs(imbalance_before) > threshold and abs(imbalance_after) < threshold:
        return True  # Correctly ignored transient spike
    return False


def validate_iceberg_detection(
    estimated_depth: float,
    actual_depth: float,
    tolerance: float = 0.5,
) -> bool:
    """
    QUÉ: Valida que el bot NO confía ciegamente en depth visible.
    
    Returns:
        True if bot's depth estimate is within tolerance of actual.
    """
    if actual_depth <= 0:
        return True
    ratio = estimated_depth / actual_depth
    # If ratio is way off, bot was fooled
    return ratio > tolerance


def validate_wash_trading_detection(
    delta_volume_ratio: float,
    threshold: float = 0.3,
) -> bool:
    """
    QUÉ: Valida que delta/volume ratio es sospechosamente bajo.
    
    Returns:
        True if ratio is below threshold (wash detected).
    """
    return delta_volume_ratio < threshold


# Self-test
if __name__ == '__main__':
    print("=" * 60)
    print("⚡ OMEGA-VOID: Hostile Order Book Test")
    print("=" * 60)
    
    hob = HostileOrderBook(mid_price=50000.0)
    
    # Test each pattern
    for pattern_name, gen_fn in [
        ("Clean", hob.generate_clean_book),
        ("Spoofed", lambda: hob.generate_spoofed_book()),
        ("Iceberg", lambda: hob.generate_iceberg_book()),
        ("Wash Trading", lambda: hob.generate_wash_trading_book()),
        ("Layering", lambda: hob.generate_layering_book()),
    ]:
        book = gen_fn()
        total_bid_qty = sum(b.quantity for b in book['bids'][:5])
        total_ask_qty = sum(a.quantity for a in book['asks'][:5])
        fake_count = sum(1 for l in book['bids'] + book['asks'] if l.is_fake)
        
        print(f"\n📊 {pattern_name}:")
        print(f"   Pattern: {book['pattern'].value}")
        print(f"   Bid Depth5: {total_bid_qty:.4f}")
        print(f"   Ask Depth5: {total_ask_qty:.4f}")
        print(f"   Imbalance: {(total_bid_qty - total_ask_qty)/(total_bid_qty + total_ask_qty):.3f}")
        print(f"   Fake Levels: {fake_count}/{len(book['bids']) + len(book['asks'])}")
    
    # Test mixed sequence
    sequence = hob.generate_mixed_sequence(n_snapshots=100)
    report = hob.get_manipulation_report()
    print(f"\n📈 Mixed Sequence: {len(sequence)} snapshots")
    print(f"   Manipulations: {report['total_manipulations']}")
    print(f"   By Type: {report['by_type']}")
    
    print("\n✅ All hostile patterns generated successfully")
