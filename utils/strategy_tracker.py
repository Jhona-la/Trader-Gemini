"""
🏆 Strategy Performance Tracker — FORENSIC-V16
═══════════════════════════════════════════════════════════════
QUÉ: Sistema de tracking granular de performance por estrategia.
POR QUÉ: El sistema anterior (StrategySelector) tenía IDs hardcoded que
   no coincidían con los strategy_ids reales, haciendo que todos los pesos
   fueran siempre neutrales (1/N).
PARA QUÉ: Tracking real de cada estrategia por día/semana/mes/año y por
   moneda, con ranking evolutivo que muta pesos basado en rendimiento REAL.
CÓMO: Cada trade se registra con strategy_id + symbol + horizon + timestamp.
   El sistema calcula rolling metrics y genera rankings periódicos.
CUÁNDO: Se actualiza en cada fill event (trade cerrado).
DÓNDE: utils/strategy_tracker.py (nuevo módulo)
QUIÉN: StrategyTracker → llamado desde Portfolio/Engine
═══════════════════════════════════════════════════════════════
"""

import time
import uuid
import threading
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils.logger import logger


@dataclass
class TradeRecord:
    """Registro atómico de un trade completado."""
    trade_id: str                              # UUID único
    strategy_id: str                           # ID de la estrategia
    symbol: str                                # Par trading
    horizon: str                               # SCALPING / SWING
    direction: str                             # LONG / SHORT
    entry_time: float                          # Unix timestamp
    exit_time: float = 0.0                     # Unix timestamp
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    gross_pnl: float = 0.0                     # PnL bruto (sin fees)
    net_pnl: float = 0.0                       # PnL neto (con fees)
    fees: float = 0.0
    pnl_pct: float = 0.0                       # PnL como porcentaje
    is_win: bool = False
    exit_reason: str = ""                      # TP/SL/TRAILING/MANUAL/KILL_SWITCH
    
    @property
    def duration_seconds(self) -> float:
        if self.exit_time > 0:
            return self.exit_time - self.entry_time
        return time.time() - self.entry_time


@dataclass 
class StrategyMetrics:
    """Métricas agregadas de una estrategia."""
    strategy_id: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_duration_seconds: float = 0.0
    last_updated: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    consecutive_wins: int = 0
    max_consecutive_wins: int = 0
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0
    
    @property
    def profit_factor(self) -> float:
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss != 0 else float('inf')
    
    @property  
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        avg_win = self.gross_profit / self.wins if self.wins > 0 else 0.0
        avg_loss = abs(self.gross_loss) / self.losses if self.losses > 0 else 0.0
        wr = self.win_rate
        return (wr * avg_win) - ((1 - wr) * avg_loss)
    
    @property
    def score(self) -> float:
        """
        Composite score para ranking.
        Combina Win Rate, Profit Factor, y penaliza rachas de pérdidas.
        """
        if self.total_trades < 3:
            return 0.50  # Neutral para estrategias nuevas
        
        wr_score = self.win_rate * 0.35
        pf_score = min(self.profit_factor / 3.0, 1.0) * 0.35
        exp_score = max(0, min(self.expectancy * 100, 1.0)) * 0.20
        streak_penalty = max(0, 1.0 - (self.consecutive_losses * 0.15)) * 0.10
        
        return float(wr_score + pf_score + exp_score + streak_penalty)


class StrategyTracker:
    """
    Sistema de tracking de performance por estrategia con ventanas temporales.
    
    Características:
    - Tracking por strategy_id + symbol + horizon
    - Métricas rolling: día, semana, mes, all-time
    - Ranking evolutivo con score compuesto
    - Thread-safe (singleton)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # All completed trades
        self.trades: List[TradeRecord] = []
        
        # Aggregated metrics by strategy_id
        self.all_time: Dict[str, StrategyMetrics] = {}
        
        # Aggregated metrics by (strategy_id, symbol)
        self.by_symbol: Dict[str, Dict[str, StrategyMetrics]] = defaultdict(dict)
        
        # Aggregated metrics by (strategy_id, horizon)
        self.by_horizon: Dict[str, Dict[str, StrategyMetrics]] = defaultdict(dict)
        
        # Rankings cache
        self._rankings: Dict[str, float] = {}
        self._last_ranking_update: float = 0
        
        # Thread safety
        self._state_lock = threading.RLock()
        
        self._initialized = True
        logger.info("🏆 [StrategyTracker] Initialized — Tracking enabled for all strategies")
    
    def record_trade(self, 
                     strategy_id: str,
                     symbol: str,
                     horizon: str,
                     direction: str,
                     entry_price: float,
                     exit_price: float,
                     quantity: float,
                     gross_pnl: float,
                     net_pnl: float,
                     fees: float,
                     entry_time: float,
                     exit_time: float,
                     exit_reason: str = "",
                     setup_type: str = "UNKNOWN",
                     strategy_version: str = "1.0.0") -> str:
        """
        Registra un trade completado.
        
        Returns:
            trade_id (str): UUID del trade registrado
        """
        with self._state_lock:
            trade_id = str(uuid.uuid4())[:8]
            
            pnl_pct = 0.0
            notional = entry_price * quantity
            if notional > 0:
                pnl_pct = net_pnl / notional
            
            record = TradeRecord(
                trade_id=trade_id,
                strategy_id=strategy_id,
                symbol=symbol,
                horizon=horizon,
                direction=direction,
                entry_time=entry_time,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                net_pnl=net_pnl,
                fees=fees,
                pnl_pct=pnl_pct,
                is_win=net_pnl > 0,
                exit_reason=exit_reason
            )
            
            self.trades.append(record)
            
            # Update all-time metrics
            self._update_metrics(self.all_time, strategy_id, record)
            
            # Update by-symbol metrics
            if strategy_id not in self.by_symbol:
                self.by_symbol[strategy_id] = {}
            self._update_metrics(self.by_symbol[strategy_id], symbol, record)
            
            # Update by-horizon metrics
            if strategy_id not in self.by_horizon:
                self.by_horizon[strategy_id] = {}
            self._update_metrics(self.by_horizon[strategy_id], horizon, record)
            
            logger.info(
                f"🏆 [Tracker] {strategy_id} | {symbol} | {horizon} | "
                f"{'WIN' if record.is_win else 'LOSS'} | "
                f"PnL: ${net_pnl:.4f} ({pnl_pct*100:.2f}%) | "
                f"WR: {self.all_time[strategy_id].win_rate*100:.1f}% | "
                f"Score: {self.all_time[strategy_id].score:.3f}"
            )
            
            return trade_id
    
    def _update_metrics(self, metrics_dict: Dict, key: str, trade: TradeRecord):
        """Actualiza métricas agregadas para una llave dada."""
        if key not in metrics_dict:
            metrics_dict[key] = StrategyMetrics(strategy_id=key)
        
        m = metrics_dict[key]
        m.total_trades += 1
        m.total_pnl += trade.net_pnl
        m.total_fees += trade.fees
        m.last_updated = time.time()
        
        # Duration tracking
        dur = trade.duration_seconds
        m.avg_duration_seconds = (
            (m.avg_duration_seconds * (m.total_trades - 1) + dur) / m.total_trades
        )
        
        if trade.is_win:
            m.wins += 1
            m.gross_profit += trade.net_pnl
            m.largest_win = max(m.largest_win, trade.net_pnl)
            m.consecutive_wins += 1
            m.consecutive_losses = 0
            m.max_consecutive_wins = max(m.max_consecutive_wins, m.consecutive_wins)
        else:
            m.losses += 1
            m.gross_loss += trade.net_pnl  # Negative value
            m.largest_loss = min(m.largest_loss, trade.net_pnl)
            m.consecutive_losses += 1
            m.consecutive_wins = 0
            m.max_consecutive_losses = max(m.max_consecutive_losses, m.consecutive_losses)
    
    def get_rankings(self, force_update: bool = False) -> Dict[str, float]:
        """
        Retorna ranking de estrategias ordenado por score (mayor = mejor).
        Se cachea por 60 segundos para performance.
        """
        with self._state_lock:
            now = time.time()
            if not force_update and (now - self._last_ranking_update) < 60:
                return dict(self._rankings)
            
            rankings = {}
            for strat_id, metrics in self.all_time.items():
                rankings[strat_id] = metrics.score
            
            # Sort by score descending
            self._rankings = dict(sorted(rankings.items(), key=lambda x: x[1], reverse=True))
            self._last_ranking_update = now
            
            return dict(self._rankings)
    
    def get_metrics_for_window(self, strategy_id: str, 
                                window_days: int = 7) -> Optional[StrategyMetrics]:
        """
        Calcula métricas para una ventana temporal específica.
        
        Args:
            strategy_id: ID de la estrategia
            window_days: 1=día, 7=semana, 30=mes, 365=año
        """
        with self._state_lock:
            cutoff = time.time() - (window_days * 86400)
            window_trades = [
                t for t in self.trades 
                if t.strategy_id == strategy_id and t.exit_time >= cutoff
            ]
            
            if not window_trades:
                return None
            
            metrics = StrategyMetrics(strategy_id=strategy_id)
            for trade in window_trades:
                self._update_metrics({strategy_id: metrics}, strategy_id, trade)
            
            return metrics
    
    def get_best_strategy_for_symbol(self, symbol: str) -> Optional[str]:
        """Retorna el strategy_id con mejor score para un símbolo específico."""
        with self._state_lock:
            best_id = None
            best_score = -1.0
            
            for strat_id, symbol_metrics in self.by_symbol.items():
                if symbol in symbol_metrics:
                    m = symbol_metrics[symbol]
                    if m.total_trades >= 3 and m.score > best_score:
                        best_score = m.score
                        best_id = strat_id
            
            return best_id
    
    def get_multiplier_for_strategy(self, strategy_id: str) -> float:
        """
        Retorna un multiplicador de sizing basado en el rendimiento de la estrategia.
        
        Rango: 0.5x (mala performance) a 1.5x (excelente performance)
        Default: 1.0x (nueva o sin datos)
        """
        with self._state_lock:
            if strategy_id not in self.all_time:
                return 1.0
            
            m = self.all_time[strategy_id]
            if m.total_trades < 5:
                return 1.0
            
            # Score range: 0.0 → 1.0
            # Map to multiplier: 0.5 → 1.5
            return 0.5 + m.score
    
    def get_dashboard_data(self) -> Dict:
        """Exporta datos para el dashboard."""
        with self._state_lock:
            rankings = self.get_rankings()
            
            strategies = []
            for strat_id, score in rankings.items():
                m = self.all_time[strat_id]
                if not m:
                    continue
                    
                strategies.append({
                    'strategy_id': strat_id,
                    'score': round(score, 3),
                    'total_trades': m.total_trades,
                    'win_rate': round(m.win_rate * 100, 1),
                    'profit_factor': round(m.profit_factor, 2) if m.profit_factor != float('inf') else 999.0,
                    'total_pnl': round(m.total_pnl, 4),
                    'expectancy': round(m.expectancy, 4),
                    'avg_duration_min': round(m.avg_duration_seconds / 60, 1),
                    'max_consecutive_losses': m.max_consecutive_losses,
                    'max_consecutive_wins': m.max_consecutive_wins,
                })
            
            # Window metrics for top strategies
            windows = {}
            for strat_id in list(rankings.keys())[:5]:
                windows[strat_id] = {
                    'day': self._metrics_to_dict(self.get_metrics_for_window(strat_id, 1)),
                    'week': self._metrics_to_dict(self.get_metrics_for_window(strat_id, 7)),
                    'month': self._metrics_to_dict(self.get_metrics_for_window(strat_id, 30)),
                }
            
            return {
                'rankings': strategies,
                'temporal': windows,
                'total_strategies_tracked': len(self.all_time),
                'total_trades_tracked': len(self.trades),
            }
    
    def _metrics_to_dict(self, m: Optional[StrategyMetrics]) -> Optional[Dict]:
        if not m:
            return None
        return {
            'trades': m.total_trades,
            'win_rate': round(m.win_rate * 100, 1),
            'pnl': round(m.total_pnl, 4),
            'score': round(m.score, 3),
        }
    
    def print_leaderboard(self):
        """Imprime tabla de ranking en consola."""
        rankings = self.get_rankings(force_update=True)
        
        if not rankings:
            print("📊 No strategy data yet.")
            return
        
        print("\n" + "═" * 80)
        print("🏆 STRATEGY LEADERBOARD")
        print("═" * 80)
        print(f"{'#':>3} {'Strategy':<30} {'Score':>7} {'WR%':>6} {'PF':>6} {'Trades':>7} {'PnL':>10}")
        print("-" * 80)
        
        for rank, (strat_id, score) in enumerate(rankings.items(), 1):
            m = self.all_time[strat_id]
            if not m:
                continue
            pf_str = f"{m.profit_factor:.2f}" if m.profit_factor < 999 else "∞"
            print(
                f"{rank:>3} {strat_id:<30} {score:>7.3f} "
                f"{m.win_rate*100:>5.1f}% {pf_str:>6} "
                f"{m.total_trades:>7} ${m.total_pnl:>9.4f}"
            )
        
        print("═" * 80 + "\n")


# Singleton instance
strategy_tracker = StrategyTracker()
