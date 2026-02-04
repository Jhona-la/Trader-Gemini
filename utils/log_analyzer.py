"""
META-LEARNING LOG ANALYZER
==========================
Tracks trade outcomes and adjusts indicator weights adaptively.
Learns from failures to reduce false signal sources.

Features:
- Trade outcome logging
- Adaptive weight system (penalize false signals)
- Decision audit trail
- Performance metrics by indicator
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from config import Config
from utils.logger import logger


@dataclass
class TradeDecision:
    """A recorded trade decision with full context."""
    timestamp: str
    symbol: str
    direction: str  # LONG or SHORT
    entry_price: float
    target_price: float
    stop_price: float
    leverage: int
    indicators_used: List[str]
    indicator_values: Dict[str, float]
    confluence_score: int
    outcome: Optional[str] = None  # WIN, LOSS, or None (pending)
    exit_price: Optional[float] = None
    pnl_pct: Optional[float] = None


class MetaLearner:
    """
    Tracks indicator performance and adjusts weights adaptively.
    
    Learning Rule:
    - WIN: Increase weight of indicators that signaled correctly (+10%)
    - LOSS: Decrease weight of indicators that gave false signal (-15%)
    
    Weight bounds: [0.1, 2.0] to prevent any indicator from dominating.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(Config.DATA_DIR, "meta_learning.db")
        
        self.db_path = db_path
        
        # Default weights (equal)
        self.weights = {
            'rsi_divergence': 1.0,
            'macd_cross': 1.0,
            'bb_rejection': 1.0,
            'orderbook': 1.0,
            'volume_anomaly': 1.0
        }
        
        # Learning rates
        self.win_boost = 0.10    # +10% on win
        self.loss_penalty = 0.15 # -15% on loss (penalize more)
        
        # Min/max bounds
        self.min_weight = 0.1
        self.max_weight = 2.0
        
        # Initialize database
        self._init_db()
        
        # Load existing weights
        self._load_weights()
        
        logger.info(f"🧠 META-LEARNER INITIALIZED")
        logger.info(f"   Weights: {self.weights}")
    
    def _init_db(self):
        """Initialize SQLite database for persistence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Weights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS indicator_weights (
                indicator TEXT PRIMARY KEY,
                weight REAL,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                last_updated TEXT
            )
        ''')
        
        # Trade decisions table (audit trail)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                target_price REAL,
                stop_price REAL,
                leverage INTEGER,
                indicators_used TEXT,
                indicator_values TEXT,
                confluence_score INTEGER,
                outcome TEXT,
                exit_price REAL,
                pnl_pct REAL
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                date TEXT PRIMARY KEY,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                total_pnl_pct REAL,
                best_indicator TEXT,
                worst_indicator TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_weights(self):
        """Load weights from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT indicator, weight FROM indicator_weights')
        rows = cursor.fetchall()
        
        for indicator, weight in rows:
            if indicator in self.weights:
                self.weights[indicator] = weight
        
        conn.close()
    
    def _save_weights(self):
        """Save weights to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        for indicator, weight in self.weights.items():
            cursor.execute('''
                INSERT OR REPLACE INTO indicator_weights (indicator, weight, last_updated)
                VALUES (?, ?, ?)
            ''', (indicator, weight, now))
        
        conn.commit()
        conn.close()
    
    def record_trade_decision(self, decision: TradeDecision) -> int:
        """
        Record a new trade decision.
        Returns the decision ID for later outcome update.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trade_decisions 
            (timestamp, symbol, direction, entry_price, target_price, stop_price,
             leverage, indicators_used, indicator_values, confluence_score,
             outcome, exit_price, pnl_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision.timestamp,
            decision.symbol,
            decision.direction,
            decision.entry_price,
            decision.target_price,
            decision.stop_price,
            decision.leverage,
            json.dumps(decision.indicators_used),
            json.dumps(decision.indicator_values),
            decision.confluence_score,
            decision.outcome,
            decision.exit_price,
            decision.pnl_pct
        ))
        
        decision_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"Recorded trade decision #{decision_id}: {decision.symbol} {decision.direction}")
        return decision_id
    
    def update_trade_outcome(self, decision_id: int, outcome: str, 
                              exit_price: float, pnl_pct: float):
        """
        Update trade outcome and adjust weights.
        
        Args:
            decision_id: ID from record_trade_decision
            outcome: 'WIN' or 'LOSS'
            exit_price: Actual exit price
            pnl_pct: Realized PnL percentage
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update the decision record
        cursor.execute('''
            UPDATE trade_decisions
            SET outcome = ?, exit_price = ?, pnl_pct = ?
            WHERE id = ?
        ''', (outcome, exit_price, pnl_pct, decision_id))
        
        # Get the indicators used in this decision
        cursor.execute('''
            SELECT indicators_used FROM trade_decisions WHERE id = ?
        ''', (decision_id,))
        
        row = cursor.fetchone()
        if row:
            indicators_used = json.loads(row[0])
            self._adjust_weights(indicators_used, outcome)
        
        conn.commit()
        conn.close()
        
        logger.info(f"📊 Trade #{decision_id} outcome: {outcome} ({pnl_pct:+.2%})")
        logger.info(f"   Updated weights: {self.weights}")
    
    def _adjust_weights(self, indicators_used: List[str], outcome: str):
        """Adjust weights based on outcome."""
        adjustment = self.win_boost if outcome == 'WIN' else -self.loss_penalty
        
        for indicator in indicators_used:
            if indicator in self.weights:
                new_weight = self.weights[indicator] + adjustment
                # Clamp to bounds
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                self.weights[indicator] = new_weight
        
        self._save_weights()
        self._update_win_loss_counts(indicators_used, outcome)
    
    def _update_win_loss_counts(self, indicators_used: List[str], outcome: str):
        """Update win/loss count per indicator."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        column = 'wins' if outcome == 'WIN' else 'losses'
        
        for indicator in indicators_used:
            cursor.execute(f'''
                UPDATE indicator_weights
                SET {column} = {column} + 1
                WHERE indicator = ?
            ''', (indicator,))
        
        conn.commit()
        conn.close()
    
    def get_weighted_confidence(self, signals: Dict[str, str]) -> float:
        """
        Calculate weighted confidence score based on current weights.
        
        Args:
            signals: Dict of {indicator_name: 'LONG'/'SHORT'/'NEUTRAL'}
        
        Returns:
            Confidence score [-1.0, 1.0]
        """
        total_weight = sum(self.weights.values())
        score = 0.0
        
        for indicator, signal in signals.items():
            if indicator in self.weights:
                weight = self.weights[indicator]
                if signal == 'LONG':
                    score += weight
                elif signal == 'SHORT':
                    score -= weight
                # NEUTRAL adds nothing
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                AVG(pnl_pct) as avg_pnl
            FROM trade_decisions
            WHERE outcome IS NOT NULL
        ''')
        
        row = cursor.fetchone()
        
        # Per-indicator stats
        cursor.execute('SELECT indicator, weight, wins, losses FROM indicator_weights')
        indicator_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_trades': row[0] or 0,
            'wins': row[1] or 0,
            'losses': row[2] or 0,
            'win_rate': (row[1] / row[0]) if row[0] and row[0] > 0 else 0,
            'avg_pnl': row[3] or 0,
            'indicator_stats': {
                stat[0]: {'weight': stat[1], 'wins': stat[2], 'losses': stat[3]}
                for stat in indicator_stats
            },
            'current_weights': self.weights
        }
    
    def get_best_and_worst_indicators(self) -> Dict:
        """Identify best and worst performing indicators."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT indicator, weight, wins, losses,
                   CASE WHEN (wins + losses) > 0 
                        THEN CAST(wins AS FLOAT) / (wins + losses) 
                        ELSE 0 END as win_rate
            FROM indicator_weights
            ORDER BY win_rate DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {'best': None, 'worst': None}
        
        return {
            'best': rows[0][0] if rows else None,
            'worst': rows[-1][0] if rows else None,
            'all': [
                {'indicator': r[0], 'weight': r[1], 'wins': r[2], 
                 'losses': r[3], 'win_rate': r[4]}
                for r in rows
            ]
        }
    
    def log_decision_reason(self, symbol: str, passed_layers: List[str], 
                            failed_layers: List[str], final_decision: str):
        """
        Log the reasoning behind a trade decision for audit purposes.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'symbol': symbol,
            'passed_layers': passed_layers,
            'failed_layers': failed_layers,
            'decision': final_decision,
            'weights_at_time': self.weights.copy()
        }
        
        # Log to file
        log_file = os.path.join(Config.DATA_DIR, "decision_audit.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        logger.debug(f"Decision logged: {symbol} → {final_decision}")


# Singleton instance for global access
_meta_learner_instance = None

def get_meta_learner() -> MetaLearner:
    """Get or create the global MetaLearner instance."""
    global _meta_learner_instance
    if _meta_learner_instance is None:
        _meta_learner_instance = MetaLearner()
    return _meta_learner_instance
