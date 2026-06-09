"""
🧠 OMNISCIENT CONSENSUS FILTER (CTOS CORE)
===========================================================
QUÉ: Cerebro y oráculo único de filtros y vetos para todo el sistema Trader Gemini.
POR QUÉ: Antes los filtros estaban dispersos y duplicados entre RiskManager, MetaCoordinator, 
  y SignalBroker, lo que causaba vetos incoherentes, lag de ejecución y discrepancias en backtests.
PARA QUÉ: Unificar todos los 19 gates atómicos (Kill Switch, Fee Drag, Regímenes, Tensión, 
  Correlación, Sentimiento, Vacío de Liquidez, Invariantes y Contagio) en un pipeline secuencial 
  de altísima velocidad (<1ms) y consistencia 100% garantizada.
CÓMO: Oráculo secuencial ordenado por costo computacional.
CUÁNDO: Ejecutado en MetaCoordinator al arbitrar intenciones y en RiskManager al generar órdenes.
DÓNDE: core/consensus_filter.py
QUIÉN: Arquitecto Senior + Risk Manager + Quant Developer + SRE
"""

import time
import logging
from typing import Tuple, Dict, Any, List

from utils.logger import logger
from core.events import SignalEvent, SignalType
from core.enums import OrderSide
from config import Config
from utils.strategy_tracker import strategy_tracker

class ConsensusFilter:
    """
    🧠 Omnisciente Consensus Filter
    
    Unifica todos los filtros y validaciones de riesgo, invariantes del sistema y 
    vetos topológicos de grafo en un solo punto centralizado de verdad.
    """
    
    def __init__(self):
        self._metrics = {
            "total_evaluations": 0,
            "passed": 0,
            "failed": 0,
            "veto_reasons": {}
        }
        logger.info("🧠 [ConsensusFilter] Omnisciente Consensus Filter Inicializado Exitosamente.")
    
    def check_signal(
        self, 
        signal_event: SignalEvent, 
        portfolio: Any, 
        current_price: float, 
        risk_manager: Any = None, 
        meta_coordinator: Any = None
    ) -> Tuple[bool, str]:
        """
        Evalúa secuencialmente todos los gates de consenso unificados.
        Retorna (True, 'APPROVED') o (False, 'MOTIVO_RECHAZO').
        """
        self._metrics["total_evaluations"] += 1
        symbol = signal_event.symbol
        sig_type_str = getattr(signal_event.signal_type, "name", str(signal_event.signal_type))
        horizon = getattr(signal_event, "horizon", "SCALPING")
        
        # 1. EMERGENCY BYPASS (EXIT signals bypass all entry filters)
        if sig_type_str == "EXIT" or getattr(signal_event, "is_exit", False):
            self._metrics["passed"] += 1
            return True, "APPROVED_BYPASS_EXIT"

        # =====================================================================
        # BANDA 1: FILTROS DE RIESGO DE BAJO COSTO (RiskManager/Global config)
        # =====================================================================
        
        # Gate 0.5: Toxic Asset Blacklist (Centralized in Config)
        TOXIC_ASSETS = getattr(Config.Risk, 'TOXIC_ASSETS', ["DOT/USDT", "ATOM/USDT"])
        norm_symbol = symbol.replace("/", "")
        toxic_normalized = [t.replace("/", "") for t in TOXIC_ASSETS]
        if symbol in TOXIC_ASSETS or norm_symbol in toxic_normalized:
            return self._fail(f"TOXIC_ASSET_BLACKLISTED ({symbol})")
        
        # Gate 0.7: Dynamic Symbol Win Rate Blacklist (FORENSIC PROTECTION)
        # FORENSIC-V154: Tightened from 20→8 trades, WR 10%→20%.
        # DATA: BTC did 17 trades at 5% WR (-$0.89) without triggering blacklist.
        # 8 trades is enough to detect a failing symbol pattern.
        try:
            symbol_trades = [t for t in strategy_tracker.trades if t.symbol == symbol and t.horizon == horizon]
            if len(symbol_trades) >= 8:
                last_n_trades = symbol_trades[-8:]
                wins = sum(1 for t in last_n_trades if t.is_win)
                wr = wins / len(last_n_trades)
                if wr < 0.20:
                    ml_confidence = getattr(signal_event, 'ml_confidence', getattr(signal_event, 'strength', 0.5))
                    if ml_confidence >= 0.55:
                        logger.info(f"🧠 [SOPHIA OVERRIDE] {symbol} bypassing Dynamic Blacklist (WR {wr*100:.1f}%) due to high AI confidence: {ml_confidence:.2f}")
                    else:
                        logger.warning(f"🛑 [DYNAMIC BLACKLIST] {symbol} suspended: Recent WR {wr*100:.1f}% < 20% on last {len(last_n_trades)} trades.")
                        return self._fail(f"DYNAMIC_SYMBOL_BLACKLIST ({symbol} Recent WR {wr*100:.1f}%)")
        except Exception as tracker_err:
            logger.error(f"❌ Error checking dynamic symbol blacklist: {tracker_err}")
        
        # Gate 0.8: Symbol Directional Preference (FORENSIC-V156)
        # QUÉ: Aplica bias direccional y confianza mínima por símbolo.
        # POR QUÉ: BTC LONG = 0% WR (-$0.91), BTC SHORT = 100% WR (+$0.26).
        #   Cada moneda tiene patrones direccionales distintos.
        # CÓMO: Ajusta la confianza de la señal con el bias del perfil del símbolo.
        try:
            _sym_profile = Config.SymbolProfiles.get(symbol)
            # OMEGA FIX: Use strength (all strategies set this) + ml_confidence as boost
            _strength = getattr(signal_event, "strength", 0.5)
            _ml_conf = getattr(signal_event, "ml_confidence", None)
            _sig_confidence = max(v for v in [_strength, _ml_conf] if v is not None) if _ml_conf is not None else _strength
            _direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
            _dir_bias = _sym_profile.get("long_bias", 0) if _direction == "LONG" else _sym_profile.get("short_bias", 0)
            _adjusted_conf = _sig_confidence + _dir_bias
            _min_conf = _sym_profile.get("min_confidence", 0.50)
            
            if _adjusted_conf < _min_conf:
                return self._fail(
                    f"SYMBOL_PROFILE_LOW_CONF ({symbol} {_direction} "
                    f"raw={_sig_confidence:.3f}{_dir_bias:+.2f}={_adjusted_conf:.3f}<{_min_conf})"
                )
        except Exception as profile_err:
            logger.error(f"❌ Error in symbol profile check: {profile_err}")
            
        # Gate 1: Kill Switch
        if risk_manager:
            if not risk_manager._validate_kill_switch():
                return self._fail("KILL_SWITCH_ACTIVE")
        elif getattr(Config, "KILL_SWITCH_ACTIVE", False):
            return self._fail("KILL_SWITCH_ACTIVE")

        # Gate 2: Fee Drag Filter
        # QUÉ: Bloquea trades donde la volatilidad actual (ATR) es tan baja que 
        #   no cubre ni siquiera las comisiones.
        # POR QUÉ: Las matemáticas del micro-scalping exigen que el mercado 
        #   tenga amplitud. Operar sin spread destruye el equity por fees.
        try:
            if getattr(Config.Execution, "USE_LIMIT_BBO_ENTRIES", True) and getattr(Config.Execution, "USE_LIMIT_BBO_EXITS", True):
                round_trip_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002) * 2
            else:
                round_trip_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375) * 2
            
            # Minimum TP expected is typically ~0.3%. ATR must support this movement
            # ATR check: Current ATR percentage must be > round_trip_fee * 3.0
            _sig_meta = getattr(signal_event, "metadata", {}) or {}
            atr_pct = _sig_meta.get("atr_pct", 0.0)
            
            if atr_pct > 0:
                if atr_pct < (round_trip_fee * 3.0):
                    logger.warning(f"🛑 [VOLATILITY BLOCK] {symbol} ATR {atr_pct*100:.3f}% < {round_trip_fee*3.0*100:.3f}% (3.0x round-trip fee).")
                    return self._fail(f"FEE_DRAG_ATR ({atr_pct*100:.3f}% < fee_buffer)")
        except Exception as e:
            logger.error(f"❌ Error in Fee Drag filter: {e}")

        # Gate 3: Frequency Limits
        if risk_manager:
            if not risk_manager._validate_frequency_limits(symbol, signal_event.signal_type):
                return self._fail("FREQUENCY_LIMIT_EXCEEDED")

        # Gate 3.5: Cooldown Check (Horizon-Aware)
        strategy_id = getattr(signal_event, "strategy_id", "Unknown")
        from utils.cooldown_manager import cooldown_manager
        can_trade_res = cooldown_manager.can_trade(symbol, strategy_id=strategy_id, horizon=horizon)
        if not can_trade_res[0]:
            return self._fail(f"COOLDOWN_ACTIVE ({can_trade_res[1]})")

        # Gate 4: Regime Veto
        if risk_manager:
            if not risk_manager._validate_regime_veto(symbol, signal_event.signal_type):
                global_regime = getattr(risk_manager, "global_regime", "UNKNOWN")
                return self._fail(f"REGIME_MISMATCH ({sig_type_str} vs {global_regime})")

        # Gate 4.5: Strategic Regime Veto (Final Quality Filter)
        if risk_manager:
            current_regime = getattr(risk_manager, "current_regime", "UNKNOWN")
            if ("VOLATILE" in current_regime or "CHOPPY" in current_regime) and strategy_id == "TECHNICAL_STRATEGY":
                return self._fail(f"STRATEGIC_REGIME_VETO_{current_regime}_TECHNICAL")
            if "TRENDING" in current_regime and strategy_id == "STATISTICAL_REVERSION":
                return self._fail(f"STRATEGIC_REGIME_VETO_{current_regime}_STATISTICAL")

        # Gate 5: Regime Tension Veto
        tension = getattr(signal_event, "tension", 0.0)
        if tension > 1.5 or tension < -1.5:
            return self._fail(f"REGIME_TENSION_EXCESSIVE (tension={tension:.2f})")

        # =====================================================================
        # BANDA 2: INVARIANTES AXIOMÁTICOS DEL GRAFO & ORÁCULOS DE MERCADO
        # =====================================================================
        
        # Gate 6: Invariantes Estricto (De invariants.py)
        if meta_coordinator:
            if hasattr(meta_coordinator, "_check_invariants"):
                if not meta_coordinator._check_invariants(signal_event):
                    return self._fail("SYSTEM_INVARIANT_VIOLATION")
        else:
            # Fallback local a invariants
            from core.invariants import invariants
            from core.structs import TradeIntent
            direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
            pseudo_intent = TradeIntent(
                symbol=symbol,
                direction=direction,
                confidence=getattr(signal_event, "confidence", getattr(signal_event, "strength", 0.5)),
                expected_mfe=0.0,
                expected_mae=0.0,
                horizon=horizon,
                regime_compatibility=1.0,
                liquidity_score=0.5,
                strategy_id=getattr(signal_event, "strategy_id", "unknown"),
                timestamp_ns=getattr(signal_event, "timestamp_ns", 0)
            )
            passed, reason = invariants.check_all(pseudo_intent)
            if not passed:
                return self._fail(f"SYSTEM_INVARIANT_VIOLATION ({reason})")

        # =====================================================================
        # BANDA 3: VETOS TOPOLÓGICOS Y ANÁLISIS DE CORRELACIÓN Y SENTIMIENTO
        # =====================================================================
        
        # Gate 7: Asset Correlation Risk
        if risk_manager and hasattr(risk_manager, "correlation_manager") and risk_manager.correlation_manager:
            active_symbols = list(set(
                v_key.split('_')[0] for v_key, pos in portfolio.virtual_ledger.items()
                if abs(pos.get("quantity", 0)) > 1e-8
            ))
            if active_symbols:
                safe, reason = risk_manager.correlation_manager.check_correlation_risk(symbol, active_symbols)
                if not safe:
                    return self._fail(f"HIGH_CORRELATION_VETO ({reason})")

        # Gate 8: Market Sentiment Veto
        if risk_manager and hasattr(risk_manager, "sentiment_processor") and risk_manager.sentiment_processor:
            mood = risk_manager.sentiment_processor.get_market_mood()
            if sig_type_str == "LONG" and mood < -0.5:
                return self._fail(f"SENTIMENT_DIVERGENCE (LONG but Mood={mood:.2f})")
            elif sig_type_str == "SHORT" and mood > 0.5:
                return self._fail(f"SENTIMENT_DIVERGENCE (SHORT but Mood={mood:.2f})")

        # Gate 9: Liquidity Vacuum Veto
        if horizon == "SCALPING" and risk_manager and hasattr(risk_manager, "liquidity_guardian") and risk_manager.liquidity_guardian:
            quality = risk_manager.liquidity_guardian.get_market_quality_score(symbol)
            if quality < 30:
                return self._fail(f"LIQUIDITY_VACUUM (Quality={quality:.1f} < 30)")

        # Gate 10: Graph Theory & Contagion Veto
        if meta_coordinator and hasattr(meta_coordinator, "graph_layer") and meta_coordinator.graph_layer:
            direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
            state = meta_coordinator.graph_layer.state_matrix.get(symbol)
            
            # 1. Contagio
            if direction == "LONG":
                contagion_risk = meta_coordinator.graph_layer.get_contagion_risk(symbol)
                if contagion_risk > 0.50:
                    return self._fail(f"GRAPH_CONTAGION_RISK (Risk={contagion_risk:.2f})")
            
            if state:
                # 2. Microstructure desbalance
                if direction == "LONG" and state.orderflow_imbalance < -0.60:
                    return self._fail(f"ORDERFLOW_IMBALANCE (Imbalance={state.orderflow_imbalance:.2f})")
                if direction == "SHORT" and state.orderflow_imbalance > 0.60:
                    return self._fail(f"ORDERFLOW_IMBALANCE (Imbalance={state.orderflow_imbalance:.2f})")
                
                # 3. Ecosystem Gravity
                ecosystem_gravity = meta_coordinator.graph_layer.get_ecosystem_gravity()
                if direction == "LONG" and ecosystem_gravity < -2.0 and state.eigenvector_centrality > 0.1:
                    return self._fail(f"ECOSYSTEM_GRAVITY_VETO (Gravity={ecosystem_gravity:.2f})")

        # Si supera todos los gates, ¡APROBADA!
        self._metrics["passed"] += 1
        return True, "APPROVED"

    def _fail(self, reason: str) -> Tuple[bool, str]:
        """Registra el fallo en las métricas y retorna la causa."""
        self._metrics["failed"] += 1
        self._metrics["veto_reasons"][reason] = self._metrics["veto_reasons"].get(reason, 0) + 1
        return False, reason

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna las métricas acumuladas de evaluación."""
        return self._metrics

# Singleton Global para uso unificado
_consensus_filter = ConsensusFilter()

def get_consensus_filter() -> ConsensusFilter:
    """Devuelve la instancia singleton de ConsensusFilter."""
    return _consensus_filter
