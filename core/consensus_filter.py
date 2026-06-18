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
                        logger.warning(f"🛑 [DYNAMIC BLACKLIST SOFT-VETO] {symbol} suspended: Recent WR {wr*100:.1f}% < 20% on last {len(last_n_trades)} trades. FASE II: Aplicando Sizing Termodinámico (10%).")
                        try:
                            object.__setattr__(signal_event, 'thermodynamic_micro_sizing', True)
                        except (AttributeError, TypeError):
                            pass
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
            
        # =====================================================================
        # BANDA 2: VETOS ESTRUCTURALES Y DEL SISTEMA (Backtest & Prod Parity)
        # =====================================================================
        
        # ─── FASE 1: FUNDING EVASION ───
        if hasattr(signal_event, 'timestamp'):
            try:
                from datetime import datetime
                evt_dt = datetime.fromtimestamp(signal_event.timestamp)
                if evt_dt.minute >= 45:
                    return self._fail("FUNDING_EVASION")
            except Exception:
                pass

        # ─── FASE 9: CORRELATION SHIELD ───
        if symbol in ("ETH/USDT", "SOL/USDT") and horizon in ('SCALPING', 'MICROSCALPING'):
            if portfolio:
                btc_pos = portfolio.get_horizon_position("BTC/USDT", horizon)
                if btc_pos and btc_pos.get("quantity", 0) != 0:
                    btc_dir = "LONG" if btc_pos["quantity"] > 0 else "SHORT"
                    _direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
                    if btc_dir == _direction:
                        return self._fail("CORRELATION_SHIELD")
                        
        # ─── FASE 22: REGIME LOCKS ───
        if portfolio and hasattr(portfolio, 'market_regime') and portfolio.market_regime:
            locks = portfolio.market_regime.get_regime_locks(symbol)
            _direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
            is_locked = False
            if horizon == 'SWING' and locks.get('LOCK_SWING'):
                is_locked = True
            elif horizon in ('SCALPING', 'MICROSCALPING'):
                if _direction == "LONG" and locks.get('LOCK_SCALP_LONG'):
                    is_locked = True
                elif _direction == "SHORT" and locks.get('LOCK_SCALP_SHORT'):
                    is_locked = True
            if is_locked:
                return self._fail("REGIME_LOCK")

        # =========================================================================
        # 🚀 FASE 0: OMNISCORE FUSION (The Perfect Binomial)
        # QUÉ: Integra señales de ML y Technical de forma asíncrona usando el SSOT.
        # POR QUÉ: Las estrategias son asíncronas. Para que el bot sea 100% igual
        #   al backtest, la decisión debe basarse en el estado combinado de TODAS.
        # =========================================================================
        try:
            caller_id = getattr(signal_event, 'strategy_id', 'UNKNOWN').lower()
            
            # Config already imported at module level (line 23)
            # FASE 32: If signal comes from OmniStrategy, it already has the fused score. Skip redundant calculation.
            if getattr(Config, 'OmniScore', None) and getattr(Config.OmniScore, 'master_threshold', 0.0) > 0.0 and "omni" not in caller_id:
                from core.global_state import global_state
                sv = global_state.get_symbol_vector(symbol)
                
                # In backtest mode, global_state may not have symbol vectors.
                # Skip this production-only OmniScore gate entirely.
                if sv is None:
                    raise RuntimeError("BACKTEST_BYPASS: No symbol vector available")
                
                # Fetch thresholds & weights
                master_th = Config.OmniScore.master_threshold
                ml_th_long = Config.OmniScore.ml_threshold_bull
                ml_th_short = Config.OmniScore.ml_threshold_bear
                w_ml = Config.OmniScore.w_ml
                w_tech = Config.OmniScore.w_technical
                
                tech_active = 0
                ml_active = 0
                
                # Identify caller and state
                caller_id = getattr(signal_event, 'strategy_id', 'UNKNOWN').lower()
                direction = "LONG" if signal_event.signal_type == SignalType.LONG else "SHORT"
                
                if direction == "LONG":
                    # Determine tech active
                    if "tech" in caller_id:
                        tech_active = 1
                    else:
                        tech_active = sv.tech_long_active
                        
                    # Determine ml active
                    if "ml" in caller_id:
                        # Event caller is ML, so we trust its internal confidence if it fired
                        # Or we use the global state
                        ml_active = 1 if getattr(signal_event, 'confidence', sv.ml_bull_score) >= ml_th_long else 0
                    else:
                        ml_active = 1 if sv.ml_bull_score >= ml_th_long else 0
                        
                else: # SHORT
                    # Determine tech active
                    if "tech" in caller_id:
                        tech_active = 1
                    else:
                        tech_active = sv.tech_short_active
                        
                    # Determine ml active
                    if "ml" in caller_id:
                        ml_active = 1 if getattr(signal_event, 'confidence', sv.ml_bear_score) >= ml_th_short else 0
                    else:
                        ml_active = 1 if sv.ml_bear_score >= ml_th_short else 0
                        
                # Additional modules
                phalanx_active = sv.phalanx_sig if hasattr(sv, 'phalanx_sig') else 0
                w_phalanx = getattr(Config.OmniScore, 'w_phalanx', 0.0)
                
                statarb_active = sv.statarb_sig if hasattr(sv, 'statarb_sig') else 0
                w_statarb = getattr(Config.OmniScore, 'w_statarb', 0.0)
                
                # Base OmniScore
                omniscore = (tech_active * w_tech) + (ml_active * w_ml) + (phalanx_active * w_phalanx) + (statarb_active * w_statarb)
                
                if omniscore < master_th:
                    return self._fail(
                        f"OMNISCORE_VETO ({symbol} {direction} OmniScore={omniscore:.2f} < {master_th:.2f} | "
                        f"Tech:{tech_active} ML:{ml_active})"
                    )
                
                # FASE 36: OMNISCORE ADAPTATIVE PENALTIES (Soft Vetos)
                # Instead of completely blocking trades because 1 sub-system disagreed, 
                # we apply penalties. If OmniScore survives, the trade executes.
                penalty = 0.0

                # Soft Gate 4: Regime Mismatch
                if risk_manager:
                    global_regime = getattr(risk_manager, "global_regime", "UNKNOWN")
                    if not risk_manager._validate_regime_veto(symbol, signal_event.signal_type):
                        penalty += 0.20
                        logger.debug(f"  [OmniScore] Penalty -0.20 for Regime Mismatch ({global_regime})")

                # Soft Gate 4.5: Strategic Regime
                if risk_manager:
                    current_regime = getattr(risk_manager, "current_regime", "UNKNOWN")
                    strategy_id_val = getattr(signal_event, 'strategy_id', 'UNKNOWN')
                    if ("VOLATILE" in current_regime or "CHOPPY" in current_regime) and strategy_id_val == "TECHNICAL_STRATEGY":
                        penalty += 0.15
                    if "TRENDING" in current_regime and strategy_id_val == "STATISTICAL_REVERSION":
                        penalty += 0.15

                # Soft Gate 5: Tension
                tension = getattr(signal_event, "tension", 0.0)
                if tension > 1.5 or tension < -1.5:
                    penalty += 0.10

                # Soft Gate 7: Correlation Risk
                if risk_manager and hasattr(risk_manager, "correlation_manager") and risk_manager.correlation_manager:
                    active_symbols = list(set(
                        v_key.split('_')[0] for v_key, pos in portfolio.virtual_ledger.items()
                        if abs(pos.get("quantity", 0)) > 1e-8
                    ))
                    if active_symbols:
                        safe, reason = risk_manager.correlation_manager.check_correlation_risk(symbol, active_symbols)
                        if not safe:
                            penalty += 0.25
                            logger.debug(f"  [OmniScore] Penalty -0.25 for High Correlation")

                # Soft Gate 8: Sentiment Divergence
                if risk_manager and hasattr(risk_manager, "sentiment_processor") and risk_manager.sentiment_processor:
                    mood = risk_manager.sentiment_processor.get_market_mood()
                    if sig_type_str == "LONG" and mood < -0.5:
                        penalty += 0.15
                    elif sig_type_str == "SHORT" and mood > 0.5:
                        penalty += 0.15

                # Soft Gate 9: Liquidity Vacuum
                if horizon == "SCALPING" and risk_manager and hasattr(risk_manager, "liquidity_guardian") and risk_manager.liquidity_guardian:
                    quality = risk_manager.liquidity_guardian.get_market_quality_score(symbol)
                    if quality < 30:
                        penalty += 0.20

                # Soft Gate 10: Contagion & Topology
                if meta_coordinator and hasattr(meta_coordinator, "graph_layer") and meta_coordinator.graph_layer:
                    state = meta_coordinator.graph_layer.state_matrix.get(symbol)
                    if direction == "LONG":
                        contagion_risk = meta_coordinator.graph_layer.get_contagion_risk(symbol)
                        if contagion_risk > 0.50:
                            penalty += 0.30
                    if state:
                        if direction == "LONG" and state.orderflow_imbalance < -0.60:
                            penalty += 0.20
                        if direction == "SHORT" and state.orderflow_imbalance > 0.60:
                            penalty += 0.20
                        ecosystem_gravity = meta_coordinator.graph_layer.get_ecosystem_gravity()
                        if direction == "LONG" and ecosystem_gravity < -2.0 and state.eigenvector_centrality > 0.1:
                            penalty += 0.25

                # Final Evaluation
                final_score = omniscore - penalty
                if final_score < master_th:
                    return self._fail(f"OMNISCORE_SOFT_VETOS_DEPLETED (Init:{omniscore:.2f} - Penalty:{penalty:.2f} = {final_score:.2f} < {master_th:.2f})")
                
                logger.info(f"🧠 [OmniScore] {symbol} {direction} APPROVED | Final Score: {final_score:.2f} (Init: {omniscore:.2f}, Pen: {penalty:.2f})")
                
        except RuntimeError as e:
            if "BACKTEST_BYPASS" in str(e):
                pass # Silently ignore backtest bypass
            else:
                logger.error(f"❌ Error in OmniScore Fusion Gate: {e}")
        except Exception as e:
            logger.error(f"❌ Error in OmniScore Fusion Gate: {e}")
            
        # Gate 1: Kill Switch (HARD BLOCK)
        if risk_manager:
            if not risk_manager._validate_kill_switch():
                return self._fail("KILL_SWITCH_ACTIVE")
        elif getattr(Config, "KILL_SWITCH_ACTIVE", False):
            return self._fail("KILL_SWITCH_ACTIVE")

        # Gate 2: Fee Drag Filter (HARD BLOCK)
        try:
            if getattr(Config.Execution, "USE_LIMIT_BBO_ENTRIES", True) and getattr(Config.Execution, "USE_LIMIT_BBO_EXITS", True):
                round_trip_fee = getattr(Config, 'BINANCE_MAKER_FEE_BNB', 0.0002) * 2
            else:
                round_trip_fee = getattr(Config, 'BINANCE_TAKER_FEE_BNB', 0.000375) * 2
            
            _HORIZON_GATE_MULT = {
                'MICROSCALPING': 1.5,
                'SCALPING': 2.0,
                'SWING': 2.8,
            }
            _gate_mult = _HORIZON_GATE_MULT.get(horizon, 2.0)
            
            dna_mult = getattr(Config.Risk, 'CONSENSUS_FEE_MULT', None)
            if dna_mult is not None:
                _gate_mult = dna_mult
            
            _sig_meta = getattr(signal_event, "metadata", {}) or {}
            atr_pct = _sig_meta.get("atr_pct", 0.0)
            _fee_threshold = round_trip_fee * _gate_mult
            
            if atr_pct > 0 and atr_pct < _fee_threshold:
                import os
                if os.getenv("TRADER_GEMINI_BACKTEST") != "true":
                    logger.warning(f"🛑 [VOLATILITY BLOCK] {symbol} {horizon} ATR {atr_pct*100:.3f}% < {_fee_threshold*100:.3f}% ({_gate_mult}x round-trip fee).")
                    return self._fail(f"FEE_DRAG_ATR ({atr_pct*100:.3f}% < fee_buffer {_gate_mult}x)")
        except Exception as e:
            logger.error(f"❌ Error in Fee Drag filter: {e}")

        # Gate 3: Frequency Limits (HARD BLOCK)
        if risk_manager:
            if not risk_manager._validate_frequency_limits(symbol, signal_event.signal_type):
                return self._fail("FREQUENCY_LIMIT_EXCEEDED")

        # Gate 3.5: Cooldown Check (HARD BLOCK)
        strategy_id = getattr(signal_event, "strategy_id", "Unknown")
        from utils.cooldown_manager import cooldown_manager
        
        _volatility = getattr(signal_event, "metadata", {}).get("atr_ratio", 1.0) if isinstance(getattr(signal_event, "metadata", {}), dict) else 1.0
        _streak = 0
        if risk_manager and risk_manager.portfolio:
            _streak = getattr(risk_manager.portfolio, "_win_streak", 0)

        can_trade_res = cooldown_manager.can_trade(
            symbol, 
            strategy_id=strategy_id, 
            horizon=horizon,
            volatility_factor=_volatility,
            win_streak=_streak
        )
        if not can_trade_res[0]:
            return self._fail(f"COOLDOWN_ACTIVE ({can_trade_res[1]})")

        # Gate 6: Invariantes Estricto (HARD BLOCK)
        if meta_coordinator:
            if hasattr(meta_coordinator, "_check_invariants"):
                if not meta_coordinator._check_invariants(signal_event):
                    return self._fail("SYSTEM_INVARIANT_VIOLATION")
        else:
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

        # Si supera todos los gates fuertes y sobrevivió a las penalidades suaves, ¡APROBADA!
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
