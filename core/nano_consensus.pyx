# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
import time
import logging
from typing import Tuple, Dict, Any, List
from utils.logger import logger
from core.events import SignalEvent, SignalType

cdef class NanoConsensusFilter:
    """
    🧠 Omnisciente Consensus Filter (Cython/C++ Optimized)
    
    Unifica todos los filtros y validaciones de riesgo, invariantes del sistema y 
    vetos topológicos de grafo en un solo punto centralizado de verdad.
    
    Optimizaciones JIT:
    - Verificación rápida de atributos de clase.
    - Early exits (bypassing overhead Python).
    - C-types para cálculos matemáticos intermedios.
    """
    
    cdef public dict _metrics
    cdef public dict _veto_reasons
    cdef public int _total_evaluations
    cdef public int _passed
    cdef public int _failed
    
    def __init__(self):
        self._metrics = {
            "total_evaluations": 0,
            "passed": 0,
            "failed": 0,
            "veto_reasons": {}
        }
        self._total_evaluations = 0
        self._passed = 0
        self._failed = 0
        self._veto_reasons = {}
        logger.info("🧠⚡ [NanoConsensusFilter] Cython JIT Compiler Engine Inicializado Exitosamente.")

    cdef tuple _fail(self, str reason):
        self._failed += 1
        if reason in self._veto_reasons:
            self._veto_reasons[reason] = self._veto_reasons[reason] + 1
        else:
            self._veto_reasons[reason] = 1
        return (False, reason)

    cpdef tuple check_signal_jit(self, object signal_event, object portfolio, object current_price, object risk_manager, object meta_coordinator, object Config, object strategy_tracker):
        """
        Evalúa secuencialmente todos los gates de consenso unificados en Cython.
        Retorna (True, 'APPROVED') o (False, 'MOTIVO_RECHAZO').
        """
        self._total_evaluations += 1
        
        cdef str symbol = signal_event.symbol
        cdef str horizon = getattr(signal_event, "horizon", "SCALPING")
        
        cdef str sig_type_str
        if hasattr(signal_event.signal_type, "name"):
            sig_type_str = signal_event.signal_type.name
        else:
            sig_type_str = str(signal_event.signal_type)
            
        cdef str direction = "LONG" if sig_type_str == "LONG" else "SHORT"
        
        # 1. EMERGENCY BYPASS (EXIT signals bypass all entry filters)
        if sig_type_str == "EXIT" or getattr(signal_event, "is_exit", False):
            self._passed += 1
            return (True, "APPROVED_BYPASS_EXIT")

        # =====================================================================
        # BANDA 1: FILTROS DE RIESGO DE BAJO COSTO (RiskManager/Global config)
        # =====================================================================
        
        # Gate 0.5: Toxic Asset Blacklist
        cdef list TOXIC_ASSETS = getattr(Config.Risk, 'TOXIC_ASSETS', ["DOT/USDT", "ATOM/USDT"])
        cdef str norm_symbol = symbol.replace("/", "")
        cdef list toxic_normalized = []
        for t in TOXIC_ASSETS:
            toxic_normalized.append(t.replace("/", ""))
        
        if symbol in TOXIC_ASSETS or norm_symbol in toxic_normalized:
            return self._fail(f"TOXIC_ASSET_BLACKLISTED ({symbol})")
            
        # Gate 0.7: Dynamic Symbol Win Rate Blacklist
        try:
            symbol_trades = []
            if hasattr(strategy_tracker, 'trades'):
                for t in strategy_tracker.trades:
                    if t.symbol == symbol and getattr(t, 'horizon', '') == horizon:
                        symbol_trades.append(t)
                        
            if len(symbol_trades) >= 8:
                last_n_trades = symbol_trades[-8:]
                wins = 0
                for t in last_n_trades:
                    if t.is_win:
                        wins += 1
                wr = float(wins) / len(last_n_trades)
                if wr < 0.20:
                    ml_confidence = getattr(signal_event, 'ml_confidence', getattr(signal_event, 'strength', 0.5))
                    if ml_confidence >= 0.55:
                        pass
                    else:
                        try:
                            object.__setattr__(signal_event, 'thermodynamic_micro_sizing', True)
                        except (AttributeError, TypeError):
                            from utils.error_handler import SystemIntegrityError
                            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
        except Exception:
            pass
            
        # Gate 0.8: Symbol Directional Preference
        try:
            if hasattr(Config, 'SymbolProfiles') and symbol in Config.SymbolProfiles:
                _sym_profile = Config.SymbolProfiles[symbol]
                _strength = getattr(signal_event, "strength", 0.5)
                _ml_conf = getattr(signal_event, "ml_confidence", None)
                if _ml_conf is not None:
                    _sig_confidence = max(_strength, _ml_conf)
                else:
                    _sig_confidence = _strength
                    
                _dir_bias = _sym_profile["long_bias"] if direction == "LONG" else _sym_profile["short_bias"]
                _adjusted_conf = _sig_confidence + _dir_bias
                _min_conf = _sym_profile["min_confidence"]
                
                if _adjusted_conf < _min_conf:
                    return self._fail(
                        f"SYMBOL_PROFILE_LOW_CONF ({symbol} {direction} "
                        f"raw={_sig_confidence:.3f}{_dir_bias:+.2f}={_adjusted_conf:.3f}<{_min_conf})"
                    )
        except Exception:
            pass

        # =====================================================================
        # BANDA 2: VETOS ESTRUCTURALES Y DEL SISTEMA
        # =====================================================================
        
        # ─── FASE 1: FUNDING EVASION ───
        if hasattr(signal_event, 'timestamp'):
            try:
                from datetime import datetime
                evt_dt = datetime.fromtimestamp(signal_event.timestamp)
                if evt_dt.minute >= 45:
                    return self._fail("FUNDING_EVASION")
            except Exception:
                from utils.error_handler import SystemIntegrityError
                raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

        # ─── FASE 9: CORRELATION SHIELD ───
        if symbol in ("ETH/USDT", "SOL/USDT") and horizon in ('SCALPING', 'MICROSCALPING'):
            if portfolio:
                btc_pos = portfolio.get_horizon_position("BTC/USDT", horizon)
                if btc_pos and btc_pos.get("quantity", 0) != 0:
                    btc_dir = "LONG" if btc_pos["quantity"] > 0 else "SHORT"
                    if btc_dir == direction:
                        return self._fail("CORRELATION_SHIELD")

        # ─── FASE 22: REGIME LOCKS ───
        cdef bint is_locked = False
        if portfolio and hasattr(portfolio, 'market_regime') and portfolio.market_regime:
            locks = portfolio.market_regime.get_regime_locks(symbol)
            if horizon == 'SWING' and locks.get('LOCK_SWING', False):
                is_locked = True
            elif horizon in ('SCALPING', 'MICROSCALPING'):
                if direction == "LONG" and locks.get('LOCK_SCALP_LONG', False):
                    is_locked = True
                elif direction == "SHORT" and locks.get('LOCK_SCALP_SHORT', False):
                    is_locked = True
            if is_locked:
                return self._fail("REGIME_LOCK")

        # =========================================================================
        # 🚀 FASE 0: OMNISCORE FUSION (The Perfect Binomial)
        # =========================================================================
        cdef float omniscore = 1.0
        cdef float master_th = 0.0
        cdef float penalty = 0.0
        
        try:
            caller_id = getattr(signal_event, 'strategy_id', 'UNKNOWN').lower()
            
            if hasattr(Config, 'OmniScore') and getattr(Config.OmniScore, 'master_threshold', 0.0) > 0.0 and "omni" not in caller_id:
                from core.global_state import global_state
                sv = global_state.get_symbol_vector(symbol)
                
                if sv is None:
                    raise RuntimeError("BACKTEST_BYPASS: No symbol vector available")
                
                master_th = Config.OmniScore.master_threshold
                ml_th_long = Config.OmniScore.ml_threshold_bull
                ml_th_short = Config.OmniScore.ml_threshold_bear
                w_ml = Config.OmniScore.w_ml
                w_tech = Config.OmniScore.w_technical
                
                tech_active = 0
                ml_active = 0
                
                if direction == "LONG":
                    if "tech" in caller_id:
                        tech_active = 1
                    else:
                        tech_active = getattr(sv, 'tech_long_active', 0)
                        
                    if "ml" in caller_id:
                        ml_active = 1 if getattr(signal_event, 'confidence', getattr(sv, 'ml_bull_score', 0)) >= ml_th_long else 0
                    else:
                        ml_active = 1 if getattr(sv, 'ml_bull_score', 0) >= ml_th_long else 0
                else:
                    if "tech" in caller_id:
                        tech_active = 1
                    else:
                        tech_active = getattr(sv, 'tech_short_active', 0)
                        
                    if "ml" in caller_id:
                        ml_active = 1 if getattr(signal_event, 'confidence', getattr(sv, 'ml_bear_score', 0)) >= ml_th_short else 0
                    else:
                        ml_active = 1 if getattr(sv, 'ml_bear_score', 0) >= ml_th_short else 0
                        
                phalanx_active = getattr(sv, 'phalanx_sig', 0)
                w_phalanx = getattr(Config.OmniScore, 'w_phalanx', 0.0)
                
                statarb_active = getattr(sv, 'statarb_sig', 0)
                w_statarb = getattr(Config.OmniScore, 'w_statarb', 0.0)
                
                omniscore = (tech_active * w_tech) + (ml_active * w_ml) + (phalanx_active * w_phalanx) + (statarb_active * w_statarb)
                
                if omniscore < master_th:
                    return self._fail(
                        f"OMNISCORE_VETO ({symbol} {direction} OmniScore={omniscore:.2f} < {master_th:.2f} | "
                        f"Tech:{tech_active} ML:{ml_active})"
                    )

                # Soft Gate 4: Regime Mismatch
                if risk_manager:
                    global_regime = getattr(risk_manager, "global_regime", "UNKNOWN")
                    if hasattr(risk_manager, "_validate_regime_veto") and not risk_manager._validate_regime_veto(symbol, signal_event.signal_type):
                        penalty += 0.20

                # Soft Gate 4.5: Strategic Regime
                if risk_manager:
                    current_regime = getattr(risk_manager, "current_regime", "UNKNOWN")
                    if ("VOLATILE" in current_regime or "CHOPPY" in current_regime) and caller_id == "technical_strategy":
                        penalty += 0.15
                    if "TRENDING" in current_regime and caller_id == "statistical_reversion":
                        penalty += 0.15

                # Soft Gate 5: Tension
                tension = getattr(signal_event, "tension", 0.0)
                if tension > 1.5 or tension < -1.5:
                    penalty += 0.10

                # Soft Gate 7: Correlation Risk
                if risk_manager and hasattr(risk_manager, "correlation_manager") and risk_manager.correlation_manager:
                    active_symbols = []
                    virtual_ledger = getattr(portfolio, 'virtual_ledger', {})
                    if virtual_ledger:
                        for v_key, pos in virtual_ledger.items():
                            if abs(pos.get("quantity", 0)) > 1e-8:
                                active_symbols.append(v_key.split('_')[0])
                    
                    active_symbols = list(set(active_symbols))
                    if active_symbols:
                        safe, reason = risk_manager.correlation_manager.check_correlation_risk(symbol, active_symbols)
                        if not safe:
                            penalty += 0.25

                # Soft Gate 8: Sentiment Divergence
                if risk_manager and hasattr(risk_manager, "sentiment_processor") and risk_manager.sentiment_processor:
                    mood = risk_manager.sentiment_processor.get_market_mood()
                    if direction == "LONG" and mood < -0.5:
                        penalty += 0.15
                    elif direction == "SHORT" and mood > 0.5:
                        penalty += 0.15

                # Soft Gate 9: Liquidity Vacuum
                if horizon == "SCALPING" and risk_manager and hasattr(risk_manager, "liquidity_guardian") and risk_manager.liquidity_guardian:
                    quality = risk_manager.liquidity_guardian.get_market_quality_score(symbol)
                    if quality < 30:
                        penalty += 0.20

                # Soft Gate 10: Contagion & Topology
                if meta_coordinator and hasattr(meta_coordinator, "graph_layer") and meta_coordinator.graph_layer:
                    state = meta_coordinator.graph_layer.state_matrix.get(symbol, None)
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
                        if direction == "LONG" and ecosystem_gravity < -2.0 and getattr(state, 'eigenvector_centrality', 0) > 0.1:
                            penalty += 0.25

                final_score = omniscore - penalty
                if final_score < master_th:
                    return self._fail(f"OMNISCORE_SOFT_VETOS_DEPLETED (Init:{omniscore:.2f} - Penalty:{penalty:.2f} = {final_score:.2f} < {master_th:.2f})")
                
        except RuntimeError as e:
            if "BACKTEST_BYPASS" not in str(e):
                pass
        except Exception:
            pass
            
        # Gate 1: Kill Switch (HARD BLOCK)
        if risk_manager:
            if hasattr(risk_manager, "_validate_kill_switch") and not risk_manager._validate_kill_switch():
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
                return self._fail(f"FEE_DRAG_ATR ({atr_pct*100:.3f}% < fee_buffer {_gate_mult}x)")
        except Exception:
            pass

        # Gate 3: Frequency Limits (HARD BLOCK)
        if risk_manager:
            if hasattr(risk_manager, "_validate_frequency_limits") and not risk_manager._validate_frequency_limits(symbol, signal_event.signal_type):
                return self._fail("FREQUENCY_LIMIT_EXCEEDED")

        # Gate 3.5: Cooldown Check (HARD BLOCK)
        strategy_id = getattr(signal_event, "strategy_id", "Unknown")
        try:
            from utils.cooldown_manager import cooldown_manager
            _volatility = getattr(signal_event, "metadata", {}).get("atr_ratio", 1.0) if isinstance(getattr(signal_event, "metadata", {}), dict) else 1.0
            _streak = getattr(risk_manager.portfolio, "_win_streak", 0) if risk_manager and risk_manager.portfolio else 0
            can_trade_res = cooldown_manager.can_trade(symbol, strategy_id=strategy_id, horizon=horizon, volatility_factor=_volatility, win_streak=_streak)
            if not can_trade_res[0]:
                return self._fail(f"COOLDOWN_ACTIVE ({can_trade_res[1]})")
        except Exception:
            pass

        # Gate 6: Invariantes Estricto (HARD BLOCK)
        if meta_coordinator:
            if hasattr(meta_coordinator, "_check_invariants"):
                if not meta_coordinator._check_invariants(signal_event):
                    return self._fail("SYSTEM_INVARIANT_VIOLATION")
        else:
            try:
                from core.invariants import invariants
                from core.structs import TradeIntent
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
            except Exception:
                pass

        self._passed += 1
        return (True, "APPROVED")

    def get_metrics(self) -> dict:
        self._metrics["total_evaluations"] = self._total_evaluations
        self._metrics["passed"] = self._passed
        self._metrics["failed"] = self._failed
        self._metrics["veto_reasons"] = self._veto_reasons
        return self._metrics
