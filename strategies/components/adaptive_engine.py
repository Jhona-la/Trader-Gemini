import numpy as np

class AdaptiveMLParameterEngine:
    """
    🧬 Adaptive ML Parameter Engine (Evolutionary Auto-Tuning) — SINGLE SOURCE OF TRUTH
    
    Auto-calibra los parámetros del ML durante backtest Y live trading
    basado en el rendimiento real de los trades.
    
    QUÉ: Motor de parámetros adaptativos que diferencia Scalping vs Swing.
    POR QUÉ: Un SL de 2% en Scalping 1m es suicida; necesitamos 0.15-0.50%.
             Un SL de 0.15% en Swing 1h es ruido; necesitamos 1.0-5.0%.
    PARA QUÉ: Garantizar que backtest y producción usen los MISMOS parámetros.
    CÓMO: Rangos diferenciados por horizonte + recalibración por ventana de trades.
    CUÁNDO: Instanciado al crear MLStrategyHybridUltimate y WalkForwardXGBoost.
    DÓNDE: strategies/components/adaptive_engine.py (Single Source of Truth)
    QUIÉN: ml_strategy.py (producción) y run_multi_horizon_backtest.py (backtest)
    
    NOTA FORENSE: La versión anterior del backtest tenía rangos sl_mult=(1.2, 3.0)
    para Scalping, lo que hacía que los backtests operaran con SL 6x más amplio
    que producción (sl_mult=0.15-0.50). Esta divergencia invalidaba los backtests.
    """
    # Rangos base para Scalping vs Swing — CORRECTOS Y FINALES
    RANGES = {
        # FORENSIC-1: Correcting Scalping bounds.
        # Scalping needs TP around 0.4% (0.4) and SL around 0.2% (0.2)
        # NANO-LATENCY TENSOR REFINEMENT: Aggressive noise rejection filters.
        'scalping': {
            'lookahead':       (10.0, 20.0),  # Midpoint = 15. Matches train_supreme.py
            'label_threshold': (0.0008, 0.0025), # REFINED: Higher base threshold to filter noise
            'retrain_interval':(90.0, 360.0),
            'dd_stress_limit': (0.40, 0.70),
            'ml_confidence':   (0.70, 0.90), # REFINED: Extreme sniper confidence required
            'vol_sensitivity': (-0.08, 0.05),
            'balance_cap':     (0.55, 0.80),
            'sl_mult':         (0.15, 0.40), # REFINED: Tighter SL for scalping
            'tp_mult':         (0.30, 0.80), # REFINED: Tighter TP for guaranteed hits
            'cooldown':        (10.0, 40.0),
            'decay_exit_threshold': (0.40, 0.60), # REFINED: Faster decay recognition
            'trail_start_pct': (0.50, 0.80),
            'trail_dist_pct':  (0.05, 0.20),
            'momentum_exit_accel': (0.005, 0.015),
        },
        'swing': {
            'lookahead':       (40.0, 80.0),  # Midpoint = 60. Matches train_supreme.py
            'label_threshold': (0.0010, 0.0050),
            'retrain_interval':(1000.0, 5000.0),
            'dd_stress_limit': (0.30, 0.50),
            'ml_confidence':   (0.65, 0.85),
            'vol_sensitivity': (0.0, 0.20),
            'balance_cap':     (0.60, 0.85),
            'sl_mult':         (1.0, 5.0),  # Swing targets (1.0% to 5.0%)
            'tp_mult':         (2.0, 10.0), # Swing targets (2.0% to 10.0%)
            'cooldown':        (25.0, 80.0),
            'decay_exit_threshold': (0.45, 0.65),
            'trail_start_pct': (0.40, 0.70),
            'trail_dist_pct':  (0.15, 0.35),
            'momentum_exit_accel': (0.012, 0.030),
        }
    }
    
    def __init__(self, horizon_str='SCALPING', alpha=0.15, learning_rate=0.02,
                 horizon_days=None, profile_override=None):
        """
        Constructor unificado para producción y backtest.
        
        Args:
            horizon_str: 'SCALPING' o 'SWING' (usado por ml_strategy.py)
            alpha: EMA smoothing factor
            learning_rate: Gradient descent step size
            horizon_days: Días del horizonte (1=scalping, 7+=swing). Backtest compat.
            profile_override: Dict con override de parámetros desde HORIZON_PROFILES.
                             Claves soportadas: 'ml_lookahead', 'ml_retrain'
        """
        # Resolver horizonte: horizon_str tiene prioridad, horizon_days como fallback
        if horizon_days is not None:
            self.is_scalping = (horizon_days <= 1) # <=1 = Scalping, 2+ = Swing
        else:
            horizon_str_upper = horizon_str.upper()
            self.is_scalping = (horizon_str_upper in ('SCALPING', 'MICROSCALPING', '1D'))
        
        self.profile = 'scalping' if self.is_scalping else 'swing'
        # Defensive copy to avoid mutating class-level RANGES
        self.r = {k: tuple(v) for k, v in self.RANGES[self.profile].items()}
        self.alpha = alpha
        self.lr = learning_rate
        
        # Iniciar en el punto medio del rango
        self.params = {k: v[0] + (v[1] - v[0]) / 2.0 for k, v in self.r.items()}
        
        # ── FIX CRÍTICO: Sincronizar parámetros con el Profile global (backtest compat) ──
        if profile_override:
            if 'ml_lookahead' in profile_override:
                val = profile_override['ml_lookahead']
                self.params['lookahead'] = val
                self.r['lookahead'] = (val * 0.5, val * 1.5)
            if 'ml_retrain' in profile_override:
                val = profile_override['ml_retrain']
                self.params['retrain_interval'] = val
                self.r['retrain_interval'] = (val * 0.5, val * 1.5)
        
        # Señas de reward suavizadas (EMA) — para producción (simple feedback)
        self.ema_reward = 0.0
        self.ema_mae = 0.0
        self.ema_mfe = 0.0
        self.last_accuracy = 0.50
        self.trades_processed = 0
        
        # Window-Based Recalibration — para backtest (avanzado)
        self.trade_history = []
        self.recalibration_interval = 50

    def get(self, param_name):
        val = self.params.get(param_name)
        if param_name in ['lookahead', 'retrain_interval', 'cooldown']:
            return int(round(val))
        return val
    
    def get_model_suffix(self):
        """Retorna sufijo para nombres de modelo: '_scalping' o '_swing'"""
        return '_scalping' if self.is_scalping else '_swing'

    def feedback_trade(self, pnl_pct, mae_pct=0, mfe_pct=0):
        """
        ♻️ PHASE 4: Window-Based Recalibration (Evolutionary Fix)
        
        Modo dual:
        - Si se pasan mae_pct/mfe_pct reales (backtest): usa recalibración por ventana
        - Si solo se pasa pnl_pct (producción): usa EMA gradient descent
        
        QUÉ: Acumula trades y recalibra cada 50 eventos usando percentiles reales.
        POR QUÉ: Gradient descent por-trade oscilaba violentamente y no convergía.
        PARA QUÉ: Convergencia estable de SL/TP basada en distribución real.
        """
        self.trades_processed += 1
        
        # Siempre acumular en ventana
        self.trade_history.append({
            'pnl': pnl_pct,
            'mae': abs(mae_pct),
            'mfe': abs(mfe_pct)
        })
        
        # Mantener ventana de 200 trades
        if len(self.trade_history) > 200:
            self.trade_history.pop(0)
        
        # EMA rewards (compatible con producción)
        reward = 1.0 if pnl_pct > 0 else -1.0
        self.ema_reward = self.alpha * reward + (1 - self.alpha) * self.ema_reward
        self.ema_mae = self.alpha * abs(mae_pct) + (1 - self.alpha) * self.ema_mae
        self.ema_mfe = self.alpha * abs(mfe_pct) + (1 - self.alpha) * self.ema_mfe
        
        # Simple gradient descent para parámetros no-SL/TP (funciona bien trade-by-trade)
        self.params['ml_confidence'] -= self.lr * self.ema_reward * 0.5
        self.params['dd_stress_limit'] += self.lr * self.ema_reward * 0.5
        self.params['cooldown'] -= self.lr * self.ema_reward * 50.0
        
        # Exit Parameters Evolution
        self.params['decay_exit_threshold'] -= self.lr * self.ema_reward * 0.1
        self.params['trail_start_pct'] += self.lr * self.ema_reward * 0.1
        self.params['trail_dist_pct'] -= self.lr * self.ema_reward * 0.05
        self.params['momentum_exit_accel'] += self.lr * self.ema_reward * 0.002
        
        # Recalibración periódica de SL/TP por ventana (más estable que gradient per-trade)
        if self.trades_processed % self.recalibration_interval == 0 and len(self.trade_history) >= 30:
            self._recalibrate()
        else:
            # Fallback: EMA gradient para SL/TP (producción sin MAE/MFE reales)
            if mae_pct == 0 and mfe_pct == 0:
                proxy_mae = min(0, pnl_pct) * 1.5
                proxy_mfe = max(0, pnl_pct) * 1.5
                target_sl_mult = (abs(proxy_mae) * 1.5) / 0.01
                self.params['sl_mult'] = self.alpha * target_sl_mult + (1-self.alpha) * self.params['sl_mult']
                target_tp_mult = (abs(proxy_mfe) * 0.8) / 0.01
                self.params['tp_mult'] = self.alpha * target_tp_mult + (1-self.alpha) * self.params['tp_mult']
        
        self._clip_bounds()
    
    def _recalibrate(self):
        """
        Calcula parámetros óptimos basados en la distribución real de la ventana.
        
        QUÉ: Recalibración periódica cada 50 trades usando percentiles reales de MAE/MFE.
        POR QUÉ: El gradient descent per-trade oscilaba violentamente.
        CÓMO: P80 de MAE para SL, P60 de MFE para TP, WR para confianza.
        """
        pnls = [t['pnl'] for t in self.trade_history]
        maes = [t['mae'] for t in self.trade_history]
        mfes = [t['mfe'] for t in self.trade_history]
        
        wr = len([p for p in pnls if p > 0]) / len(pnls)
        
        # 1. Ajustar Confianza basado en WinRate real
        target_conf = 0.50 + (0.70 - 0.50) * (1.0 - wr)
        self.params['ml_confidence'] = 0.8 * self.params['ml_confidence'] + 0.2 * np.clip(target_conf, 0.45, 0.70)
        
        # 2. Ajustar SL/TP basado en MAE/MFE reales
        # FORENSIC-V36: Usamos divisor 0.01 para que 1.0 = 1% (Consistencia con ml_strategy)
        if maes and any(m > 0 for m in maes):
            target_sl_mult = np.percentile(maes, 80) / 0.01 if np.percentile(maes, 80) > 0 else self.params['sl_mult']
            # Clip to the CORRECT bounds for this profile
            sl_lo, sl_hi = self.r['sl_mult']
            self.params['sl_mult'] = 0.7 * self.params['sl_mult'] + 0.3 * np.clip(target_sl_mult, sl_lo, sl_hi)
            
        if mfes and any(m > 0 for m in mfes):
            target_tp_mult = np.percentile(mfes, 60) / 0.01 if np.percentile(mfes, 60) > 0 else self.params['tp_mult']
            tp_lo, tp_hi = self.r['tp_mult']
            self.params['tp_mult'] = 0.7 * self.params['tp_mult'] + 0.3 * np.clip(target_tp_mult, tp_lo, tp_hi)
            
        # 3. Ajustar cooldown basado en drawdown reciente
        cumulative = np.cumsum(pnls)
        if len(cumulative) > 0:
            recent_drawdown = max(cumulative) - cumulative[-1]
            if recent_drawdown > 0.05:
                self.params['cooldown'] = min(self.r['cooldown'][1], self.params['cooldown'] * 1.2)
            else:
                self.params['cooldown'] = max(self.r['cooldown'][0], self.params['cooldown'] * 0.9)
            
        # Evolution of Exit Parameters (Smooth transition)
        self.params['decay_exit_threshold'] = 0.9 * self.params['decay_exit_threshold'] + 0.1 * np.clip(wr * 0.5, 0.35, 0.65)
        
        self._clip_bounds()

    def feedback_training(self, accuracy):
        """Recibe accuracy de entrenamiento para ajustar el retrain_interval"""
        delta_acc = accuracy - self.last_accuracy
        self.last_accuracy = accuracy
        if delta_acc < -0.05:
            # Drop > 5% -> emergency retrain
            self.params['retrain_interval'] *= 0.8
        elif delta_acc > 0:
            # Stable -> save compute
            self.params['retrain_interval'] *= 1.05
        self._clip_bounds()

    def _clip_bounds(self):
        """Asegura parámetros físicos dentro de rango"""
        for k, bounds in self.r.items():
            if k in self.params:
                self.params[k] = max(bounds[0], min(bounds[1], self.params[k]))
