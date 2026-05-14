"""
═══════════════════════════════════════════════════════════════════════
📉 SWING DCA ENGINE — Dollar Cost Averaging Automático para Swing
═══════════════════════════════════════════════════════════════════════

QUÉ: Motor proactivo que evalúa posiciones Swing en drawdown y genera
     señales de DCA (promediar precio) escalonadas.

POR QUÉ: Con $13 USD y sizing Swing al 30%, queda 70% de pólvora seca
     en el silo. Sin DCA, una posición Swing en drawdown del -3% queda
     atrapada esperando que el mercado recupere TODO el movimiento.
     Con DCA, el avg_price baja → el TP se acerca → recuperación más rápida.

PARA QUÉ: Transformar el 70% de margen libre Swing en un mecanismo
     de defensa activa que convierte drawdowns en oportunidades.

CÓMO: 3 layers escalonados (-2%, -4%, -6%) con validaciones de:
     1. Kill Switch / Régimen de mercado
     2. ATR safety (no promediar en cisne negro)
     3. Cooldown entre DCAs (30 min)
     4. Margen disponible mínimo ($0.50)
     5. Límite de layers (máx 3)

CUÁNDO: Evaluado en cada tick de check_stops() para posiciones SWING.

DÓNDE: core/swing_dca_engine.py → invocado desde risk_manager.py::check_stops()

QUIÉN: SwingDCAEngine (trigger) + RiskManager (sizing) + Portfolio (avg_price)

DEPENDENCIAS:
    - Config.Strategies.DCA → Parámetros de configuración
    - Portfolio.virtual_ledger → Estado de posiciones Swing
    - Portfolio.get_available_cash(horizon='SWING') → Margen disponible
    - RiskManager.kill_switch → Estado de emergencia
    - utils.logger → Logging estructurado
═══════════════════════════════════════════════════════════════════════
"""

import time
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from config import Config
from core.events import SignalEvent, SignalType
from utils.logger import logger


class SwingDCAEngine:
    """
    Motor de DCA automático exclusivo para posiciones Swing.
    Genera SignalEvents de tipo LONG/SHORT (misma dirección) cuando
    se cumplen las condiciones de drawdown escalonado.
    """

    def __init__(self):
        # Config shortcuts (leídos una vez, Config es inmutable en runtime)
        dca_cfg = getattr(Config.Strategies, 'DCA', None)
        self.enabled = getattr(dca_cfg, 'ENABLED', True) if dca_cfg else True
        self.max_layers = getattr(dca_cfg, 'MAX_LAYERS', 3) if dca_cfg else 3
        self.triggers = getattr(dca_cfg, 'TRIGGERS', [-0.020, -0.040, -0.060]) if dca_cfg else [-0.020, -0.040, -0.060]
        self.size_mults = getattr(dca_cfg, 'SIZE_MULTS', [0.25, 0.30, 0.35]) if dca_cfg else [0.25, 0.30, 0.35]
        self.cooldown_s = getattr(dca_cfg, 'COOLDOWN_SECONDS', 1800) if dca_cfg else 1800
        self.regime_block_bear = getattr(dca_cfg, 'REGIME_BLOCK_BEAR', True) if dca_cfg else True
        self.atr_safety_mult = getattr(dca_cfg, 'ATR_SAFETY_MULT', 2.5) if dca_cfg else 2.5
        self.recalc_tp = getattr(dca_cfg, 'RECALC_TP', True) if dca_cfg else True
        self.min_margin = getattr(dca_cfg, 'MIN_MARGIN_FOR_DCA', 0.50) if dca_cfg else 0.50

        # ── Estado interno ──
        # Tracking de DCA ejecutados por posición: {v_key: {'layers': int, 'last_dca_ts': float}}
        self._dca_state: Dict[str, Dict[str, Any]] = {}

    # ══════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def evaluate(
        self,
        v_key: str,
        pos: Dict[str, Any],
        symbol: str,
        current_price: float,
        available_cash_swing: float,
        global_regime: str,
        kill_switch_active: bool,
        atr_current: float = 0.0,
        atr_average: float = 0.0,
        now: Optional[datetime] = None,
        sophia_intelligence=None,
        returns: Optional[np.ndarray] = None,
    ) -> Optional[SignalEvent]:
        """
        Evalúa si una posición Swing debe recibir un DCA.

        Args:
            v_key: Clave del virtual_ledger (e.g. "BTC/USDT_SWING_LONG")
            pos: Diccionario de posición del virtual_ledger
            symbol: Símbolo de trading (e.g. "BTC/USDT")
            current_price: Precio actual del activo
            available_cash_swing: Cash disponible en el silo Swing
            global_regime: Régimen global de mercado (TRENDING_BULL/BEAR, etc.)
            kill_switch_active: Si el kill switch está activo
            atr_current: ATR actual del activo (para safety check)
            atr_average: ATR promedio histórico (para comparación)
            now: Timestamp actual (para tests/backtest)
            sophia_intelligence: Motor de IA predictiva (opcional)
            returns: Array de retornos recientes para análisis de Sophia (opcional)

        Returns:
            SignalEvent si se debe ejecutar DCA, None si no.
        """
        # ── Gate 0: Habilitado ──
        if not self.enabled:
            return None

        # ── Gate 1: Kill Switch ──
        if kill_switch_active:
            return None

        qty = pos.get('quantity', 0.0)
        if abs(qty) < 1e-8:
            return None

        entry_price = pos.get('avg_price', 0.0)
        if entry_price <= 0 or current_price <= 0:
            return None

        # ── Gate 2: Solo posiciones SWING ──
        horizon = pos.get('horizon', 'SCALPING')
        if horizon != 'SWING':
            return None

        # ── Calcular PnL no realizado ──
        if qty > 0:  # LONG
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            unrealized_pnl_pct = (entry_price - current_price) / entry_price

        # Si la posición es rentable, no hay DCA necesario
        if unrealized_pnl_pct >= 0:
            return None

        # ── Gate 3: Régimen de mercado ──
        if self.regime_block_bear:
            if qty > 0 and global_regime == 'TRENDING_BEAR':
                logger.debug(f"🚫 [DCA] {symbol} SWING LONG blocked: Bear market regime")
                return None
            if qty < 0 and global_regime == 'TRENDING_BULL':
                logger.debug(f"🚫 [DCA] {symbol} SWING SHORT blocked: Bull market regime")
                return None

        # ── Gate 4: ATR Safety (no promediar en cisne negro) ──
        if atr_current > 0 and atr_average > 0:
            atr_ratio = atr_current / atr_average
            if atr_ratio > self.atr_safety_mult:
                logger.warning(
                    f"🦢 [DCA] {symbol} BLOCKED: ATR Spike ({atr_ratio:.1f}x > {self.atr_safety_mult}x). "
                    f"Possible Black Swan. NO averaging."
                )
                return None

        # ── Gate 5: Layers máximos ──
        state = self._dca_state.get(v_key, {'layers': 0, 'last_dca_ts': 0.0})
        current_layers = state['layers']
        if current_layers >= self.max_layers:
            return None

        # ── Gate 6: Cooldown entre DCAs ──
        if now is None:
            now = datetime.now(timezone.utc)
        now_ts = now.timestamp() if hasattr(now, 'timestamp') else float(now)
        last_dca_ts = state['last_dca_ts']
        if last_dca_ts > 0 and (now_ts - last_dca_ts) < self.cooldown_s:
            remaining = self.cooldown_s - (now_ts - last_dca_ts)
            logger.debug(f"⏳ [DCA] {symbol} cooldown: {remaining:.0f}s remaining")
            return None

        # ── Gate 7: Margen disponible ──
        if available_cash_swing < self.min_margin:
            logger.debug(f"💸 [DCA] {symbol} blocked: Insufficient margin (${available_cash_swing:.2f} < ${self.min_margin:.2f})")
            return None

        # ══════════════════════════════════════════════════════════════
        # DETERMINAR LAYER DE DCA
        # ══════════════════════════════════════════════════════════════
        next_layer_idx = current_layers  # 0-indexed: layer 0 = primer DCA
        if next_layer_idx >= len(self.triggers):
            return None

        trigger_pct = self.triggers[next_layer_idx]
        size_mult = self.size_mults[next_layer_idx] if next_layer_idx < len(self.size_mults) else 0.20

        # ¿Se alcanzó el umbral de drawdown para este layer?
        if unrealized_pnl_pct > trigger_pct:
            return None  # Aún no suficiente drawdown

        # ══════════════════════════════════════════════════════════════
        # 🧠 SOPHIA AI: Validación de DCA (Módulo Predictivo)
        # ══════════════════════════════════════════════════════════════
        direction_str = "LONG" if qty > 0 else "SHORT"
        signal_type = SignalType.LONG if qty > 0 else SignalType.SHORT
        layer_num = next_layer_idx + 1
        
        sophia_report_dict = None
        if sophia_intelligence is not None:
            # Pseudo-setups para DCA (asume que la estructura es neutral pero el rebote es el focus)
            pseudo_setups = {'rsi': 50.0, 'bb_position': 0.5, 'confluence': 0.8}
            
            try:
                report = sophia_intelligence.analyze(
                    symbol=symbol,
                    direction=direction_str,
                    signal_strength=0.90, # Fuerte convicción por ser DCA
                    setups=pseudo_setups,
                    confluence_score=0.8,
                    tp_pct=pos.get('tp_pct', 0.045),
                    sl_pct=Config.Strategies.SWING_PARAMS['sl_pct'],
                    returns=returns,
                    ttl_seconds=self.cooldown_s,
                    regime=global_regime
                )
                
                # Para DCA, requerimos solo una probabilidad > 50% (reversión a la media)
                dca_min_prob = 0.50
                if report.win_probability < dca_min_prob:
                    logger.warning(f"🧠 [SOPHIA-DCA] VETOED {symbol} {direction_str} L{layer_num}: P(Win)={report.win_probability:.1%} < {dca_min_prob:.1%} (Entropy: {report.entropy_label})")
                    return None
                    
                logger.info(f"🧠 [SOPHIA-DCA] APPROVED {symbol} {direction_str} L{layer_num}: P(Win)={report.win_probability:.1%}")
                sophia_report_dict = report.to_dict()
                
            except Exception as e:
                logger.error(f"Error in Sophia DCA analysis for {symbol}: {e}")
                # En caso de error, permitimos el DCA fallback asumiendo que es una falla técnica no predictiva
                pass

        # ══════════════════════════════════════════════════════════════
        # 🎯 TRIGGER DCA — Generar señal
        # ══════════════════════════════════════════════════════════════
        # Calcular sizing: fracción del margen disponible
        dca_margin = available_cash_swing * size_mult
        leverage = pos.get('leverage', getattr(Config, 'BINANCE_LEVERAGE', 10))
        dca_notional = dca_margin * leverage

        if dca_notional < 5.0:
            # Pad to minimum Binance notional
            dca_notional = 6.0
            dca_margin = dca_notional / leverage

        if dca_margin > available_cash_swing:
            logger.warning(f"🚫 [DCA-L{layer_num}] {symbol} margin needed ${dca_margin:.2f} > available ${available_cash_swing:.2f}")
            return None

        dca_qty = dca_notional / current_price if current_price > 0 else 0
        if dca_qty <= 0:
            return None

        # ── Calcular nuevo avg_price proyectado ──
        old_notional = abs(qty) * entry_price
        new_notional = old_notional + (dca_qty * current_price)
        new_qty = abs(qty) + dca_qty
        projected_avg = new_notional / new_qty if new_qty > 0 else entry_price

        # ── Calcular nuevo TP si está habilitado ──
        tp_pct = pos.get('tp_pct', 0.045)
        if self.recalc_tp and tp_pct > 0:
            # TP se mantiene como % del nuevo avg_price
            # Pero recalculamos el precio absoluto de TP
            if qty > 0:
                old_tp_price = entry_price * (1 + tp_pct)
                # Mantener el mismo precio objetivo de TP pero recalcular el %
                new_tp_pct = (old_tp_price - projected_avg) / projected_avg
                # Cap: nunca menor que 1.5% para Swing
                new_tp_pct = max(new_tp_pct, 0.015)
            else:
                old_tp_price = entry_price * (1 - tp_pct)
                new_tp_pct = (projected_avg - old_tp_price) / projected_avg
                new_tp_pct = max(new_tp_pct, 0.015)
        else:
            new_tp_pct = tp_pct

        logger.info(
            f"📉 [DCA-L{layer_num}] {symbol} SWING {signal_type.name} | "
            f"Drawdown: {unrealized_pnl_pct*100:.2f}% (trigger: {trigger_pct*100:.1f}%) | "
            f"Adding ${dca_notional:.2f} notional ({dca_qty:.6f} qty) | "
            f"Avg: ${entry_price:.2f} → ${projected_avg:.2f} | "
            f"TP: {tp_pct*100:.2f}% → {new_tp_pct*100:.2f}% | "
            f"Layer {layer_num}/{self.max_layers}"
        )

        # ── Actualizar estado interno ──
        self._dca_state[v_key] = {
            'layers': current_layers + 1,
            'last_dca_ts': now_ts,
        }

        # ── Construir Metadata ──
        meta = {
            'is_dca': True,
            'dca_layer': layer_num,
            'dca_max_layers': self.max_layers,
            'dca_trigger_pct': trigger_pct,
            'dca_size_mult': size_mult,
            'dca_qty': dca_qty,
            'dca_margin': dca_margin,
            'dca_notional': dca_notional,
            'projected_avg_price': projected_avg,
            'new_tp_pct': new_tp_pct,
            'old_avg_price': entry_price,
            'unrealized_pnl_pct': unrealized_pnl_pct,
        }
        
        if sophia_report_dict:
            meta['sophia'] = sophia_report_dict

        # ── Construir SignalEvent ──
        return SignalEvent(
            strategy_id="DCA_SWING",
            symbol=symbol,
            datetime=now,
            signal_type=signal_type,
            strength=0.90,  # Alta convicción: misma dirección que la tesis original
            horizon='SWING',
            priority=2,  # No urgente (Limit order)
            metadata=meta,
            tp_pct=new_tp_pct,
        )

    def get_dca_state(self, v_key: str) -> Dict[str, Any]:
        """Retorna estado de DCA para una posición."""
        return self._dca_state.get(v_key, {'layers': 0, 'last_dca_ts': 0.0})

    def reset_position(self, v_key: str):
        """
        Resetea el estado de DCA cuando una posición se cierra.
        Debe llamarse desde Portfolio._record_closed_trade() o check_stops() al cerrar.
        """
        if v_key in self._dca_state:
            old = self._dca_state.pop(v_key)
            if old['layers'] > 0:
                logger.info(f"🔄 [DCA] Reset state for {v_key} (was at layer {old['layers']})")

    def reset_all(self):
        """Reset completo (nuevo session)."""
        self._dca_state.clear()
        logger.info("🔄 [DCA] All DCA state cleared")

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Snapshot de todos los estados DCA (para dashboard/telemetría)."""
        return dict(self._dca_state)


# ── Singleton global ──
swing_dca_engine = SwingDCAEngine()
