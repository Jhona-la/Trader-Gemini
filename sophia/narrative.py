"""
📝 SOPHIA-INTELLIGENCE §4.1: Narrative Generator

QUÉ: Genera narrativa en lenguaje humano explicando la intención del bot.
POR QUÉ: Los logs numéricos son para máquinas. Los humanos necesitan
     entender QUÉ piensa el bot en un vistazo rápido.
PARA QUÉ: "Abro LONG en BTC porque detecto absorción institucional en $50,000,
     con una confianza del 78% y un horizonte de 12 minutos."
CÓMO: Template engine parametrizado con datos del SophiaReport.
CUÁNDO: Después de analyze(), antes de emitir SignalEvent.
DÓNDE: sophia/narrative.py
QUIÉN: NarrativeGenerator, invocado por SophiaIntelligence o directamente.
"""

from typing import Dict, List, Optional
from utils.logger import logger


class NarrativeGenerator:
    """
    📝 SOPHIA §4.1: Human-Readable Trade Intention Narrator.
    
    QUÉ: Transforma datos numéricos del SophiaReport en narrativa humana.
    POR QUÉ: Un trader debe poder leer el log y entender la decisión en 5 segundos.
    PARA QUÉ: Trazabilidad, auditoría, y debugging visual.
    CÓMO: Templates parametrizados → string format con datos del report.
    """
    
    # Setup type descriptions
    SETUP_DESCRIPTIONS = {
        'long_mean_rev': 'reversión a la media alcista (precio en extremo inferior BB + RSI oversold)',
        'short_mean_rev': 'reversión a la media bajista (precio en extremo superior BB + RSI overbought)',
        'long_momentum': 'momentum alcista (MACD acelerando + tendencia UP confirmada)',
        'short_momentum': 'momentum bajista (MACD decelerando + tendencia DOWN confirmada)',
    }
    
    # Feature explanations
    FEATURE_EXPLANATIONS = {
        'RSI': 'fuerza del indicador RSI',
        'BB Position': 'posición en Bandas de Bollinger',
        'ADX': 'fuerza de la tendencia (ADX)',
        'Volume Ratio': 'ratio de volumen vs media',
        'MTF Confluence': 'confluencia multi-timeframe',
        'MACD Histogram': 'aceleración MACD',
        'Trend Alignment': 'alineación con tendencia mayor',
        'ATR %': 'volatilidad relativa (ATR%)',
    }
    
    @staticmethod
    def detect_setup_type(setups: Dict) -> str:
        """Detect which setup triggered the signal."""
        if setups.get('long_mean_rev'):
            return 'long_mean_rev'
        elif setups.get('short_mean_rev'):
            return 'short_mean_rev'
        elif setups.get('long_momentum'):
            return 'long_momentum'
        elif setups.get('short_momentum'):
            return 'short_momentum'
        return 'unknown'
    
    @staticmethod
    def generate_intention(
        symbol: str,
        direction: str,
        win_prob: float,
        expected_exit_mins: float,
        top_features: List[Dict],
        setups: Dict,
        entropy_label: str,
        tail_warning: bool,
        current_price: float = 0.0,
    ) -> str:
        """
        Generate human-readable trade intention narrative.
        
        QUÉ: La narrativa completa del "pensamiento del bot".
        CÓMO: Template → format → string.
        
        Returns:
            String like: "Abro LONG en BTC/USDT porque detecto reversión a la media
            alcista con confianza del 78% y horizonte de 12 min. Factores principales:
            RSI(+0.12), Volume(+0.08). Decisión: Alta Convicción."
        """
        dir_es = "LONG (compra)" if direction == "LONG" else "SHORT (venta)"
        
        # Setup explanation
        setup_key = NarrativeGenerator.detect_setup_type(setups)
        setup_desc = NarrativeGenerator.SETUP_DESCRIPTIONS.get(
            setup_key, 
            'señal técnica combinada'
        )
        
        # Price context
        price_str = f" a ${current_price:,.2f}" if current_price > 0 else ""
        
        # Top features narrative
        feat_parts = []
        for f in top_features[:3]:
            sign = "+" if f['contribution'] > 0 else ""
            feat_parts.append(f"{f['feature']}({sign}{f['contribution']:.3f})")
        features_str = ", ".join(feat_parts) if feat_parts else "N/A"
        
        # Warnings
        warnings = []
        if entropy_label == "Indeciso":
            warnings.append("⚠️ Decisión con alta incertidumbre")
        if tail_warning:
            warnings.append("⚠️ Fat tails detectadas — riesgo de cola elevado")
        warnings_str = ". ".join(warnings)
        if warnings_str:
            warnings_str = f" {warnings_str}."
        
        # Build narrative
        narrative = (
            f"Abro {dir_es} en {symbol}{price_str} porque detecto {setup_desc}, "
            f"con una confianza del {win_prob:.0%} y un horizonte de "
            f"{expected_exit_mins:.0f} minutos. "
            f"Factores principales: {features_str}. "
            f"Decisión: {entropy_label}."
            f"{warnings_str}"
        )
        
        return narrative
    
    @staticmethod
    def generate_post_mortem_narrative(
        symbol: str,
        direction: str,
        predicted_prob: float,
        actual_outcome: str,  # "WIN" or "LOSS"
        brier_score: float,
        pnl: float,
        duration_mins: float,
        predicted_exit_mins: float,
    ) -> str:
        """
        Generate post-mortem comparison narrative.
        
        QUÉ: Compara la predicción del bot contra el resultado real.
        """
        outcome_emoji = "✅" if actual_outcome == "WIN" else "❌"
        
        time_accuracy = ""
        if predicted_exit_mins > 0:
            time_ratio = duration_mins / predicted_exit_mins
            if 0.5 <= time_ratio <= 2.0:
                time_accuracy = "Estimación temporal PRECISA."
            elif time_ratio < 0.5:
                time_accuracy = "Salió ANTES de lo estimado."
            else:
                time_accuracy = "Tardó MÁS de lo estimado."
        
        # Calibration assessment
        if brier_score < 0.05:
            cal_status = "EXCELENTE 🎯"
        elif brier_score < 0.15:
            cal_status = "BUENA ✅"
        elif brier_score < 0.25:
            cal_status = "ACEPTABLE ⚠️"
        else:
            cal_status = "POBRE ❌ (recalibrar modelo)"
        
        narrative = (
            f"{outcome_emoji} POST-MORTEM {symbol} {direction}: "
            f"Predicho {predicted_prob:.0%} de éxito → Resultado: {actual_outcome} "
            f"(PnL=${pnl:+.4f}). "
            f"Brier Score: {brier_score:.4f}. "
            f"Calibración: {cal_status}. "
            f"Duración: {duration_mins:.1f}min (estimado: {predicted_exit_mins:.1f}min). "
            f"{time_accuracy}"
        )
        
        return narrative
    
    @staticmethod
    def format_intention_report(
        trade_id: str,
        win_prob: float,
        expected_exit_mins: float,
        top_features: List[Dict],
        entropy: float,
        narrative: str,
    ) -> str:
        """
        Format the structured intention report per protocol spec.
        
        Output format:
        [ID] | [P: XX.X%] | [T: X min]
        [Top 3 Features] | [H: X.XX]
        "Pensamiento del Bot": [Narrative]
        """
        top3 = ", ".join(
            f"{f['feature']}={f['contribution']:+.3f}" 
            for f in top_features[:3]
        ) or "N/A"
        
        report = (
            f"┌─── SOPHIA INTENTION REPORT ───────────────────┐\n"
            f"│ [{trade_id[:8]}] | P(Win): {win_prob:.1%} | E[T]: {expected_exit_mins:.0f} min\n"
            f"│ Top-3: [{top3}] | Entropía: {entropy:.2f}\n"
            f"│ 💭 \"{narrative}\"\n"
            f"└───────────────────────────────────────────────┘"
        )
        
        return report
