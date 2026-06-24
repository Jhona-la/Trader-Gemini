import math
import logging

logger = logging.getLogger(__name__)

class ExponentialSizing:
    """
    Motor de Sizing Exponencial (Quarter-Kelly Log-Continuo)
    
    Toma el output de confianza (tensor) de una red neuronal y lo transforma 
    en un tamaño de riesgo fraccional de Kelly optimizado para cuentas micro (Ej. $13 USD).
    Su objetivo es buscar la duplicación de capital (Curva de Doblamiento Exponencial).
    """
    
    def __init__(self, kelly_fraction: float = 0.25, default_b: float = 1.5, min_risk_pct: float = 0.01, max_risk_pct: float = 0.25):
        """
        :param kelly_fraction: 0.25 es Quarter-Kelly, 0.50 Half-Kelly. Recomendado 0.25 para scalping extremo.
        :param default_b: Reward-to-Risk ratio (TP/SL) por defecto esperado por el sistema (ej 1.5).
        :param min_risk_pct: Riesgo mínimo aceptable del bankroll a arriesgar (1%).
        :param max_risk_pct: Riesgo máximo del bankroll por operación (25% en Quarter-Kelly).
        """
        self.kelly_fraction = kelly_fraction
        self.default_b = default_b
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct

    @staticmethod
    def sigmoid(z: float) -> float:
        """ O(1) Sigmoid para mapear logits [-inf, inf] a probabilidades [0, 1] """
        # Limitamos Z para evitar Overflows
        z = max(min(z, 20.0), -20.0)
        return 1.0 / (1.0 + math.exp(-z))

    def calculate_kelly_risk(self, confidence_logit: float, current_capital: float, b: float = None, min_notional: float = 5.0, leverage: int = 1) -> dict:
        """
        :param confidence_logit: El score del oráculo (Ej: 1.37)
        :param current_capital: Balance en USD (Ej: 13.0)
        :param b: Relación de recompensa a riesgo. Si None, usa el por defecto.
        :param min_notional: Tamaño mínimo notional aceptado por el exchange (Ej: $5.0)
        :param leverage: Apalancamiento para el cálculo del notional.
        :return: Dict con métricas del cálculo y el porcentaje/monto a arriesgar.
        """
        import numpy as np
        
        # Enforce float64 precision to prevent drift in exponential compounding
        confidence_logit = np.float64(confidence_logit)
        current_capital = np.float64(current_capital)
        min_notional = np.float64(min_notional)
        
        if b is None:
            b = np.float64(self.default_b)
        else:
            b = np.float64(b)
            
        # Si el logit es negativo, significa que es una señal en dirección contraria (SHORT)
        # Tomamos el valor absoluto para saber la confianza en la dirección elegida
        abs_logit = np.abs(confidence_logit)
        
        # 1. Transformar Confianza en Probabilidad (P)
        p = np.float64(self.sigmoid(float(abs_logit)))
        q = np.float64(1.0) - p
        
        # 2. Calcular Fracción de Kelly (f*)
        # Fórmula: f* = p - (q / b)
        kelly_f = p - (q / b)
        
        # Si la ventaja es negativa (expectancy < 0), el tamaño de la apuesta debe ser cero
        if kelly_f <= 0.0:
            import logging
            logging.getLogger(__name__).warning(f"🛡️ [EXP-SIZING] Rejected NEGATIVE_EXPECTANCY: logit={confidence_logit:.4f}, abs_logit={abs_logit:.4f}, p={p:.4f}, q={q:.4f}, b={b:.4f}, kelly_f={kelly_f:.4f}")
            return {
                "probability": float(p),
                "kelly_f": 0.0,
                "applied_f": 0.0,
                "risk_amount_usd": 0.0,
                "action": "SKIP",
                "reason": "NEGATIVE_EXPECTANCY",
                "diag_b": float(b),
                "diag_p": float(p),
                "diag_q": float(q),
                "diag_logit": float(confidence_logit)
            }
            
        # 3. Aplicar Fraccionalidad (Quarter-Kelly)
        if current_capital < 100.0:
            # 🚀 FULL KELLY FOR MICRO-ACCOUNTS
            applied_f = kelly_f * np.float64(1.0)
        else:
            applied_f = kelly_f * np.float64(self.kelly_fraction)
        
        # 4. Limitar por Risk Management del Microcapital
        # Quitar el techo máximo en micro-cuentas para permitir agresividad total
        if current_capital < 100.0:
            applied_f = np.maximum(np.float64(self.min_risk_pct), applied_f)
            # Cap at 90% purely to avoid total liquidation math errors
            applied_f = np.minimum(applied_f, np.float64(0.90))
        else:
            applied_f = np.maximum(np.float64(self.min_risk_pct), np.minimum(applied_f, np.float64(self.max_risk_pct)))
        
        risk_amount_usd = current_capital * applied_f
        
        # 5. Protección del Capital Mínimo (Mínimo Notional de Binance)
        notional_size = risk_amount_usd * np.float64(leverage)
        
        # 🚀 MICRO-ACCOUNT BINANCE FLOOR EVASION
        if notional_size < min_notional and current_capital < 100.0:
            if current_capital * np.float64(leverage) >= min_notional * 1.2:
                # Forzar el tamaño al floor seguro para pasar la validación
                target_notional = np.float64(min_notional * 1.2)  # $6.00 if min is 5
                notional_size = target_notional
                risk_amount_usd = target_notional / np.float64(leverage)
                applied_f = risk_amount_usd / current_capital
                print(f"⚠️ [EXP-SIZING] Floor Evasion Triggered: Forced notional to ${target_notional:.2f} to bypass Binance limits.")
        
        if notional_size < min_notional:
            return {
                "probability": float(p),
                "kelly_f": float(kelly_f),
                "applied_f": float(applied_f),
                "risk_amount_usd": float(risk_amount_usd),
                "action": "SKIP",
                "reason": f"INSUFFICIENT_NOTIONAL (size: ${notional_size:.2f} < min: ${min_notional:.2f})"
            }
        
        return {
            "probability": float(p),
            "kelly_f": float(kelly_f),
            "applied_f": float(applied_f),
            "risk_amount_usd": float(risk_amount_usd),
            "action": "TRADE",
            "reason": "OPTIMAL"
        }

if __name__ == "__main__":
    # VEREDICTO DEL PRIMER ALIENTO
    engine = ExponentialSizing(kelly_fraction=0.25, default_b=1.5)
    
    capital = 13.0
    oracle_output = 1.37
    
    result = engine.calculate_kelly_risk(oracle_output, capital)
    
    print("=========================================================")
    print("🚀 EL VEREDICTO DEL PRIMER ALIENTO - MOTOR EXPONENCIAL")
    print("=========================================================")
    print(f"Capital Base          : ${capital:.2f} USD")
    print(f"Output del Oráculo    : {oracle_output}")
    print(f"Probabilidad Derivada : {result['probability']*100:.2f}%")
    print(f"Full Kelly F*         : {result['kelly_f']*100:.2f}% del Bankroll")
    print(f"Quarter Kelly (Usado) : {result['applied_f']*100:.2f}% del Bankroll")
    print(f"Dinero a ARRIESGAR    : ${result['risk_amount_usd']:>5.2f} USD (El SL debe enmarcar este límite)")
    print("=========================================================")
