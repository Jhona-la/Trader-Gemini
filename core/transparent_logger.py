import logging
import datetime
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class TransparentLogger:
    """
    Sistema de logging centralizado con visibilidad total de decisiones (Deep Vision).
    Cumple con el estándar de auditoría de la Sección X.
    """
    
    def __init__(self):
        self._setup_logging()
        
    def _setup_logging(self):
        import os
        os.makedirs("dashboard/data", exist_ok=True)
        self.sink_path = "dashboard/data/backtest_thoughts.jsonl"
        
    def _write_to_sink(self, source: str, symbol: str, data: dict):
        """FORENSIC-V43: Persist thoughts to JSONL for post-mortem analysis."""
        import json
        import numpy as np
        from config import Config
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super(NpEncoder, self).default(obj)
                
        try:
            with open(self.sink_path, "a", encoding="utf-8") as f:
                log_entry = {
                    "ts": self._get_timestamp(),
                    "source": source,
                    "symbol": symbol,
                    "is_backtest": getattr(Config, 'IS_BACKTEST', False),
                    "thoughts": data
                }
                f.write(json.dumps(log_entry, cls=NpEncoder, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error in _write_to_sink: {e}")
            
    def _get_timestamp(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def log_technical_signal(self, symbol, timeframe, price, signal, strength, analysis, indicators, confluence):
        """
        Nivel 1 - DECISIONES ESTRATÉGICAS (Technical)
        """
        self._write_to_sink("TECHNICAL", symbol, {
            "timeframe": timeframe,
            "price": price,
            "signal": signal,
            "strength": strength,
            "analysis": analysis,
            "indicators": indicators,
            "confluence": confluence
        })
        
        color = Fore.GREEN if signal == "BUY" else (Fore.RED if signal == "SELL" else Fore.YELLOW)
        
        print(f"\n{Style.BRIGHT}═══════════════════════════════════════════════════════════")
        print(f"📊 [SIGNAL GENERATED] {symbol} | {self._get_timestamp()}")
        print(f"═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}🎯 ESTRATEGIA: Technical Confluence{Style.RESET_ALL}")
        print(f"──────────────────────────────────────────")
        print(f"├─ 📍 PAR:           {symbol}")
        print(f"├─ ⏰ TIMEFRAME:     {timeframe}")
        print(f"├─ 📊 PRECIO ACTUAL: ${price:.2f}")
        print(f"├─ 🎲 SEÑAL:         {color}{signal}{Style.RESET_ALL}")
        print(f"└─ 💪 FUERZA:        {strength:.2f} ⭐⭐⭐⭐⭐")
        
        print(f"\n{Fore.CYAN}🔬 ANÁLISIS TÉCNICO COMPLETO{Style.RESET_ALL}")
        print(f"──────────────────────────────────────────")
        
        # Tendencia
        trend = analysis.get('trend', {})
        print(f"{Style.BRIGHT}📈 TENDENCIA:{Style.RESET_ALL}")
        print(f"   ├─ Timeframe actual:  {trend.get('current', 'N/A')}")
        print(f"   ├─ Timeframe superior: {trend.get('higher', 'N/A')}")
        print(f"   └─ ADX Strength:      {trend.get('adx', 0):.1f}")
        
        # Indicadores
        print(f"\n{Style.BRIGHT}📊 INDICADORES CORE:{Style.RESET_ALL}")
        rsi = indicators.get('rsi', {})
        print(f"   ├─ RSI (14):          {rsi.get('value', 0):.1f} → {rsi.get('status', 'Neutral')}")
        
        macd = indicators.get('macd', {})
        print(f"   ├─ MACD:              Hist: {macd.get('hist', 0):.4f} | Signal: {macd.get('signal', 'N/A')}")
        
        bb = indicators.get('bb', {})
        print(f"   ├─ Bollinger Bands:   %B: {bb.get('pct_b', 0):.2f}")
        
        # Confluencia
        print(f"\n{Style.BRIGHT}🔗 CONFLUENCIA:{Style.RESET_ALL}")
        print(f"   ├─ Indicadores alineados: {confluence.get('aligned_count', 0)}/{confluence.get('total_count', 5)}")
        print(f"   ├─ Score de confluencia:  {confluence.get('score', 0):.2f}")
        print(f"   └─ Nivel de confianza:    {confluence.get('confidence', 'MEDIUM')}")
        
        print(f"\n{Style.BRIGHT}🎯 DECISIÓN FINAL: {color}{signal}{Style.RESET_ALL}")
        reason = analysis.get('reason', 'N/A')
        print(f"   └─ Razón: {reason}")
        print("──────────────────────────────────────────\n")

    def log_ml_prediction(self, symbol, model_name, prediction, confidence, features, decision):
        """
        Nivel 2 - ESTRATEGIAS DE INTELIGENCIA ARTIFICIAL (ML)
        """
        self._write_to_sink("ML", symbol, {
            "model_name": model_name,
            "prediction": prediction,
            "confidence": confidence,
            "features": features,
            "decision": decision
        })
        
        color = Fore.GREEN if decision == "LONG" else (Fore.RED if decision == "SHORT" else Fore.YELLOW)
        
        print(f"\n{Style.BRIGHT}═══════════════════════════════════════════════════════════")
        print(f"🧠 [ML PREDICTION] {symbol} | {self._get_timestamp()}")
        print(f"═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        print(f"\n{Fore.MAGENTA}🤖 MODELO: {model_name}{Style.RESET_ALL}")
        print(f"──────────────────────────────────────────")
        print(f"├─ 🎯 Predicción:     {prediction:+.5f}")
        print(f"├─ 📊 Confidence:     {confidence:.1%}")
        
        print(f"\n{Style.BRIGHT}📊 FEATURES CLAVE:{Style.RESET_ALL}")
        print(f"──────────────────────────────────────────")
        for k, v in features.items():
            if isinstance(v, (int, float)):
                print(f"   ├─ {k}: {v:.4f}")
            else:
                print(f"   ├─ {k}: {v}")
            
        print(f"\n{Style.BRIGHT}🎯 RECOMENDACIÓN FINAL: {color}{decision}{Style.RESET_ALL}")
        print(f"   └─ Fuerza: {prediction:.4f}")
        print("──────────────────────────────────────────\n")

    def log_sniper_analysis(self, symbol, layers):
        """
        Nivel 3 - ESTRATEGIA SNIPER
        """
        self._write_to_sink("SNIPER", symbol, {
            "layers": layers
        })
        
        print(f"\n{Style.BRIGHT}═══════════════════════════════════════════════════════════")
        print(f"🎯 [SNIPER MODE] {symbol} | {self._get_timestamp()}")
        print(f"═══════════════════════════════════════════════════════════{Style.RESET_ALL}")
        
        # Layer A
        la = layers.get('A', {})
        print(f"\n{Fore.YELLOW}🎯 LAYER A - TECHNICAL CONFLUENCE:{Style.RESET_ALL}")
        print(f"   └─ Score: {la.get('score', 0)}/3 ({la.get('status', 'FAIL')})")
        
        # Layer B
        lb = layers.get('B', {})
        print(f"\n{Fore.YELLOW}🔍 LAYER B - ORDER BOOK ANALYSIS:{Style.RESET_ALL}")
        print(f"   ├─ Imbalance: {lb.get('imbalance', 0):+.2f}")
        print(f"   └─ Status:    {lb.get('signal', 'NEUTRAL')}")
        
        # Layer C
        lc = layers.get('C', {})
        print(f"\n{Fore.YELLOW}🐳 LAYER C - WHALE DETECTION:{Style.RESET_ALL}")
        print(f"   ├─ Anomalía Vol: {lc.get('z_score', 0):.1f}σ")
        print(f"   └─ Status:       {'WHALE' if lc.get('is_anomaly') else 'NORMAL'}")
        
        # Total
        print(f"\n{Style.BRIGHT}🎯 CONFLUENCE TOTAL:{Style.RESET_ALL}")
        print(f"──────────────────────────────────────────")
        trigger = layers.get('trigger', {})
        sig = trigger.get('signal', 'HOLD')
        col = Fore.GREEN if sig == "LONG" else (Fore.RED if sig == "SHORT" else Fore.WHITE)
        
        print(f"   🔥 TRIGGER DECISION: {col}{sig}{Style.RESET_ALL}")
        print("──────────────────────────────────────────\n")

# Global Instance
monitor_log = TransparentLogger()
