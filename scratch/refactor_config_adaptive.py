import re
import os

CONFIG_PATH = "config.py"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    content = f.read()

symbol_profiles_start = content.find("class SymbolProfiles:")
exit_marker = content.find("# MANDATORY TAGGING SYSTEM - SCALPING VS SWING")

if symbol_profiles_start == -1 or exit_marker == -1:
    print("No se encontraron los marcadores.")
    exit(1)

adaptive_engine_code = '''class AdaptiveProfileEngine:
        """
        Motor Adaptativo y Evolutivo: Genera perfiles dinámicamente para TODAS las monedas,
        dependiendo de su categoría y horizonte, reemplazando diccionarios quemados.
        """
        @classmethod
        def _get_category(cls, symbol: str) -> str:
            majors = ["BTC/USDT", "ETH/USDT"]
            memes = ["DOGE/USDT", "SHIB/USDT", "PEPE/USDT", "BONK/USDT", "FLOKI/USDT", "WIF/USDT"]
            if symbol in majors:
                return "MAJOR"
            elif symbol in memes:
                return "MEME"
            else:
                return "ALT"

        @classmethod
        def get(cls, symbol: str, horizon: str = "SCALPING") -> dict:
            category = cls._get_category(symbol)
            is_scalp = (horizon.upper() in ["SCALPING", "MICROSCALPING"])
            
            if category == "MAJOR":
                base_lev = 20 if is_scalp else 15
                max_risk = 0.05 if is_scalp else 0.03
                kelly_l = 1.0; kelly_s = 0.90
                pullback_tol = 0.40
            elif category == "MEME":
                base_lev = 5 if is_scalp else 3
                max_risk = 0.02 if is_scalp else 0.015
                kelly_l = 0.60; kelly_s = 0.50
                pullback_tol = 0.30
            else:
                base_lev = 15 if is_scalp else 10
                max_risk = 0.04 if is_scalp else 0.02
                kelly_l = 0.85; kelly_s = 0.75
                pullback_tol = 0.45

            profile = {
                "category": category,
                "base_leverage": base_lev,
                "max_risk_pct": max_risk,
                "kelly_factor_long": kelly_l,
                "kelly_factor_short": kelly_s,
                "min_confidence": 0.45 if category == "MAJOR" else 0.50,
                "long_bias": 0.0,
                "short_bias": 0.0,
                "pullback_tol": pullback_tol,
                "trail_protect_atr": 0.9 if is_scalp else 1.3,
                "trail_pursue_atr": 1.4 if is_scalp else 2.0,
                "trail_capture_atr": 0.5,
                "trail_runner": 1.0,
                "momentum_sensitivity": 1.0,
                "sl_mult": 1.4 if is_scalp else 1.6,
                "tp_mult": 1.0 if is_scalp else 1.2,
                "atr_stop_long": 1.5 if is_scalp else 2.0,
                "atr_stop_short": 1.8 if is_scalp else 2.2,
                "sc_min": 78 if is_scalp else 75, 
                "sl_min": 22 if is_scalp else 25, 
                "alpha_decay_lambda": 0.05 if is_scalp else 0.01,
                "alpha_max_ttl": 60.0 if is_scalp else 360.0,
                "rsi_overbought": 78 if is_scalp else 75,
                "rsi_oversold": 22 if is_scalp else 25
            }
            return profile

    SymbolProfiles = AdaptiveProfileEngine
    
    class Trailing:
        @classmethod
        def get_asset_profile(cls, symbol: str) -> dict:
            return AdaptiveProfileEngine.get(symbol, "SCALPING")
            
        @classmethod
        def get_family_profile(cls, family: str) -> dict:
            return {"r1_pct": 0.50, "r2_pct": 0.30, "runner_pct": 0.20}

    '''

new_content = content[:symbol_profiles_start] + adaptive_engine_code + content[exit_marker:]

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    f.write(new_content)

print("Config.py refactored successfully to use AdaptiveProfileEngine.")
