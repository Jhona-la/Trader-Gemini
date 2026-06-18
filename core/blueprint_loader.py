import os
import glob
import json
from datetime import datetime
from config import Config
from utils.logger import setup_logger

logger = setup_logger("BlueprintLoader")

class BlueprintLoader:
    """
    [PHASE 7] EVOLUTIONARY INJECTION ENGINE
    Lee el archivo omni_evolver_best_*.json más reciente generado por el Omni-Evolver 
    e inyecta sus optimizaciones cuánticas directamente en la configuración de producción.
    """
    
    @staticmethod
    def load_latest_blueprint(data_dir: str = "data") -> bool:
        """
        Escanea la carpeta de datos, encuentra el blueprint más fresco y
        aplica las mutaciones al sistema en nanosegundos antes de iniciar el motor.
        """
        pattern = os.path.join(data_dir, "omni_evolver_best_*.json")
        files = glob.glob(pattern)
        
        if not files:
            logger.info("ℹ️ [Blueprint Loader] No evolution blueprints found. Using base config.")
            return False
            
        # Ordenar por fecha de modificación descendente (el más nuevo primero)
        files.sort(key=os.path.getmtime, reverse=True)
        latest_file = files[0]
        
        try:
            with open(latest_file, 'r') as f:
                blueprint = json.load(f)
                
            logger.info(f"🧬 [Blueprint Loader] Injecting Evolutionary Genome from: {os.path.basename(latest_file)}")
            BlueprintLoader._inject_mutations(blueprint)
            return True
            
        except Exception as e:
            logger.error(f"❌ [Blueprint Loader] Failed to load blueprint: {str(e)}")
            return False
            
    @staticmethod
    def _inject_mutations(blueprint: dict):
        import os
        from config.adaptive_config import adaptive_config
        
        # 1. RISK MUTATIONS
        if "BLUEPRINT_RISK" in blueprint:
            r = blueprint["BLUEPRINT_RISK"]
            if "cvar_confidence" in r:
                setattr(Config.Risk, "CVAR_CONFIDENCE_OVERRIDE", r["cvar_confidence"])
            if "max_sector_exposure_micro" in r:
                setattr(Config.Risk, "MAX_SECTOR_MICRO", r["max_sector_exposure_micro"])
            if "max_sector_exposure_scalp" in r:
                setattr(Config.Risk, "MAX_SECTOR_SCALP", r["max_sector_exposure_scalp"])
            if "max_sector_exposure_swing" in r:
                setattr(Config.Risk, "MAX_SECTOR_SWING", r["max_sector_exposure_swing"])
            if "daily_drawdown_limit" in r:
                Config.Risk.MAX_DRAWDOWN = r["daily_drawdown_limit"] * 100.0  # Convert to percentage
                
        # 2. STRATEGY PARAMETERS (Injected into Config.Strategies.Mutations for strategies/technical.py)
        if not hasattr(Config.Strategies, 'Mutations'):
            Config.Strategies.Mutations = {}
            
        if "BLUEPRINT_SNIPER" in blueprint:
            sn = blueprint["BLUEPRINT_SNIPER"]
            Config.Strategies.Mutations['sniper_vol_mult'] = sn.get('volume_spike_multiplier', 1.5)
            Config.Strategies.Mutations['sniper_abs_pct'] = sn.get('absorption_threshold_pct', 0.5)
            
        if "BLUEPRINT_TECHNICAL" in blueprint:
            tech = blueprint["BLUEPRINT_TECHNICAL"]
            Config.Strategies.Mutations['rsi_buy'] = tech.get('rsi_oversold', 30)
            Config.Strategies.Mutations['rsi_sell'] = tech.get('rsi_overbought', 70)
            Config.Strategies.Mutations['ema_fast'] = tech.get('macd_fast', 12)
            Config.Strategies.Mutations['ema_slow'] = tech.get('macd_slow', 26)
            Config.Strategies.Mutations['ema_trend'] = tech.get('ema_trend_window', 200)

        if "BLUEPRINT_PATTERN" in blueprint:
            pat = blueprint["BLUEPRINT_PATTERN"]
            Config.Strategies.Mutations['pat_wick_strict'] = pat.get('wick_filter_strictness', 2.0)
            Config.Strategies.Mutations['pat_cons_min'] = pat.get('consolidation_candles_min', 12)
            
        # 3. LOGICAL DNA TOGGLES (Exported to ENV vars so strategies and engine can read them globally)
        if "LOGICAL_DNA" in blueprint:
            dna = blueprint["LOGICAL_DNA"]
            os.environ['DNA_RISK_DYNAMIC'] = str(dna.get('risk_dynamic_stops', True))
            os.environ['DNA_SNIPER_VOLUME'] = str(dna.get('sniper_volume_confirmation', True))
            os.environ['DNA_PATTERN_STRICT'] = str(dna.get('pattern_strict_wick_filter', False))
            os.environ['DNA_TECH_GARCH'] = str(dna.get('tech_use_garch', False))
            
        # 4. OMNISCORE SYNERGY
        if "BLUEPRINT_OMNISCORE" in blueprint:
            if not hasattr(Config, 'OmniScore'):
                Config.OmniScore = type('OmniScore', (), {})
            omni = blueprint["BLUEPRINT_OMNISCORE"]
            Config.OmniScore.w_ml = omni.get('w_ml', getattr(Config.OmniScore, 'w_ml', 1.0))
            Config.OmniScore.w_technical = omni.get('w_technical', getattr(Config.OmniScore, 'w_technical', 1.0))
            Config.OmniScore.w_phalanx = omni.get('w_phalanx', getattr(Config.OmniScore, 'w_phalanx', 0.5))
            Config.OmniScore.w_statarb = omni.get('w_statarb', getattr(Config.OmniScore, 'w_statarb', 0.5))
            Config.OmniScore.master_threshold = omni.get('master_threshold', getattr(Config.OmniScore, 'master_threshold', 1.5))
            logger.info(f"⚖️ [Blueprint Loader] OmniScore Synergy loaded -> ML:{Config.OmniScore.w_ml:.1f} | TECH:{Config.OmniScore.w_technical:.1f} | THRESH:{Config.OmniScore.master_threshold:.1f}")

        # 5. MATRIX OVERRIDES (Injected into adaptive_config.py 5D Matrix)
        if "MATRIX_OVERRIDES" in blueprint:
            mo = blueprint["MATRIX_OVERRIDES"]
            
            for horizon, data in mo.items():
                if horizon in adaptive_config.matrix:
                    # Update global_horizon (e.g. capital_allocation)
                    if "global_horizon" in data:
                        adaptive_config.matrix[horizon]["global_horizon"].update(data["global_horizon"])
                    
                    # Update por_activo (e.g. BTC, ALL)
                    if "por_activo" in data:
                        for asset, asset_params in data["por_activo"].items():
                            if asset not in adaptive_config.matrix[horizon]["por_activo"]:
                                adaptive_config.matrix[horizon]["por_activo"][asset] = {}
                            adaptive_config.matrix[horizon]["por_activo"][asset].update(asset_params)

            # Sincronizar los legacy caps para compatibilidad con el resto del sistema
            micro_cap = adaptive_config.matrix["MICRO"]["global_horizon"].get("capital_allocation_base_pct", 0.25)
            scalp_cap = adaptive_config.matrix["SCALP"]["global_horizon"].get("capital_allocation_base_pct", 0.45)
            swing_cap = adaptive_config.matrix["SWING"]["global_horizon"].get("capital_allocation_base_pct", 0.30)
            
            Config.MICROSCALPING_MARGIN_CAP = micro_cap
            Config.SCALPING_MARGIN_CAP = scalp_cap
            Config.SWING_MARGIN_CAP = swing_cap

            logger.info(f"✨ [Blueprint Loader] 5D Matrix Overrides applied. New Capital Distribution: MICRO={micro_cap*100:.1f}% | SCALP={scalp_cap*100:.1f}% | SWING={swing_cap*100:.1f}%")
