
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from core.genotype import Genotype
from utils.logger import logger
from data.data_provider import DataProvider
from core.enums import TimeFrame
from config import Config

class OracleRemediator:
    """
    💊 GENETIC DOCTOR: Rescues failing symbols by scaling the BTC Apex Genotype.
    """
    def __init__(self, audit_file="massive_audit_raw.json", apex_symbol="BTC/USDT"):
        self.audit_file = audit_file
        self.apex_symbol = apex_symbol
        self.genotype_dir = "data/genotypes"
        self.model_dir = ".models"
        
    def run_remediation(self):
        logger.info("🩺 Starting Sovereign Remediation Protocol...")
        
        # 1. Identify failing symbols
        if not os.path.exists(self.audit_file):
            logger.error(f"❌ Audit file {self.audit_file} not found.")
            return

        with open(self.audit_file, 'r') as f:
            audit_data = json.load(f)

        failures = []
        for entry in audit_data:
            if entry.get('metrics', {}).get('total_return', 0) < 0:
                failures.append(entry['symbol'])
        
        failures = list(set(failures)) # Unique symbols
        if not failures:
            logger.info("✅ No failing symbols detected. System at Peak Performance.")
            return

        logger.info(f"🚨 Detected {len(failures)} failing symbols: {failures}")

        # 2. Load APEX Genotype (BTC)
        apex_path = os.path.join(self.genotype_dir, f"{self.apex_symbol.replace('/','')}_gene.json")
        apex_genotype = Genotype.load(apex_path)
        if not apex_genotype:
            logger.error(f"❌ Apex genotype not found at {apex_path}")
            return

        # 3. Get Benchmark Volatility (BTC)
        btc_vol = self._get_volatility(self.apex_symbol)
        
        remediation_log = []

        for symbol in failures:
            logger.info(f"🧬 Remediating {symbol}...")
            
            # 4. Get Target Volatility and Scale Factors
            target_vol = self._get_volatility(symbol)
            scale_factor = target_vol / btc_vol if btc_vol > 0 else 1.0
            
            # 5. Clone and Scale ADN
            new_genes = apex_genotype.genes.copy()
            
            # Scale relevant risk parameters
            new_genes["tp_pct"] = apex_genotype.genes["tp_pct"] * scale_factor
            new_genes["sl_pct"] = apex_genotype.genes["sl_pct"] * scale_factor
            
            # Preserve identity
            new_genotype = Genotype(
                symbol=symbol,
                generation=apex_genotype.generation + 1,
                fitness_score=0.5, # Reset to neutral
                genes=new_genes
            )
            
            # 6. Inject Genotype
            target_path = os.path.join(self.genotype_dir, f"{symbol.replace('/','')}_gene.json")
            new_genotype.save(target_path)
            
            # 7. Tabula Rasa: Clear residual models to prevent "stale learning" interference
            self._wipe_models(symbol)
            
            log_entry = {
                "symbol": symbol,
                "scale_factor": round(float(scale_factor), 4),
                "new_tp": round(float(new_genes["tp_pct"]), 6),
                "new_sl": round(float(new_genes["sl_pct"]), 6)
            }
            remediation_log.append(log_entry)
            logger.info(f"✅ {symbol} Remediated (Scale: {scale_factor:.2f}x)")

        self._save_summary(remediation_log)

    def _get_volatility(self, symbol):
        """Calculates Standard Deviation of Log-Returns for robust volatility scaling."""
        try:
            import queue
            from data.binance_loader import BinanceData
            
            q = queue.Queue()
            loader = BinanceData(q, [symbol])
            
            # Fetch 10,000 bars (~7 days) for high-confidence volatility estimation
            sym_clean = symbol.replace('/', '')
            klines = loader.client_sync.get_klines(symbol=sym_clean, interval='1m', limit=10000)
            
            if not klines: return 0.0001 # Minimal noise level
            
            # Extract closing prices
            closes = np.array([float(k[4]) for k in klines])
            
            # Calculate Log Returns
            log_returns = np.diff(np.log(closes))
            
            # Standard Deviation of returns (Volatility)
            vol = np.std(log_returns)
            
            logger.info(f"📊 {symbol} Statistical Volatility: {vol:.6f}")
            return vol if vol > 0 else 0.0001
        except Exception as e:
            logger.error(f"Error calculating statistical vol for {symbol}: {e}")
            return 0.0001

    def _wipe_models(self, symbol):
        """Removes WFV models to allow fresh meta-optimization."""
        clean_symbol = symbol.replace('/', '_')
        model_path = f".models_wfv_{clean_symbol}"
        if os.path.exists(model_path):
            import shutil
            shutil.rmtree(model_path)
            logger.info(f"🧹 Wiped residual models for {symbol}")

    def _save_summary(self, log):
        summary_path = "remediation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": log
            }, f, indent=2)
        logger.info(f"📊 Remediation summary saved to {summary_path}")

if __name__ == "__main__":
    remediator = OracleRemediator()
    remediator.run_remediation()
