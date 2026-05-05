"""
═══════════════════════════════════════════════════════════════════════════════
 FORENSIC EXPORTER — Alpha Leak & Trade Replay Persistence
═══════════════════════════════════════════════════════════════════════════════

QUÉ: Módulo encargado de persistir los resultados de la auditoría forense
     en formato Parquet para análisis offline y dashboards.
POR QUÉ: Los datos de auditoría (MFE/MAE, Alpha Leak, Trade Replays) son
     demasiado voluminosos para JSON y demasiado estructurados para logs.
PARA QUÉ: Permite análisis post-hoc con pandas/polars sin re-ejecutar el
     backtest completo.
CÓMO: Recibe los diccionarios de métricas del backtest y los serializa
     en archivos Parquet comprimidos usando pyarrow.
CUÁNDO: Al finalizar cada ejecución de run_god_mode_backtest.py.
DÓNDE: utils/forensic_exporter.py
QUIÉN: QA Engineer + Quant Developer
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from utils.logger import logger


class ForensicExporter:
    """
    Exports forensic audit data to structured files for offline analysis.
    """

    def __init__(self, output_dir: str = "results/forensic"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_alpha_leak_table(
        self,
        alpha_leak: Dict[str, float],
        run_id: str,
        scenario: str = "A"
    ) -> str:
        """Export alpha leak breakdown to JSON."""
        path = os.path.join(self.output_dir, f"alpha_leak_{scenario}_{run_id}.json")
        data = {
            "run_id": run_id,
            "scenario": scenario,
            "timestamp": datetime.utcnow().isoformat(),
            **alpha_leak
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"📊 [FORENSIC] Alpha Leak Table exported to {path}")
        return path

    def export_strategy_attribution(
        self,
        strategy_attribution: Dict[str, Dict[str, Any]],
        run_id: str,
        scenario: str = "A"
    ) -> Optional[str]:
        """Export per-strategy performance attribution to Parquet."""
        if not HAS_PANDAS or not strategy_attribution:
            return None

        rows = []
        for strat_id, metrics in strategy_attribution.items():
            rows.append({
                "strategy_id": strat_id,
                "trades": metrics.get("trades", 0),
                "wins": metrics.get("wins", 0),
                "losses": metrics.get("losses", 0),
                "win_rate": (metrics["wins"] / metrics["trades"] * 100) if metrics.get("trades", 0) > 0 else 0,
                "gross_pnl": metrics.get("gross_pnl", 0.0),
                "net_pnl": metrics.get("net_pnl", 0.0),
                "sharpe_contribution": metrics.get("net_pnl", 0.0),  # Simplified
            })

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"strategy_attribution_{scenario}_{run_id}.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv")
            df.to_csv(path, index=False)
        logger.info(f"📊 [FORENSIC] Strategy Attribution exported to {path}")
        return path

    def export_symbol_attribution(
        self,
        symbol_attribution: Dict[str, Dict[str, Any]],
        run_id: str,
        scenario: str = "A"
    ) -> Optional[str]:
        """Export per-symbol performance attribution to Parquet."""
        if not HAS_PANDAS or not symbol_attribution:
            return None

        rows = []
        for sym, metrics in symbol_attribution.items():
            rows.append({
                "symbol": sym,
                "trades": metrics.get("trades", 0),
                "wins": metrics.get("wins", 0),
                "losses": metrics.get("losses", 0),
                "win_rate": (metrics["wins"] / metrics["trades"] * 100) if metrics.get("trades", 0) > 0 else 0,
                "gross_pnl": metrics.get("gross_pnl", 0.0),
                "net_pnl": metrics.get("net_pnl", 0.0),
                "fees": metrics.get("fees", 0.0),
            })

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, f"symbol_attribution_{scenario}_{run_id}.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv")
            df.to_csv(path, index=False)
        logger.info(f"📊 [FORENSIC] Symbol Attribution exported to {path}")
        return path

    def export_conflict_log(
        self,
        conflict_log: List[Dict[str, Any]],
        run_id: str,
        scenario: str = "A"
    ) -> Optional[str]:
        """Export conflict log to Parquet."""
        if not HAS_PANDAS or not conflict_log:
            return None

        df = pd.DataFrame(conflict_log)
        path = os.path.join(self.output_dir, f"conflict_log_{scenario}_{run_id}.parquet")
        try:
            df.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv")
            df.to_csv(path, index=False)
        logger.info(f"📊 [FORENSIC] Conflict Log exported to {path} ({len(conflict_log)} entries)")
        return path

    def export_top_bottom_trades(
        self,
        trades: List[Dict[str, Any]],
        run_id: str,
        top_n: int = 100,
        scenario: str = "A"
    ) -> Optional[str]:
        """
        Export the top N best and worst trades for trade replay analysis.
        """
        if not HAS_PANDAS or not trades:
            return None

        df = pd.DataFrame(trades)
        if "net_pnl" not in df.columns:
            return None

        df_sorted = df.sort_values("net_pnl", ascending=False)
        top_trades = df_sorted.head(top_n)
        bottom_trades = df_sorted.tail(top_n)

        combined = pd.concat([top_trades, bottom_trades]).drop_duplicates()
        path = os.path.join(self.output_dir, f"trade_replay_{scenario}_{run_id}.parquet")
        try:
            combined.to_parquet(path, index=False)
        except Exception:
            path = path.replace(".parquet", ".csv")
            combined.to_csv(path, index=False)
        logger.info(f"📊 [FORENSIC] Trade Replay exported to {path} ({len(combined)} trades)")
        return path

    def export_all(
        self,
        alpha_leak: Dict[str, float],
        strategy_attribution: Dict[str, Dict[str, Any]],
        symbol_attribution: Dict[str, Dict[str, Any]],
        conflict_log: List[Dict[str, Any]],
        run_id: str,
        scenario: str = "A",
        trades: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Optional[str]]:
        """Export all forensic data in one call."""
        return {
            "alpha_leak": self.export_alpha_leak_table(alpha_leak, run_id, scenario),
            "strategy_attribution": self.export_strategy_attribution(strategy_attribution, run_id, scenario),
            "symbol_attribution": self.export_symbol_attribution(symbol_attribution, run_id, scenario),
            "conflict_log": self.export_conflict_log(conflict_log, run_id, scenario),
            "trade_replay": self.export_top_bottom_trades(trades or [], run_id, scenario=scenario),
        }
