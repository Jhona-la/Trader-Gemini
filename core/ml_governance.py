import sqlite3
import json
import os
import shutil
from datetime import datetime
import logging

logger = logging.getLogger("MLGovernance")

class MLGovernance:
    """
    ⚖️ [PHASE 12] ML GOVERNANCE ENGINE
    Manages model versioning, quality gates, and production promotion.
    
    👨‍🏫 MODO PROFESOR:
    - QUÉ: Un juez y bibliotecario para los modelos de Inteligencia Artificial.
    - POR QUÉ: No queremos que un modelo que "aprendió mal" tome decisiones con dinero real.
    - PARA QUÉ: Para tener trazabilidad total (saber qué versión del modelo hizo qué trade) y seguridad.
    """
    def __init__(self, db_path="data/feature_store.db", models_root=".models"):
        self.db_path = db_path
        self.models_root = models_root
        os.makedirs(self.models_root, exist_ok=True)
        
    def register_model(self, symbol, metrics, model_paths):
        """
        Registers a new model and evaluates Quality Gates.
        Returns model_id if promoted to Production status.
        """
        sharpe = metrics.get('sharpe', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # QUALITY GATE: Institutional standards
        # Reference: Phase 10 validation
        is_production = 1 if sharpe > 1.5 and win_rate > 52 else 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get next version
        cursor.execute("SELECT MAX(version) FROM model_registry WHERE symbol = ?", (symbol,))
        last_version = cursor.fetchone()[0] or 0
        version = last_version + 1
        
        model_id = f"{symbol.replace('/', '_')}_v{version}_{datetime.now().strftime('%Y%m%d')}"
        
        # Move models to governance storage
        governance_path = os.path.join(self.models_root, model_id)
        os.makedirs(governance_path, exist_ok=True)
        for name, path in model_paths.items():
            if os.path.exists(path):
                shutil.copy(path, os.path.join(governance_path, f"{name}.joblib"))

        cursor.execute("""
            INSERT INTO model_registry (model_id, symbol, version, sharpe, win_rate, created_at, model_path, is_production)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_id, symbol, version, sharpe, win_rate, datetime.now().isoformat(), governance_path, is_production))
        
        # If this is the new Production leader, demote previous ones
        if is_production:
            cursor.execute("UPDATE model_registry SET is_production = 0 WHERE symbol = ? AND model_id != ?", (symbol, model_id))
            logger.info(f"🏆 MODELO PROMOVIDO A PRODUCCIÓN: {model_id} (Sharpe: {sharpe:.2f})")
        else:
            logger.warning(f"⚠️ Modelo {model_id} no superó Quality Gate (Sharpe: {sharpe:.2f}). Guardado como histórico.")

        conn.commit()
        conn.close()
        return model_id if is_production else None

    def get_production_model(self, symbol):
        """Retrieves the latest production-grade model path for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT model_path, version, sharpe 
            FROM model_registry 
            WHERE symbol = ? AND is_production = 1 
            ORDER BY version DESC LIMIT 1
        """, (symbol,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {'path': result[0], 'version': result[1], 'sharpe': result[2]}
        return None

    def get_performance_history(self, symbol):
        """Returns a history of all trained models for auditing."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM model_registry WHERE symbol = ? ORDER BY version DESC", conn, params=(symbol,))
        conn.close()
        return df

if __name__ == "__main__":
    gov = MLGovernance()
    print("ML Governance Engine Initialization Test: OK")
