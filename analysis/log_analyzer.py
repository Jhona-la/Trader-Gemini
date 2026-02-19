import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AI_Log_Analyzer')

class LogAnalyzer:
    """
    OMEGA PROTOCOL: AI-DRIVEN LOG ANALYSIS
    Uses Unsupervised Learning (Isolation Forest) to detect anomalies in system logs.
    """
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.model = IsolationForest(contamination=0.01, random_state=42) # 1% expected anomalies
        
    def load_logs(self, limit=10000):
        """Load recent JSON logs into DataFrame."""
        logger.info(f"📂 Loading logs from {self.log_dir}...")
        data = []
        
        # Iterar sobre todos los archivos JSON en logs/
        files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(self.log_dir, x)), reverse=True)
        
        count = 0
        for file in files:
            path = os.path.join(self.log_dir, file)
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # Extract features of interest
                        # We are looking for performance anomalies or error bursts
                        
                        # Timestamp processing
                        ts_str = entry.get('timestamp', '').split(',')[0]
                        try:
                            dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                            timestamp = dt.timestamp()
                        except:
                            timestamp = 0
                            
                        # Extract numerical details if available in message (Regex could be better, but simple is fast)
                        # E.g. "P&L $50.00", "Latency 0.5ms" - this requires structured logging of those values
                        # For now, we rely on metadata we initiated in later phases?
                        # Actually, phase 49 ensured structural logs.
                        
                        # Let's feature engineer simple proxies:
                        # 1. Message Length
                        # 2. Log Level (INFO=0, WARNING=1, ERROR=2) -> Weighted
                        level_map = {'DEBUG': 0, 'INFO': 1, 'WARNING': 5, 'ERROR': 10, 'CRITICAL': 20}
                        severity = level_map.get(entry.get('level', 'INFO'), 0)
                        
                        data.append({
                            'timestamp': timestamp,
                            'severity': severity,
                            'module': entry.get('module', 'unknown'),
                            'message_len': len(entry.get('message', '')),
                            'raw': entry.get('message', '')
                        })
                        count += 1
                        if count >= limit:
                            break
                    except:
                        pass
            if count >= limit:
                break
                
        if not data:
            logger.warning("⚠️ No logs found to analyze.")
            return None
            
        df = pd.DataFrame(data)
        logger.info(f"✅ Loaded {len(df)} log entries.")
        return df

    def detect_anomalies(self):
        """Train model and find outliers."""
        df = self.load_logs()
        if df is None or len(df) < 50:
            logger.warning("📉 Not enough data for AI analysis (need > 50 logs).")
            return
        
        # Feature Selection
        # We look for clusters of high severity or unusual message lengths
        features = df[['severity', 'message_len']]
        
        logger.info("🧠 Training Isolation Forest model...")
        self.model.fit(features)
        
        df['anomaly'] = self.model.predict(features)
        
        # Anomalies are marked as -1
        anomalies = df[df['anomaly'] == -1]
        
        if not anomalies.empty:
            logger.info("="*60)
            logger.info(f"🚨 DETECTED {len(anomalies)} SYSTEM ANOMALIES")
            logger.info("="*60)
            
            for idx, row in anomalies.iterrows():
                ts = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"🔴 [{ts}] Severity: {row['severity']} | Len: {row['message_len']} | Msg: {row['raw'][:100]}...")
                
            logger.info("="*60)
            logger.info("🔍 Recommendation: Investigate these events manually.")
        else:
            logger.info("✅ SYSTEM HEALTHY. No behavioral anomalies detected.")

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    analyzer.detect_anomalies()
