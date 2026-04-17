from datetime import datetime
from utils.health_supervisor import HealthSupervisor, get_alert_system
from utils.loss_analyzer import get_loss_analyzer
from utils.auto_correction_engine import get_auto_correction_engine
from utils.interaction_monitor import get_interaction_monitor

class DiagnosticDashboard:
    """
    SISTEMA DE AUTO-DIAGNÓSTICO: Controlador de API Interna para el Dashboard de UI.
    """
    def __init__(self):
        self.metrics = {
            "system_health": 100,
            "loss_rate": 0.0,
            "fee_efficiency": 0.0,
            "strategy_conflicts": 0,
            "auto_corrections_applied": 0
        }
        self.loss_analyzer = get_loss_analyzer()
        self.auto_corrections = get_auto_correction_engine()
        self.interaction_monitor = get_interaction_monitor()
        self.alert_sys = get_alert_system()
        
    def update_metrics(self):
        # Health score base on active alerts
        active_critical = len([a for a in self.alert_sys.active_alerts.values() if a["status"] == "ACTIVE"])
        self.metrics["system_health"] = max(0, 100 - (active_critical * 25))
        
        # Loss metrics (mocked retrieval logic since PnL needs portfolio injection)
        self.metrics["fee_efficiency"] = 100.0  # Placeholder 
        
        # Conflicts
        self.metrics["strategy_conflicts"] = len([e for e in self.interaction_monitor.interaction_log if e.get("conflict_detected")])
        self.metrics["auto_corrections_applied"] = len(self.auto_corrections.applied_corrections)

    def generate_report(self) -> str:
        self.update_metrics()
        
        issues = "\n".join([f"⚠️ {c['type']}: {c['message']}" for c in self.alert_sys.active_alerts.values() if c["status"] == "ACTIVE"])
        if not issues:
            issues = "Ningún problema detectado."
            
        corrections = "\n".join([f"✅ {c['issue']} -> {c['correction']}" for c in self.auto_corrections.applied_corrections[-5:]])
        if not corrections:
            corrections = "Ninguna corrección automática aplicada recientemente."
            
        report = f"""
        📊 DIAGNÓSTICO DEL SISTEMA EN LÍNEA - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        ========================================================================

        [MÉTRICAS CORE]
        Salud del Sistema:           {self.metrics['system_health']}/100
        Eficiencia de Fees estimada: {self.metrics['fee_efficiency']:.2f}%
        Conflictos de Estrategia:    {self.metrics['strategy_conflicts']}
        Auto-Correcciones Inyectadas:{self.metrics['auto_corrections_applied']}

        [🔍 PROBLEMAS CRÍTICOS IDENTIFICADOS]
        {issues}

        [✅ ACCIONES CORRECTIVAS APLICADAS EN MEMORIA (Live Config)]
        {corrections}

        [🎯 ESTADO DEL EVENT LOOP Y PIPELINE]
        - Loss Analyzer: Activo ({self.loss_analyzer.streak_analyzer['consecutive_losses']} racha negativa local)
        - Interaction Monitor: Trackeando {len(self.interaction_monitor.interaction_log)} eventos I/O
        """
        return report

_diagnostic_dashboard = None
def get_diagnostic_dashboard() -> DiagnosticDashboard:
    global _diagnostic_dashboard
    if not _diagnostic_dashboard:
        _diagnostic_dashboard = DiagnosticDashboard()
    return _diagnostic_dashboard
