"""
🗂️ SESSION MANAGER - Sistema de Gestión de Sesiones
====================================================

PROFESSOR METHOD:
- QUÉ: Sistema de gestión de sesiones de trading.
- POR QUÉ: Para separar datos de diferentes ejecuciones del bot y facilitar análisis.
- PARA QUÉ: Comparar rendimiento entre sesiones, evitar mezclar métricas, debugging.
- CÓMO: Organización híbrida por fecha + hora de ejecución.
- CUÁNDO: Se crea nueva sesión al iniciar el bot, se cierra al detenerlo.
- DÓNDE: dashboard/data/{mode}/sessions/YYYY-MM-DD/run_HHMMSS/
- QUIÉN: main.py inicia/cierra, portfolio.py usa paths de sesión.
"""

import os
import json
import shutil
from datetime import datetime, timezone
from typing import Optional, Dict, List
from utils.logger import logger


class SessionManager:
    """
    Gestiona las sesiones de trading con estructura híbrida:
    - Por fecha (YYYY-MM-DD)
    - Por ejecución (run_HHMMSS)
    """
    
    def __init__(self, base_dir: str = "dashboard/data/futures"):
        """
        Inicializa el Session Manager.
        
        Args:
            base_dir: Directorio base para datos (ej: dashboard/data/futures)
        """
        self.base_dir = base_dir
        self.sessions_dir = os.path.join(base_dir, "sessions")
        self.current_session_id: Optional[str] = None
        self.current_session_path: Optional[str] = None
        self.session_info: Optional[Dict] = None
        self._start_time: Optional[datetime] = None
        
        # Asegurar que existe el directorio de sesiones
        os.makedirs(self.sessions_dir, exist_ok=True)
    
    def start_session(self, mode: str = "futures", symbols: List[str] = None, 
                      initial_capital: float = 0.0) -> str:
        """
        Inicia una nueva sesión de trading.
        
        Args:
            mode: Modo de trading (futures/spot)
            symbols: Lista de símbolos monitoreados
            initial_capital: Capital inicial
            
        Returns:
            session_id: ID único de la sesión (YYYY-MM-DD_HHMMSS)
        """
        self._start_time = datetime.now(timezone.utc)
        date_str = self._start_time.strftime("%Y-%m-%d")
        time_str = self._start_time.strftime("%H%M%S")
        
        # Crear estructura: sessions/2026-02-03/run_183015/
        self.current_session_id = f"{date_str}_{time_str}"
        date_dir = os.path.join(self.sessions_dir, date_str)
        self.current_session_path = os.path.join(date_dir, f"run_{time_str}")
        
        os.makedirs(self.current_session_path, exist_ok=True)
        
        # Crear archivo de info de sesión
        self.session_info = {
            "session_id": self.current_session_id,
            "start_time": self._start_time.isoformat(),
            "end_time": None,
            "mode": mode,
            "symbols": symbols or [],
            "initial_capital": initial_capital,
            "status": "RUNNING",
            "summary": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "pnl": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "duration_minutes": 0
            }
        }
        
        self._save_session_info()
        self._create_backup()
        self._update_current_pointer()
        
        logger.info(f"🗂️ Session started: {self.current_session_id}")
        logger.info(f"   Path: {self.current_session_path}")
        
        return self.current_session_id
    
    def get_session_path(self) -> str:
        """
        Retorna el path de la sesión actual.
        Si no hay sesión activa, retorna el base_dir para compatibilidad.
        """
        if self.current_session_path and os.path.exists(self.current_session_path):
            return self.current_session_path
        return self.base_dir
    
    def get_session_id(self) -> Optional[str]:
        """Retorna el ID de la sesión actual."""
        return self.current_session_id
    
    def end_session(self, summary: Dict = None) -> Dict:
        """
        Finaliza la sesión actual y guarda el resumen.
        
        Args:
            summary: Diccionario con métricas finales (opcional)
            
        Returns:
            session_info: Información completa de la sesión
        """
        if not self.session_info:
            logger.warning("No active session to end")
            return {}
        
        end_time = datetime.now(timezone.utc)
        self.session_info["end_time"] = end_time.isoformat()
        self.session_info["status"] = "COMPLETED"
        
        # Calcular duración
        if self._start_time:
            duration = (end_time - self._start_time).total_seconds() / 60
            self.session_info["summary"]["duration_minutes"] = round(duration, 2)
        
        # Actualizar con summary proporcionado
        if summary:
            self.session_info["summary"].update(summary)
        
        self._save_session_info()
        
        logger.info(f"🏁 Session ended: {self.current_session_id}")
        logger.info(f"   Duration: {self.session_info['summary']['duration_minutes']:.1f} min")
        logger.info(f"   PnL: ${self.session_info['summary']['pnl']:.2f}")
        
        # Enviar resumen por Telegram
        self._send_session_summary()
        
        result = self.session_info.copy()
        
        # Limpiar estado
        self.current_session_id = None
        self.current_session_path = None
        self.session_info = None
        self._start_time = None
        
        return result
    
    def update_session_stats(self, trades: int = 0, pnl: float = 0.0, 
                             wins: int = 0, losses: int = 0):
        """
        Actualiza estadísticas de la sesión en tiempo real.
        """
        if not self.session_info:
            return
            
        self.session_info["summary"]["total_trades"] += trades
        self.session_info["summary"]["pnl"] += pnl
        self.session_info["summary"]["winning_trades"] += wins
        self.session_info["summary"]["losing_trades"] += losses
        
        total = self.session_info["summary"]["total_trades"]
        if total > 0:
            win_rate = (self.session_info["summary"]["winning_trades"] / total) * 100
            self.session_info["summary"]["win_rate"] = round(win_rate, 1)
        
        self._save_session_info()
    
    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Lista las sesiones más recientes.
        
        Args:
            limit: Número máximo de sesiones a retornar
            
        Returns:
            Lista de diccionarios con info de cada sesión
        """
        sessions = []
        
        if not os.path.exists(self.sessions_dir):
            return sessions
        
        # Iterar por fechas (ordenadas desc)
        date_dirs = sorted(os.listdir(self.sessions_dir), reverse=True)
        
        for date_dir in date_dirs:
            date_path = os.path.join(self.sessions_dir, date_dir)
            if not os.path.isdir(date_path):
                continue
            
            # Iterar por runs de esa fecha
            run_dirs = sorted(os.listdir(date_path), reverse=True)
            
            for run_dir in run_dirs:
                run_path = os.path.join(date_path, run_dir)
                info_file = os.path.join(run_path, "session_info.json")
                
                if os.path.exists(info_file):
                    try:
                        with open(info_file, 'r') as f:
                            sessions.append(json.load(f))
                    except:
                        from utils.error_handler import SystemIntegrityError
                        raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')
                
                if len(sessions) >= limit:
                    return sessions
        
        return sessions
    
    def _save_session_info(self):
        """Guarda la info de sesión en archivo JSON."""
        if not self.current_session_path or not self.session_info:
            return
            
        info_path = os.path.join(self.current_session_path, "session_info.json")
        try:
            with open(info_path, 'w') as f:
                json.dump(self.session_info, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session info: {e}")
    
    def _update_current_pointer(self):
        """Actualiza el puntero a la sesión actual."""
        pointer_path = os.path.join(self.base_dir, "current_session.json")
        try:
            with open(pointer_path, 'w') as f:
                json.dump({
                    "session_id": self.current_session_id,
                    "path": self.current_session_path
                }, f)
        except Exception as e:
            logger.error(f"Error updating session pointer: {e}")
    
    def _create_backup(self):
        """Crea backup de all_trades.csv al iniciar nueva sesión."""
        all_trades_path = os.path.join(self.base_dir, "all_trades.csv")
        if os.path.exists(all_trades_path):
            backup_name = f"all_trades_backup_{self.current_session_id}.csv"
            backup_path = os.path.join(self.base_dir, "backups", backup_name)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            try:
                shutil.copy2(all_trades_path, backup_path)
                logger.info(f"📦 Backup created: {backup_name}")
            except Exception as e:
                logger.warning(f"Could not create backup: {e}")
    
    def _send_session_summary(self):
        """
        Envía resumen de sesión por Telegram (Phase 4.5 Enhanced).
        Includes detailed metrics and formatting.
        """
        try:
            from utils.notifier import Notifier
            from config import Config
            
            if not Config.Observability.TELEGRAM_ENABLED:
                return
            
            s = self.session_info["summary"]
            duration = s["duration_minutes"]
            pnl = s['pnl']
            total_trades = s['total_trades']
            win_rate = s['win_rate']
            winning = s['winning_trades']
            losing = s['losing_trades']
            
            # Session result emoji
            result_emoji = "📈" if pnl >= 0 else "📉"
            pnl_sign = "+" if pnl >= 0 else ""
            
            # Duration formatting
            if duration >= 60:
                dur_str = f"{duration/60:.1f}h"
            else:
                dur_str = f"{duration:.0f}min"
            
            msg = f"🏁 *SESSION ENDED* 🏁\n\n"
            msg += f"📊 ID: `{self.current_session_id}`\n"
            msg += f"⏱️ Duración: `{dur_str}`\n\n"
            
            msg += f"*Resultado:*\n"
            msg += f"{result_emoji} PnL: `{pnl_sign}${pnl:,.2f}`\n"
            msg += f"📈 Trades: `{total_trades}` ({winning}W / {losing}L)\n"
            msg += f"🏆 Win Rate: `{win_rate:.1f}%`\n"
            
            sharpe = s['sharpe']
            max_dd = s['max_drawdown']
            if sharpe:
                msg += f"📐 Sharpe: `{sharpe:.2f}`\n"
            if max_dd:
                msg += f"📉 Max Drawdown: `{max_dd:.2f}%`\n"
            
            Notifier.send_telegram(msg, priority="INFO")
        except Exception as e:
            logger.warning(f"Could not send session summary: {e}")


# Instancia global para uso en todo el proyecto
session_manager: Optional[SessionManager] = None


def get_session_manager() -> Optional[SessionManager]:
    """Retorna la instancia global del SessionManager."""
    # Removed global session_manager
    return session_manager


def init_session_manager(base_dir: str = "dashboard/data/futures") -> SessionManager:
    """Inicializa la instancia global del SessionManager."""
    global session_manager
    session_manager = SessionManager(base_dir)
    return session_manager
