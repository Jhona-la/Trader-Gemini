"""
🔄 HOT RELOAD SYSTEM - Strategy Live Updates
Allows updating ml_strategy.py and other modules without restarting the bot.

PROFESSOR METHOD:
- QUÉ: Sistema de recarga en caliente de módulos Python
- POR QUÉ: Permitir actualizaciones de código sin perder conexiones WebSocket
- CÓMO: importlib.reload() + watchdog file observer
- CUÁNDO: Cuando se detecta un cambio guardado en strategies/
- DÓNDE: Se integra en el main loop de main.py
"""
import importlib
import importlib.util
import sys
import os
import time
import threading
import hashlib
import ast
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Optional watchdog import
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object

from utils.logger import logger


@dataclass
class ReloadResult:
    """Result of a reload operation."""
    success: bool
    module_name: str
    old_version: str
    new_version: str
    latency_ms: float
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SyntaxValidator:
    """
    Pre-validates Python files before reload.
    
    PARA QUÉ: Evitar que un error de sintaxis rompa el bot en producción.
    """
    
    @staticmethod
    def validate_file(filepath: str) -> tuple[bool, Optional[str]]:
        """
        Check if a Python file has valid syntax.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # Parse the AST to check syntax
            ast.parse(source)
            return True, None
            
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            return False, error_msg
        except Exception as e:
            return False, f"Validation error: {e}"
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get MD5 hash of file contents for version tracking."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        except:
            return "unknown"


class ModuleReloader:
    """
    Core module reloader using importlib.
    
    CÓMO FUNCIONA:
    1. Valida sintaxis del archivo modificado
    2. Guarda referencia al módulo antiguo
    3. Ejecuta importlib.reload()
    4. Verifica que la recarga fue exitosa
    5. Registra el evento en Sentinel-24
    """
    
    def __init__(self):
        self.module_versions: Dict[str, str] = {}
        self.reload_history: List[ReloadResult] = []
        self.validator = SyntaxValidator()
        self._lock = threading.Lock()
    
    def get_module_version(self, module_name: str) -> str:
        """Get current version hash of a module."""
        if module_name in sys.modules:
            module = sys.modules[module_name]
            if hasattr(module, '__file__') and module.__file__:
                return self.validator.get_file_hash(module.__file__)
        return "not_loaded"
    
    def reload_module(self, module_name: str) -> ReloadResult:
        """
        Safely reload a Python module.
        
        Args:
            module_name: Full module name (e.g., 'strategies.ml_strategy')
        
        Returns:
            ReloadResult with success status and timing
        """
        start_time = time.perf_counter()
        old_version = self.get_module_version(module_name)
        
        with self._lock:
            try:
                # Check if module is loaded
                if module_name not in sys.modules:
                    return ReloadResult(
                        success=False,
                        module_name=module_name,
                        old_version=old_version,
                        new_version=old_version,
                        latency_ms=0,
                        error=f"Module '{module_name}' not loaded"
                    )
                
                module = sys.modules[module_name]
                filepath = getattr(module, '__file__', None)
                
                if not filepath or not os.path.exists(filepath):
                    return ReloadResult(
                        success=False,
                        module_name=module_name,
                        old_version=old_version,
                        new_version=old_version,
                        latency_ms=0,
                        error="Module file not found"
                    )
                
                # PRE-VALIDATION: Check syntax before reload
                is_valid, error = self.validator.validate_file(filepath)
                if not is_valid:
                    logger.error(f"❌ [HOT_RELOAD] Syntax error in {module_name}: {error}")
                    return ReloadResult(
                        success=False,
                        module_name=module_name,
                        old_version=old_version,
                        new_version=old_version,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        error=error
                    )
                
                # RELOAD: Execute importlib.reload
                reloaded_module = importlib.reload(module)
                
                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000
                new_version = self.get_module_version(module_name)
                
                # Update version tracking
                self.module_versions[module_name] = new_version
                
                result = ReloadResult(
                    success=True,
                    module_name=module_name,
                    old_version=old_version,
                    new_version=new_version,
                    latency_ms=latency_ms
                )
                
                self.reload_history.append(result)
                
                # Log success
                logger.info(f"✅ [HOT_RELOAD_SUCCESS] {module_name}")
                logger.info(f"   Version: {old_version} → {new_version}")
                logger.info(f"   Latency: {latency_ms:.2f}ms {'🟢' if latency_ms < 100 else '🟡'}")
                
                return result
                
            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                error_msg = str(e)
                logger.error(f"❌ [HOT_RELOAD_FAILED] {module_name}: {error_msg}")
                
                return ReloadResult(
                    success=False,
                    module_name=module_name,
                    old_version=old_version,
                    new_version=old_version,
                    latency_ms=latency_ms,
                    error=error_msg
                )


class StrategyReloader:
    """
    Specialized reloader for trading strategies with state preservation.
    
    PARA QUÉ: Mantener positions/orders activas mientras se actualiza la lógica.
    """
    
    def __init__(self, engine=None):
        self.module_reloader = ModuleReloader()
        self.engine = engine
        self._strategy_states: Dict[str, Dict[str, Any]] = {}
    
    def preserve_strategy_state(self, strategy) -> Dict[str, Any]:
        """Extract state that should survive reload."""
        state = {}
        
        # Preserve key attributes
        for attr in ['symbol', 'lookback_bars', 'model', 'scaler', 'feature_columns']:
            if hasattr(strategy, attr):
                state[attr] = getattr(strategy, attr)
        
        # Preserve training data if available
        if hasattr(strategy, 'training_data'):
            state['training_data'] = strategy.training_data
        
        return state
    
    def restore_strategy_state(self, strategy, state: Dict[str, Any]):
        """Restore preserved state to new strategy instance."""
        for attr, value in state.items():
            if hasattr(strategy, attr):
                try:
                    setattr(strategy, attr, value)
                except Exception as e:
                    logger.warning(f"Could not restore {attr}: {e}")
    
    def reload_strategy(self, module_name: str, class_name: str, 
                        current_instance, **init_kwargs) -> tuple[bool, Any]:
        """
        Reload a strategy module and create new instance with preserved state.
        
        Returns:
            (success, new_instance or None)
        """
        # 1. Preserve current state
        preserved_state = self.preserve_strategy_state(current_instance)
        
        # 2. Reload the module
        result = self.module_reloader.reload_module(module_name)
        
        if not result.success:
            return False, None
        
        try:
            # 3. Get the reloaded class
            module = sys.modules[module_name]
            strategy_class = getattr(module, class_name)
            
            # 4. Create new instance with same init args
            new_instance = strategy_class(**init_kwargs)
            
            # 5. Restore preserved state
            self.restore_strategy_state(new_instance, preserved_state)
            
            logger.info(f"✅ Strategy {class_name} reloaded with state preserved")
            return True, new_instance
            
        except Exception as e:
            logger.error(f"❌ Failed to create new strategy instance: {e}")
            return False, None


class StrategyFileWatcher(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    Watches strategies/ folder for changes and triggers reload.
    
    CUÁNDO: Cada vez que se guarda un archivo .py en strategies/
    """
    
    def __init__(self, callback: Callable[[str], None], 
                 watch_path: str = "strategies",
                 debounce_seconds: float = 1.0):
        if WATCHDOG_AVAILABLE:
            super().__init__()
        
        self.callback = callback
        self.watch_path = Path(watch_path).resolve()
        self.debounce_seconds = debounce_seconds
        self._last_event_time: Dict[str, float] = {}
        self._observer: Optional[Observer] = None
        self._running = False
    
    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return
        
        filepath = Path(event.src_path)
        
        # Only watch .py files
        if filepath.suffix != '.py':
            return
        
        # Ignore __pycache__
        if '__pycache__' in str(filepath):
            return
        
        # Debounce: Skip if recently triggered
        now = time.time()
        last_time = self._last_event_time.get(str(filepath), 0)
        
        if now - last_time < self.debounce_seconds:
            return
        
        self._last_event_time[str(filepath)] = now
        
        # Convert filepath to module name
        relative = filepath.relative_to(self.watch_path.parent)
        module_name = str(relative.with_suffix('')).replace(os.sep, '.')
        
        logger.info(f"🔄 [FILE_WATCHER] Detected change in: {filepath.name}")
        
        # Trigger callback
        self.callback(module_name)
    
    def start(self):
        """Start watching for file changes."""
        if not WATCHDOG_AVAILABLE:
            logger.warning("⚠️ watchdog not installed. Hot reload disabled.")
            logger.warning("   Install with: pip install watchdog")
            return False
        
        if self._running:
            return True
        
        try:
            self._observer = Observer()
            self._observer.schedule(self, str(self.watch_path), recursive=True)
            self._observer.start()
            self._running = True
            logger.info(f"👁️ [FILE_WATCHER] Watching {self.watch_path} for changes...")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start file watcher: {e}")
            return False
    
    def stop(self):
        """Stop the file watcher."""
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._running = False
            logger.info("🛑 [FILE_WATCHER] Stopped")


class HotReloadManager:
    """
    Main manager for hot reload functionality.
    
    DÓNDE: Se instancia en main.py y se integra con el Engine.
    """
    
    def __init__(self, engine=None, strategies_path: str = "strategies"):
        self.strategy_reloader = StrategyReloader(engine)
        self.engine = engine
        self._reload_queue: List[str] = []
        self._lock = threading.Lock()
        
        # Initialize file watcher
        self.file_watcher = StrategyFileWatcher(
            callback=self._on_file_change,
            watch_path=strategies_path
        )
        
        self._active = False
    
    def _on_file_change(self, module_name: str):
        """Called when a strategy file changes."""
        with self._lock:
            if module_name not in self._reload_queue:
                self._reload_queue.append(module_name)
                logger.info(f"📥 [HOT_RELOAD] Queued: {module_name}")
    
    def process_pending_reloads(self) -> List[ReloadResult]:
        """
        Process any pending reloads. Call this from the main loop.
        
        Returns:
            List of reload results
        """
        results = []
        
        with self._lock:
            pending = self._reload_queue.copy()
            self._reload_queue.clear()
        
        for module_name in pending:
            result = self.strategy_reloader.module_reloader.reload_module(module_name)
            results.append(result)
            
            # Log to Sentinel-24 format
            if result.success:
                self._log_sentinel_event(result)
        
        return results
    
    def _log_sentinel_event(self, result: ReloadResult):
        """Log reload event in Sentinel-24 compatible format."""
        import json
        from pathlib import Path
        
        event = {
            'type': 'HOT_RELOAD_SUCCESS' if result.success else 'HOT_RELOAD_FAILED',
            'timestamp': result.timestamp,
            'module': result.module_name,
            'version_old': result.old_version,
            'version_new': result.new_version,
            'latency_ms': result.latency_ms,
            'target_met': result.latency_ms < 100
        }
        
        # Append to sentinel log
        log_path = Path('logs/hot_reload_events.json')
        log_path.parent.mkdir(exist_ok=True)
        
        try:
            events = []
            if log_path.exists():
                with open(log_path, 'r') as f:
                    events = json.load(f)
            
            events.append(event)
            
            # Keep last 100 events
            events = events[-100:]
            
            with open(log_path, 'w') as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not log to sentinel: {e}")
    
    def start(self):
        """Start the hot reload system."""
        if self.file_watcher.start():
            self._active = True
            logger.info("🔥 [HOT_RELOAD] System ACTIVE - Ready for live updates")
            return True
        return False
    
    def stop(self):
        """Stop the hot reload system."""
        self.file_watcher.stop()
        self._active = False
    
    @property
    def is_active(self) -> bool:
        return self._active


# Global instance
_hot_reload_manager: Optional[HotReloadManager] = None


def init_hot_reload(engine=None, strategies_path: str = "strategies") -> HotReloadManager:
    """Initialize and return the global hot reload manager."""
    global _hot_reload_manager
    _hot_reload_manager = HotReloadManager(engine, strategies_path)
    return _hot_reload_manager


def get_hot_reload_manager() -> Optional[HotReloadManager]:
    """Get the global hot reload manager."""
    return _hot_reload_manager


if __name__ == '__main__':
    # Test the reloader
    print("="*60)
    print("🔄 HOT RELOAD SYSTEM - Test")
    print("="*60)
    
    # Test syntax validator
    validator = SyntaxValidator()
    test_file = "strategies/ml_strategy.py"
    
    if os.path.exists(test_file):
        is_valid, error = validator.validate_file(test_file)
        print(f"Syntax check for {test_file}: {'✅ Valid' if is_valid else f'❌ {error}'}")
        print(f"File hash: {validator.get_file_hash(test_file)}")
    
    # Test module reloader
    reloader = ModuleReloader()
    
    if 'strategies.ml_strategy' in sys.modules:
        result = reloader.reload_module('strategies.ml_strategy')
        print(f"\nReload result:")
        print(f"  Success: {result.success}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Version: {result.old_version} → {result.new_version}")
    else:
        print("\nModule not loaded, importing first...")
        import strategies.ml_strategy
        result = reloader.reload_module('strategies.ml_strategy')
        print(f"Reload latency: {result.latency_ms:.2f}ms")
    
    print("="*60)
    print("✅ Hot Reload System Ready")
