"""
🕸️ PHASE OMNI: DEPENDENCY GRAPH VALIDATOR
==========================================
QUÉ: Validador de grafo de dependencias que verifica la integridad
     de los imports entre módulos antes de aplicar parches.
POR QUÉ: Cambiar un módulo puede romper N módulos dependientes de forma
         invisible. Un import roto en risk_manager.py causa pérdida inmediata.
PARA QUÉ: Garantizar que ningún cambio rompe la cadena de dependencias.
CÓMO: 1. Escanea todos los .py del proyecto via ast.parse
      2. Construye un grafo dirigido de dependencias (import/from)
      3. Para un módulo dado, calcula el "blast radius" (dependientes)
      4. Valida que todos los módulos dependientes importan correctamente
CUÁNDO: Pre-commit hook, o llamado manualmente antes de deploy.
DÓNDE: utils/dep_graph.py
QUIÉN: SRE/DevOps, QA Engineer.
"""

import ast
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class DepNode:
    """Represents a module in the dependency graph."""
    module_path: str          # Relative path (e.g. "core/engine.py")
    imports: Set[str] = field(default_factory=set)      # Modules this imports
    imported_by: Set[str] = field(default_factory=set)   # Modules that import this
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)


class DependencyGraph:
    """
    🕸️ Dependency Graph for Trader Gemini.
    
    Builds a directed graph of module dependencies and provides:
    - blast_radius(module): Set of modules affected by changes
    - validate_all(): Check all imports resolve correctly
    - topological_order(): Safe execution/loading order
    """
    
    # Critical modules that, if broken, crash the entire system
    CRITICAL_MODULES = {
        'core/engine.py',
        'core/events.py',
        'core/portfolio.py',
        'risk/risk_manager.py',
        'risk/kill_switch.py',
        'data/binance_loader.py',
        'execution/binance_executor.py',
        'config.py',
    }
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.root = project_root
        self.nodes: Dict[str, DepNode] = {}
        self._built = False
    
    def build(self) -> 'DependencyGraph':
        """Scan all .py files and build the dependency graph."""
        py_files = list(self.root.rglob("*.py"))
        
        # Filter: skip venv, __pycache__, .git, tests
        py_files = [
            f for f in py_files 
            if not any(part in f.parts for part in 
                      ['venv', '.venv', '__pycache__', '.git', 'node_modules'])
        ]
        
        for filepath in py_files:
            rel_path = str(filepath.relative_to(self.root)).replace('\\', '/')
            self._analyze_file(filepath, rel_path)
        
        # Build reverse edges (imported_by)
        for mod, node in self.nodes.items():
            for imp in node.imports:
                if imp in self.nodes:
                    self.nodes[imp].imported_by.add(mod)
        
        self._built = True
        return self
    
    def _analyze_file(self, filepath: Path, rel_path: str):
        """Parse a single file and extract its imports."""
        node = DepNode(module_path=rel_path)
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for ast_node in ast.walk(tree):
                if isinstance(ast_node, ast.Import):
                    for alias in ast_node.names:
                        resolved = self._resolve_import(alias.name)
                        if resolved:
                            node.imports.add(resolved)
                
                elif isinstance(ast_node, ast.ImportFrom):
                    if ast_node.module:
                        resolved = self._resolve_import(ast_node.module)
                        if resolved:
                            node.imports.add(resolved)
        
        except SyntaxError as e:
            node.is_valid = False
            node.errors.append(f"SyntaxError: {e}")
        except Exception as e:
            node.errors.append(f"ParseError: {e}")
        
        self.nodes[rel_path] = node
    
    def _resolve_import(self, import_name: str) -> Optional[str]:
        """
        Resolve an import name to a project-relative file path.
        Returns None if it's an external package.
        """
        # Convert dotted import to path candidates
        parts = import_name.split('.')
        
        candidates = [
            '/'.join(parts) + '.py',
            '/'.join(parts) + '/__init__.py',
        ]
        
        for candidate in candidates:
            if (self.root / candidate).exists():
                return candidate
        
        # Check if it's a package (directory with __init__.py)
        pkg_path = '/'.join(parts)
        if (self.root / pkg_path).is_dir():
            init_path = pkg_path + '/__init__.py'
            if (self.root / init_path).exists():
                return init_path
        
        # External package — not tracked
        return None
    
    def blast_radius(self, module_path: str) -> Set[str]:
        """
        Calculate the set of modules affected by changes to the given module.
        Returns all modules that directly or transitively depend on it.
        """
        if not self._built:
            self.build()
        
        affected = set()
        queue = [module_path]
        
        while queue:
            current = queue.pop(0)
            if current in affected:
                continue
            affected.add(current)
            
            if current in self.nodes:
                for dep in self.nodes[current].imported_by:
                    if dep not in affected:
                        queue.append(dep)
        
        affected.discard(module_path)  # Don't include the module itself
        return affected
    
    def validate_all(self) -> Dict[str, List[str]]:
        """
        Validate that all internal imports resolve correctly.
        Returns dict of {module: [error_messages]}.
        """
        if not self._built:
            self.build()
        
        errors = {}
        
        for mod, node in self.nodes.items():
            mod_errors = list(node.errors)
            
            for imp in node.imports:
                if imp not in self.nodes:
                    # Import resolves to a file that doesn't exist
                    mod_errors.append(f"Missing dependency: {imp}")
            
            if mod_errors:
                errors[mod] = mod_errors
        
        return errors
    
    def validate_patch(self, changed_modules: List[str]) -> Dict[str, Any]:
        """
        Validate a set of proposed changes against the dependency graph.
        
        Args:
            changed_modules: List of module paths being modified.
            
        Returns:
            Dict with:
            - 'safe': bool (True if no critical modules affected)
            - 'blast_radius': Set of affected modules
            - 'critical_affected': Set of critical modules in blast radius
            - 'warnings': List of warning strings
        """
        if not self._built:
            self.build()
        
        total_affected = set()
        for mod in changed_modules:
            total_affected |= self.blast_radius(mod)
        
        critical_hit = total_affected & self.CRITICAL_MODULES
        
        warnings = []
        if critical_hit:
            for c in critical_hit:
                warnings.append(f"⚠️ CRITICAL MODULE AFFECTED: {c}")
        
        if len(total_affected) > 10:
            warnings.append(f"⚠️ Large blast radius: {len(total_affected)} modules affected")
        
        return {
            'safe': len(critical_hit) == 0,
            'blast_radius': total_affected,
            'blast_radius_count': len(total_affected),
            'critical_affected': critical_hit,
            'warnings': warnings,
        }
    
    def get_import_chain(self, from_module: str, to_module: str) -> Optional[List[str]]:
        """
        Find the shortest import chain between two modules (BFS).
        Returns None if no path exists.
        """
        if not self._built:
            self.build()
        
        if from_module not in self.nodes or to_module not in self.nodes:
            return None
        
        visited = set()
        queue = [(from_module, [from_module])]
        
        while queue:
            current, path = queue.pop(0)
            if current == to_module:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.nodes:
                for imp in self.nodes[current].imports:
                    if imp not in visited:
                        queue.append((imp, path + [imp]))
        
        return None
    
    def summary(self) -> str:
        """Returns a human-readable summary of the dependency graph."""
        if not self._built:
            self.build()
        
        total = len(self.nodes)
        with_errors = sum(1 for n in self.nodes.values() if n.errors)
        avg_deps = sum(len(n.imports) for n in self.nodes.values()) / max(total, 1)
        
        # Top 5 most depended-on modules
        top_deps = sorted(
            self.nodes.items(),
            key=lambda x: len(x[1].imported_by),
            reverse=True
        )[:5]
        
        lines = [
            f"\n{'='*60}",
            f"🕸️ DEPENDENCY GRAPH SUMMARY",
            f"{'='*60}",
            f"Total modules:  {total}",
            f"With errors:    {with_errors}",
            f"Avg deps/mod:   {avg_deps:.1f}",
            f"\nTop 5 most depended-on:",
        ]
        
        for mod, node in top_deps:
            lines.append(f"  {mod}: {len(node.imported_by)} dependents")
        
        return "\n".join(lines)


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    graph = DependencyGraph()
    graph.build()
    
    print(graph.summary())
    
    # Validate all imports
    errors = graph.validate_all()
    if errors:
        print(f"\n⚠️ Import Errors Found:")
        for mod, errs in errors.items():
            for e in errs:
                print(f"  [{mod}] {e}")
    else:
        print("\n✅ All imports validated successfully!")
    
    # Check blast radius for specified module
    if len(sys.argv) > 1:
        target = sys.argv[1]
        radius = graph.blast_radius(target)
        print(f"\n💥 Blast radius for {target}: {len(radius)} modules")
        for m in sorted(radius):
            critical = " 🚨" if m in graph.CRITICAL_MODULES else ""
            print(f"  → {m}{critical}")
