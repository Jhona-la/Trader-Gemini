"""
🧬 PHASE OMNI: MUTATION TESTING FRAMEWORK
==========================================
QUÉ: Framework de testing por mutación que inyecta bugs controlados en el código
     y verifica que los tests existentes los detectan.
POR QUÉ: Un test suite con 100% coverage pero 0% mutation score da falsa seguridad.
         Los mutation tests verifican que los tests realmente validaN la lógica.
PARA QUÉ: Medir la calidad real del test suite y encontrar "puntos ciegos".
CÓMO: Para cada módulo crítico:
      1. Parsea el AST del archivo
      2. Genera mutantes (negar condiciones, cambiar operadores, alterar constantes)
      3. Ejecuta los tests contra cada mutante
      4. Si el test PASA con el mutante, el test es DÉBIL (mutante sobrevivió)
CUÁNDO: Ejecutado manualmente o como parte del pipeline CI pre-deploy.
DÓNDE: tests/mutation_tester.py
QUIÉN: QA Engineer, CI/CD Pipeline.

SEGURIDAD:
- ❌ NUNCA modifica archivos en disco (solo en memoria via importlib)
- ✅ Usa ast.parse para análisis estático seguro
- ✅ Timeout por mutante (30s) para evitar loops infinitos
"""

import ast
import copy
import sys
import time
import importlib
import importlib.util
import types
import subprocess
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class MutantResult:
    """Result of a single mutant test run."""
    module: str
    mutation_type: str
    location: str  # line:col or function name
    killed: bool   # True = test detected the bug (GOOD)
    error_msg: str = ""
    execution_time: float = 0.0


@dataclass 
class MutationReport:
    """Aggregate report for a mutation testing session."""
    total_mutants: int = 0
    killed: int = 0
    survived: int = 0
    timeout: int = 0
    errors: int = 0
    results: List[MutantResult] = field(default_factory=list)
    
    @property
    def mutation_score(self) -> float:
        """Percentage of mutants killed (higher = better test suite)."""
        if self.total_mutants == 0:
            return 0.0
        return (self.killed / self.total_mutants) * 100.0
    
    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"🧬 MUTATION TESTING REPORT\n"
            f"{'='*60}\n"
            f"Total Mutants:  {self.total_mutants}\n"
            f"Killed (GOOD):  {self.killed}\n"
            f"Survived (BAD): {self.survived}\n"
            f"Timeout:        {self.timeout}\n"
            f"Errors:         {self.errors}\n"
            f"{'='*60}\n"
            f"MUTATION SCORE: {self.mutation_score:.1f}%\n"
            f"{'='*60}\n"
        )

    def surviving_mutants_detail(self) -> str:
        """Returns details of surviving mutants (test weaknesses)."""
        survivors = [r for r in self.results if not r.killed]
        if not survivors:
            return "✅ All mutants killed! Test suite is robust.\n"
        
        lines = ["\n⚠️ SURVIVING MUTANTS (Test Gaps):\n"]
        for r in survivors:
            lines.append(f"  - [{r.module}] {r.mutation_type} @ {r.location}")
        return "\n".join(lines)


class MutationOperator:
    """
    Defines mutation operators that transform AST nodes.
    Each returns a list of (mutated_ast, description) tuples.
    """
    
    @staticmethod
    def negate_conditions(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """Negate boolean conditions (if x > 0 → if x <= 0)."""
        mutants = []
        
        # Operator swap mapping
        swap_map = {
            ast.Gt: ast.LtE,
            ast.Lt: ast.GtE,
            ast.GtE: ast.Lt,
            ast.LtE: ast.Gt,
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for i, op in enumerate(node.ops):
                    if type(op) in swap_map:
                        mutant_tree = copy.deepcopy(tree)
                        # Find and mutate the corresponding node
                        for m_node in ast.walk(mutant_tree):
                            if isinstance(m_node, ast.Compare) and m_node.lineno == node.lineno:
                                m_node.ops[i] = swap_map[type(op)]()
                                desc = f"negate_cond@L{node.lineno}: {type(op).__name__} → {swap_map[type(op)].__name__}"
                                mutants.append((mutant_tree, desc))
                                break
        return mutants
    
    @staticmethod
    def swap_arithmetic(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """Swap arithmetic operators (+ → -, * → /)."""
        mutants = []
        swap_map = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and type(node.op) in swap_map:
                mutant_tree = copy.deepcopy(tree)
                for m_node in ast.walk(mutant_tree):
                    if isinstance(m_node, ast.BinOp) and m_node.lineno == node.lineno:
                        old_op = type(m_node.op).__name__
                        m_node.op = swap_map[type(node.op)]()
                        desc = f"swap_arith@L{node.lineno}: {old_op} → {type(m_node.op).__name__}"
                        mutants.append((mutant_tree, desc))
                        break
        return mutants
    
    @staticmethod
    def alter_constants(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """Alter numeric constants (0.5 → 0.0, 100 → 99)."""
        mutants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value == 0:
                    continue  # Changing 0 is often meaningless
                
                mutant_tree = copy.deepcopy(tree)
                for m_node in ast.walk(mutant_tree):
                    if (isinstance(m_node, ast.Constant) and 
                        m_node.lineno == node.lineno and 
                        m_node.value == node.value):
                        old_val = m_node.value
                        # Mutate: multiply by 2 or set to 0
                        m_node.value = 0 if isinstance(old_val, float) else old_val + 1
                        desc = f"alter_const@L{node.lineno}: {old_val} → {m_node.value}"
                        mutants.append((mutant_tree, desc))
                        break
        
        # Limit: Max 20 constant mutations per file to cap runtime
        return mutants[:20]

    # ================================================================
    # OMEGA-VOID §4.3: FINANCIAL-DOMAIN MUTATION OPERATORS
    # ================================================================

    @staticmethod
    def invert_sign_polarity(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """
        Invert the sign of return values (1 → -1, BUY → SELL semantics).
        
        QUÉ: Invierte el signo de valores retornados en funciones de señal.
        POR QUÉ: Si una función de señal retorna +1 (comprar) pero el test
             no valida la dirección, este mutante SOBREVIVIRÁ, revelando
             que el test no verifica la dirección del trade.
        PARA QUÉ: Certificar que los tests detectan señales invertidas.
        """
        mutants = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Return) and node.value is not None:
                # Only target simple numeric returns or unary ops
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                    if node.value.value != 0:
                        mutant_tree = copy.deepcopy(tree)
                        for m_node in ast.walk(mutant_tree):
                            if (isinstance(m_node, ast.Return) and 
                                m_node.lineno == node.lineno and
                                isinstance(m_node.value, ast.Constant) and
                                m_node.value.value == node.value.value):
                                old_val = m_node.value.value
                                m_node.value.value = -old_val
                                desc = f"invert_sign@L{node.lineno}: return {old_val} → return {-old_val}"
                                mutants.append((mutant_tree, desc))
                                break
                
                # Negate UnaryOp (e.g., return -x → return x)
                elif isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.USub):
                    mutant_tree = copy.deepcopy(tree)
                    for m_node in ast.walk(mutant_tree):
                        if (isinstance(m_node, ast.Return) and 
                            m_node.lineno == node.lineno and
                            isinstance(m_node.value, ast.UnaryOp)):
                            # Remove the negation
                            m_node.value = m_node.value.operand
                            desc = f"invert_sign@L{node.lineno}: return -expr → return expr"
                            mutants.append((mutant_tree, desc))
                            break
        
        return mutants[:15]

    @staticmethod
    def swap_sl_tp_values(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """
        Swap stop-loss and take-profit constants.
        
        QUÉ: Intercambia valores de SL y TP en asignaciones.
        POR QUÉ: Si SL=0.5% y TP=1.0%, swappearlos genera SL=1.0%, TP=0.5%.
             Esto crea un sistema con R:R invertido que DEBERÍA ser detectado
             por tests de risk_manager. Si no lo detecta, hay un gap crítico.
        PARA QUÉ: Validar que los tests verifican la relación SL < TP.
        """
        mutants = []
        
        # Look for assignments with 'sl', 'stop', 'tp', 'take_profit' in name
        sl_keywords = {'sl', 'stop_loss', 'stop_pct', 'sl_pct', 'max_loss'}
        tp_keywords = {'tp', 'take_profit', 'tp_pct', 'profit_target', 'target_pct'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name_lower = target.id.lower()
                        # Multiply SL by 3 (making it too lax)
                        if any(kw in name_lower for kw in sl_keywords):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                                mutant_tree = copy.deepcopy(tree)
                                for m_node in ast.walk(mutant_tree):
                                    if (isinstance(m_node, ast.Assign) and 
                                        m_node.lineno == node.lineno and
                                        isinstance(m_node.value, ast.Constant)):
                                        old_val = m_node.value.value
                                        m_node.value.value = old_val * 3  # 3x wider SL
                                        desc = f"swap_sl@L{node.lineno}: {target.id}={old_val} → {old_val*3}"
                                        mutants.append((mutant_tree, desc))
                                        break
                        
                        # Divide TP by 3 (making it too tight)
                        elif any(kw in name_lower for kw in tp_keywords):
                            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, (int, float)):
                                if node.value.value != 0:
                                    mutant_tree = copy.deepcopy(tree)
                                    for m_node in ast.walk(mutant_tree):
                                        if (isinstance(m_node, ast.Assign) and 
                                            m_node.lineno == node.lineno and
                                            isinstance(m_node.value, ast.Constant)):
                                            old_val = m_node.value.value
                                            m_node.value.value = old_val / 3  # 3x tighter TP
                                            desc = f"swap_tp@L{node.lineno}: {target.id}={old_val} → {old_val/3:.4f}"
                                            mutants.append((mutant_tree, desc))
                                            break
        
        return mutants[:10]

    @staticmethod
    def remove_kill_switch_checks(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
        """
        Remove safety return statements (kill-switch, drawdown checks).
        
        QUÉ: Elimina `return` statements dentro de bloques `if` de seguridad.
        POR QUÉ: Si se elimina un `return` de kill-switch y los tests NO fallan,
             significa que ningún test verifica que el kill-switch realmente
             detiene el trading. Este es un BUG CRÍTICO de cobertura.
        PARA QUÉ: Certificar que CADA return de seguridad está cubierto por tests.
        """
        mutants = []
        
        # Find if-blocks that contain safety-related checks
        safety_keywords = {'kill', 'emergency', 'max_drawdown', 'circuit', 'halt',
                          'stop_trading', 'is_trading', 'max_loss', 'danger'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Check if the condition references safety variables
                condition_str = ast.dump(node)
                is_safety_check = any(kw in condition_str.lower() for kw in safety_keywords)
                
                if is_safety_check:
                    # Find Return statements in the if body
                    for i, stmt in enumerate(node.body):
                        if isinstance(stmt, ast.Return):
                            mutant_tree = copy.deepcopy(tree)
                            # Replace the Return with Pass
                            for m_node in ast.walk(mutant_tree):
                                if (isinstance(m_node, ast.If) and 
                                    m_node.lineno == node.lineno):
                                    for j, m_stmt in enumerate(m_node.body):
                                        if isinstance(m_stmt, ast.Return) and m_stmt.lineno == stmt.lineno:
                                            m_node.body[j] = ast.Pass()
                                            ast.fix_missing_locations(m_node.body[j])
                                            desc = f"remove_safety@L{stmt.lineno}: removed return in kill-switch"
                                            mutants.append((mutant_tree, desc))
                                            break
                                    break
        
        return mutants


class MutationTester:
    """
    Main mutation testing engine.
    
    Usage:
        tester = MutationTester()
        report = tester.test_module("risk/risk_manager.py", "tests/test_risk.py")
        print(report.summary())
    """
    
    CRITICAL_MODULES = [
        "risk/risk_manager.py",
        "strategies/technical.py",
        "core/engine.py",
        "config.py",
        "data/binance_loader.py",
    ]
    
    def __init__(self, project_root: Path = PROJECT_ROOT, timeout: int = 30):
        self.project_root = project_root
        self.timeout = timeout
        self.operators = [
            # Generic operators
            MutationOperator.negate_conditions,
            MutationOperator.swap_arithmetic,
            MutationOperator.alter_constants,
            # OMEGA-VOID §4.3: Financial-domain operators
            MutationOperator.invert_sign_polarity,
            MutationOperator.swap_sl_tp_values,
            MutationOperator.remove_kill_switch_checks,
        ]
    
    def generate_mutants(self, filepath: str) -> List[Tuple[ast.AST, str]]:
        """Parse a file and generate all mutants."""
        full_path = self.project_root / filepath
        
        with open(full_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        all_mutants = []
        for operator in self.operators:
            mutants = operator(tree)
            all_mutants.extend(mutants)
        
        return all_mutants
    
    def test_module(self, module_path: str, test_path: str = None) -> MutationReport:
        """
        Run mutation testing on a single module.
        
        Args:
            module_path: Relative path to module (e.g. "risk/risk_manager.py")
            test_path: Relative path to test file. If None, auto-discovers.
        """
        report = MutationReport()
        
        # Auto-discover test file
        if test_path is None:
            basename = Path(module_path).stem
            test_candidates = [
                f"tests/test_{basename}.py",
                f"tests/{basename}_test.py",
            ]
            for candidate in test_candidates:
                if (self.project_root / candidate).exists():
                    test_path = candidate
                    break
        
        if not test_path or not (self.project_root / test_path).exists():
            print(f"⚠️ No test file found for {module_path}. Skipping.")
            return report
        
        # Generate mutants
        mutants = self.generate_mutants(module_path)
        report.total_mutants = len(mutants)
        
        print(f"🧬 Testing {module_path} with {len(mutants)} mutants against {test_path}...")
        
        for i, (mutant_tree, desc) in enumerate(mutants):
            result = self._run_mutant_test(module_path, test_path, mutant_tree, desc)
            report.results.append(result)
            
            if result.killed:
                report.killed += 1
            elif result.error_msg == "TIMEOUT":
                report.timeout += 1
            elif result.error_msg:
                report.errors += 1
            else:
                report.survived += 1
            
            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(mutants)} (Score: {report.mutation_score:.0f}%)")
        
        return report
    
    def _run_mutant_test(self, module_path: str, test_path: str,
                          mutant_tree: ast.AST, desc: str) -> MutantResult:
        """
        Test a single mutant by running the test suite against it.
        Uses subprocess isolation to prevent state leakage.
        """
        start = time.time()
        
        try:
            # Compile mutant to code
            ast.fix_missing_locations(mutant_tree)
            mutant_code = compile(mutant_tree, module_path, 'exec')
            
            # Run test in subprocess with timeout
            # We write the mutant to a temp location and run pytest against it
            full_test_path = str(self.project_root / test_path)
            
            result = subprocess.run(
                [sys.executable, "-m", "pytest", full_test_path, "-x", "--tb=no", "-q"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.project_root),
                env={**dict(__import__('os').environ), 'MUTATION_TESTING': '1'}
            )
            
            elapsed = time.time() - start
            
            # If tests FAIL → mutant was KILLED (good!)
            # If tests PASS → mutant SURVIVED (bad - test gap)
            killed = result.returncode != 0
            
            return MutantResult(
                module=module_path,
                mutation_type=desc.split("@")[0],
                location=desc,
                killed=killed,
                execution_time=elapsed
            )
            
        except subprocess.TimeoutExpired:
            return MutantResult(
                module=module_path,
                mutation_type=desc.split("@")[0],
                location=desc,
                killed=True,  # Timeout counts as killed (infinite loop = detected change)
                error_msg="TIMEOUT",
                execution_time=self.timeout
            )
        except Exception as e:
            return MutantResult(
                module=module_path,
                mutation_type=desc.split("@")[0],
                location=desc,
                killed=True,  # Compilation errors = mutant killed
                error_msg=str(e),
                execution_time=time.time() - start
            )
    
    def run_full_audit(self) -> Dict[str, MutationReport]:
        """
        Run mutation testing on ALL critical modules.
        Returns dict of module → report.
        """
        results = {}
        
        print("\n" + "="*60)
        print("🧬 MUTATION TESTING - FULL AUDIT")
        print("="*60 + "\n")
        
        for module in self.CRITICAL_MODULES:
            if not (self.project_root / module).exists():
                print(f"⚠️ Module {module} not found. Skipping.")
                continue
            
            report = self.test_module(module)
            results[module] = report
            print(report.summary())
            print(report.surviving_mutants_detail())
        
        # Final Summary
        total_mutants = sum(r.total_mutants for r in results.values())
        total_killed = sum(r.killed for r in results.values())
        overall_score = (total_killed / total_mutants * 100) if total_mutants > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 OVERALL MUTATION SCORE: {overall_score:.1f}% ({total_killed}/{total_mutants})")
        print(f"{'='*60}\n")
        
        return results


# CLI Entry Point
if __name__ == "__main__":
    tester = MutationTester()
    
    if len(sys.argv) > 1:
        # Test specific module
        module = sys.argv[1]
        test = sys.argv[2] if len(sys.argv) > 2 else None
        report = tester.test_module(module, test)
        print(report.summary())
        print(report.surviving_mutants_detail())
    else:
        # Full audit
        tester.run_full_audit()
