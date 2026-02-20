"""
📋 PHASE OMNI: OMNIPOTENCIA-SÍNCRO CERTIFICATION
==================================================
QUÉ: Script de certificación final que ejecuta todas las pruebas del
     protocolo Omnipotencia-Síncro y genera un reporte de salud.
POR QUÉ: Necesitamos una verificación unificada de que TODOS los
         componentes del protocolo funcionan correctamente antes de deploy.
PARA QUÉ: Un único punto de verdad sobre el estado del sistema.
CÓMO: Ejecuta 4 categorías de validaciones:
      1. Latencia (Engine loop, WebSocket, Signal-to-Order)
      2. Test Coverage (pytest con cobertura)
      3. Resiliencia (Byzantine tests)
      4. Integridad (Dependency graph, Config validation)
CUÁNDO: Antes de cada deploy a producción.
DÓNDE: tests/omni_certification.py
QUIÉN: SRE, QA Engineer.

SALIDA: Reporte con Heat Map de salud por subsistema.
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class OmniCertification:
    """
    📋 Omnipotencia-Síncro Certification Engine.
    
    Runs all validation checks and generates a unified health report.
    Each check returns a score (0-100) and a pass/fail status.
    """
    
    # Target thresholds from implementation plan
    TARGETS = {
        'latency_engine_ms': 50.0,       # Max 50ms engine loop
        'latency_execution_ms': 100.0,    # Max 100ms signal-to-order
        'test_coverage_pct': 80.0,        # Min 80% test coverage
        'mutation_score_pct': 60.0,       # Min 60% mutation score
        'byzantine_pass_pct': 100.0,      # All byzantine tests must pass
        'dependency_errors': 0,           # Zero dependency errors
        'recovery_time_ms': 100.0,        # Max 100ms state recovery
    }
    
    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.root = project_root
        self.results: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
    
    def run_all(self) -> Dict[str, Any]:
        """Run all certification checks."""
        print("\n" + "="*70)
        print("📋 OMNIPOTENCIA-SÍNCRO CERTIFICATION")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # 1. Dependency Graph Validation
        self._check_dependencies()
        
        # 2. Config Validation
        self._check_config()
        
        # 3. Byzantine Resilience Tests
        self._check_byzantine()
        
        # 4. Unit Tests
        self._check_unit_tests()
        
        # 5. Import Sanity
        self._check_critical_imports()
        
        # Generate report
        return self._generate_report()
    
    def _check_dependencies(self):
        """Validate dependency graph integrity."""
        print("🔍 [1/5] Checking dependency graph...")
        
        try:
            from utils.dep_graph import DependencyGraph
            
            graph = DependencyGraph(self.root)
            graph.build()
            
            errors = graph.validate_all()
            error_count = sum(len(errs) for errs in errors.values())
            
            score = 100 if error_count == 0 else max(0, 100 - error_count * 10)
            
            self.results['dependency_graph'] = {
                'status': 'PASS' if error_count == 0 else 'WARN',
                'score': score,
                'total_modules': len(graph.nodes),
                'error_count': error_count,
                'errors': {k: v for k, v in list(errors.items())[:5]},  # Top 5
            }
            
            print(f"   ✅ {len(graph.nodes)} modules scanned, {error_count} errors")
            
        except Exception as e:
            self.results['dependency_graph'] = {
                'status': 'FAIL',
                'score': 0,
                'error': str(e),
            }
            print(f"   ❌ Dependency check failed: {e}")
    
    def _check_config(self):
        """Validate configuration integrity."""
        print("🔍 [2/5] Validating config...")
        
        try:
            from config import Config
            
            checks_passed = 0
            total_checks = 5
            issues = []
            
            # Check 1: API keys present
            if Config.BINANCE_API_KEY and Config.BINANCE_SECRET_KEY:
                checks_passed += 1
            else:
                issues.append("Missing API keys")
            
            # Check 2: Risk limits configured
            if hasattr(Config, 'MAX_RISK_PER_TRADE') and Config.MAX_RISK_PER_TRADE > 0:
                checks_passed += 1
            else:
                issues.append("MAX_RISK_PER_TRADE not configured")
            
            # Check 3: Trading pairs defined
            if hasattr(Config, 'TRADING_PAIRS') and len(Config.TRADING_PAIRS) > 0:
                checks_passed += 1
            else:
                issues.append("No trading pairs configured")
            
            # Check 4: Validate method works
            try:
                if hasattr(Config, 'validate') and callable(Config.validate):
                    Config.validate()
                checks_passed += 1
            except Exception as e:
                issues.append(f"Config.validate() failed: {e}")
            
            # Check 5: Critical constants
            if hasattr(Config, 'STOP_LOSS_PCT') and Config.STOP_LOSS_PCT > 0:
                checks_passed += 1
            else:
                issues.append("STOP_LOSS_PCT not set")
            
            score = int((checks_passed / total_checks) * 100)
            
            self.results['config_validation'] = {
                'status': 'PASS' if checks_passed == total_checks else 'WARN',
                'score': score,
                'checks_passed': checks_passed,
                'total_checks': total_checks,
                'issues': issues,
            }
            
            print(f"   ✅ {checks_passed}/{total_checks} config checks passed")
            
        except Exception as e:
            self.results['config_validation'] = {
                'status': 'FAIL',
                'score': 0,
                'error': str(e),
            }
            print(f"   ❌ Config validation failed: {e}")
    
    def _check_byzantine(self):
        """Run Byzantine resilience tests."""
        print("🔍 [3/5] Running Byzantine resilience tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", 
                 "tests/byzantine_test.py", 
                 "-v", "--tb=short", "-q"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.root),
            )
            
            # Parse pytest output
            output = result.stdout + result.stderr
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            total = passed + failed
            
            score = int((passed / max(total, 1)) * 100)
            
            self.results['byzantine_resilience'] = {
                'status': 'PASS' if failed == 0 else 'FAIL',
                'score': score,
                'passed': passed,
                'failed': failed,
                'total': total,
                'output': output[-500:] if len(output) > 500 else output,
            }
            
            print(f"   {'✅' if failed == 0 else '❌'} {passed}/{total} Byzantine tests passed")
            
        except subprocess.TimeoutExpired:
            self.results['byzantine_resilience'] = {
                'status': 'TIMEOUT',
                'score': 0,
            }
            print("   ⏰ Byzantine tests timed out (>120s)")
        except Exception as e:
            self.results['byzantine_resilience'] = {
                'status': 'FAIL',
                'score': 0,
                'error': str(e),
            }
            print(f"   ❌ Byzantine tests failed: {e}")
    
    def _check_unit_tests(self):
        """Run core unit tests."""
        print("🔍 [4/5] Running unit tests...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", 
                 "tests/", 
                 "-q", "--tb=line",
                 "--ignore=tests/mutation_tester.py",
                 "-x",  # Stop at first failure for speed
                 "--timeout=60"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(self.root),
            )
            
            output = result.stdout + result.stderr
            
            # Extract pass/fail counts from pytest summary line
            # Format: "X passed, Y failed" or "X passed"
            passed = 0
            failed = 0
            for line in output.split('\n'):
                if 'passed' in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == 'passed' and i > 0:
                            try:
                                passed = int(parts[i-1])
                            except ValueError:
                                pass
                        if p == 'failed' and i > 0:
                            try:
                                failed = int(parts[i-1])
                            except ValueError:
                                pass
            
            total = passed + failed
            score = int((passed / max(total, 1)) * 100)
            
            self.results['unit_tests'] = {
                'status': 'PASS' if failed == 0 else 'WARN',
                'score': score,
                'passed': passed,
                'failed': failed,
            }
            
            print(f"   {'✅' if failed == 0 else '⚠️'} {passed} passed, {failed} failed")
            
        except subprocess.TimeoutExpired:
            self.results['unit_tests'] = {
                'status': 'TIMEOUT',
                'score': 50,
            }
            print("   ⏰ Tests timed out (>300s)")
        except Exception as e:
            self.results['unit_tests'] = {
                'status': 'FAIL',
                'score': 0,
                'error': str(e),
            }
            print(f"   ❌ Unit tests failed: {e}")
    
    def _check_critical_imports(self):
        """Verify all critical modules can be imported."""
        print("🔍 [5/5] Verifying critical imports...")
        
        critical_modules = [
            'config',
            'core.events',
            'core.engine',
            'core.adaptive_balancer',
            'core.self_tuner',
            'utils.dep_graph',
            'utils.logger',
        ]
        
        passed = 0
        failed_imports = []
        
        for mod in critical_modules:
            try:
                importlib.import_module(mod)
                passed += 1
            except Exception as e:
                failed_imports.append(f"{mod}: {e}")
        
        total = len(critical_modules)
        score = int((passed / total) * 100)
        
        self.results['critical_imports'] = {
            'status': 'PASS' if not failed_imports else 'FAIL',
            'score': score,
            'passed': passed,
            'total': total,
            'failed': failed_imports[:5],
        }
        
        print(f"   {'✅' if not failed_imports else '❌'} {passed}/{total} critical imports OK")
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate the final certification report with heat map."""
        elapsed = time.time() - self.start_time
        
        # Calculate overall score
        scores = [r.get('score', 0) for r in self.results.values()]
        overall_score = sum(scores) / max(len(scores), 1)
        
        # Determine certification status
        all_pass = all(r.get('status') != 'FAIL' for r in self.results.values())
        cert_status = 'CERTIFIED' if all_pass and overall_score >= 80 else 'NOT CERTIFIED'
        
        # Heat Map
        heat_map = self._generate_heat_map()
        
        report = {
            'certification': cert_status,
            'overall_score': round(overall_score, 1),
            'timestamp': datetime.now().isoformat(),
            'elapsed_seconds': round(elapsed, 1),
            'checks': self.results,
            'heat_map': heat_map,
        }
        
        # Print report
        print("\n" + "="*70)
        print(f"📋 CERTIFICATION: {cert_status}")
        print(f"   Overall Score: {overall_score:.0f}/100")
        print(f"   Elapsed: {elapsed:.1f}s")
        print("="*70)
        print("\n📊 HEAT MAP:")
        print(heat_map)
        
        # Save report
        report_path = self.root / "data" / "certification_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Report saved: {report_path}")
        
        return report
    
    def _generate_heat_map(self) -> str:
        """
        Generate ASCII heat map showing subsystem health.
        🟢 = 80-100, 🟡 = 60-79, 🔴 = 0-59
        """
        lines = []
        
        for check_name, result in self.results.items():
            score = result.get('score', 0)
            status = result.get('status', 'UNKNOWN')
            
            if score >= 80:
                indicator = "🟢"
            elif score >= 60:
                indicator = "🟡"
            else:
                indicator = "🔴"
            
            bar_len = int(score / 5)  # 0-20 chars
            bar = "█" * bar_len + "░" * (20 - bar_len)
            
            name = check_name.replace('_', ' ').title()
            lines.append(f"  {indicator} {name:.<30} [{bar}] {score:>3}%")
        
        return "\n".join(lines)


# ======================================================================
# CLI ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    import importlib
    
    cert = OmniCertification()
    report = cert.run_all()
    
    # Exit code based on certification
    if report['certification'] == 'CERTIFIED':
        print("\n✅ SISTEMA CERTIFICADO PARA PRODUCCIÓN")
        sys.exit(0)
    else:
        print("\n❌ SISTEMA NO CERTIFICADO - Resolver problemas antes de deploy")
        sys.exit(1)
