import subprocess
import sys
import time
import os

# Configuración de Colores para CLI
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Fases del QA Supremo
PHASES = [
    {
        "name": "FASE 1: MICROESTRUCTURA & HORIZON ISOLATION",
        "description": "Verifica que Scalping y Swing nunca colisionen en el Virtual Ledger.",
        "commands": [
            [sys.executable, "-m", "pytest", "tests/test_horizon_isolation.py", "-v"],
            [sys.executable, "-m", "pytest", "tests/test_horizon_specialization.py", "-v"]
        ]
    },
    {
        "name": "FASE 2: FORENSE IA & PREVENCIÓN DE REGRESIONES",
        "description": "Garantiza que la IA no sufra de Data Leakage ni Repainting (Mirar al futuro).",
        "commands": [
            [sys.executable, "-m", "pytest", "tests/test_repainting.py", "-v"],
            [sys.executable, "-m", "pytest", "tests/test_ml_leakage.py", "-v"]
        ]
    },
    {
        "name": "FASE 3: SIMULACIÓN DE SEGURIDAD PnL",
        "description": "Verifica que no exista fuga de margen.",
        "commands": [
            # diagnose_margin_leak.py is a standard python script (not pytest), so we run it directly
            [sys.executable, "tests/diagnose_margin_leak.py"]
        ]
    }
]

def run_command(command, phase_name):
    print(f"\n{Colors.OKBLUE}▶ Ejecutando: {' '.join(command)}{Colors.ENDC}")
    start_ns = time.perf_counter_ns()
    
    try:
        # Aseguramos que el script corra desde el root del proyecto
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        result = subprocess.run(
            command, 
            cwd=cwd,
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True
        )
        
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        
        if result.returncode == 0:
            print(f"{Colors.OKGREEN}✅ [PASS] en {duration_ms:.2f}ms{Colors.ENDC}")
            return True, result.stdout
        else:
            print(f"{Colors.FAIL}❌ [FAIL] Error Detectado en {duration_ms:.2f}ms{Colors.ENDC}")
            print(f"{Colors.WARNING}--- OUTPUT DEL ERROR ---{Colors.ENDC}")
            print(result.stdout)
            print(f"{Colors.WARNING}------------------------{Colors.ENDC}")
            return False, result.stdout
            
    except FileNotFoundError:
        print(f"{Colors.FAIL}❌ [ERROR CRÍTICO] Comando no encontrado: {command[0]}{Colors.ENDC}")
        return False, "Command not found."

def main():
    print(f"{Colors.HEADER}{Colors.BOLD}╔══════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}║     [QA SUPREME ORCHESTRATOR] Trader Gemini V5           ║{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}╚══════════════════════════════════════════════════════════╝{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Iniciando validación de Arquitectura de Producción...{Colors.ENDC}\n")

    total_start = time.perf_counter_ns()
    failed_phases = 0

    for idx, phase in enumerate(PHASES, 1):
        print(f"{Colors.BOLD}{Colors.HEADER}===================================================={Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER} {phase['name']}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER} {phase['description']}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}===================================================={Colors.ENDC}")
        
        phase_success = True
        for cmd in phase["commands"]:
            success, _ = run_command(cmd, phase["name"])
            if not success:
                phase_success = False
                failed_phases += 1
                break # Rompe la ejecución de esta fase, pasamos al reporte
                
        if not phase_success:
            print(f"\n{Colors.FAIL}🚨 REGRESIÓN DETECTADA 🚨. La Arquitectura está comprometida.{Colors.ENDC}")
            print(f"{Colors.FAIL}Abortando ejecución de Fases posteriores para prevenir corrupción.{Colors.ENDC}")
            break

    total_duration_s = (time.perf_counter_ns() - total_start) / 1_000_000_000.0

    print(f"\n{Colors.BOLD}===================================================={Colors.ENDC}")
    if failed_phases == 0:
        print(f"{Colors.OKGREEN}🏆 [AUDITORÍA FORENSE 100% EXITOSA] - Cero Regresiones Detectadas.{Colors.ENDC}")
        print(f"{Colors.OKGREEN}El sistema puede ser liberado a Producción con Máxima Confianza.{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}☠️ [AUDITORÍA FALLIDA] - El código actual contiene violaciones arquitectónicas.{Colors.ENDC}")
    
    print(f"{Colors.OKCYAN}Tiempo Total: {total_duration_s:.2f} segundos.{Colors.ENDC}")
    print(f"{Colors.BOLD}===================================================={Colors.ENDC}\n")

    if failed_phases > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
