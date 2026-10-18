"""Exit-status contracts using only local command shims; never process availability."""
from pathlib import Path
import os,subprocess,unittest
import test_cargo_semaforo_fail_closed as fixtures

SCRIPT=Path(__file__).resolve().parents[1]/"cargo_semaforo.sh"
BASH=fixtures.BASH

@unittest.skipIf(BASH is None,"bash no disponible")
class EnumeratorExitStatusTests(unittest.TestCase):
    setUp=fixtures.SemaforoFailClosedTests.setUp
    shim=fixtures.SemaforoFailClosedTests.shim
    blocked=fixtures.SemaforoFailClosedTests.blocked
    admitted=fixtures.SemaforoFailClosedTests.admitted

    def run_enumerator(self,tool,code,stdout,after=0):
        self.shim(tool,"printf '%s\\n' '"+stdout+"'; exit "+str(code)+"\n")
        self.shim("date","""if [ "${1:-}" = "+%s" ]; then
 n=0
 if [ -f clock-counter ]; then IFS= read -r n < clock-counter; fi
 printf '%s\n' "$((n+1))" > clock-counter
 if [ "$n" = 0 ]; then printf '100000\n'; else printf '%s\n' "$((100000+__AFTER__))"; fi
else printf '2026-10-18T00:00:00Z\n'; fi
""".replace("__AFTER__",str(int(after))))
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(("GIT_","SEMAFORO_","BASH_FUNC_")) and k not in ("BASH_ENV","ENV")}
        env.update(PATH=str(self.bin),SEMAFORO_RAIZ=str(self.root),SEMAFORO_AGENTE="synthetic-status",SEMAFORO_SOLO_DECIDIR="1",SEMAFORO_ESPERA_MAX_MIN="0",SEMAFORO_PAUSA_S="0",SEMAFORO_PACIENCIA_MIN="30",SEMAFORO_RAM_GB="10")
        # Exactly our one enumerator file may resolve; no real OS process API.
        probe='for c in powershell powershell.exe pwsh pwsh.exe tasklist pgrep; do if command -v "$c" >/dev/null 2>&1; then [ "$(type -t "$c")" = file ] || exit 91; printf "%s\\n" "$c"; fi; done'
        p=subprocess.run([BASH,"--noprofile","--norc","-c",probe],cwd=self.root,env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertEqual(p.stdout.splitlines(),[tool])
        r=subprocess.run([BASH,"--noprofile","--norc",str(SCRIPT),"check"],cwd=self.root,env=env,capture_output=True,text=True,timeout=20)
        self.assertFalse(self.trace.exists(),"No Cargo, including the sentinel, is allowed")
        return r

    def test_powershell_nonzero_with_numeric_zero_is_unknown(self):
        self.blocked(self.run_enumerator("powershell",23,"0"))

    def test_powershell_success_numeric_zero_is_legitimate(self):
        self.admitted(self.run_enumerator("powershell",0,"0"))

    def test_pgrep_fatal_two_with_numeric_zero_is_unknown(self):
        self.blocked(self.run_enumerator("pgrep",2,"0"))

    def test_pgrep_fatal_three_with_numeric_zero_is_unknown(self):
        self.blocked(self.run_enumerator("pgrep",3,"0"))

    def test_pgrep_status_one_numeric_zero_is_legitimate_no_match(self):
        self.admitted(self.run_enumerator("pgrep",1,"0"))

    def test_pgrep_success_known_two_retains_numeric_patience(self):
        self.admitted(self.run_enumerator("pgrep",0,"2",after=8400))
        self.assertIn("turno por paciencia: 2 procesos",self.log.read_text(encoding="utf-8"))

if __name__=="__main__":unittest.main()
