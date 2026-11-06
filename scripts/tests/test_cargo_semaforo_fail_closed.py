"""Synthetic fail-closed contracts. Never real availability; only external dummy roots/tool mocks.

RÉGIMEN 2026-11-06 (mandato del operador: cero restricciones de recursos): la
cuenta de procesos y la RAM ya NO gobiernan el turno — sólo protegen el
cerrojo (un cerrojo vencido no se roba si hay compiladores vivos o la cuenta
es desconocida) y se anotan como telemetría."""
from pathlib import Path
import os,shutil,subprocess,tempfile,unittest

SCRIPT=Path(__file__).resolve().parents[1]/"cargo_semaforo.sh"
BASH=shutil.which("bash")

@unittest.skipIf(BASH is None,"bash no disponible")
class SemaforoFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="semaforo-failclosed-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.bin=self.root/"bin"; self.bin.mkdir()
        self.lock=self.root/"target"/".semaforo.lock"
        self.log=self.root/"logs"/"semaforo.log"
        self.trace=self.root/"cargo-unexpected.trace"
        for name in ("mkdir","cat","awk","grep","tr","rm","sleep"):
            self.shim(name,f'exec "/usr/bin/{name}" "$@"\n')
        self.shim("cargo",'printf "UNEXPECTED_CARGO\\n" >> cargo-unexpected.trace; exit 97\n')

    def shim(self,name,body):
        p=self.bin/name;p.write_text("#!/bin/bash\n"+body,encoding="utf-8");p.chmod(0o700)

    def run_case(self,processes=None,tasklist=None,after=8400,ram="10",foreign=None):
        if foreign is not None:
            self.lock.mkdir(parents=True)
            (self.lock/"dueno").write_text("foreign-owner exact\n",encoding="utf-8")
            (self.lock/"creado").write_text(str(foreign)+"\n",encoding="utf-8")
        before={p.name:p.read_bytes() for p in self.lock.iterdir()} if self.lock.exists() else None
        if tasklist is not None:self.shim("tasklist",tasklist)
        # Date is virtual; all counter/trace files stay in this case's cwd.
        self.shim("date","""if [ "${1:-}" = "+%s" ]; then
 n=0
 if [ -f clock-counter ]; then IFS= read -r n < clock-counter; fi
 printf '%s\n' "$((n+1))" > clock-counter
 if [ "$n" = 0 ]; then printf '100000\n'; else printf '%s\n' "$((100000+__AFTER__))"; fi
else printf '2026-10-18T00:00:00Z\n'; fi
""".replace("__AFTER__",str(int(after))))
        env={k:v for k,v in os.environ.items() if not k.upper().startswith(("GIT_","SEMAFORO_")) and k not in ("BASH_ENV","ENV")}
        env.update(PATH=str(self.bin),SEMAFORO_RAIZ=str(self.root),SEMAFORO_AGENTE="synthetic-failclosed",SEMAFORO_SOLO_DECIDIR="1",SEMAFORO_ESPERA_MAX_MIN="0",SEMAFORO_PAUSA_S="0",SEMAFORO_RAM_GB=ram)
        if processes is not None:env["SEMAFORO_PROCESOS"]=processes
        # Confirm the mock PATH cannot resolve real enumerators or pgrep.
        probe=subprocess.run([str(BASH),"--noprofile","--norc","-c","for c in powershell powershell.exe pwsh pwsh.exe pgrep; do if command -v \"$c\" >/dev/null 2>&1; then exit 91; fi; done"],cwd=self.root,env=env,capture_output=True,text=True,timeout=10)
        self.assertEqual(probe.returncode,0,probe.stdout+probe.stderr)
        result=subprocess.run([str(BASH),"--noprofile","--norc",str(SCRIPT),"check"],cwd=self.root,env=env,capture_output=True,text=True,timeout=20)
        self.assertFalse(self.trace.exists(),"No Cargo, including the mock, may be invoked")
        if before is not None and result.returncode==75:
            self.assertTrue(self.lock.exists(),result.stdout+result.stderr)
            self.assertEqual({p.name:p.read_bytes() for p in self.lock.iterdir()},before)
        return result

    def blocked(self,result):
        self.assertEqual(result.returncode,75,result.stdout+result.stderr)
        self.assertNotIn("ADQUIERE",self.log.read_text(encoding="utf-8"))
        self.assertIn("SIN TURNO",self.log.read_text(encoding="utf-8"))

    def admitted(self,result):
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("turno concedido",result.stdout)
        self.assertIn("ADQUIERE",self.log.read_text(encoding="utf-8"))
        self.assertFalse(self.lock.exists(),"Own synthetic lock must be released")

    def test_unknown_count_with_free_lock_still_admits(self):
        # RÉGIMEN 2026-11-06: la cuenta de procesos ya no es puerta del turno.
        self.admitted(self.run_case(processes="?"))

    def test_unknown_count_cannot_steal_stale_lock(self):
        # Cerrojo vencido por gracia (140 min) con cuenta desconocida: duda
        # conserva — el dueño puede estar compilando y no se puede saber.
        self.blocked(self.run_case(processes="?",foreign=100000))

    def test_tasklist_failure_with_free_lock_admits(self):
        self.admitted(self.run_case(tasklist='printf "enumeration failed\\n" >&2; exit 23\n',after=0))

    def test_tasklist_failure_cannot_steal_stale_lock(self):
        self.blocked(self.run_case(tasklist='printf "enumeration failed\\n" >&2; exit 23\n',foreign=100000))

    def test_successful_tasklist_no_matches_recovers_stale_lock(self):
        # 0 compiladores vivos + cerrojo vencido: huérfano legítimo, se libera.
        self.admitted(self.run_case(tasklist='printf "Image Name PID\\nSystem 4\\n"; exit 0\n',foreign=100000))

    def test_numeric_count_with_free_lock_admits_immediately(self):
        # RÉGIMEN 2026-11-06: builds ajenos vivos ya no bloquean el turno.
        r=self.run_case(tasklist='printf "cargo.exe 11\\nrustc.exe 12\\n"; exit 0\n')
        self.admitted(r)
        self.assertNotIn("turno por paciencia",self.log.read_text(encoding="utf-8"))

    def test_numeric_count_never_steals_live_compilers_lock(self):
        # Cerrojo vencido pero con 2 cargo/rustc vivos: el dueño compila, no
        # se le roba el turno.
        self.blocked(self.run_case(tasklist='printf "cargo.exe 11\\nrustc.exe 12\\n"; exit 0\n',foreign=100000))

    def test_fresh_foreign_lock_blocks_even_with_busy_machine(self):
        self.blocked(self.run_case(processes="2",foreign=108400))

    def test_no_enumerator_cannot_steal_stale_lock(self):
        # Sin ningún enumerador disponible la cuenta es «?» y un cerrojo
        # vencido NO se roba.
        self.blocked(self.run_case(foreign=100000))

if __name__=="__main__":unittest.main()
