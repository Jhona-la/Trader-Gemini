"""Synthetic fail-closed contracts. Never real availability; only external dummy roots/tool mocks."""
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
        env.update(PATH=str(self.bin),SEMAFORO_RAIZ=str(self.root),SEMAFORO_AGENTE="synthetic-failclosed",SEMAFORO_SOLO_DECIDIR="1",SEMAFORO_ESPERA_MAX_MIN="0",SEMAFORO_PAUSA_S="0",SEMAFORO_PACIENCIA_MIN="30",SEMAFORO_RAM_GB=ram)
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

    def test_unknown_after_140_minutes_with_high_ram_is_not_a_count(self):
        self.blocked(self.run_case(processes="?"))

    def test_unknown_cannot_enter_ttl_recovery_or_remove_foreign_lock(self):
        self.blocked(self.run_case(processes="?",foreign=0))

    def test_tasklist_failure_empty_stdout_before_patience_is_unknown(self):
        self.blocked(self.run_case(tasklist='printf "enumeration failed\\n" >&2; exit 23\n',after=0))

    def test_tasklist_failure_empty_stdout_after_patience_is_unknown(self):
        self.blocked(self.run_case(tasklist='printf "enumeration failed\\n" >&2; exit 23\n'))

    def test_successful_tasklist_no_matches_is_legitimate_zero(self):
        self.admitted(self.run_case(tasklist='printf "Image Name PID\\nSystem 4\\n"; exit 0\n'))

    def test_successful_tasklist_numeric_count_retains_lg25b_patience(self):
        r=self.run_case(tasklist='printf "cargo.exe 11\\nrustc.exe 12\\n"; exit 0\n')
        self.admitted(r)
        self.assertIn("turno por paciencia: 2 procesos",self.log.read_text(encoding="utf-8"))

    def test_known_numeric_count_still_blocks_before_patience(self):
        self.blocked(self.run_case(processes="2",after=0))

    def test_known_numeric_patience_still_requires_extra_ram(self):
        self.blocked(self.run_case(processes="2",ram="5"))

    def test_known_numeric_patience_never_bypasses_fresh_foreign_lock(self):
        self.blocked(self.run_case(processes="2",foreign=108400))

    def test_no_enumerator_after_patience_remains_unknown(self):
        self.blocked(self.run_case())

if __name__=="__main__":unittest.main()
