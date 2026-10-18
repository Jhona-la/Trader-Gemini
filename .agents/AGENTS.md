# Reglas de Proyecto: Trader Gemini

> Estado vivo del proyecto (qué está integrado, decisiones vigentes, pendientes,
> cómo compilar/probar desde Linux): **`.agents/MEMORIA.md`**. Léelo antes de
> empezar y actualízalo al cerrar cada ola.

<RULE[growth_over_wr]>
- NO ES NECESARIO UN WIN RATE (WR) DEL 100%.
- El objetivo principal, único y fundamental que debe superar el 100% es el **crecimiento exponencial y el interés compuesto cada 3 días**.
- El sistema debe estar optimizado para maximizar la curva de capital y lograr duplicar la cuenta (13 USD) exponencialmente, aceptando pérdidas y asumiendo un Win Rate realista (ej. 60-80%), siempre y cuando la esperanza matemática y el crecimiento compuesto resulten en la meta financiera. Sigues pensando como scalping y swing, y debes pensar como un universo multivariante continuo temporal espectral sobre el cual debemos de entender en todas sus partes espectrales.
</RULE[growth_over_wr]>

<RULE[sesiones_concurrentes]>
- Este repo se edita en PARALELO por varias sesiones de agente. Cuatro episodios
  de pérdida de trabajo lo demostraron (V7, plan-maestro, heredoc/git-checkout,
  REHAB). Protocolo obligatorio:
- `git status` ANTES de cada commit. Si hay archivos modificados que NO son de
  tu cambio, NO los toques: haz `git add` SOLO de tus archivos. `git add -A`
  está PROHIBIDO — en REHAB arrastró un test de otra sesión.
- Antes de cablear cualquier ancla (nombre de función, campo, binario), RE-GREP
  el símbolo: otra sesión pudo moverla o renombrarla entre tu lectura y tu edit.
- Antes de matar/relanzar `god_engine.exe`, verifica el PID Y StartTime del
  proceso (otra sesión pudo relanzarlo con un binario distinto al tuyo).
- Commits atómicos, un bloque funcional por commit, prefijo del bloque
  (B3.7, F8, etc.) en el mensaje.
- OPS build: `Stop-Process god_engine` ANTES de `cargo build` (exe bloqueado =
  link fallido y el binario VIEJO se relanza en silencio). Al relanzar con
  PowerShell SIEMPRE `-RedirectStandardOutput` a un archivo — sin eso el
  motor queda ciego de log (45 min de sesión perdidos el 2026-09-15).
  Sin `TG_LOG_FILE`, a `logs/demo_vNN.log`. Para que el log rote y se
  comprima solo (CL-144), define `$env:TG_LOG_FILE = "logs/demo_vNN.log"` y
  redirige stdout a `logs/demo_vNN.stdout.log` (ahí sólo quedan los
  `println!` directos), NUNCA al mismo archivo: el motor no lo abriría y el
  log seguiría entero en stdout. Los segmentos viejos quedan como
  `logs/demo_vNN.log.000001.gz`, `…000002.gz`; `logs/demo_vNN.log.lock` es
  el cerrojo del motor dueño (un segundo motor con la misma ruta escribe en
  su stdout).
- MERGE ENTRE SESIONES: un auto-merge "limpio" NO es un merge correcto. Tras
  integrar ramas de sesiones paralelas, haz diff del resultado contra CADA
  padre y compila con `--all-targets` antes de commitear: el 2026-09-25 dos
  choques semánticos (Swing/Scalping ya erradicados por U-ERR-5; F-009
  editado en un bloque que la otra rama movió a `puertas_del_continuo`)
  pasaron sin marcas de conflicto.
</RULE[sesiones_concurrentes]>

<RULE[higiene_disco_y_memoria]>
- LIMITACIÓN FÍSICA CRÍTICA: Laptop de 16 GB RAM sin GPU dedicada. El espacio en disco C se satura con facilidad por compilaciones incrementales de Rust, archivos .pdb masivos y carpetas target/worktrees olvidados.
- PROTOCOLO OBLIGATORIO DE DISCO:
  1. PROHIBIDO crear carpetas target alternativas (`target-verify`, `target_build`, `target-cf2`, etc.). Usar únicamente `target/` estándar de Cargo.
  2. WORKTREES TEMPORALES: Al fusionar una rama o cerrar una tarea, ELIMINAR el worktree con `git worktree remove --force <path>` y correr `git worktree prune`. PROHIBIDO dejar worktrees en detached HEAD abandonados consumiendo 5-15 GB cada uno (en Ola Ω134 se eliminaron 13 worktrees detached recuperando decenas de GB).
  3. BUILD DE TESTS Y SÍMBOLOS (.PDB): El `Cargo.toml` raíz fija `[profile.dev] debug = 1` y `[profile.test] debug = 1` (line-tables-only). Esto preserva trazas de pánico completas (`RUST_BACKTRACE=1`) pero reduce los archivos `.pdb` y la memoria del enlazador en >85%. PROHIBIDO forzar `debug = 2` en Cargo.toml o flags locales.
  4. Para comprobaciones rutinarias de tipos, usar `cargo check --workspace --all-targets` en vez de `cargo test` para evitar emitir ejecutables y PDBs innecesarios.
  5. NUNCA crear dumps de logs o runs masivos dentro de `.claude/worktrees/` o la raíz del proyecto. Los logs efímeros deben ir a `logs/` (ignorado en git) y ser rotados.
- PROTOCOLO OBLIGATORIO DE RAM:
  1. No mantener múltiples procesos de ejecución en segundo plano (`god_engine`, replays infinitos, listeners zombis) compitiendo por memoria.
  2. Al cerrar tareas, verificar con `Get-Process god_engine` que no haya instancias huérfanas bloqueando archivos o RAM.
</RULE[higiene_disco_y_memoria]>
