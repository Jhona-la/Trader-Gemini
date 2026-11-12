# Reglas de Proyecto: Trader Gemini

> Estado vivo del proyecto (qué está integrado, decisiones vigentes, pendientes,
> cómo compilar/probar desde Linux): **`.agents/MEMORIA.md`**. Léelo antes de
> empezar y actualízalo al cerrar cada ola.

<RULE[growth_over_wr]>
- NO ES NECESARIO UN WIN RATE (WR) DEL 100%.
- El objetivo principal, único y fundamental que debe superar el 100% es el **crecimiento exponencial y el interés compuesto cada 3 días**.
- El sistema busca maximizar el crecimiento neto del capital real de la wallet del entorno (Demo o Producción), sin fijar 13 USD ni un rango supuesto de WR. La duplicación cada 72 horas es una meta que exige evidencia OOS después de fees, funding y slippage; no es un rendimiento certificado ni un valor para forzar apalancamiento. Estudiar el universo multivariante, continuo, temporal y espectral.
</RULE[growth_over_wr]>

<RULE[sesiones_concurrentes]>
- Este repo se edita en PARALELO por varias sesiones de agente. Cuatro episodios
  de pérdida de trabajo lo demostraron (V7, plan-maestro, heredoc/git-checkout,
  REHAB). Protocolo obligatorio:
- Ejecuta `git status` con tu `GIT_INDEX_FILE` privado ANTES de cada commit.
  Si hay archivos modificados que NO son de tu cambio, NO los toques: haz
  `git add` SOLO de tus archivos. `git add -A` está PROHIBIDO — en REHAB
  arrastró un test de otra sesión.
- En el árbol compartido están prohibidos `git reset --hard`, checkouts
  forzados y `git stash` para despejar cambios. No sobrescribas, borres ni
  ocultes trabajo de otra sesión ni sus índices, locks, refs o procesos.
- Antes de cablear cualquier ancla (nombre de función, campo, binario), RE-GREP
  el símbolo: otra sesión pudo moverla o renombrarla entre tu lectura y tu edit.
- Antes de detener/relanzar `god_engine.exe`, verifica dueño de la sesión,
  PID Y StartTime del proceso (otra sesión pudo relanzarlo con otro binario).
  Sólo puedes detener el proceso propio cuya identidad acabas de comprobar;
  no mates ni relances procesos ajenos.
- Commits atómicos, un bloque funcional por commit, prefijo del bloque
  (B3.7, F8, etc.) en el mensaje.
- OPS build: si un motor propio mantiene bloqueado el ejecutable que se va a
  enlazar, detén sólo su PID verificado antes de ese build. Un proceso ajeno
  no se mata ni relanza para despejar un enlace. Al relanzar con
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

<RULE[edicion_privada_y_evidencia]>
- El mandato del operador SHA256
  `9bdf8a6de04ae0d986e4f8ffa3f8ca7724ec74d6689ae14fc6009ce2f313201a`
  (§§2,22,32) rige la concurrencia: tras reconciliar las ramas heredadas,
  trabajar sobre `main` con `GIT_INDEX_FILE=.git/index_[agent_id]` propio.
  No usar el índice canónico compartido ni intervenir índices o locks ajenos.
- Escribir primero en `.agent_tx/staging/[agent_id]/[ruta_del_archivo]`;
  conservar Before, candidato y hashes del sujeto. Para Rust, validar el AST
  completo con el parser real antes de instalar el candidato mediante
  reemplazo atómico. El parseo sintáctico no acredita tipos, tests ni CI.
  Recomparar Before con el archivo actual antes de instalar; si cambió,
  reconciliar el cambio concurrente y validar el resultado sin sobrescribirlo.
- Usar el bus reactivo asíncrono para agrupar las pasadas y publicar feedback
  en `.agent_tx/diagnostics/[archivo].json`, sin filas de admisión de agentes
  ni semáforos bloqueantes. La implementación y sus perfiles deben revisarse
  antes de promoverse; un borrador o un PASS en Temp no autoriza un servicio.
- La evidencia de compilación, tests y publicación se vincula al manifiesto
  completo de un snapshot inmutable: fuentes, configuración, herramientas,
  perfil y comandos efectivos. Mantener el sujeto protegido durante toda la
  ejecución; hashes iguales antes/después del árbol vivo no descartan ABA ni
  prueban qué bytes leyó el compilador. Los diagnósticos del árbol vivo son
  feedback; no transferir un PASS a otro sujeto, perfil o SHA.
- Fusionar con `git merge-tree --write-tree` sobre commits exactos, revisar
  conflictos y el diff contra cada padre, y validar el resultado con
  `--all-targets`. Crear el commit con sus padres correctos y actualizar
  `refs/heads/main` mediante CAS que compare el SHA anterior esperado. Si la
  ref cambió, recomponer y validar; no forzarla ni reemplazar archivos ajenos.
  El preview de merge no equivale a commit, CAS, push ni CI verde del SHA final.
</RULE[edicion_privada_y_evidencia]>

<RULE[higiene_disco_y_memoria]>
- El mandato 9bdf (§§2,22,32) sustituye los topes de RAM, SSD e hilos y las
  auditorías bloqueantes anteriores. No medir RAM libre, tamaño de carpetas
  ni espacio de disco como requisito de admisión de un build. No usar
  `scripts/cargo_semaforo.sh`, turnos de recursos ni semáforos bloqueantes.
- Usar todos los CPUs/hilos disponibles para compilar y probar, con
  `CARGO_BUILD_JOBS` acorde a esa disponibilidad, sin recortes artificiales,
  y el enlazador `rust-lld`. No imponer topes de RAM o SSD a los agentes.
- PROHIBIDO crear targets alternativos (`target-verify`, `target_build`,
  `target-cf2`, etc.). Todos usan el único `CARGO_TARGET_DIR` compartido,
  correspondiente al `target/` canónico del repositorio.
- WORKTREES: no retirar un worktree automáticamente por cerrar una tarea o
  fusionar una rama. Antes de retirar uno propio, comprobar dueño, ruta
  absoluta resuelta y pertenencia al espacio previsto, y preservar commits,
  cambios, archivos no rastreados e ignorados que sigan siendo necesarios.
  La ausencia de heartbeat no demuestra abandono. Retirar sólo el checkout
  identificado una vez rescatado su trabajo; no usar `remove --force` ni
  limpiezas automáticas a ciegas. Para trabajo de otra sesión, verificar la
  autorización ya dada y su procedencia; la inactividad no concede permiso.
- BUILD DE TESTS Y SÍMBOLOS (.PDB): Por el mandato explícito del operador del 2026-10-22, el `Cargo.toml` raíz fija `debug = 0` y `strip = "symbols"` en `[profile.dev]` y `[profile.test]`. Sustituye la pauta histórica `debug = 1`: las trazas pierden símbolos/líneas y no se prometen porcentajes de ahorro sin medir. PROHIBIDO forzar `debug = 2` o abrir perfiles/flags locales que dupliquen artefactos fuera del protocolo compartido.
- El nuevo mandato de recursos no exige modificar ese perfil existente.
- Para comprobaciones rutinarias de tipos, usar `cargo check --workspace --all-targets` en vez de `cargo test` para evitar emitir ejecutables y PDBs innecesarios.
- NUNCA crear dumps de logs o runs masivos dentro de `.claude/worktrees/` o la raíz del proyecto. Los logs efímeros deben ir a `logs/` (ignorado en git) y ser rotados.
</RULE[higiene_disco_y_memoria]>
