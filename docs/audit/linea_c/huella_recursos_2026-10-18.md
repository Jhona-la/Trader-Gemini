# Huella de recursos del PC del operador — 2026-10-18

Claude (Línea C), sesión en el PC del operador. Medido entre las 06:26 y las
07:10 UTC (01:26–02:10 Bogotá) sin compilar nada. Pedido del operador
(06:15 UTC): «el proyecto está desbordando la memoria». Presupuestos: 16 GB
de RAM y 300 GB de SSD para la carpeta del proyecto.

## 1. Resumen

- **La presión es de RAM, no de disco, y no la causan las compilaciones.**
  A las 06:26 UTC quedaban 1,7 GB libres de 23,4, con 45,7 GB de memoria
  comprometida (límite 63,9; el pagefile usaba 4,3 GB). En ese instante no
  había ningún `cargo`/`rustc` vivo.
- **Lo nuevo frente al 17-10 son VS Code y Edge.** VS Code (abierto el 17-10
  a las 21:22 UTC) ocupa 7,8 GB privados y 3,4 GB residentes, y trabaja sobre
  OTRA carpeta: `Documents/GitHub/Proyectos/Proyectos`, no Trader Gemini.
  Edge se abrió a las 06:18 UTC y ocupa 2,6 GB residentes. Juntos explican
  los ~6 GB que separan los 7–8,8 GB libres del 17-10 por la tarde de los
  1–3 GB de ahora.
- **Disco dentro del presupuesto**: la carpeta ocupa ≈ 82 GB de 300, y C:
  tiene 360 GB libres. Pero `target/` crece ≈ 7 GB cada 10 h (37 → 44 →
  51,5 GB en 20 h), sobre todo por **artefactos duplicados por worktree y por
  variante de `RUSTFLAGS`**.
- **Fuera del repo, lo grande es de las apps de los agentes**:
  `~/.zcode/cli/rollout` (36,9 GB, un único `model-io-*.jsonl` de 26,8 GB),
  `~/.gemini` (24,3 GB) y `~/.codex` (10,8 GB, con un `target/` privado de
  1,7 GB en el worktree ROOT).
- **El suelo de RAM del semáforo se está saltando**: de las 30
  adquisiciones desde las 01:44 UTC, 16 se hicieron con 1,0–2,9 GB libres
  (`jobs=1`; 12 de AGY y 4 de la Línea G), por debajo del suelo de 4 GB
  (§3.3).

## 2. RAM

### 2.1 Estado (06:26 UTC)

| Medida | Valor |
|---|---|
| Física total / libre | 23,38 GB / 1,73 GB (2,19 GB a las 06:58) |
| Caché standby / modificada | 1,71 GB / 0,03 GB |
| Comprometida / límite | 45,7 GB / 63,9 GB |
| Pagefile (asignado / en uso / pico) | 37,0 GB / 4,3 GB / 7,4 GB |
| Procesos | 552 (19,8 GB residentes, 32,8 GB privados) |
| `cargo`/`rustc`/`god_engine` vivos | 0 (06:26) · 1 de AGY a las 07:05, por el semáforo |

### 2.2 Por familia (memoria privada; WS = residente)

| Familia | N | WS (GB) | Privada (GB) | Nota |
|---|---:|---:|---:|---|
| VS Code + extensiones | 26 | 3,41 | 7,76 | extensión host 2,45 · Gemini Code Assist (`cloudcode_cli`) 1,24 · Copilot runtime 0,95 (+2 servidores MCP de node desde 06:17) · Kilo 0,77 · Pylance 0,55. **Carpeta abierta: otro proyecto** |
| ChatGPT / Codex | 73 | 3,46 | 5,49 | `codex.exe app-server` 1,45 · renderers 0,93 + 0,33 |
| Sistema y otros | 251 | 4,21 | 5,14 | Defender 0,62 · explorer 0,33 · LGHUB 0,30 · Free Download Manager 0,22 |
| Claude (app + 4 sesiones) | 15 | 2,06 | 3,79 | sesiones de 05:44 (17-10), 11:35 (Fable), 06:13 y 06:25 UTC |
| Antigravity | 12 | 1,32 | 2,18 | `language_server` 0,88 |
| ZCode (GLM) | 12 | 0,34 | 2,05 | |
| Edge | 35 | 2,62 | 2,00 | abierto 06:18 UTC desde el explorador |
| WorkBuddy | 17 | 0,91 | 1,84 | |
| shells / node / git | 97 | 1,05 | 1,12 | 9 `git fsmonitor--daemon` (≈ 0,5 GB), uno por worktree |
| Qoder | 12 | 0,42 | 1,11 | |
| Riot / Valorant | 2 | 0,02 | 0,26 | |

Los 10 procesos con más memoria privada: extensión host de VS Code (2,45 GB),
`codex.exe` (1,45), `cloudcode_cli` (1,24), `copilot-runtime` (0,95),
renderer de ChatGPT (0,93), `language_server` de Antigravity (0,88), renderer
de Claude (0,80), `kilo` (0,77), dos sesiones de Claude Code (0,70 y 0,67).

### 2.3 Evolución (RAM libre al adquirir el semáforo, `logs/semaforo.log`)

| Ventana (UTC) | RAM libre al adquirir | Qué cambió |
|---|---|---|
| 17-10 15:37–19:53 | 5,8–8,8 GB, `jobs=2` | estado de las auditorías de la Línea G y de Fable |
| 17-10 21:22 | — | se abre VS Code con 4 extensiones de IA |
| 18-10 01:44–04:10 | 1,0–6,1 GB, `jobs=1` | AGY y Línea G compilan con 1,0–2,9 GB libres |
| 18-10 04:31–05:24 | 5,4–6,9 GB | ventana mejor |
| 18-10 05:28–06:14 | — | check release `-j1` de Codex ROOT (adaptador V2, no aparece en el log) |
| 18-10 06:13–06:25 | — | dos sesiones nuevas de Claude Code; Edge a las 06:18 |
| 18-10 06:28–07:05 | 1,9–3,1 GB, `jobs=1` | AGY compila por debajo del suelo |

### 2.4 Comparación con las auditorías anteriores

| Auditoría | RAM libre | Apps de agentes | Proyecto | target |
|---|---|---|---|---|
| Línea G, 17-10 ~10:30 Bogotá (AGENT_SYNC §4) | 7,1 GB | ≈ 5,5 GB (WS) | 69 GB | 37 GB (debug 29 + release 8,3) |
| Fable, 17-10 ~11:50 Bogotá (MEMORIA, ola F1) | 6,5–8 GB | ~15 apps residentes | 76 GB | 44 GB |
| **Línea C, 18-10 01:26 Bogotá** | **1,7 GB** | **≈ 14,5 GB WS / 26,2 GB privados (sin VS Code ni Edge: ≈ 8,5 / 16,5)** | **≈ 82 GB** | **51,5 GB (debug 42,2 + release 9,3)** |

Sin VS Code ni Edge, los agentes ocupan ≈ 8,5 GB residentes, frente a los
5,5 GB del 17-10. Las sesiones duran más: cada app de agente creció, y hay
cuatro sesiones de Claude Code abiertas.

## 3. Qué explica la presión y qué conviene

### 3.1 Causa

1. **VS Code sobre otro proyecto** (7,8 GB privados, 3,4 residentes). Es la
   mayor pieza y no trabaja en Trader Gemini.
2. **Edge** (2,6 GB residentes, desde 06:18 UTC).
3. **Apps de agentes residentes** (≈ 8,5 GB residentes): ChatGPT/Codex 3,5,
   Claude 2,1, Antigravity 1,3, WorkBuddy 0,9, Qoder 0,4, ZCode 0,3.
4. Las compilaciones (0,4–0,9 GB por `rustc`) no son la causa; son las que
   sufren: con < 4 GB libres no hay turno, y quien se lo salta pagina.

### 3.2 Para el operador (sólo él puede hacerlo)

- Cerrar VS Code mientras no se use (sobre todo Copilot en modo agente,
  Gemini Code Assist y Kilo): ≈ 3,4 GB residentes y 7,8 privados.
- Cerrar Edge si no hace falta: ≈ 2,6 GB.
- Cerrar las apps de agentes que no tengan turno hoy (Qoder figura HUÉRFANO
  desde el 10-10 y su app sigue abierta: 0,4 GB; WorkBuddy: 0,9 GB).
- Con eso la RAM libre vuelve a 7–9 GB, y el semáforo con `jobs=2`
  funciona sin que nadie pagine.

### 3.3 Reglas nuevas propuestas para todos los agentes

1. **Nadie baja el suelo de RAM del semáforo por debajo de 3 GB.** Hoy se
   compila con 1,0–2,9 GB libres (`SEMAFORO_RAM_MIN_GB` rebajado o
   `jobs=1` con paciencia). Con < 3 GB libres un `rustc` pagina y se lleva
   la RAM de los demás. Si no hay turno, se espera o se avisa al operador
   para que cierre apps; no se fuerza.
2. **Una sola sesión viva por línea.** Al terminar una ola, cierra tu sesión
   (o tu app) en vez de dejarla residente. Hoy hay cuatro sesiones de Claude
   Code y cinco apps de IA abiertas a la vez.
3. **`RUSTFLAGS` único.** Usa el `.cargo/config.toml` del repo
   (`target-cpu=native`, `opt-level=3`) y no exportes otros `RUSTFLAGS`: cada
   variante duplica todas las dependencias en el `target/` compartido (§4.3).
4. **Los logs de sesión de las apps de agente son de su dueño**: GLM/ZCode
   tiene 36,9 GB en `~/.zcode/cli/rollout` (§4.4). Que cada agente mida y
   rote sus propios rastros fuera del repo al cerrar ola.
5. **Ningún `target/` privado.** El worktree ROOT de Codex tiene
   `~/.codex/worktrees/root-lg5-main-1015/Trader Gemini/target` (1,71 GB) y
   el de base-audit otro de 0,14 GB (ya señalado por la Línea G el 17-10).

## 4. Disco

### 4.1 Carpeta del proyecto (primer nivel, 06:27 UTC)

| Carpeta | MB | Nota |
|---|---:|---|
| `target/` | 51 528 | §4.2 |
| `data/` | 24 501 | tapes del operador: intocable |
| `.git/` | 3 122 | |
| `graphify-out/` | 817 | 3 instantáneas no versionadas (el máximo que conserva la política) |
| `.mimosa/` | 741 | `hook-state/` 738 MB, 45 845 archivos de la sesión `sess_0108e907` (la misma del rollout de ZCode); ignorado por git |
| `.claude/` | 519 | worktrees de Claude, ≈ 125 MB cada uno, sin `target/` propio; `med/` 18 MB (§4.6) |
| `results/` | 253 | ignorado |
| `.venv/` | 202 | |
| `logs/` | 148 | |
| `crates/` | 64 | |
| resto | ≈ 70 | |
| **Total** | **≈ 82 000** | 27 % del presupuesto de 300 GB |

Volumen C: 930 GB, 570 usados, **360 GB libres**.

### 4.2 `target/`

| Ruta | MB |
|---|---:|
| `debug/` | 42 246 |
| `debug/deps` | 29 469 |
| `debug/incremental` (1 117 directorios, ninguno con más de 3 días) | 11 961 |
| `debug/build` | 770 |
| `release/` | 9 320 |
| `release/deps` | 8 193 |
| `release/build` | 719 |

| `debug/deps` | n | GB | > 72 h |
|---|---:|---:|---|
| `*.exe` | 1 003 | 2,47 | 454 (1,08 GB; todos del 14-10) |
| `*.pdb` | 689 | 4,79 | 0 |
| `*.rlib` | 2 023 | 14,38 | — |
| `*.rmeta` | 5 274 | 5,37 | — |
| `*.dll` | 578 | 1,68 | — |

`release/deps`: 8 exe (0,04 GB), 76 pdb (0,16), 1 639 rlib (4,33), 2 953
rmeta (3,03). El exe más grande es de 58 MB (`backtest_engine-*.exe`); la
media es ≈ 2,5 MB. El peso de las pruebas está en los `.pdb` (4,8 GB), no en
los `.exe`.

### 4.3 Duplicación en el `target/` compartido

- 2 023 `rlib` para 381 crates distintos (5,3 por crate): `syn` ×44,
  `quantum_arena` ×24, `windows`/`windows_core` ×23, `reqwest` ×22,
  `os_guardian` ×21.
- `debug/.fingerprint` tiene 11 766 entradas; sólo `god-engine-core` tiene
  1 087 y `trader-gemini-v5` 938.
- Causas: cada worktree compila los crates del workspace con su propia ruta
  (huella distinta), y cada variante de `RUSTFLAGS`, features o toolchain
  rehace las dependencias. Los `rlib` por día: 6,6 GB el 15-10, 5,0 el 16-10
  y 2,4 el 17-10.
- El crecimiento de `target/` (≈ 7 GB cada 10 h) viene de aquí, no de los
  binarios de prueba. Con 360 GB libres no es urgente. La regla 3 de §3.3
  (`RUSTFLAGS` único) lo frena. Limpiar `incremental` entre olas lo recorta,
  pero no lo detiene.

### 4.4 Fuera del repo (sólo medido; nada borrado)

| Ruta | GB | Última escritura | Dueño | Candidato |
|---|---:|---|---|---|
| `~/.zcode/cli/rollout` | 36,91 | 17-10 14:24 | GLM / ZCode | `model-io-sess_0108e907….jsonl` de **26,8 GB** (17-10) y `model-io-sess_0af6bea7….jsonl` de **9,3 GB** (25-09). Son logs de E/S del modelo: rotar o comprimir |
| `~/.gemini/antigravity/brain` | 11,19 | 13-10 | AGY | revisar |
| `~/.gemini/antigravity-ide` | 9,86 | 22-09 (`brain` 7,77) y 04-10 (`conversations` 2,07) | AGY (IDE antiguo) | sin escrituras desde el 04-10 |
| `~/.gemini/antigravity/conversations` | 2,50 | 18-10 | AGY | activo |
| `~/.codex/audit-evidence` | 2,83 | 08-10 | Codex | recibos: conservar |
| `~/.codex/tmp` | 2,54 | 18-10 | Codex | `root-continuity-20261015` 1,07 activo; `skills-batch-20261015` 0,70 del 15-10 |
| `~/.codex/worktrees` | 2,55 | 15-10 | Codex | `root-lg5-main-1015` 1,90 (con `target/` privado de 1,71); `base-audit` 0,27 (con `target/` de 0,14) |
| `~/.vscode/extensions` | 5,34 | 17-10 | operador | |
| `~/.vscode/agent-plugins` | 1,79 | 16-10 | operador | |
| `~/.workbuddy-ai/workspace` | 2,91 | 07-09 | WorkBuddy | sin escrituras desde el 07-09 |
| `~/.workbuddy-ai/logs` | 1,04 | 18-10 | WorkBuddy | rotar |
| `~/.rustup` | 3,93 | — | toolchains | `nightly-2026-06-30` 1,28 (fijado en `rust-toolchain.toml`), `nightly` 1,35 (30-06), `stable` 1,30 (15-06) |
| `~/.antigravity` | 2,36 | 19-05 | AGY | `extensions` 2,36 (los 52 GB del 12-10 ya no están) |
| `%TEMP%` | 1,85 | — | varios | instaladores de WorkBuddy (0,46) y VS Code (0,46) del 15-10 |
| `~/.cargo` | 1,03 | — | registro de crates | |
| `~/.qoder` | 0,72 | — | Qoder | |
| `~/.claude` | 0,60 | — | Claude | |

### 4.5 Limpieza aplicada (paso 2)

| Acción | Resultado | Liberado |
|---|---|---|
| `bash scripts/higiene_recursos.sh` (simulación) | sólo tocaba `logs/demo_v32.log.gz` (> 30 d) y `git worktree prune` | — |
| `APLICAR=1 bash scripts/higiene_recursos.sh` | hecho (`logs/higiene.log`, 06:58:49Z) | ≈ 1 MB |
| `git branch -d` de ramas locales fusionadas sin worktree | sólo `claude/beautiful-tharp-80c51c` (la mía); las demás fusionadas tienen worktree abierto (jolly, fable, dos de Codex) | 0 |
| Exes de prueba de más de 72 h (`EXES_DIAS=2`, véase §4.6) | primer intento omitido (el guard vio a AGY compilando a las 07:05Z). A las 07:10Z, sin cargo vivo y con el cerrojo canónico del semáforo tomado por mí durante el borrado (anotado en `logs/semaforo.log`): **454 exes del 14-10 borrados** (no tenían `.pdb`) | ≈ 1,1 GB (`df`) |
| `disk_audit.sh CLEAN_INCREMENTAL=1` | **no aplicado**, a propósito | 0 (12 GB) |

Por qué no se tocó `incremental` (12 GB): la política lo limpia **entre
olas**, y ahora AGY y Fable están en bucles de editar y probar (todas sus
carpetas tienen menos de 3 días). Borrarlo forzaría a recompilar desde cero
los crates del workspace, justo cuando falta RAM. El disco no aprieta: con
360 GB libres no compensa. Que lo haga quien cierre la última ola del día.

Total liberado: ≈ 1,1 GB. El disco no era el problema.

### 4.6 Hallazgos menores

- `scripts/higiene_recursos.sh`: con `EXES_DIAS=3` usa `find -mtime +3`, que
  en `find` significa «4 días o más». Los 454 exes del 14-10 (3 d 16 h) no
  entran hasta mañana. Para la política escrita («> 3 d») haría falta
  `-mmin +4320`. El comentario «≈ 1 GB cada uno» tampoco vale ya: con
  `debug = 1` los exes son de 2–58 MB, y el peso está en los `.pdb`.
- Hay un `bash scripts/cargo_semaforo.sh check --workspace --all-targets`
  (PID 28240, desde 06:17 UTC) sin padre vivo, esperando turno. Por la hora
  y el comando parece el check que Fable encadena tras su test de las
  06:17. No lo toco.
- `.claude/worktrees/med/` (18 MB): ejecutables del medidor de la Línea C
  (CL-60, 12-10) y sus logs. No es un worktree registrado; se puede borrar
  cuando se cierre la tanda de rendimiento de la Línea C.

## 5. Rescate de la Línea C (paso 3)

| Rama / lugar | Estado | Respaldo |
|---|---|---|
| `claude/tanda-cf5` | 19 commits fuera de main (CL-109…CL-124 + doc + rescate de GLM 132), 286 por detrás | **empujada a origin** a las ~06:50 UTC (`153abeed7`) |
| Otras ramas `claude/*` locales | `jolly-montalcini-cee407` (GLM 135, fusionada), `linea-h-ola4` (Línea H, activa) | no son de la Línea C |
| stashes | ninguno | — |
| Modelos de las 18 monedas | no hay ninguna rama, stash ni worktree con ese trabajo. En `models/` hay 8 `*USDT_MOTOR` (ADA, ATOM, BNB, BTC, NEAR, SOL, XLM, XRP), 6 candidatos USDT y 10 `*FDUSD_MOTOR` del 02-07 | — |

### 5.1 Qué hay en `claude/tanda-cf5` (C-F5a) y si sigue aplicando

| Commit | Contenido | Sobre main |
|---|---|---|
| CL-109 `292f67219` | W-1 del #60: una reconexión del WS sólo corta el emparejamiento de StatArb (no reinicia sumas) | sigue faltando en main; aplica sin conflictos |
| CL-110 `952ad1070` | EST-1: sumas de retardo 1 centradas con sus propias filas | sin conflictos |
| CL-111 `fa0f5cf56` | EST-5, W-5: sin la resolución de la base, StatArb no vota | sin conflictos |
| CL-112 `b625e2628` | EST-2/3/6: documentación del estimador | sin conflictos |
| CL-113 `17491ba33` | W-2, W-4: dos pruebas de StatArb que no probaban lo que decían | sin conflictos |
| CL-114 `47044fa5c` | T1-01, T1-06 (Línea G): el freno de la estrategia mide cada cierre con la geometría de su posición | sin conflictos. La C-F5c (#70) los daba aún por abiertos |
| CL-115 `0364429a3` | T1-04, T1-05: sondas y posiciones sin stop propio no alimentan el freno | **conflicto** en `god-engine-core/src/lib.rs` y `veto_registry.rs`. Hay que revisarlo frente a LG-27b (el detector sólo aprende de cierres con SL propio) |
| CL-116 `886391d53` | T1-08: el Monte Carlo del freno duro fuera del cerrojo | **conflicto** en `drawdown.rs`. Probablemente superado por F-A4 (Fable/AGY: caché con vecino conservador e hilo de fondo) |
| CL-117 `bef09f01c` | T1-07: el sistema inmune dice cuál es el rearme real | sin conflicto propio (sólo el host) |
| CL-118 `42d9874e0` | T2-5: la evidencia de una moneda fuera del universo no se pierde al guardar | en `lib.rs`, revisar |
| CL-119 `f0b02c512` | T2-7/T2-8/T3-05: cobertura de la prueba del estado aprendido | **conflicto** en `sonda.rs` |
| CL-120 `68762e7b1` | T2-4/T2-6: el estado aprendido se escribe en otro hilo y sólo cuando cambia | sin conflicto propio |
| CL-121 `92dde1092` | T3-09: se retira la prueba de racha del drawdown | **conflicto** en `drawdown.rs` (tras CL-130 de #70) |
| CL-122 `85f7f2605` | T3-06/07/08: contratos sin comentarios, registro de vetos | **conflicto** en `cf4_freno_duro.rs` y `veto_registry.rs` |
| CL-123 `ddb79e498` | SHIB → `1000SHIBUSDT` en roster, cópulas y spot | sigue faltando: main no tiene `1000SHIBUSDT` en `bootloader.rs`. Sin conflicto propio |
| CL-124 `a8fea182e` | T3-04: el generador de cópulas exige el mes | sin conflicto propio |
| `26887a7bc` | ADR-0017, memoria, buzón | `MEMORIA.md` en conflicto (unión) |
| `153abeed7` | rescate de GLM 132: `zz_medir_duracion_sonda.rs` | — |

Fusión de prueba (`git merge-tree`) de la rama entera sobre main: 6
conflictos (`MEMORIA.md`, `god-engine-core/src/lib.rs`, `drawdown.rs`,
`sonda.rs`, `veto_registry.rs`, `cf4_freno_duro.rs`). CL-109…CL-114 entran
limpios: son el primer bloque a portar (W-1 y la revisión del #60). CL-115,
CL-116, CL-119, CL-121 y CL-122 hay que rehacerlos sobre main, teniendo en
cuenta F-A4 (Fable/AGY), CL-130…CL-134 (#70) y LG-27b (Línea G). El resto
(CL-117, CL-118, CL-120, CL-123, CL-124) se porta tal cual tras resolver el
contexto. No se borra nada de la rama.

## 6. Cómo se midió

- RAM: `Win32_OperatingSystem`, `Get-Counter \Memory\*`,
  `Win32_Process` (WorkingSet y PrivatePageCount, agrupados por ruta del
  ejecutable y línea de comando; cadenas de padres para Copilot y Edge).
- Disco del proyecto: `du -sm` de Git Bash (los worktrees, uno a uno: `du`
  no cuenta dos veces un mismo archivo entre argumentos). Fuera del repo:
  `robocopy /L /S /BYTES` (lista sin copiar; mucho más rápido que `du` sobre
  `~/.rustup`).
- `.exe`/`.pdb`: `find -maxdepth 1` con `-mtime`/`-mmin`.
- Ramas: `git log origin/main..rama`, `git cherry`, `git merge-tree
  --write-tree --name-only`.
