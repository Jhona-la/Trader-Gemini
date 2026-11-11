# MEMORIA DEL PROYECTO — Trader Gemini (estado vivo, versión corta)

> Este archivo lo importan `CLAUDE.md` y `AGENTS.md` en **cada** llamada de cada agente.
> Límite duro: **menos de 30 KB**. La historia completa NO va aquí: va a
> `.agents/memoria_archivo/` (ver índice al final). Al cerrar una ola, escribe aquí sólo
> lo que cambia el estado vigente (3 a 6 líneas) y el detalle en tu entrada de
> `AGENT_SYNC.md` o en el archivo histórico del mes.
>
> Reescrito por el integrador el 2026-11-10 (tarea 1 del encargo del operador). El
> contenido anterior (1.297.833 bytes, 11.671 líneas, SHA256
> `1184474E20D5080E9EFEA70AF3089FF750872B4DD3E80B2CF565E81A2F2FAB53`) está íntegro,
> sin cambiar una línea, en los 8 trozos del índice; su concatenación en orden
> reproduce ese SHA256.

## 1. Meta

- Objetivo único (regla `growth_over_wr` de `.agents/AGENTS.md`): crecimiento neto del
  capital real de la wallet del entorno (Demo o Producción) con interés compuesto. **No
  hace falta WR del 100 %.** La duplicación cada 72 h es una **meta** que exige
  evidencia OOS después de fees, funding y slippage; no es un rendimiento certificado
  ni una razón para forzar apalancamiento.
- Método: universo multivariante, continuo, temporal y espectral; una posición, un
  horizonte continuo (τ medida); fail-closed ante datos ausentes (`N/D`, nunca `0.0`
  inventado).

## 2. Estado actual (2026-11-10)

- **Producción: NO-GO. Demo: NO-GO. OOS: NO-GO.**
- ROI / PnL / WR / Sharpe / EV por operación: **`N/D`**. No existe certificación
  +100 %/72 h. No se ejecutó Mainnet, ni credenciales, ni attestation.
- Admisión OOS de desarrollo: recibo `data/DEV_V41_2026-06_08/admission-v4.json`
  (SHA `94aede1b…bb26ed`, 67.011.806 filas) = `INSUFICIENTE`, componentes externos
  `unverified`. Es desarrollo, no reserva OOS final.
- `main` local avanza por CAS sin push (último código integrado por Kimi V42 y Sol V43/M1
  el 2026-11-10; `origin/main` va detrás). Push sólo con permiso del operador.
- Kimi Quant Sr V43: **sin cuota** desde 2026-11-10 15:17 -05:00 (su trabajo queda
  preservado en `.agent_tx/staging/KIMI-QUANT-SR-V42` y `KIMI-QUANT-SR-V43`).
- Sol M1 paso 1 cerrado (2026-11-10 21:33): `telemetry-server::trade_ledger`, ledger por
  operación con bruto, fees, funding, slippage y neto; `NaN` = N/D. Paso 2: informe
  ROI/PnL/WR antes y después de fees; paso 3: conectar el panel (hoy el panel de
  PnL/fees muestra siempre «N/D»: `causal_report` = `None` en `mmap_viz_stream.rs` y
  `/api/viz/causal`).
- Latencias baseline documentadas (Risk Fase3 ≈ 87 µs, F1-108 temporal 325 µs, Kähler,
  Yang-Mills): no se suben límites para ponerlas en verde.
- Estado vivo de cada agente: `AGENT_SYNC.md` (entradas abiertas arriba y §1 Registro).

## 3. Decisiones vigentes (no reabrir sin evidencia nueva)

### Proceso y concurrencia (detalle en `.agents/AGENTS.md`)
- Varias sesiones editan en paralelo. Índice privado
  `GIT_INDEX_FILE=.git/index_[agent_id]`; `git add` sólo de tus archivos.
  Prohibidos `git add -A`, `git reset --hard`, checkouts forzados, `git stash`,
  `git clean`. No tocar índices, locks, worktrees ni procesos ajenos.
- Escribir primero en `.agent_tx/staging/[agent_id]/[ruta]`, comparar el Before con el
  archivo vivo y reemplazar de forma atómica; si cambió, reconciliar.
- Fusionar con `git merge-tree --write-tree`, diff contra **cada** padre y
  `--all-targets` antes de commitear: un auto-merge limpio no es un merge correcto
  (2026-09-25: Swing/Scalping y F-009 pasaron sin marcas de conflicto).
- `refs/heads/main` se avanza por CAS con el SHA anterior esperado. Sin push sin
  permiso del operador. Commits atómicos con prefijo de bloque.
- Mandato 9bdf (§§2,22,32) sustituye semáforos y topes de RAM/SSD anteriores: no usar
  `scripts/cargo_semaforo.sh`. Un único `target/` compartido; prohibidos targets
  alternativos. Worktrees: no retirar ajenos ni por inactividad.
- `god_engine.exe`: verificar dueño, PID y StartTime antes de detener; relanzar con
  `-RedirectStandardOutput`; log rotado con `TG_LOG_FILE=logs/demo_vNN.log` y stdout a
  `logs/demo_vNN.stdout.log` (nunca el mismo archivo).

### Motor (las que más se rompen al editar; historia en el archivo 06)
- **D-743 `puertas_del_continuo`**: una sola función con las cuatro puertas
  (viabilidad, invariante bayesiano D-472, ponderación ML F-009, vetos CVD/muro L2)
  para `fast_intent` y `slow_intent`. Cualquier cambio a las puertas va ahí.
- **D-745 / U-ERR-5**: una posición, un horizonte. `PositionHorizon` sólo tiene
  `Continuous`; el horizonte real es `Position::entry_tau_ms`. No reintroducir
  Swing/Scalping.
- **D-742**: el espectro temporal no opina con escalas τ no observadas.
- **D-753 + paridad**: `train_forest` reproduce el tape por
  `GodEngineCore::process_event` y exige paridad 48D con tolerancia cero contra
  `build_54d_tensor`.
- **D-752/D-756**: convicción por rama = cota de Wilson de su propio historial, sin
  literales.
- Riesgo: Kelly micro sin piso ficticio (se rechaza lo que no cabe en el tope de
  ruina), `min_notional` publicado del símbolo, correlación = Pearson real con signo.
- Evidencia financiera: gross no finito ⇒ `N/D`, nunca coerción a `0.0` seguida de
  restar fees (defecto de `causal_attribution.rs` reproducido por Sol 2026-11-09).
- Perfiles: `[profile.dev]` y `[profile.test]` con `debug = 0` y `strip = "symbols"`
  (mandato del operador 2026-10-22). No forzar `debug = 2`.

## 4. Cómo compilar y probar

### Windows (PC del operador, entorno principal)
- Toolchain fijado: `nightly-2026-06-30` (`rust-toolchain.toml`; lo comprueba
  `crates/quantum-arena/tests/toolchain_pin_contract.rs`). Target
  `x86_64-pc-windows-msvc`, enlazador `rust-lld`, `sccache`, `checksum-freshness`
  (contrato `rb_h2_frescura_por_contenido_contract.rs`).
- Revisión de tipos: `cargo check -p <crate>`; general:
  `cargo check --workspace --all-targets` (hoy ~18 min con carga; evita emitir
  ejecutables y PDB).
- Pruebas de un crate: `cargo test -p <crate> --locked`. Las CI usan
  `cargo +nightly-2026-06-30 test <paquetes> --locked -- --test-threads=1`
  (`.github/workflows/unit-suites.yml`, `replay-contracts.yml`).
- No definir `RUSTFLAGS` de entorno: `build_loop.ps1` y `run_sim_5.ps1` lo rechazan; la
  configuración compartida está en `.cargo/config.toml` (dueño: integrador).
- Pruebas Python de scripts: `python -m unittest discover -s scripts/tests`.
- Una compilación pesada por agente a la vez (24 GB de RAM para 4 a 6 agentes).

### Linux (sesiones cloud)
- `os-guardian` es sólo-Windows; `cargo check` nativo en Linux falla también en main.
- `rustup target add x86_64-pc-windows-gnu`, instalar `gcc-mingw-w64-x86-64` y
  `cargo check --target x86_64-pc-windows-gnu --workspace --all-targets`.
- Tests: `wine64` y `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=/usr/lib/wine/wine64
  WINEDEBUG=-all`, luego `cargo test --target x86_64-pc-windows-gnu -p <crate> --lib`.
- Demo desde la nube: los hosts de Binance dan 403 y no hay claves; la demo se corre
  en el PC local salvo que el operador habilite red y variables `USE_TESTNET=true`,
  `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_SECRET_KEY`.

## 5. Mapa de áreas (encargo del operador, 2026-11-10)

| Agente | Área | Crates / archivos |
| --- | --- | --- |
| Sol (WorkBuddy) | Evidencia financiera (M1) | telemetry-server, telemetry-engine, audit-engine, flight-recorder |
| Codex QRA | Ejecución y paridad (M2) | execution-engine, risk-engine, `tests/host_oms_contract.rs` |
| AGY (Antigravity) | Señales y estrategias (sus olas) | signal-engine, strategy-core, feature-engine, dark-alpha-engine |
| Codex RB (DeepSeek Harness) | Datos y OOS | data-ingest, data-pipeline, storage-engine, backtest-engine |
| Integrador (Claude Code) | Integración e infraestructura | `Cargo.toml`, `Cargo.lock`, `.cargo/`, `.github/`, `scripts/`, `AGENT_SYNC.md`, `.agents/MEMORIA.md` |
| Claude (Cowork) | Coordinación | plan "Gestión de agentes de Trader Gemini" |

Crates sin dueño explícito en el encargo (`os-guardian`, `quantum-arena`,
`evolution-engine`, `god-engine-core`, `graph-architecture`, `omniscient-registry`,
`metacortex-engine`, `src/`): coordinar en `AGENT_SYNC.md` antes de tocarlos.

## 6. Archivos grandes: no leer enteros

`AGENT_SYNC_ARCHIVO.md`, `.agents/memoria_archivo/*`, `INFORME_FORENSE_MAESTRO.md`,
`FORENSIC_INTELLIGENCE_AUDIT.md`, `ATLAS_ANALITICO.md`: buscar con `rg -n` y leer sólo
el trozo necesario. No importarlos con `@` desde `CLAUDE.md`/`AGENTS.md`.

## 7. Índice del archivo histórico (`.agents/memoria_archivo/`)

Trozos contiguos del `MEMORIA.md` anterior, en su orden original (que no es
cronológico: las entradas nuevas se escribían arriba y algunas se añadían al final).
Líneas = líneas del archivo original.

| # | Archivo | Líneas | Bytes | Contenido |
| --- | --- | --- | --- | --- |
| 01 | [MEMORIA_01_2026-11-02_a_2026-11-10.md](memoria_archivo/MEMORIA_01_2026-11-02_a_2026-11-10.md) | 1-88 | 19.553 | Kimi V42 (CAS de código, limpieza de ramas, pila F1-049/F1-053), V35 F2A/F3/F4, corte Codex 11-02 |
| 02 | [MEMORIA_02_2026-10-19_a_2026-10-30.md](memoria_archivo/MEMORIA_02_2026-10-19_a_2026-10-30.md) | 89-2378 | 277.289 | Codex QRA T1/V3r2, Kimi V33/V34, Zero-Wait, censo C-F7b (Línea C), AGY Ω164-Ω220 |
| 03 | [MEMORIA_03_2026-10-15_a_2026-10-19.md](memoria_archivo/MEMORIA_03_2026-10-15_a_2026-10-19.md) | 2379-4068 | 232.645 | Codex RB (RET-45, RB-H1), AGY Ω125-Ω168, Línea C C-F5/C-F6/C-F7, BASE15 |
| 04 | [MEMORIA_04_2026-10-10_a_2026-10-14.md](memoria_archivo/MEMORIA_04_2026-10-10_a_2026-10-14.md) | 4069-6621 | 268.568 | AGY Ω61-Ω124 (desacoplamientos, CI) |
| 05 | [MEMORIA_05_2026-10-02_a_2026-10-09.md](memoria_archivo/MEMORIA_05_2026-10-02_a_2026-10-09.md) | 6622-9437 | 264.008 | AGY Ω10-Ω60, Qoder R7 (fichas de auditoría) |
| 06 | [MEMORIA_06_2026-09-25_a_2026-09-30.md](memoria_archivo/MEMORIA_06_2026-09-25_a_2026-09-30.md) | 9438-10806 | 107.043 | Olas 1-7, integración PR #4 «espectro predictivo» (decisiones D-742…D-757, cómo compilar desde Linux, hallazgos abiertos), Codex 09-28/09-30 |
| 07 | [MEMORIA_07_2026-10-01_a_2026-10-21.md](memoria_archivo/MEMORIA_07_2026-10-01_a_2026-10-21.md) | 10807-11556 | 112.889 | Olas 8-9, R4, B15, ROOT, AGY Ω178-Ω198, F7 |
| 08 | [MEMORIA_08_2026-11-04_a_2026-11-05.md](memoria_archivo/MEMORIA_08_2026-11-04_a_2026-11-05.md) | 11557-11671 | 15.838 | Kimi V35 R13A/R14A, V37 I-13 y backoff, V38 |

`COORDINACION_CODEX_2026-09-28.md` (1,76 MB, buzón append-only) sigue en la raíz porque
se escribió el 2026-11-10 (AGY Ola 32, Kimi V42, GLM-222); se moverá a esta carpeta cuando
lleve más de 24 h sin cambios. El registro cerrado de `AGENT_SYNC.md` está en
`AGENT_SYNC_ARCHIVO.md` (raíz).

Para reconstruir el original (bytes LF, sin BOM), concatenar en orden 01…08:
`cat .agents/memoria_archivo/MEMORIA_0*.md > MEMORIA_original.md`.
