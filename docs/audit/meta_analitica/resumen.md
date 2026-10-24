# Meta-analítica del impacto por cambio (RB-A1a)

Rama `origin/main` (primer padre) en `0440796262e2`: 1088 commits, del 2025-11-26 al 2026-10-19. Libro completo en `libro_impacto.tsv`; regenerar con `python scripts/meta_analitica.py`.

## 1. Cambio por agente

| Agente | Commits | Prod + | Prod − | Tests + | Tests − | Δ funciones de test | Docs + |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trader Gemini Bot | 848 | 203596 | 66390 | 44762 | 1114 | +2743 | 561901 |
| AGY | 117 | 13801 | 4393 | 11530 | 491 | +297 | 78170 |
| Claude Línea C | 37 | 8065 | 2492 | 5923 | 560 | +205 | 8755 |
| GLM | 16 | 3417 | 1197 | 2929 | 49 | +146 | 58708 |
| Codex RB | 14 | 405 | 669 | 1144 | 0 | +13 | 1905 |
| Claude Línea H | 13 | 249 | 39 | 818 | 15 | +19 | 357 |
| Fable | 12 | 98 | 62 | 174 | 1 | +5 | 622 |
| Jhonatan Laiton | 12 | 1427 | 1008 | 1990 | 56 | +112 | 7427 |
| Claude Línea G | 10 | 367 | 41 | 733 | 0 | +14 | 509 |
| Sol | 7 | 136 | 4 | 160 | 0 | +5 | 477 |
| Qoder | 2 | 1158 | 57 | 662 | 9 | +26 | 881 |

## 2. Cómputo

Clausura de dependencias de `god-engine-core` en `Cargo.lock`: 434 → 357 paquetes (host `trader-gemini-v5`: 469 → 486; lock: 470 → 487).

| Δ clausura núcleo | Commit | Agente | Asunto |
|---:|---|---|---|
| -97 | `566a9c8c81d6` | Codex RB | RB-R1: la pila polars (y yahoo_finance_api) fuera del grafo de god-engine-core, backtest-e |
| +37 | `f6f0cb31b9c4` | Trader Gemini Bot | checkpoint(f0): estabilización — saneamiento git, seguridad y arreglos críticos |
| -13 | `e3282c47af21` | Claude Línea C | Merge PR #81: RET-2, 5, 15 y 28a, dependencias sin uso y código huérfano del censo C-F7b |
| -4 | `d18b373547e3` | Claude Línea C | sol(f9-6): clúster phase-runner RETIRADO (censo C-F7b F353-F361 + F360) |

Builds del semáforo (log local): 109 cerrados, 4 sin liberar (matados o en curso).

| Agente | Builds | Compilación (h) | Espera (h) | Fallidos |
|---|---:|---:|---:|---:|
| AGY | 44 | 2.86 | 0.49 | 14 |
| Claude-LineaC | 1 | 0.00 | 0.00 | 0 |
| Claude-LineaG | 23 | 2.81 | 0.32 | 4 |
| Codex RB | 9 | 4.08 | 1.38 | 0 |
| Fable | 28 | 2.05 | 7.60 | 5 |
| Sol | 2 | 0.18 | 0.95 | 0 |
| claude-lineaH | 2 | 0.17 | 0.45 | 0 |

## 3. Salud del código

Δ neto de funciones de test en el rango: +3585.
Sitios de pánico de producción (trinquete RB-P1): 226 → 226 desde `d8a56b15c0e5`.

## 4. Financiero

| Commit | Fecha | Agente | T-1 (genes sensibles / 144) |
|---|---|---|---:|
| `785a389f422e` | 2026-10-08 | AGY | 16 |
| `fd5959f0d890` | 2026-10-09 | GLM | 16 |
| `ab240abd3b15` | 2026-10-09 | Trader Gemini Bot | 16 |
| `f5cadac7023d` | 2026-10-09 | Trader Gemini Bot | 16 |
| `0043f8893a58` | 2026-10-10 | Qoder | 16 |
| `aa716edf25a6` | 2026-10-10 | Claude Línea C | 16 |
| `8632ca3de485` | 2026-10-11 | Claude Línea C | 16 |
| `1d1f8bdd3564` | 2026-10-11 | Claude Línea C | 16 |
| `cac9660b32d2` | 2026-10-17 | AGY | 18 |
| `44c9062a4813` | 2026-10-18 | Fable | 18 |
| `a2e28c80f60f` | 2026-10-18 | Claude Línea C | 17 |
| `24dfc19c884e` | 2026-10-18 | Fable | 18 |
| `4db48d6a6d89` | 2026-10-18 | Fable | 18 |
| `ab767f94a08e` | 2026-10-18 | Claude Línea C | 17 |
| `dac445628c7a` | 2026-10-19 | Codex RB | 16 |

## 5. Mayores cambios de producción

| Commit | Agente | Prod + | Prod − | Átomos | Asunto |
|---|---|---:|---:|---|---|
| `f6f0cb31b9c4` | Trader Gemini Bot | 25188 | 4233 |  | checkpoint(f0): estabilización — saneamiento git, seguridad y arreglos |
| `f0ec9c29c208` | Trader Gemini Bot | 22025 | 3908 |  | chore(freeze): stable state after comprehensive forensic audit and eng |
| `dc500946013e` | Trader Gemini Bot | 12688 | 3349 |  | feat: implement GodEngineCore, live trading, data pipeline, backtest e |
| `f2d9e4efcf60` | Trader Gemini Bot | 9064 | 4639 |  | f0.6/f0.7/f0.8: docs honestas, quality gate, censo maestro + fmt basel |
| `ac1366331833` | Trader Gemini Bot | 9161 | 3828 |  | feat(FMT): Ondas XXX-XXXIX — evidencia income/ejecución (FMT-282/284)  |
| `40b362014a95` | Trader Gemini Bot | 9522 | 3418 |  | f4.7: ensamble bayesiano online + online_daemon CABLEADO (eran fantasm |
| `6fdccd648fcf` | Trader Gemini Bot | 10303 | 2440 | PR #5 | merge: PR #5 — integra claude/elegant-euler-mmtht4 (D-742…D-758) con m |
| `8506be986057` | Trader Gemini Bot | 7908 | 2246 |  | R4: integra main 3dc y adapta contratos al índice triangular |
| `87a6bb302674` | Trader Gemini Bot | 3952 | 3641 |  | feat: implement parsers module and generate new backtest insight repor |
| `4057edee51c4` | Trader Gemini Bot | 96 | 3649 |  | feat: migrate core services to Rust, remove legacy Python configuratio |

## 6. Atribución por área (componentes)

Sólo producción y tests transversales a cada componente (`area()`: crate, `host`, `workspace`, `datos` o la clase transversal). Completo en `area_impacto.tsv`.

| Área | Commits | Prod + | Prod − | Tests + | Tests − | Ratio T/P | Share prod + | Primera | Última |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| host | 204 | 49615 | 26256 | 0 | 0 | 0.00 | 21.3% | 2026-06-16 | 2026-10-19 |
| god-engine-core | 323 | 39312 | 14339 | 20066 | 565 | 0.51 | 16.9% | 2026-07-01 | 2026-10-19 |
| quantum-arena | 170 | 20201 | 3519 | 5288 | 31 | 0.26 | 8.7% | 2026-06-29 | 2026-10-19 |
| risk-engine | 178 | 17659 | 5514 | 11080 | 599 | 0.63 | 7.6% | 2026-06-29 | 2026-10-18 |
| execution-engine | 95 | 15922 | 3163 | 3984 | 86 | 0.25 | 6.8% | 2026-06-29 | 2026-10-19 |
| signal-engine | 132 | 14099 | 4145 | 6822 | 40 | 0.48 | 6.1% | 2026-06-29 | 2026-10-19 |
| evolution-engine | 79 | 11491 | 4301 | 2045 | 15 | 0.18 | 4.9% | 2026-07-01 | 2026-10-18 |
| feature-engine | 53 | 8187 | 1136 | 4591 | 121 | 0.56 | 3.5% | 2026-06-29 | 2026-10-19 |
| data-pipeline | 33 | 8138 | 1775 | 388 | 3 | 0.05 | 3.5% | 2026-06-29 | 2026-10-19 |
| backtest-engine | 108 | 8090 | 2321 | 6190 | 487 | 0.77 | 3.5% | 2026-07-01 | 2026-10-19 |
| metacortex-engine | 54 | 8023 | 1627 | 4396 | 69 | 0.55 | 3.4% | 2026-08-16 | 2026-10-18 |
| strategy-core | 49 | 7654 | 2230 | 3086 | 194 | 0.40 | 3.3% | 2026-07-01 | 2026-10-18 |
| telemetry-server | 19 | 5932 | 1000 | 0 | 0 | 0.00 | 2.5% | 2026-07-01 | 2026-10-18 |
| storage-engine | 19 | 3926 | 784 | 397 | 14 | 0.10 | 1.7% | 2026-08-16 | 2026-10-07 |
| dark-alpha-engine | 30 | 3763 | 748 | 858 | 0 | 0.23 | 1.6% | 2026-07-01 | 2026-10-17 |

Concentración: el top 3 de áreas concentra el 46.9% de la producción añadida.
Áreas con producción añadida y ratio de tests < 0.05 (candidatas a deuda de pruebas): host, data-pipeline, telemetry-server, state-engine, os-guardian, graph-architecture, telemetry-engine, flight-recorder, phase-runner, graph-4d.

