# Meta-analítica del impacto por cambio (RB-A1a)

Rama `origin/main` (primer padre) en `3a8a6bf706b9`: 1076 commits, del 2025-11-26 al 2026-10-19. Libro completo en `libro_impacto.tsv`; regenerar con `python scripts/meta_analitica.py`.

## 1. Cambio por agente

| Agente | Commits | Prod + | Prod − | Tests + | Tests − | Δ funciones de test | Docs + |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trader Gemini Bot | 842 | 202737 | 65750 | 43922 | 1110 | +2722 | 561479 |
| AGY | 117 | 13801 | 4393 | 11530 | 491 | +297 | 78170 |
| Claude Línea C | 37 | 8065 | 2492 | 5923 | 560 | +205 | 8755 |
| GLM | 16 | 3417 | 1197 | 2929 | 49 | +146 | 58708 |
| Claude Línea H | 13 | 249 | 39 | 818 | 15 | +19 | 357 |
| Fable | 12 | 98 | 62 | 174 | 1 | +5 | 622 |
| Jhonatan Laiton | 11 | 1257 | 988 | 1868 | 56 | +108 | 7402 |
| Claude Línea G | 10 | 367 | 41 | 733 | 0 | +14 | 509 |
| Codex RB | 9 | 0 | 0 | 0 | 0 | +0 | 434 |
| Sol | 7 | 136 | 4 | 160 | 0 | +5 | 477 |
| Qoder | 2 | 1158 | 57 | 662 | 9 | +26 | 881 |

## 2. Cómputo

Clausura de dependencias de `god-engine-core` en `Cargo.lock`: 434 → 454 paquetes (host `trader-gemini-v5`: 469 → 499; lock: 470 → 500).

| Δ clausura núcleo | Commit | Agente | Asunto |
|---:|---|---|---|
| +37 | `f6f0cb31b9c4` | Trader Gemini Bot | checkpoint(f0): estabilización — saneamiento git, seguridad y arreglos críticos |
| -13 | `e3282c47af21` | Claude Línea C | Merge PR #81: RET-2, 5, 15 y 28a, dependencias sin uso y código huérfano del censo C-F7b |
| -4 | `d18b373547e3` | Claude Línea C | sol(f9-6): clúster phase-runner RETIRADO (censo C-F7b F353-F361 + F360) |

Builds del semáforo (log local): 103 cerrados, 5 sin liberar (matados o en curso).

| Agente | Builds | Compilación (h) | Espera (h) | Fallidos |
|---|---:|---:|---:|---:|
| AGY | 44 | 2.86 | 0.49 | 14 |
| Claude-LineaC | 1 | 0.00 | 0.00 | 0 |
| Claude-LineaG | 23 | 2.81 | 0.32 | 4 |
| Codex RB | 3 | 2.63 | 1.04 | 0 |
| Fable | 28 | 2.05 | 7.60 | 5 |
| Sol | 2 | 0.18 | 0.95 | 0 |
| claude-lineaH | 2 | 0.17 | 0.45 | 0 |

## 3. Salud del código

Δ neto de funciones de test en el rango: +3547.
Trinquete de pánicos RB-P1: aún sin línea base en este rango.

## 4. Financiero

| Commit | Fecha | Agente | T-1 (genes sensibles / 144) |
|---|---|---|---:|
| `949d29c1b823` | 2026-10-04 | Claude Línea C | 16 |
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
