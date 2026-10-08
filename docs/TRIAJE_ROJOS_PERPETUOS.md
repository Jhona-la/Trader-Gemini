# TRIAJE DE ROJOS PERPETUOS — del mapa pasivo a la cola activa

> XCVIII (2026-10-06). El F8 catalogó ~79 tests que certifican defectos
> abiertos como verde documentado. Este triaje los clasifica por
> ACCIONABILIDAD: qué se puede fix hoy, qué exige decisión, qué es
> comportamiento aceptado.

## Categoría A — FIXABLE AHORA (defecto claro, fix delimitado)

| archivo:línea | defecto | fix |
|---|---|---|
| `audit-engine/tests/auditor_open_diagnostics.rs:40` | NaN = alineación 1.0 | rechazar NaN en coherence_score |
| `quantum-arena/tests/genome_reader_diagnostics.rs:26-27` | auto-comparación siempre-verdadera | comparar contra caso de referencia |
| `evolution-engine/tests/fitness_evidence_contract.rs:94` | `2*full - full > 0.02` = `full > 0.02` tautología | aserción de no-aditividad real |
| `data-ingest/tests/dynamic_selector_contract.rs:43-44` | `variation > 0.0` tautología | probar variación real con datos |
| `god-engine-core/tests/stateful_open_diagnostics.rs:15` | prefijo `open_` en test ya CERRADO (#660) | renombrar a regression_ |
| `quantum-arena/tests/genome_gate_open_diagnostics.rs:1` | era OPEN, REPARADO (FMT-216) — prefijo miente | renombrar |

**Todos estos son repairs de HONESTIDAD del sistema de tests**: un test
tautológico o auto-comparativo es un test que miente al decir que
verifica algo. No cambian conducta — sólo dejan de certificar en falso.

## Categoría B — NECESITA DECISIÓN DE DISEÑO (cada uno = mini-ola)

| archivo:línea | defecto | decisión requerida |
|---|---|---|
| ~~`execution-engine/tests/shadow_open_diagnostics.rs:20`~~ | ~~kill-switch no bloquea nueva entrada~~ | **DRENADO (GLM 105)**: espejo CL-3 — latch permanente, 7 rutas de nuevo riesgo bloqueadas, salidas libres. Sin oráculo (stub sin cablear a dinero). |
| ~~`execution-engine/tests/registry_open_diagnostics.rs:8`~~ | ~~timeout local fabrica Expired del exchange~~ | **DRENADO (GLM 107)**: fail-closed — timeout local escribe `Unknown` (no un terminal del wire): la evidencia real tardía ya no es absorbida por `merge` y `await_resolution` sigue esperando → `resolve_via_rest`. Superficie auxiliar (0 call-sites vivos) ⇒ sin oráculo. Cancel-and-reconcile para acked enmudecidas = ola futura CON oráculo. |
| ~~`execution-engine/tests/execution_open_diagnostics.rs:44`~~ | ~~selector fabrica BTCUSDT de `[]`~~ | **DRENADO (GLM XCIX)**: fail-closed, universo vacío. |
| ~~`execution-engine/tests/execution_open_diagnostics.rs:51`~~ | ~~`assert_ne!(score, score)` — orden total inconsistente~~ | **DRENADO (GLM C)**: Ord total real, NaN menor. |
| ~~`storage-engine/tests/mmap_open_diagnostics.rs:6`~~ | ~~pérdida silenciosa de frames en wrap~~ | **DRENADO (GLM 105, con oráculo)**: stop-at-first-invalid — el cursor sólo avanza sobre frames validados; liveness por el clamp MAX_BATCH_READ. El frame commiteado-después se recupera (pérdida sistemática del dataset del Shadow Forest eliminada). |
| `signal-engine/tests/ensemble_characterization.rs:5 tests` | piso 0.7, clones inflan, baseline 0.5 fabrica lift | revisión del ensamble |
| ~~`risk-engine/tests/veto_open_diagnostics.rs:4 tests`~~ | ~~NaN-peak pasa, capital desconocido no fail-closed, cap=1 no honrado~~ | **DRENADO (GLM 106)**: 4 doctrinas — peak NaN/≤0 ≠ seguro (fail-closed), capital NaN = veto (fail-closed), caps de clúster y racha honrados EXACTOS (pisos `.max(2)` removidos). Superficies auxiliares (0 call-sites vivos) ⇒ sin oráculo. **GEMELOS VIVOS registrados como ola futura CON oráculo**: god-engine lib.rs:279 (peak NaN omite el veto inline de drawdown) y lib.rs:564+ (agregación viva de correlación). |
| `god-engine-core/tests/reality_physics_open_diagnostics.rs:3` | precio infinito, maker sin cola, latency_penalty ignorado | física del fill |

**B queda en 8 tests** (8 drenados por GLM: XCIX, C, 105×2, 106×4, 107×1). **Deuda viva registrada por los gemelos**:
lib.rs:279 y lib.rs:564+ requieren ola con oráculo si el consejo decide
endurecer también el camino vivo.

## Categoría C — COMPORTAMIENTO DOCUMENTADO (aceptado, no bug)

Los ~50 restantes: resilience/behavioral (parámetros fuera de dominio
aceptados, drops indistinguibles), strategy-core (fallback cruzado,
spread constructor), payload_open (FMT-175/219), accounting_open — el
test VERDE con prefijo open_ es la convención CORRECTA aquí: describe
límites conocidos y aceptados del diseño actual.

## Regla de drenaje

- **A** se fix en ciclos normales (sin oráculo: son tests, no conducta).
- **B** cada uno es una mini-ola con su decisión documentada + oráculo si
  toca conducta viva.
- **C** se deja: la convención es honesta mientras el nombre no mienta
  (los 2 renombrados de A cierran los únicos casos de nombre vencido).
