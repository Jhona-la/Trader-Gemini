# Revisión V39 desde la base — diseño de reconciliación y corrección causal

## Intención

Continuar el sistema multiactivo, continuo, temporal y espectral con correcciones demostrables, sin rehacer V38 ni añadir teorías por prestigio. La duplicación cada 72 horas es una hipótesis OOS neta después de fees, funding y slippage, nunca una garantía ni una orden de apalancamiento.

## Estado comprobado

- `HEAD=62df45d9bf53fb88438cc03dc293e3c0d3f3202b`, tree `e4e2696310f1844590f18c1d7fa6f5f843b2950e`; `origin/main=7afc9ed103be46f55fcd6e1387e403b0da85b538` es ancestro.
- GLM-204 fue reconciliado por unión documental: Unit `42149507213` y Replay `42149507195` SUCCESS; contrato 5/5 y workspace exit 0 en 3m39s.
- V38 quedó con revisión independiente `Ready to merge? Yes`, pero su código no tiene CAS por Clippy baseline. No se repiten backoff, PR #171 ni config/lib V37.
- Sólo `main` remota. `linea-i-i14`, `qsr-lc20r2`, `agy/wave3-omega220` y worktrees detached permanecen por WIP, commits exclusivos o propiedad no resuelta; no se borra nada por antigüedad.
- Producción NO-GO; ROI/PnL/WR/Sharpe/EV=`N/D`. No hay Demo, Mainnet, OOS, push ni attestation nuevos.

## Objetivos

1. Mantener plan y estado recuperables por cualquier agente, con sujetos, propietarios y evidencia por SHA.
2. Seleccionar un único defecto libre con llamador vivo y RED comportamental; aplicar el cambio mínimo y revisión independiente.
3. Preservar trabajo concurrente y retirar referencias sólo con todas las guardas probadas.
4. Revisar valores, modelos, vetos y recursos por consumidor y efecto real, no por presencia de literales o nombres de teorías.
5. Dejar bloqueos explícitos: baseline de Clippy, OOS, Demo y producción.

## No objetivos y antiobjetivos

- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy ni WIP ajeno sin handoff.
- No reescritura masiva, poda ciega, cambio de fixture/comparador/ABI 144/mínimo `0.110` ni teoría sin consumidor.
- No borrar ramas, worktrees, overlays, datos, modelos o logs por antigüedad/tamaño ni detener procesos ajenos.
- No ejecutar Demo/Mainnet/Producción, leer credenciales, hacer push ni certificar rentabilidad o ausencia de ruina.
- No presentar inventario mecánico o compilación como auditoría completa, CI o prueba E2E.

## Enfoques considerados

1. **Refactor masivo/teoría primero:** rechazado por mezcla de sujetos, riesgo de leakage y ausencia de atribución.
2. **Verificación primero (elegido):** reconciliar base y aplicar un átomo con RED→GREEN, regresión y revisión independiente.
3. **Sólo documentación:** insuficiente si hay defecto demostrado; `NO_ATOM` es aceptable si no hay evidencia, nunca un cambio fabricado.

El modo automático y el mandato de ejecución se interpretan como permiso para trabajar sin nuevas preguntas, con estas decisiones registradas como elecciones del ejecutor.

## Secuencia y salidas

### Fase 0 — Corte de verdad

1. Reconciliar refs, CI, agentes, WIP y handoffs. Registrar heartbeat e índice privado.
2. Unir upstream verde con V38, revisar diff contra cada padre y ejecutar contrato y workspace.
3. Materializar diseño/plan en `.agent_tx/staging/KIMI-QUANT-SR-V39/plan-v39/subject`; builder independiente sin Critical/Important.

### Fase 1 — Átomo causal

1. Revisión independiente de alcance fija archivo:línea, llamador, causa y RED, o `NO_ATOM`.
2. Escribir RED antes de producción y observar el fallo esperado. No mocks, placeholders ni formato colateral.
3. Implementación mínima; contratos del área, crate completa, `cargo check --workspace --all-targets --locked` y Clippy comparado con baseline.
4. Revisión independiente; Critical/Important exige nueva RED→GREEN. No CAS de código con baseline roja.

### Fase 2 — Mapa priorizado

1. Mantener censo por ignición, ingesta, decisión, ejecución, evolución, persistencia y observabilidad con llamadores vivos.
2. Cada teoría exige invariante analítico, causalidad temporal, coste, consumidor, baseline, ablación y criterio OOS. No integración por nombre.
3. Valores fijos/fallbacks/vetos: ausencia = `N/D`, no otro literal. RAM/SSD/CPU: acotación demostrada, zero-copy cuando sea útil y target único, sin purgas que destruyan rescate.

### Fase 3 — Demo/OOS/Producción (bloqueada)

1. Prerregistrar dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión antes de medir.
2. Demo exige `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige autorización separada y OOS favorable.
3. Reportar ROI/PnL/WR antes/después de fees, Sharpe, EV/trade, periodo y dataset o `N/D`. La meta +100 %/72 h se clasifica compatible, refutada o insuficiente.
4. B1/A4-T2 y V2 permanecen con sus dueños; cualquier ignición posterior exige handoff y pruebas reales.

## Coordinación

- Overlays propios, índice `.git/index_KIMI-QUANT-SR-V39`, nightly `2026-06-30`, `CARGO_BUILD_JOBS=16` y único `CARGO_TARGET_DIR`.
- `AGENT_SYNC.md`, `COORDINACION_CODEX_2026-09-28.md`, `.agents/MEMORIA.md` y `.agent_tx/diagnostics/base-review-v39-status.json` son los canales de verdad.
- Cada recibo liga fuentes/hashes, comandos, perfil y resultado. `check` no equivale a test, CI ni attestation.
- CAS documental permitido con expected SHA y sin push. CAS de código sólo con pruebas, revisión y baseline permitida.

## Átomo seleccionado: identidad de moneda en fees del examen

**Defecto y efecto.** La construcción de series del examen descarta los IDs antes de filtrar, y el orden por longitud vuelve a asignar IDs densos. Con una moneda plana/corta y otra elegible, la serie elegible puede heredar las comisiones de la moneda excluida o de otra moneda ordenada. `GodEngineCore::sync_reality_fees_for_coin` resuelve fees por ese ID, por lo que el error altera fitness, DSR y potencialmente la promoción del genoma.

**Cambio mínimo.** Conservar `(source_coin_id, returns)` en filtrado y ordenación. Construir `roundtrip_fees` usando el ID original y pasar esos IDs directamente a `wf_evaluate_real`, sin reasignar posiciones densas. Los arrays de precios/replay permanecen alineados por índice de serie y no se muta ningún registro global.

**RED.** Con dos monedas y tasas `0.0001` y `0.01`, la serie plana de coin 0 se excluye y la elegible de coin 1 debe cobrar coin 1. En orden por longitud, coin 1 va primero pero conserva IDs `[1,0]`; la implementación actual cobra coin 0 y asigna `[0,1]`. El test se ejecuta primero por aserción contra extracciones comportamentales existentes, no por falta de API. Tras el cambio se espera la tasa de coin 1 y los IDs originales.

**Alcance.** Sólo `online_daemon.rs` y su contrato. No se editan `god-engine-core`, `quantum-arena`, `online_random_forest.rs`, B1/A4-T2, QRA/Navier/`adversarial_gate`, V2 ni WIP ajeno. El árbol base es el candidato V38, copiado a `.agent_tx/staging/KIMI-QUANT-SR-V39/fee-identity-v1/subject`. Resultado esperado: contrato nuevo 1/1 y 236/236 en `evolution-engine`; workspace y Clippy sin diagnósticos nuevos. CAS de código sigue bloqueado por baseline.

## Supuestos

- El mandato autoriza verificación, documentación y pruebas locales; no autoriza Demo/Mainnet ni borrado de trabajo no resuelto.
- El grafo existente se consulta sin reconstrucción redundante.
- La fecha de corte es 2026-11-05. Cambios posteriores de refs exigen recomponer la fase afectada.
