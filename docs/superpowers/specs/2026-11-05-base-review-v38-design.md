# Revisión V38 desde la base — diseño de reconciliación y evolución causal

## Intención

Reconciliar el trabajo concurrente y convertir la visión multiactiva, continua, temporal y espectral en una secuencia verificable de correcciones. La duplicación del capital cada 72 horas es una hipótesis de investigación sujeta a evidencia OOS neta después de fees, funding y slippage; no una garantía, una instrucción de apalancamiento ni una autorización para operar.

## Estado comprobado al diseñar

- `HEAD=ba27c367749dae443a488006f6b79b03d7b2941e`, tree `98051a27e2f575a5ce99d70c5bcaf0c4846f8ac2`; `origin/main=6f211252fce278b0c47a0d95c0f7c3724ce1f814` y sólo `main` remota.
- V37 está cerrado: PR #171 y GLM-203 integrados; backoff con revisión monótona validado 234/234 pero sin CAS por Clippy baseline; config/lib con cuatro `NO_CHANGE`. No se repite ese trabajo ni se transfieren sus PASS a otro SHA.
- Ramas locales: `claude/linea-i-i14`, `claude/qsr-lc20r2` y `claude/qsr-lcf020` son ancestros de `main`; sus worktrees requieren guardas antes de retirarse. `agy/wave3-omega220` conserva un commit exclusivo `3ff61ae3`, por lo que se preserva para revisión/rescate.
- CI de `81fb45db`: Unit `42132984074` y Replay `42132984167` SUCCESS. No existe CI de V38 todavía.
- Recursos observados: 1,55 GiB RAM libres de 23,38 GiB y 159,66 GiB SSD libres. No se usa esa medición como admisión ni como permiso de borrado. El mayor working set observado es Memory Compression; no hay proceso de mercado que detener.
- Producción NO-GO; ROI/PnL/WR/Sharpe/EV=`N/D`. No hay Demo, Mainnet, OOS ni attestation nuevos.

## Objetivos

1. Publicar un plan común recuperable por cualquier agente, con propietarios, interfaces, fases y evidencia exigida.
2. Clasificar ramas y worktrees por valor, ancestro, PR, commits exclusivos, suciedad y propiedad; retirar sólo lo que cumpla todas las guardas.
3. Seleccionar un único defecto libre con llamador vivo, hipótesis causal y RED comportamental; corregirlo con el cambio mínimo y revisión independiente.
4. Mantener una auditoría priorizada de arquitectura, matemática, estadística, física, algoritmos y recursos sin convertir nombres de teorías en evidencia.
5. Dejar handoff máquina/legible con lo verificado, lo diferido y los bloqueos, sin inventar resultados económicos.

## No objetivos y antiobjetivos

- No rehacer PR #171, GLM-203, backoff V37, config/lib V37 ni el grafo de 19.105 nodos.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy ni WIP de otros agentes sin handoff explícito.
- No reescritura masiva, poda ciega de dependencias, teoría por prestigio, métricas inventadas ni simplificación de modelos para fingir progreso.
- No borrar ramas, worktrees, overlays, datos, modelos o logs por antigüedad/tamaño ni detener procesos ajenos.
- No ejecutar Demo/Mainnet/Producción, leer credenciales, hacer push o attestation sin autorización y puertas correspondientes.
- No declarar autoadaptación, ausencia de ruina, latencia de nanosegundos ni rentabilidad a partir de compilación o pruebas aisladas.

## Enfoques considerados

1. **Refactor masivo/teoría primero:** rechazado. Mezcla sujetos, impide atribución causal y eleva riesgo de Goodhart, leakage y sobreajuste.
2. **Verificación primero (elegido):** reconciliar Git/propiedad y aplicar un átomo con RED→GREEN por vez. Conserva rescate, revisión y evidencia por sujeto.
3. **Sólo censo/documentación:** útil para abrir la fase, insuficiente si existe un defecto demostrado. Se cerrará con comportamiento cuando haya evidencia y con `NO_CHANGE` cuando no la haya.

El operador pidió ejecución autónoma y el modo automático está activo; por ello se adopta este enfoque sin nuevas preguntas. Las decisiones quedan registradas como elecciones del ejecutor, no como preferencias atribuidas al operador.

## Secuencia y criterios de salida

### Fase 0 — Corte de verdad y sincronización

1. Leer heartbeat, memoria, handoffs, refs, CI y WIP agregado. Registrar V38 y su índice privado.
2. Auditar cada rama/worktree por HEAD, commits exclusivos, PR, ancestro y suciedad. `agy/wave3-omega220` queda preservado por su commit exclusivo.
3. Materializar el plan en `.agent_tx/staging/KIMI-QUANT-SR-V38/plan-v38/subject`, sin tocar producción compartida.
4. Revisar diseño con builder independiente y resolver cualquier Critical/Important antes del plan.

### Fase 1 — Higiene condicional de referencias

1. Para ramas ya ancestro de `main`, exigir PR/ancestría, cero commits exclusivos, worktree limpio y ausencia de archivos necesarios o reservas.
2. Retirar worktree sin `--force`, rama local y referencias remotas sólo cuando todas las guardas estén probadas. Si falta una, conservar y registrar causa.
3. Revisar el commit exclusivo de AGY para decidir rescate posterior; no integrarlo por antigüedad ni borrarlo por ser único.

### Fase 2 — Átomo causal libre

1. Selección basada en llamador vivo y efecto real, no en literal o teoría llamativa. Escribir RED antes de producción y observar el fallo esperado.
2. Implementación mínima, sin mocks/placeholders ni formato colateral. Ejecutar contratos del área, suite completa del crate y `cargo check --workspace --all-targets --locked`.
3. Clippy comparado por mensaje+archivo contra baseline; ningún diagnóstico nuevo. Revisión independiente; Critical/Important exige nuevo RED→GREEN.
4. No CAS de código con baseline roja. El parche y sus hashes quedan como candidato recuperable.

### Fase 3 — Mapa priorizado del sistema

1. Mantener censo por arquitectura/ignición, ingesta, decisiones, ejecución, evolución, persistencia y observabilidad, con archivo:línea y llamador vivo.
2. Para cada teoría o motor: invariante analítico, causalidad temporal, complejidad/costo, consumidor, baseline, ablación y criterio OOS. Ninguna integración por nombre o complejidad aparente.
3. Revisar valores fijos, fallbacks, vetos y límites por evidencia. Ausencia = `N/D`, nunca un literal sustituto.
4. RAM/SSD/CPU: buffers acotados por necesidad demostrada, zero-copy cuando reduzca costo medido, persistencia verificada y target único. No purgas que sacrifiquen rescate.

### Fase 4 — Demo/OOS/Producción (bloqueada)

1. Prerregistrar dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión antes de medir.
2. Demo requiere `DEMO_MAINNET_AUTHORIZATION.json` con SHA/config/periodo/entorno/aprobador humano. Producción exige autorización separada y OOS favorable.
3. Reportar ROI/PnL/WR antes y después de fees, Sharpe, EV/trade, periodo y dataset, o `N/D`. La hipótesis +100 %/72 h se clasifica compatible, refutada o insuficiente, nunca forzada con leverage.
4. B1/A4-T2 y V2 permanecen con sus dueños; cualquier ignición posterior exige handoff y pruebas del flujo real, no una reescritura por otro agente.

## Decisiones de arquitectura y coordinación

- Overlays propios, índice `.git/index_KIMI-QUANT-SR-V38`, nightly `2026-06-30`, `CARGO_BUILD_JOBS=16` y único `CARGO_TARGET_DIR` canónico. No targets alternativos ni semáforos.
- Cada recibo liga sujeto, fuentes/hashes, comandos, perfil y resultado. `check` no equivale a test, CI ni attestation.
- `AGENT_SYNC.md`, `COORDINACION_CODEX_2026-09-28.md`, `.agents/MEMORIA.md` y `.agent_tx/diagnostics/base-review-v38-status.json` son los canales de sincronización.
- El CAS documental puede hacerse con expected SHA y sin push. El CAS de código exige pruebas, revisión y baseline permitida. Ningún cambio se promueve por antigüedad o presión de cuota.

## Átomo seleccionado: reentrenamiento repetido de una revisión fallida

**Evidencia y llamador vivo.** `should_retrain_forest_now` usa `pending_new_obs >= 10` como condición de nivel. Tras `RetrainFailed`, el loop conserva ese contador y vuelve a intentar el entrenamiento cada 500 ms con el mismo `observation_revision`, aunque no exista deuda de escritura (`forest_persist_retry=None`). El punto de entrada vivo es `src/bin/god_engine.rs:2946`, que lanza `run_online_learning_loop`. Clonar y validar ambas secuencias retrasa además el muestreo de retornos y el watchdog.

**Cambio mínimo.** Añadir `retrain_failed_revision: Option<u64>` al daemon, inicializado a `None`. Al fallar entrenamiento se guarda la revisión actual; al tener éxito se limpia. La pierna pendiente de `should_retrain_forest_now` sólo se arma si `pending_new_obs >= 10` y la revisión almacenada difiere de la actual. La pierna periódica y la inhibición por deuda de persistencia permanecen iguales. No se borra `pending_new_obs`, no se crea reintento de persistencia y una nueva observación vuelve a armar el disparador.

**RED exigido.** Con 50 observaciones de una sola clase, `should_retrain_forest_now(10,50)` es inicialmente verdadero; `retrain_forest_and_persist` falla por variabilidad de clases; la misma revisión no debe rearmar el disparador. Una observación aceptada después incrementa la revisión y `should_retrain_forest_now(11,51)` vuelve a ser verdadero. El test se añade a `online_persist_backoff_contract.rs`, continúa el overlay V37 sin tocar `online_random_forest.rs` y no requiere extraer el loop infinito.

**Límites.** El Minor de cargar el snapshot recuperado es cobertura adicional, no un defecto separado. La corrección usa el candidato V37 actual y no cambia B1/A4-T2, QRA/Navier/`adversarial_gate`, V2 ni WIP ajeno. Resultado esperado: 6/6 contratos backoff y 235/235 en `evolution-engine`; workspace y Clippy sin diagnósticos nuevos. CAS de código sigue bloqueado por baseline.

## Supuestos adoptados por ejecución automática

- El mandato amplio autoriza verificación, documentación, pruebas locales y retirada de referencias sólo con todas las guardas; no autoriza Demo/Mainnet ni borrado de trabajo no resuelto.
- La limpieza remota ya está completa al abrir V38; las ramas locales se tratan individualmente.
- El grafo existente se consulta incrementalmente y no se reconstruye sin cambios que lo justifiquen.
- La fecha de corte es 2026-11-05. Cambios posteriores de refs exigen recomponer la fase afectada, no fingir continuidad.

## Registro de ejecución y revisión independiente

- `qsr-lcf020` fue la única referencia retirada: PR #163 MERGED, worktree limpio y blob ignorado idéntico al contrato preservado. `linea-i-i14`, `qsr-lc20r2`, `agy/wave3-omega220`, ambos worktrees detached y `tg-glm204` se preservan por WIP, valor o propiedad no resueltos.
- El átomo quedó en `.agent_tx/staging/KIMI-QUANT-SR-V38/retrain-revision-v1/subject` y `retrain-revision.patch`. RED 5/6 por la aserción objetivo → GREEN 6/6; `evolution-engine` 235/235 en 27 suites; workspace exit 0 en 3m36s; Clippy baseline 40/current 40, `ADDED=[]`, `REMOVED=[]`.
- Revisión independiente agent-45: **Ready to merge? Yes**, sin Critical/Important. Confirmó borrado correcto del marcador tras éxito directo y recuperación, preservación de persistencia/contadores y periodicidad intencional.
- Minor diferido: el test no ejecuta un éxito posterior para fijar `retrain_failed_revision=None` ni una iteración múltiplo de 20 para documentar la excepción periódica. Es cobertura adicional, no defecto demostrado; no se amplía el átomo.
- No hay CAS de código por baseline roja. No push, attestation, Demo, Mainnet ni OOS. ROI/PnL/WR/Sharpe/EV=`N/D`; Producción NO-GO.
- Handoff: `.agent_tx/diagnostics/base-review-v38-status.json` y `.superpowers/sdd/2026-11-05-base-review-v38/progress.md`. El siguiente agente debe revalidar fuentes/hashes y no repetir el átomo ni transferir PASS a otro SHA.
