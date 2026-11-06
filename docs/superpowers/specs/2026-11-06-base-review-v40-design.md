# Revisión V40 desde la base — diseño

## Intención

Continuar el universo multiactivo, continuo, temporal y espectral mediante defectos demostrables y mejoras conectadas al flujo vivo. La duplicación del capital cada 72 horas es una hipótesis que debe falsarse fuera de muestra después de fees, funding y slippage; no es una garantía, una certificación existente ni una instrucción para aumentar apalancamiento.

## Estado comprobado

- Base local inicial `f7d69533` y remota `0ce05fda`: 95 commits locales por delante y dos documentales por detrás. El remoto sólo contiene `main`.
- GLM-204 es la última punta de código remoto con Unit `42149507213` y Replay `42149507195` SUCCESS. GLM-205/206 añaden documentación A3, no una nueva certificación de código.
- La fusión candidata `96d27840e5cc1e3e2c45b44350d6d499cf0bc389` tiene padres `f7d69533` y `0ce05fda`; preserva V39 y GLM-205/206 y registra el heartbeat V40. Su validación se ejecuta antes de cualquier CAS.
- El workspace contiene 696 rutas modificadas y no se usa como fuente canónica. Los sujetos se construyen desde objetos Git inmutables.
- `linea-i-i14` y `qsr-lc20r2` no tienen commits exclusivos, pero sus worktrees contienen WIP. AGY tiene un commit exclusivo de 22 archivos y WIP adicional. Los dos detached también permanecen. No hay retirada segura en este corte.
- El grafo histórico de 19.105 nodos está desactualizado respecto al código y sólo sirve para generar hipótesis; los llamadores se vuelven a comprobar en las fuentes.
- Producción NO-GO; ROI/PnL/WR/Sharpe/EV=`N/D`. A3 pertenece a otro flujo y no se ejecuta desde V40.

## Objetivos

1. Reconciliar refs y trabajo concurrente sin borrar valor ni mezclar WIP.
2. Publicar un plan por fases que vaya de arquitectura y evidencia a código, no de nombres de teorías a parches cosméticos.
3. Seleccionar un único defecto libre con llamador vivo, invariante y efecto económico/lógico demostrable; escribir RED antes de producción.
4. Validar el cambio sobre sujeto inmutable con regresión, workspace, comparación de Clippy y revisión independiente.
5. Dejar un handoff recuperable con bloqueos, Minor, recursos y próximos pasos explícitos.

## No objetivos y antiobjetivos

- No rehacer V37–V39, PR #171, backoff, GLM-203/204 ni el censo config/lib anterior.
- No integrar por nombre las teorías del mandato, reescribir cientos de archivos ni simplificar matemática existente para salir del paso.
- No editar B1/A4-T2, QRA/Navier/`adversarial_gate`, V2, legacy ni trabajo reservado.
- No borrar ramas/worktrees por antigüedad, heartbeat o tamaño. Retirar sólo con valor/ancestría, propiedad, rescate y archivos preservados demostrados.
- No ejecutar Demo/Mainnet/Producción, leer credenciales, hacer push o attestation, ni afirmar rentabilidad.
- No usar reset/stash/checkout forzado, `git add -A`, force-push, `--no-verify`, targets alternativos o semáforos.

## Enfoques considerados

### A — Átomo causal incremental (elegido por este agente)

Reconciliar primero; revisar arquitectura y consumidores; elegir un defecto libre; demostrar RED→GREEN y revisión independiente; integrar sólo si las puertas lo permiten. Reduce blast radius y produce evidencia reutilizable. El coste es menor cobertura por ciclo.

### B — Rescate amplio de AGY/Wave 3

Los 22 archivos aportan motores y contratos nuevos, pero son trabajo activo con APIs e invariantes que necesitan revisión específica. No se descartan ni se integran ciegamente. Su coste es validación cross-crate, posible solapamiento y ausencia de atribución OOS.

### C — Reescritura masiva

Se rechaza por incompatible con el WIP extenso, la falta de una baseline económica y la regla anti-cosmética. Una lista de teorías no sustituye causalidad, medición ni llamadores vivos.

La elección es una decisión de este agente bajo ejecución automática, no una preferencia atribuida al operador.

## Fases

### Fase 0 — Verdad, propiedad y coordinación

Refs, CI, agentes, ramas, worktrees, índices privados, heartbeat y reconciliación en memoria. Cada resolución exige diff contra padres y validación del árbol. El CAS documental no implica push ni CI.

### Fase 1 — Mapa arquitectónico y económico

Inventario mecánico por archivos y consumidores: ignición, ingesta, normalización, selección de símbolos, decisión, fees/fricción, ejecución/OMS, evolución online, persistencia, telemetría y visualización. Priorizar por llamador vivo, impacto y libertad de edición. El inventario no se presentará como auditoría semántica completa. **Regla de parada:** cerrar esta fase en cuanto exista una opción priorizada que cumpla llamador vivo, propiedad libre, invariante, no duplicidad y RED disponible; el resto de candidatos queda diferido para evitar parálisis por análisis.

### Fase 2 — Un átomo de corrección

Un defecto concreto, una hipótesis causal, un test real que falle por ese mecanismo y el cambio mínimo. Prohibido crear mocks, fallbacks a cero o simplificaciones con nombres avanzados. Tests de límites y NaN/Inf cuando correspondan.

### Fase 3 — Certificación e integración

Contrato, suites vecinas y crate completa, workspace `--all-targets`, formato acotado, Clippy comparado con baseline y revisor independiente. Critical/Important exige nuevo RED→GREEN y un veredicto independiente final sobre el sujeto inmutable definitivo, con hashes y diff exactos; los tests nuevos por sí solos no cierran esos hallazgos. Minor queda con coste explícito. CAS con índice privado y expected SHA; código bloqueado mientras su baseline esté roja.

### Fase 4 — Recursos, Demo y OOS (bloqueados)

Recursos: sin topes de RAM/SSD/hilos ni semáforos/colas de admisión. Zero-wait speculative overlays, índices Git privados, `CARGO_BUILD_JOBS=16`, `rust-lld` y target único compartido; paralelismo sólo sobre superficies de escritura no coincidentes. Diagnóstico no destructivo de buffers/persistencia, sin purgas por tamaño ni limpieza del trabajo. OOS exige prerregistrar dataset/periodo, commit incumbente, fees/funding/slippage reales, walk-forward, multiplicidad y regla de decisión. Demo requiere `DEMO_MAINNET_AUTHORIZATION.json`; Producción exige autorización separada y OOS favorable. Reportar ROI/PnL/WR antes/después de fees, Sharpe, EV/trade y periodos exactos o `N/D`.

## Decisiones pendientes del constructor

El átomo concreto no se fija en este diseño para evitar elegir por intuición o duplicar trabajo. Se seleccionará después de comprobar propiedad, llamador y prueba disponible. AGY/Wave 3 requiere su propia revisión de invariantes y no es una cola de código gratuito.
