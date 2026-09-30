# CX — Causalidad del replay, calentamiento y disponibilidad de información

Fecha local: 2026-09-29. Autor: Codex. Base: main `968259dc` (PR #21).
Rama: `feat/quant-sr-codex-causalidad`. Estado: candidato con contratos
locales; integración condicionada a comprobaciones y revisión cruzada.

## Resumen y alcance verificable

La prioridad de esta pasada es la raíz de la evidencia: si el simulador
usa información futura, ninguna mejora del genoma, red neuronal o estimador
puede convertir su evaluación en evidencia fuera de muestra. Se conserva
la arquitectura espectral y se repara el orden de observación, sin añadir
modelos ni modificar límites de riesgo para producir más operaciones.

Se revisó el archivo completo `crates/backtest-engine/src/booktick_replay.rs`
y su conexión con `GodEngineCore::process_event`, la actualización temporal
de `StatefulEngine`, los consumidores de evolución/ventanas, los contratos
de paridad, el informe MX y ADR-0006. Los consumidores y el núcleo se
inspeccionaron de forma focalizada, no íntegramente. No se certifica una
auditoría archivo por archivo de todo el proyecto.

Cuatro reparaciones: precarga anticipada, warmup con aperturas, backfill
macro anterior al primer dato y aritmética del límite de warmup. Cinco
expedientes adicionales permanecen abiertos. El workflow CI nuevo se
registra aparte: existencia del YAML no equivale a ejecución aprobada.

## Grafo vivo: de la raíz de datos al resultado

```text
Tape de un activo + genoma + configuración + estado global/modelos
  └─ admisión de la fila i
      ├─ macro con corte día−1 [sin fecha de publicación/vintage: CX-05]
      ├─ features/core desde prefijo observado [precarga eliminada: CX-01]
      └─ puerta de entrada i >= warmup [CX-02]
          ├─ warmup: features sí; nuevas posiciones no
          └─ evaluación: reglas del núcleo + envolvente de riesgo
              └─ cierres → estadísticas → selección del genoma
                  [ledger, relojes y población todavía incompletos: CX-08]
```

El nodo raíz es la observación disponible; el nodo de decisión no puede
consultar el sufijo. El nodo terminal de métricas tampoco debe mezclar
poblaciones temporales o cierres. Compartir el código del núcleo no prueba
igualdad de datos, modelos, estado inicial, reloj o fills con producción.

## Contrato matemático y significado de los cálculos

Para un flujo x y otro x' con el mismo prefijo hasta k, una transición
causal debe satisfacer S_k(x) = S_k(x') si configuración, estado externo y
semilla son iguales. Equivalentemente, S_i = F(S_(i−1), x_i), sin lectura
de x_j para j > i. Es un contrato de **no anticipación**, no identificación
causal económica: no demuestra que una señal cause retornos.

Los tests observan el estado ANTES de cada evento. Comparan los 34 valores
exportados de features, precio anterior, ATR interno, Hurst, EMAs de vela,
contador, capital, margen y presencia de posición. La comparación usa bits:
los prefijos y el orden de operaciones son idénticos; tolerancias amplias
ocultarían una dependencia indebida. No se observa todo el heap del núcleo:
el alcance de esta prueba es la proyección explícita descrita.

Warmup W significa filas [0,W) consumidas una vez sin entradas, seguida de
evaluación en [W,N). La puerta es i >= W, no i >= min(W,N/10): el tamaño
futuro del archivo no debe decidir la frontera. W sigue siendo una opción
explícita del caller, no una prueba de que todas las escalas estén maduras.
600 ticks a 100 ms representan aproximadamente un minuto, no 600 velas.

La condición de tamaño conserva la política previa de más de diez filas
posteriores al warmup, pero la expresa como N.saturating_sub(W) > 10.
Evita overflow de W+10 sin adoptar diez muestras como suficiencia estadística.
La pertinencia de ese mínimo fijo sigue pendiente de contrato de readiness.

La consulta macro devuelve el último dato con fecha <= día consultado.
Si no existe, usa el sentinel legacy 0, nunca el primer dato futuro. Esto
evita un backfill concreto, pero no resuelve ausencia tipada ni vintages.

## Matriz de expedientes de esta pasada

| ID | Severidad | Estado | Evidencia y relación histórica |
|---|---|---|---|
| CX-01 | P0 | reparado, candidato | Precarga de futuro; MX-19 |
| CX-02 | P1 | reparado, candidato | Aperturas y frontera del warmup; parte de MX-18 |
| CX-03 | P1 | reparado, candidato | Primer valor macro entregado antes de existir |
| CX-04 | P1 | reparado, candidato | Overflow del límite de warmup |
| CX-05 | P0 | abierto | Macro sin publicación histórica ni vintage |
| CX-06 | P1 | abierto | ATR sembrado desde primera fila sin validar |
| CX-07 | P1 | abierto | Relojes y rechazo no atómicos entre capas |
| CX-08 | P1 | abierto | Ledger, población y duración; MX-18/20/21/22 |
| CX-09 | P1 | abierto | Evidencia insuficiente de paridad y readiness |

### CX-01 — Estado construido con el futuro antes del primer evento

**Mecanismo.** La base agrupaba hasta max(W,600) filas del mismo tape y
llamaba `process_kline` antes del bucle desde índice cero. El estado de
Hurst, precio anterior, ATR y otros estimadores ya dependía de ese prefijo
futuro. Cambiar N también podía cambiar cuánto se precargaba. Determinismo
de dos ejecuciones iguales no descubre ese defecto: ambas pueden anticipar.

**Reproducción.** Con 650 ticks sintéticos, W=100 y cadencia de un minuto,
`last_price` antes del evento cero era 95.364,99490466162, no cero.
Multiplicar por 1,7 los precios desde índice 40 alteraba el estado del
índice cero. Comparar el mismo prefijo en tapes de 180 y 800 filas también
fallaba. Son pruebas sobre el runner real, no sobre una réplica del algoritmo.

**Impacto.** Los mismos genomas se juzgan con una historia imposible en
producción. Puede cambiar señales, rechazos y clasificación de candidatos;
no se atribuye una dirección universal al sesgo ni se cuantifica alpha.

**Reparación.** Retirar la precarga y dejar que `process_event` actualice
el estado temporal existente. No agregar un segundo constructor de velas:
`StatefulEngine` ya alimenta Hurst y EMAs al avanzar el reloj. No se acelera
artificialmente readiness ni se inyectan cierres para forzar actividad.

**Cierre acotado.** Estado inicial frío e invariancia de prefijo/sufijo
aprobados en trade-only y libro. No cubre un modelo entrenado con futuro,
mutaciones concurrentes del registro global ni fuentes macro revisadas.

### CX-02 — Calentamiento que podía gastar capital y operar

**Mecanismo.** `warmup_ticks` sólo filtraba el registro de cierres. Los
eventos seguían permitiendo entradas y el capital/DD recibían sus efectos.
Además se recortaba W al 10% del archivo: el valor solicitado no gobernaba
la frontera. Esto separaba la población de PnL reportado de la contabilidad.

**Evidencia de segunda pasada.** La primera prueba corta pasó con la
precarga antigua; no se interpretó como ausencia de bug. Tras quitarla,
una prueba ampliada detectó una posición abierta antes del evento 3 en
trade-only, aunque el warmup solicitado era 19.900. El fallo estaba oculto
por otro fallo: corregir la cronología expuso la apertura indebida.

**Reparación.** Frontera exacta W y `suppress_entries = i < W` en los tres
envíos al núcleo. Se reutiliza el parámetro legacy `latency_panic`, cuyo
uso verificado dentro de `process_event` es permitir/prohibir entradas;
no se cambia la latencia medida ni el latch global de kill-switch. Esta
reutilización merece una API de admisión tipada futura para evitar ambigüedad.

**Verificación.** Hasta la frontera, capital inicial intacto, margen cero
y ninguna posición; el contador de features sí avanza. Control positivo:
con W=0, el mismo fixture abre una posición. Así el test no pasa por haber
apagado permanentemente todo el motor. No se elimina ningún veto de riesgo.

**Límite.** MX-18 no queda completamente cerrado: el reloj de las métricas
todavía incluye el archivo completo y falta reconciliación de ledger.

### CX-03 — Consulta macro anterior al primer dato devolvía futuro

**Mecanismo.** `binary_search` con `Err(0)` retornaba `s.first()`. Pedir
el día 99 a una serie cuyo primer dato es día 100 devolvía 123. El corte
día−1 no lo evitaba. Un test histórico incluso exigía ese comportamiento.

**Reparación.** Sin observación anterior se retorna 0 como sentinel legacy.
El día cero consulta −1 en i64, sin recortarlo a cero y anticipar el cierre
del mismo día. Se actualizan las expectativas incorrectas del test histórico,
manteniendo sus controles de día exacto y huecos de fin de semana.

**Límite.** 0 no es una medición ni acredita neutralidad económica. Falta
un tipo Missing/Observed con fecha de disponibilidad, fuente y cobertura.
No se llama «reparada» a toda la causalidad de datos macro por este cambio.

### CX-04 — Overflow de warmup podía saltarse el rechazo de entrada

**Mecanismo.** W+10 en usize podía desbordar: panic según perfil o wrap
en perfiles que no chequean overflow. Un valor imposible dejaba de expresar
un rechazo fiable antes de construir y ejecutar el motor.

**Reparación.** Sustracción saturada sobre N. El test exige no sólo cero
trades, sino cero eventos visitados para W=usize::MAX. La primera versión
del test sólo comprobaba cero trades y pasó en la base: era insuficiente;
queda documentado, no contado como detección dinámica inicial del defecto.

### CX-05 — Macro revisada no equivale a información disponible entonces

**Evidencia estática.** `OmniHistory::fetch` descarga `fredgraph.csv` y
conserva fecha de observación y valor; no conserva fecha de publicación,
realtime_start/end ni vintage. El fallback local también sólo lleva fecha
y nivel. Un corte por día de observación no distingue revisiones posteriores.

**Consecuencia.** Sigue siendo posible que una evaluación histórica reciba
un valor que aún no se conocía. No se cuantificó el efecto por serie ni se
afirma que todas las series del conjunto se revisen de la misma forma.

**Criterio de cierre.** Dataset point-in-time versionado con `observed_at`,
`available_at`, vintage, procedencia y prueba de invariancia frente a
publicaciones/revisiones posteriores. Ausencia explícita y comportamiento
de cada consumidor bajo datos faltantes. Un retraso fijo arbitrario no basta.

### CX-06 — Primera fila inválida puede contaminar el ATR del harness

**Evidencia estática.** `running_atr` y `prev_mid` se siembran desde
`ticks[0].mid()` antes de la aduana. Aunque el bucle omita esa fila, el ATR
puede conservar NaN/Inf y contaminar el desplazamiento de libro posterior.
No se confunde esto con el ATR interno ni se afirma una pérdida ejecutada.

**Cierre pendiente.** Admisión atómica antes de cualquier estado, siembra
desde la primera fila elegible, diagnóstico de rechazo y prueba de que
insertar una fila inválida no contamine las transiciones aceptadas.
La política sobre filas inválidas y la población de warmup debe ser explícita.

### CX-07 — Orden temporal y frontera de vela no tienen contrato único

**Evidencia estática.** El flag de minuto compara con `ticks[i−1]`, aunque
esa fila haya sido omitida. La admisión del harness no rechaza reloj
retrocedido o cantidades inválidas antes de todos los submódulos. Algunas
capas del núcleo sí rechazan transiciones, pero otras se ejecutan antes.

**Impacto potencial.** Estados parciales, calibraciones omitidas o extras,
confusión entre tiempo del evento y recepción. No se afirma que todos esos
efectos se hayan medido; requiere un test de integración por transición.

**Cierre pendiente.** Contrato de aceptación/rechazo único, secuencia y
timestamps explícitos, contadores trazables y pruebas de duplicado, empate,
retroceso, hueco, reordenamiento y fila descartada en la frontera.

### CX-08 — Salidas y métricas siguen sin un ledger único

**Evidencia.** Siguen presentes `c1.or(c2)`, nocional calculado con mid de
salida en ambas patas y duración desde primera/última fila del archivo.
MX documenta sus mecanismos. El warmup reparado elimina una mezcla de
capital preevaluación, pero no resuelve cada inconsistencia del terminal.

**Cierre pendiente.** Identidad de fill/cierre, costos por pata, tiempo
efectivamente evaluado, posiciones abiertas terminales, motivo de corte y
reconciliación PnL-equity. No modificar goldens para ocultar una discrepancia.

### CX-09 — Compartir núcleo no certifica paridad ni resolución continua

ADR-0006 llama a decisión/física/genoma «paridad por construcción». Lo que
se ha comprobado es reutilización de código. La igualdad funcional exige
igualdad del estado, eventos, entorno de modelos, datos y reglas de ejecución.
La no anticipación corregida aquí era un contraejemplo del harness.

`ReplayTick.ts_ms` tiene resolución declarada de milisegundos. No puede
representar diferencias de un nanosegundo. Los estimadores de minuto
siguen existiendo; este cambio no los convierte en un continuo de todas
las escalas ni demuestra cobertura de cien años. Una discretización debe
tener contrato de error, soporte de datos y costo, no etiquetas de universalidad.

**Cierre pendiente.** Trazas sincronizadas por evento entre replay/demo,
identidades de datos-modelo-genoma, pruebas de consumidor, métricas de
latencia, readiness por escala y validación de capacidad. No se ejecutó
trading, entrenamiento, promoción ni el oráculo T-1 en esta pasada.

## Teoría y redes neuronales: implicaciones, no promesas

Un modelo diferenciable también puede aprender de datos futuros. Antes
de añadir Transformer/GNN/Neural ODE se necesita una representación causal
versionada, labels con disponibilidad y comparación fuera de muestra contra
un baseline. La puerta de riesgo duro permanece independiente; no se vuelve
suave o entrenable sólo para facilitar retropropagación. Son requisitos de
diseño de esta propuesta, no afirmaciones de una arquitectura ya implementada.

El catálogo teórico anterior y las adendas ST/TE se conservan. No se han
implementado nuevas ecuaciones del milenio ni aceleración cuántica. La
reparación presente aporta una propiedad falsable necesaria para evaluar
cualquier candidato, no demuestra la rentabilidad de esos candidatos.

## Fuentes primarias y separación de evidencia

- [scikit-learn: data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage):
  referencia de metodología sobre información no disponible en predicción.
  El fallo local se demuestra con tests, no se deduce de esa documentación.
- [Federal Reserve Bank of St. Louis: Real-Time Periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html):
  distingue información conocida en distintas fechas y datos revisables.
  Sustenta el requisito de vintage; no cuantifica sesgo en este repositorio.
- [Manifiesto oficial Rust del 2026-06-30](https://static.rust-lang.org/dist/2026-06-30/channel-rust-nightly.toml):
  `1.98.0-nightly (096694416 2026-06-29)`, coincide con el compilador local.

CLI Firecrawl no disponible; se utilizó el conector Firecrawl para la
fuente FRED, sin instalar herramientas ni cambiar credenciales. Se contrastó
también la documentación primaria con lectura web. No hay autoridad de
«consejo» simulada: esta es una revisión Codex; la revisión cruzada es externa.

## Verificación, CI y política de integración

Contratos nuevos: primera corrida 2 aprobadas / 4 fallidas; al retirar
precarga y reparar macro, 5/1 con warmup ampliado. Candidato final: 7/0,
con control positivo de apertura y rechazo extremo comprobado antes del motor.
Regresión ampliada: **100 aprobadas / 0 fallidas / 2 ignoradas**:
44 biblioteca (incluidos los siete nuevos contratos), 7 paridad con 2
mediciones manuales ignoradas, 21 métricas, 25 labels y 3 riesgo espectral.
La biblioteca tomó 75,40 s y paridad 34,79 s; el build 56,89 s. Check
`cargo check --workspace --all-targets --locked`: éxito en 1m40s, con
warnings preexistentes. Son tiempos de prueba, no latencia del motor.
El golden existente pasa sin rebaseline. No se ejecutaron los tests de
todos los demás crates; el check de sus targets no sustituye esa ejecución.

```powershell
cargo test -p backtest-engine --locked --lib --test ex_post_metrics_contract --test label_evidence_contract --test spectral_risk_contract --test bt_vivo_parity_audit -- --test-threads=1
cargo check --workspace --all-targets --locked
```

Workflow `Replay contracts`: Windows 2022, Rust nightly-2026-06-30,
checkout fijado por SHA, permisos sólo lectura, credenciales no persistidas,
lockfile obligatorio, comprobación workspace/all-targets y regresión de replay.
No ejecuta `--ignored`, training, tapes ni procesos de trading. No cambia
las protecciones de GitHub. Debe verificarse su ejecución remota; escribir
el archivo no equivale a tener CI verde ni revisión independiente.

Las PR10/20 no modificaban booktick_replay.rs al reservar alcance. Se
preservan ramas TH, Claude y GLM; no se modifica el Cargo.lock sucio del
checkout compartido. El trabajo se publica por PR y no se fusiona sin los
checks y la revisión cruzada solicitada. Sólo se borrará la rama al constar
su incorporación, nunca mientras esté pendiente de revisión.
