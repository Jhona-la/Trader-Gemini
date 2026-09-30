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

## Cierre local y bloqueo de publicación

Commits: reparación `0ed10b4b`, CI `cc5441e8`, informe `6f474a0d`;
mainf6903e91 integrado en `5054304c`. El único conflicto era de apéndices
en coordinación: ambos textos conservados. Diff contra ambos padres
revisado y check workspace/all-targets/locked aprobado antes del commit
(3,58 s). El diff de crates y workflow entre 6f474a0d y 5054304c es vacío.

La revisión automática de permisos bloqueó push/PR por requerir aprobación
específica para publicar CX en el repositorio PÚBLICO. No se reintentó ni
se usó otra vía. La mención a conflicto pendiente en ese rechazo era
obsoleta: la verificación posterior muestra índice limpio y ninguna ruta
unmerged. Consulta remota de rama/PR CX: inexistentes en este cierre.
La CI está escrita y versionada sólo localmente; no ejecutada en GitHub.
Revisión cruzada pendiente. No hay merge CX a main ni rama CX que borrar.

## Adenda CX-06 — rechazo de precios y contaminación del ATR (2026-09-29)

Esta adenda preserva el corte anterior como historia. Actualiza CX-06 a
**reparación candidata local de la siembra del ATR**; no certifica admisión
atómica integral. Balance actualizado de los nueve expedientes CX: cinco
reparaciones candidatas y cuatro abiertos (CX-05, CX-07, CX-08 y CX-09).
La trazabilidad de rechazos y la validación conjunta de cantidades/reloj
siguen pendientes en CX-07. No se presenta esta corrección como una
auditoría archivo por archivo de todo el repositorio.

### Causa, alcance y cadena de efectos

La aduana de precios comprobaba mid finito/positivo, bid/ask positivos y
libro no cruzado. Sin embargo, antes de entrar al bucle se ejecutaban
`running_atr = 0.001 * ticks[0].mid()` y `prev_mid = ticks[0].mid()`.
El `continue` descartaba el evento, pero no deshacía esa inicialización.
El rechazo era efectivo para el núcleo y a la vez inefectivo para el
estado auxiliar del simulador: una desconexión entre raíz y ejecución.

```text
Primera fila inválida ──► semilla ATR/prev_mid ANTES del rechazo (defecto)
                               │
                         EMA posterior alterada
                               │
                      bid/ask simulados alterados
                               │
                 fills, capital o posición distintos

Candidato: fila ─► aduana de precios ─► primera semilla / siguiente EMA
                         └─ rechazo: no siembra ni actualiza ATR
```

Para NaN, la EMA conserva NaN; para infinito, también puede producirlo
al multiplicar el desplazamiento por cero. Por tanto, desactivar el
desplazamiento no neutralizaba el defecto. El core podía usar su fallback
de cotización al recibir límites no válidos. Para precios negativos o un
libro cruzado finitos, la diferencia frente al primer precio aceptado
introducía un salto inexistente; no hacía falta que apareciera NaN.

El alcance dinámico demostrado es el modo libro (`trade_only=false`). El
modo trade-only construye sus cotizaciones desde el mid aceptado y sirve
como control negativo. No se extrapolan los efectos de estas fixtures a
rentabilidad, pérdidas reales, frecuencia en producción ni todos los activos.

### Reproducción falsable antes de reparar

Base de código `2640bbbd`. Se añadieron cinco tests sin modificar el runner:
cuatro prefijos inválidos y una cinta íntegramente rechazada. El mismo
replay real procesa 80 filas válidas deterministas y una variante con una
fila inválida antepuesta. La fila añadida lleva exactamente el timestamp
de la primera válida y se usa `warmup_ticks=0`: así se aíslan el reloj de
CX-07 y la población cruda del warmup. Los índices se realinean por evento
aceptado, no por posición física en el archivo.

Se comparan bit a bit 34 features, precio/volatilidad/Hurst internos, EMAs,
contador de ticks, capital, margen y presencia de posiciones. Son 80
instantáneas antes del evento, no inspección exhaustiva del heap ni una
prueba independiente del último estado terminal. Los parámetros cruzados
son ambos modos y desplazamiento `0.0`/`0.10`.

| Primera fila añadida | Primera divergencia observada, índice aceptado base 0 | Condición |
|---|---:|---|
| bid/ask NaN | 10 | Libro, desplazamiento 0 |
| bid/ask +infinito | 10 | Libro, desplazamiento 0 |
| bid=-100, ask=-90 | 3 | Libro, desplazamiento 0,10 |
| bid=900.000, ask=600.000 | 3 | Libro, desplazamiento 0,10 |

En NaN se observó capital `1000.05353295625` sin la fila rechazada frente
a `1000.05435539213` con ella, en la instantánea 10. El sentido del error
puede favorecer un backtest: no es solamente un sesgo conservador. En el
caso negativo, en la instantánea 3 la referencia tenía posición abierta
y margen `8.28120985010707`; la variante tenía margen cero y ninguna
posición. Es evidencia de resultado simulado distinto, no una estimación
del impacto económico esperado. Los bits y aserciones de estas corridas
proceden del contrato ejecutado, no de un simulador matemático duplicado.

Resultado RED: **8 aprobadas / 4 fallidas / 0 ignoradas**, 3,47 s de tests
y 50,41 s de compilación. Incluye los siete contratos CX previos. La cinta
de 20 filas inválidas mixtas ya dejaba el núcleo frío y capital intacto;
ese control impide confundir «no hubo trades» con ausencia de contaminación
que sólo se manifiesta cuando llegan precios válidos después.

### Reparación y significado de los cálculos

La pareja `(ATR, mid anterior)` comienza como `None`. Sólo después de
superar la aduana de precios recibe estado. En la primera fila aceptada
se conserva exactamente la semilla previa `0.001 * mid` y se ejecuta la
misma primera actualización; no se utiliza cero como sustituto de datos.
Las siguientes filas aceptadas siguen la misma recurrencia:

`TR_k = max(ask_k - bid_k, abs(mid_k - mid_(k-1)))`

`ATR_k = 0.02 * TR_k + 0.98 * ATR_(k-1)`

`slip_k = ATR_k * max(shift_atr_frac, 0)`

TR expresa un rango monetario en unidades del precio cotizado; ATR es su
promedio exponencial por evento admitido; slip desplaza cada lado del
libro simulado. No es una probabilidad, un retorno ni el ATR porcentual
interno del core. La corrección cambia la procedencia del estado, no el
coeficiente, la semilla, el tamaño de posición ni una puerta de riesgo.
En una cinta válida desde el principio se conserva el orden aritmético
de cada actualización. El estado auxiliar no añade un nuevo buffer ni
un recorrido del tape; costo y memoria adicionales constantes.

Conservación no significa calibración demostrada: `alpha=0.02` tiene
semivida `ln(0.5)/ln(0.98) = 34.30961849` eventos admitidos, no segundos.
La duración física cambia con la intensidad del feed. Ni ese coeficiente,
ni la semilla del 0,1% ni el desplazamiento histórico del 10% reciben aquí
una justificación universal. Una futura sustitución por tiempo transcurrido
necesita especificar escala, unidades, política de huecos, datos de
calibración y prueba de paridad; no elegir otro número sin evidencia.

Resultado GREEN del mismo conjunto: **12 aprobadas / 0 fallidas / 0
ignoradas**, 3,51 s de tests y 24,88 s de compilación. Las cuatro pruebas
que fallaban ahora recorren todas sus combinaciones; la cinta totalmente
rechazada mantiene capital inicial y no muta el núcleo en ambos modos.

### Límites de cierre y coordinación

- No se valida aquí cantidad no finita/negativa, orden temporal, duplicados
  ni frontera de minuto calculada desde una fila descartada. CX-07 sigue abierto.
- El warmup sigue contando filas crudas; no se redefine silenciosamente
  como número de eventos aceptados. La invariancia demostrada usa W=0.
- La duración estadística sigue tomando primera/última fila del archivo;
  una cinta enteramente inválida no es evidencia de mercado aunque conserve
  capital. Población evaluada y estado «sin evidencia» pertenecen a CX-08.
- No se añaden teorías cuánticas ni se certifica crecimiento, omnisciencia,
  continuidad nanosegundo a siglo, ausencia total de bugs o paridad live.
- Se mantiene el bloqueo de publicación pública CX; no hay nuevo push/PR.
  GLM y Claude no han emitido una revisión cruzada de este candidato.

La consulta remota de esta pasada encontró main `9ed0cb8c` y sólo ramas
remotas main y las dos Claude; PR10 abierta, PR20 abierta y draft. El nuevo
commit de GLM es documental y revisa MX/PR21, no CX. Su mención de
«linaje ... corregido» para MX-20/21/22 necesita una precisión: aún están
`span` del archivo, `2 * qty * mid` de cierre y `c1.or(c2)` en el runner.
Se conserva su nota original y se añade la discrepancia; no se convierte
una valoración favorable en cierre técnico de esos hallazgos.

### Verificación ampliada y cierre de esta adenda

Commit de reparación/pruebas: `88136410`. Misma orden ampliada documentada
arriba: **105 aprobadas / 0 fallidas / 2 ignoradas** en cinco bloques
(49 biblioteca, 7 paridad + 2 ignoradas, 21 métricas, 25 labels y 3 riesgo
espectral). Biblioteca: 74,80 s; paridad: 31,79 s; build: 47,69 s.
Las dos ignoradas son mediciones manuales con tapes reales. El golden
existente se conserva y pasa. No se ejecutó la suite de todos los crates.
`cargo check --workspace --all-targets --locked` aprobado en 23,53 s;
persisten warnings previos. Estos tiempos no son benchmarks del hot path.

Main remoto `9ed0cb8c` incorporado localmente en `8f9c27aa`: conflicto sólo
en apéndices de coordinación, ambos conservados. Verificación adicional:
las 1.434 líneas del primer padre y las 1.415 del segundo se preservan
en orden en el buzón. Diffs contra ambos padres inspeccionados; el código
no cambia frente a `88136410`. Check all-targets/locked repetido antes
de cerrar el merge: aprobado en 3,55 s. Ningún conflicto sin resolver.

La nota GLM y esta precisión quedan juntas. El aviso ignorado compartido
permite localizar el trabajo, pero no acredita acuse. No se tocó el
checkout principal, los procesos de entrenamiento ni los modelos ajenos.
No había rama local ya integrada distinta de main para eliminar; las
otras ramas contienen trabajo no integrado y se preservan.

Publicación CX sigue bloqueada por falta de autorización específica para
exponer estos cambios e informes en el repositorio público. No se hizo
push ni PR y no se eludió la revisión de permisos. El workflow contiene
esta nueva suite porque ejecuta la biblioteca completa, pero no se ha
ejecutado remotamente. Revisión cruzada CX y merge a main pendientes.

## Autorización de publicación CX — actualización posterior al cierre local

El operador respondió «Sí» a la solicitud específica de publicar estos
cambios e informes CX en el repositorio PÚBLICO Jhona-la/Trader-Gemini y
abrir su PR, condicionando el merge a CI y revisión cruzada. Queda levantado
el bloqueo de publicación; las notas anteriores documentan el estado de
sus respectivos cortes y no la autorización vigente. No cambia el código
probado, los resultados 105/0/2 ni los cuatro expedientes abiertos CX.

Comprobación previa: origin/main9ed0cb8c, rama local limpia, sin PR CX
existente. No se publican modelos ni tapes, no se ejecuta entrenamiento y
no se modifica el checkout compartido. Publicar no significa integrar ni
aprobar: la URL, SHA remoto, CI y revisión deben comprobarse después.

## Publicación comprobada — PR #22

Publicación realizada tras autorización: [PR #22](https://github.com/Jhona-la/Trader-Gemini/pull/22).
SHA de apertura765b9d340ed5fdfecd1456950fad64c42d595916 idéntico en checkout,
rama remota y PR. Base9ed0cb8c; GitHub indica MERGEABLE, sin conflictos.
El diff de código y workflow frente al candidato probado88136410 es vacío;
los cambios posteriores a ese candidato fueron de documentación/coordinación.

La [CI inicial](https://github.com/Jhona-la/Trader-Gemini/actions/runs/36668221994)
comenzó en estado IN_PROGRESS. Este registro es de la apertura, no afirma
éxito de esa corrida ni sustituye el estado vivo de checks de la PR.
La consulta inicial no devuelve revisiones ni revisores asignados. Se solicita
revisión cruzada de los contratos y del candidato vigente, con SHA explícito.

No se fusiona ni se habilita auto-merge hasta CI satisfactoria y revisión
cruzada; la rama se conserva. No se requiere más autorización de publicación
CX, pero publicar no resuelve los expedientes abiertos ni certifica el sistema.

## Integración CX / XLIX-A — nueva evidencia de composición (2026-09-29)

### Corte, alcance y diagnóstico

La PR22 publicada en `0ef061d8` se comparó con main `2080e423`, que añade
la reparación GLM `281786bd` y su registro. Ambos intentan impedir operar
con datos futuros del prefijo, pero usan mecanismos distintos. La revisión
comprende el diff de esos commits, los tres archivos en conflicto, la
prueba añadida por GLM y los contratos CX. No añade una auditoría semántica
de todo el repositorio ni una afirmación de revisión por varios agentes.

| ID de seguimiento | Tipo | Evidencia | Decisión de integración |
|---|---|---|---|
| CX-I01 | P1, riesgo de composición | Sin precarga, conservar el salto `max(W,600)` deja filas sin observar; dos contratos fallan en evento 1 | Un solo recorrido causal; W suprime entradas, no observaciones |
| CX-I02 | P1, insuficiencia de prueba | El test GLM pasa con la misma combinación defectuosa | Se conserva y se añaden contratos de estado; comentarios limitados a lo que realmente prueba |
| CX-I03 | P2, integridad documental | Main2080e423 contiene tres marcadores de conflicto de stash versionados | Quitar delimitadores preservando en orden ambos textos |

Estos tres registros son hallazgos de integración asociados a CX-01/02 y
al proceso de verificación, no tres nuevas reparaciones económicas. El
inventario original CX conserva cinco candidatos y cuatro abiertos; no se
duplica MX-19 para inflar el total de defectos resueltos.

### CX-I01 — dos reparaciones válidas en intención no se componen por suma

GLM conserva la precarga de klines del prefijo y evita evaluarlo después
mediante `frontera_preload = max(warmup_ticks,600)`. CX retiró la precarga:
el mismo stream produce incrementalmente las features, y durante las
primeras W filas se impiden entradas. Conservar simultáneamente «no hay
precarga» y «omitir la historia ya precargada» elimina la única fuente de
esas observaciones. Es un fallo de composición, no un conflicto de nombres.

Formalmente, si `T(x_i,S_i)` es la transición de observación, la ruta
incremental exige `S_(i+1)=T(x_i,S_i)` para cada fila aceptada. Durante
`i<W` sólo se impone `allow_entries=false`. El salto incorrecto aplica
`S_(i+1)=S_i` a todas las filas `i<max(W,600)`: conserva un estado frío
aunque los datos existan. En una cinta de 80 filas con W=0 no observaría
ninguna. No se soluciona elevando W ni sustituyendo el estado faltante por
una constante. Tampoco debe reintroducirse la precarga futura para lograr
que una prueba de actividad vuelva a pasar.

Se construyó **una mutación local controlada, nunca commiteada ni publicada**:
el runner CX, sin precarga, más el salto de GLM. El observador se mantuvo
antes del salto para inspeccionar el estado del motor. Dos contratos nuevos
fallaron: **0 aprobadas / 2 fallidas**, 0,05 s; build 29,55 s. En ambos,
antes del evento de índice1, `tick_count` era 0 cuando debía ser 1 en
trade-only. Se reprodujo con W=0 y W=10. Esto demuestra el fallo de esa
combinación; no se atribuye falsamente esa versión combinada a main.

La resolución preserva la intención común —historia causal sin posiciones
en warmup— mediante el recorrido incremental de CX. No retiene el salto
ni un mínimo oculto de 600 filas. Agrega una explicación junto al bucle
para evitar reintroducirlo en una integración posterior. No modifica
coeficientes, genomas, límites duros, kill-switch, fees o física de fills.

### CX-I02 — determinismo y sanidad no certifican causalidad

Sobre la misma mutación defectuosa se ejecutó, sin alterar sus aserciones,
`xlixA_mx19_prefijo_consumido_una_vez_y_determinista`: **1 aprobada / 0
fallidas**, 2,28 s; build 39,95 s. Compara operaciones/PnL de dos corridas,
capital final finito/positivo y drawdown menor que uno. Un motor que omite
observaciones puede seguir siendo determinista y entregar cifras finitas.
El nombre y los comentarios atribuían al test una garantía que las
aserciones agregadas no establecen. Esta ejecución prueba el punto ciego
frente a la mutación ensayada, no invalida todo el trabajo de GLM.

Se conserva la fixture, el nombre y todas sus aserciones ejecutables.
Sólo se aclaran comentarios: con tendencia intensa durante 700 filas y
W=600, las últimas 100 quedan fuera del warmup; comparar capital finito
para W=600/700 no demuestra dominancia de actividad ni ausencia de entradas
en el prefijo. Los contratos añadidos observan los estados que faltaban.

En la resolución candidata, **14 contratos CX pasan / 0 fallan** en
3,77 s; build 26,70 s. Las nuevas comprobaciones cubren cinta de 80 filas,
W=0/10/60/69, ambos modos y llegada efectiva a la frontera. Durante warmup
exigen capital intacto, margen cero y ausencia de posiciones; el contador
de observaciones debe avanzar en cada fila. En el contrato actual del
harness hay un evento por fila en trade-only y dos en modo libro; esta
prueba no certifica que ese modelado de microestructura sea paridad live.

### CX-I03 — un índice limpio puede contener conflictos ya versionados

Main2080e423 tenía `<<<<<<< Updated upstream`, `=======` y `>>>>>>> Stashed
changes` dentro de COORDINACION_CODEX_2026-09-28.md. `git status` estaba
limpio: sólo describe el índice, no detecta contenido incorrecto previamente
commiteado. Los marcadores no eran del merge nuevo y podían confundirse
con instrucciones de resolución vigentes. Había además un bloque histórico
reincorporado desde stash; no se descartó por parecer duplicado.

Se conservaron ambos textos y se retiraron sólo los delimitadores. La
comprobación por subsecuencia preservó, en orden, las 1.508 líneas del
primer padre y las 1.529 líneas no delimitadoras del segundo. Los conflictos
de apéndices en el informe MX también conservaron ambas versiones. El
historial antiguo queda explícitamente histórico: su afirmación de usar
precarga/salto no describe la resolución actual CX.

Control preventivo añadido al workflow: checkout con profundidad2 y
`git diff --check HEAD^ HEAD` antes de compilar. En un evento PR revisa
el delta del merge candidato contra su primer padre; en push revisa el
commit frente a su padre. `git show --check 2080e423` reprodujo los tres
marcadores con código de salida2. Este control detecta problemas de diff,
no conflictos semánticos, no inspecciona toda la historia ni garantiza
que main esté protegido ante pushes directos. No se eliminó ningún check
de compilación ni ninguna prueba para introducirlo.

### Implicaciones para el genoma y las escalas

Si el estado con el que se evalúa un genoma depende de filas futuras o de
observaciones omitidas, la aptitud deja de representar la trayectoria
declarada. Corregir esa trayectoria es previo a comparar desempeño contra
demo/producción; no demuestra que el genoma tenga edge ni que se haya
resuelto toda la discrepancia entre entornos. No se reentrena ni promueve
ningún candidato en esta integración.

MX-19b de GLM se conserva abierto: número de filas no equivale a historia
en tiempo físico. Tampoco se impone un warmup global arbitrario de 512
minutos a todos los consumidores. El cierre necesita readiness verificable
por estimador/escala, timestamps aceptados, frecuencia efectiva, huecos y
linaje de modelos/features. Un cálculo avanzado o una red neuronal no
reemplaza esas condiciones de observabilidad y reproducibilidad.

La CI de `0ef061d8` no valida por sí sola este nuevo árbol integrado. Los
resultados de la regresión conjunta, check all-targets y SHA final se
registrarán al completar la validación. Revisión cruzada CX aún pendiente;
no confundir resolución local del conflicto con merge de PR22 a main.

### Cierre de validación local de la integración

Regresión ampliada: **108 aprobadas / 0 fallidas / 2 ignoradas**. Biblioteca
51/0/0 (72,20 s), paridad 8/0/2 (35,95 s), métricas21/0/0, labels25/0/0,
riesgo espectral3/0/0. Build40,58 s. La prueba de GLM pasa también en la
resolución correcta. El golden existente no se alteró; las dos ignoradas
siguen siendo mediciones manuales con tapes. No se ejecutaron todos los
tests de los demás crates ni training, promoción, exchange o T-1.

`cargo check --workspace --all-targets --locked` aprobado en20,12 s antes
de cerrar el merge, con warnings. Diffs de la resolución contra ambos
padres inspeccionados; `git diff --check` limpio y sin delimitadores en
los archivos revisados. El cambio operativo respecto al primer padre CX
es nulo: el runner sólo recibe comentarios; respecto al segundo padre se
reconcilia la reparación GLM con el recorrido incremental y sus pruebas.
El workflow sí añade el control preventivo de diff, sin quitar cobertura.

La matriz original continúa con cinco candidatos y cuatro expedientes
abiertos. CX-I01/02/03 quedan documentados como controles de integración
local, no como cierre de la arquitectura universal o del objetivo financiero.
La revisión cruzada debe referirse al nuevo SHA. No había otra rama local
ya integrada distinta de main para eliminar; CX sigue pendiente de merge.

### Revisión GLM recibida e incorporación documental posterior

Resolución de integración publicada en `a85b57e4`. Durante la validación,
GLM emitió revisión técnica favorable en PR22 sobre `0ef061d8` y recomendó
retirar `frontera_preload`, conservar CX sin precarga y mantener las
observaciones durante warmup. Esa es la resolución aplicada. La prueba
GLM se preservó; los dos contratos adicionales hacen observable su límite.
Estado de la revisión en GitHub: COMMENTED, no APPROVED. No se atribuye
al revisor una inspección del SHA nuevo que todavía no ha confirmado.

Main `b75db332` añadió únicamente el registro de esa revisión y se incorpora
conservando ambos apéndices. No cambia código ni workflow respecto a
`a85b57e4`, cuyo árbol produjo 108/0/2 y el check20,12 s. Se solicita
confirmación de la resolución integrada y CI del candidato vigente antes
de merge a main. No se habilita auto-merge ni se elimina la rama pendiente.

Se conserva una discrepancia científica explícita: que la ruta incremental
coincida no demuestra paridad del estado completo, disponibilidad de datos,
historia suficiente por escala ni latencia idéntica. Por tanto, la parte
de preparación física de estimadores de MX-19b/CX-09 sigue abierta; no se
considera resuelta sólo porque la revisión apoye la arquitectura elegida.

## Cierre de integración CX verificado — 2026-09-30

Este seguimiento actualiza el estado sin borrar los cortes anteriores.
GitHub reporta PR22 **MERGED** en c75f23ce44acbed4291237fb0617971ef2c6afa9
(12:27:51Z). CI36670361992 para e8546d60 concluyó SUCCESS: check de todos
los targets y regresiones del workflow. Codex verificó ancestralidad a
origin/main ee438edb, ningún delta src/crates/workflow frente a e8546d60
y preservación en orden del texto de coordinación. GLM realizó el merge
y dejó su comprobación postmerge en el buzón; no se la atribuye a Codex.

Rama remota retirada por GLM; referencia local CX eliminada por Codex
sólo tras verificar integración y ausencia de un worktree que la usara.
El historial se conserva en main. La revisión técnica favorable quedó
COMMENTED, no una aprobación formal APPROVED; no se reescribe ese hecho.

Cinco reparaciones CX están ahora integradas, no meramente publicadas.
CX-05/07/08/09 y los límites de readiness/paridad/linaje permanecen abiertos.
Integración y CI no son certificación integral, ni re-baseline T-1 ni prueba
de retorno. MR examina después el linaje de modelos y sus gates en
docs/AUDITORIA_REGISTRO_MODELOS_2026-09-30.md, sin cambiar los fixes CX.
