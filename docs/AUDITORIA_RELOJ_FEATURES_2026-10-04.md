# RA-I-F01 — reloj del prefijo aceptado y aislamiento de rechazos

## Dictamen acotado

Rama `codex/feature-clock-2026-10-04`, worktree separado, base
e3adf74e36aa48c122e83a72d1b7f6c3cbad90bd. Defecto P2 de transición de estado
reproducido: la API rechazaba un tick, pero publicaba su timestamp en el reloj
que consume el enfriamiento y el olvido de pérdidas. Corrección local con
RED18aprobadas/7fallos → GREEN25aprobadas/0fallos, mismas pruebas. No integrada
en main ni RA en este corte. Revisión independiente y verificación ampliada
pendientes; no certificación económica ni auditoría de todo el proyecto.

## Causa desde la base hacia sus consumidores

En `StatefulEngine::try_process_tick`, la guardia de precio era seguida por
la actualización de `last_event_ms`. Después se validaban volumen, timestamp
retroactivo, contador agotado y productos/retorno derivados no finitos. Ante
alguno de esos rechazos la función devolvía `Err`, pero el reloj ya había
cambiado. `process_tick` conserva compatibilidad ignorando el resultado: no
es un contador de errores y no revierte esa primera mutación.

El prefijo aceptado quedaba con `current_ts=1100`, precios/EMAs anteriores,
pero `last_event_ms=9100` después de un volumen inválido. La inconsistencia no
estaba cubierta por el snapshot de regresión, que sólo incluía current_ts y
otros campos. Una prueba verde de ese snapshot reducido no demostraba que
el rechazo dejara intacto todo el estado relevante.

| Nodo | Dato observado | Consecuencia del defecto |
|---|---|---|
| Raíz: guardias de entrada | Precio válido, volumen inválido o derivado no finito | Se rechaza el tick, después de publicar el reloj |
| Estado intermedio | last_event_ms y current_ts pertenecen a prefijos distintos | Tiempo aceptado incoherente aunque las features conservan su foto anterior |
| Decisión: can_open_position_ms | Diferencia entre last_event_ms y último cierre | Puede completar un enfriamiento sin observación aceptada |
| Decisión: racha_amortizada/get_active_directional_streak_ms | Escalera de olvido medida en ese mismo tiempo | Puede amortizar pérdidas antes de aceptar la observación |
| Continuación | Un tick válido posterior al prefijo, anterior al timestamp rechazado | El max del reloj seguía reteniendo el futuro rechazado |

No se demuestra una orden enviada, PnL afectado, bypass general de riesgo o
incidente en producción. `process_event` sanea trade_qty no finita/negativa a0
antes de llamar al motor de features; por eso no todo NaN de la red reproduce
este camino. Los callers directos de la API y guardias de derivados mantienen
el defecto verificable. Además el core cambia otro estado antes de llamar a
features: este arreglo **no es rollback transaccional de GodEngineCore**.

## Por qué los cálculos cambian sin tocar precios

La amortización existente usa
`m=floor(log2(1+elapsed_ms/base_ms))`, seguida de sustracción saturada de la
racha cruda. Con cierre1100ms, reloj1100ms, base1000ms, racha larga3 y corta2,
elapsed=0 y ambas rachas permanecen. Un tick rechazado fechado9100ms imponía
elapsed8000: m=floor(log2(9))=3. El olvido podía dejar ambas rachas efectivas
en0. El enfriamiento compara ese tiempo contra base*2^racha amortizada,
limitado por el ancla lenta existente; ya podía devolver permiso.

Las cifras1000/1100/9100 son un fixture de contraprueba, no parámetros de
trading propuestos. No se cambia retroceso binario, amortización, horizon
anchor, estado sin cierre, política ante base inválida ni riesgo escalar.
El reloj debe medir tiempo de **ticks aceptados**, no número de eventos, tiempo
de pared inventado, frecuencia nanosegundo o horizonte universal sin datos.

## Reparación mínima y equivalencia de datos válidos

La actualización monotónica de last_event_ms se mueve después de todos los
guards actuales, inmediatamente antes de asignar current_ts. La API sigue
devolviendo los mismos motivos de rechazo y acepta los mismos datos válidos.
No se altera el orden de las EMAs, Kalman, acumuladores, espectro, multifractal,
features ML, validación kline, sizing, genoma ni veto.

Para una entrada válida, los guards no mutan estado y el bloque del reloj
ejecuta el mismo max sobre los mismos valores antes de los cálculos. Para un
rechazo, no ejecuta ninguna publicación de reloj. Empates mantienen orden del
caller y timestamp0 sigue siendo un reloj inicializado válido. La rama warmup
y reset se preserva. No se introduce un timestamp sustitutivo ni se clampa un
evento inválido para convertirlo en una observación aceptada.

## Oráculo efectivo y trazabilidad

Se amplía el snapshot con last_event_ms, reloj de último cierre y rachas
generales/direccionales. Se agregan tres pruebas: rechazos futuros no liberan
enfriamiento/olvido; rechazo futuro no eclipsa la siguiente aceptación;
contador agotado no adelanta relojes. Se refuerzan los controles existentes de
timestamp0 y empates. La suite ampliada tiene25 tests:22 anteriores+3 nuevos.

```text
cargo +nightly-2026-06-30 test --locked --offline -j 2 -p god-engine-core --test stateful_transition_contract -- --test-threads=1 --nocapture
```

Windows, nightly2026-06-30, cache DEV del checkout principal. RED compiló
en5m33s y ejecutó25 tests en0,02s; exit101 por18pases/7fallos/0ignoradas.
No error de compilación contado como fallo lógico. Los siete fallos:

- cumulative_volume_overflow_is_rejected_before_mutation;
- exhausted_tick_counter_rejects_without_advancing_any_clock;
- future_rejection_does_not_shadow_the_next_accepted_clock;
- invalid_tick_volume_does_not_mutate_state;
- overflowing_notional_does_not_mutate_state;
- overflowing_relative_return_does_not_mutate_state;
- rejected_future_ticks_do_not_release_cooldown_or_forget_directional_losses.

GREEN usó el mismo comando sin `--nocapture`, compiló en39,45s, ejecutó los25
en0,02s:25pases/0fallos/0ignoradas, exit0. Pruebas idénticas por SHA256:
`BB96D9C922FDD1F1DE6F1823F8BC4170FAC24A5C698C384B64858756DD4CF706`.
Fuente vieja estatal:
`A41A75456CB241C9AA16C6BF6707F2E222FDB0E1058AD7F4B349C981E03773CB`;
fuente corregida:
`5A5940CBEEE5F5EC7984449BBDF86157A6C56F8BA3B840120A1E0DC5ED318A5C`.

Logs locales ignorados, no borrar durante integración:
`target/feature-clock-red.log` SHA256
`A5D7F5C1EDCA8E8C2076742246ACF1B6C0D026ACAEBDE1FB6E9CBFBD50DE3FF9`;
`target/feature-clock-green.log` SHA256
`E8780D48CE5431AEEB7B9C6A96D103FD7FCFB95ACF999AC7CD59159807E5DF76`.
El informe registra hashes cotejados de las fuentes asociadas a cada ejecución;
los logs cargo por sí solos no incluyen un sello automático del commit/hash.
La fuente no se cambió durante cada corrida y el control RED precedió al patch.

## Verificación ampliada y límites

En curso contratos internos `--lib stateful_engine` y check del workspace con
`--all-targets`, locked/offline. No ejecutar otro build ni matar procesos para
sortear una cola de cache. La prueba T1 RA y los bloques MG/forest permanecen
aislados: sus resultados no certifican esta fuente ni ésta sus fórmulas.

El snapshot de prueba incluye campos observables concretos, no toda memoria
privada, threads, arena o posición. La inspección del orden de los guards
demuestra que esos rechazos preceden a la primera mutación en esta función;
no prueba que cualquier secuencia finita deje todos los cálculos posteriores
numéricamente sanos. Tampoco certifica monotonía de productores externos,
desbordamiento de cualquier reloj del host, sincronía multiasset o toda la
política de cooldown. Las reglas heredadas de base inválida y ausencia de
cierre no se modifican en este bloque.

Integración condicionada a revisión del hash exacto, all-targets, comparación
con main vigente, pruebas pertinentes y CI. No se declara certificación total,
motor omnisciente, observación continua1ns–100años ni duplicación cada72h.

## Cierre técnico local posterior

Revisión independiente sólo lectura del hashsource5A5940CB/testBB96D9C9,
estables antes/después/cierre: sin bloqueadores nuevos hallados. Confirma
prioridad de errores, todos los Err antes de mutación, equivalencia para
entradas válidas, cero/empates/warmup/reset. No ejecutó cargo ni verificó
GREEN histórico; el oráculo es del padre, no un segundo resultado independiente.

Comprobación ampliada del padre (sesión27451):

```text
cargo +nightly-2026-06-30 test --locked --offline -j 2 -p god-engine-core --lib stateful_engine -- --test-threads=1
cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j 2
```

Unitarios18pass/0fail/0ignored/145filtered,0,05s, build3m13s; all-targets
exit0 en3m25s. Logs feature-clock-stateful-lib.log y feature-clock-all-targets.log
en target, ignorados. La misma fuente de GREEN/review siguió intacta. Estado
local listo para revisión/integración dentro del alcance; CI y llegada a main
siguen pendientes, no se transfieren resultados de otras ramas.

Residual destacado por review y cotejado en writer del core: timestamp0 válido
para ticks no distingue un cierre en0 del sentinel `last_scalp_exit_ms==0`
usado como «sin cierre». No cierre en0 ejecutado ni impacto real demostrado;
definir presencia explícita con compatibilidad antes de cambiar ese contrato.
El core sigue descartando Result y modificando OFI/otros estados; este patch
no certifica atomicidad del evento completo, estimadores downstream o panic.

## Ejecución explícita del contrato en CI

El commit local `f3018b35` conserva la reparación y sus recibos RED/GREEN.
El workflow heredado compila todos los targets, pero no ejecutaba las pruebas
de `stateful_transition_contract`. Se añade un paso explícito con el mismo
target de 25 pruebas GREEN, el mismo compilador y el lock existente. No se
retiran las regresiones previas, no se activan tests ignorados y no se altera
el timeout ni la fuente de producción/pruebas ya revisada.

La adenda CI establece qué debe ejecutar el candidato remoto; no demuestra
que CI haya terminado ni convierte un check de compilación en un oráculo.
Publicación pública consultada y merge condicionado a los gates pertinentes.
