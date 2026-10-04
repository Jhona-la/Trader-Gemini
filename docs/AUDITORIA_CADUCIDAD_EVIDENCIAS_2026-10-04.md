# MG02/MG05 — retirada causal de evidencias de riesgo

## Estado y alcance

Parche local en `codex/evidence-expiry-2026-10-04`, base
c6ce7333f670c021db56b2c987680bbd7ebe8c7a. No integrado en RA ni main en
este corte. Dos defectos ya identificados, no nuevos IDs: MG02 retención de
la raíz de Lundberg al perder evidencia; MG05 reutilización de IC de otra
escala o de una publicación anterior. Revisión cruzada independiente en curso.
RED/GREEN efectivos y revisión del padre no equivalen a CI, prueba genética,
paridad completa, benchmark ni autorización para operar.

### Recibo cruzado posterior

Revisor independiente leyó producción, consumidores y tests sin ejecutar cargo,
ni editar: no encontró bloqueadores introducidos en el alcance del parche.
Hashes lib/helper/test estables antes, después y al cierre, iguales a los
registrados abajo. No es aprobación humana GitHub ni reproducción independiente
de las corridas. Los logs no llevan sello automático del hash: se cotejan con
las copias de fuente guardadas y el recibo del worker; esa limitación se conserva.

Cambios exclusivamente en:

- `god-engine-core/src/lib.rs`: fronteras de publicación existentes.
- `god-engine-core/src/evidence_publication.rs`: helper numérico por clave/coin.
- `god-engine-core/tests/evidence_expiry_contract.rs`: 13 controles ejecutables.

No modifica fórmulas del estimador, política R>0, lector de riesgo, registry,
umbrales, genomas, modelos, tamaño de posiciones, mínimo de operaciones o CI.

## MG02: una cota existente no prueba que siga existiendo edge

El estimador de siniestros observa retornos por nocional. Cuando tiene muestras
y deriva suficientes devuelve una raíz R; al perder esa condición devuelve
`None`. La publicación anterior sólo escribía `Some(R)`. Por ello, después de
una trayectoria de pérdidas, el lector podía seguir usando la última raíz
positiva aunque el estimador ya no justificara ninguna cota. El registro de
vetos descriptivo no elimina esa memoria: el error está en la transición entre
productor, registro y lector, no en el texto de la ficha.

La corrección publica R=0 al recibir `None` o R no finito. El lector ya convierte
R no positivo en ausencia: se conserva su política y su conversión de unidades
R_capital=R_nocional/L. El diagnóstico `lundberg_margen_5pct` pasa a0 cuando
falta evidencia; **no significa una cota medida de pérdidas nulas**. Con R
finito se conserva la fórmula ln(20)/R y el criterio R>0 del lector. El 5% es
política existente, no un nivel de confianza calibrado por esta reparación.

No se elimina la clave de la moneda: el lookup hace fallback a una clave
global. Borrarla podría reactivar un R global positivo. Se deja una máscara
numérica local0. Tampoco se usa NaN como máscara: el registro normaliza valores
no finitos a0. Para publicación válida se escribe primero el diagnóstico y
luego R; al retirar se escribe primero R y después el diagnóstico. Ese orden
reduce exposición a la cota anterior, pero **no crea una transacción conjunta**.

## MG05: IC debe corresponder a la escala vigente

El lector de grupo sólo endurece rho cuando el IC SIGNED publicado supera su
base escalar medida. La publicación anterior se hacía dentro de depth y sólo
cuando la escala dominante tenía pares maduros. Si tau cambia de A madura a B
fría, la falta de IC en B no es la continuidad del IC de A. También una tau
inválida debe retirar el valor anterior; pasar por eventos sin depth o dual
directo no puede congelar una evidencia sólo por omitir esa rama.

El helper conserva la búsqueda de la escala más cercana y el cálculo de
`coherencia_media_con_todas`. Publica IC=-1 cuando falta evidencia finita o tau
es inválida; con evidencia vuelve a publicar su valor SIGNED intacto. -1 es
el piso del dominio de la base[-1,1], por lo que no endurece ni una base negativa.
Usar0 como ausencia endurecería injustificadamente una base<0. La máscara local
bloquea el fallback global. Numéricamente coincide con anticorrelación perfecta,
que tiene el mismo efecto en el lector actual de **sólo endurecimiento**; por
ello no acredita que la telemetría pueda distinguir ausencia de IC=-1 medido.

Se publica tras las maduraciones en `process_event`, incluso sin depth; en
`process_tick_dual` para sus callers directos; y tras el feedback de cierre que
recalcula la dominante. La publicación de evento/dual precede al retorno por
kill-switch. No duplica maduración de IC: en evento seguido de dual el update
del espectro con dt=0 mantiene su idempotencia. No autoriza entradas cuando el
kill-switch está activo, ni cambia su política.

## Reproducción efectiva: mismo test, variantes del productor

Nightly2026-06-30, Windows, locked/offline, -j2 y test-threads=1. Cache DEV del
checkout principal; no se inicia host/live ni entrenamiento. El control RED
introduce el helper como extracción del comportamiento anterior (escritura
sólo ante `Some` finito y writer IC dentro de depth); no debe describirse como
un test que ya existiera en la base sin ese módulo. GREEN reemplaza publicación
y cableado, manteniendo el mismo archivo de pruebas.

```text
cargo +nightly-2026-06-30 test --locked --offline -j2 -p god-engine-core --test evidence_expiry_contract -- --test-threads=1
cargo +nightly-2026-06-30 test --locked --offline -j2 -p risk-engine --lib lundberg -- --test-threads=1
```

| Control ejecutado | Propiedad comprobada | RED → GREEN |
|---|---|---|
| mg02_some_none_clears_r_and_margin_on_first_loss_of_evidence_and_recovers | Estimador real positivo→ausente→positivo; R y diagnóstico retirados al primer None | falla → pasa |
| mg02_nonfinite_some_is_absence_and_finite_some_recovers | Some(NaN/±Inf) retira R/margen y una raíz finita recupera la evidencia | falla → pasa |
| mg05_a_cold_b_return_a_does_not_reuse_a_at_b | Pares reales maduros A→B fría→A; B no hereda IC=1 | falla → pasa |
| mg05_invalid_tau_invalidates_previous_ic_and_valid_tau_recovers | tau0/negativa/NaN/±Inf retira IC; tau válida recupera | falla → pasa |
| mg05_trade_event_without_depth_expires_previous_ic_in_real_core | Core real, evento trade sin depth y sin entradas | falla → pasa |
| mg05_direct_dual_tick_expires_previous_ic_in_real_core | Caller dual directo no hereda IC de una publicación anterior | falla → pasa |
| mg05_event_expires_ic_before_kill_switch_early_return | Invalida antes del retorno por kill-switch, sin operar | falla → pasa |
| expired_coin_masks_global_without_expiring_other_coins_or_symbol_scope | Máscara coin0 conserva coin1, símbolo y fallback legítimo de coin2 | falla → pasa |
| mg02_real_risk_reader_stops_using_expired_r_even_with_positive_global | Lector real retira el veto causado por R caducado; no borra contador histórico | falla → pasa |
| expiring_one_arena_does_not_expire_another_arena_with_the_same_coin_id | Dos arenas distintas no comparten las claves retiradas | falla → pasa |
| mg05_real_risk_reader_stops_tightening_on_cold_b_and_resumes_on_a | Lector real deja de sumar aprietes en B y reanuda en A | falla → pasa |
| legacy_finite_scoped_values_and_global_fallback_still_reach_real_risk_reader | Evidencia finita y fallback global legítimo conservan el lector | pasa → pasa |
| concurrent_readers_see_expiry_after_publication_without_cross_coin_reactivation | Lecturas por clave completas; tras handoff, publicación terminada visible; otra moneda/global no reactiva la máscara | falla → pasa |

Resultado RED:1 aprobado/12 fallos/0 ignoradas, exit101. Resultado GREEN:
13 aprobados/0 fallos/0 ignoradas, exit0,0,21s de ejecución tras1m19s de build.
Regresiones Lundberg:6 aprobadas/0 fallos/0 ignoradas/122 filtradas, exit0.
Un intento anterior tuvo cinco errores E0596 de mutabilidad de fixtures;
se corrigieron y **no se contabilizan como RED del comportamiento**.

SHA256 del mismo test RED/GREEN/final:
`797AF948C6E4B82EA385B9DB518956B34FC013E48EDBB2779E253E25718C77E3`.
Fuente GREEN lib:
`5D0F90F1659D6FE2974A1444EBF79D62B04766874844541EE8FBF4CBD743DF42`;
helper:
`78EAF1C0871415D03C49019D22D413AA623E572B6126A06A001C02FF0E639654`.
Fuente RED lib:
`D85F17644E942DD259980F9B5C21DEFDAB42A2EC6F06D07B93B3DBCCE67CB757`.

Logs locales ignorados, cotejados por el padre:
`target/evidence-expiry/negative-contract.log` SHA256
`1C530099D63F8CAD3C9538719C2500BB5CB78F98D72287EC9DE2BAA061F295FB`;
`positive-contract.log`
`D70DA59C6FFD650D6ACAFF217AC89DF8FD36EBC69B26EAF6D356D37E44215A79`;
`lundberg-units.log`
`D6D29CFF08F2B2A365AFB4C936103D1E2B93D181AECC907AE372F01DAB13916C`.
Las copias de fuentes están en `target/evidence-expiry/{red,green}-contract-src`;
preservarlas antes de cualquier limpieza, que no se realiza aquí.

## Check del candidato reconciliado con main

Main e3adf74 se incorporó por merge sin commit sobre basec6ce7333, sin conflicto
textual. Diffs leídos contra ambos padres: frente a main sólo el patchMG y sus
archivos nuevos; frente a base también fichas descriptivas de veto y bitácoras
ajenas preservadas. No cambios adicionales de fuente funcional. Hashlib/helper/
test del review intactos tras la reconciliación. Check del candidato:
`cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j 2`,
exit0,2m33s, sesión23272. Log local `target/evidence-expiry/all-targets-candidate.log`.
Las advertencias existentes no se eliminan mediante cargo fix/global fmt.
Este check compila todos los targets; no ejecuta sus suites ni prueba paridad.

## Límites, integración y siguiente contrato

La concurrencia probada usa un escritor por clave coin y un handoff de canal
al lector de riesgo. Admite generaciones mezcladas entre claves durante una
publicación. No prueba snapshot conjunto R/margen/IC/tau/capital/posición/orden,
linealización del ciclo completo, coherencia ante múltiples cores escribiendo
la misma moneda en el mismo registry, ni revocación de una orden ya validada.
Compartir registry comparte las claves: aislar arenas es distinto de aislar
cores sobre una misma arena. No se ejecutó Loom/Miri, stress ni benchmark.

Los mensajes TG_GENOME_ENV no definido y DarkAlpha fallback pertenecen al
fixture local: no se configura entorno ni se cargan/promueven modelos para
silenciarlos. No trasladar este resultado a producción/demo o a rentabilidad.
MG05 no certifica frescura indefinida de pares, toda la política de grupo ni
los límites económicos heredados. Ausencia vuelve al riesgo escalar existente;
no significa riesgo cero ni garantía de que operar sea seguro.

Residual concreto del estimador: `observar_maduracion` excluye un bloque nuevo
desalineado, pero no caduca necesariamente un enlace que ya acumuló muestras.
`coherencia_par` no recibe tiempo as-of; puede devolver IC histórico en A
después de maduraciones no contemporáneas. Esta reparación retira cuando la
consulta devuelve None/no finito o la tau vigente no tiene evidencia; **no
garantiza que un Some en la misma escala sea reciente**. No regresión nueva:
subcontrato de vigencia MG05 aún abierto. El siguiente contrato debe medir
edad/soporte por par y tau y separar memoria estimada de evidencia consumible;
no inventar un TTL universal ni confundir rechazo de actualización con olvido.

Tampoco se ejecuta un cierre real que pierda R o un feedback real que desplace
tau: los tests ejercitan el estimador/helper/lector y eventos/dual reales por
separado. La neutralidad para base negativa se prueba por dominio e inspección
del lector, no por un fixture de base negativa directa entre estos13 tests.
La mayor frecuencia de publicación tiene coste no medido. Estas reservas
impiden llamar al bloque una certificación de todos los vetos/latencia/IC.

Antes de integrar: revisión cruzada del hash exacto, diff por padre al incorporar
main vigente, all-targets, paridad pertinente y evidencia genética/multiasset
propia si la política de promoción lo requiere. Mantener esta serie fuera de
RA mientras su T1 use la fuente congelada. No sustituir el objetivo72h por una
reparación de metadatos ni presentar13tests como certificación del sistema.
