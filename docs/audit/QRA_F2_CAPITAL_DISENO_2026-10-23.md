# QRA F2 — diseño de capital observado

**PROPUESTA SIN IMPLEMENTAR.** Fuentes H `ebe5d1b6b2e78a2b0c7d3a19074e0c797d970147` y M `68839f9683afdec64e4c992fdc6ce7e9d7c5441c`, cuyos854 archivos Rust son iguales. Diseño elaborado mediante lectura, sin Cargo ni cambios de fuentes. El cierre F0/F1 consta en [su recibo](QRA_F0_F1_RESULTADOS_2026-10-23.md); este documento no acredita la implementación ni el CI de PR106.

## Problema y aceptación observable

Un evento válido puede indicar que ya no queda capital defensivo, pero el camino actual conserva un valor positivo anterior. Hay tres filtros encadenados: `wallet_balance > 0` en el stream, `usdt > 0` en el puente y `exchange > 0` en el freno. El cero también funciona como marcador de ausencia del escenario, de modo que quitar sólo los filtros no resuelve el contrato.

Con arena y capital externo anteriores de 100, los negativos que deben distinguirse son:

| Entrada y contexto | Resultado requerido |
|---|---|
| Cuenta declarada flat mediante snapshot válido; `wb:"0"`, `P:[]` | Publicar cero observado de esa sesión; el capital del freno debe ser cero, nunca 100. |
| Wallet 10 y posición válida con `up:-10` | Equity cero; activar la misma defensa. |
| Wallet 10 y posición válida con `up:-20` | Equity negativa finita; conservar una defensa `<=0`, nunca recuperar el arena positivo. Preferir conservar -10 para diagnóstico. |
| USDT sin `wb`, o `wb:null`, texto, NaN/Inf/overflow | Error de evidencia, no un cero observado. No publicar ni mutar parcialmente el cache. |
| Escenario explícito sin cuenta adjunta | Mantener fallback al capital del arena. No liquidar un backtest por ausencia de exchange. |
| Cuenta adjunta de nueva sesión, sin snapshot válido | Desconocido: bloquear admisión, sin inventar insolvencia ni heredar el valor de la sesión anterior. |
| Evento o REST de ejecutor/generación anterior | Rechazar antes de mutar evidencia, cash, telemetría vigente o admisión. |

La ecuación del freno permanece `min(capital_arena, capital_externo_observado)` cuando existe observación finita vigente. Con pico 100 y capital externo cero, `freno_duro` ya calcula caída 1; con -10 calcula 1.1. No se necesita modificar sus umbrales actuariales. El host ya define insolvencia como `cap <= 0` y dispone de un kill-switch con rearme humano.

## Mapa de código completo para este átomo

Anclas verificadas en el HEAD del corte; volver a buscar antes de editar:

| Productor, consumidor o frontera | Ruta y anclas | Trabajo necesario |
|---|---|---|
| REST: cartera requerida, finita y no negativa | `crates/execution-engine/src/execution_evidence.rs:79–103` | Preservar su dominio de wallet y el error ante ausencia. El nombre `parse_account_equity` no cambia la semántica real de `totalWalletBalance`. |
| Transporte account actual | `crates/execution-engine/src/executor.rs:3989–4025` | Reutilizar la misma consulta account; no añadir otra cuenta, endpoint o fallback económico. |
| Bootstrap del host | `src/bin/god_engine.rs:1439–1454` | Llevar el token junto a la respuesta que inicializa el capital. Un error sigue reintentando, sin reemplazarlo por 0 o 13. |
| Puente de transporte escalar | `src/bin/god_engine.rs:1647`, `1809–1819` | Sustituir la autoridad del `AtomicU64` por una observación coherente; mover el puente a `host_oms` para ejercitar exactamente su implementación productiva. |
| Stream y sus dos constructores reales | `crates/execution-engine/src/user_data_stream.rs:735–817`; `src/bin/god_engine.rs:1895–1918`, `4270–4296` | Campo `wb` requerido; equity firmada finita; ambos streamers deben usar el mismo puente protegido por sesión. |
| Presencia y valor actuales en arena | `crates/quantum-arena/src/state.rs:730–735`, `837` | Reemplazar el significado doble de cero por estado explícito y una sola publicación atómica. |
| Único writer productivo de `capital_exchange` | `src/bin/god_engine.rs:2171–2181` | Quitar el relay que copia cada 5 s desde el escalar. La defensa debe leer el mismo snapshot que publicó el puente. |
| Único reader físico de `capital_exchange` | `crates/risk-engine/src/drawdown.rs:221–230` | Separar cuenta ausente, cuenta desconocida y valor observado, incluido cero/negativo. |
| Tres consumidores del resultado | `src/bin/god_engine.rs:2181–2210`; `crates/risk-engine/src/lib.rs:389–421`; `crates/god-engine-core/src/lib.rs:9622–9626` | Host, riesgo y consejo F1-049 deben usar el mismo plano. El bloque del core calcula drawdown tras una orden admitida por riesgo; también debe manejar ausencia si cambia el retorno. |
| Identidad ya disponible | `src/host_oms.rs:105–115`, `140–180`, `260–275` | Reutilizar generación `u64` y `Arc::ptr_eq` del ejecutor; publicación y reemplazo se excluyen mediante el mutex de sesión existente. |
| Refresh de cartera | `src/bin/god_engine.rs:4313–4330` | Manejar `Err` de forma explícita; la respuesta tardía no puede publicar para otra sesión ni tapar un evento posterior. |
| Admisión OMS final | `src/host_oms.rs:188–238` | Exigir capital conocido y positivo para la sesión, además del contrato vigente de fees/exposición. El papel explícito sin cuenta conserva su política de escenario. |
| Publicación tardía del core | `crates/god-engine-core/src/lib.rs:10130–10145` | Revisar el mismo snapshot bajo el gate si se pretende garantizar que una propuesta aún no publicada quede vetada por un cero recibido después de evaluar riesgo. |

Los otros readers del escalar del host son `god_engine.rs:4342/4543–4544` (DB/telemetría) y `5469` (profiler). Deben leer el snapshot o quedar como espejos diagnósticos sin autoridad de riesgo; no convertir `None` a cero en un mensaje que afirme capital observado. Las escrituras REST a `arena.unified_capital` y `base_capital` en 4321–4322 pertenecen al cash local y no deben cambiarse a equity con PnL abierto, lo que mezclaría planos.

Escritores adicionales encontrados de `capital_exchange`, todos en tests: `crates/risk-engine/tests/cf5c_pico_plano_y_venue.rs:56/67/69` y `crates/god-engine-core/tests/f1_047_049_050_whiplash_council_vecm_contract.rs:136`. No hay otro productor productivo en el grep Rust del corte.

## Representación propuesta: presencia, sesión y valor juntos

Preferir un snapshot inmutable compartido mediante `ArcSwapOption`, dependencia que `quantum-arena` ya utiliza para el libro de fees. No añadir `AtomicBool observed` junto a un `AtomicU64 value`: aunque cada campo sea atómico, un lector podría combinar presencia nueva con saldo viejo o identidad de otra generación. Tampoco usar NaN, infinito, cero o un valor negativo como sentinela.

Esquema de API propuesto, no código implementado:

```rust
// Una sola raíz atómica compartida entre OmsSession, bridge y arena.
type CapitalBook = Arc<ArcSwapOption<SessionCapitalState>>;

struct SessionCapitalState {
    generation: u64,
    observation: Option<CapitalObservation>,
}

struct CapitalObservation {
    equity: f64, // Constructor exige finito; admite cero y negativos.
    source: CapitalSource, // REST account o estimación UDS; no afirmar equivalencia completa.
}
```

Interpretación: raíz `None` significa escenario sin cuenta; raíz `Some` con `observation:None` significa cuenta adjunta todavía desconocida; raíz `Some` con observación significa evidencia admitida de esa generación. El constructor controla finitud antes de crear una fila. El snapshot es runtime y no se restaura como prueba de cuenta desde el registro genérico de parámetros.

`OmniscientRegistry::get_value_fast` ofrece `Option`, pero hoy no hay una clave registrada de capital y sus snapshots pueden restaurar parámetros. Añadir una clave numérica de presencia allí no demuestra identidad/frescura. La propuesta usa el mecanismo de snapshots existente, sin convertir una generación `u64` a `f64` ni crear una segunda bandera global.

Crear el book en `OmsSession` antes del stream y compartir ese mismo `Arc` con `GlobalArena` antes de ponerla en uso. No copiar el book cada 5 s: una copia podría repoblar la generación nueva con un valor retenido del ejecutor anterior. `replace_executor`, bajo el orden de locks actual sesión → gate de publicación, instala el estado desconocido de la nueva generación antes de publicar su ejecutor. Si devuelve BUSY, conserva la identidad y observación de la sesión todavía dueña del transporte.

## API de producción y semántica REST/UDS

1. **`CapitalBridgeSink` productivo en `host_oms`.** Debe guardar `Arc<OmsSession>` y el `SessionToken` que corresponde al streamer. `on_capital` llama a una publicación explícita de equity finita firmada; el método toma el mutex de sesión y verifica generación y puntero antes del único store del book. Un evento viejo no llega a telemetría vigente a través de otro callback sin guarda. Se conserva separado el canal de cierres `AuditorCloseSink`.
2. **No ampliar el dominio de wallet.** `publish_current_balance` y el parser de cartera siguen rechazando wallet negativo. Añadir una operación distinta para equity firmada evita que aceptar -10 de equity permita -10 de cartera por accidente.
3. **Respuesta account tipada, en la misma consulta REST.** Propuesta mínima para no sobrescribir equity con wallet: un `AccountCapitalSnapshot` con `wallet_balance` para el cash local y `margin_balance` para el plano defensivo. Ambos campos son requeridos y finitos; wallet no negativo, margin balance firmado. El DTO directo de serde rechaza duplicados de campos reconocidos y permite desconocidos. Un método inherente de `OrderExecutor` devuelve ese snapshot; el trait `fetch_account_balance` puede conservar su firma y devolver la cartera del mismo helper. Bootstrap y refresh del host consumen la respuesta tipada de verdad. No hay una API nueva aislada que sólo invoquen tests.
4. **Modo y unidad explícitos.** La primera versión debe delimitar USD-M single-asset USDT. Un modo multi-assets desconocido o no soportado no se etiqueta como USDT: se rechaza para esta admisión hasta diseñar agregación/conversión. No sustituir por `availableBalance` o `maxWithdrawAmount`.
5. **UDS sin defaults que fabriquen capital.** `wb` debe estar presente. Las filas de posición usadas para construir PnL requieren sus campos de cantidad y PnL; no interpretar una omisión como cierre o PnL cero. Validar todo el evento antes de mutar el cache. Si la suma desborda o no se dispone del cache consistente, rechazar la observación en vez de sustituirla por wallet. Eliminar el `.max(0.0)` si se conserva el valor negativo para diagnóstico; nunca convertir una equity no positiva en ausencia.
6. **Cache de posiciones: precondición real.** Binance sólo envía posiciones cambiadas. `P:[]` no prueba por sí solo que la cuenta esté flat. El positivo `wb=0/P[]` del contrato debe partir de un snapshot flat declarado válido. La inicialización/recuperación debe sembrar el cache completo de esa sesión con la respuesta account/posición ya consultada, o mantener desconocida la estimación UDS hasta completar esa base. Un cache recién vacío tras reconnect no equivale a posiciones cero. Esto es necesario para no fabricar una insolvencia cuando wallet es cero pero existe PnL positivo no incluido en ese delta.
7. **Respuesta REST tardía.** Capturar el token y el `Arc` del estado de capital antes de la espera. Antes de publicar, comprobar el token y que el book sigue siendo el esperado; si un UDS ya lo cambió, descartar esa publicación o volver a consultar. Así una respuesta iniciada antes de un cero UDS no lo tapa con un saldo anterior. Esto ordena productores locales; no promete un snapshot atómico del exchange entre REST y WS.
8. **Error de refresh visible a la autorización real.** Extender el `apply_snapshot` productivo para recibir también `Result<AccountCapitalSnapshot, String>`, procedente de `refresh`, y exigir éxito vigente antes de Ready. Un `Err` mantiene la pausa/readiness bloqueada; no se disfraza con la última observación válida. Esta última puede conservarse como dato defensivo/diagnóstico, con su procedencia, pero no sirve para certificar ese refresh. Así se cierra el `if let Ok` silencioso sin inventar un TTL de antigüedad. Un error de token viejo sigue sin revocar la sesión nueva. Ajustar los fixtures existentes de `apply_snapshot` para aportar evidencia explícita de capital, no un default escondido.

La ampliación tipada REST evita un problema concreto: wallet 10 no es prueba de equity positiva si el PnL abierto es -20. Si se decide posponerla, hay que reducir el alcance del primer átomo y declarar que sólo propaga los ceros de UDS; no presentarlo como resolución de bootstrap/refresh de equity. La recomendación de este diseño es incluirla para cerrar el camino coherentemente, sin tocar sizing ni fees.

## Lectura del freno y sincronización de admisión

Propuesta de retorno de `capital_del_freno`: `Option<f64>`. En escenario sin cuenta devuelve `Some(capital_arena)` si ese capital es finito; en cuenta adjunta desconocida devuelve `None`; con observación finita devuelve `Some(min(capital_arena, equity))` si el capital local también es finito. Un capital local no finito tampoco se rescata mediante `f64::min`. Los tres consumidores existentes deben manejar esa ausencia explícitamente: host, riesgo y cálculo de drawdown del consejo en el core. En este último, una desaparición/cambio de evidencia después de evaluar riesgo debe vetar la propuesta; no convertir `None` a un drawdown de cero.

En riesgo, `None` rechaza antes de dimensionar; `Some(0)` y `Some(negativo)` alimentan el freno vigente y rechazan por drawdown. En el host, capital desconocido no actualiza pico ni se convierte en una insolvencia inventada; debe bloquear nuevas entradas y continuar atendiendo cerrojo del operador, latencia y otras defensas. **No añadir un `continue` que salte esas defensas.** El cero conocido sí alcanza `insolvent` y el kill-switch ya previsto.

Para el contrato acotado UDS → nueva evaluación de riesgo, el único snapshot y el guard de sesión bastan para evitar una mezcla de presencia/valor. Para afirmar además que una propuesta calculada antes del cero no puede publicarse después, es necesario el gate real: publicación del capital y apertura/debit del core deben compartir el gate y el core debe releer capital justo antes de publicar. La API del publicador debe disponer del arena una vez enlazado; no tomar el mutex de sesión desde el core mientras mantiene ese gate. Mantener siempre sesión → gate; los lectores de riesgo no toman el mutex de sesión. No mantener ninguno durante red ni callbacks que reentren a sesión.

La implementación puede conservar el stream inicial previo al arena para acumular evidencia bajo sesión; hasta enlazar el arena no hay publicación de entradas. Una vez enlazado, el puente usa el gate del arena para el store. Esta unión y ambos constructores deben probarse; no basta que el test use un bridge distinto o un setter de test. Las peticiones admitidas antes de la revocación, y los fills ya ocurridos en Binance, siguen requiriendo su liquidación/reconciliación existente.

## Contratos de aceptación, sin red ni modelo nuevo

Extender fixtures existentes, sin debilitar los asserts ni el golden:

| Contrato | Camino real y negativo falsable |
|---|---|
| UDS → bridge → freno → riesgo | En `tests/host_oms_contract.rs`, crear el bridge productivo, el streamer real con `with_sink`, estado completo flat y arena 100. Invocar `dispatch` con bytes `ACCOUNT_UPDATE`, no llamar manualmente al setter. Tras `wb=0/P[]`, exigir observación cero, capital del freno cero, `freno_duro(...,100,0).echado()` y orden sin admisión/margen nuevo en `RiskEngine::evaluate_quantum_order`. |
| Equity cero/negativa | Sembrar posición válida y enviar wallet 10 con PnL -10/-20. Exigir respectivamente 0/-10 y defensa; la cartera continúa siendo 10, sin autorizar wallet negativo. |
| Ausente no es cero | Tras positivo válido, enviar `wb` ausente, null, NaN, Inf, overflow y campo de posición requerido ausente. No publicar falso cero ni cambiar el cache parcialmente. Un fallo inicial mantiene desconocido; uno posterior no se etiqueta como observación fresca. |
| Backtest compatible | Adaptar `cf5c_pico_plano_y_venue`: la raíz ausente conserva 14.04 de arena; una cuenta observada a 13 conserva el freno 13; cero observado ya no es la operación usada para representar ausencia. Intentar construir observación NaN falla sin reemplazar la anterior. |
| Identidad | Ampliar `balance_publication_requires_current_executor...`: reemplazar sesión y entregar éxito/error/evento viejos; el book y la admisión nueva quedan intactos. La sesión nueva arranca desconocida, nunca con los 100 de la anterior. Una respuesta válida actual permite el camino normal sujeto a las otras guardas. |
| REST tardío | Capturar token/book para respuesta 100; publicar cero por UDS; entregar la respuesta capturada. Debe ser descartada por cambio de snapshot aunque el ejecutor siga siendo el mismo. |
| Deltas y cache | Snapshot completo con dos posiciones; delta de una conserva la otra. `P:[]` con cache desconocido no demuestra flat; reconnect invalida la estimación hasta reseed. Se conserva el test de `reconcile_after_reconnect`. |
| Publicación tardía, si se incluye la garantía fuerte | Fixture del gate de `host_oms_contract`: detener propuesta ya calculada antes de debit/open, publicar cero con el publicador real y liberar. Ningún débito/posición nuevo; tampoco segundo envío de transporte. Reutilizar el seam/barrier ya existente, sin sleeps para acertar una carrera. |

`account_update_routes_to_sink` en `user_data_stream.rs:906` aporta el control positivo 105.25 + 1.5 = 106.75. `cf5c_pico_plano_y_venue.rs:52` y `host_oms_contract.rs:438/470/550` aportan los controles de plano, sesión y gate. El writer directo positivo del test `f1_047_049_050...:136` se migra a observación válida, preservando su aserción original. Los tests de account tipado se conectan a `execution_evidence_contract.rs`, que ya forma parte de la suite real de execution.

No se ejecutó ninguna de estas pruebas al redactar. Tras implementar y revisar, exigir CI del candidato compuesto y del push real de main, recursos y contratos previos conservados. No crear un canal de certificación que sólo compile sin ejecutar el contrato nuevo.

## Reserva sugerida y límites

Fuentes necesarias: `quantum-arena/src/state.rs` (y reexport mínimo si hace falta), `risk-engine/src/drawdown.rs`, sus tres consumidores (`risk-engine/src/lib.rs`, `src/bin/god_engine.rs` y `god-engine-core/src/lib.rs:9622–9626`), `src/host_oms.rs`, `execution-engine/src/user_data_stream.rs`, `execution_evidence.rs` y `executor.rs`. La garantía fuerte del gate tardío amplía la reserva del core al bloque 10130–10145; acordar ambos bloques antes de editar. Tests: `host_oms_contract.rs`, `execution_evidence_contract.rs`, `cf5c_pico_plano_y_venue.rs`, el writer/lector directo positivo de core y los unit tests UDS existentes. No cambiar dependencias Cargo: ArcSwap ya está disponible.

El cambio del campo público de `GlobalArena` requiere revisar sus constructores y los accesos listados; no afirmar compatibilidad ABI externa de un tipo `repr(C)` tras modificar su layout. No se encontró un consumidor externo con el grep Rust; eso no demuestra ausencia universal de consumidores fuera del repositorio. No mezclar esta revisión con la corrección de los exports FFI de backtest.

La identidad propuesta acredita el ejecutor/generación local y su entorno configurado. No identifica mágicamente una cuenta Binance distinta por nombre de usuario ni repara IDs persistentes, migración de posiciones, pico guardado por cuenta o arbitraje entre fuentes remotas. Los métodos `hot_swap_credentials`/`set_testnet` existen, pero el grep actual sólo encontró sus definiciones y un test de cliente; si se añaden callers productivos deben pasar por la misma frontera de generación. No suponer que mutar credenciales dentro del mismo `Arc` conserva la prueba de identidad.

Fuentes primarias ya consultadas: [Account REST USD-M](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/account) diferencia wallet, margin balance, disponible y retirable, y distingue USD agregado de multi-assets frente a USDT single-asset; [User Data Streams](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/user-data-streams) documenta los deltas de posiciones y eventos de funding sin `P`. No se hizo consulta autenticada para este diseño.

El fallback FFI a 13 USD es una prioridad separada. Este documento no implementa ni certifica capital cero, paridad financiera completa, rendimiento, rentabilidad ni autorización para desplegar u operar fondos.
