# Revisión independiente de causalidad — hidden_liquidity

Fecha: 2026-10-15. Revisor: Codex, subagente quant_foundations. Alcance: lectura del candidato externo de ROOT y sus contratos; ninguna edición de código, Cargo, rustc ni ejecución de pruebas Rust. Este documento revisa la causa deduplicada de liquidez oculta con reserva ROOT. B15-16 es un identificador propuesto, aún pendiente de incorporarse al ledger BASE; no abre un censo paralelo.

**Dictamen:** el candidato es una reparación numérica de la neutralidad y del espejo con una política interpolada explícita. No constituye flujo firmado observado ni acredita detección de un muro pasivo oculto. No encuentro una razón para bloquear esa reparación acotada por exigir simultáneamente un cambio de alfa o un nuevo transporte de datos. Sí debe conservarse la separación entre esa reparación y la identificación causal que todavía falta. No recomiendo invertir automáticamente el signo del alfa sin validar qué magnitud consume Flow.

Revisé los bytes bajo `C:/Users/jhona/.codex/tmp/hidden-symmetry-design-20261015/proposal-cd9-4daf9f1c-c6a0-4a4e-bc1e-aba1693cfe58/`, con base declarada `cd9feaa0aaa89e6b45d6363a23a4fbf365ea41d5`:

| Archivo propuesto | SHA256 leído por esta revisión |
| --- | --- |
| `proposed/crates/dark-alpha-engine/src/hidden_liquidity.rs` | `7c1720588b6b0be50eaa9331657869e9d9baaa1f45193b879055effb52ffd79b` |
| `proposed/crates/dark-alpha-engine/tests/hidden_liquidity_symmetry_contract.rs` | `ad26922c4fd87b86acc315f41d59ee7852eecd8ed81746ccdb4ad8b1c47c87b7` |

Corte histórico: esta revisión leyó V1 y sus 13 tests declarados. ROOT rechazó después su fastpath por cuantización de amortiguamiento subnormal y preparó V2; esta nota no acredita aceptación numérica ni ejecución Rust de V1 o V2. Las observaciones de política, procedencia y signo mantienen aquel alcance.

El módulo propuesto, líneas 228–250, utiliza `z = gamma*abs(delta)/ATR`, `t = tanh(z)*sign(delta)` y pesos `0.75 ± 0.25*t`. Para profundidades positivas, ratios representables y suma superior al soporte EPS de líneas 264–275, sean `p0=(Bid-Ask)/(Bid+Ask)` y `t=tanh(gamma*delta/ATR)`. La presión matemática es:

`Pi = (p0 + t/3) / (1 + p0*t/3)`.

Con libro simétrico resulta `Pi=t/3`; en delta cero vale exactamente cero. Intercambiar Bid/Ask y negar delta niega la presión. La función de presión es suave en este dominio, incluida su derivada en cero. El factor común de amortiguación se cancela al formar el cociente. Esto no hace suaves los ratios individuales, que conservan `exp(-gamma*abs(delta)/ATR)`, ni los cambios por EPS, abstención, actividad, bandas o vetos discretos. El README del candidato ya delimita esas exclusiones; mantenerlas en el recibo final evita certificar C1 global.

Los pesos suman 1.5. Son ponderaciones de un score heredado; no deben interpretarse como volúmenes ejecutados por lado, pues con esa interpretación sumarían `1.5*V`. Los valores 0.75 y 0.25 proceden de interpolar la política anterior `(1,0.5)/(0.5,1)`, no de una estimación de Kyle. Además, gamma pasa a controlar tanto el amortiguamiento como la respuesta direccional. Por ejemplo, gamma cero elimina esta última para cualquier delta. Ese acoplamiento es una elección de política, no una consecuencia de la simetría.

El test propuesto de líneas 257–270 reconoce un cambio material de activación: para V=150, Bid=Ask=100 y delta=0 los ratios pasan a 1.125/1.125 y quedan bajo theta=1.25. No es una mera refactorización con paridad económica. Declararlo expresamente, como hace ROOT, es correcto; no corresponde retocar theta para ocultar esa diferencia.

**Qué signo está observado.** El feed permite distinguir comprador y vendedor agresor. La documentación oficial de Binance define `m` como si el comprador es maker. Por tanto, `m=false` describe compra agresora contra una oferta pasiva; `m=true`, venta agresora contra una compra pasiva. Es la misma semántica documentada en `crates/feature-engine/src/microstructure.rs:54–62` y usada en `crates/god-engine-core/src/lib.rs:3499–3525`. [Binance, documentación oficial del stream](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-streams.md#aggregate-trade-streams), [Binance, SDK oficial de futuros USD-M](https://github.com/binance/binance-connector-python/blob/master/clients/derivatives_trading_usds_futures/src/binance_sdk_derivatives_trading_usds_futures/websocket_streams/websocket_streams.py).

| Evento observado | Agresor | Contraparte pasiva ejecutada | Q del agresor |
| --- | --- | --- | --- |
| `m=false` | Comprador | Vendedor en Ask | `+V` |
| `m=true` | Vendedor | Comprador en Bid | `-V` |

Una reserva iceberg es una orden límite con cantidad parcialmente visible y cantidad restante oculta que puede renovar su exposición. Esa mecánica describe al proveedor pasivo; no identifica por sí sola su inventario restante después del fill. [Nasdaq, Market Model, sección 6.2.3, página 66](https://www.nasdaq.com/docs/2022/03/27/NEQD_Market_Model_3.0.0.pdf).

De ello se desprenden dos convenciones diferentes. `ask_exec - bid_exec` orienta flujo agresor; `bid_abs - ask_abs` orientaría soporte del comprador pasivo frente al vendedor pasivo. Si se pretende representar absorción pasiva, llamar soporte alcista a Ask ejecutado confunde ambos lados. Sin embargo, tampoco se deduce que toda compra agresora absorbida deba producir una señal Short futura. Dos estados con el mismo volumen, libro y delta pueden diferir en si la última reserva acaba de agotarse o todavía conserva cantidad. Esta es una inferencia de no identificabilidad a partir de los datos disponibles, no una afirmación de rentabilidad ni una regla contraria nueva.

El candidato aclara que no observa el agresor en líneas 24–26 y 246–247, pero el encabezado de líneas 3–19 y los comentarios del gate en 316–340 siguen hablando de institucionales, acumulación/distribución y compuerta conformal. Conviene alinear esos textos con el alcance real. El nombre público puede preservarse por compatibilidad; documentar que devuelve un proxy de score no requiere romper su API. El multiplicador de banda y el enum no aportan por sí solos alpha, calibración o cobertura conformal.

**Conexión productiva que el candidato no repara.** En el árbol principal leído:

- `GodEngineCore::process_event`, líneas 2448, 2452 y 2465, recibe `is_trade`, `trade_qty` e `is_buyer_maker`. El dato de agresor sí existe en esta frontera. Una publicación de delta plano con `is_buyer_maker=false` tendría agresión compradora observada; no equivaldría a inventarla por el signo de cero.
- `process_event:3486–3491` escribe la cantidad del evento y la pone en cero cuando no es trade. Pero `process_tick_dual:3704–3714` no recibe ese contexto: su caller del detector, líneas 5546–5552, consulta un campo de estado. Los callers directos del método público pueden conservar una cantidad anterior. No atribuyo ese riesgo a todo evento productivo: el camino normal por process_event sí limpia el campo.
- La misma llamada pasa `dir_v`. `stateful_engine.rs:89–95,837` lo define y calcula como EWMA del paso del mid; no es el desplazamiento actual. Capturar `mid_price - feature_engine.last_price` antes de `process_tick` en `lib.rs:3903` permite usar el desplazamiento actual antes de sobrescribir el precio en `stateful_engine.rs:1005`. `v_t` es un rango de 60 segundos en precio, según `stateful_engine.rs:76–79,972–992`.
- `process_event:2720–2728` puede sustituir profundidades ausentes por `trade_qty.max(0.01)`. Que el nuevo módulo rechace profundidad no positiva no prueba que su entrada productiva sea profundidad observada: el caller ya pudo enmascararla.
- `flow_excitation_confluence.rs:313,363–378` utiliza presión también sin actividad iceberg. Por tanto, un futuro API firmado con Q=±V no debe conectarse sin revisar ese uso: una presión de flujo ±1 no acredita por sí sola un bloque activo. El candidato actual conserva ese comportamiento explícitamente; no atribuirle una solución de este contrato.

La continuación conservadora para eliminar la inferencia de lado desde delta es separar datos observados de política: conservar el API unsigned con Q=0 declarado como ausencia de información sobre el agresor, y añadir un API firmado con `Vask=0.5*V+0.5*Q`, `Vbid=0.5*V-0.5*Q`. Si el API unsigned conserva presión por asimetría de profundidades, esa magnitud es un proxy del libro, no flujo firmado observado. Q debe provenir únicamente del flag actual, con V finito no negativo y `abs(Q)<=V`; no del tick rule ni de una EWMA. Un worker privado que reciba `Option<(V,Q)>` permite conservar la firma pública de process_tick_dual: process_event entrega el par fresco y los callers sin contexto entregan None. Profundidad ausente debe seguir siendo ausencia para este detector. Esta continuación no exige cambiar automáticamente la orientación predictiva del consumidor: primero hay que nombrar si el nuevo campo describe agresor o pasivo.

Mi recomendación de integración es permitir que ROOT cierre su reparación numérica con sus propios checks y su alcance explícito; no presentarla como implementación del transporte firmado observado. Si el criterio de cierre del humano exige eliminar la heurística causal, la continuación firmada es un requisito separado y concreto, no un motivo para reemplazar una heurística por otra inversión de signo sin validación. Mantener las dos entregas identificables permite revisar cada cambio sin silenciar a ROOT ni certificar una garantía que sus datos no aportan.

El archivo de contratos contiene 13 funciones de prueba declaradas y recibos PREPARED. Esta revisión examinó sus oráculos y anclas; no ejecutó ni certificó su compilación, sus resultados, admisión en Core, órdenes reales o rendimiento financiero. Los tests de continuidad/monotonía propuestos fijan la política interpolada elegida; no identifican al agresor ni prueban causalidad de un muro oculto.
