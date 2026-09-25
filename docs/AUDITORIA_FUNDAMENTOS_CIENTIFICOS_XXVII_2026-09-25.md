# Auditoría XXVII — alcance productivo, capacidades reales y evidencia de ejecución

Fecha: 2026-09-25. Continuación de [XXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVI_2026-09-25.md>). [Artefacto estructurado XXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVII_2026-09-25.json>).

## 1. Dictamen ejecutivo y alcance honesto

La principal conclusión es topológica: el host god_engine no usa execute_order para esta ruta de entrada. Toma una cantidad previamente creada por el núcleo y llama directamente a las APIs raw/maker/iceberg. Por tanto, la mejora de build_payload/execute_order de XXVI no demuestra que todas sus invariantes alcancen al camino vivo. XXVI ya limitaba su alcance; esta ronda conecta explícitamente el análisis con el consumidor del host.

Se incorpora un despachador común invocado por el host y comprobable con transporte simulado. Valida determinados dominios antes de efectos de cuenta, conserva campos con unidades diferenciadas e identidad de intención, exige configuración de leverage incluso para 1× y no envía una entrada si esa configuración no queda confirmada. El adaptador real comprueba símbolo y valor de la respuesta de leverage, en lugar de aceptar cualquier cuerpo con HTTP exitoso.

Se contiene también una capacidad no respaldada: el adaptador USD-M deja de declarar éxito paper o enviar icebergQty al endpoint /fapi/v1/order. La API se conserva, pero devuelve un motivo explícito de capacidad no soportada. No se sustituye silenciosamente por MARKET, una orden visible completa ni una secuencia de órdenes hijas no auditada. Esto reduce el conjunto de acciones admitidas por ese adaptador de forma deliberada y documentada.

No se cierra la proyección monetaria del riesgo. FMT-113 sigue abierto: el host puede cambiar leverage sin reducir la cantidad enviada. Tampoco se cierran los tres testigos de lotes/valoración de XXVI. Se documenta además FMT-222: una consulta fallida de posiciones en una rama de reconciliación puede terminar en exchange_confirmed=true. No se ha ejercitado esa rama contra una cuenta ni atribuido pérdidas reales.

Cobertura acumulada: 143/289 archivos Rust preexistentes con lectura completa registrada; quedan 146. Se añade la lectura completa de execution-engine/src/lib.rs. El módulo nuevo entry_dispatch.rs y su suite se revisaron completos, pero no inflan el denominador ni cuentan como lecturas de archivos preexistentes. god_engine.rs y executor.rs siguen revisados por tramos, no completos. El inventario base sigue siendo 1.119 archivos y 24 manifiestos; no equivale a revisión semántica de todos ellos.

## 2. Grafo vivo observado: raíz, decisión y terminal no eran una sola cadena

```text
RAÍZ: observaciones/estado/genoma
  └─ núcleo y RiskEngine → posición local provisional + new_order(q)
       └─ host: RiskEnvelope/vol_brake → exec_leverage
            └─ adaptación por margen → effective_leverage
                 └─ filtros e identidad de intención
                      └─ dispatch_entry [NUEVO, conectado al host]
                           ├─ dominios y capacidad → rechazo explícito
                           ├─ configurar L, también 1× → confirmar símbolo/L
                           └─ enviar raw / maker / iceberg habilitado por adapter
                                └─ ACK/Unknown → reconciliación/protección

Ruta API paralela: ValidatedOrder → execute_order → build_payload
  └─ fortalecida en XXVI; no era la ruta de entrada anterior del host

Auxiliar QuantumOrderRouter
  └─ definición/reexportación/tests localizados; no caller vivo localizado
```

Esta distinción evita tres inferencias incorrectas: que un crate usado en producción ejecuta todos sus helpers; que un test de un helper cubre automáticamente el consumidor real; y que un nombre de tipo «Validated» acredita invariantes posteriores. El grafo es un mapa diagnóstico de llamadas leídas, no prueba de ejecución de todas las ramas en un proceso desplegado.

El host mantiene force_maker=false en este tramo. La rama iceberg depende de nocional y umbral; corregirla no significa que se haya observado activación real. El router auxiliar conserva deudas FMT-103/104, pero no se lo confunde con el nuevo despachador conectado al host.

## 3. Matriz de hallazgos y estado

| ID | Prioridad y alcance | Resultado verificable | Pendiente |
|---|---|---|---|
| FMT-220 | P1, rama iceberg del host/adaptador | Capacidad no documentada bloqueada antes de envío; precio/cantidad visible pasan a campos nombrados; desaparece ID fijo del host | Implementación real de órdenes hijas/capacidad respaldada, sin fallback implícito |
| FMT-221 | P1, configuración previa a entrada | Se exige configurar también 1×; un fallo impide enviar; respuesta verifica símbolo y L | Concurrencia por cuenta/símbolo, tiers, incertidumbre de configuración y latencia |
| FMT-222 | P1, reconciliación del host | Lectura estática demuestra Unknown→true→exchange_confirmed | Abierto; requiere modelo de evidencia e identidad, no sólo otro booleano |
| FMT-218 | P1, admisión numérica, ampliación XXVI | raw_qty ahora rechaza NaN producido por redondeo; paper no lo oculta | Otras rutas, malla completa y diagnóstico tipado global |
| FMT-113 | P1, exposición y genoma, antecedente V | Se verifica que el host calcula L sobre q ya fijada y puede elevar su techo por margen | Presupuesto monetario y proyección final conectada al ledger |
| FMT-175/219 | P1, lote y valoración, antecedentes XVIII/XXVI | Los tres diagnósticos anteriores siguen reproduciendo deuda | Mínimo post-lote, cantidad conservadora y nocional al precio final |

No se renumera ni reemplaza la matriz histórica de 305 puntos. Se añaden tres IDs a la serie FMT y se conservan los estados históricos mediante adendas. P1 describe severidad potencial bajo activación de la ruta, no evidencia de una pérdida ya ocurrida.

## 4. FMT-220 — capacidad iceberg, unidades y trazabilidad

### 4.1 Intercambio de argumentos de la misma representación numérica

Antes, la llamada del host enviaba los argumentos quantity, maker_price, iceberg_qty a una función cuyo contrato espera quantity, iceberg_qty, price. Los dos últimos son f64, por lo que el compilador no detecta la permutación. Si q=2 unidades, precio=100 unidades de cotización/base y visible=0,2 unidades, el caller podía pedir un iceberg de 100 unidades con precio 0,2, en vez de visible 0,2 y precio 100. El efecto exacto dependía de filtros y respuesta remota; no se afirma que llegara a llenarse.

El nuevo EntryRoute::Iceberg usa price y visible_quantity como campos nombrados. El adaptador delega a execute_iceberg_limit respetando la posición de ambos parámetros. El contrato local verifica visible_quantity finita, positiva y no mayor que quantity cuando el transporte declara esa capacidad. No se prueba aquí una ejecución iceberg real, porque el adaptador USD-M no la declara disponible.

Evidencia actual: [tipo de ruta](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/entry_dispatch.rs:10>), [adaptación de campos](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/entry_dispatch.rs:148>), [construcción del host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4122>).

### 4.2 Un límite público no se convierte en iceberg porque se añada un parámetro

El contrato consultado de [Nueva orden USD-M](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#new-order) no incluye icebergQty entre los parámetros de /fapi/v1/order. El código enviaba ese parámetro y describía ocultación de volumen como si la capacidad estuviera acreditada. Además, paper devolvía Ok antes de comprobarla. Que otra sección o producto contenga un campo icebergQuantity no demuestra soporte en este endpoint.

Se declara SUPPORTS_NATIVE_ICEBERG=false para este adaptador. La API heredada permanece, incluido su código histórico de serialización detrás de la guarda, pero no se considera operativo ni certificado. El rechazo se produce antes del éxito paper, firma o envío. En el despachador, se produce también antes de cambiar leverage. No se realizó una petición real para averiguar cómo responde el exchange a un parámetro no documentado.

La semántica del veto es «capacidad no respaldada por este adaptador», no «alpha insuficiente» ni «no puede existir una estrategia iceberg». Una implementación algorítmica requeriría órdenes hijas, IDs estables, control de fills parciales, cancelación, cola, presupuesto agregado, protección del remanente y recuperación. Enviar una orden visible en silencio alteraría la intención de ejecución y la exposición informacional; no es una reparación equivalente.

Evidencia: [declaración de capacidad](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/binance_api.rs:19>), [guarda de API](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:2665>), [prueba paper](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:27>), [ausencia de efectos en transporte simulado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:168>).

### 4.3 Identidad de intención: no repetir iceberg_01

La rama usaba el literal iceberg_01 para múltiples intenciones. El contrato oficial requiere unicidad entre órdenes abiertas; un identificador fijo no conserva esa condición al solapar intenciones. La reparación genera un ID por nueva intención, de 35 caracteres, con dirección y UUIDv7, y lo transporta por los campos del mismo request. La prueba genera 1.000 IDs sin colisiones observadas y comprueba formato y longitud; no es una prueba matemática de ausencia universal de colisiones.

El mismo ID debe conservarse al reconciliar o reintentar una intención. Volver a llamar al generador para un reintento no es idempotencia. XXVII no demuestra exactamente-una-vez ni completa persistencia/reanudación de IDs. Ver [generador](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/entry_dispatch.rs:107>) y [regresión](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:324>).

## 5. FMT-221 — configurar no es confirmar y 1× no significa no hacer nada

### 5.1 Tres rupturas independientes

Primera: el host sólo invocaba set_leverage si effective_leverage>1. Si la cuenta ya tenía un valor distinto, una intención de 1× no lo restablecía. El razonamiento local sobre financiación podía usar un estado que no coincidiera con el exchange.

Segunda: cuando invocaba la función, descartaba Result mediante let _ y continuaba con la orden. Un fallo HTTP, de red o del limitador local no detenía el envío. La ruta execute_order endurecida en XXVI sí manejaba ese error, pero el host no pasaba por ella.

Tercera: set_leverage devolvía Ok para cualquier cuerpo recibido con éxito HTTP. Se ignoraban símbolo y leverage efectivos de la respuesta. El transporte confirma que recibió una respuesta, no que ésta corresponde a la configuración solicitada.

### 5.2 Reparación y fronteras de evidencia

dispatch_entry valida entrada/capacidad, configura el leverage solicitado —incluido 1— y sólo después llama submit_entry. Un error de configuración se devuelve con ENTRY_LEVERAGE_UNCONFIRMED; el siguiente paso no se ejecuta. Los errores de envío se conservan sin reemplazar AMBIGUOUS por un rechazo definitivo. El host usa ese despachador antes de su lógica existente de resolución.

set_leverage valida símbolo no vacío y entero del dominio 1..125. En paper no toca red. En modo real, validate_leverage_confirmation exige JSON con symbol y leverage sin defaults y coincidencia exacta con la petición. Un cuerpo vacío, ilegible, otro símbolo, otro leverage o un valor fraccionario no demuestra confirmación y se rechaza. El [contrato oficial de cambio de leverage](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#change-initial-leverage) contiene ambos campos y además maxNotionalValue.

El resultado no garantiza capacidad de cuenta. maxNotionalValue no se consume todavía como techo de esta orden; no se evaluaron aquí tiers, liquidación o margen cruzado. Ante timeout o ACK ilegible, el cambio de configuración puede haberse aplicado: la reparación impide enviar esta entrada, pero no afirma que la cuenta conserve el leverage anterior. El error describe falta de confirmación, no prueba de ausencia de efecto remoto.

Evidencia: [secuencia común](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/entry_dispatch.rs:94>), [parser de confirmación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/binance_api.rs:23>), [host conectado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4140>), [fallo sin envío en las tres rutas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:120>), [validación de ACK](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:297>).

### 5.3 Concurrencia y latencia todavía abiertas

El host captura un Arc del ejecutor antes de consultar filtros y conserva ese mismo objeto para configurar/despachar la entrada. Evita cambiar de instancia entre esas operaciones por una nueva carga del ArcSwap, pero no existe exclusión por cuenta/símbolo entre múltiples despachos. Otro actor podría cambiar leverage entre confirmación y envío; la prueba de orden secuencial no es una transacción distribuida. Falta identidad de cuenta/entorno y epoch de configuración para compartir una caché con garantías. La reconciliación posterior todavía vuelve a cargar el ejecutor y no queda cubierta por esta retención local.

Confirmar 1× añade una operación que antes se omitía. Tiene coste de latencia y rate limit. No se elimina por conveniencia: primero debe definirse cuándo una confirmación previa sigue siendo válida. No se midieron p50/p99 ni rendimiento. Tampoco se demostró que construir un plan tipado sea por sí mismo más rápido.

## 6. FMT-218 ampliado — la ruta raw podía declarar éxito con cantidad NaN

Con q=0,4 y step=f64::from_bits(1), el paso es positivo, pero su inverso desborda. El redondeo basado en multiplicación por el inverso y división puede producir NaN. La condición final_quantity==0 es falsa para NaN; por eso la ruta paper imprimía «raw_qty NaN» y devolvía Ok. Es una reproducción directa del consumidor que usa el host, no sólo del constructor alternativo de XXVI.

Se exige finitud y positividad del resultado en execute_raw_qty_with_client_id. El despachador también comprueba que inverso y cantidad escalada sean representables antes de configurar la cuenta. Una cantidad no representable no es oportunidad negada por política estratégica: es una acción que el adaptador no puede expresar de forma válida.

Se conservaron los algoritmos de redondeo compartidos; no se cambiaron a ciegas los usados por cierres/protección. Por tanto, la reparación no garantiza que q_final≤q_autorizada, no revalida mínimo nocional y no constituye aritmética decimal exacta. Los diagnósticos FMT-175/219 de XXVI siguen pasando porque siguen abiertos.

Evidencia: [guarda raw](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:2182>), [reproducción rojo→verde](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/entry_route_contract.rs:18>), [preflight](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/entry_dispatch.rs:30>).

## 7. FMT-222 — conservar un estado incierto no acredita confirmación

### Mecanismo observado en lectura estática

En la rama de entrada AMBIGUOUS o MAKER_CHASE_UNVERIFIED, el host consulta fetch_open_positions. Si encuentra un símbolo, adopted=true; si no lo encuentra, false. Si la consulta falla, también devuelve true para evitar un rollback potencialmente peligroso. Después, todo adopted=true conduce a exchange_confirmed.store(true).

La intención de preservar estado bajo incertidumbre es razonable; el error es reutilizar el booleano para acreditar existencia en el exchange. «No tengo evidencia suficiente para borrar» y «he confirmado una posición» son proposiciones diferentes. Este defecto puede contaminar posteriores decisiones contables, de protección o aprendizaje que interpreten exchange_confirmed como hecho observado.

Evidencia: [consulta y clasificación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4161>) y [confirmación posterior](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4198>). No se ejecutó un timeout real ni se creó una posición para demostrarlo; es una contradicción de control de flujo y significado comprobada en código.

### Problemas de identidad y tiempo asociados

Encontrar una posición por símbolo no demuestra que pertenezca a esta intención, al mismo lado o a sus fills. La ausencia de posición en una consulta tampoco prueba necesariamente que nunca existió una orden: podría estar abierta sin fill o la observación no resolver el resultado causal de una solicitud. El comentario «la orden jamás aterrizó» es más fuerte que la evidencia disponible. Deben separarse estado de orden, estado de posición y evidencia del intento.

La función rollback_positions recorre los slots abiertos del símbolo y los cierra localmente, devolviendo margen/fee según su estado. No selecciona explícitamente un reservation_id de esta intención. Es una superficie de riesgo bajo concurrencia o reutilización de slots; no se ha demostrado aquí que cierre realmente otra posición en el proceso vivo. No se modificó este contrato sin auditar todos sus lectores/escritores.

### Cierre requerido, no implementado en XXVII

Hace falta al menos distinguir confirmación vinculada a intención/fills, ausencia acreditada con semántica de orden y desconocido. Preservar localmente debe ser una decisión independiente de confirmar o contabilizar. Identidad de instrumento, posición/lado, orden, cuenta, entorno y versión deben conservarse; rollback debe afectar sólo la reserva/estado del intento correspondiente. Se requieren pruebas de timeout antes/después de aceptación, limit aceptada sin fill, fill parcial, consulta fallida, posición previa del mismo símbolo y rotación/reutilización de slot.

El nuevo despachador conserva errores de envío ambiguos sin reetiquetarlos; eso no repara la lógica posterior descrita. FMT-222 queda P1 abierto, con estas condiciones explícitas de cierre.

## 8. FMT-113 revisitado — el gen puede cambiar financiación sin cambiar riesgo

La evidencia de la ronda V sigue vigente: el host toma q de new_order y calcula N=|q|P. La envolvente y el freno de volatilidad influyen sobre exec_leverage, no proyectan directamente esa q en el tramo revisado. Después, la adaptación de margen puede aumentar effective_leverage por encima de exec_leverage. XXVII hace más fiel la configuración solicitada; no transforma ese cálculo en control correcto de exposición.

Para entender el error hay que separar tres magnitudes:

| Magnitud | Expresión simplificada | Unidad/interpretación |
|---|---|---|
| Nocional | N=|q|P | Moneda de cotización |
| Margen nominal | M=N/L | Moneda de colateral bajo supuestos simplificados |
| Pérdida al stop antes de gaps | R≈N·d+C_costes | Moneda de riesgo bajo un modelo de ejecución |

Si q, P, d y el coste asumido no cambian, modificar L cambia M pero no R en ese modelo. Por tanto, ∂R/∂L=0 a exposición fija: reducir leverage no equivale automáticamente a reducir pérdida nominal de esa posición. La liquidación, restricciones y financiación pueden cambiar; no se deduce invariancia de todos los riesgos reales.

Contraejemplo ya documentado en V: con N=100 y margen libre=100, L=1 exige 100 y supera el disparador 85%. La adaptación calcula ceil(100/80)=2, cuyo margen 50 pasa el guard 95%. Si d=2%, la pérdida nominal sigue siendo 2 antes de costes; no satisface un presupuesto de 1 porque el leverage haya cambiado. Es una refutación algebraica de una garantía universal, no resultado empírico de una cuenta.

Evidencia actual: [derivación de leverage](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3887>), [nocional de q preexistente](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4023>), [adaptación posterior](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4075>). La bootstrap con n<30 que selecciona L=1 también permanece; no se interpreta como demostración de riesgo mínimo ni como ley de exploración segura.

La corrección sistémica debe proyectar cantidad y acción completa antes de reservar/contabilizar, y resolver cómo actualizar o revertir la posición provisional si cambia q. Aplicar sólo otra fórmula en el ejecutor puede desincronizar ledger, fees y brackets. No se presenta EntryRequest como ese ExecutionPlan monetario: su documentación lo excluye expresamente.

## 9. Auditoría de razonamiento, continuidad y teorías avanzadas

### 9.1 El objetivo no son dos motores ni otra colección de regímenes duros

Se mantiene el objetivo de un estado multiactivo, multivariante y temporal-espectral. Ni scalping/swing ni una clasificación global de volatilidad deben confundirse con un particionado natural exhaustivo del fenómeno. Persisten en el sistema lectores acotados, anclas, clamps y decisiones globales descritos en XXV/XXVI. Esta ronda no afirma haberlos eliminado al crear un despachador.

El tipo de orden y la capacidad del endpoint sí son variables de contrato del actuador. Diferenciar MARKET de una orden limitada con precio, o reconocer ausencia de capacidad iceberg, no introduce un régimen de mercado discreto. Es necesario distinguir coordenadas de inferencia continua, estado inferido con incertidumbre y restricciones físicas/operativas de la acción.

### 9.2 Autoevolución: trazabilidad y sensibilidad, no sólo mutación

Un gen cambia el sistema sólo si su perturbación alcanza una acción o restricción relevante. El diagnóstico debe medir la cadena g→parámetros efectivos→predicción→acción→resultado con versiones, no sólo g→fitness de backtest. Si el tramo de host reduce una señal de riesgo a un leverage entero y luego mantiene q, regiones enteras del genotipo pueden tener efecto nulo sobre la pérdida proyectada o efecto discontinuo sólo por vetos.

Una sensibilidad local puede definirse sobre la acción efectiva a(g,x), manteniendo fijo un estado x reproducible. Las diferencias finitas deben registrar dominio factible y cambios de rama; un cociente numérico cerca de un salto no es un Jacobiano suave. Probar invariancia ante cambio de unidades, monotonicidad de presupuestos y sensibilidad causal sería más informativo que bautizar más parámetros como epigenéticos. No se ejecutó aquí un barrido genómico ni se atribuyó causalmente el gap de rentabilidad completo.

### 9.3 Uso de teoría y límites de la evidencia

Las mejoras aplicadas usan análisis dimensional, conservación de identidad, contratos de composición y separación de estados de conocimiento. No se añadieron ecuaciones de problemas del milenio ni formalismos cuánticos nominales. Una teoría propuesta debe especificar observable, unidad, hipótesis, dominio, estimador, prueba de refutación y coste computacional; una ecuación sofisticada no resuelve un argumento posicional intercambiado ni una confirmación fabricada.

El dominio formal 1 ns..100 años exige diferenciar resolución de timestamp, resolución de datos, soporte estimable y horizonte de decisión. No se promete recalcular todo cada nanosegundo ni conocer un futuro no observado. El soporte numérico de una función tampoco equivale a evidencia estadística en todos sus puntos. Ese contrato sigue pendiente, igual que la dependencia multiactivo con snapshots consistentes.

## 10. Vetos, rechazos e incertidumbre: matriz de significado

| Motivo | Objeto rechazado | Por qué tiene sentido | Qué no implica |
|---|---|---|---|
| ENTRY_INVALID_DOMAIN | Acción con datos básicos no representables | No se puede configurar/enviar de forma válida | Alpha negativo o horizonte no interesante |
| ENTRY_INVALID_ID | Identidad fuera del contrato del endpoint | Impide perder trazabilidad por un identificador inválido | Exactly-once o persistencia garantizada |
| UNSUPPORTED_NATIVE_ICEBERG | Una capacidad no respaldada por el adaptador | Evita éxito ficticio o cambio silencioso de intención | Prohibición universal de estrategias fragmentadas |
| ENTRY_LEVERAGE_UNCONFIRMED | Envío posterior a configuración no confirmada | La financiación asumida no está acreditada | Que el cambio remoto no se aplicó |
| AMBIGUOUS | Resultado desconocido del envío | Exige reconciliación, no reemisión ciega | Rechazo definitivo ni posición confirmada |
| Mínimo nocional/margen | Acción no factible bajo reglas y presupuesto | Puede justificar abstención | Permiso para aumentar exposición hasta pasar |

La nueva capacidad bloqueada debe registrarse como problema de implementación/compatibilidad, no penalizar al gen como si fuese un fracaso de predicción. Los vetos por información desconocida deben permitir que siga funcionando la gestión defensiva con la mejor evidencia disponible. Esta ronda sólo interviene entradas; no reemplaza el protocolo completo de salidas o protección.

## 11. Módulos del maestro y deuda conectada

| Módulo | Aporte XXVII | Pendiente sistémico |
|---|---|---|
| 1. Ingestión/L2 | Precio y filtros como datos externos con contrato | Snapshot, tiempo, identidad y profundidad coherentes |
| 2. IA/señales | Separar veto operativo de calidad estadística | Calibración y causalidad del impacto de cada predictor |
| 3. Multiactivo/horizontes | No confundir tipos de órdenes con regímenes estratégicos | Espectro y riesgo conjunto sin recortes silenciosos |
| 4. Ejecución/conectividad | Despacho conectado, capacidades y leverage confirmado | Reconciliación FMT-222, filtros completos, latencia y concurrencia |
| 5. Riesgo/genomas | FMT-113 conserva una cadena causal explícita | Proyección de q y pérdida monetaria hasta ledger |
| 6. Estado/telemetría | Identidad por intención y semántica de errores | Reserva atómica/versionada y rollback por intento |
| 7. Confluencia/cuántica | No se añade formalismo sin observable/contrato | Evidencia de utilidad y coste de los mecanismos existentes |
| 8. Backtest/gobernanza | Transporte falso y paper prueban secuencia sin cuentas | Paridad económica y protocolos completos bajo fallos |

## 12. Verificación realizada

Diez pruebas nuevas en entry_route_contract.rs. Dos se observaron fallar antes del cambio y pasar después: raw_qty con cantidad redondeada NaN y éxito paper iceberg no respaldado. Las otras ocho prueban contratos introducidos/reforzados; no se inventa evidencia rojo→verde previa para código que todavía no existía.

Se comprueban: fallo de configuración sin envío en tres rutas, configuración explícita de 1×, conservación de campos e identidad, capacidad no soportada sin efectos, entradas inválidas antes de mutación, conservación de AMBIGUOUS de envío, ACK de leverage, formato/unicidad muestral de IDs y ruta real del adaptador en paper. El mock registra el request nombrado y la secuencia de llamadas, no simula un libro ni certifica fills. La correspondencia posicional con la API heredada se inspeccionó en el adaptador; la rama iceberg real está bloqueada por capacidad.

Suites ejecutadas: risk-engine --lib --tests, 108 pases (100 funcionales y ocho diagnósticos abiertos); execution-engine, diez contratos nuevos, nueve de payload existentes y tres diagnósticos de payload; cinco tests preexistentes de redondeo/precio y dos de firma. Total distinto: 126 funcionales y 11 diagnósticos abiertos, 137 pases. No se suman repeticiones. Que los diagnósticos pasen confirma deuda, no cierre.

cargo check --offline pasó para god_engine, feature_exporter, train_forest y train_dark_alpha. Persisten los tres warnings preexistentes de evolution-engine (latest_ts, mode, trades). No se ejecutó toda la suite del workspace ni se repitió el test inestable de promoción FMT-216. No se generó un binario operativo ni se reinició el motor.

No hubo peticiones a cuentas, órdenes, mediciones de PnL ni benchmarks productivos. Las pruebas de red usaron sólo abstracción de transporte/paper; set_leverage en paper no realiza I/O. La prueba de HTTP del cambio real de leverage queda pendiente en un harness de transporte adecuado. Las lecturas web fueron documentación pública, no datos de usuario.

## 13. Hoja de ruta por dependencias

1. Cerrar FMT-222 con evidencia de orden/posición vinculada al intento y rollback por reserva; no convertir Unknown en confirmado.
2. Conectar ExposureBudget/plan equivalente al punto anterior a reserva local y despacho. Demostrar q/P/SL/costes coherentes en núcleo, host, replay, payload y ledger. Cerrar FMT-113/175/219.
3. Completar metadatos del venue por tipo de orden y proyección decimal; impedir que el segundo redondeo invalide el primero.
4. Definir una capacidad de ejecución iceberg real, si se desea, con presupuestos agregados, órdenes hijas, cancelación y recuperación. No habilitar el booleano sin implementar y demostrar el contrato.
5. Serializar o versionar configuración por cuenta/símbolo y definir cuándo una confirmación cacheada es válida; medir coste de latencia/rate limit sin omitir seguridad.
6. Instrumentar sensibilidad gen→acción y razones de abstención fuera de muestra, con contexto de activo, escala y estado continuo de mercado; no premiar tasas de aceptación por sí solas.
7. Completar 146 lecturas Rust pendientes y la revisión no Rust. Ningún módulo parcialmente revisado se declara auditado completo.

## 14. Preservación, fuentes y estado Git

Cuatro fuentes existentes modificadas: src/bin/god_engine.rs, execution-engine/src/executor.rs, binance_api.rs y lib.rs. Se añade entry_dispatch.rs y una suite. Se preservan los cambios anteriores; los parches de los archivos grandes son localizados y no se aplica rustfmt global al host ni al ejecutor. Los informes se amplían mediante adendas, sin sustituir diagnósticos históricos.

Estado local observado: main, HEAD 59a76de4, árbol compartido con numerosos cambios. No hubo commit, push, merge ni fetch. No se infiere el estado de ramas remotas ni trabajo de terceros no inspeccionado. No se modificaron deliberadamente genomas activos, se entrenaron/promovieron modelos ni se terminaron/reiniciaron procesos.

La habilidad Firecrawl llevó a contrastar capacidades y confirmaciones con el contrato público, en lugar de trasladar semántica Spot a USD-M. Se reutilizó el scrape oficial de XXVI, con cache 2026-09-24T14:49:44.901Z; se leyeron ahora las secciones correspondientes. La búsqueda developer adicional no aportó un contrato primario mejor y sus pasajes de terceros no fundamentan el cambio. No se presenta reutilización de cache como una nueva comprobación del servidor.

La certificación integral y la equivalencia backtest/demo/producción permanecen abiertas.

## 15. Verificación documental y preservación final

Se validaron 27 referencias de evidencia, 30 enlaces locales y 47 hashes finales: seis fuentes/tests de esta ronda y 41 modelos. Sin archivos ausentes, referencias fuera de rango ni diferencias de hash. Los 41 modelos conservan el snapshot de XXVI. Los tres prefijos documentales preservan su hash normalizado CRLF→LF; las adendas añaden 2.361, 5.417 y 1.318 caracteres de cadena .NET al atlas, maestro y XXVI. Estas cifras no son bytes.

Rustfmt --check pasó en entry_dispatch.rs, binance_api.rs, lib.rs y la suite nueva. git diff --check pasó para las fuentes y documentos versionados intervenidos. El host se volvió a comprobar con cargo check después de retener el mismo ejecutor entre filtros y despacho; no se modificó la ruta de cierre. La repetición final de los diez tests nuevos pasó. Ninguna de estas comprobaciones certifica los archivos pendientes, los contratos abiertos ni el estado de un binario ya desplegado.

## Adenda de continuidad XXVIII — 2026-09-25

[Informe XXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVIII_2026-09-25.md>) y [artefacto XXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVIII_2026-09-25.json>). FMT-222 recibe contención: en la rama incierta no se confirma por símbolo/error ni se revierte por ausencia de posición. Se conserva la reserva y se consulta el ID con el ejecutor del envío; siguen pendientes identidad generacional, fills/hijos, resolución durable y rama Ok.

Nuevos FMT-223/224 refuerzan fronteras de consulta/posiciones, sin certificar el arranque con unwrap_or_default ni los otros parsers. FMT-225 identifica feedback/capital fuera del control exchange_confirmed; queda abierto y no se confunde simulación con live. FMT-226 contiene los reemplazos maker ante error inicial o consulta todavía activa. 16 pruebas nuevas, cinco rojo→verde; 66 funcionales y 11 diagnósticos abiertos, 77 pases. Check de cuatro binarios pasa, 41 modelos preservados. Cobertura sigue 143/289 Rust, 146 pendientes. No hubo órdenes, cuentas, promoción, reinicios ni publicación Git. Se conserva íntegro el contenido precedente.
