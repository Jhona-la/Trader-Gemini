# Auditoría de fundamentos XXVIII — evidencia de ejecución, vetos y causalidad del aprendizaje

Fecha: 2026-09-25. Continuación de XXVII. Estado: **reparaciones parciales verificadas; auditoría integral y certificación operativa abiertas**.

[Artefacto estructurado XXVIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVIII_2026-09-25.json>). [Informe precedente](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVII_2026-09-25.md>). Se añaden adendas al atlas y al maestro; no se sustituye su matriz histórica de 305 puntos ni se renumeran hallazgos anteriores.

## 1. Dictamen y alcance verificable

La desconexión más importante de esta ronda no es la ausencia de una teoría matemática más compleja: es que el sistema puede convertir una observación incompleta en una afirmación de ejecución y alimentar después estados económicos o aprendizaje con otra semántica. Un predictor temporal continuo no compensa una recompensa de procedencia incierta.

Se contienen tres clases de inferencia incorrecta: respuesta de posiciones inválida interpretada como lista vacía; consulta de orden no encontrada interpretada como rechazo concluyente; y presencia/ausencia de posición agregada usada para resolver una intención individual. También se contienen dos vías de reemplazo potencialmente duplicador de maker-chase.

Se añaden cuatro IDs: FMT-223 a FMT-226. Se actualizan FMT-222, FMT-179 y FMT-180; se relacionan con FMT-185/187/188/189 sin declararlos resueltos. Son hallazgos de contrato y flujo de código, con pruebas locales donde se indica; no un informe de incidentes o pérdidas observadas en una cuenta.

Inventario versionado comprobado: 1.119 archivos, 289 Rust y 24 manifiestos Cargo. La cobertura histórica de lectura completa sigue en **143/289 Rust; 146 pendientes**. Esta ronda profundiza en segmentos de archivos grandes y añade un módulo y una suite; no completa una nueva lectura integral de un Rust preexistente. No se suman archivos nuevos al denominador histórico ni se cuenta un grep como lectura completa. La auditoría de todos los archivos y de todas las teorías sigue pendiente.

Se modifican tres fuentes existentes: executor.rs, lib.rs de execution-engine y el host god_engine.rs. Se añade execution_evidence.rs y execution_evidence_contract.rs; cuatro pruebas adicionales viven en el módulo de tests del ejecutor. No se modifica la lógica de aprendizaje del core en esta ronda.

## 2. Resumen de estados

| ID | Prioridad y superficie | Evidencia | Resultado XXVIII |
|---|---|---|---|
| FMT-222 | P1, host y reservas | Rama ambigua mezclaba conservación, adopción y confirmación | Contención: conserva sin confirmar ni hacer rollback; falta resolver por intención/generación |
| FMT-223 | P1, consulta y registro | -2013→Rejected; ACK de otra identidad→Accepted; mutación previa a validación | Contrato REST reforzado y pruebas; WS, causalidad global y motivo tipado completo pendientes |
| FMT-224 | P1, posiciones y arranque | JSON inválido→Ok vacío; epsilon borra exposición; filas omitidas | Parser estricto conectado; arranque con unwrap_or_default y segundo parser siguen abiertos |
| FMT-225 | P1, capital y aprendizaje | La condición exchange_confirmed no cubre capital/feedback/mmap | Abierto, demostrado por flujo estático; no se aplica un gate global que rompería simulación |
| FMT-226 | P1, maker-chase | Cualquier error inicial→MARKET; NEW posterior a cancel→remanente | Dos vías contenidas; ledger padre/hijos, frescura y fallback tipado pendientes |
| FMT-179/180 | P1, fronteras de mensajes | Defaults y transportes heterogéneos | Mejora específica de GET; POST/legacy no quedan certificados |
| FMT-185/187/188/189 | P1/P2, causalidad y estado | Fusión, features y generaciones incompletas; expiración local | Conservan estado previo, con deuda reproducida donde corresponde |

Los términos «contenido» y «parcial» son deliberados. Mantener una reserva incierta evita una decisión destructiva sin evidencia, pero no es reconciliación completada, protección instalada ni garantía de disponibilidad.

## 3. Grafo vivo: raíz, decisión y terminal deben conservar significado

La cadena relevante es: **respuesta externa → evidencia validada → identidad de orden → fills → asignación de posición/reserva → resultado económico → aprendizaje**. Cada flecha requiere un contrato diferente. No es correcto comprimir toda la cadena en Ok, adopted o exchange_confirmed.

| Nodo | Datos que debe conservar | Salida legítima | Salto ilegítimo encontrado o pendiente |
|---|---|---|---|
| Raíz de observación | Cuenta, entorno, activo, endpoint, instante y payload válido | Observación con alcance y procedencia | JSON inválido convertido en ausencia |
| Nodo de orden | clientOrderId, orderId, lado, estado y cantidades | Admitida, activa, terminal o no resuelta | NEW tratado como posición ejecutada |
| Nodo de ejecución | Fills deduplicados, cantidad, precio, comisión y moneda | Ejecución atribuida | Presencia agregada del símbolo acredita otro intento |
| Nodo de decisión | Presupuesto, evidencia pendiente, acción factible | Enviar, esperar, reconciliar o reducir riesgo | Timeout habilita una nueva exposición completa |
| Nodo terminal económico | Cierre conciliado, costes, identidad y generación | Recompensa observada y estado contable | Cierre local no acreditado actualiza aprendizaje vivo |

El conocimiento sobre una orden y el conocimiento sobre la cartera son complementarios, no equivalentes. Una orden LIMIT puede existir sin posición. Una posición puede existir por una orden anterior, una intervención manual o una pierna diferente. Una orden FILLED histórica tampoco demuestra que la posición siga abierta: puede haberse cerrado después.

Que el modelo de mercado sea continuo no elimina los estados discretos del protocolo. FILLED, CANCELED y una identidad contractual no son regímenes de volatilidad arbitrarios: son hechos de ciclo de vida. Suavizarlos mediante una confianza predictiva permitiría actuar contra restricciones reales. La continuidad debe aplicarse a las variables y escalas que la admiten, manteniendo explícitas las restricciones del venue y de causalidad.

## 4. FMT-222 — conservar incertidumbre sin fabricar confirmación

### Evidencia, causa y consecuencia

En XXVII, la rama AMBIGUOUS/MAKER_CHASE_UNVERIFIED consultaba posiciones agregadas. Un símbolo encontrado devolvía adopted=true; una consulta fallida también devolvía true para preservar el estado. Ambos caminos terminaban en exchange_confirmed=true. Si la lista no contenía el símbolo, se declaraba que la orden nunca había llegado y se hacía rollback.

Son tres fallos lógicos distintos: ausencia de prueba de rechazo convertida en prueba de aceptación; exposición agregada atribuida a una intención sin enlace causal; y ausencia de fill confundida con ausencia de orden pendiente. Cambiar simplemente true por false agravaría el tercer problema al autorizar rollback bajo incertidumbre.

### Contención implementada

La [rama ambigua del host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4141>) ya no consulta presencia por símbolo para confirmar o revertir. Marca protección como dirty antes de esperar una lectura, consulta la identidad original con el mismo entry_executor usado por el despacho y registra ENTRY PENDING RECONCILIATION. No escribe exchange_confirmed ni llama rollback_positions dentro de esa rama. Tampoco borra una confirmación que otro productor pudiera haber publicado.

El resultado de la consulta se conserva en el registro de órdenes cuando pasa el contrato, pero no se proyecta directamente sobre un slot sin generación. Incluso Accepted puede significar NEW. Y un terminal del padre maker no liquida por sí solo el estado de un hijo taker. No se emite una orden adicional en este manejo de incertidumbre.

### Límites y criterio de cierre

El [rollback genérico](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4026>) sigue recorriendo slots abiertos; no tiene reservation_id. La rama Ok sigue marcando una posición local como confirmada a partir del éxito de la API de despacho. El núcleo puede cerrar localmente un slot mientras el trabajo asíncrono sigue pendiente. No existe aquí una transición transaccional que ligue cuenta/entorno/símbolo/pierna/intención/fill/generación.

Esta contención puede mantener reservas más tiempo, incluida una orden finalmente rechazada. La marca dirty solicita trabajo a un mecanismo existente; no prueba que un bracket esté instalado ni garantiza recuperación inmediata. No se añade una cola duradera de pendientes. Hace falta medir antigüedad del pendiente y estado de protección sin liberar capital por mero timeout.

Cierre exigido: ledger persistente por intención con hijos, fills deduplicados, transición económica versionada, rollback de una reserva exacta y pruebas de reinicio, reordenamiento, fill parcial, reutilización de slot y cambio de cuenta. La prueba de integración completa contra un transporte simulado todavía falta.

## 5. FMT-223 — consulta fallida o identidad ajena no prueban rechazo/aceptación

### Reproducción y contrato anterior

resolve_via_rest aplicaba el ACK al registro antes de validar la relación con la consulta. Un ACK con orderId positivo se aceptaba aunque perteneciera a otro símbolo. Un objeto vacío podía terminar en Rejected. Además, cualquier error cuyo texto contuviese -2013 se convertía en rechazo definitivo. El texto explicativo afirmaba que la orden nunca había llegado.

La documentación oficial de [Query Order](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#query-order) describe consultas por símbolo e identificador y límites de retención: determinados terminales sin fills dejan de encontrarse después de tres días, y órdenes antiguas después de noventa. Por tanto, no encontrada no acredita inexistencia histórica. Esta referencia no prueba una ventana concreta de inconsistencia inmediata tras un envío; esa latencia no se ha medido aquí. Se reutiliza el scrape oficial cacheado el 2026-09-24, no se afirma una lectura nueva del servidor.

### Reparación aplicada y significado de cada validación

[classify_query_result](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/execution_evidence.rs:78>) exige símbolo e ID consultados no vacíos y coincidentes, orderId positivo, lado BUY/SELL y cantidades finitas con 0 ≤ executedQty ≤ origQty y origQty > 0. NEW requiere cero ejecutado; PARTIALLY_FILLED requiere un ejecutado estrictamente interior; FILLED requiere la cantidad original completa. Un terminal con ejecución positiva no habilita rollback de esa ejecución. Estado desconocido, identidad discordante, dominio inválido o error de consulta devuelven Timeout, que se conserva como nombre de compatibilidad para «no concluyente».

[parse_query_order_response](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/execution_evidence.rs:121>) exige presencia de campos en el JSON antes de usar el DTO legacy. Es importante comprobar executedQty explícito: su ausencia no equivale a un cero observado. Se aplica sólo a GET de órdenes con cantidad; no se obliga a todos los ACK internos/POST/condicionales a compartir un esquema universal.

[record_query_resolution](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:492>) valida antes de mutar el registro y contrasta lado, símbolo y orderId previamente conocido. Un terminal atrasado con cero no puede producir Rejected si el registro fusionado ya conserva ejecución positiva. Cuatro tests del propio ejecutor verifican ausencia de mutación en errores/identidad inválida, conservación de lado e ID, prioridad de fills conocidos y distinción NEW/terminal sin fill. No hacen llamadas de red.

### Deuda restante

Accepted sigue agrupando admisión y ejecución por compatibilidad. Timeout agrupa red, esquema, identidad y estado; falta un motivo estructurado para observabilidad y recuperación. La ruta await_resolution sobre el registro WS no comparte todas estas validaciones. Otros productores pueden seguir llamando apply_ack directamente. get, validación, apply_ack y segunda lectura no son una sola transacción; se conserva FMT-185/188.

No se valida aquí un watermark causal, autenticidad histórica del fill, generación del consumidor ni positionSide en OrderAck. La cantidad original positiva limita el alcance a órdenes ordinarias con cantidad; un formato close-all distinto requiere su propio contrato, no inventar cero. La aritmética sigue siendo f64: no se declara equivalencia decimal exacta ni validación completa de todos los campos económicos.

Cierre: enum de evidencias separadas, motivos tipados, validación coherente REST/WS, identidad por cuenta/pierna y fusión bajo propiedad/versión. La ausencia prolongada exige reconciliación y observabilidad, no una regla que la convierta automáticamente en rechazo.

## 6. FMT-224 — el parser de posiciones podía fabricar una cartera vacía

### Causa reproducida

fetch_open_positions usaba bloques if let anidados y devolvía Ok(open_positions) aunque no hubiese podido interpretar el JSON. Un objeto de error, null o texto corrupto podían parecer ausencia de posiciones. Filas con campos faltantes se descartaban silenciosamente. positionAmt=NaN se descartaba por comparación; infinito podía admitirse. El corte abs(amount)>1e-8 borraba exposición no nula por una constante sin vínculo con el instrumento.

Se reprodujeron tres tests fallidos después de extraer la lógica original a una función conectada al mismo consumidor: payload inválido, valores no finitos y cantidad ±1e-9. La extracción mantuvo la semántica anterior; no se presenta una prueba nueva de endpoint real como si hubiera fallado en producción.

### Reparación y justificación

[parse_active_positions](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/execution_evidence.rs:32>) deserializa una lista con campos necesarios, valida la totalidad y sólo entonces devuelve un resultado. Un fallo no se convierte en una lista parcial. Se comprueban símbolo, cantidad y precio finitos, precio positivo para exposición abierta y apalancamiento positivo entero. Se validan lado/signo, duplicados por pierna y mezcla de BOTH con LONG/SHORT del mismo símbolo.

Sólo cero numérico representa exposición plana en este contrato; no se elimina una cantidad por ser menor que un epsilon. Se conserva la diferencia entre «existe exposición» y «se puede enviar una orden de ese tamaño». El lote mínimo es una restricción de acción, no una licencia para borrar pasivos pequeños.

Una cartera con LONG=2 y SHORT=-2 conserva dos posiciones en la salida, aunque su neto sea cero. La validación no colapsa activos ni piernas por un escalar. Se aceptan números JSON y cadenas numéricas finitas. Una fila plana puede tener entryPrice=0. Campos extra del venue no provocan rechazo sólo por ser nuevos.

### Lo que continúa abierto

El [arranque](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1357>) aún usa fetch_open_positions().await.unwrap_or_default(): puede volver a colapsar el error explícito del parser a vacío. Además, consulta antes de descartar posiciones en paper. La segunda ruta fetch_position_risk mantiene PositionRiskEntry con defaults; no comparte esta frontera estricta. Se documentan estos consumidores y no se declara que la cartera global quedó protegida por arreglar un parser.

ActivePosition conserva dirección pero no todo el identificador de pierna/cuenta/epoch. No acredita frescura ni completitud de un snapshot. No valida el universo completo del venue ni la representabilidad de todos los productos derivados (cantidad×precio, margen). La conversión decimal a f64 mantiene límites de representación, incluido subdesbordamiento extremo; quitar 1e-8 no elimina esas limitaciones.

Rechazar un snapshot completo por una fila inválida es una política de contención explícita. Evita tomar una vista parcial por cartera completa, pero puede reducir disponibilidad: requiere conservar el último snapshot válido como desactualizado, identificar instrumentos afectados y continuar gestión defensiva con evidencia suficiente. Esa política de recuperación no queda implementada aquí.

## 7. FMT-225 — un cierre local puede actualizar capital y aprendizaje sin acreditación de ejecución

### Evidencia estática y alcance de la afirmación

En [el cierre del core](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1563>), was_exchange_confirmed limita la escritura de coin.metrics.pnl_realized. Sin embargo, fuera de ese if se ejecutan record_trade_outcome, la suma a unified_capital, apply_spectral_epigenetic_feedback_with_time, la actualización del espectro temporal y del ensamble, y el productor mmap de predicción frente a resultado. Véase [la secuencia de feedback](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1607>).

La conclusión precisa es condicional: **si esa ruta cierra una posición local no confirmada, estos consumidores no están protegidos por la condición que sí protege la métrica de PnL**. No se ha ejecutado una cuenta ni se ha medido cuántos eventos de este tipo ocurrieron. Tampoco se afirma que toda adaptación actual sea falsa.

### Por qué afecta al genoma y a la comparación backtest/demo/producción

Se pueden generar dos verdades simultáneas: la métrica visible excluye el trade, pero el capital que usan otras decisiones y la recompensa del aprendizaje incluyen su cierre local. Una ventana de red, veto, pending o rollback puede alterar la población de ejemplos. Una mejora aparente del estimador puede corresponder a la selección de operaciones simuladas y no al rendimiento realizable.

El problema no se resuelve cambiando scalping/swing por una escala continua: la recompensa debe llevar el activo, el horizonte de la intención, versión de features/modelo/genoma, decisión, estado de ejecución y procedencia. FMT-187 ya señalaba features sin propiedad generacional; esta ronda añade el problema diferente de autorización de la recompensa económica.

Para una recompensa de trading, una formulación de contrato es R = PnL_de_fills_atribuidos − costes_atribuidos, en un numerario declarado. Un retorno contrafactual calculado de precios puede ser útil como etiqueta de investigación, pero es otro estimando. No debe entrar como fill o ganancia realizada. Del mismo modo, fallo del transporte no es una etiqueta de alpha negativo.

### Por qué no se añadió un if global

El core también soporta simulación/backtest. Exigir siempre exchange_confirmed en todos los consumidores podría apagar aprendizaje legítimo simulado o impedir liquidar posiciones de un backtest. La solución requiere separar procedencias, no ocultarlas detrás de otro booleano. Asimismo, una entrada real no basta para certificar un cierre real: ambos extremos deben conciliarse.

Propuesta de contrato, aún no implementada: OutcomeEvidence con modo/procedencia, cuenta/entorno, activo/pierna, intent_id, fills de apertura/cierre, costes, tiempos, generación de slot y versiones de features/modelo/genoma. Los consumidores declaran qué evidencias admiten: PnL realizado vivo, simulación, etiquetas contrafactuales y métricas diagnósticas deben tener canales separados.

Cierre exigido: pruebas diferenciales con la misma secuencia de mercado y eventos de ejecución para backtest, paper, demo-live y producción; ningún veto operativo, fallo de consulta o close local pendiente puede incrementar PnL realizado vivo ni entrenar como fill acreditado. También se necesita reembolso/retención de reservas coherente, sin doble contabilización al llegar el fill tardío.

## 8. FMT-226 — maker-chase podía aumentar exposición bajo incertidumbre

### Dos mecanismos distintos

Primero, maker_res.is_err() disparaba execute_raw_qty por la cantidad completa. Un error de envío puede ocurrir después de aceptación remota; entonces la nueva MARKET no es un reemplazo acreditado. Ejemplo lógico: intención q=1, GTX aceptada pero respuesta perdida y MARKET=1; si ambas ejecutan, el total puede ser 2. No se trata de una simulación de pérdida ni de un incidente observado.

Segundo, después de cancelar se consultaba la orden, pero se utilizaba sólo executedQty. Una respuesta válida NEW o PARTIALLY_FILLED no prueba que la orden ya no pueda ejecutar. Si q=1, acumulado=0,4 y se envía MARKET=0,6 mientras el límite sigue vivo, los fills posteriores del remanente maker pueden llevar la cantidad agregada hasta 1,6.

### Contención conectada al ejecutor

El [error inicial](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:2460>) ya no dispara una MARKET completa: retorna MAKER_CHASE_UNVERIFIED con el motivo original. Se aplica conservadoramente a todos los errores del transporte legacy, que no tiene un contrato suficiente para distinguir con seguridad cada no-envío/rechazo. Esto puede retener una reserva incluso para un rechazo auténtico; esa pérdida de disponibilidad se documenta, no se esconde como auto-adaptación.

La [consulta posterior a cancelación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:2512>) exige identidad, cantidades y terminal mediante [terminal_maker_executed_quantity](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/execution_evidence.rs:147>). NEW y PARTIALLY_FILLED devuelven incertidumbre y no habilitan el envío. Los tests verifican activo/no reemplazo, terminal válido y discordancia de identidad. La ruta de red completa de maker-chase no se ejecutó.

Se corrigieron comentarios y telemetría que afirmaban cancelación a partir de etiquetas de error. El sondeo existente de 100×4ms se describe como presupuesto fijo más scheduling, no como adaptación demostrada ni como latencia irrelevante por asumir cinco minutos. No se cambió ese presupuesto sin datos.

### Pendientes y criterio de cierre

El helper usa el snapshot de la consulta; todavía hace falta fusionarlo causalmente con fills WS antes de calcular remanente y resolver conflictos con cantidades posteriores conocidas. No existe un agregado durable padre/hijos que restrinja la suma de fills y reservas a la cantidad autorizada. Los IDs de hijos y la recuperación tras reinicio requieren revisión. La ruta LIMIT usa un transporte legacy y su éxito WS puede significar envío, no ACK semántico. El DTO no liga positionSide a la intención consultada.

El cálculo actual sigue siendo round_lote(max(q_solicitada − q_ejecutada,0)); debe derivarse del presupuesto final firmado y del acumulado causal de todas las órdenes relacionadas. No se cierra con este helper la proyección de cantidad, tolerancias de redondeo ni el mínimo nocional (FMT-175/219).

Cierre: transporte controlable por tests para aceptación seguida de timeout, cancel sin confirmar, fill parcial tardío, ACK ajeno, consulta vieja, restart y hijo ya ejecutado. Demostrar la cota de exposición conjunta antes de permitir fallback incluso cuando el error sea un rechazo válido.

## 9. Auditoría de vetos: dominios, seguridad, evidencia y estrategia no son lo mismo

| Restricción | Qué controla | Fundamento | Recuperación necesaria |
|---|---|---|---|
| Esquema/identidad inválidos | Calidad de evidencia | No se puede atribuir una respuesta | Reconsulta trazable; no aprendizaje de alpha |
| Cantidades no finitas/inconsistentes | Dominio aritmético y semántico | No hay cantidad económica utilizable | Conservar incertidumbre, corregir productor |
| Piernas duplicadas/modos mezclados | Coherencia del snapshot | No sumar ni colapsar dos interpretaciones | Diagnóstico por cuenta y modo |
| Orden activa después de cancelar | Exclusión de exposición concurrente | Todavía puede ejecutar | Resolver terminal antes de reemplazar |
| Error de envío maker | Incertidumbre de ejecución | Error no implica no-ejecución | Ledger, consulta y fallback tipado |
| Umbral abs(q)>1e-8 | Borrado de exposición | No tenía justificación por instrumento | Eliminado en el parser intervenido |
| Sondeo de 400ms | Latencia/recursos | Presupuesto fijo, no teoría espectral validada | Medir coste/oportunidad por contexto sin omitir seguridad |

No se debe optimizar el sistema simplemente para reducir el número de vetos. Un descenso de rechazos puede ser consecuencia de eliminar una salvaguarda necesaria. Deben distinguirse oportunidades descartadas por utilidad/riesgo, entradas físicamente no representables, capacidades no soportadas y ausencia de evidencia.

Las restricciones de seguridad no deben bloquear indiscriminadamente reducción de riesgo. Esta ronda interviene entradas y evidencia; no rehace toda la política de salidas, protección, rate limit ni kill-switch. Las reservas pendientes pueden impedir nuevas oportunidades: esa restricción tiene un motivo contable y debe ser visible, no una penalización encubierta a determinado horizonte o activo.

## 10. Fundamento matemático para el continuo multiactivo y la autoevolución

### 10.1 Separar espacio de estados y espacio de conocimiento

Un estado de mercado x_a(t,τ) por activo a, instante t y escala τ puede tener coordenadas continuas de volatilidad, liquidez, dependencia y persistencia. Eso no elimina la máscara de observación m_a(t,τ), ni el error del estimador, ni la disponibilidad causal de datos. Definir τ entre 1ns y 100 años no suministra datos nuevos en cada punto. Resolución del reloj, frecuencia de muestreo, soporte histórico y horizonte de decisión son cantidades distintas.

Además del estado de mercado, la política necesita un estado de ejecución: intención, órdenes pendientes, fills y reservas. Introducir incertidumbre de ejecución como si fuera confianza del predictor mezcla dos problemas de estimación. Una predicción excelente con una reserva duplicada sigue siendo una acción incoherente.

### 10.2 Cálculos y qué significan

N=|q|P es nocional en moneda cotizada; M≈N/L es una aproximación de financiación, no pérdida máxima; R≈N·d+costes aproxima riesgo al stop bajo hipótesis explícitas y no cubre gaps ilimitados. Estas identidades no cambian porque la intención sea de microsegundos o años. Cambiar L sin proyectar q no cambia N ni el término N·d: continúa FMT-113.

Para órdenes relacionadas, la cantidad ejecutada atribuida es una suma de fills únicos, no la suma de ACKs repetidos. El acumulado de un ACK y los eventos WS que lo componen no son observaciones independientes que puedan sumarse. Una reserva debería cubrir la exposición aún posible del conjunto de órdenes vivas; un error de lectura no reduce esa exposición posible a cero. Son invariantes contables y de conjuntos de escenarios, no filtros de volatilidad.

El resultado aprendido requiere una relación causal decisión→ejecución→coste→cierre. Si se entrena sólo sobre fills aceptados sin registrar decisiones pendientes, vetadas y contrafactuales por separado, cambia la distribución condicionada de ejemplos. Si se añaden cierres imaginados como reales, cambia además el estimando. No se ha estimado aquí la magnitud de esos sesgos; se identifica el mecanismo y las pruebas necesarias.

### 10.3 Qué teoría tiene prioridad y qué no se justifica todavía

Prioridad de ingeniería: máquinas de estados con evidencia, invariantes de conservación, semántica de eventos/tiempos, estimadores con procedencia y validación de políticas bajo observación parcial. Estas herramientas conectan directamente con los fallos encontrados. No se implementaron nuevos algoritmos probabilísticos ni un optimizador de políticas en esta ronda.

Una aplicación de física, teoría espectral, geometría o algoritmos cuánticos debe declarar observable, unidades, hipótesis, identificabilidad, aproximación numérica, complejidad y prueba fuera de muestra frente a una base. Una ecuación de un problema del milenio no es por sí misma un algoritmo disponible, una solución de mercado ni evidencia de ventaja computacional. No se promete omnisciencia ni rentabilidad a partir de nomenclatura. Tampoco se descarta investigación interdisciplinaria: debe superar el mismo contrato de evidencia.

## 11. Módulos del maestro: impacto de raíz a cima

| Módulo | Contribución de esta ronda | Pendiente que impide certificarlo |
|---|---|---|
| 1. Ingestión/L2/normalización | Dominio y esquema como frontera explícita de posiciones | Relojes, completitud, secuencias y contratos L2 no reauditados íntegros |
| 2. Inferencia/modelos | Identificación del canal de recompensa no acreditada | Separación de procedencias y evidencia del impacto predictivo |
| 3. Multiactivo/espectros | Conservación de exposición pequeña y piernas distintas | Dependencia conjunta, escala estimable y sincronización causal |
| 4. Ejecución/HFT | Consulta por identidad; contención de reemplazos inciertos | Ledger de órdenes hijas, paridad REST/WS y latencia medida |
| 5. Riesgo/Kelly/genomas | No confundir veto operacional con rendimiento del gen | Proyección q/precio/costes y causalidad de recompensas |
| 6. Estado/mmap/SO | Evidencia conservada sin falsa confirmación | Generación, atomicidad compuesta, arranque y consumidor mmap |
| 7. Cuántica/confluencia | Exigencia de observables y contratos verificables | Utilidad empírica y costes de mecanismos nominalmente cuánticos |
| 8. Backtest/gobernanza | Nuevos contratos reproducibles y deuda no ocultada | Replay de fallos, separación sim/live y promoción FMT-216 |

## 12. Verificación, límites experimentales y preservación

Se añadieron 16 pruebas: doce en execution_evidence_contract y cuatro en executor::tests_query_evidence. Cinco reproducciones fallaron antes y pasaron después. Tres ejercitaron el parser extraído y conectado con semántica original; dos ejercitaron la clasificación extraída de resolve_via_rest. Las restantes son pruebas de refuerzo de los contratos nuevos. No se atribuye rojo→verde a todas las ramas ni a la integración completa del host.

Ejecución distinta de esta ronda: **66 pruebas funcionales y 11 diagnósticos abiertos, 77 pases**. Desglose: entry_route=10, exchange_response=7, execution_evidence=12, order_evidence=6, payload_admission=9, reconciliation_evidence=13, execution_open_diagnostics=6 (una regresión y cinco de deuda), registry_open_diagnostics=3 de deuda, payload_open_diagnostics=3 de deuda, tests_query_evidence=4 y tests_m4_c01=4. No se suman las repeticiones de la suite nueva.

Los diagnósticos abiertos siguen demostrando, entre otros, defaults en el parser legacy de ACK, mezcla de monedas/comisiones, fallback del selector, expiración por reloj local, reemplazo de precio por snapshot viejo, colisión de adopciones hedge y revaloración maker sin presupuesto. Su pase no significa que esos fallos se hayan cerrado.

cargo check --offline pasó para god_engine, feature_exporter, train_forest y train_dark_alpha. Persisten tres warnings previos de evolution-engine (latest_ts, mode, trades). Rustfmt --check pasó en los dos archivos nuevos. No se formatearon globalmente los archivos grandes compartidos. No se ejecutó toda la suite del workspace, un replay end-to-end de ejecución, benchmarks de latencia ni la prueba de promoción inestable FMT-216.

Las pruebas HTTP existentes usan loopback efímero con respuestas sintéticas; las nuevas son funciones puras/registro local. No hubo peticiones a cuentas, órdenes, entrenamiento, promoción, modificación deliberada de genomas, creación de binario operativo, reinicio ni terminación de procesos. Los 41 hashes de modelos del snapshot XXVII se verificaron sin cambios.

Estado Git observado: main, HEAD 59a76de4 y árbol compartido con muchos cambios previos. No hubo commit, push, merge ni fetch. No se certifica el estado remoto ni se incorpora trabajo ajeno no inspeccionado.

La habilidad Firecrawl se utilizó para consultar el contrato oficial ya obtenido, siguiendo la reutilización de fuentes. [Nota de fuente y alcance](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/.firecrawl/XXVIII-query-evidence.md>). El artefacto registra hashes y verificación de prefijos documentales; los informes anteriores se conservan mediante adendas.

## 13. Hoja de ruta por dependencias y criterios de aceptación

1. Ledger de intención/reserva/fills por cuenta y entorno, con generación y órdenes hijas. Resolver pendientes sin confirmar ni liberar por ausencia o timeout. Probar crash/restart y eventos fuera de orden.
2. Arranque y reconciliación con snapshot completo/validado o modo explícitamente no listo; eliminar colapsos de error a vacío. Mantener gestión defensiva y observabilidad bajo datos incompletos.
3. Separar OutcomeEvidence real, simulado y contrafactual; corregir capital/feedback/mmap conjuntamente. Verificar exactamente una atribución económica por fill y ausencia de recompensas reales por rechazo/pending.
4. Completar maker-chase con transporte tipado, agregación padre/hijos, cantidades causales y presupuesto. No reabrir el fallback sólo para aumentar tasa de operaciones.
5. Cerrar FMT-113/175/219 y sincronización por instrumento; demostrar que filtros, payload, ledger y brackets representan el mismo plan de riesgo.
6. Medir latencias de consulta/cancelación/protección y sensibilidad gen→acción con activo/escala/estado continuo, incertidumbre y costes. Sólo entonces adaptar presupuestos temporales con objetivos verificables.
7. Completar las 146 lecturas Rust pendientes y la auditoría no Rust, manteniendo inventario por archivo y separación entre lectura, reproducción, reparación y certificación.

No se certifica que el sistema esté libre de fallos, sea plenamente autoevolutivo o que backtest, demo y producción sean económicamente equivalentes.

## 14. Verificación documental final

Validados 21 anclajes de evidencia del artefacto, 15 enlaces locales del informe y 46 hashes (cinco fuentes/tests y 41 modelos), sin ausencias, referencias fuera de rango ni diferencias. La suma independiente de las suites confirma 66 pruebas funcionales más 11 diagnósticos de deuda, 77 pases distintos.

Los prefijos de atlas, maestro y XXVII conservan exactamente su SHA-256 después de normalizar CRLF→LF. Las adendas añaden respectivamente 2.753, 4.922 y 1.234 caracteres de cadena .NET; no son conteos de bytes. El histórico queda preservado.

La comprobación estática de cableado localiza la rama ambigua del host y verifica ausencia de escritura exchange_confirmed, ausencia de llamada a rollback_positions y de consulta agregada de posiciones, uso del ejecutor retenido y marca dirty anterior a la consulta. Es una comprobación del código, no una ejecución concurrente ni una certificación de recuperación. git diff --check pasó en las fuentes/documentos versionados intervenidos; HEAD local continúa en 59a76de4.

## 15. Continuidad documental — ronda XXIX, 2026-09-25

La [auditoría XXIX](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIX_2026-09-25.md>) y su [artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIX_2026-09-25.json>) contienen la ampliación de FMT-225 y los nuevos FMT-227…232. Se preserva el diagnóstico histórico anterior; no se reescribe su estado retrospectivamente.

Se contiene capital/aprendizaje de cierres con entrada no confirmada, se aíslan las publicaciones de cierre simulado, se elimina una doble ingestión de recompensa y se conserva el horizonte que Kelly releía después de borrarlo. El ledger de liquidación real sigue pendiente. Se reproduce una pérdida de frames mmap y se documentan validación tardía de cabecera y kill-switch que suprime gestión local de cierres.

Cuatro rojo→verde; 82 funcionales y 6 diagnósticos abiertos, 88 pases distintos, una ignorada. Check de cuatro binarios pasa. Cobertura acumulada 144/289 Rust; 145 pendientes. Se verifican 41 modelos sin cambios. No se declara completada la auditoría de todo el proyecto ni la paridad económica backtest/live.
