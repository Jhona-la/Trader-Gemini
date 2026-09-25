# Auditoría de fundamentos científicos XXXVI — vetos, evidencia y recuperación causal

Fecha: 2026-09-25. Proyecto: Trader Gemini. Continuación de la [ronda XXXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXV_2026-09-25.md>). [Artefacto estructurado XXXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXVI_2026-09-25.json>).

## 1. Dictamen y alcance verificable

No se certifica que el sistema sea autónomamente adaptativo, que haya paridad backtest/demo/producción ni que sus vetos estén globalmente justificados. Esta ronda repara contratos concretos del centinela de drift y encuentra siete grupos adicionales de problemas, FMT-268 a FMT-274. FMT-265 y FMT-266 continúan abiertos parcialmente: sus errores numéricos y de propiedad del rearme se corrigen localmente; la calidad y causalidad de su evidencia no quedan resueltas.

La revisión parte del código actual, no de las etiquetas “quantum”, “holístico”, “forense” o “institucional”. Un test que pasa sólo demuestra sus aserciones; un log de certificación sin aserciones no certifica un sistema. Tampoco se deduce que todos los problemas previos sigan presentes: se revalidan los recorridos indicados y se preservan las conclusiones históricas como snapshots.

Inventario conservador: 1.119 archivos versionados, 289 Rust preexistentes y 24 Cargo.toml. Cinco nuevas lecturas completas: state_validator.rs (111 líneas iniciales), cybernetic_resilience.rs (144), resilience.rs (59), telemetry.rs (252), forensics.rs (120), todos de audit-engine. El acumulado pasa de160 a165/289; quedan124 Rust sin lectura completa registrada. Las búsquedas sobre todo el repositorio no se contabilizan como lectura completa. Host/core/executor se recorrieron por productores, consumidores y ramas relevantes; no se suman como archivos completos nuevos. Los tests nuevos no inflan ese denominador histórico.

Se preserva main en59a76de4be726098d9af934b4d35987e9a636802, con numerosos cambios previos compartidos. No hay commit, push, merge, fetch, checkout o reset. No se verifica estado remoto. No se ejecutan órdenes, peticiones de cuenta, entrenamientos/promociones operativos, reinicios ni build del binario operacional. Los cambios al host son código fuente comprobado mediante cargo check, no desplegado.

## 2. Grafo vivo: evidencia, decisión, veto y terminal

| Nodo | Productor y consumidor observado | Contrato que debería cumplir | Resultado de esta ronda |
|---|---|---|---|
| Raíz de evidencia | Propuesta local de cierre → bloque close_was_real del host | Identidad de cierre, generación, activo, coste, fuente y confirmación | El flag de entrada real no convierte la propuesta de salida en fill confirmado |
| Normalización | pnl / (qty × current_price) | PnL y nocional finitos; unidad explícita | Nocional inválido ya no se transforma en retorno cero; usa nocional de salida, no equity |
| Comparación | real y shadow=0,95real → DriftAuditor | Pareja causal independiente o etiqueta explícita de proxy | API checked; compatibilidad de símbolo/lado; la independencia sigue ausente |
| Decisión de veto | Resultado checked → DriftRecovery | Invalididad nunca cuenta como recuperación; racha consecutiva | Nuevo estado local, reinicio en error, liberación en la décima observación |
| Veto de entrada | DriftRecovery → GodEngineCore.drift_entry_veto | Bloquea entradas antes de reservar exposición; no cancela defensas | Aplicado en la frontera de entradas del core; otras causas permanecen separadas |
| Bloqueos externos | Inmune, degradación, insolvencia, incidencias → latches global/executor | Propiedad de causa y política de salida documentada | Drift ya no escribe ni limpia esos latches; sus políticas generales siguen abiertas |
| Nodo terminal | Propuesta → despacho, fill, reconciliación, ledger, aprendizaje | El efecto confirmado debe cerrar el ciclo con la misma identidad | Sin contrato transaccional completo; rechazos pueden usar rollback de todo el activo |

La distinción importante no es “scalp frente a swing”, sino propuesta/confirmación/evidencia desconocida, y riesgo de entrada frente a riesgo de mantener o reducir una posición. Un campo continuo de predicciones puede convivir con decisiones operacionales discretas justificadas por dominio, solvencia o protocolo. Continuidad no obliga a admitir NaN, violar un filtro de exchange ni convertir evidencia ausente en certeza.

## 3. Matriz de resolución de la ronda

| ID | Prioridad | Estado exacto | Problema |
|---|---|---|---|
| FMT-265 | P1 | Reparación parcial | Drift inválido aceptado y emparejamiento insuficiente; proxy no independiente |
| FMT-266 | P1 | Reparación parcial | Rearme no consecutivo, off-by-one y borrado de causas ajenas |
| FMT-268 | P1 | OPEN; documentación precisada | “Paridad” contable sin snapshot, dominio completo ni flujos externos |
| FMT-269 | P2 | OPEN; documentación precisada | Dos contratos de salud discrepan y carecen de ventana/denominador |
| FMT-270 | P2 | OPEN; documentación precisada | ChaosMonkey no reproducible, probabilidades inválidas y latencia truncada |
| FMT-271 | P1 | OPEN; documentación precisada | Pérdidas de telemetría y fallo de worker sin resultado observable |
| FMT-272 | P2 | Cálculo OPEN; comunicación corregida | Extrapolación lineal con50 trades/día presentada como duplicación compuesta |
| FMT-273 | P1 | OPEN reproducido | Rechazar una propuesta revierte todos los slots del activo |
| FMT-274 | P1 | OPEN estático y tests smoke | Pruebas forenses proclaman éxito sin verificar la tubería real |

Las prioridades expresan impacto potencial, no una incidencia observada en cuenta. No se reescribe la matriz histórica de305 puntos ni se equiparan sus identificadores a los FMT. La ampliación es aditiva.

## 4. FMT-265 — dominio numérico, compatibilidad y significado del drift

### 4.1 Mecanismo reproducido

Antes, d=shadow.pnl_pct−real.pnl_pct. Si d era NaN, se omitía la acumulación, pero abs(d)>limit también resultaba falso y el método devolvía Ok(NaN). El consumidor interpretaba ese Ok como observación limpia. Un límite NaN o infinito tampoco representaba una política validada. Dos resultados de activos o lados distintos podían compararse como si fueran el mismo experimento. Un acumulador que desbordaba dejaba de actualizarse silenciosamente y todavía podía devolver éxito.

La nueva prueba inicial ejecutó cinco casos: cuatro fallaron y uno pasó. Los fallos cubren PnL no finito, límite inválido, incompatibilidad de símbolo/lado y overflow acumulado. No se infiere cobertura de toda posible entrada flotante de esos cuatro casos; se añaden también diferencia desbordada y acumulador ya corrupto.

### 4.2 Reparación y compatibilidad

audit_execution_checked distingue InvalidLimit, IncompatibleTrade, NonfinitePnl, NonfiniteDifference, InvalidAccumulator, AccumulatorOverflow y Exceeded{drift,limit}. Sólo las discrepancias finitas que exceden el límite incrementan mismatch_count. Una observación inválida no se incorpora al total. El CAS conserva la suma observada o devuelve un error explícito si no es representable; no inventa un cero ni un valor saneado.

El wrapper anterior conserva Result<f64,f64>: exceso→Err(drift), otras invalideces→Err(NaN), nunca Ok(NaN). Esto evita ruptura de firma pero conserva una limitación semántica para consumidores legacy. El host usa la API tipada. El constructor sigue admitiendo max_drift arbitrario para compatibilidad; la validación ocurre en cada comparación, también si se modifica el campo público. No se afirma que construir el objeto valide la política.

El umbral0,05 y la regla inclusiva abs(d)≤limit se preservan. El cero es un límite válido y exige igualdad exacta. El total sigue siendo una suma firmada, no un estadístico de discrepancia absoluta: +0,04 y−0,04 producen total0 y ambos quedan debajo del umbral. Un test OPEN reproduce esa cancelación. No se confunde este hecho con un bug de CAS: lo que falta es una definición estadística apropiada del objetivo.

### 4.3 Qué mide realmente el host

Con r=PnL/(cantidad×precio_de_salida) y shadow=0,95r, el residuo es d=−0,05r, salvo redondeo. La condición abs(d)>0,05 equivale aproximadamente a abs(r)>1. No compara predicción previa con resultado posterior y no puede detectar discrepancia modelo/mercado de forma independiente.

El denominador es nocional de salida. Sin comisiones, para un largo con precios Pe y Ps, r=(Ps−Pe)/Ps, no (Ps−Pe)/Pe; su magnitud crece cuando Ps se aproxima a cero. No es ROI del capital ni retorno de entrada. Con apalancamiento, fees y reconciliación, hace falta fijar explícitamente qué retorno se estima antes de calibrar una alarma. Se corrigió el fallback de nocional inválido a retorno0: ahora llega como evidencia no finita al contrato checked y no aporta una observación limpia.

Precios y timestamps de TradeResult siguen siendo campos descriptivos, no validados por el comparador de PnL. Los precios de entrada0 del host no se hacen pasar por precios válidos: la API documenta que no los utiliza. Símbolo/lado compatibles no prueban identidad, generación, orden temporal, fill confirmado ni sombra independiente. La rama host sigue antes de la deduplicación temporal de cierres de bracket y antes de confirmar la salida asíncrona. El flag close_was_real acredita contexto de entrada, no el resultado de ejecución de salida.

### 4.4 Cierre requerido

Emitir evidencia inmutable de decisión y ejecución, con outcome_id, position_id, generación, símbolo canónico, unidades, costes, horizonte, timestamps de evento/recepción y origen. Emparejar la predicción congelada antes del resultado con fills/reconciliación confirmados. Deduplicar antes de afectar acumuladores o recuperación. Definir residuo y escala causal por activo/posición/portafolio. Sólo después escoger contraste, tolerancia y política de recuperación.

## 5. FMT-266 — recuperación por causa sin borrar otros riesgos

### 5.1 Error previo

La rama de error incrementaba drift_kill_armed_at, pero no reiniciaba drift_clean_closes. Por tanto, nueve cierres limpios, otro fallo y dos limpios podían liberar el bloqueo sin diez observaciones consecutivas. fetch_add devuelve el valor anterior: comparar clean>=10 tras incremento liberaba en la undécima observación partiendo de cero. Finalmente se ejecutaba store(false) sobre arena.kill_switch_active, aunque otro componente hubiera activado ese booleano por insolvencia, degradación o fallo inmune.

SeqCst no resuelve propiedad de causa: ordenar escrituras de booleanos no permite saber qué productor es dueño de un bloqueo. Cambiar únicamente>=10 a>=9 habría corregido el conteo, pero mantenido el borrado de causas ajenas.

### 5.2 Estado nuevo y alcance de la corrección

DriftRecovery es un estado local del bucle del host. Recibe una política NonZeroU32 y no tiene referencia al arena ni al executor. Ante Err o incluso un Ok no finito suministrado incorrectamente, bloquea y pone clean=0. Mientras está bloqueado, cada Ok finito incrementa de forma saturada; al llegar al valor requerido libera sólo su estado y reinicia el contador. La historia limpia anterior al incidente no da crédito anticipado.

GodEngineCore incorpora un veto privado de entradas, inicializado falso y controlado por un setter explícito. La frontera entries_blocked lo combina con permisos de entrada, feed, latencia y validez as-of; la gestión local de posiciones ya se ejecutó. El veto no toca el layout de GlobalArena, mmap, genomas ni latches compartidos. El host deja de escribir true/false en kill_switch_active por esta causa. No se crea un mecanismo de “desbloquear todo”.

Las pruebas cubren la décima observación, reinicio tras nuevo fallo, ausencia de crédito previo, errores tipados, valores no finitos fabricados, ciclos repetidos, cierre defensivo con allow_entries=true y conservación de un latch global ya activo. No prueban un fill real ni la recuperación del executor.

### 5.3 Límites abiertos de operación y temporalidad

Diez observaciones es una política heredada, no una probabilidad de recuperación. El estado no recibe identidad: repetir diez veces la misma pareja puede liberarlo. Tampoco conserva activo u horizonte del incidente; observaciones de B pueden rehabilitar un problema detectado en A. El veto de cuenta puede ser deliberado si el incidente compromete contabilidad global, pero la evidencia de recuperación debe probar ese mismo alcance.

No hay expiración por reloj ni garantía de disponibilidad: si no quedan posiciones elegibles, no llegarán cierres que satisfagan la racha. No se abren posiciones de prueba ni se borra el freno para fabricar evidencia. Se requiere una vía de reconciliación o recuperación explícitamente autorizada y comprobable.

El nuevo veto toma efecto en la siguiente decisión del core. Si process_event devuelve cierre y entrada en la misma llamada, el auditor corre después: la entrada ya generada no queda cancelada por esta reparación. Se documenta esta frontera pendiente; no se usa el rollback amplio de FMT-273 para ocultarla. Resolverla requiere resultado transaccional con identidad de la propuesta o arbitraje de veto antes de comprometer estado.

Los bloqueos global/executor siguen pudiendo impedir cierres locales y nuevas protecciones. El test OPEN de FMT-232 continúa reproduciendo esa política. No se aflojan controles del executor sin separar emergencia, rate-limit, cancelación, reduce-only, brackets y apertura. Corregir drift no autoriza ignorar vetos de solvencia.

## 6. FMT-268 — validador de paridad que pierde la invalidez

Archivo: [state_validator.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/audit-engine/src/state_validator.rs>).

La identidad local es E=C0+ΣPnL_realizado−Σfees_de_entradas_abiertas, comparada con capital actual. Su utilidad es encontrar discrepancias contables bajo una convención de ledger; no demuestra igualdad de señales, costes, estados ni resultados entre backtest y producción.

Problemas: C0 no finito/no positivo se sustituye por13; PnL no finito se omite; fees no finitas, negativas o cero no se agregan; capital actual no finito causa retorno silencioso. Las sumas pueden desbordar y no se comprueba el dominio del resultado agregado. El retorno es unit: el llamador no distingue coincidencia, divergencia, evidencia inválida o ausencia de chequeo. Sus tres tests verifican principalmente que la llamada no entre en panic; no afirman los resultados de una evaluación tipada.

Las lecturas Relaxed de múltiples campos y slots no constituyen snapshot conjunto. Cambiar todo a SeqCst tampoco crearía una transacción contable. Un cierre concurrente puede mover PnL, fee, capital y estado abierto en momentos distintos y producir una alerta espuria o esconder una inconsistencia. La ecuación no representa depósitos/retiros, transferencias, funding, rebates u otros flujos si no están incluidos exactamente en el mismo ledger. No se afirma que todos ellos falten en el sistema; faltan como términos y procedencia explícitos en este verificador.

El límite absoluto0,01USD es una decisión de tolerancia, no una demostración del error flotante máximo. diff>1 sólo imprime una alerta adicional si no hay posiciones; no “fuerza sync” aunque un comentario lo sugiera. No se localizó consumidor operativo externo al módulo en la búsqueda crates/src. Se precisa la documentación y se deja la lógica abierta para no introducir alertas/bloqueos con un snapshot no coherente.

Criterio de cierre: snapshot versionado coherente, entradas tipadas, ledger de flujos completo, tolerancia monetaria ligada a redondeo y número de operaciones, resultado estructurado y pruebas de intercalaciones. La política de bloqueo debe depender de la clase de discrepancia, no de un mensaje de log.

## 7. FMT-269 — umbrales de “salud integral” sin contrato homogéneo

Archivo: [cybernetic_resilience.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/audit-engine/src/cybernetic_resilience.rs>).

audit_systemic_integrity exige engines≥1 y latencia<1000ms. audit_comprehensive_health rechaza sólo latencia>2000ms, además de drops>100, error_rate_pct inválido o>15 y fails≥5. Con latencia1500 y todo lo demás nominal, el primero rechaza y el segundo acepta. En1000 y2000 también difieren las fronteras. Si son niveles de alerta distintos, la API debe identificarlos; hoy ambos se presentan como salud sistémica.

queue_dropped_events no lleva ventana ni volumen total.101 pérdidas acumuladas en una vida larga y101 en un segundo son indistinguibles. El porcentaje de error no lleva número de observaciones, incertidumbre ni definición de evento. El contador de motores no verifica dependencias, readiness de modelos ni cobertura por activo; uno activo podría ser insuficiente. El retorno early-exit sólo entrega la primera causa y oculta fallos simultáneos.

Dos diagnósticos OPEN reproducen la discrepancia y la falta de temporalidad. No se hallaron callers operativos fuera del módulo. Se retira la promesa de cero fugas/bloqueos/pérdida de señales. No se sustituyen1000/2000/100/15/5 por otros números inventados.

Diseño requerido: separar integridad de datos, disponibilidad de dependencias, saturación y calidad estadística; ventana/event-time, contador total y denominador; lista de causas con severidad y acción. Los límites de capacidad pueden ser discretos; su legitimidad depende del presupuesto de servicio y protocolo, no de llamarlos “holísticos”.

## 8. FMT-270 — inyección de caos sin probabilidad ni reproducción controladas

Archivo: [resilience.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/audit-engine/src/resilience.rs>).

La muestra es (reloj mod100)/100, derivada de RDTSC en x86_64 o SystemTime en otras arquitecturas. No hay semilla reproducible ni garantía de uniformidad/independencia. Aun si los cien residuos fueran uniformes, el soporte discretizado altera probabilidades que no caen en múltiplos de0,01: para0<p≤1, la fracción de valores menores que p sería ceil(100p)/100. El reloj real puede añadir correlación y sesgo; no se infiere una distribución observada sin medirla.

NaN,−0,5 y−infinito hacen falsa la comparación y simulan éxito sin declarar mala configuración. Un drop_rate2 genera “Network Drop”, confundiendo configuración inválida con fallo aleatorio. Ambos comportamientos tienen diagnósticos OPEN. latency_spike.as_millis()>0 descarta cualquier duración positiva submilisegundo; para el resto thread::sleep bloquea el hilo llamador, no simula un calendario de red. El test anterior sólo fuerza429; no examina reproducibilidad ni rutas de latencia.

Se corrige el comentario que llamaba determinista a la entropía; no se sustituye la fuente aleatoria en esta ronda. No se encontraron callers operativos de este ChaosMonkey, distinto del de data-pipeline. Cierre: inyector parametrizado por semilla y reloj virtual, validación de[0,1], guiones de pérdida/reordenamiento/duplicación y fake transport. No añadir sleeps al hilo de trading.

## 9. FMT-271 — la infraestructura diagnóstica también puede quedar ciega

Archivo: [telemetry.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/audit-engine/src/telemetry.rs>).

El canal bounded reserva un millón de eventos. Se ignora tanto el resultado de spawn del worker como try_send. Cola llena y receptor desconectado se pierden sin contador ni señal de salud; ParityAlert comparte esa misma política con otros mensajes. La auditoría puede detectar un fallo y perder justamente su evidencia sin que el emisor lo sepa. Es un riesgo estático, no se provocó agotamiento de RAM, saturación de producción ni fallo real de thread en esta ronda.

La promesa de~32MB no deriva del tamaño del enum: hay variantes con varios f64/u64 y String; deben contarse tamaño real del payload, alineación, overhead del canal y buffers externos. try_send es no bloqueante en su contrato de envío, no de latencia cero. Primera inicialización Lazy, asignación y scheduling también pertenecen al coste. Se corrigen esas afirmaciones, sin inventar un benchmark.

Faltan métricas offered/accepted/dropped/disconnected, heartbeat y vida del worker, política por severidad y drenaje en apagado. Una alerta contable crítica podría necesitar persistencia o una ruta independiente; mover todo a envío bloqueante podría empeorar el hot path. La corrección necesita backpressure explícito y presupuesto medido, no una capacidad mayor elegida a ojo.

## 10. FMT-272 — escenario lineal presentado como crecimiento compuesto

En el worker, avg_pnl=ΣPnL/N, daily=avg_pnl×50 y days=capital/daily. Son USD, USD/día y días respectivamente. Ese days estima tiempo para sumar otro capital de tamaño actual bajo ganancia monetaria media constante y cadencia fija. No es una dinámica de reinversión. Bajo un retorno diario constante g>0, una identidad compuesta distinta sería ln2/ln(1+g), y aún sería sólo un escenario, no un pronóstico.

La frecuencia50 no se mide y el agregado mezcla activos, tamaños, costes y horizontes. El comentario decía “por moneda activa”, pero el cálculo no multiplicaba por monedas; se precisa que es total hipotético. Las variaciones UpdateCapital, NaN y pérdida de eventos pueden invalidar el acumulado del worker. No se encuentra productor externo de send_trade_execution en crates/src; sí hay dos llamadas del host a update_dynamic_capital. Por ello no se atribuye una predicción operativa efectivamente mostrada en esta ejecución a esa rama.

Se conserva el cálculo y se cambia su comunicación a “escenario lineal no calibrado”, con fórmula y advertencia de que no es interés compuesto. Así se mejora lo existente sin sustituirlo por otra cifra infundada. Queda pendiente construir una estimación temporal con flujo completo, tamaños y costes, dependencia, incertidumbre y evaluación fuera de muestra. Duplicar capital en tres días es un objetivo del proyecto, no una propiedad demostrada ni una restricción científicamente garantizada.

## 11. FMT-273 — un rechazo local elimina estado de otras posiciones

Código: [rollback_position](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:435>).

La función recibe sólo coin_id, itera positions.slots(), cierra todos los abiertos, libera su margen y reembolsa su entry_fee local. No comprueba posición rechazada, generación, orden, lado o exchange_confirmed. El host la utiliza, entre otras rutas, ante veto de memoria y rechazos de entrada. En un sistema multiescala con varios slots de un mismo activo, rechazar una propuesta no autoriza borrar una exposición existente.

El nuevo diagnóstico abre una posición propuesta y otra posición confirmada en un slot distinto del mismo activo; fija margen total30 y llama rollback_position(0). Ambas quedan cerradas localmente y el margen pasa a0. Todo ocurre en arena y directorio temporal aislados; no se envía una orden. El test prueba pérdida local de identidad, no que el exchange haya cerrado nada. En operación podría dejar la cuenta con exposición mientras el core cree estar plano, hasta que una reconciliación intervenga.

La tupla new_order tampoco porta slot/generación. Añadir un nuevo veto tardío utilizando ese rollback extendería el problema. Por eso no se incorporó a la nueva ruta drift. Una entrada generada antes del nuevo veto sigue siendo una limitación explícita de esa frontera, no se disimula con una reversión amplia.

Criterio de cierre: propuesta con token inmutable de reserva/posición/generación, transición compare-and-close o cancelación de reserva bajo el protocolo del slot, compensación de su margen/fee una sola vez y conservación de slots ajenos. Pruebas de dos posiciones mismo activo, parcial, confirmación concurrente, rechazos duplicados, reutilización de slot y fallo de compensación.

## 12. FMT-274 — los “tests forenses” no sustentan su certificado

Archivo: [forensics.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/audit-engine/src/forensics.rs>), leído completo.

audit_data_consistency abre un CSV fijo, cuenta timestamps decrecientes y emite logs, sin assert sobre violations, esquema o presencia del archivo. Errores de lectura y líneas cortas se omiten; parse inválido se convierte en0. El binario sólo se inspecciona por metadata y se estima número de ticks como size/48, sin contrato de layout leído aquí. El test pasa también cuando no hay datos. No verifica backtest frente a producción.

audit_ml_data_leakage_prevention usa cinco precios constantes y medias de prefijos construidas en el propio test; no invoca el feature engine real, ni el generador de etiquetas, splits o entrenamiento. Correlaciona cuatro parejas features_T/precio_T+1. Un r alto no prueba lookahead: una tendencia determinista y causal puede producirlo; un r bajo tampoco excluye leakage no lineal, joins futuros o normalización sobre todo el dataset. Rust impide ciertos accesos fuera de límites, pero permite leer un índice futuro válido del array: seguridad de memoria no equivale a causalidad temporal.

Si r≥0,99 se intenta escribir .forensic_violation; aun entonces se imprime “GOVERNANCE PASSED” y “0,00%”. No hay aserción de fallo. Los dos tests pasaron en esta ronda, pero se contabilizan como smoke heredado, no evidencia de ausencia de leakage. No se encontró archivo .forensic_violation en raíz ni en el crate al comprobar después. No se cambia ni ejecuta un mutador AST.

Se necesita causalidad verificable: truncar el futuro y exigir invariancia del prefijo producido por el pipeline real; variar el futuro sin modificar features pasadas; assert sobre esquemas y orden temporal; dataset ausente→skip explícito o fallo de precondición según job; splits purgados por intervalos de etiquetas y lineage de normalización. Un certificado debe ser resultado estructurado de pruebas concretas con hash de datos/código, no texto incondicional.

## 13. Fundamentos: qué teoría integrar y con qué obligación de prueba

### 13.1 Modelo continuo sin confundirlo con resolución ilimitada

Representar estado y evidencia como funciones de activo a, escala u=lnτ, tiempo de evento t y variable observada. La implementación necesita bases, cuadraturas o discretizaciones con error y coste medidos; una malla numérica no es por sí misma un sesgo estratégico. Sí lo es tratar etiquetas legadas como ontología exhaustiva o usar coeficientes sin validación.

No se obtiene información nueva ejecutando el mismo dato un nanosegundo después. Soporte temporal solicitado, resolución real del feed, latencia del procesamiento y horizonte de predicción son cuatro variables distintas. Una extrapolación a100 años y una observación nanosegundo no comparten grado de identificabilidad. Cada región del campo debe llevar incertidumbre y soporte, incluyendo “no observado”; no un parámetro numérico plausible que finja conocimiento.

### 13.2 Detección secuencial y múltiples activos

Las confidence sequences estudian cobertura uniforme en tiempo y permiten decisiones en tiempos de parada bajo sus supuestos; no basta repetir intervalos puntuales. La fuente principal relaciona estas construcciones con supermartingalas no negativas y desigualdad de Ville. Los pasajes consultados contrastan el coste de cobertura uniforme con inferencia pointwise. [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240).

Como propuesta matemática, para un proceso no negativo E_t con E_0=1 y propiedad de supermartingala bajo H0, una frontera1/α puede controlar la probabilidad de cruzarla mediante Ville. Aplicarlo exige definir filtración, hipótesis, variable y condiciones de momentos/colas. Aquí no se ha construido ni validado tal proceso para los residuos. No es una garantía vigente del trading. La familia de test martingales y resultados de admisibilidad son referencias pertinentes para ese diseño, no certificados del código. [Test Martingales](https://arxiv.org/abs/0912.4269), [Admissible anytime-valid inference](https://arxiv.org/abs/2009.03167).

Observaciones repetidas, selección posterior del activo, dependencia entre escalas y cambios de genoma alteran el problema. Un posible presupuesto α_a,h con suma≤α para una familia predefinida y procesos válidos puede separar alarmas simultáneas; escogerlo tras mirar resultados exige otra justificación. Esta es una derivación de diseño, no implementación ni calibración completada.

La evaluación off-policy con secuencias de confianza ofrece una línea para promoción condicionada de modelos en contextual bandits; se revisó sólo el abstract y no se traslada automáticamente a una cartera con impacto de mercado y exposición persistente. [Off-policy Confidence Sequences](https://arxiv.org/abs/2102.09540). El enfoque Bayes-assisted para medias acotadas IID es vecino relevante; tampoco convierte retornos dependientes no acotados en IID por normalizarlos a posteriori. [Bayes-Assisted Confidence Sequences](https://arxiv.org/abs/2605.07964).

### 13.3 Integraciones con criterio de falsación

| Teoría/diseño candidato | Problema concreto | Requisito antes de implementarlo |
|---|---|---|
| Máquinas de estados y propiedad de causa | Rearme que borra vetos ajenos | Invariantes de bloqueo, acción permitida y recuperación por incidente |
| Procesos secuenciales válidos | Alarmas/rearmes estadísticos | Outcome causal, variable/colas/hipótesis, presupuesto multiasset y test de falsas alarmas |
| Bases multirresolución en logτ | Campo continuo de horizontes | Error de aproximación, soporte real, aliasing y coste por evento |
| Evaluación de política y promoción controlada | Genoma bueno en replay y débil en demo/live | Contexto congelado, costes, outcomes identificados y valoración terminal |
| Fault injection reproducible | Desconexiones y rutas de rechazo | Semilla, reloj virtual, guiones de fallos y observabilidad de compensaciones |
| Contabilidad versionada | Paridad capital/PnL/fees | Ledger reconciliado, snapshot consistente y control de duplicados |

No se implementan ecuaciones de problemas del milenio por prestigio ni se afirma ventaja cuántica. Cualquier teoría física o cuántica requiere mapeo de observables/unidades, supuestos falsables, algoritmo ejecutable, comparación con baseline y coste. El cuello actual demostrado es evidencia y transición de estado; añadir complejidad sin arreglar esa base haría más difícil atribuir resultados.

## 14. Validación, preservación y límites

Resultado final:155 pruebas únicas pasan,0 fallos en la selección final. Son114 de integración y41 unitarias. Clasificación:133 contratos/compatibilidad,5 smoke heredados sin aserción de resultado forense/paridad y17 diagnósticos OPEN. No se suman ejecuciones repetidas. Se añadieron21 pruebas:14 funcionales y7 OPEN; otras3 previas de drift pasan de OPEN a regresión de reparación. Cuatro fallos reproducidos antes del cambio pasan después.

Integración: drift_validation7, drift_recovery5, auditor_open9 (6OPEN), resilience_open6, trajectory_duration2, close_outcome26 (2OPEN), darwin_genotype2 (1OPEN), darwin_margin3, decay_causality10, fitness_numeric7, liquidation_state7, stateful_transition22, evolution/fitness_evidence8 (2OPEN). Unitarias: audit-engine19, core/darwin7, core/tests_m5_h012, evolution/fitness::tests13; este último filtro incluye5 de entropy_fitness y8 de fitness, contadas una sola vez.

Comandos ejecutados con --offline -j1 y tests en un hilo:

    cargo test --offline -j 1 -p audit-engine --test drift_validation_contract -- --test-threads=1
    cargo test --offline -j 1 -p god-engine-core --test close_outcome_contract --test stateful_transition_contract --test liquidation_state_contract --test decay_causality_contract --test fitness_numeric_contract --test darwin_margin_contract --test darwin_genotype_open_diagnostics -p evolution-engine --test fitness_evidence_contract -p audit-engine --test drift_validation_contract --test drift_recovery_contract --test auditor_open_diagnostics --test resilience_open_diagnostics --test trajectory_duration_contract --no-fail-fast -- --test-threads=1
    cargo test --offline -j 1 -p audit-engine --lib -- --test-threads=1
    cargo test --offline -j 1 -p god-engine-core --lib darwin::tests -- --test-threads=1
    cargo test --offline -j 1 -p god-engine-core --lib tests_m5_h01 -- --test-threads=1
    cargo test --offline -j 1 -p evolution-engine --lib fitness::tests -- --test-threads=1
    cargo check --offline -j 1 -p trader-gemini-v5 --bin god_engine --bin evolver --bin walkforward_evolver

El primer comando corresponde al RED inicial (4 fallos/1 éxito); la selección final incluye sus7 casos actuales en GREEN. La compilación final de los tres binarios mediante check termina correctamente. No se construye ni inicia el ejecutable operacional. Hubo espera normal por lock del build compartido; no se borró cache ni se terminaron procesos.

La selección final, comandos, conteos, hashes y referencias ancladas se registran en el JSON. La reproducción inicial tuvo4 fallos/1 éxito; tras la reparación esos cuatro contratos pasan. Los diagnósticos OPEN se separan de contratos: su éxito significa que la limitación aún existe.

No se equiparan los dos tests forenses y tres del StateValidator, sin aserciones sobre el resultado de auditoría, con una certificación. El resto de pruebas heredadas también conserva su alcance particular. La compilación y los tests no prueban latencia nanosegundo, rendimiento económico, fills reales, ausencia global de carreras ni cobertura exhaustiva.

Se conservan los41 modelos de la ronda anterior y los prefijos normalizados CRLF→LF del atlas, maestro e informe XXXV. El artefacto no se hashea a sí mismo ni certifica modificaciones concurrentes posteriores. Se mantienen los warnings anteriores latest_ts, mode, trades y toxic; no se ejecuta cargo fix.

## 15. Hoja de ruta causal, desde la raíz al terminal

1. Outcome y propuesta tipados con identidad, generación, fuente, moneda, costes y timestamps; deduplicación antes de auditar o aprender.
2. Rechazo/cancelación por reserva exacta, nunca por todo el activo; probar compensaciones concurrentes y preservación de posiciones confirmadas.
3. Separar política de apertura, mantenimiento, cancelación y reducción de riesgo por cada causa global/executor; resolver FMT-232 sin saltarse límites del exchange.
4. Recuperación por incidente con evidencia del mismo alcance y una vía de reconciliación cuando no existen cierres suficientes; no abrir exposición para producir datos.
5. Reemplazar proxy0,95r por predicción congelada y ejecución confirmada comparables; calibrar residuo y decisión secuencial fuera de muestra.
6. Contrato contable versionado, métricas de pérdida de telemetría y supervisor del worker; pruebas de saturación aisladas.
7. Pipeline real de pruebas de causalidad, replay compartido con configuración/modelos congelados, valoración terminal y promoción durable.
8. Continuar lectura completa de los124 Rust pendientes y documentar cobertura no Rust por separado. No marcar el sistema “certificado” por terminar una selección de tests.

## Continuación XXXVII — 2026-09-25

[Informe XXXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXVII_2026-09-25.md>) y [artefacto XXXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXVII_2026-09-25.json>). Se conserva íntegro el snapshot de esta ronda; esta adenda actualiza, no borra, sus pendientes.

FMT-273/FMT-266 avanzan con cancelación por reserva exacta, confirmación generacional y compensación idempotente para las rutas cooperantes; se cancela la propuesta actual ante veto tardío de memoria/drift. No hay ledger durable ni migración de todos los escritores. FMT-275/276 corrigen dominios, PF inventado, pisos de sizing, gate WR40%, capital ficticio y dependencia de unidad; sin caller operacional localizado de esas APIs. FMT-277 corrige confirmación del slot fijo y sobrescritura del único retorno entre candidatos. FMT-278 conecta PnL de emergencia a la función común con signo correcto. FMT-186 y113 continúan abiertos sobre admisión espectral y presupuesto de margen.

109 pruebas únicas aprobadas:103funcionales/compatibilidad,1estática y5OPEN;24nuevas.6 RED numéricos y1 RED de cableado→GREEN. Check de tres binarios correcto,41modelos preservados. Cobertura166/289Rust,123pendientes. Sin cuenta/órdenes, despliegue, promoción, reinicio ni publicaciónGit. Los detalles de límites y evidencia diferencial están en XXXVII; no constituye certificación total.
