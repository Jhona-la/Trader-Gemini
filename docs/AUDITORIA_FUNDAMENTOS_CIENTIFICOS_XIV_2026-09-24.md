# Auditoría científica XIV — decisión única, autorización revocable y crédito causal multiescala

Fecha: 2026-09-24. Corte local: main, HEAD 59a76de4. Continuación aditiva de XIII. Se conservan los informes y la matriz histórica de 305 puntos. Se añaden FMT-155–164, se actualiza parcialmente FMT-093 y se caracteriza de nuevo FMT-023. T37 propone un contrato de expertos disponibles y feedback retardado, sin afirmar una implementación productiva de esa teoría.

## 1. Resultado y clasificación de la evidencia

Se modificaron cinco fuentes: los orquestadores de señales, fases y margen, más dos funciones numéricas de señales. Se agregaron cinco archivos de tests. No se modificaron el consumidor central concurrente, el backtest, los genomas activos ni las cuentas. Las interfaces antiguas de consenso se conservan como compatibilidad: ya no evalúan dos motores ni aplican una preferencia nominal a swing.

**84 tests distintos aprobados, 21 nuevos.** Once pruebas fallaron antes de corregir sus mecanismos. Cinco pruebas nuevas son caracterizaciones que pasan porque el defecto permanece: no deben sumarse a reparaciones. Se ejecutó la suite de biblioteca de signal-engine, no toda la suite del workspace.

**Cobertura acumulada: 114/289 Rust preexistentes leídos completos; 175 pendientes.** Se acreditan siete lecturas nuevas: god-engine-core/src/orchestrator.rs, signal-engine/src/coaxial_breakout.rs, micro_scalp_trigger.rs, turbo_scalper.rs, trend_runner.rs, strategy-core/src/momentum_booster.rs y strategy_telemetry.rs. Los orquestadores de señales/riesgo, perceptron_gate, neuro_plasticity y supersonic_shockwave ya figuraban en rondas anteriores y no se cuentan de nuevo. Las lecturas dirigidas de grandes consumidores tampoco se suman. El inventario Git sigue en 1.119 archivos, 289 Rust y 24 Cargo.toml.

| ID | Prioridad contextual | Estado XIV | Contrato afectado |
| --- | --- | --- | --- |
| FMT-093 | P1, guard conectado | Dominio numérico corregido; coherencia/delta pendientes | Margen con capital, configuración o posiciones desconocidas |
| FMT-155 | P1, control conectado | Bypass y revocación corregidos; evidencia de aprobación pendiente | Esperar ticks no autoriza operación |
| FMT-156 | P2, compatibilidad de API | Corregido y probado | Dos consultas del mismo ensamble podían dar decisiones distintas |
| FMT-157 | P2, admisión de componentes | Corregido localmente | Un init fallido no puede incorporar un votante |
| FMT-158 | P1, consumidor identificado | Abierto | CVPIN se entrega como ratio de intensidad Hawkes |
| FMT-159 | P1, consumidor identificado | Abierto, ejemplo reproducido | La base del modelo no llega al gate que dice usarla |
| FMT-160 | P2 API; P1 de interpretación temporal | Dominio del helper corregido; construcción multiescala pendiente | Cero/cero aparece como compresión máxima |
| FMT-161 | P2, estabilidad numérica | Corregido para Mach finito | Una función acotada generaba NaN por cuadrado intermedio |
| FMT-162 | P2, contrato de target | Abierto, caracterizado | El TP “expandido” puede ser inferior al TP base |
| FMT-163 | P2, API auxiliar sin caller localizado | Abierto, caracterizado | PnL negativo no impide extensión pese a su prueba/documentación |
| FMT-164 | P2, API de telemetría | Abierto | Ausencia sustituida por valores plausibles y formato dual sin coordenada espectral |
| FMT-023 | P1, score consumido | Abierto, caracterizado | Piso de confianza y dependencia del número de clones |

Las prioridades no equivalen a incidentes demostrados en mercado. Se diferencian prueba local, búsqueda de llamadas, interpretación condicionada y validación económica todavía inexistente.

## 2. Grafo de decisión: relaciones observadas y fronteras no probadas

| Origen | Transformación/nodo | Consumidor observado | Frontera pendiente |
| --- | --- | --- | --- |
| Flag de aprobación del host | PhaseOrchestrator | Permiso de entrada consultado por god_engine | Identidad/versionado de lo aprobado y revocación hasta despacho |
| Márgenes de slots espectrales | PortfolioOrchestrator::allow_trade | Asignador de riesgo | Snapshot global coherente, reservas simultáneas y riesgo nocional |
| Estrategias registradas | TensorVoteOrchestrator | Consenso continuo del núcleo | Semántica común, disponibilidad y calibración |
| CVPIN del feature engine | Argumento hawkes_ratio de Turbo | Fallback de señal del núcleo | Corregir productor, conservar comparativa de comportamiento |
| ml_prob_motor | MicroScalpTriggerEngine | Voto dentro del ensamble | Falta publicación de ml_model_base |
| Proxies llamados atr_1s/5s/1m | CoaxialBreakoutEngine | Voto dentro del ensamble | Mediciones reales de intervalos y unidades |

No basta que una flecha transporte un f64. La conexión debe conservar significado, unidad, reloj, objetivo, identidad de instrumento y versión. Varias desconexiones aquí producen valores finitos: una prueba genérica de ausencia de NaN no puede detectarlas.

## 3. FMT-093 — el guard de portafolio ya rechaza estado numéricamente desconocido

**Evidencia:** [PortfolioOrchestrator](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/orchestrator.rs:100>). El [asignador de riesgo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:826>) sí llama a este guard.

El hallazgo ya estaba documentado en IV. La reproducción ahora ejecutada muestra que capital NaN podía autorizar una solicitud válida: capital≤0 resulta falso y exposición>capital·límite también. Un margen NaN de una posición abierta contamina las sumas con el mismo efecto. Un margen negativo puede compensar artificialmente el margen positivo. La configuración de drawdown no se validaba como fracción finita.

Se exige capital finito positivo, presupuesto de drawdown finito dentro de [0,1], margen de cada posición abierta finito no negativo y suma final finita. No se convierten datos inválidos a cero. Se mantienen para estados válidos los límites y el veto long durante Crash, que son decisiones de política anteriores, no conclusiones científicas nuevas.

**Pruebas:** tres rojas→verdes para capital, presupuesto y márgenes; dos preservan la frontera exacta del límite y la acumulación entre slots/lados. Con capital 100 y presupuesto 0,1, margen total 90 sigue aceptándose y 90,01 se rechaza.

**No cerrado:** el guard suma colateral, no delta ni nocional. La fórmula heredada permite una fracción 1−min(drawdown,0,2), entre 80% y 100%; endurecer el objetivo de drawdown puede aumentar el margen autorizado. No se la renombró como garantía de drawdown. Lecturas atómicas separadas no forman una instantánea de cartera y dos decisiones concurrentes pueden aprobar sobre el mismo capital antes de reservarlo. Hace falta contrato de reserva y reconciliación; el parche no constituye una certificación de riesgo global.

## 4. FMT-155 — el tiempo de espera sustituía la autorización genética

**Evidencia:** [PhaseOrchestrator::on_tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/orchestrator.rs:38>), [consulta del host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3271>).

La fase GenomicAudit avanzaba si darwin_approved era true **o** si habían transcurrido 5.000 ticks adicionales al warmup. La siguiente fase transitaba a PaperTrading o ProductionMainnet sin otra evidencia. La prueba usa aprobación siempre falsa, warmup mínimo y 6.000 llamadas: antes llegaba a ProductionMainnet.

Además, al retirar aprobación después de alcanzar la fase terminal, is_trading_allowed seguía devolviendo true. Revocarla entre GenomicAudit y DemoVerify tampoco impedía la transición. Ambas secuencias están reproducidas.

**Reparación:** se elimina la autorización por timeout. El permiso requiere fase habilitada y flag vigente; una revocación en DemoVerify/PaperTrading/ProductionMainnet devuelve el estado a GenomicAudit en la siguiente actualización. La consulta de permiso ya responde false antes de ese tick. Una nueva aprobación recorre otra vez DemoVerify. Las lecturas usan Acquire, sin pretender que ello vuelva transaccional el genoma o el resto del sistema.

**Semántica y límites:** el cambio controla admisión, no cancela órdenes ni liquida posiciones. No modifica el motor que ya esté ejecutándose: exige una futura integración/despliegue. El productor del flag todavía lo pone a true en el [arranque](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1195>); no hay en esta API identificador de genoma, evidencia, caducidad o versión aprobada. DemoVerify sigue siendo una transición, no una prueba efectiva de paridad. El permiso puede retirarse después de consultar el booleano y antes de despachar: la autorización final requiere vínculo con la generación y una frontera de ejecución. Son obligaciones abiertas, no subsanadas por cambiar el orden de memoria.

## 5. FMT-156 — la compatibilidad dual duplicaba evaluación y mantenía un sesgo nominal

**Evidencia:** [adaptadores de consenso](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/orchestrator.rs:397>).

evaluate_scalp_consensus_for_coin y evaluate_swing_consensus_for_coin ya delegaban ambos al consenso continuo. Sin embargo, evaluate_dual_consensus_for_coin los invocaba por separado. evaluate_consensus_for_coin hacía lo mismo y favorecía la segunda salida mediante un multiplicador 1,20 atribuido a persistencia macro.

El trait permite mutabilidad interior. Dos evaluaciones no tienen por qué observar el mismo estado ni ser funciones puras: una estrategia puede registrar un evento, actualizar memoria o leer un registro modificado. El test cuenta llamadas y devuelve +0,9 la primera vez y −0,9 la segunda. Antes se observaban dos evaluaciones, decisiones opuestas y una selección influida por el nombre de una ruta que ya no representaba otro horizonte.

**Reparación:** los adaptadores consultan una vez el consenso universal. La API de pareja devuelve dos copias de esa misma decisión para preservar la firma; no son dos muestras independientes ni dos motores. La API general retorna esa decisión directamente, sin preferencia 1,20. También se evitó construir un Vec temporal de referencias en cada evaluación continua: se itera sobre la colección existente, manteniendo orden y fórmula. Es una eliminación de asignación, no un benchmark ni una garantía de latencia.

**Alcance:** el núcleo observado ya usa directamente evaluate_continuous_consensus_for_coin; no se atribuye a ese caller una duplicación que pertenece a los adaptadores. Dos llamadas externas separadas todavía pueden reevaluar el mismo evento: una garantía global exactly-once necesitaría identidad/caché por evento y versión. Los nombres legacy se mantienen para compatibilidad, no como arquitectura de operación.

## 6. FMT-157 — fallar la inicialización no impedía votar

**Evidencia:** [registro de estrategias](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/orchestrator.rs:27>).

add_strategy descartaba el Result de init y añadía el objeto siempre. Así, una implementación que informa un registro incompleto o configuración inválida participaba igualmente. La prueba usa un initializer que devuelve Err y un voto alcista: antes el ensamble emitía Long.

try_add_strategy ahora propaga el error y sólo inserta después del éxito. El wrapper conservado registra el rechazo y no añade la estrategia. Hay una prueba roja→verde y una prueba posterior que comprueba el Result de la API nueva, además de confirmar que los votantes ya válidos no se alteran.

**Límite:** no se deshacen efectos secundarios que el initializer hubiera escrito en el registro antes de fallar. El host que use el wrapper tampoco recibe un conjunto tipado de componentes ausentes. Por tanto, no se acredita un arranque completo ni transaccional; falta declaración de componentes requeridos, transacción de registro o inicialización en staging y publicación conjunta.

## 7. FMT-158 — la ruta Turbo recibe otra variable científica

**Evidencia:** [fallback del núcleo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:3244>), [contrato de Turbo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/turbo_scalper.rs:23>), [current_vpin](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/math_kernels.rs:237>).

El caller obtiene cvpin.current_vpin(), lo nombra hawkes_r y lo entrega a hawkes_ratio. Para volúmenes válidos, CVPIN aquí es |buy−sell|/(buy+sell), un desequilibrio en [0,1]. El ratio de intensidad Hawkes compara ritmo condicionado con ritmo base, tiene otro significado y puede superar uno. Ambos son adimensionales: el análisis de unidades es necesario pero no suficiente para detectar esta incompatibilidad.

Turbo usa ese valor en coherencia, un supuesto z-score y significación. El mismo módulo, cuando vota dentro del ensamble, consulta hawkes_intensity publicado desde otro productor. Por ello, las dos rutas con un nombre común no representan la misma excitación ni responden igual a los genes.

**Impacto condicionado:** puede alterar activación o bloquear la ruta bajo determinadas configuraciones; no se afirma que la haga imposible para todo genoma ni que explique por sí sola la diferencia de PnL real. Su z-score tampoco centra la variable ni estima en ese punto una desviación con distribución nula contrastada.

**Cierre pendiente:** usar una variable explícitamente tipada y causalmente alineada al instante de decisión; congelar datos y comparar las dos rutas; recalibrar umbrales con la distribución correcta. El núcleo mantiene cambios concurrentes y se preservó intacto. No se “reparó” la variable convirtiendo CVPIN mediante otra fórmula arbitraria.

## 8. FMT-159 — el gate dice usar la base propia del modelo, pero no la recibe

**Evidencia:** [lectura de ml_model_base](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/micro_scalp_trigger.rs:171>), [base calculada en el núcleo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:2171>), [publicación de probabilidades](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:2365>).

MicroScalpTriggerEngine busca ml_model_base en el registro y usa 0,5 cuando falta. El núcleo calcula la base del bosque activo, pero publica ml_prob y ml_prob_motor sin publicar esa clave. La búsqueda dirigida en crates/src encontró consumidores y datos del consejo, no un productor de registro para esa base. No es una prueba dinámica de toda posible escritura externa.

**Contraejemplo reproducido:** Hawkes=2, OBI=−0,5 y p=0,36, con base real hipotética 0,30. Sin la clave, el fallback 0,50 interpreta p como un lift negativo y produce voto short. Al registrar 0,30, el mismo conjunto deja de pasar ese gate short. El test caracteriza el problema; no configura un modelo real ni modifica el host.

El helper should_trigger_micro_scalp lee umbrales del genoma; la implementación evaluate_for_coin usa cortes fijos Hawkes≥1,2, |OBI|≥0,2 y LIFT=0,05. Los tests que prueban sensibilidad de los genes en el helper no prueban sensibilidad de la ruta del ensamble. Turbo presenta una separación semejante entre helper parametrizado y vote con pesos 0,6/0,4 y cortes fijos.

También continúa la cuestión de FMT-054: un lift de probabilidad de beneficio no se convierte automáticamente en dirección de retorno. Publicar la base correcta repara una conexión, no redefine el target aprendido. Cierre: contrato tipado de objetivo/esquema/base/versión y matriz de influencia gen→ruta→acción, antes de extrapolar “el genoma funciona” a todos los caminos.

## 9. FMT-160 — cero/cero se interpretaba como compresión casi máxima

**Evidencia:** [helper coaxial](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/coaxial_breakout.rs:27>).

Se forman razones de ATR normalizados por sqrt(1), sqrt(5) y sqrt(60); las compresiones se calculan como max(1−ATR_corto/max(ATR_largo,epsilon),0), y el score es tanh(4·comp_1·comp_2). Con tres ATR cero, ambas compresiones eran uno y el score tanh(4)≈0,999329. También se admitían negativos transformados mediante max(0). La ausencia de variación pasaba a ser casi certeza de una ruptura.

La guarda ahora requiere ATR de un segundo finito no negativo y los dos denominadores finitos positivos. Se conserva un numerador cero con denominadores observados positivos: puede representar compresión real, no una razón indeterminada. La prueba roja cubre ceros/negativos; otra posterior confirma que no se bloquea indebidamente ese numerador cero válido.

**Pendiente en la ruta de registro:** evaluate_for_coin mantiene fallbacks sintéticos para escalas ausentes o inválidas. Y el [productor central](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:2087>) no mide tres ATR independientes sobre ventanas acreditadas de 1, 5 y 60 segundos: fabrica esas claves desde micro_v, v_t, atr_pct y precio, con mezclas y pisos. Dividir una cifra llamada atr_1m por sqrt(60) no prueba que su observación corresponda a ese intervalo.

La relación de escala browniana es una hipótesis de referencia, no una propiedad universal de rangos en microestructura irregular, saltos, drift y volatilidad variable. Faltan definición del estimador, unidades de cada canal, timestamps, error y comparación contra el mismo proceso muestreado a distintas cadencias. No se sustituyeron tres ventanas por otros números arbitrarios. El helper queda protegido numéricamente; el modelo multiescala sigue abierto.

## 10. FMT-161 — desbordamiento intermedio en una función acotada

**Evidencia:** [compute_shockwave_jump](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/supersonic_shockwave.rs:44>).

Para M>1 finito se calculaba tanh((M²−1)/(M²+1)). Aunque su resultado matemático está acotado y tiende a tanh(1), M² puede desbordar y producir infinito/infinito. La prueba con M=10^155 falló porque devolvía NaN.

Se utiliza la identidad:

`(M²−1)/(M²+1) = (1−M^−2)/(1+M^−2)`.

Para M>1, 1/M está entre cero y uno; elevarlo al cuadrado no desborda. Si subdesborda a cero en la cola, la evaluación toma el límite representable adecuado. Se conservan las guardas previas para entradas no finitas y M≤1. Tests comparan la expresión original en su dominio seguro y el límite para 10^155, 10^200 y f64::MAX.

**Alcance exacto:** se estabilizó la expresión para M finito. No se corrigió el posible desbordamiento del cociente que calcula M desde velocidad/sonido ni la normalización condicionada por magnitud de FMT-020. Tampoco se derivaron ecuaciones de conservación del libro que justifiquen llamar Rankine–Hugoniot a esta transformación. Una identidad algebraica preserva la heurística; no demuestra una ley física de precios.

## 11. FMT-162 — la cota suave puede reducir el objetivo base

**Evidencia:** [HighPayoffTrendRunner](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/trend_runner.rs:28>).

Antes del último paso, final_tp se eleva al menos al valor base. Después se calcula B·tanh(final_tp/B). Para x>0, tanh(x)<x: el paso final puede deshacer el piso. Con base=0,02, Hurst=0,5, VPIN=0, ATR=0,01 y B=0,1, el resultado es 0,1·tanh(0,2)≈0,0197375, inferior al base. La caracterización reproduce ese valor.

La prueba existente con Hurst=0,70 sí pasa porque su expansión previa compensa la contracción; no demuestra el invariante para todo el dominio. La implementación evaluate añade además su propio gate Hurst>0,52 y utiliza base/techo fijos 0,02/0,08, por lo que tampoco debe extrapolarse un contraejemplo de la API pura a una orden real en ese mismo contexto.

**Decisión pendiente:** definir si se desea un target suavemente acotado o una extensión que nunca reduzca el base. Cuando base supera el techo, ambas condiciones son incompatibles y debe explicitarse precedencia o rechazo. No se eligió otra curva para mejorar un test sin resolver ese contrato. La función y el consumidor conservan comportamiento; estado abierto.

## 12. FMT-163 — la penalización de PnL negativo no desactiva todos los términos

**Evidencia:** [VolatileMomentumBooster](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/momentum_booster.rs:43>).

La función pone positive_pnl=max(pnl,0), pero el peso final incluye un término Hawkes independiente del PnL. La prueba legacy afirma que un PnL negativo no dilata el TP, usando Hawkes=1; en ese caso la alineación ya es cero y no ensaya el supuesto conflictivo.

La nueva caracterización utiliza PnL=−0,01, Hawkes=3, dirección long y parámetros positivos explícitos. Devuelve un factor mayor que uno: anular una contribución no anula el conjunto. No significa necesariamente que extender en pérdida sea siempre incorrecto; significa que la condición anunciada no es la implementada.

No se localizó caller operativo de calculate_tp_extension en la búsqueda dirigida. Falta contrato económico de extensión, atribución causal y pruebas de escenarios adversos, costes y distribución de salidas. Esta función auxiliar no debe usarse como prueba de que el genoma gobierna un TP vivo.

## 13. FMT-164 — telemetría finita puede ser evidencia inventada

**Evidencia:** [StrategyTelemetryFrame](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/strategy_telemetry.rs:19>).

El frame de 64 bytes distingue subsistemas Scalp/Swing con payloads diferentes, pero no lleva coordenada tau, duración observada, versión de esquema, identidad del genoma o máscara de calidad. Valores no finitos se reemplazan por alternativas plausibles: capital 13, volatilidad 0,01, confianza/WR 0,5 o profit factor 1. Un lector no distingue esos defaults de mediciones reales.

Los casts de identificador/conteo a f64 pierden exactitud para enteros mayores de 2^53. El reloj usa SystemTime, no una secuencia causal ni reloj monotónico; que la unidad sea nanosegundo no demuestra resolución efectiva ni coste cero. La alineación de 64 bytes está comprobada por el test, pero no valida el contenido ni la promesa de “zero-latency”.

No se localizaron consumidores operativos externos de esos constructores durante esta búsqueda. No se migró su formato: reutilizar números de subsistema o bytes reservados sin versionado puede romper lectores históricos. Cierre: especificación común continua con estado observado/ausente/imputado, identificación de evento/versión y decodificadores compatibles. Debe poder diagnosticarse “desconocido”, no llenar huecos con una cifra que luce sana.

## 14. FMT-023 y filtros relacionados — la confianza sigue sin ser probabilidad calibrada

Las [caracterizaciones del ensamble](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/tests/ensemble_characterization.rs:1>) confirman dos mecanismos previamente descritos: votos idénticos de 10^-12 reciben score≥0,70, y clonar una estrategia de magnitud 0,2 cinco veces eleva el score respecto a una sola copia. Ambos tests pasan precisamente porque esa fórmula continúa presente.

No se cambió el piso ni la escala del score en una reparación de autorización. La ruta de riesgo puede consumirlos mediante otros gates; cambiar su calibración exige comparar tasas de aceptación, costes y exposición con datos apropiados. Tampoco se garantiza invariancia a clones simplemente eliminando un boost: habría que preservar la masa/prior asignada a la fuente, no tratar una copia como información adicional.

| Filtro o cálculo | Qué calcula realmente | Rigidez/obligación |
| --- | --- | --- |
| Consenso | Acuerdo de signo más magnitud y boost por conteo | Piso 0,70, correlación entre expertos y scores heterogéneos |
| Lifetime del consenso | max(30.000, base·(1+9·score)) | Confianza convertida a duración sin estimar supervivencia temporal |
| Micro dentro del ensamble | Cortes 1,2 / 0,2 y lift 0,05 | No usa los mismos genes que el helper probado |
| Turbo vote | Pesos 0,6/0,4 y cortes de Hawkes/flujo | Difiere del helper; ruta central entrega otra variable |
| Coaxial | Ratios sobre tres claves y tanh del producto | Soporte temporal sintético, pisos y escalado browniano no validado |
| Perceptrón | Ganancia acotada con piso de salida 0,15 | Discontinuidad al origen; FMT previo, no nueva reparación |
| Margen | Colateral agregado frente a fracción de capital | No es delta ni cálculo de pérdida extrema |
| Fases | Estado y permiso booleano | Ahora sin timeout permisivo, aún sin certificado de candidato |

La taxonomía evita tratar toda rigidez igual. Finitud, denominadores válidos y autorización son invariantes de seguridad. Un umbral predictivo necesita estimación/validación. Un límite de memoria es un presupuesto computacional. Una preferencia económica requiere política explícita. Hacer una función suave no elimina su arbitrariedad ni la convierte en autoevolutiva.

## 15. T37 — expertos disponibles, masa espectral y feedback madurado

### 15.1 Objetivo y ecuaciones con significado

Propuesta de investigación/integración, **no implementada en producción en XIV**. Se utilizaron los MCP de Firecrawl Research Index; el CLI no apareció en la comprobación local. Se leyeron abstracts, relaciones bibliográficas y pasajes pertinentes, no se afirma haber leído íntegramente toda la familia de artículos.

Cada experto debe emitir un registro con identidad, instrumento, tiempo de decisión, escala tau, objetivo predicho, esquema/versión, score o distribución, estado de disponibilidad y motivo de abstención. Una salida cero no distingue dato ausente de opinión neutra. Una cifra de “confianza” puede representar peso, disponibilidad, score o probabilidad: hay que tiparla.

Para predicciones comparables, una mezcla conceptual sobre experto i y escala tau puede escribirse:

`predicción(t) = [Σ_i ∫ w_i(t,tau)·a_i(t,tau)·p_i(t,tau) dlog(tau)] / [Σ_i ∫ w_i(t,tau)·a_i(t,tau) dlog(tau)]`.

Aquí w es masa de ponderación no negativa; a indica disponibilidad/peso de participación; p es una predicción del **mismo objetivo y horizonte de evaluación declarado**. Si el denominador es cero, el resultado es ausencia de evidencia, no 0,5 ni un long. No se pueden promediar probabilidades de beneficio a escalas diferentes como si fueran una sola probabilidad del mismo evento: hace falta fijar la consulta o un modelo conjunto.

Una implementación finita necesita cuadratura, soporte y error declarados. Refinar nodos de tau debe conservar masa, y clonar un experto debe dividir su masa previa en lugar de crearla. Esta es una condición de diseño propuesta para evitar sensibilidad al número de representaciones, no una propiedad ya demostrada del ensamble actual.

### 15.2 Familia científica y condiciones de transferencia

**AdaNormalHedge / expertos que se abstienen.** El problema confidence-rated permite participación en [0,1] y exige peso cero para quien se abstiene. El algoritmo actualiza a partir de pérdidas y regret frente a competidores, no a partir del nombre de una estrategia. Ese contrato permite representar ausencia sin inventar un voto. No se identifica por ello “confidence” del artículo con probabilidad de acertar un trade. [Achieving All with No Parameters: Adaptive NormalHedge](https://arxiv.org/abs/1502.05934).

**Hedging con feedback retardado.** La formulación consultada admite conjuntos discretos o continuos de expertos, pero supone pérdidas acotadas y que se revelan los vectores de pérdida correspondientes a decisiones anteriores. El PnL realizado sólo por la política elegida no suministra ese vector contrafactual. Aplicar su garantía a retornos desconocidos de órdenes no ejecutadas sería incorrecto. [Adaptive Hedging under Delayed Feedback](https://arxiv.org/abs/1902.10433).

**Familias complementarias.** AdaHedge/FlipFlop estudian adaptación de la tasa y propiedades frente a reescalado de pérdidas; Fixed Share/mirror descent aborda competidores cambiantes; coin betting aporta otra construcción de adaptación y expertos durmientes. Los abstracts localizados justifican compararlos, no declarar que alguno domina económicamente aquí. [Follow the Leader If You Can, Hedge If You Must](https://arxiv.org/abs/1301.0534), [Mirror Descent Meets Fixed Share](https://arxiv.org/abs/1202.3323), [Online Learning for Changing Environments using Coin Betting](https://arxiv.org/abs/1711.02545).

No se importó ninguna cota de regret sin sus hipótesis. Regret pequeño frente a expertos malos puede seguir significando pérdidas. La adaptación matemática del peso no acredita solvencia, liquidez, causalidad de ejecución ni ventaja fuera de muestra.

### 15.3 Protocolo de integración antes de experimentar con capital

1. Elegir un target observable sin ejecutar órdenes para el primer baseline: por ejemplo un evento direccional definido a un horizonte, con tratamiento explícito de huecos y revisiones. No confundirlo con beneficio de una estrategia.
2. Registrar en decisión predicciones, esquema, tau, disponibilidad, genoma/política y estado relevante; madurar la etiqueta al tiempo correcto, una sola vez.
3. Usar una pérdida coherente. Brier, (p−y)^2, es acotada para p∈[0,1], y∈{0,1}; ello no convierte un score heurístico en p. Calibración y discriminación deben medirse por separado.
4. Comparar baseline uniforme sobre expertos válidos, algoritmo adaptativo y ablaciones con el mismo tape y tiempos de feedback. Medir cobertura de datos, abstención, calibración y sensibilidad a clones/partición temporal.
5. Sólo después introducir PnL/costes y evaluación off-policy cuando exista soporte identificable. No adjudicar a una operación rechazada un fill ficticio para completar una tabla.
6. Mantener políticas en shadow, linaje, rollback y límites de riesgo ajenos al mecanismo que se optimiza. Los fallos de dataset/feedback de FMT-005/006/049 no quedan resueltos por cambiar el algoritmo agregador.

### 15.4 Prueba de influencia del genoma

Se propone una matriz de incidencia medida, no sólo un grafo de nombres. Para cada gen g_j, sobre el mismo estado causal y secuencia exógena, registrar diferencias de features, score, admisión, sizing y efectos al variar g_j dentro de su dominio. Si un gate duro impide derivar, usar comparaciones por regiones y cruce de frontera; no presentar una diferencia finita como derivada universal.

Una sensibilidad cero puede ser correcta por saturación o porque ese gen no corresponde al contexto. Es sospechosa cuando la documentación atribuye a ese gen el control de una ruta que nunca lo lee. FMT-159 ofrece un ejemplo concreto. La prueba debe registrar eventos bloqueados y motivos, sin convertir un veto en evidencia de inexistencia de alpha.

## 16. Física, algoritmos, cuántica y límites del paradigma

La lectura de neuro_plasticity vuelve a mostrar RDTSC y Xorshift como fuente de perturbaciones, no hardware ni medición cuánticos. Su función de “reconexión” perturba pesos pequeños, no cambia la topología de una red. La búsqueda no localizó un caller operativo de esa rutina ni de predict_with_plasticity fuera de sus definiciones; no se acredita auto-reconfiguración viva por su presencia en un crate. Son relecturas de deuda previa, no nuevos IDs para aumentar el conteo.

Tampoco se añadió una ecuación de problemas del milenio por prestigio. Cada teoría candidata debe declarar estado, observables, unidades, hipótesis, solución/estimación numérica, coste y criterio de falsación. En esta ronda la aportación matemática ejecutada es una identidad estable, dominios válidos y contratos de estado; T37 sigue siendo una integración propuesta.

El continuo temporal debe permitir preguntas en distintas escalas con soporte y error explícitos. No equivale a observar datos nuevos cada nanosegundo, ni a identificar empíricamente cien años a partir de un warmup de minutos. El nodo raíz aporta evidencia limitada; ningún nodo posterior puede convertir ausencia o una etiqueta sintética en información observada.

## 17. Hoja de cierre y verificación

Orden propuesto por dependencias, sin autorización implícita de trading o despliegue:

1. Integrar la autorización versionada y la reserva coherente de margen, conservando gestión de posiciones/órdenes existentes bajo veto.
2. Corregir en una integración coordinada los productores de FMT-158/159; pruebas de schema y paridad por ruta.
3. Sustituir claves temporales sintéticas por estimadores con timestamps, soporte y unidades; evaluar FMT-160 junto a los defectos previos de reloj.
4. Resolver contratos de targets y telemetría antes de convertirlos en fitness.
5. Ejecutar el baseline T37 con pérdidas observables y feedback causal; después comparar candidatos económicos fuera de muestra.

| Verificación ejecutada | Resultado |
| --- | --- |
| cargo test -p signal-engine --lib --test physics_numeric_contract --test single_consensus_contract --offline -- --test-threads=1 | 61 existentes + 4 numéricos + 4 de registro/consenso |
| cargo test -p signal-engine --test ensemble_characterization --offline -- --test-threads=1 | 5 caracterizaciones de fallos abiertos |
| cargo test -p risk-engine --test portfolio_admission_contract --offline -- --test-threads=1 | 5 tests |
| cargo test -p god-engine-core --test phase_authorization_contract --offline -- --test-threads=1 | 3 tests |
| cargo test -p god-engine-core --lib orchestrator::tests --offline -- --test-threads=1 | 2 tests previos |
| cargo check --bin god_engine --offline | Pasa, sin ejecutar el motor |
| rustfmt --check según edición de cada archivo | Pasa en los diez archivos intervenidos |
| git diff --check en los cinco fuentes | Pasa |

Persisten tres warnings previos de evolution-engine: latest_ts, mode y RealWfOutcome.trades. Los tests que crean arenas mostraron el aviso de entorno de genoma no definido; no se configuró demo/prod ni se promovió un artefacto. No se ejecutó un backtest económico, benchmark p99, exchange autenticado ni suite completa del workspace.

El [artefacto JSON XIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XIV_2026-09-24.json>) conserva estados, alcance, comandos, cobertura, fuentes y hashes. Maestro, atlas y XIII reciben adendas; su contenido previo se conserva. Se protegieron los cambios concurrentes de god_engine.rs, core/lib.rs, risk/lib.rs y booktick_replay.rs.

No se hicieron commit, push, merge, fetch, despliegue, reinicio, operaciones de cuenta ni cambios de genomas activos. Main local no acredita sincronización remota. La auditoría integral y la capacidad autoevolutiva productiva permanecen sin certificar.

### Control final de integridad documental

El JSON se parseó correctamente: contiene doce registros de hallazgo, diez IDs nuevos y estados parciales explícitos. Coinciden los diecinueve hashes de fuentes, pruebas y cuatro consumidores protegidos. Los veintiún enlaces locales del informe apuntan a archivos existentes y sus referencias de línea están dentro de cada archivo. Los prefijos completos anteriores del atlas, maestro y XIII mantienen su SHA-256 tras normalizar CRLF a LF; las adendas agregaron respectivamente 3.235, 6.791 y 1.231 caracteres normalizados, sin reescribir el historial. Estas comprobaciones acreditan integridad de este corte documental, no corrección de todo el software ni sincronización remota. La advertencia de permisos del archivo global de ignore de Git no impidió el estado local; su configuración no fue modificada.

## Continuidad — ronda XV (2026-09-24)

Se añade [XV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XV_2026-09-24.md>) y su [JSON](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XV_2026-09-24.json>) sin reescribir las conclusiones ni resultados de XIV.

Nuevos FMT-165/166: fallos de estado/covarianza en Kalman corregidos y dos genes de régimen sin lectura decisoria localizada. Se rehabilita la validación conjunta del estimador auxiliar de correlación y se añade una FFT V2 opt-in con calidad, centrado y potencia normalizada. Persisten la ruta ML legacy, la parametrización física del Kalman, las copias de correlación y el veto global BTC; no se presentan como cierres completos de FMT-002/004/046/091/140.

T38 desarrolla una propuesta de estado espectral multivariante con soporte, incertidumbre y causalidad, no una nueva taxonomía de volatilidad ni un algoritmo productivo ya instalado. 93 tests distintos aprobados, 25 nuevos; siete rojo→verde y cinco diagnósticos abiertos. Cobertura acumulada 115/289 Rust completos, 174 pendientes. No ejecución del motor, genomas activos ni publicación Git.
