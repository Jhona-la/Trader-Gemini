# Auditoría científica XXIV — transiciones de estado, filtros y memoria temporal

Fecha: 2026-09-25. Continuación aditiva de [XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIII_2026-09-24.md>). Referencia local main/59a76de4, sin verificación del remoto. El árbol contiene trabajo previo/concurrente que no se atribuye a esta ronda.

## 1. Dictamen ejecutivo

Se confirma una divergencia concreta entre arranque con velas y entrenamiento por ticks: las velas escribían last_price, pero no inicializaban las EMA de ticks ni el Kalman. El primer tick vivo tomaba la rama equivocada. Con precio constante 100, las EMA resultaban 9,5238095 y 0,9950249, y fair_price era 95,2385487. No era una nueva señal de mercado: era un error de inicialización.

Se corrige esa transición, se completa el reset de tres memorias omitidas, se agregan APIs fallibles con validación previa de ticks/OHLCV y se reparan condiciones numéricas del cooldown. El exportador v2 propaga el rechazo de features como error y no publica una observación ni un footer de éxito para ese caso.

Estas reparaciones NO convierten la arquitectura actual en un modelo espectral continuo multiactivo. Persisten bandas rígidas, kernels dependientes del número de eventos, unidades ambiguas y estimadores llamados Hawkes sin calibración/observación suficientes. Cinco diagnósticos reproducen explícitamente problemas ABIERTOS. No se confunden tests verdes de diagnóstico con problemas resueltos.

La revisión mantiene separados: fallo reproducido, reparación local, limitación científica, integración pendiente y resultado económico no medido. No se entrenaron/promovieron modelos, no se ejecutaron órdenes ni se reinició ningún motor.

## 2. Cobertura acreditada y método

Nueva lectura completa: [StatefulEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs>), 1.390 líneas preexistentes. Se leyó completo el cuerpo, sus métodos auxiliares y las diez pruebas unitarias existentes. Había sido inspeccionado por fragmentos en rondas anteriores, no contado como lectura completa.

Se releyó [HawkesProcessEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hawkes.rs>), 133 líneas, ya incluido en la primera ronda: NO suma nuevamente. También se inspeccionaron llamadas concretas de core/lib, bootloader, exportador y trainer. No se afirma haber releído completos esos archivos grandes.

Cobertura acumulada: **139/289 fuentes Rust preexistentes; 150 pendientes**. El inventario base sigue siendo 1.119 archivos y 24 manifiestos. Ninguna de esas cifras acredita haber revisado cada archivo ni certificado todo el workspace. Los dos archivos nuevos de pruebas se contabilizan aparte.

Método: contraejemplos pequeños antes de corregir; regresiones rojo→verde; pruebas de continuación tras rechazos; aislamiento entre dos instancias; conciliación con FMT-001/002/004/015/025 y XXIII; referencias primarias para las condiciones del modelo Hawkes. Las pruebas usan fixtures sintéticas y no corpus operativo.

## 3. Grafo raíz → decisión → terminal

~~~text
REST OHLCV ──► validación de valores ──► warmup sin reloj de cierre certificado
                                                │
Trades / quotes ──► reloj + valores ──► primera observación de ticks
                                                │
                               EMA / Kalman / memoria / FFT / Hurst
                                                │
                        snapshot por activo + consumidor del contrato
                         │                      │
              exportador v2 fallible     host / trainer legacy
                         │                      │
                 error explícito       rechazo aún no propagado globalmente
                                                │
                                  filtros / cooldown / decisión
                                                │
                               ejecución / fill / atribución genómica
                                          [NO CERTIFICADOS]
~~~

El grafo permite localizar dónde una reparación termina: try_process_tick protege su objeto antes de mutarlo ante los errores comprobados, pero no revierte escrituras que otro nodo ya realizó ni impide por sí solo inferencia con un snapshot anterior. Los nodos terminales y la promoción no se modificaron.

## 4. Matriz complementaria

| ID | Prioridad | Estado de esta ronda | Cierre que no se acredita |
|---|---|---|---|
| FMT-206 | P1 | Reparada inicialización tick tras warmup | Paridad completa de historia/estado train-live |
| FMT-207 | P1 | Reparado reset incompleto de tres campos | Separación de memoria de riesgo, modelo y observaciones |
| FMT-208 | P1 | Guardas locales y exportador v2 reparados | Propagación de rechazo a todos los consumidores |
| FMT-209 | P1 | Pánicos/wrap/horizonte inválido reparados | Cooldown continuo, calibrado y dimensionalmente consistente |
| FMT-210 | P1 | Abierto; tres diagnósticos Hawkes reproducidos | Modelo de intensidad identificado y API temporal coherente |
| FMT-004 | P2 | Ampliación de evidencia, abierto en consumidor | Frecuencias físicas y muestreo regular acreditado |
| FMT-001 / FMT-002 | P1 | Reconfirmados, sin reparación de modelo aquí | Velocidad con signo/unidad y Q/R dimensionales |

Los cinco nuevos IDs detallan mecanismos localizados y no se agregan automáticamente a los 305 puntos históricos como si todas sus familias fueran nuevas. FMT-209 amplía la familia FMT-015; FMT-210 se distingue del estimador de otra ruta analizado en FMT-025. Se conservan las mejoras históricas y se identifican sus residuales.

## 5. FMT-206 — el warmup dejaba una transición de inicialización imposible

### Mecanismo y evidencia

process_kline escribía last_price=close, mientras un cambio anterior había eliminado deliberadamente la actualización de EMA por velas para no mezclar kernels. Esa eliminación era razonable, pero process_tick seguía usando last_price==0 como condición de inicialización de todos los estimadores de ticks. La condición ya era falsa después de cualquier vela aceptada.

Con EMA inicial cero y precio p, la primera actualización heredada calculaba EMA_fast=(2/21)p y EMA_slow=(2/201)p. La feature relativa quedaba:

~~~text
(EMA_fast − EMA_slow) / EMA_slow
= (201 / 21) − 1
= 8,57142857…
~~~

Esto ocurre incluso con una serie completamente plana. El test lo reproduce para precios 0,001, 100 y 100.000: no es exclusivo de una denominación monetaria. La rama de Kalman también actualizaba el estado inicial cero, en lugar de sembrar el primer precio de ticks. Con p=100 se observó fair_price=95,2385487096467.

### Impacto causal

El mismo precio puede producir una tendencia inicial muy distinta según si hubo warmup REST. Esto afecta entradas del modelo sin requerir que el mercado cambie; puede contribuir a divergencias backtest/demo/live. No se atribuye a este mecanismo una cantidad de pérdidas ni se afirma que el agregado necesariamente opere: faltan trazas de decisiones y fills.

### Reparación y límites

[try_process_tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:490>) utiliza tick_count==0 para inicializar las EMA y Kalman del stream de ticks. Se mantienen los coeficientes existentes para observaciones posteriores. [Pruebas de transición](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/stateful_transition_contract.rs:38>) verifican la siembra, la ausencia de tendencia falsa en una serie plana y la igualdad de las EMA entre un prefijo de ticks con y sin warmup.

La primera transición ya no introduce el retorno vela→tick en los estimadores actualizados solo en la rama posterior. Esto es un cambio deliberado del comportamiento de arranque, no una promesa de salida idéntica al binario anterior. Las muestras que el warmup ya dejó en otros estimadores siguen existiendo: no se demuestra igualdad del vector completo entre historias diferentes. El consumidor histórico conserva FFT/multifractal con observaciones de naturaleza distinta; véase sección 10.

No se cambian Q/R del Kalman, su unidad ni su dependencia por actualización: FMT-002 permanece abierto en el llamador. Antes de promoción operacional se requiere evaluación del arranque y versión del contrato de features/modelo; aquí no hubo promoción.

## 6. FMT-207 — reset parcial y dominios de memoria mezclados

### Fallo reproducido

[reset](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:414>) reinicializaba buffers y muchos campos, pero omitía last_inst_v, dir_velocity y last_trade_is_sell. Después de procesar 100→99, reset dejaba last_inst_v=1, dir_velocity>0 y la última dirección vendedora. Una secuencia posterior plana no tenía el mismo estado que una instancia nueva.

La aceleración posterior usaba memoria del periodo anterior aunque su acumulador se hubiese puesto en cero. El caso plano también podía propagar la dirección antigua en la regla de ticks. Limpiar el ring sin limpiar los estadísticos que lo resumen crea un objeto híbrido: no es historia continua ni un reinicio completo.

### Cambio realizado

Se limpian los tres campos. La regresión [reset y continuación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/stateful_transition_contract.rs:79>) compara snapshots y features después de varios ticks planos con una instancia nueva; ambos coinciden en los campos comprobados.

No se afirma equivalencia formal de cada byte privado de todos los objetos ni se valida una política de reconexión completa. El snapshot de prueba incluye contadores, reloj, cinemática, vela, precio filtrado, Hawkes y los vectores 34D/10D, no un volcado de memoria cruda.

### Residual sistémico importante

El mismo reset borra rachas/cooldowns y restablece simd_nn a Default. Eso mezcla al menos tres dominios: observación transitoria, memoria de riesgo y parámetros aprendidos. Un reinicio por feed no debería interpretarse automáticamente como evidencia de que desapareció una pérdida o de que el aprendizaje se invalidó.

No se ha demostrado que una reconexión concreta invoque ese reset ni que abra una orden; se documenta la semántica que la API ofrece. No se cambia esa política sin un contrato de ciclo de vida. Cierre requerido: operaciones distintas para reiniciar observaciones, reidentificar un estimador, restaurar un modelo y conservar/restaurar memoria de riesgo con epoch y linaje.

## 7. FMT-208 — observaciones inválidas alteraban el prefijo del estado

### Qué se reproducía

El tick solo comprobaba precio positivo/finito. Se aceptaban volúmenes NaN/infinitos/negativos y timestamps anteriores al último tick. current_ts, acumuladores, EMA y precio podían cambiar antes de cualquier filtro de un componente interno. La guarda de un estimador no protege a los otros escritores.

process_kline tenía comentarios que anunciaban descarte de precios corruptos, pero empezaba actualizando Omni y extremos sin validar el OHLCV completo. Un open=103 con high=102 y low=98 se aceptaba. Sustituir un extremo por la vela previa no resolvía la validez ni la disponibilidad.

Entradas finitas también pueden producir resultados derivados no finitos: 1e200×1e200 para notional, o (1e300−1e−300)/1e−300 para retorno. Se reprodujo mutación con ambas.

### Contrato implementado

[FeatureInputError](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:13>) distingue InvalidPrice, InvalidVolume, NonFiniteDerivedValue, BackwardTimestamp, CounterExhausted e InvalidOhlc.

- try_process_tick valida precio, volumen, monotonicidad frente al stream de ticks, capacidad del contador, producto precio×volumen, suma de volumen de vela y retorno relativo representable, antes de la primera escritura.
- Se admiten volumen cero y timestamps iguales en orden del caller; no se inventa separación temporal ni se deduplican eventos.
- Timestamp cero deja de significar «no inicializado» para la vela interna: se usa tick_count==0. El caso t=0 seguido de t=1 conserva inicio 0 y volumen acumulado 5.
- try_process_kline valida precios positivos/finitos, low≤open/close≤high, volumen finito/no negativo y retorno relativo representable antes de actualizar subcomponentes.
- Las APIs unitarias anteriores siguen disponibles y descartan el error. Son compatibilidad, no telemetría de aceptación.

No hay un umbral de retorno de mercado añadido. Se comprueban restricciones del schema y de representación numérica. Una observación extrema pero representable no se rechaza solo por ser rara.

### Integración y límite de atomicidad

El [exportador v2](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:189>) llama la API fallible y propaga el índice del registro y la causa. La prueba [rechazo de features](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:438>) entrega un tape válido para su parser, pero con producto numérico no representable: el exportador aborta sin observación ni footer complete.

Core/lib y train_forest siguen llamando la API de compatibilidad. Sus actualizaciones posteriores pueden continuar sobre un estado previo; otros estados del host pueden haber cambiado antes. No se declara rechazo atómico del grafo completo ni cierre del bloqueo de inteligencias. Se requiere un resultado de aceptación propagado hasta decisión/evaluación y una política explícita de datos rechazados.

Tampoco se prueba finitud de todos los cálculos internos para todo f64. Campos públicos pueden ser alterados externamente, los estimadores conservan sus propias guardas y las conversiones f32/variancias tienen otros límites. try_process_kline sigue sin timestamp ni flag de cierre: valida geometría numérica, no causalidad temporal ni finalización de la vela.

## 8. FMT-209 — cooldown: errores numéricos reparados, arbitrariedad no resuelta

### Reproducciones y reparación acotada

Con una pérdida y min_cooldown_ms=600.000, el código aplicaba clamp con mínimo 600.000 y máximo 300.000: el proceso entraba en pánico. Con mínimo u64::MAX aparecían además multiplicaciones no saturadas. Un tau NaN con mínimo cero podía admitir una apertura.

Se usan multiplicaciones saturadas; el límite superior del clamp no puede ser menor que su mínimo. Tras convertir a entero, el resultado se acota nuevamente por el mínimo exacto del caller para no reducirlo por redondeo f64. Un horizonte no finito o no positivo se rechaza.

El fallback de reloj multiplicaba diferencia de ticks por 100 sin saturación. El stress con tick_count=2^62 y mínimo 1 reprodujo elapsed=0 por wrap en la configuración usada. Ahora esa multiplicación satura, también en los dos lectores auxiliares de rachas. La prueba no representa una cantidad de ticks observada en producción.

No se disminuyen mínimos solicitados ni se incorpora un parámetro de trading nuevo. Los casos ordinarios mantienen la política heredada; el test de desacoplamiento espectral preexistente sigue pasando.

### Discontinuidad confirmada

[El diagnóstico de frontera](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/stateful_open_diagnostics.rs:6>) registra una pérdida a tau=59.999 ms, después un resultado sin pérdida a tau=3.600.000 ms y consulta ambas escalas cercanas. Con el mismo historial, el candidato a 59.999 ms se bloquea y a 60.000 ms se admite.

Una diferencia de un milisegundo cruza el índice de banda aunque las escalas sean casi idénticas. La regla adicional |Δln tau|<0,60 no elimina esa partición; interactúa con rachas globales y por banda.

### Rigideces todavía abiertas

Bandas 60.000/1.800.000 ms, escalas base 30.000, pisos/techos 60.000–7.200.000, cortes de media ventana, factores 0,20–5 y multiplicadores discretos no derivan en este método de una estimación/calibración documentada. tau≤10 ms positivo termina usando 30.000 ms: la API acepta un número continuo pero no conserva su significado en ese dominio.

La comparación v_t>0,0015 utiliza una magnitud en unidades de precio con un umbral absoluto. Su decisión puede cambiar al redenominar el activo; no es una condición de volatilidad invariante. El fallback ticks×100 supone una cadencia, no un reloj observado. Los lectores auxiliares y record_trade_outcome aún no comparten todas las guardas de horizonte; los incrementos u32 de rachas siguen sin saturación.

Estas limitaciones permanecen abiertas. No se eliminan protecciones de riesgo alegando continuidad. El reemplazo requiere pérdida normalizada, exposición, incertidumbre, soporte de evidencia, validación fuera de muestra y un contrato que no debilite límites duros.

## 9. FMT-210 — el Hawkes local no tiene aún un contrato de intensidad certificado

### Fallos de API comprobados sin modificar el módulo

[HawkesProcessEngine::update](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hawkes.rs:29>) no rechaza timestamps menores que last_update_ms: omite decay y agrega el impulso tardío al estado actual. El diagnóstico t=1000 seguido de t=900 confirma aumento de intensidad sin retroceder el reloj.

La inicialización usa last_update_ms==0. Un impulso en t=0 seguido de t=1000 sin impulso no decae; el cero fue confundido con ausencia de reloj. La corrección del agregador StatefulEngine no corrige automáticamente este reloj interno.

[update_batch](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hawkes.rs:82>) devuelve ratio=0 para lote vacío aunque las intensidades existentes sean distintas. Su salida no es una vista consistente del estado. Las tres pruebas diagnósticas pasan porque estos problemas siguen presentes; no porque hayan sido solucionados.

### Límites científicos y numéricos

El constructor recorta mu/alpha/beta pero inicializa intensidades con el mu original. Configuraciones no finitas o negativas no tienen un constructor fallible. El ratio puede volverse no finito tras sumas extremas; no se ensayó aquí todo el dominio numérico. dt se limita a 300 s y beta a un mínimo 0,1: esa truncación cambia la evolución exacta, aunque para parámetros rápidos el residuo pueda ser pequeño.

La excitación es alpha·[1+min(log(1+volume/normalizador),3)]·|OFI|, no necesariamente un salto de un proceso de conteo de tipo identificado. process_tick usa el OFI previo porque el host llama update_ofi después; también usa un proxy de volumen de profundidad. No se convierte en flujo observado por denominarlo Hawkes.

Dos canales bull/bear no son por sí solos una red multiactivo con excitación cruzada. No hay en este módulo likelihood, compensador, calibración por tipo de evento, identificación de matriz de kernels ni prueba de estabilidad estimada. No debe leerse alpha como fracción endógena o branching ratio sin definir el kernel y sus marcas.

### Cierre requerido

Primero corregir reloj/constructor/snapshot con APIs fallibles compartidas; después definir eventos, marcas, unidades y observabilidad. Solo entonces estimar kernels y evaluar compensadores/errores y estabilidad. Corregir el lote vacío no acreditaría alpha predictivo; añadir una ecuación multivariada sin observaciones tampoco.

## 10. Memoria física, espectro y características: FMT-004 y residuales

La lectura completa reconfirma que la FFT recibe retornos por evento y se analiza cada 64 ticks; no almacena aquí una malla regular de reloj. El warmup introduce cierres de velas en el mismo objeto espectral y luego llegan retornos de ticks. El bin no puede interpretarse sin más como una frecuencia física.

La vela interna se ancla al primer evento y se reancla al evento que cruza 60.000 ms. [Diagnóstico de malla](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/stateful_open_diagnostics.rs:39>) confirma que un cruce en 60.001 deja el origen allí, no en 60.000. Un gran hueco produce una sola actualización, no evidencia para todas las ventanas intermedias. No deben fabricarse esas ventanas como observadas.

Las EMA 20/200 por evento y 9/21/120/720 por vela, el piso de ATR y las escalas Hurst 10/25/50 no abarcan automáticamente un continuo. FMT-001 persiste: dir_velocity se forma con |Δp|, no con signo; FMT-002 persiste en unidades y Δt de Kalman. El valor get_market_regime devuelve Continuous incluso cuando el campo interno arranca Neutral: es una etiqueta, no certificado de soporte, readiness o continuidad.

Las features zerificadas por contrato y los defaults de arranque tampoco son mediciones. La ausencia explícita del exportador XXIII mejora parte del contrato, pero sus primeras 34 dimensiones aún heredan semánticas de este objeto.

## 11. Fundamentación científica y diseño propuesto, sin promoción automática

### T01/T24: evolución entre eventos y pruebas que distinguen reloj de cadencia

Para una memoria exponencial z sin nueva evidencia, una dinámica explícita es dz/dt=−z/tau. Su transición exacta es z(t+Δ)=exp(−Δ/tau)z(t), y exp(−(a+b)/tau)=exp(−a/tau)exp(−b/tau). Esta propiedad permite comprobar que subdividir un intervalo sin eventos no altera la memoria. tau, Δ y las unidades deben ser compatibles.

Para observaciones ponderadas por evento pueden mantenerse S0 y S1 con igual decay, después agregar w y w·x, respectivamente; S1/S0 es una media de evidencia con olvido. Eso NO es automáticamente invariante a duplicar mensajes: duplicar eventos cambia su peso. Hay que definir si w representa un trade, volumen, exposición, intervalo observado o confianza. La regla de actualización no se decide por prestigio de la ecuación.

Si se interpola una señal entre eventos, debe declararse si se mantiene el último valor conocido o se usa un modelo. Usar el próximo valor para rellenar retrospectivamente un intervalo introduce información futura. Las pruebas deben comparar estados as-of, duplicados, eventos simultáneos, cambios de unidad y huecos; no solo finitud.

Estas son propuestas de contrato y validación. No se reemplazaron hoy todas las EMA por decay físico, porque cambiaría el significado de las features/modelos y exigiría una migración coordinada.

### T03: Hawkes multivariado con hipótesis comprobables

En un Hawkes lineal positivo, lambda_i(t)=mu_i+sum_j integral phi_ij(t−s)dN_j(s). La integral de phi_ij tiene significado de descendencia esperada en ese modelo, no cualquier coeficiente llamado alpha. Bajo kernels causales, no negativos e integrables, el marco estacionario de Bacry/Muzy impone radio espectral de la matriz de kernels integrados menor que uno. Su identificación por estadísticos de segundo orden es condicional al modelo; no valida cualquier covarianza empírica. [Fuente primaria](https://arxiv.org/abs/1401.0903).

Martins/Hendricks analizan compensadores, ajuste de residuos y estabilidad de calibración para eventos de libro concretos. Eso orienta cómo probar un candidato, no acredita el código actual ni la rentabilidad en otros instrumentos. Reescalar tiempos por la intensidad acumulada debe producir residuos compatibles con el modelo; no rechazar una prueba no prueba que el modelo sea verdadero. [Fuente primaria](https://arxiv.org/abs/1604.01824).

La literatura también documenta sesgos de endogeneidad por cambios de parámetros/proceso y calidad de timestamps. Un sistema puede reconocer cambio continuo de contexto sin tratar etiquetas de régimen como verdades físicas. Esta fuente se consultó a nivel de abstract; no se simularon aquí sus resultados. [Fuente primaria](https://arxiv.org/abs/1308.6756).

### Continuo multiactivo y grafo vivo

Una representación de investigación puede indexarse por activo, tipo de observación, lado y log(tau), con soporte observado, incertidumbre y época de estado separados. El acoplamiento entre activos requiere identidad contractual, unidades y actualización as-of; dos instancias independientes solo prueban aislamiento local, no sincronización de toda la cartera.

La memoria de pérdida podría representarse como una función sobre log(tau) con kernel de transferencia, en vez de tres cajas. Su ancho y decaimiento deben estimarse/validarse, no trasladarse a genes con valores arbitrarios. Los límites de riesgo duros continúan fuera de ese aprendizaje.

Una malla finita es una aproximación: necesita error de refinamiento, coste y criterio de soporte. El genoma podría parametrizar kernels o bases, pero el efecto debe verificarse con perturbaciones controladas sobre features, decisión, fill y retorno neto. Ninguna de las pruebas de esta ronda demuestra esa cadena.

No se incorpora una ecuación por estar asociada a un problema del milenio, a física avanzada o a una etiqueta cuántica. Debe especificar estimando, unidades, hipótesis, identificabilidad y ganancia incremental frente a un baseline. No hay promesa de omnisciencia, cobertura observable de 1 ns a 100 años ni actualización física por cada nanosegundo.

## 12. Matriz de ocho módulos

| Módulo | Aporte XXIV | Residual principal |
|---|---|---|
| 1. Ingestión/L2 | Guardas OHLCV/tick antes de escritura local | Identidad, cierre y cobertura del feed |
| 2. IA/modelos | Siembra correcta tras warmup; no se exporta rechazo como dato | Migración del contrato y paridad completa |
| 3. Multiactivo/tiempo | Aislamiento local probado; timestamp cero válido | Acoplamiento asíncrono y memoria física común |
| 4. Ejecución/red | No se reinicia ni opera el motor | Propagación de rechazo sin alterar salidas defensivas |
| 5. Riesgo/genoma | Cooldown no viola mínimo ni admite tau inválido | Kernel de pérdidas continuo y atribución genómica |
| 6. Estado/SO | Reset de memorias omitidas y guardas de overflow | Ciclo de vida por dominio; atomicidad del grafo |
| 7. Confluencia/cuántica | Semántica y observabilidad separadas de etiquetas | Hawkes calibrado, ausencia y readiness explícitas |
| 8. Backtesting/gobierno | Regresiones rojo→verde y diagnósticos abiertos separados | Evaluación causal y económica fuera de muestra |

## 13. Pruebas y resultados

| Grupo | Funcionales aprobadas | Nuevas en XXIV | Diagnósticos abiertos |
|---|---:|---:|---:|
| stateful_transition_contract | 22 | 22 | 0 |
| stateful_engine::tests | 10 | 0 | 0 |
| feature_exporter::contract_tests | 10 | 1 | 0 |
| train_dark_alpha::contract_tests | 8 | 0 | 0 |
| train_forest::contract_tests | 25 | 0 | 0 |
| label_evidence_contract | 25 | 0 | 0 |
| spectral_risk_contract | 3 | 0 | 0 |
| stateful_open_diagnostics | 0 | 5 diagnósticos | 5 |
| Total | **103** | **23 funcionales + 5 diagnósticos** | **5** |

Se observaron 11 fallos en las primeras 12 pruebas, y cuatro fallos adicionales de cooldown antes de su reparación. Son 15 regresiones distintas observadas rojo→verde, no 15 familias independientes de bugs. Las siete restantes del contrato principal amplían especificación/cobertura sin haberse ejecutado todas contra la versión original.

La primera prueba del reloj fallback, con u64::MAX y mínimo cero, era demasiado débil: pasó pese al wrap. Se fortaleció a 2^62 y mínimo uno; falló antes de la corrección y pasó después. Se registra esta limitación del test inicial para no presentar verificación insuficiente como garantía.

Una invocación con filtro stateful ejecutó diez unitarias y cero integraciones. Se repitió correctamente sin ese filtro para ejecutar las 22 del archivo; cero pruebas no se contabiliza como evidencia.

Comandos principales:

~~~text
cargo test -p god-engine-core --test stateful_transition_contract --test stateful_open_diagnostics --offline
cargo test -p god-engine-core --lib stateful_engine::tests --offline
cargo test --bin feature_exporter --bin train_dark_alpha --bin train_forest --offline
cargo test -p backtest-engine --test label_evidence_contract --test spectral_risk_contract --offline
cargo check --bin feature_exporter --bin god_engine --bin train_dark_alpha --bin train_forest --offline
~~~

Las repeticiones no se suman. No se ejecutó toda la suite del workspace, no se midió p99/latencia de producción, no se hizo un backtest económico ni comparación de PnL. Compilar y pasar fixtures no acredita autoevolución ni calidad predictiva.

## 14. Coste, compatibilidad y riesgo del cambio

La validación nueva añade trabajo O(1) por observación, sin recorrer historial ni alocar para la guarda. Esto describe complejidad, no nanosegundos medidos. La cadena de subestimadores conserva sus costes anteriores; no se certifica que el hot path no tenga otros cuellos de botella.

Se mantienen firmas unitarias antiguas como wrappers y se añaden métodos fallibles. El layout de StatefulEngine no incorpora campos nuevos. Cambia el resultado de inicialización después de velas y de entradas previamente inválidas. No se alteran archivos de modelos ni su formato, pero una futura puesta en producción debe verificar su compatibilidad estadística con esos cambios.

No se mezcló esta ronda con una migración de todos los predictores a un nuevo kernel temporal. No se suprimieron mecanismos de riesgo ni se sustituyeron datos ausentes por nuevas constantes.

## 15. Hoja de ruta de cierre, raíz a cima

1. Propagar aceptación/rechazo del evento por todos los nodos, con counters y razones; impedir que un rechazo parcial se convierta en decisión con mezcla de epochs. Preservar salidas defensivas.
2. Dividir reset de observaciones, riesgo y modelo, con snapshots versionados y pruebas de recuperación.
3. Reparar APIs Hawkes de reloj/constructor/lotes; separar señal descriptiva de intensidad estadística calibrada.
4. Definir contrato de observación común y modelos temporales con unidades; coordinar productor, trainer, evaluador y host.
5. Sustituir bandas rígidas solo con una memoria continua identificada y validada, manteniendo límites de riesgo explícitos.
6. Probar sincronización multiactivo as-of, escalas monetarias y de cantidad, missingness y separación entre dato observado y sintetizado.
7. Medir atribución genómica y efecto neto con costes/fills en evaluación separada de selección; no promover por tests de software.
8. Continuar las 150 fuentes Rust pendientes y el resto del inventario, sin declarar una auditoría integral por acumulación de búsquedas.

## 16. Artefacto y conservación

El [artefacto XXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIV_2026-09-25.json>) contiene fuentes, hashes, hallazgos, estado de reparación, pruebas y límites. Las adendas del [atlas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/ATLAS_ANALITICO.md>), [maestro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/INFORME_FORENSE_MAESTRO.md>) y XXIII preservan sus prefijos anteriores.

Esta ronda modifica StatefulEngine y el exportador v2; añade dos archivos de pruebas y documentación. El diff del exportador contra HEAD incluye XXIII y no se atribuye entero a XXIV. Cinco fuentes de referencia y 41 modelos conservan sus hashes de comparación.

Firecrawl Research se utilizó para buscar y ampliar referencias, con lectura de pasajes relevantes de dos artículos. La nota local .firecrawl/XXIV-hawkes-evidence.json contiene paráfrasis y enlaces públicos, está ignorada por Git y no incluye código/datos privados. La investigación fundamentó los requisitos de estabilidad/calibración y la decisión de no llamar Hawkes validado al módulo actual; no incorporó un estimador nuevo.

No hubo commit, push, merge, fetch, entrenamiento, promoción, modificación de cuentas/genomas, exportación de corpus real, descarga de mercado, órdenes, reinicio ni terminación de procesos del usuario. La certificación sistémica permanece abierta.

## 17. Validación documental final

El JSON se parseó correctamente. Se verificaron sumas de pruebas/cobertura, 49 referencias de evidencia con archivo y línea, y 23 enlaces locales del informe/adendas sin destinos ausentes ni líneas fuera de rango. Se recomprobaron 50 hashes: nueve fuentes del snapshot posterior y 41 modelos. Las cinco fuentes de referencia mantienen además su hash inicial.

Los prefijos históricos de atlas, maestro y XXIII conservan su SHA-256 tras normalizar CRLF a LF. Se agregaron respectivamente 2.584, 6.874 y 1.140 caracteres de cadena .NET; no se sustituyó el texto previo. Las longitudes no son bytes ni puntos de código Unicode. Los hashes y longitudes originales se incluyen en el artefacto.

Rustfmt --check pasa para las dos suites nuevas y el exportador. No se reformateó masivamente StatefulEngine: se inspeccionó su diff localizado y git diff --check pasó para las fuentes intervenidas y documentos versionados. Se mantienen tres warnings previos de evolution-engine y el import Arc sin uso de continuous_evolution_backtest. Las esperas de compilación/lock finalizaron sin terminar procesos ajenos. No hay una medición de latencia que pueda extrapolarse desde estos tiempos de compilación.

## 18. Continuación XXV — contratos de admisión y evidencia de rechazo

[Informe XXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXV_2026-09-25.md>) y [JSON XXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXV_2026-09-25.json>) añaden FMT-211..216. Se corrigen identidad TP/SL→EV→orden, dominios numéricos seleccionados y clasificación de rechazo frente a ejecución desconocida. Se documentan leverage fraccionario posterior, límites auxiliares inconsistentes y un mutante sembrado sin banda admisible.

Se mantiene el gate de promoción: la prueba aleatoria preexistente falló una vez y pasó aislada; no se declara resuelta. Semilla 199/tasa 0,5 reproduce un candidato cuya abstención sí tiene sentido. El objetivo no es hacer pasar todo filtro, sino justificarlo con la misma evidencia que llega al terminal.

Cobertura acumulada 141/289 Rust, 148 pendientes. 16 contratos y seis diagnósticos nuevos; nueve regresiones rojo→verde. La suite global no está verde por la incidencia de promoción. Los 41 modelos permanecen intactos y no hubo operaciones de mercado ni Git remotas. Los resultados de XXIV anteriores se conservan como evidencia histórica, sin reinterpretarlos como certificación de XXV.
