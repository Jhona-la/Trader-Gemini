# Auditoría científica XII — memoria temporal, integridad estadística y diagnóstico no mutante

Fecha: 2026-09-24. Corte local: main, HEAD 59a76de4. Continuación de XI. Se agregan hallazgos y correcciones; las rondas previas mantienen su contenido y sus resultados históricos. No se declara que todo el proyecto esté auditado, que el motor sea autoevolutivo en producción ni que exista ventaja económica o cuántica demostrada.

## 1. Resultado, alcance y clasificación de evidencia

Esta ronda corrige mecanismos reproducidos de estadísticas online y un diagnóstico que podía modificar una cuenta mientras decía consultarla. Añade una primitiva de memoria temporal definida por una ecuación diferencial y verificada por propiedades matemáticas. No cambia el reloj ni las features del motor vivo, ni integra un modelo neuronal nuevo.

Se comprobaron **85 tests distintos**, incluidos 24 tests nuevos. Siete pruebas numéricas reprodujeron fallos antes de los arreglos; tres controles de código fuente reprodujeron llamadas/afirmaciones defectuosas. Un cuarto control de fuente exigía un getter nuevo todavía inexistente: su fallo inicial no se contabiliza como otra reproducción de un bug numérico. Dos tests nuevos son caracterizaciones de discrepancias legacy que continúan abiertas, no pruebas de reparación.

**Cobertura conservadora acumulada: 101/289 Rust preexistentes leídos completos; 188 pendientes.** Se añaden seis lecturas íntegras: src/features/welford.rs, src/features/ewma.rs, src/features/mod.rs, crates/quantum-arena/src/feed_health.rs, crates/quantum-arena/src/tick_source.rs y src/bin/system_health.rs. Los archivos releídos y los cinco tests nuevos no se suman a esa cobertura. También se inspeccionó src/lib.rs y conexiones dirigidas; no se infla el acumulado con esa inspección. En particular, executor.rs y god_engine.rs no se acreditan como nuevas lecturas completas.

| Identificador | Prioridad | Estado en XII | Naturaleza y alcance |
| --- | --- | --- | --- |
| FMT-092 | P2 de API | Reparación local de dominio y transición | No se localizó uso productivo de update_decay; update sí tiene consumidores |
| FMT-138 | P2 de API | Corregido | Primera muestra exponencial introducía un prior cero no declarado |
| FMT-139 | P2 de API | Corregido en construcción | EWMA aceptaba ganancias/periodos matemáticamente inválidos |
| FMT-140 | P2 de arquitectura | Abierto, caracterizado | Copias exportadas de estadísticas mantienen contratos diferentes |
| FMT-141 | P1 diagnóstico | Afirmaciones corregidas; observación IPC pendiente | Estado privado del proceso consultor se presentaba como estado del motor |
| FMT-142 | P1 operativo | Ruta mutante retirada del diagnóstico | Consultar salud podía activar hedge en la cuenta |
| FMT-143 | P2 contable | Etiquetas/cobertura corregidas; reconciliación pendiente | Filas de income se presentaban como trades y win rate |
| FMT-144 | P1 integración | Abierto | Watchdog incompleto y recuperación sin identidad de conexión/evento |
| T35 | Investigación y primitiva | Baseline escalar probado, integración pendiente | Propagación temporal exacta y estados de grafo asíncrono |

No se renumera la matriz histórica de 305 puntos. Los siete nuevos IDs FMT-138–144 no representan siete incidentes observados en mercado: cada sección distingue reproducción, análisis estático y alcance no probado.

## 2. FMT-092 — corregir la representación, no sólo ocultar varianza negativa

### Contrato matemático y defecto reproducido

En modo acumulativo, Welford mantiene n, media y M2, donde M2 es suma de cuadrados centrados. La varianza muestral se consulta como M2/(n−1). En modo exponencial, el mismo campo m2 representa directamente una varianza filtrada V. Son objetos distintos aunque ambos tengan unidades de observación al cuadrado.

La API original permitía update_decay y luego update. La primera convertía M2 a V y activaba is_decay; la segunda seleccionaba su rama sólo por count>=2000. Si el contador era menor, volvía al algoritmo de suma de cuadrados, pero variance seguía leyendo m2 como varianza. Una cifra finita podía por ello tener una interpretación equivocada.

Reproducción en [welford_mode_contract.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/welford_mode_contract.rs>): muestras 2 y 4, actualización exponencial con 6 y alpha=0,5, seguida de update(8). La media antigua resultaba 5,6666666667. Con la política predeterminada explícita de la rama exponencial, el valor esperado es 4,5034982509. La prueba falla antes y pasa después.

Además, alpha>1 invertía el peso de la historia. Tras 3 y 7, update_decay(100,2) producía count=−1, media=195 y m2=−18058. El clamp de variance devolvía cero y ocultaba la ruptura del contrato. Esto no era “adaptación rápida”: los pesos ya no definían una media convexa.

### Reparación y compatibilidad

En [welford.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/welford.rs:32>), update entra en la rama exponencial si is_decay ya está activo, independientemente del contador. Conserva la ganancia histórica 2/2001 para ese método. Quien elige otra ganancia debe continuar llamando explícitamente update_decay; no se infiere un alpha a partir de un contador cuyo significado ha cambiado.

update_decay rechaza alpha fuera de (0,1], además de no finitos. El rechazo no modifica ningún campo, como se comprueba en la prueba de estado completo. La transición automática de la ruta exclusivamente update al superar 2000 observaciones conserva su media, varianza y contador anteriores: hay una prueba de referencia en esa frontera.

Se mantiene la conversión histórica desde varianza muestral a estado inicial del filtro. No se afirma que ello reconstruya exactamente todos los pesos de una distribución empírica. La API pública sigue permitiendo modificar campos; no queda blindada frente a corrupción externa del estado.

### Unidades, contador y límites no reparados

En estado exponencial, la recurrencia V'=(1−a)V+a(x−m)(x−m') conserva una varianza alrededor de la media filtrada. No entrega automáticamente una varianza insesgada bajo dependencia temporal ni una incertidumbre de predicción.

count conserva una semántica legacy de masa con un cap 1/alpha en update_decay, y se conserva en update tras el cambio de modo. No es número de observaciones recientes ni tamaño muestral efectivo. Para pesos geométricos normalizados estacionarios, el cociente (sum w)^2/sum(w^2) resulta (2−alpha)/alpha, distinto de 1/alpha. Incluso ese cociente no mide independencia de datos correlacionados. Por ello no se utilizó count para inventar una garantía estadística nueva.

Siguen sin resolverse aquí: overflow para secuencias extremas, campos públicos corruptos, piso nominal de z_score y política de memoria de 2000 eventos. Retirar esa política del flujo operativo exige migrar los consumidores, no sustituir silenciosamente su estimando.

## 3. FMT-138 — una primera observación aprendía también de un cero inexistente

### Mecanismo

El estado recién creado tiene media cero por representación informática. La primera llamada update_decay(10.0, 0.2) aplicaba inmediatamente la recurrencia exponencial: media=2 y varianza=16. Eso equivale a atribuir masa 0,8 a una observación/prior cero que el llamador nunca proporcionó.

Una serie constante de 10 generaba así dispersión durante el arranque. Podría justificarse un prior declarado con peso e incertidumbre; aquí no había tal contrato. Se mezclaba “estado vacío” con “evidencia de cero”.

### Corrección y prueba

La primera muestra válida fija media=x, varianza=0, count=1 e is_decay=true. La ganancia gobierna la incorporación de observaciones posteriores. Un test verifica esa condición inicial y otro compara cinco actualizaciones con pesos empíricos explícitos, incluyendo alpha=1 y cambios de ganancia.

La equivalencia ponderada sólo se exige cuando el estimador se inicia directamente en modo exponencial. No se extiende falsamente a la conversión del estimador acumulativo muestral. Varianza cero con una única muestra significa dispersión observada nula; no significa que la distribución futura carezca de riesgo.

## 4. FMT-139 — parametrización EWMA inválida y contrato temporal nuevo

### Ganancia por evento: validación sin reemplazos arbitrarios

El constructor del crate aceptaba NaN, alpha=2 y periodos como −1. NaN contamina la segunda actualización; alpha>1 permite extrapolar fuera del intervalo entre estado y muestra; N=−1 genera división por cero en 2/(N+1). Ninguno corresponde al suavizado que declara la API.

[ewma.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/ewma.rs:22>) añade try_new y try_from_period con Result. Se exige alpha finito en (0,1] y periodo finito N>=1. No se sustituye una configuración inválida por 0,1 o 14. Los constructores históricos new/from_period conservan la firma, pero ahora fallan explícitamente con panic ante configuración inválida; el uso configurable debe migrar a la API fallible. Es un cambio semántico documentado.

Las rutas localizadas en Omni usan periodos literales válidos. MarketCorrelationHeatmap acepta un periodo de su llamador; un uso externo inválido ahora será rechazado, no reparado silenciosamente. No se encontró evidencia de una configuración inválida real en producción. Los tests de alpha NaN, alpha>1 y periodo negativo fallaron antes y pasan después; la secuencia válida 100→110 con N=9 sigue produciendo 102.

La validación de constructor no impide que un consumidor escriba después un alpha inválido, pues los campos siguen públicos. La actualización legacy conserva su comportamiento para observaciones válidas y no incorpora un reloj físico automáticamente.

### De eventos a tiempo físico: derivación de la primitiva

Para un estado m que relaja hacia una entrada x retenida constante:

```text
dm/dt = (x-m)/tau
m(t+dt) = exp(-dt/tau)*m(t) + (1-exp(-dt/tau))*x
alpha(dt,tau) = -expm1(-dt/tau)
```

tau es una constante de tiempo, no un periodo de muestras ni una clase de estrategia. Tras tau, queda una fracción e^-1 de la desviación inicial bajo entrada constante. Esta solución no requiere integrar cada nanosegundo.

[ewma.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/ewma.rs:58>) incorpora update_elapsed con dt>=0 y tau>0, ambos finitos en milisegundos. Rechaza observación no finita sin mutación. La primera observación establece la condición inicial incluso con dt=0; después dt=0 no cambia el estado. La API no sobrescribe alpha de la ruta por eventos. Ante resultado no representable, devuelve error sin publicar ese resultado.

**Causalidad de la entrada retenida:** el llamador debe decidir qué x estuvo vigente durante el intervalo. No puede usar retrospectivamente una observación recién llegada como si se hubiera conocido desde el principio sin declarar esa hipótesis. La primitiva resuelve una ecuación dada; no define la semántica del feed, no deduplica ticks, no detecta timestamps fuera de orden y no aprende tau.

### Qué demuestran las pruebas temporales

Las siete pruebas de [ewma_physical_time_contract.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/ewma_physical_time_contract.rs>) verifican el resultado analítico, invariancia al dividir un intervalo de entrada constante, cambio conjunto de unidades de dt/tau, rechazo de datos inválidos, ausencia de avance con dt=0 y límite de una separación temporal muy grande.

También se prueba dt=1 ns con tau=100 años convencionales de 365,25 días: desde estado cero hacia uno, la contribución es positiva y coincide con dt/tau dentro de tolerancia. expm1 evita perder esa ganancia por restar dos números casi iguales. No prueba que sumar esa contribución a cualquier estado no nulo sea representable: el redondeo de f64 sigue imponiendo límites.

**No se promete resolución de mercado de 1 ns, datos identificables de 100 años ni cómputo completo cada nanosegundo.** No se conectó update_elapsed a ningún consumidor operativo. La migración debe probar significado del reloj, señales retenidas, recalentamiento y paridad del modelo entrenado/servido. Cambiar una feature por eventos a otra por tiempo puede exigir reentrenamiento.

## 5. FMT-140 — duplicación exportada y reparaciones que no llegan a todas las rutas

Se leyeron íntegramente [welford.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/features/welford.rs>), [ewma.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/features/ewma.rs>) y [mod.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/features/mod.rs>). src/lib.rs exporta features, mientras los crates operativos utilizan feature_engine. Son implementaciones distintas, no aliases de un contrato común.

El Welford de src/features no descarta NaN: [10,NaN,20] deja su media NaN. El del crate descarta la muestra inválida y conserva media 15. El legacy tampoco cambia de modo a 2000 observaciones. El EWMA legacy transforma parámetros inválidos a 0,1 o periodo 14; el crate ahora ofrece rechazo explícito. Diferencias válidas de diseño deben tener tipos/nombres y contratos distintos, no quedar ocultas tras el mismo nombre de estructura.

[legacy_statistics_diagnostics.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/legacy_statistics_diagnostics.rs>) compila las copias reales mediante path e incluye dos caracterizaciones nuevas y tres tests ya presentes en esos archivos. Las caracterizaciones pasan porque detectan el desacuerdo: no demuestran que se haya reparado. No se editaron las copias legacy ni se igualaron algoritmos que tienen políticas de memoria distintas sin inventariar sus consumidores.

La ruta viva identificada del volumen institucional utiliza feature_engine y Omni del crate también lo hace. El alcance acreditado de la copia legacy es una API exportada; no se atribuye sin evidencia a las órdenes actuales. Cierre: inventario de usos, contrato de migración y una fuente de implementación por estimando, conservando adaptadores de compatibilidad con semántica explícita.

## 6. FMT-141 — un proceso consultor no observa la memoria privada del motor

feed_health.rs contiene un AtomicBool estático inicializado a false. system_health es otro ejecutable. Su lectura de ese átomo observaba su propia copia, no la del proceso god_engine. Como el consultor no ejecuta el watchdog, el valor inicial llevaba a imprimir “vivo” aunque el motor estuviera detenido o su feed fallara.

El mismo patrón aparece en órdenes: OrderExecutor::new construye un OrderRegistry nuevo. Sus totales locales no son el registro compartido del motor ni una consulta de órdenes abiertas al exchange. Un cero en ese objeto no acredita ausencia de órdenes reales.

En [system_health.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/system_health.rs:103>) se retira la falsa inferencia y se publica DESCONOCIDO por falta de snapshot IPC. La sección de órdenes conserva sus números, pero identifica que pertenecen a la consulta y no al motor. El genoma del almacén tampoco prueba qué generación cargó un proceso vivo; su encabezado lo explicita y la ausencia de archivo ya no afirma “baseline en operación”.

Se preservan las secciones del diagnóstico. No se fabricó un puente IPC usando un archivo sin identidad/edad ni se convirtió falta de evidencia en estado rojo o verde. Para cerrar la observabilidad real se requiere snapshot con PID/identidad de proceso, generación, timestamp, secuencia y antigüedad verificables. La corrección de las afirmaciones está probada por fuente; el IPC sigue pendiente.

## 7. FMT-142 — la consulta de salud podía cambiar el modo de la cuenta

La sección de modo llamaba ensure_hedge_mode con paper trading desactivado. En executor.rs esa función realiza un GET y, si la cuenta está en one-way, envía un POST para activar hedge. El diagnóstico admite --live: no era sólo un posible cambio de simulación.

**Impacto:** una operación presentada como inspección tenía autoridad de mutación sobre la cuenta. Que el exchange pueda rechazar el cambio con posiciones abiertas no vuelve de sólo lectura a la ruta. No se ejecutó system_health para comprobarlo, ni se consultaron credenciales o el exchange.

Se añade [executor.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:584>), fetch_hedge_mode: sólo GET firmado, parseo booleano estricto y error cuando el ejecutor está en simulación sin evidencia remota. No escribe is_hedge_mode ni envía POST. El diagnóstico consume ese método y muestra hedge/one-way como observación, sin exigir ni cambiar la política de la cuenta.

ensure_hedge_mode permanece intacto para sus consumidores explícitamente mutantes. Así se separan consulta y configuración sin retirar capacidad operativa. La nueva consulta reutiliza el helper GET existente, con su timeout/reintento. No se ha validado una respuesta real ni un servidor mock; se verificó el código, su compilación y controles de fuente de no-POST/no-store. No se presenta esa evidencia limitada como prueba completa de red o autorización.

## 8. FMT-143 — una respuesta contable no define operaciones cerradas

system_health pedía una sola página de hasta 1000 entradas de income desde una fecha de siete días atrás. Sumaba REALIZED_PNL, comisiones y funding, pero llamaba trades al número de filas REALIZED_PNL y WR a la fracción de filas positivas.

En el propio código no hay agrupación por identidad de operación, fills parciales, apertura/cierre ni reparto de costes entre operaciones. Por tanto, la fracción calculada no acredita un win rate neto por trade. Tampoco hay paginación en ese llamador: una suma de respuesta no prueba cobertura completa de la ventana.

Se preservan los cálculos y se corrigen sus etiquetas: neto de filas, filas positivas y número de filas REALIZED_PNL. Se muestra tamaño de respuesta, alcance de una página y aviso al alcanzar el límite. Si no hay filas REALIZED_PNL se siguen mostrando comisiones/funding; ya no se afirma ausencia de operaciones durante toda la ventana.

Permanece pendiente la contabilidad reconciliada: paginación con completitud explícita, deduplicación, corte temporal común e identidad de trade. Los cocientes de arrastre y funding siguen siendo descriptivos de filas; sus denominadores cercanos a cero y compensación entre cobros/cargos requieren un contrato adicional. No se convirtió aquí una proporción de filas en fitness ni se recalculó rentabilidad.

## 9. FMT-144 — vida del transporte, frescura del dato y recuperación no son equivalentes

**Análisis estático de rutas, sin simular desconexión en producción.** En el host inspeccionado, el timeout del lector WS llama stall. Las ramas de error de socket y fin de stream salen para reconectar sin esa publicación. La espera de mensajes tampoco demuestra por sí sola que llegue información de mercado utilizable.

Por otra parte, el procesador limpia el flag cuando event_time>0 y escribe latencia. Un evento encolado puede ser anterior al stall. No hay en ese booleano identidad de sesión ni prueba de que el evento certifique recuperación de la conexión que falló. El test lógico mínimo es la secuencia “watchdog marca; cola entrega evento antiguo; procesador limpia”. El orden atómico de memoria no añade causalidad o edad al evento.

Hay controles adicionales: latencia y permisos de entrada en el núcleo. Pueden bloquear una orden y no se demostró una apertura insegura efectiva. La existencia de ellos tampoco vuelve completo el contrato del watchdog. El umbral real depende de entorno/configuración; los textos antiguos de “5s” no describen universalmente esa política.

No se modifica god_engine.rs por contener trabajo concurrente. Cierre propuesto: salud por conexión/fuente requerida, estado explícito de reconexión, timestamp de último evento válido, identidad de sesión y regla de recuperación que rechace mensajes anteriores al fallo. Evitar que la actividad de una fuente sustituya la frescura de otra. Esta reparación requiere integración con el interlock, no cambiar un Ordering de AtomicBool.

## 10. T35 — dinámica temporal exacta y grafo asíncrono: integración con criterio

La consulta bibliográfica usó Firecrawl Research Index; la CLI no estaba disponible y se utilizó el conector. No se subió código. La lectura de fuentes distinguió ecuación implementada, propiedades numéricas y propuestas de modelos nuevos.

El trabajo de [Meng sobre momentos centrales online](https://arxiv.org/abs/1510.04923) fundamenta la separación entre suma central y momento normalizado, y presenta recurrencias centradas para evitar la resta inestable de segundos momentos brutos. Se leyeron dos pasajes del cuerpo; no se usa esa referencia para prometer inmunidad universal a overflow ni adaptación de régimen.

[Temporal Graph Neural Networks for Irregular Data](https://arxiv.org/abs/2302.08415) distingue propagación entre observaciones e incorporación de nodos observados. Los pasajes verificados describen soluciones cerradas de sistemas lineales y dinámicas exponenciales/periódicas, con restricciones para estabilidad. Su uso de estados latentes no prueba causalidad económica ni ventaja de trading.

Esta distinción orienta T35: propagar un estado temporal y asimilar evidencia son operaciones diferentes. La primitiva escalar implementada es un baseline clásico con solución analítica, no una implementación de ese modelo neuronal ni una demostración “cuántica”.

Familia de alternativas conservadas para una evaluación posterior:

- [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907): resumen sobre dinámica continua y tiempos de observación; no se verificaron sus experimentos completos.
- [Functional Latent Dynamics](https://arxiv.org/abs/2405.03582): resumen que propone curvas latentes para evitar resolver una ODE compleja en cada paso. Sus costes/ventajas no se trasladan a este sistema sin medirlos.
- [Graph Neural Flows](https://arxiv.org/abs/2410.14030): resumen sobre dependencias entre series irregulares y curvas continuas; una notación causal no identifica por sí sola efectos causales en mercados.
- [ASTGI](https://arxiv.org/abs/2509.23313): resumen sobre observaciones asíncronas, vecinos y consultas temporales. Es candidato, no módulo integrado.
- [ASeer](https://arxiv.org/abs/2308.16818): resumen en tráfico irregular que combina difusión espacial y codificación temporal. La transferencia al mercado es una hipótesis experimental.

No se añaden ecuaciones de problemas del milenio por prestigio. Para cualquier teoría importada se exige correspondencia entre variables, unidades, hipótesis observables, problema que resuelve, baseline, coste y criterio de falsación. Una ecuación sofisticada no repara un snapshot inexistente, una etiqueta errónea ni un gen sin consumidor.

### Experimento y criterio de promoción

Comparar el filtro por eventos, el filtro temporal analítico y un candidato de estado más rico con el mismo tape causal, presupuesto computacional y señales disponibles. Registrar por nodo tiempo de evento/recepción, fuente, escala, máscara de dato y generación. Evaluar perturbaciones de empaquetado, gaps, eventos fuera de orden, reinicios y cambios de unidad.

Sólo exigir invariancia al empaquetado cuando el estimando la tenga: para entrada constante retenida sí; para una medida ponderada por número de transacciones, duplicar eventos cambia legítimamente la población. Separar integración de estado, aprendizaje del parámetro tau y selección fuera de muestra. Medir error, soporte, calibración, latencia p50/p99 y después utilidad neta/abstenciones, sin seleccionar repetidamente contra el mismo holdout.

Las escalas no necesitan etiquetarse scalping/swing. Necesitan una coordenada temporal común y soporte identificable. Consultar una ley a 100 años no equivale a disponer de evidencia a 100 años; almacenar timestamps en nanosegundos no obliga ni habilita procesar todo el sistema mil millones de veces por segundo.

## 11. Topología diagnóstica raíz–cima

| Arista | Dato que debe conservar | Estado de esta ronda |
| --- | --- | --- |
| Fuente → nodo raíz | Identidad, sesión, reloj, validez | Watchdog FMT-144 abierto; TickEvent no prueba paridad por compartir tipo |
| Nodo raíz → estadística | Estimando, unidades, memoria | Welford reparado localmente; EWMA temporal opt-in |
| Estadística → nodo de decisión | Features versionadas, soporte, incertidumbre | Nueva semántica temporal todavía no conectada |
| Decisión → nodo terminal | Factibilidad, cantidad/riesgo finales | FMT-113 permanece abierto |
| Terminal → aprendizaje | Atribución causal y resultado reconciliado | Hallazgos previos de ledger/feedback siguen abiertos |
| Motor → diagnóstico | Snapshot interproceso con edad e identidad | Se elimina falso verde; IPC sigue pendiente |
| Diagnóstico → exchange | Lectura sin órdenes/configuración | Ruta de cambio de modo reemplazada por GET |

TickSource sólo abstrae la entrega de TickEvent. Las búsquedas encontraron su implementación mock, no una implementación operativa del trait. El registro porta timestamp sin contrato de unidad en el tipo, y no incorpora por sí mismo calidad, procedencia o secuencia de sesión. Su test comprueba entrega de un evento y fin, no identidad backtest–producción. No se renumera como otro hallazgo lo ya relacionado con contratos de replay previos.

## 12. Verificación, límites y continuidad

| Grupo | Tests aprobados | Interpretación |
| --- | ---: | --- |
| feature-engine unitarios | 57 | Regresiones existentes, incluida DFA de rondas previas |
| welford_mode_contract | 7 | Cuatro fallaban antes |
| ewma_configuration_contract | 4 | Tres fallaban antes |
| ewma_physical_time_contract | 7 | Contrato de capacidad nueva; no mejora económica |
| legacy_statistics_diagnostics | 5 | Dos diagnósticos nuevos abiertos y tres tests existentes incluidos |
| system_health_source_contract | 4 | Controles textuales acotados, no tests de HTTP |
| tick_source::tests | 1 | Contrato mock existente |
| Total distinto | 85 | No se cuentan ejecuciones repetidas |

cargo check --bin god_engine --bin system_health --offline pasó sin ejecutar ninguno. Persisten tres warnings previos de evolución: latest_ts, mode y RealWfOutcome.trades. La suite de feature-engine se ejecutó completa; no la de todo el workspace. No hubo backtest económico, requests autenticadas, medición P99, promoción de genoma, reinicio ni despliegue.

Los hashes y el manifiesto de cobertura constan en el [auditoria_fundamentos_XII_2026-09-24.json](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XII_2026-09-24.json>). Se mantienen sin editar host/core/risk concurrentes y copias legacy. La documentación histórica recibe sólo adendas. No se hizo commit, push, merge ni fetch; main local no acredita sincronización remota.

Prioridades siguientes: contrato de salud por fuente/generación y snapshot de diagnóstico; migración de memoria temporal con features versionadas; presupuesto sobre cantidad final; feedback evolutivo íntegro y atribuido. Las reparaciones estadísticas y las etiquetas honestas hacen estos contratos comprobables, pero no sustituyen su implementación.

### Integridad final del entregable

El JSON se parseó correctamente y coinciden los diecisiete hashes de archivos intervenidos, leídos o protegidos. Todos los enlaces locales de este informe existen y sus números de línea quedan dentro del archivo. Los prefijos previos completos del atlas, maestro y rondas IV/XI mantienen sus SHA-256 tras normalizar CRLF a LF: las adendas no eliminaron ni reescribieron el contenido anterior. git diff --check y rustfmt --check pasaron en el alcance comprobado; no se reformatearon los grandes consumidores concurrentes ni las copias legacy. El commit observado sigue siendo 59a76de4.


## Continuación aditiva XIII — persistencia y recuperación

La [ronda XIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XIII_2026-09-24.md>) amplía el análisis con FMT-145–154 y T36. No modifica los resultados históricos de XII. Corrige contratos numéricos de OHLCV, semántica de errores de persistencia y etiquetas desconocidas de intenciones; caracteriza checkpoints aún débiles y una fase de entrenamiento vacía. Acredita 28 tests focalizados (tres confirman fallos abiertos), compilación sin ejecución y seis lecturas completas nuevas: cobertura acumulada 107/289, 182 pendientes. Persistencia causal y equivalencia de replay no están certificadas.
