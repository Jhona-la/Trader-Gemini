# Auditoría de fundamentos científicos VIII — Continuidad de aprendizaje y contratos de generación

Fecha de intervención: 2026-09-24. Revisión local: HEAD 59a76de4. Anexo aditivo a las rondas I–VII; no sustituye sus evidencias ni reinterpreta sus estados históricos como estados actuales.

## 1. Dictamen y alcance real

Esta ronda identifica seis hallazgos adicionales, FMT-124 a FMT-129. Repara localmente tres y la sobrescritura de sigma ya registrada como FMT-013. Añade dieciséis tests; cinco demostraron el defecto antes de la reparación. La verificación final de esta intervención suma **25 tests distintos aprobados** y cargo check satisfactorio del host.

El resultado NO acredita rentabilidad, autoevolución integral, ejecución cuántica ni cobertura de todo el repositorio. Se releyeron completos CMA, el bucle auxiliar de evolución y MetaEvolver, que ya estaban incluidos en la cobertura anterior. Por tanto, la cobertura acumulada de lectura completa sigue en **92 archivos Rust preexistentes distintos de 289**. Los dos nuevos archivos de prueba se cuentan aparte. Los 1.119 archivos versionados y 24 manifiestos Cargo son inventario, no certificación de revisión exhaustiva.

**Distinción operativa decisiva:** [el host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1994>) llama a OnlineEvolutionDaemon::run_online_learning_loop. La búsqueda de referencias Rust no encontró un llamador operativo de [EvolutionEngine::start_evolution_loop](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/lib.rs:62>). Los usos localizados de CmaEsOptimizer son ese bucle y sus pruebas. La reparación compila con el host, pero eso no demuestra que el host ejecute CMA. No se conectó automáticamente una segunda autoridad de promoción al sistema vivo.

Los defectos del daemon previamente documentados siguen siendo prioritarios. Una mejora de una ruta auxiliar no puede presentarse como explicación resuelta de la diferencia entre backtest, demo y producción. La ausencia de referencias estáticas tampoco es una prueba universal sobre binarios externos o consumidores fuera de este repositorio.

## 2. Paradigma de grafo vivo y topología verificable

### 2.1 Dos rutas que no deben confundirse

~~~mermaid
flowchart TD
  H["Nodo raíz: host god_engine"] --> D["OnlineEvolutionDaemon: llamada localizada"]
  D --> E["Evaluación / promoción / watchdogs: pendientes III y V"]

  P["Supervisor heurístico: win rate y configuración"] --> S["Nivel externo de exploración s"]
  S --> C["Sigma efectivo: sigma aprendido × s nuevo / s previo"]
  C --> M["Muestreo híbrido CMA/PSO"]
  M --> V["Vectores normalizados y evaluación por ID"]
  V --> B["Preflight de generación completa"]
  B -->|rechazo| R["Sin update ni promoción de esa generación"]
  B -->|válida| U["Selección, caminos, covarianza y sigma"]
  U --> C
  U --> G["Nodo de decisión: gates de candidato"]
  G --> T["Nodo terminal: solicitud a GenomeEnvelope"]

  A["Ruta CMA auxiliar: sin llamador operativo localizado"] -. contexto .-> P
~~~

El diagrama representa enlaces observados en código, no actividad medida de procesos ni confirmación de que una promoción haya ocurrido. Los nodos terminales de ejecución de órdenes no se modificaron. El gate del almacén no sustituye un experimento económico independiente; FMT-058 sigue abierto.

La verificación de una telaraña de componentes requiere, por cada arista, identidad, unidad, reloj, versión, origen de evidencia, política de rechazo y autoridad. En esta ronda se reforzó identidad estructural dentro de una generación y continuidad del estado de búsqueda. No se añadieron timestamps de mercado al optimizador ni un linaje de evaluación completo.

### 2.2 Estado consolidado de esta intervención

| Punto | Prioridad de seguimiento | Estado local | Qué se ha demostrado |
|---|---|---|---|
| FMT-013, anterior | P1 de diseño; ruta auxiliar | Reparada la sobrescritura | Dos generaciones conservan sigma aprendido; el supervisor compone cambios relativos |
| FMT-124 | P2 actual; alto impacto si se integra | Reparado el contrato estructural | IDs únicos/completos, dimensiones y genes válidos antes de mutar |
| FMT-125 | P2 actual; sesgo de búsqueda | Reparada la atracción sin evidencia | Memorias PSO no evaluadas o con fitness no finito no atraen muestras |
| FMT-126 | P2 actual; error de ecuación | Reparada la amortiguación | d_sigma coincide con la expresión contrastada |
| FMT-127 | P2 actual; comparabilidad | Abierto | Las memorias históricas no identifican el objetivo/ventana con que se midieron |
| FMT-128 | P2 actual; adaptación aparente | Abierto, ahora observable | Lambda efectiva se mantiene aunque cambie la solicitud de población |
| FMT-129 | P2 actual; incoherencia de escala | Abierto | Muestreo y actualización pueden usar sigmas distintos |
| FMT-012 | P1 de diseño, anterior | Abierto | Persiste blanqueamiento diagonal con covarianza completa y otras desviaciones |
| FMT-048/058/059 | Anteriores | Abiertos | Abstención, promoción y cronología/equity no quedan resueltas |
| FMT-113/117 | Anteriores | Abiertos | Cantidad ejecutable y soporte temporal siguen requiriendo cierre |

P2 expresa la exposición demostrada en la ruta auxiliar, no inocuidad matemática. No se crea un incidente de producción basándose sólo en una función disponible. Tampoco se marca como resuelto un hallazgo global porque pase un test local.

## 3. FMT-013 — Reparación de continuidad de sigma y autoridad del supervisor

### 3.1 Causa y significado del cálculo

El objeto CMA sobrevivía entre iteraciones, pero el caller imponía sigma=mutation_rate antes de cada población. Se conservaban covarianza y caminos; se descartaba precisamente la magnitud de exploración aprendida. La existencia de estado persistente no demostraba adaptación efectiva.

Sigma mide dispersión de propuestas en coordenadas genómicas normalizadas; no es probabilidad de mutación, volatilidad monetaria, duración temporal ni confianza estadística. La regla CSA modifica log(sigma) según la longitud del camino evolutivo respecto de una referencia gaussiana. La [formulación de Hansen](https://arxiv.org/abs/1604.00772) permite contrastar esta recurrencia; las alteraciones del híbrido deben validarse por separado.

### 3.2 Composición implementada

[apply_exploration_level](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:183>) separa dos cantidades: sigma aprendido y nivel externo s. Antes de muestrear:

~~~text
sigma_muestreo(t) = sigma_aprendido(t) · s(t) / s(t−1)
sigma_aprendido(t+1) = sigma_muestreo(t) · exp(a(t))
a(t) = (c_sigma/d_sigma) · (norma_camino / norma_referencia − 1)
~~~

La segunda línea describe la intención CSA; la implementación conserva clamps y geometría no canónicos, detallados en FMT-012/129. La primera es una política de composición explícita de esta intervención, no un teorema de optimalidad. Si s no cambia, la operación es un no-op exacto, sin redondeo adicional. Cambiar 0,2 a 0,3 multiplica la dispersión aprendida por 1,5; repetir 0,3 no vuelve a multiplicarla. Volver a 0,2 revierte sólo el factor de supervisor, no el aprendizaje intermedio.

Esta operación no reinicia media, covarianza, caminos, generación ni memorias PSO. Un verdadero reinicio exige política independiente con causa registrada. Los niveles y sigmas deben ser finitos y positivos. Si el cociente intermedio desborda pero el resultado es representable, se calcula en logaritmos. Si el resultado real desborda o subdesborda a cero, se rechaza sin modificar sigma ni la referencia privada del supervisor.

El [caller](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/lib.rs:187>) consume el resultado y registra antes/después, nivel previo/nuevo, generación y población efectiva/solicitada. El constructor de configuración dinámica usa try_new; un error impide empezar esa generación.

### 3.3 Evidencia experimental y límites

Traza determinista obtenida con semilla 93, dos dimensiones, cuatro candidatos y PSO desactivado:

| Generación | Sigma al muestrear | Sigma tras selección/CSA | Sigma conservado para el siguiente muestreo |
|---|---:|---:|---:|
| 1 | 0,2000000000000000 | 0,1837367367778880 | 0,1837367367778880 |
| 2 | 0,1837367367778880 | 0,1495963716456293 | 0,1495963716456293 |

No se exige que sigma siempre decrezca: esa sería otra rigidez incorrecta. El test prueba conservación de lo aprendido en este fixture, no convergencia económica. También se comprueban cambios relativos, repetición idempotente, inversión del ajuste, geometría intacta y rechazo transaccional de entradas inválidas.

La política existente que genera s sigue usando win rate agregado, un umbral derivado de ml_threshold_long, factor 1,5 y techo 0,5. No se ha demostrado que eso detecte cambio de régimen ni que una tasa de aciertos por sí sola mida utilidad neta. Tampoco se identifica toda observación con su genoma activo. Corregir la composición evita borrar aprendizaje; no legitima automáticamente el supervisor.

## 4. Hallazgos nuevos: mecanismos, impacto y condiciones de cierre

### FMT-124 — Una generación carecía de contrato estructural y podía seleccionar repetidamente el mismo candidato

**Evidencia.** update recibía vectores y tuplas con índices sin verificar que formasen un lote completo uno-a-uno. Los accesos a población y memorias se realizaban por ese índice. Un ID repetido podía ocupar varias posiciones parentales; un ID fuera de rango causaba panic; un genoma mal dimensionado o no finito podía entrar en memorias y actualizaciones.

**Contraejemplos reproducidos.** Con lambda=4 y mu=2, sustituir el ID de la segunda evaluación por 0 produjo generación=1 en lugar de rechazo. El mismo candidato acumuló los dos pesos parentales aunque no fueran dos observaciones distintas. Cambiar el primer ID a 4 causó acceso fuera de límites. Un primer genoma con NaN también avanzó la generación. El fitness finito no subsana la invalidez del objeto al que se atribuye.

**Impacto lógico.** Hay tres niveles: disponibilidad del proceso por panic, contaminación persistente de memoria y sobreponderación de evidencia. Un promedio ponderado con pesos normalizados no corrige duplicidad de identidad: sólo normaliza el error. Una evaluación que llega antes o después debe conservar su asociación, sin ganar peso por el orden de llegada.

**Reparación.** [validate_batch](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:360>) exige estructuras internas de tamaño compatible, sigma positivo finito, exactamente lambda genomas de dimensión n, genes finitos en [0,1] y exactamente lambda evaluaciones con índices únicos dentro del rango. El preflight ocurre antes de modificar scores o estado. Las dos APIs devuelven UpdateOutcome: Updated, InsufficientValidParents o Rejected con CmaInputError.

try_new rechaza dimensión cero, lambda menor que dos y sigma no finito/no positivo. El constructor histórico permanece para parámetros estáticos válidos y falla explícitamente si se viola ese contrato. No se sustituye configuración inválida por una constante financiera inventada.

**Pruebas.** Casos rojo→verde de duplicación, índice inválido y genoma inválido; además, lotes incompletos, matriz de forma corrupta, permutación del orden de resultados y preservación de todos los campos y scores ante rechazo estructural. La prueba de genoma ahora recorre NaN, dimensión corta y valor superior a uno.

**Límite preciso.** Esto no prueba que el score corresponda al hash del genoma ni a la misma cinta, comisión, reloj o ventana. Un caller puede entregar un índice válido con datos semánticamente equivocados. Los campos públicos todavía permiten corrupción numérica interna fuera del contrato; no se acredita inmunidad ante toda mutación arbitraria del objeto. Se conserva el tratamiento anterior de fitness no finitos después del preflight: puede marcar/ordenar scores y devolver padres insuficientes sin mutar el optimizador.

**Cierre restante de identidad científica.** Incorporar identificadores de propuesta y evaluación, hash del genoma, versión del decodificador y del simulador, datos as-of, costes, entorno, semilla y objetivo. Rechazar resultados de otra generación y repetir tests con ejecución paralela, resultados tardíos y reintentos.

### FMT-125 — PSO atraía hacia ceros de almacenamiento antes de observar resultados

**Evidencia.** El constructor inicializaba personal_bests y global_best como vectores cero y sus fitness como f64::MIN. sample_population aplicaba ambas fuerzas sin consultar esas marcas. El caller auxiliar sustituía global_best por la media, pero tampoco había medido su fitness; las memorias personales seguían en cero.

**Significado del sesgo.** En un hipercubo normalizado, cero es la frontera inferior de cada gen, no un valor neutral universal. Añadir c1·r1·(0−x) introduce contracción sistemática sin evidencia de utilidad. Al decodificar genes heterogéneos puede favorecer mínimos de parámetros con significados distintos. No se infiere que todos reduzcan riesgo o aumenten operaciones: eso depende del mapa genómico.

**Reproducción.** Se añadió RNG inyectable sin alterar inicialmente el algoritmo. Con semilla 81, los mismos gaussianos, media 0,5 y sin observaciones, el primer candidato pasó aproximadamente de (0,77398; 0,63288) sin PSO a (0,62391; 0,53816) con fuerzas no evaluadas. La desigualdad fue una regresión roja, no una inspección visual del código.

**Reparación.** Las [fuerzas cognitiva y social](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:298>) sólo se aplican si su fitness asociado es finito y mayor que la marca excluida. Se siguen consumiendo las mismas variables aleatorias, permitiendo comparación con ruido común. El término de inercia no se elimina; en la inicialización sus velocidades son cero.

**Pruebas y límites.** Con memorias no observadas, activar coeficientes PSO deja exactamente las mismas muestras del baseline bajo idéntico RNG. Fitness NaN e infinitos tampoco constituyen evidencia. Tras una evaluación válida, las memorias sí influyen: no se desactivó PSO silenciosamente. Una puntuación finita antigua sigue necesitando comparabilidad temporal, objeto de FMT-127. No se ha demostrado estabilidad o superioridad del híbrido completo.

### FMT-126 — La ecuación de amortiguación de CSA estaba transcrita con otra función

**Evidencia.** El constructor utilizaba:

~~~text
d_anterior = 1 + 2·max(0, (mu_eff−1)/(n+1)) + c_sigma
d_contrastado = 1 + 2·max(0, sqrt((mu_eff−1)/(n+1))−1) + c_sigma
mu_eff = 1 / sum_i(w_i²), con sum_i(w_i)=1
~~~

Faltaban la raíz y el desplazamiento menos uno. d_sigma amortigua la variación logarítmica del paso; no estima volatilidad de mercado. Para n=2 y lambda=4, el caso rojo obtuvo 1,7154953653521605 frente a 1,4089687727837696. El cociente c_sigma/d_sigma cambiaba y, por tanto, la velocidad de adaptación también.

**Reparación y fundamento.** Se corrigió [la expresión](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:144>) según el pseudocódigo del apéndice C de [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772). La prueba cubre varias combinaciones de dimensión y población, incluyendo razones grandes de mu_eff/n. No se eligió un nuevo multiplicador por ensayo informal.

**Alcance.** Se restaura esa ecuación, no todas las invariancias del algoritmo publicado. Permanecen el blanqueamiento diagonal, reflexión acotada por diez iteraciones, clamps de gaussianos/covarianza, mezcla PSO y diferencia de normalización de h_sigma. Presentar este arreglo como CMA canónico completo sería incorrecto. Cierre global: suites de funciones rotadas y mal condicionadas, residuos de factorización, diversidad, presupuesto de evaluaciones y comparación con una implementación de referencia.

### FMT-127 — Las memorias personales y globales comparan puntuaciones históricas sin identidad de objetivo

**Evidencia.** [Las memorias PSO](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:473>) se actualizan únicamente cuando el score actual supera al máximo histórico. El caller persiste ese objeto, pero construye nuevas ventanas recientes, consulta capital/configuración y vuelve a simular. No existe en la memoria una versión de objetivo, datos, costes o época de evaluación.

**Contraejemplo lógico.** Un candidato A gana 100 en la ventana W1. En W2, todos los candidatos nuevos puntúan entre −1 y 1. El máximo histórico conserva A=100 sin medir cuánto vale A en W2. Incluso una transformación aditiva común del objetivo entre rondas, que no altera el ranking dentro de W2, puede impedir actualizar la memoria indefinidamente. Cambiar la distribución económica vuelve más serio el problema, no lo origina.

**Impacto.** Una fuerza de búsqueda aparentemente aprendida puede estar gobernada por una comparación que ya no existe. Además, el slot i contiene memoria entre propuestas CMA nuevas; esa semántica no equivale automáticamente a una partícula PSO con trayectoria e identidad estables. Es una decisión de híbrido que necesita definición y experimento, no sólo un nombre.

**Estado.** Abierto. No se borraron memorias en cada iteración como arreglo indiscriminado, porque eso eliminaría aprendizaje útil cuando el objetivo sí sea comparable. Tampoco se inventó una vida media de olvido.

**Cierre requerido.** Registrar objetivo y ventana; reevaluar incumbentes en el mismo escenario que los candidatos o declarar una política causal de invalidación por cambio de contexto. Comparar esas alternativas con igual presupuesto, incluyendo coste de reevaluación. Probar objetivo estacionario, traslación de fitness, cambio de óptimo, etiquetas retrasadas y datos solapados. La publicación encontrada sobre PSO dinámico aporta un vecino bibliográfico, pero la consulta de texto completo no devolvió pasajes; no se atribuye a ella una política concreta verificada.

### FMT-128 — La población adaptativa solicitada no cambia la población efectiva

**Evidencia.** [pop_size](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/lib.rs:162>) se recalcula entre 40 y 100 usando min_trades_per_day. Sin embargo, lambda, mu, pesos, mu_eff y buffers se construyen sólo al crear el optimizador. Las siguientes poblaciones se muestrean con el lambda persistido. Cambiar capacidad reservada de un Vec no cambia su número de candidatos.

**Impacto y contraejemplo.** Si el primer valor es 40 y el siguiente 100, se siguen produciendo 40 propuestas; si el primer valor es 100 y luego se solicita 40, siguen siendo 100. El controlador puede atribuir un efecto a una acción que no llegó al optimizador. La frecuencia deseada de operaciones tampoco define por sí sola el tamaño estadístico o computacional apropiado de una población.

**Mejora limitada aplicada.** La reserva de genomas usa el número real de muestras. El log distingue lambda efectiva de solicitada. Esto elimina una ambigüedad operativa, no implementa redimensionamiento adaptativo.

**Cierre requerido.** Especificar si lambda es fija por experimento, cambia mediante reinicio o se adapta con una transición explícita de pesos, caminos y memoria. Comparar mejora por evaluación, tiempo de pared, memoria y latencia p99, no sólo por generación. No asignar lambda directamente sin reconstruir el resto del estado dependiente ni reiniciar la covarianza a escondidas.

### FMT-129 — El paso usado al generar propuestas puede diferir del usado al aprender de ellas

**Evidencia.** El muestreo usa self.sigma; la actualización obtiene [safe_sigma](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:506>) recortándolo a [10^-6,100]. La salida de CSA admite un mínimo de 10^-9 y no impone ese mismo máximo. El contrato permite, por tanto, observar un desplazamiento generado con una escala y normalizarlo con otra.

**Demostración dimensional.** Con sigma=10^-9 y un desplazamiento de media 10^-9 lejos de fronteras, el paso normalizado pertinente es 1. La división por 10^-6 produce 0,001. Su contribución cuadrática a un término de covarianza puede reducirse por un factor de un millón antes de los demás pesos/clamps. En el extremo superior se puede producir el sesgo inverso.

**Impacto.** El camino aprende el comportamiento de una distribución distinta de la muestreada. Sigma puede seguir siendo finito, y el test de finitud pasar, mientras se altera la realimentación. Este punto es distinto del blanqueamiento diagonal de FMT-012, aunque ambos errores se combinan.

**Estado y cierre.** Abierto. El arreglo del supervisor no lo oculta ni autoriza sigma extrema como científicamente válida. Se necesita un snapshot de distribución por lote, con sigma y factor de covarianza efectivos, y una política de límites aplicada coherentemente antes de muestrear. La aceptación debe comprobar igualdad de distribución generadora/normalizadora, límites de precisión, saturaciones explícitas y ausencia de reparaciones silenciosas. No basta quitar un clamp aislado sin revisar las divisiones, covarianza y mezcla PSO.

## 5. Revisión por los ocho módulos del sistema

| Módulo | Resultado de esta ronda y conexión pendiente |
|---|---|
| 1. Ingestión, parsers y L2 | No se reauditaron todos sus archivos. La ventana usada por evolución necesita snapshot coherente, IDs y timestamps; un lote estructuralmente válido no acredita calidad de mercado |
| 2. Inferencia y señales | No se corrige aquí el proxy ML ni el contrato de features FMT-003/052. No confundir la mejora del optimizador con información predictiva nueva |
| 3. Estrategia, régimen y horizonte | Las propuestas deben parametrizar política sobre escala continua con soporte observable. Persisten reloj global y etiquetas auxiliares FMT-059; renombrarlas no corrige la cronología |
| 4. Ejecución y conectividad | No se cambiaron payloads ni órdenes. FMT-113 permanece; ningún test de CMA acredita cantidad ejecutable, fill, comisión o latencia real |
| 5. Riesgo y genomas | Se fortaleció validación de propuestas y continuidad de aprendizaje. Permanecen abstención penalizada y promoción sin contraste completo del incumbente |
| 6. Estado, memoria y SO | Rechazo estructural sin mutación parcial; no equivale a transacción distribuida ni snapshot atómico de arena. Campos públicos y presupuestos de recursos requieren contrato adicional |
| 7. Orquestación y confluencia | Se documentó separación entre la autoridad llamada por el host y CMA auxiliar. No se añadió otro promotor concurrente |
| 8. Backtesting y gobernanza | Cinco regresiones rojo→verde, pruebas de dos generaciones, fuentes y límites. Persisten reutilización de validación, atribución económica y watchdogs bloqueados FMT-051/071 |

El supervisor MetaEvolver también merece seguimiento: el caller termina pasando alpha.3, un conteo, a audit_system_architecture, cuyo argumento se interpreta como Sharpe. Sus mensajes infieren causas estructurales desde umbrales sin un diagnóstico causal. Es una ampliación de los problemas de semántica FMT-059/T30, no una prueba de que deba eliminarse RSI o dividirse una arquitectura. No se implementó esa recomendación automática.

## 6. Evolución científica: qué integrar y qué medir

### 6.1 Continuidad temporal no equivale a enumerar nanosegundos

Un modelo puede parametrizar una política para todo tau>0 sin almacenar un gen por cada instante. Por ejemplo, como propuesta de diseño:

~~~text
theta(tau,z) = sum_j a_j(z) · phi_j(log(tau/tau_ref))
~~~

tau es escala de decisión, tau_ref fija la unidad del logaritmo, z es contexto observable, phi son funciones base y a son coeficientes aprendidos. La elección de base, regularización y resolución debe evaluarse; no se presenta esa expansión como modelo ya implementado o universalmente óptimo.

La resolución de timestamp, la frecuencia del feed, la frecuencia de cómputo, el horizonte predictivo y el horizonte de riesgo son variables diferentes. Producir un número para 1 ns o 100 años no demuestra disponer de observaciones ni incertidumbre calibrada a esa escala. Un dominio continuo se puede aproximar con refinamiento guiado por error y soporte; exigir recomputar literalmente cada nanosegundo no genera la información ausente.

El contrato deseado por arista es: política/estimación, escala, reloj, soporte, coste y estado de extrapolación. Los nombres scalping/swing no deben gobernar motores independientes, pero quitar cadenas del código sin migrar estas variables podría conservar exactamente la misma partición bajo otro nombre. Esta ronda no declara erradicadas todas esas referencias.

### 6.2 T31 — Geometría de información e invariancias como criterio de diseño

[Information-Geometric Optimization Algorithms: A Unifying Picture via Invariance Principles](https://arxiv.org/abs/1106.3708) conecta optimización basada en distribuciones, gradiente natural y reglas por ranking. Los pasajes consultados explican el flujo IGO y la recuperación de variantes de CMA/NES y otros métodos mediante familias probabilísticas; las propiedades del flujo no se transfieren sin más a discretizaciones, clamps o híbridos.

Aplicación propuesta, no desplegada: utilizar invariancias como tests de diseño antes de aumentar complejidad. Si dos objetivos tienen el mismo orden dentro de una evaluación, una regla puramente por ranking no debe seleccionar padres diferentes sólo por su escala numérica. Si se rota un problema, un método que afirma adaptación completa de covarianza debe justificar cómo cambia su comportamiento. Si cambia el objetivo entre ventanas, el score histórico necesita nueva interpretación, no una comparación desnuda.

Experimento mínimo: mismos seeds y presupuesto sobre cuadráticas rotadas, mal condicionadas y objetivos móviles; medir error, diversidad, evaluations-to-target y coste p99. Comparar CMA de referencia, híbrido actual y ablaciones sin PSO. Un resultado favorable en funciones sintéticas es requisito de ingeniería, no evidencia de alpha financiero.

La expansión bibliográfica recuperó también [MAP-Elites](https://arxiv.org/abs/1504.04909), [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) y [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944). Se conservan como alternativas de archivo diverso, evaluación escalable y búsqueda costosa respectivamente, basándose en sus resúmenes. No se verificaron aquí sus cuerpos ni se implementaron sus algoritmos.

### 6.3 T30, ampliación — Registro de objetivo, propuesta y evidencia

El próximo contrato debería distinguir generación, propuesta, genoma normalizado, fenotipo aplicado, datos, objetivo y evidencia. Un hash de genoma no reemplaza el hash del escenario. Un score no reemplaza el número de consultas hechas sobre el holdout. Un índice de vector no identifica una operación ejecutada.

La extensión debe permitir invalidación razonada, reevaluación comparable de memorias e integración del incumbente. Sólo después corresponde decidir qué componente evoluciona, con qué límite de riesgo y qué condiciones autorizan promoción. Los actuales UpdateOutcome y StepSizeAdjustment son avances locales y observables, no ese ledger completo.

### 6.4 Matemática, física y cuántica: alcance verificable

No se añadió una ecuación de un problema del milenio como decoración ni se equiparó complejidad formal con capacidad predictiva. Para usar una PDE, un potencial, un operador espectral o una analogía física se necesitan variables, unidades, condiciones de contorno, identificabilidad, error numérico y una hipótesis refutable sobre datos. Resolver una ecuación no demuestra que el mercado obedezca ese modelo.

El código intervenido sigue siendo optimización clásica. Una eventual investigación cuántica necesitaría problema formulado, circuito/operador, recursos, medición y baseline clásico bajo igual presupuesto; ninguna de esas garantías se obtiene renombrando un fallback como colapso cuántico.

La habilidad Firecrawl guio búsqueda, expansión y contraste primario. La CLI no estaba disponible y se utilizó el índice conectado; no se instaló software ni se envió código privado. Su influencia material fue corregir una ecuación publicada y mantener explícita la diferencia entre teoría, híbrido y evidencia operativa. La consulta del cuerpo del artículo PSO dinámico no devolvió pasajes; ese límite se conserva en vez de rellenarlo por inferencia.

## 7. Pruebas y resultados reproducibles

~~~text
cargo test -p evolution-engine --offline --test cma_generation_contract
cargo test -p evolution-engine --offline --test cma_generation_contract --test cma_penalty_contract
cargo test -p evolution-engine --offline --test cma_supervision_contract -- --nocapture
cargo test -p evolution-engine --offline --lib cma_es::tests
cargo check -p trader-gemini-v5 --bin god_engine --offline
~~~

La primera ejecución, antes de reparar los mecanismos, produjo cinco fallos esperados: duplicate ID, ID fuera de rango, genoma inválido, PSO sin evidencia y damping. Sólo se había añadido la inyección de RNG y las pruebas; no se atribuye ese fallo a cambios del algoritmo reparador.

Resultado final, sin duplicar reejecuciones:

| Suite | Tests aprobados | Naturaleza |
|---|---:|---|
| cma_generation_contract | 5 | Nuevas regresiones rojo→verde |
| cma_supervision_contract | 11 | Nuevos contratos de estado, configuración, supervisor y evidencia |
| cma_penalty_contract | 6 | Regresiones de la ronda VII conservadas |
| cma_es::tests | 3 | Pruebas existentes de muestreo, singularidad y fitness inválido |
| Total | 25 | 16 nuevos; no es toda la suite del workspace |

cargo check del host terminó correctamente. Persisten avisos anteriores sobre latest_ts, mode y RealWfOutcome.trades; no se aplicó cargo fix global. Hubo espera por lock de compilación; no se interrumpió otro proceso. Las pruebas puras no ejecutan el loop, no promocionan genomas y no envían órdenes.

No se ejecutaron backtests económicos retenidos, campañas de optimización, pruebas en demo/producción ni benchmarks de rendimiento. No se usa el número de tests como sustituto de esos ensayos. La adaptación, las ecuaciones y los rechazos tienen evidencia local; los efectos económicos permanecen sin medir.

## 8. Manifiesto, preservación y límites de certificación

| Archivo intervenido | Líneas finales | SHA-256, prefijo |
|---|---:|---|
| crates/evolution-engine/src/cma_es.rs | 667 | 526D3E8BA5388101 |
| crates/evolution-engine/src/lib.rs | 665 | C340A623B9DDF8B6 |
| crates/evolution-engine/tests/cma_generation_contract.rs | 82 | DEFD43DCBDE6B56E |
| crates/evolution-engine/tests/cma_supervision_contract.rs | 225 | A3B412684A8D8E97 |

Los dos fuentes ya contenían cambios propios de VII. Antes de continuar se verificaron sus hashes contra el cierre de esa ronda: CMA 98ACE3FBC23FA9DA y lib FE66BE8328214F24. No se atribuyeron cambios concurrentes a esta intervención ni se borraron para limpiar el árbol.

Los hashes de tres archivos ajenos prioritarios permanecieron iguales en la comprobación posterior a las pruebas: host F2BA92C4283C66E6; replay 736CC1DBB38E21D7; risk/lib E9883455B6BEBEFA. No se editaron graphify-out, genomas, archivos de configuración operativa ni fuentes ajenos sucios. No se formateó globalmente evolution/lib.

Se ejecutaron rustfmt --check de CMA y ambos tests nuevos, y git diff --check de los fuentes intervenidos. Las adiciones a ATLAS, maestro y VII son append-only; los estados de rondas anteriores conservan su significado histórico. El informe maestro contiene bytes NUL preexistentes: no se normalizan ni eliminan en esta tarea.

Sin trading, despliegue, promoción, reinicios, merge, commit, push ni fetch. HEAD sigue siendo 59a76de4. Las ramas y el remoto no se certifican actualizados con base sólo en esta revisión local.

La comprobación final confirmó los cuatro hashes completos de esta ronda y los tres hashes ajenos prioritarios. Se validaron los diez enlaces locales de fuente y los seis encabezados FMT nuevos. El SHA-256 de todo el prefijo anterior de ATLAS, maestro y VII se conservó tras las adiciones, normalizando únicamente CRLF/LF para la comparación. Las adiciones respectivas fueron 1.650, 2.174 y 1.081 caracteres; no se sustituyó contenido anterior. rustfmt --check y git diff --check volvieron a pasar.

## 9. Hoja de ruta sistémica de rehabilitación 1-a-1

1. Priorizar la ruta realmente llamada por el host: cerrar identidad de observaciones, diferencias del simulador, validación consumida por búsqueda y aislamiento de watchdogs, conservando FMT-049/050/051/052/055/056/057/071 como pendientes hasta evidencia de cierre.
2. Coordinar los fuentes compartidos para resolver FMT-113: presupuesto, cantidad realizable, reserva y payload deben referirse a la misma operación. No modificar sobre cambios ajenos sin revisión.
3. Completar T30 para que ningún resultado de otra ventana, genoma, entorno o versión pueda actuar como evidencia intercambiable. Eso permite abordar FMT-127 con política explícita.
4. Unificar distribución generadora y normalizadora de CMA, incluyendo FMT-012/129; contrastar referencia, híbrido y ablaciones antes de conectar la ruta auxiliar.
5. Definir política de población y supervisor mediante pérdidas, incertidumbre y presupuesto computacional; no equiparar win rate con utilidad ni tamaño de Vec con adaptación.
6. Migrar horizonte a contrato continuo con soporte observable, resolviendo FMT-117 y cronologías auxiliares. Evaluar extrapolación por separado de observación.
7. Reanudar la cobertura archivo por archivo: quedan 197 Rust preexistentes sin acreditar lectura completa en estas rondas, además del resto de tipos de archivo. No declarar auditoría integral hasta cerrar inventario y evidencia correspondiente.

**Conclusión:** hay progreso verificable en continuidad de aprendizaje, atribución estructural y fidelidad de una ecuación. Persisten fallos científicos y de integración suficientemente importantes para impedir una certificación global. La siguiente mejora no debe consistir en añadir más teoría a una arista sin contrato, sino en hacer comparable, trazable y refutable lo que esa arista transmite.

## Continuación aditiva — ronda IX (2026-09-24)

El [anexo IX](AUDITORIA_FUNDAMENTOS_CIENTIFICOS_IX_2026-09-24.md) prioriza
el daemon que sí llama el host. Repara componentes de FMT-055/056:
estadístico descriptivo, degeneración/error explícitos, EWMA por revisión
nueva y autoridad de parada que no permite al daemon rearmar causas ajenas.
No se certifican inferencia global, rollback transaccional ni despliegue.

FMT-130–132 documentan problemas de entrega durable, journal y construcción
de retornos. La cobertura pasa a 93 Rust preexistentes distintos por la
lectura completa de evolution_ledger.rs. Las 25 pruebas CMA de VII/VIII
siguen pasando; IX suma 22 nuevas y verifica 47 distintas sin duplicar
reejecuciones. El cambio de lib.rs en IX es la exportación del módulo
de evidencia; sus hashes actuales constan en el nuevo manifiesto.
