# Auditoría científica XXI — medida coherente, separación temporal y rutas de aprendizaje

Fecha de corte documental: 2026-09-24, zona del proyecto America/Bogota. Continuación aditiva de la ronda XX. Esta entrega modifica localmente un entrenador, documenta cinco hallazgos nuevos y amplía tres pendientes; no certifica el proyecto entero ni un despliegue.

## 1. Resultado y límites de la intervención

Se repararon dos mecanismos concretos de train_forest: la mezcla de poblaciones al calcular ganancias de árboles (FMT-192) y el presupuesto de muestras no vinculante (FMT-194). FMT-193 queda parcialmente reparado: ahora se conservan ventanas de información, se purga el ajuste contra selección y la promoción exige una prueba cronológicamente posterior. Falta gobierno persistente de experimentos, incertidumbre de la evaluación y validación económica.

Los defectos no eran una falta de ecuaciones más sofisticadas: se estaban combinando cantidades que no pertenecían a la misma población, evento o reloj. Esas inconsistencias pueden degradar cualquier modelo posterior, por avanzado que sea. La corrección del cálculo no demuestra por sí sola que cambien las decisiones del genoma en demo o producción.

Resultado de verificación: **37 pruebas funcionales distintas aprobadas**: 25 del entrenador —18 nuevas y 7 anteriores— y 12 del contrato de inferencia ya existente. Cuatro pruebas nuevas fallaron con las fórmulas anteriores y pasan tras la corrección. Cargo check de god_engine y train_forest pasa. No se entrenó un modelo con datos reales, no se ejecutó el motor y no se promovió ningún artefacto.

Persisten hechos importantes: un horizonte de etiqueta por ejecución, barreras constantes, descarte de timeouts, targets dependientes del número de eventos, otra ruta neuronal sin holdout y un validador auxiliar con supuestos intrabar optimistas. No se presenta el resultado como sistema universal continuo, autoevolución completa, superioridad cuántica ni capacidad de cumplir una rentabilidad objetivo.

## 2. Cobertura archivo por archivo y preservación

Inventario versionado recontado: **1.119 archivos, 289 Rust y 24 Cargo.toml**. Cobertura acumulada conservadora: **134/289 Rust preexistentes leídos completos; 155 pendientes**. El incremento frente a XX es exactamente tres. Una búsqueda de texto no equivale a leer un archivo completo ni a demostrar ausencia de consumidores dinámicos.

| Archivo | Alcance de XXI | Resultado |
|---|---|---|
| src/bin/train_forest.rs | Relectura completa, reparación y pruebas; no aumenta cobertura histórica | FMT-192/193/194; extensión FMT-028 y FMT-199 |
| src/bin/feature_exporter.rs | Nueva lectura completa, 217 líneas; sin cambios | FMT-195 |
| src/bin/feature_validator.rs | Nueva lectura completa, 147 líneas; sin cambios | FMT-196 |
| src/bin/train_dark_alpha.rs | Nueva lectura completa, 382 líneas; sin cambios | FMT-197/198 |
| crates/god-engine-core/tests/ml_model_contract.rs | Relectura completa de harness previo; no es nuevo archivo preexistente auditado | 12 regresiones, inventario ignorado |
| Cargo.toml y referencias en src/crates/scripts | Búsqueda de declaraciones y consumidores, no nueva lectura completa | Binarios auxiliares y enlace CSV localizados |

Se preservan por SHA-256 los tres binarios auxiliares y cuatro fuentes de referencia: predictor, núcleo, daemon evolutivo y genoma. Se comparan también los 41 archivos JSON/bin de models contra su estado inicial. Su conservación no significa que sean válidos: los defectos del inventario de XX continúan pendientes.

Solo se edita una fuente de producción en esta ronda: train_forest.rs. No se alteran configuraciones, genomas, fuentes de ejecución o parámetros de cuentas. Atlas, maestro y XX reciben adendas; sus prefijos se verifican normalizando CRLF a LF. La matriz histórica de 305 puntos no se sustituye ni se infla sumando estos IDs como si fueran independientes de todas las taxonomías anteriores.

Git local permanece main/59a76de4, con numerosos cambios previos/concurrentes. No se realizaron commit, push, merge ni fetch. Esto no verifica la sincronización del remoto ni quién resolvió otros cambios.

## 3. Matriz de hallazgos y estado

| ID | Prioridad / capa | Estado XXI | Objeto preciso |
|---|---|---|---|
| FMT-192 | P1, estadística del ajuste | Reparado localmente | Ganancia y mínimos de hijos sobre la población completa del nodo |
| FMT-193 | P1, evaluación temporal | Parcial | Purga y test posterior implementados; gobierno experimental pendiente |
| FMT-194 | P2, presupuesto/latencia | Reparado localmente | Stride no densifica y contador limita intentos por archivo |
| FMT-028 | P1, objetivo probabilístico | Abierto, reconfirmado | Condicionamiento a toque y complemento largo/corto no equivalentes |
| FMT-195, nuevo | P1, etiqueta/esquema CSV | Abierto | 500 ticks llamados 5m, TP inalcanzables y contexto macro fabricado |
| FMT-196, nuevo | P2, diagnóstico auxiliar | Abierto | Proxy OHLC llamado OBI, trayectorias intrabar ambiguas y veredicto sin inferencia estadística |
| FMT-197, nuevo | P1, contrato de entrenamiento neuronal | Abierto | Filas de dimensión variable y datos inválidos convertidos en evidencia |
| FMT-198, nuevo | P1, publicación neuronal | Abierto | Ajuste sobre todo el CSV y escritura al destino de modelo sin evaluación independiente |
| FMT-199, nuevo | P1, semántica de objetivos temporales | Abierto; descripciones corregidas | RMS y profundidad ponderados por eventos, no por tiempo físico |

“Reparado localmente” exige pruebas del mecanismo, no significa desplegado. P1 se refiere a posible invalidez de decisiones/evidencia si la ruta se utiliza; no afirma una pérdida observada ni que un binario auxiliar esté activo.

## 4. Grafo vivo: de la raíz a la decisión de publicación

El grafo comprobado en esta ronda tiene tres recorridos diferentes. No deben confundirse por compartir nombres de features.

    Raíz: ticks con reloj y procedencia
      → validación de orden / valores
      → presupuesto por archivo
      → features + etiqueta + intervalo [inicio, fin]
      → ajuste purgado ──→ bosque
      → selección temporal ──→ elección de rondas
      → artefacto congelado y validado
      → test posterior independiente en esta ejecución
      → nodo terminal de publicación: candidato o destino solicitado

    Ruta auxiliar distinta:
    ticks → feature_exporter → *_FEATURES.csv
          → train_dark_alpha → models/DarkAlpha_<activo>.json
                               [sin test/gate independiente localizado]

    Diagnóstico auxiliar distinto:
    barras OHLCV → feature_validator → consola con “ventaja”
                                      [no equivale a libro L2 observado]

Las aristas verificadas son declaraciones, llamadas y rutas por defecto del código. No se midió qué recorrido ejecuta hoy un proceso productivo. Tampoco se probó que la publicación neuronal implique activación automática: el hallazgo confirmado llega hasta la escritura al destino del modelo.

| Módulo de la arquitectura histórica | Conexión de XXI | Lo que no queda certificado |
|---|---|---|
| 1. Ingestión/L2/normalización | Orden de ticks; proxy OHLC; saneamiento CSV | Procedencia real de libros, unidades de reloj y disponibilidad macro |
| 2. IA/modelos/señales | Ganancia GBDT; esquema neuronal; selección/test | Calibración, representación universal y distribución productiva |
| 3. Multiactivo/horizontes | Ventanas explícitas y dependencia del reloj | Modelo conjunto entre activos y horizontes continuos |
| 4. Ejecución/conectividad | No intervenido | Fills, costes, colas, rechazo y latencia p99 |
| 5. Riesgo/genomas | Se identifica evidencia que no debe promoverse como fitness | Transferencia causal gen→decisión→resultado |
| 6. Estado/telemetría/SO | Presupuesto, errores numéricos y persistencia pendiente | Publicación atómica y recuperación ante fallo |
| 7. Confluencia/cuántica | Contratos previos a cualquier nueva integración | Ventaja cuántica o fusión de probabilidades heterogéneas |
| 8. Backtesting/gobernanza | Purga, holdout y crítica del POC | Corrección por búsqueda repetida y evidencia económica externa |

## 5. FMT-192 — una partición no puede mezclar medidas estadísticas

**Evidencia actual:** [candidate_gains](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:157>) y [build_tree](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:367>). En la versión anterior el padre sumaba todas las filas de idx, pero el hijo izquierdo sumaba solo probe, de hasta 2.048 extracciones con reemplazo. El derecho se definía por resta entre esas poblaciones distintas. Los mínimos de hijos también se contaban sobre probe.

Sea I el multiconjunto bootstrap del nodo, incluyendo repeticiones. Para un umbral c, L y R deben particionar ese mismo I. Con gradiente negativo g y curvatura h:

    G_A = suma de g_i en A
    H_A = suma de h_i en A
    Score(A) = G_A² / (H_A + lambda + epsilon)
    Ganancia = Score(L) + Score(R) − Score(I)

La expresión utilizada conserva la escala del score del programa; no se presenta como una nueva pérdida exacta ni elimina las aproximaciones de construcción del árbol. La condición necesaria aquí es G_I=G_L+G_R y H_I=H_L+H_R sobre las mismas observaciones y multiplicidades. Restar una estadística de una submuestra a una estadística total no crea el complemento de esa submuestra bajo la partición real.

**Reproducción numérica ejecutada:** 4.096 filas, dos mitades de 2.048, gradientes +1 y −1, hessiano 1, lambda 1 y un probe que toma una de cada dos filas. La fórmula anterior da 1.364,2232339891889; la partición completa da 4.094,000976085894. No es solo un cambio de redondeo. Un segundo caso tiene 80 filas de un hijo, pero solo 20 en probe: min_child=40 lo descartaba pese a que el hijo completo cumplía el mínimo.

**Consecuencia:** elección y descarte de umbrales no correspondían a la función optimizada sobre el nodo. La distorsión cambia con tamaño, repetición bootstrap y composición del probe; un genoma o selector puede terminar comparando modelos cuyos errores provienen del estimador interno de particiones.

**Reparación:** el probe conserva únicamente la función de proponer cuantiles. Para cada feature se ordena la población completa una vez; un barrido de prefijos y sufijos obtiene las sumas de ambos hijos. Los sufijos evitan obtener un hijo pequeño por resta de dos acumulados grandes. Se mantienen multiplicidades bootstrap, se cuentan hijos completos y se excluyen ganancias no finitas, curvaturas negativas o estadísticas inválidas.

**Complejidad:** por feature/nodo, O(n log n + q log q + n + q), memoria auxiliar O(n+q). Evita un escaneo completo por cada umbral, pero puede ser más costoso que el algoritmo anterior basado en probe. No se ejecutó benchmark de entrenamiento real; no hay afirmación de aceleración. El precio de corregir el estimador debe medirse.

**Pruebas y límite:** cinco tests cubren contraejemplos, comparación con suma directa con duplicados, estadísticas inválidas y un árbol de un nivel con 4.096 filas. Permanecen propuesta aproximada de umbrales, bootstrap 80%, subconjunto de features 70%, 24 cuantiles, epsilon 1e−12 y corte de ganancia 1e−9. Esas políticas siguen requiriendo análisis de sensibilidad; no se han convertido en adaptación espectral.

## 6. FMT-193 — separación por información, no solo por posición de fila

**Evidencia actual:** [TrainingSamples](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:258>), [purga](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:306>), [holdout posterior](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:334>) y [gate del artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:1142>).

Antes, un split 80/20 ordenaba las features pero olvidaba qué futuro había utilizado cada etiqueta. Con horizonte 300.000 ms y stride 50.000 ms, etiquetas cercanas al corte pueden observar precios que pertenecen al periodo de selección. Además, selección de rondas y criterio de publicación reutilizaban las mismas observaciones. Cambiar de archivo no verifica por sí solo ni cronología ni disjunción.

La muestra i ahora lleva un intervalo cerrado I_i=[t_i,e_i]. Se conserva para todos los objetivos el deadline completo t_i+horizonte, incluso si la barrera se tocó antes: es una cota conservadora, no el tiempo exacto de resolución. Si b es el primer instante de selección, se conserva en ajuste solo:

    e_i < b

La igualdad se purga porque comparte un instante. Se revisan todas las filas: con horizontes diferentes, los fines no tienen por qué estar ordenados aunque los inicios sí lo estén. Se preserva la correspondencia entre features, target e intervalo.

**Flujo implementado:**

1. Validar timestamps positivos y no decrecientes antes del motor de features. Se rechaza cronología rota; no se reordena silenciosamente.
2. Validar contrato no vacío, dimensiones consistentes, finitud y orden estricto de inicios muestreados.
3. Conservar selección 80/20 o archivo --val-in; exigir que sus inicios sean posteriores a todos los inicios de ajuste y purgar etiquetas solapadas.
4. Rechazar ajuste vacío tras purga o insuficiente para el bootstrap implementado. Rechazar clasificación monoclase, que produciría logit infinito.
5. Elegir rondas con selección; congelar exactamente el modelo serializable y evaluar ese artefacto.
6. Si se solicita --promote, exigir --test-in antes de leer datos. --val-in nunca sustituye al test.
7. Exigir primer inicio del test > máximo fin de las etiquetas utilizadas por ajuste y selección. El test se construye y puntúa después de congelar el modelo; sus labels no ajustan parámetros ni el baseline en esa ejecución.

**Ejemplo de prueba:** train [(1,3),(4,10),(7,11)] y selección desde 10 conservan solo la primera muestra. Para train [(1,100),(2,3),(4,50),(5,6)] y selección desde 10 se conservan las muestras 2 y 5: purgar solo un sufijo sería incorrecto. Un test en 100 se rechaza si una etiqueta previa termina en 100; uno que empieza en 101 cumple la separación temporal comprobada.

**Interpretación del score:** para regresión, el baseline usa la media del ajuste, no la media del test. Por tanto 1−MSE_modelo/MSE_baseline es un skill score relativo al baseline fijado en train, no necesariamente el R² convencional centrado en la media del test. El texto y logs fueron corregidos. Se retira el piso absoluto 1e−12 del denominador del gate: con pérdidas 0,8e−20 y 1e−20 la mejora relativa sigue siendo 20%, no cero. Baseline cero, valores no finitos, pérdidas negativas y márgenes inválidos no aportan aprobación.

**Cambio de interfaz deliberado:** una invocación antigua con --promote y sin --test-in falla antes de I/O. Sin test se puede crear un candidato de investigación con diagnóstico de selección; no se habilita promoción. Fallar el gate conserva la política de escribir solo candidato y terminar con código 2, nunca destino promovido. No se ejecutó ninguna de estas publicaciones sobre modelos reales.

**Por qué sigue parcial:** no hay registro persistente de cuántas veces se reutilizó un test ni del conjunto de hiperparámetros/activos/horizontes explorados. No hay sello de procedencia y disponibilidad de cada feature, manifiesto de corpus, intervalos de incertidumbre con dependencia temporal, evaluación de coste económico ni protocolo de activación atómica. Un test posterior no vuelve independientes sus propias filas solapadas; no habilita inferencia IID. Tampoco demuestra robustez bajo cambios de distribución.

La referencia externa contrastada establece que el conjunto de prueba no debe dirigir elecciones del modelo y que el preprocesamiento aprendido debe ajustarse sin observarlo. La purga por intervalos y su implementación son diseño y verificación locales de esta auditoría, no una atribución a esa página. [Scikit-learn: common pitfalls, data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## 7. FMT-194 — presupuesto de muestras verificable

**Evidencia:** [effective_stride y SamplingBudget](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:216>) y [construcción por archivo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:676>).

La condición anterior conservaba stride cuando el span era demasiado grande para el presupuesto y, en el caso contrario, podía reducir el stride solicitado. No había contador de aceptación que impusiera un máximo. Multiplicar stride por budget también introducía riesgo de overflow; budget=0 llevaba a división por cero.

**Reproducciones ejecutadas:** span=10.000.000 ms, budget=100, stride solicitado=1.000 ms producía stride=1.000, incompatible con un máximo de 100 intentos en una malla densa. Para span=1.000.000, budget=1.000, stride=50.000, el algoritmo lo bajaba a 1.000: aumentaba cincuenta veces la densidad solicitada.

Para un span inclusivo S y presupuesto B>0, la regla nueva es:

    stride_efectivo = max(stride_solicitado, floor(S/B)+1)
    número de puntos de una malla densa <= floor(S/stride_efectivo)+1 <= B

La derivación usa aritmética entera y extremos inclusivos, no un nuevo umbral empírico. El caso extremo de saturación de u64 se protege además por un contador independiente. La fecha de siguiente intento usa suma comprobada: overflow no vuelve al pasado.

**Semántica precisa:** max-samples limita intentos elegibles reservados por archivo. Un intento consume presupuesto aunque después se descarte por macro ausente, target inválido o timeout. Así etiquetas aceptadas ≤ intentos ≤ presupuesto. No significa “exactamente B etiquetas”, ni limita todos los ticks leídos, ni la suma de train+val+test a un único B. Los logs ahora lo dicen explícitamente.

**Latencia residual:** la cronología recorre el archivo, el motor procesa ticks previos a los muestreos y cada etiqueta puede recorrer futuros ticks hasta su deadline. En el peor caso, ese trabajo sigue dependiendo del número de intentos y longitud de ventanas; no existe aquí un SLA temporal. El presupuesto limita materialización de muestras y consultas futuras, no toda la memoria del proceso ni un tiempo máximo absoluto.

**Cobertura y sesgo:** una malla acotada no garantiza representación de activos, estados o eventos raros. Los mínimos heredados de 50.000 ticks y 5.000 etiquetas permanecen; un presupuesto pequeño o muchos descartes puede abortar de forma legítima. No se inventan etiquetas para completar cupos. Cuatro tests cubren casos históricos, extremos inclusivos, series irregulares, cero y overflow.

## 8. FMT-195 — el CSV no comparte evento ni esquema con el bosque

**Evidencia:** [cabecera y contrato de exportación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:89>), [barreras](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:113>), [constantes macro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:188>) y [consumidor CSV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:54>).

Hay tres incongruencias verificadas en esta ruta:

- El encabezado target_5m corresponde a 500 ticks futuros, no a cinco minutos de reloj. Con cadencias distintas, el mismo número de eventos describe horizontes físicos diferentes. El CSV no guarda timestamps ni intervalos que permitan reconstruir ese horizonte.
- Se comprueba SL largo −0,18% antes del TP corto −0,36%, y SL corto +0,18% antes del TP largo +0,36%. Para precio positivo, todo toque del TP ya satisface la condición previa y sale del bucle. Ambos TP son ramas inalcanzables. Hay además fallback a retorno terminal ±0,04%, distinto del target del bosque.
- Los 54 canales son 34 universales más proxies/constantes, mientras el bosque construye 34+10 espectrales+4 macro. Las columnas 41=1,04, 42=1,02, 43=1,00 y 44=0,75 no son observaciones macro de cada fecha. Una cabecera numérica feature_i no expresa esa semántica.

**Contraejemplo por lectura de flujo:** desde mid=100, un movimiento a 100,20 activa label=1 por SL corto; nunca necesita alcanzar 100,36. Un largo cuyo objetivo es +0,36% no ha ganado por ese hecho. Si se termina sin toque y retorno >0,04%, el fallback también puede producir positivo sin el evento buscado por el bosque.

**Impacto condicionado:** el consumidor localizado por defecto es train_dark_alpha, no train_forest. La discrepancia invalida la afirmación de alineación 1:1 y dificulta comparar/fusionar sus probabilidades. No se ha demostrado que todas las columnas neuronales coincidan con un consumidor activo: esa traza queda pendiente. Compartir las primeras 34 llamadas tampoco iguala disponibilidad de fuentes.

**Integridad secundaria:** el exportador no filtra todos los ticks inválidos antes de actualizar estado; convierte features no finitas a cero, escribe sobre el destino CSV directamente, ignora errores individuales de filas y no comprueba el flush final antes del mensaje “clean rows”. El número contado mide escrituras aceptadas por el buffer, no certificación durable completa. No se simuló un disco lleno.

**Cierre requerido:** esquema versionado con activo, reloj, unidades, intervalo, fuente y objetivo; label por acción con causalidad compartida; estados de ausencia y censura explícitos; exportación transaccional con reporte de rechazos. Cambiar solamente target_5m por otro nombre no corrige el evento ni migra modelos previos. Estado abierto; no se regeneró CSV.

## 9. FMT-196 — el validador auxiliar prueba otro observable y otro proceso de fills

**Evidencia:** [proxy OHLC](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_validator.rs:70>), [entrada/salida](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_validator.rs:85>) y [veredicto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_validator.rs:139>).

El programa toma barras de BTCUSDT_6M.parquet y deriva p=clip((close−low)/(high−low),0,05,0,95). Luego define bid_qty=V p y ask_qty=V(1−p). Para V positivo, el cociente habitual (bid_qty−ask_qty)/(bid_qty+ask_qty) se reduce a 2p−1. El volumen se cancela: el supuesto OBI es una función de localización del cierre dentro de la barra. No es profundidad observada del libro. El caso de volumen nulo y los fallbacks requieren tratamiento separado.

La señal usa la barra anterior, lo cual evita una forma particular de lookahead; eso no valida el simulador entero. Al abrir en current_open dentro de la rama position==0, no evalúa TP/SL de esa misma barra. En barras posteriores comprueba TP antes de SL para largos y cortos. Si ambos fueron tocados, selecciona beneficio sin conocer el orden intrabar.

**Contraejemplo:** posición larga abierta en 100, una barra posterior con high=101 y low=99 y barreras ±0,2%. Ambas barreras están dentro del rango. La implementación suma +0,2%; OHLC por sí solo no identifica si fue primero 100,2 o 99,8. No puede certificarse ni ganancia ni pérdida cierta sin trayectoria o una política de ambigüedad explícita. Si esos extremos ocurren en la barra de entrada, ambas posibilidades se omiten.

Se agregan retornos fijos sin comisiones, no se modelan gaps ni fills, no se liquida/valora la posición terminal y pnl es suma simple de retornos, no riqueza compuesta. Los comentarios reconocen ausencia de costes, pero el veredicto final afirma ventaja estadística independiente si WR>55% y pnl>0, sin precisión, dependencia, tamaño mínimo justificado o evaluación externa.

**Severidad y límite:** P2 de diagnóstico; se localizó un binario POC, no una autorización operacional automática. El daño plausible es orientar diseño/selección con evidencia demasiado optimista o mal nombrada. No se ejecutó ese backtest ni se afirma que su resultado actual tenga sesgo cuantificado.

**Cierre:** llamarlo proxy OHLC, probar trayectoria/ambigüedad, incluir entrada y terminal, modelar costes y comparar contra un baseline en datos retenidos. El umbral WR no reemplaza la distribución de retornos y sus costes; con payoffs distintos, mayor WR no implica mayor crecimiento.

## 10. FMT-197 — entradas neuronales malformadas pueden convertirse en aprendizaje

**Evidencia:** [parser](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:81>) y [dimensionamiento y momentos](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:121>).

El parser acepta filas con al menos 26 columnas y crea un vector de min(columnas−1,54). Luego fija input_dim con la primera fila y usa x[i] para todas las demás. No verifica dimensión constante ni nombres del encabezado.

**Reproducción estática mínima:** cabecera; primera fila con target+54 features; segunda fila con target+25 features. Ambas pasan el filtro inicial. El bucle de medias indexa la segunda hasta i=53 y puede entrar en panic al llegar a i=25. Si la primera fila es corta y otras más largas, la interpretación de capacidad queda determinada por el primer registro, no por un esquema.

Un target que se parsea como NaN pasa el parseo de f64 y la comparación target>0,5 es falsa: produce clase 0. Infinito positivo produciría clase 1. Valores como 0,5 también se asignan a 0, sin distinguir neutralidad. Features ilegibles, NaN o infinitas se convierten en cero sin máscara ni contador de imputación. Errores de lectura de línea se omiten. Así “valid samples” no equivale a datos semánticamente válidos.

**Impacto:** interrupción determinista con ciertos CSV, corrupción silenciosa de etiquetas con otros y estimaciones de escala afectadas por imputación invisible. Esta ruta es especialmente relevante si se añade otro exportador o un esquema distinto durante una refactorización. No se ejecutó una carga malformada que pudiera escribir un modelo.

**Cierre:** contrato estricto de cabecera/dimensiones/unidades, rechazo o cuarentena con razón por fila, labels tipados y finitos, estadísticas de calidad y tests de NaN, dimensiones, orden y CSV truncado. La política de imputación debe aprenderse y registrarse por canal; cero no es universalmente ausencia. Estado abierto.

## 11. FMT-198 — ajuste neuronal y publicación carecen de evaluación independiente

**Evidencia:** [normalización](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:128>), [entrenamiento](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:183>) y [persistencia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:369>).

Se calculan media y desviación sobre todos los inputs; se normalizan todos y se barajan para ocho épocas de Adam. La pérdida impresa es de entrenamiento. No hay separación temporal, purga, early stopping por selección ni evaluación externa dentro del binario. Tras sanitize_denormals, escribe models/DarkAlpha_<símbolo>.json, sin destino candidato diferenciado ni flag de promoción.

**Precisión metodológica:** ajustar un scaler con todos los datos destinados realmente a train no es por sí mismo fuga de test; aquí no existe test. El defecto confirmado es ausencia de evaluación independiente y de un control de publicación en esta ruta. Sería fuga adicional si posteriormente se presentase una partición de ese mismo CSV como no observada manteniendo el scaler/modelo ya ajustados.

**Impacto:** terminar ocho épocas o bajar BCE no evidencia generalización, robustez multi-activo, sincronía de features ni efecto económico. La ruta puede sobrescribir un modelo anterior sin comparación. No se demostró que un watcher activo lo recargue; se demuestra el destino de escritura, no la activación.

**Cierre:** separar ajuste, selección y test por intervalos y procedencia; aprender transformaciones solo con ajuste; medir el artefacto serializado real; escribir candidato versionado con manifiesto y exigir decisión de promoción trazable. La escritura directa no es publicación atómica; el mismo tipo de riesgo de persistencia queda pendiente también en el bosque. Antes de activar debe probarse recuperación, rollback y política de modelo no disponible.

Estado abierto: el parche de train_forest no protege train_dark_alpha. No se trasladó automáticamente una protección de un binario a todo el sistema en el informe.

## 12. FMT-199 y extensión FMT-028 — un reloj configurable no equivale a un objetivo continuo

**Evidencia:** [targets vol/volu](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:858>) y [barreras constantes](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_forest.rs:721>).

El objetivo vol es 100·sqrt(sum(r_i²)/n), con retornos simples entre eventos válidos. Es RMS de retornos por evento en una ventana; no resta media, no pondera duración y no normaliza por tiempo físico. No es automáticamente desviación estándar ni volatilidad anualizada/por segundo. El objetivo volu es sum(bq_i+aq_i)/n, profundidad media por evento, no volumen ejecutado ni exposición temporal media del libro.

**Contraejemplo exacto de muestreo:** una secuencia contiene un retorno no nulo r y n−1 retornos cero. Insertar n observaciones repetidas de precio agrega retornos cero sin cambiar el movimiento observado de principio a fin. sum(r_i²) permanece r², pero n se duplica y el RMS se divide por sqrt(2). El objetivo cambia porque el proveedor emitió más eventos, no porque cambie el desplazamiento. Para profundidad, un estado 10 durante 9 segundos y 100 durante 1 segundo tiene media temporal 19; con una observación por estado, la media por evento es 55. Replicar mensajes altera aún más esa media.

**Impacto multi-activo:** dos fuentes/activos con distintas cadencias, unidades base o tamaños de contrato pueden producir targets no comparables. Usar un deadline de reloj no elimina esa dependencia del estimador. Cambiarlo silenciosamente exigiría regenerar labels, reentrenar y migrar consumidores; no se hizo.

Las descripciones del código ahora indican RMS/profundidad por evento. El comentario antiguo que convertía 500 ticks a 75 ms en aproximadamente 2 s también se corrigió: son aproximadamente 37,5 s. Se trata de una corrección explicativa, no de evidencia de frecuencia real del feed.

**FMT-028 continúa:** al descartar timeouts, dir estima el evento de TP largo condicionado a resolución antes del horizonte. Su complemento es SL del largo bajo esa condición, no éxito del corto con otros pagos. Se retiró del comentario la equivalencia injustificada con cierre en break-even. No se cambiaron targets ni órdenes; siguen faltando masa de no resolución, lado, costes y acción en el contrato compartido.

**Cierre científico requerido:** decidir la magnitud objetivo antes del estimador: riesgo acumulado de horizonte, tasa de variación por tiempo, profundidad temporal, volumen de trades o tiempo de primer paso. Después declarar unidad, disponibilidad y proceso de observación. Deben probarse invariancias bajo duplicación de mensajes, refinamiento temporal controlado, cambio de unidades y cambio de activo, sin imponer igualdad cuando se ha observado información nueva real.

## 13. T44 — contratos de medida y evidencia antes de ampliar la teoría

T44 es una propuesta de arquitectura científica, no una nueva implementación de alpha. Complementa T38–T43 y la representación continua ya discutida.

### 13.1 Estado multivariante y soporte observado

Una coordenada logarítmica s=log(h/h0) permite parametrizar funciones de horizonte sin crear motores excluyentes por etiquetas. La representación finita necesita una base, tolerancia de aproximación y presupuesto de cómputo explícitos. No debe confundir el dominio matemático del modelo con resolución observada del feed o fiabilidad de extrapolación.

Propuesta de contrato por muestra/nodo: activo e instrumento, instante de evento y disponibilidad, dominio de horizonte, unidad de cada observable, máscara de ausencia, soporte efectivo, esquema/versión, incertidumbre y objetivo/acción. Volatilidad y dependencia se representarían como funciones estimadas y sus incertidumbres, no como una categoría que monopoliza el estado.

Un modelo puede admitir una coordenada de 1 ns o 100 años sin poseer evidencia para ambos extremos. Iterar literalmente cada nanosegundo de cien años de 365,25 días representa aproximadamente 3,16×10^18 pasos; no se ha implementado ni justificado ese cálculo. Debe separarse precisión de timestamp, cadencia de eventos, actualización del estimador y horizonte de decisión. La arquitectura por eventos con aproximación multiescala es una hipótesis de ingeniería evaluable, no permiso para fabricar observaciones.

### 13.2 Evolución con juez estable y trazable

La unidad que debe evolucionar es una hipótesis funcional evaluada sobre evidencia identificable. Cadena necesaria:

    gen/hipótesis versionado
      → expresión funcional aplicada al activo y escala
      → predictor/decisión con soporte e incertidumbre
      → orden y fills conciliados
      → outcome neto/censurado atribuido
      → comparación causal con baseline y candidatos
      → promoción registrada o rechazo reversible

FMT-192 protege una operación dentro del ajuste; FMT-193 protege parte del juez; FMT-194 acota muestreo. Ninguno demuestra por sí solo que un gen mutado alcance un consumidor decisorio, que el feedback llegue a su generación correcta o que backtest y producción observen el mismo mundo. Las deudas de rondas anteriores mantienen prioridad.

### 13.3 Criterio para integrar teoría matemática, física o cuántica

Cada incorporación debe especificar observable, hipótesis, ecuación, unidades, método numérico, error controlado, baseline, ablation y coste. Los problemas del milenio no se convierten en componentes útiles por dificultad o prestigio. No se trasladó una PDE física ni una dinámica cuántica al motor en esta ronda.

Una analogía requiere identificar qué entidades observadas corresponden a sus variables y qué predicción falsable añade. En cuántica se debe diferenciar heurística inspirada, simulación clásica y ejecución física; ningún test de esta ronda acredita ventaja cuántica. En algoritmos, un cambio solo es mejora bajo una métrica declarada: corregir ganancia puede aumentar tiempo de entrenamiento y aun así ser obligatorio para coherencia estadística.

### 13.4 Umbrales que siguen siendo política, no teoría

Persisten 80/20 de selección, 80% de bootstrap, 70% de features, 24 cuantiles, 2.048 propuestas, 100 ticks de warmup, mínimos 50.000/5.000, barreras y default de horizonte/stride. No se eliminan sin sustitución evaluada. Cada uno debe inventariarse con unidad, justificación, sensibilidad, fuente y efecto sobre cobertura.

Un límite de memoria, una precondición de finitud o una prohibición de usar futuro no son el mismo tipo de rigidez que una división arbitraria de estados de mercado. Adaptar no significa admitir valores inválidos ni relajar autorizaciones. La función universal debe conservar restricciones contables y de ejecución comprobables.

## 14. Pruebas, resultados negativos y alcance técnico

| Grupo | Nuevas | Anteriores | Resultado |
|---|---:|---:|---|
| Estadísticas de partición | 5 | 0 | 5 pasan; 2 reprodujeron fallo anterior |
| Presupuesto de muestreo | 4 | 0 | 4 pasan; 2 reprodujeron fallo anterior |
| Intervalos, alineación y purga | 6 | 0 | 6 pasan |
| Requisito de test de promoción | 1 | 0 | Pasa |
| Gate numérico/escala | 1 | 0 | Pasa |
| Contrato de ticks | 1 | 0 | Pasa |
| Exportación/residual/predictor retenido de XX | 0 | 7 | 7 pasan |
| Contrato de inferencia de XX | 0 | 12 | 12 pasan |
| Total funcional distinto | 18 | 19 | 37 pasan |

Se ejecutaron cargo test --bin train_forest --offline y cargo test -p god-engine-core --test ml_model_contract --offline. El inventario manual de modelos está ignorado por diseño y no se volvió a ejecutar; no se cuenta como prueba funcional aprobada. Una invocación inicial usó el nombre del harness como filtro y ejecutó cero tests: se corrigió a --test, sin computar ese cero como evidencia.

Para las cuatro reproducciones se extrajeron mecánicamente las fórmulas antiguas a helpers y se observaron sus aserciones fallidas con los siete tests previos pasando. La primera compilación de esa extracción tuvo un error de paréntesis en un cast antes de una comparación; se corrigió antes del rojo funcional y no se atribuye al código previo. La pasada final mantiene las 25 pruebas del entrenador en verde; repetirlas no aumenta el total.

Cargo check --bin god_engine --bin train_forest --offline pasa. Rustfmt y git diff --check de la fuente intervenida pasan. Continúan tres warnings anteriores en evolution-engine: latest_ts, mode y RealWfOutcome.trades sin uso. Se eliminaron del entrenador los dos bindings innecesarios observados en XX, sin tratarlo como reparación funcional independiente.

No se ejecutó el CLI con corpus reales ni un flujo completo de entrenamiento/exportación/publicación. Las pruebas usan datos sintéticos y funciones de contrato. El harness previo de inferencia crea y elimina dos archivos temporales sintéticos de modelo inválido; no borra material del usuario. Compilar/checkear el host no valida su proceso desplegado ni su latencia.

## 15. Hoja de ruta de rehabilitación verificable

1. Terminar el contrato de evidencia: manifiesto de dataset/modelo/target, unidades, reloj, soporte y registro inmutable de experimentos. Registrar hashes de corpus y ventanas usadas, no solo nombre de archivo.
2. Reparar la ruta exportador→DarkAlpha sin asumir que hereda los controles del bosque. Separar ausencia, invalidación y censura; validar dimensiones antes de cualquier estadística.
3. Sustituir el POC optimista por un evaluador con ambigüedad intrabar explícita, costes, terminal y métricas adecuadas; conservar sus resultados históricos como diagnósticos limitados.
4. Migrar objetivos por evento a magnitudes justificadas por reloj/acción donde corresponda. Versionar el cambio y demostrar paridad de entrenamiento e inferencia antes de activar.
5. Cerrar atribución gen→parámetro aplicado→decisión→fill→outcome y comparar backtest/demo/producción con los mismos contratos; no calibrar ocultando fallos de datos.
6. Medir sensibilidad, coste y cobertura de la representación multiescala/multiactivo; solo después ampliar familias teóricas bajo un protocolo de comparación previamente fijado.
7. Continuar los 155 Rust pendientes y el resto del inventario, con estado explícito por archivo. No declarar finalizada una auditoría raíz→cima por completar este bloque.

## 16. Artefacto y conservación de historial

El [artefacto XXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXI_2026-09-24.json>) contiene hallazgos, pruebas, referencias, hashes iniciales/finales, cobertura y limitaciones. El [atlas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/ATLAS_ANALITICO.md>) y el [informe maestro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/INFORME_FORENSE_MAESTRO.md>) reciben únicamente adendas.

Firecrawl se utilizó para consultar documentación primaria pública sobre fuga de información; orientó la separación entre ajuste, selección y evaluación. No se enviaron código privado, genomas, datasets ni credenciales. No se incorporaron citas como sustituto de pruebas locales.

La certificación de esta entrega se limita a los contratos comprobados y a la integridad de lo preservado. La auditoría completa, la publicación Git, la integración de ramas y cualquier activación operacional siguen fuera de lo ejecutado en esta ronda.

## 17. Cierre de verificación documental

El JSON se deserializa, contiene nueve hallazgos —cinco IDs nuevos— y sus grupos suman 37 pruebas funcionales. Los ocho hashes finales de fuentes coinciden; siete son idénticos al inicio de XXI. Los 41 modelos conservan su SHA-256. Se verificaron los 25 enlaces locales del informe y las referencias de evidencia del JSON: destinos existentes y líneas dentro de rango.

Los prefijos históricos del atlas, maestro y XX conservan su SHA-256 normalizado. Se agregaron respectivamente 3.163, 6.465 y 1.348 caracteres bajo normalización CRLF→LF, sin reescribir contenido histórico. Los hashes y longitudes están en el artefacto.

Después del último ajuste se repitieron las 25 pruebas del trainer y el check de ambos binarios: siguen pasando. Formato y diff-check pasan. El diff de Git contra HEAD incluye trabajo previo de otras rondas; no debe atribuirse íntegramente a XXI. El alcance de esta ronda se registra contra los hashes iniciales capturados, no como un commit aislado inexistente.

## Adenda de continuidad XXII — actualización posterior sin borrar XXI

[Informe XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXII_2026-09-24.md>) · [Artefacto XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXII_2026-09-24.json>).

FMT-197 queda reparado localmente: CSV legacy 54D estricto, targets 0/1 finitos, dimensiones estables y rechazo sin imputación. FMT-198 pasa a contención parcial: salida neuronal solo como candidato fuera de models/, sin evaluación temporal independiente aún. El formato sigue sin intervalos/procedencia; FMT-195 y la semántica target_5m no se consideran reparados.

Se actualizan FMT-073/074/076 de IV con congelamiento determinista, validación numérica, abstención y preservación de pesos al preparar buffers. Se agregan FMT-200/201 sobre fidelidad/seguridad de dos diagnósticos auxiliares que no se ejecutaron por sus posibles escrituras de caché. La conexión del antiguo destino neuronal al host se confirma en fuente, no como activación de un proceso observado.

La pasada XXII aprueba 94 tests distintos, 26 nuevos, y checks de cuatro binarios. Cobertura acumulada 136/289 Rust preexistentes, 153 pendientes. train_forest y otras siete fuentes protegidas conservan el hash inicial de XXII, al igual que 41 modelos. Sin entrenamiento real, promoción ni operaciones Git remotas. Esta adenda actualiza estados posteriores; no modifica la evidencia ni los resultados históricos de XXI.
