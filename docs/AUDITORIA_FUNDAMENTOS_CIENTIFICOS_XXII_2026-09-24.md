# Auditoría de fundamentos científicos XXII — identidad del predictor, evidencia neuronal y fidelidad diagnóstica

Fecha: 2026-09-24. Proyecto: Trader Gemini. Continuación aditiva de [XXI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXI_2026-09-24.md>) y actualización de hallazgos de [IV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_IV_2026-09-24.md>). Estado local observado: main, HEAD 59a76de4, árbol con cambios anteriores. No se verificó sincronización remota.

## 1. Dictamen ejecutivo y límites de certificación

Esta ronda confirma fallos reproducibles en la identidad matemática del modelo servido: preparar buffers eliminaba pesos normales pequeños; congelar la normalización no impedía aprender durante determinadas llamadas de evaluación; validaciones incompletas y activaciones podían convertir corrupción en una predicción aparentemente válida. Se reparan esos contratos locales y se añade rechazo explícito de evidencia CSV inválida. El entrenador neuronal ya no escribe su resultado directamente en el espacio de modelos operativos.

No se certifica que el sistema completo sea autoevolutivo, multiactivo generalizable o continuo en todo horizonte. Tampoco se atribuye el desfase de resultados entre backtest, demo y producción a una única causa: los defectos aquí probados son mecanismos capaces de alterar la función inferida, no una atribución causal sobre operaciones reales. No se ejecutaron órdenes, motores, entrenamiento con corpus reales ni promoción.

Resultado verificable: 94 pruebas distintas aprobadas, de las cuales 26 son nuevas. Son 93 pruebas funcionales y un control de tiempo medio. Doce nuevas reprodujeron el fallo antes del parche. Los cuatro binarios comprobados compilan. Ocho fuentes de referencia y 41 modelos conservan su SHA-256 inicial. Los informes previos se conservan mediante adendas; la matriz histórica de 305 puntos no se sustituye ni se declara cerrada.

## 2. Cobertura raíz → cima y método contra sesgos

El inventario de referencia conserva 1.119 archivos versionados, 289 fuentes Rust y 24 manifiestos Cargo. Las nuevas lecturas completas son [ml_path_probe.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/ml_path_probe.rs:1>), 74 líneas, y [test_ml.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/bin/test_ml.rs:1>), 247 líneas. La cobertura acumulada de fuentes Rust preexistentes pasa de 134 a **136/289**; faltan **153** lecturas completas. Los archivos no Rust no se consideran íntegramente certificados por ese contador.

Se releen completos el núcleo DarkAlpha y su entrenador. Se trazan aristas concretas del host, exportador, entrenador del bosque, carga de NanoForest y daemon neuronal. Una búsqueda o lectura parcial no aumenta la cobertura; tampoco lo hace crear un test. El artefacto registra diferencias entre evidencia estática, reproducción ejecutada, hipótesis descartada y propuesta pendiente.

Hipótesis descartada: no se confirmó doble normalización Scaler más Welford en la inferencia ordinaria. La rama Scaler excluye las ramas Welford. El problema real es la coexistencia de contratos y la posibilidad de cambiar de coordenadas, no una composición que el código no ejecuta. Esto evita contar una sospecha como bug.

## 3. Paradigma de grafo vivo: identidad de cada arista

~~~text
Nodo raíz: ticks / activo / reloj / unidades
  └─ feature_exporter → CSV legacy sin intervalos ni manifiesto       [FMT-195 abierto]
       └─ parser estricto → TRAIN declarado → Scaler + Adam          [FMT-197 reparado local]
            └─ candidato de investigación fuera de models/          [FMT-198 contención parcial]
                 └─ evaluación causal + autorización                 [NO implementadas aquí]
                      └─ snapshot {pesos, transformación, esquema, soporte}
                           └─ inferencia → evidencia válida o ausencia
                                └─ ensemble / nodo de decisión
                                     └─ orden / fill / terminal / outcome
                                          └─ atribución genómica y promoción
~~~

El grafo distingue una arista observada en fuente de una arista activada en un proceso. El host contiene una lectura de models/DarkAlpha_BTCUSDT.json al construir el motor, en [lib.rs:233](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:233>). Esto confirma conectividad estática del antiguo destino del trainer; no demuestra que un proceso actualmente desplegado haya cargado ese archivo ni qué versión mantiene en memoria.

Un nodo raíz necesita procedencia; un nodo de transformación necesita unidades y versión; un nodo de decisión debe distinguir ausencia de evidencia de probabilidad 0,5; el nodo terminal necesita liquidación/costes/outcome. Una curva de pérdida del entrenamiento no cierra el ciclo evolutivo entre estos nodos.

### Matriz de estado complementaria

| ID | Prioridad | Estado de XXII | Alcance que queda pendiente |
|---|---|---|---|
| FMT-073 | P1 | Reparación local del congelamiento y selección de coordenadas | Snapshot semántico, actualizaciones por reloj, otras APIs de aprendizaje |
| FMT-074 | P1 | Reparación parcial: finitud, formas, estado y rollback de fit | Esquema, unidades, autorización, recursos, cuantización y política de ausencia |
| FMT-076 | P2 | Reparación local de poda implícita y clasificación numérica | Modelos históricamente podados no recuperados; limpieza explícita sigue siendo una transformación |
| FMT-197 | P1 | Reparación local del contrato CSV legacy | Procedencia, intervalos, significado del target y generalización multiactivo |
| FMT-198 | P1 | Contención parcial de publicación | Test independiente, manifiesto, ledger y promoción aprobada |
| FMT-200, nuevo | P2 | Abierto, diagnóstico auxiliar | Sonda 44D no equivalente a la ruta 48D del bosque |
| FMT-201, nuevo | P2 | Abierto, diagnóstico auxiliar | Lectura binaria, supuestos de forma y comparación de coordenadas diferentes |

No hay siete fallos nuevos: son dos nuevos y cinco actualizaciones. FMT-075, FMT-077/078, FMT-195/196/199 y FMT-028 siguen pendientes; las reparaciones del bosque de XXI no se trasladan automáticamente a DarkAlpha.

## 4. FMT-076 — preparar memoria alteraba la función aprendida

### Evidencia y mecanismo

La rutina denominada sanitize_denormals anulaba valores con magnitud inferior a 1e−7 y también valores no finitos. init_buffers la invocaba durante preparación/carga. Esto confundía una política de poda aproximada con clasificación del formato numérico y permitía ocultar corrupción.

En binary64, 1e−8 es un número normal, no subnormal. La documentación oficial fija el mínimo positivo normal aproximadamente en 2,225·10⁻³⁰⁸ y ofrece is_subnormal para identificar la categoría correcta. Esta es una propiedad de representación, no un hiperparámetro financiero. [Rust: f64 e is_subnormal](https://doc.rust-lang.org/std/primitive.f64.html#method.is_subnormal).

Una red escalar con primer peso 1e−8 y segundo peso 1e8 conserva preactivación 1 para entrada 1. Su salida es σ(1) ≈ 0,7310586. Anular el primer peso cambia esa salida a σ(0)=0,5. El peso parece pequeño aisladamente, pero su contribución depende de la composición de capas. Por tanto, un umbral absoluto por coeficiente no acota el error de la función.

### Reparación y pruebas

En [sanitize_denormals](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:115>) se usa is_subnormal; NaN e infinitos permanecen detectables. [init_buffers](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:701>) ya no poda parámetros. Las pruebas comprueban igualdad de pesos y predicción al inicializar buffers, preservación de normales pequeños, detección de NaN y roundtrips JSON/bincode en memoria.

La limpieza explícita de verdaderos subnormales continúa siendo optativa y puede cambiar la función; no se declara pérdida cero. El host y un diagnóstico todavía la llaman expresamente. Esta reparación no recupera coeficientes que ya se hayan eliminado y guardado en artefactos históricos. No se reescribió ninguno de los 41 modelos examinados por hash.

Criterio local de cierre: una operación de preparación de memoria no cambia pesos ni la salida de la fixture; una rutina de clasificación no borra errores antes de validarlos. No equivale a demostrar reproducibilidad bit a bit en toda plataforma, backend o cuantización.

## 5. FMT-073 — freeze, disponibilidad estadística y coordenadas del modelo

### Problema y consecuencia

Antes coexistían excepciones de warmup y reglas diferentes entre predict y predict_for_coin. Un modelo congelado podía incorporar observaciones de evaluación para completar sus estadísticas. Aunque los pesos permanecieran constantes, cambiar media y escala alteraba las entradas de la primera capa. Un backtest y una secuencia live con distinto historial podían terminar evaluando funciones diferentes.

La palabra congelado tampoco garantizaba que un estado global ya utilizable fuese elegido de la misma manera por ambas APIs. Los números de slot no certifican activo ni dominio de entrenamiento.

### Cambio aplicado

[predict_in_context](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:747>) concentra validación, selección de coordenadas y forward. Cuando existe Scaler, se usa su transformación fija. Sin Scaler y con freeze:

1. Para cada canal se prefiere el estado global válido con count ≥ 2.
2. Si falta, predict_for_coin puede usar el estado válido del slot solicitado.
3. Si no hay estado disponible, se devuelve None, sin aprender con la observación ni crear slots estadísticos.
4. La preparación de scratch buffers no modifica el estado serializado.

Dos observaciones son el mínimo algebraico para definir la varianza muestral con denominador n−1, no una garantía de precisión, estabilidad ni tamaño efectivo suficiente. No se sustituyó el antiguo warmup por una afirmación de confianza con dos muestras. Con varianza cero se conserva la política legacy de canal cero; su idoneidad financiera sigue sin probarse.

El uso de coordenadas globales para un slot desconocido significa únicamente que la transformación existe. No autoriza aplicar el modelo a un activo nuevo. El registro de instrumentos y el host deben imponer soporte y autorización.

### Estado adaptativo y límites

La rama no congelada verifica sobre copias la actualización de todos los canales antes de comprometer sus momentos. Si un canal posterior desborda, no quedan actualizados solo los anteriores. Una observación con NaN/Inf se rechaza antes de tocar el estado estadístico.

Este contrato no es una transacción de todo el predictor: una observación numéricamente válida puede actualizar estadísticas y después desbordar en la red; ese fallo posterior no revierte esos momentos. La creación de un slot frío puede preceder al rechazo aritmético. Ambas limitaciones se dejan explícitas.

Freeze controla normalizadores en inferencia, no es una prohibición general de fit, mutación ni plasticidad. fit sin Scaler todavía recorre y actualiza estadísticas por muestra y época; repetir épocas no equivale a recibir nueva información del mercado. No se ha implementado adaptación por tiempo físico ni un estimador con tamaño muestral efectivo.

Pruebas: congelamiento con count 2 y 20, igualdad entre APIs, abstención con estado frío, fallback local sin tocar otros activos, slot extremo sin asignaciones estadísticas en modo congelado y preflight multicanal. Esto acredita contratos locales, no paridad completa backtest/demo/live.

## 6. FMT-074 — números válidos, evidencia ausente y aprendizaje transaccional

### Defectos confirmados

La validación de dimensiones no bastaba: podían existir pesos/biases no finitos, Scaler incompleto o con desviaciones negativas y momentos estadísticos inválidos. Una red con varias salidas se aceptaba aunque la API devolviese solo la primera como probabilidad binaria. Además, una activación ReLU puede transformar un NaN en cero si solo evalúa la comparación sum > 0.

Parámetros y features finitos tampoco garantizan que productos y sumas intermedios sean finitos. Saturar un infinito o neutralizar un NaN puede convertir un fallo de aritmética en una señal utilizable. La ausencia de evidencia debe propagarse, no inventar una probabilidad.

### Reparación aplicada

[validate](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:634>) exige capas compatibles, una sola salida, parámetros finitos, Scaler de dimensión exacta con desviaciones no negativas y estados estadísticos finitos/no negativos. Se admiten vectores estadísticos vacíos por compatibilidad legacy, pero no longitudes parciales diferentes de la entrada.

Las activaciones de DenseLayer propagan NaN cuando la suma es no finita; predict devuelve None si la evidencia o el cálculo no sirven. El Scaler tiene [scale_checked](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:450>) y comprueba desbordamiento antes de winsorizar. El wrapper antiguo scale marca la salida completa con NaN ante error para que no se mezcle evidencia transformada con ceros plausibles.

fit valida el lote completo, targets dentro de [0,1], dimensiones exactas y learning rate finito/positivo antes de actualizar. Conserva una copia del modelo y restaura el estado si una actualización genera corrupción detectable. La API legacy devuelve unidad, no un error tipado: el llamador aún necesita un resultado de entrenamiento auditable. Las pruebas incluyen lote inválido tras una fila válida y desbordamiento con entradas finitas.

### Compatibilidad, coste y deuda

No se cambiaron las escalas absolutas heredadas del Scaler: divide por s si s>1e−4, usa 1e−4 si 1e−6<s≤1e−4 y anula el canal si s≤1e−6. Tampoco se eliminó el clipping de Welford a ±5 ni la compresión de colas. Mantenerlos evita una migración silenciosa de modelos; no demuestra invariancia a unidades ni optimalidad. Cambiar la unidad de una feature puede cruzar un umbral aunque no cambie la información económica.

scale_checked puede modificar parte de su buffer antes de devolver error; sus consumidores nuevos descartan ese buffer. No se promete atomicidad de esa API por sí sola. Las APIs cuantizadas y auxiliares no heredan automáticamente todas estas guardas.

La inferencia conserva compatibilidad de prefijo: acepta features de longitud al menos input_dim y utiliza ese prefijo. La dimensión no identifica semántica. Tampoco hay un manifiesto de activo, reloj, target, normalizador, estado genómico o dataset, ni límites globales de tamaño de artefacto/slots.

Validar pesos y estados en cada llamada añade O(P+C·d), con P parámetros y C slots estadísticos. La prueba aislada observó 8.272 ns por llamada, promedio de 20.000 invocaciones en build de tests sin optimización con debuginfo; el control existente exige menos de 25.000 ns de media. No es p99, ni latencia end-to-end, ni comparación controlada antes/después. Una futura validación de carga cacheada exige inmutabilidad/versionado real: los campos públicos permiten hoy mutaciones posteriores.

## 7. FMT-197 — el dataset neuronal ya no se corrige silenciosamente al leerlo

Antes el trainer permitía filas de tamaños diferentes y después indexaba según la primera; también podía binarizar targets inválidos o sustituir errores de features por cero. El riesgo incluía panic, pérdida de significado y aprendizaje sobre observaciones fabricadas.

[read_training_csv](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/train_dark_alpha.rs:62>) exige el encabezado legacy exacto target_5m,feature_0…feature_53 y 55 columnas por fila. Rechaza targets distintos de 0/1, números no finitos, campos no numéricos, errores de lectura, dimensiones mixtas y dataset vacío. El mensaje indica fila/columna; no omite registros malformados para presentar un corpus aparentemente limpio.

La rigidez de ese esquema es una precondición de compatibilidad de este exportador, no una segmentación de mercados. Ampliarlo requiere un esquema versionado, no aceptar longitudes arbitrarias. El encabezado target_5m sigue siendo una etiqueta sintáctica: no valida cinco minutos reales ni corrige los 500 eventos y las barreras defectuosas de FMT-195.

### Para qué sirve el cálculo del Scaler

fit_scaler calcula momentos sobre las filas declaradas de entrenamiento:

~~~text
δ = xₙ − μₙ₋₁
μₙ = μₙ₋₁ + δ/n
M2ₙ = M2ₙ₋₁ + δ·(xₙ − μₙ)
s = sqrt(M2ₙ/n)
~~~

M2 representa la suma de desviaciones cuadráticas y s la desviación poblacional del conjunto de ajuste. No es una estimación de retorno, incertidumbre predictiva ni evidencia fuera de muestra. Se rechazan formas/números inválidos y desbordamiento. Se conserva el suelo de serialización 1e−8, documentado como política heredada, no como ruido aprendido. La fixture [1,3] devuelve media 2 y desviación 1; un canal constante conserva el suelo.

El Scaler se ajusta sobre todo el CSV porque este se declara TRAIN, sin presentar ese mismo corpus como test. El trainer informa explícitamente evidencia de ajuste. No se creó una separación aleatoria o cronológica ficticia a partir de un CSV que no contiene intervalos informacionales. Faltan hashes de corpus, timestamps, identidad por fila, límites de memoria y un ledger persistente de rechazos. FMT-197 queda reparado en su contrato sintáctico-numérico local, no en la validez económica de los datos.

## 8. FMT-198 — separar candidato de investigación y autorización operacional

El trainer anterior normalizaba y entrenaba todo el CSV, reportaba BCE de ajuste y escribía DarkAlpha_<symbol>.json en models/. La conexión estática de ese directorio con el host convierte esa escritura en una arista relevante de gobierno, aunque no se haya demostrado activación en un proceso concreto.

Ahora produce exclusivamente artifacts/training/DarkAlpha_<SYMBOL>_CANDIDATE_<run_id>.json. El símbolo debe ser ASCII mayúscula/numérico para evitar que se use como componente de ruta arbitrario. create_new impide sobrescribir un destino existente; write_all y sync_all propagan errores. El run_id en nanosegundos de reloj distingue nombres, pero no demuestra unicidad matemática: una colisión provoca error. No se acepta ni se añade --promote en esta ruta.

La red se valida y congela antes de serializar. Se eliminó el sembrado que equiparaba el activo entrenado al slot numérico cero. El Scaler es la transformación usada por este entrenador; los slots no son un registro de símbolos.

La persistencia del candidato no es una publicación atómica mediante rename: un fallo de I/O puede dejar un candidato incompleto. Está fuera del espacio operativo y no debe interpretarse como autorizado. Los helpers públicos save/save_json y otros trainers tampoco quedaron gobernados por esta modificación.

Los ocho epochs, arquitectura 54→64→32→1, tamaño de lote, learning rate, regularización y Adam permanecen. Validar el artefacto final no certifica todos los momentos del optimizador ni que cada log de pérdida fuese válido. No hay evaluación causal independiente, calibración, corrección por experimentos repetidos ni registro de soporte. Por ello FMT-198 es contención parcial, no cierre.

Este cambio altera deliberadamente el destino del CLI: cualquier workflow que esperase una actualización directa en models/ debe pasar por un futuro gate explícito. No se entrenó ni se produjo un candidato real en esta ronda.

## 9. FMT-200 — una sonda presentada como exacta usa otro contrato de entrada

**Localización:** [ml_path_probe.rs:1](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/ml_path_probe.rs:1>), carga línea 9, vector líneas 44–48 y diagnóstico de None líneas 50–66. Prioridad P2 por falsa confianza diagnóstica; no es prueba de órdenes incorrectas en el host.

La sonda declara reproducir el camino exacto de inferencia, pero concatena 34 features universales y 10 espectrales en 44 posiciones. El entrenador del bosque construye además cuatro canales macro, formando 48D. La sonda no incluye ese contexto ni prueba paridad de estado/warmup/as-of. Si un árbol requiere alguno de los índices macro, un None puede deberse a dimensión, no a datos no finitos. Un bosque que no use esas posiciones puede devolver un número; eso tampoco acredita equivalencia integral del pipeline.

La salida de diagnóstico solo cuenta índices no finitos al recibir None. Por tanto, puede presentar un rechazo sin índice culpable y orientar la investigación hacia la causa equivocada. Si falta el modelo global, esa rama ni siquiera incrementa el contador de None. Los filtros bid<=0, ask<=0 y bid>ask no rechazan NaN por sí solos.

Se fija un único activo/archivo y se muestrea por índice de evento cada 137 ticks; no se declara una medida temporal equivalente al host. El replay usa layout nativo de BinTick y trunca sobrantes al dividir tamaño; no es un parser portable de contrato binario.

**Efecto lateral relevante:** load_global llama a load_model; este puede generar/sobrescribir la caché binaria tras leer JSON válido, en [ml_inference.rs:176](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/ml_inference.rs:176>). Ejecutar la sonda no sería una inspección estrictamente de solo lectura.

**Criterio de cierre:** construir features mediante la misma función versionada y snapshot causal que el consumidor; comparar vectores y scores antes de interpretar distribuciones; clasificar ausencia de modelo, dimensión, datos y aritmética por separado; usar carga sin activación ni escritura de caché y fixtures sintéticas. No se ejecutó ni modificó esta sonda.

## 10. FMT-201 — el diagnóstico auxiliar puede comparar otro problema o fallar antes de medirlo

**Localización:** [test_ml.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/bin/test_ml.rs:1>). Es un binario main, no un conjunto de pruebas unitarias aprobado por llamarse test. Prioridad P2 por estar en una ruta auxiliar; el riesgo de memoria requiere atención antes de usarlo.

### Lectura y seguridad de representación

La ruta de ticks consume ocho bytes incondicionalmente sin verificar TGMTICK1. Ante un archivo legacy sin encabezado elimina parte del primer registro: puede fallar la lectura posterior o reinterpretar campos desplazados si hay suficientes bytes. No se informa ese desajuste como causa.

Luego crea Vec<u8> y fabrica &[BinTick] mediante from_raw_parts. Una asignación solicitada para bytes no establece por el tipo la alineación de BinTick. La precondición de alineación es obligatoria incluso si el asignador observado suele satisfacerla incidentalmente. La documentación oficial la exige expresamente; si no se cumple, el comportamiento es indefinido. No se observó ni se provocó UB en este equipo. [Rust: seguridad de from_raw_parts](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html#safety).

Tampoco se codifica endianness del archivo; repr(C) no convierte bytes externos a formato portable. La reparación propuesta es un lector compartido con encabezado validado, tamaños/truncamiento explícitos y from_le_bytes por campo, o un contrato de alineación demostrado y verificado. No basta con cambiar el cast.

### Formas y semántica de las entradas

Se indexan 32 pesos de layer3 tras deserializar sin comprobar que esa longitud exista. Un modelo pequeño válido puede hacer fallar el diagnóstico aunque la API soporte su arquitectura. También se recorren medias e indexan desviaciones sin validar correspondencia. Los unwrap de carga/predicción no separan abstención esperada de corrupción.

En el replay neuronal, pseudo_maker usa ask_qty>bid_qty, mientras exportador y sonda usan bid_qty>ask_qty. Esto demuestra una inversión entre rutas, no que cualquiera de esas aproximaciones identifique realmente el lado agresor. El libro no sustituye automáticamente un trade observado.

Las features de diagnóstico incluyen constantes macro copiadas; no son un snapshot del contexto del host. La primera simulación mezcla kline/OFI/flujo y ceros en un vector 54D sin acreditar paridad causal. Probar ceros, unos o rampas sirve como smoke test aritmético, no como prueba de relevancia económica.

### Comparación no controlada

La segunda variante clona pesos, elimina Scaler y activa Welford. Cambia la función de entrada y su adaptación, no solo una opción de rendimiento. Comparar medias, rangos o porcentaje p≥0,5 entre ambas variantes no indica cuál tiene alpha, calibración o menor riesgo. No hay etiquetas externas ni evaluación de pérdidas/costes común en ese diagnóstico.

La carga de NanoForest también puede escribir cachés. Se preservó el archivo sin ejecutarlo. Para cerrar FMT-201 se requiere replay seguro, esquema/estado/transformación idénticos, errores tipados, índices derivados de formas validadas y comparación de outputs contra un oráculo congelado. Solo después procedería una evaluación predictiva fuera de muestra.

## 11. Implicaciones multiactivo, temporales y genómicas

La ruta [del host:2309](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:2309>) restringe inferencia neuronal al símbolo BTCUSDT. Fuera de ese símbolo no ejecuta la red, pero crea Some(0.5) y lo envía al ensemble como DarkAlphaNN. Esto no es ausencia a nivel de interfaz. Cuánto modifica pesos/confianza/decisión requiere seguir el contrato del ensemble; no se cuantifica aquí. No se retiró esa protección por activo ni se habilitó inferencia indiscriminada.

La construcción 54D tampoco es por sí sola universal: hay rutas 12D/34D y un fallback al tensor 54D. El exportador usa constantes donde el host puede disponer de macro/funding/contexto real. Un canal constante en entrenamiento tiene escala prácticamente nula y puede quedar anulado por el Scaler aunque en vivo varíe. Eso no se arregla introduciendo una ecuación más compleja en la capa siguiente.

Un sistema espectral multivariante necesita conservar simultáneamente activo, unidad, reloj, soporte temporal, representación y estado adaptativo. No es necesario asignar categorías scalping/swing para representar horizontes; tampoco basta con renombrar variables o eliminar todas las restricciones. Las precondiciones numéricas, causales y de ejecución son distintas de una partición arbitraria de mercados.

La expresión genómica debe poder trazarse hasta parámetros efectivamente usados, decisión, fill y outcome con versiones. Si el predictor cambia al cargar o el normalizador aprende de la evaluación, se pierde control sobre la intervención atribuida al genoma. Las reparaciones reducen dos fuentes de esa pérdida, pero no prueban que expliquen el rendimiento real observado. Los genomas y el daemon evolutivo no fueron modificados.

Representar horizontes parametrizables entre nanosegundos y décadas no acredita resolución observacional a nanosegundos, frecuencia de cálculo por nanosegundo ni identificabilidad a cien años. Esas tres propiedades requieren presupuestos y evidencia diferentes. La cobertura científica debe declarar dónde estima, dónde extrapola y dónde se abstiene.

## 12. Fundamento matemático: profundización de T25 y T44, sin duplicar teorías

Se conserva T25 de IV: un cambio de normalización es una intervención sobre el predictor. Para una transformación estrictamente afín z=D⁻¹(x−μ), con D diagonal invertible, y primera capa a=Wz+b, conservar a al usar μ′,D′ requiere:

~~~text
W′ = W D⁻¹ D′
b′ = b + W D⁻¹(μ′ − μ)
W′ D′⁻¹(x − μ′) + b′ = W D⁻¹(x − μ) + b
~~~

La igualdad resulta de sustituir y cancelar; no es una teoría nueva en esta ronda. Explica por qué no se puede anunciar adaptación inocua cuando se cambian estadísticas manteniendo pesos. Sirve para diseñar una prueba metamórfica de conservación de logits.

No se implementa ese transporte automáticamente: los suelos de escala, clipping y winsorización rompen el supuesto afín global; escalas cero no son invertibles; estados de Adam también tienen coordenadas. El congelamiento corregido establece un contrato más limitado y comprobable. Una migración futura necesita versión conjunta de transformación/pesos/optimizador, soporte, límites de error y rollback.

T44 añade unidad de evidencia e intervalos: cada muestra debe registrar cuándo empezó y terminó la información de su etiqueta. El CSV actual no permite purgar solapamientos ni demostrar independencia temporal. Un número BCE representa ajuste de una probabilidad a una etiqueta definida; si esa etiqueta mezcla barreras incompatibles y horizonte por eventos, una BCE baja responde a ese problema, no necesariamente al objetivo operable.

El plan científico sigue siendo formular un fallo, identificar variables/estimador, construir baseline, probar invariantes, evaluar fuera de muestra y medir coste. No se incorporan ecuaciones de problemas del milenio, analogías físicas ni terminología cuántica como sustituto de una integración falsable. No se ha demostrado ventaja de hardware cuántico ni de una formulación cuántico-inspirada en esta ronda.

## 13. Ocho módulos: qué cubre esta ronda y qué no

| Módulo del informe maestro | Arista revisada | Estado de evidencia |
|---|---|---|
| 1. Ingestión, parsers, L2 | CSV neuronal y replay auxiliar | Parser corregido; ticks/labels del exportador pendientes |
| 2. Inferencia y señales | Scaler, Welford, capas, carga | Contratos numéricos probados; calibración no certificada |
| 3. Multiactivo y horizontes | Símbolo/slot, 44D/48D/54D y target_5m | Incompatibilidades trazadas; no nuevo motor espectral |
| 4. Ejecución HFT y Binance | Posibles consumidores de evidencia | Sin cambios de red/órdenes; sin latencia productiva |
| 5. Riesgo, Kelly y genomas | Atribución dependiente del snapshot predictor | Vínculo lógico documentado, no retorno causal medido |
| 6. Estado, memoria y SO | Freeze, scratch, desbordamiento y cast auxiliar | Pruebas locales; riesgo portable del diagnóstico abierto |
| 7. Cuántica y confluencia | Ausencia frente a voto 0,5 | Trazabilidad parcial; sin ventaja cuántica acreditada |
| 8. Backtesting y gobernanza | Candidato separado, pruebas y conservación | Contención de publicación; falta evaluación causal neuronal |

## 14. Pruebas ejecutadas y límites de la evidencia

| Grupo | Nuevas | Existentes | Resultado |
|---|---:|---:|---|
| neural_evidence_contract | 18 | 0 | 18 pasan; primeras 12 fallaban antes |
| Contratos de train_dark_alpha | 8 | 0 | 8 pasan |
| Biblioteca dark-alpha-engine, funcionales | 0 | 30 | 30 pasan |
| Control de media de inferencia | 0 | 1 | Pasa; no p99 |
| Contratos previos de train_forest | 0 | 25 | 25 pasan |
| ml_model_contract | 0 | 12 | 12 pasan; 1 inventario manual ignorado |
| Total distinto | 26 | 68 | 94 pasan |

Comandos ejecutados: cargo test -p dark-alpha-engine --test neural_evidence_contract --offline; cargo test -p dark-alpha-engine --lib --offline; cargo test --bin train_dark_alpha --bin train_forest --offline; cargo test -p god-engine-core --test ml_model_contract --offline. La medida aislada usó test_inference_speed con nocapture. Repeticiones no aumentan el total.

Cargo check --bin god_engine --bin train_dark_alpha --bin train_forest --bin auto_trainer_daemon --offline pasa. Persisten tres warnings previos en evolution-engine: latest_ts, mode y RealWfOutcome.trades sin uso. No se ejecutó un test completo de todo el workspace ni el flujo real CSV→Adam→archivo→host.

Las primeras doce pruebas se ejecutaron contra el comportamiento anterior y fallaron; luego pasaron. Las seis neuronales posteriores amplían cobertura, pero no se presentan como reproducciones históricas ejecutadas. Dos unit tests existentes se actualizaron porque exigían la antigua imputación neutral de corrupción; la continuidad del nombre no convierte ese contrato antiguo en válido.

Los tests usan fixtures sintéticas. El harness previo de NanoForest crea y elimina archivos temporales de modelo inválido; no elimina modelos del usuario. Las pruebas de serialización neuronales son roundtrips en memoria, no un experimento de crash-consistency del filesystem. No se ejecutaron los dos diagnósticos auxiliares con sus efectos laterales.

## 15. Hoja de ruta de rehabilitación verificable

1. Reparar FMT-195 con esquema versionado: objetivo operable, reloj/unidades, intervalos, barreras/censura y procedencia. Regenerar datos solo después de tests del contrato; no reinterpretar silenciosamente el CSV antiguo.
2. Cerrar FMT-198: train/selección/test purgados, evaluación congelada, costes y calibración; manifiesto y ledger de reutilización de evidencia; promoción explícita y reversible.
3. Corregir FMT-200/201 antes de usar los diagnósticos como certificadores. Compartir decoder y constructor de features; carga de solo lectura; aserciones de paridad, no histogramas sin target.
4. Completar FMT-074: versiones de esquema y unidades, soporte por activo/horizonte, snapshot de normalizador/pesos, autorización y límites de recursos; tipar abstención y error hasta el ensemble.
5. Resolver FMT-075 y FMT-077/078 sin borrar historia: inicialización, formas del daemon y coordenadas train/serve necesitan sus propias pruebas. No activar un trainer porque compila.
6. Medir el coste de validación por versión inmutable y distribución de latencia, con p50/p99 y carga representativa. La optimización no debe volver a transformar corrupción en neutralidad.
7. Cerrar la atribución gen→parámetro→decisión→fill→outcome bajo el mismo contrato en backtest/demo/producción; después comparar adaptadores continuos multiescala con baselines.
8. Continuar las 153 fuentes Rust y demás archivos pendientes. Ni esta ronda ni una lista de tests autorizan afirmar auditoría total o ausencia de bugs.

## 16. Conservación, artefacto y uso de fuentes

[Artefacto XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXII_2026-09-24.json>): IDs, evidencia, estados, pruebas, fuentes, hashes iniciales/finales y límites. Se agregan adendas al [atlas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/ATLAS_ANALITICO.md>) y al [informe maestro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/INFORME_FORENSE_MAESTRO.md>), además de una actualización al informe XXI.

Cambian dos fuentes preexistentes y se añade un archivo de tests. Se preservan ocho fuentes de referencia y los 41 modelos por SHA-256. Los prefijos históricos se comprueban normalizando CRLF a LF, no como identidad byte a byte del formato de finales de línea. El diff contra HEAD incluye trabajo previo: no todo lo que muestra Git pertenece a XXII.

Firecrawl se utilizó, con CLI no disponible y conector alternativo, para consultar documentación primaria de Rust sobre subnormales y precondiciones de memoria. Esas consultas influyeron en la corrección numérica y en la clasificación prudente del riesgo de alineación. No se subió código privado, corpus, credenciales ni genomas. Las fuentes no sustituyen reproducciones locales.

No se hizo commit, push, merge, fetch, despliegue, reinicio, cambio de cuentas, genomas, órdenes ni promoción. La solicitud histórica de unificar ramas no se interpreta como permiso para mezclar este árbol sucio sin revisión de conflictos y pruebas. El estado remoto sigue sin comprobarse.

## 17. Cierre de verificación documental

El JSON se deserializa y registra siete hallazgos —dos IDs nuevos—; los grupos de pruebas suman 94. Se verificaron los 18 enlaces locales del informe y todas las referencias de evidencia del artefacto: destinos existentes y líneas dentro de rango. Los once hashes finales de fuentes coinciden, incluidas dos fuentes modificadas, ocho preservadas y un archivo de tests nuevo. Los 41 modelos conservan el hash inicial.

Los prefijos históricos normalizados de atlas, maestro y XXI conservan sus SHA-256. Se agregaron respectivamente 3.520, 7.843 y 1.517 caracteres bajo normalización CRLF→LF. Los controles de formato de las tres fuentes Rust intervenidas y diff-check acotado pasan. La inspección final mantiene HEAD 59a76de4; no es un commit creado por esta ronda.

La validación documental no demuestra ausencia de otros bugs. Los estados abiertos y parciales se mantienen, así como la prohibición de interpretar un candidato de investigación o un test sintético como autorización operacional.

## Adenda de continuidad XXIII — contrato de datos y productores

[Informe XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIII_2026-09-24.md>) · [Artefacto XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIII_2026-09-24.json>).

FMT-195 pasa a reparación parcial mediante exportación JSONL v2 de investigación: horizontes de reloj, barreras por lado, censura, procedencia, hash e intervalos. No cambia el CSV previo ni se conecta automáticamente al trainer legacy; FMT-198 sigue pendiente en evaluación/promoción.

Las nuevas lecturas del descargador y conversor OHLCV documentan FMT-202–205: pseudo-libro tratado como REAL, fugas temporales residuales, inversión maker→flujo y adquisición/orden incompletamente gobernados. Respecto a FMT-201, ahora se confirma que aq>bq reproduce el indicador maker del generador específico, mientras bq>aq lo invierte; no se valida el diagnóstico completo ni una regla general para L2. V2 omite canal 6 ante ausencia de indicador fiable.

70 pruebas pasan, 34 nuevas. Cobertura 138/289 Rust preexistentes, 151 pendientes. Los doce archivos de referencia protegidos y 41 modelos mantienen hashes iniciales de XXIII. Sin corpus real exportado, entrenamiento, promoción, genomas, operaciones Git remotas ni órdenes. Los resultados y archivos de XXII se mantienen como evidencia histórica.
