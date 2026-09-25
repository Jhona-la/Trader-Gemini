# Auditoría científica XXIII — procedencia, primer paso y superficie temporal de etiquetas

Fecha de inicio: 2026-09-24; cierre documental: 2026-09-25. Continuación aditiva de [XXII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXII_2026-09-24.md>). Base local: main, HEAD 59a76de4, con cambios anteriores/concurrentes preservados. Sin verificación remota.

## 1. Dictamen y alcance

Se repara parcialmente FMT-195 mediante un exportador de investigación versionado que conserva horizontes de reloj, desenlaces por lado, censura, procedencia, hash del input e intervalos de información. Se conserva el espacio de 54 posiciones de features, pero los canales no disponibles se representan con null; no se inventan constantes macro ni un lado agresor a partir de profundidad. No se modifica ni reinterpreta un CSV ya guardado.

La nueva salida no es compatible silenciosamente con train_dark_alpha: ese consumidor sigue admitiendo exclusivamente el CSV legacy y su ruta de candidato de XXII. No se ha entrenado un predictor multihorizonte, acreditado un modelo continuo universal, completado la promoción ni demostrado paridad backtest/demo/producción. Se implementa un contrato de evidencia necesario para esas etapas, no su resultado.

Hallazgo central de la raíz: el precio/tiempo de aggTrades puede proceder del exchange, pero los bid/ask y cantidades del tape se construyen artificialmente. En el generador de velas siguen presentes dependencias de valores futuros. Estas limitaciones invalidan una certificación de microestructura observada basada solamente en el nombre REAL o en una cabecera.

Resultado técnico: **70 pruebas funcionales distintas aprobadas, 34 nuevas**. Check de feature_exporter, god_engine, train_dark_alpha y train_forest aprobado. La ayuda del CLI se ejecutó sin abrir datasets. No se ejecutaron exportaciones de datos reales, descargas, entrenamiento, órdenes, promoción, reinicio ni cambios de genomas.

## 2. Cobertura y método

Dos nuevas lecturas completas:

| Fuente preexistente | Líneas al inicio | Resultado |
|---|---:|---|
| src/bin/binance_vision_sync.rs | 483 | Procedencia, reconstrucción, adquisición y orden de eventos |
| src/bin/parquet_to_bin.rs | 199 | Disponibilidad causal de OHLCV y generación de subticks |

La cobertura acumulada pasa de 136 a **138/289 fuentes Rust preexistentes**; **151** siguen pendientes. El inventario base de 1.119 archivos y 24 manifiestos no se considera auditado íntegramente por esa cifra. Las relecturas del exportador, TickSource y tick_replayer, así como consultas parciales del host/StatefulEngine/OrderFlowTracker, no aumentan el contador.

Se inspeccionan las condiciones matemáticas y sus consumidores, no solo palabras como scalping, swing o quantum. Los nuevos tests comprueban la implementación nueva; no se presentan como 34 regresiones ejecutadas contra el programa anterior. Los contraejemplos históricos se distinguen como trazas de código/aritmética. Una prueba de compilación no certifica observabilidad, causalidad, generalización ni latencia productiva.

## 3. Grafo vivo de la evidencia

~~~text
Exchange / OHLCV
  ├─ aggTrades → reconstrucción de pseudo-libro → TGMTICK1        [FMT-202/204/205]
  └─ vela final → subticks con marcas anteriores a disponibilidad [FMT-203]
        ↓
Tape de bytes + origen declarado + orden de archivo
        ↓
Decoder LE validado → hash del snapshot
        ↓
Estado de features por activo declarado, usando solo prefijo
        ↓
Ancla t → superficie de resultados por h y por lado
        ↓
Intervalo de información + censura + ausencia explícita
        ↓
JSONL v2 de investigación
        ↓
Adaptador / evaluación causal / promoción / host                [PENDIENTES]
        ↓
Decisión → ejecución → terminal → atribución genómica            [NO CERTIFICADOS]
~~~

El nodo raíz necesita observaciones y metadatos verificables. El nodo de decisión no debe recibir una probabilidad con otro target. El nodo terminal necesita fills, costes y outcome económico. Los resultados del nuevo módulo son eventos sobre midpoints muestreados, no operaciones terminales.

## 4. Matriz complementaria y conciliación histórica

| ID | Prioridad | Estado | Relación con historial |
|---|---|---|---|
| FMT-195 | P1 | Reparación parcial del exportador mediante v2 | CSV, barreras, horizonte y features de XXI |
| FMT-069 | P2 | Nueva ruta estricta; lector legacy permanece abierto | Procedencia y validación del replay de III |
| FMT-202 | P1 | Contención en v2; productores/consumidores históricos abiertos | Reconfirmación de familia D-233/D-721: no es un fenómeno enteramente nuevo |
| FMT-203 | P1 | Abierto | Residuales de D-691/D-722 en generación de velas |
| FMT-204 | P1 | Canal abstiene en v2; otras rutas abiertas | Amplía la inversión entre rutas observada en FMT-201 |
| FMT-205 | P1 | Abierto | Integridad de adquisición/orden; relación con D-724 |

Los cuatro IDs FMT nuevos organizan trazas concretas de archivos recién leídos; no se suman sin conciliación a la matriz histórica de 305 puntos. FMT-200 y FMT-201 no se cierran: no se modificaron ni ejecutaron los diagnósticos auxiliares. FMT-198 sigue sin evaluación/promoción neuronal completa.

## 5. FMT-202 — aggTrades no identifica el libro L2

### Evidencia de fuente y mecanismo

En [binance_vision_sync:204](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/binance_vision_sync.rs:204>) se calcula base_depth=max(0,25·q,0,1). El lado del trade decide a cuál cantidad sumar q; el semi-spread se fija en 0,00005·precio. La rama mensual duplica esa lógica. Después se escribe la cabecera TGMTICK1 y se presenta el contenido como REAL.

La respuesta oficial de aggTrades contiene identificadores, precio, cantidad, timestamp e indicador comprador-maker. No contiene el libro bid/ask ni su profundidad. Por tanto, reconstruir esos campos requiere supuestos adicionales y no constituye una medición del L2. Se contrastó con la [documentación primaria de Binance](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#compressed-aggregate-trades-list).

Cambiar el suelo antiguo en dólares por un spread relativo evita una distorsión de escala, pero no convierte ese spread en observado. El suelo de cantidad 0,1 también depende de la unidad del activo. No existe una sola profundidad compatible con un precio y tamaño negociados; numerosos libros pueden producir el mismo trade.

### Impacto y límites

OBI, OFI, slippage o probabilidades de fill entrenadas sobre esos campos caracterizan el generador y sus supuestos. Una estrategia rentable bajo ese mundo no queda validada en el libro del exchange. Esto es un mecanismo de divergencia plausible, no una cuantificación causal del PnL observado en producción.

La documentación histórica D-233 ya describía esta familia en otro cargador; D-691 reconocía que el tape REAL no tenía libro. XXIII no borra esa evidencia ni anuncia una primera detección universal. Confirma que el problema sigue en este productor y que la cabecera por sí sola no contiene una garantía de medición.

### Contención y cierre pendiente

La nueva [Tape::parse](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/label_evidence.rs:36>) clasifica TGMTICK1 como TradeDerivedUnverifiedBook; TGMSYNT1 como CandleSynthetic y legacy como LegacyUnknown con opt-in. El manifiesto declara observed_l2_certified=false y el target como midpoint muestreado sin costes.

No se migra la cabecera de archivos existentes ni se cambia TickOrigin::es_real en todos los consumidores. El formato no contiene símbolo verificado, contrato, resolución certificada, trade ID, unidad de cantidad ni indicador maker original. Cierre sistémico: persistir los campos observados y distinguirlos de transformaciones, versionar el productor, adquirir L2 cuando el estimador lo requiera y probar cada consumidor contra esa procedencia.

## 6. FMT-203 — persiste futuro en los subticks de OHLCV

### Dependencias identificadas

[parquet_to_bin:94](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/parquet_to_bin.rs:94>) usa high/low actuales cuando no hay barra previa; esos valores no son conocidos en la apertura. Incluso cuando hay rango previo, el fallback y el suelo del spread dependen de close de la barra actual. Si falta open, se usa close como apertura.

[quarter_vol:111](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/parquet_to_bin.rs:111>) distribuye el volumen final de la vela desde el subtick t+0. Con igual apertura y pasado, dos futuros con distinto volumen generan cantidades diferentes en el instante inicial. Es una violación de disponibilidad causal, independientemente del orden elegido para los extremos.

Los extremos finales high/low se asignan a t+15 s y t+35 s; el cierre final se asigna a t+55 s. OHLCV no identifica cuándo ocurrieron los extremos dentro de la barra ni garantiza que el cierre final ya se conociera cinco segundos antes de terminar un minuto. Elegir el orden usando la apertura previa elimina una dependencia particular del signo futuro, no las demás.

### Contraejemplos y riesgo

Si el rango previo es cero, manteniendo open=100 y el mismo pasado, close=100 induce suelo de spread 0,01 mientras close=200 induce 0,02. La apertura recibe dos spreads distintos según el futuro. Con volumen final 4 frente a 40, las cantidades iniciales pasan de 1 a 10. Son dependencias visibles por sustitución en la fórmula; no se ejecutó el conversor sobre datos del usuario.

Además quedan imprecisiones de schema/tipos, imputación de nulos y sumas ts+15.000/35.000/55.000 sin comprobación de overflow. No se afirma que un timestamp real actual haya desbordado. La cabecera TGMSYNT1 informa síntesis y es una mejora histórica real, pero no vuelve causal el camino generado.

### Estado y criterio de cierre

Abierto. No se regeneró ni alteró ningún tape. Un cierre válido requiere separar simulación intrabar de observación: mantener OHLCV en su tiempo de disponibilidad, modelar explícitamente la incertidumbre de trayectoria o trabajar con datos más granulares observados. Un interpolador o una ecuación física sofisticada no identifica de forma única el orden perdido. Las pruebas necesarias deben fijar el pasado y cambiar el futuro, exigiendo que no cambie ninguna entrada emitida antes de su disponibilidad.

## 7. FMT-204 — profundidad reconstruida, indicador maker y signo de flujo

### Cadena algebraica confirmada

Para un trade con is_buyer_maker=true, el productor genera bq=base y aq=q+base. El exportador legacy y el trainer del bosque pasan bq>aq como si fuera is_buyer_maker: el booleano resultante es false. En [OrderFlowTracker:31](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/microstructure.rs:31>), false suma volumen comprador; true suma vendedor. Por tanto, bajo esa transformación concreta, un trade vendedor termina alimentando volumen comprador. Para is_buyer_maker=false ocurre el caso opuesto.

También se pasa bq+aq=q+2·base, no q. El error no es solo de nombre: cambia signo y peso de la estadística. El indicador original y los IDs no se persisten en BinTick. Invertir la comparación podría recuperar la orientación bajo el productor exacto, pero no certifica un archivo de procedencia desconocida ni un libro realmente observado.

### Revisión de una conclusión anterior

FMT-201 de XXII probaba que el diagnóstico utilizaba la comparación opuesta al exportador, sin determinar cuál era económicamente correcta. Ahora se traza el productor y el consumidor. Para el tape generado por este código, aq>bq concuerda con el indicador original, mientras bq>aq lo invierte. Esto corrige la interpretación pendiente; no convierte el diagnóstico completo en válido ni permite inferir agresor desde cualquier snapshot L2.

### Cambio y deuda

En v2 no se llama update_trade_flow a partir de profundidad y la posición 6 de features se emite como null. La posición 37 conserva un proxy de desequilibrio de cantidades, separado del flujo agresor y sin certificación L2. No se modifican modelos entrenados, train_forest, el host ni los diagnósticos; el problema sigue abierto allí.

Cierre global: conservar trade price/quantity/maker/identity como observaciones separadas de quotes, construir flujo únicamente desde trades y comparar el mismo contrato en replay/live. Las hipótesis de reconstrucción deben quedar identificadas por versión y no mezclarse con datos observados.

## 8. FMT-205 — adquisición parcial y orden temporal insuficientemente gobernados

### Evidencia

El parser [parse_aggtrades_zip](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/binance_vision_sync.rs:173>) retorna true tras recorrer un ZIP incluso si no produjo registros; errores de entradas/lectura/filas se omiten. skip(1) es incondicional: si el contenido carece de cabecera se pierde un registro válido. Esto no afirma que los archivos oficiales inspeccionados carezcan de cabecera; no se descargó ninguno en esta ronda.

El modo diario avanza ante descargas o ZIP inválidos y escribe el resultado parcial. No hay ledger de días completos/fallidos ni requisito de cobertura para publicar el tape. El HTTP de aggTrades no usa error_for_status antes de consumir bytes. El parseo de fechas hace aritmética civil sin comprobar rigurosamente mes/día/forma exacta.

La rama diaria usa sort_by_key estable, pero la mensual mantiene [sort_unstable_by_key:459](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/binance_vision_sync.rs:459>). No se dice que cambie aleatoriamente cada ejecución: no garantiza preservar el orden de los eventos empatados. En primeras llegadas, reordenar un +TP y un −SL con igual timestamp puede cambiar el resultado. El ordinal del archivo no acredita secuencia del exchange.

### Impacto

Un éxito de transporte o un archivo creado no prueba un dataset completo. Huecos pueden parecer periodos sin actividad y alterar memoria, volatilidad, etiquetas o evaluación. La pérdida de IDs impide distinguir duplicados de eventos distintos con igual reloj. File::create sobrescribe destinos conocidos; varios errores de escritura se silencian y no hay protocolo completo de publicación durable/atómica. Estas rutas no se ejecutaron.

### Cierre

Abierto. Se necesita parser de schema declarado con contabilidad de filas, IDs y errores; ledger de cobertura por archivo/rango; validación de HTTP/ZIP/fechas; orden causal identificado; persistencia exclusiva o generación versionada y commit verificable. El hash que añade v2 identifica bytes recibidos, pero no prueba completitud del proceso que los produjo.

## 9. FMT-195 — reparación versionada del contrato de etiqueta

### Por qué era incorrecto

El CSV llamaba target_5m a 500 eventos. Verificaba SL largo antes del TP corto y SL corto antes del TP largo; con TP=0,36% y SL=0,18%, todo toque de TP satisfacía primero la condición del otro lado. Un retorno de +0,20% podía etiquetarse positivo sin alcanzar el TP largo. Los casos sin barrera se mezclaban con otro criterio de retorno terminal o se descartaban.

Una única clase no representaba los dos lados ni la masa de no resolución. Cambiar solo el encabezado no resolvía esa pregunta probabilística ni permitía purgar ventanas de información.

### Definición nueva y significado de cálculos

Para un ancla t₀, midpoint p₀>0 y muestra futura pⱼ:

~~~text
rⱼ = (pⱼ − p₀)/p₀
T⁺long  = primer j posterior al ancla con rⱼ ≥ a
T⁻long  = primer j posterior al ancla con rⱼ ≤ −b
T⁺short = primer j posterior al ancla con rⱼ ≤ −a
T⁻short = primer j posterior al ancla con rⱼ ≥ b
~~~

a y b son barreras explícitas declaradas por el llamador; no están optimizadas aquí. Se restringen a (0,1) para que las barreras de ambos lados sean compatibles con midpoints estrictamente positivos. Se usan fracciones de precio de entrada, no retorno neto sobre margen ni beneficio después de costes.

Cada lado conserva el primer evento que le corresponda. El stop de un lado no termina la búsqueda del otro. Por eso ambos pueden alcanzar stop en una trayectoria que primero sube y luego baja: los resultados no son complementos. La comparación incluye las muestras exactamente en el deadline. Las marcas del mismo milisegundo se ordenan por ordinal del tape, limitación declarada.

### Tiempo, censura y disponibilidad

[label_surface](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/label_evidence.rs:161>) admite una lista positiva y estrictamente creciente de horizontes en milisegundos. No contiene categorías scalping/swing ni cortes de régimen. Los horizontes son consultas sobre una trayectoria compartida; la malla finita no certifica todos los horizontes del universo.

Si no se observa una barrera y el tape alcanza el deadline, el resultado es NoObservedHit. Si termina antes, es RightCensored. Un hit ya observado conserva su identidad aunque el horizonte completo quede censurado. No se imputa una pérdida ni una victoria a la censura y no se descartan esos registros.

Si el último punto anterior al deadline está en t=2 y el siguiente en t=20 para h=5, la salida no utiliza el precio de t=20 como si existiera en t=5. Conserva retorno as-of de t=2, confirma el alcance temporal con timestamp 20 y registra information_end_ms=20. Ese tiempo debe entrar en la purga; usar solo el deadline podría mezclar información de otro conjunto.

horizon_covered significa únicamente que el tape llega al horizonte, no que no existan huecos, eventos faltantes o barreras no observadas. Se registra el mayor gap observado y el tiempo del último punto. No se interpola precio ni se supone un límite de staleness validado. Los hits son del camino muestreado, no del proceso continuo no observado.

## 10. Parser, features y persistencia del nuevo exportador

### Ingestión

El decoder nuevo interpreta explícitamente registros de 40 bytes little-endian sin cast de alineación. Rechaza vacío, truncamiento, cabecera desconocida, no finitos, precios no positivos/cruzados, cantidades negativas/suma desbordada y tiempo decreciente. Conserva empates y orden; no ordena ni deduplica silenciosamente.

El límite de registros se declara antes de la lectura. La entrada se lee hasta presupuesto+1 para detectar exceso y se rechaza completa si supera el presupuesto: no se exporta un prefijo presentado como todo el archivo. El límite es de recursos, no una ley del mercado. No hay garantía universal contra OOM si el operador declara un presupuesto irrazonable. Los lectores legacy no se reemplazaron globalmente: FMT-069 permanece parcial.

### Features

[feature_snapshot](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:126>) conserva posiciones legacy, primeras 34 calculadas por StatefulEngine y derivados 34–39, pero no acredita que todos los defaults internos de StatefulEngine sean observaciones. El canal 6 está ausente por falta de agresor fiable. Los canales 40–53 son null porque este tape no contiene las fuentes externas necesarias.

Los nombres del manifiesto describen v_t/mid, ratio de ATR, compresión de cantidad, desequilibrio de cantidades y transformaciones derivadas; no llaman log-return a cualquier cociente ni Parkinson a min(100·ATR,5). Se conservan transforms heredadas como límites 0,1/5 y escala 1.000 de cantidad: siguen siendo deuda de unidades/calibración, no leyes validadas. No se ha demostrado invariancia de todas las features aunque sí de los labels probados.

Un desbordamiento se identifica antes de clipping para no convertirlo en evidencia plausible. JSON conserva precisión numérica sin redondeo fijo a seis decimales; valores no disponibles se representan como null. Se prueba que cambiar el precio futuro no cambia el snapshot previo, manteniendo iguales sus observaciones previas.

### Manifiesto y finalización

El JSONL empieza con schema tgm.midpoint_barrier_surface.v2 y registra símbolo, unidad de reloj, barreras, horizontes, stride, procedencia, hash SHA-256 y cantidad de bytes/registros. Símbolo y unidad son declaraciones del caller, no metadatos certificados por el binario original. Cada observación guarda ordinal, tiempo de features, inicio/conteo de historia e intervalo de etiqueta. El footer complete registra filas y trabajo usado.

[El CLI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/feature_exporter.rs:223>) crea exclusivamente un destino nuevo, propaga write/flush/sync_all y no sobrescribe un CSV. Un fallo puede dejar salida parcial: no hay publicación atómica por rename. El footer es condición necesaria, no prueba suficiente de fsync exitoso ni autorización de uso. Deben comprobarse salida completa y éxito del proceso. No existe consumidor v2 aprobado en esta ronda.

## 11. Interfaz y cambio de compatibilidad deliberado

La invocación antigua con argumentos posicionales ya no genera un CSV ambiguo. Falla con error de uso. Los datasets y modelos existentes no se borran ni migran. El comando --help se verificó y explica los argumentos:

~~~text
--symbol SYMBOL --input PATH --output NEW_PATH --timestamp-unit ms
--horizons-ms H1,H2,... --stride-ms N
--tp-return A --sl-return B
--max-records N --max-work N
[--allow-legacy]
~~~

No se suministran parámetros de trading por defecto. stride-ms selecciona el siguiente evento disponible tras el intervalo mínimo desde el ancla anterior; no es una observación interpolada de una rejilla perfectamente regular. El estado recibe los eventos previos procesados; el label usa el mismo tape completo validado.

El presupuesto de trabajo cuenta visitas de registros futuros más horizontes emitidos. Agotarlo produce error sin un footer de éxito; no reduce silenciosamente el espectro pedido. No limita por sí solo todo el tiempo de lectura, hashing, cálculo de features, serialización o memoria.

Train_dark_alpha mantiene su contrato legacy estricto y rechazará el encabezado JSONL si se lo pasa como CSV. No hay conversión automática de cuatro estados por lado a una etiqueta binaria. El siguiente adaptador necesita seleccionar una pregunta estadística, conservar censura y particiones, validar el schema y permanecer fuera del motor hasta superar evaluación independiente.

## 12. Teoría integrada y objetivos de investigación

Se materializa una parte del contrato de T17 —primer paso y riesgos competitivos— y T44 —unidad e intervalo de evidencia—, sin añadir nombres de teoría duplicados. Para cada lado y barreras fijas, las probabilidades de TP primero, SL primero y no resolución son funciones del horizonte. El código calcula desenlaces observados, no esas probabilidades ni sus intervalos de confianza.

Una futura curva F⁺(h), F⁻(h), S(h) debe satisfacer F⁺+F⁻+S=1 en el dominio definido y tratar explícitamente censura/observación. Eliminar timeouts cambiaría el objeto a probabilidad condicionada a resolución. La censura dependiente del mercado no se vuelve inocua porque tenga un enum: requiere método y diagnóstico propios.

Desde la raíz, cualquier modelo avanzado sigue condicionado por el operador de observación. Si dos trayectorias producen la misma vela OHLCV, esa vela no identifica cuál tocó primero una barrera ni el estado L2. Una red más grande, una ecuación física o un término cuántico no recuperan de manera única información no observada. Las técnicas útiles deben declarar qué incertidumbre modelan y qué nueva evidencia aportan.

Para el continuo multiactivo: separar X_activo(log h,t), soporte observacional, incertidumbre y política de decisión; probar error al refinar malla y presupuesto; combinar activos con as-of/identidad/unidades, no concatenar archivos y asumir sincronía. Esta ronda etiqueta un activo declarado por tape; no implementa dependencia cruzada, un proceso tensorial conjunto ni un genoma con influencia validada en todas las escalas.

T17/T44 se complementan con T24: tests metamórficos de cambio de unidad de precio y traslado/dilatación de reloj. Estos tests pasan para los labels de las fixtures, lejos de ambigüedades de redondeo; no se extrapolan a todas las features o a instrumentos con reglas diferentes. Los límites de hardware, resolución milisegundo y datos disponibles siguen vigentes. No se acredita soporte de 1 ns a 100 años ni una iteración por nanosegundo.

## 13. Matriz de ocho módulos

| Módulo | Resultado de XXIII | Pendiente |
|---|---|---|
| 1. Ingestión, parsers, L2 | Decoder estricto y procedencia explícita en nueva ruta | Adquisición causal/completa y L2 realmente observado |
| 2. IA/modelos/señales | Se define qué etiqueta y qué evidencia se entrega | Adaptador v2, calibración y modelo probabilístico multihorizonte |
| 3. Multiactivo/horizontes | Lista de horizontes de reloj, lados independientes | Estado conjunto entre activos y soporte empírico continuo |
| 4. Ejecución/red | No se atribuyen fills a midpoints | Slippage, costes, liquidez y replay de ejecución |
| 5. Riesgo/genomas | Se evita llamar outcome económico al label | Atribución gen→expresión→decisión→fill→resultado |
| 6. Memoria/estado/SO | Sin cast en nuevo decoder, budgets, errores de escritura | Todos los lectores legacy y publicación atómica |
| 7. Confluencia/cuántica | Ausencia de canal diferenciada de cero | Consumidores tipados y evidencia de ventaja incremental |
| 8. Backtesting/gobierno | Schema/hash/intervalos y pruebas | Ledger de experimentos, purga/evaluación/promoción v2 |

## 14. Pruebas y verificación

| Grupo | Nuevas | Previas | Resultado |
|---|---:|---:|---|
| label_evidence_contract | 25 | 0 | 25 aprobadas |
| feature_exporter::contract_tests | 9 | 0 | 9 aprobadas |
| train_dark_alpha::contract_tests | 0 | 8 | 8 aprobadas |
| train_forest::contract_tests | 0 | 25 | 25 aprobadas |
| spectral_risk_contract | 0 | 3 | 3 aprobadas |
| Total distinto | 34 | 36 | 70 aprobadas |

Los tests incluyen header/legacy/endianness/alineación, truncamiento, finitud, orden, midpoint sin overflow, barreras alcanzables, ambos stops, clock frente a eventos, límites inclusivos, información posterior de confirmación, censura, casos planos, ordinales empatados, overflow temporal, presupuesto compartido, una sola pasada por múltiples horizontes, escala de precio, cambio de reloj, ausencia de macro/agresor, precisión de serialización, causalidad del snapshot y propagación de errores de writer/flush.

Comandos: cargo test -p backtest-engine --test label_evidence_contract --offline; cargo test --bin feature_exporter --bin train_dark_alpha --bin train_forest --offline; cargo test -p backtest-engine --test spectral_risk_contract --offline. Repeticiones no aumentan el total. Se verificó cargo check de los cuatro binarios y cargo run --bin feature_exporter --offline -- --help. La ayuda no lee datasets ni publica resultados.

La primera compilación del test nuevo tuvo un literal float sin tipo explícito para to_le_bytes; se corrigió a f64. Fue un error de la fixture nueva, no un bug histórico. Un primer rustfmt recibió os error 1224 por una sección de archivo mapeada en Windows; se reintentó al finalizar la compilación y pasó. No se mató ningún proceso del usuario.

Persisten tres warnings previos en evolution-engine y uno de import Arc sin uso en continuous_evolution_backtest. Formato de las tres fuentes nuevas/intervenidas del bloque y diff-check acotado pasan. No se ejecutó toda la suite del workspace ni se midieron latencia, alpha o PnL productivos.

Para A anclas, H horizontes y hasta N puntos posteriores por ancla, el módulo comparte el barrido entre horizontes y cuesta O(A·(N+H)) en el peor caso; no O(A·N·H). Sigue pudiendo ser cuadrático en N con muchas anclas. El budget contiene trabajo, no acredita suficiencia del límite elegido ni una optimización HFT.

## 15. Hoja de ruta de cierre

1. Conservar eventos observados con identidad y unidades; distinguir trades de quotes y reconstrucción de medición. Cerrar FMT-202/204/205 en productores y consumidores, no solo en la exportación.
2. Resolver FMT-203: no generar prefijos usando cierre, volumen o extremos futuros; cuando solo existe OHLCV, representar incertidumbre y disponibilidad.
3. Diseñar adaptador v2 con censura, tiempo de disponibilidad y contratos por activo; separar train/selección/test, purgar intervalos completos y registrar reutilización de evidencia.
4. Estimar y calibrar curvas por horizonte bajo soporte observable; comprobar coherencia, baselines, coste y error de discretización antes de integrarlas con el genoma.
5. Conectar terminales reales y medir atribución causal de cambios genéticos entre backtest/demo/producción. Ningún test de parsing demuestra ese efecto.
6. Reparar los diagnósticos FMT-200/201, compartir constructores sin escrituras de caché y verificar outputs contra snapshots idénticos.
7. Continuar 151 fuentes Rust y demás archivos pendientes; mantener reapariciones y contradicciones históricas documentadas.

## 16. Artefacto, conservación y límites operativos

[Artefacto XXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIII_2026-09-24.json>) contiene estados, referencias, pruebas, hashes, cobertura y límites. Se agregan adendas al [atlas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/ATLAS_ANALITICO.md>), [informe maestro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/INFORME_FORENSE_MAESTRO.md>) y XXII, sin reescribir sus prefijos históricos.

Cambian feature_exporter y una línea de exportación de módulo en backtest-engine/lib; se añaden label_evidence y su suite. Se agrega .firecrawl/ a .gitignore para una nota pública de fuente. Doce fuentes de referencia conservan su SHA-256 inicial, al igual que 41 modelos. El diff contra HEAD contiene trabajo anterior y no se atribuye entero a XXIII.

Firecrawl se usó tras comprobar que su CLI no estaba disponible; el conector obtuvo documentación oficial pública, sin enviar datos privados. Esa fuente orientó la separación trade/libro y la decisión de no fabricar agresor. La nota de procedencia se guarda en .firecrawl/XXIII-binance-source.json y queda ignorada por Git.

No hubo commit, push, merge, fetch, descarga de mercado, ejecución del conversor, exportación de corpus real, entrenamiento, promoción, cambio de cuentas/genomas, órdenes ni reinicio. El nuevo exportador no cierra por sí solo el sistema autoevolutivo ni autoriza resultados de rentabilidad.

## 17. Cierre documental verificado — 2026-09-25

Se repitieron los cinco grupos funcionales al cierre: 25 + 9 + 8 + 25 + 3 = 70 aprobadas, sin fallos. La repetición no incrementa el total ni sustituye una ejecución completa del workspace. Rustfmt acotado y git diff --check de los archivos intervenidos versionados pasan.

El artefacto se parseó como JSON y se validaron las sumas de pruebas/cobertura, 36 referencias de evidencia con archivo y línea dentro de límites, 14 enlaces locales de este informe y 6 enlaces de las nuevas adendas, sin roturas. La comprobación de enlaces no implica haber auditado por completo todos los archivos enlazados.

Se recomprobaron 58 hashes: 17 fuentes/configuración del manifiesto posterior y 41 modelos, sin diferencias frente al snapshot posterior registrado. Dentro de ese conjunto, las 12 fuentes de referencia y los 41 modelos mantienen además su hash previo a XXIII. Los tres prefijos históricos se conservan íntegros tras normalizar CRLF a LF:

| Documento | Caracteres de prefijo conservados | Caracteres agregados |
|---|---:|---:|
| ATLAS_ANALITICO.md | 84.775 | 2.713 |
| INFORME_FORENSE_MAESTRO.md | 661.706 | 7.364 |
| Informe XXII | 36.287 | 1.420 |

Las longitudes se miden como caracteres de cadenas .NET; los hashes corresponden a UTF-8 del prefijo normalizado. No equivalen a bytes ni a número de puntos de código Unicode. Se mantiene la referencia local main/59a76de4; no se contrastó el remoto ni se atribuyeron cambios concurrentes a esta ronda. La fecha del nombre de archivo identifica el inicio de la ronda y se conserva para no romper las referencias existentes.

## Continuación XXIV — 2026-09-25

Se conserva íntegro XXIII. La revisión del consumidor StatefulEngine confirma y corrige la transición de warmup que dejaba EMA/Kalman mal inicializados, tres campos omitidos en reset, validación previa de ticks/OHLCV y errores numéricos de cooldown. El exportador v2 ahora propaga try_process_tick: un registro rechazado no se presenta como observación ni éxito del corpus.

La nueva ronda no certifica las features heredadas ni conecta v2 al entrenamiento. Continúan bandas rígidas, reloj por eventos, rechazo no propagado globalmente y fallos de la API Hawkes. Se agregan cinco diagnósticos de deuda abierta, separados de las 103 pruebas funcionales aprobadas. Quince regresiones se observaron fallar antes de la corrección.

Cobertura 139/289 fuentes Rust preexistentes; 150 pendientes. No se modificaron modelos/genomas ni se reinició u operó el motor. Detalles: [XXIV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIV_2026-09-25.md>) y [JSON](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXIV_2026-09-25.json>).
