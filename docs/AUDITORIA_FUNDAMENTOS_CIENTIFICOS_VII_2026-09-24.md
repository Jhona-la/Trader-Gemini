# Auditoría científica VII — soporte espectral, evidencia y selección evolutiva

Fecha: 24 de septiembre de 2026. HEAD local de referencia: **59a76de4**, workspace con cambios concurrentes. Continuación de las rondas I–VI. Este documento registra reparaciones locales autorizadas, hallazgos abiertos y límites; no sustituye los informes anteriores.

## 1. Resultado ejecutivo

La intervención se concentra en dos contratos conectados: **qué significa el exponente que llega al motor** y **qué evidencia determina el orden de los genomas**. No crea motores scalping/swing, no añade dependencias, no modifica genomas ni ejecuta evolución/trading. El nombre de una técnica no se acepta como prueba de sus garantías.

Se corrigen cuatro fuentes y se añaden dos archivos de tests:

- HurstDfa: estabilidad de retornos, homogeneidad numérica, publicación de pendiente sin clipping y soporte efectivo; rechazo de interpretaciones fuera del modelo adoptado.
- CmaEsOptimizer: penalización coherente con fitness de ambos signos, invalidación de puntuaciones no representables y preservación del optimizador cuando no hay suficientes padres válidos.
- evolution-engine/lib: ruta explícita de backtest sin fingir que crecimiento de capital es Sharpe en vivo.
- math_kernels: documentación del consumidor que distingue R² descriptivo de confianza estadística.
- Tests: ocho nuevos en DFA, seis en CMA y dos del consumidor RecursiveHurst; dieciséis en total.

Hay **seis regresiones de defectos preexistentes observadas fallando antes del arreglo**. Una séptima regresión detectó un defecto de precisión introducido en una versión intermedia de esta intervención y quedó corregido antes del cierre. Se documenta aparte: no se presenta como un séptimo fallo heredado.

**No se cierra la auditoría integral ni FMT-113.** Host, replay y risk-engine/lib conservan sus cambios ajenos. La cantidad final de ejecución sigue requiriendo integración coordinada. La existencia de mejoras locales no certifica equivalencia backtest/demo/producción, rentabilidad, cobertura temporal universal o autoevolución de extremo a extremo.

## 2. Matriz de resolución de esta ronda

| Referencia | Problema | Estado |
|---|---|---|
| FMT-011, previo | Penalizar fitness negativo mejoraba su clasificación | Corregido localmente en CMA, con ranking reproducido |
| FMT-117, nuevo | Soporte de muestras presentado como soporte temporal y confianza universal | Abierto operacionalmente; documentación y diagnóstico mejorados |
| FMT-118, nuevo | Validez DFA dependía de amplitud por umbral absoluto y cuadrados | Corregido localmente con normalización homogénea |
| FMT-119, nuevo | Cociente de precios perdía retornos logarítmicos finitos | Corregido localmente, incluida precisión de grandes caídas |
| FMT-120, nuevo | Clipping convertía pendiente fuera del modelo en Hurst válido | Corregido localmente; estimación sin clipping conservada |
| FMT-121, nuevo | Innovaciones de tests supuestamente centradas eran todas negativas | Corregido; contrato de las pruebas revisado |
| FMT-122, nuevo | Crecimiento de capital se interpretaba como Sharpe live | Corregido en el caller localizado; no añade evidencia live |
| FMT-123, nuevo | Sustitución −1e9 podía hacer ganar un fitness inválido | Corregido localmente; padres inválidos no actualizan memoria |
| FMT-003/012/013 | Proxy ML, geometría CMA y sobrescritura de sigma | Siguen abiertos |
| FMT-113 | El presupuesto no limita todavía la Q final en todo el flujo | Sigue abierto |

Las siete fichas nuevas no se agregan ciegamente al antiguo total de 305 fallos. Una matriz global exige deduplicación por causa y corte de código. “Corregido localmente” no significa desplegado ni validado financieramente.

## 3. Grafo diagnóstico desde la raíz hasta la decisión

~~~text
Precio + identidad + reloj
  ├─ process_tick: umbral de vela interna → muestra para RecursiveHurst
  └─ process_kline: cierre recibido       → muestra para RecursiveHurst
       ↓
  HurstDfa: retornos → perfil → F(s) → pendiente cruda + R² + soporte
       ↓
  RecursiveHurst: dominio del modelo + política R² → salida / fallback
       ↓
  Estado del motor → consumidores de régimen, horizonte y riesgo
       ↓
  Acción propuesta → FMT-113 aún pendiente en la cantidad final

Genoma candidato → replay → fitness canónico + métricas
       ↓
  update_backtest_only: el crecimiento se conserva como metadata
       ↓
  ranking admisible → memoria PSO/CMA → candidato escogido
       ↓
  gates de promoción existentes: no ejecutados en esta auditoría
~~~

El nodo raíz no es “el precio” aislado: incluye procedencia y reloj. El nodo de decisión no puede interpretar una pendiente sin su soporte ni una métrica por la posición de una tupla sin verificar su significado. El nodo terminal financiero no queda certificado por una batería verde de estimadores.

La conexión de DFA es real, no un módulo huérfano: [RecursiveHurst](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/math_kernels.rs:516>) encapsula HurstDfa y recibe muestras desde [process_tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:607>) y [process_kline](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:710>). Son lecturas dirigidas del flujo, no afirmación de auditoría completa de stateful_engine.

## 4. Hallazgos nuevos, mecanismos y criterios de cierre

### FMT-117 · P1 · El soporte observado no equivale al universo temporal representable

**Evidencia.** HurstDfa sólo recibe precio, retiene hasta 1.024 retornos y ajusta escalas de 4 a 256 muestras; no conoce timestamps, huecos ni edad del último dato. El caller por tick actualiza una vez cuando event_time_ms−kline_start_ms≥60.000 y luego desplaza el origen al evento recibido. El caller de warmup recibe close sin timestamp en su firma. La selección por reloj reduce dependencia del número de eventos, pero no demuestra que cada muestra cubra exactamente el mismo intervalo ante irregularidad o huecos.

**Contradicción concreta.** El comentario anterior decía que 512 muestras garantizaban cuatro ventanas de la escala mayor. Para s=256 sólo hay dos; el código la omite. A 512 retornos el máximo realmente utilizado es 128, con seis escalas. Sólo a 1.024 se admite 256. La regresión verifica ambas situaciones y ahora se publican scales_used y max_scale.

**Consecuencia matemática.** Un exponente ajustado sobre s no identifica automáticamente la misma ley sobre segundos. Incluso con cadencia exacta de un minuto, el ajuste abarcaría aproximadamente 4–256 minutos; no nanosegundos, décadas ni todo el historial. Usar ese H para extrapolar es una hipótesis adicional de estabilidad/escalamiento, no una medición de esas escalas.

**Acción realizada.** Documentación del estimador y del wrapper corregida, soporte efectivo visible y R² descrito como ajuste. Se conservan las ventanas como discretización finita del estimador, no como tipos de operación. El umbral R²=0,85 del consumidor y la cadencia se mantienen como políticas existentes; no se presentan como valores derivados de un teorema.

**Pendiente.** Contrato temporal con timestamps y procedencia por observación; política explícita ante huecos; propagación de soporte, edad y versión hasta la decisión. No se rellenaron huecos con precios repetidos, pues eso fabricaría dependencia. No se retiraron indiscriminadamente filtros de calidad para aumentar operaciones.

**Cierre exigible.** Secuencias idénticas bajo distintas particiones de transporte, huecos deliberados, cambios de cadencia y warmup/replay deben producir un diagnóstico de soporte coherente, no sólo números parecidos. La extrapolación fuera del dominio observado debe distinguirse de una estimación identificada.

### FMT-118 · P2 · Una ley homogénea perdía validez al cambiar únicamente la amplitud

**Evidencia.** recompute descartaba escalas con F(s)≤10⁻¹⁵. Además, elevar residuos pequeños o enormes al cuadrado producía underflow/overflow. Por ello podía aceptar una serie y rechazar su múltiplo positivo, aunque la pendiente de DFA no cambia algebraicamente con ese múltiplo.

**Reproducción.** contract_dfa_is_invariant_to_return_amplitude falló primero con factor 10⁻²⁰⁰. La batería final compara 1, 10⁻²⁰⁰, 10⁻¹⁶, 10⁻⁸, 10⁸ y 10²⁰⁰, verificando validez, pendiente y R² con tolerancia numérica. Estos extremos se inyectan en el buffer de retornos de un test interno: no representan mercados observados ni precios plausibles.

**Reparación y fundamento.** Se normaliza por A=max|rᵢ|>0 antes de promediar, integrar y elevar al cuadrado. DFA es homogéneo: F(r/A)=F(r)/A. Por tanto ln F cambia en una constante −ln A y conserva pendiente/R². Los inputs normalizados están acotados y no se impone un piso absoluto de volatilidad para decidir qué escalas existen. Se exige F>0 finito.

**Frontera.** Normalizar aquí no es normalizar volatilidad económica del portafolio: sólo estabiliza un estimador de forma/escalamiento. No recupera información perdida al cuantizar precios ni demuestra invariancia exacta en todo binary64. Una serie sin fluctuación no identifica un exponente; se invalida y limpia el diagnóstico anterior.

**Evidencia de código.** [recompute](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hurst_dfa.rs:185>) y [regresión de amplitud](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hurst_dfa.rs:536>).

### FMT-119 · P2 · Retornos finitos desaparecían por la forma de calcular el logaritmo

**Mecanismo previo.** ln(Pₜ/Pₜ₋₁) requiere primero representar el cociente. Dos precios positivos finitos pueden tener un cociente infinito o redondeado a cero, aunque ln(Pₜ)−ln(Pₜ₋₁) sea finito. El código actualizaba el ancla y descartaba ese retorno, alterando la secuencia estadística.

**Reproducción heredada.** MIN_POSITIVE → MAX → MIN_POSITIVE produjo cero retornos aceptados; el contrato esperaba dos. Es un caso de frontera numérica, no un incidente acreditado del exchange.

**Reparación final.** Para precios próximos, el cálculo usa log1p de la variación relativa; para cociente positivo finito alejado de uno, usa ln del cociente; ante underflow/overflow del cociente, usa diferencia de logaritmos. El intervalo de razón [1/2,2] es un criterio numérico de proximidad favorable a la sustracción flotante, no un umbral de trading o régimen.

**Autoverificación de esta intervención.** La primera versión usaba log1p también para variaciones próximas a −1. La sustracción perdía precisión: en 1 → 10⁻¹⁵ obtuvo −34,53957599234088 en lugar de −34,538776394910684. Se añadió contract_large_price_decline_preserves_log_return_accuracy, se observó fallar y se corrigió la selección de fórmula. Ese fallo fue de la versión intermedia, no se atribuye al código heredado.

**Límite.** Estabilidad numérica no valida el dato económico. Un salto extremo puede requerir cuarentena por procedencia, pero no debe desaparecer silenciosamente por una identidad mal evaluada.

**Código y pruebas.** [update](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hurst_dfa.rs:131>), [retornos extremos](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hurst_dfa.rs:551>) y [precisión de caídas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/hurst_dfa.rs:632>).

### FMT-120 · P1 · El clipping ocultaba la incompatibilidad entre estimación y modelo

**Mecanismo.** Toda pendiente finita se recortaba a [0,05,0,95] y se publicaba is_valid=true. Una pendiente aproximadamente dos, causada por retornos con tendencia determinista, quedaba convertida en un Hurst de 0,95 aparentemente admisible. Un R² alto no detecta ese cambio de significado.

**Reproducción.** El perfil de rᵢ=i es cuadrático; DFA1 sólo elimina su parte lineal local. La regresión de esa serie fue aceptada antes y el nuevo test falló. No se requiere aleatoriedad para reproducir la contradicción.

**Reparación.** raw_exponent conserva la pendiente finita sin recortar, incluso fuera del dominio adoptado. is_valid exige 0<pendiente<1 para interpretarla bajo el modelo de retornos estacionarios con escalamiento usado por esta API. La salida histórica mantiene su acotación sólo dentro del dominio admisible. Fuera de él, el consumidor recibe el fallback 0,5 y confidence=0; la pendiente cruda sigue disponible en el estimador para diagnóstico.

**Distinción necesaria.** Rechazar esa interpretación no demuestra que la serie sea no estacionaria. Un AR(1) estacionario de memoria corta puede mostrar pendiente aparente >1 en un rango finito por crossover. Precisamente por eso no se debe convertir automáticamente la pendiente en evidencia de memoria larga. Tampoco 0,5 de fallback demuestra un proceso browniano.

**Prueba del consumidor.** La nueva suite de RecursiveHurst alimenta precios exp(10⁻⁶t²), cuyo retorno cambia determinísticamente, y exige salida neutral sin confianza. También comprueba precios constantes. No se ejecuta un motor ni se construye una cuenta de trading.

**Pendiente.** Propagar el diagnóstico completo a telemetría y contratos de decisión, calibrar soporte y estudiar sensibilidad económica del fallback. No se mide aquí qué fracción de eventos reales cambiaría su clasificación.

### FMT-121 · P2 · El generador de pruebas tenía drift negativo estructural

**Evidencia.** El generador desplazaba un u64 33 bits: dejaba 31 bits. Luego dividía por u32::MAX, con lo que su supuesta uniforme centrada estaba aproximadamente en [−0,5,0), no [−0,5,0,5). Sumar doce de esas variables no da una innovación centrada: su media teórica aproximada era −3 y su varianza 1/4.

**Impacto sobre aseguramiento.** Las pruebas de paseo aleatorio y AR(1) incorporaban drift fuerte no declarado. Que pasaran sus umbrales no acreditaba el experimento descrito. El fallo no era del RNG de producción: pertenece al generador local de los tests de Hurst.

**Reproducción y arreglo.** En 20.000 innovaciones no apareció ninguna positiva. Se cambia a 53 bits divididos por 2⁵³ antes de centrar. El test exige ambos signos, media cercana a cero y varianza próxima a uno para la suma de doce uniformes. Esa distribución sigue siendo Irwin–Hall centrada, una aproximación, no una normal exacta.

**Efecto descubierto.** Dos expectativas antiguas dejaron de pasar al corregir las innovaciones y distinguir pendiente cruda de Hurst válido. Se revisaron para comprobar separación de pendientes finitas y admisibilidad del modelo, manteniendo sus diferencias numéricas exigidas. No se ensancharon indiscriminadamente tolerancias para esconder un fallo. La prueba de estabilidad por longitud también usa la pendiente cruda para no pasar trivialmente comparando dos fallbacks 0,5.

**Cierre local.** El generador y su contrato están probados. Sigue pendiente una campaña de calibración con múltiples semillas, procesos de memoria larga conocidos, ARMA, cambios de régimen y microestructura, separada de los datos usados para ajustar el algoritmo.

### FMT-122 · P1 · Una tupla conectaba crecimiento patrimonial con una ecuación de Sharpe live

**Evidencia de raíz a consumidor.** El evolver define velocity=final_cap/initial_capital, construye la tupla (índice, fitness, PnL, trades, Sharpe, velocity) y la pasaba a update. Este último interpretaba su sexto campo como live_sharpe y aplicaba reality_gap_adversarial_score. El mismo número se usaba después legítimamente como crecimiento para is_high_velocity.

**Fallo lógico.** Dos magnitudes adimensionales no son intercambiables: el cociente de riqueza no es media de retornos dividida por desviación, ni constituye observación live del candidato. Ese campo alteraba el ranking mediante una comparación nunca medida. Además el contador de trades era el del replay, no una muestra live independiente.

**Corrección integrada.** Se añade update_backtest_only y se cambia la llamada real del evolver. La metadata de crecimiento se preserva intacta, pero no participa en una penalización de brecha live inexistente. La API histórica update permanece para callers que realmente dispongan de esa referencia; se explicitan sus limitaciones de conteo y comparabilidad.

**Verificación.** El test varía únicamente la metadata entre 0, 1, 100 y NaN y exige exactamente el mismo fitness/ranking y la conservación de sus bits. NaN aquí prueba que un campo explícitamente opaco no se usa en el optimizador; no autoriza un estado financiero inválido ni sustituye la validación que deba hacer el caller.

**Lo que no se hizo.** No se añadió un posterior bayesiano, no se validó el candidato en vivo ni se cambiaron los gates existentes de promoción. Quitar una comparación mal tipada no demuestra ausencia de overfitting. El motor mantiene otros controles; la validación independiente de candidatos sigue pendiente.

**Código.** [CMA: ruta explícita](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/cma_es.rs:227>) y [caller reparado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/lib.rs:532>).

### FMT-123 · P1 · La recuperación numérica podía seleccionar una evaluación inválida

**Evidencia reproducida.** Un fitness NaN se sustituía por −10⁹. Ante otro candidato finito con fitness −10¹², el inválido quedaba primero. La regresión observó ganador con índice 1 cuando el candidato con evidencia válida era el índice 0.

**Reparación.** Se usa el extremo finito mínimo como marca excluida de parentalidad, se validan fitness y PnL, y en la API live también las referencias comparadas. Se ordena con total_cmp. Si no existen al menos μ padres admisibles, no se modifican centroide, sigma, generación ni memorias personales/globales. El slice de resultados sí queda ordenado y marcado: no se promete inmutabilidad de ese argumento de salida.

**Verificación adicional.** Se prueban overflow de una penalización sobre −10³⁰⁸, referencia live NaN/infinita y población con sólo un padre válido para μ=2. El último caso exige que la memoria del optimizador permanezca intacta.

**Límites.** El extremo mínimo finito se reserva como marca y no sirve como puntuación económica utilizable. La API todavía usa tuplas y no devuelve un diagnóstico tipado de rechazo. No se certifican constructor de dimensión cero, shapes arbitrariamente corruptos, IDs duplicados o correspondencia histórica de personal_bests entre ventanas distintas. Son contratos pendientes de una siguiente intervención, no garantías implícitas de este arreglo.

## 5. Reparación de FMT-011 — penalizar sin premiar el signo negativo

Para fitness f que se maximiza y retención r en (0,1]:

~~~text
P(f,r) = f·r,  si f ≥ 0
P(f,r) = f/r,  si f < 0
~~~

Se obtiene P(f,r)≤f para ambos signos. Al disminuir r, la puntuación no aumenta. La elección usa el signo de la utilidad canónica, no el de otro PnL con definición y costes diferentes. No se cambia la unidad de fitness ni se afirma invariancia a trasladar arbitrariamente su cero.

El caso rojo produjo **−0,948517463551336 a partir de −10**, adelantando al candidato sin deterioro. Después del arreglo la penalización no lo mejora. Los tests cubren ambos signos del PnL auxiliar, fitness de ambos signos y magnitudes entre 10⁻¹⁰⁰ y 10³⁰⁰; treinta y cinco combinaciones de score/severidad comprueban monotonía e identidad cuando r=1.

El factor de retención sigue siendo una heurística histórica, con su piso y umbral de muestra; no se convierte en probabilidad posterior porque se corrija su aplicación. Se eliminó un cálculo de PnL auxiliar cuyo signo sólo escogía entre dos clamps equivalentes para el rango de retención realmente devuelto. El parámetro de fee se conserva por compatibilidad: el fitness recibido debe incluir costes desde su evaluación.

Esto repara un orden matemáticamente incoherente, no todo CMA. Siguen pendientes el blanqueamiento diagonal frente a covarianza completa, la composición con PSO, la sobrescritura de sigma por el supervisor y el protocolo científico de promoción. No se cambió esa política de supervisor sin definir su relación con la adaptación aprendida.

## 6. Cálculos, unidades y significado operacional

### 6.1 DFA1 implementado

1. rᵢ=ln(Pᵢ/Pᵢ₋₁): retorno adimensional por observación admitida.
2. A=max|rᵢ|: normalización numérica; no nueva observación ni presupuesto de volatilidad.
3. Yⱼ=Σᵢ≤ⱼ(rᵢ/A−media): perfil integrado normalizado.
4. Para cada s, ajuste lineal local Y≈a+bi y fluctuación cuadrática media residual F(s).
5. Regresión ln F(s)=intercepto+α ln s; α es la pendiente cruda.
6. R² describe esa regresión. Soporte y dominio de interpretación se publican por separado.

La política de output y la validez del modelo no son el mismo objeto. La finitud de α tampoco acredita identificación de H en un mercado no estacionario. Si A=0, no existe pendiente identificable; el sistema no debe transformar ausencia de movimiento en fuerte convicción.

### 6.2 Corrección de explicaciones heredadas

También se corrigieron afirmaciones del comentario histórico: E|r|/sqrt(E r²) es un descriptor marginal, no curtosis; la relación ln10/ln50 entre desviaciones exige que ambas ventanas compartan el mismo descriptor y no es constante para toda serie; un contador acotado por ventana no diverge con la edad del proceso. Estas precisiones no rehabilitan el proxy como Hurst: delimitan correctamente por qué no lo es.

Las referencias antiguas a scalping/swing permanecen sólo donde explican una historia o nombres heredados; esta ronda no introduce una partición operativa nueva ni declara erradicadas todas las etiquetas del repositorio. Migrar el proxy que alimenta ML requiere versionar sus features y reentrenar; sustituirlo unilateralmente sólo en producción rompería train/serve.

## 7. Contraste teórico y ampliaciones propuestas

### 7.1 Evidencia primaria consultada

[Consistency of detrended fluctuation analysis](https://arxiv.org/abs/1609.09331) relaciona DFA con autocovarianza/variograma y condiciones de escalamiento, y analiza sesgo de tamaño finito. Los pasajes consultados distinguen procesos estacionarios de entradas no estacionarias con incrementos estacionarios; también advierten que completar huecos por interpolación puede introducir correlaciones. No autorizan tratar cualquier pendiente alta como memoria larga.

[Effect of Trends on Detrended Fluctuation Analysis](https://arxiv.org/abs/physics/0103018), localizado por su resumen, estudia crossovers causados por tendencias. La expansión bibliográfica recuperó [Detecting Long-range Correlations with DFA](https://arxiv.org/abs/cond-mat/0102214) y [Comparison of detrending methods for fluctuation analysis](https://arxiv.org/abs/0804.4081). Se conservan como familia pertinente: órdenes de detrending, correcciones de pequeña escala y comparación con métodos independientes. No se implementaron aquí esos métodos ni se revisaron íntegramente sus cuerpos.

[The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) describe adaptación del paso mediante longitud de caminos evolutivos comparada con selección aleatoria. Se consultaron pasajes sobre ese mecanismo; no se afirma que el híbrido actual preserve todas las propiedades del algoritmo canónico.

La habilidad Firecrawl guio búsqueda, expansión y verificación de pasajes. La CLI no estaba disponible; se usó el índice conectado, sin instalar software ni enviar código privado. Su influencia material fue separar pendiente aparente, modelo estadístico, reloj y garantía; de ahí la publicación de diagnóstico crudo y la revisión de las expectativas de prueba.

### 7.2 T29 — Diagnóstico multiescala con soporte y falsación de modelos

Propuesta, no implementación completa: transportar por nodo una estimación junto con reloj, rango de escalas observable, muestra efectiva, error y versión. Contrastar una ley única con pendientes locales/crossovers y alternativas de memoria corta. La selección del rango no debe optimizarse sobre el mismo resultado que luego se presenta como prueba independiente.

La física útil aquí es la conexión entre correlación, estructura de fluctuaciones y escalamiento bajo un modelo especificado. No basta introducir una ecuación de difusión y llamar universal a su exponente. La familia DFA/wavelets/variograma permite diagnósticos complementarios; un resultado concordante tampoco sustituye validación fuera de muestra.

Experimento de aceptación: simulaciones retenidas con white noise, ARMA, ruido gaussiano fraccional, cambios de régimen, tendencias y huecos; comprobar error por escala, falsa detección, estabilidad al variar cadencia y latencia p50/p99. La política continua puede usar una discretización adaptativa para cálculo, pero las fronteras deben obedecer soporte/error, no nombres de estilos de trading.

### 7.3 T30 — Contratos tipados de evidencia y utilidad evolutiva

Propuesta de integración: reemplazar gradualmente tuplas semánticamente ambiguas por registros que distingan fitness neto, crecimiento de riqueza, Sharpe, población evaluada, rango temporal y referencia live opcional. Cada penalización debe declarar entrada, unidad, signo, monotonía y política de evidencia ausente.

La implementación de esta ronda aporta un paso concreto: una ruta de backtest que no inventa comparación live. Queda pendiente un registro de evaluación con identidad genómica, datos as-of, costes observados, número de consultas y motivo de promoción/rechazo. Añadir optimización más avanzada sin ese registro sólo optimizaría una función cuya interpretación sigue cambiando.

### 7.4 Cuántica, complejidad y alcance real

CMA/PSO sigue siendo cómputo clásico. No se añade una etiqueta cuántica a las reparaciones ni se reclama ventaja sobre un baseline clásico. Tampoco se necesita resolver un problema del milenio para demostrar homogeneidad, coherencia dimensional o monotonía de una penalización. Los catálogos T01–T28 permanecen; cualquier teoría nueva debe aportar una propiedad falsable y un coste medido.

## 8. Pruebas y aseguramiento

Comandos ejecutados:

~~~text
cargo test -p feature-engine --offline --lib
cargo test -p evolution-engine --offline --test cma_penalty_contract
cargo test -p evolution-engine --offline --lib cma_es::tests
cargo test -p god-engine-core --offline --test dfa_contract
cargo test -p god-engine-core --offline --lib hurst
cargo check -p trader-gemini-v5 --bin god_engine --offline
~~~

**Resultado final: 69 tests aprobados**, sin duplicar ejecuciones: 57 de features, seis nuevos de CMA, tres existentes de CMA, dos nuevos del consumidor y uno existente de muestreo por reloj. La última reejecución de las suites CMA/consumidor terminó correctamente sobre los fuentes finales. cargo check del host terminó con éxito. No es la suite completa del workspace ni una prueba de rentabilidad.

Regresiones heredadas rojo→verde: cuatro de DFA y dos de CMA. Regresión adicional de la intervención: una de precisión logarítmica. Las nueve pruebas nuevas restantes refuerzan soporte, offset, umbral de calidad, metadata, monotonía, padres válidos, referencias inválidas y consumo. No se cuentan recompilaciones ni ejecuciones repetidas como pruebas diferentes.

No se ejecutaron campaña de backtest, proceso de trading, evolver, testnet, promoción de genomas o despliegue. Hubo espera por el lock de compilación y un bloqueo transitorio de Windows al formatear un fuente; se verificó el archivo y se reintentó, sin matar procesos. Los avisos existentes latest_ts/mode y RealWfOutcome.trades permanecen; no se aplicó cargo fix global.

## 9. Cobertura y manifiesto

Se leyó completo HurstDfa, nuevo respecto de la cobertura anterior, y se releyó completo CMA. Se revisaron nuevamente contratos ya cubiertos y tramos dirigidos de evolución, math_kernels y stateful_engine. Se revisaron los nuevos tests. **Cobertura acumulada: 92 Rust preexistentes distintos de 289**, no todos los archivos del proyecto. Los archivos nuevos de prueba se cuentan aparte; búsquedas por texto y tramos parciales no aumentan ese total.

| Archivo propio | Líneas finales | SHA-256, prefijo |
|---|---:|---|
| crates/feature-engine/src/hurst_dfa.rs | 643 | 92733384B87B2EBD |
| crates/evolution-engine/src/cma_es.rs | 481 | 98ACE3FBC23FA9DA |
| crates/evolution-engine/src/lib.rs | 635 | FE66BE8328214F24 |
| crates/god-engine-core/src/math_kernels.rs | 1205 | 49A86F374DFAA958 |
| crates/evolution-engine/tests/cma_penalty_contract.rs | 101 | 4862D322C639383E |
| crates/god-engine-core/tests/dfa_contract.rs | 24 | 3A23EA5F739DC895 |

Los cuatro fuentes estaban limpios al empezar esta intervención; se comprobó su estado antes de editar. Hashes iniciales: HurstDfa 0B812C4DE4BD10E1; CMA 5A4604C0A2A90231; evolution/lib DBA12D6F3703FE84; math_kernels 2C53CCD6A0CC72C9. En math_kernels sólo se cambiaron comentarios; en evolution/lib se corrigió el caller y se explicó su significado, sin formatear todo el archivo.

La verificación final confirmó que los seis archivos propios conservan los hashes anteriores y que los tres archivos ajenos prioritarios mantienen los iniciales: host F2BA92C4283C66E6; replay 736CC1DBB38E21D7; risk/lib E9883455B6BEBEFA. Los prefijos completos anteriores de ATLAS, informe maestro y anexo VI conservaron su SHA-256 tras las adiciones, normalizando únicamente CRLF/LF. No se borró ni reescribió el historial de esos informes.

Se verificaron los diez enlaces locales del anexo —archivo existente y línea dentro del archivo— y los siete encabezados FMT nuevos, sin duplicados. rustfmt --check pasó para HurstDfa, CMA y los dos tests nuevos; git diff --check pasó para los cuatro fuentes y los dos índices versionados. No se formatearon globalmente math_kernels ni evolution/lib para evitar cambios ajenos al contrato intervenido. El inventario local sigue siendo 1.119 archivos versionados, 289 Rust y 24 manifiestos Cargo.

## 10. Prioridades restantes

1. Cerrar FMT-113 mediante cantidad realizable, reserva y payload coherentes, sobre una ventana coordinada con las sesiones que modifican host/replay.
2. Cerrar FMT-117: reloj y soporte completos hasta la decisión; no extrapolar como si se hubiera observado.
3. Validar económicamente los cambios de clasificación y selección con datos retenidos y costes; una corrección matemática no demuestra mejora de PnL.
4. FMT-012/013: revisar geometría de muestreo, adaptación de sigma y supervisor como un único protocolo explícito; conservar trazabilidad por generación.
5. FMT-003/050/051: esquema ML consistente, mundo de simulación independiente del candidato y validación no consumida por la búsqueda.
6. Continuar aprendizaje/contabilidad FMT-100/101/109–111 y la cobertura de archivos restante. Las reparaciones de esta ronda no borran esos pendientes.

**Conclusión:** el avance verificable consiste en que los cálculos conserven su significado al cruzar nodos. Esta ronda repara defectos numéricos y de selección, expone incertidumbres antes ocultas por clipping y hace explícita la ausencia de evidencia live. No convierte por declaración el sistema en universal, omnisciente o completamente autoevolutivo.

## Continuación aditiva — ronda VIII, 2026-09-24

El [anexo VIII](AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VIII_2026-09-24.md)
actualiza el pendiente FMT-013: ya no se sobrescribe sigma aprendido;
el supervisor aplica cambios relativos y se prueba persistencia durante
dos generaciones. FMT-124–126 reparan contratos estructurales de lote,
atracción PSO sin observaciones y amortiguación CSA mal transcrita.
FMT-127–129 describen comparabilidad de memorias, población efectiva y
escala generadora/normalizadora todavía pendientes.

Las seis pruebas de penalización de VII vuelven a pasar en VIII; sus
resultados no se suman dos veces a una misma suite. VIII verifica 25 tests
distintos, 16 nuevos, y cargo check del host. Los hashes de CMA/lib de
esta ronda VII son históricos; el nuevo manifiesto consta en VIII.

La relectura no aumenta cobertura: permanecen 92 Rust preexistentes
distintos. Se explicita que el host llama a OnlineEvolutionDaemon y no
se localizó un caller operativo del bucle CMA auxiliar. No se atribuye
a estas reparaciones un efecto demostrado en demo/producción.
