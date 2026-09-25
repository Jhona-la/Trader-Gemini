# Auditoría científica XVI — identidad multiactivo, evidencia conjunta y contratos de asignación

Fecha: 2026-09-24. Corte local: main, HEAD 59a76de4. Continuación aditiva de XV. El maestro, el atlas y la matriz histórica de 305 puntos se conservan; los identificadores FMT constituyen una serie complementaria y no se suman ciegamente a esa matriz.

## 1. Resultado ejecutivo, alcance y estado de resolución

Se modifican cuatro fuentes y se agregan cinco archivos de pruebas. Se corrigen doce contraejemplos reproducidos antes del cambio, agrupados en contratos de asignación, estado de cesta, identidad de instrumento y cotización. Se añaden cuatro identificadores: FMT-167/168/169/170. FMT-088/034/036 reciben reparaciones parciales; FMT-033/050 reciben nuevas comprobaciones, no cierres.

**48 pruebas distintas pasan; 22 son nuevas. Doce muestran rojo→verde y seis son caracterizaciones de deuda ABIERTA.** Las cuatro pruebas nuevas restantes amplían los contratos de error, equivalencia y transacción. Un test que reproduce con éxito una deficiencia no acredita que esa deficiencia esté resuelta. Los dos tests preexistentes del orquestador se ejecutan dentro de un harness de signal-engine y se cuentan una sola vez.

**Cobertura conservadora: 116/289 archivos Rust preexistentes leídos completos; 173 pendientes.** Solo maker.rs se incorpora como nueva lectura acreditada. StatArb ya figuraba en la ronda II; no se vuelve a sumar. Tampoco se suman los nuevos tests al denominador preexistente. El inventario base comprobado conserva 1.119 archivos versionados y 24 manifiestos Cargo. No se ha auditado de extremo a extremo cada uno de ellos, ni se certifica el sistema completo.

La compilación de god_engine pasa en modo check, sin ejecutar ni relanzar el motor. Permanecen tres advertencias de evolution-engine: latest_ts, mode y RealWfOutcome.trades sin uso. No se midió rentabilidad, impacto financiero real, latencia p99 ni robustez en todas las combinaciones de features del workspace.

| Registro | Prioridad / alcance | Resultado XVI |
| --- | --- | --- |
| FMT-088 | P2, asignador auxiliar | Capital inválido, exposición fabricada y normalización corregidos; factibilidad y optimización de cartera abiertas |
| FMT-167, nuevo | P2, API de cesta | Estado finito/transaccional y normalización L1 corregidos; modelo estadístico no certificado |
| FMT-034 | P2, API de cesta | El rechazo ya no incrementa muestras aceptadas; bloqueo tras desplazamiento persistente abierto |
| FMT-033 | P2, API de cesta | Confirmado que el timestamp no afecta al estimador; sin reparación del reloj OU |
| FMT-036 | P1 arquitectónico; reparación P2 del adaptador auxiliar | No se reutiliza el par por ticks ajenos ni por prefijos ambiguos; edad por pata y lead–lag siguen abiertos |
| FMT-168, nuevo | P1, biblioteca llamada desde núcleo | Suelo de precio eliminado en libros válidos; fallback de libro inválido aún puede devolver quote cruzada |
| FMT-169, nuevo | P2, StatArb auxiliar | Demostración y prueba de umbral inalcanzable para ventana dos; abierto |
| FMT-170, nuevo | P1, conexión genoma→decisión→ejecución | Quote calculada y descartada en process_event; contratos maker divergentes; abierto |
| FMT-050 | P1, evaluación evolutiva | Reconfirmado que el gen maker_spread_pct modifica el mercado sintético; abierto |

La prioridad distingue un fallo de biblioteca de su propagación demostrada. No se ha encontrado consumidor operativo del asignador epigenético ni de MultivariateCointegrationEngine en la búsqueda estática actual de crates/src. No equivale a probar ausencia de consumidores externos o dinámicos. Maker sí tiene un consumidor en el núcleo, pero su salida no atraviesa la ruta process_event hacia el host.

## 2. Paradigma de grafo vivo: diagnóstico desde la raíz al terminal

```text
Evento (instrumento, fuente, reloj, secuencia, calidad)
    ├─ Adaptador auxiliar BTCUSDT/ETHUSDT → caché por pata → StatArb → SignalIntent de A
    │     └─ XVI corrige identidad; faltan edades, hedge y transacción de ambas patas
    ├─ StatefulEngine por activo → ATR/precio → MakerEngine → MakerQuote
    │     └─ process_tick_dual → process_event descarta _maker → sin quote en retorno al host
    └─ Datos del evaluador + candidato.maker_spread_pct → libro sintético → fitness
          └─ el parámetro de acción también modifica el entorno de evaluación

Scores + metilación + capital → asignador auxiliar → fracciones
    └─ XVI corrige validez; no hay consumidor operativo localizado ni matriz de riesgo conjunta

Precios[4] + pesos[4] → spread de cesta → proxy OU por evento → intención escalar
    └─ XVI protege estado; no estima rango de cointegración ni reloj físico
```

Este grafo es un trazado manual de las rutas inspeccionadas, no una declaración de cobertura de todas las aristas del proyecto. Nodo raíz significa procedencia observable; nodo de decisión, transformación con contrato verificable; nodo terminal, efecto confirmado en órdenes/estado. Un símbolo o gen presente en un archivo no prueba que alcance el terminal.

La identidad de instrumento no es un régimen financiero arbitrario. Distinguir BTCUSDT de BTCUSDC o de un token apalancado es conservar unidades y significado. Eliminar una clasificación temporal rígida no permite mezclar instrumentos con numerarios y mecanismos distintos. El diseño universal pendiente debe usar claves estructuradas —venue, mercado, instrumento y versión—, no extender un starts_with.

## 3. FMT-088 — Asignación sin evidencia, capital ficticio y normalización no homogénea

**Evidencia:** [asignador y API fallible](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/epigenetic_capital_alloc.rs:8>), [contratos reproducibles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/epigenetic_allocation_contract.rs>).

### Mecanismo anterior y contraejemplos

Cada score se multiplicaba por 0,5+methylation. Después, todo activo seleccionado recibía al menos 0,01. Así, diez scores cero con capital 1.000 producían diez pesos de 0,1. Un vector de negativos o NaN podía acabar igualmente invertido. La ausencia de evidencia positiva se convertía en una orden implícita de diversificación.

Capital cero, negativo o no finito era reemplazado por 13. El helper no distinguía “la cuenta tiene 13 USD” de “la cuenta no fue observada válidamente”. Con scores uniformes y capital inválido asignaba 0,5 a dos activos. Tampoco la API dinámica conservaba identidad dimensional: usaba min(len(scores),len(methylation)) y truncaba silenciosamente el universo.

Con scores [1,0,−1], la salida anterior era aproximadamente [0,9868421; 0,006578947; 0,006578947]. El suelo otorgaba capital incluso a evidencia nula o negativa. Al multiplicar scores finitos cercanos a MAX por 1,5, se generaba infinito; la normalización podía producir NaN. Reescalar positivamente todos los scores modificaba además sus proporciones cuando cruzaban el suelo 0,01.

### Reparación y significado de los cálculos

Se introduce AllocationError para capital inválido, dimensiones diferentes, score no finito y metilación no finita, con índice cuando corresponde. Las APIs try_ permiten distinguir el error de un vector válido sin oportunidades positivas. Los wrappers antiguos devuelven ceros ante error y conservan la cardinalidad de scores, por lo que pierden el motivo de abstención: esa limitación queda explícita.

Para sᵢ finitos se calcula M=max(0,maxᵢ sᵢ). Si M=0, toda asignación es cero. En otro caso:

- uᵢ = max(sᵢ,0)/M · (0,5+clip(mᵢ,0,1))/1,5.
- Se aplica la selección K heredada.
- wᵢ = uᵢ / Σⱼ∈seleccionados uⱼ para los elegidos; cero para los demás.

Las divisiones comunes por M y 1,5 cancelan al normalizar. Mantienen u entre cero y uno y evitan formar primero un producto desbordado. Hay al menos un score máximo positivo y su multiplicador está entre 1/3 y 1, de modo que el denominador seleccionado es positivo. No se introduce un epsilon financiero. La invariancia de escala está limitada por la representación de los inputs y los cocientes en f64; no se promete exactitud real-aritmética para ratios que subdesbordan.

La implementación top10 utiliza arrays de tamaño fijo; el universo usa memoria O(n) y ordenación O(n log n). Se elimina la afirmación de “fracciones óptimas”: no se está resolviendo Kelly multivariante ni una optimización convexa con restricciones de liquidez.

### Deuda pendiente y criterio de cierre

Los escalones K=2/5/10/N permanecen como política de compatibilidad, no como modelo adaptativo. Scores [1,10⁻⁸] y capital 13 producen una segunda asignación positiva muy inferior a 5 USD: K≤2 no garantiza mínimos de orden. El test lo documenta, no valida el mínimo actual de ningún venue.

Se conserva explícitamente la API sin capital que asume 13 USD; no debe confundirse con sustitución de un capital inválido recibido. Se mantienen clipping de metilación, desempate por índice y tensor de plasticidad legacy. En un empate que corta K, permutar índices puede cambiar qué activo queda seleccionado.

Cerrar FMT-088 exige estimandos comparables por activo/escala, costes y tamaños de contrato observados, restricciones verificables, presupuesto de exposición, matriz conjunta válida y una ruta consumidora auditada. Conectar el helper por existir no demuestra conveniencia. La expresión “epigenético” describe aquí un multiplicador, no evidencia de adaptación biológica, aprendizaje causal o mejora económica.

## 4. FMT-167 y FMT-034 — Cesta: transacción numérica no equivale a validez OU

**Evidencia:** [actualización de cesta](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/multivariate_coint.rs:61>), [asignaciones por pata](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/multivariate_coint.rs:199>), [pruebas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/tests/basket_state_contract.rs>).

### Fallo de estado y de rango numérico

El spread se forma como S=Σwᵢ ln Pᵢ. Pesos finitos no garantizan productos o suma finitos. Con cuatro pesos MAX y precios 100, la primera actualización anterior incrementaba count e instalaba un spread no representable. En actualizaciones posteriores, productos de innovaciones podían desbordar después de que media y contador ya hubieran cambiado.

Tras veinte observaciones estables, un salto rechazado incrementaba count de 20 a 21. El siguiente dato aceptado heredaba una tasa de actualización calculada con una muestra que no había entrado en los momentos. No es solo una diferencia de telemetría: cambia el peso estadístico del futuro.

La normalización de patas usaba Σ|w| y un suelo 10⁻⁶. Con [ε,−ε,ε,−ε], ε=10⁻²⁰⁰, devolvía aproximadamente los pesos originales, no [0,25,−0,25,0,25,−0,25]. Con ε=10³⁰⁸, la suma desbordaba y colapsaba las asignaciones. Cambiar la magnitud común de un vector de cobertura alteraba indebidamente su exposición bruta normalizada.

### Reparación

Se validan los campos públicos relevantes y precios/pesos antes de actualizar; checked_add protege el contador. Se calculan contador, media, varianza, theta y z candidatos en variables locales y solo se confirman si los resultados necesarios son finitos. No hay commit parcial al rechazar una innovación no representable. El término “transacción” significa atomicidad lógica de la función con &mut self; no es una garantía de sincronización entre hilos o persistencia durable.

La asignación usa m=max|wᵢ| y aᵢ=signo·(wᵢ/m)/Σ|wⱼ/m|. Para vector cero, señal Flat o pesos públicos no finitos devuelve ceros. Para un vector finito no nulo conserva dirección y exposición bruta uno, salvo redondeo. No convierte esos pesos en cantidades negociables: faltan precios, lotes, divisas, multiplicadores y restricciones de patas.

Los tests cubren también desbordamiento en la segunda observación con pesos 10¹⁵⁵, un peso público NaN y count=usize::MAX. La abstención ante estado público ya corrupto no lo repara automáticamente. El constructor legacy todavía sustituye pesos inválidos por 0,25; no se presenta esa sustitución como semánticamente válida.

### FMT-033/034 continúan abiertos

El timestamp sigue sin participar en el ajuste. Dos secuencias de precios idénticas con intervalos separados por un factor de un millón producen exactamente iguales momentos y half_life. Theta sigue siendo una razón por evento, forzada positiva mediante clipping; ln(2)/theta no puede presentarse como duración física inferida.

La media usa 1/min(count,500) y la varianza un decaimiento distinto. No es el estimador insesgado de Welford, ni un ajuste de OU irregular por máxima verosimilitud. Se corrige el comentario para no dar esa certificación. Permanecen pisos de varianza, amplitud esperada fija 0,015, confianza recortada y half-life operativa limitada a 1–100 con filtro ≤50.

Después del historial estable, 180 observaciones de un nuevo nivel legítimo pueden seguir rechazadas contra el último spread aceptado. Ahora count permanece correctamente en 20, pero el bloqueo persiste. No se subió arbitrariamente el umbral ni se resetearon los momentos para esconderlo. Se necesita una política explícita de cuarentena y reidentificación, con evidencia para separar cambio estructural, split, error de feed y salto real; su recuperación debe probarse sin admitir indiscriminadamente datos corruptos.

## 5. FMT-036 — Identidad de instrumento y observaciones artificiales del par

**Evidencia:** [adaptador auxiliar](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/multi_asset_orchestrator.rs:36>), [harness de identidad](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/tests/multi_asset_identity_contract.rs>).

Anteriormente starts_with("btc") aceptaba BTCUSDC y BTCDOWNUSDT como si fueran el mismo instrumento BTCUSDT. starts_with("eth") podía mezclar ETHBTC con ETHUSDT. No era generalización multiactivo: sobreescribía una única celda de caché con unidades/activos diferentes.

Además, tras poblar BTC y ETH, cada tick de SOL u otro símbolo recorría la evaluación del par aunque ninguna pata hubiera cambiado. La prueba alimenta ambas patas una vez y 110 ticks de SOL; la versión previa acababa emitiendo la señal Flat con confidence=1 por una ventana llena de copias del mismo spread. Un reloj de otro instrumento fabricaba madurez del modelo.

El adaptador ahora admite únicamente sus dos instrumentos declarados, con comparación ASCII sin distinguir mayúsculas y sin crear un String por tick. Otros instrumentos retornan antes de tocar el par. Se rechazan libros bloqueados o cruzados y se evita la suma de extremos finitos al calcular el mid mediante bid/2+ask/2. No se transforma este adaptador en el motor universal; se explicita su alcance real.

Tres pruebas fallaron y ahora pasan. Los dos tests nominales anteriores también pasan. Quedan abiertas la frescura por pata, desorden de eventos, secuencias duplicadas, coherencia entre venues y coste/hedge de dos órdenes. Una nueva observación válida de BTC todavía puede combinarse con un ETH antiguo. La reparación no cierra el componente lead–lag de FMT-036 ni prueba sincronización física.

## 6. FMT-168 — Cotización maker: el suelo viola las unidades; el fallback sigue siendo peligroso

**Evidencia:** [MakerEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/maker.rs:24>), [pruebas de cotización](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/tests/maker_quote_contract.rs>).

Con un libro válido bid=10⁻¹², ask=2·10⁻¹², la regla max(10⁻⁸) elevaba la compra por encima del ask observado. Contradecía la promesa “nunca cruzar” aunque todos los inputs fueran finitos y positivos. Un epsilon de precio absoluto no es una propiedad universal de los activos.

Para la rama de libro válido, la compra final se limita al bid observado y la venta al ask; el propio libro proporciona positividad y orden. Si el precio óptimo calculado no es utilizable, se conserva el límite observado correspondiente. También se calcula el mid evitando bid+ask. El test del precio diminuto falla antes y pasa después.

**No se cierra la validez general del quote.** Un libro inválido bid=101, ask=100 entra en un fallback que devuelve esos dos precios positivos, todavía cruzados. MakerQuote contiene únicamente dos números y no porta estado de calidad. El test open_debt_invalid_book_fallback_can_be_crossed pasa precisamente porque el defecto permanece. La validación externa del adaptador auxiliar no protege automáticamente a todos los callers de la biblioteca.

Persisten otros contratos sin resolver: cantidades negativas no se rechazan aquí; la suma de cantidades finitas puede desbordar; parámetros no finitos se sustituyen; no hay tick size, vencimiento ni identidad de instrumento en la salida. El OFI se actualiza y su valor no se usa. El constructor conserva _base_spread_bps sin consumo efectivo: dos motores con 1 y 100 generan exactamente el mismo quote con idénticos argumentos. No se elimina el parámetro sin una migración de API.

La fórmula aplicada es half_spread=mid·(spread_pct+volatility·poly_b). La suma representa un **semispread**, no el spread completo; el comentario nuevo lo explicita. El inventario se divide por 100 USD y se recorta a ±5; OBI activa un sesgo por umbral. Estos mecanismos no constituyen por sí mismos un modelo de probabilidad de fill, toxicidad o utilidad de inventario.

Cierre requerido: API fallible de cotización, consumidores que propaguen abstención, contrato de unidades y costes, rounding por instrumento que preserve post-only, y replay de confirmación/rechazo/cancelación. No se habilita una ruta ejecutora para “hacer visible” el efecto sin esas garantías.

## 7. FMT-169 — Una ventana corta puede hacer imposible el umbral de StatArb

**Evidencia:** [cálculo de StatArb](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/stat_arb.rs:38>), [caracterización](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/tests/pair_model_characterization.rs>).

La observación actual entra en la ventana antes de calcular su media y desviación poblacional. El z resultante es internamente estandarizado, no una innovación predictiva respecto de parámetros anteriores.

Sean dᵢ=xᵢ−media, Σdᵢ=0 y s²=Σdᵢ²/n. Por Cauchy–Schwarz, Σⱼ≠ᵢdⱼ² ≥ dᵢ²/(n−1), luego zᵢ²=dᵢ²/s²≤n−1. Con n=2, |z|≤1 para una ventana no constante en aritmética exacta. El constructor permite n=2 y el fallback de umbral es 1,5: no puede superarse con ese estadístico. Los pisos y el redondeo no deben utilizarse como mecanismos para producir señales fuera de ese límite.

La prueba recorre shocks de ambos signos y escalas extremas con n=2 y umbral 1,5: todos retornan Flat. Es una configuración permitida que silencia la estrategia por diseño, no por falta de divergencia económica. No se afirma que n=100 con umbral 2 sea imposible.

También se observa un spread ln A−ln B con beta fijo uno, una barrera constante 0,002 en log-spread y una intención solo para A. Ni correlación ni ese residuo acreditan cointegración; el comentario de que B “debe operar al contrario” no es una transacción de dos patas. SignalIntent no contiene el conjunto de instrumentos/pesos que permitiría comprobar esa propiedad.

Cierre: separar señal predictiva y diagnóstico descriptivo; validar configuraciones contra la distribución del estimador; contrastar un residuo de modelo entrenado con datos anteriores y comprobar cobertura fuera de muestra; expresar hedge, tamaños, fees y restricciones de las dos patas. Reducir el umbral hasta que opere solo cambia la tasa de activación. No resuelve especificación, costes ni calibración. No se cambia esa estrategia en XVI sin evidencia que elija el nuevo estimando.

## 8. FMT-170 y FMT-050 — Genoma con efecto distinto en evaluación y ruta operativa

**Evidencia conectada:** [núcleo descarta _maker](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:825>), [núcleo construye quote](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:4785>), [host consume process_event](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3216>), [mercado sintético dependiente del candidato](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:125>).

La ruta inspeccionada del núcleo calcula el quote si last_ws_latency_ms≤25. process_tick_dual lo devuelve en una tupla; process_event lo desestructura como _maker y solo retorna new_order/closed_order. El host invoca process_event y no recibe ese quote. Existe cómputo, pero no está demostrada la arista terminal para su ejecución. No se confunde la heurística maker con otras entradas limit/maker que puedan existir en el motor.

En el evaluador wf_evaluate_real, candidate.maker_spread_pct determina el half-spread del libro sintético que se entrega al núcleo. Esto reconfirma FMT-050: el candidato altera tanto un parámetro de comportamiento como el entorno que enfrenta. Su sensibilidad en fitness puede deberse a ese mundo modificado, aunque su quote sea descartado por la ruta observada. **Es una explicación causal plausible del desacople backtest/demo, no una cuantificación de su contribución al PnL real.**

Los dos callers maker inspeccionados tampoco entregan lo mismo. El núcleo usa get_features()[3], que StatefulEngine construye como v_t/last_price, una medida relativa de rango/ATR; el adaptador auxiliar usa (ask−bid)/bid, un spread relativo. Ambos se llaman volatility, pero no son estimandos intercambiables. El auxiliar además lee dynamic_ofi_threshold donde el núcleo usa maker_obi_threshold. OFI y OBI no se convierten en el mismo observable por compartir tipo f64.

Se preservan los consumidores en esta ronda. Sus hashes se comparan antes/después. Cierre: instrumentar cada gen con productor, unidad, consumidor decisorio y efecto terminal; comparar candidatos en un tape exógeno idéntico; separar sensibilidad directa de efectos por selección/entorno; probar paridad por instrumento y versión; medir coste del quote descartado antes de eliminarlo o conectarlo. Las políticas de despliegue no deben promover por un fitness cuyo significado cambió entre entornos.

## 9. T39 — Evidencia multiactivo asíncrona: teoría candidata con condiciones

Firecrawl Research Index se utilizó para contrastar fuentes primarias. Se le enviaron preguntas científicas, no código, secretos ni datos de cuentas. El CLI local no estaba disponible y se usó el conector de investigación. Se leyeron pasajes del cuerpo de dos trabajos; otros resultados se registran solo como abstracts, no como verificación completa.

El trabajo [Streaming Approach to Quadratic Covariation Estimation Using Financial Ultra-High-Frequency Data](https://arxiv.org/abs/2003.13062) formula estimadores de covariación como formas cuadráticas y estudia cómputo streaming con ancho de banda fijo. Ese ahorro de memoria conlleva rendimiento estadístico subóptimo frente a elecciones que crecen con la muestra. Trata sincronización por refresh times; copiar el último precio en cada evento no es equivalente. Discute sesgo por asincronía y diferencias entre precisión y garantía semidefinida positiva. Esto respalda evaluar costes y propiedades de la matriz, no insertar una correlación de snapshots repetidos y declararla universal.

[Koike, Estimation of integrated covariances in the simultaneous presence of nonsynchronicity, microstructure noise and jumps](https://arxiv.org/abs/1302.5202) propone pre-averaging y truncamiento sobre Hayashi–Yoshida para estimar la covarianza integrada de las partes continuas, separándola de co-saltos bajo hipótesis explícitas. No estima beneficios futuros ni demuestra cointegración. Su validez asintótica no elimina la necesidad de estudiar error finito, ruido y mecanismo de muestreo en nuestros datos.

### Propuesta de integración; no implementada ni calibrada en XVI

1. Definir el estimando: covariación realizada, covarianza predictiva, dependencia de colas, lead–lag o residuo de cointegración. No publicar uno con el nombre de otro.
2. Conservar observaciones originales con tiempos de evento/recepción, IDs de instrumento/venue y calidad. No contabilizar copias de una caché como nuevas muestras independientes.
3. Construir baselines de sincronización y ruido con tapes reproducibles; comparar métodos asíncronos/prepromediados sobre la misma evidencia.
4. Emitir matriz, intervalo efectivo de observación, error/incertidumbre y estado de disponibilidad. Un par sin soporte no vale correlación cero.
5. Verificar simetría, diagonales, rango y semidefinición positiva antes de usar una matriz como varianza de cartera. Si se regulariza o proyecta, registrar el cambio y su impacto; no ocultar una matriz incoherente tras un clamp elemento a elemento.
6. Medir memoria, operaciones por evento, colas y latencia bajo el universo real antes de añadir esta ruta al hot path. Una matriz d×d tiene O(d²) elementos; “streaming” no convierte automáticamente toda la cartera en O(1).
7. Someter asignación y riesgo a replay conjunto causal con costes y clocks preservados; promover solo por mejoras fuera de muestra definidas de antemano.

Los tres abstracts adicionales localizados —[pre-averaging de matriz ex-post](https://arxiv.org/abs/2602.19645), [covariación multivariante con ruido y asincronía](https://arxiv.org/abs/2602.19658), [muestreo endógeno](https://arxiv.org/abs/1507.01033)— son una cola de lectura, no dependencia implementada ni evidencia suficiente para escoger hiperparámetros.

## 10. Universo temporal-espectral y límites de las analogías avanzadas

El objeto de diseño no es “un motor rápido y otro lento”, ni una etiqueta única de volatilidad. Es un campo de evidencia por instrumento, escala temporal y tiempo de observación, acompañado por dependencias entre instrumentos. Una representación propuesta es E(a,b,t,τ; fuente,versión), cuyo valor y calidad dependen del estimando. Una matriz de covarianza a τ y una función de respuesta de precios no comparten unidades ni reglas de actualización aunque ocupen el mismo grafo.

La representación puede admitir consultas de 1 ns a 100 años sin inventar información. Cien años de 365,25 días contienen aproximadamente 3,15576·10¹⁸ nanosegundos. Guardar o recorrer todos esos puntos no es requisito para representar una función continua. Se puede aproximar en coordenadas log(τ) y actualizar por eventos, pero deben acotarse error de interpolación, resolución observable, ventanas efectivas y memoria. Una función continua tampoco convierte eventos repetidos en datos nuevos.

Los extremos sin soporte deben exponerse como extrapolación/prior/no identificable. “Continuous” en un enum, o devolver duración cero por Default, no demuestra un análisis espectral universal. Tampoco todo umbral es un sesgo eliminable: límites de rango numérico, solvencia y contrato de exchange pueden ser restricciones explícitas; los cortes de conveniencia requieren justificación/calibración y trazabilidad.

La traslación de física o teoría avanzada exige correspondencia verificable entre estados, observables, ecuación, unidades, condiciones de contorno, identificabilidad, estabilidad y coste. En XVI no se deriva una PDE de liquidez ni una ventaja cuántica; no se agregan ecuaciones de problemas del milenio por prestigio. Los conceptos de energía, entropía, tensor o epigenética solo aportan si su operación y objetivo son medibles. Un modelo complejo con raíz observacional incoherente conserva ese defecto.

Autoevolución requiere cerrar un ciclo trazable: evidencia causal → hipótesis/versiones → evaluación independiente → decisión de promoción → efecto operativo observado → resultados atribuidos. Mutar parámetros, aceptar todos los scores o dibujar más nodos no cierra el ciclo. La incertidumbre es parte del estado; no existe una certificación de omnisciencia o de duplicación periódica de capital en estos tests.

## 11. Matriz por los ocho módulos del maestro

| Módulo | Trabajo XVI | Frontera todavía pendiente |
| --- | --- | --- |
| 1. Ingestión, parsers, L2 y normalización | Identidad y rechazo de libro cruzado en adaptador; estudio de asincronía | No auditoría nueva de todos los parsers; timestamps, secuencias y frescura por pata |
| 2. IA, modelos y señales | Separación de confianza, z descriptivo y predicción; FMT-169 | Calibración conjunta y errores de etiquetas históricos, incluido FMT-027 |
| 3. Multiactivo, estado y horizontes | Trazado par/cesta, escala de pesos, campo de evidencia propuesto | Modelo temporal físico y cartera universal no implementados |
| 4. Ejecución HFT y conectividad | Quote válido sin suelo absoluto; arista descartada localizada | API fallible, fill model, TTL, post-only y confirmación de patas |
| 5. Riesgo, Kelly y genomas | Asignación con abstención/error; explicación del desacople genético | Covarianza válida, costes, mínimos por instrumento, promoción causal |
| 6. Estado, memoria, telemetría y SO | Actualización lógica transaccional de cesta; hashes protegidos | No certificación de atomics/mmap globales ni benchmarks p99 |
| 7. Orquestación y confluencia | Grafo manual y rutas auxiliares versus operativas | VECM/registry y grafos sintácticos previos no rehabilitados en XVI |
| 8. Backtesting y gobernanza | 48 tests, 12 rojo→verde; FMT-050 reconfirmado | Backtest multiactivo exógeno, OOS independiente y revisión archivo por archivo completa |

No se recategoriza el estado global de los 305 puntos a partir de cuatro archivos modificados. Las filas indican lo que esta ronda aporta y lo que no revisa.

## 12. Verificación reproducible y mapa de archivos

| Ejecución offline | Pruebas distintas | Lectura del resultado |
| --- | ---: | --- |
| strategy-core --lib | 21 | Suite unitaria del crate, no toda la suite workspace |
| basket_state_contract | 7 | Tres rojo→verde, dos refuerzos y dos limitaciones abiertas |
| maker_quote_contract | 3 | Un rojo→verde y dos limitaciones abiertas |
| pair_model_characterization | 1 | Limitación abierta de ventana/umbral |
| risk-engine --lib epigenetic_capital_alloc | 3 | Dos nominales y contrato actualizado de capital inválido |
| epigenetic_allocation_contract | 8 | Cinco rojo→verde, dos refuerzos y un límite abierto |
| multi_asset_identity_contract | 5 | Tres rojo→verde y dos tests anteriores incluidos por path |
| Total sin reejecuciones duplicadas | 48 | 22 nuevos; 12 rojo→verde; seis diagnósticos abiertos |

Comandos exactos:

```powershell
cargo test -p strategy-core --lib --test basket_state_contract --test maker_quote_contract --test pair_model_characterization --offline -- --test-threads=1
cargo test -p risk-engine epigenetic_capital_alloc --lib --offline -- --test-threads=1
cargo test -p risk-engine --test epigenetic_allocation_contract --offline -- --test-threads=1
cargo test -p signal-engine --test multi_asset_identity_contract --offline -- --test-threads=1
cargo check --bin god_engine --offline
```

Se ejecuta rustfmt --check con skip_children=true sobre los nueve archivos propios, para no reformatear dependencias incluidas por path. git diff --check se limita a las cuatro fuentes modificadas.

Incidencias de verificación: el primer harness maker usó un import inexistente en el reexport raíz; se corrigió a strategy_core::maker::MakerEngine antes de reproducir el fallo conductual. Ese error de compilación no se cuenta como un bug del sistema ni como rojo→verde. Cargo tuvo una espera transitoria por lock del directorio de build; no se mató ningún proceso. En las ejecuciones rojas del harness de arena apareció el aviso de TG_GENOME_ENV no definido; no se alteró el entorno para habilitar producción, ni se escribieron genomas.

La nueva lectura completa acreditada es [maker.rs](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/strategy-core/src/maker.rs>). Se releen, sin sumar cobertura, epigenetic_capital_alloc, multivariate_coint, stat_arb, vecm_arbitrage, graph-4d/lib, correlation_guard y multi_asset_orchestrator, además de interfaces pequeñas. Los fragmentos de core/lib, stateful_engine, online_daemon y host no se convierten en lecturas completas.

El [artefacto estructurado XVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVI_2026-09-24.json>) contiene hallazgos, mecanismos, reparaciones, límites, tests, bibliografía y hashes. Conserva la distinción entre demostrado por test, trazado estático y propuesta pendiente.

## 13. Hoja de ruta de cierre, sin sustituciones cosméticas

1. **Raíz:** contrato de instrumento, procedencia, relojes y calidad; replay determinista conjunto. Probar duplicados, desorden, silencios y cambios de identidad antes de estudiar alpha.
2. **Modelo:** validar configuraciones imposibles; separar residuo predictivo y z descriptivo; ajustar dependencia multiactivo y tiempo físico con incertidumbre. Contrastar contra baselines simples.
3. **Estado:** cuarentena/reidentificación de cesta y APIs fallibles en los puntos que aún fabrican fallback. Exigir que rechazo no madure evidencia y que recuperación no admita datos corruptos.
4. **Decisión:** asignación neta de costes y restricciones sobre una matriz de riesgo defendible. Conservar abstención y soporte; no reemplazar K por una curva suave sin función objetivo.
5. **Terminal:** reconstruir el contrato de MakerQuote y de intención multípata antes de conectarlos a ejecución. Trazar y medir outputs descartados, confirmaciones parciales e inventario real.
6. **Evolución:** separar gen de acción y parámetros exógenos del replay; auditar causalidad y sensibilidad hasta el fill. Evaluar candidatos sobre el mismo mundo, con atribución y prueba independiente.
7. **Gobernanza:** continuar los 173 Rust preexistentes pendientes y los demás archivos del inventario. Mantener el informe acumulativo; no usar cobertura parcial ni cantidad de ecuaciones como certificación.

No hubo commit, push, fetch, merge, operación de cuenta, despliegue ni reinicio. Main y HEAD son estados locales; el remoto no se consultó. No se asegura que otra rama o persona haya resuelto lo pendiente. El repositorio conserva cambios ajenos/concurrentes. La verificación de hashes documenta los archivos protegidos concretos, no demuestra que todo el árbol permaneció inmóvil.

### Control final de integridad

El JSON se parsea con nueve registros de hallazgo y cuatro IDs nuevos. Coinciden los veinte hashes del inventario; los cinco consumidores/configuraciones protegidos conservan sus hashes iniciales. Los diecisiete enlaces locales del informe resuelven y sus líneas están dentro del archivo correspondiente. Tras normalizar CRLF a LF, los prefijos completos del atlas, maestro y XV conservan exactamente su SHA-256: se añadieron 2.247, 5.984 y 1.447 caracteres, respectivamente. El carácter NUL histórico del maestro no se limpia ni se reescribe su contenido. La verificación de formato/diff se aplica a los archivos propios; el estado ajeno no se normaliza ni restaura.

## Continuación aditiva XVII — universo e identidad

La [ronda XVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVII_2026-09-24.md>) y su [artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVII_2026-09-24.json>) añaden FMT-171 a FMT-177 sin sustituir XVI. Ocho nuevas lecturas completas elevan cobertura a 124/289 Rust, con 165 pendientes.

Se reparan contratos de capital/dimensiones/finitud, prioridad forzada, duplicados, mínimos de nocional/leverage y bitmap. DynamicSelector deja de inventar diez símbolos ante feed inválido. Permanecen scores duales, objetivos de ranking no identificados, IDs mutables frente a caché inicial, validador de lotes incorrecto, políticas divergentes y publicación sin epoch.

T40 contrasta disponibilidad, feedback y coste de transición con fuentes primarias; no es un algoritmo incorporado ni garantía económica. 35 tests distintos pasan, 23 nuevos; doce rojo→verde y cinco diagnósticos abiertos. Dos fuentes modificadas, tres archivos de pruebas, check sin ejecutar motor. Sin despliegue ni publicación Git.
