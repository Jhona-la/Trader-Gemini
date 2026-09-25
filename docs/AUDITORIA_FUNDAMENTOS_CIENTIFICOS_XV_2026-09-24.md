# Auditoría científica XV — estado continuo, volatilidad espectral y validez de los estimadores

Fecha: 2026-09-24. Corte local main, HEAD 59a76de4. Continuación aditiva de XIV. No se sustituye el informe maestro ni la matriz histórica de 305 puntos.

## 1. Resultado ejecutivo y límites

Esta ronda interviene tres fuentes de feature-engine y agrega cinco archivos de pruebas. Dos nuevos identificadores, FMT-165/166, complementan reparaciones o nuevas evidencias sobre FMT-002/004/046/091/140. No se cuentan otra vez los mecanismos ya documentados como si fueran descubrimientos independientes.

**93 tests distintos aprobados, 25 nuevos, siete reproducciones rojo→verde.** Cinco de los nuevos tests son diagnósticos que pasan porque permanece una limitación; no acreditan reparación. La comprobación de compilación del binario pasa sin ejecutarlo. No se evaluó rentabilidad, latencia p99, ni toda la suite del workspace.

**Cobertura acumulada conservadora: 115/289 archivos Rust preexistentes leídos completos; 174 pendientes.** La nueva lectura acreditada es src/features/correlation.rs. Se releen completos Kalman, FFT, correlación del crate, multifractal, régimen, capital_regime, EWMA y varias interfaces ya revisadas. No se suman nuevamente ni se contabilizan búsquedas o fragmentos de los grandes consumidores como lecturas completas. El inventario base sigue siendo 1.119 archivos y 24 manifiestos Cargo; esta ronda no certifica todos esos archivos.

La afirmación central del usuario tiene un correlato preciso en el código: la taxonomía global BTC continúa imponiendo un veto discreto aunque otra API diga Continuous. Además, dos genes de régimen recorren almacenamiento/mutación pero no tienen lecturas decisorias localizadas. Ello no prueba que toda discretización sea incorrecta: representación estadística, cuadratura numérica y restricciones de seguridad cumplen funciones distintas.

| ID | Prioridad y alcance | Estado XV |
| --- | --- | --- |
| FMT-165, nuevo | P1 de biblioteca conectada: estado/covarianza Kalman | Cinco fallos numéricos reproducidos y corregidos; API fallible agregada |
| FMT-166, nuevo | P1 de arquitectura evolutiva: genes sin consumidor decisorio localizado | Abierto; no se altera esquema genético |
| FMT-002 | P1: parametrización de Kalman en el consumidor | Sin piso absoluto en el helper; unidades de R y reloj Q siguen abiertos |
| FMT-004 | P1 de feature consumida | API espectral V2 verificada; ruta ML legacy permanece abierta |
| FMT-046 | P2 de API auxiliar de correlación | Validación/transacción local corregidas; estimando, sincronización y matriz multiactivo pendientes |
| FMT-091 | P1: veto global BTC | Tres diagnósticos de la API alternativa; política central sin modificar |
| FMT-140 | P2: bibliotecas duplicadas | Ampliado con correlación root; reproducción de divergencia, sin migración de esa ruta |

## 2. Grafo vivo: de la observación a la restricción

| Nodo raíz / evidencia | Transformación | Nodo decisorio o consumidor | Pérdida de significado pendiente |
| --- | --- | --- | --- |
| Eventos de precio | EMAs de BTC y Hurst | Cuatro códigos globales y veto long | Sin posterior, antigüedad ni sensibilidad al instrumento/escala de la orden |
| Precios de un instrumento | Kalman escalar | fair_price en StatefulEngine | R proporcional a precio en vez de unidad al cuadrado; Q por evento |
| Retornos por tick | FFT legacy de 64 eventos | Features espectrales ML | Bin de eventos confundible con frecuencia física; DC y pisos |
| Misma ventana | Nueva FFT V2, opt-in | API de diagnóstico/prueba | Aún sin consumidor ML, timestamps ni incertidumbre estadística |
| Corte de precios conjunto | Correlación EWMA con la propia cesta | API auxiliar, no caller productivo localizado | No matriz de pares, beta, PSD multiactivo ni reloj sincronizado |
| Genoma/configuración | Copias, mutación y vectorización | No lectura operativa localizada de dos genes | Dimensión genética no equivale a influencia en la política |

Una flecha transportando números finitos no prueba identidad semántica. El nodo terminal debe recibir una decisión y su evidencia, no una etiqueta convertida tácitamente en verdad global. Ninguno de los arreglos siguientes convierte el sistema en omnisciente.

## 3. FMT-165 — el Kalman escalar violaba su propio dominio numérico

**Evidencia:** [implementación y APIs fallibles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/kalman.rs:21>), [regresiones](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/kalman_numeric_contract.rs:1>). El [consumidor StatefulEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:467>) sí utiliza el filtro.

Para observación directa y paseo aleatorio, las ecuaciones usadas son:

- P⁻ = P + Q: incertidumbre predicha.
- K = P⁻/(P⁻+R): fracción de información de la observación en la actualización.
- x⁺ = (1−K)x + Kz: posterior escalar.
- P⁺ = P⁻R/(P⁻+R): varianza posterior bajo esas hipótesis.

P, Q y R son varianzas no negativas, con las unidades de x al cuadrado. Estas ecuaciones no son cuánticas, ni identifican por sí mismas microtendencia, horizonte óptimo o capacidad predictiva.

### Reproducciones previas y mecanismo

1. **Suma del denominador desbordada:** P=R=10^308, Q=0, x=0, z=10. Antes P+R era infinito, K se volvía cero y x permanecía en cero, aunque el resultado matemático es x⁺=5 y P⁺=5·10^307.
2. **Innovación desbordada:** x=−10^308, z=10^308, P=R=1. Restar z−x producía infinito; con K=0,5 se publicaba infinito en lugar de cero.
3. **Piso dimensional:** P=R=10^-30, Q=0. Elevar P a 10^-12 cambiaba una ganancia esperada de 0,5 por una cercana a uno. No es una regularización invariante a unidades; domina al modelo cuando las unidades son pequeñas.
4. **Varianza pública negativa:** R=−0,5 con P=1 generaba K=2; observar 20 desde x=10 producía 30. Un filtro de observación directa válido no debe extrapolar por introducir ruido negativo.
5. **Mutación parcial con observación rechazada:** update_with_dynamic_r(NaN,100) no cambiaba x, pero sí R desde 1 hasta 100. La siguiente observación heredaba una adaptación asociada a un evento inválido.

Las cinco pruebas fallaron antes del cambio y pasan ahora. No son simulaciones de PnL ni una demostración de cuánto afectaron cuentas reales.

### Reparación y compatibilidad

try_new valida x finito y varianzas finitas no negativas. new conserva su firma y falla explícitamente ante configuración inválida; se documenta el panic y se ofrece la alternativa fallible. Como los campos son públicos, cada actualización vuelve a validar el estado.

La ganancia y P⁺ se evalúan por razones pequeñas. Si P⁻≥R, se usa r=R/P⁻, K=1/(1+r) y P⁺=R/(1+r); en la otra rama se usa r=P⁻/R. Esto evita la suma P⁻+R y el producto P⁻R desbordados. La media se actualiza como combinación convexa, evitando la resta desbordada de observaciones con signos opuestos.

Se eliminan pisos absolutos de P y R. x, P y el nuevo R se publican sólo cuando la actualización completa es válida. Si P+Q no es representable o la innovación es singular, se devuelve error sin modificar el estado aceptado. No se pretende resolver cualquier magnitud arbitraria: el rechazo explícito es parte del contrato.

El wrapper update mantiene el último x ante error; no publica calidad. Si el usuario corrompe x directamente, ese wrapper no lo repara ni garantiza una salida finita: debe usarse try_update y manejar Err. El wrapper de R dinámico conserva la compatibilidad de reutilizar R anterior cuando el parámetro dinámico no es válido. No es un mecanismo de autoaprendizaje de R.

### FMT-002 permanece abierto en la integración

El consumidor sigue usando R≈precio·0,0005 y Q fijo por llamada. Haber quitado pisos en el helper no vuelve correcta esa unidad ni introduce Δt. El método de modulación por volatilidad mantiene su fórmula heurística y no estima separación entre ruido observacional y ruido del proceso.

Cierre requerido: especificar proceso latente/observación, expresar difusión y R en unidades coherentes, registrar innovaciones y cobertura, comparar invariancia al cambio de unidades y reempaquetado temporal. No se asigna automáticamente toda volatilidad de precio a ruido de medición: podría representar cambios reales del proceso. Los diez tests actuales de StatefulEngine pasan, pero no verifican ese contrato estadístico.

## 4. FMT-046 — observación multiactivo inválida contaminaba varias memorias

**Evidencia:** [MarketCorrelationHeatmap](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/correlation.rs:59>), [tests](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/correlation_input_contract.rs:1>).

Antes se actualizaban last_prices y componentes del estimador mientras se recorría el vector, sin un rechazo conjunto. Algunos EWMAs podían ignorar un NaN, mientras otros sí avanzaban y los precios cambiaban. Eso produce memorias con conjuntos de observaciones diferentes. El siguiente resultado puede ser finito y, aun así, estar contaminado.

Se reprodujo con dos instancias idénticas. Tras un historial común, una recibe [NaN,150] y luego ambas reciben [120,110]. Antes divergen: aproximadamente 0,39955 frente a 0,41924. Otro test usa un cociente entre 10^300 y 10^-300: el retorno no representable contaminaba la referencia siguiente. Ambas pruebas ahora verifican igualdad respecto al control que nunca aceptó el evento inválido.

### Contrato nuevo

try_update devuelve Result<Option<f64>>:

- Err: corte incompleto, precio inválido o momento no representable.
- Ok(None): baseline o varianza insuficiente para identificar correlación.
- Ok(Some(c)): estadístico definido bajo la construcción de la cesta.

Se valida todo el corte. Retornos, medias, varianzas y covarianzas se calculan en memoria temporal reutilizable; sólo se confirman precios y momentos cuando termina la operación. No hay asignaciones dinámicas por update después de construir el objeto, aunque la memoria aumenta O(N). La transacción es lógica y local a este objeto bajo &mut self, no una transacción distribuida ni una garantía de sincronización del feed.

El primer vector establece precios, sin inventar retornos cero como observaciones iniciales. El constructor rechaza una cesta vacía o periodo inválido. Un test adicional fuerza desbordamiento del segundo momento y comprueba que no se confirma estado parcial. Otro conserva correlación uno para retornos idénticos no constantes.

### Lo que NO calcula

Sigue siendo la media de corr(r_i, media(r)), con cada activo incluido en su benchmark. No es beta, una matriz por pares ni un objeto suficiente para riesgo multiactivo. Para activos independientes y de igual varianza, la autocontaminación poblacional con su propia cesta da 1/sqrt(N); no debe interpretarse como correlación de pares.

Los comentarios ahora describen el estimando y coste O(N). No se localizó un caller productivo de este módulo; no se atribuye el arreglo al guard de posiciones. Tampoco hay timestamp, máscara por activo, edad, tratamiento de llegadas asíncronas ni intervalos de incertidumbre. Rechazar un corte completo conserva coherencia local, pero puede reducir cobertura si el caller entrega snapshots parciales; requiere observabilidad antes de integración.

El wrapper legacy sigue devolviendo cero tanto para Err como para None. Esa ambigüedad se conserva por firma y queda explícita; la interfaz fallible es obligatoria si la ausencia afecta decisiones. Periodo=1 no puede identificar covarianza histórica en esta formulación; no basta que sea un periodo admitido por EWMA.

## 5. FMT-004 — evolución espectral V2 sin migración silenciosa de features

**Evidencia:** [SpectralCycleEngine y EventSpectrumV2](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/spectral.rs:9>), [API V2](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/spectral.rs:124>), [pruebas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/event_spectrum_v2_contract.rs:1>).

El consumidor [actual](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:504>) continúa con analyze_spectrum, cada 64 ticks. Se preserva su semántica ML porque cambiar centrado y escala de potencia sin versionar el dataset/modelo sería otra ruptura de paridad.

Un test diagnóstico confirma dos defectos legacy: una entrada constante no nula tiene potencia fuera de DC por multiplicación con un taper; una sinusoidal de amplitud 10^-20 conserva bin dominante pero devuelve centroide cero por el piso absoluto de potencia. Pasar ese test **no** significa que esos defectos hayan sido arreglados en producción.

### Qué añade V2 y para qué sirven sus cálculos

1. Se registra una máscara de validez por evento. Un NaN no se confunde con un cero observado: V2 rechaza la ventana hasta que ese evento sale del buffer. No se elimina el evento comprimiendo el índice temporal.
2. Se resta la media aritmética antes del taper. Una constante no nula deja de parecer una oscilación. No se eliminan tendencias polinómicas ni se declara estacionariedad.
3. Se conserva el taper izquierdo de medio coseno. Se corrige su nombre documental: no es Planck-Taper. Su forma y longitud siguen siendo una elección de ventana finita, no óptimo aprendido.
4. Para la FFT Y_k, se publica potencia unilateral P_k=c_k|Y_k|²/(N·sum(w²)), donde c_k=2 en bins interiores y 1 para DC/Nyquist. Por Parseval, la suma equivale a energía de la señal centrada y ventaneada dividida por sum(w²).
5. Bin dominante y centroide excluyen DC. El centroide utiliza potencias normalizadas sin un piso absoluto, de modo que cambiar amplitud no cambia la forma espectral dentro de precisión representable.
6. Se escala la entrada antes de las mariposas y se restaura la potencia con orden de operaciones controlado. Overflow o subdesbordamiento de un bin positivo a cero se informan como UnrepresentablePower, no como potencia cero.

No se publica PSD por hertz: las unidades son entrada² por bin. k significa k/64 ciclos por observación. Para hertz hace falta especificar muestreo físico y calidad; timestamps irregulares no se corrigen multiplicando por una tasa media sin estudiar el sesgo.

### Evidencia y limitaciones

Ocho tests: constantes hasta f64::MAX; invariancia de forma y escalado cuadrático de potencia; invariancia a offset; Parseval con componente Nyquist; warmup/NaN/expiración; overflow y underflow explícitos; orden del ring tras envolverse; y diagnóstico legacy abierto.

La máscara incorpora 64 booleanos de estado lógico y su actualización al push. No se afirma latencia constante universal ni mejora p99; no hubo benchmark. El cálculo mantiene memoria fija y FFT O(N log N). No agrega nuevos parámetros genómicos ni conexión productiva.

V2 es una API implementada y probada para preparar migración, **no un motor universal continuo**. Mantiene 64 muestras, una sola ventana y sin intervalos de confianza. El problema de FMT-003 —proxy de momentos absolutos presentado como Hurst en tres ventanas— permanece; esta ronda lo relee pero no lo reimplementa bajo otro nombre.

## 6. FMT-091 — el régimen global sigue siendo una clasificación operativa rígida

**Ruta real:** [productor global BTC](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:748>), [consumo en riesgo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:827>), [veto long](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/orchestrator.rs:125>).

La tendencia relativa de dos EMAs y el Hurst de BTC generan BullRun/Crash/Chaotic/Range, con cortes ±0,015 y límites genéticos recortados. El veto se aplica al portafolio sin pedir el instrumento, tau o papel de cobertura de la intención. El término “Chaotic / Mean Reverting” no prueba equivalencia entre caos, antipersistencia y reversión condicional.

La API alternativa [RegimeDetector](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/regime.rs:60>) no es ese productor. Tres tests diagnósticos constatan:

- Código 255 se decodifica como Range, perdiendo distinción entre estado desconocido y clasificación válida.
- Una observación NaN conserva Crash sin metadatos de edad/calidad.
- Cambiar correlación 0,6 a 0,6+10^-12 cambia Range a BullRun; incluso correlación 2, fuera del dominio, produce una etiqueta válida.

Estos tres tests siguen pasando porque el comportamiento permanece. No se atribuye la correlación 2 al caller BTC real ni se reemplaza automáticamente Crash por una función suave.

**Criterio de rehabilitación:** separar representación continua del estado e incertidumbre, estimación de cambio, y política de admisión. Los límites de solvencia, orden válida y autorización pueden seguir siendo duros. Si se conserva una taxonomía para UI, debe ser una proyección diagnosticable; no el estado completo ni el único input de sizing. La retirada de un veto exige una política sustituta comprobada, no sólo autorización genérica de “evolucionar”.

## 7. FMT-166 — dos dimensiones genéticas carecen de lectura operativa localizada

**Evidencia:** [campos de configuración](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/config.rs:56>), [campos del genoma](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:140>), [mutación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:1855>).

La búsqueda de regime_duration_ms y regime_atr_multiplier en Rust bajo crates y src produce 22 líneas, concentradas en config.rs y genome.rs. Incluye declaración, defaults, aleatorización, snapshot/aplicación, mutación, vectorización y reconstrucción. No localiza un estimador de régimen que consuma su valor para decisiones. El productor real del régimen lee otros genes de Hurst.

**Impacto:** optimizar el vector y comprobar que se serializa no demuestra que estas coordenadas modifiquen observación, score, veto o sizing. Tampoco cabe afirmar efecto total exactamente cero sobre el proceso evolutivo: sus coordenadas pueden participar indirectamente en distancias, normalizaciones o selección de candidatos. La búsqueda estática no demuestra ausencia de todo consumidor externo/dinámico.

No se borran genes ni se compactan índices: hacerlo puede romper cromosomas persistidos, CMA y modelos históricos. Cierre: catálogo gen→consumidor con identidad de esquema; pruebas pareadas de influencia y regiones de saturación; política de deprecación/migración para genes realmente inactivos. Conectarlos a una nueva fórmula porque “deben hacer algo” sería otra arbitrariedad.

## 8. FMT-140 — una corrección del crate no alcanza la biblioteca duplicada

**Evidencia:** [correlación root](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/features/correlation.rs:32>), [export root](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/features/mod.rs:1>), [caracterización](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/tests/legacy_correlation_diagnostics.rs:18>).

src/features/correlation.rs contiene otra implementación. Filtra precios inválidos individualmente, pero imputa retorno cero y sigue actualizando la cesta/otros activos; además asigna un Vec por llamada. No es un alias del crate corregido.

Se compiló esa misma fuente en un harness de test y se reprodujo cambio del resultado futuro tras un snapshot [NaN,150]. Sus tests históricos de finitud también pasan: ambas cosas son compatibles. Finitud no equivale a conservación del estado observado.

La divergencia amplía FMT-140, ya existente para estadísticas duplicadas. No se cuenta como un nuevo ID. No se editó la copia root ni se redirigió su export sin inventariar consumidores. Su migración debe preservar intención pública o declarar cambios de semántica; una búsqueda por nombre compartido no establece identidad de implementación.

## 9. T38 — evolución del espectro de volatilidad y dependencia, no un catálogo de regímenes

### 9.1 Base científica contrastada

Se usó Firecrawl Research Index, tras comprobar que el CLI no estaba disponible; no se instaló ni se enviaron datos del repositorio. La búsqueda y expansión incluyeron familias vecinas. Se leyeron pasajes pertinentes de los dos trabajos principales, no todos los artículos íntegros.

**Escalas continuas y observaciones irregulares.** Continuous Time Locally Stationary Wavelet Processes distingue la representación continua de su estimación a partir de muestras finitas. Define un espectro no negativo a partir de amplitudes, impone regularidad/integrabilidad y formula un problema integral inverso con regularización. Su escala máxima estimable depende del soporte observado; no se obtiene todo el continuo simplemente aumentando un rango numérico. [Fuente primaria](https://arxiv.org/abs/2310.12788).

**Dependencia multivariante localizada.** El trabajo mvLSW describe espectros matriciales, coherencia y coherencia parcial. La corrección de sesgo puede perder positividad y exige regularización. El estimador presentado utiliza suavizado simétrico alrededor del instante y la implementación descrita requiere una serie regular de longitud diádica. Copiarlo directamente como predictor online introduciría información futura o hipótesis de muestreo no satisfechas. [Fuente primaria](https://arxiv.org/abs/1810.09810).

**Familias complementarias identificadas por abstract:** espectros por cuantiles para dependencia no capturada por covarianza; adaptación local de espectros wavelet; inferencia espectral bayesiana con likelihood Whittle dinámica. No se implementan ni se declara superioridad entre ellas. [Quantile Spectral Analysis](https://arxiv.org/abs/1404.4605), [Locally adaptive estimation](https://arxiv.org/abs/0808.1452), [Bayesian nonparametric spectral analysis](https://arxiv.org/abs/2303.11561). La expansión también localizó bases localmente estacionarias y shrinkage Haar-Fisz: son opciones de estimación/regularización, no evidencia de alpha. [Basis Processes](https://arxiv.org/abs/2106.03533), [Haar-Fisz](https://arxiv.org/abs/1309.2435). La variante de texturas 2D se conserva como antecedente de otro dominio, no como diseño transferido a trading.

### 9.2 Contrato propuesto: separar estado, incertidumbre y política

Lo siguiente es síntesis de diseño para este sistema, **no una implementación ni una garantía extraída de los artículos**:

1. Representar el contexto como observaciones de un campo temporal multivariante y su incertidumbre: amplitudes, dependencia entre activos, profundidad/costes, timestamps y soporte. “Volatilidad alta/baja” puede servir para presentación, pero no sustituye el campo.
2. Mantener un espectro matricial S(t,s) por tiempo y escala con unidades y medida de integración declaradas. S(t,s) debe ser semidefinida positiva para interpretar energía de cualquier combinación a; aᵀS(t,s)a expresa su energía local. No es automáticamente la pérdida monetaria de esa cartera.
3. Si se integra sobre escala, especificar ds o dlog(s). No se pueden cambiar las medidas omitiendo su Jacobiano. Tampoco se identifica energía wavelet con varianza física sin la normalización del modelo y el kernel correspondiente.
4. Separar ese contexto del horizonte de decisión tau. La escala donde se mide una fluctuación y el horizonte donde se predice un payoff son coordenadas distintas.
5. La política genética puede parametrizar una función de tau, contexto e incertidumbre, con presupuesto computacional y restricciones económicas explícitas. Usar splines/bases no prueba adaptación: deben existir parámetros identificables, pérdida causal, actualizaciones y validación.
6. Si se usa una variable latente de cambio, conservar distribución/incertidumbre o estadísticas suficientes. No convertir un argmax prematuro en cuatro “verdades” mutuamente excluyentes para todos los activos.

Este contrato se conecta con T37: expertos que predicen el mismo target pueden combinarse con disponibilidad explícita. No se mezclan probabilidades de eventos a distintos horizontes como si fueran una sola probabilidad.

### 9.3 Plan de evaluación falsable

- **Antes del modelo:** observaciones as-of, reloj físico/eventos declarado, huecos, revisiones y máscaras; ninguna imputación silenciosa.
- **Ensayo controlado:** señales constantes, sinusoides fuera de bin, chirps, cambios de amplitud, saltos y dependencias cruzadas conocidas; estudiar resolución frente a coste y sesgo de borde.
- **Causalidad:** prohibir ventanas centradas con datos posteriores a decisión. Un estimador offline puede servir de referencia retrospectiva, no de feature disponible entonces.
- **Multiactivo:** verificar simetría/PSD, invariancia por unidades y permutación de activos; medir cómo regularización altera dependencias y riesgo. Una matriz invertible no acredita identificación.
- **Comparación:** baseline actual, V2 y candidato T38 en shadow sobre el mismo tape. Medir cobertura, errores, retraso, memoria y p50/p99, además de desempeño predictivo fuera de muestra.
- **Genoma:** registrar diferencias observables por perturbación, no sólo que haya cambiado su hash. Mantener versión del estimador, target y política junto al feedback.
- **Promoción:** primero paridad de esquema y causalidad; después evaluación económica con costes y soporte. No inferir ventaja por menor error in-sample o mayor complejidad matemática.

## 10. Matemática avanzada, física y cuántica: criterio de integración

Una ecuación de un problema del milenio no es un componente de alpha por su dificultad. Antes de transferir una teoría hacen falta variables observables, hipótesis, condiciones iniciales/de frontera, solución numérica verificable y una pregunta falsable del sistema. Si se propone una analogía con fluidos, debe definirse qué magnitud se conserva y qué término representa órdenes, cancelaciones y ejecuciones. Si se propone una evolución cuántica, debe distinguirse simulación clásica, heurística inspirada y ejecución en hardware cuántico.

Esta ronda corrige la etiqueta del Kalman: es clásico. T38 aprovecha teoría de procesos estocásticos no estacionarios y problemas inversos por pertinencia al muestreo, no por prestigio. No demuestra solvencia, predictibilidad universal, observación nanosegundo a nanosegundo o evidencia empírica a cien años.

La autoevolución necesita una cadena cerrada: propuesta identificable → efecto causal medible → evaluación comparable → aprobación versionada → despliegue controlado → feedback atribuido → rollback. Siguen abiertos varios eslabones descritos en las rondas anteriores. Añadir otro estimador no los cierra.

## 11. Matriz de módulos y hoja de cierre

| Módulo del maestro | Aporte XV | Trabajo aún necesario |
| --- | --- | --- |
| 1. Ingestión/normalización | Validación conjunta de snapshots auxiliares | Relojes, huecos y sincronización real |
| 2. Inferencia/señales | Kalman numérico; V2 y legacy distinguidos | Reparametrizar R/Q; versionar/reentrenar features |
| 3. Estrategia/horizontes | Régimen global separado del estado continuo | Política por instrumento/tau/contexto validada |
| 4. Ejecución/red | Sin cambios ni consultas autenticadas | Validación extremo a extremo, no auditada íntegramente aquí |
| 5. Riesgo/genomas | Genes sin consumidor y veto global trazados | Influencia, reservas, política sustituta del veto |
| 6. Estado/telemetría | Rechazo sin commit parcial en estimador local | Calidad visible hasta consumidor; no equivale a atomicidad distribuida |
| 7. Cuántica/confluencia | Etiquetas científicas y T38 precisadas | Evidencia de integración y ventaja medible |
| 8. Backtest/gobernanza | Regresiones, diagnósticos y hashes | Pruebas económicas, migración, cobertura completa |

Orden por dependencias: calidad/tiempo → estimando y unidades → esquema de features → consumidores y genoma → evaluación causal → política de riesgo → promoción. No retirar guardas antes de que la alternativa responda a la misma obligación de seguridad.

## 12. Verificación y estado de entrega

| Comprobación | Resultado |
| --- | --- |
| feature-engine --lib, excluyendo quantum_tensor_store::tests | 54 tests, 3 excluidos para evitar la superficie de persistencia del tensor |
| kalman_numeric_contract | 8 tests nuevos; cinco rojo→verde |
| correlation_input_contract | 5 nuevos; dos rojo→verde |
| event_spectrum_v2_contract | 8 nuevos; uno caracteriza deuda legacy |
| legacy_correlation_diagnostics | 1 diagnóstico nuevo + 4 tests históricos incluidos por path |
| regime_taxonomy_diagnostics | 3 diagnósticos nuevos de deuda abierta |
| god-engine-core --lib stateful_engine::tests | 10 tests existentes |
| cargo check --bin god_engine --offline | Pasa sin ejecución |
| rustfmt --check y git diff --check | Pasan en archivos intervenidos |

Los 93 son pruebas distintas de esta ronda; no se suman ejecuciones repetidas. Los 25 nuevos incluyen cinco diagnósticos abiertos. No se ejecutó suite completa, tensor persistente, backtest económico, benchmark ni integración de cuentas. Permanecen warnings anteriores de latest_ts, mode y RealWfOutcome.trades.

El [JSON XV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XV_2026-09-24.json>) registra alcance, fuentes, comandos, pruebas, estados y hashes. Se agregan adendas al maestro, atlas y XIV. Los seis consumidores/configuraciones protegidos mantienen su contenido: core/lib.rs, risk/lib.rs, stateful_engine.rs, god_engine.rs, genome.rs y config.rs. No se modifican genomas activos ni la fórmula del veto.

Sin commit, push, merge, fetch, despliegue, reinicio, cambio de cuentas ni publicación de modelo. Main local no demuestra sincronización remota. El resultado es una mejora local verificable y una ampliación de la auditoría, no certificación de universalidad o autoevolución productiva.

### Control final de integridad

JSON parseado: siete registros de hallazgo y dos IDs nuevos. Coinciden los dieciocho hashes del inventario, incluidos los seis archivos protegidos. Los veinte enlaces locales del informe resuelven y sus líneas están dentro de los archivos. Los prefijos completos del atlas, maestro y XIV conservan su SHA-256 después de normalizar CRLF a LF; se añadieron respectivamente 2.772, 5.997 y 1.189 caracteres normalizados. La apertura del informe en el panel puede quedar en cola; no equivale a una validación científica. Los errores transitorios de escritura del parche se verificaron leyendo su resultado antes de continuar; no se duplicaron las adendas ni se restauró el repositorio con comandos destructivos.

## Continuación aditiva XVI — identidad y evidencia multiactivo

La [ronda XVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XVI_2026-09-24.md>) y su [artefacto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XVI_2026-09-24.json>) amplían esta auditoría sin sustituir las conclusiones XV. Añaden FMT-167/168/169/170, reparan subcontratos de FMT-088/034/036 y reconfirman FMT-033/050.

Se corrigen exposición fabricada por el asignador, capital inválido convertido en 13, desbordamientos y commits parciales de cesta, normalización dependiente de unidades, alias de instrumentos, repetición de pares por ticks ajenos y suelo de quote que cruzaba libros diminutos. Continúan OU sin reloj, bloqueo persistente por salto, K rígido, fallback maker inválido y quote descartada por process_event. El umbral de StatArb puede ser inalcanzable por la cota de su estadístico internamente estandarizado.

48 tests distintos pasan, 22 nuevos; doce rojo→verde y seis diagnósticos abiertos. Cuatro fuentes y cinco archivos de pruebas; check sin ejecutar el motor. La cobertura avanza a 116/289 Rust completos, 173 pendientes, contando solo maker.rs como nueva lectura acreditada. T39 propone covariación multiactivo asíncrona con contratos de evidencia, sin implementación productiva ni ventaja económica demostrada. No hubo publicación Git, despliegue ni operación de cuenta.
