# Auditoría científica XI — paridad del genoma y adaptación estadística comprobable

Fecha: 2026-09-24. Rama local main; corte HEAD 59a76de4. Adenda de la ronda X: se conserva su diagnóstico histórico y se actualiza aquí el estado de reparación. Esta ronda no acredita auditoría integral, rentabilidad ni despliegue.

## 1. Resultado y alcance

Se unificaron los lectores temporales de SuperGenotype y QuantumConfig bajo las fórmulas que ya utilizaba la configuración operativa. Esto corrige una divergencia concreta entre genotipo consultado y parámetro servido, sin cambiar límites ni fórmulas de ejecución de QuantumConfig. También se eliminó la dependencia de cachés derivadas obsoletas en las consultas del genoma.

En un módulo adicional leído íntegramente, adaptive_quantiles.rs, se corrigieron la construcción de un estimador con probabilidad NaN y desbordamientos intermedios que rompían su comportamiento ante cambios de unidades. Se explicita que un cuantil acumulativo no es automáticamente un detector de régimen, una estimación de probabilidad de ganar ni un mecanismo de olvido temporal.

**Pruebas:** 63 tests seleccionados aprobados, de los cuales 15 son nuevos. Siete regresiones fallaron antes de corregir el código: cinco de paridad y dos de cuantiles. Los tres testigos de discrepancia de X se transformaron en regresiones de igualdad y ahora pasan como tales. cargo check del host finalizó correctamente sin ejecutar el motor.

**Cobertura conservadora acumulada:** 95 Rust preexistentes distintos con lectura completa, sobre 289 del inventario base; quedan 194 sin acreditar lectura íntegra. Se añade adaptive_quantiles.rs, 278 líneas antes y 314 después. Los nuevos módulo/tests no aumentan el denominador de código histórico. Las lecturas dirigidas del núcleo, riesgo y genoma no equivalen a lectura completa de esos archivos.

**Fuera de esta intervención:** cantidades enviadas, hot-swap transaccional, almacenes activos, procesos de trading, conexiones de exchange, fusiones de ramas y publicación Git. No se alteraron host, replay ni los lib.rs de riesgo y núcleo, que contienen trabajo concurrente.

### Matriz de estado

| ID | Prioridad | Estado XI | Qué se ha probado |
| --- | --- | --- | --- |
| FMT-134 | P1 | Reparación local de lectura; paridad concurrente global pendiente | Mismas consultas deterministas y mismos genes autoritativos |
| FMT-135 | P2, API auxiliar | Corregido en construcción | NaN no crea un estimador inerte; constructor validado sin panic disponible |
| FMT-136 | P2, numérico auxiliar | Corregido en los casos reproducidos | Cambio multiplicativo de unidades, extremos finitos y orden de marcadores |
| FMT-137 | P2, diseño/afirmación auxiliar | Descripción corregida; adaptación temporal no implementada | El objeto estima historia acumulada, no una ventana reciente |
| FMT-040/112/133 | Histórica | Reparaciones de X conservadas | Regresiones de banda e interpolación vuelven a pasar |
| FMT-070/042 | Histórica | Abiertos | Igualar fórmulas no crea un snapshot ni una promoción transaccional |
| FMT-113 | P1 | Abierto | No se ha aplicado aquí el presupuesto a cantidad/payload finales |
| CES-008/009 | P1 | Abiertos parcialmente | No se elimina falta de soporte, recortes de decisión ni esquema de dos anclas |

Los tres nuevos identificadores son FMT-135–137. No se suman como tres incidentes de producción: el estimador de cuantiles no tiene consumidor operativo localizado en las búsquedas de crates y src. La matriz histórica de 305 puntos no se renumera ni se declara resuelta por esta adenda.

## 2. FMT-134 — cerrar el desacuerdo sin ampliar exposición operativa

### 2.1 Hechos que permiten elegir una política común

La ronda X demostró que SuperGenotype recortaba tau a 30 s–12 h, mientras QuantumConfig evaluaba sus coeficientes con un piso de 1 ns y sin ese techo. También aplicaban distintos pisos a OBI y trailing. Esta ronda rastreó los consumidores: el host, replay, riesgo y gestión de posiciones usan QuantumConfig; las consultas directas de SuperGenotype aparecieron en definición/tests, no en esas rutas.

Ejemplos de conexión: [gestión de posiciones](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1081>), [trailing](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1250>) y [normalización del riesgo](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:343>). No se editaron estos consumidores.

La decisión de migración es explícita: **conservar la política efectivamente servida y hacer que el genoma la represente al consultarlo**. No se escogió una nueva tasa de riesgo ni se retiró un cap del exchange. El comportamiento anterior de las consultas de SuperGenotype fuera de banda y en regiones saturadas sí cambia; es una modificación semántica de esa API, documentada y probada, no un cambio invisible.

### 2.2 Implementación y significado de cada cálculo

[horizon_policy.rs](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/horizon_policy.rs>) centraliza la evaluación:

```text
tau_eff = max(tau_ms, 10^-6 ms)
x = ln(tau_eff / 1 ms)
raw = exp(a + b*x)
served = clip(raw, bounds_parameter)   cuando esa familia tiene límites
```

a es el logaritmo del valor a la referencia de 1 ms y b es elasticidad respecto a tau. Los parámetros de TP/SL son fracciones del precio; los de trailing son múltiplos de ATR. La evaluación de Kelly devuelve un parámetro acotado del modelo, no prueba que sea una fracción óptima de riqueza. OBI es un umbral de indicador, no una probabilidad.

| Familia | Política operativa conservada | Consecuencia para interpretación |
| --- | --- | --- |
| TP/SL | Sin recorte de salida en este lector | Necesitan validación de geometría/ejecución posterior |
| Kelly | [0,01;3] | Saturación posible; el valor no acredita edge |
| Trailing mult./activación | [1,5;6] | Fuera de ese intervalo el gen queda saturado |
| Trailing paso | [0,5;4] | La continuidad no evita regiones de sensibilidad nula |
| OBI | [0,10;0,95] | Un gen menor que 0,10 se sirve como 0,10 |
| Máximo trailing | clip(1,5·activación,2,7) | Derivación de política conservada, no una cuarta curva aprendida |

La constante runtime-v1 identifica estas fórmulas de lectura. **No es un ID de generación**, no se persistió como contrato transaccional y no hace atómicas las cargas separadas. Sirve para localizar y explicar una política que antes estaba duplicada.

El evaluador común permite que todas las familias se consulten sobre la misma coordenada continua sin decidir por pertenencia a scalping o swing. Las anclas históricas aún parameterizan una ley de potencia; no constituyen por sí mismas dos motores, pero siguen limitando la familia de funciones representable. Los nombres de serialización se conservan para no romper genomas existentes.

### 2.3 Cachés derivadas obsoletas

Hay una segunda divergencia bajo el mismo contrato: QuantumConfig::from_genome y apply_to_arena ya reconstruían Kelly/trailing/OBI desde sus genes; los getters de SuperGenotype consultaban directamente las curvas almacenadas. Modificar genes o deserializar un objeto con curvas derivadas antiguas podía cambiar qué modelo se veía según el lector.

El test prepara genes válidos y sustituye sólo las curvas derivadas por constantes incompatibles. Antes, Kelly del genoma devolvía 0,99 y la configuración aproximadamente 0,157232 para tau=600.000 ms. La prueba falló antes del cambio.

Ahora [los getters](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:2051>) reconstruyen la curva pertinente mediante la misma conversión de anclas que sync_continuous_curves. No se clona todo el genoma por consulta. TP y SL conservan sus coeficientes autoritativos; no se reconstruyen desde vistas legacy. Esto alinea el significado de la consulta con la conversión operativa existente.

**Compatibilidad:** escribir únicamente kelly_horizon_curve u otra caché derivada ya no cambia el valor servido por el getter del genoma. Ese cambio es deliberado: tampoco cambiaba lo servido por from_genome. Quien quiera una nueva familia autoritativa debe migrar esquema, vector evolutivo y consumidores, no modificar sólo una caché.

### 2.4 Verificación, coste y límites

[Seis tests nuevos](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/horizon_reader_parity.rs>) comprueban todas las familias sobre las 32 escalas, cachés obsoletas y equivalencia con las fórmulas anteriores de QuantumConfig en 257 horizontes logarítmicos. Cinco fallaron antes y pasan después; el sexto protegió desde el principio las salidas operativas válidas.

[Los tres testigos de X](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/genome_reader_diagnostics.rs>) ahora exigen igualdad. Los resultados históricos de desigualdad siguen documentados en X y en su artefacto, cuyos hashes pertenecen a aquel corte, no al estado actual.

No se añadió asignación heap a los lectores de configuración. El trailing sigue calculando un solo logaritmo para sus tres curvas; los helpers son inline. La consulta directa del genoma reconstruye curvas derivadas y por ello tiene coste adicional respecto a leer una caché. No se midió latencia P99 ni uso de CPU del proceso operativo; estas observaciones de código no sustituyen un benchmark.

**Cierre acotado:** se repara paridad determinista para un mismo genoma y política. La lectura simultánea con una mutación todavía puede mezclar a/b o familias de distintas generaciones. El riesgo FMT-070 sigue abierto. Las cargas Relaxed individuales y el nombre runtime-v1 no resuelven esa consistencia.

La API de compatibilidad tampoco es un validador: f64::max convierte tau NaN/no positiva al piso y exp puede desbordar. Este comportamiento se mantiene como en QuantumConfig anterior y se hace explícito. Antes de usar resultados como autorización de orden se necesita validación tipada y una ruta de rechazo observable. No se declara aquí que todos los inputs posibles sean seguros.

## 3. FMT-135 — un percentil NaN congelaba silenciosamente el estimador

**Ámbito:** P2Quantile, [constructor](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/adaptive_quantiles.rs:25>). Prioridad P2 en esta API auxiliar; no se localizó conexión productiva.

El constructor antiguo aplicaba p.clamp(0,01,0,99). NaN sigue siendo NaN tras ese clamp. Las posiciones deseadas y sus incrementos quedaban contaminados; después de inicializar, el desplazamiento d=np[i]−n[i] era NaN y las comparaciones que deciden mover marcadores eran falsas. El proceso aceptaba observaciones y aumentaba contadores sin realizar el ajuste correspondiente al percentil pretendido. Una salida finita no acreditaba un estimador válido.

No se debe confundir la probabilidad objetivo p con un valor observado x. El código ya descartaba muestras no finitas; esa defensa no validaba la configuración. El test existente llamado NaN immunity sólo ejercitaba muestras inválidas, de modo que no cubría este fallo de constructor.

### Corrección

[try_new](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/adaptive_quantiles.rs:31>) devuelve Result y admite exactamente 0<p<1 con p finito. No recorta probabilidades válidas como 0,001 o 0,999. Los extremos 0 y 1 no se representan mediante el marcador interior de P²: para ellos corresponde distinguir mínimo/máximo.

new conserva el recorte histórico a [0,01;0,99], incluidos infinitos, para no alterar consumidores ya configurados. La excepción es NaN: ahora produce un fallo explícito de construcción. Se documenta el panic; los consumidores de configuración externa deben utilizar try_new y gestionar el error. Los consumidores localizados del wrapper usan literales válidos, no entradas de usuario.

**Pruebas:** la regresión que exige rechazo de NaN no produjo el panic esperado antes del arreglo y pasa después. Otras pruebas verifican el constructor fallible, probabilidades válidas extremas, recortes legacy y conservación de todo el estado ante muestras inválidas.

**Limitación:** los campos de P2Quantile siguen siendo públicos y mutables. Un llamador aún puede corromper p, count o las posiciones después de construirlo. Validar el constructor no es una garantía universal contra mutación arbitraria del estado. Tampoco crea un nivel de precisión estadística.

## 4. FMT-136 — las unidades alteraban la trayectoria de P² por overflow

**Ámbito:** [interpolación parabólica y lineal](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/adaptive_quantiles.rs:129>), P2Quantile. P2 numérico auxiliar.

### 4.1 Propiedad y reproducción

El ajuste de altura de un marcador es homogéneo respecto a un cambio multiplicativo positivo de unidades. Si todos los datos se multiplican por k, el cuantil y sus marcadores deberían escalar por k, salvo error de coma flotante. Las posiciones de orden no deberían cambiar por ese simple cambio de unidades.

Se alimentaron 500 observaciones repetidas de −1, −0,5, 0, 0,5 y 1 y la misma secuencia multiplicada por 10^308. Para p=0,1, el estimador original produjo aproximadamente −0,9999410143 en unidades base y −0,9998493536 al reconvertir la versión escalada. La regresión de tolerancia 10^-10 falló.

No es evidencia de cotizaciones de 10^308 en Binance. Es un contraejemplo del contrato numérico y de la afirmación de inmunidad al recibir datos finitos. También ilustra por qué comprobar exclusivamente is_finite del resultado no detecta todas las alteraciones del algoritmo.

### 4.2 Causa matemática

La fórmula multiplica separaciones de marcadores por diferencias de alturas antes de dividir. Un producto intermedio puede desbordar aunque el resultado final matemático sea representable. Si la propuesta parabólica se vuelve infinita/NaN, el algoritmo cae a su rama lineal, modificando la trayectoria del estimador.

Además, restar extremos finitos de signos opuestos puede producir infinito. El fallback lineal original usa precisamente esa diferencia. Un dato finito no garantiza intermediarios finitos.

### 4.3 Reparación

Se conserva exactamente la fórmula ordinaria cuando su propuesta es finita. Sólo ante una propuesta no finita se reevalúa la misma expresión con las tres alturas divididas por su máximo absoluto y se restaura la escala al terminar. La regla que rechaza una parábola que cruza marcadores vecinos sigue intacta: no se acepta un overshoot porque venga normalizado.

En la rama lineal, cuando la diferencia desborda, se usa una combinación convexa:

```text
lambda = d / (n_neighbor - n_i), con 0 < lambda < 1
q_new = (1-lambda)*q_i + lambda*q_neighbor
```

Los signos de d y de la diferencia de posiciones coinciden en el movimiento permitido. Esta forma evita restar extremos opuestos y mantiene el valor entre ellos bajo las precondiciones del algoritmo. No se introduce un epsilon dependiente de dólares, ni un clip artificial de observaciones, ni un retorno cero que invente neutralidad.

La consulta de arranque usa ahora una copia fija de cinco alturas en la pila en lugar de construir un Vec. Se preserva el estadístico de orden que se devolvía para menos de cinco observaciones. La comprobación es de valores y la ausencia de Vec se constata en el código; no se midió un contador de asignaciones ni una mejora de latencia.

### 4.4 Evidencia y límites

[Nueve tests de cuantiles](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/quantile_numeric_contract.rs>) cubren construcción, muestras inválidas, warmup, cuatro percentiles bajo reescalado, marcadores finitos ordenados y secuencias con ±f64::MAX. La prueba de unidades falló antes y pasa después. La prueba de extremos opuestos se añadió después de la reparación; no se presenta como otra regresión rojo→verde.

No se ha probado uniformemente todo f64 ni trayectorias arbitrariamente largas. count/u64, posiciones/i64 y np/f64 conservan límites de representación. No se restauran estados corruptos ni se incorpora una política de reset. Es una reparación numérica de los casos demostrados, no una prueba formal de convergencia del estimador bajo cualquier régimen.

## 5. FMT-137 — “dinámico” no significa olvido temporal ni ausencia de constantes

**Estado:** afirmaciones corregidas; capacidad temporal pendiente. **Prioridad:** P2 de diseño auxiliar.

El encabezado previo afirmaba eliminar completamente los umbrales hardcodeados. Sin embargo, AdaptiveQuantileEngine selecciona p=0,80 para OFI/OBI, p=0,85 para la señal holística y p=0,50 para ATR; aplica pisos 0,02/0,10 y fallbacks 0,15/0,40. Ninguna de esas elecciones se estima automáticamente por P².

El estimador no recibe timestamps, no tiene ventana móvil y no elimina observaciones antiguas. Estima un cuantil acumulativo por eventos aceptados. Dos secuencias con iguales valores y orden, pero separaciones temporales muy diferentes, son indistinguibles para él. Eso impide describir su memoria como una función identificada de tiempo físico.

### 5.1 Contraejemplo de régimen y madurez

La prueba introduce 10.000 ceros seguidos de 100 unos. El cuantil exacto de las últimas 100 observaciones es 1; el estimador acumulativo devuelve menos de 0,5. No se exige que P² olvide una historia que su contrato conserva: el fallo es atribuirle una adaptación temporal no implementada o usarlo como si respondiera esa otra pregunta.

initialized sólo significa que llegaron cinco datos. No expresa muestra efectiva, cobertura temporal, error de rango ni precisión de un percentil de cola. Con cinco observaciones, el marcador central se inicializa con el tercer orden estadístico para todos los p; es parte del arranque del algoritmo, no una certificación de que el percentil 85 esté bien identificado.

P80 tampoco significa 80 % de aciertos en trading. Un percentil marginal de |OFI| describe magnitud relativa en una población; no es P(retorno neto positivo | OFI, estado, acción). Convertirlo en umbral de entrada requiere una hipótesis predictiva y evaluación causal separadas.

### 5.2 Conectividad y criterio de cierre

La búsqueda de P2Quantile, AdaptiveQuantileEngine y los tres getters dinámicos encontró definiciones, exports y tests, no uso operativo. Los campos con nombres similares en QuantumConfig son genes/atómicos independientes: compartir una palabra no crea una conexión. No se afirma que este auxiliar esté bloqueando inteligencia en el motor actual.

Antes de conectarlo, definir el estimando: distribución acumulada, ventana por eventos, ventana por tiempo o distribución con olvido continuo. Separar cobertura/estado de calidad del valor numérico; registrar política de p, piso y fallback, y persistir o reinicializar con un contrato explícito.

El comentario corregido describe lo que existe. No se reemplaza el histórico por una ventana arbitraria ni se conecta un estimador recién reparado a la ejecución sin validar su cometido.

## 6. Fuentes primarias e integración científica T34

### 6.1 Qué fundamenta P² y qué no

El [artículo original de Jain y Chlamtac](https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf) presenta P² como un estimador heurístico de almacenamiento fijo basado en cinco marcadores y ajustes parabólicos, con fallback lineal para conservar su orden. Se consultaron las secciones de desarrollo, algoritmo y evaluación del PDF, incluida la indicación de errata del ejemplo. La fuente no convierte cinco observaciones en evidencia suficiente ni aporta un reloj físico al código auditado.

Las posiciones objetivo dependen del número de observaciones: por ejemplo, la del cuantil interior es 1+(n−1)p. Los marcadores resumen una población histórica sin almacenar sus muestras. Esta lectura de la fuente orientó la corrección: conservar el algoritmo y sus supuestos, reparar intermediarios numéricos y retirar afirmaciones de adaptación inexistente.

### T34 — Cuantiles condicionados por memoria física y evaluación de umbrales

**Estado: propuesta, no implementada.** Especializa las líneas anteriores sobre causalidad, soporte temporal y validación estadística; no sustituye T01–T33.

Una representación investigable es un cuantil de una distribución empírica causal ponderada por escala:

```text
w_i(t,tau) = exp(-(t-t_i)/tau), t_i <= t
F_hat_t,tau(x) = sum_i w_i * 1[X_i <= x] / sum_i w_i
Q_hat_t,tau(p) = inf{x : F_hat_t,tau(x) >= p}
```

Esta definición, propuesta aquí para el diseño, especifica qué se quiere aproximar; no afirma que el P² actual la calcule ni que se mantenga exactamente en memoria constante. El denominador es masa de observación, no probabilidad de éxito de una acción. Deben decidirse unidades, tratamiento de eventos duplicados/gaps y si la intensidad de eventos debe formar parte de la medida o corregirse.

Para evaluar error de cuantiles puede usarse pérdida asimétrica de cuantiles —pinball— sobre una variable objetivo definida y disponible posteriormente:

```text
rho_p(u) = u * (p - 1[u < 0])
u = observacion_objetivo - cuantil_predicho
```

La utilidad de trading necesita además costes, payoff, selección de acciones y factibilidad. Minimizar esa pérdida no demuestra máximo crecimiento ni reemplaza restricciones de exposición.

**Familia de métodos a contrastar, no lista de módulos conectados:**

- [Quantile Tracking Using a Generalized Exponentially Weighted Average](https://arxiv.org/abs/1901.04681): los pasajes consultados distinguen pesos constantes para seguir una media de pesos dependientes del estado para seguir cuantiles. El teorema consultado es estacionario y usa un límite de paso; no certifica precisión con paso constante en cualquier mercado cambiante.
- [Sequential Quantiles via Hermite Series Density Estimation](https://arxiv.org/abs/1507.05073): el resumen propone estimación secuencial de distribución y cuantiles, incluida una expansión con ponderación exponencial. Se conserva como alternativa para varios percentiles; no se verificó todo el cuerpo ni se atribuyen garantías adicionales.
- [Two maximum entropy based algorithms for running quantiles](https://arxiv.org/abs/1411.2250): el resumen trata streams no estacionarios y uso de memoria mediante histogramas. Es una alternativa para comparar con el seguimiento escalar, no prueba de ventaja en trading.
- [Frugal Streaming for Estimating Quantiles](https://arxiv.org/abs/1407.1121): el resumen presenta estimadores de memoria muy pequeña y análisis bajo streams estocásticos independientes. La independencia no se presume en datos de mercado.
- [Multiplicative Update Methods for Incremental Quantile Estimation](https://doi.org/10.1109/TCYB.2017.2779140): localizado como método relacionado; sólo se recuperó un resumen parcial. No se extrapolan sus requisitos de dominio ni sus tasas de convergencia sin lectura adicional.

**Experimento exigible:** mismo tape causal, memoria y cómputo comparables; distribuciones con cuantiles conocidos, cambios permanentes, colas, masas discretas y cadencia irregular. Medir error de valor y rango, demora tras cambios, sensibilidad a unidades, estabilidad entre percentiles y coste por evento. Si se estiman varios p, comprobar orden de cuantiles y no ocultar cruces con una etiqueta de “tensor”.

La selección de tau/p y el cambio de método deben evaluarse fuera de los datos usados para escogerlos. La propuesta no incorpora un algoritmo nuevo por prestigio ni llama cuántico a un acumulador clásico. Los problemas del milenio no sustituyen la definición del estimando, la identificación de parámetros ni la evidencia de utilidad.

## 7. Topología diagnóstica raíz–cima

| Transición | Contrato necesario | Estado observado |
| --- | --- | --- |
| Evento → estimador | Unidades, reloj, secuencia, calidad | P² sólo recibe valor; no modela tiempo físico |
| Genoma → curvas | Fuente autoritativa y esquema | Getters y sincronización ahora reconstruyen las mismas curvas derivadas |
| Curvas → parámetro servido | Política única y dominio explícito | Reparación local FMT-134; runtime-v1 conserva límites operativos |
| Parámetro → decisión | Evidencia predictiva y factibilidad | Masa/percentil no son probabilidad; quedan límites y gates históricos |
| Decisión → terminal | Cantidad, riesgo, redondeo y payload coherentes | FMT-113 sigue abierto |
| Terminal → evolución | Outcome causal, íntegro y atribuible | Hallazgos previos de feedback/ledger permanecen abiertos |

La tabla describe conexiones inspeccionadas y obligaciones pendientes, no certifica sincronía del grafo vivo. No hubo nueva auditoría completa de ingesta, conectividad Binance, mmap o hardware cuántico en esta ronda.

## 8. Verificación y manifiesto

| Grupo offline de quantum-arena | Tests distintos aprobados |
| --- | ---: |
| horizon_reader_parity | 6 nuevos |
| quantile_numeric_contract | 9 nuevos |
| genome_reader_diagnostics | 3 de X, actualizados a igualdad |
| temporal_band_contract | 11 |
| spectral_interpolation_contract | 7 |
| temporal_spectrum::tests | 17 |
| genome::tests | 6 |
| adaptive_quantiles::tests | 4 |
| Total | 63 |

cargo check --bin god_engine --offline pasó. Persisten tres warnings previos de evolución: latest_ts, mode y RealWfOutcome.trades. No se ejecutó la suite completa del workspace ni un backtest económico. Las esperas del lock de compilación no se interpretan como latencia del trading.

| Archivo | Alcance XI | Líneas finales | SHA-256, prefijo |
| --- | --- | ---: | --- |
| quantum-arena/src/horizon_policy.rs | Nuevo; leído completo | 80 | 7FBCE50ED4D7CD01 |
| quantum-arena/src/lib.rs | Export del módulo | 30 | AE90612C29F0EAFD |
| quantum-arena/src/genome.rs | Getters/sincronización; no lectura íntegra | 3347 | C3EB9752B757C6EE |
| quantum-arena/src/config.rs | Getters; lectura completa ya contada en X | 495 | 2D4D3A2B98B070D6 |
| quantum-arena/src/adaptive_quantiles.rs | Lectura íntegra nueva y reparación | 314 | 8DA0D5A4E27D7D87 |
| quantum-arena/tests/horizon_reader_parity.rs | Nuevo | 118 | D55242D4CAE114CA |
| quantum-arena/tests/quantile_numeric_contract.rs | Nuevo | 119 | FF79981A1DB3CDD9 |
| quantum-arena/tests/genome_reader_diagnostics.rs | Testigos de X convertidos | 61 | 9D4511CA85C4B5BC |

El prefijo de directorio de la tabla es crates/. Los hashes completos, pruebas, estados y archivos protegidos constan en el [artefacto XI](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XI_2026-09-24.json>). El artefacto X conserva hashes de su propio corte; no se reescribe para hacerlos coincidir con estos cambios.

La investigación utilizó Firecrawl Research Index y, al no localizar el P² original en ese índice, el PDF alojado por su autor mediante Firecrawl Scrape. La CLI no estaba disponible; se utilizó el conector sin instalar herramientas y sin subir código. La habilidad influyó en distinguir evidencia primaria de garantías no leídas y en documentar la familia T34 como experimento, no integración activada.

## 9. Pendientes que esta reparación no debe ocultar

La igualdad de fórmulas no basta para un sistema autoevolutivo. Faltan coherencia multivariable al publicar generaciones, objetivos y etiquetas compatibles, identidad causal del crédito, reconciliación de resultados y evaluación no fabricada de candidatos. No se reclasifican como resueltos los hallazgos IX/X ajenos a estos helpers.

El siguiente contrato de integración debe acoplar coeficientes, política y versión en una lectura coherente, sin perder la protección de cantidad final. La migración desde pares de anclas hacia bases temporales más ricas necesita regularización, soporte de datos y equivalencia en dominios ya validados. Debe medirse cuándo un gen cambia el parámetro servido y cuándo ese cambio alcanza una acción; más mutaciones de un gen saturado no constituyen evolución efectiva.

No hubo commit, push, merge, fetch, promoción de genomas, trading ni despliegue. La rama main local y los tests aprobados no certifican el estado remoto ni el binario que está en producción. Se preservó el trabajo concurrente y se amplió la documentación sin eliminar evidencia previa.

### Comprobación final de integridad documental

El JSON se parseó correctamente. Los doce hashes del manifiesto coinciden con los archivos: ocho intervenidos y cuatro consumidores protegidos. Se comprobó que todos los destinos locales enlazados en este informe existen. Los prefijos completos anteriores del atlas, informe maestro y ronda X conservan sus SHA-256 tras normalizar CRLF a LF; sólo se añadió contenido a esos tres documentos. Esta comprobación preserva también el contenido preexistente del informe maestro, sin limpiarlo ni reescribirlo.

git diff --check pasó en quantum-arena y los dos informes rastreados. rustfmt --check pasó en el módulo nuevo y los tres archivos de pruebas creados/actualizados para esta ronda. No se reformatearon íntegramente archivos grandes ajenos al alcance.

## Continuación XII — 2026-09-24

La [auditoría XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XII_2026-09-24.md>) conserva esta evidencia y amplía el circuito de memoria
estadística y observabilidad. Corrige FMT-092, documenta FMT-138–144 y añade
T35: primitiva temporal analítica probada, integración de grafo todavía
propuesta. No modifica las conclusiones históricas de XI ni declara cerrado
el snapshot del genoma, el presupuesto sobre cantidad final o el feedback.

85 tests distintos seleccionados pasan y ambos binarios verifican compilación
sin ejecutarse. Cobertura acreditada: 101/289 Rust preexistentes; 188 pendientes.
Los detalles de compatibilidad, métodos fallibles, alcance de las pruebas
de fuente y hashes están en el [artefacto XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XII_2026-09-24.json>). Sin trading, publicación Git
ni modificación de genomas activos.
