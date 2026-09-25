# Auditoría de fundamentos científicos XXXII — identidad de decisión, aprendizaje y vetos
Fecha: 2026-09-25. Snapshot local; continuación aditiva de XXXI. No es una certificación integral ni una auditoría de rentabilidad.

## 1. Dictamen ejecutivo y estado de resolución

La reparación principal conecta una deliberación con sus propios votos y con la generación de la posición que puede recibir crédito al cerrar. Antes se ejecutaban dos evaluaciones, con win rates distintos, se guardaba la primera antes de la admisión final y el cierre consumía un array reutilizable sin identidad de generación. Este defecto permitía aprender de un candidato no ejecutado o atribuirle el resultado de otro ocupante del slot.

Se añaden siete identificadores, FMT-246 a FMT-252, sin renumerar ni sustituir la matriz histórica de 305 puntos. FMT-243 avanza parcialmente: se corrige la doble evaluación y se captura conjuntamente el contador y los pesos del tracker; sigue abierto el uso de n agregado con wr por activo. Hay reparaciones locales en FMT-246/247/248, pero no se declara resuelto el problema económico completo de atribución.

El hallazgo abierto de mayor urgencia de esta ronda es FMT-249: una cascada puede ser consumida por las features antes de que el consejo la observe. El canal es global, carece de símbolo y timestamp y colapsa eventos por máximo. No basta con haber reparado el parámetro del veto en XXXI: el dato tiene que llegar al decisor con significado, identidad y vigencia.

Resultado de la selección ejecutada: 66 pases distintos, de los cuales 60 son funcionales/compatibilidad y 6 reproducen limitaciones ABIERTAS. Se incorporan 23 tests: 21 funcionales y 2 diagnósticos abiertos. Ocho aserciones se observaron fallar antes de su reparación: cuatro del consejo, una de atribución de cierre y tres del agregado de flujo. No se cuentan recompilaciones, repeticiones ni errores de compilación como bugs reproducidos.

Cobertura acumulada: 151 de 289 Rust preexistentes leídos por completo, 138 pendientes. Inventario histórico comprobado: 1.119 archivos versionados y 24 manifiestos Cargo. Estos números NO significan que los 151 archivos estén libres de defectos ni que se haya revisado exhaustivamente el contenido no Rust.

## 2. Paradigma de grafo vivo y topología diagnóstica

~~~text
RAÍCES OBSERVABLES
  mercado/activo/tiempo ─────────────► snapshot del consejo
  historial de desempeño ─────────► n y pesos bajo un mismo read-lock
                                      │
NODOS DE TRANSFORMACIÓN                ▼
  agentes por rol ───────────────► una evaluación / agente
  validación de salida                │ opiniones completas + wr efectivo
                                      ▼
NODO DE DECISIÓN                  CouncilDecisionTrace
  aprobación/dirección/vetos ─────┬────┘
  ML + capital + orden           │
  publicación local válida ─────▼
                          vínculo símbolo/coin/slot/generación/lado
                                      │ votos congelados de ESA evaluación
NODO TERMINAL LOCAL                   ▼
  cierre elegible + identidad ──► consumir vínculo una vez
        ├─ sin vínculo coherente: cerrar según ruta existente, NO atribuir al consejo
        └─ con vínculo: crédito orientado por lado y retorno neto local
                                      │
DEUDA ECONÓMICA ABIERTA                ▼
  intención → orden → fills → fees/funding → settlement → outcome versionado
~~~

La flecha terminal sigue terminando en una estimación local de cierre en la ruta heredada, no en una liquidación reconciliada. El vínculo añadido es una protección de atribución del consejo; no convierte una estimación en cash-flow confirmado. Simulación y ExchangeLocalEstimate conservan la separación de contexto de XXIX.

Un grafo científicamente útil necesita identidad y semántica en las aristas. Coincidir en la longitud del vector, en la posición de un nodo dentro de una lista o en una etiqueta de horizonte no demuestra que dos observaciones correspondan al mismo evento. El orden de llegada de dos activos tampoco debe decidir a cuál se imputa un shock global.

### 2.1 Matriz consolidada de esta adenda

| ID | Prioridad y exposición | Estado al cierre de la ronda | Contrato afectado |
|---|---|---|---|
| FMT-243, heredado | P1, consejo conectado al core | Doble evaluación reparada; población estadística abierta | Misma evaluación para decidir y atribuir; wr y n compatibles |
| FMT-246 | P1 condicional a topología/salida inválida | Validación e identidad local reparadas | Un rol, un voto; pesos por identidad y no por orden |
| FMT-247 | P1, consumidor del core | Vínculo local reparado con límites de concurrencia/settlement | No aprender de candidatos rechazados ni de ocupantes anteriores |
| FMT-248 | P2, helper sin caller operativo localizado | Aritmética reparada | Invariancia de unidades y estabilidad del cociente |
| FMT-249 | P1, productor y consumidores operativos | ABIERTO; documentación numérica corregida | Evidencia de cascada observable, con alcance y tiempo |
| FMT-250 | P2, sincronizador conectado | ABIERTO por inspección de ruta | Offset con calidad, frescura y reloj adecuado |
| FMT-251 | P2 latente, sin caller operativo localizado | ABIERTO | Promoción respaldada por evidencia y transición real |
| FMT-252 | P3, helper sin caller operativo localizado | Descripción corregida; aritmética extrema ABIERTA | Nombre/garantía honestos y resta sin wrap |

## 3. FMT-243 — una sola evaluación; población aún incompatible

**Evidencia.** [deliberar_traced](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:935>) y [consumidor del core](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:4536>). XXXI ya documentó el contraste entre extracción con wr crudo y deliberación con shrinkage. Esta adenda no lo cuenta como un hallazgo nuevo.

**Fallo anterior.** Con wr=0,1 y cero outcomes, el probe de extracción devolvía 0,1, mientras la deliberación usaba 0,5. Guardar la extracción y posteriormente evaluar el resultado no representa el estado que produjo la decisión. Con un agente stateful, la segunda llamada también puede observar un estado interno diferente aunque ambos argumentos coincidieran. El trabajo de evaluar once agentes se duplicaba, incluyendo sus cálculos y construcción de justificaciones.

**Cambio.** CouncilDecisionTrace devuelve el consenso, todas las opiniones, los votos elegibles para aprendizaje indexados por rol, el wr efectivo y el contador agregado observado. El core llama una vez a deliberar_traced. Los wrappers históricos deliberar y deliberar_with_weights conservan sus firmas; extract_senior_signals conserva la suya pero inicia una nueva deliberación y queda documentado que no debe combinarse con otra llamada para reconstruir un mismo trace.

El contador y los pesos adaptativos se leen bajo una única adquisición del RwLock. El lock se libera antes de invocar agentes, para no mantenerlo alrededor de callbacks. Si el lock está poisoned se devuelve rechazo de integridad; no se interpreta el estado potencialmente inconsistente como un cold start sano. No se implementa recuperación automática del tracker ni se garantiza tiempo máximo de espera.

**Significado del cálculo.** La expresión vigente es:

~~~text
wr_efectivo = (n_global · wr_activo + 0,5 · k) / (n_global + k)
~~~

La mezcla reduce el extremo observado hacia 0,5 cuando n es pequeño. Una lectura de posterior Beta requiere ensayos y proporción de la misma población; k=8 sería una concentración simétrica Beta(4,4) en ese modelo. Aquí un activo puede alterar el n utilizado para regularizar el wr de otro. Corregir la doble lectura del lock no convierte esa mezcla en inferencia calibrada. El diagnóstico que inserta diez outcomes agregados y cambia 0,5 a 5/18 sigue reproduciendo la incompatibilidad.

**Pruebas.** Un agente contador es llamado exactamente una vez; sus votos coinciden con la opinión evaluada y el wr efectivo. Un payload NaN se rechaza antes de callbacks. El trace conserva las once opiniones cuando un veto de cascada rechaza la decisión. Se comprueba explícitamente el tracker poisoned. El test OPEN de población permanece abierto; únicamente se actualizó su aserción secundaria de extracción para reflejar la reparación real, sin borrar la evidencia pendiente.

**Residual y cierre pendiente.** Falta un suficiente estadístico por población compatible o un modelo de pooling explícito y validado. No se congelan en almacenamiento durable todas las features, versiones de modelo/genoma, política, identidad de decisión y propensiones de admisión. Tampoco se ha medido dependencia temporal, sesgo de selección o calibración fuera de muestra.

## 4. FMT-246 — el orden de los agentes alteraba la identidad del aprendizaje

**Localización.** [validación de roles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:953>) y [evaluación e indexación](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/metacortex-engine/src/consejo_seniors.rs:987>).

**Mecanismo.** agents es un Vec público. El tracker y las máscaras utilizan posiciones canónicas de SeniorRole, pero el bucle original aplicaba multiplicadores por enumerate(). Reordenar la misma colección cambiaba qué peso recibía cada agente. La extracción guardaba señales por orden, por lo que la máscara de moduladores podía actuar sobre otro rol. Una colección con más de once agentes también podía escribir fuera del array de extracción. Una duplicación del mismo rol añadía capacidad como si existiera una observación adicional.

**Reproducciones rojas.** Invertir el orden de los once agentes alteró final_signal bajo un vector externo de pesos no uniforme; también invirtió posiciones del array de señales. Añadir un segundo Microestructura no produjo un rechazo de integridad. Un agente sintético con signal_direction=100 fue aprobado después de clamping, a pesar del contrato [-1,1].

**Reparación.** Se verifica unicidad de roles antes de callbacks. Se exige que el rol de la opinión coincida con el rol declarado por el agente. Multiplicadores externos, pesos aprendidos y señales se indexan por rol. Se validan señal [-1,1], confianza [0,1] y peso finito no negativo. Peso cero es válido: expresa capacidad nula, no corrupción. Con confianza o peso cero, el voto elegible para aprendizaje es cero; la opinión cruda se mantiene en el trace.

También se comprueba finitud después de multiplicar el peso y después de acumular señal/capacidad. De otro modo, inputs finitos grandes podrían producir infinito y esconder un estado mal definido detrás de un clamp. No se añade un techo arbitrario nuevo al peso para ocultar el overflow.

**Por qué el rechazo tiene sentido.** No se veta un estado de mercado ni una escala temporal: se rechaza una topología ambigua o una salida que viola su contrato numérico. Un duplicado no puede convertirse en evidencia adicional sólo por estar repetido. Una subcolección con roles únicos sigue siendo aceptable como topología; no se exige exactamente once agentes ni se introduce un nuevo quorum estadístico.

**Pruebas y límites.** Se prueban pesos externos y aprendidos con orden inverso, salida inválida, rol incoherente, overflow, duplicados y peso cero. Estos casos no acreditan independencia entre Microestructura, Metacognitivo y Teleonomia. FMT-244 permanece: transformaciones de una misma raíz pueden dar apariencia de consenso total. Tampoco se resuelven pánicos dentro de callbacks, identidad versionada de implementaciones ni un registro dinámico ilimitado de roles.

## 5. FMT-247 — persistencia de votos antes de la admisión y sin identidad de posición

**Evidencia.** [CouncilEntryEvidence](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:68>), [consumo en cierre](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1590>) y [publicación y vinculación de apertura](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:4742>).

**Fallo anterior.** El core sobrescribía last_senior_signals[coin][slot] antes de comprobar el consejo, el gate ML y el margen. El array sobrevivía al cierre, rollback, adopción o reapertura. El cierre lo trataba como evidencia del trade actual. En el fixture, una posición abierta directamente y un voto legacy +1 bastaban para insertar un outcome; no existía decisión del consejo asociada a esa posición.

**Impacto lógico.** Se mezclaban tres hechos distintos: “un candidato fue evaluado”, “una posición fue publicada” y “un resultado pertenece a esa decisión”. Eso puede invertir el crédito del aprendizaje, reforzar votos nunca admitidos o rellenar n con cierres sin origen decisional. Que el PnL sea positivo no repara una asociación causal equivocada.

**Cambio implementado.** Se añade un vínculo local por coin/slot con símbolo, generación, lado y votos de la única deliberación. Se escribe después de aprobar las demás condiciones y de obtener éxito de open_with_tau_and_fee. El array legacy queda como espejo de diagnóstico; ya no es la fuente para entrenar el consejo.

La generación esperada se obtiene antes de la transición como g+1. Si se intercala otra transición y no coincide con la observada al publicar, se descarta la atribución. Al cerrar se consume el Option una vez, incluso si luego el contexto económico rechaza aprender. Se contrastan símbolo, generación, lado, precio/cantidad devueltos por el cierre y el avance de generación observado. Una discrepancia no reconstruye ni inventa evidencia.

**Apertura fallida.** El retorno bool antes ignorado ahora se respeta. Se validan precio/cantidad finitos positivos antes del cargo local. Se mantiene reserva/fee antes de publicar la posición; si la API devuelve false, se compensan únicamente los importes de ese intento. No se publica orden, contador de éxito ni vínculo de aprendizaje para el intento rechazado. Esta compensación no constituye un ledger transaccional de reservas concurrentes.

**Frontera del veto.** Si falta vínculo, el cierre local conserva la ruta defensiva existente y las reglas OutcomeContext. Se omite exclusivamente el crédito al consejo y aumenta diag_unattributed_council_closes para cierres económicamente elegibles sin atribución. No se convierte “no sé qué decisión originó esta posición” en “debo mantenerla abierta”.

**Pruebas.** La prueba roja de posición sin vínculo pasa de un outcome indebido a cero. Las pruebas nuevas cubren consumo único, reutilización de slot, generación/lado/símbolo incorrectos, entrada no confirmada y mutación del espejo legacy. El corto ganador de XXXI conserva su prueba de orientación, ahora con vínculo explícito del fixture. Los tests construyen evidencia sintética dentro de su propia arena; no afirman haber generado fills reales.

**Residual importante.** El vínculo vive en memoria del core y es público como otros estados actuales; no es una firma ni una frontera contra callers que lo falsifiquen. No se serializa el trace completo, ni hay vínculo durable a order_id/fill_id/genoma. La identidad por símbolo mitiga reutilización de índices, pero no equivale a versión inmutable del registro. El chequeo de generación es una contención, no una prueba de linealizabilidad de todas las lecturas/cierres: la ruta económica aún usa campos leídos separadamente y el cierre no es una operación compare-and-close sobre un snapshot suministrado por el caller. No se realizó prueba concurrente de extremo a extremo.

FMT-225 sigue abierto: ExchangeLocalEstimate con entrada confirmada puede aprender de una estimación local de salida; falta settlement real. Los demás consumidores de aprendizaje no pasan automáticamente a usar CouncilEntryEvidence. El residual online/Kalman, PPO, mmap y datasets conservan sus deudas previas.

## 6. FMT-248 — el flujo cambiaba con las unidades y podía desbordarse

**Evidencia.** [aggregate_order_flow](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/order_flow_aggregator.rs:11>) y [regresiones numéricas](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/spectral_admission_diagnostics.rs:5>).

**Fórmulas y significado.** Para volúmenes no negativos B y S, r=B/(B+S) es la fracción compradora y Δ=B−S es el desequilibrio absoluto. r no tiene unidades; Δ sí conserva las unidades de volumen. Por construcción, r(aB,aS)=r(B,S) para a>0, siempre que los inputs y resultados sean representables.

**Defectos reproducidos.** Con B=S=f64::MAX, la suma anterior era infinito y r daba 0 en lugar de 0,5: equilibrio extremo se convertía en sesgo vendedor. Con B=3e−20 y S=1e−20, el filtro B+S≤1e−12 devolvía neutralidad 0,5 y Δ=0, destruyendo un desequilibrio real. La elección de unidad podía por sí sola alterar la señal.

**Reparación.** Se calcula m=max(B,S); si m=0, la convención heredada es (0,5,0). En otro caso, r=(B/m)/[(B/m)+(S/m)]. Uno de los términos escalados es 1 y el denominador no desborda. No se añade un umbral de liquidez; si se necesita tal filtro, debe expresarse separadamente con unidades y evidencia. Se conserva Δ=B−S.

**Exposición y límites.** La búsqueda de callers en crates/src sólo encontró los tests y la definición del helper. No se afirma que el arreglo cambie operaciones actuales. Las entradas negativas/no finitas aún se sustituyen por cero por compatibilidad: dato inválido sigue pudiendo confundirse con ausencia de volumen. Ese contrato merece una API checked separada; no se cablea ahora el helper al motor operativo. Se retira la promesa documental de latencia sistémica <1,1µs: no había benchmark de esta ronda que la sostuviera.

Tres aserciones rojas quedan verdes: grandes equilibrados, pequeños no nulos e invariancia de unidades. También pasan los dos tests legacy de equilibrio/sesgo y sanitización.

## 7. FMT-249 — el veto de cascadas puede observar un buffer ya vaciado

**Ruta observada.** El [parser operativo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:2902>) calcula notional desde p y q y llama a bump. El [canal global](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/liquidation_feed.rs:28>) guarda el máximo pendiente en un AtomicF64. El core consume con take_pending en la ruta macro, tanto en [eventos depth](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:754>) como en [actualización macro por tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1021>). Más adelante, [el payload del consejo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:4516>) usa peek_pending.

**Reproducción mínima.** bump(0,95) → take_pending()=0,95 → peek_pending()=0. Ese orden representa el camino donde macro absorbe el evento antes de deliberar y no llega otro evento entre ambas llamadas. El test usa el static de su proceso de pruebas, no la memoria del motor operativo. No demuestra que toda deliberación vea cero: si otro productor interviene puede ver otro evento, precisamente otra dependencia del interleaving.

**Pérdida multiactivo.** bump sólo recibe severidad. No conserva símbolo, dirección, timestamp, secuencia ni origen. El primer activo procesado puede consumir el evento de otro, mientras los restantes no lo observan. Si la intención es modelar contagio global, distribuirlo mediante “el primero gana” tampoco constituye un modelo cross-asset. max no preserva conteo, volumen agregado ni distribución temporal: diez shocks menores se vuelven indistinguibles de uno del mismo máximo.

**Normalización.** La función actual es s=clamp[ln(N/1USD)/ln(10^6),0,1]. Da 2/3 para 10.000 USD y 5/6 para 100.000 USD, no los valores 0,33/0,67 que indicaba su comentario. Se corrige la descripción, no el cálculo ni el umbral. Para s>0,85 el notional del evento debe superar 10^5,1≈125.893 USD. No es un percentil empírico, una probabilidad ni una calibración por profundidad/capitalización del activo. Cambiar sin estudio el logaritmo o su referencia alteraría sustancialmente qué eventos vetan.

**Relación con XXXI.** FMT-239 reparó activación y protección del threshold configurado dentro del consejo. Eso sigue siendo correcto para un payload válido. FMT-249 revela que la evidencia del payload puede haber desaparecido o pertenecer al contexto equivocado. No se marca FMT-239 como una reparación económica integral.

**Diseño requerido y criterio de cierre.** Separar evento consumible por features de estado observable por decisiones. Conservar evento_id, símbolo/alcance, tiempo de mercado/recepción y valor con unidades; definir explícitamente reducción y expiración. Un snapshot compartido por una decisión debe alimentar tanto las features como su veto sin depender de quién lea primero. Modelar por separado señal local y contagio global. Probar dos activos con órdenes de llegada invertidos, burst de eventos, duplicados, retraso, reordenamiento y ausencia de nuevos datos.

No se añade un TTL arbitrario ni se mantiene indefinidamente el último máximo: ambas decisiones podrían sustituir un falso permiso por un bloqueo permanente. La reparación funcional de esta ruta queda ABIERTA.

## 8. FMT-250 — offset de reloj sin calidad ni vigencia explícitas

**Evidencia.** [start_ntp_synchronizer](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/ntp.rs:9>); uso en [timestamp firmado](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:117>) y en [estimación de latencia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/data-pipeline/src/ws_client.rs:414>). Se verificó también el caller del sincronizador en el host; no se hicieron peticiones de tiempo al exchange.

**Qué calcula.** Se observan t0 y t1 mediante SystemTime y un server_time s. El estimador es offset=s−[t0+(t1−t0)/2], usando saturating_sub para el tramo local. Esto aproxima el offset con el punto medio del round-trip; requiere interpretar el timestamp del servidor y la demora de transporte. Un offset escalar no separa desfase del reloj y asimetría de red.

**Defectos de contrato.** El tiempo transcurrido se obtiene de un reloj de pared susceptible de ajuste. Si retrocede, saturating_sub oculta el salto fijando RTT a cero en lugar de invalidar la muestra. Cualquier respuesta válida de fetch_server_time reemplaza el offset sin calidad/RTT asociado, incertidumbre o edad. En error se imprime un aviso, pero el anterior offset conserva vigencia implícita. Los consumidores no reciben “sin sincronizar”, “estimación caducada” o un intervalo de error.

**Impacto condicionado.** Un ajuste incorrecto puede afectar timestamps enviados o la estimación de frescura usada por filtros de latencia. No se observó un rechazo Binance real ni se midió el jitter de esta cuenta; el hallazgo es estructural y operativo por el cableado localizado. No se afirma que todo rechazo de tiempo proceda de aquí.

**Cierre pendiente.** Medir duración con reloj monotónico, detectar discontinuidades del reloj de pared, conservar instante/calidad de la muestra y propagar validez. Definir presupuesto de incertidumbre compatible con ventanas del protocolo y objetivos de latencia; no inventar un umbral universal en milisegundos. Probar retraso asimétrico, fallo prolongado, saltos y reordenamiento de muestras con reloj inyectable. No se modifica configuración temporal operativa en esta ronda.

## 9. FMT-251 — “estasis” y hot-swap no prueban promoción a producción

**Evidencia.** [evaluate_estasis](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/hot_swap.rs:18>) y [execute_swap](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/hot_swap.rs:30>). Es un tipo diferente del HotSwapController de metacortex; no deben confundirse por compartir nombre.

**Problema matemático.** La admisión compara wr>0,55 y EV>threshold. No recibe número de muestras, incertidumbre, horizonte del retorno, costes, estabilidad, límites de cartera, evaluación fuera de muestra o identidad de candidato. Un wr bajo puede coexistir con expectativa positiva; uno alto no demuestra utilidad neta ni seguridad. Incluso el dominio del wr no está acotado a [0,1], sólo se comprueba finitud. El threshold del constructor no se valida y no expresa unidades.

**Problema funcional.** execute_swap publica un booleano y un mensaje de transición a MAINNET, pero el comentario del cuerpo describe el cambio de conexión como trabajo futuro. No hay validación de precondiciones dentro de execute_swap ni transición transaccional de conexiones, credenciales, órdenes, riesgo y rollback.

**Exposición.** La búsqueda de evaluate_estasis/execute_swap encontró definición y test local, no uso operativo en el host. Por tanto es deuda latente de API y de afirmaciones, no evidencia de una promoción real insegura ya ejecutada. No se llamó execute_swap ni se cambiaron credenciales o entorno.

**Cierre.** Separar diagnóstico de candidato, autorización explícita de despliegue y ejecución del despliegue. El evaluador debe explicar muestra, objetivo neto, incertidumbre y límites; un booleano local no puede certificarlos. La promoción requiere un protocolo de estado con rollback y reconciliación. No se conecta este helper a producción como “mejora” automática.

## 10. FMT-252 — un cociente no es un acelerador SPSC

**Evidencia.** [descripción y funciones](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/latency_accelerator.rs:1>) y [diagnóstico del timestamp extremo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/tests/spectral_admission_diagnostics.rs:45>).

El módulo no implementa cola, atomics, acquire/release, IPC ni separación entre motores. Contiene un cociente de latencias con clamp y una resta de timestamps. La documentación anterior prometía IPC subnanosegundo y hablaba de comunicación scalping/swing sin implementación que lo respaldase. Se sustituye por una descripción de helpers aritméticos; se conserva la API para no romper usos externos no visibles.

La resta convierte u64 a i64 antes de operar. Para local=0 y server=u64::MAX, el resultado actual es −1, cuando la diferencia matemática es positiva y su saturación al intervalo vigente sería +5000. El diagnóstico OPEN conserva esa reproducción. Es una prueba del dominio de la API, no un argumento de que los timestamps Unix actuales alcancen ese extremo.

El cociente devuelve neutralidad 1 ante ciertos inputs inválidos y .01 para raw_latency negativo; además satura magnitudes válidas en [.01,100]. Esas convenciones no aceleran el sistema y no se convierten aquí en política operacional. No se localizaron callers de producción de estos helpers. Quedan pendientes resta en dominio amplio, validación explícita y contrato separado entre métrica y decisión; no se cambia la política de latencia por un nombre atractivo.

## 11. Auditoría de vetos, rechazos y arbitrariedades

| Condición | Naturaleza | Tratamiento en esta ronda |
|---|---|---|
| Payload/política no finitos o fuera de dominio | Integridad | Rechazo explícito previo; se preserva XXXI |
| Rol duplicado, opinión incompatible o overflow | Integridad/topología | Rechazo explícito nuevo; no calibración de mercado |
| Peso cero o confianza cero | Abstención/capacidad nula | Válidos; no se fabrican votos de aprendizaje |
| Falta de vínculo decisional en cierre | Evidencia insuficiente para atribuir | Omitir crédito al consejo; no bloquear por ello la salida |
| Cascada por encima del parámetro | Política de riesgo | Consejo conserva veto; feed/procedencia abiertos en FMT-249 |
| Rechazo ML y presupuesto de margen | Decisión económica/ejecución | Preservados; no se recalibran por decreto |
| Kill-switch global | Política de seguridad | FMT-232 sigue reproduciendo supresión de propuesta defensiva |
| WR>0,55 y EV>threshold para “estasis” | Heurística de promoción | Deuda latente, no habilitada |
| Volumen total≤1e−12 en ratio | Umbral dependiente de unidades | Eliminado del cociente; no era condición matemática necesaria |

No todos los estados discretos son regímenes de mercado arbitrarios. Una entrada confirmada, una generación de slot o un dato inválido son hechos contractuales. El enfoque temporal continuo no elimina esos estados ni permite usar datos incoherentes. Tampoco justifica convertir vetos de integridad en una opinión sobreescribible por mayoría.

Persisten umbrales económicos legacy, pesos heurísticos y clamps temporales documentados en XXXI. Esta ronda no demuestra que estén óptimamente calibrados; evita sustituirlos por otros números sin ensayo causal, coste y fuera de muestra.

## 12. Fundamento científico y evolución sin inflación terminológica

La ampliación teórica útil de esta ronda es contractual y matemática: identidad, una evaluación reproducible, invariancia de unidades, diferencia entre evento y estado, y separación entre estimador y evidencia económica. No se implementan ecuaciones de problemas del milenio ni se afirma ventaja cuántica. La complejidad de una ecuación no suple variable observable, modelo de medición, hipótesis ni una comparación falsable.

Para evolucionar el sistema como función multivariante de escala, activo y estado, faltan al menos estos contratos:

1. Representación temporal: distinguir τ continuo como argumento de una función del soporte finito que los datos permiten estimar. FMT-245 mantiene el colapso de extremos del consejo a 30s–12h; no se soluciona extendiendo constantes a 1ns–100 años.
2. Representación de activo: usar identidad y procedencia en la raíz, no sólo índices de arrays reutilizables. El vínculo de esta ronda mejora una arista; el canal de liquidaciones demuestra otra arista todavía global y ambigua.
3. Observación versus intervención: un veto y una ejecución seleccionan qué resultados se observan. El tracker aprende acuerdo con operaciones seleccionadas, no resultados contrafactuales de las rechazadas. No puede interpretarse sin más como precisión universal del agente.
4. Estadística consistente: proporción y número de ensayos deben describir la misma población; si se desea compartir información entre activos/escalas, declarar el pooling y validar cuánto se comparte.
5. Hipótesis medibles: “ortogonal”, “cuántico”, “resonante”, “Bayesiano” y “epigenético” deben señalar una operación verificable y sus condiciones. No son garantías por sí mismas.
6. Cierre del lazo: genoma/modelo/política versionados → features as-of → decisión → intención → ejecución → cash-flow reconciliado → actualización reproducible. Sin ese lazo, un backtest exitoso no acredita impacto equivalente en demo o producción.

Se releyeron online_learning, fases_autonomous, hot_swap_controller y lib de metacortex. Sus problemas ya estaban registrados en II/III; no se reenumeran como descubrimientos nuevos. Entre ellos siguen relevantes FMT-039 (innovación escalar replicada con denominadores por canal, learning_rate sin efecto en esa actualización), generación/compilación que no demuestra integración o hot-swap y validaciones no globales en la máquina de fases. La ruta online del core vuelve a usar features actuales al cierre; esta ronda no la reescribe ni convierte su salida diagnóstica en probabilidad servida.

No se ha realizado una revisión bibliográfica nueva ni un barrido completo de todas las teorías del proyecto. Las derivaciones numéricas anteriores se contrastan con el código y los tests indicados; no se presentan como resultados empíricos del mercado.

## 13. Recorrido por los ocho módulos del informe maestro

| Módulo | Evidencia/avance de XXXII | Frontera no certificada |
|---|---|---|
| 1. Ingestión, parsers, L2 y normalización | FMT-248; productor de liquidaciones localizado | Parser completo, pérdidas/reordenamiento y validez de datos de origen |
| 2. Inferencia de IA y señales | Una evaluación por agente; validación de outputs | Calibración ML y dependencia entre modelos |
| 3. Estrategia multiasset y horizontes | Identidad por símbolo/slot/generación; FMT-249 | Espectro funcional completo, pooling y contagio |
| 4. Ejecución y conectividad | Retorno de apertura respetado; revisión NTP | Ledger de fills, reservas atómicas y reloj con incertidumbre |
| 5. Riesgo y genomas | No atribuir outcomes a candidatos no admitidos | Calibración de vetos, transmisión causal de genes y settlement |
| 6. Estado, memoria y telemetría | Consumo único de vínculo; snapshot del tracker | mmap, concurrencia integral, persistencia y latencia medida |
| 7. Orquestación y confluencia | Roles estables y sin duplicación | FMT-244, raíz de señales compartida; ninguna certificación cuántica |
| 8. Backtesting, auditoría y gobernanza | Regresiones sintéticas y límites explícitos | Suite completa, paridad live/backtest, promoción y rentabilidad |

## 14. Inventario de lectura y límites de cobertura

Nuevas lecturas completas de archivos Rust preexistentes, sin contarlos por segunda vez:

| Archivo | Líneas base | Líneas finales | Resultado |
|---|---:|---:|---|
| execution-engine/src/ntp.rs | 42 | 42 | FMT-250, sin cambio operativo |
| execution-engine/src/hot_swap.rs | 65 | 65 | FMT-251 latente |
| god-engine-core/src/order_flow_aggregator.rs | 54 | 59 | FMT-248 reparado localmente |
| god-engine-core/src/latency_accelerator.rs | 56 | 56 | FMT-252 descripción corregida; defecto extremo abierto |
| god-engine-core/src/liquidation_feed.rs | 61 | 62 | FMT-249; comentario de normalización corregido |

Total añadido: 5, acumulado 151/289, pendientes 138. Las relecturas de metacortex no incrementan el contador; constaban en II/III y XXXI. Core/lib, position, executor, ws_client y host se inspeccionaron por rutas y anclas, sin declarar lectura completa nueva. Los archivos nuevos de tests no inflan el denominador histórico. Tampoco se confunde buscar símbolos en todo el workspace con analizar semánticamente todos sus archivos.

## 15. Verificación reproducible y significado de los pases

~~~text
cargo test --offline -j 1 -p metacortex-engine --test council_evidence_contract -- --test-threads=1
cargo test --offline -j 1 -p metacortex-engine --lib consejo_seniors::tests -- --test-threads=1
cargo test --offline -j 1 -p god-engine-core --test close_outcome_contract --test spectral_admission_diagnostics -- --test-threads=1
cargo test --offline -j 1 -p god-engine-core --lib order_flow_aggregator::tests -- --test-threads=1
cargo check --offline -j 1 --bin god_engine --bin feature_exporter --bin train_forest --bin train_dark_alpha
~~~

| Selección | Funcionales/compatibilidad | Diagnósticos OPEN | Total |
|---|---:|---:|---:|
| council_evidence_contract | 31 | 3 | 34 |
| consejo_seniors::tests | 8 | 0 | 8 |
| close_outcome_contract | 15 | 1 | 16 |
| spectral_admission_diagnostics | 4 | 2 | 6 |
| order_flow_aggregator::tests | 2 | 0 | 2 |
| Total único | 60 | 6 | 66 |

Los seis OPEN son: población global/per-activo, consenso dependiente, saturación temporal, kill-switch defensivo, pérdida del evento de liquidación tras consumo y wrap de timestamp en helper. Su pase significa que la limitación sigue presente. Cero ignorados en las selecciones; no se ejecutó toda la suite del workspace.

Los ocho rojo→verde se observaron antes de editar su implementación, no son inferidos a posteriori. Las regresiones de apertura/cierre usan fixtures aislados; la ruta completa desde tape hasta nueva entrada con modelo validado no se reprodujo de extremo a extremo. El respeto del bool de apertura y el punto de persistencia se verificaron por inspección y compilación, con tests dinámicos de consumo/atribución.

Cargo check de los cuatro binarios pasa con los tres warnings previos latest_ts, mode y trades de evolution-engine. No se ejecutó cargo fix. No hay benchmark de p50/p99/p999 ni comparación de PnL. Quitar una segunda evaluación ahorra trabajo estructural, pero no autoriza prometer nanosegundos: persisten allocations, strings, RwLock y scheduling del host.

## 16. Preservación, Git y seguridad de operación

Se agregan adendas al atlas, al informe maestro y a XXXI. Sus prefijos anteriores se verifican normalizando CRLF→LF, sin reescribir conclusiones históricas. El artefacto JSON mantiene hashes, cobertura, pruebas, estados y referencias de esta ronda.

Los 41 hashes de modelos coinciden con el snapshot de XXXI y con la comprobación final de esta ronda. No se editan deliberadamente genomas activos, credenciales ni configuración de cuenta. Rama local main y HEAD 59a76de4. El árbol ya tenía numerosos cambios ajenos; no se hizo commit, push, merge, fetch, reset ni checkout. No se afirma que main remoto incluya estas reparaciones ni que todas las ramas estén integradas.

No hubo peticiones de cuenta, órdenes, entrenamiento/promoción operativos, terminación/reinicio del motor ni build del ejecutable operativo. Sí hay aprendizaje sintético en los tests. Los temporales que elimina el fixture de cierre son exclusivamente sus directorios únicos bajo Temp validado; no se borran cachés compartidas, modelos, datasets o traumas operativos.

## 17. Hoja de ruta sistémica de rehabilitación 1-a-1

1. FMT-249: especificar evento/estado de liquidación por activo y alcance global, y un snapshot común de decisión. Es prioritario porque un veto correctamente calculado no protege si su evidencia se vacía antes.
2. FMT-225/247: llevar la identidad local a intención/orden/fill/outcome durable, con reservas y cierre condicional por generación; no ampliar aprendizaje con estimaciones no reconciliadas.
3. FMT-243: corregir población de n/wr, explicitar pooling por activo/escala y persistir contexto/política/modelo en el trace.
4. FMT-244/245: diseñar confluencia por procedencia y una representación temporal con soporte/error medibles. No equiparar mayor número de transformaciones con más evidencia independiente.
5. FMT-250: introducir estado de sincronización con edad, incertidumbre y pruebas de saltos. Preservar autoridad de riesgo y condiciones del protocolo.
6. FMT-251/252: separar diagnósticos/helpers de capacidades reales de despliegue o latencia; cerrar dominio numérico sin activar rutas latentes.
7. Continuar el inventario pendiente de 138 Rust y archivos no Rust, conservando evidencia por archivo. No hay declaración de “todo solucionado”.

La mejora acreditada es trazabilidad local y aritmética consistente. La autoevolución multiactivo temporal-espectral sigue necesitando datos causales, arquitectura de estado, validación estadística y ejecución reconciliada; no se certifica por la cantidad de ecuaciones ni por 66 tests en verde.

## Adenda de continuidad XXXIII — 2026-09-25

[Informe XXXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXXIII_2026-09-25.md>) · [Artefacto XXXIII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXXIII_2026-09-25.json>). Esta adenda conserva el estado histórico de la ronda anterior y precisa sus avances posteriores.

FMT-249 recibe reparación local: productor con identidad y ejecución reportada, estado por instancia/símbolo y vista as-of compartida no destructiva. El buffer global deja la ruta operativa; se reproduce antes la contaminación0,95y se verifica después su aislamiento. Se documenta la política de duplicados, atrasados y mismo milisegundo, y se mantiene la capacidad de cierre defensivo frente al nuevo interlock de evidencia futura.

NuevosFMT-253a256: parser/requested-vs-executed reparado en productor operativo, límites de observabilidad/calibración aún abiertos, retroceso de reloj del kernel genérico OPEN y overflow de spread corregido. API legacy y sus diagnósticos se conservan. El documento explica los cálculos, unidades, consecuencias del umbral y semivida, la diferencia de datos entre entrenamiento/servicio y los requisitos de un paradigma espectral verificable.

26tests nuevos; tres aserciones válidas rojo→verde. Selección100pases=91funcionales/compatibilidad+9OPEN, check offline de cuatro binarios correcto. 41modelos preservados; dos nuevas lecturas integrales→153/289Rust,136pendientes. Sin operaciones de cuenta, órdenes, promoción operativa, reinicios o publicación Git. No se declara todo reparado ni se reemplaza la matriz histórica.
