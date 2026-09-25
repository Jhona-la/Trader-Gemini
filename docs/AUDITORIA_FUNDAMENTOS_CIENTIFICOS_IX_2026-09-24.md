# Auditoría de fundamentos científicos IX — Evidencia operativa, autoridad de parada y trazabilidad

Fecha: 2026-09-24. Base local HEAD 59a76de4. Continuación aditiva de I–VIII; se conserva íntegro su historial.

## 1. Dictamen ejecutivo y alcance de la certificación

Esta intervención prioriza la ruta de evolución que **sí está conectada en el código del host**: LiveEvolutionDaemon. Corrige mecanismos de FMT-055 y FMT-056, sin declararlos globalmente cerrados. Añade FMT-130–132 sobre entrega durable de eventos, semántica del registro y construcción de observaciones económicas. Los tres nuevos hallazgos permanecen abiertos; no se confunde documentarlos con repararlos.

Resultado local: 22 pruebas nuevas, cuatro con reproducción rojo→verde; 47 tests distintos aprobados al incluir las regresiones CMA anteriores. cargo check del host pasó. Son pruebas de cálculo y control, no una campaña económica ni una medición en un proceso de trading desplegado.

Se leyó completo evolution_ledger.rs, adicional a la cobertura previa, y se releyó online_daemon.rs. La cobertura acreditada asciende a **93 Rust preexistentes distintos de 289**. Los archivos nuevos de implementación y pruebas no inflan ese denominador. Faltan 196 Rust por acreditar como leídos íntegramente, además de archivos de otros tipos. El inventario de la ronda anterior —1.119 archivos versionados y 24 manifiestos Cargo— no equivale a cobertura de auditoría completa.

**Cambios funcionales concretos:**

1. Estadístico de media con denominador muestral y factor sqrt(n) correctos, cálculo normalizado sin piso absoluto de desviación y estados explícitos de degeneración/error.
2. EWMA que sólo incorpora nuevas revisiones de observación; cero ya no significa “no inicializado”.
3. Autoridad de parada monotónica para el daemon: puede activar el latch compartido, pero no desactivarlo por la recuperación de su score.
4. Telemetría tipada y motivos de promoción que describen un gate heurístico, sin atribuirle confianza bayesiana superior al 95 %.

No se cambiaron capital, apalancamiento, payloads, armados por entorno, genomas persistidos ni el proceso en ejecución. No se desplegó el binario. No se promete duplicar capital ni se presenta un algoritmo clásico como cuántico.

## 2. Grafo vivo: raíz, decisiones, terminales y límites de autoridad

[El host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1967>) construye el daemon y [lanza su bucle](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1994>). Esta conexión es diferente de la ruta CMA auxiliar auditada en VIII, para la que no se localizó un caller operativo.

~~~mermaid
flowchart TD
  H["Nodo raíz: host"] --> D["LiveEvolutionDaemon"]
  P["PnL realizado agregado + capital actual"] --> O["Observaciones por polling — FMT-132"]
  D --> O
  O --> E["ReturnEvidence: lote completo"]
  E -->|no finito / insuficiente / rango numérico| X["Sin estadístico utilizable; no promoción en esta evaluación"]
  E -->|constante| C["t indefinido; signo observado explícito"]
  C -->|pérdidas y política de muestra| L["Latch de parada: sólo activar"]
  E -->|studentizado finito| R["EWMA por revisión nueva"]
  R -->|degradación heurística| L
  R --> A["Armado / búsqueda / gates existentes"]
  A --> G["Nodo de decisión: GenomeEnvelope"]
  G --> W["Nodo terminal documental: save_weight"]
  W --> Q["Cola acotada sin ack — FMT-130"]
  Q --> S["Upsert SQLite de pesos — FMT-131"]
  I["Host: drawdown / latencia / otras causas"] --> L
~~~

El grafo refleja aristas del código, no prueba que un proceso concreto haya realizado una promoción. El daemon mantiene otras mutaciones y gates ya documentados; esta intervención no los certifica.

El control de parada no pertenece a una sola métrica. [El supervisor del host](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:1721>) activa el flag por drawdown o latencia, y el executor dispone de sus propias barreras. Que antes el daemon borrase el flag de arena no demuestra que desactivara simultáneamente todas esas barreras ni que se enviara una orden. Sí era una violación demostrable de autoridad sobre estado compartido.

La conexión de retorno a los nodos raíz sigue incompleta: las observaciones no identifican operación, decisión, entrada/salida ni genoma causante. Esa falta impide deducir que una mejora global de PnL sea aprendizaje causal de un candidato concreto.

## 3. Matriz de estado: qué cambió y qué permanece abierto

| Referencia | Estado de esta ronda | Evidencia y pendiente |
|---|---|---|
| FMT-055: fórmula de media studentizada | Reparación local probada | sqrt(n), finitud, homogeneidad de unidades y degeneración explícita |
| FMT-055: actualizaciones por reloj | Reparación local probada | Repetir revisión no cambia la EWMA; no significa independencia de ventanas |
| FMT-055: falsa confianza | Descripción corregida; gate pendiente | No se publica posterior >95 %; permanece la fórmula heurística |
| FMT-056: rearme por el daemon | Escritura insegura retirada | La función de seguridad sólo escribe true; no restablece causas ajenas |
| FMT-056: rollback | Abierto | No hay compare-and-swap de generación; un error sigue consumiendo el watchdog en la rama histórica |
| FMT-130 | Nuevo, P1, abierto | Encolar no implica persistir y el emisor no conoce pérdidas |
| FMT-131 | Nuevo, P1 para uso de auditoría, abierto | Upsert de pesos no constituye historial íntegro de decisiones |
| FMT-132 | Nuevo, P1, abierto | Deltas seleccionados / capital posterior no equivalen a retornos reconciliados |
| FMT-049/050/051/052/057/071 | Abiertos | Mundo simulado, holdout, features, bypass y watchdogs bloqueados siguen pendientes |
| FMT-113/117 | Abiertos | Cantidad ejecutable y soporte temporal no se reparan con un estadístico nuevo |

Los avances parciales no se renumeran como hallazgos nuevos. Se conservan los identificadores FMT-055/056 para mantener trazabilidad con III. FMT-130–132 amplían el registro, sin renumerar la matriz histórica de 305 puntos del informe maestro.

## 4. FMT-055 — Reparación matemática y del reloj de evidencia

### 4.1 Qué calcula ahora el módulo, y qué no calcula

[summarize_returns](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/return_evidence.rs:46>) recibe un lote finito de observaciones escalares x_i. Calcula:

~~~text
media = sum(x_i) / n
s² = sum((x_i − media)²) / (n − 1)
T_descriptivo = (media / s) · sqrt(n), si s > 0
~~~

La media expresa localización, s expresa dispersión muestral y T compara la media con su escala de error bajo la normalización habitual. La fórmula puede calcularse para n≥2 y dispersión positiva; eso no garantiza que su distribución sea Student en los datos del sistema. La inferencia exacta habitual necesita supuestos que aquí no están acreditados. El resultado tampoco es un Sharpe anualizado: no incorpora una base temporal de anualización ni corrige autocorrelación.

La fórmula anterior multiplicaba por sqrt(n−1), aunque ya dividía la varianza entre n−1. Para el fixture [1,2,…,10], devolvía **5,449770637375485** frente al valor **5,744562646538029**. No es la mayor fuente de sesgo del sistema, pero sí un error reproducible de ecuación.

### 4.2 Homogeneidad, rango numérico y conservación de colas

Se normaliza cada observación por a=max|x_i| antes de calcular media y varianza, y se usa suma compensada para la media. Para c>0, en aritmética ideal T(c·x)=T(x). La implementación se comprueba entre factores 10^-250 y 10^250 sobre un fixture representable; no se afirma invariancia fuera del rango de punto flotante.

El piso anterior s≤10^-12 devolvía cero. Una serie con información relativa idéntica, expresada en unidades pequeñas, quedaba neutralizada. La regresión con factor 10^-20 falló antes de reparar. El nuevo módulo no introduce otro piso de volatilidad financiera para ocultar la degeneración.

No se eliminan colas negativas ni observaciones finitas extremas. Una prueba conserva una pérdida grande que vuelve negativa la media de treinta pequeños beneficios. Retenerla no convierte el estimador en robusto frente a contaminación; evita declarar favorable una muestra recortando precisamente el riesgo que debía medir.

Se distinguen pérdida de rango numérico y constancia real. Una muestra no constante cuya desviación restaurada no es representable devuelve NumericalRange, no Constant ni cero. Por ejemplo, la desviación muestral de [-MAX,MAX] no cabe en f64. No se oculta ese fallo detrás de una puntuación neutral.

### 4.3 Datos inválidos y degeneración no son ausencia de edge

ReturnEvidence contiene n, media, desviación y MeanStatistic. El estado Constant indica que todos los valores son iguales; studentized() devuelve None. Una muestra vacía o de un elemento devuelve InsufficientObservations. Un NaN o infinito rechaza todo el lote y conserva su índice en NonFiniteObservation.

Antes se eliminaban los elementos no finitos y se calculaba sobre lo restante. Sin un contrato de faltantes, eso cambia la población y puede favorecer el resultado. Ahora el módulo no imputa ni descarta observaciones silenciosamente. **Límite:** el productor de PnL todavía puede filtrar datos antes de formar el lote; esta validación no recupera observaciones que ya se perdieron aguas arriba.

Veinticinco valores −0,01 no producen un t finito: la desviación es cero. Pero tampoco constituyen evidencia observada neutral. El consumidor distingue pérdidas constantes y puede activar la parada por una política de seguridad explícita, sin inventar un p-valor ni un t=−infinito. Una constante positiva o cero no produce aprobación estadística: la rama de búsqueda/promoción sale sin fabricar score.

El watchdog aplica el mismo diagnóstico. Para una muestra constante negativa, activa el latch de seguridad, no ejecuta un rollback atribuyéndole significancia inexistente. Para muestras variables conserva su umbral operativo −2 y sus pendientes de linaje. No se certifica la seguridad integral del rollback.

### 4.4 Reloj de observación frente a reloj del scheduler

[La ingesta](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:505>) incrementa una revisión al añadir una observación válida al histórico. EvidenceEwma sólo modifica su estado si esa revisión es posterior a la última consumida. La longitud de la ventana no sirve como versión: al estar acotada, pueden entrar datos nuevos sin cambiar n.

[observe](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/return_evidence.rs:122>) distingue valor no inicializado mediante Option. Cero es un valor legítimo; una primera observación de score cero seguida de score 2 con peso 0,1 produce 0,2, no un reinicio a 2.

Contraejemplo reproducido en la nueva suite: inicializar en −0,4, incorporar una nueva revisión con −2 y leerla otras mil veces conserva **−0,56**. Las lecturas no empujan el estado progresivamente hacia −2. También se rechazan revisiones antiguas sin modificar el resultado. Peso inválido o score no finito no consumen la revisión ni mutan estado.

Esta operación sigue siendo una EWMA de scores de ventanas solapadas, no una acumulación de evidencia independiente. El peso 0,1 se conserva como política heredada; no se deriva como óptimo ni como corrección de dependencia. Persisten repetición de búsquedas sobre datos compartidos y falta de un ledger completo de consultas.

### 4.5 Cambios de interfaz y política que deben conocerse

Se añade shadow_return_evidence como vista tipada. El escalar histórico shadow_sharpe_ratio se conserva por compatibilidad, pero contiene NaN cuando T está indefinido. La búsqueda de referencias sólo localizó su declaración, inicialización y escritura en el daemon; no se identificó un consumidor de decisión externo de ese escalar en el repositorio.

Las dos vistas tienen locks separados: no se promete snapshot atómico al leer ambas juntas. La vista tipada es la autoritativa para interpretar estado y unidades. Tampoco contiene aún reloj de mercado, versión de genoma o duración efectiva.

La evaluación no inicia búsqueda con menos de diez observaciones, haciendo explícito el mínimo que antes tenía el estimador pero que el caller transformaba indirectamente en score cero. Diez, veinte y veinticinco son políticas de consumidores ya presentes, no teoremas de suficiencia. No se añaden nuevos números para aparentar significancia.

Se sustituyó el motivo de promoción “confianza bayesiana >95 %” por los valores reales del score heurístico y su umbral. La fórmula sigue siendo heurística y requiere sustitución metodológica; corregir su etiqueta no convierte el gate en una inferencia calibrada.

## 5. FMT-056 — Autoridad monotónica de parada y límites del rollback

### 5.1 Defecto y reparación

El daemon escribía false en arena.kill_switch_active cuando su EWMA superaba −0,5. El mismo flag podía haber sido activado por drawdown, latencia o una intervención distinta. Una recuperación de score no demuestra resolución de esas causas.

[latch_degradation](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/return_evidence.rs:151>) tiene una única facultad: si hay petición de degradación, hace una transición atómica hacia true. Si no hay degradación, no escribe nada. Devuelve si esta llamada produjo la transición false→true, para evitar duplicar el mensaje de activación.

La propiedad local es:

~~~text
estado_final = estado_inicial OR petición_de_parada
~~~

No permite la transición true→false. Una prueba exhaustiva cubre los cuatro casos booleanos. Otra alimenta cien revisiones saludables con la parada ya activa: permanece activa. Una prueba concurrente de dieciséis hilos verifica una sola transición de activación; todos los hilos se esperan mediante join, sin confundir lanzamiento con finalización.

### 5.2 Consecuencia operativa deliberada

El daemon ya no rearma automáticamente operaciones, incluso si fue quien detectó inicialmente la degradación. La recuperación requiere una autoridad que conozca y libere las causas correspondientes. Esto favorece preservación de seguridad, pero puede mantener el sistema detenido hasta una intervención autorizada. No se disfraza ese coste como autonomía resuelta.

La evolución puede seguir haciendo trabajo auxiliar cuando otras condiciones lo permiten; la reparación no enciende ni apaga por sí sola los flags del executor. Tampoco puede impedir que otro escritor borre el booleano. La garantía demostrada se limita al escritor corregido, no a todos los caminos de seguridad del proyecto.

Arquitectura de cierre propuesta: conjunto de razones de parada con propietario, epoch, evidencia, condición de despeje y registro de acción; la autorización efectiva es la conjunción de todas las razones despejadas. Un score agregado no debe tener autoridad universal de rearme. La migración requiere revisar todos los escritores y consumidores, no sólo cambiar el tipo del flag.

### 5.3 Pendientes que impiden cerrar FMT-056 completo

[La rama histórica del watchdog](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:569>) aún no condiciona la reversión a que la generación vigilada siga siendo la actual mediante una transacción indivisible. Una comprobación previa aislada no resolvería la carrera con una promoción concurrente.

También permanece el borrado de promoted_generation y post_promo_returns tras intentar rollback, incluso si falla. No se mueve ese borrado sin definir reintentos, idempotencia, backoff y estado de fallo persistente: hacerlo de forma incompleta puede producir un nuevo intento cada ciclo y seguir revirtiendo el linaje equivocado. La seguridad de esa máquina de estados requiere una intervención específica.

No se ejecutó ningún rollback real ni se abrió el almacén para modificarlo durante las pruebas. La revisión de genome_store fue dirigida a sus operaciones relevantes; no se suma como nuevo archivo leído completo.

## 6. FMT-130 — Entrega al registro evolutivo sin confirmación de persistencia

**Prioridad:** P1 para trazabilidad de decisiones; no prueba de pérdida ya ocurrida.

**Evidencia.** [save_weight](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/evolution_ledger.rs:101>) usa try_send sobre un canal acotado a 100.000 eventos e ignora el resultado. La creación del hilo tampoco propaga fallo al constructor. El writer registra ciertos errores SQLite por log, pero el emisor no recibe confirmación de apertura, aceptación, escritura ni commit.

**Mecanismo.** Hay al menos tres estados distintos: solicitud creada, solicitud encolada y transacción confirmada. La API retorna unit en todos los casos. Canal lleno o desconectado pueden descartar el evento sin que el caller sepa que falta. Un batch extraído de la cola se pierde si falla la transacción; no se observa reencolado ni protocolo de replay.

**Consecuencia.** [El daemon](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:1200>) solicita registrar una promoción después de aplicarla. La ausencia posterior del registro no permite distinguir “no hubo promoción” de “falló la escritura”. Que SQLite use WAL no convierte una llamada no confirmada en durable ni hace cero el bloqueo: existe busy_timeout y un hilo escritor con coste propio.

**Acotación.** No se forzó saturación, disco lleno ni fallo de DB del usuario. El defecto se establece por las ramas de error y el contrato de retorno, no por una afirmación de incidente observado. El canal está acotado en número de eventos, no en bytes máximos de todos los String posibles. No se afirma que el productor normal alcance ese límite.

**Cierre requerido.** Separar aceptación y durabilidad con identificador estable y ack; decidir qué eventos exigen persistencia antes de aplicar un cambio. Registrar contadores de descartes, backlog, edad máxima y errores. Probar worker ausente, cola llena, commit fallido, apagado, reintento y recuperación. No convertir todo el hot path en IO síncrono sin presupuesto de latencia; una cola durable o un outbox transaccional necesitan diseño y medida.

## 7. FMT-131 — Un almacén mutable de pesos se usa como si fuera historial de decisiones

**Prioridad:** P1 para auditoría; el upsert puede ser correcto para su función original de vista de pesos.

**Evidencia.** [La tabla genome_weights](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/evolution_ledger.rs:45>) tiene clave primaria (symbol,strategy), y el INSERT usa ON CONFLICT DO UPDATE. Campos no finitos se convierten en cero antes de encolarlos y de escribirlos. No contiene event_id, generation aplicada, parent_generation, hash de genoma, versión de datos, motivo de decisión ni resultado de aplicación.

**Contraejemplo operativo.** El caller de rollback usa symbol=gen_rollback_PARENT y strategy=rollback. Dos reversiones al mismo padre tienen la misma clave y la segunda reemplaza a la primera. La promoción usa un símbolo basado en generación, que evita algunas colisiones, pero no hace append-only toda la tabla ni reconstruye la relación causal completa.

**Impacto.** Una vista del último peso puede responder al estado actual, no al conjunto ordenado de eventos. La imputación de NaN como 0 colapsa “dato inválido” y “peso cero válido”, perdiendo información necesaria para investigar decisiones. El timestamp se obtiene al procesar el evento en el writer, no al observar el mercado ni al aplicar el genoma.

**Límites para evitar falsos positivos.** Los comentarios y tests todavía dicen scalp/swing, pero strategy es texto libre: esta tabla no obliga por esquema a dos motores. El path del host termina en .redb, aunque el writer usa SQLite; la extensión por sí sola no cambia el formato ni demuestra fallo de lectura. Tampoco se afirma que todo el historial genómico se pierda: GenomeEnvelope conserva otro mecanismo de historia, con sus propios contratos pendientes.

**Cierre requerido.** Mantener la vista de pesos existente si se necesita, y añadir un journal de decisiones inmutable con IDs únicos, entorno, generación, hashes, métricas tipadas, clocks y estado de persistencia/aplicación. Deduplicar reintentos por ID, no por padre o nombre de estrategia. Probar dos rollbacks al mismo padre, errores de datos y reconstrucción de orden bajo commits diferidos. No migrar ni sobrescribir las DB existentes sin plan y copia recuperable.

## 8. FMT-132 — El productor no entrega una serie de retornos reconciliada

**Prioridad:** P1; afecta observación, selección, riesgo y capacidad de atribuir aprendizaje.

**Evidencia.** [sample_realized_returns](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/evolution-engine/src/online_daemon.rs:489>) toma diferencias del PnL realizado acumulado por moneda entre polls, omite delta=0 y divide por el capital agregado actual. Ese mismo vector alimenta estadísticos y búsquedas. No conserva capital antes de cada outcome, flujos externos, IDs de fills o genoma de decisión.

**Diferencia algebraica del denominador.** Bajo el modelo simplificado sin flujos ni otros cambios, sea r=delta/C_antes y C_después=C_antes(1+r). Dividir por el capital posterior produce:

~~~text
rho = delta/C_después = r/(1+r), con r > −1
d²rho/dr² = −2/(1+r)³ < 0
~~~

No es la misma variable. En dos escenarios independientes que parten de 100, un outcome +10 genera +10/110≈0,090909 y uno −10 genera −10/90≈−0,111111. La media de r es cero; la de rho es aproximadamente −0,010101. La concavidad explica la asimetría bajo ese modelo y sus condiciones, sin extrapolarla como magnitud exacta del sesgo del sistema real.

En el daemon real, el capital leído puede incluir cambios de otras monedas y puede no ser un snapshot coherente con los contadores. La fórmula simple anterior ilustra un error de interpretación, no afirma que toda lectura corresponda exactamente al cierre inmediatamente anterior.

**Agregación y selección.** Dos fills de +1 y −1 dentro del mismo intervalo dejan delta=0 y no añaden observación, aunque hubo actividad, costes y dispersión. Varios cierres de una moneda pueden convertirse en una sola muestra; dos monedas se incorporan secuencialmente a una lista que no conserva sus timestamps originales. Por tanto, n cuenta deltas observados no n trades independientes.

**Faltantes y resets.** El mapa last_realized_by_coin no lleva epoch de contador. Un reset puede parecer una pérdida. Un PnL no finito puede contaminar temporalmente el baseline guardado; el filtro posterior de ret finito no reconstituye la evidencia omitida. El nuevo validador estadístico sólo valida el lote que recibe, no estos mecanismos de producción.

**Relación con FMT-049/052.** FMT-049 ya documenta usar resultados de la estrategia como si fueran retornos de mercado para simular otro genoma. FMT-132 añade el problema previo: ni siquiera la serie de resultados propios corresponde necesariamente al retorno económico, reloj o unidad de observación que se declara. FMT-052 añade incompatibilidad de features; ninguna de las tres capas se corrige multiplicando por sqrt(n).

**Cierre requerido.** Construir eventos de outcome reconciliados, con capital/base definidos, inventario marcado, costes, flujos, tiempos de decisión/ejecución y linaje. Conservar eventos de retorno cero y faltantes tipados. Separar retorno por operación, incremento de riqueza por intervalo y exposición ponderada; no mezclarlos en una misma inferencia. Reproducir escenarios multiactivo, cierres simultáneos, resets y capital externo antes de sustituir la serie activa.

## 9. Ciencia aplicable: inferencia secuencial y evaluación de políticas

### 9.1 Ampliación de T22 — Monitorizar continuamente exige un protocolo de inferencia

[Time-uniform, nonparametric, nonasymptotic confidence sequences](https://arxiv.org/abs/1810.08240) distingue cobertura puntual y uniforme en el tiempo. Los pasajes consultados muestran por qué consultar intervalos puntuales repetidamente no conserva automáticamente su garantía, y discuten el significado de parámetros que cambian.

Esto orienta la corrección del reloj, pero no proporciona un certificado para nuestra EWMA. Una futura implementación debe especificar estimando, filtración, condiciones de momentos/colas, límites de recompensa, política de resets y qué información se usa para elegir apuestas o límites antes de cada observación. “No paramétrico” no significa “sin supuestos”.

La búsqueda y expansión recuperaron [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686), [Admissible anytime-valid sequential inference must rely on nonnegative martingales](https://arxiv.org/abs/2009.03167) y [Game-theoretic statistics and safe anytime-valid inference](https://arxiv.org/abs/2210.01948). Se consideran familias relevantes por sus resúmenes; no se implementaron sus procedimientos ni se verificaron sus cuerpos en esta ronda.

La investigación Firecrawl guio búsqueda, expansión y verificación de pasajes primarios. La CLI no estaba disponible; se usó el índice conectado, sin instalar software ni enviar fuentes privados. Su influencia fue exigir separación entre score, evidencia y garantía, no añadir una fórmula con atribución de seguridad no demostrada.

### 9.2 T32 — Evaluación fuera de la política observada, con soporte y causalidad explícitos

[Waudby-Smith y colaboradores](https://arxiv.org/abs/2210.10768) estudian inferencia off-policy para bandits contextuales adaptativos. Los pasajes consultados explican que evaluar una política diferente de la que generó los datos es un problema contrafactual y conservan condiciones sobre el proceso, incluso cuando permiten contextos dependientes y políticas cambiantes.

En este sistema, los beneficios del incumbente no son por sí solos la respuesta que habría obtenido un mutante. Se necesita registrar contexto previo, acción, política que la eligió, probabilidad de elección cuando exista, costes y recompensa posterior. Un cociente de propensiones no puede recuperar información de acciones sin soporte en los datos. Esta última es una condición de diseño para la integración, no una propiedad ya verificada del daemon.

No se propone introducir aleatoriedad en operaciones reales para obtener cobertura de acciones sin un protocolo de riesgo autorizado. Tampoco se reduce automáticamente trading con inventario, impacto y recompensas diferidas a un bandit de una sola decisión. Cuando las acciones alteran el estado futuro, hay que formular el problema secuencial correspondiente.

Experimento previo a integración: entorno retenido con políticas registradas, soporte conocido y política objetivo identificada; comparar estimaciones con el resultado contrafactual disponible en esa simulación, medir sesgo/cobertura y detectar explícitamente falta de soporte. Después comprobar si las hipótesis sobreviven a los datos y mecanismos reales. El resultado de simulación no basta para trasladar el certificado al mercado.

### 9.3 Universo continuo temporal y multivariante: contrato, no etiqueta

Esta intervención no añade motores scalping/swing. El módulo de evidencia no toma decisiones según esas etiquetas, pero sigue siendo un resumen escalar de observaciones agregadas. No se presenta como una representación completa del espectro.

Para avanzar hacia el dominio temporal continuo hacen falta estados condicionados por escala, activo, exposición y contexto, con timestamps y soporte efectivo. El reloj de revisión corregido sólo identifica novedad del batch; no mide nanosegundos, no reconstruye el mercado entre mensajes y no observa cien años de evolución. La frecuencia del scheduler no es el horizonte económico ni una fuente adicional de información.

Una política continua puede representarse funcionalmente y evaluarse con resolución adaptativa, como se discutió en VIII; su malla debe obedecer error y soporte. La inferencia multivariante necesita además dependencias entre activos, señales, horizontes y decisiones. Multiplicar tests escalares sin controlar consultas o correlación no convierte el sistema en omnisciente.

### 9.4 Física, algoritmos y cuántica: criterio de admisión

Se conserva la aspiración de aprovechar teoría avanzada. El criterio para integrar es una predicción refutable o una propiedad de cálculo/control que pueda medirse: unidades correctas, estabilidad, causalidad, cota de error, reducción de coste o mejor resultado fuera de muestra bajo el mismo presupuesto.

No se incorporan ecuaciones de problemas del milenio por prestigio. Un potencial o una PDE necesita variables identificadas y validación del modelo; una técnica cuántica necesita formalización, recursos y comparación clásica. Los arreglos de esta ronda son estadística descriptiva, concurrencia y contratos de evidencia clásicos.

## 10. Revisión por módulos y rehabilitación 1-a-1

| Módulo | Conexión observada y siguiente cierre |
|---|---|
| 1. Ingestión/L2 | Outcome reconciliado y clocks; evitar reemplazar mercado por PnL propio |
| 2. IA y señales | Features coherentes entre entrenamiento/inferencia, FMT-052; no calibrar sobre muestras contaminadas |
| 3. Estrategia y temporalidad | Estado continuo condicionado por soporte; no inferir resolución temporal a partir de polling |
| 4. Ejecución/conectividad | FMT-113 sigue prioritario: quantity, reservas, payload y fills coherentes |
| 5. Riesgo/genomas | Parada monotónica local reparada; identidad de promoción y rollback todavía pendiente |
| 6. Estado/telemetría/SO | FMT-130/131: entrega durable, orden, snapshots y límites medidos |
| 7. Orquestación | Autoridades de parada por razón; no rearmar causas ajenas desde una señal agregada |
| 8. Backtest/gobernanza | Protocolos de selección y holdout; medir cobertura bajo la búsqueda completa, no sólo una fórmula |

Orden recomendado de implementación:

1. Cerrar el evento económico de FMT-132 y su atribución a genoma/decisión. Sin ello, más sofisticación estadística aprende sobre una variable ambigua.
2. Añadir journal durable y acknowledgments manteniendo la vista de pesos, FMT-130/131.
3. Resolver rollback como transacción condicional de linaje y preservar evidencia ante fallo; auditar todos los escritores del latch.
4. Separar búsqueda pesada, ingesta y watchdogs, FMT-071; medir edad de evidencia, tiempo de ciclo y latencia p99.
5. Sustituir gates heurísticos por un protocolo inferencial explícito cuando se satisfagan sus supuestos. Mientras tanto, reportar scores como scores.
6. Validar el proceso completo de evolución frente al incumbente con datos retenidos, costes y multiplicidad; no promover basándose en una sola etiqueta OOS.
7. Continuar la cobertura archivo por archivo. Esta ronda no certifica que los restantes módulos estén libres de bugs.

## 11. Pruebas, manifiesto y preservación

Comandos ejecutados:

~~~text
cargo test -p evolution-engine --offline --lib evidence_regressions
cargo test -p evolution-engine --offline --test return_evidence_contract
cargo test -p evolution-engine --offline --test cma_penalty_contract --test cma_generation_contract --test cma_supervision_contract
cargo test -p evolution-engine --offline --lib cma_es::tests
cargo check -p trader-gemini-v5 --bin god_engine --offline
~~~

| Suite | Tests distintos aprobados |
|---|---:|
| evidence_regressions, nuevos | 4 |
| return_evidence_contract, nuevos | 18 |
| Tres suites de contrato CMA, anteriores | 22 |
| cma_es::tests, anteriores | 3 |
| Total | 47 |

No se duplican las reejecuciones finales en el conteo. Las primeras cuatro pruebas fallaron contra la función anterior. En los casos de constancia/error, el oráculo se migró de “no devolver un número neutral válido” a exigir el estado tipado correspondiente; no se pretendió conservar una API escalar que ocultaba precisamente la distinción.

Las pruebas cubren unidades, signo, varianza casi nula, muestra constante, colas, no finitos, insuficiencia, overflow, revisiones repetidas/antiguas, inicialización en cero, parámetros de EWMA inválidos y concurrencia del latch. No son pruebas end-to-end de exchange, promoción o rollback; no crean DB de usuario ni ejecutan el daemon infinito.

| Archivo propio | Líneas | SHA-256, prefijo |
|---|---:|---|
| crates/evolution-engine/src/online_daemon.rs | 1254 | EEA91F108543B66A |
| crates/evolution-engine/src/lib.rs | 666 | F43E052ADEDCED10 |
| crates/evolution-engine/src/return_evidence.rs | 153 | FCF62A34C4DD61E4 |
| crates/evolution-engine/tests/return_evidence_contract.rs | 202 | 9F3E44CBF57F9774 |

online_daemon.rs estaba limpio al comienzo, hash 2796CF7361F127E8. lib.rs contenía cambios propios de VIII y su hash previo C340A623B9DDF8B6 se verificó antes de añadir únicamente la exportación del nuevo módulo. No se formatearon globalmente esos fuentes; rustfmt --check se aplicó al módulo y test nuevos. git diff --check pasó para los fuentes modificados.

El archivo adicional leído completo, no modificado, es storage-engine/src/evolution_ledger.rs: 247 líneas, SHA-256 CE46371CCBEADE63. Los tramos de host y genome_store son lecturas dirigidas, no nuevas unidades de cobertura completa.

Los hashes ajenos prioritarios se conservaron: host F2BA92C4283C66E6; booktick_replay 736CC1DBB38E21D7; risk/lib E9883455B6BEBEFA. Se preservaron cambios concurrentes de configuración, genomas y graphify-out. Se añaden referencias a ATLAS, maestro, III y VIII sin borrar contenido anterior, incluidos bytes NUL históricos del maestro.

Permanecen avisos de compilación preexistentes sobre latest_ts, mode y RealWfOutcome.trades. No se ejecutó cargo fix. Las esperas por el lock de compilación no se resolvieron matando procesos.

**Conclusión:** esta ronda mejora una ruta realmente conectada, pero conexión en código no equivale a despliegue ni eficacia económica. El cálculo ya distingue lo observado de lo indefinido, y la recuperación de una señal no libera una parada ajena. La observación económica, su linaje y su persistencia siguen impidiendo certificar autoevolución integral.

Sin trading, despliegue, cambios de genomas, reinicios, commit, push, merge ni fetch.

### Verificación final de preservación

Se comprobaron los doce enlaces locales de fuente, los tres identificadores nuevos y los hashes completos de los cuatro archivos propios, del ledger leído y de los tres fuentes ajenos prioritarios. Las reejecuciones finales de los 22 tests nuevos y cargo check terminaron correctamente. Los prefijos completos anteriores de ATLAS, maestro, III y VIII conservaron su SHA-256, normalizando sólo CRLF/LF para comparar; se añadieron respectivamente 1.583, 2.199, 1.052 y 844 caracteres. HEAD e inventario permanecieron en 59a76de4, 1.119 archivos versionados, 289 Rust y 24 manifiestos Cargo. La solicitud de mostrar el anexo en el panel fue encolada por la aplicación; no se presupone que el usuario lo haya abierto.
