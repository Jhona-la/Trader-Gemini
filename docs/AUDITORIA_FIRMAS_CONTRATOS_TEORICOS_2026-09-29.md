# Auditoría ST — firmas de caminos y contratos de transferencia científica

Fecha: 2026-09-29. Autor: Codex. Rama local: codex/signature-contract-audit.
Base examinada: 49fc995c2275ba80500671f8410b484337e4159b; el módulo de GLM
entró en fb6412c98ed90e684cd2ba8d056e932d3376032a.

## 1. Dictamen y límites del aseguramiento

Se reproducen cinco pruebas fallidas asociadas a cuatro defectos numéricos y
temporales del adaptador de firmas. Se corrigen localmente sin alterar el kernel
algebraico ni conectar señales nuevas al trading. Otras tres afirmaciones de
documentación requieren corrección: unicidad de la truncación, fórmula
incompleta y alcance de la invariancia al remuestreo.

Esta ola registra **18 hallazgos y limitaciones**, no 18 bugs de ejecución:
4 reparaciones de código, 3 correcciones documentales y 11 cuestiones abiertas
de diseño, contrato o validación científica. Ninguna reparación local equivale
a publicación, merge, despliegue, validación OOS o garantía de rentabilidad.

La búsqueda de consumidores en crates/ y src/ sólo encuentra la exportación,
definiciones y pruebas de firmas. Por tanto, los defectos están en un módulo
integrado al repositorio, pero **no se demuestra impacto operativo actual**.
Tampoco se puede afirmar que haberlo implementado haga al genoma más adaptativo.

Se enumeraron 1.350 archivos versionados en la base. Esa enumeración NO significa
que se leyera semánticamente cada archivo. La revisión profunda de esta ola cubre
path_signatures.rs y sus tests, TRIAGE_TEORICO, los contratos citados de Hurst,
las limitaciones previas CF/OA y el estado de integración. El resto conserva
su condición de no re-auditado en esta ola. Compilar todos los targets tampoco
certifica todos sus comportamientos.

La prioridad P2 de los fallos de firmas refleja la ausencia de consumidor
operativo encontrado. Su prioridad subiría si un consumidor externo no
versionado las utilizara: esa posibilidad no se ha verificado.

## 2. Raíz → representación → decisión → terminal

~~~mermaid
flowchart LR
    A["Raíz: precios y reloj del feed"] --> B["Adaptador: dominio, orden y duración"]
    B --> C["Nodo de representación: firma de nivel 2"]
    C --> D["Seis coordenadas; sin consumidor operativo encontrado"]
    D -. "propuesta, no conexión existente" .-> E["Modelo versionado + activo + τ + frescura"]
    E -.-> F["Nodo de decisión: utilidad y riesgo de cartera"]
    F -.-> G["Nodo terminal: ejecución y ledger"]
    G -. "etiqueta causalmente alineada" .-> E
~~~

El error de raíz se propaga aunque Chen y reverso pasen: una identidad algebraica
puede ser correcta sobre coordenadas de entrada mal definidas. Las líneas
discontinuas son requisitos de integración futura, no evidencia de conexión.

## 3. Matriz de hallazgos

| ID | Tipo y prioridad | Resumen | Estado de esta rama |
|---|---|---|---|
| ST-01 | Numérico/temporal, P2 | La escala depende del epoch absoluto | Código corregido localmente |
| ST-02 | Integridad temporal, P2 | Retrocesos internos aceptados | Código corregido localmente |
| ST-03 | Representación, P2 | Conversión prematura pierde deltas enteros | Código corregido localmente |
| ST-04 | Estabilidad numérica, P2 | Dos logaritmos grandes cancelan retornos pequeños | Código corregido localmente |
| ST-05 | Teoría, P2 | Unicidad completa atribuida al nivel 2 | Documentación corregida |
| ST-06 | Especificación, P2 | Fórmula, contorno y reverso no coinciden | Documentación corregida/adenda |
| ST-07 | Muestreo, P2 | Invariancia geométrica confundida con remuestreo | Documentación corregida/adenda |
| ST-08 | Diseño estadístico, P2 | Seis coordenadas no son seis variables independientes | Abierto/documentado |
| ST-09 | Semántica temporal, P2 | La normalización elimina duración absoluta | Abierto/documentado |
| ST-10 | Integración, P2 | No hay consumidor de producción encontrado | Abierto/experimental |
| ST-11 | API numérica, P2 | Kernel público no valida finitud ni overflow | Abierto/precondición explícita |
| ST-12 | Diagnóstico de rechazos, P2 | Option mezcla causas de invalidez | Abierto |
| ST-13 | Coste, P2 | O(n) por ventana no es presupuesto multiescala | Abierto/no benchmark |
| ST-14 | Estimando, P2 | Hurst de precio no calibra rough σ | Abierto/corrigendo científico |
| ST-15 | Información temporal, P2 | Transfer entropy no elimina historia ni lag | Abierto/corrigendo científico |
| ST-16 | Inferencia, P2 | Lineage del outcome no identifica intervención | Abierto; relacionado con OA |
| ST-17 | Selección, P2 | Logloss aislada no demuestra MDL | Abierto/corrigendo científico |
| ST-18 | Garantías, P2 | Existencia de ACI no certifica cobertura desplegada | Abierto; relacionado con CF |

No se renumeran ni se cierran automáticamente las matrices históricas de
305 puntos, CF, OA o TH. Los ID ST son un ámbito nuevo, con referencias cruzadas.

## 4. Defectos reproducidos y reparación

### ST-01 — El denominador temporal era una fecha, no una duración

**Evidencia.** El adaptador original definía t_norm = timestamp / timestamp_final.
Calculaba span, pero sólo lo empleaba para comprobar que fuese positivo.
Para [1000,2000,3000], el incremento temporal total era 2/3; para la misma ventana
desplazada en 1.000.000 ms se convertía aproximadamente en 0,001994017946.
La prueba de traslación imprime 0,6666666666666667 frente a 0,001994017946161497.

**Mecanismo.** Si el origen se desplaza c, el denominador cambia de tN a tN+c.
La normalización de una ventana debe usar T=tN−t0. El error contamina S^t y S^tt,
y también las componentes cruzadas con precio. Dos observaciones idénticas
salvo fecha dejan de ser comparables. A epochs de mercado grandes el tiempo
normalizado de ventanas cortas casi desaparece.

**Corrección.** Cada incremento temporal es (t_i−t_{i−1})/T, con restas enteras.
Es la misma geometría que u_i=(t_i−t0)/T, pero evita restar dos fracciones
casi iguales. No se establece ningún horizonte fijo ni filtro económico.

**Falsación y cierre.** window_time_is_a_fraction_of_elapsed_duration exige
S^t=1 y S^tt=1/2; moving_the_epoch_does_not_change_window_features compara todas
las coordenadas tras la traslación. Ambas fallaban y ahora pasan.
La normalización cambia la semántica de features; cualquier consumidor externo
necesitaría versionado y revalidación, no una sustitución silenciosa.

### ST-02 — Sólo se validaban los extremos del reloj

**Evidencia.** [1000,900,2000] era aceptado porque el último instante supera
al primero. El segundo segmento retrocede en la coordenada temporal.

**Mecanismo e impacto.** La firma algebraica sigue siendo calculable, pero ya
no representa una trayectoria ordenada causalmente por el tiempo del feed.
Un pipeline puede confundir un evento fuera de orden con geometría real
del precio. Ordenar timestamps sin las observaciones asociadas sería aún peor.

**Corrección.** checked_sub entre instantes consecutivos rechaza retrocesos.
Los timestamps iguales se permiten cuando la ventana completa tiene duración
positiva: varios eventos pueden compartir la resolución del reloj. Se conserva
su orden de llegada y no se inventa un dt positivo.

**Pruebas.** backward_internal_clock_is_rejected_without_sorting_prices era RED;
equal_timestamps_preserve_event_order_when_total_span_is_positive confirma
que no se añadió una restricción de unicidad arbitraria.
Rechazar una ventana desordenada no sustituye una política upstream de
reordenamiento por secuencia/watermark; esa política queda fuera del parche.

### ST-03 — El epoch se convertía a f64 antes de restarlo

**Evidencia.** Los enteros 2^53 y 2^53+1 son distintos, pero la conversión
individual a f64 puede hacerlos iguales. La ventana positiva de 1 ms devolvía None.

**Mecanismo.** El contrato admite u64, mientras la resta operaba sobre una
representación con 53 bits de precisión. Es un error de dominio representable,
no evidencia de que los timestamps actuales en milisegundos alcancen ese valor.

**Corrección.** T y cada dt se calculan primero con checked_sub de u64;
sólo las duraciones se convierten para formar el cociente. La identidad
de diferencias enteras se conserva hasta esa conversión.

**Prueba y límite.** integer_elapsed_time_survives_absolute_epoch_above_f64_precision
pasa. No se afirma precisión infinita de cocientes f64, ni que dt de 1 ns
sea observable cuando el API recibe ms. Cambiar el nombre de la unidad no
recupera información que el feed no contiene.

### ST-04 — Cancelación del retorno logarítmico

**Evidencia.** Para p=1e300 y el siguiente f64 representable q>p, ln(q)−ln(p)
devolvía cero en la plataforma probada. El movimiento existe en la entrada,
pero se pierde al restar resultados redondeados de magnitud grande.

**Corrección final.** Para precios cercanos dentro de un factor dos se usa
ln1p((q−p)/p). La región se motiva por la condición de resta exacta de Sterbenz,
no por una clasificación de volatilidad. Fuera de ella se usa ln(q)−ln(p),
evitando ratios no representables y r redondeado cerca de −1. No se recortan
retornos ni se introduce un epsilon económico.

**Control de la propia corrección.** La primera propuesta usaba ln1p siempre
que r fuese finito y mayor que −1. Una prueba adicional q/p=1e−10 la rechazó:
−23,02585084720009 en vez de −23,025850929940457. Ese defecto intermedio NO
pertenecía al main original y no se cuenta como un quinto bug suyo. Se corrigió
antes del commit. Esto documenta una revisión adversarial real, no sólo un GREEN.

**Pruebas.** adjacent_representable_prices_keep_their_log_return,
large_price_declines_do_not_lose_ratio_precision_near_minus_one y
extreme_positive_prices_do_not_require_a_representable_price_ratio cubren
movimientos pequeños, caídas grandes y ambos sentidos entre el mínimo subnormal
positivo y f64::MAX. No prueban por enumeración todos los pares f64.

Referencia numérica: [prova automatizada de implementaciones math.h, §5.2](https://cs.stanford.edu/people/sharmar/pubs/popl18.pdf).
Se consultó el pasaje de Sterbenz, no se auditó ni adoptó el verificador del artículo.

## 5. Teoría de firmas: qué calcula y qué no

Para incrementos v_k=(ΔlogP_k,Δu_k), la representación calcula:

- Nivel 1: s = Σ_k v_k, desplazamiento total.
- Nivel 2: Q = Σ_{a<b} v_a⊗v_b + (1/2)Σ_k v_k⊗v_k.
- Concatenación: (s_A,Q_A)*(s_B,Q_B) =
  (s_A+s_B, Q_A+Q_B+s_A⊗s_B).
- Área orientada: A=(Q_xt−Q_tx)/2.

Estas ecuaciones describen el camino lineal por partes elegido a partir de
los datos. Chen permite verificar composición; no prueba capacidad predictiva.
El marco y las identidades se contrastaron con el
[primer de Chevyrev y Kormilitzin](https://arxiv.org/abs/1603.03788).

### ST-05 — La truncación no hereda la unicidad de la firma completa

**Evidencia.** El comentario y el triage atribuían al nivel 2 caracterización
del camino salvo equivalencia tree-like. Añadir «unicidad no afirmada» no
corrige esa premisa. El resultado de
[Hambly y Lyons](https://arxiv.org/abs/math/0507536) concierne a la firma completa
de caminos de variación acotada y su equivalencia; no hace inyectivo este vector.

**Contraejemplo propio reproducible.** Los incrementos
[(1,1/4),(-1,1/4),(-1,1/4),(1,1/4)] y [(0,1)] tienen el mismo nivel 1, (0,1),
y el mismo nivel 2, con Q_tt=1/2 y el resto cero. El reloj es estrictamente
creciente en ambos. Sin embargo, la coordenada de nivel 3
S^{x,x,t}=(1/2)∫x(t)^2dt vale 1/6 en el primero y 0 en el segundo.
No son equivalentes por la firma completa.

**Reparación.** Se retira la atribución incorrecta del comentario ejecutable
y se añade un corrigendo al triage sin borrar su histórico. La prueba
level_two_collides_for_distinct_strictly_time_augmented_paths preserva
el contraejemplo. No se añade nivel 3 por defecto: antes debe existir una
hipótesis de utilidad y un presupuesto de dimensión.

### ST-06 — Especificación inconsistente con el kernel

**Evidencia.** Dos comentarios mostraban sólo Σ_{a<b}, omitiendo la mitad
diagonal que el código sí sumaba. El triage decía <3 puntos, pero el API admite
dos; su explicación del reverso mezclaba invariancia y cambios de signo.

**Impacto.** Un consumidor o una reimplementación podía implementar otro objeto,
rechazar ventanas válidas o comparar reversos incorrectamente. Los cuatro tests
originales defendían el kernel, pero no eliminaban la contradicción documental.

**Corrección.** Fórmula completa en el módulo y el campo level2; contorno <2;
reverso s→−s y Q→Qᵀ. Para dos puntos hay una firma de segmento válida. Para
precio constante sólo las coordenadas de precio son nulas: el tiempo sigue
acumulando 1 y su término cuadrático 1/2. No se cambia el kernel original.

### ST-07 — Reparametrizar un camino no es observar otro remuestreo

**Evidencia.** Se afirmaba que la frecuencia de muestreo no contaminaba la
representación. Una subdivisión exacta de un segmento conserva su geometría;
eliminar o insertar una excursión del precio cambia el camino interpolado.

**Alcance correcto.** La reparametrización del recorrido transforma todas sus
coordenadas simultáneamente. Cambiar sólo el reloj de un camino aumentado
(logP,u) modifica su geometría. La invariancia a subdivisión no elimina aliasing,
microestructura, errores de interpolación ni diferente resolución de fuentes.

**Corrección y prueba.** Se acota el comentario y el triage; una prueba
subdivide cada segmento de un camino no recto y compara todas las coordenadas.
No se promete invariancia general ante downsampling de ticks. El consumidor
deberá registrar cómo se construyó e interpoló cada camino.

## 6. Limitaciones abiertas de integración y representación

### ST-08 — Redundancia exacta: dimensión de salida no es dimensión informativa

**Evidencia algebraica.** Q_ij+Q_ji=s_i·s_j. Con tiempo normalizado, s_t=1,
Q_tt=1/2, Q_xx=s_x²/2 y Q_xt+Q_tx=s_x. Por tanto, los seis valores sólo
contienen dos grados de libertad algebraicamente independientes: retorno total
y una coordenada de área. La prueba shuffle_identity_exposes_redundant_level_two_coordinates
verifica esta restricción, no un beneficio predictivo.

Dos grados de libertad algebraicos no significan rango lineal dos de la matriz
de features: x² puede aportar una dirección lineal distinta de x.

**Riesgo futuro.** Tratar las seis columnas como seis fuentes independientes
puede hacer singular una covarianza sin regularización, inflar confianza o
duplicar votos. No se elimina el vector: columnas redundantes pueden resultar
útiles a ciertos modelos y conservarlo evita romper el API.

**Cierre requerido.** Definir la representación consumida, su rango efectivo,
estandarización y regularización; comparar con el baseline (retorno, área),
controlando presupuesto de parámetros y selección. No resolverlo simplemente
sumando una constante diagonal sin estudiar su efecto.

### ST-09 — Falta el contrato de horizonte y soporte observable

**Evidencia.** to_features devuelve seis f64, sin activo, T, resolución, edad,
origen ni versión. La prueba normalized_features_alone_do_not_identify_absolute_horizon
construye la misma representación para duración 2 ms y 2.000.000 ms.

**Interpretación.** Es una consecuencia intencional de normalizar la forma,
no un defecto que deba corregirse devolviendo al epoch absoluto. Sin metadatos,
un genoma podría tratar como equivalentes trayectorias con costes, riesgo,
liquidez y targets radicalmente distintos.

**Cierre requerido.** Envolver cada observación con identidad del activo,
rango temporal as-of, τ realizado, unidad del reloj, resolución efectiva,
política de interpolación, calidad y versión. Evaluar por horizonte continuo
o bandas de soporte de datos; no reintroducir motores scalping/swing.
Extrapolar hacia 1 ns o 100 años debe declararse extrapolación, no observación.

### ST-10 — Implementación aislada no equivale a impacto del genoma

**Evidencia.** La búsqueda de firma_ventana_logprecio, firma_nivel2_incrementos
y path_signatures en crates/ y src/ de la base sólo encuentra módulo, export
y tests. No se encuentra conexión al entrenamiento, selección, inferencia
o feedback operativo.

**Consecuencia.** Puede afirmarse «feature experimental implementada».
No «genoma enriquecido en producción», «paridad backtest/demo» ni «alpha probado».
Este hueco explica por qué una mejora matemática aislada puede tener impacto cero
en operación, pero no se afirma que sea la causa de toda discrepancia del sistema.

**Cierre requerido.** Un único contrato de extracción, compartido por replay y
serving, con versión en el artefacto del modelo. Probar misma observación→mismos
features→misma predicción antes de añadir el feature al voto. Después, ablación
temporal OOS con costes y latencia; nunca conectar un peso positivo por intuición.

### ST-11 — Kernel algebraico sin dominio validado

**Evidencia.** firma_nivel2_incrementos acepta pares arbitrarios. Un incremento
f64::MAX genera overflow en productos; NaN se propaga. La prueba correspondiente
es diagnóstica: PASA al confirmar esta limitación, no al afirmar que fue reparada.

**Alcance del parche.** El adaptador de precios exige precios finitos positivos,
reloj válido y resultado finito. El kernel sigue siendo un API de bajo nivel
con precondiciones explícitas; no se altera su firma ni se ocultan valores
inválidos con ceros. No se añade una política nueva a consumidores inexistentes.

**Cierre requerido.** Si se expone a otras fuentes, definir un API checked con
errores tipados, rangos/unidades y comportamiento ante overflow acumulado.
Decidir si el API sin checks será interno o permanecerá público documentado.
Añadir fuzz/property tests incluyendo subnormales y acumulación larga.

### ST-12 — Rechazos sin causa diferenciada

**Evidencia.** None mezcla longitudes incompatibles, muestra insuficiente,
precio inválido, duración cero, reloj retrocedido y salida no finita.
No existe contador/identificador causal en este API.

**Impacto potencial.** Un consumidor no puede distinguir warmup de pérdida de
integridad del feed. Un fallback uniforme puede convertir fallos de datos en
falsa neutralidad o esconder un bloqueo persistente. Es un problema de
observabilidad de rechazo, no autorización para aceptar entradas inválidas.

**Cierre requerido.** API tipada o wrapper diagnóstico compatible que preserve
Option para llamadores existentes si fuese necesario. Registrar motivo y soporte,
sin multiplicar contadores por reintentos de la misma observación. Coordinarlo
con el registro de vetos de GLM, no crear un catálogo incompatible.

### ST-13 — El coste por ventana no es un presupuesto end-to-end

**Evidencia.** El adaptador asigna un Vec de n−1 incrementos y recorre la ventana;
el kernel vuelve a recorrerlos. O(n) es correcto para una ventana, pero no
establece latencia de m activos por k escalas ni coste de reconstrucción continua.

**Riesgo.** Recalcular m·k ventanas tras cada evento puede consumir mucho más
que la simple suma prefija sugiere. No hay benchmark de colas, p99, memoria,
recuperación ni presupuesto de muestreo en esta ola. No se inventa una cifra
de latencia ni se declara un cuello de botella medido.

**Cierre requerido.** Medir en hardware objetivo; separar extracción de la
ruta crítica; evaluar composición por Chen e inversa del prefijo para ventanas
móviles, verificando deriva numérica. La identidad algebraica por sí sola
no asegura una implementación incremental estable ni gratuita.

## 7. Corrigenda del triage científico

Estas correcciones se añaden al documento de GLM; no borran sus propuestas.
Distinguen una hipótesis prometedora de un supuesto ya contrastado.

### ST-14 — Rough volatility: estimar la variable correcta

**Evidencia local.** El triage propone H<0,1 contrastado contra DFA.
HurstDfa::update recibe precios y forma retornos; su pendiente no es una
estimación automática de la regularidad de log-volatilidad.

**Contraste primario.** [Gatheral, Jaisson y Rosenbaum](https://arxiv.org/abs/1410.3394)
estudian escalado de momentos de incrementos de log σ. Los pasajes consultados
incluyen estimaciones tanto inferiores como superiores a 0,1 y discuten sesgos
de ventana/suavizado. No justifican imponer H<0,1 universal a precios o activos.

**Transferencia admisible.** Construir un estimador causal de volatilidad con
ruido de medición declarado, estimar E|Δlogσ|^q por varias escalas y separar
estimando de precio y de volatilidad. Comparar predicción OOS y estabilidad
contra un baseline simple. No se implementó rough Bergomi ni una nueva ley
de TP/SL en este parche.

### ST-15 — La transferencia de entropía necesita historia y condicionamiento

**Evidencia.** El triage y el aviso de coordinación dicen que mide dirección
sin ventana de lag. La definición de
[Schreiber](https://arxiv.org/abs/nlin/0001042) compara distribuciones del futuro
del destino condicionadas a su pasado, con y sin el pasado de la fuente.
Los órdenes de historia y el muestreo siguen siendo decisiones del estimador.
El artículo contempla además condicionamiento por una fuerza común.

**Riesgo.** Usar dependencia simultánea como TE puede confundir shocks comunes,
asincronía o autocorrelación con transmisión entre activos. Ser direccional
no basta para certificar causalidad intervencional.

**Cierre requerido.** Declarar reloj/event time, desfase físico, historias,
conditioning set, estimador y sesgo de muestra finita. Contrastar nulos que
preserven dependencia temporal y asincronía, con corrección de selección
entre pares/escalas. TE queda como candidato, no como veto ya validado.

### ST-16 — Atribución contable/temporal no es identificación causal

**Evidencia.** El triage llama diseño causal a la atribución por posición.
El informe OA ya advierte que snapshots y generación evitan calificar una
opinión equivocada; no demuestran contribución económica causal de un modelo.

**Distinción.** Conservar quién predijo antes de una operación es indispensable
para lineage. Estimar qué habría ocurrido al intervenir sobre una política
requiere un estimando contrafactual y supuestos identificables. La completitud
del do-calculus opera sobre un modelo causal y su estructura, no sólo sobre IDs:
[Huang y Valtorta](https://arxiv.org/abs/1206.6831).

**Cierre requerido.** Formular tratamiento, outcome, unidad, confusores y política
de selección. Si se propone evaluación off-policy, registrar propensiones y
soporte; no afirmar identificación cuando se vetan de forma determinista
todas las acciones de una región. No se experimenta con dinero real para
«crear evidencia» ni se eliminan controles protectores.

### ST-17 — Un gate de logloss no documenta un criterio MDL

**Evidencia.** El triage describe logloss del trainer como MDL implícito.
La formulación revisada de
[Grünwald](https://arxiv.org/abs/math/0406077) distingue codificación del modelo
y de los datos, o un código universal con complejidad incorporada. Una pérdida
predictiva aislada no especifica ninguno de esos mecanismos.

**Consecuencia.** Podría adjudicarse control de complejidad a un gate que sólo
compara ajuste. La selección repetida de genomas añade además multiplicidad
que no desaparece por llamar MDL al score. Esto no afirma que el trainer
carezca de otros controles: sólo rechaza esa equivalencia no demostrada.

**Cierre requerido.** Especificar código/prior, unidades (nats/bits),
precisión de parámetros, complejidad efectiva y protocolo de evaluación.
Comparar contra el gate existente sin reutilizar el test para elegir penalidad.
No se añade una penalidad numérica arbitraria en esta ola.

### ST-18 — La garantía conformal sigue siendo condicional a su contrato

**Evidencia.** El triage marca Conformal/ACI como incertidumbre calibrada existente.
El informe CF documenta por separado recorte de ACI, feedback retrasado,
selección de outcomes y aceptación por singleton. Que haya código y tests
no prueba todas las hipótesis de cobertura en esa cadena.

**Riesgo.** Confundir cobertura marginal con error condicionado a operar puede
legitimar un veto o tamaño de posición sin evidencia suficiente. La denominación
«calibrada» requiere especificar sobre qué población, ventana y política se mide.

**Cierre requerido.** Mantener los pendientes CF y no cerrarlos por esta ola:
objetivo de cobertura, retrasos, población elegible, abstenciones, dependencia
y soporte por activo/τ. Revalidar sobre decisiones completas, no sólo trades
seleccionados. Esta fila es una referencia cruzada, no un nuevo defecto
independiente que deba sumarse otra vez al total CF.

## 8. Diseño continuo multiactivo y condiciones de transferencia

La propuesta es parametrizar un estado Z(a,τ,t): activo a, horizonte τ>0 e
instante causal t, con incertidumbre y soporte observable. El genoma puede
parametrizar funciones sobre log τ y covariables en vez de seleccionar categorías
scalping/swing. Es una propuesta de contrato, no una propiedad certificada de main.

Un continuo matemático se aproxima con cómputo finito. La malla debe justificar
error, cobertura y coste: refinarla debería estabilizar decisiones dentro de
tolerancias declaradas. No equivale a ejecutar todos los cálculos cada nanosegundo
ni a disponer de cien años de etiquetas para cada activo.

| Etapa | Evidencia exigida | Condición para no promover |
|---|---|---|
| Datos | Identidad, unidad, secuencia, as-of y resolución | Soporte desconocido |
| Operador | Ecuación, estimando, aproximación y coste | Nombre de teoría sin objeto medible |
| Verificación | Casos cerrados, límites, invariantes y adversarios | Sólo escenarios favorables |
| Estadística | Nulo, dependencia, incertidumbre y multiplicidad | Score sin población/comparación |
| Consumo | Extractor y versión compartidos en replay/serving | Feature aislada |
| Promoción | Ablación OOS, costes, capacidad y rollback | Test reutilizado para selección |
| Operación | Riesgo de cartera, controles duros y diagnóstico | Más actividad sin evidencia |

Concatenar firmas individuales no construye automáticamente una firma conjunta.
Hace falta un camino vectorial causalmente sincronizado, tratamiento de
asincronía/ausencias y presupuesto de dimensión. Una correlación promedio tampoco
sustituye una medida de pérdidas conjuntas. No se introduce una cópula ni una red
tensorial sin validar esas convenciones.

### Auditoría de vetos: clases que no deben confundirse

- Integridad: precios no finitos, reloj inválido, identidad ausente. Conservar el
  rechazo y explicar causa; no convertir el dato defectuoso en señal neutra.
- Viabilidad operativa: filtros del mercado, saldo, compatibilidad de órdenes.
  Exigen datos vigentes; no se sustituyen por pesos aprendidos.
- Riesgo: exposición, liquidez, pérdida. Política explícita y autoridad; no
  relajar límites automáticamente por observar menos operaciones.
- Evidencia estadística: muestra, calibración, drift. Expresar incertidumbre y
  registrar abstenciones; observación en sombra no implica ejecutar órdenes.
- Heurística: umbral sin derivación/validación. Candidato a ablación y calibración,
  no a eliminación indiscriminada de todas las barreras.

El registro XLVIII-C de GLM no se modifica. Un catálogo de motivos no prueba que
cada retorno lo emita correctamente ni que sus FP/FN estén medidos: requieren
trazado de consumidores. Este parche no desactiva vetos, límites ni kill-switch.

### Prioridad científica y exclusiones

Firmas: baseline retorno+área frente a representación ampliada con igual coste.
Rough volatility: estimar el proceso correcto de log-volatilidad.
TE: historias/retardos físicos y nulos con dependencia.
MDL: código de modelo más pérdida, con selección anidada.
HJB/BSDE, Koopman, TDA, RG, EVT y cópulas requieren un ciclo propio con estimando,
datos y baseline; no se afirma haber auditado todos sus usos ni probado su utilidad.

La clasificación especulativa de computación cuántica se conserva.
[HHL](https://arxiv.org/abs/0811.3171) presupone acceso/preparación de datos,
simulación de la matriz y condiciones de precisión/condicionamiento. Su salida
es un estado proporcional a la solución: leer todos los componentes clásicos
no es gratis. Una ventaja bajo ese modelo no prueba aceleración end-to-end
de una cartera concreta.

[Grover](https://arxiv.org/abs/quant-ph/9605043) mejora consultas a un oráculo de
búsqueda; construir y ejecutar el oráculo económico también cuesta. No resuelve
cualquier optimización de trading en tiempo polinómico. Para Grover se consultó
metadata/abstract, no todo el cuerpo del trabajo.

Redes tensoriales clásicas, fórmulas inspiradas en física y hardware cuántico son
clases distintas. No se añade hardware, servicio ni simulación. Los problemas
del milenio no se usan como justificación de alpha: la transferencia exige
operador verificable, no semejanza verbal.

## 9. Verificación reproducible

Commit de código: f60e182036f26d44127c22ccb92187db450df4cf.

| Corrida | Alcance | Resultado |
|---|---|---|
| RED original | 12 nuevas pruebas contra fb6412c9 | 7 pasan, 5 fallan |
| Revisión adversarial del parche | Caída 1→1e−10, ln1p demasiado amplio | 1 falla; corregida antes de commit |
| GREEN específico final | 13 contratos ST | 13 pasan |
| Compilación | workspace --all-targets | Pasa en 1m18s, con warnings |
| Regresión ampliada | Seis crates --all-targets | 1.098 pasan, 0 fallan, 6 ignoradas; 87 bloques |
| Replay | backtest-engine --lib | 37 pasan, 0 fallan; 24,50s de ejecución |

Total de las dos suites finales disjuntas: **1.135 aprobadas, 0 fallidas,
6 ignoradas**. Las 13 ST ya están dentro de 1.098. Cinco ignoradas son de
testnet, una del inventario de modelos. Diagnósticos que confirman defectos
abiertos NO se cuentan como reparaciones. No se ejecutó T-1, testnet ignorada,
motor vivo/demo, entrenamiento ni promoción. Golden intacto.

~~~powershell
cargo test -p feature-engine --test path_signature_contract
cargo check --workspace --all-targets
cargo test -p feature-engine -p signal-engine -p risk-engine -p god-engine-core -p execution-engine -p evolution-engine --all-targets
cargo test -p backtest-engine --lib
~~~

Resultados sobre f60e1820, base 49fc995c; no sobre cualquier main posterior.
SHA-256 de bytes locales (puede variar con CRLF respecto de otro checkout):

- Original: 77E4CA054EFBBC92A1EA77D3D0B1D805419AA276B725391D9C922879E14C8D4A.
- Módulo final: 017C0934CBAF2DEF4D4267C71673765CB3B8B4153394097FF17B5C9282C39B68.
- Tests: C19D4C3281BA8FB76A1896FAE405812516B9E993457C17CA2710602A44601F5A.

Investigación mediante Firecrawl: CLI no instalada, conector existente utilizado.
Fuentes bajo .firecrawl/st-audit/, ignoradas por Git; se citan trabajos primarios,
no se incorpora su texto completo al commit.

## 10. Git, coordinación y pendientes

La rama feat/quant-sr-codex-horizonte conserva 6209704a y el informe TH.
Su política de rechazo estricto de τ contradice el fallback intencional de Claude:
no se integra con ST. Babysit exige aclarar ese conflicto de intención, no
seleccionar un lado silenciosamente.

Lectura remota de esta ola:

- [PR10](https://github.com/Jhona-la/Trader-Gemini/pull/10):
  abierto, no draft, head f87ad3fc, no fusionado.
- [PR20](https://github.com/Jhona-la/Trader-Gemini/pull/20):
  abierto, draft, head 43ed1411, no fusionado.
- git ls-remote: main = 303e7ef24f6be0bbde0b6da799dfb078044b4ec6;
  contiene firmas fb6412c9 y registro cff240d7.
- GLM pasó a glm/xlviii-d-transfer-entropy: rama activa que no debe borrarse
  porque su HEAD momentáneo ya esté contenido en main.
- Backups backup-before-cleanup, v7-unificacion-wip y rama TH preservados.

No se emitieron nuevos comentarios ni push/PR de ST. Tras la denegación previa
del control de aprobación sigue pendiente autorización informada para publicar
en el repositorio público. No se elude por otro canal. Un registro local no
implica recepción ni aprobación de Claude/GLM.

La siguiente integración local debe comparar ambos padres y compilar todos los
targets antes del commit. Publicación y merge remoto son estados separados.
No se borra una rama propia pendiente, una ajena activa ni backups exclusivos.

## 11. Hoja de ruta de cierre

1. Revisar e integrar ST con autorización de publicación, CI y recibo remoto.
2. Motivos de invalidez compatibles con el registro GLM (ST-12).
3. Feature versionada con activo, τ, as-of, soporte y reloj (ST-09).
4. Paridad replay/serving antes de modelos nuevos (ST-10).
5. Medir rango informativo y presupuesto multiescala/multiactivo (ST-08/13).
6. Estudios separados para ST-14…18, conservando pendientes CF/OA.
7. Retomar TH con decisión de τ y revisión de replay/multislot.

Hay una mejora comprobada y una auditoría trazable. No hay certificación global,
omnisciencia, lectura semántica de cada archivo ni evidencia de duplicación
consistente del capital. La meta evaluable es crecimiento geométrico con riesgo,
costes, capacidad e incertidumbre explícitos.
