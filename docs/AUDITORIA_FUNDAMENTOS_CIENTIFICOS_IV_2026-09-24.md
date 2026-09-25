# Auditoría de fundamentos científicos IV — aprendizaje servido, microestructura y controles de decisión

Fecha: 24 de septiembre de 2026. Repositorio observado: Trader Gemini. HEAD: `59a76de4`.

Esta ampliación documenta **23 hallazgos adicionales, FMT-072–094**, con evidencia de código, mecanismos de fallo, condiciones de activación y criterios de cierre. No sustituye ni elimina los informes anteriores. Los identificadores son nuevos dentro de la serie FMT; no implican que cada tema sea desconocido para todos los informes históricos. En particular, FMT-094 muestra una reparación anterior que no alcanzó todos los consumidores.

**Estado: auditoría y propuestas; no reparación ni certificación de producción.** Se leyeron completos 15 Rust adicionales, 4.781 líneas, y tramos de conexiones grandes. Las cuatro rondas reúnen 82 Rust distintos de los 289 Rust / 1.119 archivos versionados observados. No se ha revisado íntegramente todo el proyecto. Pasaron 95 tests existentes seleccionados; no prueban ausencia de los defectos descritos. No se modificó código operativo, configuración, modelos, genomas, procesos, armado o ramas. No se ejecutó trading ni una campaña nueva de rentabilidad.

Documentos relacionados: [ronda I](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_2026-09-24.md>), [ronda II](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_II_2026-09-24.md>), [ronda III](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_III_2026-09-24.md>), [ATLAS](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/ATLAS_ANALITICO.md>) e [informe maestro](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/INFORME_FORENSE_MAESTRO.md>).

## 1. Paradigma de grafo vivo: qué se exige realmente

El objeto objetivo sigue siendo un campo multivariante condicionado por activo, estado, escala temporal y calidad de observación; no dos motores llamados scalping y swing. Sin embargo, quitar esas palabras no convierte automáticamente el sistema en continuo, adaptativo o científicamente válido. Esta ronda detecta tres rupturas particularmente importantes:

1. **Representación:** precios equivalentes bajo un cambio de unidad no generan siempre features equivalentes; entrenamiento y validación pueden usar coordenadas distintas.
2. **Aprendizaje:** una actualización por símbolo puede terminar consumiéndose globalmente; congelar un modelo no siempre congela sus estadísticas; el rollback no restaura todo el estado que determina la siguiente actualización.
3. **Decisión:** una señal infinitesimal puede adquirir una magnitud mínima; una interpolación suave puede anular por completo la sensibilidad a un gen; un dato inválido puede transformarse en aparente evidencia o permiso.

Para cada arista del grafo hacen falta, como mínimo, identidad de activo, esquema, unidades, tiempo del evento, tiempo de disponibilidad, generación de modelo/genoma y validez. Un escalar finito no acredita ninguno de esos contratos por sí solo.

### 1.1 Continuidad temporal no significa un bucle por nanosegundo

El intervalo de 1 ns a 100 años abarca aproximadamente 3,16×10¹⁸ en razón de escalas. Es un dominio posible para una representación matemática, no una promesa de observabilidad ni de identificabilidad de parámetros en todos sus puntos. Deben distinguirse:

- La resolución del reloj o del timestamp.
- La frecuencia y calidad de observaciones efectivamente disponibles.
- La evolución latente entre eventos, definida por un modelo.
- La escala de memoria, el horizonte de predicción y el horizonte de ejecución, que no son necesariamente iguales.
- La precisión numérica y el presupuesto de actualización.

Una familia de kernels sobre log-escala puede evaluarse por propagación entre eventos y aproximación controlada. Debe declarar error de truncamiento/discretización y masa espectral no identificada. Repetir un estado un millardo de veces por segundo no crea información. Tampoco hay evidencia aquí que permita atribuir predicciones fiables a cien años. Los límites de latencia y de datos son parte del modelo, no obstáculos que deban ocultarse con nombres.

## 2. Matriz de hallazgos y resolución

P1: defecto de corrección o conexión de alta prioridad en una ruta relevante, aunque su impacto dependa de activación. P2: contrato auxiliar, fragilidad o defecto científico que requiere validación antes de usarlo. “Viva” significa conexión localizada en el código; **no** demuestra que la instancia desplegada esté ejecutando esa rama. “Entrenador” es el ejecutable disponible, no un proceso cuya actividad se haya comprobado. Todos los puntos siguientes están **abiertos**.

| ID | Prioridad | Hallazgo | Alcance comprobado |
|---|---|---|---|
| FMT-072 | P1 | El perceptrón amplifica una señal infinitesimal hasta ±0,15 | Estrategia registrada y consumidor continuo |
| FMT-073 | P1 | Freeze admite mutación y dos contratos de normalización | DarkAlpha; condiciones de arranque y carga |
| FMT-074 | P1 | Validar dimensiones no valida el contrato científico del modelo | Carga/API de DarkAlpha |
| FMT-075 | P2 | Inicialización He/Xavier con varianza incorrecta para la distribución usada | Constructores de capas |
| FMT-076 | P2 | Sanitización de “subnormales” poda pesos normales y altera el artefacto | Carga/inicialización de buffers |
| FMT-077 | P1 | El entrenador carga dimensiones variables y ejecuta índices fijos 54×64×32 | auto_trainer_daemon |
| FMT-078 | P1 | Entrena sin Welford y valida con Welford cuando no hay Scaler | auto_trainer_daemon |
| FMT-079 | P1 | Predicciones ausentes pueden mejorar la pérdida de validación | Gate de promoción del entrenador |
| FMT-080 | P1 | Rollback parcial conserva Adam y estadísticas de una generación rechazada | Ciclo del entrenador |
| FMT-081 | P1 | Piso nominal de precio destruye invariancia entre activos/unidades | Features Omni del vector 34D |
| FMT-082 | P1 | Restar señal MACD a precio produce una feature casi constante | Feature Omni 15 / universal 27 |
| FMT-083 | P1 | Rangos solapados y relojes distintos cambian el significado de volatilidad | Omni, ticks y warmup |
| FMT-084 | P1 | OFI no inicializa profundidad previa y fabrica flujo con cantidades inválidas | Modelo OFI consumido por el núcleo |
| FMT-085 | P2 | Ocho vectores nulos pueden agotar las categorías adaptativas | Clustering auxiliar |
| FMT-086 | P2 | Carga de clustering permite índices fuera del array y parámetros inválidos | Persistencia auxiliar |
| FMT-087 | P2 | Libro L2 conserva colas antiguas incompatibles con sus nuevos totales | API auxiliar L2 |
| FMT-088 | P2 | Asignador epigenético rígido asigna capital aun sin scores válidos | API auxiliar, sin consumidor vivo localizado |
| FMT-089 | P1 | La escasez anula el gen de colchón; dos consumidores usan entradas distintas | Riesgo y apertura del núcleo |
| FMT-090 | P2 | Entropía auxiliar convierte ausencia de datos en observaciones de orden | ShannonEntropyEngine, no el Shannon del núcleo |
| FMT-091 | P1 | Régimen BTC discreto sigue imponiendo un veto global unilateral | Núcleo → risk-engine → allow_trade |
| FMT-092 | P2 | EW-Welford admite alpha inválido y mezcla representaciones al alternar métodos | API de estadísticas |
| FMT-093 | P1 | El guard de portafolio puede devolver permiso con capital NaN | API conectada; ejecución final no demostrada |
| FMT-094 | P1 | Una segunda lectura Hebbiana aún usa la clave errónea y fallback global | Estrategia perceptrón del consenso |

## 3. Descripciones forenses

### FMT-072 — Exploración mínima no equivale a evidencia mínima

**Evidencia:** [PerceptronGateEngine::infer](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/perceptron_gate.rs:27>) aplica signo(s)·clamp(tanh(5·(|s|w−0,5)), 0,15, 1). Con s=0 devuelve cero; con cualquier s positivo suficientemente pequeño devuelve 0,15. Se registra en [el núcleo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:201>) y participa en el consenso continuo.

**Mecanismo y reproducción:** para s=10⁻¹² y w=1, la salida es 0,15: amplificación de 1,5×10¹¹ y discontinuidad en cero. Reducir el peso tras pérdidas no permite que ese voto se acerque continuamente a cero. El comentario lo presenta como exploración, pero no hay allí aleatorización, probabilidad de explorar ni presupuesto de riesgo: es una opinión direccional mínima determinista.

**Impacto:** ruido de signo puede adquirir peso material en la votación. No significa que una señal aislada abra necesariamente una orden: existen otras estrategias y gates. FMT-023 ya documentó que el consenso puede conferir confianza alta a magnitudes débiles; este punto identifica una fuente concreta y distinta de esa amplificación.

**Aprendizaje asociado:** update_weight incrementa o decrementa según el signo del PnL, sin producto entrada-salida ni crédito de la contribución individual. Puede llamarse heurística de refuerzo, pero no se ha demostrado aquí una regla Hebbiana o un gradiente de utilidad. El núcleo le pasa ATR porcentual como std_dev, no una desviación estimada de las recompensas.

**Cierre:** continuidad y comportamiento de abstención explícitos; exploración separada de confianza; curva de ganancia medida; atribución del resultado a la señal realmente usada. Regresión pendiente: barrido ±ε, cero y pesos tras rachas de pérdidas.

### FMT-073 — El congelamiento tiene excepciones que cambian las coordenadas

**Evidencia:** [freeze](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:635>) promete inferencia determinista sin drift. Sin embargo, [predict](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:750>) llama normalize —actualiza y transforma— mientras el contador global sea menor que 500. [predict_for_coin](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:838>) usa estadísticas globales desde count≥20, con calentamiento local y fallback diferentes.

**Contraejemplo de estado:** un canal con observaciones previas [0,2] tiene media 1 y desviación √2. Transformar x=4 con estado fijo da 2,12132; incorporarlo primero da media 2, desviación 2 y z=1. Freeze no garantiza que la misma secuencia se evalúe con la misma función que el artefacto cargado. Para contadores globales entre 20 y 499, predict puede actualizar el global mientras predict_for_coin lo usa sin actualizarlo: dos contratos públicos distintos.

**Límite importante:** con global entrenado y disponible, la corrección MOD2/7-035 sí conserva ese global al completar el warmup local. No se reabre como si siguiera vigente el antiguo cambio obligatorio a estadísticas locales. Tampoco toda actualización causal de normalización es leakage; el defecto es prometer inmutabilidad y comparar funciones que cambian sin registrar esa diferencia.

**Impacto:** dependencias del orden de inferencia, warmup y validación; reproducibilidad incompleta para modelos fríos o sin Scaler. Cierre: separar warmup, adaptación y serving; serializar el estado necesario; congelamiento comprobable por igualdad de estado; pruebas de paridad entre las dos APIs para las condiciones declaradas.

### FMT-074 — Coherencia de tensores no equivale a modelo válido

**Evidencia:** [DenseLayer::is_valid](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:55>) comprueba tamaños; [layers_valid/validate](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:606>) comprueba encadenamiento. No exige finitud de pesos/sesgos, salida escalar, integridad de Scaler, estado estadístico válido o identidad del esquema de features. [Scaler::scale](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:441>) deja sin transformar dimensiones que excedan sus vectores.

**Mecanismo:** una red con dos salidas puede pasar validate aunque predict y fit sólo utilicen out[0]. Un Scaler incompleto produce una mezcla de entradas estandarizadas y crudas. En una capa ReLU, una suma NaN cae al else y se convierte en cero; el chequeo de probabilidad final puede no revelar esa corrupción. La carga bincode admite representaciones no finitas que el validador dimensional no excluye; posteriormente init_buffers puede reemplazarlas, cambiando el modelo en vez de rechazarlo.

**API de entrenamiento:** fit comprueba dimensiones de las capas, pero no la finitud de learning_rate/targets ni la longitud de channel_normalizers antes de indexarlo. Un estado público incoherente o un target NaN puede corromper la actualización. No se afirma que los datasets actuales contengan esos valores.

**Cierre:** distinguir validación estructural, numérica y semántica; declarar esquema y objetivo; rechazar o migrar explícitamente estados incompatibles; fallar antes de modificar pesos. Conservar las guardas D-613/D-614, que sí evitan lecturas fuera de límites por tensores truncados.

### FMT-075 — He/Xavier: se confunde desviación estándar con semiancho uniforme

**Evidencia:** [constructores](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:62>) generan pesos aproximadamente uniformes entre −scale y scale, con scale=√(2/n) o √(2/(n+m)). Para U(−a,a), Var=a²/3. Por tanto, el primer constructor tiene varianza nominal 2/(3n), no 2/n; el segundo, 2/(3(n+m)), no 2/(n+m). Además, [DarkAlphaEngine::new](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:647>) comenta Tanh pero usa esas capas con ReLU.

El análisis de [He et al.](https://arxiv.org/abs/1502.01852) deriva para ReLU una condición de preservación de escala bajo supuestos de independencia y simetría; no afirma que cualquier inicialización distinta imposibilite aprender. Para n=34, las varianzas nominales comparadas son 0,0588235 y 0,0196078. El PRNG determinista introduce además una realización finita, no una garantía de independencia exacta.

**Impacto y límite:** escala inicial de activaciones/gradientes diferente de la anunciada y experimentos mal especificados. No se cuantificó pérdida de rentabilidad ni se probó que ésta sea la causa dominante del rendimiento actual. Cierre: distribución, ganancia y activación coherentes; tests de momentos y propagación de escala; baseline reproducible. El test existente sólo exige pesos finitos.

### FMT-076 — El filtro de subnormales modifica valores normales

**Evidencia:** [sanitize_denormals](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/dark-alpha-engine/src/lib.rs:111>) pone a cero todo peso/sesgo con magnitud menor que 10⁻⁷. Ese límite no identifica subnormales f64: el mínimo positivo normal es aproximadamente 2,225×10⁻³⁰⁸. El método se ejecuta al inicializar buffers y cargar modelos.

**Consecuencia:** guardar y cargar puede cambiar la función aprendida; un peso normal de 10⁻⁸ se elimina. Si se pretende pruning o regularización, debe ser un tratamiento del modelo con error medido, no una operación supuestamente neutra de mantenimiento de memoria. La sensibilidad de un peso depende de su entrada y de capas posteriores; la magnitud aislada no acredita irrelevancia.

**Cierre:** separar rechazo de corrupción, tratamiento real de subnormales y compresión. Verificar roundtrip de predicciones con tolerancia justificada y registrar cualquier transformación del artefacto. El test actual comprueba eliminación de valores del orden de 10⁻³¹⁶; no comprueba conservación de valores normales pequeños. No se atribuye el factor de latencia mencionado en comentarios a una medición realizada en esta auditoría.

### FMT-077 — El entrenador admite un modelo y presupone otro

**Evidencia:** [auto_trainer_daemon](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:65>) carga cualquier DarkAlpha estructuralmente válido. Si no puede cargarlo crea 54×64×32×1, pero no exige esa arquitectura después de una carga satisfactoria. Adam y los bucles [de forward](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:220>) tienen dimensiones literales.

**Reproducción estructural:** un modelo válido 34×64×32×1 tiene 2.176 pesos en la primera capa. El índice i·54+j llega a 2.176 en i=40, j=16 y excede el vector. Antes de ese punto también interpreta incorrectamente el layout. Hay modelos 34D previstos por default_model y por la inferencia del núcleo; no es un formato imposible. No se arrancó el daemon para provocar el fallo.

**Impacto:** abortar el entrenamiento, acceso fuera de rango con panic y actualización incompatible antes de fallar. Cargo define panic=abort en release. El proceso afectado sería el entrenador ejecutado, no se deduce automáticamente que el motor de trading comparta ese proceso.

**Cierre:** contrato explícito de arquitectura y features; derivación de todos los buffers desde ese contrato; migración o rechazo antes del forward. Test de integración pendiente con modelos 12D/34D/54D y anchuras alternativas.

### FMT-078 — Paridad train/serve incompleta en la rama sin Scaler

**Evidencia:** [escalado del entrenamiento](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:213>) sólo actúa con Some(Scaler). Si es None, usa features crudas. [val_bce](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:160>) invoca predict, que aplica Welford y winsorización en esa misma condición. Un modelo nuevo empieza precisamente con scaler=None.

**Mecanismo:** los gradientes se calculan para fθ(x), mientras el criterio de selección mide fθ(Ts(x)), donde Ts incluye estadísticas y compresión de colas. Aunque se compartan pesos, no es la misma hipótesis. Además, la validación previa puede actualizar normalizadores, y los normalizadores no se entrenan mediante el forward manual del daemon.

**Impacto:** el aprendizaje puede optimizar una representación que nunca sirve; la mejora aparente puede depender del movimiento del transformador. El comentario de paridad no cubre esta rama. Este hallazgo es distinto del desajuste de etiquetas documentado anteriormente.

**Cierre:** una transformación canónica versionada, evaluada con estado congelado y compartida por entrenamiento/validación/inferencia; comparar vectores transformados y logits, no sólo probabilidades finitas. Incorporar normalización no basta si se ajusta con el holdout o cambia entre baseline y candidato.

### FMT-079 — Ausencia de predicción contabilizada como pérdida cero

**Evidencia:** [val_bce](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:161>) suma BCE sólo cuando predict devuelve Some, pero divide entre todas las filas. Si todas las predicciones son None, el resultado es cero, que aparenta perfección. [La promoción](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:375>) sólo rechaza si val_after≥val_before.

**Contraejemplos:** con diez filas, cinco predicciones perfectas y cinco ausentes, la pérdida no penaliza esas cinco ausencias; con todas ausentes devuelve 0. Un candidato que pase de una BCE positiva a ausencia total puede superar el gate. Por separado, en aritmética IEEE una comparación NaN≥a es falsa, por lo que el predicado no rechaza un valor NaN si llega hasta él.

**Alcance:** se demuestra la lógica del gate, no la promoción observada de un artefacto corrupto. Los pasos de serialización o carga posteriores podrían fallar; eso no convierte el criterio previo en correcto. El parser admite f64 parseables sin comprobar finitud, por lo que también falta una barrera de datos.

**Cierre:** métricas con cobertura explícita, rechazo de cualquier comparación inválida, número de observaciones elegibles común y criterio de abstención definido. La métrica no debe mejorar por perder capacidad de inferir.

### FMT-080 — Rollback restaura pesos, no el proceso de aprendizaje

**Evidencia:** el [snapshot](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:155>) guarda tres capas; el [rechazo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/auto_trainer_daemon.rs:377>) sólo restaura esas capas. Momentos Adam, contador t y estadísticas modificadas por validación permanecen. La iteración siguiente reutiliza el CSV completo y el mismo esquema de holdout.

**Mecanismo:** la siguiente actualización parte de pesos antiguos con momentos de gradientes de una trayectoria rechazada. La unidad de rollback debería ser el estado que determina el futuro: pesos, transformador, optimizador y metadatos de evidencia. Conservar momentos podría ser una decisión deliberada, pero entonces no es restaurar el incumbente y exige validación específica.

**Problemas vinculados:** se reutiliza un holdout para elegir generaciones repetidamente, extensión concreta de FMT-051/T22. No se observa ledger de consultas ni delimitación de datos nuevos. La pérdida de entrenamiento se divide por num_samples aunque sólo se entrenan train_size filas, subestimando su media por ese factor; no es el gate principal, pero distorsiona telemetría.

**Cierre:** snapshot transaccional completo; replay de una generación rechazada debe dejar idéntico el estado posterior del incumbente. Identificar datos nuevos, consultas de validación y presupuesto de selección. Restaurar sólo pesos no resuelve la contaminación del criterio.

### FMT-081 — Piso de un dólar en una feature supuestamente relativa

**Evidencia:** [OmniStrategyEngine::extract_features](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/omni_strategies.rs:150>) define safe_last_price=max(last_price,1). Se usa para normalizar MACD, ATR, distancias y rango. Las 22 features se integran en el vector universal mediante [get_universal_features](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:815>).

**Contraejemplo:** una serie con precio 100 y ATR 1 produce ATR normalizado 0,01. La misma serie escalada por 10⁻⁴ tiene precio 0,01 y ATR 0,0001, pero produce 0,0001. El riesgo relativo es idéntico; la feature cambia cien veces. Para activos de precio muy bajo, el piso puede atenuar sistemáticamente múltiples canales.

**Impacto:** la representación aprende denominación nominal además de dinámica; transferencia entre activos y evaluación de genomas dejan de ser comparables. Un normalizador posterior no garantiza reparación, especialmente si hay thresholds, clipping, dispersión casi nula o mezcla de activos.

**Cierre:** denominadores con unidades y dominio declarados; estado inválido separado de precio válido pequeño. Regresión metamórfica: escalar OHLC por c>0 y verificar los canales declarados adimensionales, sin exigir invariancia a canales deliberadamente monetarios.

### FMT-082 — Precio menos señal MACD no es un oscilador MACD

**Evidencia:** la [feature de índice 15](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/omni_strategies.rs:184>) calcula (precio−EMA_señal_MACD)/safe_last_price y la recorta a [−0,1,0,1]. MACD_signal es una media de diferencias entre medias de precio; no una media del nivel de precio.

**Contraejemplo:** en un mercado plano de precio 100, macd_line y macd_signal valen cero. La expresión da 1 antes del clipping y 0,1 después, permanentemente. En una tendencia moderada, la señal MACD suele ser pequeña respecto al precio y el canal sigue saturado. Con precio 0,01, el defecto FMT-081 lo transforma además en 0,01: una aparente información de activo que procede de la denominación.

**Consecuencia:** una de las entradas 34D —índice universal 27— puede ser casi constante pese al nombre de indicador. Constancia no es automáticamente dañina si el modelo la ignora; sí contradice la información anunciada y consume capacidad, calibración y pruebas.

**Cierre:** definir si se desea histograma MACD, distancia a EMA de precio u otra cantidad. Cada una tiene una ecuación distinta. Medir varianza, saturación y contribución incremental por activo/escala; no cambiarla sin migrar/reentrenar los consumidores.

### FMT-083 — Volatilidad de rangos sin intervalo homogéneo

**Evidencia:** [Omni::update](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/omni_strategies.rs:79>) actualiza ATR y el proxy Parkinson con cada llamada. En [process_tick](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:565>) recibe el high/low acumulado de una vela interna y se llama en cada tick. En [process_kline](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:689>) recibe el cierre actual con extremos de la vela anterior.

**Mecanismo:** el mismo rango acumulado se reutiliza muchas veces como si fueran observaciones del estimador. La cantidad de ticks y el orden de ampliación del rango cambian el peso efectivo de un mismo intervalo. Esos rangos se solapan y no tienen la misma duración; el factor 1/(4 ln 2) no corrige esa diferencia. En warmup se combina además un cierre con extremos de otro intervalo.

**Interpretación correcta:** puede existir una feature heurística de rango intravela actualizada por evento; lo que no se acredita es una varianza por unidad de tiempo comparable entre modos. RSI/MACD usan periodos por llamada, no segundos; el “local extrema decay” usa 10⁻⁶ por llamada. Una discretización finita no es el problema: lo es no declarar el reloj y el objeto estimado.

**Cierre:** contrato de intervalo, timestamps y causalidad; separar barras cerradas de actualización parcial; ensayar tapes con el mismo camino y distinto empaquetado de eventos. La prueba debe preservar la salida sólo para los canales cuya definición lo exija. Este punto amplía, no renumera, los problemas espectrales por reloj de FMT-004.

### FMT-084 — OFI: profundidad inicial omitida y dato inválido convertido en flujo

**Evidencia:** [OFIModel::update](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/microstructure.rs:118>) inicializa precios/cantidades en el primer evento y retorna sin asignar prev_depth. El segundo evento usa max(current_depth,prev_depth). Cantidades no finitas se convierten en cero, mientras precios inválidos pueden conservar el precio previo.

**Contraejemplo de profundidad:** primer libro con cantidades bid=999,5 y ask=0,5; segundo con bid=0,5 y ask=0,5, precios iguales. El flujo neto es −999. La profundidad estabilizadora debería incluir 1.000, pero queda en 1 y el cociente se recorta a −10. Con EMA α=0,1, la salida es −1 en vez de −0,0999 para la regla documentada. No se propone cambiar el estimador: se exige inicializar su propio estado.

**Dato inválido:** si sólo desaparece la cantidad bid por corrupción, sustituirla por cero simula una cancelación. El dato siguiente válido puede simular reposición. Recuperar finitud evita envenenamiento persistente, pero no elimina esos flujos artificiales.

**Cierre:** inicialización consistente, validez por snapshot y política explícita de rechazo/reconciliación. Conservar el arreglo D-709 de precios saneados; añadir tests de segundo evento con colapso de profundidad y de ausencia de datos sin transacciones inventadas.

### FMT-085 — El clustering aprende ocho categorías de ausencia

**Evidencia:** [AdaptiveResonanceClustering](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/microstructure.rs:454>) normaliza el vector cero a cero. Su similitud con cualquier prototipo nulo es cero y no alcanza vigilancia positiva. Cada llamada crea categoría nueva con match=1 hasta llegar a ocho; después asigna la más cercana sin actualizar si no hay resonancia.

**Reproducción lógica:** ocho llamadas con [0,0,0,0] llenan la capacidad. Un vector informativo posterior tiene producto escalar cero con todos los prototipos; se asigna al índice inicial con score cero y no aprende. NaN/Inf se convierten primero en cero. Una secuencia de ausencia puede bloquear permanentemente la supuesta autoorganización.

**Alcance:** búsqueda de consumidores localiza definición, reexport y test, no integración viva de este clasificador. El efecto demostrado es del algoritmo disponible, no un bloqueo ya observado del motor. ART-2 es además una denominación más fuerte que lo acreditado por un clustering coseno con capacidad fija.

**Cierre:** abstención para vectores sin información, inicialización separada de evidencia de resonancia y política de renovación/olvido con evaluación de estabilidad. Los tests actuales sólo cubren dos patrones informativos.

### FMT-086 — Persistencia de clustering sin validar invariantes

**Evidencia:** [load_from_disk](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/microstructure.rs:536>) acepta active_categories sin exigir ≤8, valores f64 parseables sin finitud y filas faltantes que quedan en cero. classify_and_adapt recorre 0..active_categories e indexa un array de ocho.

**Contraejemplo estructural:** un archivo parseable con active_categories=9 puede cargar y provocar panic al clasificar. No se generó ni se consumió un archivo corrupto durante esta auditoría. Tasas/vigilancia cargadas tampoco pasan por los clamps del constructor; un constructor seguro no protege automáticamente una ruta de deserialización.

**Impacto:** falta de recuperación segura ante artefactos truncados o incompatibles y discontinuidad entre estado guardado/cargado. Mantener prototipos sin comprobar norma también cambia el significado de “coseno”.

**Cierre:** validación completa y atómica antes de reemplazar estado, categoría dentro de capacidad, filas completas, parámetros admisibles y norma controlada. Hash y versión ayudan a identidad, pero no sustituyen validación semántica.

### FMT-087 — El libro L2 representa dos snapshots incompatibles

**Evidencia:** [update_from_raw_depth](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/microstructure.rs:591>) sobrescribe sólo los niveles presentes. Si antes hubo 20 niveles y después uno, las otras 19 entradas permanecen, aunque total_bid_usd/total_ask_usd se recalculan sólo con el nuevo input.

**Consecuencia:** sum(bid_notionals) puede diferir del total_bid_usd publicado. Un consumidor de arrays ve liquidez antigua; otro de totales ve la actual. La validación sólo comprueba primer precio de cada lado y spread; no acredita orden completo, duplicados o overflow de p·q.

**Alcance:** esta API normalizada es auxiliar en las búsquedas realizadas; el parser vivo y su sincronización de secuencias no se certifican por revisar esta clase. SpoofingDetector está deliberadamente desconectado según su propio comentario: no se debe conectarlo a un único nivel fingiendo un libro completo.

**Cierre:** limpiar niveles ausentes, registrar longitud efectiva, comprobar identidad entre arrays/totales y definir si la entrada es snapshot o delta. Un método de snapshots no puede interpretarse como reconciliación de deltas sin secuencias y semántica adicional.

### FMT-088 — Asignación epigenética no identifica oportunidad ni factibilidad

**Evidencia:** [allocate_epigenetic_universe_margin](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/epigenetic_capital_alloc.rs:100>) escoge K=2/5/10/N con cortes 30/100/300. El equivalente top10 tiene cortes 30/100. Todo peso seleccionado se eleva a 0,01 y luego se normaliza a suma uno. Capital inválido se reemplaza por 13.

**Contraejemplos:** scores todos cero o negativos producen asignación positiva; capital cero/NaN también. Para $13 y scores muy desiguales, seleccionar dos activos no garantiza que ambos superen un mínimo de notional. No se usan mínimos por símbolo, apalancamiento, lotes, comisiones ni reserva de efectivo. min(scores.len,methylation.len) trunca silenciosamente universos desalineados.

**Interpretación:** la metilación es aquí un multiplicador 0,5–1,5, no una solución de optimización de portafolio ni una garantía de payoff. Su continuidad por score no elimina la rigidez del top-K ni convierte scores en retornos esperados.

**Alcance y cierre:** API sin consumidor vivo localizado. No atribuirle exposición real actual. Antes de conectarla: identidad del universo, opción cash, factibilidad discreta de exchange, restricciones conjuntas y objetivo estimable. No basta sustituir los escalones por una sigmoid si se mantienen los defectos de factibilidad.

### FMT-089 — Una curva suave puede silenciar el genoma

**Evidencia:** [margin_cushion](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/capital_regime.rs:147>) calcula, dentro del rango del gen, c(g,w)=(1−w)g+0,98w. [micro_weight](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/capital_regime.rs:94>) fija w=1 cuando capital/min_notional≤3.

**Resultado algebraico:** ∂c/∂g=1−w. Con w=1 es exactamente cero: genes 0,50, 0,70 y 0,90 producen todos 0,98. Se conserva continuidad respecto al capital, pero desaparece identificabilidad/efecto del gen en ese contexto. El comentario que afirma que el colchón “sale SIEMPRE del gen” omite esta degeneración.

**Conexión adicional:** [riesgo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:317>) calcula escasez con allocated_capital y config.min_notional; [núcleo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:4618>) usa current_cap y mínimo efectivo por símbolo. Compartir función no garantiza compartir resultado. Esas cantidades pueden tener propósitos distintos, pero deben diferenciarse como restricciones y explicarse al atribuir rechazos.

**Impacto:** genomas evaluados con holgura pueden no tener el mismo margen de acción en demo/producción; no es prueba suficiente de la causa total del gap. Cierre: medir Jacobiano gen→fenotipo→acción condicionado por capital/símbolo/escala; distinguir gen silenciado de gen perjudicial; declarar qué límites son invariantes de seguridad. No se recomienda relajar reservas para “activar” genes.

### FMT-090 — Shannon: datos ausentes transformados en orden aparente

**Evidencia:** [ShannonEntropyEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/shannon_entropy.rs:29>) convierte no finitos en cero y los contabiliza. Tras 20 entradas inválidas puede devolver entropía cero con warmup completo. El test de NaN exige precisamente resultado finito tras repetir entradas inválidas.

**Mecanismo:** se estima la dispersión de un histograma marginal, no la tasa de entropía del proceso ni transferencia entre activos. Un ciclo perfectamente predecible que visita todos los bins puede tener entropía marginal alta; un stream corrupto imputado al mismo bin puede tener entropía baja. Por eso no debe interpretarse automáticamente como desorden dinámico o evidencia de predictibilidad.

**Reloj:** 0,995 es retención por llamada. Su constante de decaimiento es −1/ln(0,995)≈199,50 eventos: a 10 eventos/s equivale a unos 19,95 s; a 1.000 eventos/s, a 0,1995 s. La clase no recibe Δt, por lo que no acredita un decaimiento físico por nanosegundo.

**Alcance:** no confundir esta clase auxiliar con ShannonEntropy de math_kernels usado por stateful_engine; esa implementación distinta no se releyó completa aquí. Cierre: calidad y masa válida separadas; unidades de memoria explícitas; elegir entropía marginal, condicional o tasa según la pregunta. Véase T26.

### FMT-091 — El régimen continuo local convive con un veto discreto global

**Evidencia:** [stateful_engine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/stateful_engine.rs:764>) devuelve MarketRegime::Continuous. Pero [GodEngineCore](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:747>) publica otro régimen global a partir de BTC: cortes de tendencia ±0,015 y thresholds de Hurst. [PortfolioOrchestrator::allow_trade](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/orchestrator.rs:110>) veta toda intención long cuando ese régimen es Crash.

**Problema científico:** clasificar Hurst bajo como “Chaotic / Mean Reverting” mezcla antipersistencia, reversión y caos, que no son equivalentes. No se calcula allí incertidumbre del estimador, probabilidad posterior del régimen ni sensibilidad multiactivo. Una regla sistémica puede ser deliberada, pero debe justificarse como restricción de seguridad y no presentarse como inferencia universal adaptativa.

**Rigidez y alcance:** el veto es unilateral y global aunque las oportunidades, coberturas y escalas difieran. Su mera existencia no es un bug por ser discreto: órdenes y límites duros son legítimos. El fallo es la afirmación de universalidad frente a una taxonomía heurística con consecuencias decisorias y sin contrato probabilístico. La escala de tendencia viene además de EMAs por evento.

**API alternativa:** RegimeDetector en risk-engine usa correlación <0,2 como “caos”, sin consumidor vivo localizado. No se debe atribuir ese criterio concreto al núcleo: son implementaciones distintas. Cierre: rastrear productor correcto, justificar política, medir falsos vetos y reacción al riesgo; una posterior de cambio como T07 no autoriza por sí sola a retirar el cortacircuitos.

### FMT-092 — Dos representaciones de varianza mezclables sin transición válida

**Evidencia:** [WelfordOnline::update_decay](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/feature-engine/src/welford.rs:53>) sólo exige alpha>0 finito, no alpha≤1. Para estado nuevo, val=1 y alpha=2 produce media 2, m2=−2 y count=0,5. variance devuelve max(m2,0): oculta la invalidez como varianza cero.

**Segundo mecanismo:** update_decay transforma m2 a varianza y marca is_decay=true. Si después se usa update con count<2.000, éste entra en la rama del Welford acumulativo y trata el mismo m2 como suma de cuadrados. variance sigue interpretándolo como varianza por is_decay. La API permite alternar operaciones que no conservan significado.

**Alcance:** no se localizaron llamadas productivas a update_decay; update estándar sí tiene consumidores. No se afirma que las estadísticas actuales hayan sufrido esa secuencia. Es un defecto de contrato latente relevante antes de conectar adaptación de alpha al genoma.

**Cierre:** estados o tipos distintos para acumulación y decaimiento, conversión explícita, dominio de alpha y semántica de count. Pruebas de equivalencia con cálculo ponderado de referencia y alternancia de métodos. Clampear una varianza negativa no sustituye rechazar una transición inválida.

### FMT-093 — Permiso de portafolio con estado financiero desconocido

**Evidencia:** [allow_trade](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/orchestrator.rs:100>) valida required_margin, pero comprueba capital únicamente con capital≤0. Con capital=NaN, esa comparación y las posteriores “exposición > capital·límite” son falsas. Con régimen no Crash, margen solicitado válido y sin posiciones, puede llegar a true. Márgenes de posiciones no finitos también contaminan sumas sin una comprobación final.

**Impacto y límite:** un guard conectado carece de una precondición necesaria para dar permiso. Otros controles aguas arriba/abajo pueden impedir la orden; no se demostró apertura efectiva con NaN. Cada frontera de seguridad debe rechazar o señalar desconocimiento por su propio contrato, en lugar de depender silenciosamente de otra capa.

**Otros desajustes de significado:** la exposición sumada es margen, no delta de mercado, a pesar del comentario de net delta. Posiciones con margen igual y apalancamiento distinto no tienen igual exposición nocional. El límite se deriva de global_max_drawdown mediante 1−min(g,0,2), por lo que un gen de drawdown más pequeño permite una proporción mayor de margen: esa transferencia de significado requiere justificación independiente.

**Cierre:** estado financiero finito y snapshot coherente, unidades explícitas, límites de margen y delta separados, motivos de rechazo trazables. Test pendiente de capital NaN/Inf, márgenes inválidos y portfolios con igual margen pero distinta exposición.

### FMT-094 — La reparación Hebbiana no alcanzó el segundo consumidor

**Productor:** [cierre de operaciones](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:1904>) actualiza {símbolo}_hebbian_weight y también perceptron_hebbian_weight global. Una lectura del núcleo fue corregida en [MOD2/7-005](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:2125>) para consultar la clave por símbolo.

**Consumidor omitido:** [PerceptronGateEngine](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/signal-engine/src/perceptron_gate.rs:123>) sigue pidiendo “perceptron_hebbian_weight” mediante get_scoped_parameter. La [resolución real](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/omniscient-registry/src/lib.rs:195>) busca {símbolo}_perceptron_hebbian_weight, después c{id}:perceptron_hebbian_weight y finalmente el global. No hay un alias implícito a {símbolo}_hebbian_weight.

**Contraejemplo de conexión:** BTC posee peso 1,2 bajo BTCUSDT_hebbian_weight. Cierra DOGE y escribe 0,6 en su clave y en el global. Sin otro productor de la clave larga de BTC, el perceptrón de BTC consume 0,6; el otro consumidor del núcleo sigue leyendo 1,2. La búsqueda en crates/src no encontró un escritor de la clave larga. No se inspeccionaron valores de una sesión viva para afirmar que el escenario ya ocurrió.

Como réplica numérica, para señal candidata 0,8, la fórmula del perceptrón da aproximadamente 0,980096 con peso 1,2 y 0,15 con peso 0,6. El cambio de símbolo que alimenta el fallback puede alterar sustancialmente el voto sin que cambie esa señal candidata. No es una medición de órdenes ni PnL.

**Impacto:** aislamiento aparente por símbolo pero acoplamiento real al orden de cierres; el mismo aprendizaje entra con dos valores en el grafo. Una regresión que prueba sólo la lectura ya corregida deja pasar el defecto.

**Cierre:** contrato canónico de claves, política explícita de fallback y test productor→todos los consumidores con cierres intercalados de dos símbolos. No es necesario eliminar el aprendizaje compartido si se desea; sí distinguirlo del aprendizaje local y validarlo como transferencia, no como accidente.

## 4. Mejoras existentes que deben conservarse

No todos los nombres heredados implican dos motores. evaluate_scalp_consensus_for_coin y evaluate_swing_consensus_for_coin redirigen al mismo consenso continuo; PerceptronGate y SwingConformalFilter declaran horizonte Continuous. El núcleo usa directamente evaluate_continuous_consensus_for_coin. Los wrappers duales todavía evalúan dos veces el mismo conjunto si alguien los invoca; es deuda auxiliar de coste/semántica, no prueba de dos motores vivos en la ruta localizada.

Se conservarían también las guardas de dimensión de DarkAlpha, el filtro por dirección conformal, la aproximación numérica de p bilateral normal probada, la corrección de OFI ante precios NaN y la continuidad de micro_weight. Sus tests pasan. Eso no demuestra normalidad del z-score recibido, cobertura conformal universal, calibración de confianza o conservación del significado entre productores y consumidores.

La inferencia DarkAlpha del núcleo también restringe explícitamente el modelo entrenado en BTC a BTC; fuera de ese símbolo devuelve voto neutral. Los problemas de representación multiactivo no se presentan como prueba de que esa red concreta siga infiriendo sobre todas las monedas. La rama de modelos 12D/34D/54D sí confirma la relevancia de exigir dimensiones en el entrenador.

El nombre “cuántico” no convierte operaciones f64, ReLU, histograma o producto escalar en computación cuántica. Las objeciones previas FMT-018–024 y T19 siguen vigentes; esta ronda no aportó evidencia de hardware cuántico ni de una ventaja computacional certificada.

## 5. Integraciones teóricas propuestas y explicación de los cálculos

Son propuestas, no funciones implementadas ni mejoras de rentabilidad acreditadas. T01–T24 permanecen en los anexos previos. Añadir una teoría exige un fallo objetivo, contrato de datos, estimador identificable, coste, baseline y criterio de rechazo.

### T25 — Transporte de coordenadas para adaptación sin cambiar silenciosamente el predictor

**Problema:** FMT-073/078/080. Si un modelo usa z=D⁻¹(x−μ), modificar μ o la diagonal D cambia sus entradas aunque los pesos permanezcan iguales. Adaptar un normalizador es modificar parte del modelo.

Para una primera capa afín a=Wz+b, una transformación de estadísticas (μ,D)→(μ′,D′) permite conservar exactamente su preactivación, en aritmética ideal y con escalas positivas, mediante:

~~~text
W′ = W D⁻¹ D′
b′ = b + W D⁻¹(μ′ − μ)
W′ D′⁻¹(x − μ′) + b′ = W D⁻¹(x − μ) + b
~~~

**Qué significa:** los pesos se transportan al nuevo sistema de coordenadas. Esto separa adaptación de representación de aprendizaje predictivo. La igualdad se deriva por sustitución; no implica que el predictor sea bueno ni que una nueva distribución esté cubierta por la calibración anterior.

**Límites decisivos:** winsorización, clipping, features muertas o desviación cero rompen la transformación afín global. No se puede aplicar esta identidad a toda la tubería actual sin separar su parte no lineal. Los momentos del optimizador también están expresados en coordenadas: deben transformarse con justificación o reinicializarse bajo una política evaluada. Cambiar pesos sin tratar Adam recrearía FMT-080.

**Integración propuesta:** nodos distintos para estado estadístico, transformación versionada y función predictiva; commit atómico de sus versiones. Coste O(d·h) al reparametrizar la primera capa, no necesariamente en cada tick. Validación: igualdad de logits en un conjunto de referencia antes/después, prueba de colas no afines, y posterior evaluación prequential independiente. Rechazar la propuesta si el error de transporte o su coste excede el presupuesto declarado.

### T26 — Información dirigida en tiempo continuo para auditar aristas del grafo

**Problema:** FMT-090, además de los problemas de causalidad y sincronización de rondas anteriores. H(X) describe dispersión marginal; H(Xfuturo|Xpasado) describe incertidumbre remanente; la información condicional añadida por el pasado de Y plantea una pregunta de conexión distinta. Ninguna de estas cantidades equivale automáticamente a retorno operable.

[Spinney, Prokopenko y Lizier](https://arxiv.org/abs/1610.08192) desarrollan transferencia de entropía en tiempo continuo usando medidas sobre trayectorias y cocientes de Radon–Nikodym, con aplicaciones a procesos de salto/puntuales. Un caso puntual simple puede expresarse como log-cociente de likelihoods:

~~~text
T[Y→X; 0,T] =
  Σ(ti en eventos de X) log(λX|X,Y(ti) / λX|X(ti))
  − ∫₀ᵀ [λX|X,Y(u) − λX|X(u)] du
~~~

Las intensidades tienen unidades de eventos/tiempo; cociente y contribución integrada son adimensionales. El término de espera importa: considerar sólo eventos omite información de la ausencia de eventos. El cociente requiere soporte compatible e intensidades admisibles; la formulación general impone condiciones sobre medidas/historias. No se deben usar intensidades futuras.

**Diseño propuesto, no conclusión del paper:** comparar dos modelos causales de intensidad sobre el mismo tape: uno con historia del objetivo y otro con historia adicional de la fuente. Añadir contexto común observado, incertidumbre, timestamps de disponibilidad y validación fuera de muestra. En el grafo, etiquetar una arista como “incremento predictivo condicional medido”, no como causalidad económica demostrada.

**Coste y refutación:** el coste depende del número de aristas, memoria y estimador; no es O(1) para un grafo creciente. Comenzar con candidatos escasos fuera del hot path, comparar contra Hawkes multivariante y modelos nulos, y controlar búsquedas múltiples. Rechazar aristas cuyo incremento desaparezca con contexto común, cambio de periodo o costes de ejecución. La ventaja predictiva de un feed puede no ser explotable tras su latencia de recepción.

### Ampliación de T07 — Posterior de cambio en lugar de renombrar umbrales

[Adams y MacKay](https://arxiv.org/abs/0710.3742) infieren una distribución de run length —observaciones desde el último cambio— mediante un modelo predictivo y una función de hazard. Su formulación supone independencia de parámetros entre segmentos. No basta poner “Bayes” al resultado de comparar Hurst con un literal.

**Aplicación propuesta:** modelar cambio en residuos/calibración o intensidades, no convertir automáticamente esa probabilidad en Long/Short. Mezclar predicciones sobre hipótesis de duración preserva incertidumbre sobre el estado. Una implementación exacta sin poda puede acumular hipótesis; limitar a K requiere registrar masa descartada/error y cuesta en función de K y del modelo de observación.

El prior de duración y el likelihood deben estimarse o justificarse: mover umbrales al genoma no elimina arbitrariedad si el objetivo de entrenamiento es inválido. Medir demora de detección, falsas alarmas y daño de resets; compararlo con un detector sencillo. Esta es una concreción de T07, no una teoría nueva contada dos veces.

### 5.1 Milenio, física y cálculo cuántico: criterio de transferencia

Se volvió a consultar el [catálogo oficial de Clay](https://www.claymath.org/millennium-problems/): distingue Navier–Stokes en “Active problems”, cinco entradas en “Unsolved problems” y Poincaré en “Solved problems”. No se deduce de esa clasificación que una prueba haya sido verificada por esta auditoría. Tampoco todos los Problemas del Milenio son ecuaciones importables.

La transferencia útil es de herramientas con un contrato identificable: estabilidad y reducción de modelos para una dinámica definida; complejidad y aproximación para búsqueda restringida; análisis de operadores para modos observables; geometría de sensibilidad para genes. Importar una fuerza de fluidos sin variables, fuentes, unidades y condiciones de frontera identificadas no demuestra una ley de precios. Hodge discreto en un grafo no equivale a la conjetura de Hodge; una descomposición tensorial clásica no acredita ventaja cuántica.

**Criterio de inversión científica:** integrar sólo cuando una hipótesis más simple falla de manera reproducible y la nueva teoría predice qué mejora y cuándo debe dejar de funcionar. Para esta base, cerrar FMT-077/078/079/089/094 tiene prioridad sobre añadir ecuaciones cuya evidencia seguiría atravesando esas conexiones rotas.

## 6. Visualización diagnóstica: raíz, decisión y terminal

~~~mermaid
flowchart TD
  R["Raíz: eventos, símbolo, reloj y calidad"] --> O["OFI / Omni / normalización"]
  O -->|"FMT-081–084: unidad, intervalo, estado"| X["Features versionadas"]
  X --> N["Nodo predictivo: DarkAlpha"]
  T["CSV + arquitectura + transformador"] -->|"FMT-077/078"| A["Entrenador Adam"]
  A -->|"FMT-079/080: gate y rollback"| N
  N -->|"FMT-073–076"| D["Nodo de decisión continuo"]
  P["PnL de cierre por símbolo"] --> H["Escritor de peso Hebbiano"]
  H --> K["Lectura corregida del núcleo"]
  H -->|"FMT-094: clave distinta / fallback global"| G["Perceptrón del consenso"]
  G -->|"FMT-072: piso de magnitud"| D
  K --> D
  C["Genoma + capital + mínimo exchange"] -->|"FMT-089: sensibilidad cero"| Q["Riesgo / margen"]
  B["Régimen global BTC"] -->|"FMT-091: veto discreto"| Q
  D --> Q
  Q -->|"FMT-093: validez financiera"| E["Nodo terminal: orden validada / abstención"]
  E --> P
~~~

El diagrama muestra conexiones de código inspeccionadas, no telemetría de ejecución ni cobertura total. Las APIs auxiliares ART/L2/Shannon/asignador epigenético se mantienen fuera de este camino vivo: dibujarlas conectadas sería inventar capacidad.

### 6.1 Correspondencia con los ocho módulos del informe maestro

| Módulo | Aporte de esta ronda | Límite de cobertura |
|---|---|---|
| 1. Ingestión, parsers, L2 y normalización | FMT-081–084/087/090/092 | No auditoría integral nueva de parsers ni reconciliación de secuencias |
| 2. IA y modelos | FMT-073–080/085/086 | No benchmark de precisión/latencia de producción |
| 3. Estrategia, régimen y horizontes | FMT-072/083/091/094 | Contratos continuos localizados; persisten proxies por evento |
| 4. Ejecución y conectividad | Efectos posibles de señales/guards sobre admisión | No auditoría nueva de red Binance o fills |
| 5. Riesgo y genomas | FMT-088/089/093 | Sensibilidad estática; falta replay controlado por entorno |
| 6. Estado y telemetría | FMT-073/074/080/086/087/094 | Sin certificado global de atomicidad o snapshot coherente |
| 7. Confluencia y señales cuánticas | FMT-072/094 y reconocimiento de wrappers | Sin demostración de ventaja cuántica |
| 8. Validación y gobernanza | FMT-075–080; 95 tests y criterios nuevos | Pasar tests existentes no cierra los hallazgos |

## 7. Verificación, manifiesto y límites

### 7.1 Pruebas existentes ejecutadas

Comandos ejecutados en el workspace compartido, sin acceso a red de Cargo:

~~~text
cargo test -p dark-alpha-engine -p feature-engine --lib --offline -- --skip test_inference_speed --skip test_quantum_tensor_store_read_write
cargo test -p risk-engine --lib --offline capital_regime::tests
cargo test -p signal-engine --lib --offline perceptron_gate::tests
cargo test -p signal-engine --lib --offline swing_conformal_filter::tests
~~~

Resultados: DarkAlpha 30; feature-engine 48; capital_regime 9; perceptrón 3; conformal 5. **Total 95 aprobados, cero fallidos.** Se excluyó el test de velocidad por depender del entorno y el de lectura/escritura del tensor para evitar su efecto sobre ficheros. No se añadieron tests al repositorio. Cargo generó artefactos ordinarios de compilación; no se ejecutó el entrenador ni el motor.

Se calcularon por separado, mediante réplica numérica de las fórmulas, el piso del perceptrón, varianzas de inicialización, cambio de z-score al actualizar estadísticas, profundidad inicial OFI, invariancia nominal, sensibilidad del colchón, constante de decaimiento y alpha=2. Son **contraejemplos algebraicos/numéricos, no nuevos tests de integración Rust aprobados**. Las reproducciones que requerían archivos corruptos o provocar panic se mantuvieron como especificaciones.

Los tests actuales verifican varias propiedades útiles y otras demasiado débiles: finitud no acredita significado; probabilidad en [0,1] no acredita calibración; suma de pesos uno no acredita factibilidad; ausencia de saltos en capital no acredita sensibilidad al gen. Esas diferencias explican por qué una batería verde puede coexistir con defectos.

### 7.2 Archivos leídos completos

Hashes SHA-256 abreviados a 16 hexadecimales, para identificar el contenido observado; no firmas de certificación.

| Archivo relativo al repositorio | Líneas | SHA-256, prefijo |
|---|---:|---|
| crates/dark-alpha-engine/src/lib.rs | 1432 | 034BD95616118112 |
| crates/feature-engine/src/lib.rs | 36 | 59CB7DDBCE66B8CC |
| crates/feature-engine/src/shannon_entropy.rs | 132 | 47D768531F9119DF |
| crates/feature-engine/src/welford.rs | 109 | D24594F33EFB90BE |
| crates/feature-engine/src/omni_strategies.rs | 209 | 07B2D357F4192CA3 |
| crates/feature-engine/src/microstructure.rs | 727 | 252B0B8430580E1E |
| crates/risk-engine/src/regime.rs | 81 | D16AB5DCCDD9E87B |
| crates/risk-engine/src/capital_regime.rs | 258 | F94D67FB91BE9904 |
| crates/risk-engine/src/epigenetic_capital_alloc.rs | 246 | 91D89868FD487DD0 |
| crates/risk-engine/src/orchestrator.rs | 168 | 05D5F71C3741B7B2 |
| crates/signal-engine/src/lib.rs | 23 | 743C64C6B52466BF |
| crates/signal-engine/src/perceptron_gate.rs | 191 | 34A87FEE7CA8D8D4 |
| crates/signal-engine/src/swing_conformal_filter.rs | 203 | 0502D2AD97AC3DA0 |
| crates/signal-engine/src/orchestrator.rs | 555 | 98157C241C9CBAF5 |
| src/bin/auto_trainer_daemon.rs | 411 | DFF4BB060A8382F0 |

Lectura dirigida adicional: god-engine-core/lib.rs (registro, carga, régimen, aprendizaje, claves y apertura), stateful_engine.rs (tick/kline/features), risk-engine/lib.rs (escasez y orquestador), omniscient-registry/lib.rs (resolución de claves), manifiestos Cargo y documentos previos. Estos archivos grandes no se cuentan como leídos completos por revisar tramos.

### 7.3 Integridad y reproducibilidad de la documentación

El workspace contenía cambios operativos y artefactos concurrentes antes y durante la revisión. El HEAD no describe por sí solo el código auditado: hacen falta contenido del workspace y hashes. No se atribuyen esos cambios a esta ronda. Se añadió este informe al ATLAS y al maestro y se agregó el enlace de continuación en la ronda III. La comprobación SHA-256 de los tres prefijos anteriores, normalizando sólo CRLF/LF, confirmó su conservación íntegra.

Verificación documental: 23 encabezados FMT únicos y consecutivos, un grafo Mermaid y ningún enlace local a fichero inexistente o número de línea fuera de rango. Los 15 hashes del manifiesto permanecieron iguales al cerrar la revisión. El diff documental comprobado no presentó errores de whitespace. Estas verificaciones acreditan integridad del anexo, no corrección del sistema auditado.

No se hicieron commit, push, merge ni fetch. Las solicitudes históricas de unificación de ramas no se ejecutan como efecto secundario de una revisión científica. Tampoco se afirma que otros autores hayan resuelto estos puntos o que los hallazgos de rondas anteriores sigan todos idénticos: requieren comprobación individual sobre la versión que vaya a promoverse.

La investigación usó el índice de papers de Firecrawl, expansión de referencias y pasajes internos para las afirmaciones teóricas centrales; el catálogo de Clay se consultó directamente. Las propuestas de integración y los contraejemplos del código son inferencias de esta auditoría, diferenciadas de resultados de los artículos.

## 8. Hoja de ruta propuesta: rehabilitación desde la raíz hasta la acción

1. **Congelar contratos, no el desarrollo.** Versionar esquema, unidades, objetivo, política de normalización y modelo junto al genoma. Distinguir datos inválidos de valor cero. Establecer un tape de replay común con tiempos de evento/disponibilidad y calidad. No cambiar features sin migrar sus consumidores.
2. **Cerrar la paridad del aprendizaje.** Atender FMT-077/078/079 primero: arquitectura explícita, transformación idéntica y validación que no premie ausencia. Después, FMT-073/074/076/080: snapshot completo, carga validada y reproducción de la función servida.
3. **Cerrar la arista de crédito.** FMT-094 requiere verificar todas las lecturas de cada escritura aprendida, no sólo el primer consumidor. Definir si una señal aprende por símbolo, por contexto o de forma compartida, y probar que el fallback corresponde a esa definición.
4. **Reparar la representación medible.** FMT-081–084 exigen invariancias declaradas y estado inicial consistente. Usar pruebas metamórficas de unidad, reempaquetado y warmup, con tolerancias y excepciones explicadas. Medir varianza efectiva/saturación de cada canal antes de atribuirle inteligencia.
5. **Restituir adaptabilidad identificable.** FMT-089: mapear qué genes tienen derivada nula o quedan recortados, por entorno y escala. Comparar acciones y rechazos sobre el mismo tape; no comparar sólo PnL final ni asumir que genes distintos generan políticas distintas.
6. **Revisar filtros con función de decisión explícita.** FMT-072/091/093: abstención, restricciones duras de seguridad, incertidumbre estadística y preferencias de riesgo no son el mismo objeto. Retener controles prudentes mientras se demuestra una alternativa; no borrar un veto sólo por ser discreto.
7. **Mantener capacidades auxiliares bajo cuarentena científica.** FMT-085–088/090/092 deben validarse antes de conectarse. “Desconectado” no es necesariamente bug si el contrato aún no es seguro. Conectar todo sin semántica común puede empeorar el sistema.
8. **Evaluar teorías por ablación y coste.** T25 puede resolver cambio de coordenadas bajo límites precisos; T26 puede medir información incremental de aristas; T07 puede representar incertidumbre sobre cambios. Ninguna sustituye la política de riesgo, el ledger de evidencia o la validación fuera de muestra. Medir latencia p50/p99 en un entorno controlado: esta auditoría no proporciona esas cifras.

### Criterio profesional de cierre

Un hallazgo no se cierra porque desaparezca una palabra, se suavice una función, se conecte una clase o pasen los tests ya existentes. Se cierra cuando existe una regresión que habría detectado el defecto, una corrección del contrato matemático, una prueba productor→consumidor, evidencia reproducible y una revisión de impacto en los entornos afectados. Para declarar mejora científica se necesita, además, superar un baseline con incertidumbre y costes incluidos.

**Conclusión:** el sistema contiene avances continuos reales, pero todavía combina transformaciones incompatibles, evidencia incompleta y conexiones de aprendizaje parciales. El objetivo defendible es un sistema adaptativo con límites, estados y resultados verificables; no una certificación de omnisciencia. Esta ronda amplía la evidencia y las propuestas sin eliminar lo existente ni cambiar el comportamiento operativo.

## Continuación científica V — 24 de septiembre de 2026

La [ronda V](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_V_2026-09-24.md>) añade FMT-095–113 sin modificar esta ronda.
Sigue la geometría TP/SL hasta cantidad ejecutada, envolvente, ledger
y aprendizaje; distingue defectos vivos de auxiliares y amplía con
T27 (factibilidad/seguridad) y T28 (reducción por observabilidad).
8 Rust completos adicionales, 3.615 líneas; 34 tests existentes
aprobados. El acumulado de manifiestos FMT es 90 Rust distintos:
no constituye cobertura integral del proyecto ni cierre de hallazgos.

## Adenda de reparación posterior XII — FMT-092, 2026-09-24

Se preserva el diagnóstico anterior. La [auditoría XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XII_2026-09-24.md>) repara localmente
el dominio de alpha y la mezcla de representaciones al alternar métodos.
También añade FMT-138, arranque exponencial con prior cero no declarado,
y FMT-139–144 en configuración, duplicación y diagnóstico.

Siete tests de contrato Welford pasan; cuatro fallaban antes. La transición
automática histórica y el cálculo acumulativo válido se conservan. No se
localizó consumidor productivo de update_decay. count sigue siendo un
indicador legacy, no tamaño muestral efectivo; no se acredita inmunidad
a todo overflow ni estado público corrupto. La nueva API temporal EWMA es
opt-in, no una migración operativa de los normalizadores. El [artefacto XII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XII_2026-09-24.json>)
identifica el corte actual; los hashes anteriores siguen siendo históricos.
