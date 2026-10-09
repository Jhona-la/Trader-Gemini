# R2 — Revisión integral de selection_stats, 2026-10-08

Estado: **427/427 líneas leídas; revisión semántica abierta, no cierre del archivo**. Fuente fija `e9c6a435eb1780da7f2c6e29103f3bd7d227d94c`. Este trabajo no modifica el candidato congelado, no retrasa su publicación y no constituye un cambio de pipeline. Sin Cargo, compilación, tests, repetición de testigos ni datos privados. Sólo lectura de objetos Git, derivación matemática, contraste de fuentes primarias y escritura de este informe ignorado.

Archivo: `crates/risk-engine/src/selection_stats.rs`; blob `1e2ba9b194cf133bc614071c168003287558145d`; SHA256 `a1995b82b66671b5083397e4e96a466020e13aadf32ec836808c868ffc9b88d6`. Es idéntico al módulo de los testigos R4 de18bb; esa igualdad permite enlazar sus resultados con procedencia, no afirmar una ejecución nueva sobre e9. La revisión independiente `pearson_domain_review` comprobó las desigualdades y el extremo de N, también sin ejecutar código.

## 1. Mapa completo de lectura

Los rangos siguientes forman una partición de todas las líneas, incluidos comentarios, blancos y tests. Su lectura no acredita cobertura de comportamiento ni de las dependencias del workspace.

| Líneas | Superficie | Evaluación |
|---|---|---|
| 1–27 | Doctrina, fórmulas, `ReturnMoments` | API pública sin invariantes de construcción; las afirmaciones de edge real exceden calibración disponible. |
| 28–49 | `compute_moments` | Filtro finito, n>=20, media, varianza muestral, momentos y floor de sd examinados; errores y convenciones abajo. |
| 50–71 | `sharpe`, `psr` | SR de una observación del caller, sin frecuencia/rf incorporados; `psr` acepta un SR externo y colapsa SE pequeño a0.5. |
| 72–106 | `sharpe_std_error` y documentación | Floor/fallback; positividad exacta derivada; falta validación numérica y estadística. |
| 107–127 | Euler y `expected_max_sharpe` | Aproximación entre ensayos; N=0/1 reinterpretados, sigma sin dominio y cancelación de cola para N enorme. |
| 128–144 | `dsr` | Reutiliza SE del ganador como dispersión entre ensayos; sin información de la familia. |
| 145–201 | Inversa normal, coeficientes, tres ramas | Leídos todos los coeficientes y branches; dominio inválido se confunde con infinitos de frontera. No precisión global certificada aquí. |
| 202–213 | CDF normal | Aproximación explícita; NaN se propaga, infinitos dan extremos; tests puntuales no prueban la cota global comentada. |
| 214–249 | Umbral, verdict, `edge_survives_multiplicity` | Umbral nominal0.95, motivos ambiguos y sin n efectivo/datos descartados; NaN no pasa la comparación, pero puede salir como score. |
| 250–427 | Los11 tests, comentarios y cierre | Todos leídos; detalle en§5. No ejecutados en esta revisión. |

## 2. Convención de momentos y Pearson correcto

Para los n valores retenidos, escribir `m_k = (1/n) sum (r_i-r_bar)^k`. El módulo define:

- `s² = n/(n−1) * m_2`, desviación **muestral**.
- `a = m_3/s³`, campo skewness.
- `b = m_4/s⁴`, campo kurtosis, no exceso.
- `c=(n−1)/n`, por lo que `a=c^(3/2)*gamma_3` y `b=c²*gamma_4`, donde gamma3/gamma4 son los momentos estandarizados de la distribución empírica usando `sqrt(m_2)`.

No son exactamente los coeficientes Fisher–Pearson empíricos habituales ni sus versiones corregidas por sesgo. La definición convencional usa divisor n dentro de la desviación para estos coeficientes; las variantes deben declararse antes de comparar implementaciones. La discrepancia finita desaparece asintóticamente, pero puede importar cerca del mínimo20. Esto es una elección de estimador sin contrato documentado, no prueba por sí sola de fallo aritmético. [NIST, definiciones de skewness y kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm).

**Derivación para estos campos, no trasplante de una desigualdad poblacional.** Sea Y=(r−r_bar)/s, con promedio empírico E. En aritmética exacta, E[Y]=0, E[Y²]=c, E[Y³]=a y E[Y⁴]=b. Cauchy–Schwarz sobre Y e Y²−c da:

`a² <= E[Y²] E[(Y²−c)²] = c (b−c²)`.

Por tanto, **`b >= c² + a²/c`**. Sólo para una variable estandarizada a varianza1 corresponde directamente `gamma4 >= gamma3²+1`. La advertencia R4 sobre el test manual era válida para momentos poblacionales; esta fórmula precisa el dominio muestral realmente implementado.

Con `g=max(b,3)`, se obtiene:

`g−1−a² >= g−1−c*b+c³ >= (1−c)*g−1+c³ >= (1−c)²*(c+2) >0`.

Por completar el cuadrado:

`q(SR)=1−a*SR+(g−1)*SR²/4`

`= (g−1)/4 * [SR−2a/(g−1)]² + 1−a²/(g−1) >0`.

Esto vale para cualquier muestra exacta no degenerada n>1 y SR real finito, incluido n>=20 del productor. **El fallback q<=0 no es una región alcanzable por sus momentos exactos coherentes** después del floor a3. Puede recibir estructuras manuales incoherentes, NaN o errores numéricos; no retirarlo ni modificarlo ahora sin definir el contrato público.

El unit de407–426 usa n100, a2,b3: requeriría b>=0.99²+4/0.99=5.020504…, luego no procede de ese productor exacto. Además, media0/sd1 produciría SR0 en `dsr`, no el SR2 manual del test. El test cubre una rama defensiva artificial, no una región estadística legítima ni una calibración de muestras débiles.

**q positivo no implica q>=1.** El término −a*SR puede reducirlo aun con b>3. El contraejemplo Rust R4 ya existente (40 datos; SR0.5, b36.148125, a5.858119…) da SE0.08284959 <1/sqrt39=0.16012815. No se repitió. Además, para normalidad con skew0,kurt3 y SR no cero, la propia fórmula da SE=sqrt((1+SR²/2)/(n−1)); llamar a1/sqrt(n−1) el SE gaussiano general omite el término de SR. Es el valor en SR0, no en todo SR.

## 3. Dominio y hallazgos separados por naturaleza

### R2-SS-01 — Fallos numéricos no distinguidos de estadística, MED de contrato

Líneas29–49 filtran entradas no finitas, pero no validan los resultados de suma, resta, cuadrados, desviación ni momentos. Entradas finitas no garantizan salidas finitas.

Contraejemplos por traza de operaciones, **no ejecutados** aquí:

- Veinte valores1e308: la suma positiva rebasa f64 aunque el valor medio matemático sea finito; media y sd pueden quedar Inf, y las estandarizaciones Inf/Inf producen NaN. `sd <=1e-12` no detecta NaN/Inf y se retorna `Some(ReturnMoments)` inválido. La serie matemáticamente constante debería distinguirse como degenerada.
- Veinte valores alternando±1e200: la suma puede quedar0, pero cada cuadrado1e400 rebasa f64; sd=Inf y las desviaciones estandarizadas pasan a0. Se emiten momentos incompatibles con el vector original. Es aritmética finita, no falta de calibración de DSR.

En la ruta `edge`, un DSR NaN no supera0.95 (`passes=false`); **no se afirma que esos ejemplos abran promoción**. El problema es que `Some`/score/notas no identifican corrupción numérica y que funciones públicas aceptan el objeto inválido. El dominio económico normal y la alcanzabilidad de esos extremos en los callers no están establecidos por esta lectura.

La tolerancia fija `sd<=1e-12` rompe invariancia frente a cambio de escala positivo cuando se cruza ese umbral. Puede ser una política de degeneración numérica válida si se declara en las unidades del retorno; actualmente `None` no distingue esa causa de n<20. La suma ingenua también puede depender del orden en coma flotante; no se cuantificó ese efecto en rangos económicos.

### R2-SS-02 — Penalización de multiplicidad vuelve a cero, LOW operativo / MED de API

Líneas117–126: bajo usize64 y redondeo IEEE habitual, N=2^53 se convierte exactamente a f64, pero `1/(N*e)<2^-54`, por lo que `1−1/(N*e)` redondea a1. La inversa normal devuelve Inf, e_max queda Inf y el fallback retorna **0** para sigma finito. Para N2 y sigma>0 el benchmark es positivo: se viola monotonía al crecer el número de ensayos. El problema ya existe antes de que el cast pierda enteros adyacentes; a partir de N>2^53 también aparece esa pérdida. N=2^54 hace que la primera cola redondee a1.

En usize32 no se alcanzan esos extremos. Los contadores de callers saturan pero no hay evidencia de que alcancen billones de ensayos en operación; no elevar este caso a incidencia financiera. Es un contraejemplo algebraico del dominio público, sin test nuevo. No confundir la saturación del entero con un benchmark numéricamente seguro.

Otras fronteras: N0 y N1 se convierten silenciosamente en2, mientras `SelectionVerdict.n_trials` conserva el original; sigma negativa retorna benchmark negativo y sigma NaN/Inf se transforma en cero. El producto sigma*e_max no se comprueba tras multiplicarse y puede desbordarse aun con factores finitos. Definir errores/casos frontera en vez de interpretar cero como «sin penalización».

### R2-SS-03 — Información descartada y diagnóstico falso, MED de trazabilidad

El filtrado de NaN/Inf ya está reproducido por el testigo R4 existente:403 entradas se reducen a400 y conservan aprobación. El módulo no devuelve conteo descartado, posiciones temporales ni causa de pérdida. Esto requiere contrato explícito con el consumidor; no puede decidir si una repetición numérica es una observación duplicada sin identidad temporal/origen.

Veinte retornos constantes producen `None` por sd0, pero `edge` informa «muestra insuficiente (<20 retornos finitos)». Ese mensaje es falso para ese caso; no implica que se deba aprobar una varianza cero. Un scoreNaN se describe genéricamente como ruido. Las etiquetas «edge REAL al95%» y la garantía general leptocúrtica exceden lo comprobado. Separar insuficiencia, degeneración, contaminación, error numérico, falta de identificabilidad y no superar un umbral nominal.

### R2-SS-04 — Estructura pública no validada, MED condicional

`ReturnMoments` expone todos sus campos. `sharpe` devuelve0 para sd negativa/pequeña, propaga NaN en otros casos; `sharpe_std_error` no comprueba consistencia de n,a,b ni SR. La rama else recibe también qNaN: puede ocultar un skewNaN eliminándolo; kurtosisNaN se convierte a3 mediante `max`. La semántica de `f64::max` devuelve el argumento no NaN cuando sólo uno es NaN. [Documentación oficial Rust](https://doc.rust-lang.org/std/primitive.f64.html#method.max).

Ejemplo de API manual, no ejecución ni ruta actual demostrada: con mean0.1,sd1,n10000,skewNaN,kurtNaN, el fallback vuelve a un SE finito y puede producir una decisión positiva, en vez de señalar momentos inválidos. `git grep` no encontró construcción externa de `ReturnMoments` en crates/src fuera de este módulo; los callers esenciales usan `edge`, no esa estructura manual. Mantener alcance condicional.

`psr` admite un SR que no coincida con mean/sd y da0.5 si SE<=1e-12; `dsr` sólo exige n>=2 al recibir estructuras manuales. La inversa normal trata NaN y+Inf como−Inf por el primer guard, y p fuera de[0,1] como un extremo. Eso mezcla errores de dominio con límites válidos p0/p1. `normal_cdf` no filtra NaN. No se afirma que todos esos casos sean alcanzables desde entradas sanas de `edge`.

### Q4 / R2-SS-05 — Calibración y selección, HIGH de inferencia; no bug aritmético aislado

`dsr` usa SE del ganador como sigma de la distribución entre ensayos. El artículo DSR distingue varianza de Sharpes entre candidatos, número efectivo independiente de ensayos y momentos/longitud del ganador. El conteo bruto puede ser conservador bajo supuestos específicos de dependencia entre ensayos, pero no demuestra cobertura universal con candidatos heterogéneos, selección en embudo o reutilización adaptativa. [Bailey y López de Prado, DSR, §2 y apéndice3](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).

Los momentos marginales y n no registran orden, autocovarianzas, frecuencia, activos, solape o selección de cierres. Lo distingue inferencia i.i.d. y estacionaria dependiente y restringe reglas simples de agregación temporal. [Lo, The Statistics of Sharpe Ratios](https://traders.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf). El testigo de duplicación existente exhibe insuficiencia de un contrato, no una tasa de error medida del bot.

No proponer HAC como sustitución automática de n. Haría falta definir retornos y reloj, estadístico/función de influencia, condiciones de momentos/dependencia, selección de lags/kernel, límites de muestra y experimento de calibración; después resolver además la dependencia entre ensayos y el presupuesto de consultas. Un HAC de la media por sí solo no es la varianza asintótica del cociente Sharpe ni corrige selección o contaminación.

## 4. Callers esenciales y alcance leído

`risk-engine/src/lib.rs:22` publica el módulo; `evolution-engine/src/lib.rs:18` reexporta exactamente esa fuente. No apareció una implementación paralela en la búsqueda de símbolos. Blobs: risk lib `554e02d060059d9d15247a2194df6d65f0f3ea36`; evolution lib `6c60560c2be84f6e11e9978eca821b7a0b2893ce`.

**Darwin**, blob `847a6e96e9b4de13cb0d473c25e159662a7e6cf0`. Leídos320–415,426–520,625–696, además de referencias de tests/localización. El vector se forma de cambios de saldo realizado sobre cadencia disparada por ticks, sin marcas ni duración por observación; filtra r no finito y puede terminar con vector parcial cuando el capital se vuelve inválido. La selección IS evalúa20×5 y usa el campeón en el segmento OOS con baseline; N suma100 aunque élites puedan repetirse. Esa cifra es evaluaciones, no prueba de100 ensayos independientes/identidades únicas. Flags de legacy y promoción delimitan alcance; no se inspeccionó su valor operativo.

Las líneas649–657 actualizan un contador atómico mediante load/store y llaman a `edge`; el host recuperado mantiene instancia entre rondas y espera el worker. Eso no prueba seguridad ante callers externos concurrentes ni persistencia por linaje tras reinicio. `expected_max_sharpe(total,1)` se usa como telemetría; la compuerta real usa DSR. El comentario «erradica optional stopping» no deriva de acumular N. No es necesario reabrir Q1 corregido dentro del proceso para mantener este límite.

**LiveEvolutionDaemon**, blob `b2ba04b247e51e0aae35d7aeae9449946d8b536c`. Leídos245–282,412–435,448–475,554–605,634–778,1296–1318,1433–1477,1478–1505,1526–1722,1765–1875; referencias adicionales de contador obtenidas por búsqueda. No se declara revisión total de sus2431 líneas.

- `trade_return_on_equity`419–427 produce `pnl_net/(capital_after−pnl_net)` y sólo conserva retorno finito con denominador positivo. `wf_replay_segment`575–595 lo registra únicamente al cierre y cuando `puntua`: **observaciones por trade**, no retornos de cartera de1s como Darwin. n cuenta cierres, y la demora/selección de cierres y posiciones abiertas requieren contrato propio.
- `wf_evaluate_real`634–725 calienta y evalúa series en bucles por activo, usa saldo para scoring y entrega un único vector sin tiempos. La serie no demuestra sincronización multiactivo real; el estado persiste entre pasadas/activos. Esto limita interpretar el score como proceso de cartera de frecuencia común.
- El pre-screen recorre0..=2000, incluyendo incumbente, y topK24 más incumbente llegan al motor; sus returns de pre-screen se descartan. N usa `max(prescreen_count,evaluated)` y un acumulador saturante. No hay deduplicación de genomas canónicos ni muestra de Sharpes entre ensayos en el API. No inferir un undercount aritmético por contar2001 en lugar de2002: el incumbente ya está en pre-screen; queda por definir qué constituye ensayo en las dos etapas.
- La ruta1831–1845 usa returns simulados del ganador y corta ante `!passes`;1853–1865 promueve primero al almacén y después aplica al arena. La política de arming se lee en738–768 y el guard1303; no se alteró. La nota de promoción ya indica ausencia de garantía posterior, aunque otros comentarios del mismo caller afirman edge real categórico.

## 5. Los11 tests leídos y sus límites

| Test | Qué comprueba por lectura | Qué no demuestra |
|---|---|---|
| `psr_edge_real_supera_umbral` | Serie determinista con media positiva arroja PSR alto. | Existencia de edge fuera de muestra o cobertura del95%. |
| `psr_ruido_puro_no_supera` | Un vector determinista produce PSR bajo. | Distribución nula, tasa de falsos positivos o dependencia. |
| `dsr_mas_pruebas_menor_dsr` | Orden no estricto para N10/5000 en un vector de media alta. | Monotonía global ni resolución si la CDF satura. |
| `edge_survives_rechaza_muestra_corta` | Diez valores no pasan. | Frontera19/20, varianza cero, contaminación o overflow. |
| `inverse_normal_cdf_valores_conocidos` | Ocho cuantiles centrales/moderados. | Fronteras, subnormales, p inválida o cota de error total. |
| `d741_umbral_sube_respecto_de_la_formula_vieja` | Una referencia N2000 y comparación con fórmula anterior. | Familia de ensayos real o N enorme. |
| `d741_el_dsr_es_mas_exigente_que_el_viejo` | DSR nuevo menor que el anterior en un vector. | El comment anuncia cambio de aprobación; el assert sólo compara dos scores, no exige que uno pase y otro falle. |
| `expected_max_sharpe_crece_con_las_pruebas` | Orden estricto en10/2000/200000. | Dominio usize64 completo, sigma inválida o producto desbordado. |
| `normal_cdf_correcta` | Tres puntos0,±1.96. | Cota uniforme, colas ni NaN. |
| `omega15_h1_4_dsr_sharpe_std_error_leptocurtico` | Un caso de cola pesada con SE>baseline. | La afirmación universal, refutada por otro skew/SR válido. |
| `r5_b3_discriminante_degenerado_jamas_sub_gaussiano` | Rama fallback y un caso de fórmula manual. | Alcanzabilidad de sus momentos desde datos coherentes o validez estadística. |

Estos son análisis de los cuerpos presentes; no resultados nuevos de ejecución ni sustitutos de CI/T1.

## 6. Próximos contratos concretos, sin cambios en esta ola

1. **Especificación R2 antes del fix:** declarar qué objeto representa cada retorno (saldo/MTM, simple/log/exceso, periodo/trade), política de faltantes/ruina, unidades, definición exacta de momentos, n y N. Mantener por separado el objetivo económico72h y una compuerta estadística nominal.
2. **Contrato numérico productor:** fixture conocido19/20; constantes; escala positiva con tolerancia declarada; traslación de momentos centrales (el Sharpe sí cambia); grandes offsets, sumas/cuadrados que desbordan; NaN/Inf en distintas posiciones. Una salida válida exige todos los campos finitos, sd>0 y Pearson muestral con tolerancia de redondeo. Preservar motivo explícito si no puede estimarse.
3. **Contrato público:** `ReturnMoments` manual incoherente/n0/n1/n2, sd negativa/NaN/Inf, skew/kurt inválidas y SR incompatible; definir rechazo o precondición en lugar de fallback silencioso. El ejemplo del unit manual puede quedar como prueba de entrada inválida con el comportamiento acordado, sin venderlo como calibración física.
4. **Contrato benchmark:** N0/N1/N2, rangos habituales y extremos dependientes de arquitectura; sigma0/negativa/NaN/Inf/muy grande; no disminuir benchmark al crecer N dentro del dominio declarado y nunca convertir error numérico en ausencia de penalización. Oráculo independiente de alta precisión para cuantiles/comparaciones de cola; no congelar como esperado el valor actual errado.
5. **Contrato de decisión y trazabilidad:** score válido finito en[0,1] o estado no identificado; `passes` sólo con evidencia válida; número de observaciones originales/aceptadas/descartadas y razón. Conservar los controles sanos al cambiar mensajes o errores. Identidad temporal/origen se valida en el caller, no deduplicando valores iguales.
6. **Calibración separada:** nulos y alternativas preregistrados con dependencia temporal/multiactivo, colas, missingness, cierres y selección en embudo; registro de todas las consultas y genomas. Comparar estimadores/costes y cobertura antes de presentar95% nominal como frecuencia de error. HAC, bloques o e-procesos son candidatos con hipótesis distintas, no una lista de parches intercambiables.

La publicación del candidato actual puede continuar bajo sus recibos existentes. Este lote añade evidencia y contratos propuestos: no declara solucionados Q4, probabilidades de ruina, OOS sellado ni crecimiento de100% cada72h; tampoco propone aflojar vetos para obtener un score deseado.
