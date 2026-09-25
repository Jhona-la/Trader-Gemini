# Auditoría científica VI y rehabilitación verificable del riesgo

Fecha: 24 de septiembre de 2026. Referencia Git local: **59a76de4**, con cambios concurrentes. Este anexo continúa la serie FMT y registra **correcciones de código**, no sólo diagnósticos. No sustituye ni elimina las rondas I–V.

## 1. Resultado ejecutivo y límite de certificación

Se corrigieron contratos locales en tres archivos de risk-engine y se añadieron dos suites de integración. La intervención conserva las firmas existentes y añade una API de presupuesto monetario. No introduce motores separados por scalping/swing, constantes de horizonte nuevas, dependencias externas ni un optimizador cuántico sin evidencia.

Cambios realizados:

- El RR genómico ya no puede reducir el TP base ni cambiar su probabilidad contractual de 0,40 a 0,55; un producto no representable no contamina una geometría base finita.
- La envolvente no reactiva automáticamente riesgo micro cuando su evidencia prescribe cero y no permite excepciones al presupuesto para satisfacer un notional mínimo.
- Datos no finitos, parámetros estadísticos inválidos y cálculos no representables dejan de convertirse en autorización de exposición.
- Se eliminan pisos artificiales de probabilidad y payoff que fabricaban edge. La varianza beta se evalúa evitando productos intermedios que desbordaban.
- Se añade ExposureBudget con presupuesto, notional máximo y proyección de cantidad sobre lotes, mínimos y costes explícitos.
- El guard de notional valida el filtro recibido y el producto margen×leverage, sin asumir un mínimo universal.

**Frontera crítica:** FMT-113 sigue **abierto**. La nueva proyección existe y está probada, pero el host y el replay todavía no la consumen para modificar la cantidad final. Sus archivos ya tenían cambios de otras sesiones y se preservaron conforme a las reglas del repositorio. El bootstrap explícito n<30 del consumidor tampoco desaparece al corregir la envolvente. No es válido anunciar control de riesgo extremo a extremo ni desplegar suponiendo que ese contrato ya está cerrado.

No hubo trading, modificación de capital, edición de genomas, reinicio de motores, despliegue, commit, push, merge o fetch. Se compilaron y ejecutaron pruebas locales; cargo check validó el host sin lanzarlo.

## 2. Matriz de estado, sin borrar historia

“Corregido localmente” significa implementación y pruebas sobre el workspace observado. No significa desplegado, rentable, completamente integrado o certificado bajo todos los estados concurrentes.

| Referencia | Estado de esta ronda | Evidencia / trabajo restante |
|---|---|---|
| FMT-096 | Corregido localmente | Monotonía del TP, mismo SL/piso/rr_required y manejo del overflow del target |
| FMT-098 | Corregido en la envolvente; consumidor maduro probado | Micro sin edge ya se veta en replay; bootstrap n<30 externo sigue pendiente |
| FMT-099 | Corregido en la envolvente | Mínimo que excede presupuesto se rechaza; no se certifican otras capas de sizing |
| FMT-113 | Abierto, con API preparatoria implementada | Falta llevar cantidad proyectada a host, replay, reserva local y payload |
| FMT-114 | Nuevo; corregido localmente | Datos inválidos y overflow estadístico podían producir exposición |
| FMT-115 | Nuevo; corregido localmente | Filtro de notional inválido / producto infinito pasaban el guard |
| FMT-116 | Nuevo; corregido localmente | Pisos de p y payoff creaban edge ficticio |
| FMT-095/097 | Abiertos | Conflicto de piso difusivo, cap de barreras y escalón micro |
| FMT-100/101 | Abiertos | Doble EMA y epsilon monetario en aprendizaje/restauración |
| FMT-102–112, excepto los aspectos aquí indicados | Abiertos | No se reclama reparación de simulador, mmap, ledger o interpolación |

Se agregan FMT-114–116 como fichas científicas nuevas. No se suman mecánicamente a la matriz histórica de 305 puntos: una consolidación global requiere deduplicación causal y verificación de estados.

## 3. Contrato matemático implementado y significado de cada cálculo

### 3.1 Probabilidad: estimación, incertidumbre y decisión son objetos distintos

Para el posterior beta, con a>0 y b>0:

~~~text
S = a + b
media = a/S
varianza = (a/S)(b/S)/(S+1)
p_inferior_aprox = clamp(media − z·sqrt(varianza), 0, 1)
~~~

a y b son parámetros del posterior; no son probabilidades. El valor n=a+b−1 representa evidencia neta bajo el prior Jeffreys Beta(1/2,1/2) utilizado por esta implementación. La media y su desviación expresan la distribución del parámetro Bernoulli bajo ese modelo. No identifican automáticamente una probabilidad de primera llegada al TP ni resuelven dependencia temporal, selección o cambios de régimen.

El cálculo de varianza es algebraicamente el mismo que ab/[S²(S+1)], pero evita multiplicar números enormes para después dividirlos. Con a=b=10²⁰⁰, la expresión anterior producía intermediarios infinitos; la nueva obtiene aproximadamente 3,5355339059327376·10⁻¹⁰¹ de desviación. Eso es estabilidad numérica, no evidencia de que existan 10²⁰⁰ trades.

El recorte [0,1] impone el dominio probabilístico sin elevar artificialmente probabilidades pequeñas. El anterior piso 0,01 no era prudencia: cuando la probabilidad genuina es menor, la aumentaba. La aproximación normal sigue sin ser un cuantil beta exacto; se corrigió su documentación, no se le adjudicó una cobertura estadística que no tiene.

Implementación: [kelly_envelope.rs:66](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:66>) y [kelly_envelope.rs:90](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:90>).

### 3.2 Kelly: utilizar el payoff recibido, no uno más favorable

Sea p la probabilidad inferior aproximada y r>0 el payoff medio ganador/perdedor:

~~~text
f_Kelly = p − (1−p)/r
f_shrunk = f_Kelly · n/(n+k)
f_final = mínimo entre f_shrunk, guard de racha y tope de política
~~~

p y r son adimensionales; f representa una fracción. La reformulación p−q/r evita el producto innecesario p·r. Si r es pequeño, el resultado puede ser negativo o no representable: en esta API se rechaza exposición. No se eleva r a 0,05 para obtener un resultado aceptable.

El shrinkage n/(n+k) es una política de contracción con k explícito, no una nueva observación ni una prueba de edge. El guard de racha y el máximo 0,25 son restricciones de política existentes; se conservaron y se dejó de presentarlas como un teorema conjunto de probabilidad de ruina. Tampoco se cambió el requisito histórico n≥3 de esta función; tres observaciones no acreditan por sí solas la calidad de la estrategia.

La validación comprueba finitud, positividad de parámetros beta, número de observaciones representable, z≥0, k≥0, payoff positivo y representabilidad de n+k. Si no hay evidencia admisible, devuelve cero. La exploración, si se autoriza, debe poseer presupuesto, estado y observaciones propios; no puede aparecer como una sustitución oculta de ese cero.

Implementación: [kelly_envelope.rs:230](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:230>).

### 3.3 Presupuesto monetario, exposición y financiación

Definiciones:

| Símbolo | Magnitud | Unidad |
|---|---|---|
| C | Capital de referencia suministrado | Moneda de cuenta |
| f | Fracción autorizada por la envolvente | Adimensional |
| B=Cf | Presupuesto de pérdida nominal | Moneda de cuenta |
| Q | Cantidad de activo | Unidades del activo |
| P | Precio de valoración | Moneda/unidad de activo |
| N=|Q|P | Notional | Moneda de cuenta |
| d | Distancia relativa de entrada a stop | Adimensional |
| c | Reserva proporcional de costes ida/vuelta | Adimensional |
| L | Multiplicador de margen configurado | Adimensional |

Se implementa B=Cf y N_max=B/(d+c). Bajo ese **modelo nominal**, una cantidad admisible satisface |Q|P(d+c)≤B. La nueva API exige d>0 y c≥0 finitos, y rechaza overflow/underflow que impida representar positivamente el presupuesto o su denominador.

Esto no demuestra que un stop se ejecute en su precio, ni cubre costes fijos, funding incierto, saltos no acotados o impacto no lineal. El caller debe construir una reserva c apropiada y, si el modelo lineal es insuficiente, sustituirlo por una restricción de coste más general. No se inventa c desde una constante local de conveniencia.

En cambio, el margen aproximado es N/L. Cambiar L manteniendo Q y P no modifica la pérdida al stop. Por eso max_leverage conserva su nombre por compatibilidad, pero se documenta como techo de exposición N_max/C. No es una instrucción para configurar automáticamente el leverage del exchange.

La variante de compatibilidad conserva el piso prudencial histórico de stop de 5 bps y usa c=0 porque la firma antigua no recibe costes. Se explicita esta limitación; no se declara que el método antiguo ya incluya las comisiones. Para sizing completo se preparó exposure_budget con coste explícito.

Implementación: [kelly_envelope.rs:273](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:273>) y [kelly_envelope.rs:311](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:311>).

### 3.4 Proyección sobre lotes: continuidad de la política, discreción física de la ejecución

La política puede ser función continua del estado y de τ; la orden del exchange debe respetar una retícula de cantidades. Esa discreción no equivale a separar motores por horizonte.

Para propuesta Q₀, precio P y step s>0:

~~~text
Q_límite = mínimo(Q₀, N_max/P)
n_lotes = floor(Q_límite/s)
Q = n_lotes · s
~~~

Después se verifica nuevamente: Q>0, Q≤Q₀, Q≥min_qty, QP≥min_notional, QP≤N_max y QP(d+c)≤B. No se eleva Q para satisfacer el mínimo. Si el conjunto factible está vacío, la respuesta es None.

Ejemplo: N_max=50, precio=30 y step=1. Existe un notional continuo de 50, pero los lotes disponibles dan 30 o 60. Si el mínimo es 50, no hay orden factible. Retornar 60 para “desbloquear inteligencia” rompería el presupuesto; retornar 30 incumpliría el filtro.

Los productos y divisiones binary64 pueden sobrepasar una desigualdad por redondeo. La API permite retroceder un lote y revalida; nunca añade un epsilon al presupuesto para hacer pasar una orden. Rechaza conteos superiores a 2⁵³−1, límite de representación entera exacta relevante, no umbral económico. Puede rechazar o reducir conservadoramente una cantidad en una frontera numérica; no promete el óptimo decimal exacto.

El tipo ExposureBudget tiene campos privados y acceso de lectura. Conserva el presupuesto observado en el momento de su creación; no garantiza frescura de capital/precio después. El caller debe manejar versión del estado, reservas concurrentes, precio de ejecución, inventario total y liquidación.

Implementación: [kelly_envelope.rs:111](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:111>) y [kelly_envelope.rs:137](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:137>).

## 4. Hallazgos nuevos: evidencia, impacto y reparación

### FMT-114 · P1 · Datos inválidos y desbordamiento numérico podían autorizar riesgo

**Estado:** corregido localmente en la decisión de riesgo; no equivale a validación universal de todos los productores de datos.

**Mecanismo anterior:** risk_fraction no verificaba todos los parámetros de la beta, z, k, payoff ni los resultados intermedios. En una cadena con NaN, comparar ≤0 no rechaza necesariamente; operaciones min/max posteriores pueden devolver el otro operando finito. Se podía transformar una estimación inválida en una fracción positiva. max_leverage tampoco validaba finitud de capital, stop y mínimo del exchange de manera completa.

**Prueba roja real:** contracts_invalid_posterior_or_uncertainty_cannot_create_edge observó 0,014867039231272083 frente al cero requerido. contracts_invalid_capital_stop_and_exchange_minimum_fail_closed observó (25,true) frente a (0,false). Esas pruebas fallaron antes del arreglo; no son sólo ejemplos escritos después.

**Variante numérica:** con parámetros beta muy grandes pero finitos, ab y S²(S+1) desbordaban. La regresión de varianza observó una desviación inválida y falló antes de reemplazar la fórmula por sus cocientes normalizados.

**Reparación:** validación previa, rechazo de resultados no finitos, varianza reordenada algebraicamente y comprobación de representabilidad de B y N_max. El posterior corrupto no se “cura” inventando una operación: la decisión falla cerrada. Los métodos estadísticos públicos no constituyen un parser de artefactos ni una reparación de la persistencia; esa frontera sigue requiriendo validación propia.

**Impacto comprobado:** los tests de consumidor verifican que un posterior inválido no admite una nueva candidata y que el replay devuelve la reserva local al vetarla. No se provocó corrupción en un motor real. Pruebas: [spectral_risk_contract.rs:104](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/spectral_risk_contract.rs:104>), [spectral_risk_contract.rs:357](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/spectral_risk_contract.rs:357>) y [spectral_risk_contract.rs:74](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/tests/spectral_risk_contract.rs:74>).

### FMT-115 · P1 · El guard de notional aceptaba filtros inválidos y productos infinitos

**Estado:** corregido localmente; consumidor localizado en risk-engine/lib.rs, que no fue editado.

**Evidencia:** enforce_minimum_notional comprobaba margen y leverage, pero no el mínimo recibido ni la finitud del producto. Con mínimo=NaN, N<min_notional es falso y la función devolvía autorización. Un producto margen×leverage infinito podía superar cualquier mínimo finito y también ser aceptado.

**Prueba roja real:** el nuevo test obtuvo (true,10) al exigir (false,0) para un filtro inválido. Se añadieron casos de infinito, mínimo no positivo, overflow del producto y underflow, además de un caso válido exactamente en la frontera para impedir un endurecimiento indiscriminado.

**Reparación:** filtro finito y estrictamente positivo; producto finito, positivo y suficiente. Se documentó que el primer argumento representa margen monetario, no cantidad de activo ni notional ya multiplicado. El helper no dimensiona el riesgo, no eleva volumen y no presupone que todos los símbolos tengan mínimo 5.

**Límite:** otros callers pueden validar previamente estos datos; el hallazgo demuestra un defecto del contrato de esta API, no que una orden concreta con filtro NaN haya llegado al exchange. La validación de contratos individuales aporta defensa, pero no cierra por sí sola el problema de cantidades de FMT-113.

Código: [guard.rs:107](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/guard.rs:107>). Regresión: [spectral_risk_contract.rs:310](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/spectral_risk_contract.rs:310>).

### FMT-116 · P1 · Pisos de probabilidad y payoff convertían desventaja económica en edge

**Estado:** corregido localmente en la envolvente; el aprendizaje/restauración del payoff conserva otros problemas FMT-100/101.

**Primer mecanismo:** b=max(payoff,0,05) sustituía el cociente económico recibido. Con a=99.900,5, b_beta=100,5 y payoff=0,0001, una tasa de aciertos alta no compensa la magnitud de pérdida. La prueba roja observó una fracción autorizada de **0,25**, cuando Kelly con ese payoff debe ser no positivo.

**Segundo mecanismo:** mean y lcb se recortaban al mínimo 0,01. Con a=100,5, b_beta=99.900,5 y payoff=200, la media real es aproximadamente 0,001005, menor al break-even 1/201≈0,004975. El piso de 1 % inventaba ventaja; la prueba roja obtuvo **0,005047476261869065** de riesgo en vez de cero.

**Reparación:** media beta dentro del dominio [0,1], límite inferior aproximado capaz de llegar a cero y payoff positivo original, sin elevación artificial. La división q/payoff se valida mediante el resultado de Kelly, sin recurrir a un payoff ficticio cuando es muy pequeño.

**Significado científico:** estabilizar una ecuación no permite alterar los parámetros hacia una región favorable. Un floor numérico legítimo debe preservar una cota conservadora o declarar un modelo distinto; aquí ocurría lo contrario. Eliminar estos pisos no demuestra calibración del modelo, pero devuelve coherencia entre sus inputs y la decisión.

**Riesgo residual:** payoff_ratio continúa siendo una estimación suavizada con epsilon monetario; esta ronda no corrige ese estimador ni su restauración en el host, porque debe hacerse conjuntamente y sobre los cambios concurrentes adecuados. Tampoco reemplaza el LCB normal por un cuantil exacto o una secuencia de confianza válida bajo dependencia.

Pruebas: [spectral_risk_contract.rs:324](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/spectral_risk_contract.rs:324>) y [spectral_risk_contract.rs:339](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/spectral_risk_contract.rs:339>).

## 5. Reparaciones de hallazgos anteriores: qué quedó demostrado

### 5.1 FMT-096 — monotonía genómica sin cambio encubierto de hipótesis

Con SL=0,002, coste=0,001 y p=0,40, el resultado base requiere RR=2,75 y TP=0,0055. Antes, target_rr=2 activaba otra constante de probabilidad y bajaba TP a 0,004. Ahora un target menor que el RR aplicado deja la base intacta. Un target mayor sólo se aplica si su producto es finito y no reduce TP.

Las regresiones verifican mismo SL, mismo flag de piso, mismo rr_required, monotonía en target y EV condicional bajo el mismo p. La malla de ensayo contiene 129 valores logarítmicos entre 10⁻⁶ ms y 3,15576·10¹² ms, con ocho targets por escala: 1.032 combinaciones. Es un ensayo algebraico del dominio representable de 1 ns a cien años, **no** observación a resolución nanosegundo ni calibración con historia centenaria.

Se conserva el algoritmo base de stop. Corregir FMT-095 estrechando/ampliando barreras sin resolver la cantidad final podría modificar riesgo de manera no controlada; no se anuncia reparación de ese punto. También queda pendiente estimar cómo cambia la probabilidad de primera llegada cuando aumenta el TP.

Código: [tp_sl.rs:193](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/tp_sl.rs:193>).

### 5.2 FMT-098 — cero riesgo no se convierte automáticamente en exploración financiada

La rama que sustituía f≤0 por 0,015·peso_micro fue retirada de max_leverage. Dos pruebas rojas mostraron que tanto ausencia de evidencia como 500 pérdidas permitían (3,true) en una cuenta de 13 con stop 0,5 %. Ambas pasan ahora exigiendo (0,false).

Además se añadió prueba del consumidor real del replay, no sólo del helper. Con historial maduro 50/50 y payoff 1:1, una candidata micro se veta; su posición local se cierra y capital/margen se restauran. Otra prueba conserva aceptación de una candidata factible con evidencia positiva. El cambio no convierte cualquier cuenta pequeña en inoperable por una regla de tamaño.

**Importante:** el host/replay conservan un bootstrap propio cuando n<30. Esa política debe revisarse con presupuesto explícito y procedencia de evidencia; no se oculta bajo el estado “corregido localmente”. Un motor autoevolutivo debe poder distinguir desconocimiento, desventaja y restricción de ejecución. Obligar actividad no es demostrar aprendizaje.

### 5.3 FMT-099 — el mínimo del exchange no concede permiso para romper el presupuesto

Se eliminó la excepción que llamaba safe_micro_leverage a una autorización después de fallar la desigualdad de riesgo. La regresión usa evidencia positiva con f=0,015, C=13, d=10 % y mínimo=5: B=0,195, pero la posición mínima pierde nominalmente 0,50. Antes retornaba (1,true); ahora retorna (0,false).

El resultado no es “falta de inteligencia”: no existe acción positiva bajo esas restricciones. La API nueva permite distinguir factibilidad continua de lotes realizables. La exploración que quiera aceptar más pérdida debe modificar un presupuesto autorizado y auditable, no contradecir silenciosamente el actual.

La firma antigua mantiene el piso conservador de stop; la API nueva recibe d y c explícitos. Los contratos documentan esta diferencia para evitar una falsa afirmación de paridad por compartir un nombre.

## 6. Grafo vivo: frontera de integración que sigue abierta

~~~text
Genoma + estado espectral + datos/clock/versiones
                  │
                  ▼
Geometría / probabilidad / cantidad candidata
                  │
        ┌─────────┴─────────────────────────┐
        ▼                                   ▼
Envolvente corregida                  Host/replay actuales
  p, payoff, f                        bootstrap n<30 externo
  B=Cf, N_max                         cantidad del core intacta
        │                             adaptación de margen
        ▼                                   │
ExposureBudget.project_quantity             │
  lotes, mínimos, coste                     ▼
  rechazo de conjunto vacío             payload / ledger
        │
        └─ PENDIENTE: transportar Q proyectada, reservas y versión
                      a ambos consumidores sin carreras ni doble fee
~~~

La nueva API se usa internamente para el presupuesto de max_leverage y en pruebas de proyección. **project_quantity no tiene todavía un consumidor operativo**. No se dibuja esa arista como implementada para aparentar conexión.

La integración siguiente requiere, como una sola unidad revisable:

1. Obtener un snapshot coherente de capital disponible, propuesta, geometría y filtros del símbolo.
2. Diferenciar cantidad, exposición N/C y margen N/L; no derivar una de otra mediante clamps que pierdan el presupuesto.
3. Proyectar cantidad con fees/reserva de ejecución, mínimos, step y límites de cartera.
4. Recalcular reserva de margen, fees estimadas y estado de posición para la cantidad autorizada.
5. Revalidar precio y cantidad tras serialización/redondeo y antes del envío; definir qué ocurre si el mercado cambió.
6. Preservar la misma identidad de intención al recibir fills parciales o respuestas ambiguas.
7. Replicar la misma semántica en replay, además de comprobarla contra un oráculo de contabilidad independiente.
8. Mantener un presupuesto separado y finito de exploración, si se autoriza, con causa de aprobación y límites acumulados.

No se modificó el host para incorporar sólo el paso 3 dejando los restantes desalineados. El protocolo de concurrencia del proyecto impide tocar los cambios ajenos como si pertenecieran a esta tarea. Queda documentada la frontera para una ventana coordinada de integración; no constituye un bloqueo de los arreglos locales ya completados.

## 7. Teoría avanzada: integración con condiciones, no acumulación de nombres

### 7.1 Concretización de T10 — presupuesto nominal no es probabilidad de drawdown

Se contrastó [Risk-Constrained Kelly Gambling, Busseti–Ryu–Boyd](https://arxiv.org/abs/1603.06183), incluidos pasajes de sus secciones de modelo y cota de drawdown. El artículo trabaja con retornos IID, fracciones de apuesta y una alternativa de efectivo. La condición de momento negativo E[(rᵀb)^−λ]≤1, con λ=log(β)/log(α), es suficiente para acotar la probabilidad de caer por debajo de una fracción α de la riqueza inicial bajo sus hipótesis. No es la misma restricción que sobrevivir una racha estimada ni la misma métrica que drawdown desde el máximo histórico.

**Inferencia aplicada al proyecto:** no se puede reemplazar esa esperanza por una constante de racha y conservar el nombre de garantía. El motor tiene costes, selección adaptativa, dependencia, cambios de política y riesgo de ejecución; necesita una formulación compatible con esos elementos. Un stop nominal no determina toda la distribución de retorno.

Esta ronda implementó un presupuesto realizable bajo un modelo de pérdida nominal, no el optimizador del artículo. Para integrar después una restricción probabilística se necesitan distribución o conjunto de incertidumbre, condición de riqueza positiva, tratamiento de dependencia, presupuesto de evidencia y test de violaciones fuera de muestra. La alternativa de no abrir exposición debe seguir siendo factible.

### 7.2 Familias relacionadas que no deben confundirse

La búsqueda recuperó [Distributional Robust Kelly Gambling: Optimal Strategy under Uncertainty in the Long-Run](https://arxiv.org/abs/1812.10371). Su resumen plantea maximizar el peor crecimiento logarítmico entre distribuciones admisibles y señala que convexidad no implica tratabilidad general. Es una propuesta relevante para incertidumbre de modelo, no permiso para escoger después el conjunto que produzca el resultado deseado.

La expansión bibliográfica también localizó [Robust Analysis in Stochastic Simulation: Computation and Performance Guarantees](https://arxiv.org/abs/1507.05609): el resumen trata errores de especificación de distribuciones de entrada y análisis de peor caso. Aplicación candidata: someter evaluación de genomas a familias justificadas de slippage, latencia y rutas de fill, en vez de certificar con un único simulador favorable. No se implementó ni se validaron aquí todos sus teoremas.

[Robust valuation and risk measurement under model uncertainty](https://arxiv.org/abs/1407.8024) aporta una familia basada en incertidumbre media/volatilidad y modelos de super/subcobertura. Es un vecino matemático pertinente, pero su problema de cobertura/valoración no equivale automáticamente a predicción de alpha. Se conserva como referencia exploratoria basada en su resumen, no como componente recomendado para producción.

Las fuentes relacionadas se distinguen por alcance: el cuerpo de Busseti–Ryu–Boyd sí fue consultado para las afirmaciones anteriores; los otros trabajos se localizaron por metadata/resúmenes. No se anuncia una revisión integral de toda esa literatura.

### 7.3 T27 — primer componente implementado; garantía restringida

ExposureBudget implementa una proyección escalar de factibilidad nominal con retícula de lotes. No se vende como un QP multiactivo, una función de barrera ya demostrada ni un controlador estocástico completo. Su utilidad es verificable: evita confundir un mínimo continuo con una cantidad ejecutable y hace visible el presupuesto monetario.

La siguiente ampliación natural es control de riesgo agregado y reservas concurrentes, después un modelo de costes no lineal si los datos lo justifican. Añadir un solver antes de corregir la semántica de Q, N y L sólo complicaría el mismo error. Tampoco conviene aprender a saltarse restricciones por aumentar la frecuencia de trades.

### 7.4 Espectro continuo, física y computación cuántica

Las pruebas numéricas atraviesan el dominio temporal sin categorías scalping/swing. No se cambiaron el reloj de entrada, el soporte observacional, las curvas genómicas ni la calibración de Hurst. Un valor τ=1 ns aceptado por una función no convierte datos milisegundo en observaciones nanosegundo; un horizonte de cien años necesita extrapolación/prior y escenarios, no coeficientes supuestamente aprendidos sobre una historia inexistente.

Las analogías físicas son útiles cuando sus variables, unidades y condiciones representan un mecanismo. Una ecuación de difusión exige especificar drift, varianza, memoria, saltos y condiciones de frontera. Un grafo exige distinguir dependencia sintáctica, flujo causal de datos y dependencia predictiva. La conectividad perfecta de un diagrama no corrige una atribución errónea de PnL.

Los problemas del milenio no son un requisito para arreglar estas contradicciones. La complejidad matemática admisible debe mejorar una propiedad medible —incertidumbre, coste, estabilidad, identificación o control— frente a un baseline. No se formuló ninguna afirmación nueva sobre el estado de resolución de esos problemas.

No se implementó hardware cuántico ni se demostró ventaja cuántica. Se mantienen las propuestas anteriores T01–T28 y sus condiciones. Esta ronda prioriza correcciones matemáticas que ya tienen contraejemplos reproducibles; la ambición del sistema no justifica inventar garantías ni llamar omnisciencia a una aproximación finita.

## 8. Pruebas y evidencia reproducible

### 8.1 Secuencia rojo → verde observada

Se añadieron regresiones **antes** de reparar las causas:

| Bloque | Tests nuevos que fallaron inicialmente | Resultado anterior representativo |
|---|---:|---|
| RR, overflow, bootstrap, presupuesto y validación | 8 | TP menor; (3,true) sin edge; (1,true) pese a mínimo inviable |
| Mínimo de notional inválido | 1 | (true,10) con filtro NaN |
| Pisos de payoff/probabilidad | 2 | f=0,25 y f≈0,00504748 en casos sin edge |
| Varianza beta | 1 | Resultado no finito con parámetros grandes válidos |

**Total: 12 regresiones observadas fallando antes y pasando después.** No se cuenta un fallo de compilación como reproducción del bug.

Además se añadieron nueve tests de presupuesto/proyección y tres del consumidor de replay: **24 tests nuevos** en total. El barrido de proyección evalúa 768 combinaciones de capital, stop, precio, step y propuesta; verifica un número no trivial de aceptaciones para evitar un test vacuo que pase rechazando todo. No es una prueba exhaustiva de todos los binary64.

Propiedades verificadas: no aumentar la propuesta para alcanzar mínimos; no superar B después de redondear; mayor coste no aumenta cantidad; coherencia dimensional al cambiar unidades monetarias en casos seleccionados; representación entera de lotes; no permitir exposición por ausencia/corrupción de evidencia; rollback de candidata vetada y aceptación con evidencia positiva.

### 8.2 Comandos ejecutados

~~~text
cargo test -p risk-engine --offline --lib --test spectral_risk_contract
cargo test -p backtest-engine --offline --test spectral_risk_contract
cargo test -p backtest-engine --lib --offline envelope_
cargo check -p trader-gemini-v5 --bin god_engine --offline
rustfmt --check --edition 2021 [los cuatro archivos tocados de risk-engine]
rustfmt --check --edition 2024 crates/backtest-engine/tests/spectral_risk_contract.rs
git diff --check -- crates/risk-engine/src
~~~

Resultados finales: **49 tests existentes de risk-engine + 21 nuevos de risk-engine + 3 nuevos de backtest-engine + 3 existentes del gate = 76 aprobados**. Los tres tests existentes del gate fueron envelope_bootstrap_mantiene_entrada_que_sostiene_margen, envelope_autoritativa_veta_cuando_no_hay_edge y envelope_gate_ignora_posicion_ya_evaluada. cargo check del host terminó con éxito. Son pruebas seleccionadas, no la suite completa del workspace.

Advertencias observadas fuera de los cambios: variables latest_ts/mode y campo trades no usados en evolution-engine; import Arc no usado en continuous_evolution_backtest. No se ejecutó cargo fix ni se tocaron esos archivos ajenos para limpiar avisos.

Las pruebas del consumidor construyen arenas locales en memoria y no lanzan el proceso de trading. Cargo generó artefactos habituales de compilación; la suite de integración puede compilar binarios del paquete, pero no se ejecutaron evolvers ni binarios operativos. Hubo esperas por el lock de build; no se terminaron procesos ajenos.

### 8.3 Qué no está demostrado

No se hizo campaña completa de backtest, prueba financiera de rentabilidad, testnet, ejecución real, benchmark p99, simulación de pérdida de alimentación ni prueba completa de concurrencia. Las cifras de aciertos de tests no son evidencia de alpha, calibración universal, ausencia de bugs o capacidad para duplicar capital.

Tampoco se comprobó el cierre de todos los IDs históricos. FMT-095/097/100/101/109–113, entre otros, conservan trabajo material pendiente. La política resultante puede operar menos al eliminar autorizaciones injustificadas; eso requiere evaluación económica posterior, no revertir la matemática para alcanzar un conteo de órdenes.

## 9. Alcance de archivos y preservación del workspace

Se revisaron íntegramente de nuevo tp_sl.rs y kelly_envelope.rs; guard.rs se añadió a la cobertura completa de archivos preexistentes (122 líneas antes, 136 después de validación y formato). Los dos archivos de tests fueron creados y revisados en esta ronda; no se usan para inflar la cobertura del proyecto anterior.

**Cobertura acumulada FMT de archivos preexistentes: 91 Rust distintos sobre los 289 Rust versionados del corte.** Persisten 1.119 archivos versionados y 24 manifiestos Cargo; no se auditó el proyecto completo. Los tests nuevos todavía no versionados se contabilizan separadamente. Tramos del host, replay, estado de arena y callers siguen siendo lecturas dirigidas, no lectura integral.

### 9.1 Manifiesto de cambios propios

SHA-256 abreviado para identificar el contenido final observado, no firma de despliegue:

| Archivo | Líneas | SHA-256, prefijo |
|---|---:|---|
| crates/risk-engine/src/kelly_envelope.rs | 478 | 93A7D4C23EB12605 |
| crates/risk-engine/src/tp_sl.rs | 402 | AD0B72FB86901D53 |
| crates/risk-engine/src/guard.rs | 136 | AAE534AEBF9D2F57 |
| crates/risk-engine/tests/spectral_risk_contract.rs | 366 | 9D2BCA2FEA0DBABC |
| crates/backtest-engine/tests/spectral_risk_contract.rs | 92 | BD51D0AC5119CA06 |

Los tres archivos fuente estaban limpios al inicio de esta intervención. Hashes previos: tp_sl 9A3A34A9E0A12E01; kelly_envelope 40957EB957D3F427; guard E8AAB6C2D92E153F. Se comprobó su estado antes de editar y se usaron cambios dirigidos más formato local, no formateo de todo el workspace.

### 9.2 Archivos ajenos preservados

No se editaron src/bin/god_engine.rs, crates/backtest-engine/src/booktick_replay.rs ni crates/risk-engine/src/lib.rs. Sus hashes se contrastaron durante el trabajo con los del inicio:

| Archivo | SHA-256, prefijo inicial |
|---|---|
| src/bin/god_engine.rs | F2BA92C4283C66E6 |
| crates/backtest-engine/src/booktick_replay.rs | 736CC1DBB38E21D7 |
| crates/risk-engine/src/lib.rs | E9883455B6BEBEFA |

Se conservaron configuraciones, genomas y artefactos concurrentes. Los informes se amplían mediante adiciones; la declaración histórica “sólo documentación” de la ronda V describe esa ronda, no la VI. La continuación explicita el nuevo estado para evitar confundir auditoría anterior con reparación presente.

La investigación se hizo mediante la habilidad Firecrawl y su índice conectado de papers porque la CLI no estaba disponible. Se usaron preguntas científicas genéricas; no se enviaron archivos del proyecto al servicio ni se instaló software. Su contribución material fue separar presupuesto nominal, heurística de racha y garantía probabilística, y documentar qué integración avanzada exige supuestos todavía no verificados.

### 9.3 Cierre de integridad documental y técnica

Después de añadir esta ronda se verificaron los SHA-256 de los prefijos completos anteriores de ATLAS_ANALITICO.md, INFORME_FORENSE_MAESTRO.md y el anexo V, normalizando únicamente CRLF/LF: los tres permanecen idénticos. Los cinco archivos propios conservan los hashes de 9.1 y los tres archivos ajenos de 9.2 conservan sus hashes iniciales. Esta comprobación acredita esos contenidos concretos, no ausencia de cambios de otras sesiones en todo el repositorio.

Se comprobaron los 15 enlaces locales de este anexo —archivo existente y línea dentro del archivo— y los tres encabezados nuevos FMT-114–116, sin duplicados. También se validaron los 37 enlaces locales con ese formato del anexo V y los cinco de cada índice. Los checks de formato de los cinco Rust y git diff --check sobre los tres fuentes e índices terminaron sin errores. El informe maestro ya contenía bytes NUL antes de esta ronda; la comparación de prefijo demuestra que no se introdujeron ni se eliminaron con esta ampliación. No se normalizó ese documento histórico en una edición colateral.

## 10. Hoja de ruta de cierre profesional

1. **Coordinar FMT-113.** Integración atómica de cantidad proyectada, reserva, fees, filtros, adaptador de margen y payload en host/replay sobre sus cambios actuales. Es la prioridad pendiente, no una recomendación de desplegar esta ronda aislada como solución global.
2. **Revisar la exploración externa n<30.** Definir presupuesto acumulado, selección de observaciones y criterio de detención. No mezclar evidencia desfavorable con ignorancia.
3. **Corregir la geometría conjuntamente.** FMT-095/097: piso difusivo, cap de stop, probabilidad de primera llegada y coste. Evitar ampliar stops sin redimensionar la cantidad.
4. **Corregir la evidencia de aprendizaje.** FMT-100/101/109–111: resultados crudos, normalización monetaria, identidad de operación, journaling y recuperación. No convertir un cambio de kernel estadístico en una migración implícita del estado persistido.
5. **Validar cambios con secuencias retenidas y coste.** Separar corrección de invariantes de mejora de rentabilidad. Medir discrepancia de decisiones, rechazos, riesgo nominal, fills y latencia de colas; conservar todas las consultas de selección.
6. **Continuar la auditoría completa.** Mantener manifiesto de cobertura real y estados por evidencia. No presentar 91 archivos como “todos”, ni introducir más teorías antes de identificar qué propiedad concreta falta.

**Conclusión:** esta ronda entrega reparación local, regresiones reproducidas y una base de proyección monetaria con unidades claras. No transforma por decreto el sistema en autoevolutivo, omnisciente o cuántico. La evolución verificable exige que los genes afecten acciones factibles y que los resultados regresen como evidencia íntegra; las aristas pendientes están identificadas y no se ocultan tras una batería verde.

## Continuación VII — soporte y evidencia en los estimadores y la selección

La [Auditoría científica VII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VII_2026-09-24.md>) añade FMT-117–123 y repara localmente FMT-011. Se corrigen homogeneidad y log-retornos DFA, validez separada del clipping, RNG de pruebas, semántica del campo de evidencia live y ranking de evaluaciones inválidas. El exponente publica pendiente cruda y soporte; el evolver usa un caller explícito de backtest que no confunde crecimiento con Sharpe live.

Se agregan dieciséis tests. Seis regresiones de defectos anteriores fallaron antes del arreglo; otra detectó un defecto de precisión de la propia versión intermedia, corregido y documentado separadamente. El nuevo informe contiene pruebas, fuentes primarias, fórmulas, efectos y límites. La cobertura completa sube a 92 Rust preexistentes distintos, no al proyecto entero.

FMT-117 continúa abierto para reloj y propagación de soporte. FMT-113 tampoco se cierra: se preservaron los cambios ajenos del host, replay y risk-engine/lib. No hubo trading, modificación de genomas, despliegue ni operaciones Git de publicación/integración. El estado y los resultados de esta ronda VI se conservan como evidencia de su propio corte.
