# Auditoría XXVI — acción ejecutable, vetos económicos y consistencia del genoma

Fecha: 2026-09-25. Continuación de [XXV](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXV_2026-09-25.md>). Artefacto: [evidencia estructurada XXVI](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVI_2026-09-25.json>).

## 1. Dictamen y límites de aseguramiento

Se reparan contratos locales de admisión: el apalancamiento aceptado por la ruta principal de riesgo queda entero y acotado por el genoma y el instrumento, el coste sobre margen se comprueba después de los ajustes, y el constructor de órdenes rechaza determinados dominios numéricos inválidos antes de firmarlos. La ruta execute_order deja de truncar silenciosamente leverage. Su éxito paper ya no precede a las comprobaciones básicas de precio, margen, paso, señal, producto finito y leverage.

No se certifica la seguridad integral, la equivalencia backtest/demo/producción ni una capacidad autoevolutiva demostrada. Tres diagnósticos nuevos reproducen deuda real: el lote puede perder el mínimo nocional, el ajuste por tolerancia puede aumentar cantidad y el precio maker puede modificar el nocional sin volver a presupuestarlo. El último caso produce 44 de nocional límite a partir de 40 calculado al precio de referencia. Son ejemplos sintéticos deterministas; no pruebas de pérdidas ni de órdenes reales enviadas.

Se añaden FMT-217, FMT-218 y FMT-219. Se actualizan FMT-214 y FMT-175; FMT-215 y FMT-216 siguen abiertos. La matriz histórica de 305 puntos y los informes anteriores no se reemplazan: esta es una ampliación con estados de alcance explícito, no una renumeración ni un certificado universal.

Cobertura acumulada de la serie: 142/289 archivos Rust preexistentes con lectura completa registrada; 147 siguen pendientes. Se añade la lectura completa de binance_api.rs. El inventario base de 1.119 archivos y 24 manifiestos no equivale a revisión lógica de todos ellos. executor.rs se ha leído por rutas y tramos: no se cuenta como lectura completa. Tampoco se cuenta un rg, un hash o una compilación como auditoría semántica archivo por archivo.

## 2. Paradigma de grafo vivo: dónde se rompe la conservación de la decisión

La unidad que interesa conservar no es una etiqueta de estrategia, sino una acción identificable bajo un estado y una evidencia definidos. El grafo diagnóstico de esta ronda es:

```text
RAÍZ: instrumento + observaciones + tiempos + versión del genoma/spec
  │
  ├─ espectro multivariante → intención (dirección, horizonte, confianza)
  │                              │
  └─ estado de cartera ───────────┤
                                 ▼
DECISIÓN: payoff, coste, capital, margen m, leverage L, TP/SL
  │                         [contratos locales reforzados]
  ▼
PROYECCIÓN: precio ejecutable P*, cantidad q*, filtros y cuenta
  │                         [minN/lote/valoración todavía desconectados]
  ▼
TERMINAL: firma → envío → ACK/Unknown → fills → posiciones/protección
  │
  └─ atribución económica y de rechazo → aprendizaje/versionado → RAÍZ
                                [cierre causal no certificado]
```

Un nombre como ValidatedOrder no demuestra que la orden conserve sus propiedades después de proyectar precio y cantidad. Tampoco firmar demuestra que el exchange la admita, ni un ACK demuestra un fill. La reparación XXVI se sitúa entre decisión y construcción de payload; no integra por sí sola todas las aristas posteriores.

La continuidad temporal y la discretización de un actuador son conceptos diferentes. La inferencia puede parametrizar horizonte, intensidad de volatilidad y dependencia de forma continua; el mercado exige cantidades y precios pertenecientes a mallas y este endpoint exige leverage entero. El error no es reconocer esa restricción, sino calcular riesgo con una acción y ejecutar otra sin recalcular ni conservar evidencia.

## 3. Matriz de resolución de esta ronda

| ID | Prioridad y alcance | Evidencia | Estado |
|---|---|---|---|
| FMT-214 | P1, ruta principal de riesgo y execute_order | Rescate de mínimo devolvía L=4,121212121212122 incluso con techo genómico 1 | Contrato entero/techo reparado localmente; proyección conjunta de exposición abierta |
| FMT-217 | P1, veto de coste | Presupuesto de coste bajo podía omitirse si la orden no entraba en ramas de rescate; valores inválidos no tenían rechazo explícito | Validación de dominio y evaluación final reparadas; política micro y modelo de costes abiertos |
| FMT-218 | P1, constructor y admisión básica del ejecutor | L fraccionario o fuera de dominio, productos/lotes no representables, compra maker desplazada por mínimo tick | Reparación local; rutas alternativas y filtros completos no certificados |
| FMT-175 | P1 para esta evidencia en caller operativo; antecedente P2 en helper | N=5,10255 termina en 3; q=0,3999999999 termina en 0,4 | Se amplía la deuda abierta del ejecutor; no se atribuye al helper ya reparado |
| FMT-219 | P1, valoración maker | m=10, L=4, P_ref=100, tick=10: q=0,4 y P*=110 → N*=44 | Abierto y reproducido |
| FMT-215 | Antecedente, vetos auxiliares/cartera | Cuatro diagnósticos siguen reproduciendo desconocidos admitidos o máximo 1 convertido en 2 | Sin corrección en XXVI |
| FMT-216 | Antecedente, factibilidad del genoma | Semilla 199 de XXV y prueba de promoción intermitente | No se relaja el veto ni se declara resuelto; no se repite aquí la suite completa de quantum-arena |

P1 expresa severidad potencial si la ruta y las condiciones se activan; no atribuye incidencia productiva sin trazas. Un test diagnóstico que pasa porque el defecto sigue presente no se contabiliza como contrato funcional reparado.

## 4. FMT-214 — el entero inicial no sobrevivía a los rescates posteriores

### Causa, propagación y efecto

La ruta cuantizaba la salida de QuantumLeverageMatrix al principio, pero volvía a asignar leverage con divisiones destinadas a alcanzar mínimo nocional. El primer rescate calculaba aproximadamente L_c=(N_min/m)×1,02. Si esa cifra era fraccionaria, el entero inicial dejaba de representar la acción final. Además, el rescate no conservaba el techo del genoma en todas sus reasignaciones.

El fixture sintético con capital 13, precio 100, ATR 1 y mínimo 5 devolvió margen 1,2375 y L=4,121212121212122 antes de la reparación. El barrido con techo genómico 1 también observó esa salida. La cuantización inicial no bastaba. En el ejecutor, convertir posteriormente L mediante as u32 daba 4, mientras la cantidad seguía calculada con el leverage fraccionario.

En unidades de cotización, N=mL. Si se dimensiona N con L_d y se configura el exchange con L_e, el margen nominal implícito pasa a m_real=N/L_e=m×L_d/L_e, antes de fees, tier y otras exigencias. Para 4,121212…→4 la diferencia es aproximadamente 3,03%; para un techo genómico 1, admitir más de 4 es además una violación independiente del techo. Son dos contratos distintos: representación y autorización de riesgo.

### Reparación aplicada

El valor genómico se valida antes de combinarlo con el techo del instrumento, para que NaN o un valor no positivo no se conviertan en otro límite por min o un fallback. La salida de la matriz debe ser finita y al menos 1. Cada reasignación intervenida se acota por el techo y se cuantiza antes de evaluar sus consecuencias. Al final se vuelve a comprobar finitud, integridad y techo, y el mínimo se verifica mediante el producto final, no únicamente una división intermedia.

Se incorpora [ValidatedOrder::integer_leverage](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:36>): devuelve Option y no trunca ni satura. Representa enteros positivos que caben en u32; el adaptador Binance impone adicionalmente el máximo 125 del esquema. Ese máximo no sustituye las restricciones inferiores del instrumento ni de la cuenta. La [documentación primaria del endpoint](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade#change-initial-leverage) exige un entero entre 1 y 125 y devuelve también maxNotionalValue.

Evidencia de ruta: [techo genómico](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:493>), [primer rescate](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:832>), [segundo rescate y comprobación final](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:881>), [contratos reproducibles](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/leverage_admission_contract.rs:48>).

### Lo que no queda resuelto

Floor preserva un techo, pero no resuelve por sí solo el problema conjunto de factibilidad y utilidad. El algoritmo conserva una ampliación de margen para alcanzar el mínimo, un buffer absoluto 0,1 y un multiplicador 1,0005. Sigue pudiendo modificar la exposición calculada inicialmente. No se ha demostrado que cada orden acepte únicamente una pérdida monetaria menor o igual al presupuesto original.

Tampoco se ha demostrado completitud: un rescate truncado puede rechazar aunque exista otra combinación entera admisible de L y margen. Elegir ceil indiscriminadamente sería incorrecto porque puede aumentar riesgo. El cierre requiere buscar dentro del conjunto admisible con todos los presupuestos, no cambiar un redondeo para incrementar la tasa de aceptación.

## 5. FMT-217 — un veto económico dependía del camino de control

### Error lógico

El coste sobre margen se comprobaba en algunas ramas que intentaban rescatar una orden demasiado pequeña. Una orden ya suficientemente grande podía omitir esa restricción. A la inversa, comprobar un leverage candidato antes de limitarlo podía rechazar por el coste de una acción que nunca se enviaría. Validar un intermedio y omitir el resultado final no implementa un presupuesto de acción.

El test usa capital 100, métricas sintéticas maduras y max_fee_pct=0,0001. El roundtrip de la fixture es 0,00082 por unidad de nocional; cualquier L≥1 tiene coste sobre margen superior a ese presupuesto en el extremo estándar de la política. La orden debía abstenerse independientemente de que necesitase rescatar el mínimo. El test fallaba antes y pasa después. También se prueban NaN, infinito y valor negativo del presupuesto, que ahora provocan rechazo de entrada inválida.

### Unidades y significado del cálculo

Si c es coste fraccional de ida y vuelta sobre nocional, C=cN es coste monetario y C/m=cL es la fracción de margen consumida por ese coste. No es probabilidad de ruina, no es pérdida total al stop y no incorpora por definición financiación o impacto que el estimador c no modele. La implementación ahora comprueba esa magnitud después de todas las reasignaciones de leverage intervenidas.

Se conserva explícitamente la política previa B(w)=(1−w)b_gen+w×0,035, donde w es el peso micro de asignación. Se valida b_gen y se verifica cL_final≤B(w). Por tanto, no se afirma que max_fee_pct sea un techo duro universal del genoma: cerca del extremo micro, el literal 0,035 puede dominar el gen y relajar o endurecer su valor. Ese desacoplamiento de semántica evolutiva sigue abierto y debe resolverse con un contrato de política, no ocultarse en el informe.

Evidencia: [definición de presupuesto](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:806>), [veto final](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:900>), [regresión independiente del rescate](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/tests/leverage_admission_contract.rs:91>).

### Requisitos de cierre económico

Falta vincular el coste a la acción ejecutable: maker/taker efectivo, instrumento, tamaño, spread observado, profundidad, latencia, comisiones por moneda, financiación y liquidación. Hace falta distinguir coste estimado, incertidumbre y coste realizado, y registrar qué política/versión fijó B. Menos rechazos no prueba mejor política; deben medirse falsos aceptados y falsos rechazados mediante evaluación causal fuera de muestra, sin realizar órdenes para probar una hipótesis de auditoría.

## 6. FMT-218 — el nodo terminal confiaba demasiado en el nombre ValidatedOrder

### Dominio numérico y contrato externo

build_payload es una API pública y puede recibir órdenes de consumidores distintos de la ruta principal. Antes podía firmar cantidades calculadas con leverage fraccionario o con aritmética no representable. Un paso positivo subnormal f64::from_bits(1) hace que el inverso desborde; comparar solamente final_quantity==0 no rechaza NaN. Una firma criptográfica válida no convierte NaN en una cantidad negociable.

Ahora se exigen margen y precio positivos finitos, paso positivo finito, leverage entero del dominio del adaptador, producto de nocional finito positivo, cociente de cantidad finito positivo y cantidad redondeada finita positiva. En maker se exige además tick válido, precio final positivo finito y geometría de lado respecto de la referencia. En execute_order se comprueba la admisión básica antes del éxito paper, la mutación de leverage y el envío. Se rechaza Flat y overflow de mL en esa entrada.

El tick se valida cuando se usa para una orden límite maker. Se añadió un control positivo que permite construir MARKET con tick no utilizable, porque ese payload no contiene precio límite: rechazarlo en esta función por un dato que no usa sería un veto espurio. La ruta operativa todavía consulta un filtro con tick antes de build_payload; este cambio no demuestra que haya desaparecido esa dependencia más arriba.

### Compra maker por debajo del primer tick

Con referencia 0,005 y tick 0,01, el helper aplicaba max(tick) y obtenía precio 0,01: no cero. Una prueba inicial describía incorrectamente el motivo del rechazo esperado. Se corrigió el nombre y la explicación manteniendo el testigo: una compra maker no debe desplazarse por encima de su referencia a causa del fallback al primer tick. Se contiene el caso en build_payload, sin modificar globalmente el helper que usan otras rutas.

La nueva comprobación de lado no prueba pasividad contra el libro. Puede faltar bid/ask contemporáneo, el precio recibido puede no ser el mid y el libro puede cambiar antes de la llegada. Se corrigió el comentario que decía «nunca cruza»; la garantía post-only corresponde al exchange bajo GTX. Los [filtros oficiales y time-in-force](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/common-definition) fundamentan la distinción entre restricciones por tipo de orden y condiciones del mercado, no una supuesta certeza sobre ejecución.

Evidencia: [builder](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:1385>), [geometría maker](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:1448>), [conversión exacta en execute_order](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:1988>), [suite terminal](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/payload_admission_contract.rs:19>).

### Alcance de seguridad que falta

No se unificaron execute_raw_qty, limit, IOC, iceberg, maker-chase ni protección/cierre. El helper de redondeo permanece compartido y algunas rutas sólo comparan contra cero; eso es una superficie pendiente, no una afirmación de que todas envíen NaN en condiciones reales. Cambiar su semántica sin revisar las salidas defensivas podría impedir reducir riesgo.

execute_order conserva un éxito paper anterior al cálculo completo del payload y un cambio de leverage anterior a obtener/proyectar todos los filtros. Por ello aún puede haber divergencia paper/real y efectos de cuenta antes de un rechazo local posterior. La reparación sólo garantiza que los defectos básicos enumerados no lleguen a ese punto.

## 7. FMT-175 ampliado — la proyección conservadora no está integrada en el terminal

### Dos testigos distintos

Primero: m=1,2756375, L=4 y P=100 dan N=5,10255 y q_raw=0,0510255. Con paso 0,03, build_payload devuelve q=0,03 y N_final=3. El resultado ya no satisface un mínimo 5, aunque la decisión previa superaba 5,1. La firma se construye porque build_payload no recibe el mínimo nocional ni revisa ese contrato después del lote.

Segundo: q_raw=0,3999999999 y paso 0,1 terminan en q=0,4. snap_floor aproxima al entero próximo dentro de una tolerancia relativa. Ese comportamiento ayuda a evitar ciertos residuos de serialización, pero no conserva la desigualdad q_final≤q_autorizada. La diferencia del ejemplo es pequeña; no se exagera su impacto monetario. Lo relevante es el contrato, especialmente si otras etapas confían en una proyección siempre conservadora o amplifican el resultado.

Evidencia: [diagnóstico de mínimo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/payload_open_diagnostics.rs:20>), [diagnóstico de cantidad aumentada](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/payload_open_diagnostics.rs:38>), [tolerancia del ejecutor](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:44>).

### Diferencia con la reparación XVIII

La ronda XVIII mejoró un helper de proyección y documentó que no estaba unificado con round_to_step_size del ejecutor. XXVI no revierte aquella mejora ni vuelve a contarla como nueva. Ahora se reproduce el desacople en el constructor operativo: el estado de FMT-175 debe distinguir API auxiliar reparada y consumidor terminal todavía abierto.

Los filtros públicos tienen mínimo, máximo, paso y origen; MARKET_LOT_SIZE no se reduce necesariamente a LOT_SIZE. El [contrato oficial USD-M](https://developers.binance.com/en/docs/products/derivatives-trading-usds-futures/common-definition) también especifica la referencia del mínimo nocional para MARKET. Un único precio local sin tipo, origen y tiempo no acredita esa comprobación.

SymbolFilter sólo representa step_size, tick_size y min_notional. La lectura revisada conserva min_notional con fallback 5 en error de parseo y no transporta minQty/maxQty, el filtro separado de mercado ni todos los límites de precio. Un valor ausente, desactivado o corrupto no debería confundirse con un mínimo universal inventado. Esto se registra como continuación de la deuda de filtros, no como corrección realizada. Ver [estructura](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:102>) y [parser](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:1610>).

### Criterio de cierre

Una proyección común debe calcular con metadatos versionados del instrumento y del tipo de orden, preservar la cantidad/riesgo autorizados, representar exactamente la malla decimal en la serialización y revalidar las restricciones finales. El mismo componente y las mismas razones de rechazo deben usarse en simulación y ejecución, con diferencias explícitas de la fuente de precio y del modelo de fills. No basta copiar el helper reparado si pierde la identidad de filtros o se vuelve a redondear después.

## 8. FMT-219 — el precio límite modifica el valor de la acción

El builder calcula primero q=floor_lote(mL/P_ref) y después obtiene el precio maker P*. Esa secuencia no garantiza qP*≤mL. En una venta sintética con m=10, L=4, P_ref=100, paso 0,001 y tick 10 se obtiene q=0,4 y P*=110. El valor del límite es 44, un 10% superior al nocional de referencia 40.

El ejemplo no demuestra un consumo real de margen de 11 ni una pérdida realizada: el exchange puede usar mark price, tiers y reglas específicas, y la orden puede no llenarse. Sí demuestra que el plan usa dos valoraciones diferentes sin expresar cuál autoriza el presupuesto. En una compra, un precio menor puede a su vez cruzar el mínimo nocional hacia abajo. La geometría de TP/SL y el payoff evaluado también pueden cambiar al alterar entrada; no basta corregir la división para afirmar equivalencia estadística.

Evidencia reproducible: [test maker](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/tests/payload_open_diagnostics.rs:53>). No se repara a ciegas cambiando q por mL/P*: falta saber qué exposición quiere conservar el host, cómo reconcilia cantidad/margen y con qué referencia calcula riesgo el venue. La corrección debe producir un ExecutionPlan único con precio/valoración, cantidad, coste y pérdida máxima aproximada bajo supuestos declarados, y verificarlo antes de efectos de cuenta.

Para cierre se necesitan controles positivos de ambas direcciones, ticks grandes/pequeños, borde del primer tick, min/maxQty, mínimo nocional, rejillas desplazadas, tasas, targets explícitos y revalidación si cambia la referencia. Después, pruebas de integración con un proveedor simulado que compruebe ausencia de efectos antes de aceptación; no pruebas improvisadas enviando órdenes a una cuenta.

## 9. Auditoría de vetos: qué significa cada rechazo

| Familia | Justificación válida | Qué no demuestra | Estado observado |
|---|---|---|---|
| Entrada numérica | La acción no es representable o sus datos esenciales son inválidos | Falta de ventaja de la estrategia | Mejorada en las rutas enumeradas; faltan motivos tipados comunes |
| Dominio del venue | Leverage entero, lotes y precios dentro de filtros aplicables | Que un horizonte sea válido o inválido por sí mismo | Entero reforzado; proyección de filtros incompleta |
| Coste presupuestado | cL supera el presupuesto explícito de margen | Probabilidad de ruina o EV calibrado | Gate final añadido; c y política micro pendientes |
| Mínimo de orden | La acción autorizada no pertenece al conjunto admisible | Permiso para aumentar exposición hasta lograr un fill | Upsizing histórico y post-lote abiertos |
| Post-only | La orden tomaría liquidez según el libro al llegar | Error que deba eludirse convirtiendo automáticamente a taker | Geometría local contenida; evidencia del libro pendiente |
| Datos/estado desconocidos | No hay evidencia suficiente para abrir riesgo | Confirmación de un rechazo remoto ni permiso para bloquear gestión defensiva | Antecedentes XXV y rutas auxiliares abiertos |
| Régimen/cluster | Sólo si corresponde a un presupuesto de dependencia explícito | Un estado físico discreto del mercado o independencia entre horizontes | Discretizaciones globales y cardinalidad persisten |
| Factibilidad de genoma | El candidato no admite acciones bajo el dominio declarado | Obligación de relajar el gate para que toda mutación opere | FMT-216 abierto; no se debilitó el filtro |

Un veto adaptativo debe tener entidad, versión, datos y motivo; no basta un contador agregado. Propuesta de registro pendiente: decision_id, instrumento, entorno, timestamps de evento/recepción, versión de genoma/spec, etapa, acción candidata/final, restricción, unidades, observado/límite, incertidumbre y estado accepted/rejected/unknown. Es un contrato propuesto, no telemetría implementada en esta ronda.

El denominador importa: una elevada fracción de rechazos puede venir de datos corruptos, candidatos imposibles, políticas demasiado rígidas, costes altos, saturación o un alpha débil. No se debe atribuirla a «inteligencia bloqueada» sin separar esas causas. La optimización tampoco debe premiar sólo aprobación: eso incentiva eliminar defensas en lugar de mejorar oportunidades.

## 10. Revisión matemática y semántica evolutiva

### 10.1 Espectro continuo y actuación finita

Se mantiene abierto el recorte de expected_duration_ms a 1.000..43.200.000 en un lector de riesgo ([evidencia](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:974>)). Cambiar nombres de scalping/swing no amplía ese soporte. Un contador en milisegundos tampoco representa distinciones submilisegundo sólo por documentarlas como continuas.

Una especificación útil separaría tiempo de evento t, horizonte τ, resolución observada y soporte de estimación. Definir u=log(τ/τ_ref) da una coordenada adimensional para evaluar funciones sobre escalas, no datos nuevos ni resolución infinita. Una implementación finita podría refinar su aproximación donde el error/importancia lo justifique; debe declarar extrapolación y error fuera del soporte. Representar formalmente desde 1 ns a 100 años no significa disponer de información en todos esos horizontes ni poder recalcular todos los activos cada nanosegundo.

La variación de volatilidad y dependencia también puede describirse mediante magnitudes continuas e incertidumbre. El consumidor de market_regime sigue convirtiendo un estado global a una categoría antes de allow_trade; XXVI no elimina esa discretización. Su reemplazo exige pruebas de los presupuestos que actualmente impone, especialmente cartera conjunta y acciones defensivas.

### 10.2 Del genotipo a la acción realmente evaluada

En el código revisado global_leverage y max_fee_pct se cargan desde el genoma y se publican en atomics. FMT-214 muestra que publicar un gen no garantiza respetarlo después de modificar la acción. FMT-217 muestra que una política interpolada puede dominar otro gen según el entorno de capital. Por eso una buena puntuación de backtest no prueba que el mismo fenotipo se ejecute en demo/producción.

Para atribuir una mejora evolutiva es necesario poder reconstruir candidato → configuración efectiva → intención → proyección → resultado, conservando versión y contexto. Las lecturas atómicas individuales tampoco prueban un snapshot compuesto del mismo genoma/mercado. XXVI no modifica el publicador ni demuestra coherencia entre todos los lectores. No se entrenó, mutó ni promovió un genoma activo.

### 10.3 Factibilidad como problema explícito, no cascada de rescates

Contrato matemático propuesto para una acción a=(dirección,q,P,L,TP,SL), no implementado globalmente:

```text
q ∈ malla del instrumento/tipo de orden; P ∈ malla si es LIMIT
1 ≤ L ∈ enteros ≤ min(techo genómico, techo de instrumento/cuenta)
N_ref(a) satisface los límites del venue con la referencia apropiada
riesgo_estimado(a, estado, costes) ≤ presupuesto autorizado
margen_estimado(a, reglas de cuenta) ≤ margen reservado disponible
datos y parámetros esenciales finitos, con identidad/versiones compatibles
```

La admisión es pertenencia a ese conjunto, no la maximización de un score por sí sola. La elección de una acción requiere una función objetivo separada y validada; si no hay acción factible debe abstenerse sin inventar capital ni exposición. El número finito de enteros del endpoint hace posible explorar ese componente con límites conocidos, pero no resuelve automáticamente el resto de la optimización ni justifica maximizar leverage.

La expresión orientativa q×|P_entrada−P_stop|+costes aproxima pérdida hasta una barrera bajo supuestos de ejecución; no es una cota garantizada ante gaps, slippage o fallos de protección. El informe no usa esa aproximación como garantía de solvencia ni sustituye riesgo conjunto por sumar stops.

### 10.4 Criterio para integrar teorías avanzadas

No se incorporaron ecuaciones de problemas abiertos, operadores denominados cuánticos ni bibliotecas nuevas para añadir complejidad nominal. Antes de integrar una teoría deben existir variable observable, unidades, dominio, supuestos, mecanismo causal o predictivo contrastable, coste computacional y un test capaz de refutar su contribución. El nombre matemático no compensa una desconexión entre nocional aprobado y payload.

Se corrigió una afirmación concreta de complejidad en binance_api.rs: que HMAC produzca un digest de longitud fija no hace O(1) el procesamiento del mensaje/clave. La función conserva su implementación; sólo se precisó la descripción. No se midió su latencia ni se demostró una mejora de rendimiento.

## 11. Recorrido por módulos y asuntos pendientes

| Módulo del informe maestro | Conexión relevante en XXVI | Límite de lo verificado |
|---|---|---|
| 1. Ingestión/L2/normalización | Precio de referencia y filtros deben tener identidad y frescura | No se auditó de nuevo todo el L2; no se fabrican bid/ask para probar pasividad |
| 2. IA/modelos/señales | p usada por EV no cambia contractualmente al modificar precio/targets | No se certifica calibración ni se reentrenan modelos |
| 3. Estrategia multiactivo/horizontes | Persiste recorte temporal y régimen global; cardinalidad no mide dependencia | No se declara sistema continuo universal completado |
| 4. Ejecución/red | Entero y dominio de payload reforzados; lote/precio/filtros abiertos | No se enviaron órdenes ni se probaron todas las rutas |
| 5. Riesgo/Kelly/genomas | Techo de leverage y coste final reparados en ruta principal | Presupuesto de pérdida, upsizing, política micro y snapshot pendientes |
| 6. Estado/telemetría/SO | No hay plan final versionado común; contadores no bastan para causalidad | Sin prueba de sincronía global, p99 ni benchmark productivo |
| 7. Confluencia/cuántica | No se agregó formalismo nominal para sustituir contratos | Sin evidencia nueva de ventaja cuántica ni omnisciencia |
| 8. Backtest/gobernanza | Paper salta todavía la validación completa; tests distinguen reparación/deuda | Paridad productiva y auditoría integral siguen abiertas |

En latencia, fetch_all_symbol_filters conserva hasta dos intentos configurados a 10 s separados por 250 ms. Es un presupuesto de espera de esa ruta, no una medición p99 ni un límite exacto de tiempo total. Un miss de caché puede introducir I/O de exchangeInfo en el camino operativo; se debe medir y separar actualización de metadatos, frescura y decisión de admisión. No se cambió la política de reintento ni se invocó ese endpoint desde las pruebas.

## 12. Pruebas y evidencia negativa

Se añadieron 14 contratos funcionales: cinco de riesgo y nueve de construcción/admisión terminal. Se añadieron tres diagnósticos que reproducen deuda abierta. Un diagnóstico de XXV sobre leverage fraccionario pasó a ser regresión del contrato entero; los otros cuatro permanecen diagnósticos abiertos.

Hay ocho reproducciones distintas con fallo observado y pase tras reparación que se contabilizan conservadoramente: cinco de riesgo, leverage terminal inválido/fraccionario, cantidad no representable con paso subnormal y el testigo maker del primer tick. Este último se renombró porque la explicación inicial «precio no positivo» era incorrecta: el valor era positivo pero del lado equivocado. La prueba mixta de dominios numéricos se afinó para evaluar tick sólo en maker y se añadió un control MARKET; no se suma como una novena reproducción rojo→verde idéntica tras cambiar ese alcance. Errores de fixture o de descripción no se atribuyen a producción.

Ejecuciones ampliadas: risk-engine --lib --tests da 108 pases, de los cuales ocho reproducen deuda abierta (tres de taxonomía, uno de top-k, cuatro vetos auxiliares). execution-engine aporta nueve contratos nuevos, tres diagnósticos nuevos, cinco pruebas preexistentes de redondeo/precio y dos de firma. Total de pruebas distintas: 116 funcionales y 11 diagnósticos abiertos, 127 pases observados. Las repeticiones no se suman. Esto no equivale a toda la suite del workspace verde.

cargo check --offline pasó para god_engine, feature_exporter, train_forest y train_dark_alpha. Persisten los tres warnings preexistentes de evolution-engine: latest_ts, mode y campo trades. El fallo intermitente de promoción registrado en XXV no se borra por no haber repetido esa suite aquí. No hubo benchmark, atribución de PnL, simulación de cien años ni certificado de tiempo real.

Pruebas nuevas y diagnósticos usan datos sintéticos y credenciales de prueba, con construcción local y paper explícito. No hubo órdenes, llamadas a cuentas ni conexión operativa deliberada al exchange. Las esperas de compilación se resolvieron sin terminar procesos ajenos.

## 13. Hoja de ruta sistémica de cierre

1. Unificar ExecutionPlan antes de cualquier efecto de cuenta: cantidad, precio/referencia, leverage, filtros, presupuesto monetario, identidad y versión. Cerrar FMT-175/219 sin perder salidas defensivas.
2. Sustituir rescates secuenciales por factibilidad explícita; demostrar que no aumenta pérdida autorizada y distinguir rechazo correcto de ausencia de búsqueda completa. Mantener controles hasta verificar el reemplazo.
3. Integrar filtros por tipo de orden y metadatos validados; demostrar serialización decimal, límites y paridad de la proyección en replay, paper, demo y producción simulada.
4. Separar política genómica, restricciones del venue y políticas operativas. Explicar/validar 0,035, 0,1, 1,0005 y límites micro, sin convertir una interpolación suave en evidencia de optimización científica.
5. Instrumentar vetos con denominador y evidencia versionada; evaluar señales rechazadas en shadow causal sin desactivar defensas productivas.
6. Completar soporte temporal y dependencia multiactivo: resolución observada, extrapolación, presupuesto de aproximación y riesgo conjunto, no sustitución de palabras.
7. Reparar la generación/factibilidad de genomas de FMT-216 y validar promoción reproducible; no relajar un veto justificado para ocultar la falta de soluciones admisibles.
8. Completar las 147 lecturas Rust pendientes y la revisión no Rust. Las mejoras aquí realizadas no autorizan el rótulo «todo auditado».

## 14. Preservación, Git y trazabilidad

Tres fuentes productivas intervenidas: risk-engine/src/lib.rs, execution-engine/src/executor.rs y execution-engine/src/binance_api.rs. Tres suites nuevas y un diagnóstico anterior actualizado. El árbol ya contenía numerosos cambios de otras rondas/sesiones; el diff contra HEAD no representa solamente XXVI. No se revirtió ni formateó masivamente executor.rs.

El estado local comprobado fue main, HEAD 59a76de4, con cambios sin consolidar. No hubo commit, push, merge ni fetch; no se infiere el estado de ramas remotas ni que otro colaborador haya corregido o no estas rutas. La integración de todas las ramas no se ejecuta implícitamente sobre un árbol compartido sucio.

Los 41 modelos coinciden por SHA-256 con el snapshot XXV. Se preservan los informes mediante adendas y el artefacto conserva hashes anteriores y finales. No se modificaron deliberadamente genomas activos, entrenamientos, promociones ni procesos productivos. Firecrawl se usó para contrastar los contratos oficiales de leverage y filtros; influyó en la validación del entero y en separar restricciones por tipo de orden. No demuestra rendimiento financiero.

La certificación integral permanece abierta.

## 15. Comprobación documental final

Se verificaron las 31 referencias de evidencia del JSON, 48 hashes finales (siete fuentes/tests y 41 modelos) y 27 enlaces locales del informe y sus adendas, sin archivos ausentes, líneas fuera de rango ni hashes divergentes. Los tres prefijos históricos coinciden tras normalizar CRLF→LF; se añadieron 2.701, 5.058 y 1.448 caracteres de cadena .NET al atlas, maestro y XXV respectivamente. Son longitudes de texto, no bytes. leverage_matrix.rs y genome.rs conservan además sus hashes previos a esta ronda.

Rustfmt --check pasó en las seis fuentes/tests formateadas. executor.rs recibió parches localizados, no un formateo global; git diff --check pasó para las fuentes y documentos versionados intervenidos. La repetición final de cinco contratos de leverage y cinco pruebas de vetos pasó, confirmando en el fixture L=4 y margen=1,2756374999999998. Los avisos de TG_GENOME_ENV ausente corresponden a las arenas de prueba; no se solucionaron cambiando el entorno productivo ni promoviendo un genoma.

## 16. Continuación XXVII — alcance del consumidor real

El [informe XXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXVII_2026-09-25.md>) verifica que el host usaba APIs raw/maker/iceberg, no execute_order/build_payload. Amplía FMT-218 a raw_qty y conecta un despachador común al host. FMT-220 contiene capacidad iceberg no respaldada, argumentos intercambiados e ID fijo; FMT-221 exige configuración/confirmación de leverage incluso a 1× y bloquea envío si falla. No se implementa iceberg mediante órdenes hijas ni se sustituye por MARKET.

FMT-113 sigue abierto: el host modifica financiación sin proyectar cantidad sobre pérdida monetaria. Los tres diagnósticos FMT-175/219 siguen reproduciendo deuda. Nuevo FMT-222: consulta de reconciliación fallida puede terminar marcando exchange_confirmed=true; queda P1 abierto, sin cambiar a ciegas la conservación/rollback.

Diez contratos nuevos, dos rojo→verde, 126 funcionales y 11 diagnósticos abiertos con pase; check de cuatro binarios pasa. Cobertura 143/289 Rust, 146 pendientes. [Artefacto XXVII](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XXVII_2026-09-25.json>). Se conserva todo el histórico XXVI. No se enviaron órdenes, accedió a cuentas ni publicaron cambios Git.
