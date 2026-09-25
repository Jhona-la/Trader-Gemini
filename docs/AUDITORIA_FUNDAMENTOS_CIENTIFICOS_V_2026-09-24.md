# Auditoría de fundamentos científicos V — riesgo realizable, ejecución y memoria temporal

Fecha: 24 de septiembre de 2026. Continuación aditiva de las rondas I–IV. Referencia Git observada: **59a76de4**, con un workspace modificado y trabajo concurrente. Este documento describe el contenido observado, no certifica el commit ni atribuye cambios ajenos a esta auditoría.

**Dictamen:** hay avances reales hacia una representación temporal continua, pero sus garantías se pierden en varias composiciones de funciones. Destaca una desconexión entre el presupuesto de Kelly y la cantidad efectivamente enviada: ajustar apalancamiento de margen no equivale a ajustar riesgo monetario. También persisten incompatibilidades en la geometría TP/SL, el aprendizaje del payoff, la reconstrucción de cierres y capacidades auxiliares de simulación/persistencia.

Se añaden **19 fichas FMT-095–113**, todas abiertas. Las fichas amplían la serie científica; no se afirma que cada causa sea desconocida en todo el histórico ni se suman sin deduplicación a la matriz histórica de 305 puntos. Se conservan hallazgos, explicaciones y correcciones anteriores. Revisar y documentar no es reparar.

Se leyeron completos **8 Rust, 3.615 líneas**, además de tramos dirigidos de productores y consumidores. El acumulado de los manifiestos FMT I–V es **90 Rust distintos** frente a **289 Rust, 1.119 archivos versionados y 24 Cargo.toml** del corte. Quedan 199 Rust fuera de esos manifiestos de lectura completa, además de la cobertura no Rust pendiente. Búsquedas globales no cuentan como lectura integral. No se ha revisado todo el proyecto archivo por archivo ni se ofrece certificación de producción.

## 1. Paradigma de grafo vivo: de raíz a resultado terminal

El sistema debe modelarse como un grafo de contratos verificables, no sólo como un grafo de imports o nombres científicos. Cada arista transporta valores, unidades, tiempo de disponibilidad, identidad y versión. El “nodo terminal” no es una intención aprobada: es un resultado ejecutado, reconciliado y atribuible.

~~~text
RAÍZ: evento observado + reloj + símbolo + identidad + versión de datos
  └─ representación multivariante Z(t, ln τ, activo, canal)
       ├─ estado estadístico y su incertidumbre
       └─ genoma → fenotipo y geometría de barreras
            └─ DECISIÓN: cantidad, dirección, barreras, coste, plazo
                 └─ conjunto factible de riesgo, margen y filtros
                      └─ intención identificada → ejecución → fills
                           └─ TERMINAL: posición/caja/PnL reconciliados
                                ├─ evidencia durable y recuperable
                                └─ crédito causal → adaptación versionada
~~~

La misma ecuación en backtest y vivo puede reproducir el mismo error. La propiedad exigible no es únicamente igualdad entre entornos: también deben conservarse presupuesto de pérdida, identidad de operación, balance de cantidades, causalidad y semántica de las estimaciones.

### 1.1 Separar tres cantidades que hoy se confunden

Sea C el capital de referencia, Q la cantidad, P el precio, N=|Q|P el notional, d la distancia relativa al stop y L el apalancamiento de margen. Bajo una aproximación lineal sin gaps:

- Pérdida de precio al stop: R=N·d.
- Margen inicial simplificado: M=N/L.
- Presupuesto de riesgo: B=C·f, con f una fracción autorizada.
- Factibilidad: R más costes y reserva de ejecución no debe superar B; el margen y los filtros se comprueban adicionalmente.

Con N y d fijos, cambiar L modifica M, pero no R. No se puede llamar “reducción de riesgo de posición” a una operación que sólo cambia el denominador de margen. Un stop tampoco garantiza el precio efectivo de salida: la desigualdad anterior es nominal y necesita tratamiento de gaps, slippage y liquidez.

## 2. Resumen de resolución y matriz de esta ampliación

P1 indica prioridad alta por seguridad, corrección económica, memoria o evidencia; P2 indica contrato/validación que requiere corrección antes de confiar en la capacidad. La prioridad de una API auxiliar no demuestra exposición productiva actual.

| ID | Prioridad | Problema | Alcance comprobado |
|---|---|---|---|
| FMT-095 | P1 | Imponer RR≥2 estrecha el supuesto piso difusivo | Función consumida por riesgo |
| FMT-096 | P1 | Objetivo genómico cambia p de diseño y puede reducir TP | Función consumida por riesgo |
| FMT-097 | P1 | Cap micro de 55 bps reintroduce discontinuidad y no prueba riesgo USD | Camino de riesgo conectado |
| FMT-098 | P1 | Bootstrap micro reabre riesgo con evidencia negativa madura | Envolvente usada por host/replay |
| FMT-099 | P1 | Excepción permite riesgo mínimo superior al presupuesto | Envolvente usada por host/replay |
| FMT-100 | P2 | Payoff recibe dos suavizados consecutivos | Host y dos backtests |
| FMT-101 | P2 | Regularización monetaria rompe invariancia del payoff | Envolvente conectada |
| FMT-102 | P2 | Compounder convierte PF≤1 en edge positivo | API auxiliar sin consumidor vivo localizado |
| FMT-103 | P1 | Ruta de pánico pierde identidad estable de intención | Router auxiliar |
| FMT-104 | P2 | Redondeo IOC puede superar la tolerancia declarada | Router auxiliar |
| FMT-105 | P1 | Simulador fabrica salida favorable fija | Simulador auxiliar |
| FMT-106 | P1 | Simulador no conserva ledger entre rutas y cierres parciales | Simulador auxiliar |
| FMT-107 | P1 | Anillo mmap permite escritores simultáneos sobre el mismo slot | Almacén auxiliar |
| FMT-108 | P1 | Reinicio, retención y durabilidad temporal sin protocolo | Almacén auxiliar |
| FMT-109 | P1 | Diario asíncrono confunde enqueue con persistencia | Contabilidad conectada |
| FMT-110 | P1 | Contexto de entrada recuperado sin identidad ni límite as-of | Recuperación conectada |
| FMT-111 | P1 | Descartes contados no invalidan el aprendizaje consumidor | Contabilidad y host |
| FMT-112 | P2 | Interpolación no conserva identidades del estado continuo | API espectral; contradicción de contrato |
| FMT-113 | P1 | Kelly limita leverage, no cantidad; margen puede elevar el límite | Host y replay conectados |

**Reconocimientos:** REJ_TP_SL_FLOOR sí está conectado en el corte actual; no se reabre CES-018 como si siguiera desconectado. La geometría esperada se reutiliza en el gate y la orden de riesgo; los defectos 095–097 afectan a esa geometría compartida. Los pesos epigenéticos sí llegan a consumidores espectrales revisados y existen tests de consistencia en nodos. Ninguna mejora local demuestra todavía el contrato global.

## 3. Hallazgos detallados

### FMT-095 · P1 · El piso difusivo deja de ser piso al imponer una razón de barreras

**Evidencia:** [tp_sl.rs:131](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/tp_sl.rs:131>) calcula σ=ATR·(τ/60.000 ms)^H y SL_difusivo=kσ. Tras aplicar el piso de fricción, [tp_sl.rs:164](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/tp_sl.rs:164>) reemplaza SL por TP/2 cuando la razón inicial es inferior a dos. El test existente acepta explícitamente ese estrechamiento.

**Contraejemplo:** ATR=0,005, τ=240.000 ms, H=0,5, k=1 y coste f=0,001 producen σ=0,01. Con p=0,40, RR requerido=1,75 y TP=0,0175. La segunda regla devuelve SL=0,00875: queda 12,5 % por debajo del supuesto piso difusivo. El cálculo puede ser numéricamente finito y satisfacer su EV condicional; eso no restituye la garantía de distancia que se había anunciado.

**Impacto:** una barrera más cercana cambia la distribución de primera llegada y la frecuencia de salida. No es legítimo conservar automáticamente la misma probabilidad de ganar tras modificar las barreras. El problema no consiste en que RR≥2 sea siempre incorrecto: consiste en declarar simultáneamente dos invariantes que la composición no preserva.

**Cierre exigible:** decidir si σk es piso, propuesta o escala de referencia; formular conjuntamente objetivo, stop y probabilidad condicionada. Si es un piso duro, no puede estrecharse para satisfacer otro objetivo. Añadir una regresión que verifique el conjunto de invariantes en todo el dominio y documente cuándo la solución factible no existe. No basta con comprobar RR después de modificar SL.

### FMT-096 · P1 · “Objetivo más ambicioso” puede rebajar TP y cambiar la hipótesis de EV

**Evidencia:** [tp_sl.rs:190](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/tp_sl.rs:190>) parte del resultado calculado con WORST_TOLERATED_WR=0,40, pero recalcula rr_design con DESIGN_WIN_RATE=0,55. Si target_rr supera max(rr_design,2), sobrescribe tanto TP como rr_required, aunque target_rr sea menor que el RR original.

**Reproducción algebraica:** SL=0,002 y f=0,001 dan RR_min(0,40)=2,75 y TP=0,0055. La variante admite target_rr=2 porque RR_min(0,55)=1,72727. Devuelve TP=0,004. Entonces EV(0,40)=0,40·0,004−0,60·0,002−0,001=−0,0006 por unidad de notional. Con p=0,55, la misma fórmula da +0,0003. No hay contradicción aritmética: hay un cambio no explícito de la probabilidad contractual.

**Alcance:** la función está conectada a risk-engine; otros filtros pueden rechazar una intención. No se afirma que este contraejemplo haya generado una orden real. Sí se refuta que la variante sólo pueda aumentar el objetivo y conserve automáticamente la garantía de la función base.

**Cierre:** parametrizar y versionar la probabilidad usada, no sustituirla mediante un nombre de constante diferente. Probar monotonía respecto del target cuando ése sea el contrato; comprobar EV bajo la misma hipótesis antes/después. Una condición algebraica de EV no demuestra que el p supuesto esté calibrado para el activo, las barreras y el horizonte. La teoría de primeras llegadas T17 ya propuesta resulta pertinente, sin que la denominación “difusiva” pruebe sus supuestos.

### FMT-097 · P1 · El cap micro vuelve discreta la geometría y no demuestra una pérdida máxima

**Evidencia:** [lib.rs:621](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/lib.rs:621>) impone SL=0,0055 cuando micro_w_alloc>0,5 y el stop calculado excede ese valor; TP se reconstruye usando RR y un mínimo 2,25. El comentario justifica el cambio mediante una pérdida máxima aproximada de 0,028 USD asociada al mínimo notional.

**Mecanismo:** el peso micro puede ser suave, pero convertirlo en el predicado w>0,5 reintroduce un escalón. Con el mismo stop propuesto de 1 %, a un lado de la frontera se devuelve 0,55 % y al otro 1 %. En la función de régimen, w=0,5 corresponde a C/M=√30; con M=5, la frontera es aproximadamente 27,3861. La variable C usada por el consumidor es la asignación pertinente, no necesariamente todo el equity.

**Error económico:** M_min·d es una pérdida nominal mínima para esa barrera, no una cota superior para una orden con N≥M_min. Si N=50, 55 bps suponen 0,275 USD antes de costes, no 0,028. Además, estrechar SL altera la probabilidad de salida y puede destruir la interpretación de piso difusivo de FMT-095.

**Cierre:** calcular riesgo con cantidad final, precio y costes, no con el mínimo del exchange. Documentar el compromiso de barreras y evaluar discontinuidades de la función compuesta, no sólo del helper smoothstep. Eliminar el escalón no autoriza a desactivar límites: primero se necesita un conjunto factible y una abstención explícita cuando no pueda satisfacerse.

### FMT-098 · P1 · El bootstrap micro no distingue incertidumbre inicial de evidencia adversa

**Evidencia:** [kelly_envelope.rs:204](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:204>) sustituye toda fracción f≤0 por 0,015·micro_w. No comprueba el número de operaciones ni la razón que produjo cero. El host usa max_leverage en [god_engine.rs:3805](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3805>) y el replay en [booktick_replay.rs:657](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/booktick_replay.rs:657>).

**Contraejemplo:** después de 500 pérdidas válidas, risk_fraction continúa devolviendo cero por ausencia de edge. Para C=13 y mínimo=5, micro_w=1; max_leverage vuelve a f=0,015. Como n≥30, ya no es el arranque explícito del host, pero la envolvente puede devolver operable=true. La evidencia negativa madura se trata como si fueran pocas muestras.

**Impacto:** la adaptación pierde una propiedad esencial: poder disminuir exposición hasta cero ante evidencia contraria. No se deduce que todo el motor vaya a operar —quedan filtros—, pero esta capa deja de representar la decisión estadística que anuncia.

**Cierre:** separar estado “sin información” de “información desfavorable”, asignar presupuesto acumulado y finito a exploración y conservar la causa del veto. Definir de dónde provienen observaciones exploratorias y cómo se evalúan sin forzar dinero real. Un mecanismo de aprendizaje no necesita garantizar actividad perpetua; también debe aprender a abstenerse. Una rama condicionada sólo a capital no representa epistemología bayesiana.

### FMT-099 · P1 · La excepción de notional mínimo permite lo que la desigualdad acababa de prohibir

**Evidencia:** [kelly_envelope.rs:217](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:217>) compara B=Cf con M_min·d. Cuando B es menor, una excepción micro puede retornar leverage≥1 y operable=true. La comprobación M_min≤5C sólo limita una relación de financiación; no demuestra que la pérdida quepa en B.

**Contraejemplo:** C=13, f=0,015, d=0,10 y M_min=5 dan B=0,195 USD y riesgo mínimo nominal de 0,50 USD. La excepción retorna (1,true), aun cuando el mínimo supera 2,564 veces el presupuesto. Este ejemplo prueba el contrato de la API; el cap de stop de otra capa puede impedir ese d en algunas entradas reales. Un consumidor no debe depender de una restricción externa no expresada para que “operable” signifique lo prometido.

**Impacto:** “safe_micro_leverage” no es prueba de seguridad. Si no existe cantidad positiva que satisfaga simultáneamente el mínimo y el riesgo, subir o bajar leverage de margen no crea esa cantidad. El coste de entrada/salida empeora la desigualdad.

**Cierre:** representar explícitamente conjunto vacío. La exploración, si se autoriza, requiere otro presupuesto declarado y no debe falsificar el dictamen de la envolvente. Regresión: para toda respuesta operable, verificar una cantidad realizable que respete filtros y pérdida presupuestada; comprobarla de nuevo después de redondear y antes del envío.

### FMT-100 · P2 · El payoff se suaviza dos veces y su velocidad de adaptación no es la declarada

**Evidencia:** [god_engine.rs:3324](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3324>) y [god_engine.rs:3666](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3666>) actualizan avg_win_abs/avg_loss_abs mediante EMA 0,95/0,05 y pasan esos promedios a record_trade. [kelly_envelope.rs:139](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:139>) vuelve a aplicar la misma EMA. La misma composición aparece en [lib.rs:393](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/lib.rs:393>) y [booktick_replay.rs:511](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/booktick_replay.rs:511>).

**Demostración:** si ambos estados valen 1 y la siguiente ganancia vale 9, una EMA produce 1,4, pero la segunda devuelve 1,02. Se implementa un filtro de segundo orden H(z)=[0,05/(1−0,95z⁻¹)]², no una EMA única. Su retraso de baja frecuencia es aproximadamente 38 actualizaciones de la clase correspondiente, frente a 19 de una etapa. Ganancias y pérdidas sólo avanzan sus respectivos estados: ese retraso no equivale a segundos ni a número total de trades.

**Consecuencia:** la magnitud del payoff reacciona mucho más lentamente que la evidencia binaria. No se señala aquí divergencia host/replay, porque ambos repiten el doble suavizado; se señala un contrato de adaptación erróneo compartido. Reiniciar/restaurar estados también debe especificar cuáles de las dos etapas se recuperan.

**Cierre:** definir si la API recibe resultados crudos o estadísticas suficientes. Si se desean dos escalas de memoria, hacerlas explícitas y justificarlas. Comparar respuesta a escalón, a rachas y a cambio de tamaño; probar igualdad de estados aprendidos al reproducir la misma secuencia completa, no sólo igualdad del número de operaciones.

### FMT-101 · P2 · El payoff depende de la escala monetaria por un epsilon absoluto

**Evidencia:** [kelly_envelope.rs:155](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/kelly_envelope.rs:155>) calcula b=(avg_win+0,001)/(avg_loss+0,001). El valor 0,001 tiene unidades de PnL, aunque el resultado b es adimensional. Los floors de 10⁻⁶ también son monetarios.

**Contraejemplo:** ganancias medias 0,002 y pérdidas medias 0,001 dan b=1,5, frente a la razón económica 2. Multiplicar ambas por diez produce b=1,90909. La estrategia porcentualmente idéntica recibe otro Kelly simplemente por cambiar tamaño o denominación. El efecto es especialmente relevante cuando se pretende transferir entre cuentas grandes de backtest y cuentas pequeñas.

**Matiz estadístico:** regularizar una razón mal identificada cuando faltan pérdidas puede ser necesario. El defecto no es regularizar, sino hacerlo sin modelo ni unidades explícitas y presentar el resultado como payoff comparable. La probabilidad acumula observaciones históricas mientras los tamaños usan ventanas exponenciales; tampoco son automáticamente estimaciones del mismo régimen. El LCB implementado es una aproximación media−z·desviación de una beta, no un cuantil beta exacto para cualquier muestra.

**Cierre:** definir recompensa normalizada por exposición/riesgo de entrada, manejar ausencia de una clase mediante prior explícito y exigir invariancia a cambios de unidad. Separar garantía estadística, aproximación computacional y política prudencial. Validar sensibilidad a tamaños heterogéneos, costes mínimos y cambios de régimen antes de interpretar cambios en b como evolución del edge.

### FMT-102 · P2 · El compounder auxiliar transforma estrategias perdedoras en candidatas con exposición positiva

**Evidencia:** [capital_compounder.rs:85](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/risk-engine/src/capital_compounder.rs:85>) reemplaza PF por max(PF,1,01). Calcula Kelly con ese valor, aplica un mínimo 0,005 y, tras penalidades, vuelve a imponer kelly_clamp_min si la fracción sigue siendo positiva.

**Contraejemplo:** con win_rate=0,5 y PF=0,8, la fórmula sin sustitución da 0,5·(1−1/0,8)=−0,125. La implementación usa 1,01, obtiene una fracción positiva y puede elevarla al mínimo configurado. Penalidades no nulas por drawdown, correlación o racha pueden quedar neutralizadas por ese último piso. Una penalidad exactamente cero sí permanece cero: no se afirma que esa rama también se resucite.

**Alcance:** se localizaron definición y tests, no invocación viva de CapitalCompounderEngine. Es una capacidad auxiliar no apta para conectarse sin corregir su contrato, no una explicación acreditada del sizing productivo actual. Extiende el patrón FMT-014 a una implementación diferente.

**Cierre:** no alterar una estadística adversa para superar su propio criterio. Separar exploración de sizing por edge y probar monotonicidad: aumentar drawdown o reducir PF no debe aumentar exposición ni quedar oculto sin política explícita. Los nombres “compounder”, “continuo” o “exponencial” no convierten un conjunto de clamps en una ley de crecimiento óptimo.

### FMT-103 · P1 · La ruta de pánico descarta el identificador estable de la intención

**Evidencia:** [router.rs:56](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/router.rs:56>) usa execute_raw_qty cuando latencia>500 ms o spread>0,005, ignorando client_order_id recibido. La ruta normal sí llama la variante con ID. [executor.rs:2128](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/executor.rs:2128>) genera un UUID nuevo cuando el ID está vacío.

**Mecanismo:** dos llamadas del router para la misma intención pueden generar dos identidades distintas justo bajo red degradada. Un ID aleatorio por intento permite rastrear ese intento, pero no deduplicar la intención original. No se afirma que todo reintento interno del ejecutor duplique órdenes: el problema probado es la pérdida de identidad al repetir la llamada de alto nivel.

**Alcance:** QuantumOrderRouter se reexporta, pero la búsqueda en src y crates sólo localizó construcción/uso en sus tests. No se equipara este auxiliar al camino directo de entrada del host. La severidad describe el riesgo al conectarlo.

**Cierre:** preservar un identificador de intención estable en todas las ramas y separar intent_id, attempt_id y fill_id. Simular resultado de transporte ambiguo y comprobar que la reconciliación decide antes de reemitir. Además, MARKET bajo spread grande no ofrece por sí mismo una garantía de protección: urgencia de salida y apertura de exposición son objetivos distintos que el contrato debe distinguir.

### FMT-104 · P2 · El redondeo IOC no conserva un límite unilateral de precio

**Evidencia:** [router.rs:94](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/router.rs:94>) deriva una tolerancia, forma raw_ioc_price y redondea al tick más cercano. El log presenta el resultado como máximo slippage.

**Contraejemplo:** compra a referencia 100, tolerancia 0,0026 y tick 0,1: límite teórico 100,26, redondeado 100,30. La tolerancia pasa de 26 a 30 bps. Para una venta, redondear hacia abajo puede violar simétricamente el mínimo aceptable. No es un error de representación de céntimos: es la dirección incorrecta para conservar una desigualdad.

**Alcance:** defecto del router auxiliar. La ausencia de fill o filtros posteriores puede evitar una ejecución, pero no repara el límite que construye la API. Tampoco IOC garantiza que haya ejecución, por mucho que se use un precio más agresivo.

**Cierre:** proyectar sobre ticks factibles con redondeo orientado al lado de la restricción, y revalidar el límite después. Si no existe un tick que cumpla agresividad y tolerancia, devolver esa incompatibilidad. Probar límites a ambos lados de medio tick, símbolos con ticks grandes relativos al precio, datos inválidos y cambios de filtros. Conservar la semántica del máximo es más importante que lograr un precio cercano.

### FMT-105 · P1 · El simulador fabrica una salida favorable de veinte puntos básicos

**Evidencia:** [simulator.rs:330](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/simulator.rs:330>) calcula exit_price=entry_price·1,002 para cierre long y entry_price·0,998 para cierre short. No consulta la trayectoria de mercado ni el último mid para valorar esa salida.

**Mecanismo:** con dirección de cierre coherente y cantidad positiva, la contribución bruta es favorable por construcción. Una caída del mercado durante un long no cambia ese signo. Las comisiones pueden reducir o superar la ganancia, pero no devuelven al modelo la distribución de pérdidas por precio. La latencia simulada sólo retrasa la respuesta; no hace evolucionar el precio de salida mediante un tape causal.

**Alcance crucial:** la búsqueda actual sólo encontró SimulatedExecutor en su definición y tests; no acredita que run_backtest_native o demo real lo utilicen. Por tanto, este hallazgo invalida su aptitud como simulador económico, pero **no prueba que explique el buen resultado del backtest principal**. La afirmación histórica genérica “el simulador explica la divergencia” necesita identificar el backend concreto.

**Cierre:** salida por evento de mercado as-of, precio ejecutable, lado, profundidad y costes; pruebas de long que pierde, short que pierde, gaps y trayectorias que no tocan barreras. Evaluar la política sobre el mismo tape y semillas registradas, no sobre precios de cierre construidos a partir de la recompensa deseada.

### FMT-106 · P1 · El simulador no conserva cantidades, precios medios ni obligaciones entre métodos

**Evidencia:** execute_order suma cantidad pero conserva el precio inicial y sobrescribe is_long ([simulator.rs:135](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/simulator.rs:135>)). execute_raw_qty crea entradas a precio 1 sin la contabilidad de fees de execute_order. Varias rutas limit/IOC/maker/iceberg retornan Ok sin actualizar el mapa de posiciones; OCO/trailing no mantienen órdenes condicionales. reduce_only elimina toda la posición aunque quantity sea parcial.

**Contraejemplos:** una compra de una unidad a 100 y otra a 200 debe producir precio medio 150, no conservar 100; vender contra un long no debe sumar cantidad y cambiar sólo la etiqueta de dirección. Cerrar 0,2 de una posición de 1 no debe borrar el 0,8 restante. Con quantity superior a la posición, el PnL usa la cantidad solicitada sin representar un residual coherente.

**Consecuencia:** cambiar el tipo de ejecución cambia las leyes del ledger. Un Ok puede significar aceptación, fill o sólo log según el método; el consumidor no recibe suficiente semántica para reconciliarlo. El kill-switch sólo imprime, por lo que tampoco simula su efecto de bloqueo.

**Cierre:** adoptar una máquina de estados explícita para órdenes, fills y posiciones; separar aceptación de ejecución. Exigir conservación de cantidad firmada, coste medio, caja y fees; tests con parciales, inversión de lado, cancelación y fallos. No conectar esta clase como sustituto económico de producción hasta que esos invariantes sean comunes a todas las rutas.

### FMT-107 · P1 · El wrap-around mmap invalida la exclusión entre escritores

**Evidencia:** [temporal_store.rs:63](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/temporal_store.rs:63>) reserva un contador atómico, pero obtiene el slot aplicando módulo de la capacidad. La API acepta &self y declara Sync manualmente. La escritura del bloque usa cuatro stores de 128 bits o una copia no atómica, sin reserva por generación ni protocolo de publicación.

**Interleaving adversarial:** con capacidad útil 64 bytes, A reserva offset 0 y se suspende. B reserva 64, cuyo módulo vuelve a cero, y escribe. A reanuda mientras B todavía escribe o después de su publicación. Ambos comparten el mismo payload; el contador atómico no impone exclusión sobre esos bytes. Con capacidad mayor ocurre cuando el contador da una vuelta antes de concluir un escritor lento.

**Impacto:** se admite una carrera de datos no atómicos y un bloque mezclado; además de integridad, la declaración de seguridad de memoria exige revisión. _mm_sfence ordena stores del hilo, pero no serializa escritores ni convierte 64 bytes en una transacción.

**Cierre:** definir propiedad del slot, generación, reserva y publicación; impedir reutilización mientras exista escritura/lectura incompatible. Modelar el interleaving mínimo y utilizar pruebas de concurrencia adecuadas antes de habilitarlo. No se provocó una carrera real ni se escribió a un almacén productivo. El anillo corrige el agotamiento D-517, pero introduce obligaciones que el contador por sí solo no satisface.

### FMT-108 · P1 · El almacén temporal no ofrece recuperación ni durabilidad por devolver Ok

**Evidencia:** [temporal_store.rs:24](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/storage-engine/src/temporal_store.rs:24>) abre con truncate(false), pero ejecuta set_len(prealloc_bytes) y siempre inicializa write_cursor=0. No persiste cursor, generación, formato, checksum o marca de commit. El método de escritura no realiza flush del mapping ni sincronización del archivo.

**Consecuencias distintas:** reabrir comienza a sobrescribir desde el principio sin reconstruir el orden; pedir una capacidad menor puede truncar el archivo mediante set_len aunque el flag de apertura diga lo contrario; el wrap elimina historia sin exponer política de retención. Sustituir NaN/Inf por cero también destruye la distinción entre dato válido cero y dato corrupto.

**Límite físico:** una barrera de memoria de CPU no acredita persistencia en SSD, recuperación tras caída ni latencia “picosecond”. La ejecución de instrucciones SIMD y la durabilidad son contratos diferentes. No se midió latencia de almacenamiento ni se ensayó pérdida de alimentación.

**Alcance y cierre:** no se localizó consumidor vivo de TemporalObjectStore. Antes de conectarlo, especificar si es caché descartable o registro durable. En el segundo caso, hacen falta formato versionado, secuencias, commits verificables, retención y recuperación; en el primero, no debe alimentar aprendizaje como evidencia histórica infalible. Probar reinicio, último bloque incompleto, tamaño distinto y detección de corrupción en archivos temporales aislados.

### FMT-109 · P1 · El diario asíncrono no garantiza la evidencia que sus comentarios prometen

**Evidencia:** [trade_accounting.rs:153](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/trade_accounting.rs:153>) crea un canal mpsc sin cota y un hilo de I/O. [trade_accounting.rs:189](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/trade_accounting.rs:189>) ignora errores de write_all; journal_append sólo detecta que el canal se haya desconectado. record_bracket_close continúa al aprendizaje después de encolar, no tras confirmar persistencia.

**Escenarios:** si el disco rechaza una escritura y el hilo sigue vivo, send puede haber devuelto éxito y la línea se pierde sin confirmación al productor. Una caída después de enqueue y antes del write pierde evidencia mientras el posterior ya pudo incorporar el cierre. Si la entrada supera sostenidamente la capacidad de disco, la cola sin cota crece: memoria y latencia de recuperación no están limitadas. Un write parcial seguido de error puede dejar una línea JSON inválida.

**Impacto:** “diario primero” describe orden de llamadas, no write-ahead durable. La copia adicional de line para send amplifica coste, pero el defecto principal es semántico, no unos microsegundos sin benchmark.

**Cierre:** niveles de confirmación separados —aceptado, persistido, consumido— con secuencia y política de durabilidad. Definir backpressure, spool, alarma y recuperación; reconciliar el posterior con offsets confirmados. Probar fallos de disco y caída del proceso en un entorno aislado. No se ejecutaron tests de este módulo que escriben en data/trade_fills.jsonl.

### FMT-110 · P1 · La reconstrucción de entrada puede usar otra operación o un estado posterior al cierre

**Evidencia:** [trade_accounting.rs:306](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/trade_accounting.rs:306>) selecciona el último precio por símbolo y lado con mayor timestamp. No recibe ID de posición, ID de entrada ni timestamp límite del cierre. [user_data_stream.rs:538](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/user_data_stream.rs:538>) lo utiliza cuando falta el precio local; el PnL calculado alimenta al host.

**Contraejemplo:** operación A entra a 100, cierra a 110 en t1, pero el fill llega tarde. Antes de procesarlo se registra una nueva entrada B a 120 en t2>t1. Si se usa el fallback sin contexto local, se elige 120 y se reconstruye una pérdida de 10 por unidad en lugar de la ganancia de 10 de A. Incluso limitar a t≤t1 no resuelve posiciones simultáneas del mismo símbolo/lado sin identidad.

**Coste de recuperación:** la caché evita reparsear sin cambios, pero cualquier append invalida la firma y dispara lectura/parseo de todo el diario bajo un mutex; después se escanea el vector completo. Es O(longitud del diario) en esos accesos, no consulta indexada constante. No se midieron p99 ni se atribuye una latencia concreta.

**Cierre:** vincular fills a posición/intención y recuperar contexto as-of de esa identidad, incluidos cantidad y fees. Ante contexto ambiguo, conservar PnL desconocido, no inventar uno desde el registro más reciente. Mantener índice incremental y probar reordenación, reconexión, reapertura inmediata, parciales y múltiples posiciones. La integridad del crédito evolutivo depende de esta arista.

### FMT-111 · P1 · Contar cierres descartados no corrige la muestra que aprende Kelly

**Evidencia:** [trade_accounting.rs:239](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/execution-engine/src/trade_accounting.rs:239>) descarta cierres cuando PENDING alcanza 1.024, incrementa DESCARTADOS y avisa. La búsqueda de cierres_descartados localiza API y test, no un consumidor operativo que invalide el posterior, reponga registros o marque la muestra incompleta. El host sigue drenando y aprendiendo de lo recibido.

**Mecanismo:** la probabilidad de inclusión depende de congestión y velocidad del consumidor. En mercados turbulentos, esos factores pueden relacionarse con PnL; por tanto, no puede suponerse descarte aleatorio. El posterior incorpora sólo el subconjunto disponible con un contador n que no representa toda la actividad.

**Distinción:** D-710 mejoró observabilidad: ya no es un descarte completamente silencioso. La mejora no acredita que el aprendizaje se haya vuelto íntegro. El diario podría servir de reparación, pero FMT-109 impide asumir persistencia garantizada y no se localizó aquí un replay de faltantes.

**Cierre:** secuencias continuas entre ledger y aprendizaje, detección de huecos y recuperación deduplicada. Un estado de evidencia incompleta debe ser explícito y determinar qué decisiones siguen autorizadas; eso no exige liquidar posiciones por una alarma de telemetría. Validar que introducir congestión cambia latencia, no el conjunto final de observaciones aprendidas, o documentar formalmente el esquema de muestreo.

### FMT-112 · P2 · La interpolación conserva continuidad, pero no las identidades del estado representado

**Evidencia:** [temporal_spectrum.rs:542](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:542>) interpola momentum_z y signal por separado. En los nodos signal=tanh(clamp(z)); entre nodos, en general I[tanh(z)]≠tanh(I[z]). [temporal_spectrum.rs:500](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:500>) declara E(τ)=w(τ)|s(τ)|, pero interpola las masas ya calculadas en la malla.

**Ejemplo:** z0=0 y z1=2, en el punto medio logarítmico, producen signal interpolada=0,48201379 y tanh(z interpolado)=0,76159416. Con señales opuestas ±a y pesos iguales positivos, la energía interpolada es positiva aunque signal_at sea cero; evaluar w(state_at)·|signal_at| daría cero.

**Interpretación profesional:** no toda interpolación componente a componente es un bug. Puede ser una convención válida si signal, z y masa son campos independientes. El fallo confirmado es presentar simultáneamente identidades funcionales que esa convención no conserva. No se demuestra aquí una orden incorrecta causada por consultar precisamente el punto medio.

**Cierre:** elegir y documentar variables primitivas y derivadas. O bien derivar señal/energía desde un estado continuo común, o declarar explícitamente interpolación de señales y de masas como observables diferentes. La corrección CES que iguala energía en nodos permanece válida; esta ficha exige cierre entre nodos, signos opuestos y extrapolación. Refinamiento de malla y error de decisiones deben medirse, no sustituirse por el adjetivo “continuo”.

### FMT-113 · P1 · La envolvente de riesgo no dimensiona la cantidad enviada y puede perder su techo de leverage

**Evidencia conectada:** [god_engine.rs:3842](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3842>) calcula kelly_frac y un techo derivado de env_lev. [god_engine.rs:3887](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3887>) construye exec_leverage. Sin embargo, [god_engine.rs:3990](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:3990>) toma final_qty de new_order; [god_engine.rs:4023](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4023>) calcula N=|final_qty|·precio, y [god_engine.rs:4136](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4136>) envía esa misma cantidad. No se observa en ese tramo una proyección de Q sobre el presupuesto Kelly/vol_brake.

**Segunda ruptura de la cadena:** [god_engine.rs:4074](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/src/bin/god_engine.rs:4074>) puede elevar effective_leverage por encima de exec_leverage para satisfacer margen, sin volver a aplicar el techo de la envolvente. El replay repite la lógica en [booktick_replay.rs:708](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/backtest-engine/src/booktick_replay.rs:708>) y retorna true sin redimensionar qty. Esto invalida una interpretación fuerte del comentario “envolvente autoritativa”.

**Contraejemplo de composición:** con N=100 USD y margen libre 100, un exec_leverage=1 exige margen 100>85. El adaptador eleva a ceil(100/80)=2, cuyo margen 50 pasa el guard de 95 %. Un techo previo de 1 deja de ser techo. Si d=2 %, la pérdida nominal sigue siendo 2 USD para cualquier L: un presupuesto Kelly de 1 USD no se cumple por escoger 2x ni por haber calculado previamente 1x. Se trata de valores para refutar la propiedad universal de la composición, no de una orden observada.

**Impacto sobre el genoma:** hay regiones donde cambiar f o vol_brake modifica sólo financiación, o queda absorbido por floor entero a 1 y ajuste posterior. El gen estadísticamente “adaptado” puede tener derivada nula respecto a cantidad y riesgo. Si un guard veta algunas órdenes, sí cambia el conjunto operado; no equivale a un control continuo de exposición.

**Cierre de máxima prioridad:** distinguir apalancamiento económico N/C del apalancamiento de margen configurado en el exchange. Proyectar la cantidad candidata a riesgo monetario y filtros, actualizar estado local/fees/reserva coherentemente, y verificar la desigualdad sobre cantidad realmente enviada. Si el mínimo realizable excede el presupuesto, retornar inviabilidad. Cualquier adaptación de margen debe conservar esa cantidad limitada y las restricciones de liquidación. La regresión debe recorrer propuesta→gate→redondeo→adaptador→payload→ledger; probar sólo max_leverage no alcanza.

## 4. Teoría temporal: qué significa realmente un universo continuo

La representación defendible es una familia multivariante condicionada por escala, no dos motores discretos. Una coordenada ℓ=ln(τ/τ₀) permite estudiar memoria, dependencia cruzada e incertidumbre como funciones de escala. τ₀ fija unidades; no es una “escala verdadera” del mercado. Es admisible aproximar esas funciones con un número finito de bases y refinar donde el error de aproximación o decisión lo justifique.

### 4.1 Cuatro capacidades diferentes

1. **Representar un horizonte:** que el tipo numérico y la función acepten 1 ns o cien años.
2. **Observar información:** disponer de eventos con resolución, sincronía y contenido suficientes.
3. **Identificar un modelo:** distinguir parámetros a partir de esa información con incertidumbre.
4. **Actuar:** disponer de una decisión factible, una infraestructura y una latencia compatibles.

Aceptar una τ no prueba las otras capacidades. El banco revisado usa 32 memorias EWMA y timestamps en milisegundos; un bucle por nanosegundo no crea eventos observados. Hacer 10⁹ actualizaciones vacías por segundo no añade información y puede agravar latencia/consistencia. Actualización por eventos con transiciones analíticas de Δt permite una semántica de tiempo continuo sin fingir muestreo infinito.

Para τ mucho menor que el intervalo observado, α=1−exp(−Δt/τ) se aproxima a uno: muchas escalas rápidas se vuelven indistinguibles para la observación disponible. Para τ muy grande respecto de la historia, apenas hay masa aprendida y dominan inicialización/prior. Esto es identificabilidad y condicionamiento, no falta de un nombre más avanzado. La corrección numérica con expm1 preserva pequeñas masas; no inventa evidencia secular.

### 4.2 Qué se revalida y qué no se reabre

CES-009 ya trató anclas y dominio operativo; CES-014 trató crédito/memoria; FMT-040/041 trataron banda y significado de fricción. Esta ronda no renumera esos problemas. Constata que el espectro representable es más amplio que la banda 30 s–12 h usada por ciertas vistas, y que expected_duration puede seleccionar otra τ. No se resume el sistema como “sólo dos horizontes”, porque sería falso.

La presencia de nombres scalp/swing puede ser compatibilidad de esquema, vista diagnóstica o decisión efectiva. La auditoría debe distinguirlos por flujo de datos. Borrar una etiqueta no amplía observabilidad; conservar una etiqueta no prueba por sí solo bifurcación. El criterio es si una función de escala gobierna efectivamente la decisión final y si su sensibilidad sobrevive a clamps, pisos y conversiones.

El requisito práctico es publicar soporte temporal, warmup, error, resolución de datos y dominio de acción de cada canal. Los resultados fuera del soporte deben llamarse extrapolaciones/priores, no conocimiento omnisciente. Para cien años hacen falta escenarios y riesgo de modelo; no hay base en estas lecturas para afirmar calibración empírica a ese horizonte.

## 5. Integraciones teóricas nuevas y contratos de admisión

Las rondas anteriores ya proponen T01–T26: estado con Δt, wavelets, Hawkes, rough paths, grafos/Hodge, aprendizaje online, cambio bayesiano, conformal, evidencia secuencial, crecimiento robusto, control de inventario, evolución restringida, Koopman, Volterra, cointegración, covariación asíncrona, primeras llegadas, identificabilidad genómica, modelos cuántico-inspirados, optimización estable, evaluación off-policy, holdout reutilizable, drawdown, pruebas metamórficas, transporte de coordenadas e información dirigida. No se sustituye ese catálogo por una lista de ecuaciones prestigiosas.

### T27 · Viabilidad y filtros de seguridad: proyectar la propuesta sobre acciones realmente factibles

**Fuente primaria contrastada:** [Ames, Xu, Grizzle y Tabuada, Control Barrier Function Based Quadratic Programs for Safety Critical Systems](https://arxiv.org/abs/1609.06408). Su marco relaciona condiciones diferenciales de barrera con invariancia de conjuntos y combina seguridad y desempeño mediante optimización. Las garantías dependen del modelo, regularidad y controles admisibles; no son automáticamente garantías para mercados con saltos.

**Aplicación propuesta, inferencia de esta auditoría:** definir K(x) como conjunto de acciones válidas en el estado observado: cantidad, lado, precio límite, stop, reserva de costes, margen y límites de exposición. Hallar una acción próxima a la propuesta genómica dentro de K(x), no modificar sucesivamente restricciones hasta que alguna orden pase. Si K(x) no contiene una entrada positiva, la abstención es una solución legítima.

Una formulación inicial no requiere solver sofisticado: para stop y costes dados, resolver N·d+coste(N)≤B y proyectar Q sobre el step_size; luego revalidar. Cantidad mínima, ticks y modos de orden introducen restricciones discretas/no convexas. Por eso no se puede afirmar que todo el problema sea un QP. La incertidumbre de precio/slippage exige escenarios, cotas o restricciones probabilísticas, cuya fiabilidad debe evaluarse aparte.

**Qué significan los cálculos:** h(x)≥0 describe una reserva de seguridad; su evolución sirve para verificar que la acción no salga del conjunto permitido bajo las hipótesis. El objetivo de proximidad preserva, cuando sea factible, la intención del modelo. El solver no produce alpha: organiza restricciones que hoy se contradicen.

**Coste y prueba de valor:** comenzar con proyección escalar determinista y medir p99 antes de añadir optimización de cartera. Comparar contra la política vigente sobre propuestas idénticas y verificar cero violaciones nominales post-redondeo; después ensayar gaps y datos atrasados. Rechazar la integración si sólo reduce rechazos cambiando el presupuesto, si no detecta conjunto vacío o si el coste de cómputo incumple el deadline.

**Límites:** una garantía determinista de invariancia no sobrevive sin más a saltos no acotados, fallos de contraparte, ejecución parcial o datos corruptos. La seguridad operativa debe incluir monitor y reconciliación. No se propone eliminar los frenos actuales mientras se construye este contrato.

### T28 · Reducción de orden orientada a observabilidad: gastar cómputo donde las escalas son distinguibles

**Fuentes primarias localizadas:** [Gramians, Energy Functionals and Balanced Truncation for Linear Dynamical Systems with Quadratic Outputs](https://arxiv.org/abs/1909.04597) y [Numerical computation and new output bounds for time-limited balanced truncation of discrete-time systems](https://arxiv.org/abs/1902.01652). Sus resúmenes indexados describen reducción mediante controlabilidad/observabilidad y cotas ligadas al problema. La consulta del cuerpo del segundo artículo no completó respuesta utilizable en esta ronda; no se presenta como leído íntegramente ni se importan sus teoremas sin verificar hipótesis.

**Problema local:** un banco enorme de filtros puede gastar memoria/latencia en estados casi equivalentes para la entrada observada o irrelevantes para la salida de decisión. Una representación continua no obliga a mantener todas las escalas con igual densidad ni a actualizarlas mediante un reloj imposible.

**Diseño a evaluar:** congelar una versión del núcleo lineal de filtros ẋ=Ax+Bu, y definir salidas Cx que correspondan a observables concretos, no a un score seleccionado después de ver el resultado. Para un horizonte de estudio T, los integrales P_T=∫₀ᵀ e^(At)BBᵀe^(Aᵀt)dt y Q_T=∫₀ᵀ e^(Aᵀt)CᵀCe^(At)dt describen qué estados puede excitar la entrada y cuáles afectan a esas salidas. Son definiciones del modelo lineal propuesto, no cálculos ya implementados en el motor.

La combinación de ambas estructuras orienta qué estados redundantes comprimir. Un estado poco excitado no es necesariamente inútil en un cambio de régimen; debe conservarse exploración/monitorización y poder ampliar la base. Tampoco una pequeña energía de error en señales implica pequeño error económico si la política contiene umbrales discontinuos como FMT-097.

**Integración y coste:** estimación/actualización de la base fuera del camino crítico, snapshots versionados y evaluación de un banco reducido en línea. El modelo completo incluye normalizaciones, tanh, ganancias epigenéticas y reloj irregular: no es el LTI de la derivación. La reducción lineal es un candidato para un subbloque; sus garantías no se transfieren al pipeline completo.

**Falsación:** comparar base original, refinada y reducida sobre replay retenido; medir error por escala, discrepancia de acciones, violaciones de riesgo, memoria y p99. Rechazar si la reducción sólo mejora tiempo medio a costa de colas o pierde señales relevantes fuera del periodo usado para escoger la base. Esta propuesta complementa T14 y T13; no demuestra que más reducción ni más dimensiones produzcan rentabilidad.

### 5.1 Física, cuántica y problemas del milenio: criterio de transferencia

Una teoría de otro dominio debe aportar un mecanismo identificable: variables observables, dinámica, unidades, condiciones y predicción falsable. No basta con colocar un término de Navier–Stokes, una función de onda o una etiqueta “topológica” dentro de una puntuación. El precio no se convierte en velocidad de un fluido por renombrarlo; un balance financiero exige una ley de conservación del ledger, no una analogía verbal.

Para ecuaciones de transporte o difusión, especificar qué distribución evoluciona, cuál es el generador, cómo se estiman drift/difusión/saltos y qué condiciones de frontera representan órdenes y liquidez. Una ecuación de Fokker–Planck o un modelo de primeras llegadas puede ser pertinente si sus predicciones superan un baseline; su uso no necesita resolver un problema del milenio ni permite adjudicarse sus garantías.

Para teoría cuántica, distinguir hardware cuántico, simulación clásica de sistemas cuánticos y métodos clásicos inspirados en sus estructuras. Exigir observable, preparación/entrada, salida y coste total, incluidos carga de datos y medición. En los archivos de esta ronda no se demuestra ventaja cuántica. Tensores, SIMD, números complejos, entropía o un nombre Quantum no bastan.

Los problemas del milenio delimitan cuestiones matemáticas precisas, no constituyen una librería de alpha. Su prestigio no altera el criterio de admisión. La revisión oficial de estado de Clay ya quedó documentada en la ronda IV; aquí no se anuncia ninguna nueva resolución ni se condiciona la rehabilitación del sistema a resolver esas cuestiones.

La mejora científica inmediata es más concreta: ecuaciones con significado verificable, decisiones factibles y feedback correcto. T27 y T28 se eligen porque enfrentan fallos observados; deben competir experimentalmente con alternativas simples. Si no aportan valor bajo coste e incertidumbre, no se integran por complejidad.

## 6. Matriz modular raíz–cima y hoja de rehabilitación

| Módulo del informe maestro | Evidencia ampliada aquí | Criterio de cierre siguiente |
|---|---|---|
| 1. Ingestión, parsers, L2, normalización | Resolución y soporte temporal; contexto as-of | Preservar identidad y orden de disponibilidad, no sólo valores |
| 2. IA, modelos, señales | FMT-100/101/111 | Estadísticas con población, memoria y faltantes explícitos |
| 3. Estrategia, régimen y horizontes | FMT-095–097/112 | Familia de barreras y estados consistente entre escalas |
| 4. Ejecución HFT y conectividad | FMT-103–106/110/113 | Intención→payload→fill con cantidad y límites preservados |
| 5. Riesgo, Kelly y genomas | FMT-095–102/113 | Probar riesgo monetario realizable, no sólo un leverage |
| 6. Memoria, atomicidad y telemetría | FMT-107–111 | Concurrencia, durabilidad y recuperación verificables |
| 7. Orquestación y confluencia | FMT-112 y composición de gates | Distinguir campos derivados, masa, acuerdo e incertidumbre |
| 8. Backtesting y gobernanza | Simulador auxiliar frente a replay real; T27/T28 | Paridad causal más corrección independiente y evidencia retenida |

### Orden de rehabilitación propuesto, no ejecutado

1. **Cantidad y riesgo terminal:** FMT-113, seguido de 098/099. Establecer una única restricción sobre pérdida monetaria final, incluyendo costes, filtros y efectos del adaptador de margen.
2. **Geometría coherente:** FMT-095–097. Barreras, probabilidad y escala deben evaluarse conjuntamente; conservar las correcciones CES sin atribuirles garantías adicionales.
3. **Crédito íntegro:** FMT-109–111. Recuperar identidad y evidencia antes de usar resultados para evolución; un algoritmo más sofisticado aprende peor si recibe resultados de otra posición.
4. **Memoria del estimador:** FMT-100/101. Definir población, unidad y reloj de adaptación; comparar derivadas del fenotipo frente al gen en cada entorno.
5. **Capacidades auxiliares:** FMT-102–108. Validar antes de conectar. Que una clase exista en un crate no significa que mejore el sistema ni que explique su comportamiento vivo.
6. **Representación y coste:** FMT-112, T28 y el catálogo previo. Medir soporte y error entre escalas; evitar tanto categorías temporales arbitrarias como promesas de resolución inexistente.
7. **Validación de promoción:** casos adversariales, replay determinista, evaluación retenida y benchmark de latencia/recursos. Mantener un registro de hipótesis/ensayos para no convertir selección repetida en falsa certeza.

Cada reparación futura necesita test que falle antes, corrección, prueba de integración productor→terminal, comparación de decisiones y revisión de efectos. Un control de riesgo no se cierra con un test de que el resultado es finito; un gen no se considera activo sólo porque se serializa; una teoría no queda validada por una nomenclatura.

## 7. Verificación realizada, trazabilidad y límites

### 7.1 Tests existentes ejecutados

~~~text
cargo test -p risk-engine --lib --offline tp_sl::tests
cargo test -p risk-engine --lib --offline kelly_envelope::tests
cargo test -p risk-engine --lib --offline capital_compounder::tests
cargo test -p quantum-arena --lib --offline temporal_spectrum::tests
~~~

Resultados: **9 + 6 + 2 + 17 = 34 tests aprobados; cero fallidos.** No se añadieron tests ni se cambió código operativo. Cargo generó artefactos normales de compilación. No se ejecutó el motor, el evolver, una orden, un test de concurrencia destructivo ni los tests de diario que escriben en data.

Se calcularon separadamente réplicas numéricas de TP/SL, cambio de p, frontera micro, factibilidad de riesgo mínimo, doble EMA, regularización monetaria, redondeo IOC e interpolación no lineal. Son contraejemplos de fórmulas leídas, no nuevas pruebas Rust integradas. Los interleavings y fallos de disco se documentan como razonamiento adversarial y regresiones pendientes, no como experimentos realizados.

Por qué una batería verde no cierra estos puntos:

- El test de determinismo TP/SL llama dos veces a la misma función; no recorre consumidores ni probabilidades distintas.
- El test de energía comprueba nodos; no demuestra identidad entre nodos.
- Las pruebas de Kelly no cubren todo el adaptador de margen ni la cantidad final.
- El test llamado “Monte Carlo” de la envolvente contiene una trayectoria determinista de pérdidas; su nombre no acredita una estimación de probabilidad de ruina.
- Los tests del simulador que comprueban Ok no demuestran conservación de posiciones ni fidelidad de recompensa.

### 7.2 Manifiesto de lectura completa

SHA-256 abreviado a 16 caracteres: identifica la versión observada; no es una firma de certificación.

| Archivo relativo al repositorio | Líneas | SHA-256, prefijo |
|---|---:|---|
| crates/risk-engine/src/tp_sl.rs | 400 | 9a3a34a9e0a12e01 |
| crates/risk-engine/src/capital_compounder.rs | 222 | 85685740667bd63e |
| crates/execution-engine/src/router.rs | 301 | c154fb30fbd1b124 |
| crates/storage-engine/src/temporal_store.rs | 180 | ed8c41848bcf19d9 |
| crates/quantum-arena/src/temporal_spectrum.rs | 1153 | 079677d8b9c6c828 |
| crates/risk-engine/src/kelly_envelope.rs | 363 | 40957eb957d3f427 |
| crates/execution-engine/src/simulator.rs | 505 | 7e1645b93e6bf356 |
| crates/execution-engine/src/trade_accounting.rs | 491 | b23699dc9a88d46f |

### 7.3 Lecturas dirigidas: no cuentan como archivos completos

Se siguieron funciones y llamadas en host, risk-engine/lib, executor, user_data_stream y los dos backtests. También se revisaron constantes genómicas, helper de régimen ya cubierto en IV, documentación CES y anexos históricos. Los siguientes hashes identifican el corte de los archivos grandes enlazados, no prueban su lectura completa:

| Archivo | SHA-256, prefijo |
|---|---|
| src/bin/god_engine.rs | f2ba92c4283c66e6 |
| crates/risk-engine/src/lib.rs | e9883455b6bebefa |
| crates/execution-engine/src/user_data_stream.rs | 33c2cc1aefc61d74 |
| crates/backtest-engine/src/booktick_replay.rs | 736cc1dbb38e21d7 |
| crates/backtest-engine/src/lib.rs | 297c921255045fa7 |
| crates/execution-engine/src/executor.rs | 4e5722ab137d6da2 |

La investigación utilizó la habilidad de Firecrawl para búsqueda de artículos y extracción de pasajes primarios. La CLI no estaba disponible; se usó el índice conectado conforme al fallback de la habilidad, sin instalación. T27 tiene pasajes del cuerpo consultados; T28 se presenta como propuesta apoyada en resúmenes indexados y con verificación del texto completo pendiente. No se enviaron archivos del proyecto a ese servicio.

### 7.4 Integridad y estado de trabajo

Se añadió este anexo al ATLAS y al informe maestro y se enlazó desde IV, conservando el contenido anterior. La verificación de prefijos normaliza sólo CRLF/LF. El repo ya contenía cambios operativos y artefactos concurrentes; no se sobrescribieron ni se atribuyen a esta ronda.

Comprobación final: los tres prefijos históricos conservan su SHA-256 normalizado; los ocho archivos de lectura completa y los seis de lectura dirigida mantienen los hashes capturados. Se verificaron 19 encabezados FMT únicos, 36 enlaces locales con archivo y línea válidos y 90 rutas Rust distintas en los manifiestos acumulados. El chequeo de whitespace de los índices versionados no informó errores. Estas verificaciones acotan la reproducibilidad del anexo, no certifican el sistema entero.

No hubo cambios de configuración/genomas, refactorización operativa, despliegue, trading, reinicios, commit, push, merge o fetch. La inspección de Git se limitó a estado local, inventario y HEAD. Por tanto, tampoco se certifica que todos los trabajos anteriores estén publicados ni que otras ramas hayan resuelto los hallazgos.

**Conclusión:** un sistema verdaderamente adaptativo necesita que el aprendizaje cambie una decisión ejecutable y que esa decisión vuelva como evidencia correcta. Esta ronda demuestra varias aristas donde eso no se conserva. La prioridad no es añadir ecuaciones por prestigio, sino cerrar esos contratos, medir sus límites y después admitir teorías nuevas por valor experimental. La auditoría integral sigue pendiente; estos 19 puntos quedan documentados, no reparados.

## Continuación VI — reparación posterior, sin alterar este corte histórico

La [Auditoría científica VI y rehabilitación verificable del riesgo](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_VI_2026-09-24.md>) registra cambios posteriores autorizados. FMT-096 se repara localmente; FMT-098/099 se corrigen en la envolvente, con pruebas del consumidor maduro. Se añaden FMT-114–116 por datos inválidos/overflow, validación de notional y pisos estadísticos que inventaban edge; los tres reciben correcciones locales. La declaración anterior de 19 puntos abiertos describe el corte V y se conserva como historia, no como estado actualizado de cada helper.

La nueva proyección monetaria ExposureBudget prepara FMT-113, pero **no lo cierra**: falta aplicarla a cantidad final, reserva y payload en host/replay, cuyos cambios concurrentes se preservaron. También permanece el bootstrap externo n<30. No se repararon por esa sola API el conflicto de barreras, doble EMA, identidad de cierres ni durabilidad del diario.

Se aprobaron 76 tests seleccionados, incluidos 24 nuevos; 12 regresiones fallaron antes del arreglo y pasaron después. cargo check del host terminó sin ejecutar el motor. La cobertura completa de lectura sube a 91 Rust preexistentes distintos, más dos tests nuevos contados por separado. Se documentan supuestos, unidades, incertidumbre y fronteras nominales del riesgo; no hay certificación de rentabilidad, auditoría integral ni despliegue. No se modificaron genomas ni se realizaron commit, push, merge o fetch.

## Continuación X — semántica explícita entre nodos para FMT-112

[Auditoría científica X](<C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_X_2026-09-24.md>) cierra la ambigüedad de representación de FMT-112
eligiendo y documentando interpolación independiente de observables nodales.
Se conservan señales, masas y pesos existentes: I[tanh(z)] no se iguala
artificialmente a tanh(I[z]), y masa positiva no se interpreta como ausencia
de cancelación direccional. Dos tests reproducen esas propiedades.
No se demuestra con ello la utilidad económica del arbitraje por masa.

FMT-133 repara un helper distinto: derivada local del interpolante frente
a secante que mezclaba tramos. FMT-134 registra discrepancias todavía
abiertas entre lectores del genoma. FMT-113 permanece abierto: no se cambió
cantidad final, payload ni adaptadores del host/replay. Esta adenda amplía
el informe previo sin alterar su evidencia histórica.
