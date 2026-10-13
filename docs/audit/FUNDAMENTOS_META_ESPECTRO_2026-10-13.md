# Fundamentos de la meta, el espectro y la evidencia

Fecha: 2026-10-13. Alcance: especificación matemática y revisión parcial; implementación y validación económica abiertas. Este documento no modifica límites ni demuestra duplicación del capital.
El protocolo de las secciones 1–4 procede del recibo revisado el 2026-10-11, con fuentes congeladas en fcda9d262a7ec529cfe2c2d16e38d994e2ea3258 y árbol 39123720ee202f756642a4a81585802658202f1a. Se reproduce aquí para que su consulta no dependa de un log ignorado.
El contraste documental y de APIs de las secciones 5–7 corresponde a main 3dc88468114205b0257b5e1fd82688a652dede2f, árbol de98a1257e0db515b4d4471a865c508a0d27b8f3. No confundir esos snapshots con una certificación del árbol de trabajo o de cambios posteriores.

## 1. Patrimonio observable y objetivo a 72 horas

Sea E(t) el patrimonio neto MTM de la cartera conjunta, en un numerario común: efectivo, posiciones y obligaciones. Comisiones, financiación, préstamo y ejecución se contabilizan una sola vez. La valoración y el coste estimado de liquidación se distinguen; no se resta otra vez un coste ya incluido en los fills.
Cada precio, tipo de cambio, fill y flujo necesita identidad y tiempo de disponibilidad. Una marca futura o de otro instante no es capital conocido al decidir. La conciliación de posiciones y pasivos corresponde al contrato contable, no al estimador de rendimiento.
Para excluir flujos externos, sea U(t)=E(t)/N(t), con unidades N(t)>0. Un flujo F(t), positivo para aportar y negativo para retirar, modifica N por F(t)/U(t−), valorando inmediatamente antes del flujo. Con E(t+)=E(t−)+F(t), U permanece igual.
La continuación exige U(t−)>0 y N(t+)>0. Un reembolso total deja E=N=0 y U indefinido: es cierre por reembolso, no insolvencia. Registrar su censura con una regla prefijada; omitir esos bloques cambia el estimando.

Con H=72 horas, definir para capital positivo al inicio:

\[
G_t^H=\log\frac{U(t+H)}{U(t)},\qquad \Delta_t=G_t^H-\log 2.
\]

Duplicar corresponde a Δ≥0. Registrar tanto bloques consecutivos como ventanas móviles, con anclajes y reglas para marcas faltantes fijados antes de evaluar.
La identidad de referencia es g=log(2)/3=0,2310490602 día⁻¹; el rendimiento simple diario equivalente es exp(g·1 día)−1=0,2599210499. Son identidades de capitalización del objetivo, no previsiones.
E[G]≥log(2), P(G≥log(2)) y éxito en todas las ventanas son afirmaciones diferentes. La esperanza sólo se reporta si está definida; una media sin control de colas no adquiere una garantía por llamarse crecimiento logarítmico.
G=−∞ corresponde a U(t)>0 y U(t+H)=0. Una ventana iniciada sin patrimonio positivo tiene log no definido. Insolvencia permanece como fracaso absorbente en el indicador de éxito; conservar los fallos en el denominador de evaluación.
Patrimonio negativo requiere registrar insolvencia y deuda, no un log ficticio. Ninguna de estas situaciones se resuelve sustituyendo capital por un epsilon.

## 2. Drawdown, ruina y frontera de políticas

Fijar el origen del seguimiento del pico y conservarlo entre ventanas:

\[
M(t)=\sup_{s\le t}U(s),\quad D(t)=1-\frac{U(t)}{M(t)},\quad D_T^{\max}=\sup_{t\le T}D(t).
\]

Insolvencia contable, liquidación contractual e imposibilidad operativa de continuar son eventos distintos. Un rechazo de entrada o un freno protector no equivale automáticamente a ruina.
El rango d∈[0,1] presupone riqueza no negativa. Si se conserva patrimonio negativo, DD puede superar uno: extender el dominio y registrar deuda, o declarar absorción en cero y registrar deuda aparte.
Una barrera respecto de riqueza inicial no es una barrera desde pico. El modelo de [Busseti, Ryu y Boyd, §§2.1 y 3](https://web.stanford.edu/~boyd/papers/pdf/kelly.pdf) supone retornos IID y define el mínimo respecto de riqueza inicial normalizada; su garantía no se transfiere directamente al DD móvil.

Sea Π el conjunto de políticas causales admisibles, incluyendo activo, horizonte, estrategia, tamaño, ejecución y actualización. Capital, lotes y margen imponen restricciones; capacidad, impacto, colas, latencia, fills y liquidez futura requieren estimación y estrés.
Para cada política π, capital inicial, horizonte T y modelo P contrastable, fijar t o una distribución de orígenes antes de evaluar. Cuando las magnitudes estén definidas, estimar conjuntamente:

\[
\left(E_P[G_t^H],\ P_P(G_t^H\ge\log2),\
\{P_P(D_T^{\max}>d)\}_d,\ P_P(\text{ruina antes de }T)\right).
\]

Construir la frontera no dominada con incertidumbre y repetirla por capital y condiciones de ejecución. Presenta intercambios crecimiento–riesgo; no convierte el riesgo mínimo estimado en riesgo aceptable.
Si se utiliza una familia de modelos \(\mathcal P\), distinguir inf_P de crecimiento/probabilidad y sup_P de colas/ruina. El peor caso sólo cubre los modelos incluidos, no una dinámica real que quede fuera.
Una búsqueda finita sin candidato no demuestra imposibilidad matemática; una prueba de inviabilidad necesita una cota válida para todo Π bajo supuestos declarados.
La tolerancia de pérdida, la probabilidad de éxito exigida y el error inferencial son elecciones de política. La meta fija log(2); no fija un DD universal, un 5 % ni una frecuencia mínima. Mostrar la frontera permite avanzar sin inventarlos.

## 3. Por qué debe explicitarse el horizonte del DD

Supuestos exclusivos del ejemplo: infinitas operaciones; indicadores de pérdida IID, q∈(0,1); fracción fija f∈(0,1); patrimonio positivo y finito en cada instante discreto; sin flujos externos. Cada pérdida multiplica el patrimonio por 1−f. No es un modelo certificado del sistema adaptativo.
Para cualquier d∈[0,1), elegir un entero L>log(1−d)/log(1−f). Tras L pérdidas desde W(a)>0:

\[
W(a+L)=W(a)(1-f)^L,\quad M(a+L)\ge W(a)
\ \Longrightarrow\ D(a+L)\ge1-(1-f)^L>d.
\]

No hace falta que la racha empiece en un pico. Cada bloque disjunto de L operaciones es íntegramente perdedor con probabilidad q^L; la probabilidad de no observar ninguno entre n bloques es (1−q^L)^n→0.
Cada umbral d<1 se supera casi seguramente. La intersección contable de estos sucesos para racionales d↑1 prueba sup_k D(k)=1 casi seguramente.
En cada instante finito D(k)<1: el supremo puede ser uno sin alcanzarse, sin patrimonio cero y sin ruina. Es compatible con crecimiento logarítmico positivo si otros supuestos lo establecen. También subsiste condicionado a no tocar una barrera inicial cuando ese suceso tiene probabilidad positiva; no garantiza esa supervivencia.
Para N operaciones fijo sólo se obtiene la cota inferior P(D_N^max>d)≥1−(1−q^L)^floor(N/L), no la probabilidad exacta. Convertir N a tiempo calendario exige modelar la actividad.
La derivación no se traslada automáticamente a fracciones variables, adaptación, dependencia o interrupciones. Distingue eventos matemáticos; no demuestra un fallo del sistema ni justifica retirar controles.

## 4. Evidencia causal y una cota secuencial explícita

Antes del tramo OOS, congelar el algoritmo entero, incluida su regla de actualización. Las decisiones usan información disponible; las etiquetas sólo actualizan cuando están disponibles.
Elegir arquitectura o hiperparámetros después de ver resultados OOS requiere evidencia posterior no utilizada en esa elección, con su dependencia temporal declarada. Reutilizar adaptativamente el holdout puede sobreajustarlo: [Dwork et al.](https://arxiv.org/abs/1506.02629).
Activos, motores y horizontes que comparten mercado o capital no son réplicas independientes. Tampoco lo son ventanas de 72 horas solapadas. El remuestreo por bloques multivariantes requiere supuestos contrastables de dependencia y estabilidad.
Candidato y control comparten tramos, capital y restricciones; comparar diferencias pareadas, conservando fallos. Un replay es una realización bajo su ejecución simulada; no identifica por sí solo probabilidades poblacionales.

Para K candidatos prefijados y bloques consecutivos de 72 horas, sea X(k,j)=1 si se cumple la duplicación, con insolvencia como fracaso absorbente. Reembolsos o marcas faltantes requieren diseño explícito de censura; no seleccionar silenciosamente sólo bloques completos.
Definir m(k,j)=E[X(k,j)|F_inicio,j] bajo la ley verdadera, no una predicción puntual asumida calibrada.
Para esta prueba, X(k,j) debe ser medible respecto de la filtración antes del inicio del bloque siguiente. Si etiquetas o conciliaciones llegan tarde, la información disponible para operar puede no satisfacerlo.
Declarar una filtración de auditoría válida y definir m respecto de ella, distinguiéndola de la información usada por la política causal, o adaptar el diseño. Retrasar solamente la publicación de la banda no repara la condición.

Para 0<α<1 y n≥1, Hoeffding condicional da dos colas con cota 2 exp(−2nε²): X tiene rango de longitud uno y X−m es diferencia de martingala. Elegir

\[
\epsilon_{n,K,\alpha}=
\sqrt{\frac{\log(K\pi^2 n^2/(3\alpha))}{2n}}.
\]

Entonces 2 exp(−2nε²)=6α/(Kπ²n²), y la unión sobre candidatos y tiempos satisface

\[
\sum_{k=1}^{K}\sum_{n=1}^{\infty}\frac{6\alpha}{K\pi^2n^2}=\alpha.
\]

Con probabilidad al menos 1−α, simultáneamente para todo k y n:

\[
\left|n^{-1}\sum_{j=1}^{n}(X(k,j)-m(k,j))\right|\le\epsilon_{n,K,\alpha}.
\]

La banda para n⁻¹Σm(k,j) es [X̄−ε,X̄+ε]∩[0,1]. No requiere IID bajo la filtración declarada, pero acota el promedio de probabilidades condicionales de los bloques evaluados: no prueba una ley futura constante, todas las ventanas móviles ni riesgo de ruina.
α es error inferencial, no riesgo económico. G no acotado no hereda esta cota. Añadir candidatos adaptativos fuera de K requiere otro presupuesto y diseño inferencial válidos.
[Howard et al.](https://arxiv.org/abs/1810.08240) desarrollan secuencias de confianza bajo condiciones explícitas; la fórmula anterior es la derivación elemental mostrada aquí, no una fórmula atribuida a ese artículo.

## 5. Corrigendum del plan y memoria del 13 de octubre

Referencias de líneas del snapshot 3dc88468: [plan maestro](../PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-13.md) y [.agents/MEMORIA.md](../../.agents/MEMORIA.md). La clasificación siguiente distingue una identidad, una especificación incompleta y una hipótesis; no declara auditadas todas sus implementaciones.

| Ancla | Corrección o límite |
|---|---|
| Plan 108–110; memoria 7 | El plan separa correctamente g=0,231049 día⁻¹ de 25,9921 % simple diario; la memoria los vuelve a igualar. Usar exp(g·Δt)−1 con unidades temporales explícitas. |
| Plan 62 y 318; memoria 7 | d*=1−0,05^(1/3)=0,63159685 y 13(1−d*)=4,78924 son aritméticamente correctos. No constituyen preservación absoluta, garantía del DD desde pico ni prueba de inexistencia del edge tras un evento de cola. |
| Plan 118–123 | 2·fee·L expresa coste como fracción del margen cuando fee es fracción del nocional y L=nocional/margen. No es una cantidad en R sin dividir por el riesgo monetario que define R. El valor 0,10 R no queda derivado. |
| Plan 140 | max(MinNotional,f·C/SL) ordena nocional positivo aun si f=0. Si el presupuesto no alcanza el mínimo ejecutable, el conjunto factible puede no contener ninguna orden; no se eleva el presupuesto mediante max. Hallazgo documental, no atribución de esa fórmula al runtime. |
| Plan 79 y 172; memoria 8 | Se mezclan 4^k μs y 100 μs·2^(k/2). La segunda fórmula acaba en 4,634095 s para k=31, no 24 h. Ninguna coincide con la unidad inicial del código. |
| Plan 164–178; memoria 8 | L² necesita dominio/medida, producto interno, integrabilidad y representación de los datos. Una variedad riemanniana necesita métrica y tratamiento del borde del simplex. Pesos baricéntricos no prueban que todas las decisiones sean C∞. |
| Plan 184–239 | Las analogías físicas y parámetros conductuales son hipótesis que requieren operacionalización y contraste; escribir una ecuación no demuestra ventaja, causalidad ni ejecución cuántica. |
| Plan 54 y 301 | Latencia <25 ns, cero asignaciones y exhaustividad requieren inventario y mediciones reproducibles. No son conclusiones medidas por este recibo ni consecuencias de O(1). |

Para un stop como fracción de precio SL y coste por nocional c adoptado en el modelo, pérdida monetaria modelada=N(SL+c). Un presupuesto de pérdida f·C da N≤f·C/(SL+c); usar SL solo mide pérdida bruta. Los costes inciertos y gaps requieren su propio modelo. Si el payoff ya es neto, no volver a restar las mismas fricciones.
Esperanza aritmética positiva no implica crecimiento log positivo a cualquier fracción. Sin costes, p=1/2, b=2,25 y f=0,8 dan E[R]=0,625, pero E[log(W'/W)]=½log(2,8)+½log(0,2)=½log(0,56)<0. Es un contraejemplo algebraico, no una estimación de mercado.
“Medio Kelly” es una elección de fracción/riesgo bajo un modelo; no el óptimo universal. La fórmula de barrera asociada a Kelly fraccionario no cambia el evento inicial por DD móvil. Suponer un modelo y fijar α tampoco convierte la meta en un teorema.

La malla de [temporal_spectrum.rs, 70–102](../../crates/quantum-arena/src/temporal_spectrum.rs) es τ_k=10⁻⁶ ms·4^k=(1 ns)·4^k, k=0,…,31: llega a 4,611686018427×10¹² ms, aproximadamente 146,14 años.
Es una malla discreta logarítmica de base 4; 4^k=2^(2k) puede describirse como submalla diádica. La interpolación de τ no vuelve continua la observación. El reloj del core es u64 en ms: constantes sub-ms no acreditan resolución física sub-ms.
El código del espectro, 32–35, reconoce restricciones operativas heredadas [30 s,12 h]; no presentar su eliminación como universalmente demostrada.

En el modelo hidrodinámico, especificar coordenadas, unidades, condiciones de frontera y ecuaciones complementarias antes de usar Reynolds como magnitud identificada; sus umbrales de mercado requieren validación.
En gauge/Hodge, definir conexión, operadores, métrica y bordes. Asociar componentes exactas, coexactas o armónicas con reversión, manipulación o tendencia añade hipótesis empíricas; la descomposición por sí sola no identifica esas causas.
El K de la integral de Feynman es un propagador de amplitud, no directamente una probabilidad financiera. La relación con probabilidades requiere módulo cuadrado y normalización/estados adecuados; no basta una suma compleja. [Fuente primaria: Feynman, Leighton y Sands, III.1](https://www.feynmanlectures.caltech.edu/III_01.html).
No se ofrece aquí resolución de problemas del milenio ni se deriva de ellos una garantía de rentabilidad. La simulación clásica de una ecuación inspirada en física no demuestra uso de hardware cuántico ni ventaja cuántica.
Para el e-producto del plan 273–275, una condición suficiente exige r_max,t>μ, 0≤φ≤1, R_t≤r_max,t, apuestas/cotas predecibles y E[R_t|F_(t−1)]≥μ, con integrabilidad. Así los factores son no negativos y su esperanza condicional ≤1. La esperanza incondicional escrita no basta.
“Conformal libre de distribución” tampoco significa libre de supuestos de dependencia: declarar intercambiabilidad o el método apropiado para datos no intercambiables. [Angelopoulos y Bates, §§4.6 y 5.3](https://arxiv.org/pdf/2107.07511).

## 6. LH12: edad de cada cierre y contrato de consulta

En 3dc88468, [calibration.rs](../../crates/god-engine-core/src/calibration.rs) guarda sólo (score,won), línea 68; update_at, 170–175, conserva el máximo timestamp; fit, 192–209, da peso uno a cada cierre real. calibrate_at, 130–141, envejece el mapa completo desde el último cierre.
Un cierre reciente restaura el peso completo del mapa, aunque contenga cierres antiguos. La mezcla sí actúa entre cierres: a una hora, con semivida 12 h, el peso es 2^(−1/12)≈0,944. “No actúa nunca mientras se opera” es demasiado fuerte; el defecto es ausencia de edad individual en el ajuste.
La API añadida calibrate_at_tau, 147–167, cambia la semivida de esa misma mezcla, con límites heredados. La búsqueda del símbolo en Rust sobre 3dc88468 encontró únicamente su definición; no prueba integración productiva ni soluciona el peso de cada cierre.

Contrato propuesto para el ajuste temporal, con x_i=logit(s_i), p_i=σ(a·x_i+b), a≥0 y pérdida logística ℓ:

\[
L_t(a,b)=\sum_i w_i(t)\ell(y_i,p_i)
+\frac{\kappa}{n}\sum_i\ell(s_i,p_i)
+\frac{\lambda}{2}[(a-1)^2+b^2],
\qquad w_i(t)=2^{-(t-t_i)/H_c}.
\]

Aquí κ conserva la masa total de pseudo-observaciones y λ el ridge vigente. H_c=12 h conserva una política heredada; no es una semivida óptima demostrada ni H=72 h de la meta.
No normalizar los w_i por Σw_i: se perdería el descenso de masa real frente al prior. Objetivo, gradiente y Hessiana deben usar por fila q_i=w_i+κ/n y etiqueta (w_i y_i+(κ/n)s_i)/q_i.
El reloj de ajuste no retrocede; una observación tardía debe conservar su edad respecto de ese reloj. Las observaciones sin timestamp necesitan una convención de compatibilidad declarada, no una edad histórica inventada. Probar también update_at→update→update_at.
Conservar una mezcla O(1) entre cierres es una aproximación explícita, no el MAP temporal exacto en cada consulta. Consultar now anterior al último dato no reconstruye un estado histórico causal.
[estado_aprendido.rs, 57–58 y 160–177](../../crates/god-engine-core/src/estado_aprendido.rs) ya conserva triples (score,won,ts); 317–332 restaura por update_at. La nueva representación interna debe preservar ese contrato y su reloj.

El oráculo causal separa historias con iguales scores, resultados, recuento y último timestamp, pero distintas edades previas, por debajo de WINDOW para aislar envejecimiento de expulsión. El modelo anterior no las distingue.
Con score=0,5, W=Σw_i y Y_w=Σw_i y_i, un oráculo escalar resuelve (W+κ)σ(b)−(Y_w+κ/2)+λb=0; así incluye el ridge en vez de exigir una media Beta exacta.
Añadir pruebas de cadencia activa, traslación del reloj, cierre reciente que no rejuvenece los anteriores, consultas sin mutación, timestamp repetido/tardío, invalidez transaccional y restauración. Son contratos propuestos; este documento no ejecuta ni declara verdes esas pruebas.
LH12 no resuelve LG-5: el core etiqueta PnL neto>0 (lib.rs:4348), que no equivale a TP antes que SL. Tampoco corrige selección de operaciones ejecutadas, soporte local/global por recuento, ni identificación contrafactual del libro sombra. Permanecen trabajos separados.

## 7. Estado implementado y trazabilidad parcial

La revisión histórica de metrics.rs:114–174 encuentra CAGR de extremos sin flujos externos y DD suministrado por el llamador; sus ratios Sharpe/Sortino se declaran proxies de PnL monetario por trade. fitness_contract.rs:17–54 calcula log(Cf/Ci) menos penalización cuadrática de DD, sin duración.
Las APIs inspeccionadas no construyen G72 unitizado ni su frecuencia. ex_post_metrics_contract.rs verifica dominios y oráculos numéricos, no rentabilidad. Sus blobs no cambiaron entre fcda9d26 y 3dc88468; no se presume que todo otro consumidor sea idéntico.
Siguiente evidencia necesaria: contabilidad unitizada con flujos/censura; recibos por bloque con datos disponibles y procedencia; trayectoria conjunta con pico persistente; comparación pareada y frontera con incertidumbre. La auditoría contable y de ejecución conserva sus responsables.
No se ejecutaron Cargo, entrenamiento, trading, pruebas económicas ni benchmarks para este documento. T1 o una suite de contratos prueban sus respectivos oráculos, no +100 % cada 72 h. Estado por archivo: whole_file_closed=false.

| Fuente | Blob del contraste | Lectura decisiva |
|---|---|---|
| Plan maestro 2026-10-13 | 89dc15e89a5ecac0ea3e8919483d65c47484c290 | 1–425; revisión documental |
| .agents/MEMORIA.md en 3dc | c0bdae8a71248d7422deacc38abe57b7183fc313 | 7–8 y entradas LH12; parcial |
| calibration.rs en 3dc | b40e35a98fbf4e86ebf15e43c0b67287f2f8874e | 1–283 y pruebas relevantes; delta desde 6bb |
| temporal_spectrum.rs | 6abdef00e41fc6ed6270c1209536c00f506726af | 32–107; unidades y compatibilidad |
| god-engine-core/src/lib.rs | 3c3a500aacc5e72bdd02218702b1c96507f0e747 | 3302–3320, 4348, 4483–4489, 8178–8186 |
| estado_aprendido.rs | 70119a185559fe25cb751a31d04150423affa887 | 57–58, 160–191, 317–332 |
| backtest-engine/src/metrics.rs | bf99c0bc722e9efda25c5769ee7f0690a1b4e52e | 1–344; revisión histórica de API |
| ex_post_metrics_contract.rs | 4f813f5db0286e9303204d6b5f7e2f9303bb3ce1 | 1–219; leído, no ejecutado aquí |
| fitness_contract.rs | f249ea5d6ef8107b62e63af2d4849a350c08ff02 | 1–55; utilidad interna, no G72 |
