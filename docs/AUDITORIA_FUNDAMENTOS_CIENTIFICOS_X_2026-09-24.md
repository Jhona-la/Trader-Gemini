# Auditoría científica X — contrato temporal del genoma y semántica del campo espectral

Fecha: 2026-09-24. Corte local: rama main, HEAD 59a76de4. Continuación aditiva de las rondas I–IX y de la auditoría CES. Las fichas históricas se conservan; el estado actualizado de los puntos tratados se consigna aquí.

## 1. Resultado ejecutivo y alcance real

Se corrigió el cálculo de la banda de fricción del genoma para todos los signos de la pendiente y se sustituyó una secante por la pendiente local del interpolante que la API dice representar. Se aclararon las variables primitivas y derivadas del espectro **sin cambiar sus señales, pesos, masas ni decisiones de fusión**. Se añadieron pruebas reproducibles de una divergencia todavía abierta entre dos lectores del mismo genoma.

Esta ronda no certifica un motor universal operativo, ni autoevolución rentable, ni ventaja cuántica. Las restricciones temporales heredadas siguen presentes. El progreso verificable es acotado: contratos algebraicos reparados, semántica explícita, contraejemplos ejecutables y un diseño de investigación con hipótesis comprobables.

**Verificación:** 44 tests seleccionados aprobados; 21 son nuevos. De ellos, 12 regresiones fallaron contra el código previo y pasaron después. Otros tres tests nuevos caracterizan deliberadamente una divergencia semántica entre APIs todavía abierta: que pasen confirma el problema, no su reparación. La comprobación de compilación del host pasó sin ejecutar el motor.

**Cobertura de lectura:** 94 Rust preexistentes distintos con lectura completa acumulada, sobre la base de 289 Rust versionados; quedan 195 sin acreditar lectura íntegra. Esta ronda añade config.rs, 493 líneas. Releer temporal_spectrum.rs no incrementa el conteo; los tramos de genome.rs no equivalen a sus 3.372 líneas completas. Los tres tests nuevos se cuentan por separado. Los 1.119 archivos versionados y 24 manifiestos Cargo del inventario anterior no se convierten en archivos auditados por haberlos enumerado.

**Seguridad y concurrencia:** no se editaron genomas activos, host, replay, risk-engine/lib.rs ni god-engine-core/lib.rs. No hubo trading, promoción, despliegue, reinicios ni ejecución del evolver. No se realizó commit, push, merge ni fetch. Que main sea la rama local actual no prueba que las demás ramas estén integradas ni que el remoto contenga estos cambios.

### Matriz de resolución de esta ronda

| ID | Severidad de la ficha | Estado en X | Evidencia y límite |
| --- | --- | --- | --- |
| FMT-040 | P1 | Reparación algebraica local verificada | Banda creciente, decreciente, plana y vacía; no certifica ejecución ni extrapolación segura |
| FMT-041 | P1, conservada de la ronda II | Parcial | Comentarios y diagnóstico del gate corregidos; q=0,65 y la política operativa se conservan |
| FMT-112 | P2 | Contrato de interpolación aclarado y probado | Se elige interpolar observables nodales por separado; no se impone otra señal al motor |
| FMT-133 | P2 | Reparado localmente | Gradiente por tramos, derivada derecha en nodos y extrapolación constante |
| FMT-134 | P1 de paridad de lectores | Abierto | Tres testigos ejecutables prueban distintos dominios/pisos; amplía CES-009 |
| CES-008/009 | P1 | Abiertos | Resolución, soporte por escala, recortes operativos y genes de dos anclas |
| FMT-113 | P1 | Abierto, sin nueva reparación aquí | El presupuesto debe llegar a cantidad enviada; no lo cierra arreglar una banda temporal |

FMT-133 y FMT-134 son las dos fichas nuevas de esta ronda. No se renumeran los 305 puntos históricos ni se presenta esta matriz parcial como un recuento íntegro de incidencias. Las prioridades describen el riesgo del contrato, no la prueba de un incidente real.

## 2. FMT-040 — la geometría admisible dependía incorrectamente del signo del gen

### 2.1 Contrato, unidades y derivación

El genoma representa el stop mediante una ley de potencia:

```text
x = ln(tau / 1 ms)
SL(x) = exp(a + b*x)
f <= q*SL(x)
q = MAX_FRICTION_SHARE_OF_RISK = 0.65
c = ln(f/q)
a + b*x >= c
```

SL y f son fracciones del nocional; a es el logaritmo del valor de SL a la referencia de 1 ms; b es elasticidad del stop respecto al horizonte. El dominio de representación es el intervalo de la malla, aproximadamente 1 ns–146,15 años. La escala de referencia fija unidades: cambiar ms por segundos sin transformar a cambia el modelo.

Si b>0, la restricción genera un mínimo temporal. Si b<0, genera un máximo. Si b=0, admite el intervalo entero o ninguno. En todos los casos debe intersectarse con el dominio representado. La frontera procede de la desigualdad elegida; **la elección q=0,65 no procede de un teorema de rentabilidad**.

### 2.2 Fallos reproducidos y consecuencia

El [cálculo previo, documentado en la ronda II](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_II_2026-09-24.md:178>) devolvía siempre el extremo superior del espectro. Con a=−4, b=−0,1 y f=0,001, la raíz es aproximadamente 57.194.502.976 ms: 1,81 años. El código incluía hasta 146 años, donde el stop ya incumple la política. No se infiere que una posición de esa duración haya sido abierta.

La nueva regresión encontró además una segunda manifestación de la misma causa: para una curva decreciente bajo el piso en todo el espectro, el mínimo infinito se recortaba con min(hi), fabricando una banda [hi,hi]. El test con a=−10,5 y b=−0,1 lo reprodujo. Vacío y singleton tienen significados diferentes: el segundo puede superar un filtro que únicamente pregunta si existe banda.

También se reproducen la sustitución de pendientes no nulas menores que 1e−12 por curvas planas, la admisión de un intercepto infinito en una curva plana y la falta de contracción de la banda decreciente al aumentar la comisión. Las siete regresiones rojas pertenecen al mismo contrato; no se cuentan como siete incidentes.

El riesgo atraviesa la validación de candidatos y el reparador de RR: ambos consumen tradeable_band_ms. Un dominio ampliado o ficticio cambia qué extremos validan y qué genomas se reparan. Una cantidad final segura sigue requiriendo otros contratos; la banda no es un dimensionador monetario.

### 2.3 Reparación y fronteras numéricas

[tradeable_band_ms](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:2432>) ahora:

1. Rechaza coeficientes no finitos y un piso no representable positivo.
2. Evalúa los dos extremos en coordenadas logarítmicas.
3. Devuelve intervalo completo o vacío cuando ambas evaluaciones deciden lo mismo.
4. Calcula la raíz únicamente cuando hay cruce dentro del dominio.
5. Devuelve el lado correcto de la raíz según el extremo que satisface la desigualdad.

[min_tradeable_tau_ms](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:2417>) deriva su respuesta de esa misma banda; no mantiene un segundo solucionador. Un mínimo finito no implica autorización para todos los horizontes superiores. Si no hay banda, retorna infinito incluso si la raíz matemática de una curva creciente estaría más allá del dominio.

No se introduce un umbral nuevo para decidir si b es cero. La clasificación por extremos evita dividir innecesariamente por pendientes diminutas y exponenciar raíces externas enormes. Los extremos retornados siguen siendo aproximaciones f64, **no intervalos certificados por redondeo dirigido**. La prueba de pertenencia omite un margen de 1e−12 alrededor de la frontera para separar redondeo de clasificación; ese margen sólo existe en el test, no modifica la política.

Se conserva deliberadamente el fallback de fee cero/negativo/no finito a REFERENCE_ROUNDTRIP_FEE. Es una convención heredada comprobada por test, no una simulación sin comisiones. Tampoco se eliminan los recortes de los lectores de parámetros, ni se valida con esta función que exp(a+b*x) sea realizable como bracket, que haya evidencia o que el exchange lo acepte.

### 2.4 Pruebas y propiedad metamórfica

[Once tests de banda](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/temporal_band_contract.rs>) cubren ambos signos, pendientes ±5e−13, curvas planas, igualdad, raíz externa, coeficientes inválidos y fallback de costes. El barrido contrasta 25 curvas con 513 horizontes cada una; no es una prueba económica.

La transformación SL→k·SL y f→k·f conserva la desigualdad. En coeficientes corresponde a a→a+ln(k), b sin cambios. El test usa k=0,001, 2 y 1.000. Aumentar sólo f debe contraer el conjunto admisible, independientemente del signo de b. Estas relaciones son más exigentes que comprobar únicamente que el resultado sea finito.

## 3. FMT-041 — presupuesto, esperanza y probabilidad de toque no son lo mismo

Se corrigieron la descripción de [q y del piso](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:2389>), el comentario del baseline y el mensaje del [gate de promoción](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome_store.rs:263>). Ya no afirman que fuera de la política ninguna operación pueda tener EV positiva.

El contraejemplo histórico sigue vigente: p=0,4, TP=0,004, SL=0,0001 y f=0,001 dan EV=0,00054 en el modelo binario. Este cálculo refuta la supuesta necesidad universal del piso; no prueba que la probabilidad p sea alcanzable para esas barreras y ese mercado.

Se conserva el chequeo de extremos de EV y se corrige su justificación. Para p fijo en (0,1), costes f>=0 constantes y dos leyes de potencia, definir:

```text
h(x) = ln(p) + a_T + b_T*x
       - ln((1-p)*exp(a_S+b_S*x) + f)
```

La última expresión logarítmica es convexa; h es cóncava. h>=0 en ambos extremos implica h>=0 en el intervalo. Ésta es la justificación de la cobertura continua bajo ese modelo; la monotonía de TP/SL sola no basta porque el RR exigido depende del stop. La derivación completa ya estaba en II y se conserva.

La propiedad deja de estar demostrada si p o f dependen de tau, si hay otras barreras/salidas o si se reemplazan las curvas por splines sin nuevas cotas. Elevar TP generalmente cambia la distribución del tiempo de llegada y la probabilidad de ganar. El reparador geométrico no crea edge manteniendo p por decreto.

**Estado parcial:** no se retira el guard de fricción, no se demuestra una tasa óptima q y no se revisan todos los comentarios de riesgo. El cierre económico requiere registrar rechazos, predicciones causales de payoff, costes y resultados de alternativas comparables. La seguridad no debe relajarse para hacer visible un gen que fue restringido deliberadamente.

## 4. FMT-112 — contrato elegido para el continuo entre nodos

### 4.1 Qué representa cada consulta

En un nodo actualizado, momentum_z es sorpresa relativa dividida por una EWMA de desviación absoluta. No es un z-score gaussiano. La señal se calcula como tanh(clamp(z,−5,5)). El peso es una función heredada de persistencia y ganancia epigenética. La masa nodal es E_i=w_i·|s_i|.

El [interpolante de señal](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:315>) conserva los valores nodales y es lineal por tramos en x=ln(tau/1 ms). Lo mismo ocurre por separado con el momentum. [state_at](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:567>) devuelve una vista de esas consultas, no una EWMA nueva integrada a la escala solicitada. prev_dev=0 tampoco permite tratar esa vista como un checkpoint reanudable.

La convención adoptada preserva el comportamiento existente:

| Campo consultado | Operación | No equivale necesariamente a |
| --- | --- | --- |
| Señal | I[s_i] | tanh(clamp(I[z_i])) |
| Momentum | I[z_i] | Una sorpresa normalizada recalculada desde precio/volatilidad interpolados |
| Masa espectral | I[w_i·|s_i|] | w(I[estado])·|I[s_i]| |
| Estado | Conjunto de observables interpolados | Estado dinámico de un filtro independiente a esa tau |
| Gradiente | Pendiente local de I[s_i] | Derivada temporal del precio o sensibilidad económica de una orden |

### 4.2 Contraejemplos convertidos en contratos ejecutables

Para z0=0 y z1=2, el punto medio logarítmico da I[z]=1, I[tanh(z)]≈0,4820 y tanh(I[z])≈0,7616. Ambas consultas pueden ser útiles; afirmar su igualdad sería incorrecto. El test protege la decisión explícita de no cambiar la señal servida.

Con señales +0,8 y −0,8 y pesos unitarios, el punto medio tiene señal cero y masa 0,8. La masa retiene actividad en los nodos vecinos aunque se cancele la dirección interpolada. El código de [continuous_energy_density](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:514>) no se altera; se corrige su documentación.

**Cierre acotado:** la ambigüedad funcional de FMT-112 se resuelve eligiendo una representación y probándola. No se demuestra que la interpolación elegida sea óptima ni que su masa prediga rentabilidad. No se añade un cálculo decorativo desconectado ni se cambia el campo que consume el núcleo.

### 4.3 Uso descendente que sigue necesitando validación

El núcleo compara masas al arbitrar candidatos, en [lib.rs](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/god-engine-core/src/lib.rs:3732>) y el bloque de candidatos opuestos próximo. Por tanto, documentar que la masa no equivale a confianza no valida el arbitraje económico. Se necesita medir selección, dirección, soporte y retorno neto as-of.

No se editó ese archivo, que ya tenía cambios concurrentes. Tampoco se retiran aquí el veto por entropía, los umbrales de coherencia ni las proyecciones fijas. La entropía entre escalas puede ser máxima con acuerdo direccional perfecto; la corrección de una descripción no acredita que un consumidor use correctamente el descriptor. Son límites de las fichas CES existentes, no nuevas incidencias contadas dos veces.

## 5. FMT-133 — derivada local sustituida por una secante no local

**Estado: reparado localmente; P2.** El [helper](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/temporal_spectrum.rs:540>) decía devolver la derivada respecto a ln(tau), pero calculaba (s(2tau)−s(tau/2))/(2ln2). Ese cociente promedia pendientes cuando el intervalo atraviesa un quiebre. También mezclaba la extensión constante con el primer/último segmento.

**Reproducción:** en una señal nodal 0→1→1, consultar tau=1,1·tau_del_nodo_central cae en el tramo plano, cuya pendiente es cero. La implementación devolvió aproximadamente 0,31108. Un tau inválido produjo −0,54101 en una señal constante de 0,75, porque uno de los extremos se convertía al mínimo por max mientras el otro era neutralizado como inválido. Para f64::MAX, multiplicar por dos podía crear infinito y una secante falsa.

**Corrección:** se busca el intervalo sobre los nodos físicos y se calcula (s[i+1]−s[i])/(ln(tau[i+1])−ln(tau[i])). En un nodo se documenta y retorna la derivada derecha. En el último nodo y fuera del dominio, la extensión constante tiene pendiente cero. No existe una derivada bilateral ordinaria en todos los quiebres; la API ya no afirma lo contrario. NaN, infinitos y horizontes no positivos retornan neutralidad, coherente con las consultas de señal.

Se elige el lado usando los nodos, no floor de un logaritmo redondeado. No se multiplica tau y no se introduce un ancho de diferencias finitas arbitrario. El coste es una búsqueda en 32 nodos más aritmética local; no se midió una distribución de latencia real, por lo que no se publica una mejora de nanosegundos.

**Cinco regresiones rojo→verde:** tramo plano cerca de quiebre, pendiente de un tramo en cinco posiciones, convención derecha, extrapolación y entradas inválidas. Se encuentran en [spectral_interpolation_contract.rs](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/spectral_interpolation_contract.rs>), junto a las dos caracterizaciones de interpolación.

**Alcance:** la búsqueda de consumidores localizó definición/documentación/tests, no llamadas productivas al gradiente. Se corrige un contrato público para uso futuro; no se atribuye una mejora observada del PnL ni de la latencia del motor.

## 6. FMT-134 — un genoma, dos dominios de lectura y diferentes límites de salida

**Estado: abierto; P1 de paridad contractual.** Amplía CES-009 con evidencia ejecutable. No es una afirmación de que todos los backtests usen un lector y producción otro: ambos pueden usar QuantumConfig. El hecho demostrado es que APIs equivalentes en nombre no son equivalentes en resultado.

### 6.1 Evidencia y mecanismos

[SuperGenotype](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/genome.rs:2064>) recorta tau a 30 s–12 h antes de evaluar TP, SL, Kelly, trailing y OBI. [QuantumConfig](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/src/config.rs:403>) evalúa los coeficientes con tau>=1 ns sin ese límite superior. from_genome copia las curvas de TP/SL y reconstruye las demás desde los mismos genes de ancla; igualar coeficientes no iguala la política del lector.

Dentro del dominio compartido también divergen:

| Familia | SuperGenotype | QuantumConfig |
| --- | --- | --- |
| OBI | Salida limitada a [0,05;0,95] | [0,10;0,95] |
| Trailing multiplicador | [0,01;20] | [1,5;6] |
| Trailing activación | [0,01;20] | [1,5;6] |
| Trailing paso | [0,01;20] | [0,5;4] |
| Trailing máximo | No retornado por este método | Derivado de activación·1,5 y limitado a [2;7] |

En [tres testigos diagnósticos](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/crates/quantum-arena/tests/genome_reader_diagnostics.rs>), construidos sin disco ni arena viva:

- La misma curva con pendiente 0,1 coincide dentro de banda, pero difiere en más de un factor e al consultar 1 ns o cien años.
- Un umbral OBI plano de 0,075 se lee como 0,075 en el genoma y 0,10 en la configuración.
- Trailing plano (1;0,2;0,1) se lee como (1,5;1,5;0,5) en la configuración.

Los tests pasan porque verifican esos desacuerdos actuales. Están marcados como caracterizaciones de un defecto abierto, no como invariantes deseadas. Al unificar el contrato deben reemplazarse por tests de paridad.

### 6.2 Impacto y límite de atribución

Un optimizador o diagnóstico que use un lector puede evaluar un fenotipo distinto al del consumidor que usa el otro. Un gen dentro de un tramo saturado tiene sensibilidad nula respecto al valor servido; mutarlo puede no cambiar ninguna acción. Esto es una causa plausible y localizada de falta de impacto del genoma, no una explicación cuantificada de todo el gap backtest/demo/producción.

Hay consumidores vivos de QuantumConfig en riesgo, host, replay y trailing. La existencia de esos consumidores confirma relevancia de esa API; no demuestra por sí sola que la otra ruta haya causado una orden discrepante. Se requiere seguir la misma decisión, genoma, estado, tau y política por ambos entornos.

### 6.3 Criterio de cierre sin ampliar riesgo accidentalmente

Definir un evaluador compartido con tres salidas distinguibles: parámetro matemático bruto, dominio con evidencia y parámetro admisible por política. La política necesita versión e identidad; no debe quedar oculta dentro de cada método. El snapshot de coeficientes y política debe ser coherente con una sola generación, enlazando FMT-070/FMT-042.

Conservar límites vigentes hasta tener una migración que explique qué barrera reemplaza cada uno. Unificar haciendo que ambos lectores extrapolen sin límites podría reabrir brackets excesivos. Unificar copiando siempre el clamp de 30 s–12 h conservaría la limitación que el usuario quiere superar. La solución necesita cobertura y errores por escala, validación de barreras y trazabilidad del valor efectivo, no solamente una operación textual.

Probar equivalencia entrada→consulta→bracket→cantidad→payload para el mismo snapshot; incluir OBI bajo 0,10, trailing cerca de cada límite, tau fuera de banda, NaN y hot-swap concurrente. Registrar tanto discrepancias como motivos de rechazo. No se implementa aquí una elección de política global sin esa evaluación.

## 7. Diseño científico: representación, evidencia, decisión y ejecución

La siguiente tabla es el mapa diagnóstico raíz–cima de esta ronda, no un grafo extraído ni una medición de sincronía:

| Nivel/nodo | Objeto que debe transmitir | Ruptura relevante | Avance en X |
| --- | --- | --- | --- |
| Raíz: eventos | Precio, tiempo, resolución, secuencia y procedencia | Milisegundos no observan nanosegundos; gaps y arranque | Se conserva el diagnóstico CES-008 |
| Estado multiescala | Observables, masa y cobertura por escala | Confusión entre interpolación y dinámica | Contrato explícito FMT-112 |
| Genoma | Funciones de escala, dominio y versión | Banda errónea y políticas de lectura divergentes | FMT-040 reparado; FMT-134 abierto |
| Decisión | Payoff/objetivo, evidencia y factibilidad | Masa no es confianza ni coste/SL un teorema de EV | Descripciones corregidas; validación pendiente |
| Terminal | Cantidad/fills/costes ligados a la decisión | El presupuesto puede no llegar a Q enviada | FMT-113 no se declara cerrado |
| Feedback a raíz | Outcome íntegro, identidad y versión causal | Atribución, pérdidas de ledger y selección | Ronda IX permanece vigente; sin reparación adicional |

### T33 — Espacio de escalas causal y selección temporal: experimento, no reemplazo activado

Esta propuesta especializa T02/T14/T29; no sustituye esas líneas ni añade un indicador al motor sin consumidor. El objetivo es distinguir una escala característica observable de un centroide que puede estar dominado por memoria de arranque, número de nodos o saturación.

Lindeberg construye representaciones causales y recursivas mediante filtros de primer orden **en cascada**. Bajo elecciones específicas, el núcleo límite posee covariancia para cambios de escala temporal asociados a potencias del parámetro de distribución; los filtros discretos son aproximaciones con condiciones de resolución y demora. Esto no equivale al banco actual de 32 EWMAs independientes. Se verificaron estos requisitos en pasajes de [A time-causal and time-recursive scale-covariant scale-space representation](https://arxiv.org/abs/2202.09209).

La familia incluye el trabajo anterior de [receptores espaciotemporales causales y recursivos](https://arxiv.org/abs/1504.02648), cuyo resumen indexado describe la distribución logarítmica de escalas y sus propiedades de respuesta. Se conserva como referencia de diseño, no como una implementación ya presente. El trabajo cercano [Separable time-causal and time-recursive receptive fields](https://arxiv.org/abs/1504.01502) pertenece a esa misma familia; no se contabiliza como un resultado económico independiente.

La selección basada en extremos sobre escala de **derivadas temporales normalizadas** está desarrollada en [Temporal scale selection in time-causal scale space](https://arxiv.org/abs/1701.05088). Se consultaron pasajes que distinguen normalización por varianza y por norma Lp. El gradiente reparado en X es derivada respecto a ln(tau) de una señal interpolada, no esa derivada temporal del artículo. Confundir ambas cantidades trasladaría una teoría sin implementar su operador.

**Diseño propuesto, inferencia nuestra y todavía no ejecutada:**

1. Elegir una variable observable y causal por canal: logprecio, flujo firmado o spread, con unidades, timestamps y máscara de calidad; no mezclar todos por nombre.
2. Comparar el banco actual con un candidato de cascada a presupuesto de memoria y latencia equivalente. Definir la convención de reconstrucción de entrada entre eventos irregulares antes de discretizar.
3. Declarar por filtro edad, masa observada y rango identificable. Una tau representable sin evidencia sigue siendo prior/extrapolación.
4. Seleccionar escalas por un objetivo explícito y normalizado. Medir respuesta a pulsos, rampas, cambios de régimen, ruido, gaps y reescalado temporal, separando interior del dominio y efectos de truncamiento.
5. Comparar error de predicción y decisión al refinar la malla. El coste de actualizar todas las escalas debe justificarse por reducción de error o utilidad, no por frecuencia nominal.
6. Evaluar fuera de muestra con selección múltiple registrada y costes de ejecución comunes. Rechazar el candidato si el beneficio desaparece al imponer disponibilidad causal, demora y misma capacidad computacional.

No se prometen covariancia exacta para cualquier factor temporal ni invariancia de decisiones financieras. Tampoco se añade una cascada y se la llama cuántica. El trabajo externo fundamenta las preguntas experimentales; no prueba alpha.

### Matemática avanzada y complejidad justificada

El criterio de integración sigue siendo estimando, operador, unidades, supuestos, error, coste y criterio de rechazo. Ecuaciones de difusión, primer paso, control estocástico, firmas de caminos, grafos y métodos cuántico-inspirados ya tienen propuestas anteriores; no se declara una nueva solución a problemas del milenio ni se usan como certificado de trading.

La aspiración multivariante exige más que ampliar el eje tau del precio: definir interacciones entre variables y activos, dependencia, observaciones asíncronas y objetivos. Una ley de potencia de dos parámetros es una familia continua restringida, no cualquier función posible. Bases más ricas exigen regularización e identificación; aumentar dimensión sin soporte amplía el espacio de sobreajuste.

## 8. Evidencia de pruebas, metodología y artefacto

| Comando ejecutado en modo offline | Resultado |
| --- | --- |
| cargo test -p quantum-arena --test temporal_band_contract | 11/11 |
| cargo test -p quantum-arena --test spectral_interpolation_contract | 7/7 |
| cargo test -p quantum-arena --test genome_reader_diagnostics | 3/3, testigos de discrepancia abierta |
| cargo test -p quantum-arena --lib temporal_spectrum::tests | 17/17 |
| cargo test -p quantum-arena --lib genome::tests | 6/6 |
| cargo check --bin god_engine | Correcto, sin ejecutar binario |

Total: 44 tests distintos. Antes del arreglo, el primer grupo tuvo siete fallos y el segundo cinco; el primer comando combinado se detuvo al fallar el segundo binario, y se ejecutó el grupo de banda por separado para obtener su evidencia roja. No se suman ejecuciones repetidas como tests nuevos.

Persisten tres warnings ya conocidos en evolución: latest_ts y mode no usados, y RealWfOutcome.trades no leído. No se modificaron por esta ronda. Hubo espera de bloqueo de la carpeta de compilación al ejecutar comprobaciones; no se interrumpió otro proceso. No es un benchmark de latencia, un backtest económico, una prueba de todos los crates ni una validación del ejecutable desplegado.

La investigación usó las habilidades Firecrawl y Firecrawl Research Index. La CLI no estaba disponible, por lo que se consultó el índice conectado, sin instalación y sin subir código del proyecto. Su influencia material fue exigir causalidad, construcción del operador y normalización antes de proponer T33. Los resúmenes indexados y pasajes consultados se distinguen arriba; no se reclama lectura íntegra de todos los artículos.

El [artefacto estructurado X](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_X_2026-09-24.json>) enlaza estados, pruebas, alcance y hashes para seguimiento local. No se publicó en ningún servicio.

### Manifiesto de código de este corte

| Archivo | Lectura/cambio X | Líneas | SHA-256 (prefijo de 16 dígitos hex) |
| --- | --- | ---: | --- |
| crates/quantum-arena/src/config.rs | Completa; sin editar | 493 | 4AD9B0BD4DAE722B |
| crates/quantum-arena/src/genome.rs | Tramos dirigidos; reparación local | 3372 | FF9308D33C34D9B2 |
| crates/quantum-arena/src/genome_store.rs | Tramo de validación; comentarios/mensaje | 762 | CDA733774807C110 |
| crates/quantum-arena/src/temporal_spectrum.rs | Relectura; helper/documentación | 1178 | D65659D7926D31E04 |
| crates/quantum-arena/tests/temporal_band_contract.rs | Nuevo, leído completo | 170 | D025551D88A17C07 |
| crates/quantum-arena/tests/spectral_interpolation_contract.rs | Nuevo, leído completo | 98 | 23BCA36E72D24EB02 |
| crates/quantum-arena/tests/genome_reader_diagnostics.rs | Nuevo, leído completo | 61 | F84F14C5393E2C50 |

Los hashes de host F2BA92C4283C66E65, replay 736CC1DBB38E21D7 y risk/lib E9883455B6BEBEFA coinciden con el corte protegido anterior. No se mezclaron sus cambios con esta reparación. Los índices y adendas históricas reciben sólo apéndices; el contenido anterior, incluidos los NUL preexistentes del maestro, se conserva.

## 9. Hoja de rehabilitación inmediata

Primero, resolver el contrato de parámetro efectivo de FMT-134: snapshot coherente, política versionada y paridad comprobable. En paralelo lógico, mantener FMT-113 como prioridad de seguridad: la cantidad real debe satisfacer el presupuesto después de redondeos y adaptaciones. No ordenar estas tareas como si una banda correcta ya protegiera dinero.

Después, instrumentar soporte por escala y generación/cadencia de observaciones, completar la causalidad del feedback y del juez evolutivo, y evaluar T33 frente al baseline. Sólo entonces decidir si ampliar el dominio operativo, modificar bases o retirar restricciones heredadas.

La evidencia necesaria para cerrar el proyecto sigue siendo mayor que la de esta ronda. Un reporte detallado y 44 tests aprobados no autorizan a declarar que todos los archivos fueron auditados, que todas las ramas fueron integradas o que el sistema ya es autoevolutivo en producción.

## Adenda de estado posterior — ronda XI, 2026-09-24

El contenido anterior y el artefacto X representan el corte de esta ronda
y se preservan. La [auditoría XI](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XI_2026-09-24.md>) repara localmente FMT-134:
unifica fórmulas de lectura según QuantumConfig, reconstruye curvas
derivadas desde genes autoritativos y convierte los tres testigos de
desacuerdo de X en regresiones de igualdad. Los hashes históricos de X
no deben compararse como si fueran un manifiesto del código posterior;
el estado nuevo consta en el [artefacto XI](</C:/Users/jhona/Documents/Proyectos/Trader Gemini/docs/artifacts/auditoria_fundamentos_XI_2026-09-24.json>).

La paridad determinista no resuelve el snapshot coherente ni la promoción
transaccional: FMT-070/042 siguen abiertos. FMT-113 también permanece
pendiente. XI añade FMT-135/136, reparaciones del constructor y cálculo numérico P²,
y FMT-137, corrección de una afirmación de adaptación temporal no implementada.
No se encontró consumidor operativo de ese auxiliar. T34 es propuesta
experimental, no nuevo módulo conectado ni prueba de ventaja científica.

XI verifica 63 tests distintos (15 nuevos; siete rojo→verde) y cargo check
del host. La cobertura acreditada avanza a 95/289 Rust preexistentes;
no se afirma lectura completa de los restantes 194. Sin publicación Git,
cambio de genomas activos, trading ni despliegue.
