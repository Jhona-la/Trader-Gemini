# RA — Revisión desde la base: contratos, causalidad y composición

## 1. Dictamen y alcance verificable

Fecha: 2026-10-04. Base de investigación: `949d29c1b823b14839be9e17a230a5aa3cd212b6`
(main tras PR27, CL-36…42). Rama propia: `codex/root-audit-2026-10-04`.
Se reutiliza un worktree aislado; no se modifica el checkout operativo compartido.
La serie MW anterior sigue separada en `codex/model-reload-contract`.

La revisión produce **16 expedientes**, no 16 cierres ni 16 descubrimientos
independientes de todo el historial. Cinco familias tienen reparaciones locales
en curso de verificación: I-01/02/03, R-02 y E-05. Los demás conservan estado
abierto. Los antecedentes se enlazan; una reparación vieja correcta no acredita
que todos los contratos vecinos también lo sean.

Participaron tres subagentes con alcances disjuntos: riesgo/admisión,
estimación espectral y genoma/evolución. El agente principal revisa sus evidencias,
trabaja en ingesta, integra y valida. Son revisiones automatizadas, no una
certificación externa. Los contraejemplos aritméticos y las lecturas estáticas
no se presentan como ejecuciones del sistema completo.

El inventario de la base contiene **1427 archivos versionados**. No se afirma
haber auditado semánticamente cada archivo. Tampoco se ejecutaron órdenes,
training, promociones, pruebas contra un exchange ni experimentos de rentabilidad.
Los resultados históricos de otros agentes se identifican como tales.

## 2. Grafo de contratos desde la raíz

```mermaid
flowchart LR
  A[Bytes y reloj de mercado] --> B[Parser: valor y ausencia distintos]
  B --> C[Estado por activo y escala]
  C --> D[Señal fijada antes del resultado]
  D --> E[Decisión y admisión de riesgo]
  E --> F[Envío y reserva efectiva]
  F --> G[Fill, costes y trayectoria de capital]
  G --> H[Evaluación causal del candidato]
  H --> I[Publicación de identidad evaluada]
  I --> C
```

El fallo de una arista puede invalidar un cálculo local correcto: un booleano
inventado contamina flujo; un voto fijado tarde contamina habilidad; una reducción
de leverage aumenta margen; un drawdown actual no representa máximo histórico.
Continuidad temporal y multiactivo exigen contratos entre estas etapas, no sólo
más dimensiones o más ecuaciones.

## 3. Matriz de la ola

| ID | Nivel | Estado del primer corte | Contrato |
|---|---|---|---|
| RA-I-01 | P2 | reparación local, RED/GREEN | entero completo, rango y ausencia de reloj |
| RA-I-02 | P2 | reparación local, RED/GREEN | booleano completo del agresor |
| RA-I-03 | P2 | reparación local, RED/GREEN | profundidad íntegra bajo espaciado y suma finita |
| RA-I-04 | P2 | abierto, latente | capacidad del token bucket frente a crédito temporal |
| RA-R-01 | P1 | abierto, estático/aritmético | techo de cartera después de adaptar leverage |
| RA-R-02 | P2 | reparación local en validación | estado de riesgo desconocido no equivale a calma |
| RA-E-01 | P1 | abierto, estático | identidad del símbolo examinado |
| RA-E-02 | P1 | abierto, estático/aritmético | reconstrucción intrabar condicionada al cierre |
| RA-E-03 | P2 | abierto, estático | latencia evaluada frente a latencia publicada |
| RA-E-04 | P1 | abierto, estático/aritmético | máximo drawdown frente a drawdown actual |
| RA-E-05 | P2 | reparación local en validación | frontera IS/OOS y warmup efectivo |
| RA-S-01 | P1 | abierto, estático/aritmético | voto tardío puntuado contra retorno ya observado |
| RA-S-02 | P1 | abierto conocido, revalidado | predicción emitida frente a predicción recalculada |
| RA-S-03 | P2 | abierto, estático/aritmético | soporte histórico de escalas largas |
| RA-S-04 | P2 | abierto, estático/aritmético | tamaño efectivo de muestra bajo olvido |
| RA-S-05 | P3 | abierto, matemático | salto de TrendRunner frente a afirmación C∞ |

La clasificación P1 indica riesgo para integridad de decisión/evaluación; no
prueba una pérdida monetaria observada. Ningún total de tests equivale al número
de bugs corregidos.

## 4. Ingestión y normalización

### RA-I-01 — Un prefijo numérico no es un timestamp válido

**Fuente:** `crates/data-pipeline/src/parser.rs`, extractores públicos y
`BookTickerEvent::parse_from_json`; consumidor `ws_client.rs:285–439` en la base.

El extractor devolvía cero si no había dígitos y aceptaba prefijos como 12 en
`12x`, `12.5` o `12e3`. La multiplicación decimal no comprobaba overflow.
En el runner RED, 2^64 no se rechazó (no fue un panic aritmético observado);
un offset `usize::MAX` sí provocó panic al cortar el slice. Un espacio válido
tras el separador podía convertirse en timestamp cero. La finitud de los
precios no valida el reloj que acompaña a esos precios.

La reparación comprueba límites del slice, presencia de dígitos, overflow,
ceros iniciales y frontera del token. Acepta los cuatro espacios JSON dentro
del alcance del scanner. Si la clave compacta E está presente pero es inválida,
rechaza el evento: no la trata como ausencia ni recurre a T. El fallback a T
cuando E falta, y el cero legacy cuando ambos faltan, se conservan; la validación
posterior sigue siendo responsable de ese cero.

**Alcance:** se exige un entero sin signo para este campo; no se sostiene que
los exponentes sean inválidos en JSON general. La gramática JSON también admite
fracciones y exponentes, y define los cuatro caracteres de whitespace y los
literales completos ([RFC8259, §§2,3,6](https://www.rfc-editor.org/rfc/rfc8259)).
No se convirtió el scanner en un validador estructural completo: claves
duplicadas, rutas anidadas y formatos no compactos requieren trabajo adicional.

### RA-I-02 — Un agresor no se clasifica mirando una sola letra

**Fuente:** `parser.rs`, AggTradeEvent; `ws_client.rs:443–453` consume el
booleano para actualizar el flujo. Antes, cualquier cosa no iniciada en t era
false; `null`, 0 o un string se aceptaban como un lado válido. Un salto de línea
antes de true se interpretaba como false. Esto es distinto de rechazar ruido:
se fabrica una observación direccional.

El arreglo exige true/false completos y una frontera de token, y reconoce
whitespace JSON. Se conserva la validación previa de precio/cantidad. Relación
con el histórico D-117: la corrección de espacios/tabuladores no cerraba la
validación léxica completa ni CR/LF. No se atribuye al arreglo una tasa de
corrección de trades en producción; no se midió su frecuencia real.

### RA-I-03 — El fast path puede borrar un lado válido del libro

**Fuente:** `DepthEvent::parse_from_json` y `extract_wall_sum`.
Con bids compactos y asks espaciados, el lado compacto daba una suma positiva
y se evitaba el fallback; asks quedaba artificialmente en cero. El caso simétrico
también falla. Además, sumar dos cantidades finitas de 1e308 producía infinito,
aceptado tanto en el fast path como en el fallback.

Ahora se exige reconocer las dos claves compactas para tomar el fast path;
con espaciado mixto se usa el fallback ya existente. Se rechazan agregados no
finitos en ambos caminos. El testigo pasa de (2,0) a (2,3).
**No cierre global de L2:** sigue siendo un agregado de cantidades, no un libro
con secuencia, precios por nivel y reconstrucción snapshot/delta. D-118 y el
contrato completo de L2 no quedan resueltos por este parche.

### RA-I-04 — El crédito temporal puede reutilizarse sin respetar la capacidad

**Fuente:** `crates/data-ingest/src/lib.rs:65–78`. Antecedentes: #47, #186,
#524 y #1490 del informe histórico; el CAS único sí resuelve la carrera antigua,
pero no demuestra la corrección de la ecuación de reposición.

Con capacidad 5, tasa 100 tokens/ms y un milisegundo transcurrido, se añaden 5
tokens; el tiempo consumido se trunca a `floor(5/100)=0`. La siguiente llamada
observa el mismo crédito y repone otra vez. Incluso con tasas bajas, la historia
acumulada mientras estaba lleno no se descarta necesariamente al saturar.
Es un contraejemplo de la fórmula, no una prueba de carga concurrente.

La búsqueda de consumidores sólo encontró definición/tests dentro del proyecto;
no se presenta como el rate limiter operativo del exchange. Sigue abierto:
hay que especificar fracciones, saturación, wrap temporal y concurrencia antes
de elegir otra representación. No se introduce un mutex o un límite arbitrario
sin medir ni se altera el control de red real.

## 5. Riesgo: validez local frente a validez de la composición

### RA-R-01 — Menos leverage puede consumir más margen que el admitido

**Fuentes:** `risk-engine/src/orchestrator.rs:243`, `envio.rs:96`,
`src/bin/god_engine.rs:4203`, `booktick_replay.rs:772` (base investigada).
CL-41 subordina el apalancamiento al validado; CL-41b/c mantienen la reserva
real sin margen fantasma. Esos arreglos no revalidan el techo de cartera para
la misma entrada una vez que cambia su margen.

Contraejemplo aritmético: capital 100, techo de margen 50, reserva 14, leverage
validado 5, nocional 70. El envío exploratorio a 1 necesita 70. Cabe bajo el
disparador 85 y la guarda 95 del margen libre 100, pero excede el techo 50.
El margen efectivo es `M=N/L`: disminuir L reduce proximidad a liquidación,
pero aumenta M. No son el mismo invariante.

Se necesita una admisión de sustitución de reserva —sin contar dos veces la
propia— contra el mismo techo, coherente con la cartera, antes de confirmar el
envío. Permanece abierto; no se cambia el umbral 0,95 para ocultarlo ni se
presenta una comprobación textual del helper como test de composición.

### RA-R-02 — NaN/Inf borran la presión de riesgo

**Fuente:** `risk-engine/src/orchestrator.rs:154–186` en la base.
Los componentes sistémicos no finitos se convertían en cero; el veto exigía
finitud para activarse. También se descartaban flujos espectrales no finitos.
Por tanto, cambiar únicamente p_crash=0,92 por NaN podía pasar de rechazo a
admisión en régimen Range con capital/margen válidos.

La reparación local aplica rechazo conservador ante estado no finito y conserva
la política de datos finitos y el arranque frío válido. Su evidencia ejecutada
se registra en la adenda de validación, no se presume aquí. No se demuestra que
el productor operativo publique actualmente NaN/Inf. Las lecturas atómicas
individuales tampoco constituyen un snapshot coherente de toda la cartera.

## 6. Genoma, examen y atribución del rendimiento

### RA-E-01 — Elegir una cinta no cambia automáticamente la identidad del slot

**Fuentes:** `src/bin/evolution.rs:44,209`, `quantum-arena/src/symbol_registry.rs:84`,
`backtest-engine/src/booktick_replay.rs:418`, `god-engine-core/src/lib.rs:4126`.
El CLI registra BTC antes de procesar --symbol. Registrar después ETH es aditivo;
el replay sigue enviando coin_id=0. En un registro recién iniciado, ETH puede
estar en otro índice mientras la cinta elegida contiene ETH y el núcleo consulta
el modelo/especificaciones del índice BTC. Las mayúsculas de CL-37 no reparan
este desacuerdo de identidad.

**Cierre requerido:** un binding explícito cinta→símbolo→índice→spec→modelo que
recorra el examen; probar al menos dos símbolos sin depender de un registro
global previo. Es estático; no se ejecutó el CLI ni se cargó modelo real.

### RA-E-02 — El examen intrabar inventa un prefijo usando el cierre futuro

**Fuente:** `evolution-engine/src/online_daemon.rs:481–500`, consumidor de examen
alrededor de 1733. Se calcula el precio de cierre con el retorno de la barra y
se generan ocho precios lineales, separados por 2000 ms, que recibe el núcleo.

Con apertura 100 y cierre 108, el primer evento vale 101. Cambiar sólo el cierre
cambia ese evento anterior al cierre, las features y el lado del agresor derivado
de la pendiente. Que omni use prev_ret no hace causal esa reconstrucción.

**Alcance del dictamen:** un puente sintético condicionado a los dos extremos
puede servir como escenario declarado; no acredita el camino intrabar histórico
ni aptitud OOS sobre microestructura observada. Seleccionar candidatos con ese
camino puede premiar regularidad inventada. Se requiere separar escenarios
sintéticos de evidencia de promoción y ejecutar un test de independencia de
prefijos. No se afirma haber medido cuánto PnL espurio produce.

### RA-E-03 — Latencia del genoma examinado no coincide con la publicada

**Fuente:** `evolution-engine/src/random_forest.rs:75` y replant alrededor de 278.
El constructor aplica la mutación, fuerza latency_penalty_ms=25 y conserva el
genoma original para cosecharlo. Replant no reproduce esa sobrescritura.

Un candidato con 50 ms puede examinarse primero a 25, publicarse con 50 y después
examinarse con 50 al replantar. Llamar 25 «peor caso» no lo convierte en una cota.
**Cierre:** hacer explícito si la latencia es gen, perfil de estrés o dato exógeno;
guardar la identidad efectiva del examen y probar constructor/replant/publicación.
No se cambia una cifra por otra ni se acredita una latencia del proveedor.

### RA-E-04 — Recuperar capital borra la penalización del drawdown histórico

**Fuente:** `random_forest.rs:165–187`. La cosecha mantiene peak_capital pero
envía `(peak-capital_actual)/peak` al campo max_drawdown_pct. Falta conservar
el máximo de esa magnitud durante la trayectoria.

Con suficientes cierres para pasar el mínimo, control 13→14 y mutante
13→6,5→15 producen drawdown actual cero para el mutante recuperado, aunque su
máximo observado fue 50 %. Con la utilidad vigente
`ln(Cf/C0) − 4 ln(2)·DD²`, el mutante obtiene aproximadamente 0,1431 si DD=0
y −0,5500 si DD=0,5; el control obtiene 0,0741. Cambia la elección del ganador.
Es aritmética sobre la función declarada, no una simulación de trading.

**Cierre:** acumular máximo drawdown en cada evento de valoración relevante,
definir su reinicio por cohorte y transferir ese mismo observable a la cosecha.
Observar sólo en harvest además puede perder caídas intermedias. Permanece
abierto: añadir un campo sin conectarlo a la valoración no bastaría.

### RA-E-05 — Warmup fijo puede consumir o anular la validación OOS

**Fuente:** `src/bin/evolution.rs:529–548` y guardia del replay alrededor de 278.
El contexto es la cola IS de longitud min(train_len,50000), pero warmup_ticks
se fijaba en 50000 incluso cuando ese prefijo era menor.

Con 10000 ticks y split 70/30, sólo existen 7000 de contexto y 3000 OOS; la
guardia devuelve estadísticas por defecto. Con 60000 hay 42000 de contexto
y se saltan 8000 filas OOS adicionales. Es una frontera incorrecta, no una
demostración de que la estrategia no opera ni de que el OOS sea bueno.

**Corrección local:** derivar rango IS y warmup de la misma longitud efectiva,
sin leer el futuro y sin cambiar el máximo histórico 50000 ni el split.
La adenda registra pruebas del helper productivo y cableado. Esto no certifica
que 50000 observaciones sean suficientes estadísticamente ni cierra RA-E-01.

## 7. Estimación espectral: tiempo, soporte y significado estadístico

### RA-S-01 — Voto fijado tarde, retorno puntuado desde antes del voto

**Fuentes:** `signal-engine/src/skill_motores.rs:156–190` y
`god-engine-core/src/lib.rs:2177`. La maduración en un trade cierra un bloque,
pero el snapshot de votos se rearma en el siguiente depth. Si ese depth llega
dentro del siguiente bloque, su voto se puntúa después contra todo el retorno
desde el cierre anterior, no sólo contra el retorno posterior a su emisión.

Ejemplo: bloque de 68720 ms, apertura 100; depth a 60000 ms, precio 101 y voto
positivo; cierre 101. Se acredita ln(101/100) aunque después de emitir no hubo
retorno. La repetición con movimientos alternantes puede producir habilidad
aparente perfecta. El deduplicado de qo-648 resuelve doble conteo, no el desfase.

**Cierre:** ligar voto, precio de referencia, instante y horizonte de su emisión.
O conservar el voto disponible en el inicio real del bloque, o medir sólo el
intervalo posterior al nuevo voto; son contratos distintos que necesitan pruebas
de retraso de depth. No se arregla cambiando el umbral de significancia.

### RA-S-02 — El score usa una predicción recalculada al vencimiento

**Fuente:** `quantum-arena/src/spectral_tape.rs:551–584`.
**Antecedente:** el informe forense ya pedía congelar predicción y climatología
al emitir (alrededor de línea 13383 en la base). Se revalida; no es hallazgo nuevo.

Pending conserva features/persistencia, pero al vencer se calcula de nuevo la
predicción con coeficientes actuales. Con ventanas desplazadas h/4, actualizaciones
intermedias pueden contener parte del mismo intervalo. Un ejemplo RLS con x=1,
coeficiente emitido 0, P=1, lambda=0,75 y una actualización con residual 1 lleva
el coeficiente a 4/7. Para objetivo 1 se puntúa error 9/49 en lugar del error
1 de la predicción realmente emitida.

**Cierre:** guardar y puntuar ambos pronósticos emitidos, con identidad de
modelo, antes de actualizar; probar que una actualización intermedia no cambia
el error atribuido a una predicción pendiente. No basta con puntuar antes de la
actualización de ESA muestra si otras ya cambiaron el modelo.

### RA-S-03 — Tener coordenada temporal no equivale a tener historia en ella

**Fuentes:** gate de observabilidad en `skill_motores.rs:247`, composición en
`voto_espectral.rs:132` y su consumo en el núcleo. El gate excluye escalas por
debajo de resolución, pero no mide historia disponible para escalas muy largas.
La composición normaliza cada columna de escala independientemente.

El contraejemplo 100@0→101@1000 ms da masa diminuta en la escala 31 (≈146 años),
pero puede dar momentum_z≈1 y voto solitón≈0,7616. Trece votos de 0,8 en otras
escalas y 0,9 allí hacen dominante a esa escala aunque sólo haya un segundo
de historia. No se demuestra una decisión ejecutada: qo-652 y otras compuertas
pueden impedir ese horizonte. El defecto está en la evidencia que declara la
composición antes de esos consumidores.

CL-35 sí corrigió masa en otra ruta. Multiplicar TODOS los pesos de una columna
por el mismo soporte no corrige ésta: el factor se cancela al dividir por su suma.
**Cierre:** propagar soporte/incertidumbre como magnitud no cancelable y probar
maduración entre escalas sin convertirlas en etiquetas scalping/swing.

### RA-S-04 — Muestra acumulada no es muestra efectiva de momentos con olvido

**Fuente:** `temporal_spectrum.rs:449–456,604`. Los momentos usan alpha=1/64;
la significancia sigue leyendo skill_n acumulado. Qo-648 corrigió esta clase
de error en SkillMotores, no en este selector.

Para pesos geométricos q=63/64 e independencia,
`Neff=(sum w)^2/sum(w^2) → (1+q)/(1−q)=127`.
El contador puede llegar a 5000 mientras la memoria informativa no crece así.
En el contraejemplo del acumulador, la magnitud llamada IC es ≈0,0604: supera
un umbral ≈0,0283 con n=5000, pero no ≈0,1796 con 127.

La independencia es un supuesto favorable, no acreditado; autocorrelación,
selección entre escalas y normalización no centrada exigen análisis adicional.
El nombre IC y un corte t≥2 no constituyen por sí solos un test estadístico
calibrado. **Cierre:** pesos/Neff verificables y validación bajo dependencia;
no reemplazar el contador por otro número sin explicar la derivación.

### RA-S-05 — TrendRunner no es C∞ en la frontera implementada

**Fuente:** `signal-engine/src/trend_runner.rs:102`, afirmación alrededor de 215.
Se define h=tanh(max(H−0,5,0)/0,04), pero se retorna cero mientras h≤10^-4.
La frontera H*=0,5+0,04·atanh(10^-4)≈0,5000040000000133 tiene límite derecho
positivo para desplazamiento/TP no nulos y valor izquierdo cero. No es C0 allí,
luego tampoco C∞. Hay además un corte explícito de desplazamiento cerca de cero.

El salto es pequeño y su impacto económico no está medido. La prioridad P3
refleja sobreafirmación matemática y discontinuidad, no una catástrofe demostrada.
**Cierre:** describir la función real o justificar y validar otra familia;
no sustituir automáticamente cada condicional por tanh.

## 8. Revisión adversarial, antecedentes y límites de cierre

La revisión independiente del primer parche I-03 encontró una regresión:
`{"bids":[["10","1e308"] ],"asks" : [["11","1e308"]]}` era rechazado.
El scanner rápido atravesaba el cierre espaciado, sumaba ambos lados y la
guardia nueva validaba ese agregado provisional antes de elegir el fallback.
El refinamiento elige primero la ruta y sólo valida las sumas de la ruta usada.
Se añadió el contraejemplo y su espejo como contrato versionado. Resultado
posterior: 19/19 contratos y 5/5 sondas adicionales, con serde_json 1.0.150.
Las siete pruebas de módulo repetidas dentro de las sondas no se suman como
evidencia independiente.

**Residual I-03 explícito:** ambos encabezados compactos con cierres `] ]`
siguen pudiendo confundir el scanner por su búsqueda de `]]`. Este defecto
estructural previo queda abierto; se cerraron el lado perdido por encabezados
mixtos y el agregado infinito, no todos los formatos posibles de profundidad.
No se atribuye latencia nanosegundo a este parche: no se hizo benchmark.

Antecedentes de los expedientes espectrales, para evitar contar reaperturas como
hallazgos inéditos:

| Expediente | Antecedente y diferencia comprobada |
|---|---|
| S-01 | qo-648/H6 describe cierre y rearme siguiente; RA añade el contraejemplo del movimiento ya observado antes del snapshot. |
| S-02 | Codex Ola 10, FORENSIC alrededor de 13383, ya exige congelar predicción y climatología al emitir. Mismo defecto abierto. |
| S-03 | CL-35 protege la masa espectral; RA identifica otra ruta en la composición de votos donde no se propaga esa protección. |
| S-04 | qo-648/H5 ya corrige muestra efectiva en SkillMotores; falta propagación a TemporalSpectrum. |
| S-05 | AGY-AUD-P22 trató la frontera anterior H≈0,52; RA identifica el salto residual cerca de 0,500004. |

R-02 rechaza estados no finitos de todos los slots, incluso sin posición y en
la dirección aparentemente favorable. Es una política conservadora de nueva
admisión ante corrupción/desconocimiento, no la estimación de presión=100 %.
Los ceros de arranque válidos siguen admitidos. No convierte lecturas atómicas
separadas en una transacción coherente; tampoco añade diagnóstico tipado del
motivo de rechazo. Ambos son límites pendientes. Los clamps finitos y los
umbrales legacy se conservan: mantener compatibilidad no certifica su teoría.

## 9. Criterio matemático para una evolución útil

Un dominio temporal continuo puede aproximarse con una malla finita y refinable.
Eso no acredita información observada a 1 ns ni a 100 años. Cada activo/escala
necesita unidades, soporte temporal, disponibilidad causal, incertidumbre,
coste computacional y error de aproximación explícitos. Interpolar una función
continua y disponer de evidencia a todas sus escalas son propiedades distintas.

Los siguientes cálculos tienen funciones distintas y no deben confundirse:

- **Neff:** estima cuánta información conservan unos pesos. La fórmula geométrica
  127 presupone independencia; dependencia serial y selección requieren ajuste.
- **R² prequential:** compara errores de predicciones fijadas antes de conocer
  la etiqueta con un baseline igualmente causal. Recalcular la predicción a
  posteriori cambia el objeto que se evalúa, aunque la ecuación SSE sea correcta.
- **Máximo drawdown:** mide la peor caída desde un pico en la trayectoria, no
  sólo el déficit actual. Recuperarse no borra el riesgo sufrido durante selección.
- **Margen=N/L:** reducir leverage L reduce exposición financiada por unidad de
  margen, pero aumenta el margen necesario para un nocional N fijo. Debe
  readmitirse la reserva resultante contra el mismo techo de cartera.
- **Warmup IS:** aporta estado pasado; su longitud exacta determina qué filas
  quedan excluidas del score OOS. No puede incluir filas futuras por un literal.

La agenda de investigación debe priorizar estimación multiescala causal,
calibración bajo dependencia, incertidumbre y control de cartera compuesto.
Una nueva teoría necesita un estimando definido, hipótesis comprobables,
baseline, prueba de ablación y validación OOS neta de costes. El nombre de un
problema del milenio, una analogía física o la etiqueta «cuántico» no sustituye
esa correspondencia ni prueba superioridad. Esta ola no añade teoría ornamental.

La meta de duplicar en 72 horas exige multiplicar por `2^(t/3)` tras t días.
Es una especificación aspiracional del usuario, no un rendimiento demostrado,
una garantía ni un criterio para relajar vetos de solvencia o integridad.
Esta revisión no midió rentabilidad, capacidad, ruina ni costes de ejecución.

## 10. Recibos de pruebas: no mezclar runners ni snapshots

| Corte | Ejecución | Resultado | Qué acredita / qué no |
|---|---|---|---|
| Parser inicial RED | módulo real, harness aislado, serde_json 1.0.151 | 8 pasan / 10 fallan, 0,51 s | Reproduce contratos rotos; no runtime completo. |
| Parser primer GREEN | mismo harness | 18/0, 0,82 s | No detectaba todavía el contraejemplo de la revisión cruzada. |
| Parser revisión | serde_json 1.0.150, igual al lock raíz | 18/0 y sondas 4/1 | Detecta regresión en la validación de la ruta descartada. |
| Parser refinado | serde_json 1.0.150, lock temporal | 19/0, 0,01 s; 5/0 sondas adicionales | Cierra contraejemplos versionados; no JSON completo ni benchmark. |
| Riesgo RED/GREEN | rustc, código real de riesgo con arena sintético | 22/4 → 26/0 | Incluye 7 nuevos + 7 cartera + 12 régimen de capital. No sustituye arena real. |
| OOS RED/GREEN | rustc, helper de producción std-only | 2/4 → 6/0, GREEN 0,90 s | Prefijo exacto y OOS íntegro; no acredita PnL ni todo evolution. |
| Workspace intermedio | check --workspace --all-targets --locked --offline -j 2 | exit 0, 8m24s | Hubo ediciones durante la compilación: no es recibo final del candidato. |
| MW anterior recuperado | sesión antigua, rama MW | exit 0, 41m04s | Sólo aquella serie; no se imputa como prueba de RA. |

Tiempos de tests excluyen compilación salvo filas que dicen check. El lock raíz
no se modificó. La primera resolución aislada de serde_json 1.0.151 se conserva
en el historial de evidencia y se revalidó con 1.0.150, no se oculta.

El workflow añade tres contratos seleccionados sin retirar los anteriores ni
debilitar umbrales. Las ejecuciones Cargo con crates reales y el check final de
integración se anotarán en una adenda con su resultado exacto. No se ejecutó
el T-1 completo, paridad operacional ni una suite completa del workspace.

## 11. Inventario Git y reconciliación, por objetos inmutables

Una primera enumeración coincidió con la eliminación/integración de una rama
GLM. Ese resultado se descartó y se repitió fijando OIDs y base. Las diferencias
no se calculan contra nombres que pueden moverse durante la enumeración.
Snapshot de inventario: main `89d0d3b63bda0f55cf3c29c1c572150e3c7d3e82`.

| Referencia al corte | OID | Commits exclusivos frente al main del corte | Disposición |
|---|---|---:|---|
| backup-before-cleanup | 26afeb794811c456cfd12c4659218ccc2d5c972b | 3 | Preservar backup; no equivalencia demostrada. |
| codex/model-reload-contract | 3139f444a2dcfd7b8ec1aca2aebb54c48ec46ff6 | 4 | MW separado, publicación no autorizada en este permiso RA. |
| codex/root-audit-2026-10-04 | 949d29c1b823b14839be9e17a230a5aa3cd212b6 | 0 antes de estos commits | Rama activa de esta ola. |
| feat/quant-sr-codex-horizonte | 6209704acc77bbc0c088cfd206fd82118aaed3d0 | 4 | Preservar correcciones TH y documentación. |
| main | 89d0d3b63bda0f55cf3c29c1c572150e3c7d3e82 | 0 | Rama principal protegida del borrado. |
| qoder/ola54-d0-distribucion | 949d29c1b823b14839be9e17a230a5aa3cd212b6 | 0 en aquel momento | Worktree activo; no borrar por mera ancestralidad. |
| v7-unificacion-wip | 48421129a9f03a101b1ace0e1789e35aff402148 | 1 | Preservar WIP. |
| origin/claude/auditoria-deslizamiento-apalancamiento-sqtc08 | 6f02388e822401884e15368b6348ee4e260e4019 | 1 | Commit documental posterior al head de PR27; no borrar. |
| origin/main | mismo OID main del corte | 0 | Referencia principal. |

`origin/HEAD` es simbólica y no se cuenta como rama de trabajo. El head integrado
de PR27 fue `15fb393ab07908f12214d09995e2d2b582dae64c`; la rama Claude se reutilizó
y avanzó a 6f02388e. Una PR merged no autoriza a eliminar commits añadidos después.
En este corte no había otra rama inactiva y enteramente integrada elegible para
borrado. No se borró ninguna rama en esta ronda.

PR24 GO y PR25 MP ya están integradas; los recibos históricos se preservan en sus
informes. CI del main inicial949: run37194885331 SUCCESS. Ese éxito no cubre RA.
GLM dejó revisión favorable local de MW y pidió T-1/paridad al integrarlo; MW no
se incorpora a esta ola para simular una certificación no ejecutada.

Durante la revisión main avanzó a f9fbcbd5e y luego a 0580e267a. El segundo
incluye qo-654 (momentos EWMA de D0 por moneda), además de votes_export tau-matched
y documentación de GLM. El inventario anterior conserva su fecha lógica y no
se presenta como estado final. La reconciliación exige comparar ambos padres
y compilar todos los targets antes de cerrar el merge en nuestra rama.

## 12. Cobertura y siguiente orden de trabajo

Rutas examinadas semánticamente en esta ola (no certificación de todo su contenido):
parser y validación/data-pipeline, ws_client/data-ingest, token bucket/data-ingest;
orchestrator/envio/risk-engine; tramos de entrada/reserva/publicación en
god-engine-core y booktick_replay; CLI evolution y registro de activos;
evolution-engine/random_forest y online_daemon; skill_motores, temporal_spectrum,
spectral_tape, voto_espectral y trend_runner; contratos nuevos, workflow y
antecedentes documentales citados. El inventario de los demás archivos no es
equivalente a lectura semántica. Las líneas citadas son de la base de investigación.

Orden recomendado por dependencia:

1. Integridad de admisión compuesta R-01 y explicación tipada del rechazo.
2. Identidad E-01, fijación causal S-01/S-02 e interpretación E-02 antes de
   optimizar resultados: sin estos contratos, una mejor fitness puede ser falsa.
3. Trayectoria máxima E-04 y equivalencia de genoma evaluado/publicado E-03.
4. Soporte temporal S-03 y estadística bajo olvido S-04, con pruebas de prefijo,
   maduración y ruido nulo; después, evaluar teorías adicionales por ablación.
5. Residuales de estructura del parser, crédito temporal y continuidad declarada.

Los cinco parches locales no resuelven esos once expedientes abiertos ni el
residual estructural de I-03. No se declara autonomía evolutiva certificada,
ausencia total de bugs, auditoría completa de 1427 archivos ni rentabilidad.

## 13. Publicación autorizada y verificación analítica del informe

El usuario autorizó expresamente publicar **RA** y abrir su PR pública;
no amplió ese permiso a MW. Se publicó el head inicial `ad3f09a5` y se abrió
[PR28](https://github.com/Jhona-la/Trader-Gemini/pull/28) como **borrador**.
CI del candidato: [run37209982889](https://github.com/Jhona-la/Trader-Gemini/actions/runs/37209982889),
observada en curso; whitespace/conflict markers e instalación del compiler
pasaron, all-targets seguía en ejecución. No es un recibo SUCCESS.

La revisión cruzada estática de 44e1d4fd y f7b3830c no confirmó bloqueadores en
su alcance. Comprobó equivalencia de `train_replay` con la conversión anterior,
frontera `i < warmup`, y conservación del comportamiento finito de riesgo.
No es una aprobación GitHub de GLM/Claude ni una certificación del runtime.
Se pidió revisión mediante el buzón compartido; no se presume acuse.

La ejecución Cargo seleccionada local de los tres contratos **no terminó**:
se interrumpió deliberadamente su primera compilación de dependencias para
liberar el lock del check final. Se identificó y detuvo exclusivamente el
árbol propio cargo19552 y sus rustc, comprobando PID/inicio/padre; ninguna
prueba había llegado a ejecutarse. El exit0xffffffff resultante es interrupción,
no evidencia de regresión funcional ni resultado verde. No se interrumpieron
la paridad ni las compilaciones de otros agentes. CI conserva los tres targets.

La skill `validate-data` se aplicó como QA del informe existente: 16 IDs únicos,
rutas de evidencia existentes y reconciliación 6 P1 +9 P2 +1 P3 =16; 4 parches
acotados +1 parcial +11 abiertos =16. Se recalcularon independientemente los
ejemplos: Neff=127, margen14→70 frente a techo50, y H*=0,500004000000013.
El umbral implementado es `2/sqrt(n−3)` (no `n−2`): 0,02829276035 con5000 y
0,1796053020 con127. La fitness del ejemplo es 0,07410797215 para el control,
0,1431008436 para el mutante con drawdown actual0, y −0,5500463369 con DDmax0,5.
Se mantienen separados aritmética, política heredada y calibración estadística.

Dictamen de comunicación: **compartible con salvedades** como auditoría y
candidato; **no listo para integrar** mientras falten sus controles. No se
convirtió el Markdown solicitado en una app ni se reemplazaron informes previos.

## 14. Segundo corte de ramas y plan concurrente

Referencia remota fijada: `7c4cea4036ee67e8b73809fac7c8763280a2291a`.
Frente a ella, backup conserva3 commits exclusivos; MW4; TH4; WIP1; Claude1;
GLM `glm/lxxxiv-l2v1`1, tanto local como remoto; RA5 en su publicación inicial.
`main` local estaba en f9fbcbd5e, tres commits detrás del remoto y sin exclusivos.
No se movió el main compartido mientras GLM trabaja. Se preservó también el
worktree detached `.t1-regresion`, cuya base7c4 no acredita resultados de RA.
La rama/worktree Qoder de la primera foto ya no aparecía: no fue borrada por Codex.
No había otra rama inactiva completamente integrada para eliminar con seguridad.

El nuevo `PLAN_MAESTRO_SINCRONIZACION.md` (qo-655) es documentación, no resultado
experimental. Se conserva su contenido; sus afirmaciones de certificación y
rentabilidad no se adoptan como pruebas propias. La paridad10/10 que reporta
GLM cubre escenarios/versiones definidos, no todos los comportamientos futuros.
El requisito T-1 del plan para cambios de pipeline vivo se registra como
**pendiente RA**; el envío inicial de PR28 precedió a la lectura de ese plan.
No se marca ready ni se integra omitiendo ese control.

El main0580 añadió qo-654 al core y la exportación tau-matched de GLM. Se
inspeccionaron ambos diffs: los cambios RA no ocupan esos bloques. Qo-655 añade
sólo memoria, coordinación y el plan; no cambia Rust. Los recibos de compilación
y reconciliación final se añaden a continuación una vez observados.

## 15. Recibo local final de integración

Merge de main0580 en RA: `b558816`, después de comparar ambos padres y
`cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j 2`
con **exit0, 11m32s** (tiempo reportado por Cargo, incluye espera del lock).
Merge documental de main7c4: `c57376b`, mismo check **exit0, 8,16s** antes del
commit. Ambos conservaron los cambios de cada padre sin conflicto textual.
Hay advertencias; no se ejecutó cargo fix ni se maquilló CI para quitarlas.

La fuente Rust final queda ligada a c57376b. Las adendas posteriores son sólo
documentación/evidencia. Este recibo acredita compilabilidad de todos los targets,
incluidos los contratos nuevos, **no su ejecución integral**. La compilación
Cargo de tests interrumpida y el T-1 pendiente siguen declarados como tales.

Commits funcionales: parser `0d3dcece`, riesgo `44e1d4fd`, OOS `f7b3830c`,
CI `741d7995`; informe inicial `ad3f09a5`. Sólo la rama RA se publica. El merge
de main hacia RA no significa que RA haya llegado a main. La PR se mantiene
en borrador hasta recibir CI, T-1 y revisión del candidato correspondiente.

## 16. Adenda de sincronización y CI efectivamente ejecutada

Nuevo corte remoto observado: main `856e59ba`, sólo plan/coordinación frente a
7c4cea; GLM publicó localmente su plan `d777175f` sobre su rama de trabajo.
Se integraron ambos en RA por `f10434d6` y `381e5f7d`, con unión de las entradas
en conflicto. Contra el primer padre los merges sólo modifican documentos;
contra el otro se conservan los cambios Rust RA y Qoder. All-targets antes de
cada commit: exit0/1m30s y exit0/15,14s. No se movió el checkout compartido.
GLM sigue activo: su rama no se borra aunque estos dos commits ya estén en RA.

[CI37211025915](https://github.com/Jhona-la/Trader-Gemini/actions/runs/37211025915)
terminó SUCCESS, job111462096917: all-targets y todas sus suites enumeradas
pasaron. El log de los targets nuevos confirma parser19/0, riesgo7/0 y OOS6/0,
ninguna ignorada en esos32 contratos. Este recibo con crates reales supera la
limitación de los runners aislados; **no borra** el historial del Cargo local
interrumpido. La suite rápida T1_measurement9/0 no es T-1 completo.

Revisión automatizada independiente de496f902d contra7c4cea: sin bloqueadores
nuevos en parser0d3d, riesgo44e1, OOSf7b3 y CI741d. Revisó consumidores, fallback
E→T, política finita y frontera warmup. No es aprobación humana; el CLI completo
de evolución, snapshots concurrentes, latencia y operación real no quedaron
certificados. El residual Depth sigue abierto, no se oculta como corrección total.

El T-1 completo local fue iniciado sobre496f902d, con release/locked/offline,
dos jobs, test-threads1 y log en target/ra-t1-2026-10-04.log. En este corte
continúa **compilando dependencias**, sin tests ejecutados ni resultado.
Los merges documentales no cambian su fuente Rust; no se atribuye el T-1 de
Qoder sobre7c4 ni la paridad GLM de PR27 al candidato RA.

Se amplió el [plan compartido](PLAN_MAESTRO_SINCRONIZACION.md) con contratos
G0–G8 y se actualizó la sección Codex del [plan operativo GLM](PLAN_MAESTRO_2026-10-04.md).
Las propuestas de reparto son propuestas, no compromisos aceptados. El buzón
local registra el aviso y la solicitud de revisión; no presupone acuse externo.
No hay merge RA→main todavía, ni eliminación de ramas con commits exclusivos.

## 17. Revisión matemática multiactivo MG: defectos y validaciones separados

Fuente: código496f902d, idéntico en los bloques citados tras los merges
documentales. Revisor delegado sólo lectura, seguido de lectura del publicador,
estimador, consumidor y antecedentes por Codex y recálculo independiente de
los contraejemplos. Evidencia **estática y aritmética**, no prueba de frecuencia
operacional, simulación de pérdidas ni tests Cargo nuevos. Los16 expedientes RA
conservan sus IDs y conteos: los cinco expedientes MG no son cinco bugs nuevos.

### MG-01 — Validación del significado y capacidad de vinculación de Lundberg

**Tipo:** contrato matemático/política heredada; pendiente, no nuevo bug del solver.
`cramer_lundberg.rs:149` define `m=ln(1/ε)/R`, reserva mínima que permite
`exp(−Rm)≤ε` para el proceso lineal y condiciones declaradas. En
`correlation_guard.rs:552` se usa `min(tope_streak,m)` como techo del riesgo
de un evento. Por tanto cambiar una tolerancia de ruina a otra menor puede
relajar el veto: R40, riesgo0,08, ε0,05 produce m0,0748933 y veta; ε0,01
produce m0,1151293 y permite. El cálculo cumple el código pero no demuestra
por sí mismo una cota de exposición de cartera ni ruina compuesta.

El consumidor `risk-engine/src/lib.rs:609` convierte R por leverage máximo
del spec. Con R≤100, ε0,05 y Lmax≥9, m_capital≥0,2696159046; un techo
streak≤0,25 siempre domina. Es un resultado condicionado a esos parámetros:
**tener R medido no significa que Lundberg esté causando rechazos**.
Además, la conversión real de nocional a capital requiere `f=N/C`; leverage
máximo permitido no es automáticamente la exposición realmente reservada.

Antecedente #604 ya reconoce la interpretación heurística del min y mezcla
de horizontes; no se contabiliza de nuevo como descubrimiento. Cierre:
especificar reserva/proceso/exposición y condiciones; demostrar la garantía
anunciada o mantener explícitamente una heurística. Telemetría debe separar
R disponible, techo activo y rechazo atribuible. No se propone eliminar el veto.

### MG-02 — Coeficiente publicado obsoleto tras perder su raíz válida

**Tipo/severidad:** estado y vigencia de evidencia, P2; defecto verificado abierto.
El estimador devuelve None cuando la media deja de ser positiva, faltan
muestras o no hay cruce válido (`cramer_lundberg.rs:93`). El publicador
`god-engine-core/src/lib.rs:3495` escribe R y margen **sólo con Some** y no
invalida las claves con None. El lector `risk-engine/src/lib.rs:617` interpreta
cualquier R positivo retenido como utilizable en la admisión posterior.

Reproducción aritmética de su anillo/bisección:256 cierres alternos
+0,015/−0,01, después256 pérdidas−0,01, recalculando tras cada cierre. La última
raíz publicable es0,78074798165 tras reemplazar50 elementos; al terminar el
estimador devuelve None, pero la escritura condicional conserva esa raíz.
No es un fallo de convergencia ni la deuda de cobertura estadística de R̂:
es una transición de estado `Some→None` mal propagada al consumidor.

Impacto probado: evidencia vigente y evidencia servida divergen. El efecto
monetario no se acredita por este ejemplo; MG01 puede hacer no vinculante
ese término en algunos specs. Cierre: invalidación explícita de R y sus métricas
derivadas, prueba de transición a None y consumo exacto del respaldo vigente;
registrar ventana/versión/tiempo. La estimación de incertidumbre de R es otra deuda.

### MG-03 — La proyección equicorrelacionada no es la matriz del grupo

**Tipo:** validación de aproximación declarada, pendiente; no duplicar ADR-0002.
`dependency_exposure` mide aristas contra la candidata y `rho_efectivo_grupo`
las promedia (`correlation_guard.rs:571`); `calcular_riesgo_grupo:508` aplica
esa únicaρ a todos los términos cruzados. Así, las aristas internas y su
relación con pesos desiguales no se preservan en general.

Contraejemplo PSD:ρCA=ρCB=0,9, ρAB=1, pesos/riesgos[0,122;0,122;0,01],
umbral de grupo0,85, sin inflación de cópula ni R. La equicorrelaciónρ0,9
da0,2470854103; la forma cuadrática con todas las aristas da0,2530375466.
Quedan a lados distintos de0,25. La matriz es PSD (determinante0 y menores
principales no negativos); no se fabricó una correlación imposible.

El ADR-0002 acepta esa proyección. El ejemplo refuta equivalencia general o
conservadurismo universal, **no** la existencia del diseño aceptado. Los riesgos
al stop tampoco son automáticamente desviaciones típicas calibradas. Cierre:
definir estimando/garantía, contrastar matrices completas y pesos desiguales,
documentar error y fallback. Cambiar el modelo necesita experimento propio.

### MG-04 — Coseno sin centrar e interpretación como correlación de riesgo

**Tipo:** contrato entre estimando y consumidor, pendiente; no error de implementación
del IC de #607. `espectral_multiactivo.rs:71` acumula E[a²], E[b²], E[ab]
con olvido1/64, no medias centradas; el cociente es un coseno de retornos.
El lector `risk-engine/src/lib.rs:630` lo usa para apretarρ del riesgo de grupo.

Con256 bloques contemporáneos, a=0,01+0,001·[+,+,−,−] y
b=0,01+0,001·[+,−,+,−] periódicos, el acumulador da IC0,9900773327,
mientras la correlación centrada de ese conjunto es0. La deriva común explica
la discrepancia; una señal de dirección/co-movimiento no es necesariamente
covarianza de fluctuaciones. No se declara que centrar sea siempre la política
correcta ni que la ecuación documentada sea defectuosa.

Cierre: definir qué dependencia necesita el veto, unidades y tratamiento de
deriva/volatilidad; comparar ambos estimandos con esos bloques y retornos
constantes, sin confundir incertidumbre de selección con riesgo de capital.

### MG-05 — Cambio de escala hereda evidencia delτ anterior

**Tipo/severidad:** semántica temporal y falsa atribución de evidencia, P2;
defecto verificado abierto. El publicador `god-engine-core/src/lib.rs:1874`
elige la escala más cercana alτ dominante y escribe `qo_613_rho_tau` sólo
si hay IC finito. Un cambio a escala sin pares maduros no borra ni identifica
el IC publicado antes. El consumidor `risk-engine/src/lib.rs:631` no verifica
escala/edad: ausencia inicial da respaldo, pero ausencia posterior puede dar
el dato de otra escala. Eso contradice el fallback descrito en #651.

Trigger:τA publica0,9 y luego dominaτB sin evidencia. Con grupo ya identificado,
ρ escalar0,2, riesgos[0,14;0,14], R ausente y techo0,25, el respaldo produce
0,2168870674 y permite; el IC retenido produce0,2729102417 y veta. Se demuestra
un rechazo distinto **por el dato de otra escala**, no rentabilidad perdida
ni frecuencia real. El arranque frío anterior a la primera publicación sí funciona.

Cierre: pruebaτA→τB no madura debe recuperar el respaldo; publicar identidad
de escala y vigencia junto con el valor, o invalidarlo explícitamente de manera
compatible con los lectores. Probar tambiénτ inválido, pares que caducan y
retorno aτA. No cambiar deliberadamente la política finita durante esta reparación.

MG02/MG05 se añaden como dependencias de G4/G2 en el plan. Las tres validaciones
matemáticas no se convierten en cambios de riesgo sin especificación y contraste.
No se modificó Rust mientras T-1 certifica el candidato acotado RA.

## 18. Segundo corte de sincronización, cobertura y Git

Esta adenda conserva los cortes previos. Base remota observada main
`c6ce7333f670c021db56b2c987680bbd7ebe8c7a`; GLM completó su merge LXXXIV.
Integración propia `12456048d7c5cbafb23a75a530b3e030d53be913`, padres b6dfb374 y
c6ce7333: conflicto sólo en bitácora, resuelto por unión. Comparaciones de
ambos padres realizadas; `cargo +nightly-2026-06-30 check --workspace
--all-targets --locked --offline -j 2` exit0, 37,88 s. Diff de crates, src,
Cargo, workflow y configuración frente a496f902d vacío. No es RA integrado
en main ni prueba de equivalencia operacional en demo/producción.

### 18.1 Evidencia de coordinación, sin aprobación inferida

GLM `253d0cd2` acusa recibo de firma/enlaces y anuncia calibración L2 fase3,
reconciliación de planes y review PR28. La revisión anunciada aún no consta
como resultado. ADR-0010 LXXXIV declara gate parcial: dirección relativa
mejor, logloss peor; cableado bloqueado. Se mantiene esa distinción en el
plan §13. Sep-14 observado no se vuelve ciego por congelar el ajuste en agosto;
la propuesta separa re-gate histórico de una nueva confirmación no seleccionada.
No se verificaron aquí filas, dependencia estadística o dataset del trainer;
las cifras L2 se atribuyen al recibo GLM, no a una medición independiente RA.

MG02/MG05 se reparan en otra rama propia sobre c6ce; **siguen abiertos en RA**
hasta recibir oráculos y verificar integración. La clave ausente puede recuperar
el global y NaN se normaliza a0 en omniscient-registry: publicar un NaN o borrar
un escalar no demuestra invalidación correcta. El contrato debe cubrir también
los lectores, ámbito, retorno a evidencia válida y transiciones frías.

### 18.2 Censo completo de rutas; revisión semántica incompleta

[Artefacto TSV](audit/RA_COBERTURA_2026-10-04.tsv),
[semántica del recibo](audit/README_RA_COBERTURA_2026-10-04.md): 1435 entradas
en12456048,433 en crates; ruta/modo/tipo/OID y estado inventariado. El propio
censo es posterior al snapshot. No cubre archivos ignorados, datos externos
o modelos en ejecución. No suma búsquedas, tests o lectura de tramos como
1435 archivos auditados ni modifica retroactivamente el inventario1427 original.
La auditoría debe avanzar con recibos por ruta/blob/contrato y consumidores.

### 18.3 Todas las referencias observadas frente a main c6ce

| Referencia / OID congelado | Exclusivos | Decisión y fundamento |
|---|---:|---|
| backup-before-cleanup /26afeb794 | 3 | Conservar: incluye artefactos y cambios antiguos extensos; no certificar equivalencia por comparación del árbol completo. |
| codex/model-reload-contract /3139f444 | 4 | Conservar MW, publicación fuera del permiso RA; evidencia de recarga requiere su propio gate. |
| feat/quant-sr-codex-horizonte /6209704a | 4 | Conservar TH: recibos/horizontes pendientes de reconciliación; no integrados por título. |
| v7-unificacion-wip /48421129 | 1 | Conservar: preservación WIP extensa; integrar sin auditoría puede reintroducir rutas eliminadas después. |
| origin/claude/auditoria-deslizamiento-apalancamiento-sqtc08 /6f02388e | 1 | Conservar adenda de memoria posterior a PR27; PR merged no implica rama actual enteramente integrada. |
| glm/lxxxv-l2fase3 y origin equivalente /253d0cd2 | 1 | Conservar: rama ocupada por GLM en checkout operativo, trabajo activo. |
| codex/root-audit-2026-10-04 /12456048 | 12 | Conservar: PR28 draft, faltan gates; main no contiene RA. |
| origin/codex/root-audit-2026-10-04 /b6dfb374 | 11 | Antes de publicar1245; además faltan dos commits de main en ese corte remoto. |
| codex/evidence-expiry-2026-10-04 /c6ce7333 | 0 | Conservar: worktree ocupado, reparación nueva en curso. |
| main y origin/main /c6ce7333 | 0 | Raíz de comparación, no candidato de limpieza. |
| glm/lxxxiv-l2v1 /eb617f548 | 0 | Retirada local tras ancestry+ausencia de worktree+CAS; remota ya ausente. Commit preservado en main. |

`origin/HEAD` simbólico no es una rama independiente. Censo de todos los refs
presentes después del fetch, no prueba de ausencia de ramas en repositorios
ajenos. Los OIDs no cambian aunque otros agentes muevan sus nombres después.
No se borró ningún worktree, archivo, dataset o commit. El nombre local GLM
puede recrearse desde eb617f548 si fuera necesario.

### 18.4 Gates que permanecen abiertos

CI37221376454 del head b6df estaba en ejecución en el corte leído: all-targets
y publicación/carga pasaron; contratos RA y suites posteriores todavía no
finalizaban. CI37211025915 SUCCESS32/0 sigue siendo recibo de otro candidato.
El T1 completo local29055 continúa compilando dependencias de release, sin
resultado de tests. No se cancela ni se sustituye por fastT1, por el T1 Qoder
o por paridad GLM. El nuevo head documental exige su CI correspondiente.
Review humana de RA no recibida en GitHub. Se mantiene draft y no se fuerza merge.

## 19. Recibo posterior de revisión GLM y gates no equivalentes

Main5ab07f3589a1ab273794bf373b40b9f120a5b438 integró GLM LXXXVec7b314d.
Sólo cambian bitácora, plan operativo y ADR-0010; no Rust/CI/CLI frente a c6ce.
Las entradas del nuevo ciclo se conservan junto a las propias por unión.
Los OIDs/tablas anteriores son cortes históricos, no nombres que se actualizan
retroactivamente. El censo TSV sigue fijado a12456048, no al último main.

### 19.1 Qué revisión llegó

La entrada GLM LXXXV FINAL aprueba la **dirección** de scanner y del rechazo
de estados no finitos; reconoce la frontera JSON incompleta y validación de
todos los pares antes del filtro direccional. Exige T1 y paridad al integrar.
Su recibo describe PR28/2050 líneas/735 de informe, alcance funcional igual
a496f; los heads siguientes sólo añadieron documentos. No se infiere revisión
de todas las líneas, del OOS, ni validación económica de los modelos. Es review
cruzada de agente, no review humana GitHub. Se reconoce ahora su recepción sin
borrar los cortes anteriores que todavía la esperaban.

### 19.2 Qué no prueba una suite llamada paridad

`crates/backtest-engine/tests/bt_vivo_parity_audit.rs` contiene diez tests,
dos ignorados de medición manual con tape real. Los restantes fijan knobs,
dirección adversa del slippage, sampler, sensibilidad y determinismo/prefijo.
Algunos contrastan el mismo helper o fórmulas locales: su éxito no demuestra
que host y replay recorran todas las transiciones con idénticas reservas,
timestamps, parseo, llenados, modelo cargado y reintentos. El propio test de
prefijo declara que agregados no demuestran por sí solos ausencia de anticipación.
La CI antigua8/0/2ignored es recibo de esos contratos, no paridad operacional
universal. No activar ignorados cambiando CWD/fixture a ciegas ni contabilizarlos
como ejecutados; una paridad nueva necesita inputs/snapshots y traza observada.

T1 es otra medición: un extremo por coordenada, un fixture BTC, predictor
direccional sintético y ocho estadísticas; no aptitud canónica ni inercia global.
No mide diversidad multiasset ni rentabilidad real. El runner completo continúa
en release; el bloqueo del check documental es contención del directorio de
compilación, no resultado fallido del código ni razón para cancelar otro agente.

### 19.3 Calibración y limpieza selectiva

GLM informa T=2,393: agosto logloss0,6771<0,6812, septiembre0,7078>0,6910.
El gate completo L2v1 queda parcial definitivo y cableado bloqueado. El fallo
de transferencia no se remedia llamando espectro a un régimen; necesita
validación del estimador y del proceso de selección en contextos nuevos.
No se reprodujo aquí su dataset ni se acredita una fuga sólo por esta diferencia.

Retirada local glm/lxxxv-l2fase3, OIDec7b314df7ed223b60367bfe1890dec3d844a965,
ancestro de5ab, cero exclusivos, no ocupada, eliminación CAS. Remota ya ausente
tras fetch/prune; commit recuperable desde main. No se borraron datos/commits.

## 20. Revalidación adversarial de expedientes existentes sobre el nuevo main

Un revisor delegado independiente examinó R-01/E-03/E-04 por objetos Git de
`c6ce7333f670c021db56b2c987680bbd7ebe8c7a`; los merges hasta5ab son documentales.
Codex releyó las anclas de riesgo/envío/reserva/bosque/fitness y recalculó los
ejemplos siguientes. El reporte RA no existe en c6ce: para definir la alegación
se consultó el documento posterior, pero las líneas Rust pertenecen al snapshot
fijado. Método estático y aritmético, sin ejecutar host, promociones o backtests.
Son **tres expedientes ya abiertos**, no tres bugs nuevos ni tres reparaciones.

### 20.1 R-01: una reserva contablemente correcta puede violar el techo admitido

La cadena es RiskEngine→PortfolioOrchestrator (`risk-engine/src/lib.rs:1293`)
→reserva del core (`god-engine-core/src/lib.rs:7262,7365`)→adaptación del host
(`src/bin/god_engine.rs:4171–4214`)→`EntryReservation::reajustar_margen`
(`entry_reservation.rs:64`)→`Position::reajustar_margen_generation`
(`quantum-arena/src/position.rs:433`). El reajuste comprueba generación,
estado abierto y confirmación; suma la diferencia a used_margin bajo el cerrojo
antes de publicar el margen de la ranura. **No recibe ni verifica el techo**
que el orquestador admitió. La guarda de envío del95% del libre no es ese techo.

Ejemplo: capital100, mínimo nocional5 ⇒ room20 ⇒ peso micro0. Gen colchón0,5,
presión direccional0 y sin posiciones ajenas ⇒ techo50. Reserva14 a5× ⇒ N70.
Envío exploratorio1×, libre sin su propia reserva100 ⇒ margen70: no cruza el
disparador85 ni la guarda95, se acepta por `envio.rs:81–106`. El reajuste
publica70 y contabiliza70, violando el techo50. No necesita intercalación de
hilos; la carrera de snapshot es otro contrato. El replay repite el reajuste
sin readmisión (`booktick_replay.rs:754–777`). No se afirma que todos los gates
anteriores produzcan esta orden en datos reales ni que haya causado un fill.

CL-41/41b/41c sí limitan leverage, corrigen N/L y ordenan la publicación;
esos arreglos no se deshacen. Falta readmitir **el resultado final del sizing**
con la misma restricción de cartera y semántica de exclusión de reserva propia,
o un contrato atómico equivalente, tanto host como replay. Debe haber oráculos
de aceptación/veto, no-mutación al rechazo, generación y posiciones coexistentes.

### 20.2 E-03: el defecto de equivalencia está en el ShadowForest inicial

`evolution-engine/src/random_forest.rs:65–79` aplica mutation al arena y luego
sobrescribe latency_penalty_ms con25. Guarda mutation sin esa sobrescritura.
La cosecha devuelve self.genomes[best_idx] (`:215`) y el host promueve/relee/aplica
ese genoma (`god_engine.rs:4675–4708`). El almacén no sustituye la latencia por
la efectivamente examinada. `replant` (`random_forest.rs:278–279`) aplica la
mutación sin override; por eso el desajuste es **de la construcción inicial**
y depende de si ya hubo replantación. CL-40 fija la política del candidato,
pero no convierte25ms en el genoma almacenado ni en peor caso demostrado.

Con gen latency50ms, examen inicial usa25 y publicación/replantación puede
usar50. El término de difusión dependiente de sqrt(latencia), a ATR constante,
crece por sqrt(2)=1,4142135624. Es prueba de distinta configuración, no de
inversión efectiva del ganador ni de latencia medida del proveedor.

No extender esta alegación al CLI: evolution.rs:317–322 evalúa test_cfg,
conserva el clon ganador, lo evalúa en OOS y promueve best_config. La latencia
no se sustituye por25 en ese recorrido. Cierre: equivalencia del genoma entero
almacenado/evaluado/devuelto antes y después de replantar; si se usa un escenario
de estrés, debe identificarse separado del fenotipo y probarse su dominio.

### 20.3 E-04: recuperación de capital borra el riesgo de selección en el bosque

`random_forest.rs:171–186` actualiza el pico sólo al cosechar y calcula el
drawdown **actual** `(peak−cap)/peak`. No conserva el peor drawdown de la
trayectoria. Lo entrega como max_drawdown_pct a fitness::compute. La función
compartida calcula correctamente lo que recibe, pero no reconstruye historia
perdida: F=ln(Cf/C0)−λ·DD², λ=4ln2=2,7725887222, preferencia heredada de riesgo,
no probabilidad de ruina ni promesa de Kelly. Con30 cierres válidos y crecimiento
OOS positivo, el factor OOS no modifica estos ejemplos:

| Trayectoria | DD usado al final | DD máximo observable | F resultante |
|---|---:|---:|---:|
| Control13→14 | 0 | 0 | 0,0741079722 |
| Mutante13→6,5→15, cosecha actual | 0 | 0,5 omitido | 0,1431008436 |
| Mismo mutante con contrato máximo | 0,5 | 0,5 | −0,5500463369 |

La selección se invierte. Cosechar durante la caída no conserva después su
penalización. Incluso el pico puede perder una subida no observada entre
cosechas; medir cada evento es una cuestión de cobertura de valoración.
No generalizar a todos los backtests: backtest-engine/lib.rs:431–442 y
booktick_replay.rs:577–584 acumulan máximo, y evolution-engine/lib.rs:494–526
lo toma de su equity_curve (sin que eso certifique cobertura completa OOS).
Cierre: observar pico/DD máximo en el contrato de trayectoria de cada universo,
definir reinicio al replantar y comprobar caída/recuperación entre cosechas,
control, mínimo de trades y selección. No medir sólo el déficit de la foto final.

Estos hallazgos explican mecanismos posibles de brecha demo/backtest; aún no
cuantifican frecuencia o pérdidas operacionales. Mantener R-01 en G4 y E-03/E-04
en G3/G7; no atribuir la brecha económica total a estos tres contraejemplos.

## 21. Topología del workspace desde manifiestos, con frontera de interpretación

Se añadió un [artefacto estructural JSON](audit/RA_TOPOLOGIA_DECLARADA_2026-10-04.json)
y su [lectura por niveles](audit/README_RA_COBERTURA_2026-10-04.md#topología-declarada-recibo-estructural-no-grafo-vivo-certificado).
El snapshot12456048 contiene24 paquetes (23 crates y raíz),202 targets y88
aristas internas declaradas. Se cotejaron los24 blobs de manifiestos y todos
los atributos de aristas con `cargo metadata --no-deps --locked --offline`.
El subgrafo de dependencias normales es acíclico; la raíz alcanza23 paquetes.
El paquete independiente flight-recorder no está en ese cierre transitivo.
Esto abre una **verificación de consumidor**, no acredita otro bug.

Se distingue de la topología operacional: dependencia Cargo no es entrega de
evento, target no es motor activo y aciclicidad no demuestra coherencia causal
de una red con realimentación. Las features efectivas, bibliotecas externas,
IPC, canales de registros/arena y calendario de ejecución no se certifican
por este artefacto. Por eso su estado es estructural, no revisión semántica
completa ni prueba de omnisciencia. Para convertirlo en grafo vivo verificable,
G0–G8 exige por cada enlace productor→consumidor: activo, unidad, reloj,
versión/genoma, vigencia, invalidación, rechazo explicable y recibo de entrega.
Esta evidencia no cambia los16 expedientes RA ni cierra R01/E03/E04/MG02/MG05.

## 22. CI posterior terminada: identidad y alcance exacto

La ejecución [37221376454](https://github.com/Jhona-la/Trader-Gemini/actions/runs/37221376454)
terminó SUCCESS el2026-10-04T18:50:22Z. Su head es b6dfb374b95e6d7cda7ca9f39421d5580f241fae,
no0ddb ni el merge documental5ab en preparación. Checkout/log y API de commits
coinciden en el merge probado7c1ae865ea5ba633e97b3d4d60e9386d4be65d56, padres
856e59ba267c150ec98a1a3cbf0ebc3ff3935564 y b6dfb374b95e6d7cda7ca9f39421d5580f241fae.
No atribuir este resultado a una referencia que todavía no se examinó.

El log contiene13 resultados de targets: **187 pasan,0 fallan,3 ignoradas**.
Desglose: publicación/carga1+12+14 (una ignorada); contratos RA19+7+6, cero
ignoradas; inventario9; backtest-lib53; paridad8 con2 manuales ignoradas;
métricas expost21, etiquetas25, riesgo espectral3 y fast-T1 measurement9.
All-targets terminó en verde. El test rápido no ejecuta el T1 completo;
las ignoradas no se contabilizan como ejecución, y los filtros no certifican
todos los targets/pruebas del workspace ni el desempeño operacional.

El merge propio main5ab→RA está resuelto por unión documental y espera el
check local de todos los targets; T1 completo release sigue compilando
sobre Rust496 sin resultado. PR28 permanece draft, sin GitHub review humana.
Hay recibo favorable condicionado de GLM, descrito en§19. Para integrar se
necesita el recibo del candidato publicado posterior, T1 y paridad pertinentes;
no se rebaja ese criterio por observar la CI anterior en verde.

## 23. Revalidación de telemetría: dos implementaciones, seguridad y consumidor

Corte Rust: main `5ab07f3589a1ab273794bf373b40b9f120a5b438`. Se leyeron
completos los dos módulos y sus pruebas; se buscaron referencias en todo el
Rust versionado con `git grep`, y se contrastó la semántica de las primitivas
con documentación oficial. Método estático: **no se ejecutó un interleaving,
Miri/Loom ni un proceso de producción**. No es un fallo introducido por RA.
Revalida D-127/D-237 y el consumidor #80/#101, sin añadir IDs duplicados.

### 23.1 D-127/D-237: la secuencia no proporciona seguridad al payload

`crates/flight-recorder/src/lib.rs`, blob
`2e499f222b7f35ea54270c43b905efd23bcfa3a9`, expone `record_with_time` y
`get_recent_records` como API segura y declara Send/Sync. El cursor y la
secuencia son atómicos; el frame de64 bytes en UnsafeCell no lo es. El writer
usa `write_volatile` después de incrementar la secuencia; el reader copia con
`read_volatile` y sólo después vuelve a verificarla. Si el writer empieza entre
la primera comprobación del reader y su copia, puede superponer accesos al mismo
payload. Descartar una lectura después de la carrera no hace segura esa lectura.
Tras diez spins, el reader incluso copia el frame sin validación de secuencia.

Hay además colisión de productores al reutilizar el mismo slot. El cursor
reparte tickets, no reserva el payload hasta completar su escritura. Con
capacidad1, una intercalación permitida por las operaciones actuales es:

| Paso | Productor/lector | Estado relevante |
|---:|---|---|
| 1 | W1 recibe ticket0/slot0, incrementa secuencia0→1 y se suspende | Slot ocupado, contador impar |
| 2 | W2 recibe ticket1/slot0, incrementa1→2 | No hubo CAS/adquisición exclusiva; contador par con W1 pendiente |
| 3 | W1/W2 escriben, o R ve2 y copia | Payload no atómico con accesos potencialmente superpuestos |
| 4 | Cada writer incrementa al terminar | Un contador final estable no demuestra una escritura indivisible |

La [documentación de read_volatile](https://doc.rust-lang.org/std/ptr/fn.read_volatile.html)
y [write_volatile](https://doc.rust-lang.org/std/ptr/fn.write_volatile.html)
explica que no sincronizan hilos ni vuelven atómicos los accesos; la
[referencia de Rust](https://doc.rust-lang.org/reference/behavior-considered-undefined.html)
clasifica las carreras de datos como comportamiento indefinido. Por ello la
alegación es **seguridad de memoria P1 bajo uso concurrente de la API**, no
sólo precisión de telemetría. Alinear64 bytes o cambiar Relaxed por SeqCst en
el contador no convierte todo el frame en atómico ni excluye a los escritores.
El comentario FIX#946 no acredita cierre del contrato. Las cinco pruebas del
módulo son funcionales de un hilo; no certifican este dominio concurrente.

Cierre necesario antes de cablearlo: una representación/política con accesos
legales al payload y propiedad exclusiva de escritura por slot, más semántica
de sobrecarga, generación del ticket, wrap y lector. Si se conserva una copia
optimista, debe operar sobre almacenamiento válido para concurrencia, no sobre
una copia ordinaria que luego se descarta. Añadir oráculos de productores/lector
solapados y control de overwrite; utilizar un detector/model checker compatible
como evidencia adicional, no como sustituto del argumento de seguridad.
La complejidad/latencia de cualquier alternativa debe medirse. No se aplica aquí
un mutex, un umbral de spins nuevo ni un cambio de hot-path sin ese contrato.

### 23.2 #80/#101: consumidor distinto y evidencia de desconexión acotada

`crates/telemetry-server/src/flight_recorder.rs`, blob
`72f30c7d36f07dd7086bce5d02e40870a69cf177`, tiene otro FlightEvent: timestamp,
trace_id, event_type u8 y47 bytes. El crate anterior usa timestamp_ns,
event_type u16, coin_id, flags y seis f64. **No son formatos intercambiables**.
La implementación mmap sólo tiene head atómico y copia64 bytes al slot; dos
productores separados por capacity tickets pueden escribir el mismo slot
sin exclusión. Tampoco se acredita seguridad multihilo por unsafe Sync/Send.
No se declara probada su frecuencia ni corrupción observada en un mmap vivo.

El núcleo contiene `Option<Arc<telemetry_server::FlightRecorder>>`
(`god-engine-core/src/lib.rs:756`), lo inicia enNone (`:1020`) y llama record
sólo dentro del Some (`:5827`). La búsqueda de todo el Rust versionado no
encontró asignación/constructor del recorder fuera de sus tests. El contrato
de cableado de#80 sigue sin demostración en esa inicialización. La topología
Cargo por sí sola no bastaba para concluir esto; aquí se añadió el consumidor.
No se afirma que un binario externo/no versionado no pueda inicializarlo.

La descripción antigua de#101 como plantilla `add(2,2)` ya **no corresponde
a este blob**: existe una implementación y cinco tests, aunque no dependa de
ella la raíz. Se preserva el expediente histórico y se corrige su interpretación
mediante esta adenda. Tampoco se traslada aquí la vieja alegación de f32:
el payload de ese crate es f64, no necesariamente el de sus otros consumidores.
Los cambios de truncado y capacidad en mmap están
presentes; no son el defecto que acaba de revalidarse. Consolidar propiedad,
formato, reloj y consumidor sin borrar el otro historial; conectar primero un
recorder inseguro no sería una rehabilitación. Estos recibos no aumentan los
16RA ni certifican todos los contratos de los tres archivos del recorrido.

## 24. Segunda revisión cruzada del candidato RA y seguimiento OOS

Un revisor delegado read-only contrastó los objetos de
`0ddb3d4230ff7048b1e1c3392b5098182813880f`, funcionalmente iguales a496f,
y los cambios parser0d3dcece, riesgo44e1d4fd y OOSf7b3830c. No encontró
bloqueadores nuevos **introducidos por esos cambios en su alcance**. Codex
releyó los callers y anclas señalados. El dictamen no es revisión humana
GitHub, ejecución Rust ni aprobación de todos los archivos.

El helper OOS devuelve colaIS+OOS y mide W antes de anexar el futuro. El CLI
crea particiones posicionalmente disjuntas y pasa W al replay; allí `i<W`
suprime entradas y `i>=W` delimita cierres. Recálculo separado: IS/OOS de
7000/3000,42000/18000,50000/1000 y70000/30000 dan W de7000,42000,50000 y50000,
con todas las filas OOS conservadas. Los seis tests productivos del ensamblado
no ejecutan el CLI/motor completo. W es cantidad de filas, no cantidad de
observaciones válidas: el replay descarta precios inválidos antes de actualizar
ATR. Calentamiento suficiente es otro contrato de soporte, no una garantía de W.

Se revalidó el residual Depth ya declarado: con bids/asks compactos y cierres
espaciados, `{"bids":[["10","2"] ],"asks":[["11","3"] ]}` puede producir
(5,3) en vez de(2,3). La prueba del revisor fue una traslación aislada del
scanner, no otro cargo test. Riesgo valida todo par no finito antes de filtrar
dirección; la política finita examinada se conserva. Eso no da un snapshot
transaccional de todas las lecturas atómicas.

### 24.1 RA-OOS-F01: tape de una fila admitido hasta índice inválido

P2, abierto, preexistente. `src/bin/evolution.rs` del candidato, blob
`e8d8052c1294b56b4079cc9182e894ef1d22e85a`, sólo rechaza len0 antes del split.
Con una fila, `train_len=floor(0,7*1)=0`; las particiones vacías se construyen
legalmente, pero el primer ensayo calcula `timestamps[train_len-1]` (`:346`),
y el informe vuelve a hacerlo (`:485,487`). Se produce underflow/índice inválido
en vez de un rechazo de dataset explicado, antes del nuevo helper OOS.
No se ejecutó el binario: lectura y recálculo del índice, sin red/modelos/tapes.
La ruta necesita preflight de particiones/soporte y retorno tipado antes de
ensayar, consultar red o promover, con tests de cero/una/pocas filas, máximo
de contexto y prueba de que un rechazo no publica nada. No imponer aquí un
nuevo mínimo económico sin especificar su soporte.

### 24.2 RA-OOS-F02: duración de contexto incluida en el panel post-OOS

P2 de diseño/medición, abierto, preexistente; no regresión del cálculo puro. Replay blob
`21b08076fa93abfcea3a82ef89fa50f51f51bed6`, `booktick_replay.rs:609`, envía
`last.ts-first.ts` de **todo** colaIS+OOS a ex_post_metrics. PnL/trades se
cuentan desde W. Así, el numerador de la actividad OOS y el denominador del
reloj no describen el mismo período cuando el caller interpreta el panel
como OOS exclusivo. CAGR, proxies anualizados, Calmar y turnover/día dependen
de ese span; WR/PF no reciben directamente esta alteración de duración.
El helper arregla la frontera de entrada, no este contrato de reporte.
La búsqueda no encontró consumo de `oos_rep.metrics` en el CLI evolution:
no se acredita aquí que haya alterado su ranking/promoción, que el panel se
muestre mal rotulado actualmente, ni que la función sea incorrecta si se define
deliberadamente como medición del replay completo. Lo abierto es la semántica
del período para reutilizarlo como evidencia OOS exclusiva.

Ejemplo puramente dimensional:10 días de contexto y1 día OOS convierten
turnover/día OOS en1/11 del valor del mismo nocional/capital en el día medido,
y el factor de anualización por trade en1/sqrt(11)≈0,3015113446. Para CAGR
se reduce por11 la tasa log anualizada, **no** necesariamente el porcentaje
CAGR simple por11. No es una rentabilidad observada ni una cota universal.

Cierre: etiquetar/declarar el período evaluado y su frontera temporal, sin
retrotraer a IS la duración económica de OOS; decidir manejo de filas inválidas,
timestamps iguales/desordenados y ausencia de OOS antes de medir. Invariancia
del panel exclusivo ante variar sólo contexto inocuo y pruebas de denominador
cero. OOS≤10 filas retorna stats por defecto (`:278`), no medición válida; las
métricas ex-post por defecto son NaN. Su política de soporte y señal de estado
deben ser explícitas, no presentarse como resultados económicos de esa ventana.

Estos **dos contratos de seguimiento adicionales** (un fallo de entrada y una
validación de diseño/período) no modifican los16 IDs
del corte original ni se presentan como regresiones causadas por RA. Se
registran separados en JSON y en G5/G8; no están reparados o ejecutados.

## 25. Recibo local del merge documental, antes de publicación posterior

Checkout candidato: HEAD0ddb3d4230ff7048b1e1c3392b5098182813880f con
MERGE_HEAD5ab07f3589a1ab273794bf373b40b9f120a5b438. Conflicto de bitácora
resuelto por unión; cambios de GLM en plan/ADR conservados, no sustituidos.
Comparación contra cada padre y fuente496: ninguna diferencia nueva en
Rust/CLI/Cargo/CI. Adendas propias son evidencia/documentación, no parches MG
o ShadowForest ni modificación de los originales auditados.

Comando ejecutado desde RA:

```powershell
$env:CARGO_TARGET_DIR = 'C:\Users\jhona\Documents\Proyectos\Trader Gemini\target'
cargo +nightly-2026-06-30 check --workspace --all-targets --locked --offline -j 2
```

Sesión43677: **exit0, dev3m44s**. Se usó la caché disponible separada del
lock ocupado por las pruebas DEV; check no enlazó ni ejecutó el motor operativo.
El checker propio redundante39632/sesión21251, todavía esperando lock, se
retiró sólo tras validar PID, nombre, hora13:50:25 y comando. No se detuvieron
el compilador MG, el runner del bosque ni T1 release; no es un fallo de tests
ni una modificación de gates. Una compilación verde tampoco certifica runtime.

T1 completo sobre fuente496 terminó compilación release en134m33s y empezó
sus dos tests; todavía sin resultado final en este corte. No cambiar predictor,
fixture, vector o mínimo para obtener verde. El nuevo commit documental deberá
tener su propia CI/revisión de identidad; el recibo b6df permanece histórico.
RA sigue draft hasta sus gates; main→RA no acredita RA→main. MG02/MG05 y
E03/E04 conservan ramas/checkout separados y verificación pendiente.

## 26. Acuse Qoder y segundo merge documental verificado

El siguiente avance de main,3979f58b575264ffb7a6cdb7902240909249fe81,
contiene únicamente coordinación/plan. Se integra sobre RA
03a7a6b436a1b9b6d0e8ca7cfc31175e8feacfe1. Los dos conflictos de texto se
resuelven conservando las entradas de ambas bitácoras, las tres filas de la
tabla§5 y las adendas propias G0–G8. Se mantiene el mapa de funciones de los
tres documentos: contratos de trabajo, estado/ruta crítica y contratos raíz.
La revisión de diferencias contra cada padre no encuentra nueva modificación
Rust/CLI/Cargo/CI: la fuente funcional sigue equivalente a496f902d.

El mismo comando all-targets de§25, desde RA y con caché target separada,
terminó en **exit0,13,96s** (sesión95854). Es check del candidato de merge,
no una ejecución de tests ni autorización para usar sus modelos en vivo.

Qoder declara T‑1 **16/144 en4221s** del tip7c4cea40, regresión828/0 y roster
18MOTOR. Se registra como recibo atribuido a ese agente/corte, no como
reproducción propia de RA. El test propio RA completo sigue ejecutándose
sobre496f; el vector tiene144 genes (`SuperGenotype::DIMENSION`), no una
selección arbitraria de16. El número16 del sello representa sensibilidad
bajo su fixture, no cobertura universal del espacio de genomas ni retorno.
La review GLM preservada mantiene sus condiciones T‑1/paridad. Ninguno de
estos recibos prueba duplicación de capital cada72h o elimina los contratos
abiertos. No se activa motor ni se promueve modelo/genoma.

La CI37229237858 corresponde al head03a7 y su estado se registró en progreso;
el nuevo merge publicado requerirá identidad/CI propias. La unión documental
main→RA no integra todavía RA→main. MG y bosque mantienen su aislamiento,
oráculos RED/GREEN pendientes y sus cambios fuera de esta PR.

### 26.1 Actualización GLM del registro de vetos, sin cambio de política

El avance posterior e3adf74e36aa48c122e83a72d1b7f6c3cbad90bd añade la
bitácora LXXXVI y seis textos causa/datos/responsable de V-RISK-002/005.
Se conserva por unión sobre4c817fc26e4d28d5d4529a2f878ee15d15398f70.
Los consumidores reales contrastados son `risk-engine/src/lib.rs`:629–649
(IC sólo incrementa la dependencia base) y:290–299 (drawdown contra
lerp de cota medida y tolerancia micro0,85). No cambian fórmula, umbral,
clase, estado ni gate; se describen conductas ya presentes.

Su afirmación283/283 antes/después es recibo GLM, no ejecución propia.
La clasificación textual «medido» no implica que cada componente sea
estimado:0,85 y epsilon0,05 son política declarada, no inferencia científica
por llamarlos continuos. La vigencia de IC sigue abierta en MG05; actualizar
la ficha no borra una evidencia caducada ni valida toda la matriz de vetos.

Diff contra ambos padres verificado. Contra fuente496, la única diferencia
Rust son esas seis cadenas; CLI/Cargo/CI permanecen iguales. El all-targets
propio del candidato (sesión58434, comando§25) pasó **exit0,1m52s**. No se
transfiere este check a otros worktrees. RA completo sigue ejecutando T‑1,
sin cambiar su binario/fixture; esta actualización requiere CI de su nuevo
head. Main→RA sigue sin acreditar merge RA→main.

## 27. Revisión matemática del oscilador: densidad, dominio y analogía

Corte b245cdcf, módulo completo `signal-engine/src/quantum_oscillator.rs`,
blob d6c23aff84c9d2579155fa5faa23d424a7145817; voto espectral blob
a170fcce199beae30e312bd6327ee266cf441788. También leídos tests del módulo,
physics_numeric_contract y escritores/consumidores del core. Sin editar Rust,
ejecutar motor ni atribuir estas verificaciones a una suite cargo.

### 27.1 RA-Q-F01: densidad recortada presentada como probabilidad

P2 matemático/API latente, estáticamente confirmado. El helper público
`compute_superposition_probability` (:62–82) evalúa
`sqrt(alpha/pi)*exp(-alpha*x²)`, con alpha restringido a[0,01,10], pero recorta
el valor a[0,1]. La expresión sin ese recorte es una **densidad gaussiana**,
sigma=1/sqrt(2alpha), no una probabilidad de un evento puntual. Una densidad
puede superar1; la probabilidad de un intervalo se obtiene integrándola.
La distinción y fórmula de referencia están en el
[manual de distribución normal de NIST](https://itl.nist.gov/div898/handbook/eda/section3/eda3661.htm).

Para alpha10, la densidad central correcta es1,78412411615277; el helper
devuelve1. Cuadratura Simpson independiente en PowerShell sobre[-4,4],
80000 subintervalos/h=0,0001, integrando min(1,densidad), da0,7631290823
aproximadamente en vez de1. Es recálculo numérico de la expresión, no ejecución
del Rust ni tolerancia certificada de integrador. El recorte central pierde
masa. Además clampa x a[-10,10]: con alpha1, todos los x>10 devuelven el
valor constante2,09882811567721e-44. Sobre todo R esa cola constante positiva
no es integrable; si se declara sólo un dominio finito, tampoco se renormaliza
explícitamente su densidad truncada. Sanitizar NaN a x0 también representa un
centro conocido, no evidencia de una posición observada.

Búsqueda versionada Rust: sólo se encontraron tres invocaciones del helper,
todas en sus tests; ninguno prueba integral, unidades o cola fuera del dominio.
No se demuestra aquí que afecte órdenes, consenso, ranking o rentabilidad.
Los índices de graphify antiguos no son consumidores productivos actuales.
El bug es el contrato matemático de la API bajo su interpretación documentada;
si se desea un score acotado, debe explicarse como score, no como densidad
normalizada/probabilidad de colapso. Cierre: escoger semántica, declarar soporte
y estado inválido, y comprobar masa de intervalos, normalización y fronteras
alpha>pi/|x|>10. No se cambia la política viva por este hallazgo latente.

### 27.2 Paridad de envolvente: válida en su dominio, no idéntica sin límites

La reparación #650 sí iguala la forma exp(-alpha*x²) entre sombra y vivo para
la misma entrada finita |x|≤10 y alpha válido. No se refuta ese arreglo.
La API directa sombra clampa x antes de la envolvente; evaluate_for_coin
clampa x en la fuerza pero conserva pos original en la envolvente (:207–214).
Con misma entrada x12,k1,lambda0,1,alpha0,1, fuerza común-410:
vivo=-0,000228530051400478 y sombra=-0,0186139712026188, razón81,450868665.
Con alpha≤0 también difieren las políticas de respaldo si se invoca la API
directamente. Es contraprueba aritmética de paridad universal, no trade real.

El caller actual core clampa desplazamientos espectrales a[-10,10] (:1908)
y alpha a[0,01,10] (:1923); el escritor vivo clampa pos_dev a[-3,3] (:4733).
Por tanto no se acredita ejecución de ese contraejemplo en tales rutas actuales.
Además momentum_z por banda y desviación EMA/ATR no son la misma observación:
igualar el kernel no obliga a igualar votos de entradas distintas. Seguimiento
de diseño/dominio, no nuevo bloqueador operativo ni segundo bug de trading.
Antes de reutilizar la API fuera del dominio, especificar sanidad común y tests
de equivalencia punto a punto, distinguiendo transferencia espectral de física.

### 27.3 Qué cálculo se implementa realmente

La fuerza-(k*x+4lambda*x³) es la derivada negativa del potencial clásico
V=k*x²/2+lambda*x⁴. Se usa como kernel de reversión acotado y amortiguado;
no se integra aquí una trayectoria cuántica ni se resuelve Hpsi=Epsi.
La [formulación del oscilador armónico de MIT](https://www.ocw.mit.edu/courses/5-61-physical-chemistry-fall-2017/29b4eef1ae44a28c9d2e506f9170caa6_MIT5_61F17_lec8.pdf)
incluye el operador cinético y condiciones de estado. Añadir lambda*x⁴ no
conserva automáticamente su estado fundamental gaussiano: al dividir Hpsi por
una psi gaussiana aparece ese término x⁴, no una energía constante.
Esta última observación es nuestra comprobación algebraica, no una afirmación
del material citado sobre este código. La cabecera «estado fundamental» y
«cero desfase» necesita una definición/prueba o etiquetarse como analogía:
evaluación instantánea sin estado no demuestra filtrado estocástico de fase0.
Sin evidencia OOS/ablación no se atribuye ventaja predictiva a nombres físicos.

RA-Q-F01 se guarda separado de los16 IDs originales; 27.2–27.3 son límites
de dominio/interpretación. Ninguna corrección o medición operacional se declara.

## 28. Revisión cruzada del parche aislado ShadowForest E03/E04

Revisor independiente sólo lectura, worktree forest-evaluation con base5ab07f35.
Parche local de random_forest.rs SHA256
E8C7D240A4562A5EAD4450220C2DF07D9B8F5B743EF49B66272878EB179344AD
idéntico antes/después/cierre; Codex recotejó el hash. Los once archivos de
contexto cotejados por el revisor tampoco cambiaron. No es commit integrado
ni lectura de main; los tests en vuelo todavía no constituyen un GREEN.

No se hallaron bloqueadores nuevos en el alcance leído: constructor aplica y
guarda la misma mutación sin override25ms; replant preserva correspondencia;
núcleo aislado con RecargaGenoma::Fija y promoción del clon guardado sin tocar
latencia. Observe_capital conserva máximo DD entre broadcasts/cosechas, también
al suspender por memoria. Harvest usa una lectura de capital para observación
y fitness. Capital inválido deja NaN/inviable persistente en la época; same-gen
preserva historia y replant reinicia pico/DD/baseline de operaciones.

Lambda, mínimo15, filtro de entorno y promoción permanecen iguales. Se conserva
la diferencia heredada entre trades acumulados usados en fitness y elegibilidad
de15 cerrados **desde replant**. O(1) sin asignaciones por universo, O(n) para
todo broadcast/suspensión; sin benchmark de latencia. Tests leídos abarcan
50/75ms, caída/recuperación, observación entre cosechas, misma/nueva generación,
0/14/15 cierres, no finitos/no positivos, depth/kline y memoria suspendida.
No fueron ejecutados por el revisor. El DD es realizado **observado**, no MTM
ni extremo intraproceso. Fixtures que inyectan capital no acreditan fills.
El control-Inf puede seguir siendo superado bajo política heredada; no se
añadió aquí un criterio de control finito/OOS. Antes de integrar: RED real con
producción anterior y nuevos tests, restauración por hash, GREEN y suite
pertinente. La review estática no reemplaza esos recibos ni una review humana.

## 29. Semántica del multifractal/Hurst y continuidad temporal real

Módulo feature-engine/src/multifractal.rs completo, blob
fe95ca70affc680051b8330dbd9c3ece6fe40b30; consumidores StatefulEngine blob
60aa06f21974f537058e04eebce1bbe85a83fc0b. Corte b245. Se contrastaron tests,
publicación de features, ensamblado train/serve y telemetría del core. Sin
ejecutar Rust, entrenar, alterar modelos o cambiar unidades del vector vigente.

### 29.1 RA-S-F01: proxy de magnitud interpretado como persistencia Hurst

P2 de contrato estadístico/interpretación, abierto y preexistente. Update
toma |ln(price/last_price)|, acumula sumas de magnitudes y sus cuadrados y
devuelve `H_n=0,5+ln(mean_abs/(RMS*0,7978845608))/ln(n)` con floors/clamps.
Fija h_q2=0,5. No estima la escala de incrementos para varios lags, rango
reescalado de sumas con signo, autocorrelación o dependencia temporal. El orden
y signo desaparecen de estos dos momentos de la ventana. Llamarlo Hurst y
afirmar >0,55=tendencia/<0,45=reversión no está respaldado por ese cálculo.

Contraprueba algebraica: retornos +a constantes y retornos ±a alternados
tienen igual mean_abs=RMS=a, pese a trayectorias acumuladas muy distintas.
Con n50 por encima de floors, ambos dan H_n=0,557717286512636. Recálculo en
PowerShell de la fórmula, no resultado de una prueba Rust ni H real asignado
a esas series deterministas. La feature no distingue ese cambio; no se afirma
que todo el sistema pierda dirección, pues otras features sí contienen orden.
La [investigación de la Reserva Federal sobre memoria corta/larga](https://www.federalreserve.gov/pubs/ifdp/2008/956/ifdp956.htm)
describe el rango reescalado mediante sumas temporales centradas y las
limitaciones por correlación de corto plazo. Usar una alternativa validada
requiere especificar proceso/estimador; tampoco basta sustituirlo ciegamente
por R/S para prometer capacidad predictiva.

El efecto de interpretación alcanza datos realmente publicados: StatefulEngine
:771–774 guarda h_mic/h_mes/h_mac en f32; get_spectral_ml_features(:1193)
los devuelve en[3..6]. Core consume ese bloque (:4201,4240,4276) y
train_forest:374–375 lo ensambla en índices globales37,38,39 del vector54.
No se midió que un modelo cargado use esos splits, su peso predictivo, PnL o
promociones afectadas. Compartir helper conserva paridad de valores train/serve,
pero no certifica la semántica Hurst. Cambiar feature viva sin reentrenar el
mismo contrato rompe otra propiedad ya pagada; no se hace aquí.

Cierre: declarar estos valores como proxies de forma/concentración de magnitud
si ésa es la intención, o validar un estimador de dependencia/escala con lags,
soporte, sesgos y estados sin evidencia. Versionar semántica e índices y probar
signo/permutación, IID, dependencia corta/larga y cadencia irregular; retrain y
OOS sólo si cambia el vector. Sin inventar un nuevo umbral de persistencia.

### 29.2 Lo que aún queda fijo y lo que no debe confundirse

MultiScaleHurstConfluence crea ventanas10/25/50 **observaciones**, sin recibir
timestamps. No son horizontes físicos de1ns–100años. La misma ventana puede
cubrir milisegundos o minutos según feed/activo. Comentarios aún dicen
Scalp/Swing, pero eso no prueba que el enum Continuous se haya revertido ni
que haya dos motores de trading separados. Los flags y el score0,4/0,3/0,3
son ignorados por los callers StatefulEngine leídos (_score/_micro_p/_macro_p),
por lo que no se atribuye a esos pesos un veto activo. La migración del tensor
de features a soporte temporal continuo es un contrato pendiente, no un simple
renombrado o destrucción de ventanas útiles de aproximación numérica.

| Salida | Qué calcula este módulo | Interpretación que no está acreditada |
|---|---|---|
| H_micro/meso/macro | Cociente de momentos de magnitud por ventana10/25/50 | Hurst/persistencia direccional medidos o cobertura temporal universal |
| D0 | Ocupación de cajas de1/2/4 observaciones vía pendiente q0, limitada[0,1] | Detector general de clustering de amplitud o dimensión asintótica certificada |
| Ancho f(alpha) | Legendre en q{-2,-1,0,1,2} y tres tamaños, soporte finito | Separación IID/cascada robusta por ancho o prueba de gran desviación asintótica |

D0 ignora amplitud mientras las cajas permanezcan positivas: con n50 y
todos los retornos no nulos, Z0 es50/25/12 y tau0=-1,02944684453 se clampa a1,
independientemente de magnitudes. Eso es **coherente con ocupación**; no es un
nuevo bug por no medir otra cosa. El test de #597 sí distingue huecos EXACTOS
de una serie de soporte lleno y la calibración ya reconoce ancho ruidoso. Una
ráfaga con retornos pequeños positivos entre bloques puede conservar D0=1:
no usar la palabra «racimado real» como certificado general de volatilidad.
El cache1/16, EWMA deD0 1/64 y ventanas por evento necesitan reloj/muestra
efectiva para cualquier futuro gate; repetir un valor cacheado no crea muestra
independiente. No se encontró en esta revisión un consumidor de política deD0;
sus escritores/telemetría #654 no equivalen a un sizing multiactivo ya adaptado.

QuantumKellyRiskEngine fue leído y su ausencia de callers productivos
reconfirmada por búsqueda Rust: sólo pub módulo/tests. La isla #653 ya está
declarada y no se conecta para dar apariencia de grafo completo; duplicaría
el sizing de risk-engine. Tampoco se generaliza RA-S-F01 a RecursiveHurst por
klines: es un cálculo independiente no revalidado por esta contraprueba.

Este seguimiento se registra separado de los16 expedientes históricos. No
certifica la teoría completa ni todos los archivos, y no justifica aumentar
exposición para intentar alcanzar la meta72h.

## 30. MG02/MG05: oráculos efectivos recibidos, reparación todavía aislada

Worktree evidence-expiry, rama codex/evidence-expiry-2026-10-04, basec6ce7333.
El padre leyó el diff completo, helper, los13 tests y controles RED/GREEN
conservados. SHA256 tests idéntico en ambas variantes y fuente final:
797AF948C6E4B82EA385B9DB518956B34FC013E48EDBB2779E253E25718C77E3.
Fuente GREEN lib5D0F90F1659D6FE2974A1444EBF79D62B04766874844541EE8FBF4CBD743DF42;
helper78EAF1C0871415D03C49019D22D413AA623E572B6126A06A001C02FF0E639654.
Informe detallado propio del bloque `docs/AUDITORIA_CADUCIDAD_EVIDENCIAS_2026-10-04.md`
en ese checkout; no se presupone que ya exista en RA/main.

RED efectivo:1pass/12fail/0ignored,exit101. GREEN:13pass/0fail/0ignored,exit0;
Lundberg6pass/0fail/0ignored/122filtered,exit0. Un intento anterior con cinco
errores de mutabilidadE0596 no se cuenta como fallo lógico de la implementación.
El RED extrae la publicación anterior a un helper que escribe sólo Some
finito y conserva el writerIC dentro de depth; no es una suite que existiera
sin ese módulo en el commit base. Misma prueba, variantes del productor y
frontera de publicación, conservadas bajo target/evidence-expiry.

El cambio retira R/margen al perder raíz y IC al cambiar a escala fría/invalidar
tau. Máscaras por moneda permanecen para impedir que el fallback global reactive
la evidencia caducada. R=0 conserva la ausencia que interpreta el lectorR>0;
IC=-1 no endurece base escalar alguna en[-1,1]. IC=0 no sería neutro ante una
base negativa. No cambia fórmula ln(20)/R, conversión de unidades, lector,
estimator, política epsilon ni genoma. Margen0 significa ausencia diagnóstica,
no cota de riesgo nulo; IC=-1 no distingue ausencia de anticorrelación medida.

Pruebas con estimador/registro/lector reales: Some→None→recuperación; R no
finito; A madura→B fría→A; tau inválida; evento trade sin depth; dual directo;
retorno kill-switch; máscaras global/coin/símbolo; separación de arenas;
veto/aprietes efectivos en RiskEngine; compatibilidad finita; lecturas
concurrentes y handoff tras publicación terminada. El padre cotejó los logs,
no reprodujo por segunda vez los13 controles. La revisión independiente del
parche exacto sigue en curso. No se suma un nuevo ID ni se declara MG integrado.

Coherencia por clave/coin con un escritor, no snapshot conjunto
R/margen/IC/tau/capital/posición/orden. Las pruebas permiten mezclar generaciones
entre claves durante publicación. Compartir registry comparte estado; no
revocan órdenes ya validadas. No all-targets final, T1MG, stress, benchmark,
Loom/Miri, modelo cargado ni rendimiento financiero certificado en este corte.
Warnings de entorno genómico no definido/DarkAlpha fallback son del fixture;
no se silencian alterando entorno o fuentes de la prueba genética RA.

### 30.1 Recibo cruzado MG posterior: publicación no es frescura del estimador

Review independiente del hash lib/helper/test de§30, estable antes/después/al
cierre: sin nuevos bloqueadores hallados. No reejecutó cargo. El reader coin→
global y base[-1,1] sostienen la máscara, pero los logs por sí solos no traen
sello automático del OID/hash; el recibo se acompaña de copias RED/GREEN
cotejadas. No se simula aprobación humana o integración final.

Residual MG05: `coherencia_par` no recibe as-of y devuelve el IC acumulado
si n≥madurez/denominador válido; excluir pares nuevos desalineados no caduca
un enlace ya maduro. Un Some de la misma escala puede ser histórico. Este
parche retira None/no finito/tau fría o inválida, no acredita frescura por
tiempo. Subcontrato heredado abierto, no nuevo ID/regresión ni justificación
para TTL inventado. Medir último soporte contemporáneo por par/escala y
definir consumo as-of antes de certificar «evidencia vigente» completa.
Tests13 no incluyen cierre real que pierda R, feedback real que cambie tau,
base escalar negativa directa o coste de publicar más a menudo. Registrar
límites junto al resultado favorable, no ocultarlos tras el número de tests.

### 30.2 Commit local MG tras cotejo de padres y all-targets

MG605a4d7f3c2a138ca7e826b7cb131a096f393f90, padresc6ce7333 y main e3adf74.
Diff frente a ambos padres, fuentesreview intactas y check del candidato
all-targets/locked/offline exit0,2m33s (23272). Exactas rutas propias stageadas;
bitácoras/ADR/fichas descriptivas upstream preservadas por merge. No PR/pushMG
ni integración a main en este corte. Sin T1MG/paridad económica certificada.

## 31. Nuevo seguimiento de reloj aceptado: RA-I-F01

P2 transición de estado, confirmado por lectura de main e3adf74, blob
stateful_engine60aa06f21974f537058e04eebce1bbe85a83fc0b. `try_process_tick`
(:674) valida precio y después publica last_event_ms antes de verificar
volumen, contador y productos/retorno finitos. Un rechazo posterior deja
current_ts y features anteriores, pero adelanta este reloj. Contradice el
contrato público de no mutación ante rechazo y crea dos edades distintas de
un mismo prefijo aceptado. Los tests de transición omitían last_event_ms.

No es sólo telemetría: racha_amortizada(:493), can_open_position_ms(:527) y
get_active_directional_streak_ms consumen ese reloj. En un prefijo válido con
último evento/cierre1100ms, base1000ms y rachas3, un rechazo por volumenNaN
con timestamp9100ms adelanta8bases; floor(log2(1+8))=3 borra pérdidas efectivas
y puede liberar enfriamiento. Es contraprueba algebraica/API, no orden
operacional observada. La ruta process_event sanea volumen no finito a0 antes
del caller; no se afirma que cualquier NaN externo llegue a esta API. El
notional/retorno derivado y callers directos aún requieren su propia frontera.

Reserva disjunta Codex feature-clock, sobre main e3 en un cuarto worktree.
Sólo stateful_engine.rs y stateful_transition_contract.rs; no lib.rsMG ni
random_forest.rs. Snapshot ampliado, tres controles nuevos de reloj/cooldown/
continuación/contador y frontera cero/empate. RED en compilación, sin resultado
todavía; la fuente productiva aún no se movió. Reparación prevista: publicar
reloj después de TODOS los guards, preservar fórmulas/políticas y aceptar los
mismos datos válidos. GREEN, revisión, all-targets y gates de integración
requeridos; sin unir a RA durante T1. Seguimiento separado del inventario16.

### 31.1 Reparación local del reloj: RED efectivo y GREEN efectivo

Misma suite SHA256BB96D9C922FDD1F1DE6F1823F8BC4170FAC24A5C698C384B64858756DD4CF706.
Fuente originalSHA256A41A75456CB241C9AA16C6BF6707F2E222FDB0E1058AD7F4B349C981E03773CB:
RED compiló5m33s,25tests18pass/7fail/0ignored,exit101,0,02s. Fallan volumen
inválido, nocional/retorno/volumen acumulado no finitos, contador agotado,
rechazo futuro seguido de aceptación y reloj/cooldown/rachas futuros. No error
de compilación atribuido a lógica. Fuente corregidaSHA256
5A5940CBEEE5F5EC7984449BBDF86157A6C56F8BA3B840120A1E0DC5ED318A5C:
GREEN25pass/0fail/0ignored,exit0,0,02s tras39,45sbuild. Sólo mueve el max
del reloj después de los guards y precisa «aceptado» en su comentario.

El nuevo informe AUDITORIA_RELOJ_FEATURES_2026-10-04.md está en feature-clock,
no en RA/main todavía. Inputs válidos, EMA/Kalman/ML/fórmulas y errores tipados
conservados. Review independiente, unitarios stateful y all-targets en curso.
No rollback completo del core, orden enviada, PnL ni genoma certificado. RA
mantiene su fuente496 y T1; no copiar este GREEN a sus recibos congelados.

### 31.2 Revisión cruzada posterior y residuales fuera del arreglo

Hashsource5A5940CB y testBB96D9C9 estables antes/después/cierre de revisión
independiente, sin bloqueadores nuevos hallados. No ejecutó cargo ni cotejó
GREEN: esos recibos son del padre. Todos los Err preceden ahora a la primera
mutación en esta función y los guards no consultan last_event_ms; mover su
max conserva inputs válidos. Reloj de ticks aceptados, no tiempo de pared.
Unitarios internos stateful:18pass/0fail/0ignored/145filtered,0,05s, build3m13s.
Check all-targets aún en ejecución en el corte de este recibo.

Residual heredado: last_scalp_exit_ms==0 sigue representando «sin cierre» aunque
timestamp0 sea válido para ticks. El writer de cierre del core asigna el
event_time_ms; un cierre en0 sería indistinguible del estado inicial en los
lectores de cooldown/racha. No cierre real en0 ejecutado ni impacto operativo
observado; requiere presencia tipada o marcador independiente con compatibilidad,
no sustituir0 por un tiempo arbitrario. `record_trade_outcome` mantiene además
otros relojes y bandas; no se afirma que esta reparación los haya unificado.
El wrapper del core descarta Result y continúa OFI/otras mutaciones: rechazo
atómico del whole-core, saneamiento downstream y recuperación de panic pendientes.

### 31.3 Cierre técnico local del reloj, sin integración

Commitf3018b35b2eeb99e0c41d1a842d96012d86558c2, padre maine3adf74.
All-targets exit0,3m25s (27451), fuente/hashreview/GREEN intactos. Incluye
sólo estado, tests, informe propio y memoria de esa serie. No push/PR en este
corte; consentimiento de publicación pública solicitado. No se une a RA para
atribuirle T1 de una implementación distinta. Revisar el candidato conjunto
cuando se integren las series, no sumar certificados aislados como atomicidad.

## 32. Resultado T1 completo propio de RA, identidad y alcance

Sesión29055 finalizó exit0. Comando ejecutado desde RA sin alterar fixture,
predictor, entorno, comparador ni trinquete:

```text
cargo +nightly-2026-06-30 test -p backtest-engine --release --locked --offline -j 2 --test t1_cobertura_genetica -- --test-threads=1 --nocapture
```

Build release134m33s; dos tests completos2pass/0fail/0ignored en4636,44s.
El oráculo reporta16/144=11,1111% perturbaciones con cambio de estadísticas,
por encima del mínimo0,110. Baseline roundtrip changed_slots=[].
Los128 sin cambio bajo ESTE extremo/fixture NO son128 genes universalmente
muertos. Campos alterados por from_vector pueden estar acoplados; ejprobe143
no realiza el valor pedido y cambia otros slots. Las trazas conservan esa
limitación, no se interpreta como sensibilidad independiente de coordenadas.

Fuente funcional congelada496f902d668b9a1914fe27488246b6a56ef1c1a7. Merges
documentales posteriores hasta b245 sólo añaden en Rust seis cadenas de fichas
de veto; el binario release no se reconstruyó para esos textos. Reloj/MG/bosque
no incluidos. Nightly2026-06-30, Windows. TestSHA256
A10C43BA6F881B92CDDB83D96AC4AF8D59E9DB7FDADB1B35B7DB4C1E3FE602CE;
helper de fixture/comparador1D874D34C30E3D7FCDB2A35C28D1C869837F6AB0939C5D9749A4645C73CC71C0.
Ejecutable t1_cobertura_genetica-d2e218a7f6c0d233.exe SHA256
BB837CDB756110F68FF3B1D5707D6A77AF70F60614A6C7DA792782D12642AC7C.
Log local ignorado target/ra-t1-2026-10-04.log SHA256
C6414A989C15F87F31B2A85FC9D543D21EA0341DBF464AE52CCDEB858D4AE9F6.

Predictor sintético direccional base0,5, salida ±3 por feature0, datos de
serie(3000), BTCUSDT, capital inicial1000. El ruido histórico no centrado,
especs de fixture y neutralización del gate de ML siguen intactos. Diagnóstico
separado:235aperturas/cierres,20vetosconsejo,0vetosML, capital1814,81267 y
WR1 en esa serie. Son salidas de fixture, NO consistencia financiera, WRreal,
72horas multiasset ni cuenta13USD. TG_GENOME_ENV sin definir/DarkAlpha fallback
documentados, no alterados para aprobar la prueba.

Este resultado sí elimina el pendiente «T1 RA en ejecución» para la fuente
congelada; no certifica head futuro o nueva política. CI37230580226 de b245
aún en curso en la consulta posterior; review cruzada de agente y revisión
automatizada no equivalen a aprobación humana. PR28 sigue draft y RA no llegó
a main en este corte. Nuevo recibo publicado exige verificar su CI/identidad;
paridad pertinente y los fallos abiertos conservan sus propios gates.

### 32.1 Estado operativo de integración y ramas tras el recibo

Fetch remoto posterior confirma origin/main e3adf74, PR28/headb245 aún draft
con CI37230580226 en progreso. MG605a y reloj f3018 están sólo locales, cada
uno con1commit exclusivo; RA publicada tiene16 exclusivos. Bosque sin commit,
sourcehashE8C7 y pruebas efectivas aún pendientes. MW4, TH4, V7wip1, backup3
y rama remotaClaude1 conservan exclusivos y no se borran/mergean a ciegas.
Worktrees ocupados no se limpian por tener0 commits exclusivos. Ningún merge
abierto quedó en los checkouts propios ya cerrados; no intervención en raíz.
La limpieza anterior sólo retiró2refs locales GLM integradas, commits
recuperables desde main; remotas ya ausentes. No datos/archivos borrados.

## 33. Brecha de CI: compilar un target no ejecutaba sus nuevos contratos

La revisión de los workflows de las series aisladas encontró que
`check --all-targets` incluía sus tests en compilación, pero los comandos
`cargo test` no ejecutaban los targets nuevos de caducidad ni reloj. El
workflow tampoco tenía un paso explícito de la biblioteca del bosque.
Un all-targets aprobado no podía usarse como prueba automática de esos
contratos conductuales. Se añade la ejecución explícita, sin retirar ninguna
regresión previa, cambiar compilador/timeout o activar pruebas ignoradas.

| Serie local | Commit funcional | Adenda de CI/último head | Ejecución que CI deberá realizar |
|---|---|---|---|
| MG02/MG05 | 605a4d7f | 7aadf509 | evidence_expiry_contract: 13 controles GREEN locales |
| Reloj aceptado RA-I-F01 | f3018b35 | 36b0063c | stateful_transition_contract: 25 controles GREEN locales |
| Bosque RA-E03/E04 | cace007d | cace007d | Biblioteca evolution-engine: 73 controles GREEN locales |

MG y reloj conservan byte a byte la fuente/pruebas ya revisadas en las
adendas CI. El padre comprobó la conservación de todas las líneas anteriores
del workflow y la identidad del comando/target; es validación estructural,
no una ejecución de GitHub Actions. El bosque agrega su paso al mismo contrato
heredado. Los tres heads son locales y no están incluidos en RA publicada.

La CI37230580226 de RA/headb245 continúa en ejecución en el último cotejo.
No cancelar esa ejecución por una adenda documental ni atribuirle los
contratos de estas tres implementaciones distintas. Si se publican sus PRs,
se exigirá CI de su candidato y una unión explícita de todos los pasos cuando
se integren, sin reemplazar una serie por la otra.

## 34. E03/E04: control negativo efectivo y cierre técnico local del bosque

### 34.1 Qué resultado llegó y qué se preservó

El worker terminó sus runners offline y entregó los logs del worktree
forest-evaluation. Fuente final E8C7D240 idéntica a la revisión estática§28.
El padre releyó el diff completo, los once tests añadidos, la fuente de
fitness y las aserciones fallidas; cotejó hashes y resultados, sin segunda
ejecución de esos tests. Las cinco funciones antiguas están intactas y el
baseline del archivo es idéntico en 5ab07f35 y main e3adf74e.

| Control ejecutado | Aprobadas | Fallidas | Ignoradas | Exit |
|---|---:|---:|---:|---:|
| GREEN inicial del bosque | 16 | 0 | 0 | 0 |
| Biblioteca inicial | 73 | 0 | 0 | 0 |
| RED: producción baseline + pruebas ampliadas | 7 | 9 | 0 | 101 |
| GREEN del bosque después de restaurar | 16 | 0 | 0 | 0 |
| Biblioteca final | 73 | 0 | 0 | 0 |

RED es conductual: compila y falla en nueve aserciones, incluidas 25≠50 ms,
pérdida/recuperación, broadcasts depth/kline, suspensión de memoria e
invalidez que se ocultaba tras recuperar. Dos controles nuevos de replantación
ya pasaban antes: no se pretende que fallen todos los casos. Primera compilación
97m45s; tests GREEN del bosque1,10s, restaurados0,98s; biblioteca final1,59s.
Test-module SHA256527D991D1BAD59A01181748271059C5784542D5A2DF97C8254D01C15AFD861AA.

### 34.2 Reparación, cálculo y límites económicos

E03 elimina el override25ms y mantiene el valor del genoma aplicado/guardado.
50/75ms son fixtures de contraprueba, no nuevos umbrales. E04 registra pico y
máximo drawdown del capital realizado **observado** después de broadcasts y
en cosecha, incluso cuando se suspende el motor por memoria. En cosecha usa
la misma lectura para observar y puntuar; no hace atómico todo el ciclo.

Para la trayectoria13→6,5→15, el máximo drawdown sigue siendo0,5 al recuperar;
el déficit final es0. Con lambda heredada4ln2, el componente de fitness cambia
de ln(15/13)=0,1431008436 a−0,5500463369. El control13→14 obtiene0,0741079722.
Ese contraejemplo explica para qué se conserva la historia: atribuir al
candidato el riesgo que realmente se observó, no sólo la última foto.

NaN/Inf/capital no positivo dejan historia inviable hasta replant. El mismo
genoma y cambios sólo del número de generación no borran su muestra. Replant
reinicia pico/DD/baseline manteniendo CL40 y el gate de15 operaciones. No
cambia lambda, el proxy OOS, el filtro de entorno ni el almacén/promoción.
Conteo estructural: 8 bytes/universo más cabecera Vec y O(1) por observación;
sin benchmark. Capital/contadores inyectados no son fills, MTM, DD intraevento,
validación OOS del control ni certificación de superioridad de un candidato.

### 34.3 Candidato conjunto propio, no integración en main

El padre incorporó e3 en el checkout propio, comparó ambos padres y ejecutó
all-targets/locked/offline con nightly2026-06-30: exit0 en2m30s, sesión96306.
Commit local `cace007d7f790b4b0d31a8038e1b8d9a81f90e15`, padres5ab07f35
y e3adf74e, sin merge abierto ni cambios propios pendientes. Contra e3 sólo
bosque, su pasoCI, informe y memoria; las adendas de main se conservan contra5ab.

Informe detallado propio: AUDITORIA_TRAYECTORIA_BOSQUE_2026-10-04.md en su rama.
Logs locales ignorados, hashes cotejados después del runner:

- RED:88FE3ADEE9697F2CCD53EB8941CF90D6A8980A013B3EAB42BFADC053F9A96023.
- GREEN restaurado:D3B30A6217C6B9721B0C74FCEA77B38061A7B26066493380BFA7A88552A18ACA.
- Biblioteca final:58694C9555618D6423395F5BF31C32D3C2474B77CFCB0DAE1F7BDB9066E314A3.
- All-targets:6A40DEA9531E7FEB6E3EA75D4468CBB93A58790749431083A5903F5E6DC99155.

Los logs no sellan automáticamente cada fuente/OID; se conserva la distinción
entre control ejecutado por el worker, cotejo del padre y review estática.
Publicación pública del bosque consultada por separado y aún sin respuesta;
CI/T1/paridad pertinentes e integración en main pendientes. RA-T1 fuente496
no incluye esta reparación. Queda cerrado el pendiente de ejecución§28 para
este hash, no el drawdown económico universal ni todos los expedientes RA.

## 35. Reconciliación verificable de refs y PRs, sin borrar exclusivos

Censo posterior 2026-10-04T21:06:21Z: 12 refs no simbólicas, con OID,
divergencia contra origin/main y ocupación de worktree. origin/HEAD queda
excluido como alias; main local/remoto coinciden en e3adf74e. El JSON conserva
cada fila, no sólo un recuento. Ninguna referencia adicional cumple el criterio
de eliminación. Los worktrees propios cerrados no tienen merge abierto;
RA sólo mantiene las adendas documentales por guardar en este corte.

| Familia de refs | Commits exclusivos respecto de main | Decisión al censo |
|---|---:|---|
| RA local/remota, b245 | 16 en cada ref | PR28 abierta/draft; conservar |
| MG local, 7aadf509 | 2 | Tests y CI propios, publicación pendiente |
| Reloj local, 36b0063c | 2 | Tests y CI propios, publicación pendiente |
| Bosque local, cace007d | 1 | Candidato con main incorporado; gates pendientes |
| MW local | 4 | Serie separada no autorizada por RA; conservar |
| TH local | 4 | Rama antigua divergente; revisar antes de integrar |
| V7wip local | 1 | Trabajo exclusivo; no borrar |
| backup local | 3 | Trabajo exclusivo, no evidencia de redundancia |
| Claude remota | 1 | PR27 merged no acredita este commit posterior |

Consulta GitHub y prueba de ancestry local confirman que los commits de
merge de PR23(MR), PR24(GO), PR25(MP), PR26 y PR27(Claude) son ancestros de
origin/main. Fechas/OIDs exactos en JSON; PR28 no está mergeada. Es evidencia
de integración de **esos commits**, no prueba de que cada efecto permanezca
sin cambios posteriores. El exclusivo actual de la rama de Claude impide
borrarla sólo por el estado merged de PR27.

La limpieza anterior sigue limitada a las dos refs locales GLM ya integradas;
no se eliminaron commits, archivos o refs exclusivas en este seguimiento.
Un revisor de sólo lectura analiza los exclusivos legacy para distinguir
anatomía de cambios y equivalencia de parches; no se asume su resultado ni
se hace un merge automático de cientos de commits antiguos faltantes.

QA del corte documental: JSON válido y claves únicas por objeto; las23
propiedades publicadas de b245, incluidos16 hallazgos originales, se conservan.
Enlaces/anclas nuevos y orden de subsecciones verificados. Sólo adendas de
informes/plan/memoria en RA; su Rust/CLI/Cargo/CI siguen intactos. La revisión
de enlaces, el inventario por blob y la compilación no son lectura semántica
de todos los archivos ni certificación matemática/económica del sistema.

## 36. CI b245 final: recibo confirmado, no transferencia al siguiente head

CI [37230580226](https://github.com/Jhona-la/Trader-Gemini/actions/runs/37230580226)
terminó SUCCESS. Job20:02:48Z→21:06:19Z del2026-10-04, con13 resultados
de targets:187aprobadas,0fallidas,3ignoradas. Log/API confirman headb245cdcf
y merge exacto `62c2ea0d7fc0c72595ea237d7ceff702f200dc4b`, padres main
e3adf74e y b245cdcf. Los pendientes de los recibos anteriores son históricos;
éste los resuelve sólo para esa identidad.

Parser19/0, admisión no finita7/0 y contexto OOS6/0, sin ignoradas. Regresión
backtest-lib53/0, paridad8/0/2ignoradas, métricas21/0, etiquetas25/0 y
riesgo espectral3/0. Los contratos de modelos suman1+12+14pases, con una
inspección de inventario real ignorada; inventario estructural9/0 y
medición genética rápida9/0. Todos los pasos de workflow y all-targets
terminaron correctamente. El padre recalculó el agregado desde los13 resultados.

Las tres ignoradas conservan su finalidad: inspección local de modelos reales,
medición de radio en tape real y medición de brecha de la meta con campeón
en tapes reales. No se habilitaron para esta CI ni se cuentan como aprobadas.
La paridad aquí cubre ocho contratos ejecutados, no todas las transiciones
del host, fills reales o rentabilidad; el fastT1 no sustituye al completo§32.

Log local ignorado `target/ra-ci-b245-37230580226.log`, SHA256
14F08965268EA9B1C05F19FB7E306FB5C7FB1574C5BACD0C16E3DA980386DB00.
Nightly2026-06-30, runner windows-2022. El warning del runtime Node de la
acción no es un test fallido; no se alteró el workflow para ocultarlo.

La publicación autorizada de las adendas documentales agrega recibos, no
cambia Rust/CLI/Cargo/CI. Al cambiar el head de PR28 se exigirá la CI nueva:
este SUCCESS no se renombra como prueba de ese head. La fuente funcional
del T1 propio permanece496, con la diferencia descriptiva de seis cadenas
de vetos; MG/reloj/bosque no se incluyen. Publicación de esas tres series
sigue consultada, main aún e3 y merge final de RA condicionado a sus gates.

## 37. Anatomía de exclusivos legacy: recibo local posterior a la publicación

Revisor independiente de sólo lectura, contra main fijo e3adf74e. Las cinco
refs objetivo y la ref de control conservaron sus OIDs durante su análisis.
No consultó el servidor; el fetch/censo§35 es del padre. Se revisaron commits
exclusivos, numstat, nombres de rutas y patch-id; para Claude el diff completo.
Esto **no** es una auditoría semántica de todas esas fuentes ni una ejecución
de sus contratos. No se modificaron refs, índices, archivos o procesos.

| Ref y exclusivos/faltantes | Anatomía informada por el revisor | Decisión de conservación |
|---|---|---|
| backup26afeb79:3/850 | Python, reorganización legacy, fuentes Rust y gran volumen generado/binario | No importar masivamente ni tratar un título de commit como prueba de funcionalidad |
| TH6209704a:4/448 | Helper de horizonte, contratos de posición/envolvente, documentos y merge de limpieza | Revisar contratos contra APIs actuales; equivalencia sólo del componente identificado |
| V7wip48421129:1/780 | 82 rutas, varias capas de ejecución/riesgo/replay/evolución/configuración | WIP transversal; preservar y descomponer antes de integrar |
| MW3139f444:4/52 | Helper/reload, host, tests, workflow y recibos | Serie aparte de RA; preservar sus contratos y políticas pendientes de revisión |
| Claude6f02388e:1/21 | Sólo memoria, +7/−3 líneas; recibo histórico PR27/CI/tests | Reconciliar delta por unión; no sustituir la memoria vigente |

### 37.1 Qué significan los tamaños y qué no prueban

Los numstats son del delta contra padre o fork, no la diferencia indiscriminada
entre main actual y una rama antigua. El conteo grande se hizo sin detección
de renames; los binarios no tienen cifras de líneas de texto. No mide esfuerzo,
bugs ni valor económico.

Backup: ca4dde0b cambia strategies/ml_strategy.py +3/−3; su asunto describe
un alcance mayor que ese diff. 36ec67e0 abarca1704rutas/508binarias y
+2741320/−4149476líneas;26afeb79 abarca16596rutas/3621binarias y
+20899/−7322408líneas. Mezcla generados, target/target_locked, reorganizaciones
y fuentes antiguas/nuevas. Es un archivo histórico, no un paquete listo de
mejoras vigentes. No se han leído millones de líneas ni validado esos binarios.

TH:56a00f46 altera core/risk y añade helper/contrato;f11575d3 agrega diagnóstico
de envolventes;6209704a añade1070líneas en7documentos. Neto fork→tip:
12rutas,+1500/−60. Los tres archivos nuevos de helper/contratos no estaban
bajo esos nombres en main fijo; eso no demuestra ausencia de una reimplementación.

V7:48421129 tiene82rutas,+4108/−1923:59bajo crates,6bins,5configs y otros
artefactos/documentos. Corelib+314/−726, reconciliation.rs+237/0 y replay
+327/−53 ilustran que no es un único cambio aislado de nomenclatura.

MW:4commits,d1cfe0b7/eaabd7a7/e7bd8f78/3139f444; neto11rutas,+974/−59.
Helpermodel_reload.rs+100/0, contrato+253/0 y cambios del host/workflow.
Los nombres nuevos no están en main fijo; sin certificar reload, watcher,
atomicidad o compatibilidad con el inventario de modelos actual.

### 37.2 Equivalencia parcial y frontera de integración

git cherry marca «+» los12exclusivos no-merge. No existe equivalencia completa
demostrada de esos commits; «+» tampoco refuta que parte de su funcionalidad
haya sido subsumida por cambios posteriores. El merge570a7bf3 no entra en ese
conteo. Su delta contra56a00f46 tiene el mismo patch-id estable que01f8ca68,
ancestro de main: c485000ba73965ca531601b8f74269eed408e3db. Esa normalización
de parche acredita la limpieza temporal específica, **no toda la rama TH**.

Claude6f sólo cambia memoria: no contiene código Rust/CLI/CI exclusivo.
Sus cifras históricas de tests/CI son afirmaciones del autor, no verificadas
por este revisor. Sustituir el archivo entero desde esa ref perdería entradas
posteriores Qoder/Codex; debe preservarse el delta por unión y mantener su
corte de versión. Ninguna de estas refs se elimina por este recibo.

La adenda se guarda localmente después de publicar ff99849a. La nueva CI
37235240759 está en curso sobre ese head; no se cancela para publicar metadatos.
No se afirma que este recibo adicional ya esté en GitHub/main. Siguiente paso:
reconciliación semántica por contrato y candidato conjunto, no merge masivo.

## 38. Reinicio sobre main44bc8: evidencia vigente y conflicto preservado

Corte de investigación: 2026-10-04T22:25Z. Esta ronda vuelve a contrastar
fuente y recibos vigentes; **no reinicia ni borra la historia anterior**.
Al comienzo se observó main e3; el fetch y la concurrencia lo llevaron a
44bc8ab98b7d23a44add3c9d0b9b54d34e9f6a1e, padres e3adf74e y7d51ea83.
El delta LXXXVII de GLM se leyó completo:19líneas de bitácora,28 de ADR-0010
y una descripción de V-RISK-006 (+48/−1); ninguna fórmula/política nueva.

GLM documenta un pre-gate negativo de convicción por rango en septiembre y
mantiene bloqueado el cableado L2v1. Se conserva esa conclusión operativa,
pero sus porcentajes/correlaciones son mediciones atribuidas a GLM, no
reproducidas por Codex en esta ronda. Dos cortes no prueban imposibilidad
universal de todos los modelos; tampoco autorizan activar uno ni sustituir
el continuo por motores discretos. Nuevas hipótesis requieren datos/ablaciones
y gates propios, no inferir ventaja de una teoría sofisticada.

PR28 seguía headff998, draft, sin reviews humanos registrados. CI37235240759
ya tenía checkalltargets y contratos de publicación aprobados; al corte
22:21:48Z aún estaba en curso. main avanzó después y GitHub informó conflicto.
El resultado de una CI contra base e3 no debe presentarse como validación
del candidato reconciliado con44bc8.

Se incorporó44bc8 en el checkout RA propio con merge sin commit. El único
conflicto fue la bitácora: se eliminaron marcadores, conservando íntegros los
bloques Codex y GLM por unión. Contra el padre RA, el delta es exactamente
el LXXXVII (+48/−1,3archivos). Contra el padre main permanecen las reparaciones
y adendas RA. El candidato se somete a checkworkspacealltargets antes de
cerrar el merge; el recibo final se añade posteriormente. No se modifica
la fuente compartida ni se decide por encima de una integración ajena.

El censo actualizado sólo ofreció main/originmain como refs ya alcanzadas,
no ramas de trabajo elegibles para borrar. Claude6f, MW, TH, V7, backup,
MG, reloj, bosque y RA retienen exclusivos o ocupación. OOS tiene una nueva
rama propia. Ninguna se borra porque una PR histórica usó el mismo nombre.
Los commits PR23–27 ya probados como ancestros permanecen recuperables;
eso no prueba ausencia de regresiones posteriores.

## 39. Revalidación causal: memoria estadística no equivale a evidencia vigente

### 39.1 MG05 — IC histórico de la misma escala sobrevive a la desincronización

Seguimiento del ID existente MG05, **no otro hallazgo duplicado ni cierre**.
Fuentes main44bc8 y MG7aad. Productor
d28407b00055bfa5b851ef229adfd371293538f8; lector
9c89cf3a45e3a5b2cd16a0b1d476012f7d5a3e68; helper MG
b0d34fe01fe4e08476ed05e8def2996fc4b3d30b.

observar_maduracion rechaza productos entre bloques separados más de0,5τ.
Ese guard regula la incorporación de muestras. No retira/envejece el
acumulado si dejan de existir parejas contemporáneas. coherencia_par consulta
n y momentos, sin recibir as-of. Por tanto, None/cold no significa lo mismo
que un Some maduro pero antiguo. MG corrige el primero y la escala fría,
pero conserva el segundo. Volver A→B fría→A en su test restaura A y no
certifica recencia. Parameter.timestamp registra creación de la clave:
set_value no lo actualiza, por lo que no sirve como timestamp del soporte.

Contraejemplo calculado independientemente, no ejecución Rust/fills:
escala19,τ=274877,906944ms;40pares de cierres separados275000ms con retornos
iguales alternando±0,02. EWMA con olvido1/64 produce n40 y
ea²=eb²=eab=0,00018694927809238183, por tanto IC=1.
A continuación100cierres sólo del activo0, separación final27500000ms,
100,0444171950221τ. Ninguno incorpora una nueva pareja; los momentos/IC siguen
idénticos. El parent cotejó producer/reader y recalculó los momentos.

MG puede republicar ese1. El lector sólo recibe f64: si alcanza la rama con
ρescalar0,6, max(ρ,IC)=1 y aumenta qo_613_aprietes. Es endurecimiento causal
del coeficiente, no demostración de orden rechazada ni pérdida económica.
El módulo aún afirma en su cabecera «cero consumidores de política» aunque
ya existe este lector; esa descripción observacional está desactualizada.

Contrato propuesto: separar archivo estadístico histórico de evidencia
consumible. Devolver procedencia por par/escala: ts de ambos cierres, epoch,
muestras, τ y as-of solicitado. El consumidor debe verificar admisibilidad
de ese soporte, no la hora de republicación. Conservar incertidumbre/ausencia
sin usar −1 como identidad semántica única. No imponer otro TTL universal
ni borrar necesariamente un estimador histórico útil. La elección de soporte
exige coordinación productor→publicador→lector; no se cambió en esta ronda.

### 39.2 RA-SA-F01 — Persistencia de inversión de signo en otro consumidor

P2, abierto, preexistente. Relacionado con FMT-011, que documentaba CMA;
su reparación posterior no elimina este consumidor independiente.
src/bin/evolution.rs calcula score=10000·max(ln(C/C0),−20) y, si
DD>DDmax/3, lo multiplica por d=clamp(exp(−20·exceso),0,01,1).
Selecciona scores descendentes. Para score<0, disminuir d **aumenta**
el score: una penalización de riesgo puede favorecer mayor drawdown.

Recálculo JS independiente del bloque, no backtest:
C0=13,C=11,7,DDmax=0,3; scorebase=−1053,605156578264.
DD0,11:d0,8187307530779817; DD0,20:d0,13533528323661262.
Con igual número suficiente de trades, idéntica supervivencia y la penalización
lineal de pérdida de13000puntos, los scores son−13862,618943292175 frente a
−13142,589952285081: el segundo, más arriesgado, clasifica mejor por
720,028991007094puntos. Ambos DD están bajo0,3; la penalización absoluta no
se activa y no corrige ese orden. Trayectorias algebraicamente admisibles:
13→11,57→11,7 y13→10,4→11,7. No se generaron fills para ellas.

Esto prueba una incoherencia de selección IS y aceptación del SA. No prueba
que se haya promovido ese perdedor: el gate OOS neto y otras penalizaciones
siguen existiendo. La misma fórmula con score positivo sí decrece, por lo
que no debe afirmarse fallo en todo candidato. El nombre «Bayesiana» del
comentario no aporta posterior, likelihood ni estimador que explique el coste.

Corrección propuesta: una transformación cuya puntuación nunca mejore al
aumentar severidad, para ambos signos, con unidad/cero explícitos; mantener
los umbrales actuales mientras se valida esa propiedad. Añadir pruebas de
ranking negativo/positivo/cero, identity sin exceso y extremos finitos;
cotejar contra la utilidad canónica y medir OOS antes de atribuir edge.
No se implementa a ciegas otra constante o teoría ni se declara CMA regresado.

## 40. RA-OOS-F01: cierre estructural local, fuera de PR28

Rama codex/oos-partition-2026-10-04; base5a543831, commit
9a3bb756529002b21f1e85e3571e2ffd9a6fbeac. Toca helper oos_context,
su contrato y CLI evolution, más un informe propio detallado.
RED9/5→GREEN14/0 con las mismas14pruebas; alltargets exit0 en4m28s.
El RED extrae el guard len0 del CLI al helper real; **no es ejecución
integral del CLI sin cambios** ni un error de compilación.
El informe de la rama conserva hashes, logs, límites y explicación formal.

El invariante 0<train_len<total_len protege train_len−1,train_len y la
resta de longitudes. Se conserva70/30, máximo300000 y el comportamiento
válido; preflight antes de modelo/FRED/trials/promote, después de mmap/setup.
Dos filas admitidas estructuralmente no acreditan aprendizaje. Revisión
independiente estática0nuevosblockers; no repitió pruebas ni Cargo.

Sin ejecución del CLI ni prueba de efectos externos con fixture de una fila;
sin T1 nuevo, OOS económico, red, promoción o trading. Tampoco cambia la
política histórica de return/exit0 para errores de datos. RA-OOS-F02,
calidad del tape y soporte estadístico continúan abiertos. El arreglo no
está en PR28 ni main y no hereda automáticamente sus sellos.

### 40.1 Cobertura de esta ronda y criterio de preparación

Dictamen: **Necesita revisión; parcial**. Alcance acotado: particiones,
vigencia del IC, signo del fitness SA, identidad de integración y naturaleza
de la meta. Inventario de1435archivos anterior no equivale a semántica completa.
Las tablas de cobertura se añaden desde un registro de unidades/evidencias;
sus denominadores son de esta ronda, no número de archivos, consultas ni tests.
No se certifica omnisciencia, toda teoría, todas las escalas o crecimiento72h.

### 40.2 Calidad de la auditoría y robustez: cinco unidades, no el proyecto entero

«Defectos observados» cuenta unidades con defecto pendiente / inventario
acotado aplicable. 0/N no significa N aprobaciones ni cobertura exhaustiva.
Las categorías se solapan y no deben sumarse en un porcentaje de calidad.

Calidad y utilidad del artefacto:

| Categoría | Defectos observados | Evaluación |
| --- | --- | --- |
| Utilidad/completitud | 0 / 5 | Se contestan cinco preguntas acotadas; evidencia económica72h sigue incompleta. |
| Claridad analítica | 0 / 5 | Se distinguen cálculo, reparación local, historia e integración; no certificación total. |
| Visual/interacción | N/A | MD/JSON; sin interfaz ni aceptación visual renderizada. |

Corrección analítica y robustez:

| Categoría | Defectos observados | Evaluación |
| --- | --- | --- |
| Autoridad/fuentes | 0 / 5 | Fuente/blobs/recibos identificados; meta sin medición nueva. |
| Valores/cálculos | 1 / 3 | Inversión SA demostrada; IC recalculado algebraicamente, no Rust/fills. |
| Acuerdo de gráficos | N/A | No se añadieron gráficos cuantitativos. |
| Procedencia | 0 / 3 | Tres trazas con fórmulas, hashes y límites; no todos los consumidores. |
| Consistencia entre artefactos | 0 / 2 | Historia JSON preservada y estados RA/OOS separados; control final necesario. |
| Calidad de datos | 1 / 3 | IC sin as-of abierto; particiones arregladas localmente; soporte económico distinto. |
| Conclusiones | 0 / 5 | Afirmaciones acotadas; medición de la meta incompleta, no éxito observado. |

Registro reproducible propio: target/ra-review-coverage-2026-10-04-2225.json.
El helper de cobertura confirmó su contabilidad, no su verdad o exhaustividad.

## 41. Recibo de verificación y nuevo avance documental de main

Check del candidato RA con padre44bc8 terminó exit0 en2m49s, antes del cierre
del merge. Logtarget/ra-main44bc8-check-2026-10-04.log, SHA256
099AF3FC960167ECFD9854527C2EB54EBD20945C8180AF12BDB2411F5A1446E9.
Las adendas posteriores sólo modifican documentación/JSON/memoria; no Rust,
pruebas, Cargo ni CI. No se atribuye a este check una ejecución de tests.

CI37235240759/headff998 terminó SUCCESS22:24:35Z; todos sus pasos aprobados.
Es resultado del corte publicado, no del nuevo merge/main ni del arreglo
OOS-F01. Mientras corría, main avanzó a2302278b496aeafbdb71d5516a39933426ea424f
(LXXXVIII GLM). El delta44bc8→230 se leyó completo:3documentos,+80/−0,
sin fuente/CI. Registra pre-medición condicionada negativa y borrador FDUSD.
Se conserva el bloqueo L2 y la decisión FDUSD reservada al dueño; no se
elimina ni mueve ningún modelo. Atribución de medidas GLM, no reproducción.

La descripción «cierre hermético» caracteriza su campaña experimental;
no certifica imposibilidad universal de aprendizaje continuo ni sustituye
prueba económica neta/capacidad. La división p_range≥0,95 es una hipótesis
de esa pre-medición, no nuevo motor universal adoptado por Codex.
El siguiente candidato requiere unión documental con230 y sus gates antes
de declarar PR lista o mergeada. No usar CI verde de base anterior como sello
de una integración que todavía no ocurrió.

## 42. Candidato RA con main230 comprobado; identidades de CI separadas

Segundo merge local preparado desde cd2e00324c3fccdb9750e4f1794154d0e2911817
y main2302278b496aeafbdb71d5516a39933426ea424f, sin conflictos de contenido.
Contra cada padre se revisó el delta: GLM LXXXVIII se conserva entero; las
adendas RA permanecen. Sólo se retiraron dos líneas vacías nuevas al EOF,
sin quitar información. Checkworkspacealltargets exit0 en5,41s, antes
del commit; logSHA256E29844A035F318FD63041AE147F7064C04901ABDFDF8AF1D24015F2297B85E56.
Ninguna fuente Rust/CI difiere del candidato ya comprobado con44bc8.

El log de CIff998 confirma187aprobadas,0fallidas,3ignoradas,13resultados.
Merge realmente probado:800c5f67f24b427d735d4b2530492c24d5329c0e;
padres e3adf74e36aa48c122e83a72d1b7f6c3cbad90bd y
ff99849a20ec0780f402c59bd88a612cf68e5d95, cotejados por API.
LogSHA256277D11D0D183F7C546F014DA8B8BC05BF8A7ECFF9A1BEDB14E905D324A17490D.
Las tres ignoradas son inventario real manual y dos mediciones de tapes;
no se habilitaron ni contaron como aprobadas. No es el T1 completo.

La publicación del nuevo head RA debe conservar draft y pedir nueva CI;
no fusionar a main por extrapolar el SUCCESS anterior. OOS9a3, MG7aad,
reloj36b0 y bosquecace continúan separados y locales. La revisión global
sigue parcial. Nada autoriza trading, promoción, entrenamiento o borrado
de modelos. El dueño conserva la decisión FDUSD; no se ejecutó A ni B.

## 43. Reconciliación con el barrido F0 de Qoder y verificación de identidad

### 43.1 Alcance del candidato, preservación y coordinación

Padres del merge local: RA74be3ed561d155ea4d3e349c53fca46a5b6385a2 y
main62afe0f714a0e047f0712efac829e90e09682c0d. Main57cb8f incorpora el
barrido por fases; main62afe añade su checklist F0 y ADR-0014. Sólo hay
cambios documentales frente a RA74be. El único conflicto fue la adición
concurrente al final de COORDINACION: se eliminaron únicamente los tres
marcadores y se conservó la unión, no una elección ours/theirs.

QA de subsecuencia de líneas no vacías, en orden por cada padre:
COORDINACION conserva3618/3541 líneas y PLAN_MAESTRO_SINCRONIZACION
581/106, respectivamente. Barrido y ADR son idénticos al padre main62
normalizando CRLF. No hay delta Rust, Cargo, pruebas ni workflow contra RA74.
La preservación documental no convierte las cifras de otros autores en
mediciones reproducidas ni hace universal un principio de una ADR.

Qoder prepara F1 en qoder/f1-matematica y checkout .f1 del repositorio
compartido: rama activa, no candidata a borrado aunque sea ancestral a main.
El aviso de Codex reserva SA/utilidad en su propio checkout y OOS/preflight
en otro; no modifica el IC de Qoder ni el bloque de riesgo de Claude.
No se ha recibido un acuse nuevo de esos editores. Responder por ID,
SHA, ancla y estado es requisito de sincronización, no un hecho supuesto.

### 43.2 Hallazgo de verificación: caché compartida no acredita fuente

Primer checkalltargets falló exit101 en25,8880011s: E0433 para
backtest_engine::oos_context. El módulo y su exportación SÍ estaban en la
fuente RA. Entre tanto, SA/main57 había compilado una versión sin esa API
en el mismo CARGO_TARGET_DIR; el primer log no reconstruía ese crate.
La reutilización de artefactos de otro checkout es la explicación inferida
compatible con la traza y con la reconstrucción posterior, no una prueba
de que Cargo esté universalmente defectuoso ni de que falte el módulo RA.

Se refrescaron los mtimes de nueve fuentes Rust existentes, propias y
distintas entre estos cortes, comprobando SHA256 antes/después sin cambio
de bytes. No se borró la caché, no se tocaron fuentes/índice compartidos ni
se interrumpieron runners ajenos. Repetición del MISMO comando:
nightly-2026-06-30, check --workspace --all-targets --locked --offline -j2,
perfil dev; exit0 en130,408228s. Compilar todos los targets no ejecuta tests.

| Recibo local ignorado en target | SHA256 |
| --- | --- |
| ra-main62-check-2026-10-04.log, intento fallido conservado | 7F3BE22B81FAF699FA927C668859C18324CE643368A47B203ECDC82C43BA29F1 |
| ra-main62-check-refreshed-2026-10-04.log, reconstrucción exit0 | 9635D9805D12E855648C4E1B5C3F15A1851AB62B73045CA61541BB284905D48A |

Criterio de cierre de este problema de evidencia: candidato identificable,
artefactos atribuibles a su fuente y gates del SHA integrado. Para próximas
composiciones usar target aislado cuando sea viable, o refresco acotado con
hashes y reconstrucción explícita; un check rápido cacheado no es por sí
solo certificación cruzada. Esta ronda no modifica la política global de CI.

### 43.3 Frentes locales independientes y cobertura aún parcial

OOS684e47082b57fb962cbd4fad05fac5b998be15ba integra RA74be con el guard
de particiones de9a3; su revisor reprodujo14/0 contratos y checkalltargets
exit0/137,44s. No incluye main62 ni la corrección SA; fuente/modelos no se
ejecutaron en producción. SA28c000 más seguimiento local corrige ranking
signado y dos regresiones encontradas por revisión: enfriamiento omitido
al rechazar y centinela finito que superaba candidatos válidos muy negativos.
Revisor independiente22/0 contratos; check final exit0/124,7735498s.
Son recibos locales, no reparaciones presentes en RA ni en main remoto.

Conteo de rutas .rs seguidas por Git en este checkout:465 total,409 bajo
crates,139 en directorios tests y326 fuera de ellos. No son conjuntos
disjuntos entre crates y tests. El universo de rutas y corte debe acompañar
el conteo: no equiparar esta cifra a ~377 del barrido ajeno sin reconciliar
sus exclusiones. Enumerar rutas no es auditar semánticamente cada archivo.

La regla documental None→0 requiere revisión por consumidor: cero puede
significar independencia o neutralidad medidas; ausencia no es una medida.
MG05/IC same-τ sin as-of continúa abierto. La política de SA conserva pesos,
umbrales y calendario heredados; su reparación de signo no acredita esas
constantes, unidades monetarias, estimación espectral o rendimiento72h.
No quitar límites de seguridad para lograr la meta. CI37240563716/head74be
seguía en curso al recibo; conservar draft y exigir CI/revisión del nuevo SHA.
Sin push de SA/OOS todavía, sin merge remoto nuevo ni ramas eliminadas.

## 44. Sincronización F1 recién publicada y control de ruina reproducido

### 44.1 Inventario ajeno preservado; estados y responsabilidades

Main23701ecb05486eb13b661d3d50afaadbc9a0015a aporta sólo documentación:
62afe→23701 cambia tres archivos,+120/−1. Qoder registra23 hallazgos F1
(2HIGH,8MED,13LOW), incluyendo n vitalicio frente a momentos EWMA,
gate de volatilidad contra climatología pero no persistencia, muestras de
D0 cacheadas repetidas y rearme prequential con información posterior.
Se leyó íntegro el delta. No se confirma aquí cada hallazgo mediante nueva
ejecución: inventario del autor, no reparación ni cobertura completa de F1.

El merge local9bada+23701 conserva memoria y COORD por unión, sin ours/theirs;
barrido actualizado idéntico al nuevo padre. El cambio de su fila F0 es una
actualización explícita de Qoder, no eliminación por Codex. F2 física/cuántica
es su siguiente frente. La rama temporal F1 ya no figura en el censo actual:
no la borró Codex ni se atribuye su desaparición a nuestra limpieza.

Propuesta de revisión cruzada, pendiente de acuse: Qoder revisa F1-A1/C2/C4
en su productor; GLM/Codex coordina el nulo/admisibilidad F1-C1/C3 antes de
tocar spectral_tape; Claude mantiene GENOME-GATE y límites de dimensionado.
Codex conserva SA/OOS/contratos raíz aislados. Los fallos de temporalidad,
evidencia y unidades preceden nuevas teorías; nombres avanzados no reparan
sesgo prequential ni dependencia de escala de una función de utilidad.

### 44.2 RA-RUIN-F01 — cap de dimensionado deja salir no finitos

Relacionado con F1-B4 del inventario Qoder, pero reproducido aquí sobre el
helper REAL crates/risk-engine/src/ruin.rs:67. Su rama
if !f.is_finite() || f<=0 {return f} devuelve NaN,+Inf y−Inf sin aplicar
streak_cap ni tope0,25. El nombre clamp y su descripción de aplicación
uniforme no garantizan el rango para ese dominio. No es una probabilidad
de ruina que pueda reemplazarse libremente por0: f es fracción de sizing.

Prueba aislada rustc/nightly2026-06-30 incluye ruin.rs real y copia sólo las
constantes SURVIVAL_FLOOR=.05 yTRADE_HORIZON=200 de kelly_envelope.rs para
evitar cargar arena/modelos. Tres tests heredados pasan y tres contratos
de salida finita nuevos fallan:3/3,exit101. Este RED confirma un defecto,
no una suite GREEN ni reparación. No se ejecutaron órdenes o procesos vivos.

| Procedencia | SHA256 |
| --- | --- |
| ruin.rs real, sin modificar | 7AE99C6174DE9AC4B8D59E4EC264D10FEE35C2AEC1BC47BB7D6D0044B7ACCBCB |
| probe target/ra-ruin-evidence/ruin_contract_probe.rs | 24D21CA31BE174D69AA8B118E67BB51882345DEC597765B31D0ED37A0DC29427 |
| log target/ra-ruin-evidence/red.log | 36C30F74DC4C74E15DD4492AB58D9FFCDFC65528FC4FC71139658508696EEB72 |

Camino de impacto: risk-engine/lib.rs:384 remite el cap a evaluate_single_intent;
lib.rs:525 usa el cap seguido por max(0), ylib.rs:536 rechaza raw_exposure
no finita. Eso impide inferir automáticamente una orden con exposición
infinita: hay otra defensa. En f64, max puede enmascarar NaN como0 y acabar
en rechazo por cero sin razón de invalidez. Otros lectores son Kelly,
leverage_matrix y envolvente: requieren sus propios contratos de frontera.
No se ha demostrado pérdida real ni que todos estos caminos dejen pasar.

Severidad de contrato: MEDIA, impacto operacional extremo condicionado a
consumidores/defensas. Abierto en fuente; reservado a Claude/riesgo para
revisión cruzada, no parche simultáneo. Cierre: veredicto no finito explícito
o salida segura que NO autorice exposición; conservación de entradas finitas
y de semántica de negativos/sin señal; tests por consumidor e integración.
No suavizar/eliminar el control para aumentar frecuencia de entradas.

### 44.3 Errores de explicación y límites de la teoría de rachas

En ruin.rs:29 la frase «menos q ⇒ ... f_cap menor» contradice fórmula/tests:
para 0<q<1 yH>1, al bajar q baja |ln(H)/ln(q)| y A^(1/k),A∈(0,1),
produce un cap MAYOR (sin cambiar clamps). Es defecto documental de signo,
no evidencia de código invertido. Las etiquetas LCB de q también merecen
revisión: bajar probabilidad de pérdida relaja el cap; no llamarlo conservador
por usar el límite inferior sin revisar qué posterior y convención consume.

ln(H)/ln(1/q) estima un tamaño de racha mediante aproximación de conteo;
no calcula la distribución exacta del máximo de rachas dependientes ni
certifica una probabilidad de ruina de cartera con gaps/funding/liquidación.
TRADE_HORIZON200, floor.05 ycap.25 son políticas explícitas heredadas, no
leyes universales. Conservar controles; documentar calibración, supuestos,
dependencia temporal, incertidumbre y unidades antes de rediseñarlos.

Composición/CI de RA actualizada sigue su gate separado. SA701fa incluye
su informe/artefacto local; OOS684e sigue distinto. Main yorigin/main23701
coinciden en el último fetch; sólo1remota Claude yRA permanecen además de
main. Locales con exclusividad o checkout ocupado no se borran. No prometer
que todo cambio está en main ni que el barrido inventariado resolvió bugs.

## 45. Mandato ampliado: plan ruta-a-ruta y recibo del candidato main237

El nuevo PLAN_REVISION_EXHAUSTIVA_2026-10-04 conserva planes existentes
y fases F0–F8, con subfases metas, conceptos, matemática, estadística,
física/cuántica y algoritmos ANTES de cambiar consumidores. La raíz de
datos/reloj se contrasta antes del núcleo aunque el barrido la enumere F6.
Cada ruta del árbol main237 entra en un nuevo TSV/JSON, sin excluir scripts,
docs/config/binarios/datos versionados. Fuera del Git snapshot se requiere
anexo por recurso/formato/hash; no cargar modelos ni exponer secretos.

Clasificación automática, lectura, reproducción, reparación y certificación
son estados diferentes. El censo no aumenta por sí mismo los archivos
leídos ni cierra los23F1 del autor. Recibos separados por ruta/OID/contrato,
historia aditiva y revalidación de consumidores al cambiar OID. No tomar
búsquedas, mtime o check global como lectura. La plantilla obliga a explicar
para qué sirve cada fórmula, dominio/unidad, significado y límites; incluye
señal/veto/genoma/latencia y raíz→decisión→terminal. El espectro requiere
soporte/resolución/error:1ns o100años no inventan datos o información futura.

QA anterior al commit del merge9bada+main237: ambos padres preservados
(MEMORIA1774/1596,COORD3661/3564,PLAN603/106 líneas no vacías);
barrido actualizado idéntico al padre237 y39raícesJSON históricas iguales.
Diff contra RA sólo documental. Checkalltargets tras refresco acotado9fuentes/
hashestables exit0/154,188755s, perfildev,offline,locked,j2. Log en
target/ra-main237-check-2026-10-04.log, SHA256
DE2D2B2B212AE96A81CDF29B29CE954ACC5C85CEDD2FC78D60DA1075EBA5A0D9.
Adendas/plan/censo posteriores sólo documentales, sin cambiar Rust/CI.

CI74be seguía en curso23:23Z: alltargets ycontratosdepublicación/modelos
completados; entrada/riesgo/OOS en ejecución. No cancelarla por churn de
docs ni trasladar su resultado a este merge. Nueva CI/review antes de main.
SA local22/0 yOOS14/0 separados; sizingRUIN sigueRED3/3, no ocultarlo entre
suites aprobadas. Cobertura parcial, sin rendimiento72h ni operaciones.
