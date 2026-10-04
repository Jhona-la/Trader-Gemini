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
