# Auditoría de integración, EWMA y reloj de evidencia — 2026-09-28

> **Continuación publicada:** [PR11](https://github.com/Jhona-la/Trader-Gemini/pull/11).
> §23: reparación RMT y significado de los cálculos; §24: contratos Hawkes;
> §25–26: respuesta del trainer, testigos ejecutados e integración por SHA.
> Los cortes anteriores son históricos y se conservan, incluidos sus bloqueos.
> **Recibo remoto en §29:** PR11 fusionado en main92534a9e; 875 pruebas
> pasan, cero fallos, un inventario ignorado; rama fuente retirada.
> **Nuevo delta posterior:** §30–31 documentan E0432 en main99a, su corrección
> mínima en PR12 y 956 pruebas aprobadas sobre el código integrado actualizado.

> Lectura por cortes: la cabecera original se conserva debajo. El estado más
> reciente empieza en §17: rama codex/ewma-w1-audit, worktree independiente;
> §20 reabre contratos del trainer perdidos durante un merge y §21 revisa GLM.
> Ni la cobertura histórica ni los tests de una revisión certifican otra.

Estado: reparación local verificada por contratos focalizados; regresión amplia
y publicación se registran por cortes al final. No certificación integral ni
autorización de operación. Rama de trabajo: main.

[Artefacto](artifacts/auditoria_integracion_ewma_reloj_2026-09-28.json) ·
[Admisión multiactivo](AUDITORIA_ADMISION_MULTIACTIVO_2026-09-28.md) ·
[Coordinación](../COORDINACION_CODEX_2026-09-28.md) ·
[PR #8 revisado](https://github.com/Jhona-la/Trader-Gemini/pull/8).

## 1. Dictamen y universo revisado

Una integración que compila puede violar contratos entre productor y consumidor:
el productor sembró momentos de entropía en uno; el nuevo consumidor los divide
por una masa que empezó en cero. Cada función por separado parece razonable;
juntas producen media 9,7676 para observaciones constantes de 0,5. Éste es un
fallo de composición, no una diferencia aceptable de calibración.

También se reprodujeron defectos del reloj y del estado probabilístico de W1:
un dato inválido bloquea la observación válida del mismo instante, un evento
atrasado altera el estado posterior y una excursión numérica puede dejar el
filtro sin masa para siempre. El fallo puede manifestarse como silencio de la
inteligencia o falsa transición; no implica haber medido pérdidas operativas.

Se revisaron los nueve diffs del PR #8 head 47a93fe6, sus comentarios (ningún
hilo de revisión pendiente en la consulta inicial), los tres módulos fuente
corregidos y las dos nuevas suites completas. Se inspeccionó también el
orquestador de cartera para planificar riesgo monetario, sin implementar ese
modelo en esta ola. No se declara lectura de todos los archivos del proyecto.

Reglas de trabajo: main, sin operaciones ni cambio de modelos/genomas activos.
Se conserva código de Claude, Hawkes de Qoder y documentos históricos. Se
avisa por buzón local y por
[comentario del PR](https://github.com/Jhona-la/Trader-Gemini/pull/8#issuecomment-5877204118).
La guía babysit se usó para revisar estado, comentarios, conflictos y pruebas;
un listado vacío de checks NO se presenta como CI exitoso.

## 2. Estado de integración: historia Git frente a contenido

Base remota inicial confirmada por fetch/ls-remote: e7bb4c59. Aunque el mensaje
de ese commit describe integración multiagente, sólo añadió una de las cuatro
suites nuevas de riesgo. Seguían fuera 31 regresiones, el informe de admisión y
cuatro JSON enlazados por documentación ya publicada.

Codex creó 6b7c6d37 exclusivamente con correlation_admission_contract (15),
correlation_numeric_contract (10) y correlation_open_contracts (6). Antes se
revalidó risk-engine completo: 195 pruebas pasan, ninguna ignorada. No se hizo
add -A ni add -u global. Este commit local no se equipara a publicación remota.

Antes del commit documental, otra sesión inició merge del PR #8: MERGE_HEAD
47a93fe6, nueve archivos staged. Codex detectó el índice ajeno y detuvo su
operación de staging documental. Los archivos preparados por la otra sesión
no fueron quitados del índice, sobrescritos, ni incluidos a ciegas en un commit.

Tras reproducir ocho fallos, se anunció corrección sobre el working-tree en
bloques exactos de state.rs, calibration.rs y W1ChangepointObserver, sin tocar
el índice de ese merge. Los cambios de Claude permanecen; el parche propio es
un diff adicional que requiere su validación e integración explícita. Esto
evita confundir el árbol staged con el que prueban los comandos cargo, que leen
el working-tree. Ningún resultado del segundo certifica automáticamente el primero.

Los dos backups locales conservan commits exclusivos; la rama del PR #8 no
era ancestro de main en el corte inicial. No hay autorización implícita para
borrar trabajo no integrado porque un PR anterior haya sido cerrado.

## 3. Matriz de hallazgos de esta ola

IDs EWMA-W1 locales; no renumeran los 305 puntos históricos ni cuentan dos veces
los defectos ya documentados como SPECTRAL. Prioridad técnica, sin pérdidas
materializadas estimadas.

| ID | Prioridad | Estado inicial → reparación / límite |
|---|---|---|
| EWMA-W1-01 | P1 | Semilla incompatible con masa: RED → inicialización coherente |
| EWMA-W1-02 | P1 | Momentos imposibles aceptados: RED → dominio y resultados finitos |
| EWMA-W1-03 | P2 | Calentamiento con alpha minúsculo: RED → log1p/expm1, masa real positiva |
| EWMA-W1-04 | P1 | Dato inválido consume timestamp: RED → sólo aceptar reloj tras actualizar |
| EWMA-W1-05 | P1 | Eventos atrasados reescriben posterior: RED → monotonía temporal |
| EWMA-W1-06 | P1 | Posterior destruido por underflow/umbral absoluto: RED → normalización logarítmica transaccional |
| EWMA-W1-07 | P1 | Masa de cola descartada pero sumada al denominador: corregida; test de conservación |
| EWMA-W1-08 | P2 | Olvido de Platt no es olvido por observación: OPEN |
| EWMA-W1-09 | P2 | Fisher/semivida dependen de banda fija: OPEN, política no teoría universal |
| EWMA-W1-10 | P1 | Working-tree probado distinto del índice/remoto: OPEN hasta cerrar la integración |
| EWMA-W1-11 | P2 | Hazard por evento, reloj ms y ruido no calibrados: OPEN |
| EWMA-W1-12 | P1 | Riesgo por conteo/EWMA, no pérdida monetaria actual: OPEN heredado |
| EWMA-W1-13 | P2 | Un extremo por gen no demuestra inercia global: OPEN metodológico |
| EWMA-W1-14 | P2 | Registro confunde ausencia/frío/fallo con cero o valor antiguo: OPEN |
| EWMA-W1-15 | P2 | Deriva espectral no alimentada por los dos publicadores del core: OPEN |

## 4. EWMA-W1-01 — semilla sin masa: el prior escondido

Productor: CoinArena::new, entropy_mean_ewma=1, entropy_sq_ewma=1,
campo_ewma_peso=0. Consumidor: ewma_con_peso seguido de
momentos_ewma_corregidos en la puerta de entropía.

Para alpha constante, la implementación acumula:

```text
M_t = (1-alpha) M_(t-1) + alpha x_t
S_t = (1-alpha) S_(t-1) + alpha x_t²
W_t = (1-alpha) W_(t-1) + alpha
media = M_t / W_t
varianza = S_t / W_t - media²
```

Si M_0=S_0=W_0=0, normalizar por W_t=1-(1-alpha)^t da momentos de los datos
ponderados. Si M_0=1 pero W_0=0, aparece masa fantasma:

```text
media_constante = x + (1-alpha)^t / (1-(1-alpha)^t)
```

Para alpha=0,0033, x=0,5 y t=31 se obtuvo 9,76757075091401. La varianza se
volvía materialmente negativa y max(0) la ocultaba como sigma cero. Con datos
alternos 0,2/0,8 y 200 eventos se obtuvo media 1,5678397524921845, fuera del
soporte observado. Un prior legítimo necesitaría registrar su masa y segundo
momento; sembrar sólo el numerador no lo implementa.

Corrección: ambos acumuladores de entropía empiezan en cero, igual que su peso
y los otros acumuladores corregidos. No se cambia layout, campos públicos,
alpha o lógica de estrategia. Tests construyen el arena real y ejecutan las
funciones públicas de actualización; no prueban sólo una copia idealizada del
helper que ya arrancaba correctamente.

## 5. EWMA-W1-02/03 — dominio, redondeo y calentamiento

El helper anterior verificaba finitud de entradas pero permitía peso mayor que
uno, segundo momento negativo, varianza materialmente imposible y overflow de
la división posterior. Some((infinito,0)) no es una medida utilizable.

Además 1-(1-alpha)^30 cancela a cero cuando alpha es minúsculo. Con alpha=1e-20,
peso=0 y momentos cero podía superar el calentamiento y producir una media
NaN. El problema es numérico, no evidencia de un régimen especial.

Ahora se exigen alpha en (0,1), masa en (0,1], segundo momento no negativo y
resultados normalizados finitos. Se evalúa la masa mínima mediante
-expm1(30*log1p(-alpha)). Una muestra con alpha pequeño PERO masa real suficiente
sigue siendo admisible; no se ha creado un veto contra todo alpha pequeño.

Sólo se lleva a cero una varianza negativa dentro de
64*EPSILON*max(segundo_normalizado, media²). Es un presupuesto relativo de
redondeo, no un umbral de mercado ni un intervalo de confianza. Negatividad
material produce None; el consumidor conserva su fallback explícito. Se
mantienen medias negativas válidas porque la función también sirve al flujo.

Treinta actualizaciones es una política de calentamiento heredada. NO son
necesariamente treinta observaciones independientes ni tamaño efectivo; la
correlación serial y los pesos requieren otro análisis. Se corrigió esa
afirmación en la documentación del helper, sin alterar el calentamiento para
optimizar artificialmente admisiones.

## 6. EWMA-W1-04/05 — el reloj forma parte del contrato estadístico

observe_at marcaba last_ts ANTES de validar W1. Repetir para cada timestamp
primero NaN y después 0,2 dejaba el detector sin calentar, mientras el control
con sólo 0,2 producía probabilidad 0,0017299563863949633. El dato inválido
consumía el único permiso de actualización de ese instante.

También se impedían sólo timestamps iguales, no atrasados. Después de 40
eventos a 0,2, insertar el evento antiguo (ts=1,W1=3) y continuar alteraba
p de aproximadamente 0,00173 a 0,999999989. No era una diferencia esperable
de orden: se estaba reescribiendo el futuro con un evento rechazable por reloj.

Corrección: validar antes de consumir el timestamp; sólo confirmar last_ts si
la actualización incorporó evidencia. Timestamps iguales o anteriores no
actualizan el filtro. Un fallo de normalización devuelve None sin tocar estado.
Se conserva la política legacy de un valor por milisegundo: distinguir eventos
diferentes con la misma marca exige identidad/secuencia o agregación explícita
aguas arriba. No se promete resolución nanosegundo cambiando una condición.

## 7. EWMA-W1-06/07 — probabilidad pequeña no significa ausencia de modelo

El código anterior calculaba densidades exponenciales y sólo normalizaba si
total>1e-12. Si no, publicaba los acumuladores sin normalizar de todas formas.
Con una excursión f64::MAX, todas las densidades quedaban cero y el siguiente
evento ya no tenía ninguna masa desde la que crecer. Incluso después de 40
observaciones normales, un salto válido a 3 devolvía p=0: estado absorbente.

Un W1=40 también es importante: el soporte completo de las 32 escalas log base
4 tiene diámetro 31*ln(4)>40. No puede descartarse diciendo que la magnitud
física siempre vive por debajo de cuatro unidades logarítmicas. Su evidencia
puede ser muy pequeña pero representable. El test exige normalización y
detección, no congelación o probabilidad minúscula sin significado.

La reparación calcula log-pesos (incluidos 1/sigma y hazard), resta el máximo
común antes de exponenciar y normaliza la masa representada. Restar una
constante común no altera probabilidades relativas. Si ni el cálculo
logarítmico puede representar la evidencia, devuelve None y conserva
posterior, medias, contador y último valor. No lo transforma en certeza de
estabilidad. No se introduce un nuevo corte de probabilidad arbitrario.

Otro defecto: el crecimiento desde el último bin no se almacenaba, pero se
sumaba a total. Así la distribución podía tener suma menor que uno. Ahora el
último bin representa una cola saturada y recibe ese crecimiento. Su media
con edad acotada es una APROXIMACIÓN declarada, no el posterior exacto sobre
todas las edades posibles. Un test privado verifica no negatividad, finitud
y suma uno durante 2.048 eventos, más allá de capacidad 128 y con saltos.

Los parámetros de emisión, hazard y regla de media legados se conservan; la
reparación de integridad probabilística no los convierte en un modelo
correctamente calibrado del mercado. No se certifica la probabilidad de
“cambio real” a partir de su valor numérico.

## 8. Deudas de teoría y diseño que no se deben disfrazar de cierre

**Platt y olvido.** La propuesta mezcla p calibrada con el score crudo según
tiempo desde el ÚLTIMO resultado. update_at restablece peso completo de toda
la ventana aunque casi todos sus datos sean antiguos; obs no conserva fechas
individuales. No es una verosimilitud con descuento por observación. Puede
romper el estado absorbente a costa de volver a un score no calibrado. Falta
validar selección de operaciones, datos contrafactuales, drift y error OOS.

**Semivida universal de 12 h.** Usar TAU_ANCHOR_SLOW_MS como tiempo de olvido
no demuestra que todas las monedas o estrategias cambien a esa velocidad.
Es un acoplamiento de política a ancla temporal, no aprendizaje continuo.
Separar velocidad de adaptación, horizonte de posición y antigüedad de datos;
calibrarlos con evidencia causal sin eliminar límites de solvencia.

**Fisher de escala.** El bound discreto y la fórmula para un bloque interior
uniforme son comprobables, pero “masa concentrada en mitad de una banda fija”
no demuestra identificabilidad de un modelo. El inverso Fmax/F no es tamaño
efectivo universal: depende de forma, bordes, ponderación y malla. El PR
mejora un gate casi inalcanzable, pero introduce otra política que requiere
ablation, cobertura y comparación fuera de muestra, no un título más avanzado.

**BOCPD por evento.** Hazard=1/100 por observación implica tasa por tiempo
físico diferente para activos con frecuencias distintas. Medias puntuales,
sigma fija y cola comprimida son aproximaciones. El W1 viene de observaciones
solapadas; iid no se obtiene por normalizar un vector. El nombre bayesiano
no basta para acreditar las hipótesis.

**Riesgo de cartera.** dependency_exposure corrigió dirección/slots; aún cuenta
apuestas con una EWMA histórica. Margin-used es colateral, no pérdida al stop,
delta ni riesgo de liquidación. El snapshot por slot no reserva presupuesto
frente a dos aperturas simultáneas. Se mantiene SPECTRAL-010 como OPEN; esta
ola no desplaza una solución sin prueba al camino operativo.

**Interpretación de la meta.** Las afirmaciones financieras de otros informes
y del cuerpo del PR se mantienen como aportaciones históricas, no resultados
verificados aquí. No se deduce un Sharpe alcanzable, una probabilidad universal
de drawdown o crecimiento asegurado de tests sintéticos. +100% cada 72 horas
sigue sin validación OOS/capacidad/costes. No se alteraron vetos para forzar esa
cifra ni se implementaron teorías ajenas sin correspondencia entre variables.

## 9. Evidencia focalizada y no equivalencias

Primera ejecución EWMA: 1 pasa/5 fallan. Primera ejecución temporal: 0 pasan/
3 fallan. Tras parche: los mismos 9 tests pasan. Luego se añadieron tres
pruebas de cobertura sin afirmar RED previo: masa real con alpha pequeño,
W1=40 representable y no consumir reloj tras fallo numérico. Resultado:
7 EWMA + 5 temporales pasan. La prueba privada de masa es la número 13 nueva;
su resultado se registra con la suite amplia, no se presupone.

```powershell
cargo test --offline -p god-engine-core --test ewma_initialization_contract -- --test-threads=1
cargo test --offline -p god-engine-core --test bocpd_temporal_contract -- --test-threads=1
cargo test --offline -q -p god-engine-core -p risk-engine -p quantum-arena -p backtest-engine -p evolution-engine --all-targets -- --test-threads=1
```

Los barridos internos no se cuentan como miles de tests. Reejecuciones no son
nueva cobertura. Un contrato de dominio no demuestra beneficio económico.
Se conservan goldens y tests ajenos; si la regresión amplia discrepa, se debe
investigar la causa, no actualizar los números sólo para obtener verde.

## 10. EWMA-W1-13 — el oráculo no demuestra inercia global de un gen

Durante la regresión se revisó también el método de
`crates/backtest-engine/tests/t1_cobertura_genetica.rs`. El test perturba una
coordenada hasta el extremo más lejano de su banda, reconstruye el genoma y
compara un vector de estadísticas de UNA serie sintética de BTCUSDT. Su
comentario «si con eso no cambia nada, no cambia con nada» no se deduce del
experimento. El test sí mide diferencias detectadas bajo ese fixture y esa
perturbación; no identifica todos los genes sin consumidor ni su influencia
en producción.

**Contraejemplo lógico, no medición de pérdidas del sistema.** Sea g0 el valor
base y b el extremo elegido. Una respuesta f(g)=(g-g0)(g-b) tiene el mismo
resultado en g0 y b, pero no en valores interiores. Ni siquiera continuidad
garantiza monotonía. En dos dimensiones, f(g,h)=(g-g0)(h-h0) no responde a
ninguna perturbación de una sola coordenada desde la base, aunque depende
de ambas conjuntamente. Puertas lógicas, saturaciones y productos pueden
producir este patrón sin que el gen esté desconectado.

Además, igualdad de estadísticas agregadas no implica igualdad de trayectoria:
dos secuencias de operaciones pueden dar el mismo PnL y conteo. La comparación
usa tolerancia relativa y tampoco es una prueba de igualdad exacta. La
reconstrucción `SuperGenotype::from_vector` puede reparar restricciones conjuntas;
por ello debe registrarse el vector solicitado Y el efectivamente expresado
antes de atribuir sensibilidad causal a una sola coordenada.

**Clasificación.** P2, OPEN metodológico. No se declara falso el resultado
numérico previo 19/144 ni se convierte «125 no sensibles en ese experimento»
en «125 genes muertos». No se modifica su umbral 0,115, su fixture ni sus
goldens. La ejecución actual conserva el test tal cual.

**Criterio de mejora.** Ampliar evidencia con perturbaciones interiores y de
ambos extremos, interacciones seleccionadas por dependencia funcional,
diversidad de activos/trayectorias y trazabilidad gen solicitado → gen reparado
→ lectura → veto/intención → riesgo → fill → fitness. Separar cobertura
estructural, sensibilidad condicionada y diferencia de comportamiento OOS.
Los ensayos deben tener presupuesto y sensibilidad de detección definidos;
un barrido finito no certifica todos los estados de un sistema continuo.

## 11. Corte de validación intermedia — 15:01 America/Bogota

- `cargo check --offline --workspace --all-targets`: termina con código 0,
  con advertencias preexistentes. Incluye compilación de todos los targets,
  no ejecución de todo el workspace.
- `cargo test --offline -q -p god-engine-core -p quantum-arena -p risk-engine
  --all-targets -- --test-threads=1`: 616 pasan, 0 fallan, 1 ignorada, código 0.
  Son 49 bloques de resultados, no 49 crates. La prueba privada de
  conservación de masa está incluida y pasa: las 13 pruebas nuevas pasan.
- La ignorada es `inspect_saved_models_without_activation_or_cache_writes`,
  en `ml_model_contract.rs`: inventario local no portable de modelos, no un
  test retirado por este parche. No se activa ni carga ningún modelo operativo.
- La regresión de cinco crates sigue en `t1_cobertura_genetica`, cuyo propio
  archivo advierte un coste aproximado de 22 minutos. No se declara bloqueada
  ni aprobada por silencio. Proceso propio identificado; el test --workspace
  de otro editor se preserva.
- El commit 6b7c6d37 fue publicado de forma explícita a refs/heads/main.
  `ls-remote` confirma main=6b7c6d37 y rama Claude=47a93fe6. Queda rectificada
  la condición puramente local descrita en el corte inicial del §2.
- PR #8 continúa OPEN; la consulta devolvió mergeable UNKNOWN y ningún check.
  Eso NO es prueba de CI verde ni de conflicto. No hay hilos de revisión sin
  resolver en la consulta. El índice de otro editor sigue en merge y no
  contiene los parches de working-tree aquí probados.

## 12. EWMA-W1-14/15 — límites de integración todavía abiertos

### EWMA-W1-14 (P2): ausencia de evidencia publicada como cero

En ambos puntos de publicación del núcleo (`process_event` y
`process_tick_dual`), `w1_medido.unwrap_or(0.0)` publica transporte cero cuando
no hay medida. Dentro del bloque con W1 medido, el resultado de
`observe_at(...).unwrap_or(0.0)` publica probabilidad cero cuando el detector
está frío o rechaza una actualización por error numérico. Si no llega W1
medido, no se actualiza esa probabilidad y puede permanecer un valor antiguo.

El parche de esta ola preserva el estado interno y devuelve None honestamente;
**eso no arregla la semántica del publicador**. Cero, desconocido, calentamiento,
rechazo numérico y dato antiguo siguen colapsando en un escalar sin estado ni
fecha de validez. El actuador directo de confianza está explícitamente
desconectado en el bloque inspeccionado (`let _ = p_transition`), por lo que
no se afirma que esta publicación esté vetando operaciones hoy. El impacto
demostrado es ambigüedad diagnóstica y riesgo de integrar futuros consumidores
sobre un contrato incorrecto.

Cierre exigible: esquema con valor opcional, estado de evidencia y timestamp
de última actualización aceptada; pruebas productor-registro-consumidor que
distingan todos los casos. No introducir NaN como sentinel sin auditar
serialización y consumidores. Estado OPEN; no se cambió el registry en esta ola.

### EWMA-W1-15 (P2): coordenada de deriva declarada pero no alimentada

`SpectralRegimeField::from_spectrum` necesita el ln(tau) anterior y un delta
temporal válido para calcular `dominant_drift`. Los dos publicadores reales
inspeccionados pasan siempre `None, 1.0`. Por construcción, la deriva queda
cero; el término `0.35 * accel_fast` no puede aportar al crash_flux publicado.

Con entradas finitas dentro de los dominios declarados, los otros tres términos
están acotados por 0,30, 0,20 y 0,15. Por tanto, ese camino no supera 0,65,
aunque los comentarios describen un campo capaz de llegar a uno. La vista
`legacy_view` exige más de 0,80 para Crash/BullRun y no puede alcanzar esas
categorías **si se alimenta exclusivamente de ese camino**. No se equipara
esto a desactivar el veto de Crash de todo el sistema: el orquestador recibe
también una etiqueta externa y el repositorio puede producirla por otras rutas.

El core consume crash_flux para reducir free_cap y el orquestador consume
su máximo entre monedas bajistas. Ambas modulaciones sí pueden actuar; lo
ausente es la contribución anunciada de deriva. El test de crash existente
construye manualmente el campo y después su crash_flux: no valida que los
publicadores reales transporten memoria temporal.

Cierre exigible: identidad/estado anterior por símbolo, delta temporal de
eventos aceptados, invalidación al cambiar símbolo y contrato equivalente
backtest/demo/live. Antes de cablearlo, medir distribución y efectos en margen:
activar de golpe el término faltante endurecería vetos y no es una reparación
inocua de telemetría. Estado OPEN, evidencia estática por inspección y cota
algebraica; no se presenta como un nuevo test RED ejecutado.

### Topología diagnóstica de esta revisión

```text
Evento del símbolo + timestamp
  └─ espectro temporal medido
      ├─ W1 opcional → reloj BOCPD → posterior normalizado
      │                 [arreglado]          │
      │                                      └─ registry escalar [OPEN: falta estado/edad]
      ├─ momentos EWMA + masa → media/sigma → veto de flujo/campo
      │      [semilla/dominio arreglados]       [políticas todavía por calibrar]
      └─ campo espectral + memoria previa ausente
                         └─ crash_flux con deriva cero [OPEN]
                                      └─ margen local y cartera
Resultado de operación → Platt → confianza/Kelly
                           [OPEN: antigüedad por observación/selección]
Perturbación de gen → reconstrucción → mismo motor → estadísticas agregadas
                           [OPEN: alcance causal del oráculo T-1]
```

El estado de una arista importa tanto como la fórmula de un nodo. Tener todos
los módulos conectados no garantiza que transporten evidencia válida,
sincrónica y del mismo significado. No se añadieron ecuaciones de otras áreas
como sustituto de estos contratos faltantes.

## 13. Revalidación de modelos guardados — FMT-037/190 sigue abierto

Se ejecutó explícitamente el inventario que la suite portable deja ignorado:

```powershell
cargo test --offline -q -p god-engine-core --test ml_model_contract inspect_saved_models_without_activation_or_cache_writes -- --ignored --exact --test-threads=1 --nocapture
```

Resultado del diagnóstico: un test pasa porque el INVENTARIO se ejecuta; eso
no significa que todos sus modelos sean válidos. Resultado de los archivos:
12 bosques estructuralmente aceptados, 8 rechazados y un JSON de otro esquema.
Se reproduce la deuda ya documentada en
[Fundamentos XX, §5.4](AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XX_2026-09-24.md);
no se inventa un ID nuevo para contar ocho veces un problema histórico.

Los rechazados siguen siendo ADAUSDT_MOTOR_CANDIDATE, ATOMUSDT_MOTOR,
ATOMUSDT_MOTOR_CANDIDATE, BNBUSDT_MOTOR, BNBUSDT_MOTOR_CANDIDATE,
NEARUSDT_MOTOR, NEARUSDT_MOTOR_CANDIDATE y NEARUSDT_VOL, todos JSON. Motivo
uniforme observado: `split has missing or cross-tree child`. Un hijo debe
pertenecer al intervalo del árbol que lo contiene; el validador rechaza
correctamente esta corrupción. Eliminar ese rechazo para desbloquear
inteligencias serviría un predictor de significado distinto, no repararía
el aprendizaje.

Los doce aceptados no representan doce entrenamientos independientes: diez
archivos FDUSD tienen el mismo SHA-256 y los dos BTCUSDT (MOTOR/CANDIDATE)
comparten otro. La igualdad binaria no prueba por sí sola un fallo: puede
existir una decisión de reutilizar un modelo. Sí impide inferir especialización
por activo sólo a partir del nombre de archivo. Debe existir procedencia,
contrato de features/target y evidencia de transferencia entre instrumentos.

Alcance estricto: JSON → from_data → validación estructural en proceso de
prueba. No se llamó load_model, load_global ni se escribieron caches, ni se
inspeccionó qué modelos están activos en demo/producción. DarkAlpha_BTCUSDT usa
otro esquema y no queda validado por NanoForest. Los hashes de los 21 archivos
leídos se guardan en el artefacto; aceptación estructural no implica calibración,
habilidad OOS o equivalencia entre backtest y producción. No se reentrenó,
reexportó ni promovió un modelo sin resolver primero procedencia y evidencia.

## 14. Traspaso de integración y límites de certificación

El usuario asignó explícitamente a **GLM el cierre del merge actual**. Codex
no cerrará, abortará ni rehará ese merge, ni modificará su índice. El buzón
contiene el manifest de las tres fuentes corregidas, dos suites nuevas,
documentación y artefactos que no pertenecen al índice inicial del PR. Un
commit sólo de ese índice NO publicaría el árbol de trabajo validado aquí.

Criterio mínimo de integración: preservar ambos padres y las reparaciones
posteriores; verificar diff contra cada padre, compilar todos los targets,
ejecutar las regresiones sobre el árbol que realmente se va a publicar y
confirmar el SHA remoto. Sólo entonces se puede declarar publicado. No
borrar la rama Claude antes de comprobar ancestralidad; los backups con 3/1
commits exclusivos siguen protegidos.

La corrección de peso EWMA elimina el sesgo de ARRANQUE de los momentos, no
produce por sí sola una varianza muestral insesgada. Bajo observaciones iid
con varianza sigma² y pesos normalizados fijos q_i, la esperanza de
`sum(q_i*x_i²) - sum(q_i*x_i)²` es
`sigma² * (1 - sum(q_i²))`. La masa W no almacena por sí sola sum(q_i²).
Corregir este factor bajo iid tampoco resolvería dependencia serial,
muestreo endógeno o drift. Se conserva el estimador legado y se delimita su
significado: no se denomina intervalo de confianza ni tamaño efectivo.

Tampoco se acredita latencia nanosegundo: el filtro mantiene coste O(128),
pero ahora calcula logaritmos y normalización estable. Faltan mediciones
p50/p95/p99 en release, por símbolo, carga y frecuencia real. La compilación
y las pruebas funcionales no son un benchmark de la ruta caliente. La malla
1 ns–146 años y un reloj de eventos ms describen objetos diferentes;
interpolar no crea observaciones ni permite medir lo que el feed no entrega.

Una futura teoría se aceptará por correspondencia entre variables, unidades,
hipótesis, estimador, evidencia causal y coste. Un nombre físico/cuántico o
una ecuación célebre no cierra estos contratos. Esta ola no implementó nuevas
teorías sin falsación ni convirtió la meta económica en resultado garantizado.

## 15. Resultado final de regresión local — 15:18 America/Bogota

La ejecución de cinco crates terminó con código 0:

```powershell
cargo test --offline -q -p god-engine-core -p risk-engine -p quantum-arena -p backtest-engine -p evolution-engine --all-targets -- --test-threads=1
```

**777 pasan, 0 fallan, 1 ignorada**, sumados desde los 60 bloques de resultados.
El bloque T-1 de dos pruebas pasa en 1.405,43 segundos (aproximadamente 23,4
minutos). No se cambió su fixture, su umbral ni los goldens. La salida quiet no
expone la cifra de genes sensibles: no se reutiliza el 19/144 histórico como
si fuera una medición nueva.

Desglose de esta misma ejecución: backtest-engine 61, evolution-engine 100,
god-engine-core 254 más un inventario ignorado, quantum-arena 166 y risk-engine
196. La suma es 777. Las reejecuciones focalizadas no se añaden otra vez.
El inventario ignorado se ejecutó manualmente aparte (§13); su aprobación
significa inventario completado, NO veinte modelos válidos.

Las 13 regresiones nuevas pasan. También pasa workspace check all-targets
con advertencias. Se verifican 9 hashes de fuentes/referencias y 21 de modelos,
los enlaces locales del informe, las ubicaciones del artefacto y whitespace
del diff propio y del índice. No hay procesos de prueba propios pendientes al
cerrar este corte. No se intervino la ejecución de otro editor.

La evidencia funcional se refiere al working-tree que contiene el merge
47a93fe6 MÁS los parches locales de Codex. No certifica el índice sin esos
parches, un ejecutable operativo ni un head remoto posterior.

## 16. Avance remoto concurrente: dos commits todavía fuera del árbol probado

La comprobación final encontró main remoto en **6b7c6d37**, pero la rama del
PR #8 avanzó de 47a93fe6 a **6ebb2857**. Fetch confirmó dos commits adicionales:
3c5b0c1a (fricción compartida) y 6ebb2857 (puntuación del bosque por dirección
bruta). Abarcan cinco archivos, +136/−52. Se revisó su diff completo en lectura;
NO se aplicaron al checkout ni se ejecutaron sus tests en esta ola. PR #8 sigue
OPEN/MERGEABLE, sin checks publicados en la consulta; MERGE_HEAD local sigue
47a93fe6. Se avisa a GLM: completar ese merge no integra automáticamente este
delta posterior. La propiedad del merge sigue siendo suya por decisión del usuario.

### Observaciones de revisión sobre el delta, no reparaciones acreditadas

**R8-A — dirección terminal y etiqueta de barreras son objetivos distintos.**
El cambio corrige una confusión real: una subida bruta puede perder dinero
tras costes. Sin embargo, `direccion_realizada` clasifica el signo del precio
al cierre, mientras `train_forest --label dir` usa el PRIMER toque de TP largo
o SL largo dentro de un plazo y descarta los timeouts. El ejemplo del PR,
+3 pb al cierre, puede no tocar ninguna barrera y ser excluido del entrenamiento,
aunque el nuevo registro lo puntúa como subida. Una trayectoria puede tocar
una barrera y terminar después en otro signo. Por eso el comentario que lo
equipara al target entrenado no prueba paridad. Requiere resolver etiquetas
sobre el mismo intervalo/barreras de cada predicción, distinguiendo timeout,
cierre anticipado y selección de operaciones. No se retira la mejora del signo
bruto: se mantiene OPEN la correspondencia estadística del registro con el
target (familia FMT-054/187).

**R8-B — función compartida no garantiza dominio ni entradas equivalentes.**
`roundtrip_friction` centraliza la fórmula y elimina tres expresiones
incompatibles; no demuestra que host, gate y daemon usen la misma volatilidad,
latencia ni snapshot. Su comentario dice que no sanea entradas, pero llama a
`latency_slippage_pct`, que devuelve cero para ATR/latencia no finitos y para
overflow. Así, con taker y piso finitos, un ATR NaN puede producir una fricción
finita sin el término de latencia; rechazar sólo el resultado no finito no
detecta esa ausencia. Se debe definir si es rechazo, ausencia o fallback
explícito y probarlo en cada consumidor, no sólo comparar una fórmula con sí
misma. Además ATR no es automáticamente sigma y la ley difusiva es una
hipótesis, no una certificación de coste ejecutado. Observación estática
notificada; no se inventa un resultado RED de un código no ejecutado aquí.

No se añade este delta al total de 15 hallazgos locales ni al de 13 regresiones
propias. Son observaciones de revisión posteriores sobre código remoto distinto.
Se conservan ambos backups locales: 3 y 1 commits exclusivos, todos con signo
+ en git cherry; no hay base para declarar que ya estén mergeados. No se borró
ninguna rama ni se publicó otro commit más allá de 6b7c6d37.

## 17. Nuevo corte de integración: aislamiento por agente

El merge anterior fue finalizado por GLM: main/origin-main observados en
417554220712de3f92e75c6469a21ec54c386de8. Sus padres conservan 6b7c6d37 y
47a93fe6. Este commit NO contiene las reparaciones EWMA/W1 de este informe:
cerrar el merge anterior no las publicó. Se crea la rama
`codex/ewma-w1-audit` desde ese SHA, en un worktree independiente. El
directorio compartido ya está en `glm/xlv-effective-bets`; su modificación
de `risk-engine/src/random_matrix.rs` no se copia ni se incorpora.

Se transfieren por manifiesto 16 archivos propios o de documentación
compartida ya inspeccionada. Los cinco hashes de fuentes y regresiones
coinciden exactamente con los de la validación de §15. Los originales se
preservan: copiar no significa descartarlos del directorio compartido.
La rama aislada usa su propia caché de compilación; no se reutiliza un
ejecutable operativo ni se detiene ningún proceso de otro editor.

El inventario de la BASE comprende 1.309 archivos versionados, 400 Rust y
24 Cargo.toml (23 crates más el paquete raíz). Una búsqueda léxica encuentra
78 archivos Rust bajo crates/src con scalp/swing. Este resultado NO demuestra
78 motores separados ni acredita lectura exhaustiva de los 1.309 archivos.
Hay nombres heredados, variables operativas, comentarios y contratos que
deben clasificarse siguiendo sus consumidores. Cambiar nombres no prueba
continuidad temporal; tampoco una rejilla finita representa un continuo
observado a resolución nanosegundo.

### Estado de comunicación y publicación en este corte

El aviso se añadió al buzón local, incluyendo el worktree y el alcance.
El conector GitHub rechazó el comentario con 403; la alternativa fue
bloqueada por la revisión de autorización del destino externo. El comentario
NO fue publicado. Se solicita autorización explícita para publicar código,
informes y coordinación en Jhona-la/Trader-Gemini. No se interpreta una
intención de publicación como push realizado. La publicación remota y la
eliminación de ramas quedan pendientes de esa autorización y de las pruebas.
Las actualizaciones posteriores se registrarán como cortes adicionales.

## 18. R8-A, segunda revisión: el motivo de salida tampoco identifica el target

**Estado: OPEN, revisión estática y contraejemplos algebraicos reproducidos.
No es un test del motor remoto ni una reparación aplicada a la rama Claude.**

Árbol revisado: PR #8 / df01c82c60d7e9cda2216583e0ca0b2894f62d5d.
XLIV-9c introduce `etiqueta_barrera(is_long, reason_code)`: TP largo pasa a
true, SL largo a false, TP corto a false; las demás salidas no se puntúan.
Esto evita algunos errores de signo terminal, pero conserva una pérdida de
información que una tabla de motivos no puede resolver.

### Contrato matemático que sí necesita el registro

En la ruta aggTrades inspeccionada de `src/bin/train_forest.rs`, las barreras
son **literales**: tp_pct=0.0036 y sl_pct=0.0018 (líneas 1346–1347 en la base).
No dependen allí de genes, contrariamente a la descripción del helper remoto.
Para precio inicial p0 se calculan U=p0*(1+0.0036), L=p0*(1-0.0018);
dentro del horizonte H, el primer toque de U da 1, de L da 0, y ningún toque
se excluye (líneas 1638–1675). El objeto aprendido está condicionado a resolver
alguna barrera dentro del plazo; NO es simplemente P(precio final>p0).

Formalmente, T_U=inf{t>t0: p(t)>=U} y T_L=inf{t>t0: p(t)<=L}.
La etiqueta vale 1 si T_U<T_L y T_U<=t0+H; vale 0 en el caso simétrico.
Sin toque dentro del plazo falta una etiqueta binaria de ese contrato.
El estado necesario incluye precio de referencia, ambos límites, horizonte,
orden temporal, versión del modelo y trayectoria de observaciones válidas.

La gestión viva usa SL/TP de posición o sus fallbacks dependientes de tau,
ATR y configuración (`core/lib.rs`, bloque de gestión por slots). La función
de dos argumentos no recibe ninguno de esos elementos. Por tanto una
inferencia universal de primer toque a partir de reason_code es imposible:
dos trayectorias distintas pueden compartir motivo de salida y tener targets
opuestos. Denominarla aproximación documenta la limitación, pero no demuestra
que su tasa de error sea aceptable para accionar el veto del bosque.

### Dos contraejemplos deterministas

Con p0=100, U=100.36 y L=99.82:

| Trayectoria observada | Brackets operativos hipotéticos | Primer toque del target | Tabla remota |
|---|---|---|---|
| 100 → 99.81 → 100.60 | Largo, SL=99.70, TP=100.60 | SL largo, y=0 | TP largo, y=1 |
| 100 → 100.40 → 99.00 | Corto, SL=100.50, TP=99.00 | TP largo, y=1 | TP corto, y=0 |

Ambas geometrías operativas tienen ratio distancia TP/SL=2 y no requieren
una combinación negativa o imposible de barreras. Los valores se eligieron
para demostrar no-identificabilidad, NO como configuración recomendada.
Una evaluación aritmética independiente del primer toque reprodujo ambas
contradicciones. No se ejecutó el resto de filtros del motor: podrían salir
antes; eso no repara la falta de información del helper ni acredita incidencia
operativa de estos escenarios. Debe medirse la frecuencia con trazas reales.

También puede alcanzarse el TP operativo después de H sin ningún toque
durante H: la etiqueta de entrenamiento es ausente, aunque el motivo sea TP.
Y una salida trailing no implica timeout: la trayectoria pudo tocar antes
una barrera del target mientras los brackets operativos eran distintos.
La exclusión por motivo introduce una selección dependiente de la política.

### Consecuencia sobre aprendizaje y veto

Un voto correcto del bosque puede registrarse como fallo, o uno incorrecto
como acierto. La tasa de acierto alimenta un intervalo de Wilson y el freno
de confianza; por tanto el error de target se convierte en permiso/veto de
entrada. Wilson no elimina errores de etiquetado, selección de operaciones,
dependencia temporal ni cambios del modelo. Su intervalo no es una garantía
de ventaja financiera: la tasa base puede no ser 1/2 y los pagos/costes son
asimétricos. Se requiere separar exactitud del clasificador, calibración y
esperanza neta condicionada a la acción.

### Criterios de reparación y prueba

1. Conservar un identificador de predicción y versión de target por
   símbolo/slot/generación, no sólo el último voto por activo.
2. Resolver un observador causal de las barreras DEL TARGET desde el instante
   de predicción, aunque la posición operativa cierre antes o después.
3. Persistir timeout, cobertura insuficiente y precio inválido como estados
   explícitos; no convertir ausencia en fallo, éxito o etiqueta terminal.
4. Probar los dos casos anteriores, toque fuera de horizonte, duplicados,
   eventos desordenados, gap de datos y cierre operacional anticipado.
5. Evaluar el registro contra un replay con etiquetas independientes y
   trazabilidad de población seleccionada; sólo después alimentar un veto.
6. Versionar la semántica y separar los contadores contaminados: no mezclar
   historia etiquetada por PnL, signo terminal y primer toque como si midiera
   una única Bernoulli estacionaria.

No se desactiva el veto ni se modifica el bloque de Claude en paralelo.
Esta revisión amplía R8-A; no se cuenta como otro de los 15 hallazgos EWMA/W1
ni como una regresión Rust verde. La reserva de trabajo y el problema quedan
documentados para integración serializada.

## 19. Lo que esta ola no certifica

La reparación numérica preserva masa y causalidad del reloj; no vuelve exacto
el modelo BOCPD aproximado ni estima automáticamente su hazard/ruido.
La normalización EWMA no resuelve memoria efectiva bajo autocorrelación.
No hay auditoría exhaustiva de todos los archivos, comprobación de cada veto,
benchmark de latencia en producción ni validación de crecimiento +100%/72 h.

La evolución científica requiere contratos falsables por componente:
dominio/unidades, hipótesis, parámetros estimados, ausencia de evidencia,
causalidad, coste y validación fuera de muestra. Añadir una ecuación física,
cuántica o de problemas del milenio sin ese mapa no cierra los fallos del
grafo de decisión. Se preservan las propuestas anteriores como hipótesis,
sin presentarlas como resultados demostrados.

## 20. Regresión de integración confirmada: contratos del trainer desconectados

**Ámbito: fuente en main 41755422, no sólo una propuesta del PR #8. Estado
ABIERTO. Se reabren partes de FMT-190/193/194; no se inventan IDs nuevos ni
se borran los informes de las reparaciones que existieron en otra revisión.**

La compilación independiente completó `cargo check --offline --workspace
--all-targets` con código 0 en 6 min 54 s. Sus avisos de código no usado
motivaron rastrear los consumidores. No se deduce el defecto del warning
aislado: se verificaron main(), las llamadas efectivas, el camino a
File::create, las funciones probadas y el diff contra ambos padres del merge.

### Evidencia causal de la pérdida

El merge **6fdccd648fcf970bfc150a5c95323a474328c2b2** (PR #5) tiene padres
ac1366331833ec8245b3c24fead09c7771b216b4 y
3ee49b051ce38a58457e526b1e8114a61f592fb2. Frente al primer padre, el diff de
`src/bin/train_forest.rs` elimina estas llamadas de la ruta ejecutable:

- Lectura de --test-in y require_promotion_holdout antes de I/O.
- SamplingBudget::new, su reserva vinculante y transporte de intervalos.
- purge_training y evidence_end derivados del fin de información.
- serving_predictions sobre el objeto NanoForestData congelado.
- Construcción/ordenación del holdout posterior y su evaluación para el gate.

Las definiciones y tests sobreviven. El segundo padre aportaba el camino de
features compartido con GodEngineCore, calentamiento y mejoras de targets.
La integración retuvo esa ruta, pero no recompuso las protecciones del
primer padre sobre ella. Es un conflicto SEMÁNTICO resuelto incorrectamente,
no una marca textual de Git pendiente. Los commits posteriores inspeccionados
no restauran el cableado. No se atribuye intención ni culpa personal.

### FMT-193 reabierto, P1: promoción sin prueba independiente conectada

En main(), sólo se lee --promote (línea 1057), no --test-in. El gate usa
best_val de la misma selección que determina best_rounds; `promote && gate_ok`
(línea 2052) habilita el destino operativo `models/{symbol}{suffix}.json`
(línea 2081), y File::create escribe después (2091–2093).

El helper require_promotion_holdout se ejecuta únicamente desde sus pruebas.
Por ello pasar --test-in no aporta por sí mismo ninguna evidencia: no hay
consumidor del argumento en este main. El mensaje que exige ese archivo y
la cabecera de validación contradicen el camino real. No se ejecutó una
promoción para demostrarlo: el análisis de flujo evita tocar los modelos.

Adicionalmente, `--val-in` asume que otro archivo equivale a otro mes y
omite purga/comprobación de cronología (bloque 1812–1817). Dos rutas diferentes
pueden contener los mismos ticks o intervalos solapados. El split interno
sí conserva purge_end, pero sólo elimina fin>inicio, no fin==inicio, mientras
el contrato de intervalos cerrados documentado exige excluir también igualdad.
No se afirma ausencia total de purga: se distinguen ambas rutas.

**Impacto:** el selector puede evaluar su propia decisión, reutilizar
información temporal y habilitar un modelo con evidencia optimista. Una
puntuación estadística en selección no equivale a capacidad causal futura
ni a aptitud económica neta. El mismo error puede hacer pasar o rechazar un
genoma dependiendo de la distribución del fixture.

**Cierre exigido:** transportar inicio/fin por etiqueta, purgar siempre por
información (también entre archivos), reservar test cronológicamente posterior,
congelar el modelo antes de puntuar y conectar el resultado del test al destino.
Prueba del CLI real: --promote sin evidencia debe fallar antes de I/O; un
test solapado debe impedir escribir; un candidato sin promoción puede seguir
siendo una salida de investigación explícitamente no certificada. Faltan
también gobierno de reutilización, incertidumbre y validación económica.

### FMT-194 reabierto, P2: presupuesto de muestras sólo en una función no usada

El main restaura la fórmula antigua (líneas 1185–1189): si stride*cap<span,
conserva stride, precisamente cuando span/stride supera cap. En la otra rama
puede sustituirlo por una cadencia menor que la solicitada. No hay contador de
reservas vinculado a SamplingBudget. Por ejemplo, span=10.000.000 ms,
stride=1.000 ms, cap=100 elige 1.000 ms y permite muchos más de 100 intentos
si existe contexto utilizable suficiente. Es testigo aritmético, no una
ejecución del trainer con un tape nuevo.

**Impacto:** tiempo/memoria no acotados por la opción anunciada, distribución
de muestras distinta a la solicitada, más dependencia entre etiquetas y
resultados de entrenamiento no comparables por carga/densidad. Las pruebas
de effective_stride/SamplingBudget pueden pasar sin proteger esta ruta.

**Cierre exigido:** reservar intentos causalmente antes de resolver etiquetas,
aplicar un contador vinculante, validar cap/stride positivos y overflow,
mantener el límite aunque se descarten etiquetas, y comprobar el número real
de intentos del CLI. No volver a confundir un test del helper con un test
del consumidor.

### FMT-190 reabierto parcialmente, P1: juez y artefacto vuelven a separarse

El exportador serialize_forest y sus correcciones estructurales permanecen:
no se declara reabierto el bug de offsets que ya fue reparado. Sin embargo,
el gate vuelve a evaluar forest_raw/eval_tree y best_val antes de serializar,
no NanoForest a través de serving_predictions. La serialización aplica sus
propias conversiones y shrinkage; equivalencia algebraica ideal no prueba
identidad de aritmética finita en el límite de decisión.

El comentario «Already serialized and validated before the gate above» no
corresponde al orden real: serialize_forest está después del gate, antes de
File::create. Se conserva la protección estructural previa a escribir, pero
se perdió la evaluación del mismo objeto que consumirá el serving.

**Cierre exigido:** serializar/validar primero, reconstruir todas las
puntuaciones del modelo retenido mediante NanoForest, usarlas en diagnósticos
y gate independiente y escribir exactamente ese objeto. Incluir pruebas de
umbral próximo, shrinkage, float32 y exportación de varios árboles.

### Medida de coordinación y límites de la reparación

No se restaura el archivo completo desde un padre: eso borraría el camino de
features/paridad del otro. La reparación debe recomponer ambas intenciones,
con pruebas del flujo ejecutable y comparación contra los dos padres.
Esta ola conserva las reparaciones acotadas EWMA/W1; no declara resueltos
estos tres contratos del trainer, no entrena, no promueve y no modifica
modelos existentes. La rama queda lista para una reparación serializada
posterior, con estos puntos como prioridad antes de otra promoción.

## 21. Revisión del avance GLM: contrato de espectro y alcance de N_eff

Main remoto fue verificado por `git ls-remote --heads origin` en
**7da858ef352663a4aaacc4dcde11a3151ce59a92**. Incluye 8389432c (espectro/N_eff)
y 7da858ef (nuevo módulo Hawkes cruzado). La rama Codex conserva base 41755422
y sus dos commits propios; las pruebas iniciadas no incluyen estos cambios.
El remoto Claude avanzó a 4833d45c; el delta desde df01c82c sólo modifica
memoria (cinco líneas), no repara R8-A ni el trainer.

Se revisó completo el diff de 8389432c, dos archivos y 201 líneas añadidas.
**No se revisó exhaustivamente ni se probó aquí Hawkes cruzado.** No se
atribuye un resultado de otro agente a esta pasada.

### GLM-RMT-A — nuevo solver público omite el contrato de correlación, P1 potencial

largest_eigenvalue valida diagonal unitaria, finitud, rango y simetría ANTES
de reconciliar sólo asimetría de redondeo. El nuevo full_spectrum comprueba
forma, promedia ambas mitades sin validar asimetría y admite cualquier
diagonal finita no negativa al converger. effective_bets lo llama directamente.

Testigos deducidos exactamente de las ramas de la función, no ejecutados
como regresión Rust en esta ola:

- C=4*I_3 tiene norma fuera de diagonal cero: full_spectrum devuelve [4,4,4].
  Con T=512, los tres autovalores superan el borde y effective_bets devuelve 3.
  No era una correlación de diagonal unitaria; largest_eigenvalue la rechaza.
- C=[[1,2],[-2,1]] se convierte por promedio en I_2 y se acepta. Se destruye
  una asimetría material y el rango inválido, inventando otra matriz.
- Dos funciones públicas del mismo módulo dejan de compartir el dominio
  anunciado en su cabecera; la ampliación futura del consumidor puede
  convertir ausencia/entrada corrupta en dimensión aparentemente válida.

El problema es el dominio aceptado, no que el método Jacobi sea inadecuado.
Reparación exigida: factorizar y reutilizar validación/solver, rechazar entradas
inválidas antes del promedio y añadir tests de diagonal, asimetría, rango,
finitud, indefinición y PSD singular. No basta probar un factor común.

### GLM-RMT-B — métrica auxiliar no equivale a apuestas efectivas vivas

Para autovalores retenidos positivos, (sum(sqrt(lambda)))²/sum(lambda) es una
participación espectral adimensional. Depende de la regla de selección MP,
no de pesos, lados ni riesgo monetario de posiciones. Con la misma matriz,
una cartera concentrada en un activo y otra distribuida reciben el mismo
resultado: no puede por sí sola medir apuestas efectivas de la cartera real.

La cabecera sugiere N para N activos independientes, pero la función sobre
I_N devuelve None porque ningún autovalor supera el borde. Esa abstención
puede ser una política válida sobre modos retenidos; entonces el nombre y
la interpretación deben distinguirla de dimensión total o independencia
demostrada. El borde asintótico no valida automáticamente un factor observado.

En risk-engine/lib.rs se añade `_n_eff_telemetry: Option<f64> = None` con TODO.
No hay cálculo ni publicación de esa telemetría en ese bloque. La existencia
del helper y tres tests no acredita cambio de sizing, veto o decisión.
Se requiere trazar un consumidor con procedencia y matriz alineada; no se
propone conectarlo a descuentos de riesgo antes de corregir su contrato.

### GLM-RMT-C — trabajo auxiliar sin consumidor

largest_eigenvalue crea y ordena un vector adicional y lo asigna a
last_spectrum justo antes de devolver sólo el máximo. Esa variable no se
consume ni se devuelve. Debe retirarse o unificarse con un solver compartido,
sin duplicar validación. Es trabajo inútil en la fuente; no se afirma una
latencia medida: el compilador optimizado podría eliminar parte o todo.
La magnitud del coste exige benchmark del binario real.

Estos puntos se dejan en el buzón local de coordinación; no se edita el
archivo de GLM ni se desconoce su autoría. Permanecen como revisión estática
pendiente, fuera de las 13 regresiones propias. No se borran ramas con
commits exclusivos ni se declara que todo el proyecto está integrado.


### Confirmación ejecutable posterior de GLM-RMT-A

Los dos testigos estáticos anteriores se ejecutaron después en un binario
mínimo compilado con rustc, sobre una copia aislada de random_matrix.rs.
Se verificó su identidad contra 8389432c antes de compilar: blob Git
`2d1413207425a19f4ef74a642431d256e5e87355`. Código de salida 0 del diagnóstico:

```text
scaled_identity: largest=None; full=Some([4.0, 4.0, 4.0]); effective_bets=Some(3.0)
asymmetric: largest=None; full=Some([1.0, 1.0]); effective_bets=None
```

Esto confirma el incumplimiento del dominio por full_spectrum y el resultado
inválidamente admitido por effective_bets para 4*I_3. En el caso asimétrico,
effective_bets se abstiene porque la matriz inventada queda bajo el borde;
NO se afirma que toda entrada inválida alcance Some en el consumidor.
El fuente de GLM permaneció intacto. La copia y el ejecutable están bajo
target/audit_rmt_8389432c del worktree, fuera de Git. No se activaron modelos,
no se ejecutó trading y no se suman estas dos consultas a las 13 regresiones.
El resultado no prueba que exista actualmente una posición mal dimensionada:
el consumidor de telemetría sigue siendo un placeholder.

## 22. Cierre verificable de esta pasada: commits locales y pruebas aisladas

**Rama:** codex/ewma-w1-audit. **Base probada:** 41755422.
Commits de fuentes creados en el worktree, sin tocar el índice compartido:

- fbf299ee — EWMA: semillas/momentos y siete regresiones, tres archivos.
- b3ba8d80 — W1: posterior/reloj y seis regresiones (cinco de integración,
  una privada), dos archivos.

La reejecución independiente terminó con código 0: **616 pasan, 0 fallan,
1 ignorada**, agregados desde 49 bloques de salida de
`cargo test --offline -q -p god-engine-core -p risk-engine -p quantum-arena
--all-targets -- --test-threads=1`. El inventario ignorado es la inspección
manual de modelos, no una regresión EWMA/W1 omitida. Las 13 nuevas pruebas
pasan. La caché nació en el worktree: no se reutilizó una compilación operativa.

Check workspace all-targets pasó antes (6m54s, advertencias). Se reconfirmaron
los cinco hashes de fuente/tests, el JSON, enlaces relativos y whitespace.
Los 777 tests de cinco crates y T-1 de §15 siguen siendo resultados HISTÓRICOS;
no se suman a 616 ni se afirma haber reejecutado T-1 en esta pasada.
No quedan procesos de pruebas propios en curso.

**Estado Git comprobado:** main/origin-main=7da858ef; Claude remoto=4833d45c.
Esos dos commits nuevos de main NO pertenecen al árbol de las 616 pruebas.
Las correcciones Codex no están en main: están guardadas localmente en su rama.
El commit documental conserva informes y artefactos previos, agrega esta
evidencia y la reapertura de contratos; no incorpora fuentes de GLM/Claude.

**Bloqueo de publicación:** conector 403 y rechazo de auto-revisión al intento
de aviso vía CLI. Se pidió autorización explícita para enviar el contenido
interno de auditoría al repositorio configurado Jhona-la/Trader-Gemini.
No se reintenta por otro canal, no se crea PR, no hay push/merge remoto Codex
en esta pasada. El buzón local contiene los avisos, sin presumir lectura
confirmada por los otros agentes.

**Ramas:** no hay otra rama desechable demostrada. Los backups conservan
commits exclusivos; la de Claude sigue avanzando; la propia no está fusionada.
No se elimina ninguna, ni se descartan las copias originales del directorio
compartido. Tras autorizar publicación, falta reconciliar el main más reciente,
comparar ambos padres, revalidar el árbol integrado, publicar/fusionar el PR
propio y sólo entonces eliminar su rama comprobando ancestralidad.
El worktree se conserva para continuar, no se archiva trabajo pendiente.

Quedan abiertas la regresión del trainer (§20), la correspondencia del target
(§18), el dominio/cableado de RMT (§21) y las ocho deudas locales originales.
Esto es un cierre de pasada y preservación de resultados, no certificación
completa del sistema, reparación de todos sus fallos ni validación de rentabilidad.

## 23. Continuación autorizada: contrato espectral único y evidencia roja

El usuario autorizó expresamente enviar correcciones, informes y avisos a
Jhona-la/Trader-Gemini, reconciliar main, fusionar el PR propio y retirar sólo
la rama propia integrada. El bloqueo de publicación de §22 es HISTÓRICO:
el aviso de coordinación se publicó en
[PR8](https://github.com/Jhona-la/Trader-Gemini/pull/8#issuecomment-5879667925).
No se amplía esa autorización a desplegar, operar, entrenar o promover modelos.

PR8 se fusionó externamente en 092452616fafa1a677e8c9c6b80738d1e653bfd7.
Antes de ello, Codex había integrado 7da858ef en 09c86127 con comparación de
ambos padres y check workspace all-targets aprobado (37,58 s, advertencias).
Los avances posteriores de main requieren otra integración y sus propias
pruebas; no quedan cubiertos por los 616 resultados históricos de §22.

### 23.1 GLM-RMT-A: reproducción, causa y corrección acotada

**Gravedad:** alta en el contrato de la API, impacto operativo no acreditado.
**Archivos:** crates/risk-engine/src/random_matrix.rs y su nueva suite
tests/full_spectrum_contract.rs. **Introducción observada:** 8389432c.
**Reproducción Cargo antes del parche:** 5 pasan, 2 fallan, código 1:
covariance_scale_is_rejected_by_every_spectral_entrypoint y
material_asymmetry_is_not_averaged_into_a_different_matrix.

La primera prueba presenta 4·I, una covarianza perfectamente PSD pero NO una
correlación: sus diagonales valen 4. full_spectrum devolvía [4,4,4]; al pasar
ese espectro a effective_bets se obtenía 3 con T=512. largest_eigenvalue,
correctamente, rechazaba la misma entrada. La segunda prueba detecta que
promediar entradas opuestas antes de comprobar simetría convierte datos
inválidos en una matriz distinta; no es una reparación estadística autorizada.
También se cubre una asimetría con ambas entradas dentro de [-1,1].

La raíz no es el algoritmo Jacobi en sí, sino haber duplicado un solver sin
copiar su contrato de entrada. La corrección extrae diagonalize_correlation:
ambas APIs comparten dimensionalidad, finitud, diagonal unitaria, simetría,
rango, convergencia y positividad semidefinida dentro del presupuesto numérico.
Se conserva tolerancia 64·epsilon·N y presupuesto de 64 barridos; no se
agregan clipping de autovalores, diagonal loading ni imputaciones. La
reconciliación de asimetría queda limitada al redondeo previamente validado.

El retorno interno conserva la matriz diagonalizada. largest_eigenvalue sólo
reduce su diagonal; full_spectrum la extrae y ordena. Se elimina la asignación
last_spectrum, que construía y ordenaba un vector sin consumidor en la ruta
del máximo. Esto elimina trabajo comprobable por lectura, NO acredita una
reducción de latencia medida. No se cambia el consumidor de riesgo ni se
conecta el placeholder de telemetría.

### 23.2 Qué calcula effective_bets y qué no significa

Sea K el conjunto de autovalores estrictamente por encima del borde MP,
p_i=sqrt(lambda_i)/sum_K sqrt(lambda). Entonces el escalar implementado es
1/sum_K(p_i^2) = (sum_K sqrt(lambda))^2 / sum_K lambda.
Para k modos retenidos positivos, Cauchy-Schwarz acota el resultado en [1,k].
Es adimensional y mide participación de ESOS modos, con la raíz como peso.
No es el participation ratio usual (sum lambda)^2/sum lambda^2, ni incorpora
las exposiciones de una cartera. Por tanto no estima automáticamente el
número real de apuestas independientes del portafolio.

Para I_N y T>N, el borde supera 1 y no queda ningún modo: la API devuelve
None, NO N. Un modo retenido produce 1; dos bloques comunes iguales producen
2. La documentación previa que afirmaba independencia→N era incompatible
con la selección implementada. Se conserva el nombre por compatibilidad y
se precisa su alcance en los comentarios, sin atribuciones bibliográficas
no justificadas ni calificar los modos de estadísticamente “limpios”.

None sigue mezclando entrada inválida, falta de soporte, no convergencia y
ausencia de modos retenidos. Esa pérdida de diagnóstico queda ABIERTA para
una API tipada futura. Ninguna de esas ausencias autoriza independencia,
riesgo cero o descuento del tamaño de posición.

### 23.3 Cobertura falsable añadida

Siete tests cubren: covarianza escalada; asimetría material; forma/rango/NaN/
infinitos; matrices indefinidas; espectros analíticos de equicorrelación con
N=2,5,30 y casos singulares; invariancia frente a permutación/cambio de signos
de exposición; y significado de la participación de modos retenidos.
Se comparan traza y autovalor máximo, no sólo un booleano de aceptación.
No son una auditoría de datos históricos, de la hipótesis iid del borde MP
ni de la PSD de matrices estimadas de forma asíncrona. Resultado verde e
integración remota se registrarán en un corte posterior, sin reescribir el rojo.

## 24. Hawkes cruzado y amplificación: auditoría del modelo, no sólo del código

**Corte:** hawkes_cross.rs de 7da858ef y correlation_guard.rs de 5cdc0fe0.
**Estado:** observaciones estáticas verificadas y revisión bibliográfica;
sin calibración empírica ni modificación de esos archivos por Codex. GLM
ha reservado su siguiente evolución matricial en el buzón. Se le avisó
localmente; no se presume acuse de lectura. La búsqueda en 5cdc0fe0 encuentra
amplificar_por_contagio sólo en definición y tests: el título del commit
NO demuestra que esté conectado a un veto operativo.

### 24.1 Objeto estimado y unidades

El bucle incrementa hits a lo sumo una vez por líder: estima la fracción
de ventanas (t,t+delta] con al menos un evento seguidor. No cuenta todos los
eventos y no estima una función de intensidad. Al ampliar delta, las ventanas
se anidan: sumar sus tasas cuenta otra vez parte de las mismas coincidencias.
Esa suma no es una masa de contagio invariante al refinamiento de la rejilla.
Añadir puntos de lag podría cambiarla sin que aparezca información nueva.

En el Hawkes lineal multivariado, la intensidad es
lambda_i(t)=mu_i+sum_j integral phi_ij(t-s)dN_j(s). El kernel tiene unidades
de tasa; su integral forma una matriz adimensional. La condición de radio
espectral menor que uno pertenece al modelo y sus supuestos, no a cualquier
matriz de coincidencias. La estimación no paramétrica mediante estadísticas
de segundo orden requiere resolver su relación integral; no equivale al
conteo de ventanas del código.
[Bacry y Muzy, secciones II–III](https://arxiv.org/html/1401.0903).

**Inferencia de auditoría:** es útil conservar el conteo como diagnóstico
descriptivo, pero llamarlo kernel estimado/contagio causal atribuye al resultado
propiedades que la implementación no calcula. La precedencia observada no
elimina un factor común ni distingue efectos directos de cascadas indirectas.

### 24.2 Referencia probabilística incompatible con el numerador

El código compara hits/n con density·delta. El primero es frecuencia del
indicador “al menos uno”; el segundo es un número esperado de eventos. Incluso
bajo un nulo Poisson homogéneo ideal, si m=lambda·delta,
P(N>=1)=1-exp(-m), no m. Por ejemplo m=0,5 produce ~0,393469, frente a 0,5
en la referencia; m=2 produce ~0,864665, mientras la implementación recorta
a casi 1. El recorte a [1e-9,1-1e-9] oculta que se excedió el dominio de una
probabilidad y altera el error estándar. Estos números son deducción de la
distribución hipotética, NO tasas medidas del mercado ni una calibración.

Sustituir una fórmula no bastaría: ventanas solapadas reutilizan seguidores,
las llegadas pueden depender temporalmente, la intensidad puede variar y
se estima la tasa base con la misma muestra. La varianza binomial usada
presupone una estructura que el contrato no comprueba. No hay corrección
por seleccionar el máximo z entre múltiples lags y múltiples pares.

### 24.3 Soporte observacional y falsación insuficiente

- follower_span_ms=0 se reemplaza por 1: un intervalo sin soporte declarado
  pasa a tener una exposición inventada. Debe distinguirse dato inválido.
- No hay inicio/fin de observación de ambos activos: una ventana final
  parcialmente observada se cuenta como si estuviera completa; no puede
  corregirse censura ni comprobarse exposición común sólo con un span.
- El avance monotónico del índice presupone timestamps ordenados. El contrato
  lo exige por texto, pero no lo valida; tampoco identifica duplicados.
- Se exigen cinco eventos follower globales, NO cinco coincidencias como
  afirma el comentario. El soporte efectivo por lag permanece desconocido.
- El test de independencia usa una sola realización y acepta Some si z<6.
  Eso no estima error de tipo I ni valida el umbral de aceptación z>=3.

Un programa de validación debería separar simulación bajo nulos, procesos
con excitación conocida, tasas variables/factor común, censura, timestamps
empatados y estabilidad frente a la rejilla. Debe medir cobertura y potencia
con intervalos Monte Carlo, además de residuos fuera de muestra. La literatura
sobre ajuste en libros de órdenes evalúa residuos y estabilidad, no únicamente
un pico en una serie que copia otra.
[Estudio de ajuste multivariado en LOB](https://arxiv.org/abs/1604.01824).
Como línea de investigación multi-activo también se identificó la estimación
de la matriz de normas del kernel aplicada a dos activos; no se implementó
ni se presupone válida para estos datos.
[Matriz de branching en flujos LOB](https://arxiv.org/abs/1706.03411).

### 24.4 Amplificador 5cdc0fe0: discrepancias de contrato aún abiertas

La fórmula ejecutada es r+(1-r)·min((z-3)/10,0,5), no la fórmula z/10 del
comentario. Para r=-0,4 y z=8 devuelve +0,3: el propio test exige ese valor,
aunque su título dice que la cobertura no se invierte. Para r=-1 el aumento
máximo es +1, no +0,5 en valor absoluto; 0,5 limita el coeficiente, no el
incremento. No se valida r finito/en [-1,1], ni se calibra la transformación
entre una evidencia de frecuencia de eventos y una correlación de retornos.

El estadístico de eventos no recibe el signo de retornos ni las exposiciones.
No puede, por sí mismo, identificar si las dos posiciones son la misma apuesta
o una cobertura. Tampoco hay garantía de que amplificar pares de forma
independiente conserve PSD de una matriz global. Esto último es un requisito
a validar para un consumidor matricial, no una ejecución de cartera fallida
ya observada. No conectar ese escalar a sizing/descuento de riesgo basándose
sólo en que sus tres tests reproducen la fórmula elegida.

## 25. Revisión de reparación concurrente del trainer (PR10)

Claude publicó c6862683/6f78aed8/905413be y reconcilió con main en e1edc1b3.
El diff restituye --test-in antes de E/S, presupuesto con contador, purga
entre archivos, evaluación del artefacto serializado y test posterior.
Es respuesta concreta a §20: se reconoce el avance y NO se duplica su código.
[Coordinación y observación de frontera](https://github.com/Jhona-la/Trader-Gemini/pull/10#issuecomment-5879876894).

Permanece una diferencia estática en ese head: el split interno usa purge_end
con end > first_val_ts; el contrato de intervalos cerrados del camino externo
usa >=. La igualdad de frontera no es cubierta por el test nuevo de
como_intervalos. Se avisó al autor para comprobar ambos caminos. El helper
además construye filas con zip; no confundir alineación del resultado con
validación de las longitudes originales. Este segundo punto es una deuda de
robustez del adaptador, no prueba de que build esté produciendo desalineación.

No se ejecutaron entrenamiento, promoción ni evaluación de un nuevo modelo.
La reparación se considera propuesta en PR10 hasta comprobar fusión y pruebas
del árbol resultante. El target primer-toque de §18 continúa abierto: restaurar
holdout no resuelve etiquetas de feedback que usan reason_code de otros brackets.

## 26. Evidencia ejecutada del avance matricial y coordinación posterior

**Fuente observada:** e2e0c8ef, blob Hawkes
02e892bdee59da2a5063eb478f8f45ae54f77507. Se comparó git hash-object del
archivo con el blob antes de compilar un harness aislado bajo target.
No se modificó la implementación ajena. Código de salida del diagnóstico: 0.

Casos y salidas:

| Entrada | Salida observada | Interpretación acotada |
| --- | --- | --- |
| Dos series vacías, spans [0,0], lag [100] | Some([[0,0],[0,0]]) | La matriz pierde la ausencia de soporte; sólo su consumidor podría evitar interpretar ceros como mediciones. |
| Roles de [[0,NaN],[0,0]] | Some con emitted/received/net_role NaN | Se valida forma, pero no dominio numérico del resumen. |
| 20 líderes separados 1000 ms, seguidor a +1 ms, lag 2 ms, span 20000 | rate=1; z≈99,89995 | Testigo determinista de coincidencia, NO validación de significancia. |
| Los mismos líderes en orden inverso | rate=0,05; z≈4,804807 | La precondición de orden importa; la API no detecta su incumplimiento. No se prueba que un llamador real la incumpla. |

La búsqueda de consumidores en 32e04af3 localiza cross_excitation,
contagion_matrix y contagion_roles únicamente en ese módulo y sus tests.
La matriz N×N aún no acredita integración multiactivo operativa. El adjetivo
“T03 completo” del comentario/commit no cierra la identificación estadística,
la conexión al grafo, la causalidad, la calibración ni las pruebas de soporte.

El resumen por filas/columnas suma z-scores positivos seleccionados. Esa suma
no es una intensidad, probabilidad ni integral de kernel. Cambia al añadir
activos o al variar el tamaño muestral; una doble contabilización de factores
comunes podría alterar el ranking de “líderes” sin nueva relación causal.
Se requiere representar por celda validez, exposición y tamaño efectivo,
definir la normalización pertinente al objetivo y calibrar la selección de
pares/lags. No se implementa esa transformación como un nuevo literal.

El aviso con los dos primeros testigos se dejó en el buzón compartido y el
PR11 publica las correcciones propias. El PR10 de Claude incorporó c7da6e70:
purge_end usa ahora >= y la prueba compara su frontera con purge_training.
Por lectura del diff se considera atendida la observación específica de §25;
la validación ejecutable y fusión de ese PR no se presuponen por ello.

### 26.1 Corte de integración Codex publicado

32e04af3 conserva e2e0c8ef y las reparaciones propias, con diff revisado contra
los dos padres; check --offline --workspace --all-targets pasa en 30,93 s.
La rama se publicó sin force y el [PR11](https://github.com/Jhona-la/Trader-Gemini/pull/11)
se abrió en borrador. En la primera consulta no hay hilos de revisión activos,
comentarios ni checks remotos. “Sin checks” NO significa “CI verde”.
Las suites de cinco crates se ejecutan antes del cierre, con su resultado
registrado en el siguiente corte.

La política de borrado sigue basándose en ancestralidad comprobada y ausencia
de uso. backup-before-cleanup contiene tres commits exclusivos;
v7-unificacion-wip, uno; la rama del PR9 cerrado, uno según git cherry frente
a e2e0c8ef. El cierre de un PR o un mensaje de commit no prueba integración.
Se preservan esos trabajos y la rama activa de Claude. Los parches originales
en el checkout compartido tampoco se descartan para hacer parecer limpio Git.

## 27. Validación ampliada y respuesta de GLM

Sobre 32e04af3, cargo test --offline -q -p god-engine-core -p risk-engine
-p quantum-arena -p feature-engine -p evolution-engine --all-targets
-- --test-threads=1 terminó con código 0: **844 pasan, 0 fallan, 1 ignorada**,
65 bloques de resultados. Incluye las 20 nuevas regresiones propias
(EWMA 7, W1 6, RMT 7); no se suman sus ejecuciones focalizadas otra vez.
La prueba ignorada es el inventario manual de modelos, no una regresión nueva.
No se reejecutó T-1; los 777 anteriores siguen siendo otro corte y otro alcance.

Durante esa compilación main avanzó a 5109f357. GLM confirmó en el buzón
que atendió los casos vacíos y NaN. El diff añade comprobación de orden
estricto en la matriz y finitud en roles. Son avances concretos: las pruebas
de 32e04af3 no prueban aún ese commit posterior.

El cambio suma→media/(N-1) define un promedio por contraparte del universo
elegido, no una magnitud invariante al universo o al soporte muestral.
Añadir un activo con enlaces cero reduce el promedio de un nodo previo,
y z conserva su dependencia del tamaño de muestra. No se solicitó esa
división como solución estadística ni se considera cerrada la calibración.
La matriz admite series de longitud 2 aunque cross_excitation requiere
20 líderes/5 seguidores: aún puede codificar soporte insuficiente como cero.
No se presume que este contrato alcance un consumidor operativo.

La incorporación de 5109f357, su check y reejecución se registrarán después;
esta separación evita trasladar los 844 resultados a un árbol no probado.

## 28. Cierre de validación del árbol integrado antes de fusionar PR11

**Árbol de código:** 8ac27783b1a23a2a9032bd4e529ffbde328b3e3e, que incorpora
main 5109f357 conservando las reparaciones propias. Se comparó el merge contra
ambos padres; el delta nuevo respecto de 3ee954af se limita a hawkes_cross.rs.
Check workspace all-targets: código 0, 34,89 s, con advertencias existentes.
No se ejecutó cargo fix ni se eliminaron advertencias ajenas indiscriminadamente.

Pruebas finales, todas con --offline y --test-threads=1:

| Alcance | Pasan | Fallan | Ignoradas | Evidencia |
| --- | ---: | ---: | ---: | --- |
| core, risk, arena, feature, evolution — all-targets | 844 | 0 | 1 | 65 bloques, código 0, reejecución completa tras 5109f357 |
| backtest-engine — lib | 31 | 0 | 0 | Código 0; replay/golden/unitarios, 19,84 s de ejecución |
| Total de estos ámbitos distintos | 875 | 0 | 1 | No incluye T-1 ni suma repeticiones focalizadas |

Las 20 regresiones propias ya están incluidas. Los testigos aislados Hawkes
son diagnósticos, no se suman al total. No se ejecutaron entrenamiento,
promoción, trading, despliegue ni cambios de modelos o genomas operativos.
Compilar todos los targets NO equivale a probar todos los crates ni leer
todos los archivos; no se certifica ausencia global de bugs ni la meta financiera.

### 28.1 Reprueba de la respuesta GLM y límites que quedan

Con blob 024bb3d5d9d16ae16ca3ce324bb38227112fbfe4 de 5109f357, el mismo
harness devuelve empty_observations=None y nonfinite_roles=None: esos dos
testigos específicos quedan atendidos. La matriz con dos eventos por activo
([1,2] y [3,4], spans 10, lag 1) sigue devolviendo Some(ceros): no hay
soporte para el estimador de pares, pero desaparece esa distinción en la salida.
Con entradas finitas [[0,MAX,MAX],[0,0,0],[0,0,0]], contagion_roles devuelve
emitted/net_role infinitos por sumar antes de dividir. Es una deuda numérica
de la API pública; no se afirma que el generador de z haya producido MAX.

La normalización por N-1 no cierra el modelo probabilístico, las unidades,
la selección múltiple ni los factores comunes de §24. La revisión reconoce
dos correcciones sin extenderlas a esos otros contratos. Tampoco reabre como
si siguieran iguales las entradas vacías o NaN que acabamos de verificar.

### 28.2 Estado de entrega y prioridades restantes

PR11 publica las correcciones fbf299ee, b3ba8d80 y 57bae732 y las adendas
documentales, con integración explícita de los avances remotos. Este corte
se escribe ANTES de ejecutar el merge remoto; la confirmación posterior se
consulta en [el PR11](https://github.com/Jhona-la/Trader-Gemini/pull/11).
No se afirma fusión basándose en un push o en que GitHub diga mergeable.

Orden de seguimiento: (1) verificar integración y ejecución del trainer
propuesto por Claude en PR10; (2) reconstruir targets causales de primer toque,
no deducirlos de reason_code; (3) diferenciar ausencia/invalidación/valor medido
en APIs espectrales y matriciales; (4) calibrar nulos, selección y costes
antes de conectar nuevas teorías al veto/sizing; (5) continuar cobertura
archivo por archivo con estado de lectura explícito. Se preservan los
backups con commits exclusivos y los originales locales compartidos.

La guía babysit determinó la revisión de hilos, checks, padres y conflictos.
La guía de investigación Firecrawl permitió contrastar definiciones con
fuentes primarias; sus resultados se incorporan como límites del modelo,
no como autorización para activar una estrategia.

## 29. Recibo de integración remota y eliminación segura de la rama

PR11: merged=true, merged_at=2026-09-28T22:50:17Z,
merge_commit_sha=92534a9ee7e394fe08264b43319dec7544d5c216.
git ls-remote confirmó ese SHA en refs/heads/main.
git merge-base --is-ancestor confirmó c90679dc dentro de origin/main;
git diff c90679dc origin/main no mostró diferencias de árbol.
EWMA/W1/RMT y los informes publicados SÍ llegaron a main. Las 875 pruebas
de §28 corresponden a su código; no se atribuye CI inexistente.

Sólo después se separó el worktree de la rama y se eliminaron las referencias
local y remota codex/ewma-w1-audit. No se borraron archivos ni commits:
son recuperables desde main/92534a9e y c90679dc. El worktree se conserva
para reutilización. Este recibo se prepara en codex/ewma-w1-receipt,
sin cambios de fuentes/tests ni activación del sistema.

### 29.1 Incidente real de coordinación, sin pérdida de fuentes

GitHub registró closed a 22:47:08Z y head_ref_deleted a 22:47:09Z en PR11,
cuando merged=false y main5109f357 NO contenía fbf299ee. Los eventos usan
la cuenta compartida; no se identificó qué proceso/agente hizo la limpieza.
No se atribuye a Claude o GLM por inferencia.

Los commits seguían en el worktree aislado. Codex republicó c90679dc,
verificó la referencia remota, reabrió el mismo PR y contrastó otra vez
base/head antes del merge protegido por SHA.
[Registro de recuperación](https://github.com/Jhona-la/Trader-Gemini/pull/11#issuecomment-5880140159).

La limpieza exige conjuntamente ancestralidad del head exacto en main,
ausencia de uso y preservación de cambios pendientes. Un PR cerrado,
un nombre antiguo o una referencia movida NO satisfacen ese contrato.
La recuperación documenta este incidente, no descarta otras acciones
concurrentes ni certifica automáticamente otras ramas.

### 29.2 Límites de la entrega

PR10 seguía abierto al consultar; su resultado 39/39 pertenece a Claude.
No se duplicó su trainer ni se borró su rama. Los backups mantienen commits
exclusivos. El checkout compartido conserva originales y avisos: no se usaron
reset, stash, clean ni checkout-discard. Main REMOTO integrado no significa
que ese directorio esté limpio.

No se afirma revisión de todos los archivos/crates. Persisten los contratos
abiertos de target, soporte, calibración y consumidores. No hubo operación,
entrenamiento, promoción ni despliegue. Las pruebas no certifican rentabilidad
ni duplicación del capital cada tres días.

## 30. Regresión posterior de main: import roto del modulador (PR12)

Tras verificar PR11, main avanzó a 99a19bfb, que contiene f0fcf08a.
Ese delta agrega signal-engine/src/contagion_modulator.rs y lo declara
públicamente en signal-engine/src/lib.rs. El nuevo archivo importa
crate::hawkes_cross::ContagionRole, pero el propietario real es feature-engine;
signal-engine no declara ni reexporta hawkes_cross en su raíz.

**Fallo reproducido:** cargo check --offline -p signal-engine termina con
código 1, E0432, en contagion_modulator.rs:21. El hecho de que la fórmula
tenga cuatro tests no los hace ejecutables si el crate no compila.
El error alcanza a los consumidores del crate; no depende de activar una
estrategia ni de alcanzar una condición rara de mercado.

**Corrección mínima:** importar feature_engine::hawkes_cross::ContagionRole,
dependencia ya existente. No se duplicó el tipo, no se añadió dependencia,
no se cambió el descuento ni se conectó el modulador a decisiones.
Merge 0f31d628 incorpora main99a y el arreglo, con diff contra ambos padres.
cargo check --offline --workspace --all-targets pasa después en 19,25 s,
con advertencias existentes. El PR12, inicialmente sólo recibo documental,
se amplía explícitamente con este arreglo de una línea; ya NO es sólo docs.

### 30.1 Contrato económico todavía abierto

La fórmula preservada usa 0,30, escala 5 y cap50. No se halló en el delta
calibración de esos parámetros ni evidencia que transforme diferencias de
z en pérdida de probabilidad de acierto o retorno esperado. Saber que eventos
de A preceden a B no demuestra que toda señal de B tenga menor edge: un
retardo predictivo puede ser precisamente su fuente de oportunidad.
La redundancia de cartera y la confianza de una señal son objetos distintos.

Además, el cap50 hace que el descuento ejecutable máximo sea
0,30·50/55≈0,272727, no que alcance 0,30. El límite documentado de 30%
puede servir como cota, pero no describe el máximo alcanzable de esa
implementación. Los tests comprueban ejemplos de la fórmula elegida,
no validez probabilística ni mejora fuera de muestra. Una función acotada
no pasa a estar calibrada por llamarla sigmoide.

En main99a la búsqueda sólo encuentra definición y cuatro tests del
modulador; no hay consumidor operativo localizado. Este hallazgo no prueba
que una orden real haya sido penalizada. Su conexión debe seguir pendiente
hasta definir variable objetivo, soporte y falsación; no se modifica aquí
para inventar otra forma o constante en respuesta a la auditoría.

### 30.2 Conflictos compartidos y atribución posterior del incidente

GLM añadió después un reconocimiento explícito de que borró la rama Codex
tras fallar el merge de un PR draft. Se registra como declaración del agente,
posterior a §29, sin alterar lo que se sabía en ese corte. Su mensaje decía
que no se podía reabrir; el recibo GitHub demuestra que el mismo PR11 sí se
reabrió y fusionó correctamente desde c90679dc, no desde el head restaurado5109.

En el checkout compartido aparecieron cuatro documentos UU tras restaurar
stash: .agents/MEMORIA.md, ATLAS_ANALITICO.md, COORDINACION_CODEX_2026-09-28.md
e INFORME_FORENSE_MAESTRO.md. Se comunicó en
[PR12](https://github.com/Jhona-la/Trader-Gemini/pull/12#issuecomment-5880217946).
Codex no cerró un índice ajeno ni borró un lado documental. El propio checkout
aislado no tiene esos conflictos. El estado final compartido debe verificarse
por separado; un merge remoto correcto no resuelve automáticamente un stash.

## 31. Validación final del arreglo adicional para PR12

Código probado: merge 0f31d628, que incorpora main99a19bfb y modifica
únicamente el import del modulador frente a ese main. Check workspace
all-targets pasó (19,25 s) después de reproducir E0432 antes del arreglo.

Reejecución offline con --test-threads=1:

| Alcance | Pasan | Fallan | Ignoradas |
| --- | ---: | ---: | ---: |
| core/risk/arena/feature/evolution/signal — all-targets | 925 | 0 | 1 |
| backtest-engine — lib | 31 | 0 | 0 |
| Total de ámbitos distintos del mismo árbol | 956 | 0 | 1 |

Los seis crates producen 70 bloques de resultados; signal aporta 81 pruebas.
Backtest lib terminó en 16,65 s de ejecución. Ambos comandos salen con 0.
Las 875 de §28 son un corte anterior, NO se suman a 956. Los cuatro tests
del modulador están incluidos; pasar esos ejemplos no calibra su fórmula.
No se reejecutó T-1 ni se entrenó/promovió/operó ningún modelo.

Se amplió el PR12 de forma explícita: recibo + una corrección de import.
La aprobación de compilación no se traslada a un main futuro; verificar SHA
y ancestralidad después del merge en
[PR12](https://github.com/Jhona-la/Trader-Gemini/pull/12).
Se actualizaron MD/JSON y memoria por adendas. La resolución del stash
compartido requiere un único integrador; se consultó al usuario mientras
continúa la entrega aislada, sin tocar ese índice ni descartar documentación.
