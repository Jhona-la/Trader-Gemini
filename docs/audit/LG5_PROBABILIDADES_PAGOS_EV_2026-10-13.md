# LG5 — probabilidades, pagos y esperanza neta

Fecha: 13 de octubre de 2026. Base congelada: `3327300a24a6c65bac2971fce236140e4781260a`.
Rama: `codex/lg5-payoff-evidence-1013`, worktree `spectral-evidence-1011`.
Estado inicial: implementación y validación en curso; este plan no es un certificado.

## Diagnóstico y objetivo

El productor Platt recibe `(score, net_trade_pnl > 0, close_time)`.
Aprende la etiqueta de cierre positivo, incluidas salidas intermedias.
No observa explícitamente lado, TP/SL ni horizonte. El consumidor EV mezclaba
esa probabilidad con los pagos de dos barreras que la política no realiza
siempre. El docstring anterior de SignalIntent sobredeclaraba condicionamiento.

La definición de calibración depende de la etiqueta y de la población evaluada;
calibrar la frecuencia de una clase no cambia el evento que representa.
[Guo et al., sección 2](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

Contraejemplo exacto, en unidades de un stop inicial: 15 % a +2,25 R,
40 % a +0,02 R y 45 % a −1 R. La fracción positiva es 0,55 y la media
es −0,1045 R. Usar 0,55 como probabilidad de cobrar 2,25 R produce
+0,7875 R. Descartar las salidas intermedias tampoco estima la política completa.

## Contrato económico

Por operación elegible se conserva

\[
Y_i=\frac{PnL_{neto,i}}{precio_{entrada,i}\,cantidad_{cerrada,i}},
\qquad \widehat\mu=\frac{1}{n}\sum_{i\in\mathcal W}Y_i.
\]

PnL neto ya incluye las comisiones contabilizadas de entrada y salida.
La unidad es retorno por nocional de entrada, no retorno sobre margen,
fracción de capital ni R respecto de un stop móvil. No hay clipping de colas.
El denominador usa la cantidad efectivamente cerrada tras confirmación parcial.

La ventana retiene como máximo 512 observaciones, el límite de almacenamiento
heredado de calibración. Pesos uniformes: no se introduce otra semivida o kernel.
Cada consulta separa moneda, lado y procedencia: simulación local o estimación
local de un host con entrada confirmada. La segunda no acredita liquidación del
exchange. Scores, horizontes continuos y geometrías iniciales quedan congelados
como atributos; la media los agrupa y **no afirma condicionamiento por ellos**.
No se utiliza una media global de otras monedas para inventar soporte local.

`SignalIntent.net_payoff` transporta media, dispersión descriptiva, recuento,
moneda/lado, tiempo de disponibilidad y procedencia. `None` significa desconocido;
una media cero o negativa es evidencia económica válida. La dispersión no es un
error estándar, una LCB ni una garantía de ventaja futura.

Cada cierre consume una identidad de apertura por ranura/generación una sola vez.
El reloj de disponibilidad y los watermarks por ranura/procedencia sobreviven a
la expulsión de filas y a la persistencia. Las observaciones posteriores al
instante consultado no se hacen pasar por información disponible.

## Gate y costes

La puerta madura consume directamente la media neta, no `p·TP−(1−p)·SL`.
Para conservar la prima vigente sobre fricción:

\[
EV_{bruto}>m c\quad\Longleftrightarrow\quad EV_{neto}>(m-1)c.
\]

Comparar un retorno ya neto con `m c` volvería a cobrar el coste base.
El control adicional `c·L` sigue midiendo comisión por margen; no es un débito
contable nuevo. Sin soporte maduro se rechaza por falta de evidencia, sin
fallback a score, prior 0,55 o frecuencia histórica de positivos. La sonda
presupuestada existente conserva su contrato, incluidos sus límites y geometría.

## Persistencia y límites

El estado previo sólo contiene tripletas de clasificación. No permite recuperar
magnitudes, lados ni geometrías económicas: se lee sin fabricar retornos.
Los campos nuevos son aditivos y se restauran por símbolo; un estado corrupto
no se convierte en evidencia ausente para desbloquear la cuenta.

El estimador se actualiza con cierres de operaciones seleccionadas y ejecutadas.
Su transferencia a una orden nueva presupone estabilidad de la cohorte; las
variaciones de horizonte, geometría, política, tamaño e impacto requieren
contraste fuera de muestra. La disponibilidad y el soporte observados no son
una prueba contrafactual ni una garantía causal de rentabilidad.

La optimización de Kelly necesita la distribución del retorno y sus hipótesis
de escala, no sólo la media o el porcentaje de victorias.
[Busseti, Ryu y Boyd, secciones 2–3](https://arxiv.org/pdf/1603.06183).
El sizing vigente y la mezcla de probabilidades con PF se auditan separadamente;
esta corrección no se presenta como un nuevo óptimo logarítmico. No se modifica
el detector LG8/e-CUSUM ni se reutiliza su retorno bilateralmente recortado.

## Validación y coordinación

- CI de la base 332: workflows Unit and contract suites `38373292385` y Replay
  contracts `38373292396`, ambos completed/success sobre el SHA completo.
- RED de LG5 existente: ejecutar la prueba ignorada sobre la base sin cambios,
  conservando log, snapshot y versión. Pendiente al iniciar este documento.
- GREEN: igual tasa de positivos con pagos distintos; salidas intermedias;
  ventana y pesos iguales al oráculo; cero frente a desconocido; unidades;
  causalidad; replay tras expulsión; persistencia/legacy; consumidor de riesgo
  con control positivo y motivo de rechazo exacto; coste contado una vez.
- Comparar contra ambos padres en cualquier merge; release all-targets antes
  del commit de integración. Congelar índice/HEAD/inputs para cada ejecución.
- La caché release del otro Codex se usa sólo con workspace/target explícitos
  y ventana exclusiva acordada. Un build frío interrumpido no cuenta como test.

Consejo: integración/core/persistencia, riesgo/admisión y revisión matemática
independientes. Reservas preservadas: AGY F7/Hodge/LH02, C-F5/estado y freno,
Línea H, Línea G8, Qoder y Sol. No se importan sus commits exclusivos ni se
reemplazan sus hunks por la reserva de LG5. La meta +100 %/72 h sigue abierta.
