# LG5 — probabilidades, pagos y esperanza neta

Fecha: 13 de octubre de 2026. Base congelada: `3327300a24a6c65bac2971fce236140e4781260a`.
Rama: `codex/lg5-payoff-evidence-1013`, worktree `spectral-evidence-1011`.
Estado inicial: implementación y validación en curso; este plan no es un certificado.

## Reconciliación del 15 de octubre: ejecución histórica y composición nueva

Las ejecuciones del 13 de octubre sí terminaron sobre
`971ec9ed212eabb888a3f306739d37e84be44e57`. Los recibos completos, el gate
estricto y las copias de binarios delimitan estos resultados:

| Recibo | Resultado | SHA256 del JSON |
| --- | --- | --- |
| `015-lg5-fixture-full-core-risk` | 1006 aprobadas, 0 fallidas, 53 ignoradas; 137 binarios | `ad72f47e81cc20d7ac839084613752ca53d1de502689daa06e67fed67c046a15` |
| `016-lg5-fixture-backtest-t1` | 137 aprobadas, 0 fallidas, 2 ignoradas; 8 binarios | `5f9fe9c40485ba0e003d8a0e0ac3b9fd662237bedb6fb564cea4c2c2b617ed74` |

Ambos registran exit 0 y las 12 invariantes y 8 guardas verdaderas. T1 ejecutó
sus dos pruebas, con el umbral vigente íntegro. Al faltar `--nocapture`, el
log no fija el número exacto de genes: no se le atribuye el 18/144 histórico
de R4. Model Before/After 016 conservan identidad
`122d97e15473e37f4e23d9b44fde83bb8ac1b8d12a9ff53801e6b2ee8f0c401f`;
ese inventario no prueba carga de modelos ni escrituras transitorias.

ROOT retuvo ambos recibos, 145 binarios y un bundle de los seis commits LG5
en `C:/Users/jhona/.codex/audit-evidence/trader-gemini/lg5-terminal-015-016-2026-10-15-8649302e-7e22-4a2d-9f52-807d0f2207f7`.
Manifest SHA256 `bc3b665e4e7b9ed4e1cacee0fbc47a6ed3616c71ee2cbebb0657e353a18449d0`;
su sello y hash fueron releídos durante esta reconciliación. El bundle requiere
la base 332 y las copias no certifican procedencia de la caché al ejecutable.

Main avanzó a `310f729ff7c998c5b3d3a0f1722422910bcffebb`. El candidato compuesto
ROOT `e443c4d526ef41c5865c5a3d7ee597a07afac163` conserva LG5 y los cambios
posteriores LH04, OU, SPRT y headroom. Su [informe de integración](INTEGRACION_LG5_MAIN_2026-10-15.md)
gobierna ese frente. Los resultados 015/016 no se transfieren al árbol nuevo;
faltan sus pruebas dinámicas y publicación. El CI de 332 fue verde; el de
main310 observado el 15 de octubre tenía Unit success y Replay pending,
sin certificar el candidato LG5 compuesto.

El presupuesto fijo de cinco cierres descrito abajo corresponde a la base
332 y al corte 971. Main310/e443 usa un máximo configurable en [1,50] y salida
anticipada mediante SPRT sobre signos. Esa prueba no demuestra esperanza neta
positiva ni sustituye la evidencia de pagos. La recuperación de cohortes sin
evidencia sigue pendiente. La prima `(m−1)c` evita restar por segunda vez los
costes ya incluidos; no ajusta automáticamente cambios futuros de comisiones,
slippage, geometría o política de salida.

Revisión estática independiente de e443: Matrix es idéntica a 971 y el gate
económico mantiene moneda/lado/disponibilidad y ausencia distinta de cero.
LH04 usa PF puntual, Wilson y su propio orden de multiplicadores y topes;
Matrix usa PF_LCB y mezcla heurística. No se intercambian sus helpers:
`half*clamp(f)` y `clamp(half*f)` pueden diferir. El riesgo de stop realizado
es `margen*L*abs(entry−stop)/entry`; el objetivo LH04 basado en SL genética
debe distinguirse de la geometría final y de los topes de margen. Las pruebas
de unidades/caps del candidato siguen bajo validación del integrador.

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
LG5 también corrige la mezcla de probabilidades con PF histórico: recupera
`b = PF_hist·(1−p_hist)/p_hist`, y evalúa `max(0,p_new−(1−p_new)/b)`.
La identidad `p·(1−1/PF)` sólo sirve con p y PF de la misma cohorte.
El consumidor vigente suministra PF_LCB y WR publicado/posterior: b es un
proxy sintético, no el pago histórico recuperado ni una LCB demostrada de b.
Con `p_hist=.5`, `PF_hist=1.05`, `p_new=.38`, la fórmula anterior daba una
fracción positiva y la corregida devuelve cero. La transferencia de b sigue
suponiendo pagos estables y unidades compatibles; la mezcla 70/30 es heurística.
Esta corrección algebraica no se presenta como un nuevo óptimo logarítmico. No se modifica
el detector LG8/e-CUSUM ni se reutiliza su retorno bilateralmente recortado.

## Validación y coordinación

- CI de la base 332: workflows Unit and contract suites `38373292385` y Replay
  contracts `38373292396`, ambos completed/success sobre el SHA completo.
- RED de LG5 existente reproducido sobre 332 sin cambios: 0 passed/1 failed,
  Cargo 101 esperado; fórmula standalone +0.766 R frente a media −0.104 R.
  Scope: Platt real y fórmula del fixture, sin llamada al risk-manager vivo.
  Recibo `001-lg5-baseline-red.json`, SHA256
  `2e775999443fb032c8f03b958b60e8df6fa7a104c609eb76c68fa9f51958fd60`.
- Contratos previstos (no atribuyen ejecución): igual tasa de positivos con pagos distintos; salidas intermedias;
  ventana y pesos iguales al oráculo; cero frente a desconocido; unidades;
  causalidad; replay tras expulsión; persistencia/legacy; consumidor de riesgo
  con control positivo y motivo de rechazo exacto; coste contado una vez.
- Comparar contra ambos padres en cualquier merge; release all-targets antes
  del commit de integración. Congelar índice/HEAD/inputs para cada ejecución.
- La caché release del otro Codex se usa sólo con workspace/target explícitos
  y ventana exclusiva acordada. Un build frío interrumpido no cuenta como test.

Primer corte `5a3461b8`: all-targets release exit 0, 670 inputs/643 Rust,
12 invariantes verificadas. Recibo `002-lg5-candidate-all-targets.json`, SHA256
`cc7a7722c2a48924a83ce48991c0acfa8c5513e1d916c62e548cad345e00177e`.
La revisión independiente detectó la falta del reloj explícito en riesgo; las
ejecuciones 003/004 se detuvieron en compilación para corregirlo, sin pruebas
ejecutadas ni GREEN. Sus JSON registran `command_failed`, exit −1 y freeze
estable. El stdout del orquestador no sustituye esos recibos: el padre pwsh no
propagó correctamente una salida negativa. Los siguientes cortes comprueban
estado y ambos exits del JSON antes de continuar, además del resultado Cargo.

Corte `2d95e605`: all-targets 005 PASS (SHA256
`6153127919990891d5c2cf037d2eafa365f171f3312b889396464211fc8a2186`).
007 registró Cargo 101: 2 unidades Kelly y 3 contratos aprobados; 6 contratos
fallaron en la aserción `leverage > 1`, con órdenes admitidas de volumen
71,88974 y leverage 1. Recibo SHA256
`aff3435d4db9d3eed2b1b9b6eae50211349caf493d226d6fafe64bda42353ab6`.
La revisión independiente obtiene PF_LCB=1,56256319, b=0,52085440,
p_mix=0,69 y Kelly=0,09482407. Con genes baseline, L continuo≈1,60803
se cuantiza por floor a 1; esos genes no están todos fijados por la fixture,
por lo que la cifra L no es un oráculo incondicional. La admisión exige tamaño
superior a sonda y límites válidos; no exige aumentar leverage. Se corrige
sólo esa aserción, manteniendo motivos de veto, contrastes económicos y
límites de volumen/nocional. El padre abortó antes de 006/008/009.

Corte `33d4617d`: 010 all-targets PASS; 011: 2 unidades Kelly y 9 contratos
LG5 PASS; 012: 11 del modelo, 5 de restore/cierre y 1 contraejemplo activado
PASS. 013 full terminó con 367 aprobados, 2 fallidos y 40 ignorados, antes de
la suite completa de riesgo. Los dos fallos son controles positivos de
`evidence_expiry_contract`: la fixture madura no declaraba pagos netos y
LG5 rechazó la orden. Recibo 013 SHA256
`e1e281939239a0262060b19a36bd5b4b343ad53738e8a1be55aa61214accedf5`;
54 binarios retenidos. No se reporta full GREEN ni replay 014 ejecutado.
La migración de fixtures conserva expiración, aislamiento y motivos de veto;
aporta retornos sintéticos explícitos, nunca inferidos de score/PF.

Consejo: integración/core/persistencia, riesgo/admisión y revisión matemática
independientes. Reservas preservadas: AGY F7/Hodge/LH02, C-F5/estado y freno,
Línea H, Línea G8, Qoder y Sol. No se importan sus commits exclusivos ni se
reemplazan sus hunks por la reserva de LG5. La meta +100 %/72 h sigue abierta.

## Hallazgos que permanecen abiertos

La migración legacy conserva sus contadores de operaciones, pero no tiene
pagos reconstruibles. Fuera de la sonda existente, la ausencia madura se veta
con `SIN_EVIDENCIA`; lo mismo sucede con un lado o fuente sin filas retenidas.
No se resuelve este bloqueo reseteando historial, infiriendo magnitudes de
etiquetas ni concediendo sondas ilimitadas. Falta un protocolo separado de
exploración por cohorte, con presupuesto persistente y exclusión global.
La sonda vigente admite cinco cierres totales por moneda. Una media negativa
tras agotar las sondas también puede crear un estado absorbente: sin nuevas
órdenes no entran nuevos cierres. El paso del tiempo no altera esta media.
El protocolo pendiente debe contabilizar aperturas/intentos realmente
aceptados, y no rearmar presupuesto al expulsar filas o reiniciar.

La media reúne políticas y geometrías de salida distintas. Su uso en admisión
es descriptivo y requiere validar transferencia fuera de muestra; no demuestra
la esperanza condicional de cada candidato. Riesgo comprueba moneda/lado y
orden interno de fechas y `as_of <= decision_time_ms` en la API explícita
`evaluate_quantum_order_at`. Core pasa el evento aceptado; no confunde ese reloj
con la última actualización de profundidad. El wrapper legacy toma el reloj
disponible del coin una sola vez y puede rechazar si está retrasado: llamadores
de trades/directos deben aportar tiempo de decisión mediante la API explícita.
Este argumento delimita sólo evidencia neta; los otros relojes permanecen con
su contrato previo. El productor Core aporta procedencia y disponibilidad.
El tipo público y el argumento de tiempo por sí solos no certifican productor,
liquidación del exchange, snapshot atómico ni futuras órdenes rentables.

Los contratos de integración son segmentados: cierre local elegible y restore
real por símbolo. No prueban el ciclo entero apertura automática, confirmación
parcial, cierre, siguiente intención y riesgo, ni reanudación del ledger de
posiciones abiertas. Ese ledger efímero no se restaura desde archivos legacy.

La matriz recibe PF calculado con sumas de PnL neto en dólares y WR publicado
por moneda, sin el filtro lado/procedencia de NetPayoff. Son cocientes
adimensionales, pero sus ponderaciones y cohortes difieren. 50 cierres +1 %
sobre nocional 1000 y 50 cierres −2 % sobre nocional 100 dan PF_USD=5 y razón
de pagos medios USD=5; por retorno, la razón es 0,5 y la media uniforme
−0,005. Incluso con p empírica exacta, PF*(1−p)/p reconstruye el pago en USD,
no necesariamente el pago por nocional. PF_LCB y WR posterior agregan otras
aproximaciones. El proxy conservado es sensible al sizing histórico y no
representa la misma evidencia ni acredita Kelly óptimo.
