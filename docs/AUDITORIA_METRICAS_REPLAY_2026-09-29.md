# Auditoría MX — métricas, evidencia y causalidad del replay

Fecha local: 2026-09-29 (America/Bogota). Autor: Codex.
Estado: **Needs revision — revisión parcial**. Correcciones locales verificadas;
no certificación de rentabilidad, de todos los archivos ni de producción.

## Resultado ejecutivo y alcance

Se revisó completamente `metrics.rs`, la construcción de sus entradas en
`booktick_replay.rs` y los consumidores de `ReplayStats` en evolución y
reporte de ventanas. La revisión de esos consumidores fue focalizada, no
una lectura completa de todos sus archivos. Las comprobaciones científicas
y numéricas se limitan a este grafo. No se reauditaron los 1.358 paths del
inventario histórico ni se los cuenta como aprobados.

Registro: **26 observaciones**: 14 defectos previos reparados en código local,
una regresión detectada y reparada durante el propio desarrollo (MX-26),
dos contratos aclarados y nueve problemas abiertos. Son familias de causa,
no número de tests. Los casos abiertos condicionan cualquier interpretación
financiera del panel, aunque la aritmética del módulo ahora pase las pruebas.

Cambios de código: sólo `crates/backtest-engine/src/metrics.rs` y el nuevo
`tests/ex_post_metrics_contract.rs`. Se preservan los cuatro tests originales.
No se cambian fitness, límites de riesgo, pesos de genoma, golden, datos ni
políticas de admisión. No se ejecutan trading, demo, entrenamiento, promoción
o tapes T-1. No se añaden ecuaciones avanzadas sin variable, unidad,
identificabilidad y contraste: aquí el problema material es medición y linaje.

La skill `validate-data` guió la separación de valor, ausencia, población y
denominadores; Firecrawl aportó pasajes de fuentes primarias. No se simulan
revisores humanos ni agentes especializados que no hayan intervenido.

## Grafo diagnóstico: de la raíz a los consumidores

```text
Tape (timestamps y cotizaciones)
 ├─ prefijo hasta max(warmup,600) → process_kline
 └─ bucle vuelve al índice 0                         [MX-19]
      ├─ core / cierres c1,c2 → capital y DD         [MX-18]
      └─ c1.or(c2), sólo tras warmup → PnL y qty     [MX-22]
           ├─ 2 × qty × mid_cierre → nocional        [MX-21]
           ├─ ex_post_metrics + span del archivo    [MX-20]
           │    └─ panel diagnóstico reparado
           └─ sharpe antiguo → evolución / ventanas [MX-23]
                                └─ tasa 3d          [MX-24]
```

Este grafo representa dependencias de código observadas, no un trazado de una
operación real. El punto de decisión operativo sigue fuera del módulo reparado.
Los nodos terminales aún no consumen uniformemente el contrato diagnóstico.

## Qué calcula cada métrica y para qué sirve

Sea p_i el PnL NETO monetario de cada cierre, n el número de observaciones,
D el tiempo observado en días y Y=365,25. No confundir p_i con un retorno
fraccional r_i ni con un flujo de equity mark-to-market.

| Métrica | Operador y unidad | Uso legítimo y límite |
| --- | --- | --- |
| CAGR | exp(Y/D × log(C_fin/C_ini))−1; fracción/año | Equivalente geométrico entre extremos, sin cashflows externos. No pronóstico. |
| SharpePnL proxy | mean(p)/sigma_population(p) × sqrt(nY/D); adimensional | Dispersión por trade escalada; no Sharpe de cartera. Sigma usa n, no n−1: se preserva convención descriptiva. |
| SortinoPnL proxy | mean(p)/sqrt(sum(min(p,0)^2)/n) × sqrt(nY/D) | Downside contra objetivo 0, denominador n total. Mismos límites temporales del proxy. |
| Calmar | CAGR/MaxDD | Cociente entre crecimiento y caída; necesita la misma población y curva. |
| MaxDD | Valor fraccional recibido del caller | El módulo valida dominio; NO reconstruye la curva ni valida su mark-to-market. |
| CVaR95 mostrado | max(0, ES_95 de pérdidas monetarias por trade) | Media de cola empírica, no probabilidad de ruina ni estimación de capacidad de cartera. |
| Turnover/d | nocional_bruto/C_ini/D | Intensidad de operación respecto al capital inicial; no usa equity medio. Requiere nocional real. |
| WR | count(p>0)/n | Frecuencia de ganancias; p=0 no es ganador. No mide tamaño ni crecimiento. |
| PF | sum(p>0)/abs(sum(p<0)) | Relación monetaria entre ganancias/pérdidas. No reemplaza equity ni riesgo temporal. |

Para ES, ordenar p_(1)≤...≤p_(n), q=n/20, k=floor(q), r=q−k.
La media de peor cola es (sum_{i=1..k} p_(i) + r p_(k+1))/q.
El término fraccional es cero si r=0. Se cambia el signo y se aplica el piso
de presentación 0. Esta fórmula es nuestra especialización algebraica de la
integral de cuantiles a la distribución empírica equiponderada, no una
afirmación de que la muestra reproduzca la cola poblacional.
[Acerbi y Tasche, proposición 3.2](https://arxiv.org/html/cond-mat/0104295#S3.Thmtheorem2).

La regla de raíz temporal depende de la covarianza entre incrementos.
Para retornos aditivos estacionarios,
Var(sum_{t=1..q} r_t)=q gamma_0+2 sum_{k=1..q−1}(q−k)gamma_k.
Omitir los términos cruzados requiere justificación; disponer de un conteo
de trades no la proporciona. La identidad se usa aquí para identificar la
deuda, no para fijar un lag HAC ni introducir un nuevo motor.
[Benhamou, sección 3.6.2](https://arxiv.org/html/1808.04233#S3.SS6.SSS2).

### Ausencia, degeneración y límites de representación

- NaN significa dato inválido, ausencia o ratio no definido; no cero.
- Un PnL no finito invalida el conjunto completo de métricas por trade.
  No se descartan selectivamente filas.
- Capital/tiempo/MaxDD/nocional se validan por dependencia, no con una
  salida global que borre estadísticas independientes.
- Sigma cero deja Sharpe NaN. Sortino/PF con ganancias y sin pérdidas
  mantienen +Inf por contrato; 0/0 es NaN.
- Default no tiene rendimiento observado. Una lista vacía puede coexistir
  con extremos de capital conocidos: CAGR es calculable sólo bajo el
  contrato de ausencia de cashflows, no por deducción del listado vacío.
- Los NaN públicos cambian la semántica de default; consumidores deben usar
  is_finite/is_nan, no igualdad ni conversión a cero. PartialEq con NaN no
  implica igualdad reflexiva; no se ha cambiado ese trait.
- Mínimo 20 y piso positivo del ES son políticas preservadas y explícitas.
  No son garantías de precisión. Calmar con DD cero no prueba perfección.
- Costo: ordenación O(n log n), memoria O(n) para cola, recorridos O(n)
  para momentos. Es ex-post; no se midió latencia p99 ni rendimiento HFT.
- El horizonte de la API está en milisegundos u64. Una aspiración de 1 ns
  a 100 años no añade observaciones ni resolución física al tape. No se
  declara continuo ilimitado lo que esta interfaz no representa.

## Matriz de observaciones

| ID | Prioridad | Estado local | Problema |
| --- | --- | --- | --- |
| MX-01 | P1 | fixed_local | Un PnL no finito podía detener la auditoría por panic |
| MX-02 | P1 | fixed_local | Duración cero se confundía con pérdida total |
| MX-03 | P1 | fixed_local | Dominio de capital incompleto y propagación engañosa |
| MX-04 | P1 | fixed_local | Drawdown inválido era saneado a cero riesgo |
| MX-05 | P1 | fixed_local | Nocional negativo o infinito entraba como rotación válida |
| MX-06 | P1 | fixed_local | Default y muestra vacía fabricaban rendimiento observado |
| MX-07 | P1 | fixed_local | Epsilons absolutos rompían invariancia monetaria |
| MX-08 | P1 | fixed_local | Momentos y sumas monetarias desbordaban antes del cociente |
| MX-09 | P1 | fixed_local | Downside cuadrático podía desaparecer con Sortino finito |
| MX-10 | P1 | fixed_local | La cola redondeada no era exactamente el peor 5 % empírico |
| MX-11 | P1 | fixed_local | La suma de cola desbordaba aunque su media fuera finita |
| MX-12 | P1 | fixed_local | CAGR perdía información en el ratio de extremos |
| MX-13 | P1 | fixed_local | Turnover finito se perdía por división intermedia |
| MX-14 | P1 | fixed_local | Ratios 0/0 y sigma cero se presentaban como ceros medidos |
| MX-15 | P1 | documented | El nombre Sharpe no especificaba que la variable era cash PnL |
| MX-16 | P2 | documented | Veinte trades no certifican una cola bien estimada |
| MX-17 | P1 | open | Anualización por conteo no modela dependencia ni reloj económico |
| MX-18 | P1 | open | Capital y estadísticas de trades usan poblaciones distintas |
| MX-19 | P0 | open | Precarga del prefijo y replay desde el inicio violan causalidad temporal |
| MX-20 | P1 | open | El span solicitado no necesariamente es el span procesado |
| MX-21 | P1 | open | Turnover reconstruye ambas patas con el precio de salida |
| MX-22 | P1 | open | Option::or descarta un segundo cierre si ambos existen |
| MX-23 | P1 | open | Panel nuevo desconectado de consumidores históricos |
| MX-24 | P1 | open | Equivalente de tres días compone una tasa diaria aritmética |
| MX-25 | P1 | open | Panel por símbolo no acredita riesgo multiactivo de cartera |
| MX-26 | P2 | fixed_candidate_regression | El candidato dejaba que -0 de DD invirtiera Calmar |

Los estados no implican integración en main. `documented` sólo cierra
ambigüedad de contrato, no demuestra utilidad predictiva.
`fixed_candidate_regression` registra un error del candidato de esta ola.

## Expedientes detallados

### MX-01 — Un PnL no finito podía detener la auditoría por panic

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:177`.

**Causa y condición.** La versión base ordenaba con partial_cmp(...).unwrap(). Un NaN vuelve incomparable el par y, con al menos 20 trades, el cálculo de cola aborta; Inf podía contaminar los momentos sin una señal coherente de invalidez.

**Consecuencia.** Un registro corrupto puede impedir producir el informe o dejar unos campos aparentemente válidos y otros no. Filtrar la fila sería otro sesgo porque modifica n y la población.

**Reparación / ruta propuesta.** Validar la muestra completa antes de calcular métricas por trade; conservar n de entrada y devolver NaN para todos sus estadísticos. Los extremos de capital, proporcionados aparte, conservan su cálculo independiente.

**Verificación o evidencia de aceptación.** nonfinite_pnl_invalidates_trade_metrics_without_panicking_or_dropping_rows

**Límites.** RED: panic en unwrap; GREEN: NaN/+Inf/-Inf sin panic, n=20 conservado. No se demuestra que el feed real ya esté libre de corrupción.

### MX-02 — Duración cero se confundía con pérdida total

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:147`.

**Causa y condición.** El else del CAGR asignaba -1 tanto a capital final no positivo como a un span de cero. A su vez Sharpe y turnover se convertían en cero por falta de tiempo.

**Consecuencia.** Un problema de reloj podía registrarse como ruina de -100 %, o como ausencia de riesgo/rotación, alterando la interpretación del replay.

**Reparación / ruta propuesta.** Con span=0, CAGR, Sharpe/Sortino anualizados, Calmar y turnover quedan NaN. WR, PF y cola pueden seguir identificados por la muestra.

**Verificación o evidencia de aceptación.** zero_span_is_unknown_annualization_not_ruin; finite_ruin_remains_minus_one_only_with_valid_clock

**Límites.** No se reconstruyen timestamps ni se inventa un día mínimo. La validación de orden del caller sigue pendiente en MX-20.

### MX-03 — Dominio de capital incompleto y propagación engañosa

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:147`.

**Causa y condición.** El chequeo inicial capital>0 admitía infinito. Capital final NaN se convertía en ruina, +Inf en crecimiento infinito y capital inicial inválido provocaba un panel de ceros.

**Consecuencia.** La ausencia de saldo o una lectura corrupta podían parecer rentabilidad medible. Invalidar todas las métricas por un saldo malo también ocultaba información independiente.

**Reparación / ruta propuesta.** Exigir capital inicial finito y positivo; final finito. Con tiempo válido, insolvencia finita conserva el piso histórico -1. No se usa ese piso para representar datos ausentes.

**Verificación o evidencia de aceptación.** invalid_initial_capital_does_not_invent_zero_growth; nonfinite_final_capital_is_not_ruin_or_infinite_growth

**Límites.** El CAGR de extremos presupone ausencia de aportes/retiros y no prueba reconciliación contable. Un saldo negativo no se vuelve económicamente aceptable por reportarlo como ruina.

### MX-04 — Drawdown inválido era saneado a cero riesgo

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:126`.

**Causa y condición.** El constructor sustituía DD negativo, NaN o infinito por cero. Un CAGR positivo entonces se dividía conceptualmente por ausencia de caída y producía Calmar infinito.

**Consecuencia.** El panel premiaba una entrada inválida como si fuera un recorrido sin drawdown.

**Reparación / ruta propuesta.** Aceptar sólo DD finito no negativo; inválido produce MaxDD y Calmar NaN. No imponer techo 1 sin política de deuda/apalancamiento. El módulo recibe DD, no lo estima.

**Verificación o evidencia de aceptación.** invalid_drawdown_is_not_zero_risk

**Límites.** La curva usada para producir el DD sigue siendo responsabilidad del caller. Esto no añade un veto operativo.

### MX-05 — Nocional negativo o infinito entraba como rotación válida

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:163`.

**Causa y condición.** Turnover dividía directamente cualquier notional_total por capital y días. Un acumulador corrupto podía generar rotación negativa o infinita.

**Consecuencia.** La presión de costos y capacidad quedaba mal medida; una cifra negativa no representa nocional bruto.

**Reparación / ruta propuesta.** Validar finitud y no negatividad de nocional, junto con capital/tiempo. Cero explícito con denominadores válidos continúa siendo cero; dato inválido es NaN.

**Verificación o evidencia de aceptación.** invalid_notional_is_not_valid_turnover

**Límites.** No se sustituye un nocional desconocido por cero ni se corrige la aproximación del caller: MX-21 permanece abierto.

### MX-06 — Default y muestra vacía fabricaban rendimiento observado

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:70`.

**Causa y condición.** Default derivado inicializaba todo a cero y la salida temprana por lista vacía ocultaba incluso un cambio válido entre extremos de capital.

**Consecuencia.** Un replay que nunca calculó métricas parecía tener crecimiento, riesgo y cola iguales a cero. La misma salida borraba crecimiento de extremos conocidos.

**Reparación / ruta propuesta.** Default usa NaN para las nueve métricas. Con lista vacía se calculan sólo las dependencias independientes conocidas: CAGR/Calmar/turnover y DD recibido; no WR ni ratios de trades.

**Verificación o evidencia de aceptación.** default_has_no_measured_performance; empty_sample_preserves_endpoint_growth_but_not_trade_statistics

**Límites.** n=0 y span_days=0 siguen siendo metadatos, no evidencia de rendimiento. NaN no debe ser ordenado como cero por futuros consumidores.

### MX-07 — Epsilons absolutos rompían invariancia monetaria

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:182`.

**Causa y condición.** Los cortes 1e-12 sobre sigma, downside, pérdidas y DD confundían pequeños valores válidos con ceros. Expresar el mismo PnL en otra unidad cambiaba Sharpe, Sortino o PF.

**Consecuencia.** El sistema parecía responder a diferencias económicas cuando sólo cambiaba la escala de representación. Un DD 1e-15 daba Calmar infinito en vez del cociente finito.

**Reparación / ruta propuesta.** Normalizar PnL por máximo absoluto para ratios y comprobar cero exacto en los dominios degenerados; eliminar epsilon de Calmar. No sustituirlo por otra constante arbitraria.

**Verificación o evidencia de aceptación.** dimensionless_ratios_are_invariant_to_currency_scale; tiny_valid_drawdown_is_not_clamped_to_zero

**Límites.** Verificado en escalas 1e-300, 1e-20, 1, 1e150 y 1e300 con ratios cerrados. No es prueba exhaustiva de todo f64.

### MX-08 — Momentos y sumas monetarias desbordaban antes del cociente

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:184`.

**Causa y condición.** Sumar ganancias/pérdidas y elevar desviaciones monetarias al cuadrado puede desbordar aun cuando media/sigma o PF sean representables.

**Consecuencia.** Un panel sobre entradas finitas producía Inf/Inf, cero o NaN artificiales, confundidos con propiedades del mercado.

**Reparación / ruta propuesta.** Estadísticos adimensionales sobre PnL normalizado; suma compensada de Neumaier para reducir cancelación. La unidad monetaria se restituye sólo donde corresponde, como la cola.

**Verificación o evidencia de aceptación.** dimensionless_ratios_are_invariant_to_currency_scale; large_finite_tail_does_not_overflow_intermediate_sum

**Límites.** La aritmética sigue siendo f64: no se garantiza precisión arbitraria ni se ocultan resultados realmente fuera de rango.

### MX-09 — Downside cuadrático podía desaparecer con Sortino finito

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:193`.

**Causa y condición.** Para [1,-1e-200], el cuadrado de la pérdida se redondea a cero. La base devolvía cero por epsilon; la primera reparación normalizada aún daba infinito.

**Consecuencia.** El ratio correcto bajo la convención mantenida es aproximadamente 1e200 en un año con dos trades: representable, pero distinto de ambas salidas defectuosas.

**Reparación / ruta propuesta.** Acumular la norma de pérdidas con hypot y dividir por sqrt(n), evitando formar primero cuadrados subnormales. El denominador incluye todos los trades.

**Verificación o evidencia de aceptación.** downside_does_not_underflow_before_a_representable_sortino

**Límites.** Regresión de segunda pasada: RED sobre candidato con Inf, GREEN con ratio/1e200≈1. No se presenta 1e200 como calidad de inversión.

### MX-10 — La cola redondeada no era exactamente el peor 5 % empírico

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:211`.

**Causa y condición.** El algoritmo tomaba ceil(.05*n) registros completos y promediaba. Para n=21 esto asignaba 9,52 % de masa en vez de 5 %.

**Consecuencia.** Con PnL [-100,-20,19×1], la salida era 60, pero el ES empírico de pérdidas al 95 % es (100+.05×20)/1.05=96.19047619047619.

**Reparación / ruta propuesta.** Integrar exactamente n/20 observaciones: cociente entero completo y resto fraccional. Mantener piso de presentación en cero y mínimo histórico n≥20.

**Verificación o evidencia de aceptación.** empirical_tail_uses_fractional_mass_not_ceil_average; empirical_tail_handles_ties_and_sample_floor

**Límites.** El cambio afecta el estimador diagnóstico en n no múltiplo de 20. No altera un límite de riesgo ni demuestra precisión poblacional del ES.

### MX-11 — La suma de cola desbordaba aunque su media fuera finita

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:215`.

**Causa y condición.** Dos pérdidas -1e308 suman -Inf, pero su media es -1e308. El código sumaba antes de dividir.

**Consecuencia.** CVaR podía marcar riesgo infinito por un intermediario aritmético, no por el valor del funcional.

**Reparación / ruta propuesta.** Normalizar únicamente la cola por su máximo absoluto, sumar con compensación y restaurar escala después de dividir por su masa.

**Verificación o evidencia de aceptación.** large_finite_tail_does_not_overflow_intermediate_sum

**Límites.** Se preservan ganancias en la cola antes del piso final: no se reemplazan todos los PnL positivos por cero, que definiría otra variable aleatoria.

### MX-12 — CAGR perdía información en el ratio de extremos

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:151`.

**Causa y condición.** final/inicial puede ser infinito o cero antes de la potencia anualizadora, aunque el resultado anual sea representable.

**Consecuencia.** 1e-300→1e300 en 1000 años devolvía Inf; el resultado es 10^0.6-1≈2.9810717055. El trayecto inverso tampoco es necesariamente ruina.

**Reparación / ruta propuesta.** Usar diferencia de logaritmos para razones extremas, ln_1p para cambios relativos pequeños y exp_m1 al restar uno. Mantener convención 365,25 días.

**Verificación o evidencia de aceptación.** cagr_avoids_intermediate_ratio_overflow_and_underflow

**Límites.** Fixture numérico de estrés, no escenario de trading ni respaldo de predicción a mil años. La duración real observada sigue siendo requisito.

### MX-13 — Turnover finito se perdía por división intermedia

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:163`.

**Causa y condición.** notional/capital podía desbordar antes de dividir por días. N=1e300, C=1e-10, D=1e6 tiene turnover 1e304, no infinito.

**Consecuencia.** La implementación agregaba un diagnóstico extremo falso a un resultado representable.

**Reparación / ruta propuesta.** Conservar división directa en rango normal y recurrir al dominio logarítmico ante overflow o underflow intermedio. Validar todos los dominios antes.

**Verificación o evidencia de aceptación.** turnover_avoids_intermediate_ratio_overflow

**Límites.** El fallback no limita rotación; Inf sigue posible si el resultado final genuino excede f64.

### MX-14 — Ratios 0/0 y sigma cero se presentaban como ceros medidos

Prioridad: P1. Estado: `fixed_local`. Evidencia: `crates/backtest-engine/src/metrics.rs:174`.

**Causa y condición.** Muestra constante, cero absoluto o ausencia de pérdidas disparaban ramas con valores sintéticos. Sigma cero no define un Sharpe finito, y PF=0/0 no es un PF de cero.

**Consecuencia.** La interfaz no distinguía falta de denominador de una estrategia con rendimiento/riesgo nulo.

**Reparación / ruta propuesta.** Sharpe con sigma cero queda NaN. PF y Sortino sin denominador pero numerador positivo conservan +Inf; 0/0 queda NaN. Calmar usa división explícita, sin esconder degeneración.

**Verificación o evidencia de aceptación.** all_zero_trades_have_undefined_risk_ratios_not_fabricated_zeros; zero_dispersion_has_no_finite_sharpe

**Límites.** La semántica queda documentada; no equivale a añadir una regla de admisión ni a convertir infinito en aprobación.

### MX-15 — El nombre Sharpe no especificaba que la variable era cash PnL

Prioridad: P1. Estado: `documented`. Evidencia: `crates/backtest-engine/src/metrics.rs:92`.

**Causa y condición.** La muestra es PnL neto monetario por cierre, no retorno normalizado por equity disponible. Distintos tamaños/exposiciones pueden cambiar la interpretación.

**Consecuencia.** Leer el campo como Sharpe de cartera sobrestima lo que se midió. La convención objetivo cero tampoco se deduce de operar cripto 24/7.

**Reparación / ruta propuesta.** Conservar API y agregar basis=cash_pnl_per_trade y annualization=iid_proxy en panel_line; documentación explicita unidad, varianza poblacional y benchmark cero como política.

**Verificación o evidencia de aceptación.** panel_exposes_cash_pnl_basis_and_proxy_annualization

**Límites.** Clarificación verificada, no sustitución por un Sharpe de cartera. Migración de consumidores y estimación financiera: MX-17/MX-23/MX-25.

### MX-16 — Veinte trades no certifican una cola bien estimada

Prioridad: P2. Estado: `documented`. Evidencia: `crates/backtest-engine/src/metrics.rs:112`.

**Causa y condición.** La base decía que con menos de 20 la cola no es estimable. El funcional empírico existe; n=20 sólo aporta una observación de masa de cola.

**Consecuencia.** Una regla de cobertura mínima podía interpretarse como teorema de imposibilidad o como garantía de confianza.

**Reparación / ruta propuesta.** Mantener MIN_TRADES_CVAR=20 por compatibilidad y describirlo como política explícita; explicar piso positivo frente a ES firmado.

**Verificación o evidencia de aceptación.** empirical_tail_handles_ties_and_sample_floor; metrics module documentation

**Límites.** No se estimó error estándar, intervalo ni estabilidad out-of-sample. Bajar el mínimo o automatizarlo requiere decidir precisión y dependencia objetivo.

### MX-17 — Anualización por conteo no modela dependencia ni reloj económico

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/metrics.rs:186`.

**Causa y condición.** sqrt(n/años) escala un proxy bajo supuestos de incrementos comparables no correlacionados. La API no recibe retornos temporales, autocovarianzas, duración/exposición ni benchmark por intervalo.

**Consecuencia.** Un número computado correctamente no valida su uso para comparar genomas con solapamiento, autocorrelación o tamaño variable.

**Reparación / ruta propuesta.** Proponer retornos de equity netos y alineados a un reloj declarado; estimar varianza de largo plazo/covarianza y evaluar incertidumbre con bloques temporales. Comparar ambas medidas fuera de muestra antes de migrar fitness.

**Verificación o evidencia de aceptación.** Fuente primaria Benhamou, sección 3.6.2; inspección de firma y factor anual

**Límites.** No implementado: faltan serie de equity/exposición y contrato de muestreo. No se introduce HAC con un lag arbitrario para aparentar corrección.

### MX-18 — Capital y estadísticas de trades usan poblaciones distintas

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:542`.

**Causa y condición.** PnL/trades/turnover se acumulan sólo si i≥warmup. Capital se lee de arena y DD se actualiza en cada iteración procesada, incluido el calentamiento. La función no reinicializa equity al iniciar la población reportada.

**Consecuencia.** Un CAGR/Calmar de un período puede aparecer junto a WR/PF/cola de otro. La diferencia puede afectar reconciliación y comparación de genomas.

**Reparación / ruta propuesta.** Definir inicio efectivo único; separar entrenamiento/calibración, warmup sin órdenes y evaluación; ledger de cierres y curva contable reconciliables. Añadir fixture con pérdida/ganancia antes y después del corte.

**Verificación o evidencia de aceptación.** Fuente leída: booktick_replay.rs:329,542,563,589

**Límites.** Desajuste estructural confirmado; no se cuantificó impacto con tape. Cambiarlo altera resultados del replay y exige rebaseline explícito, no editar el golden para ocultarlo.

### MX-19 — Precarga del prefijo y replay desde el inicio violan causalidad temporal

Prioridad: P0. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:268`.

**Causa y condición.** Se alimenta process_kline con hasta max(warmup_ticks,600) ticks antes de iniciar el for sobre ticks desde índice 0. El corte de estadísticas usa min(warmup_ticks,len/10), una frontera distinta.

**Consecuencia.** Los primeros eventos pueden usar estado construido con observaciones posteriores de ese mismo prefijo. Un backtest determinista puede seguir teniendo look-ahead. Además 600 ticks no equivalen a 512 cierres de un minuto.

**Reparación / ruta propuesta.** Consumir warmup una sola vez, sólo con historia anterior a la evaluación, y empezar el replay después de esa frontera; readiness por observaciones temporales realmente cerradas. Probar invariancia causal al modificar un sufijo futuro.

**Verificación o evidencia de aceptación.** Fuente leída: booktick_replay.rs:268,329,331; llamadas process_kline anteriores al bucle

**Límites.** Confirmado el orden de alimentación; no se afirma un tamaño de alpha espurio ni rentabilidad inflada medida. La sensibilidad concreta del score y el nuevo golden requieren prueba dedicada.

### MX-20 — El span solicitado no necesariamente es el span procesado

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:595`.

**Causa y condición.** El caller resta timestamps del último y primer tick del archivo; el bucle omite cotizaciones inválidas y puede cortar antes por insolvencia. saturating_sub convierte orden invertido en cero.

**Consecuencia.** CAGR, frecuencia anual y turnover pueden usar tiempo que no produjo eventos válidos, o esconder desorden temporal. El fix de duración cero sólo impide inferir ruina falsa.

**Reparación / ruta propuesta.** Registrar primer/último evento elegible y motivo de término; verificar orden y cobertura del feed; distinguir duración de observación, duración activa y duración de riesgo.

**Verificación o evidencia de aceptación.** Fuente leída: booktick_replay.rs:331,563,595

**Límites.** No basta reemplazar por tiempo del último trade: eso excluiría inactividad real. La convención debe fijarse antes de cambiar la métrica.

### MX-21 — Turnover reconstruye ambas patas con el precio de salida

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:559`.

**Causa y condición.** Se acumula dos veces qty*mid del cierre. El nocional real sería suma de cada ejecución de entrada/salida, con su precio y cantidad; fills parciales o abiertas al final también requieren tratamiento.

**Consecuencia.** La aproximación es dependiente de dirección y movimiento. Comprar una unidad a 100 y cerrar a 110 opera 210, mientras esta fórmula registra 220.

**Reparación / ruta propuesta.** Acumular nocional desde fills ejecutados, con id único y timestamps; reconciliar contra ledger y costos por pata. Mantener esta cifra marcada como proxy mientras no exista ese dato.

**Verificación o evidencia de aceptación.** Fuente leída: booktick_replay.rs:543-559; ejemplo algebraico independiente

**Límites.** No se modifica fee_est ni se afirma que comisiones reales estén ausentes de pnl_net: el core ya puede cobrarlas. El fallo auditado es de estimación/linaje del panel.

### MX-22 — Option::or descarta un segundo cierre si ambos existen

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:521`.

**Causa y condición.** Tanto actualización de posterior como estadísticas usan c1.or(c2). La semántica de or conserva c1 cuando es Some, aunque c2 también sea Some.

**Consecuencia.** Si los dos caminos cierran en un evento, el ledger estadístico y la adaptación pierden una observación, mientras el capital puede haber contabilizado ambas.

**Reparación / ruta propuesta.** Procesar ambos outcomes exactamente una vez con identidad de posición/fill; probar cierre simultáneo y evitar duplicados cuando dos caminos refieran al mismo outcome.

**Verificación o evidencia de aceptación.** Inspección de las dos expresiones en líneas 521 y 543

**Límites.** Defecto condicional de agregación confirmado; frecuencia y alcanzabilidad simultánea en configuración actual no medidas. No declarar que todas las ejecuciones pierden un cierre.

### MX-23 — Panel nuevo desconectado de consumidores históricos

Prioridad: P1. Estado: `open`. Evidencia: `src/bin/evolution.rs:328`.

**Causa y condición.** ReplayStats conserva sharpe antiguo con epsilon y cálculo monetario independiente. evolution copia rep.sharpe a out_stats[4]; backtest_windows imprime s.sharpe. La búsqueda del panel/campos nuevos sólo encuentra módulo y pruebas, además de su asignación en replay.

**Consecuencia.** Una reparación del panel no corrige automáticamente los resultados mostrados por esos bins ni la ruta de métricas de evolución. No hay prueba de impacto productivo de este cambio diagnóstico.

**Reparación / ruta propuesta.** Inventariar consumidores, versionar la semántica de cada métrica y añadir pruebas productor→consumidor. Migrar tras acordar retorno/clock/uncertainty, no reemplazar silenciosamente el slot de fitness.

**Verificación o evidencia de aceptación.** booktick_replay.rs:585; evolution.rs:328,455,558; backtest_windows.rs:287; rg sobre src y crates

**Límites.** No se afirma que out_stats[4] determine por sí solo el fitness final. Aquí se confirma transporte/visualización de la métrica antigua, no sensibilidad completa del optimizador.

### MX-24 — Equivalente de tres días compone una tasa diaria aritmética

Prioridad: P1. Estado: `open`. Evidencia: `src/bin/backtest_windows.rs:263`.

**Causa y condición.** print_window calcula daily=ROI/days y luego (1+daily/100)^3. Si se pretende un equivalente geométrico, se requiere (1+ROI)^(3/days)-1, con dominio y cashflows definidos.

**Consecuencia.** Con capital 100→121 en dos días, el código muestra 34.9232625 % a tres días; el equivalente geométrico es 33.1 %. El sesgo crece con retornos/duración.

**Reparación / ruta propuesta.** Centralizar crecimiento geométrico de horizonte y etiquetar extrapolación histórica, nunca previsión. Probar casos de pérdida, ruina, ausencia y duración cero.

**Verificación o evidencia de aceptación.** Ejemplo independiente 1.105^3-1 frente a 1.21^1.5-1; fuente backtest_windows.rs:263

**Límites.** No reparado en este lote para mantener el alcance del módulo puro. Tampoco convierte una muestra corta en evidencia de duplicación sostenible.

### MX-25 — Panel por símbolo no acredita riesgo multiactivo de cartera

Prioridad: P1. Estado: `open`. Evidencia: `crates/backtest-engine/src/booktick_replay.rs:239`.

**Causa y condición.** El contrato declara un símbolo, coin 0. Una lista de PnL por cierres no contiene la trayectoria sincronizada de exposición, liquidez, funding y correlaciones del universo.

**Consecuencia.** Agregar o promediar Sharpe/CVaR de símbolos no reconstruye los de la cartera. El panel no certifica adaptación multivariante continua ni paridad demo/live.

**Reparación / ruta propuesta.** Construir ledger multiasset ordenado por event time, valuación consistente y curvas mark-to-market con incertidumbre/capacidad; evaluar agregación antes de extraer métricas y comparar modos con misma identidad de datos.

**Verificación o evidencia de aceptación.** Contrato explícito de run_booktick_replay y firma ex_post_metrics

**Límites.** Un harness por símbolo puede ser intencional y útil; el defecto es usarlo como evidencia suficiente de la propiedad global. No se dice que todo el repositorio sea monoactivo.

### MX-26 — El candidato dejaba que -0 de DD invirtiera Calmar

Prioridad: P2. Estado: `fixed_candidate_regression`. Evidencia: `crates/backtest-engine/src/metrics.rs:138`.

**Causa y condición.** Tras eliminar saneamientos engañosos, el primer candidato aceptó -0.0 como no negativo y lo conservó. IEEE positivo/-0 produce -Inf.

**Consecuencia.** Regresión introducida durante esta reparación, no defecto atribuido retrospectivamente a la base: crecimiento positivo aparecía con Calmar negativo infinito.

**Reparación / ruta propuesta.** Canonicalizar cero válido de DD a +0 con abs después de rechazar negativos/no finitos. No aplicar abs a un DD negativo real.

**Verificación o evidencia de aceptación.** signed_zero_drawdown_cannot_reverse_calmar_sign

**Límites.** RED del candidato: -Inf; GREEN: +Inf. Se conserva esta evidencia de segunda pasada en vez de declarar que el primer arreglo fue suficiente.

## Verificación reproducible y alcance de la evidencia

Corte de reparación: `a0ad0a0b`, sobre `99192101`. Integración local de
main `7796326a` en `7bdd39a8`, conservando ambos avisos del único conflicto
documental y el registro de modelos de GLM. Cargo.lock añadió únicamente
la dependencia sha2 0.10.9 de god-engine-core que el manifiesto ya exigía.

| Ejecución | Resultado observado |
| --- | --- |
| RED contra base, 19 contratos | 1 aprobado, 18 fallidos; incluye panic NaN y cola 60 vs 96,19 |
| Segunda pasada sobre candidato, 21 contratos | 19 aprobados, 2 fallidos; downside y -0 detectados |
| Contratos finales del módulo | 21 aprobados, 0 fallidos |
| Ampliada antes de integrar GLM | 37 biblioteca +21 métricas +25 labels +3 riesgo espectral =86 aprobados; 0 fallidos/ignorados |
| Check workspace all-targets previo | Correcto, 33,37 s; warnings preexistentes conservados |
| Check all-targets antes de cerrar merge | Correcto, 30,19 s; comparación contra ambos padres |
| Repetición tras integración | Se registra en el cierre de este informe |

Comandos: `cargo test -p backtest-engine --lib --test ex_post_metrics_contract --test label_evidence_contract --test spectral_risk_contract -- --test-threads=1`
y `cargo check --workspace --all-targets`. No se ejecutó el conjunto T-1
ni mediciones manuales de tapes; no cuentan como aprobadas ni como fallidas.
Los golden existentes pasan en la ejecución previa; no se cambió su valor.
No se infiere rendimiento HFT ni prueba estadística de alpha de una compilación.

Huellas SHA-256 de archivos físicos validados (antes/después de merge):
`metrics.rs = CB48AA8C7749EF2754498A07B2D2078E1C825EC54995FE24F678FEDD6378BB98`;
`ex_post_metrics_contract.rs = 49F4E544DD18CBB59F00D16DD791B94A3A9EE79723E68E53131BAA30F87E3D9D`;
`booktick_replay.rs = 321F607898D3E4A997D888215796BC40475500D6F66182D35D9720F1EE66696D`.
El caller se conserva intacto. No se atribuyen a GLM los defectos del candidato MX.

## Cobertura y evaluación de calidad

Los denominadores son unidades del alcance declarado, NO archivos del
repositorio ni porcentaje de certificación. 0/N significa que no quedan
defectos demostrados en los controles enumerados; no prueba todos los casos.
Las categorías se solapan y no se suman. Estado general **Needs revision**,
completitud **Partial**: falta evidencia de impacto predictivo y causalidad
end-to-end. Registro interno: `.firecrawl/mx-audit/coverage.json`, fuera del
artefacto de entrega; tabla obtenida con summarize-coverage.mjs.

### Calidad y utilidad del artefacto

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Utilidad y completitud | 3 / 4 | Pendientes causalidad, riesgo conjunto y conexión a consumidores. |
| Claridad analítica | 0 / 4 | Variables, ausencias, estados y expedientes explicitados. |
| Consistencia visual/interacción | N/A | Código y documentos, sin dashboard gráfico evaluado. |

### Corrección y robustez analítica

| Category | Observed defects | Assessment |
| --- | --- | --- |
| Autoridad/confianza de fuentes | 4 / 9 | CAGR/Calmar/DD/turnover heredan período, curva o nocional del caller no reconciliados. |
| Exactitud del cálculo puro | 0 / 9 | Contrastes cerrados y dominios verificados; no certificación de inputs productivos. |
| Concordancia dentro de gráficos | N/A | No hay gráficos cuantitativos ni tooltips. |
| Detalle interactivo de fuentes | N/A | No existen tales superficies; referencias estáticas conservadas. |
| Consistencia entre componentes | 2 / 3 | Población mixta y consumidores legacy; MD/JSON contrastados por ID/estado. |
| Controles de calidad de entradas | 0 / 9 | Sólo controles del módulo puro; cobertura del feed fuera de esta unidad. |
| Sustento de conclusiones | 3 / 5 | No se sostienen extrapolaciones a cartera/causalidad; impacto predictivo además no verificado. |

## Hoja de ruta priorizada y criterios de cierre

1. **P0, MX-19: causalidad del replay.** Un sufijo futuro modificado no debe
   alterar features ni decisiones de un prefijo ya observado. Warmup histórico
   procesado una sola vez; readiness por tiempos cerrados. Registrar nueva
   semántica antes de rebaseline, sin convertir el valor viejo en un objetivo.
2. **P1, MX-18/20/21/22: ledger único.** Cada fill/cierre tiene identidad,
   instante de observación y contabilización. Equity y PnL reconcilian dentro
   de la misma población; entradas abiertas y fallos de cobertura explícitos.
   Contrastar nocional/fees por pata, cierre doble y corte temprano.
3. **P1, MX-17/23/25: métrica → selección → ejecución.** Especificar retorno,
   numerario y reloj de riesgo para cartera multiasset; probar dependencia,
   incertidumbre y consumidor. Mantener proxies visibles mientras se compara,
   no conectar automáticamente un ratio anual a un slot histórico por trade.
4. **P1, MX-24: presentación geométrica.** Corregir equivalente de horizonte
   con dominio validado y etiqueta de extrapolación, no promesa. Cerrar con
   casos deterministas de retorno positivo, negativo, ruina y ausencia.
5. **Evidencia de mejoras financieras.** Walk-forward purgado con versiones
   de datos/genoma/modelo, costos, funding, profundidad, colas de latencia,
   sensibilidad a tamaño/capacidad y comparación demo/live sincronizada.
   No es necesario recurrir a ecuaciones de problemas del milenio para
   resolver estos defectos identificados; complejidad sin validación puede
   empeorar identificabilidad y costo.

La resolución de un defecto no habilita operar ni demuestra la meta de
crecimiento. No se eliminaron vetos, límites o kill-switches.

## Integración y coordinación

Rama de trabajo `feat/quant-sr-codex-metricas`; trabajo previo ST/TE conservado.
La política τ de TH continúa en su rama y NO se mezcla silenciosamente.
GLM trabajó en `glm/xlviii-g-model-registry`; se leyeron los nuevos cambios
para integrar sin colisiones. La inclusión de un manifest no certifica por sí
sola calidad predictiva, disponibilidad en serving ni el enlace de cada señal
al hash cargado. No se regeneró el inventario de modelos.

PR #10 abierta y #20 abierta/draft al consultar; no se creó, comentó ni modificó
una PR. Aviso compartido ignorado
`.firecrawl/coordination-codex-metricas-2026-09-29.md`, sin acuse demostrado.
Las ramas locales ST/TE/TH/backups no están integradas en main y se preservan.
Una rama aún usada por otro editor no se borra por estar momentáneamente
alcanzable desde main. No se declara completado push/merge/publicación:
permanece pendiente autorización específica para publicar en repositorio
público tras el rechazo previo de aprobación.

## Cierre de validación

Validación sobre `7bdd39a8`: 86 aprobadas, 0 fallidas, 0 ignoradas
(37 biblioteca, 21 MX, 25 labels, 3 riesgo espectral). Biblioteca: 67,19 s;
los tres contratos: 0,01/0,01/0,05 s. Golden existente intacto y aprobado.
El build de esta repetición tomó 1m27s. No son tiempos de latencia del motor.

Main remoto verificado por API: `7796326acc2ff8a4cd51a60c1728a327727d4b5f`.
Ambos padres están preservados; no hubo conflicto de código. El único ajuste
de dependencia generado en Cargo.lock corresponde al manifest recibido.
Los campos/métodos y el caller tienen las huellas arriba indicadas.

MD y JSON: 26 IDs, títulos y estados alineados; JSON parseable. Cobertura
calculada por el helper sin inventar denominadores de todo el proyecto.
Los informes históricos y memoria recibieron adendas, no sustituciones.
Sigue abierta la integración/publicación de la rama Codex y los nueve
expedientes marcados open. No hay certificación total ni promesa de retorno.

## 2026-09-29 — Publicación ST/TE/MX autorizada por el operador

El operador respondió «Hazlo» a la petición explícita de publicar la rama
`feat/quant-sr-codex-metricas`, incluidos los cambios e informes ST/TE/MX,
en el repositorio PÚBLICO `Jhona-la/Trader-Gemini` y tramitar su integración
a main. Queda levantada la anterior falta de autorización de publicación;
los avisos anteriores se conservan como registro histórico, no como estado
vigente de permisos.

Alcance de publicación: los cambios acumulados frente a main7796326a,
incluidas sus pruebas y documentos. TH6209704a permanece separada por su
política pendiente; no se modifica el Cargo.lock sucio del checkout compartido.
No se amplía esta autorización a operar, entrenar o promover modelos.

La publicación no resuelve los hallazgos abiertos ni acredita rentabilidad.
Se revisan diferencias, PR/comentarios y requisitos de integración. GitHub
no reporta protección/ruleset de main ni existen workflows versionados en
este corte; ausencia de CI no se describe como «CI verde». Se realiza además
regresión local conjunta antes de integrar. El resultado definitivo y la URL
de la PR quedarán en el cierre de publicación.
