# R4 — Fundamentos de medición, selección y autoevolución

Fecha local: 2026-10-07, America/Bogota. Autor: Codex / quant_foundations.
Base revisada: `adeb8d1b1f8171b14fffa8abc9f54c396e4794dd`.

Esta revisión reabre contratos que olas anteriores daban por cerrados. Se
inspeccionaron los rangos citados, sus productores y sus consumidores; no se
declara recorrido completo de estos archivos ni de todo el repositorio.
No se cambió runtime, Cargo, configuración, arming ni procesos vivos.

Evidencia reproducible: `scripts/audit_selection_assumptions.py` y
`docs/audit/FUNDAMENTOS_R4_2026-10-07.json`. El script es un **port matemático
Python y un modelo de trazas**, no ejecuta Rust ni reproduce operaciones del
motor. Los resultados prueban limitaciones de supuestos/expresiones, no una
tasa de fallo observada del bot. El JSON incluye HEAD y SHA-256 de fuentes.

## Meta y contrato de evidencia

La meta solicitada de duplicación en 72 h equivale a `ln(2)/3 =
0.2310490602` de crecimiento log por día y a un retorno diario compuesto
constante de `2^(1/3)-1 = 25.992105%`. Es un objetivo, no un resultado validado.

La cifra histórica de Sharpe diario aproximadamente `0.68` sólo surge de
un modelo idealizado: GBM de drift y volatilidad conocidos y constantes,
tipo libre de riesgo cero, fricción cero, sin límites de apalancamiento y
Kelly pleno. Si `g(f)=f*mu-f²*sigma²/2`, entonces `f*=mu/sigma²` y
`g*=mu²/(2*sigma²)`; igualar `g*` a `ln(2)/3` da
`mu/sigma=sqrt(2*ln(2)/3)=0.6797779934` en unidades diarias del modelo.
No es una condición suficiente en mercados con saltos, costos, estimadores
inciertos, liquidaciones o restricciones. Tampoco garantiza duplicación en
cada ventana de 72 h. El plan debe medir distribución OOS del crecimiento
log neto, incertidumbre, drawdown, ruina y capacidad. No se validó aquí la
comparación histórica de Sharpe de fondos presente en otros planes.

## Hallazgos confirmados

| ID | Severidad | Contrato roto | Alcance / activación |
|---|---|---|---|
| R4-Q1 | HIGH, condicionado | El contador Darwin se reinicia en cada ronda del host | Legacy off por defecto; `ENABLE_LEGACY_DARWIN_DAEMON=true`; promoción además requiere `ENABLE_ONLINE_DARWIN_MUTATION=true` o `1` |
| R4-Q2 | HIGH, condicionado | Retornos llamados MTM y drawdown omiten pérdidas flotantes | Evaluador Darwin; mismos flags para alcanzar publicación |
| R4-Q3 | MED, condicionado | Serie llamada periódica mezcla reloj de 1 s y reloj de cierres | Evaluador Darwin; no es una serie homogénea |
| R4-Q4 | HIGH de validación | DSR no controla dependencia temporal por incluir skewness/kurtosis | Módulo canónico compartido por Darwin y LiveEvolutionDaemon; no se limita al legacy |

### R4-Q1 — La acumulación Ω15 H1-3 no sobrevive al caller

Evidencia en la base: `src/bin/god_engine.rs:1953-1965` construye una
instancia exterior y **otra `DarwinDaemon::new(...)` dentro de cada vuelta**.
`darwin.rs:400-403` inicializa `cumulative_trials` a cero. La evaluación
`darwin.rs:636-644` suma `20*5=100`, pero la instancia se descarta al terminar.
Por tanto tres rondas elegibles del host usan `100,100,100`, no
`100,200,300`. No hace falta reiniciar el proceso para perder la memoria.

El test `omega15_h1_3_darwin_daemon_multiplicidad_acumulada_monotona`
(`darwin.rs:866`) modifica el contador de una misma instancia manualmente;
no reproduce el ciclo de vida del host y no detecta este defecto.

Seguimiento 2026-10-07: el agente coordinador anunció la corrección del
caller con un `Arc<DarwinDaemon>` persistente en la rama de recuperación.
**Estado en este recibo: propuesta/en preparación; pendiente de validar**.
Los números, hashes y líneas anteriores describen la base `adeb8d1b`, no
certifican ni rechazan una implementación posterior. El cierre exige recibos
del contrato de ciclo de vida, compilación y T-1 aplicable sobre el árbol
integrado; no basta con anunciar el arreglo.

Corrección mínima propuesta al dueño de host: envolver el daemon una sola
vez en `Arc<DarwinDaemon>` fuera del loop, `Arc::clone(&daemon)` dentro y
mover ese clon a `spawn_blocking`. Conservar el `.await` evita solapar rondas.
La subauditoría no editó runtime; la coordinación R4 aplicó después este
cambio mínimo en el host, aún pendiente del recibo de compilación/regresión
e integración. Prueba de comportamiento deseable: dos invocaciones de la
misma función de reserva/registro utilizada por el caller producen 100 y
200; un contrato de ciclo de vida debe fallar si el loop vuelve a `new()`.
Después diseñar persistencia por experimento/linaje entre reinicios. El
contador en RAM, aun arreglado, no constituye por sí solo una prueba de
validez anytime ni de independencia del holdout.

### R4-Q2 — El evaluador usa saldo realizado, no equity MTM

`darwin.rs:352-354` describe una serie marcada a mercado, pero
`darwin.rs:364,379,388` lee exclusivamente `arena.unified_capital`.
El productor confirma la semántica: `god-engine-core/src/lib.rs:3602-3604`
le añade `net_realized_pnl` al cerrar y `:7519` le resta el fee de entrada;
las posiciones abiertas actualizan **otro campo**,
`coin.metrics.pnl_unrealized`, en `:4045-4064`. El fitness sólo actualiza
pico y drawdown al cerrar (`darwin.rs:369-372`).

Contraejemplo de cuenta, sin pretender que el motor genere esas órdenes:
saldo 100, flotante -40 en t=1 s y recuperación a 0 en t=2 s. Los retornos
del evaluador son `0,0`; las equities son `60,100`. Añadiendo cierres que
llevan saldo a 101 y luego a 100, el drawdown registrado es 0.990099%,
mientras el drawdown marcado fue 40%. Las pérdidas abiertas terminales
también quedan fuera del capital final si no se cierran antes del corte.

Corrección propuesta: contrato único de equity de replay que sume saldos,
marcado de **todas** las posiciones y costos exigibles, calculado sobre un
snapshot coherente. No sumar ciegamente `pnl_unrealized` sin auditar que
agrega todos los slots y su frescura. Medir drawdown en cada observación de
equity; política explícita para posiciones aún abiertas al terminar OOS.
Prueba de comportamiento pendiente: posición abierta pierde y recupera sin
cierre; el marcado y drawdown deben reflejar la excursión, también con dos
activos y varios slots. El defecto de marcado es directo; el impacto sobre
promociones reales todavía requiere replay.

### R4-Q3 — Reloj de muestreo dependiente de cierres

`darwin.rs:375-385` emite cuando `time_elapsed || closed.is_some()` y
siempre reasigna `last_sample_ts = tick.timestamp`. Un cierre a 2100 ms y
otro a 2200 ms producen intervalos de 100 ms y desplazan la siguiente
observación regular a 3200 ms. La traza reproducible emite tiempos
`1000,2000,2100,2200,3200`, con deltas `1000,1000,100,100,1000`.
Cierres simultáneos pueden producir observaciones con duración cero.
Un salto largo de mercado produce una sola observación larga, no el
número de intervalos de 1 s que afirma el comentario.

Consecuencia: comparar momentos por observación mezcla duraciones y la
frecuencia elegida por la estrategia; `N>=20` no basta para afirmar veinte
observaciones homogéneas. El test Ω15 de precios constantes sólo comprueba
cantidad de observaciones, no duración, equity, cierre simultáneo ni huecos.

Corrección propuesta: reloj exógeno con buckets temporales, agregar cierres
dentro del bucket y política explícita de frescura/huecos. Mantener retornos
por operación como otra métrica con unidad propia. No rellenar huecos con
información futura ni declarar independientes los ceros. Probar invariancia
al particionar un fill en varios cierres del mismo instante, regularidad de
timestamps y ausencia de lookahead.

### R4-Q4 — Cuatro momentos no corrigen autocorrelación

`selection_stats.rs:82-93,123-131` calcula el error estándar sólo a partir
de N, Sharpe, skewness y kurtosis. No hay autocovarianzas; la estructura
`ReturnMoments` ni siquiera conserva el orden de los datos. Ω15 H1-4 corrige
no-normalidad bajo sus supuestos, pero **no** resuelve dependencia temporal.
La ruta primaria llama la misma función desde
`evolution-engine/src/online_daemon.rs:1831-1835`.

El port de la fórmula sobre `[0.0026,-0.0014]*20` da DSR `0.2483574281`.
Repitiendo diez veces cada observación, sin nuevas innovaciones
independientes, da `0.9997245805`. Un experimento sintético estacionario
AR(1) de media poblacional cero, 500 observaciones y `n_trials=100`, semilla
20261007, obtiene:

| Autocorrelación AR(1) | Series que pasan DSR >= 0.95 / 1000 |
|---|---:|
| 0.00 | 0 |
| 0.90 | 189 |
| 0.95 | 245 |

No son 100 búsquedas ni una estimación FWER del sistema: es un diagnóstico
de calibración del gate sobre series nulas individuales, usando el mismo
parámetro de multiplicidad para las tres familias. La magnitud contradice
calificar universalmente la dependencia como optimismo leve; no cuantifica
la autocorrelación ni la tasa de promociones falsas de los datos del bot.

Corrección propuesta: estimación de varianza de largo plazo para el
**estimador de Sharpe**, o bootstrap por bloques apropiado a la dependencia
y al diseño de selección. Bajo condiciones estacionarias, la función de
influencia de Sharpe es
`psi_t=(r_t-mu)/sigma - mu*((r_t-mu)^2-sigma^2)/(2*sigma^3)`;
un estimador HAC debe tratar sus autocovarianzas, no sólo multiplicar N por
una heurística basada en autocorrelación de retornos. La elección de lags,
ventanas y cambio de distribución requiere validación. Antes de promover:
nulos IID, AR, volatilidad agrupada, colas pesadas, datos multiactivo
asíncronos y medición de error de la familia completa de selección.

## Discrepancias de modelo que requieren diseño, no parche mecánico

**Varianza entre ensayos.** El paper DSR define el benchmark con la
varianza de los Sharpes **entre ensayos** y su número efectivo independiente
(Bailey–López de Prado 2014, pp. 8–9 y apéndice 3). El código de
`selection_stats.rs:129` sustituye esa dispersión por el error estándar
del ganador. Pueden coincidir bajo un nulo homogéneo adicional, pero no
se ha demostrado aquí para genomas heterogéneos. El API actual sólo recibe
momentos del ganador y cantidad de ensayos: dos familias con igual ganador
y N pero distinta dispersión reciben el mismo DSR. No se propone inventar
una dispersión; registrar distribución de resultados y contrastar el modelo
nulo, las correlaciones entre candidatos y el costo de prescreen.

**Holdout adaptativo.** Un corte 50/50 y un contador ascendente no prueban
por sí solos generalización causal. Ventanas reutilizadas, modelos cargados
dinámicamente y realimentación de promociones exigen linaje de datos/modelos
y evaluación externa intacta. El comentario que dice que el split garantiza
generalización (`darwin.rs:442-443`) excede la evidencia. Esto es deuda de
validación, no se afirma haber demostrado una fuga específica nueva.

**Trasplantes de física/cuántica.** Requerir para cada teoría: observable,
unidades, operador, hipótesis, estimador, costo, baseline y predicción
falsable OOS. Descomposición de Hodge sobre grafos, ecuaciones de difusión
y filtros espectrales pueden tener contratos útiles sin resolver sus
problemas homónimos del milenio. La analogía física y el nombre del módulo
no certifican ventaja de trading ni computación cuántica real. La página
oficial de Clay consultada sigue marcando Yang–Mills/mass gap como abierto;
no extrapolar ese estado a todos los problemas sin comprobación individual.

## Orden recomendado de fases

1. **F0, objetivo y lenguaje:** unidades de la meta, distribución y no
   promesa; separar tests de expresividad T-1 de resultados financieros OOS.
2. **F1, medición:** equity, costos, clocks, posiciones terminales y muestras
   faltantes antes de recalibrar Sharpe/DSR o vetos.
3. **F1/F5, inferencia:** nulos dependientes y familia de ensayos; HAC o
   bloques con hipótesis documentadas; holdout externo y linaje reproducible.
4. **F3/F5, autoevolución:** ciclo de vida del contador, commit de promoción
   antes de publicación, monitoreo y rollback; paridad replay/vivo.
5. **F2, modelos avanzados:** sólo incorporar teorías cuya ablación mejore un
   baseline externo con costo y riesgo medidos. Mantener el continuo
   multiactivo/temporal sin confundir continuidad con información disponible.
6. **F8:** contratos Rust que fallen antes del arreglo; checks all-targets,
   tests de zona, paridad y T-1 del commit exacto según protocolo del repo.

Toda fase mantiene inventario **por archivo** con hash, rangos inspeccionados,
invariantes, pruebas, hallazgos, dueño y estado. Ningún test de este informe
habilita operación ni relaja vetos. Esta ola aporta diagnóstico/documentación
y deja las correcciones del pipeline para una ola coordinada con validación.

## Reproducción

Desde la raíz del checkout de esta revisión:

```text
python scripts/audit_selection_assumptions.py --output docs/audit/FUNDAMENTOS_R4_2026-10-07.json
```

Usa sólo biblioteca estándar. La semilla, orden de escenarios y parámetros
quedan en el script/JSON; pequeñas diferencias entre versiones de las
aproximaciones normales no cambian los contraejemplos. No se ejecutó Cargo,
el motor, T-1 ni un ensayo con dinero en esta subauditoría.

## Recibo de revisión independiente del plan y ledger

Se revisó la versión en elaboración de
`PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md`, el diff de
`PLAN_MAESTRO_QUANT_SR_2026-10-05.md` y el de
`PLAN_MAESTRO_SINCRONIZACION.md`, más el generador de inventario y su JSON.

Comprobación independiente con `git ls-tree` y el `Cargo.toml` del SHA base:
1.434 rutas, 460 archivos `.rs`, 655 rutas `graphify-out/`, 23 miembros del
workspace más el paquete raíz. El generador clasifica 656 entradas como
generadas porque incluye otra ruta fuera de `graphify-out/`; no contradice
el conteo de 655 de ese directorio. `audit_inventory.py check` terminó con
exit 0, cobertura exacta y cero errores. Las 1.434 filas tienen estado
`inventariado`; no hay afirmación automática de revisión realizada.

La nueva tasa por segundo `2.67417894e-6` es correcta. La ecuación
`exp(a+b*ln(tau/1ms))` coincide con `HorizonCurve::eval` para su dominio
positivo relevante; el código además aplica un suelo de `1e-6` al valor
numérico en milisegundos.

Observaciones enviadas al coordinador para reconciliar antes de publicar:

- El plan histórico conservaba afirmaciones específicas de H1-2 como MTM
  homogéneo y H1-3 como contador persistente/anytime. Requieren notas R4 junto
  a esas afirmaciones; una advertencia genérica en cabecera no basta.
- Las fases nuevas R0–R9 y las clasificaciones históricas F0–F8 del ledger
  requieren una correspondencia explícita. No son el mismo identificador.
- `audit_inventory.py generate --previous` conserva el estado si el blob y
  modo propios no cambiaron. No modela dependencias: un consumidor intacto
  puede requerir revalidación tras cambiar su productor. La invalidación
  semántica transitiva necesita recibos/manualidad o un grafo adicional;
  `check` sólo acredita metadatos, como declara correctamente el script.
- La banda `0.01s..1e6s` del plan histórico no debe confundirse con la banda
  operable `TAU_ANCHOR_FAST_MS..TAU_ANCHOR_SLOW_MS` de la base: 30s..12h.
  La malla representable, la banda propuesta y la observada son objetos
  diferentes; deben quedar etiquetados.

Estas observaciones describen la versión inspeccionada en elaboración, no
atribuyen acuse ni cierre posterior. No se ejecutaron pruebas Rust ni Cargo
durante esta revisión cruzada.

## Fuentes primarias consultadas

- [Lo (2002), The Statistics of Sharpe Ratios, artículo original](https://traders.berkeley.edu/papers/The-Statistics-of-Sharpe-Ratios.pdf): distingue inferencia IID de retornos estacionarios dependientes; la agregación temporal depende de autocorrelaciones.
- [Bailey y López de Prado (2014), The Deflated Sharpe Ratio, versión de los autores](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf): separa no-normalidad del ganador, dispersión entre ensayos y multiplicidad efectiva; el split solo no elimina sobreajuste.
- [Dwork et al. (2015), Generalization in Adaptive Data Analysis and Holdout Reuse](https://arxiv.org/abs/1506.02629): reutilizar el holdout adaptativamente necesita control adicional.
- [Busseti, Ryu y Boyd (2016), Risk-Constrained Kelly Gambling](https://arxiv.org/abs/1603.06183): formula crecimiento log con restricciones de riesgo de drawdown; una teoría candidata a contrastar, no evidencia de rentabilidad de este sistema.
- [Clay Mathematics Institute, Yang–Mills & the Mass Gap](https://www.claymath.org/millennium/yang-mills-the-maths-gap/): estado y formulación oficial del problema; no constituye una validación de analogías financieras.
