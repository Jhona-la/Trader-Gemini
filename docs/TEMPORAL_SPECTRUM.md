# ESPECTRO TEMPORAL CONTINUO (F8) — diseño y migración

> Directriz del operador: el binario scalp/swing es un corte arbitrario del
> tiempo. El sistema debe observar TODO el espectro en conjunto, con valores
> de trading apropiados a cada horizonte, calculando continuamente.

## 1. Fronteras físicas honestas (qué significa "de 1ns a 100 años")

- **Cota inferior ~1ms**: el inter-arribo de ticks de Binance Futures y el RTT
  de red. Por debajo de 1ms no existe DECISIÓN operable — solo el proceso
  interno del motor, que ya corre a cadencia de nanosegundo por evento (el
  espectro se actualiza en CADA evento).
- **Cota superior ~2.18 años**: los ciclos macro relevantes para futuros
  USDT-M. "100 años" existe como CONTEXTO (las series FRED de la capa macro
  llegan a décadas), pero no como horizonte operable con data de 2019+.
- **Entre ambas: CONTINUO log-espaciado base 4** — 19 escalas sin huecos ni
  bandas prohibidas: 1ms, 4ms, …, ~18.6h, ~3.1d, …, ~795d.

## 2. Qué existe hoy (post-F8 núcleo)

### `quantum-arena/src/temporal_spectrum.rs`
- `TemporalSpectrum`: 19 escalas por símbolo; por escala: EWMA(τ), vol DE LA
  DESVIACIÓN a esa escala (denominador estadísticamente correcto: caminata
  aleatoria ⇒ z ~ O(1) en todas las escalas), momentum z, señal tanh(z),
  persistencia (autocorrelación de signos de sorpresas consecutivas —
  tendencia ⇒ +1, reversión ⇒ −1, ruido ⇒ 0).
- **Fusión por paridad de riesgo**: score = Σ w_i·señal_i con w_i ∝ 1/vol_i —
  cada horizonte aporta según su ruido, no por pertenecer a un bucket.
- `signal_at(τ)`: interpolación log-lineal — el espectro es una FUNCIÓN
  continua, no una lista.
- `HorizonCurve`: param(τ) = exp(a + b·ln τ) — el gen del que todo se deriva.

### Genoma: las curvas mandan
- `tp_horizon_curve` y `sl_horizon_curve` (a, b) por familia; serialización
  con defaults compatibles (genomas viejos cargan sin ruptura).
- `apply_to_arena`: TP/SL de las bandas fast/slow se DERIVAN de las curvas —
  los campos `scalp_*`/`swing_*` son ahora VISTAS del continuo. Todo lector
  del arena (core, OCO, sizing) recibe valores del continuo SIN cambios.
- El GA muta (a, b): la PENDIENTE define cómo escala el parámetro con τ —
  una decisión continua sobre todo el espectro. `from_vector` deriva las
  curvas de las anclas post-reparo RR (coherencia garantizada).

### Motor: observación continua
- `GodEngineCore.temporal_spectrum[coin]`: actualizado en CADA `process_event`
  (O(19), ~120 FLOPs). La escala dominante y la fusión quedan disponibles
  para señales, ensamble y telemetría.

## 3. Migración (el binario muere por absorción, no por big-bang)

66 archivos hablan scalp/swing. La estrategia es hacer el binario IRRELEVANTE
en vez de reescribirlo a mano:

- **FASE 8a (hecha)**: núcleo + curvas de genoma + puente en apply_to_arena +
  espectro por evento. Los caminos scalp/swing operan igual pero alimentados
  por el continuo.
- **FASE 8b**: señales del core consumen `fused_score` + `signal_at(τ)`; los
  TP/SL por posición usan `tp_horizon_curve.eval(τ_efectivo)` con τ de la
  escala dominante que originó la señal; `PositionHorizon::Continuous` (slot
  0, ya existe) se puebla con la τ dominante como atributo continuo.
- **FASE 8c**: más familias de parámetros migran a curvas (thresholds,
  kelly, sizing), los 20 genes ancla legacy se deprecian formalmente y el
  backtest evalúa genomas por su rendimiento ESPECTRAL (agregado por escala).

## 4. Por qué la fusión paridad-de-riesgo y no "pesos por bucket"

Cada escala tiene ruido distinto (la lenta menos por evento). w ∝ 1/vol
normaliza el aporte de información por unidad de ruido — análogo a
inverse-variance weighting en ensambles, estándar en la literatura. El
resultado: ninguna escala domina por diseño; la que predice mejor gana peso
por EVIDENCIA (vía el ensamble F4.7 y la evolución), no por jerarquía dura.
