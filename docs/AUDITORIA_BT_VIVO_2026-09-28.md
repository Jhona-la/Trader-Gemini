# Auditoría bt↔vivo — censo de divergencias del envoltorio (2026-09-28)

**Autor**: GLM (Ola XLVI·A). **Rama**: `glm/xlvi-bt-vivo-auditoria`.
**Pregunta**: ¿qué separa la aptitud que mide el backtest del PnL que produce el
motor vivo? Es la "brecha bt/vivo" anotada como pendiente desde la auditoría
decimocuarta.

## Arquitectura verificada (lo que YA es paridad)

`run_booktick_replay` (booktick_replay.rs:219) **conduce el núcleo de
producción** (`GodEngineCore::process_event`), no una réplica. Por
construcción comparten:

| Dimensión | Estado | Evidencia |
|---|---|---|
| Lógica de decisión (gates, vetos, council, ML) | **por construcción** | mismo `GodEngineCore` |
| Física de fills (slippage, latencia, fees) | **por construcción** | `reality.calculate_market_entry(…, lat_ms)` en ambos mundos |
| Genoma (incl. `latency_penalty_ms=30.68ms`) | **por construcción** | `genome.apply_to_arena` en replay; mismo genotipo activo |
| Envolvente de sizing | **replicada + testeada** | `live_envelope_gate` (B3.19, D-442/116/382) |
| Curva VOL-BRAKE | **testeada** | M8-C01 (`vol_brake_factor_replica_curva_del_vivo`) |
| Convención maker/agresor | **corregida** | D-717 (ambos modos del replay) |
| Corte macro t-1 | **paridad** | B3.16 (`OmniHistory::day_of`, día previo) |
| Frontera de vela 1m | **paridad** | D-705/D-708 (minuto, no día) |

## Censo de divergencias (lo que NO es compartido)

### DIV-1 — Desplazamiento adverso de precios SOLO en el harness del replay

**Vivo** (lib.rs:1343): `eff_bid = bid`, `eff_ask = ask` — precios CRUDOS del
libro entran al core.
**BT** (booktick_replay.rs:309-311): `sim_bid = bid − running_atr·0.10`,
`sim_ask = ask + running_atr·0.10` — el harness ENSANCHA el spread antes de
alimentar el core.

Consecuencias: (a) toda feature que consume bid/ask (spread, OBI) ve un libro
más ancho en bt que en vivo; (b) el fill de entrada paga el desplazamiento
ADEMÁS de la física `calculate_market_entry` (que ya aplica `slip_floor` +
`lat_ms`). **Dirección del sesgo: bt MÁS pesimista en fills de entrada** —
no explica por sí sola un bt mejor que el vivo, pero contamina la comparación
de features de libro.

**Signpost falsable**: si se elimina el desplazamiento del harness, el OBI
efectivo visto por el core en bt debe igualar el OBI crudo del tick.

### DIV-2 — Latencia estática vs lognormal (colas no modeladas)

**Vivo**: RTT real a Tokyo ≈ 25ms base con jitter lognormal (σ≈0.35) y pérdidas
de paquete — el simulador canónico del repo (`NetworkJitterSimulator`,
network_jitter.rs) existe para esto.
**BT**: el core usa el valor ESTÁTICO del genoma (`30.68ms`) en la física de
fills. `NetworkJitterSimulator` **es código muerto en la ruta de decisión**:
grep en todo el workspace → 0 callers (sólo re-export).

Consecuencia: la física del core cobra una latencia fija; las COLAS (P99 ≈
2× la base con σ=0.35) no existen en bt. Los stops que sobreviven por
milisegundos en vivo aparecen como sobrevividos limpios en bt.
**Dirección del sesgo: bt MÁS optimista en colas de latencia** — el sospechoso
clásico de "bt rentable, vivo no".

**Signpost falsable**: P99 determinista de `NetworkJitterSimulator(25, 0.35,
0.001)` > `latency_penalty_ms` estático del genoma activo. Pinned en test
`xlvia_p99_latencia_lognormal_supera_penalizacion_estatica`.

### DIV-3 — Ejecución TP/SL: cruce de tick vs órdenes algo del exchange

**BT**: los brackets se resuelven dentro del core al cruce del tick stream.
**Vivo**: TP/SL viajan como órdenes algo en Binance (migración -4120), disparan
con mark price del exchange y sufren RTT + requeued. Ya registrado como
**R8-A abierto** en el PR #10 de Claude ("el motivo de salida vivo no
reconstruye el primer toque cuando los brackets difieren; XLIV-9c es una
aproximación"). **No se duplica aquí**; se referencia.

**Dirección**: vivo toca barreras antes/coje fills peores en el primer toque
que la reconstrucción por tick de bt.

### DIV-4 — Klines de calentamiento sintéticos

**BT**: los 1m klines del warmup se sintetizan agregando MIDS de ticks por
minuto (booktick_replay.rs:238-266).
**Vivo**: klines reales del exchange (last-trade price, no mid).
**Impacto**: acotado al calentamiento (Hurst necesita 512 cierres); sesgo de
nivel pequeño en hurst/EMAs de arranque. **Menor, documentado, sin acción**.

### DIV-5 — Universo: 1 símbolo (coin 0) vs roster completo

**Por diseño** (GA por-símbolo contra SU data). La interacción
multi-activo (matriz de contagio XLV, agregación de exposición) NO se ejerce
en replay. La aptitud por-símbolo es válida; la aptitud de CARTERA no se mide
aquí. Pendiente estructural conocido, no un bug.

### DIV-6 — Fees: pisos conservadores vs VIP real

BT usa `live_maker_fee/live_taker_fee` con pisos `.max(0.0002/0.0004)`.
Vivo paga el tier VIP real del exchange. Con pisos ≥ tier real, **bt es
conservador** (cobra ≥ que el vivo). Sesgo pesimista acotado, aceptable.

## Tests de contrato añadidos

`crates/backtest-engine/tests/bt_vivo_parity_audit.rs`:

1. `xlvia_genoma_compartido_misma_latencia_en_bt_y_vivo` — mismo genoma ⇒
   mismo `latency_penalty_ms` leído del arena en ambas rutas (DIV-2 queda
   reducida EXACTAMENTE a "estática vs distribución", no a config divergente).
2. `xlvia_desplazamiento_del_harness_solo_empeora` — el desplazamiento ATR del
   harness es monótono adverso: a mayor ATR, bid más bajo y ask más alto; nunca
   mejora el precio (invariante de dirección de DIV-1).
3. `xlvia_p99_latencia_lognormal_supera_penalizacion_estatica` — cuantifica la
   cola: P99 determinista del simulador > penalización estática del genoma
   activo (30.68ms). La magnitud de DIV-2, medida, no intuición.

## Síntesis para la meta (+100%/3d)

La promoción de genomas se decide sobre este replay. El replay es
**pesimista en fills de entrada (DIV-1, DIV-6) y optimista en colas de
latencia (DIV-2) y primer toque de barreras (DIV-3)**. Los dos sesgos
optimistas atacan justo donde la meta es más frágil — stops y ejecución en
fricción alta — y son los candidatos primarios de la brecha bt/vivo.

**Acción recomendada (ordenada por relación señal/coste)**:
1. Cerrar DIV-2: alimentar `lat_ms` de `calculate_market_entry` con una muestra
   determinista (seed = ts·coin) de `NetworkJitterSimulator` en vez del valor
   estático. El simulador YA existe y está probado; es cableado, no física
   nueva.
2. Eliminar DIV-1 del lado de features (pasar bid/ask crudos al core; dejar el
   desplazamiento SÓLO en el precio de fill vía física del core) — requiere
   separar "precio de feature" de "precio de fill" en `process_event`.
3. DIV-3 sigue el curso de R8-A (Claude).

Ninguna acción sobre DIV-4/5/6 (acotados o por diseño).
