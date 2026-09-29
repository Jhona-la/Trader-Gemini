# ADR-0001 — Latencia de fills: RTT lognormal determinista, no estático

- **Estado**: Aceptado
- **Fecha**: 2026-09-29
- **Responsables**: GLM (XLVI·B, 3cb195b4); revisión de pares: Claude (CL-14
  preservó el sampler al reescribir la entrada simulada como MARKET)

## Contexto

La física de fills cobraba la latencia ESTÁTICA del gen
(`latency_penalty_ms` ≈ 30.68 ms) mientras el RTT real a Tokyo/AWS es
lognormal (σ≈0.35): p50 < estática < p99. El simulador estocástico del
repo (`NetworkJitterSimulator`) era código muerto — 0 callers. El bt no
conocía la cola derecha: los stops que sobreviven por milisegundos
salían gratis en backtest.

## Decisión

1. `risk_engine::tp_sl::sample_latency_lognormal_ms(base, seed)` —
   réplica BIT-EXACTA del simulador canónico (contrato de igualdad en
   `bt_vivo_parity_audit`). El gen calibra la MEDIA del RTT; σ=0.35 es
   física de la ruta.
2. Cableado en ENTRADA taker y SALIDA taker del núcleo con semilla
   determinista (event_time_ms, coin_id): el replay conserva
   mismo-input ⇒ mismo-output.
3. No se tocaron: friction floors D-750 (doctrina medida),
   `calculate_maker_entry` (post-only no paga latencia), packet loss
   (cambiaría semántica de decisión).

## Consecuencias

- Efecto medido sobre la ley difusiva: mediana ×0.97, media ×0.985
  (Jensen), p95 +29%, p99 +46% — el cuerpo no se penaliza, la cola se
  cobra.
- La aptitud del replay cambió (re-baseline): el oráculo T-1 re-medido
  VERDE tras el cambio (cobertura ≥ trinquete 11%).
- Regla derivada: cualquier cambio de física de fills exige re-medir el
  oráculo ANTES de promocionar genomas.
