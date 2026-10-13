# Ola 2 — correcciones al inventario de los lectores (por los escépticos)

Los archivos `ola2_lectores.tsv`, `ola2_lazos.tsv` y `ola2_predicciones.tsv`
son la salida de los lectores tal cual (procedencia). Lo que los escépticos
refutaron con cita se anota aquí; no se reescriben los originales.

| Dónde | Qué decía | Qué es verdad (cita) |
|---|---|---|
| `ola2_lazos.tsv` fila 9 (A1, `epigenetic_threshold_modifier`) y `ola2_lectores.tsv` A1-03 | «No tiene lector de decisión; `get_active_epigenetic_threshold` sólo aparece en tests» | Sí tiene lector: `crates/risk-engine/src/lib.rs:1109-1120` lo lee y multiplica `min_required_confidence` por `epi_thresh.clamp(0.75, 1.35)` antes de `REJ_CONFIDENCE`. Es inerte en el T-3 porque la confianza y el umbral comparten el techo 0,98 (H2-N4). |
| `ola2_lectores.tsv` A2-02 (envolvente, ventana ×6) | La ventana no monótona 0 < f < f_min se ensancha ×6 con margen retenido | La envolvente sólo evalúa con la cartera vacía (`booktick_replay.rs:692`, `prev_open`), así que `cap_now ≈ 12,1` y la ventana ×6 es inalcanzable; la ventana en sí ya estaba registrada (H1-N3). |
| `ola2_lectores.tsv` LH-A3-7 | Cita `t3_ventaja_conocida.rs:102` | La sal está en `:98` (error de línea; el hallazgo se confirma). |
| Predicciones E1 de las tres lentes | «Con nocional mínimo el PnL por cierre es constante» | El pago por cierre de E1 crece ×1,6 (gana 0,0633 / pierde 0,0460 frente a 0,0393 / 0,0282) con b constante y capital sólo ×1,3: hay un canal de tamaño/geometría que cambia con la experiencia y que el instrumento todavía no atribuye (A1-N1, LH-A3-N2). |
