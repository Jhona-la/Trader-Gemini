# R4: deuda confirmada durante la revisión de ramas

Fecha: 2026-10-07. Estado: inspección estática, no reproducción Rust.
Estos pendientes no se declaran reparados por integrar los satélites. No se
ha medido aquí su incidencia económica. Se añaden al plan R0–R9.

## R4-I1 — AggTrade no distingue timestamp ausente de timestamp inválido

MED, fase R4/datos. En el candidato de integración root-audit sobre
`adeb8d1b` + `d0e02da1`, blob del parser
`6c47744a94f5bea150e8a65c750b2a2f58c73cc4`, `AggTradeEvent::parse`
convierte `T` presente pero inválido a cero mediante `unwrap_or(0)`.
Ejemplo: `{"p":"10","q":"1","m":true,"T":null}`.
El timestamp válido sí se conserva; el merge no perdió ese cambio de Qoder.

La ruta `ws_client.rs` transmite ese reloj a `update_agg_trade`; la rama de
reloj no creciente puede sumar volumen sin decaimiento. La alcanzabilidad
y el efecto en el host operativo necesitan replay; no se afirman pérdidas.
El parser BookTicker ya distingue presente-inválido de ausente, pero esa
corrección no cubre automáticamente AggTrade.

Siguiente contrato: `T` ausente con política explícita, cero válido o inválido
según dominio, entero válido, `null`, fracción, negativo, overflow y token
incompleto. Probar también el consumidor: un evento sin reloj confiable no
debe parecer una nueva observación temporal válida. No normalizar un error
a cero silenciosamente para que la prueba pase.

## R4-I2 — Mutación de anclas sin efecto en curvas del CLI evolution

MED, fase R6/genoma. Base `adeb8d1b`, `src/bin/evolution.rs:251–286`:
se mutan y acotan `scalp_sl_base` y `scalp_tp_base`; el candidato se evalúa
sin reconstruir las curvas usadas por `SuperGenotype::apply_to_arena`
(`genome.rs`, campos `tp_curve_*`/`sl_curve_*`). El cambio de Ω16 en
`continuous_evolution_backtest.rs` es otro caller y no cierra esta ruta.

El merge OOS/SA mejora partición y selección; no debe anunciar que todas
las dimensiones genéticas de este CLI tengan efecto. Antes de corregir:
test de sensibilidad que muta una sola dimensión, compara parámetros
publicados en arena en varios valores de tau y mantiene controles de RR,
fricción y demás genes. Decidir si se mutan curvas canónicas o se reconstruyen
desde anclas; evitar dos fuentes de verdad y conversiones acumulativas.

## R4-I3 — Alcance limitado del watcher recuperado

Los contratos MW01/02/03 de `codex/model-reload-contract` mejoran selección
de archivos y reintentos. La inspección de `3139f444` contra base `7fdd12dd`
y `adeb8d1b` no identificó regresión bloqueante en ese alcance. Esto no
certifica identidad completa de modelos:

- `(ruta, mtime, tamaño)` no detecta contenido nuevo con metadata idéntica.
- El loader puede servir un BIN por política de mtime aunque se solicitó
  el JSON; falta identidad fuente/cache (MW05).
- Revocación o retirada de un modelo y preferencia de artefactos necesitan
  contratos específicos, no sólo éxito de recarga.
- El watcher DarkAlpha independiente del host no fue reparado por MW;
  todavía actualiza mtime antes del éxito de carga en la base revisada.
- La carga síncrona dentro de una tarea Tokio periódica no demuestra una
  cota de latencia. Medir presión del runtime y publicación RCU.

Fases R4/R6/R7. Mantener los casos OPEN históricos y los modelos sintéticos
en las pruebas; no ensayar estas correcciones cargando modelos reales en vivo.

## Alcance de la revisión cruzada

Dos revisores independientes comprobaron estáticamente que los merges deben
preservar Ω12, la fuente canónica de `selection_stats` de Ω11, curvas Ω16,
la historia `control_realized_pnl_pct` de XCIV y el gate de Ville actual.
La prueba antigua de evidencia con 40 pares debe madurar 120 pares y probar
rechazo con 40; se adapta el fixture sin rebajar el umbral de familia.

La compilación del primer merge raíz pasó; la evidencia de los demás merges,
suites y T-1 pertenece al recibo final de integración, no a este informe.
Revalidar anclas y blobs cuando cambie el candidato.
