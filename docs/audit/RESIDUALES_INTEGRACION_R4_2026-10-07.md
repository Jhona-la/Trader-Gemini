# R4: deuda confirmada durante la revisión de ramas

Fecha inicial: 2026-10-07. I1–I4 se documentaron por inspección estática.
La adenda del 8 de octubre añade dos diagnósticos Rust aislados para I5;
no son una corrida del workspace, del host vivo ni una medición económica.
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

## R4-I4 — Dominio del conteo de familia del veto multiactivo

MED condicional, fases R1/R2/R5. Revisión estática del candidato `f3f8696`,
sin reproducción Rust ni afirmación de exceso de rechazos o pérdidas.
`FAMILIA_VETO_GRUPO` usa `C(MAX_COINS=30,2)*5=2175`, con umbral 43500
a alfa 0.05. Su comentario presupone nodos 18…22. El publicador de
`evidence_publication.rs:64–72` busca el vecino más cercano en distancia
absoluta entre los 32 nodos: con `tau_dom=30000 ms` selecciona k17,
`17179.869184 ms`. La banda [30 s,12 h] permite seleccionar seis índices,
17…22; la frontera hacia k23 es `43980465.11104 ms`, superior a 12 h.

No hay exclusión estructural de k17: `EnlacePar` almacena 32 procesos;
maduración y consulta validan índice, recencia, consumo y evidencia, no
pertenencia a 18…22. Los tres callers de publicación (`lib.rs:1655,2612,3627`)
pasan `spec.dominant_tau_ms`; el fallback de `refresh_fusion` puede dar 30 s.
La API pública acepta además horizontes fuera de esa banda. Falta reproducir
la composición con evidencia madura; observar un selector no demuestra por
sí solo el comportamiento estadístico completo del sistema.

**Límite esencial de la conclusión:** el roster de arranque configura 26
símbolos, por lo que `C(26,2)*6=1950 <2175`. Este conteo no demuestra
subcobertura del host configurado. Sí contradice la justificación declarada
para capacidad completa: `C(30,2)*6=2610 >2175`. No se concluye una tasa de
error real de 6%, ni un fallo global del gate, a partir de esa aritmética.

Siguiente contrato: separar consultas generales/telemetría del gate
operativo; declarar y validar capacidad y dominio; compartir el selector
y su conjunto alcanzable con el cálculo de familia. Probar límites y
vecindades, el fallback de 30 s y maduración k17 con bloques únicos.
Para capacidad 30 y esa banda, el conteo sería 2610 y el umbral 52200,
condicionado a reproducir y fijar esos supuestos. No se cambia un umbral
de producción ni se propone escoger 32 escalas automáticamente en esta ola.

Coordinación: la rama activa de Qoder, observada en `7dc80d8f`, introduce
`ESCALAS_BANDA_PAR=5` mediante `b56874a5`; no modifica esta selección.
Los archivos revisados estaban limpios allí. Se preservó su trabajo y se
informó al integrador; no se presume acuse externo.

Las familias de 32 escalas (umbral 640) y 13 motores×32 escalas (416,
umbral 8320) siguen siendo distintas, por instancia/activo. Bonferroni no
exige independencia entre procesos, pero requiere una familia completa y
procesos válidos: factores positivos, IC poblacional cero o bloques sin
solape no demuestran esperanza condicional del factor ≤1 respecto de la
filtración. Activos adicionales, reinicios y búsquedas necesitan su propio
presupuesto; conservar constantes no certifica una garantía global.
Marco de referencia: Ramdas, Grünwald, Vovk y Shafer,
[Game-Theoretic Statistics and Safe Anytime-Valid Inference](https://arxiv.org/abs/2210.01948).
La fuente formaliza el marco; no valida esta implementación financiera.

## Revalidación del 8 de octubre y residual OU

La [revalidación de fundamentos](REVALIDACION_FUNDAMENTOS_R4_2026-10-08.md)
fija I1–I4 sobre remoto18bb y candidato f26, distinguiendo los cambios de
main de las recuperaciones ausentes en ese padre. Las referencias f3
anteriores son históricas. La coherencia se publica en los callers nuevos
del core; no se cambió el dominio de seis nodos seleccionables descrito en I4.

I5, API OU opt-in: el modo físico cae al evaluador por evento cuando no
emite y timestamps duplicados/regresivos mutan el estado SDE. Dos
diagnósticos Rust aislados reproducen Short con duración cero, incluso
con z físico cero y sólo dos observaciones. No se encontró caller
productivo de esa opción en crates/src; no se atribuye incidencia en vivo.
El [informe de composición y OU](REVISION_MERGE_CORE_OU_REGISTRY_2026-10-08.md)
incluye blobs, rangos y alcance. Antes de activar la API, decidir modo
exclusivo/híbrido y probar rechazo sin mutación, cold/startup, huecos,
vida media larga y duración de cada señal. El cambio runtime se reserva
para un lote propio con revisión y regresiones; no se oculta en este merge.
