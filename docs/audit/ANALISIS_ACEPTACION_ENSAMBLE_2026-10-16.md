# Análisis de aceptación del ensamble (a)(b) — GLM 127, 2026-10-16

Petición del TABLERO: «(a)(b) ensamble (piso 0.70 + clones) al consejo
con análisis de aceptación». Este documento ES ese análisis; el fix de
(b) lo ejecuta el mismo ciclo. Toda afirmación fue verificada en el
código de main `3291ca4c`, no en documentos previos.

## (a) El piso 0.70 — estado REAL (dos vías, dos verdades)

**Vía escalar (orchestrator.rs:514-533): DRENADA desde Ω97/F3.1-03.**
La fórmula viva es `net = (prob_l − prob_s) · effective_conviction`
SIN piso; un voto infinitesimal produce convicción ~0 y queda sub-corte
(test `tiny_positive_scores_do_not_have_false_floor`, verde).

**Vía espectral (orchestrator.rs:590-594): EL PISO SIGUE VIVO —**
`net = v_dom · (0.70 + 0.30 · coherencia_inter)` cuando el consenso
espectral está activo (τ operable y v_dominante ≠ 0). La fórmula que el
mapa C-F3 :621 describía con piso 0.70 NO era historia: es esta rama,
que Ω97 no tocó. Un v_dominante pequeño compra 0.70 de su magnitud en
convicción — el mecanismo análogo al (a) escalar, en la vía espectral.

**Decisión que queda al consejo**: drenar el piso espectral exige el
análisis de aceptación de XIV §14 sobre esa vía (tasas de aceptación,
coste y exposición con el piso vs sin él) — NO lo ejecuta este ciclo
porque NO es no-op: cambia la calibración de TODAS las decisiones
dirigidas por el espectro (la fracción `qo_624_fraccion_espectral`
del censo lo dimensiona). Es el siguiente ítem de la familia.

## (b) Clones — diagnóstico y fix (ejecutado)

`ensemble_boost = 1 + (min(5, activos) − 1)·0.1` contaba INSTANCIAS:
k copias de una fuente compraban boost 1.4 sin evidencia nueva, y
`avg_conviction = Σ|v|/n_instancias` inflaba la masa con cada copia.

**El fix (XIV §202: «clonar un experto debe dividir su masa previa en
lugar de crearla»)**: en `evaluate_continuous_consensus_for_coin`, los
votos se agrupan por FUENTE (`name()`); cada grupo aporta la MEDIA de
sus votos (k copias de v ⇒ masa v — exactamente una). El denominador
de `avg_conviction` y el total del `ensemble_boost` pasan a ser
`n_fuentes` (grupos únicos), no instancias. Zero-alloc (buffer stack
cap 32; el enjambre real ~20). El censo por instancia (#611) se
mantiene — es telemetría.

**Primer intento y su lección**: agrupar sólo el numerador dejaba
`avg_conviction = masa_de_una / n_instancias` ⇒ 5 clones daban 1/5 de
la convicción (anti-invariante inverso). El test lo cazó en minutos:
«5 clones == 1 fuente: 0 vs 0.2». La invariancia exige numerador Y
denominador por fuente.

## Aceptación: por qué esto NO cambia la calibración del sistema real

El enjambre de producción (god-engine-core lib.rs:1249-1313) registra
19 motores + NN — **todos tipos únicos, sin duplicados**. Con fuentes
únicas: `n_fuentes == all.len()` y el voto de grupo == el de instancia
⇒ **bit-paridad estructural** con el régimen por-instancias. La
exigencia de XIV §14 («comparar tasas de aceptación, costes y
exposición») queda satisfecha por paridad: no hay clones que agrupar
en producción; el cambio sólo acota el ESPACIO de configuraciones
futuras (nadie podrá inflar convicción registrando copias).

El oráculo T-1 con testigo hace la demostración empírica.

## Contrato (RED→GREEN, mismo commit)

`ensemble_characterization.rs` 7/7:
1. `clones_divide_mass_instead_of_creating_it` — conviction(0.5,5) ==
   conviction(0.5,1) bit-casi-exacto (el test (b) previo asertaba `>`
   — RED histórico documentado).
2. `distinct_concordant_sources_still_boost` — 5 fuentes INDEPENDIENTES
   concordantes > 1 fuente: la concordancia real sigue premiada.
3. `mixed_signs_in_one_source_attenuate_by_mean` — copias discrepantes
   de una fuente se cancelan por media (masa 0): la discordancia
   interna no es evidencia.

## Residuales (documentados, no tocados)

- **Piso 0.70 espectral (:590-594)** — abierto al consejo (arriba).
- **Votantes estructuralmente mudos** en el denominador (coaxial,
  choque): calibración de eff, ítem distinto de clones.
- **P7 hot-path** (try_symbol/format! por evento): frente de
  rendimiento con propuesta P5 ya escrita en el mapa.
