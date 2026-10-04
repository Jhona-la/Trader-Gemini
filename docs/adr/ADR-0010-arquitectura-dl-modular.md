# ADR-0010: Arquitectura DL-modular espectral — la escalera falsable

- **Estado**: Propuesto (GLM, LXXXI, 2026-10-03) — presentado al consejo
- **Contexto**: la visión del operador — el sistema como red modular
  espectral con capas matemáticamente informadas + deep learning +
  control óptimo estocástico + kill-switch duro. Este ADR la convierte
  en una ESCALERA FALSABLE: cada capa con su medición y su gate; nada
  se cablea por superar el puerto de medición de su capa anterior.

## El mapa de capas (qué existe, qué falta)

| capa | rol | estado |
|---|---|---|
| **L0 — Sustrato espectral** | 32 filtros EWMA base-4 (1ns→146 años), persistencia por bloques, masa espectral honesta | **EXISTS** (CL-30..35, quantum-arena) |
| **L1 — Votos por escala** | 13 motores resueltos por escala (VotoEspectral), consenso espectral integrado al orquestador, pesos por skill IC-prequential | **EXISTS** (qo-609..650, AGY-P29/P30) |
| **L2 — Agregación aprendida** | un modelo entrenado que mapea (votos×escalas, features de camino, contexto) → (dirección, convicción) reemplazando/aumentando la modulación fija `0.70+0.30·convicción` | **FALTA — este ADR** |
| **L3 — Control estocástico** | sizing Kelly fraccional sobre el edge medido, multivariante con λ̂ de cópula; kill-switch duro independiente | **PARCIAL** (Kelly existe; multiactivo mejorado en LXXII-LXXIII; kill-switch existe) |

## L2: el primer experimento medible (el único paso que este ADR autoriza)

**Pregunta**: ¿una agregación APRENDIDA de los votos del L1 supera a la
modulación FIJA del consenso, fuera de muestra?

- **Datos**: los 13 votos por escala ya se publican como telemetría
  (`sombra_*`, `consenso_espectral_*`). Se necesita un export de
  (votos, escala dominante, τ, dirección realizada) sobre los tapes
  2026-06/07/08/09 con la misma rejilla de la plantilla honesta — el
  `feature_exporter` es el vehículo natural (extensión, no nuevo bin).
- **Modelo**: v1 = regresión logística / NanoForest pequeño sobre
  (votos×escalas + τ + régimen p∈Δ³) → dirección. El objetivo NO es
  sofisticación: es SI/NO supera a la fija.
- **Gate (plantilla honesta integral)**: split cronológico
  train/selección/test-posterior; el candidato L2 compite CONTRA la
  modulación fija en el MISMO test posterior. Pasa ⇒ gana el diseño de
  cableado (ola propia con oráculo T-1 + paridad, como toda conducta
  viva). No pasa ⇒ **negativo documentado: la fija basta** — y el L2
  vuelve a dormirse hasta que el L1 evolucione.
- **Cópulas y L3**: el sizing multivariante ya consume λ̂; si L2
  despierta, su convicción alimenta el mismo camino — sin bypass del
  veto (el kill-switch y el veto estructural NO se aprenden: son duros
  por diseño del operador).

## Consumidores que este ADR desbloquea (cuando L2 pase su gate)

1. **Firmas de camino (path_signatures, HUÉRFANAS desde XLVIII)**:
   features de entrada del L2 — el consumidor que la doctrina
   anti-decoración les negó hasta existir una hipótesis medible.
2. **Isla evolutiva (qo-605: neat/moe/crossover/anti_bias)**: moe_neat
   como generador de candidatos del L2 (arquitecturas de agregación) —
   el fitness sería el test posterior del L2, no el fitness del replay.
3. **La queja del operador "no es verdaderamente autoevolutivo"** queda
   respondida en arquitectura: L0/L1 evolucionan por genes (oráculo),
   L2 evoluciona por plantilla honesta, L3 por medición de edge — tres
   relojes evolutivos con sus gates.

## Reglas no negociables

- El veto estructural, el kill-switch y el tope Lundberg NUNCA entran
  en el gradiente: riesgo-duro es código, no peso aprendido.
- Ninguna capa se cablea sin superar el puerto de la capa anterior.
- Paridad bt↔vivo: el L2 entrenado debe ser la MISMA función en replay
  y en vivo (artefacto congelado + watcher, como los modelos MOTOR).

## Relacionado

ADR-0003 (meta geométrica), ADR-0008 (revalidación de modelos — el L2
hereda el reloj), ADR-0009 (cópulas mensuales — input del L3),
TRIAGE_TEORICO §LXXVIII (huérfanos), qo-605 (isla evolutiva).

---

## Adenda LXXXII (2026-10-03) — fase 1 ejecutada: dataset servible + apriori HOSTIL (y por eso valioso)

Dataset v1: BTC junio, 178,539 puntos (stride 15s), genoma ACTIVO g2,
replay real, features point-in-time. Reproducible:
`TG_GENOME_ENV=backtest votes_export BTCUSDT --in data/BTCUSDT_2026-06_REAL.bin --out ...`

**Medición apriori (antes de entrenar nada)**:
- corr(consenso_dom, r_fwd_5m) = +0.006; hit direccional 49.1% PLANO en
  todos los pisos de convicción (|voto|>0.05 … >0.3).
- Ninguna sombra individual tiene señal: corr −0.003..+0.005, hit
  48.8-50.3%.
- Condicionado por régimen NO rescata: no-range 50.0%, range profundo
  47.7% (peor). Única chispa: p_chaos>0.10 → 52.7% (n=3,476; ~3.2σ
  SIN corrección por ~10 comparaciones — pista débil, no hallazgo).
- Contexto: junio fue 97% régimen range (p_range mediana 0.969).

**Defecto de diseño v1 detectado**: etiqueta a horizonte FIJO 5m cuando
la τ del consenso VARÍA por muestra (la posición vive a
consenso_espectral_tau). La pregunta correcta es dirección A LA ESCALA
PROPIA de cada muestra.

**Redirección de la fase 2 (la honesta)**: (a) exportar r_fwd a la τ
por-muestra (y/o multi-horizonte 15m/1h); (b) sólo entonces entrenar;
(c) la hipótesis deja de ser "amplificar señal lineal" (no hay) y pasa
a ser "estructura condicional/no-lineal" — con prior hostil declarado.
Si la fase 2 también da cero a τ propia ⇒ negativo documentado y el L2
vuelve a dormir hasta que el L1 evolucione: la modulación fija no es el
cuello de botella.

---

## Adenda LXXXIII (2026-10-04) — fase 2 τ-matched: el apriori HOSTIL se REVIERTE — hay estructura

Dataset τ-matched (175,679 filas, etiqueta a la τ propia de cada
muestra, genoma activo g2, BTC junio):

- **sombra_osc SOLA: hit 53.0%** a escala propia (n=175k, ±0.1 — ~30σ).
- **Consenso con convicción |voto|>0.5: hit 53.8%** (n=60,803, ±0.2).
- Pero el consenso AGREGADO completo: 48.7% — y la banda 5m-1h
  **anti-correlaciona** (46.1%): la agregación fija AHOGA la señal del
  oscilador en convicción media. La agregación es PEOR que su mejor
  componente.
- Distribución τ: bimodal (42% <30s, 36% >1h — el clamp de banda
  operativa). sombra_res y sombra_entropia anti-señal (47.2-47.4%).

**La hipótesis L2 queda JUSTIFICADA con evidencia**: una agregación
aprendida podría conservar la señal del oscilador en vez de diluirla.
La fase de entrenamiento (v1 logística sobre votos vs modulación fija,
plantilla honesta con meses separados) pasa de "prior hostil" a "prior
estructurado": el objetivo explícito es NO PERDER lo que sombra_osc ya
sabe. Caveats declarados: un mes, un símbolo, sin costos — el gate de
la plantilla decide.
