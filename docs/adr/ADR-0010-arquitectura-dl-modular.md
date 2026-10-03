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
