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

---

## Adenda LXXXIV (2026-10-04) — el GATE del L2 v1: VEREDICTO PARCIAL — dirección SÍ (fuerte), calibración NO

Split honesto (train jun n=76,437 / eval ago / test sep-14, umbral
decisivas ±0.05%, logística Newton 11-dim, pesos congelados tras train):

| split | hit aprendida | hit fija | Δ | logloss vs constante |
|---|---|---|---|---|
| ago | 58.2% | 53.2% | **+5.0** | 0.700 vs 0.681 (peor) |
| sep-14 | 48.1% | 42.8% | **+5.3** | 0.752 vs 0.691 (peor) |

- **La DIRECCIÓN transfiere OOS**: +5.3 puntos sobre la modulación fija
  en test posterior, consistente con agosto (+5.0). La fija en
  septiembre fue ANTI-predictiva (42.8%) — la aprendida sostiene 48.1%
  en un mes hostil donde su baseline colapsó.
- **La CALIBRACIÓN falla**: probabilidades sobrefiadas (logloss peor
  que la constante — los errores vienen con p≈0.9). El gate como fue
  escrito (hit ∧ logloss>base) NO pasa completo.
- Los pesos confirman la física del hallazgo: sombra_osc +6.65,
  sombra_res +6.31, consenso_media **−3.50**, consenso_dom −0.22 — la
  agregación aprendida cabalga los motores y DESCUENTA el agregado fijo.

**Veredicto**: PARCIAL. Como agregador de DIRECCIÓN el L2 v1 ya supera
a la fija OOS; como estimador de PROBABILIDAD no sirve sin calibrar.
**Camino de cierre (fase 3)**: calibración de temperatura/Platt sobre
el mes de selección + re-gate — es un parámetro unidimensional, no un
rediseño. El cableado al consenso SIGUE bloqueado hasta que el gate
completo pase (y entonces: oráculo + paridad, como toda conducta).

---

## Adenda LXXXV (2026-10-04) — fase 3 (Platt): el PARCIAL es DEFINITIVO en v1

Calibración de temperatura ajustada en AGOSTO (selección; sep-14
intacto), T=2.393:

| split | hit calibrada | hit fija | logloss vs constante |
|---|---|---|---|
| ago | 58.1% | 53.2% | 0.6771 vs 0.6812 **MEJOR** (donde se ajustó) |
| sep-14 | 47.7% | 42.8% | 0.7078 vs 0.6910 **PEOR** (no transfiere) |

- **La DIRECCIÓN es robusta**: +4.9 a +5.3 pts sobre la fija OOS en dos
  meses, dos calibraciones, y un septiembre hostil donde la fija fue
  anti-predictiva (42.8%).
- **La CALIBRACIÓN NO TRANSFIERE entre regímenes**: ajustar en
  selección arregla selección, no test. El logloss OOS queda peor que
  la constante.
- **VEREDICTO FINAL v1: PARCIAL DEFINITIVO.** El gate como fue escrito
  (hit ∧ logloss OOS) NO pasa. El cableado al consenso SIGUE BLOQUEADO.

**Fase 4 (si el consejo la pide)**: convicción por RANGO (ordinal, no
probabilidad — la modulación fija pondera por orden, no por calibración
absoluta) o modelo no lineal con regularización estacional. Ambas son
olas nuevas con gates propios; NINGUNA se promete. El hallazgo
estructural queda firmemente documentado y estable: la agregación fija
ahoga a sus mejores componentes; una agregación mejor EXISTE como
información direccional; capturarla como PROBABILIDAD estable es el
problema abierto.

---

## Adenda LXXXVII (2026-10-04) — pre-gate de fase 4 (convicción por rango): NEGATIVO — la saga L2 v1 cierra completa

Métrica: hit del quintil superior por cada medida de convicción
(orden, no probabilidad — la hipótesis de que la modulación pondera
por ORDEN y la calibración absoluta no importa):

| split | rango modelo top-q | spread | convicción fija top-q | spread |
|---|---|---|---|---|
| ago | 66.2% | **+10.0 pts** | 62.7% | +5.6 |
| sep-14 | 49.2% | +1.4 | 50.4% | **+2.9** |

En el quiebre de régimen de septiembre la ORDENACIÓN de confianza del
modelo se desacopla de su utilidad (corr con la convicción fija cae de
0.496 a 0.010) y la fija separa mejor. **Ni probabilidad ni rango
transfieren** — el régimen hostil destruye la estructura de confianza
completa del modelo, no sólo su calibración.

**CIERRE DE LA SAGA L2 v1**: dirección robusta (+5 pts OOS constante,
dos calibraciones, dos métricas de convicción) / probabilidad y ordinal
NO transfieren / cableado BLOQUEADO definitivamente en v1. La única
estructura no probada queda registrada: **modelos por régimen**
(entrenar condicionado al símbolo p∈Δ³ del simplex — si el consejo
alguna vez la pide, es una ola con gates propios). El L2 vuelve a
dormir con la evidencia completa: la información direccional existe,
el quiebre de régimen es el enemigo.

---

## Adenda LXXXVIII (2026-10-04) — pre-medición régimen-condicionada: NEGATIVA — la saga L2 v1 cierra HERMÉTICA

Cubetas por símbolo del simplex (A: p_range≥0.95 / B: resto),
logísticas separadas entrenadas sólo con sus muestras de junio:

| split | global | por-cubeta | matiz B_resto | fija |
|---|---|---|---|---|
| ago | 58.2% | 57.8% | **63.2%** (n=25k) | 53.2% |
| sep-14 | 48.1% | 48.3% | **51.0%** (n=7.2k) | 42.8% |

El condicionado NO supera al global (el global ya tiene el régimen
como feature y aprende la partición solo) — la última estructura
abierta queda medida y negativa. **CIERRE HERMÉTICO de la saga**:
ni global, ni calibrado, ni rango, ni condicionado transfieren
completamente el quiebre de régimen de septiembre. Matiz valioso para
cualquier futuro: la señal se CONCENTRA en las muestras no-range
(cubeta B: +8.7 pts sobre la fija en sep-14) — si el consejo alguna
vez cablea algo de esta familia, el valor está en convicción alta en
régimen NO-range, nunca en range profundo.
