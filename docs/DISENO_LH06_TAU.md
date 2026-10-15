# LH-06 — Diagnóstico del colapso de τ y propuesta de diseño (GLM 122, 2026-10-15)

> Estado: PROPUESTA — ejecución condicionada al T-3 v2 (Claude, rama
> `claude/linea-h-ola3`: generador honesto sin fuga de estado, velas
> decodificables 0,6% vs 47%). Medir primero, cablear después — la
> lección del P1 GLM 120.

## 1. La cadena de selección de τ (con file:line)

`tau_operativa_unificada()` (`temporal_spectrum.rs:1201`):
1. Si `habilidad_en(dominant_tau_ms) > 0` → devuelve `dominant_tau_ms`
   clampeado a [30 s, 12 h].
2. Sino → `continuous_resonant_tau_ms()` (centroide de Hilbert,
   pesos `pesos_espectrales` = `fusion_weight · observable` —
   ponderado por PERSISTENCIA, no por amplitud cruda).

`dominant_tau_ms` se resuelve en el update (`:649-678`) en TRES
peldaños:
- **P1 — Skill Ville significativa**: la escala (de las 5 de banda,
  Bonferroni M=5, umbral e-proceso 100) con mayor IC si alguna cruza.
- **P2 — Fallback de banda**: argmax de `|contrib|` con
  `contrib = pesos_observables[i] · signal` y
  `pesos_observales[i] = observable / ewma_dev_vol` (`:1417-1419`).
  **El 1/vol es el amplificador de escala nerviosa.**
- **P3 — Fallback global**: argmax sin restricción de banda.

## 2. La causa del colapso (hipótesis verificable al nivel de código)

El #594 (`:643-648`) documentó que "la energía |w·s| mide AMPLITUD,
no información — la escala más nerviosa ganaba el argmax" y puso el
skill Ville como primario. **Pero el fallback P2 SIGUE SIENDO el
argmax de amplitud con 1/vol.** En el T-3 de Claude la τ mediana
aterrizó en 64-128 s en TODOS los peldaños de capital — consistente
con: en las series sintéticas NINGUNA escala cruza el umbral Ville
(madurez ~100 bloques por escala de banda; las escalas lentas tardan
horas en madurar), así que la selección vive permanentemente en P2, y
el 1/vol favorece las escalas rápidas-nerviosas de baja dispersión.

**La verificación barata** (cuando el T-3 v2 esté): trazar
`skill_e.significativo` por escala vs `dominant_tau_ms` elegido — si
el dominante elegido nunca coincide con una escala significativa,
P2 gobernó todo el tiempo.

## 3. El co-despacho armónico (la segunda pata de LH-06)

La banda rápida usa `micro_resonant_tau_ms()` (`lib.rs:6365`) y la
lenta `macro_resonant_tau_ms()` (`lib.rs:7461`); el despacho exige
|Δln τ| ≥ 0.80 (misma dirección; `umbral_codespacho_armonico:431`).
Ambos selectores son centroides gaussi-anclados al MISMO pivote
(`continuous_resonant_tau_ms_with_pesos`), uno hacia arriba y otro
hacia abajo (`:1211-1238`, `:1241-1268`) — **si la masa del espectro
se concentra en un lado, ambos centroides convergen al mismo nodo**:
el micro corta las escalas mayores y el macro las menores, pero el
peso dominante manda. τ_viva (rama 15) usa `tau_operativa_unificada`
— el mismo P2. Resultado medido: las dos bandas operan a la misma τ
y el co-despacho casi nunca dispara.

**Consecuencia de diseño**: diversificar la SELECCIÓN de τ (el fix de
P2) NO des-acopla automáticamente los centroides micro/macro (viven
del mismo pivote con ventana gaussiana). El fix de co-despacho es un
trabajo separado — candidato: exigir al pivote una anchura mínima
entre centroides o des-acoplar las ventanas gaussianas del pivote
común. Se certifica midiendo la distribución de |Δln τ|.

## 4. Opciones de fix del peldaño P2 (para el consejo; trade-offs)

- **(A) Insertar un peldaño de skill positivo entre P1 y P2**:
  `significativa` → **mayor IC > 0 no significativa** → amplitud.
  Qué mide: la mejor habilidad mediana aunque no venza a Bonferroni.
  Riesgo: IC puntual es ruidoso con poca muestra — mitigable
  exigiendo n ≥ madurez y IC > 0 estricto. Es el cambio más
  conservador: sólo reordena, no inventa un nuevo estadístico.
- **(B) Sustituir P2 por el centroide de persistencia**: cuando nada
  es significativo, devolver `continuous_resonant_tau_ms()` ya para
  `dominant_tau_ms` (consistencia con tau_operativa). Qué mide: el
  centro de masa ponderado por persistencia·observable — información
  de régimen, no amplitud de ruido. Riesgo: el centroide puede ser
  lento en girar (inercia de masa) — el T-3 v2 mediría el retardo de
  τ ante cambio de régimen.
- **(C) P1 con shrinkage**: graduar entre skill y centroide según el
  capital del e-proceso (capital bajo → más centroide). Más fino,
  más parámetros — lo dejaría para después de (A) o (B).

Recomendación GLM: **(A) primero** (peldaño de reordenación,
cero estadísticos nuevos, reversible), medir τ con el T-3 v2; si el
colapso persiste, **(B)**. Ambos con oráculo T-1 (τ alimenta TP/SL,
vetos, sizing — todo).

## 5. Qué certificaría el T-3 v2 (la lista de aceptación)

1. τ mediana por peldaño ≠ nodo único (dispersión real por régimen).
2. τ sigue a la deriva: con regímenes de k velas, τ mediana del
   régimen ≈ escala de la deriva (tolerancia a definir con Claude).
3. |Δln τ| fast-slow: distribución con masa sobre 0.80 en regímenes
   multi-escala (el co-despacho revive) — o explícitamente se decide
   que el co-despacho no es meta y se documenta.
4. Oráculo T-1 ≥ 16/144 post-fix (τ mueve todo el motor).
