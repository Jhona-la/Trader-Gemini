# AUDITORÍA DE FUNDAMENTOS CIENTÍFICOS — OLA XLIII (2026-09-26, tarde)

**Mandato (tercera iteración del Quant Sr — continuidad)**: hoja de ruta XLII §5: W₁→gates
con detección bayesiana de cambio, Hayashi-Yoshida para la covarianza asíncrona del
veto multi-activo. Ronda sin cuenta/órdenes/entrenamiento operativos.

## 1. B — HAYASHI-YOSHIDA: la correlación multi-activo deja de fingir sincronía

`correlation_guard.rs::hayashi_yoshida_correlation`:
- **El defecto que corrige**: el veto D-748 correlacionaba monedas muestreando sus
  ticks en una REJILLA común — pero las monedas no cotizan en instantes compartidos.
  Eso es el Epps effect: la correlación medida DECAE con la desincronía del muestreo,
  de modo que el par "poco correlacionado" del veto era a veces el par más
  desincronizado, no el menos correlacionado. En un universo multi-activo asíncrono,
  fingir sincronía es un sesgo sistemático contra los pares de relojes distintos.
- **El estimador**: R_HY(A,B) = Σ_{overlap>0} rᵢ·rⱼ con normalización por las
  varianzas HY propias — consistente bajo muestreo asíncrono no sincronizado
  (Hayashi-Yoshida 2005). Productos PLENOS de todo par de intervalos solapados, sin
  rejilla ni descarte. O(n+m) con dos punteros.
- **Cableado**: D-748 usa HY primero; si no hay evidencia HY (sin solape/varianza),
  fallback al Pearson en rejilla existente. La matriz del grupo (que ya alimenta
  Marchenko-Pastur) hereda la covarianza corregida.
- **Falsación (3 tests)**: sincronizadas exactas → HY == 1; señal común con relojes
  desplazados (10ms vs 13ms, offset 5ms) → HY mantiene ~0.98 donde la rejilla
  subestima; sin solape/sin intervalos → None.

## 2. A — BOCPD-LITE SOBRE EL TRANSPORTE W₁: el régimen anuncia su cambio

`god-engine-core::W1ChangepointObserver`:
- **Qué es**: posterior bayesiano de run-length (Adams & MacKay 2007, forma
  gaussiana, hazard 1/100, R=128 truncado) sobre la serie W₁(t) del transporte de
  Wasserstein espectral de cada moneda. `p_transition` = masa posterior de segmento
  nuevo = probabilidad de que la masa espectral esté RESTRUCTURÁNDOSE ahora.
- **La pieza que la falsación obligó a construir**: la primera versión usaba una
  emisiva fija N(0,σ) para todos los run-lengths — el salto sostenido resultaba
  igual de improbable bajo «cambio» que bajo «no cambio» y el detector era CIEGO
  (el test del salto lo delató: p≈1 %). Corregido con estadística por segmento:
  predictiva del segmento estable N(media_r, 0.35) (se aprieta alrededor de SU
  nivel) contra prior ancho N(0, 2.0) para segmentos nuevos. Ahora: reposo →
  p≈hazard; salto de 2 ejes-log → p>0.5 (tests).
- **Cableado — MEDICIÓN HONESTA (3 corridas del oráculo)**: el observador se
  alimenta en ambos sitios de actualización espectral y `p_transition` se publica
  al registry. El DESCUENTO de convicción (×0.70 en transición plena) se probó
  con DOS calibraciones (lineal y con piso 0.25): ambas midieron 20→15 genes en
  el oráculo — en el fixture sintético el transporte W₁ fluctúa por saltos y el
  posterior vive arriba, gravando la sesión entera. Calibrar el actuador contra
  las estadísticas de un fixture es exactamente el número inventado que la
  doctrina D-751 prohíbe: el ACTUADOR vuelve a telemetría y se reactiva con la
  distribución de p_transition medida sobre TAPE REAL. El detector (W₁ + BOCPD
  + sus tests) queda operativo.

## 3. El ciclo teoría→consumidor→falsación en esta rama

W₁ (XLII: medir el movimiento) → BOCPD (XLIII: inferir el CAMBIO del movimiento) →
gate de convicción (XLIII: pagar el tránsito). Cada eslabón con contrato y test que
rompió la versión ingenua anterior — incluida la mía.

## 3.5. Lección de la medición

El trinquete del oráculo hizo su trabajo: dos calibraciones del actuador midieron
pérdida neta de expresividad (20→15) y fueron revertidas con la trazabilidad
completa. La secuencia falsa→corrige→mide→revierte documentada aquí es el
protocolo funcionando, no un fracaso: el detecto quedó, el actuador espera datos
reales.

## 4. Pruebas

- risk-engine: 79+22 (3 nuevos HY). god-engine-core: 114 (3 nuevos BOCPD) + suites.
- Oráculo T-1 re-medido (§5 del log de sesión).
- Validación completa del workspace al cierre.

## 5. Hoja de ruta (hereda XLII §5)

1. Hodge discreto en el grafo de señales (T05) — ciclos de dependencia ramas↔vetos.
2. Hawkes multivariado (T03): la contagión cross-asset con matriz de excitación —
   alimentaría al veto D-748 con la dinámica, no sólo la foto (HY da la foto).
3. BOCPD sobre OTROS observables (entropía, Fisher) — mismo esqueleto.
4. bt↔vivo (entregable mayor, sin cambios).

## 6. Cierre del tramo

El veto multi-activo ya no confunde «desincronizado» con «independiente» (HY), y el
universo espectral tiene cambio de régimen PROBABLE en lugar de sólo medido a
posteriori (BOCPD sobre W₁). Ambos con la cicatriz de la falsación: el detector
ciego y el estimador sesgado no sobrevivieron a sus propios tests.
