# AUDITORÍA DE FUNDAMENTOS CIENTÍFICOS — OLA XLIV (2026-09-26, noche)

**Mandato (cuarta iteración del Quant Sr — continuidad)**: la agregación de riesgo
del veto multi-activo pasa de contar apuestas a agregar varianza con la ESTRUCTURA
medida. Ronda sin cuenta/órdenes/entrenamiento operativos.

## 1. El defecto: k·σ lineal era "todo es la misma apuesta" por defecto

`veto_por_exposicion_direccional(n, σ, q)` computaba el riesgo del grupo como
**(n+1)·σ — lineal siempre**, aunque el propio veto ACABARA de medir la matriz de
correlaciones del grupo (HY desde XLIII) y la hubiera juzgado con Marchenko-Pastur
(desde XLI). Es decir: medíamos la estructura para decidir SI contar, y después
ignorábamos la estructura al contar. Consecuencia práctica: un grupo de 5
posiciones genuinamente independientes pagaba 5·σ — el crédito de diversificación
(√5·σ ≈ 2.24·σ) que la teoría de carteras da desde Markowitz estaba negado por un
literal, y el motor sobre- vetaba exactamente los grupos diversificados que un
universo multi-activo debería preferir.

## 2. La reparación: agregación de varianza con ρ̄ estructural

`veto_por_exposicion_estructural(k, σ, q, ρ̄_efectivo)`:
- **σ_grupo = σ·√(k + k(k−1)·ρ̄)** — la fórmula clásica de varianza de cartera con
  correlación media. ρ̄→1 reproduce EXACTO el k·σ histórico (misma apuesta);
  ρ̄→0 acredita √k (independientes); ρ̄<0 (cobertura REAL, estructura consistente)
  baja hasta el piso de una apuesta individual — jamás por debajo (clamp var ≥ 1).
- **ρ̄ = None ⇒ forma lineal histórica** (fail-safe: sin matriz medida no se
  inventa estructura). La firma legada delega con None — todos los tests
  históricos intactos.
- **AllNoise (MP) ⇒ ρ̄ = 0 explícito**: coherencia con "el ruido ni veta ni
  agrava" de XLI — si toda la correlación es ruido, el grupo ES independiente.

## 3. Curl de Hodge / balance de grafos firmados: la inconsistencia se cobra

`curl_share_desbalanceado`: fracción de triángulos del grupo con
**w_ij·w_jk·w_ki < 0** (grafo firmado no balanceado, Harary; el "curl" del
1-cochain sobre la 2-celda del clique en Hodge discreto). Un triángulo A~B, B~C
pero A≁C es estructura que ningún factor común explica: la correlación por pares
subestima el riesgo del cluster justo cuando parece diversificado.

`rho_efectivo_para_agregacion`: ρ̄ crudo **inflado hacia 1 por el desbalance**
(ρ̄≥0: ρ+(1−ρ)·curl; ρ̄<0: la cobertura se acredita sólo ×(1−curl) — una
"cobertura" montada sobre triángulos inconsistentes no es cobertura). La
inconsistencia no es diversificación; es información que ρ̄ por pares no captura
y se cobra.

## 4. Cadena completa del veto multi-activo (XLI→XLIV)

HY mide sin sesgo de reloj → MP separa estructura de ruido (el ruido no veta,
ρ̄=0) → Hodge/curl detecta inconsistencia estructural (la cobrada) → agregación
de varianza decide con la estructura completa. Cada eslabón añadido por una ola,
cada uno con tests que rompieron la versión anterior.

## 4.5. BONUS de la validación: r11/FMT-216 — el "flaky" era un bug real con testigo

La validación de esta ola dejó de dar por bueno el `test_r11_evolution_pipeline`
("flaky por RNG", veníamos documentándolo desde XLI): fallaba 3/4 corridas. Un
diagnóstico DETERMINISTA (seeded, 20 000 semillas) clavó el testigo exacto
(seed=199, rate=0.5) y la cadena causal completa:

1. El PASO 2 de `enforce_curve_rr` satisface el invariante RR **estrechando la
   SL sin límite inferior**: podía dejar ambas anclas por debajo del piso de
   fricción — RR contento, banda operable VACÍA, gate de promoción rechazando.
2. La normalización ingenua post-reparación no sobrevivía: `from_vector`
   re-ejecuta el mismo `enforce_curve_rr` sobre la curva serializada y volvía
   a hundirla.

**Reparación (tres piezas coherentes)**: (a) guard de piso en el PASO 2 — si el
recorte dejaría cualquier ancla bajo el piso, se deshace y el RR queda para el
PASO 3 (curvas paralelas, que re-garantizan banda); (b) normalización por anclas
como ÚLTIMO escritor de la mutación (no-op bit-exacto si ya cumple — el contrato
D-658 "rate=0 no toca curvas" exige cero deriva); (c) el diagnóstico abierto
FMT-216, que había CONGELADO al seed=199 como testigo del defecto, se cierra como
contrato de regresión (su propio mensaje lo ordenaba: "re-evaluate this open
diagnostic"). El fixture de D-658 usaba curvas que violaban piso Y RR a la vez —
actualizado a un fixture admisible (el contrato "rate=0 intacto" sólo es exigible
a curvas ya admisibles).

**Estado**: r11 6/6 verde; diag determinista 20k semillas verde; FMT-216 CERRADO.

## 5. Pruebas

4 tests nuevos: factor común balanceado (curl 0); triángulo inconsistente
detectado (curl 1, ρ̄ efectivo > crudo); interpolación ρ̄=1/0/None contra el
k·σ y √k exactos con firma legada intacta; cobertura consistente vs
independientes separadas en el umbral correcto. risk-engine: 83+22 verdes.

## 6. Hoja de ruta (hereda XLIII §5)

1. Effective number of bets (Σ√λ)²/Σλ con deflación espectral — el diagnóstico
   fino de dimensión del grupo (hoy ρ̄ basta; el paso a λ's es directo).
2. Hawkes multivariado (T03): contagio cross-asset dinámico.
3. Reactivar actuador BOCPD con distribución de p_transition de tape real.
4. Hodge sobre el grafo de SEÑALES interno (ramas↔vetos) — el T05 original.
5. bt↔vivo (entregable mayor).

## 7. Cierre del tramo

El último veto que contaba en lugar de medir ya mide. La cadena multi-activo del
universo espectral queda completa: relojes honestos (HY), estructura vs ruido
(MP), consistencia estructural (Hodge) y agregación de varianza con la forma
funcional exacta de la teoría de carteras — sin literales, con ρ̄=None como único
fallback y documentado.
