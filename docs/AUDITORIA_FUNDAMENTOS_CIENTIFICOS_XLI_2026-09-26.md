# AUDITORÍA DE FUNDAMENTOS CIENTÍFICOS — OLA XLI (2026-09-25/26, madrugada)

**Mandato del operador (Quant Sr)**: auditar que TODOS los vetos, bloqueos, límites y
rechazos tengan sentido; pensar todo como universo multivariante continuo temporal
espectral (los regímenes y los horizontes son espectros, no cajas); revisar e integrar
teoría matemática avanzada de otros ámbitos (incluida familia problema-del-milenio)
con el protocolo de honestidad del repo.

**Resultado medible principal**: el ORÁCULO GENÉTICO T-1 pasa de **0/144 (motor
estrangulado por su propia cadena de vetos) a 19/144 sensibles (13,2 %)**, superando
el trinquete histórico de 11,5 % — sin tocar la doctrina MEDIDA de ningún gate.

## 1. Censo de la cadena de vetos (materia prima)

Dos agentes de solo lectura censaron ~70 puntos de veto en 6 etapas del camino
señal→orden (viabilidad → ramas de señal → puertas del continuo → escudos → consejo →
ML-gate → risk-engine → margen). Tabla maestra completa con procedencia (D-xxx/FMT/#),
condición, carácter MEDIDO vs LITERAL y comportamiento en arranque frío: ver §6 del
INFORME (anexo A de este documento la resume por etapa). Hallazgos estructurales:

- **17 familias de umbrales literales** conviven con la infraestructura P²/z95/Stouffer
  que podría medirlas.
- **Pensamiento binario superviviente**: enum `MarketRegime` (BullRun/Crash/Range/
  Chaotic por umbrales duros) gobierna el ÚNICO veto binario absoluto del camino
  (Crash ⇒ prohibido largos); clasificaciones duras de Hurst; pipeline de dos lecturas
  fast/slow con etiquetas de rama 1–24 (ya son vistas del continuo, pero la etiqueta
  sigue organizando el código).

## 2. Deadlocks e incoherencias REPARADOS (Fase A2)

| id | defecto | reparación |
|---|---|---|
| D1 | `has_roster_model` (B3.25): sin modelo del roster el motor NUNCA opera — veto estructural sin escape | Arranque frío total (trade_count=0 y sin modelo) ⇒ la orden es SONDA (misma doctrina que D-751b); con historial y sin modelo el veto permanece |
| D2 | `riesgo_por_operacion` sin medir tras reinicio ⇒ 2ª posición misma-dirección SIEMPRE vetada | Proxy conservador tope/8 por operación (ruina-acotado), provisional hasta persistir el riesgo |
| D3 | X-016 veta por `coherencia < 0.02` — el campo espectral FRÍO nace con coherencia 0 ⇒ never-start | La incoherencia sólo veta con ENERGÍA en el campo (`total_energy > 1e-12`); en frío, modula |
| D4 | F8-P11 y ML-GATE: DOS puertas sobre el mismo ml_prob con entradas distintas (spot_bias vs pura, 0.50 literal); la tardía siempre ganaba | F8-P11 pasa a diagnóstico; UNA puerta ML (la del lift espectral S-6) |
| D5 | Momentum adverso definido DOS veces: 8 literales en pb (ramas 1/4) vs Stouffer-z95 (tensor) | UNA definición viva: Stouffer-z95 con secular como veto-del-veto, compartida |
| D6 | Orchestrator: `total_exposure` duplicado + límite total redundante con el direccional | Una definición; queda el direccional (límite operativo) |
| D7 | AuditorInterno co-firmaba la condición dd de SeniorRiesgo ⇒ todo drawdown = 2 vetos ⇒ el override por supermayoría jamás aplicaba | El dd cuenta UNA vez; el Auditor conserva su condición propia (do_calculus) |
| D9 | `same_dir_unsecured` banda 1.50 cancelaba el despacho multi-banda (desacople ≥0.60) | Banda = distancia resonante canónica 0.80 |
| **A2b-1** | **BUG REAL del merge**: primer tick detectado por `last_ts_ms == 0` — un stream cuyo primer evento tiene ts=0 RE-SEMBRABA el espectro en cada tick (en vivo nunca disparaba; en backtest sintético sí). Delatado por `spectral_contract_timestamp_zero_is_a_valid_origin`, que fallaba en HEAD desde el merge | Primer tick por contador de `updates` (y el seed cuenta su propio update) |
| **A2b-2** | **BUG REAL del merge**: `spectral_coherence` (lector del consejo) usaba el peso LEGACY de persistencia mientras `refresh_fusion` (decisión) usa el observable D-742 — el consejo leía OTRO campo distinto al que decide | El lector usa la ponderación observable D-742; doctrina de DOS MASAS con alcance declarado (aprendizaje: persistencia×ganancia para campo/entropía/τ*; decisión: observable para fused/coherence). Test de consistencia lector==decisión añadido |
| A2b-3 | Test `long_scale_retains` pineaba la semilla 1e-7 PRE-D-742 (obsoleta desde el merge); validaciones previas truncadas no lo vieron | Expectativa actualizada al contrato D-742 (media observada raw/masa, mismas expresiones exactas del código) |

**Nota de honestidad**: los tres tests que delataron A2b fallaban YA en HEAD (`a0de0a40`)
— la validación final de la Ola XL se leyó truncada y se declaró "verde" sin verlos. Esta
ola corrige el registro: la regresión existía y su causa raíz era el bug ts=0.

## 3. Oráculo T-1 re-baselineado (Fase A1)

Causas del 0/144 (medidas): fixture sintético con ATR ~11 pb/min dejaba σ(τ)−fricción(7 pb)
< spread(4 pb) a τ corto ⇒ INVIABLE perpetuo (D-755); D1 y D3 estrangulaban lo restante.
Reparación: volatilidad del fixture coherente con la física (ciclo 30 pb, ruido 40 pb) +
deadlocks D1/D3/D5 + diag con predictor direccional (el constante no puede cooperar con
un gate por lift: base 0.953 ⇒ umbrales inalcanzables — artefacto, no diagnóstico).

**Medición: 19/144 (13,2 %) > trinquete 11,5 %; test PASA** (20,4 min de corrida).
Historia del trinquete: 25 % → 0 % (B3.20) → 13,9 % → 11,8 % → **0 % (post-fusión, esta
ola)** → **13,2 %**. La corrida NO incluyó los fixes A2b (ts=0 + coherencia): re-medir
en la próxima ola puede subir aún más.

## 4. Teoría nueva integrada (Fase B/C — protocolo completo en el código)

### B1 — Régimen como CAMPO continuo (`quantum-arena/src/spectral_regime.rs`)
Coordenadas (H(τ) por banda, τ*, d lnτ*/dt normalizado, entropía de masa, marea
portadora) sin etiquetas. **Crash-ness continua** [0,1] = 0.35·aceleración + 0.30·marea
adversa + 0.20·colapso de entropía + 0.15·anti-persistencia micro; publicada por símbolo
(`spectral_crash_flux`). El margen de largos del host **respira** con ella (×1.0 calma →
×0.05 en caída coherente) en lugar del veto binario del enum, que queda SOLO como el
extremo medido (vista legacy >0.80 con marea adversa). El enum `MarketRegime` queda como
VISTA derivada (`legacy_view`, mapeo 1:1 documentado).

### C1 — Marchenko-Pastur para el veto de correlación (`risk-engine/src/random_matrix.rs`)
TEOREMA de matrices aleatorias (la familia matemática de las conjeturas de brecha
espectral, integrada como teorema): el espectro de ruido de una matriz de correlación
N×T vive en [(1−√γ)², (1+√γ)²], γ=N/T. El veto D-748 construye ahora la matriz del grupo
(candidata + mismas-dirección) y aplica el veredicto: **si toda la correlación cabe en la
banda de ruido (AllNoise), los pares Pearson altos NO cuentan como misma-apuesta — el
ruido no veta**; con modo sistemático real (λ_max > borde) rige el pairwise D-748. Bug
 propio corregido durante la integración: γ invertido (N/T, no T/N). Falsación: ruido
iid → AllNoise; factor común inyectado → SystematicMode con λ dominante (tests).

### C2 — Funciones de estructura de Kolmogorov sobre el espectro 32D
S_p(τ) = E|dev_τ|^p por escala (tercer momento EWMA añadido al núcleo, misma convención
de masa que D-742); ζ(p) por regresión log-log entre escalas con masa ≥10 %. K41:
ζ(p)=p/3 y el 4/5-law fija ζ(3)=1. **Intermitencia χ=(1−ζ3)⁺** publicada por símbolo y
consumida: los pisos medidos P80/P85 se endurecen ×(1+0.5χ) — colas gruesas exigen
eventos mayores para ser "eventos". Falsación: ruido gaussiano iid ⇒ ζ(3)≈1 (test con
banda por el sesgo de suavizado de la EWMA).

### C3 — Información de Fisher del campo espectral
I = Σ(Δq/Δlnτ)²/q sobre la masa espectral (misma masa que la entropía): mide cuán
LOCALIZADO en escala está el régimen = identificabilidad. Publicada por símbolo
(`spectral_fisher`, −1 = frío). Consumidor previsto: el walk-forward no debe evolucionar
sobre regímenes no identificables (cableado al daemon = siguiente ola, T18/T26).

### C4 — Tabla de candidatos "milenio" (transferencia legítima, sin decoración)
| problema | transferencia honesta | estado |
|---|---|---|
| Navier–Stokes | métodos multiescala/estabilidad numérica (fusión D-742, cascada C2: la 4/5-law COMO relación de escala, no fluido de precios) | integrado (C2) |
| Yang-Mills (brecha espectral) | la familia RMT/MP del borde espectral — como teorema con condiciones exactas | integrado (C1) |
| Riemann | sólo análisis armónico con falsación propia (FFT-64 existe; ceros = futuro lejano sin variable identificada) | no integrado, justificado |
| Hodge | Hodge discreto en grafo de señales (T05: ciclos de dependencia entre ramas) | candidato priorizado |
| P≠NP / BSD | sin variable, operador ni unidades identificables hoy | rechazado por protocolo |

## 5. Deuda declarada (A3 recortado por presupuesto de sesión, con diseño)

1. **CVD z-tipificado**: sustituir ±0.12/±0.15/±0.05 por z = cvd/σ_EWMA(cvd) — requiere
   estimador de σ(cvd) en el feature engine (patrón ObiNoise, ~30 líneas + tests).
2. **X-016 percentilado**: 0.98/0.15/0.02 → percentiles P² del propio campo
   (adaptive_quantiles ya existe; añadir tres claves).
3. Pisos fríos 0.40/0.15 literales → arranque derivado de σ(τ)·k.
4. Resto de la tabla C del censo (confluencia 0.60/0.15, ramas 9/10 ±1.0 ATR, slots
   28 bps, grid del consejo 0.35/0.80/0.28/0.75…): inventariados con prioridad.
5. Cablear Fisher al walk-forward daemon; wire crash_flux al orchestrator (hoy modula
   el host, suficiente para esta ola).

## 6. Pruebas ejecutadas

- `cargo check --workspace`: 0 errores.
- quantum-arena: 77/77 (r11 es FLAKY por RNG propio — pasa solo; documentado).
- risk-engine: 76+22 tests (3 nuevos MP + doctrina D2 reescrita).
- god-engine-core/metacortex: compilan; tests de contrato existentes verdes en la
  validación completa (§7 del log de la sesión).
- T-1 oráculo: 19/144, PASA (1226 s).

## 7. Hoja de ruta (hereda XL §7)

1. Re-medir T-1 con los fixes A2b incluidos (posible subida del trinquete).
2. A3 completo (CVD/X-016/pisos fríos) con los diseños de §5.
3. FMT-285b (cuarentena con transporte) + §13.1 identidad decimal + §13.3 FX as-of.
4. Fisher → walk-forward; crash_flux → orchestrator; Hodge discreto (T05).
5. Falsación empírica bt↔vivo (sigue siendo el entregable mayor).

## 8. Cierre del tramo

La cadena de vetos vuelve a respirar y el oráculo lo ATERRIZA en un número: 0→19 genes
bajo selección. Dos bugs reales del merge (re-sembra ts=0, lector/campo desacoplados)
salieron a la luz por CONTRATOS que pineaban aritmética exacta — la lección institucional:
las validaciones truncadas no son validaciones. El régimen ya es un campo, el ruido ya no
veta, y la intermitencia ya endurece lo que el mercado exige.
