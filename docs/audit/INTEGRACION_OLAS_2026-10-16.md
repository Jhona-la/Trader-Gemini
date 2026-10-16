# Auditoría de integración de las olas matemáticas Ω136-Ω143 (GLM 130, 2026-10-16)

Encargo del operador: «AGY está implementando mucha matemática… revisa
que la integración sea perfecta». Este informe ES esa revisión. Base:
main `0bf99f60a` + exploración con anclas de código verificadas (no
documentos). Ω143 (Wasserstein) estaba SIN commitear en el working tree
al momento de la exploración — se incluye con esa salvedad.

## El mapa: solo 1 de 8 olas llega viva a la decisión

Los caminos de decisión son dos: (1) la **sombra espectral**
(god-engine-core lib.rs:2950-3370 → `consenso_espectral_dominante`, la
vía que MANDA — orchestrator.rs:388-407 la consume antes que los votos
escalares) y (2) el **orchestrator de votos escalares** (19 fuentes,
:1266-1328 → encuesta por tick/moneda :6551).

| Ola | Módulo (crate) | Instanciado | Consumidor vivo | Veredicto |
|---|---|---|---|---|
| Ω136 | OnlineNewtonGbdt (evolution-engine) | daemon de evolución (no core) | freno del bosque (sizing, lib.rs:8203) + umbrales ML del daemon | **PARCIAL** — camino lateral; su feed `shadow_evaluate` legacy usa features CONSTANTES |
| Ω137 | HelmholtzHodge (dark-alpha) | sí, per-coin, datos reales | **sombra espectral** (:3148 `voto_espectral_con_helmholtz`) | **CABLEADO** — la única ola viva en la decisión |
| Ω138 | YangMills SU(N) (strategy-core) | motor físico preexistente + voter duplicado | voter-passthrough y confluence — ambos censados 0 votos | **PARCIAL** — consumidores muertos |
| Ω139 | Kähler/Hodge (strategy-core) | sí, multiactivo real | solo confluence (mudo) | **ISLA** |
| Ω140 | Riemann/GUE (risk-engine) | sí, per-coin | solo confluence (mudo) | **ISLA** — además su input «energía» es sintético dominado por la escala del precio (BTC ~1e5) |
| 141 | BSD/Frobenius (strategy-core) | sí, per-coin, fase plausible | solo confluence (mudo) | **ISLA** |
| 142 | Atiyah-Singer/Dirac (dark-alpha) | sí, per-coin | solo confluence (mudo) | **ISLA** |
| 143 | Wasserstein (quantum-arena) | sí — SIN commitear | solo confluence (mudo) | **ISLA** (WIP) |

## Los 3 root-causes transversales

1. **La sombra se clavó en Ω137**: `voto_espectral_con_helmholtz`
   (:3154) fue la ÚLTIMA actualización del voto vivo; la cadena de
   delegación hace que ym_current, kahler, riemann, bsd, dirac y
   wasserstein defaulteen a 0.0/1.0 (NEUTRO) en todos los votos de la
   sombra. Seis olas de matemática produjeron exactamente cero
   modulación sobre la decisión.
2. **Todo el eje Ω138-143 cuelga de dos motores censados en 0 votos**
   (GLM-129): los multiplicadores bonifican/atenúan un voto que nunca
   se emite. El gate del confluence exige SIMULTÁNEOS hawkes ≥
   1.6+(gen−0.5)·2 (~1.7), |OBI| ≥ p80·gen, y lift de ML
   `ml_prob ≥ base+0.05` — con base = 0.5 para monedas sin modelo del
   roster (lift inalcanzable en la práctica). La Fase B de este ciclo
   mide cuál puerta domina.
3. **Los contract tests certifican con puertas sintéticas**:
   `build_base_long_registry()` fuerza hawkes 2.0 / OBI 0.5 /
   ml_prob 0.85 vs base 0.30 (¡+55 pp de lift!) — certifican el
   multiplicador, no que el motor hable. Suite verde mientras el
   sistema real calla: exactamente la divergencia que el censo GLM-129
   expuso al medir tape REAL.

## Matices del vehículo de medición (honestidad sobre el propio censo)

- **StatArb**: el replay del censo no corre feed SPOT (futures tape) →
  0 pares → abstención eterna EN EL REPLAY. En vivo el feed SÍ arranca
  (god_engine.rs:1982 → spot_feed). No asumir muerto en producción sin
  censo de live.
- **YangMills**: las corrientes J_i nacen de 3-ciclos de monedas
  distintas; el replay es unimoneda (coin 0) → 0 triángulos. El roster
  vivo (7 símbolos) sí produciría corrientes. El voter-passthrough
  además depende de que `yang_mills_current` sea ≠0.
- **FlowExcitation**: ese sí es mudo por gate (las claves SÍ se
  publican antes de la encuesta — no es race): es el cuello de botella
  real y el objeto de la Fase B.

## Qué hace cada parte de este ciclo con el mapa

- **Fase B** (medición de puertas): el recibo que localiza el sub-gate
  dominante del confluence — insumo para que AGY/consejo reabra la
  puerta con diseño (no lo cambio unilateralmente: motor recién
  cableado por Ω138).
- **Fase C** (Hedge por fuente): teoría nueva GLM que NACE integrada
  (al orchestrator vivo, no como isla #9) y que además devalúa
  automáticamente a cualquier fuente muda o contraproducente —
  paliativo estructural mientras las islas se deciden.

## Fase B — EL RECIBO: la puerta L es imposible por construcción en backtest

`tests/puertas_confluence_medicion.rs` (#[ignore], tape REAL BTC sep
2M ticks, 781.9s, genoma activo, core de producción):

| Puerta | Apertura sola |
|---|---|
| (H) hawkes ≥ ~1.7 | **55.6%** (p50=1.74, p90=2.48 — hay ráfaga la mitad del tiempo) |
| (O) \|OBI\| ≥ p80·gene | **20.5%** |
| (L) lift ML ±0.05 | **0.000%** — \|lift\| p50=p90=p99 = **0.0000 EXACTO** |
| H∧O | 10.99% |
| TODAS | **0.000%** |

**Root-cause exacto** (la identidad, no la escasez): en el vehículo de
backtest/evolución no hay modelos de disco ⇒ `ml_model_base =
active_forest.map(base_prob).unwrap_or(0.5)` (lib.rs:5714-5717) cae a
**0.5**, y el ensamble sin modelos también es neutral **0.5** (CL-15,
:5897 zona `base_ml_prob`) ⇒ `ml_prob_motor ≡ ml_model_base ≡ 0.5` ⇒
lift ≡ 0 con spot_bias=0 (sin feed spot en replay). La puerta L no
está "cerrada por rareza": es MATEMÁTICAMENTE imposible en todo el
pipeline de backtest/evolución.

**Consecuencias**:
1. FlowExcitationConfluence es estructuralmente mudo en TODA evaluación
   de evolución — el genoma no recibe gradiente de ese motor ni de
   ninguna de las olas Ω138-143 colgadas de él (el mismo patrón de
   «gen muerto arrastrado» que el comentario #535 documentó para otro
   gen).
2. En VIVO la puerta puede abrir (modelos del roster cargados ⇒ base
   ≈ 0.18-0.30 HOST-010, ensamble real oscila; spot_bias ±0.05 con
   feed fresco) — falta censo live para cuantificar CUÁNTO.
3. La paradoja del assurance: la suite verde (contracts con puertas
   forzadas) + backtest ciego + vivo tal vez vivo = tres verdades
   distintas del mismo motor.

**Propuesta al consejo**: (i) el replay necesita un modo «con modelos
del roster» (cargar `models/{SYM}_MOTOR.json` en backtest) para que la
evolución vea la puerta L — hoy evoluciona con un motor ciego;
(ii) mientras tanto, el Hedge del 130 (Fase C) ya devalúa fuentes
mudas/erróneas con realización medible — el consenso aprende aunque el
confluence no hable.

## Recomendación al consejo (con el mapa en la mano)

1. Congelar NUEVAS islas hasta que la sombra consuma las existentes
   (un commit «sombra × {ym,kahler,riemann,bsd,dirac}» vale más que
   una ola matemática nueva).
2. Reabrir la puerta del confluence con el recibo de la Fase B.
3. Los contracts de confluence deben incluir UN caso con puertas
   realistas (distribuciones medidas), no sólo las forzadas.
