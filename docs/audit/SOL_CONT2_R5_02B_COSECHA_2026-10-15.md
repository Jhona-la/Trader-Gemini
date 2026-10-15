# SOL-CONT2-2026-10-15 — SOL-R5-02-B: la cosecha del bosque sombra puntúa equity MARCADA

**Fecha**: 2026-10-15 · **Autor**: Sol (línea Backtest & Accounting / R6 aprender-medir)
**Base final**: `44561f59` (main con GLM-123/124 y Ω131-134) · **Ola**: R7-R6
**Rama**: `sol/r7-r6-evidencia-2026-10-15` (worktree aislado `.sol-r6-2026-10-15`)

## 0. Qué pasó con SOL-R5-02 y por qué existe este documento

GLM cerró SOL-R5-02 en el camino del **replay** (`booktick_replay.rs`,
MTM del capital final con fricción de salida — verificado, correcto).
Medí el mismo defecto en el camino de la **cosecha del bosque sombra**,
que es el que **promueve** genomas: `ShadowForest::harvest_best_genome`
leía `unified_capital` (caja) para puntuar cada universo. Es la novena
confirmación del patrón «dos caras sin reconciliar»: dos caminos de
selección con contabilidades distintas.

## 1. Hallazgo — evidencia medida (RED, antes del fix)

Anclas en `crates/evolution-engine/src/random_forest.rs` (base `b46dd286`):

- **Caja pura en la cosecha** (línea 229, régimen anterior):
  `let cap = engine.arena.unified_capital.load(...)` — con una posición
  abierta ganadora al momento de la cosecha, su marca NO entraba en el
  fitness. Medido por el contrato:
  `score 0.000000 != esperado 0.073558` (largo entry 100, qty 0.1, mid
  110 ⇒ equity marcada 13.9923; la caja plana 13.0 puntuaba igual y el
  universo ganador no ganaba la cosecha).
- **Amplificador de pérdidas IS disfrazado de OOS** (líneas 238-239,
  régimen anterior): `oos_start_capital: self.initial_capital,
  oos_end_capital: cap` — el slot OOS cargaba la trayectoria IS. El
  factor `1 + |ln(cap/inicial)|` multiplicaba pérdidas que `ln` ya
  penaliza. Medido: universo −50 % IS con dd observado 0.5 puntuaba
  **−2.3472** donde el crecimiento log puro con su drawdown da
  **−1.3863** (×1.6931 exacto del factor). No es degradación OOS: es
  doble conteo de la misma pérdida.
- Precedentes que fijan la doctrina correcta: el examen del demonio
  (`online_daemon.rs:776-777`, LG-7) **ya** marca a mercado con la misma
  fricción (taker + piso de deslizamiento), y la convención OOS sin
  split está documentada en `fitness.rs` («endpoints iguales y positivos
  mantienen la convención legacy sin split»), usada por el pre-screen
  (`online_daemon.rs:1915-1916`) y por el `evolver` en su flujo de
  selección.

## 2. Fix (GREEN)

`crates/evolution-engine/src/random_forest.rs`:

1. Nueva función libre `capital_marcado(engine)`: caja + valor a
   mercado de las posiciones abiertas al último precio transmitido
   (`coin.current_price`, vivo por `broadcast_tick` — sin problema de
   reloj), neto de fricción de salida (taker + `base_slippage_floor`).
   Sin posiciones abiertas es bit-idéntico a la caja.
2. `harvest_best_genome` puntúa y observa drawdown sobre la equity
   MARCADA; slot OOS con endpoints iguales (convención legacy, factor
   1.0) — el amplificador de pérdidas IS eliminado.
3. La rama de pánico de memoria (`panic_memory_dump`) usa la misma
   marca.
4. Helper de tests `expected_fitness` alineado a la convención.

## 3. Pruebas (recibos)

| prueba | resultado | recibo |
|---|---|---|
| Contrato RED `sol_r5_02_b_la_cosecha_marca_la_posicion_abierta` | **FALLÓ**: score 0.000000 vs 0.073558 | `outputs/Sol_r6_sol_r5_02_b_RED.log` |
| Contrato RED `sol_r5_02_b_sin_split_el_factor_oos_no_amplifica_perdidas_is` | **FALLÓ**: −2.3472 vs −1.3863 | ídem |
| Contratos GREEN (mismos, post-fix) | **2/2 ok** | `outputs/Sol_r6_sol_r5_02_b_GREEN.log` |
| Suite `evolution-engine` completa | **153 ok, 0 failed** (lib 68/68 + 11 targets de integración) | `outputs/Sol_r6_evolution_suite.log` |
| Oráculo T-1 (toca evolución) | ver §4 | `outputs/t1_oraculo_sol_r6_2026-10-15.log` |

**Alcance honesto**: el oráculo mide cobertura genética/expresividad
sobre el fixture, NO rentabilidad; las suites verdes no certifican
crecimiento económico. Cero trading, cero entrenamiento en vivo, cero
promoción fuera del entorno `backtest`.

## 4. Oráculo T-1

- **Incidente de herramienta**: `scripts/t1_oraculo.sh` aborta en este
  sandbox con `[safe-delete][SAFE_DELETE_INVALID_PATH] embedded drive
  prefix is not allowed` — su `rm -f "$tmp"` del índice temporal recibe
  una ruta con prefijo de unidad Windows embebido. No es un fallo del
  repo: es el guardia del entorno. Workaround usado (misma corrida, sin
  el `rm` del script): invocación directa
  `cargo test --release -p backtest-engine --locked --test
  t1_cobertura_genetica -- --exact
  t1_cobertura_genetica_del_oraculo_de_aptitud --nocapture
  --test-threads=1` con `CARGO_TARGET_DIR=<worktree>/target-portable`,
  `T1_THREADS=4` — el script sólo añade el sello de árbol sobre ese
  comando. Reportado al consejo para quien quiera arreglar el script
  (p. ej. `tmp` vía `mktemp` dentro del target sin conversión de ruta).
- **VEREDICTO: PASA — exit 0, 18/144 genes sensibles = 12,5 % ≥
  trinquete 11,0 %** (corrida 1965,82 s; total con build 8950 s).
  Lista sensible: `[1, 10, 11, 17, 18, 20, 24, 27, 33, 39, 68, 69, 129,
  130, 131, 141, 142, 143]`. Delta contra la canónica de 16
  (`[1,10,11,17,18,20,24,27,32,33,68,69,129,130,131,141]`): entran 39,
  142, 143 (genomas de trailing/margen ahora discriminados por el
  objetivo marcado) y sale 32 — coherente con un cambio DELIBERADO del
  objetivo de cosecha, no con una pérdida de expresividad. Recibo:
  `outputs/t1_oraculo_sol_r6_2026-10-15.log`. El oráculo mide
  expresividad genética sobre el fixture, NO rentabilidad.

## 5. Residuales reportados a GLM (su archivo, no tocado)

- **SOL-R4-01 CERRADO por GLM-124** (`glm/124-sol-r4-01-span`, en
  `44561f59`): campos `executed_ticks`, `executed_last_ts`,
  `close_reason` — exactamente lo que pedía mi recomendación. Verificado
  por diff. **Gracias, GLM.**
- **Reloj en el MTM del replay — SIGUE ABIERTO** (fix GLM-123):
  `booktick_replay.rs:744` (`aplicar_mtm_final` con
  `ultimo_mid_por_moneda(ticks)`) valúa las posiciones al último mid de
  la CINTA COMPLETA también tras el `break` por ruina (línea 730) —
  dato posterior al break en la valuación del fitness. Con GLM-124 la
  dirección es ahora trivial: el loop ya trackea `executed_last_ts`;
  basta capturar el mid del último tick **procesado** y pasarlo a
  `aplicar_mtm_final`. Caso raro (ruina), principio exacto (ningún dato
  futuro en el fitness).
- **`evolver.rs:104`**: `Evaluacion.final_capital` sigue en caja pura.
  Documentado (LG-17b: el evolver no promueve todavía); si algún día
  promueve, necesita la misma marca.

## 6. Lote R6 leído completo — sin hallazgo

`evolution-engine/src/fitness.rs` (utilidad log penalizada por drawdown²;
contracción bayesiana documentada; tests doctrinales), `return_evidence.rs`
(suma de Kahan correcta; t-estadístico convencional `mean/σ·√n`; vigilante
de degradación de Ville con evidencia de Monte Carlo en comentarios),
`score_retention.rs` (monotonía de la penalización firmada correcta).
Lectura completa con rangos; sin defecto medido.
