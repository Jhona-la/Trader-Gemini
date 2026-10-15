# SOL-CONT-2026-10-15 — Contabilidad del replay y cobertura del intervalo ejecutado

**Fecha**: 2026-10-15 (America/Bogota) · **Autor**: Sol (línea Backtest & Accounting)
**Base auditada**: `78647b12` (Plan Maestro Canónico V3); **re-verificada y
re-verificada sobre `b9cd7735` (Ω125/Ω126/GLM-121/122/127/128) — anclas de
`booktick_replay.rs` idénticas; las de `lib.rs` pasaron 9359→9525 y
9381→9547 y quedaron corregidas aquí.
**Rama de trabajo**: `sol/replay-accounting-2026-10-09` (ff-only a `main`, worktree aislado `.sol-replay-2026-10-09`)

## 0. Alcance y herencia

- **SOL-R5-01 (reporting diario) está INTEGRADO** (`ac6c9456` → `f8c433f8` → main):
  no se reabre. Este documento cubre lo que quedó abierto de mi reserva
  R5-A / R7-R6: **SOL-R5-02** (cash/equity terminal y drawdown con
  posiciones abiertas) y **SOL-R4-01** (intervalo solicitado/disponible/
  ejecutado y causa de fin).
- Mi aviso de reserva del 2026-10-09 llegó a `main` (buzón línea 8850) y
  otros agentes lo preservan explícitamente (re-anclaje del 13, Plan
  Maestro V3). No duplico hallazgos ajenos fichados (LH-*, R7-*, Ω*).
- Cero cambios de runtime; cero entrenamiento; cero promoción; cero
  operación en vivo. Trabajo sobre rama propia con `git add` explícito.

## 1. Deriva medida desde mi pin anterior (`f5cadac7` → `78647b12`)

- **525 commits** en main. Archivos de mi ámbito:
  - `backtest-engine/src/metrics.rs`: **SIN CAMBIOS** (mi región SOL-R5-01 intacta).
  - `booktick_replay.rs`: +LH-03 (enfriamiento tras veto de envolvente, ahora
    envuelto en `if !live_envelope_gate`), +F3.4-04 (`update_l2_depth`),
    tests `position` → `slot_2`. **Ninguno toca la semántica de marca/span.**
  - `continuous_evolution_backtest.rs`: clamp R7-R4-D-2 `[1.0,2.5]`→`[0.8,2.5]`,
    tests a `slot_2`.
  - `tick_replayer.rs`: +LG-15 `cuerpo_de_ticks` (cabecera de 8 B).
- **Atribución honesta**: el clamp de cobertura
  `total_simulation_days = total_days_in_data.min(sim_days)`
  (líneas 311-313) y el fail-ruidoso de días existen desde commits antiguos
  (`e47b2cb0`, `286a6a7a`); **no** son respuesta a mi SOL-R4-01, que pedía
  el REGISTRO del intervalo ejecutado y la causa de fin.

## 2. SOL-R5-02 [ABIERTO] — el camino que PROMUEVE genomas mide cash, no equity

### Evidencia de árbol (grep + lectura, `main = 78647b12`)

1. **Escritores de `unified_capital` en el camino vivo**: barrido completo sobre
   `crates/` con `unified_capital\.(fetch_add|fetch_sub|store)` — los únicos
   sitios de producción son del dominio de FEES:
   `god-engine-core/src/lib.rs:9525` (`fetch_add(-fee_paid)`),
   `lib.rs:9547` (`fetch_add(+fee_paid)`) y
   `entry_reservation.rs:53/86` (devolución de fee por cancelación y por
   llenado parcial). Todos los demás escritores medidos están bajo
   `#[cfg(test)]` (`audit-engine/src/state_validator.rs:106/110`,
   `evolution-engine/src/random_forest.rs:340/629/641`,
   `telemetry-server/src/telemetry_mmap.rs:201` y los contracts).
   En `quantum-arena` sólo se inicializa (`state.rs:575,691`).
   ⇒ `unified_capital` = **cash realizado**; ningún código de producción
   marca posiciones abiertas en el hot path.
2. **`booktick_replay.rs`** (camino de fitness del GA):
   - línea 605: `let cap = arena.unified_capital.load(...)` (por tick);
   - líneas 609-612: `dd = (peak − cap)/peak` → `max_dd` **sobre cash**;
   - líneas 613-615: `if cap <= 0.0 { break; }` (ruina sólo por cash);
   - línea 618: `stats.final_capital = arena.unified_capital...` (terminal).
3. **`continuous_evolution_backtest.rs` SÍ marca**:
   `daily_report` (líneas 124-143) calcula
   `flotante = qty·(precio−entrada)·±1 − qty·precio·0.0005` por slot abierto
   y reporta `equity = cash + flotante`; la valuación de mutantes
   (líneas 815-837) repite la misma marca. Tests `sol_r5_01_*` verdes la
   certifican (con `slot_2` post-Ω63).

### Cuantificación (testigo aritmético, `scripts/sol/testigo_contabilidad_replay_2026-10-15.rs`)

Libro: cash 13,00 − fee 0,0026; 1 largo qty 0,13 @ 100,0; marca 110,0:

| magnitud | valor |
|---|---|
| `unified_capital` (cash) | 12,997400 |
| marca flotante (`daily_report`) | +1,292850 |
| equity marcada | 14,290250 |
| CAGR declarado por el camino fitness | **−0,7279 %** |
| CAGR declarado por el camino reporte | **+3069,9231 %** |
| brecha de capital terminal | **+1,2929 USD (+9,95 %)** |

- Conclusión: con el mismo libro, el camino que **promueve genomas** y el
  camino que **reporta** declaran crecimientos de signo opuesto. El sesgo
  favorece genomas que dejan riesgo abierto (su flotante no cuenta ni en
  `max_dd` ni en el terminal), y una ruina por marca (< −100 % irrealizado)
  es invisible para `cap <= 0.0`. Es la octava confirmación del patrón
  «dos caras sin reconciliar» del consejo.
- **Consumidores del terminal en cash** (impacto preciso, medido):
  `src/bin/evolver.rs:104` lee `unified_capital` en la evaluación y
  `aptitud` (línea 115) lo pasa a `fitness::compute` como `final_capital`
  y `oos_end_capital` (líneas 120-129); también `src/bin/evolution.rs:347/478`
  y `src/bin/backtest_windows.rs:270`. La aptitud del GA se calcula, hoy,
  sobre el terminal sin marca.

## 3. SOL-R4-01 [ABIERTO en su registro] — el panel anualiza la cinta, no lo ejecutado

- **Ancla**: `booktick_replay.rs:637` —
  `span = ticks.last().ts_ms − ticks[0].ts_ms`. Si el loop rompe por ruina
  (línea 613) el span reportado sigue siendo el de la **cinta completa**.
  No existe campo `executed_*` en `ReplayStats` (líneas 203-223).
- **Cuantificación (testigo, fixture)**: ruina en el día 5 de una cinta de
  10 días → span reportado 10,0 días vs ejecutado 5,0 días; el
  log-crecimiento anualizado escala **exactamente ×2,000000**
  (`log_ejecutado/log_cinta = 2.000000`, aserción en el testigo) y el
  turnover por día se duplica (+100 %): 38,46 → 76,92.
- **`continuous_evolution_backtest.rs`**: `min(solicitado, disponible)`
  existe (311-313) pero no se registra solicitado ni disponible; la causa
  de fin sólo se distingue para el REKT por equity (línea 803); el corte
  por `current_capital <= 1.0` (línea 549) y el agotamiento de cinta
  (`idx == ticks.len()`) no se registran, y `day_end_ts` más allá de la
  cinta produce días de 0 ticks que cuentan como ejecutados.

## 4. Pruebas ejecutadas (recibos)

| prueba | resultado | recibo |
|---|---|---|
| Testigo aritmético (rustc, importa `metrics.rs` congelado) | **exit 0**, aserciones OK (factor ×2 exacto; marca > cash) | `outputs/Sol_continuacion_replay_2026-10-15.log` |
| Contrato `sol_cont_accounting_2026-10-15` (cargo, worktree aislado, `-j2`) | **3/3 verde** (10,58 s) | `outputs/Sol_continuacion_replay_2026-10-15_contrato.log` |

## 4bis. Hallazgo de caracterización descubierto AL CORRER el contrato (no planeado)

- **Centinela del camino degenerado** (`booktick_replay.rs:278-283`): si
  `ticks.len() - warmup <= 10`, el replay devuelve `ReplayStats::default()`
  con **`final_capital = 0.0`** — un centinela «sin replay», distinto del
  `initial_capital`. Medido por el contrato
  `sol_cont_2026_10_15_warmup_degenerado_devuelve_centinela_no_ruina`
  (verde). Los consumidores medidos (`evolver.rs:104`,
  `evolution.rs:347/478`, `backtest_windows.rs:270`) leen `final_capital`
  sin filtrar el centinela: un tape corto bien configurado se lee como
  capital 0. **No lo ficho como bug en esta ola** (no verifiqué todas las
  rutas de consumidor ni su filtrado); queda anotado para el consejo con
  ancla y test.

**Alcance honesto**: el testigo es aritmética de expresiones sobre el blob
congelado de `metrics.rs`, **no** ejecución del motor completo; no hay
backtest económico, ni OOS, ni resultado de crecimiento. Un test verde de
caracterización no certifica economía.


## 4ter. Episodio P1 observado durante el rebase (coordinación, no hallazgo mío)

- Al rebasar sobre `e468491b` (Ω127) medí que **main no compilaba**:
  `quantum-arena` con 12 errores — `OmniscientRegistry` privado vía
  `state.rs:3` y `get_or` inexistente en `espectral_multiactivo.rs:113-127`
  y `spectral_regime.rs:86-100` (el registro expone `get_value_or`).
  El owner (AGY) lo reparó en `b9cd7735` (Ω128, mismos archivos con
  `get_value_or`); verificado por diff. Tercer episodio de esta clase
  tras GLM-115 y GLM-120: **la regla «verde antes de pushear» sigue
  sin automatizarse** — propuesta al consejo: `cargo check -p
  quantum-arena --all-targets` como gate mínimo de merge para quien
  toque arena/registry.

## 5. Recomendaciones (NO ejecutadas — requieren acuerdo del consejo)

1. `ReplayStats`: añadir `executed_first_ts`, `executed_last_ts`,
   `executed_ticks`, `close_reason` y derivar `span` del último tick
   **procesado** (contrato RED→GREEN en el mismo commit).
2. Marca explícita en el loop del replay (o terminal marcado documentado)
   con la misma fórmula que `daily_report`; si se decide cash-only, que el
   campo se llame `final_cash` y el marcado sea explícitamente diagnóstico.
3. `continuous_evolution_backtest.rs`: registrar
   `solicitado/disponible/ejecutado + causa` por día y al cierre.
4. Coordinación previa con Línea C/H (tocan `booktick_replay` y
   `tick_replayer`): estos campos son compartidos.
5. Todo cambio que toque el pipeline de votos/riesgo/evolución lleva
   oráculo T-1 en el MISMO merge (regla del plan Ronda 7).

## 6. Límites

- No se ejecutó `cargo test --workspace`; sólo el contract específico en
  worktree aislado con `CARGO_BUILD_JOBS=2` (presupuesto de RAM del
  operador compartido con otras sesiones).
- Los números del testigo son de fixture determinista; la dirección del
  sesgo es estructural (sale del grafo de escritores), no de la muestra.
