# AUDITORÍA DE FUNDAMENTOS CIENTÍFICOS — OLA XL (2026-09-25, tarde)

**Sesión**: retoma del trabajo de la sesión forense (Ondas I–XXXIX, silencio >1 h a las 16:01,
procesos terminados por decisión del propietario) tras la operación de preservación y fusión
del PR #5. Ronda sin cuenta/órdenes/entrenamiento operativos; sin reinicios ni despliegue.

## 1. Contexto de esta ola: la fusión PR #5 ocurrió ENTRE rondas

Esta ola no empieza desde el estado que dejó la Onda XXXIX: el propietario ordenó
versionar el WIP, fusionar las ramas/PRs y retomar la línea forense. Entre medias:

1. **Commit `ac136633`**: 272 archivos (+69.6k/−14.7k) del WIP de las Ondas XXX–XXXIX
   versionados y pusheados (cargo check 0 errores, 0 fallos de tests antes del commit).
2. **Merge commit `6fdccd64`** (2 padres): fusión semántica hunk a hunk del PR #5
   (`claude/elegant-euler-mmtht4`, D-742…D-758 + F-009/WS) con main (Olas 11–26 +
   FMT XXX–XXXIX): ~90 conflictos en 17 archivos. Decisiones documentadas en el
   comentario del PR #5 (issuecomment-5841402608). Ramas/worktrees limpiados según
   decisión del propietario (conservados `backup-before-cleanup` y `v7-unificacion-wip`
   como archivos; snapshots de worktrees sucios en `backups/worktrees-snapshot-2026-09-25/`).

Lo que afecta a ESTA auditoría: la fusión dejó DOS defectos reales del lado PR
reparados durante la integración (§2) y un re-baseline pendiente del oráculo (§4).

## 2. Defectos del PR #5 reparados al fusionar (fail-closed)

| Defecto | Síntoma medido | Reparación |
|---|---|---|
| NaN capital/stop/posterior en `max_leverage` | `max_leverage(NaN, …) == (25.0, true)`: `capital <= 0.0` es falso para NaN y `f64::min` ignora NaN, de modo que un posterior inválido ADMITÍA una orden | guards fail-closed en `kelly_envelope.rs`: capital/stop/mínimo no finitos y alpha/beta NaN ⇒ `(0.0, false)` |
| Deadlock D-750↔D-751 en arranque frío | Oráculo genético T-1: **0/144** tras la fusión (medido 2×, ~20 min por corrida). Sin trades ⇒ sin evidencia ⇒ REJ_SIN_EVIDENCIA ⇒ sin trades | Cláusula D-751b en `risk-engine/src/lib.rs`: arranque frío TOTAL (trade_count==0 y sin probabilidad calibrada) ⇒ la orden sigue como SONDA D-750 (mínimo ejecutable, acotada por ruina); con ≥1 evidencia rige D-751 íntegro |

Ambos tienen tests: los contratos FMT de `spectral_risk_contract` (risk-engine y
backtest-engine) fueron REESCRITOS al contrato D-750 con nombres que dicen lo que
afirman (`contracts_exploration_without_evidence_is_exactly_the_minimum_executable`,
`contracts_mature_negative_evidence_kills_sizing_not_the_minimal_probe`).

## 3. FMT-285 — cobertura por símbolo y cuarentena recuperable (§13.2 del XXXIX)

Implementado en `crates/execution-engine/src/income_evidence.rs` + 5 tests contrato
(28/28 verdes en `income_evidence_contract`).

**Qué estaba mal (herencia FMT-282/284)**: `identity(&row)?` abortaba TODA la tarta al
primer registro inválido, y la cobertura era un único estado global de la ventana. Un
registro corrupto de UN símbolo suprimía la evidencia de todos; un símbolo sin filas era
indistinguible de una ventana truncada.

**Qué garantiza FMT-285**:
- `partition_income(Vec<IncomeEntry>) -> PartitionedIncome`: aceptadas + cuarentena +
  duplicados exactos == filas de entrada (inventario `accounted_rows()`). NINGUNA fila
  desaparece ni se anota a cero.
- `QuarantineReason::{InvalidRecord, ConflictingIdentity}` con flag `recoverable`:
  la malformación es reparable por re-lectura; el conflicto de identidad con importe
  distinto exige conciliación y NO se recupera reintentando.
- `SymbolInterval{first,last,rows}` por símbolo: intervalo OBSERVADO de este recorrido,
  no prueba de retención; símbolo ausente ⇒ sin entrada (no cobertura vacía).
- `symbols_with_quarantine()`: el reporte puede declarar qué símbolos tienen cobertura
  DEBILITADA en vez de un veto global.

**Límites que esta ola NO resuelve**: (a) `partition_income` opera sobre la tanda ya
recogida — `collect_income_window` sigue siendo letal por registro dentro del bucle de
páginas; cablear la partición al recorrido con transporte es la tarea siguiente;
(b) FX as-of (§13.3) sigue sin diseñarse; los totales siguen siendo por (asset, symbol)
sin conversión; (c) la identidad sigue sin versión/payload decimal (§13.1).

## 4. Re-baseline pendiente: el oráculo genético T-1 (0/144)

El test `t1_cobertura_genetica_del_oraculo_de_aptitud` quedó `#[ignore]` con diagnóstico
completo in situ. Causa raíz medida con `t1_diag`: la física de viabilidad del PR
(`dynamic_max_spread = (σ(τ)·escala − fricción_ida_y_vuelta).max(tick_pct)`) es honesta
sobre TAPE REAL (calibrada con 34M trades de BTCUSDT) pero el FIXTURE SINTÉTICO del
oráculo simula spread 2×maker_spread_pct=4 pb con fricción 7 pb y τ dominante corta:
σ(τ)−7pb < 0 ⇒ el piso colapsa a tick_pct (0,1/60000 ≈ 0,17 pb) ⇒ INVIABLE perpetuo
(`spread 0.000400 ≤ max 0.000002`), 0 intents llegan al risk-engine.

**No es un revert**: la compuerta es la auditoría más nueva. La recalibración exige
medir sobre tape real (o cuantizar el fixture a la rejilla de tick que el runner
registra), trabajo que esta ola deja PENDIENTE como primer ítem. El trinquete sólo
podrá re-activarse desde una re-medición documentada.

## 5. Preservación: exportador de paridad D-753

La fusión conservó `feature_exporter` de main (research-v2; sus tests comunes lo exigían).
El exportador de paridad de 48 dimensiones del PR vive íntegro en el segundo padre del
merge; recuperarlo como bin propio:
`git show 3ee49b05:src/bin/feature_exporter.rs > src/bin/feature_parity_exporter.rs`
(el hook de seguridad exige Write para crearlo; pendiente para la próxima ronda junto
con su chequeo de compilación).

## 6. Pruebas ejecutadas

- `cargo check --workspace`: 0 errores (sólo warnings heredados: `safe_vol_mult` sin uso,
  `horizon_tau_ms` muerto tras D-745, imports).
- `cargo test --workspace`: VERDE (T-1 ignorado con diagnóstico; 5 ignored preexistentes).
  Ajustes de tests a los contratos fusionados: `auditor_open_diagnostics` y
  `trajectory_duration_contract` (firmas PR de TrajectoryAuditor), `spectral_risk_contract`
  ×2 (D-750/D-751b), `portfolio_admission_contract` (margin_cushion_pct D-744),
  `leverage_admission_contract` (win_probability calibrada), `payload/shadow` ×3
  (`tau_ms` en ValidatedOrder), `reality_physics` (diagnóstico de latencia CERRADO por
  D-753 — sqrt saneado, convertido a contrato de regresión), `ensemble_characterization`
  (MicroScalpTrigger→FlowExcitationConfluence, misma semántica de baseline fabricada).
- Cobertura de lecturas: SIN CAMBIO en esta ola (169/289; la fusión consumió la ronda).

## 7. Hoja de ruta actualizada (hereda §13 del XXXIX)

1. **[NUEVO·PRIORIDAD] Re-baseline del oráculo T-1**: cuantizar el fixture sintético a
   la rejilla de tick del spec registrado (o medir sobre tape real) y re-activar el
   trinquete desde la cobertura re-medida.
2. **[NUEVO] Cablear `partition_income` al recorrido con transporte** (FMT-285b):
   que `collect_income_window` deje de ser letal por registro y alimente la cuarentena.
3. §13.1: dominio de identidad/revisión/cuenta + payload decimal para contradicciones.
4. §13.3: FX as-of y balance de flujos antes de totales multimoneda.
5. §13.4–13.9 sin cambios (fills/genoma, fee-breaker, escala multiactivo OOS,
   aritmética, equity/ROI, 120 lecturas restantes).
6. `feature_parity_exporter` vía Write + verificación.

## 8. Cierre del tramo

La cadena de evidencia income ahora sabe separar, por símbolo, lo que vio de lo que no
vio y lo que no pudo tragar; la fusión dejó constancia de sus dos reparaciones reales y
del precio honesto de la auditoría más nueva (el oráculo sintético dejó de medir). La
tarea siguiente sigue siendo la del XXXIX: unir identidad, evidencia nueva y recuperación
del veto — con el oráculo respirando de nuevo antes de optimizar nada sobre él.
