# Plan de Línea C — revisión desde la base, por fases y archivo por archivo (2026-10-11)

> **Dónde vive y base.** Documento de la Línea C (Claude, sesión cloud, prefijo CL-n). El plan maestro (docs/PLAN_MAESTRO_SINCRONIZACION.md) lo resume en la sección que le toque al fusionar; la numeración de ese plan se mueve porque otras líneas empujan a main directamente. Aquí «§1»…«§9» son apartados de este documento; «§30.x» y demás números de dos cifras remiten al plan maestro. Se escribió sobre `ciclo10` (5338eda1) y se re-ancló el 2026-10-11 a aa716edf (merge del PR #29, padres 0000eb8d y 5338eda1, con #713 y #714). Toda cita `ruta:línea` es de aa716edf salvo que nombre otra revisión; `lib.rs` sin ruta es `crates/god-engine-core/src/lib.rs` y `god_engine.rs` es `src/bin/god_engine.rs`. Las comprueba `python3 scripts/plan_linea_c.py citas docs/PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md --rev aa716edf --alias lib.rs=crates/god-engine-core/src/lib.rs --alias god_engine.rs=src/bin/god_engine.rs`. Después de re-anclar, main avanzó a 03b3f1ca (PR #32), 07688cf9 (#715) y 1025f750 (#716). Antes de editar un archivo se vuelve a hacer grep del símbolo.

**Base medida en aa716edf**
- 1 559 archivos versionados, 526 archivos Rust, 160 933 líneas Rust y 23 crates miembros. En 5338eda1 eran 1 557, 524 y 160 506.
- T-1: 16/144 genes sensibles (11,1 %) sobre 5338eda1, según el mensaje de aa716edf. El trinquete está en 11,0 %.
- T-1 sobre aa716edf: 16/144 (11,1 %), con la misma lista de genes inertes que 5338eda1 (medido por Línea C el 2026-10-11 en release, 3 078 s). Línea E informa el mismo 16/144 sobre 05f58e87, que es el árbol de 03b3f1ca. Que el T-1 no se mueva con #713 y #714 no prueba que su conducta sea correcta: el fixture no ejercita esas ramas (C-F3, pregunta 8).
- Alcance del T-1: llama a `run_backtest_native` con 1 000 USD (crates/backtest-engine/tests/t1_cobertura_genetica.rs:50-52) y no ejecuta el host. No ve el régimen de 13 USD, ni la envolvente, la IOC, el kill-switch o `close_was_real`.
- Edge fuera de muestra: sin medir.
- Ninguna fase de este plan autoriza operar, entrenar, promover genomas o modelos, ni cambiar el golden o el trinquete.

**Nombres que usa este documento**
- Fases: C-F0 … C-F11.
- Hallazgos: LC-Fn-mm.
- Commits: CL-n, desde CL-55. El último usado es CL-54 (f40de784).
- No confundir con:
  - F0–F8 del BARRIDO;
  - R0–R9, que tiene dos significados distintos (Qoder/AGY y Codex/Sol);
  - G0–G8 de Codex;
  - QS-R0…R6 de Línea E.

## 1. Cómo encaja con los otros planes

### 1.1 Fases ajenas que se reutilizan

| Fase C | Equivale a | Qué se toma como entrada (no se rehace) |
|---|---|---|
| C-F0 metas | R0 de Qoder/AGY (plan cuántico Parte VI :331-375); R0 de Codex (PLAN_REVISION_ARCHIVO_POR_ARCHIVO §4 :190-205); G0; QS-R0 (§30.2) | QS-R0 meta→requisitos; ADR-0016 |
| C-F1 conceptos | R1 de Ronda 7 (cerrada, 9 fichas); R1 de Codex (topología); G1–G2 | fichas R1; ADR-0011 (identidad) |
| C-F2 teoría | R2 (41 fichas, 9 HIGH) y R3 (17 fichas, 2 HIGH) de Ronda 7, cerradas; R2–R3 de Codex; plan cuántico Parte IV (:235-300) | las fichas; contrato G-TEO §30.1 (:1112-1136) para toda teoría nueva |
| C-F3 decisión viva | R4 de Ronda 7 (cerrada en ce5e0897, 7 fichas, 4 HIGH); QS-R1/R2; G4 | ledger LEDGER_DECISION_VIVA (81 factores, 13 verificados en su primer corte) como índice, sin editarlo y sin heredar sus estados. AGY marcó R-15, C-02 y C-W como resueltos (ledger :135, :42, :34), y Línea E dice que R-15 «no cierra la causa» (§7.3 del ledger en la rama del PR #32, 05f58e87:280). |
| C-F4 dinero | R5 (AGY Ω55 #689); R5 de Codex; G4/G8; QS-R3 = ADR-0016 D1; plan cuántico Parte V (:302-329) | ADR-0016 D1–D4 |
| C-F5 ejecución | R5; G6; hoja de ruta propia (HOJA_DE_RUTA_CIMIENTOS §6 :403-424) | ADR-0013, ADR-0015 |
| C-F6 datos | R7 de Ronda 7 (ABIERTA); R4 de Codex (datos y tiempo); G2 | fichas R7 cuando se publiquen |
| C-F7 aprendizaje | R6 de Ronda 7 (Qoder #690, anunciada en el buzón :8309-8314 y en curso en el worktree .r7r6; sin §R7-R6 publicada en BARRIDO). AGY ya declaró resueltos R7-R6-IG-1, EV-1 y C-7 en Ω60/#695 (plan sinc. §22 :952-958); esos estados no se heredan. R6 de Codex; G3/G5/G7; QS-R4/R5 | §R7-R6 cuando se publique; ADR-0008, ADR-0012 |
| C-F8 telemetría | R7 (telemetría) y R9 (plataforma) de AGY | — |
| C-F9 herramientas | R8 de Codex (UI, scripts, docs) | — |
| C-F10 suites | R8 de Ronda 7 (paridad + T-1 final, Qoder); QS-R6; QS-P (§30.7: 05f58e87:1271; en main desde 03b3f1ca) | `scripts/t1_oraculo.sh` (no está en aa716edf; entró en main con el PR #32, 03b3f1ca) |
| C-F11 limpieza | R9 de Ronda 7 y de Codex (recursos, Git) | — |

Línea C adopta el recibo por archivo de Codex/Sol (PLAN_REVISION_EXHAUSTIVA §5 :109-136). Un archivo no cuenta como revisado por aparecer en una lista. Cuenta cuando tiene recibo: OID del blob, preguntas respondidas, tests y veredicto.

### 1.2 Qué posee Línea C y qué revisa de otros

**Posee** (edita, siempre con test):
- `src/bin/god_engine.rs`: ingesta WS, kill-switch, IOC, reservas, sistema inmune y envolvente de envío.
- `crates/execution-engine/`.
- En god-engine-core: `entry_reservation.rs`, `darwin.rs`, `reset_engines` y el kill-switch del núcleo.
- En quantum-arena: `genome_store.rs`, `paths.rs` y `protection_health.rs`.
- El dimensionado en espacio de riesgo (ADR-0016 D1) y la sonda única de cartera (D4).
- Los contratos G3, G4 y G6 (plan sinc. §8 :348-351) y RA-RUIN-F01.

**Revisa sin editar.** Cada hallazgo se entrega al dueño como pedido con un test que falla:

| Dueño | Zona |
|---|---|
| Línea E | ramas 1–10 de `god-engine-core/src/lib.rs`; C-22; admisión D2/D3 de risk-engine; escritura de modelos de `train_forest`; `src/model_manifest.rs`; `unit-suites.yml` |
| Codex | `correlation_guard`, `random_matrix`, calibración, conformal, `replay-contracts.yml` |
| Qoder | espectro, consenso, Hawkes; R7-R6 |
| AGY | consejo, Prospect, Navier-Stokes, Yang-Mills |
| GLM | `veto_registry` (compartido con AGY) y tubería de promoción |
| Sol | `booktick_replay.rs` y `metrics.rs` |

**Aviso de solape.** AGY empujó #713 y #714 directo a main, sin PR y sin T-1 registrado en sus commits. Los dos tocaron zonas reservadas, y el PR #29 de Línea C también tocó zonas ajenas. Posiciones en aa716edf:
- #713 (8f6bcc8a) editó `lib.rs` en `puertas_del_continuo` (:1591-1606), en la zona de las ramas 11–15 (:6371-6467) y en el ML-GATE (:7771-7793), además de `calibration.rs`.
- #714 (0000eb8d) editó `src/bin/god_engine.rs`, que es zona C (el freno del host, :1726-1737). En `lib.rs` tocó :3953-3968, :4283-4301, :4818-4826 y :5893-5895. Este último es la rama 1 corta (K-27), dentro de las ramas 1–10 que reserva Línea E; K-27 es un ítem de QS-K. También editó `calibration.rs` (`p_win_directional`), zona de Codex (calibración y conformal). Los hunks :3953-3968 y :4818-4826 (conformal) y :4283-4301 (innovación online) son la proyección K-26 que la usa.
- El PR #29 tocó `lib.rs` en :29, :222-250, :1262-1264, :1905-1910, :2921-2926, :2952, :3724-3727, :7588-7605 y :8258-8299, ninguno en las ramas de entrada. También tocó `src/bin/train_forest.rs:1485-1493`, el archivo donde Línea E reserva la escritura de modelos. El cambio es sólo el comentario y el mensaje del aborto por kill-switch, no la escritura de modelos.

El PR #29 ya integró ese main en aa716edf. Su mensaje registra, sobre el merge de prueba con 0000eb8d, el check del workspace y god-engine-core 465/0, sin T-1. El PR #32 integró aa716edf en 05f58e87, y su mensaje registra `check --workspace --all-targets` en verde. Al re-anclar, el PR #32 seguía en borrador (head 05f58e87). Después se fusionó en main como 03b3f1ca (2026-10-11), con el mismo árbol que 05f58e87. Su descripción informa T-1 16/144 sobre esa unión; sin reproducir por Línea C.

### 1.3 Pedidos dirigidos a Línea C

| # | Origen | Pedido | Respuesta de Línea C |
|---|---|---|---|
| 1 | Línea E (buzón :233-235; ADR-0016 :114-115) | Freno inmune del host con d* = 0,632 | **Cambio de AGY en 0000eb8d (#714), sin test y con semántica aún distinta.** El host aplica `drawdown_maximo(...).min(d_falsacion)` (god_engine.rs:1729-1737), pero sin el lerp por `micro_w` que aplica el riesgo (risk-engine/src/lib.rs:336-341). #714 no añadió ningún test del host: su único test es `conformal_direction_symmetry_contract.rs`. Además, host y riesgo usan picos distintos. El host usa un pico local que nace con el capital del arranque (god_engine.rs:1682, :1704-1711), y el riesgo usa `RiskEngine.peak_capital` (risk-engine/src/lib.rs:284-286); ninguno se persiste. Línea C lo completa en C-F4: una función única de umbral y pico, una persistencia mínima del pico con un test de reinicio, y un test que exija la misma salida en host y riesgo (LC-F4-10). |
| 2 | Línea E (§30.2 :1145; §30.6 :1260-1261; buzón :236-238) | Implementar D1 y D4 | **Acuse.** Primero D4: una sola sonda abierta en toda la cartera, con la orden mínima y ≤ 5 por moneda. Hoy hay dos puertas, las dos por moneda: `trade_count < 5` en riesgo (risk-engine/src/lib.rs:1136) y la sonda de arranque frío del núcleo, `arranque_frio_sin_roster` (lib.rs:7750-7770, usada en :7792), que se arma con `trade_count == 0` y sin posición abierta en la moneda. `trade_count` no se persiste (lib.rs:7756-7757), así que tras cada reinicio cada moneda sin `_MOTOR` cargado (lib.rs:4581) puede volver a abrir una sonda. El universo de la lista de arranque es de 26 (bootloader.rs:320-347). En el repo sólo están versionados los `_MOTOR` de ATOM, BNB y NEAR (models/ está en .gitignore:49). Cuántas monedas no tienen modelo en el PC del operador está sin verificar: el plan sinc. :206 registró un roster de 18 MOTOR. D4 unifica las dos puertas. Test: una sola sonda en toda la cartera, también tras reiniciar. Después D1: una función de dimensionado compartida por el host y `live_envelope_gate`, acordada con Sol y en modo sombra. Se activa sólo con edge OOS medido y con el libro sombra cableado. |
| 3 | Línea E (buzón :222-225; §30.3 :1168-1169) | Acuse antes de cablear QS-R4b en el núcleo | **Acuse con cuatro condiciones.** (a) La salida de `process_event` es idéntica con y sin libro; un test lo comprueba. (b) No hay asignación por evento en el camino caliente: buffer fijo. (c) `riesgo_por_operacion` deja de escribirse antes de los vetos del consejo, del ML-GATE y del host, pero sin salir del núcleo. Su único escritor está en `evaluate_single_intent` (risk-engine/src/lib.rs:1434-1446), que llama `evaluate_quantum_order` (:435), y lo leen risk-engine/src/lib.rs:324 y :646, god_engine.rs:1734 y online_daemon.rs:1131. Escribirlo sólo en el host al confirmar el envío rompería la paridad. El T-1, `booktick_replay`, el examen del demonio y ShadowForest pasan por `process_event` sin host (que llama a `evaluate_quantum_order` en lib.rs:7499). Lo dejarían en su valor inicial 0 (quantum-arena/src/state.rs:611), y `drawdown_maximo` caería al gen (risk-engine/src/drawdown.rs:112-119). La escritura pasa a un punto común del núcleo y del replay, después del consejo y del ML-GATE, con reversión si veta el host y su espejo en `live_envelope_gate`. Test en `bt_vivo_parity_audit` (LC-F4-03). (d) En el ciclo 11 el gancho sólo registra en sombra. Alimentar con él al calibrador, al consejo o al EV va después del paso 2 de §5 (etiquetas). Falta el acuse de Qoder: no consta en el buzón de aa716edf. |
| 4 | Línea E, PR #32 (buzón de la rama del PR #32, 05f58e87:69-70; no está en aa716edf) | ¿Hay algo de Línea C en vuelo en las ramas 1–10? | **Sin objeto.** El PR #29 se fusionó como aa716edf y Línea C no tiene nada en vuelo en las ramas 1–10. Sus hunks en `lib.rs` (§1.2) no caen en las ramas de entrada. En cambio, #714 de AGY sí tocó la rama 1 corta (K-27, lib.rs:5893-5895), con el mismo código que QS-K según el merge 05f58e87 del PR #32. Al re-anclar, el PR #32 seguía en borrador (head 05f58e87) y ya había integrado aa716edf. Después se fusionó en main como 03b3f1ca (2026-10-11), con el mismo árbol que 05f58e87. Su descripción informa T-1 16/144 sobre esa unión; sin reproducir por Línea C. |
| 5 | Qoder R7-R3-D-1/F-1 (BARRIDO :3220-3221) | `_is_mean_reversion_vortex` | Se acepta en C-F5. Sigue en aa716edf, en god_engine.rs:4267, y el grep sólo encuentra esa definición. Se retira con un test de fuente (LC-F5-07). |
| 6 | Qoder R7-R4-C-1 (BARRIDO :3441-3445) | Residual del flag por ranura | Se acepta en C-F5. Hoy `close_was_real` recorre las ranuras con `.any(...)` y consume el primer flag confirmado que encuentra, sea o no el de la ranura que cerró (god_engine.rs:3669-3687). `closed_order` pasa a llevar el índice de ranura, y `close_was_real` consume sólo el flag de esa ranura. Test: dos ranuras confirmadas y cierre de la segunda (LC-F5-02). |
| 7 | Qoder R7-R4-A-3 (BARRIDO :3451-3453) | `QuantumOrderRouter` muerto | **Propuesta: retirarlo**, porque sólo lo usan los tests de su propio módulo (router.rs:184-303); fuera de él sólo queda el re-export de execution-engine/src/lib.rs:32. En commits separados, y tras confirmarlo con grep, retirar también `simulator`, `shadow`, `quantum_multiplexer`, `quantum_socket`, `hot_swap` y `dynamic_symbols`. La τ de la decisión viaja en `ValidatedOrder::tau_ms` (risk-engine/src/lib.rs:48), no en `expected_lifetime_ms`. Necesita el visto bueno del dueño (zona CL-14/B3.29). |
| 8 | Plan sinc. §1 (:153-157) | GENOME-GATE al cargar | Se acepta en C-F7. `load_active` y `desde_json` (genome_store.rs:140-182) llamarán a `validate` (:248) sobre el genoma tal como se leyó, antes de que `from_vector` (:180) lo recorte a las mismas cotas (genome.rs:2764-2770). Si está fuera de rango, no se aplica. Frente a F4-H3 de AGY («normalizar», :149 y :174-177), la propuesta es normalizar sólo dentro de la tolerancia de redondeo y rechazar fuera de ella (LC-F7-16). Si una fase anterior (C-F4) retira o conecta genes, GENOME-GATE entra antes que ella, porque eso cambia el vector de 144 genes que comprueba `validate` y los `active.json` de demo y prod. Si no entra antes, retirar genes espera a C-F7. |
| 9 | Codex: §14.6 (:833-842), §14 (:744-748), §13.5 (:559-560) y §8 (:358-359) | RA-RUIN-F01/F1-B4; F1 A3/A4 y B1/B2/B4; revisión cruzada; R01 | Se acepta. RA-RUIN-F01 va a C-F4: error explícito con NaN, ±Inf o sin exposición. F1 y la revisión cruzada de identidad, admisión, reserva, envío y trayectoria económica entran como preguntas de C-F1, C-F4 y C-F5. R01 (readmitir el margen final antes del envío) va a C-F5. |
| 10 | Línea E §30.6 (:1266-1267), dirigido a Codex/Sol | K-18/R-17: la confianza entra 4 veces en el tamaño | Es zona C y se acepta en C-F4. Primero, un test que cuente cada multiplicación por la confianza entre la intención y la cantidad. El arreglo de fondo es D1: la confianza entra una sola vez, como p calibrada. |

### 1.4 Decisiones del dueño que condicionan el plan (con la propuesta de Línea C)

- **Fuente única de invariantes.** ADR-0016 choca con el plan cuántico Parte II (:155-181): SL de 55 pb, 2 posiciones, suelo de 3 USD y 5×. ADR-0016 ya dice que el suelo de 3 USD no existe en el código (:111-112). **Propuesta:** rige ADR-0016 y el plan cuántico se corrige por adenda.
- **Integración de AGY sin PR ni T-1** (Ω69–Ω77, de #706 a #715). Ningún commit de esas olas (#706 a #715) tiene un PR asociado en GitHub, y ninguno registra un T-1 en su mensaje. En los push de #710 a #714 la CI sí corrió. «Unit and contract suites» cerró con éxito, pero ningún run de «Replay contracts» terminó en verde:
  - «Unit and contract suites» dio success en 759f44ec, 0043f889 (merge del PR #31), ab0967e8, 95412dd7, 8f6bcc8a y 0000eb8d.
  - «Replay contracts» falló en 759f44ec (#710) por el marcador `<<<<<<< HEAD` que ese commit dejó en el plan maestro. En 0043f889, ab0967e8, 95412dd7, 8f6bcc8a y 0000eb8d quedó cancelled, porque cada push a main cancela el run anterior (`cancel-in-progress: true` con grupo por `github.ref`, replay-contracts.yml:13-15). El `check --workspace --all-targets` (replay-contracts.yml:51-52) pasó en ab0967e8, 95412dd7 y 8f6bcc8a. No llegó a correr en 759f44ec y quedó cancelado en 0043f889 y 0000eb8d. Las suites de contratos que van detrás del check no terminaron en ninguno: se saltaron o se cancelaron.
  - El paso de marcadores sólo mira `git diff --check HEAD^ HEAD` (replay-contracts.yml:40-44), así que no podía parar los push siguientes.
  - El marcador estuvo en main desde 759f44ec hasta 0000eb8d. En aa716edf ya no queda ninguno.

  **Propuesta:** proteger main con un ruleset y checks requeridos (`replay-contracts`, `unit-suites`), o que todo entre por PR con esos checks. Y exigir T-1 adjunto en los PR que cambian conducta viva (plan sinc. §2 :170). En main 07688cf9, #715 (AGY, sin PR) ya hizo que el push a main no cancele Replay contracts (replay-contracts.yml:18 en 07688cf9, como ya hacía `unit-suites.yml:26-31`) y añadió un hook pre-commit contra marcadores (`.githooks/pre-commit`). Lo hizo directo en main y en `replay-contracts.yml`, que es zona de Codex. Queda el ruleset con checks requeridos, o PR obligatorio, y el T-1 adjunto.
- **R-15 (veto de primer toque).** AGY lo marcó como resuelto en el ledger (:135 y :204, Ola Ω73), pero la revisión de Línea E dice que «mejora, pero no cierra la causa» (05f58e87:280). **Propuesta:** pasarlo a modo sombra hasta tener μ y σ con habilidad medida, como ya pedía §30.6 (:1253) (LC-F4-04).
- **Kill-switch** (ADR-0015 :86-87). ADR-0015 nombra tres escritores de `kill_switch_active` (:13-15): el núcleo (lib.rs:7122), el sistema inmune (god_engine.rs:1802) y X-009 (god_engine.rs:4757). Pero el demonio también arma ese latch global, por pérdidas constantes (online_daemon.rs:1222, :1235 y :1324) o por una EWMA del estadístico t por debajo de −1,5 (:1343). Los de :1324 y :1343 no dependen del armado de la evolución (online_daemon.rs:737-738) y valen también en prod. Los de :1222 y :1235 sólo corren tras una promoción vigilada (:1198-1201), y en prod la evolución está desarmada por defecto (:758-770). Son ocho puntos de escritura: los siete anteriores y el drenaje de apagado del host (god_engine.rs:4947-4950), que ADR-0015 :13-15 tampoco enumera.
  - **Propuesta:** una pregunta y un test de fuente: todo escritor de `kill_switch_active` figura en ADR-0015 con su criterio.
  - Mientras un estadístico descriptivo pueda armar el latch, Línea C no recomienda el latch permanente con rearme manual. La histéresis automática sólo llega cuando el disparo sea un e-proceso con ARL0 medido.
- **Meta.** Rige la reformulación del dueño: maximizar el crecimiento geométrico neto bajo límites de riesgo, drawdown, costes y capacidad, subordinado a la supervivencia del capital. §9 la escribe como requisito medible con su falsación; «duplicar cada 3 días» queda como cota de referencia que se reporta como brecha, no como puerta. Hoy hay dos doctrinas aceptadas a la vez (LC-F0-01): fijarla en un ADR nuevo que reemplace la meta de ADR-0016 es decisión del dueño.
- **La rama 11 vive a la escala que habló (CL-85, F3.2-05).** Decisión del dueño, tomada en una tarjeta el 2026-10-12 a las 05:12 UTC: «escala que habló».
  - Con el veredicto espectral, la orden de la rama 11 vive a τ_k, la escala que decidió su dirección (contrato #624).
  - La convergencia al centroide (#565) y la modulación de Hurst no mueven esa τ.
  - La τ final se fija antes de las puertas (CL-74).
  - Sin veredicto espectral sigue la micro τ.
- **Sesión viva con la familia honesta.** **Propuesta:** no autorizar, porque el edge OOS está sin medir.
- **FDUSD y los ~51 GB de .antigravity/target** (52 576 MB según BARRIDO :1821). No son zona C. Si se pide opinión: archivar (opción B) antes que borrar, que es también la recomendación registrada para FDUSD (plan sinc. :211).
- **graphify-out.** La propuesta está en C-F11.
## 2. Fases C-F0 … C-F11

**Regla de asignación**
- Cada archivo tiene una sola fase dueña, que es donde recibe su recibo.
- Otras fases pueden leer bloques concretos; esos bloques se citan por línea.
- Las líneas citadas son de aa716edf, que es origin/main tras fusionar el PR #29 y ya incluye #713 y #714. El borrador se escribió sobre `ciclo10` (5338eda1) y sus citas se re-anclaron a aa716edf; una cita que no se pudo comprobar lleva «(sin re-anclar)».
- El comprobador de citas va en C-F0. Además, antes de cada edición se vuelve a hacer grep del símbolo, porque otra sesión puede haberlo movido.
- Un archivo versionado que no aparezca abajo va a la fase de su directorio más cercano, y se anota en el recibo.
- La lista archivo por archivo está en `docs/audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv`: una fila por cada uno de los 1 561 archivos de 1025f750 (main al publicar), con su fase, la regla que la asigna y el estado del recibo (todos «pendiente» al publicar). Son los 1 559 de aa716edf más `.githooks/pre-commit` (#715) y `scripts/t1_oraculo.sh` (PR #32), los dos en C-F10. La genera `python3 scripts/plan_linea_c.py asignacion --rev <base> --salida <tsv>`, que falla si un archivo queda sin fase, si dos fases empatan o si una ruta exacta del plan no existe. Se regenera al empezar cada fase. Al publicar, ningún archivo cae en la regla del directorio más cercano: todos están en alguna lista.

| Fase | Archivos | De ellos, Rust |
|---|---|---|
| C-F0 metas | 30 | 0 |
| C-F1 conceptos | 49 | 24 |
| C-F2 teoría | 42 | 42 |
| C-F3 decisión viva | 39 | 39 |
| C-F4 dinero | 21 | 20 |
| C-F5 ejecución | 27 | 27 |
| C-F6 datos | 47 | 44 |
| C-F7 aprendizaje | 73 | 49 |
| C-F8 telemetría | 71 | 53 |
| C-F9 herramientas | 80 | 42 |
| C-F10 suites | 191 | 186 |
| C-F11 limpieza | 891 | 0 |
| **Total** | **1 561** | **526** |

**Puerta común** (para toda fase que toque código)
- **Compilación:** `cargo check --target x86_64-pc-windows-gnu --workspace --all-targets` sin errores ni advertencias nuevas.
- **Conducta:** un test que falla en la base antes de cada arreglo y pasa después. Además, las suites de los crates tocados, bajo wine.
- **Resultados:**
  - el golden del backtest no cambia, o se re-certifica con autorización del dueño;
  - T-1 release ≥ 11,0 %, con comparación gen a gen contra la lista de 16/144 medida en 5338eda1 (todo gen ganado o perdido se explica). Sobre aa716edf también da 16/144 con la misma lista;
  - `bt_vivo_parity_audit` en verde, y paridad 48D del entrenador si se toca una feature.
- **Lo que el T-1 no ve.** El T-1 llama a `run_backtest_native(…, "BTCUSDT", 1_000.0)` (backtest-engine/tests/t1_cobertura_genetica.rs:50-52).
  - Con 1 000 USD no ejercita el régimen micro de 13 USD: ni el tope de 1,20-2,60 USD (risk-engine/src/lib.rs:1323), ni el techo micro de apalancamiento (:1269-1270), ni el peso micro de `capital_regime`.
  - Tampoco ejecuta el código del host (god_engine.rs): ni la IOC (god_engine.rs:4434-4463), ni `close_was_real` (:3677-3971), ni los escritores del kill-switch del host (:1802, :4757). La envolvente sólo entra por su espejo `live_envelope_gate` (backtest-engine/src/lib.rs:370-382), y del kill-switch sólo corre el latch del núcleo (god-engine-core/src/lib.rs:2926, :7122).
  - Para cambios en esas zonas (C-F5 y los literales micro de C-F4), la puerta es un fixture de 13 USD con tests del host, o un test nuevo de 13 USD en `bt_vivo_parity_audit` si el cambio no toca god_engine.rs. Las pruebas actuales de esa suite arrancan con 1 000 USD (backtest-engine/tests/bt_vivo_parity_audit.rs:39, :47, :224, :282, :327, :417, :494, :508) y no ejecutan god_engine.rs, así que tampoco cubren esos cambios. El recibo dice que el T-1 no cubre esos cambios.
- **Falsación:** la prueba propia de cada fase. Donde no hay medición se escribe «sin medir».

---

### C-F0 — Metas y su falsación medible

**Objetivo.** Una sola fuente de verdad para la meta y los invariantes, cada uno con la observación que lo refuta, antes de tocar código.

**Archivos:**
- `.agents/` (AGENTS.md, MEMORIA.md, rules/*, workflows/*) y `.agent/` (rules/trade.md, workflows/*);
- `AGENTS.md` y `CLAUDE.md` de la raíz;
- `TABLERO.md`;
- `COORDINACION_CODEX_2026-09-28.md` (buzón);
- `docs/adr/` (ADR-0001…0016 y README).

**Preguntas del consejo**
1. ¿La meta es «duplicar cada 3 días» (RULE growth_over_wr) o el requisito de ADR-0016 (+0,3 R por operación, ~16 operaciones/día a ½ Kelly)?
   - MEMORIA calcula que la meta exige una tasa log de 0,231/día, es decir un Sharpe diario de 0,68 con Kelly pleno (.agents/MEMORIA.md:4092-4093).
   - La familia honesta de GLM tiene una mediana OOS de +0,008 (TABLERO.md:23-24).
   - Hay que fijar qué tasa log por día se exige y con qué tope de ruina.
2. ¿Qué invariantes rigen? ADR-0016 (sin tope de 55 pb, d* = 0,632, piso de 4,79 USD) o la Parte II del plan cuántico (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md), con los invariantes en II.2 (:155-181).
3. ¿Qué fricción es la real?
   - El código opera como taker: `live_taker_fee` 0.0005 (quantum-arena/src/config.rs:208) y `force_maker = false` (god_engine.rs:4257-4269). Eso da ≥ 10 pb de ida y vuelta más deslizamiento.
   - El plan cuántico supone 4 pb maker de ida y vuelta (II.2, docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:178) y p = 70 % (II.3, :183-196).
   - Rige la del código hasta que haya fills maker medidos.
4. ¿Qué observación refuta la meta y en qué plazo? Candidatas: caída d*, e-proceso sobre R ejecutados, EV neto OOS en las dos mitades del forense. La última necesita tapes reales, así que hoy está bloqueada por datos externos.
5. ¿Qué no está medido? Edge OOS, latencia del camino caliente y cobertura de la calibración en tape real. Los tapes viven en el PC del operador y desde la nube Binance responde 403.

**Puertas.** No hay código del motor. Cada requisito debe tener: estimador con su path:line, umbral con su origen, prueba que lo refuta y estado (medido, sin medir o bloqueado por datos externos). C-F0 puede cerrarse con requisitos bloqueados; no puede cerrarse con un requisito sin estado. Lo revisan dos escépticos. Toda contradicción entre ADR y código abre un hallazgo en la fase dueña del código.

**Comprobador de citas.** Se adelanta aquí desde C-F11, porque el borrador ya traía citas desplazadas. Es `scripts/plan_linea_c.py citas`, que comprueba que cada cita path:line de este documento existe en aa716edf. Al publicar: 640 citas correctas, 8 ambiguas (nombres como `orchestrator.rs` sin directorio), ninguna inexistente y ninguna fuera de rango. Por sí solo no detecta un símbolo que se movió dentro del mismo archivo; para eso sigue el grep antes de editar.

**Entregable**
- Subsección «§9 Requisitos de la meta»: hecha (40 requisitos y los hallazgos LC-F0-01…35).
- Un ADR nuevo de la meta que reemplace el Contexto de ADR-0016, cuando el dueño lo fije (LC-F0-01).
- MEMORIA y TABLERO de Línea C al día. En aa716edf faltaban los bloques de los ciclos 9 y 10, y la fila de Claude de Línea C (TABLERO.md:26) seguía con fecha 2026-10-03. Hecho al publicar este plan.
- Hecho al publicar este plan: el comprobador de citas y la asignación archivo por archivo (`scripts/plan_linea_c.py`, 14 tests en `scripts/tests/test_plan_linea_c.py`).

---

### C-F1 — Conceptos y contratos

**Objetivo.** Fijar unidades, identidad, «desconocido», horizonte único, simetría espejo, el principio de «continuo, no regímenes» y la versión (G0).

**Archivos:**
- `crates/quantum-arena/src/`: `state.rs`, `config.rs`, `horizon_policy.rs`, `symbol_registry.rs`, `symbols.rs`, `feed_health.rs`, `paths.rs`, `atomic_float.rs`, `lib.rs`;
- `crates/omniscient-registry/src/lib.rs`;
- `crates/strategy-core/src/{lib.rs, types.rs}`;
- `crates/execution-engine/src/{binance_api.rs, order_types.rs, lib.rs}`;
- `crates/{storage-engine, data-pipeline, signal-engine, feature-engine}/src/lib.rs`;
- `src/{config.rs, env_manager.rs, lib.rs, risk/mod.rs}`;
- `crates/risk-engine/src/veto_registry.rs`;
- `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.cargo/config.toml`, `template.env`;
- `crates/*/Cargo.toml`: 20 de los 23 de aa716edf. Los de `graph-architecture`, `graph-4d` y `phase-runner` van con su crate a C-F9, para que cada archivo tenga una sola fase dueña.

**Preguntas**
1. **Unidades.** ¿Cada magnitud declara si es fracción, pb, USD, unidades de precio o ms? Ejemplos: `atr_1m` ×7.746 y los suelos de ATR en unidades de precio (god-engine-core/src/lib.rs:4422-4427).
2. **Identidad** (G1, ADR-0011). ¿Algún código supone que `coin_id 0` es BTC? Casos: `banda_paga_friccion` (lib.rs:1525-1529) y líderes por prefijo (lib.rs:3017-3038).
3. **Desconocido no es cero.** El registro:
   - convierte NaN y ±Inf en 0 (omniscient-registry/src/lib.rs:73-76);
   - cae al valor global, que es el de la última moneda que escribió (:244-251);
   - no actualiza el timestamp (:48, :65).
4. **Un horizonte por posición.** Sólo existe `Continuous` (strategy-core/src/types.rs:15-18; quantum-arena/src/position.rs:18-20). Residuos discretos que inventariar:
   - nombres scalp/swing en `state.rs` y `config.rs`;
   - alias en `signal-engine/src/orchestrator.rs:308` y `:317`;
   - centros táctico/swing/secular de 60 s / 4 h / 30 d (temporal_spectrum.rs:1149-1163);
   - Micro/Meso/Macro (stateful_engine.rs:118);
   - bandas fast/slow;
   - régimen micro/estándar;
   - modos de Kelly;
   - `MarketRegime` discreto;
   - genotipos YAML SCALPING/SWING.
5. **Espejo** (G-TEO punto 3). ¿Cada contrato del largo tiene su corto? Con barreras asimétricas, 1−p no es la probabilidad del corto.
6. **G0.** `rust-toolchain.toml:2` dice `channel = "nightly"` sin fecha. La CI ya fija `nightly-2026-06-30` (replay-contracts.yml:45-52; unit-suites.yml:68-74), así que sólo falta este archivo.
7. **`veto_registry`.** Hoy `buscar()` (veto_registry.rs:382) sólo se llama desde sus tests (:636). ¿Debe ser el índice de todos los `REJ_*` vivos (ADR-0004)?

**Puertas**
- Test que falla hoy y debe pasar:
  - el registro conserva NaN como desconocido;
  - leer la moneda A no devuelve el valor de B;
  - con el universo permutado (BTC fuera del índice 0) no cambia el gen que usa `banda_paga_friccion`;
  - test de fuente: todo `REJ_*` está en `veto_registry`.
- `rust-toolchain.toml` = `nightly-2026-06-30`, el mismo que la CI. Un test de fuente compara las tres cadenas: `rust-toolchain.toml`, `replay-contracts.yml` y `unit-suites.yml`.

**Entregable**
- «Contrato de conceptos», que citan todas las fases siguientes.
- CL-n que fija `rust-toolchain.toml` a `nightly-2026-06-30` (pendiente del ciclo 8).
- Propuesta sobre el registro, anunciada en el buzón porque es transversal.

**Estado (ciclo 11, base ae6d51a2).** Hecha la revisión: siete revisores, uno por concepto, y dos escépticos por dimensión.
- Contrato: `docs/CONTRATO_DE_CONCEPTOS_LINEA_C_2026-10-11.md`, con siete apartados: unidades, identidad, desconocido, horizonte continuo, espejo, versión y construcción, y vetos.
- Hallazgos: `docs/audit/HALLAZGOS_C-F1_2026-10-11.md`, LC-F1-01…73. Hay 42 confirmados y 31 disputados (4 altos, 36 medios y 33 bajos), y otros 6 refutados. Cada uno trae el test que debe fallar hoy, que sale de leer el código: no se ejecutó ninguno.
- Recibos: los 49 archivos de C-F1 tienen recibo con el OID de su blob en `docs/audit/PLAN_LINEA_C_RECIBOS_2026-10-11.tsv`, y hay 21 lecturas de apoyo anotadas en la fase de cada archivo.
- Los cuatro altos:
  - LC-F1-01: el papel sin banderas guarda la envolvente Kelly en `data/prod`. Es un defecto de CL-48.
  - LC-F1-02: specs inventados servidos como oficiales.
  - LC-F1-03: el slot 0 recibe el gen del stop de BTC.
  - LC-F1-04: el perceptrón vota con el peso hebbiano de otra moneda.
- En `main` c4e8e285, #720 cierra LC-F1-34 y, en parte, LC-F1-35 (adenda de los hallazgos).
- Arreglos: los hallazgos cuya fase dueña es C-F1 empiezan en el ciclo 12, después del medidor, cada uno con su test en rojo y en verde. Los demás esperan a su fase dueña.

---

### C-F2 — Teoría por dominio

**Objetivo**
- Comprobar que cada módulo teórico implementa la teoría que nombra, con sus supuestos y unidades.
- Probarlo contra datos sintéticos de solución conocida.
- Fijar su siguiente paso, en el sentido en que gradient boosting es el siguiente paso de random forest.
- El mercado se trata como un universo continuo, multivariante y multiactivo en tiempo y escala.

**Entradas.** Las fichas R2 y R3 de Ronda 7 (no se rehacen) y la Parte IV del plan cuántico. Toda teoría nueva pasa por G-TEO §30.1 (docs/PLAN_MAESTRO_SINCRONIZACION.md:1112-1136) y entra en modo sombra.

**Archivos por dominio**

| Dominio | Archivos |
|---|---|
| **F2.1 Matemática del riesgo y de la cartera** | `crates/risk-engine/src/{cramer_lundberg.rs, random_matrix.rs, hodge.rs}` |
| **F2.2 Estadística, calibración y decisión secuencial** | `crates/risk-engine/src/{evidence.rs, selection_stats.rs, ville_e_process.rs}`; `crates/quantum-arena/src/{evalues.rs, adaptive_quantiles.rs}`; `crates/god-engine-core/src/{calibration.rs, conformal.rs, diffusion.rs}`; `crates/feature-engine/src/{welford.rs, ewma.rs, correlation.rs, normalizer.rs}` |
| **F2.3 Física y espectro continuo** | `crates/quantum-arena/src/{temporal_spectrum.rs, spectral_tape.rs, espectral_multiactivo.rs, spectral_regime.rs}`; `crates/god-engine-core/src/math_kernels.rs`; `crates/feature-engine/src/{hurst_dfa.rs, multifractal.rs, spectral.rs, navier_stokes.rs, hodge_flow.rs, kalman.rs, tensor_ring.rs, shannon_entropy.rs, path_signatures.rs}`; `crates/strategy-core/src/yang_mills_gauge.rs` |
| **F2.4 Procesos puntuales, dependencia y contagio** | `crates/signal-engine/src/hawkes_bessel.rs`; `crates/feature-engine/src/{hawkes.rs, hawkes_cross.rs, lead_lag.rs, transfer_entropy.rs, copulas.rs}` |
| **F2.5 Conducta humana y de masas** | `crates/metacortex-engine/src/prospect_theory.rs`; `crates/feature-engine/src/microstructure.rs` |
| **F2.6 «Cuántico» y analogías sin consumidor** | `crates/signal-engine/src/{feynman_propagator.rs, navier_stokes_manifold.rs}`; `crates/feature-engine/src/{quantum_tensor_store.rs, simd_neural_network.rs}` |

**Preguntas por dominio**

- **F2.1**
  - ¿El R de Lundberg se convierte con la exposición real N/C?
  - ¿Marchenko-Pastur tiene consumidor? Hoy sólo lo usan tests, vía `rho_promedio`.
  - ¿El curl de Hodge predice co-movimiento antes de inflar ρ?
- **F2.2**
  - ¿El umbral Bonferroni M/α = 8320 deja los pesos en el piso?
  - ¿Platt aprende sólo de lo ejecutado?
  - ¿La cobertura conformal está medida?
  - ¿Cada e-proceso apuesta en la dirección de la hipótesis que se quiere detectar?
- **F2.3**
  - ¿Todos los lectores usan un mismo vector de pesos? Hoy `fused_score` usa `pesos_observables` (temporal_spectrum.rs:608, en `refresh_fusion`), mientras que el campo, τ* y W1 usan `fusion_weight` (:247-257) a través de `pesos_espectrales` (:1389-1396).
  - ¿Una escala anti-persistente invierte su señal?
  - ¿Hurst y la FFT están en tiempo físico o de eventos?
  - ¿Navier-Stokes y Yang-Mills dan algo distinto de cero con retornos asíncronos?
- **F2.4**
  - ¿μ̂ de Hawkes es la tasa que dice ser? Los pesos 1−e^{−dt/60} ≈ dt/60 cancelan el 1/dt (hawkes_bessel.rs:195-202), así que μ̂ es una tasa de conteo y no E[1/dt]. Por lectura tiene dos sesgos, y los dos bajan μ̂:
    - el tope `MU_INST_MAX` = 200 (:116, :195) recorta los dt < 5 ms;
    - con dt = 0 no se actualiza μ̂ (`ts > self.last_ts`, :192), aunque el evento sí entra en λ (:207).

    En ráfagas, los dos suben λ/μ̂. La magnitud está sin medir.
  - ¿El ancla 1,6 es la correcta? El código documenta λ_ss = μ + α·r/β con μ̂ → r, la tasa total observada, de modo que λ/μ̂ → 1 + α/β = 1,6 (:87-91, :169-176). El test `qo_535_mu_empirico_es_escala_libre` lo afirma (:421-466). El 2,5 = 1/(1−α/β) sólo valdría si μ fuera la base exógena, y aquí no lo es. Lo que queda por medir es cuánto mueven esa ancla los dos sesgos de μ̂.
  - ¿El contagio devuelve None con timestamps repetidos?
  - ¿Los líderes BTC/ETH son fijos?
- **F2.5**
  - ¿Qué dato de la masa entra y cuántas veces? `ls_account_ratio` entra dos veces en EnteMercado: como `crowd_ls_ratio` (god-engine-core/src/lib.rs:7607-7610; consejo_seniors.rs:581-593) y, sin presión de Prospect publicada, otra vez en la presión de respaldo (god-engine-core/src/lib.rs:7658-7664). Además, `entity_factor.clamp(0.20, 1.0)` (consejo_seniors.rs:615) anula el rango [1,0; 1,30] de prospect_theory.rs:180-184.
  - De funding, OI, liquidaciones, LS ratio, ballenas y spoofing, ¿cuál tiene IC medido? Sin medir.
- **F2.6**
  - ¿Hay mecánica cuántica o sólo nombres? El «espacio de Hilbert» son 32 escalas reales. Feynman y NavierStokesManifold sólo tienen `pub use` (signal-engine/src/lib.rs:30, 32) y tests.

**Puertas.** Pruebas sintéticas de solución conocida:
- caminata aleatoria: persistencia ≈ 0 e IC ≈ 0;
- OU: IC > 0 y vida media recuperada;
- fBm con H = 0,7: error < 0,05;
- Poisson irregular, con varios trades en el mismo ms, muestreado en fase uniforme como qo_535 (hawkes_bessel.rs:421-466): λ/μ̂ ≈ `STEADY_STATE_RATIO` (1,6) y `excitacion_hawkes_norm` ≈ 0;
- Hawkes simulado con n = 0,6, evaluado en cada evento justo después de registrarlo, como en vivo (`record_event` en god-engine-core/src/lib.rs:2639; lectura en :4235): excitación media mayor que la de un Poisson de la misma tasa evaluado igual (magnitud sin medir). Con muestreo de fase uniforme y μ̂ igual a la tasa de conteo, la media de λ/μ̂ es 1,6 con cualquier proceso estacionario, así que ahí no se exige excitación > 0;
- Hawkes simulado: α y β recuperados dentro de su IC por el estimador del siguiente paso (§3 #11);
- desfase plantado: recuperado;
- ciclo con muestreo irregular: detectado a la frecuencia correcta.

Además:
- nulo por permutación o surrogados para todo umbral de significancia;
- cobertura empírica para todo intervalo.

Esta fase no cambia conducta viva, así que el T-1 no debe moverse.

**Entregable**
- Una ficha por fila de la tabla §3.
- Lista de teorías a retirar. Cada retirada cita la prueba de falsación que falló, o «sin consumidor» con el grep que lo demuestra, y retira también sus tests. Las retiradas que ejecuta C-F4 (`capital_compounder` y `epigenetic_*`) se llevan además `risk-engine/tests/compounder_admission_contract.rs` y `risk-engine/tests/epigenetic_allocation_contract.rs`.
- Pedidos con test que falla a sus dueños: Qoder (espectro, Hawkes), Codex (calibración, correlación) y AGY (Feynman, Navier-Stokes, Yang-Mills, Prospect).

---
### C-F3 — L3, decisión viva

**Objetivo.** Seguir una intención desde el evento hasta la orden validada y responder cuatro cosas:
- cuántas fuentes de información independientes hay;
- si la τ es una sola de punta a punta;
- si cada confianza es una probabilidad calibrada que entra una sola vez;
- si cada veto tiene espejo y bloquea EV negativo medido.

**Archivos**

`crates/god-engine-core/src/lib.rs`, por bloques (líneas de aa716edf):

| Bloque | Líneas | Detalle |
|---|---|---|
| `process_event` | 1657-2733 | ρ(τ) :1749; consenso y habilidad :2431-2504; Hawkes :2638; contagio :2716; maker descartado :2719 |
| `puertas_del_continuo` | 1544-1655 | K-06 de #713 en :1591-1606 |
| `process_tick_dual` | 2810-8128 | ver filas siguientes |
| ↳ espectro | 2834-2845 | segunda publicación de ρ (:2840) |
| ↳ gestión y cierre | 3182-4335 | `break` :4332-4333 |
| ↳ realimentación | 3815-4310 | Lundberg :3863-3874; ρ :3883; consejo :4160; PPO :4268; K-26 de #714 en :3953-3968 y :4283-4301 |
| ↳ otros | — | ATR :4422-4427; ML :4571-4581; K-26 conformal de #714 :4818-4826; Stouffer :5048-5061; compuesto :5288-5374 |
| ↳ ramas 1–10 | ~5866-6129 | reserva de Línea E; K-27 de #714 en :5893-5895 |
| ↳ ramas 11–15 | 6131-6594 | lift ML de #713 en :6371-6373, :6445 y :6467 |
| ↳ freno del bosque | 6759-6811 | — |
| ↳ acondicionamiento espectral | 6901-7104 | — |
| ↳ riesgo, consejo y ML-GATE | 7499-7805 | sonda de arranque frío :7764-7770; K-23 de #713 en :7771-7793 |
| ↳ colchón | 7815-7832 | — |
| ↳ apertura | 7958 | — |
| ↳ maker | 8086-8124 | — |
| ↳ duración por Hurst | 8169, 8491-8492 | — |

Resto de archivos:
- `crates/god-engine-core/src/{ml_inference.rs, entes_consejo.rs, evidence_publication.rs, contagion_publisher.rs}`;
- `crates/metacortex-engine/src/consejo_seniors.rs`;
- `crates/signal-engine/src/{orchestrator.rs, voto_espectral.rs, skill_motores.rs, flow_impulse.rs, flow_excitation_confluence.rs, quantum_oscillator.rs, soliton_wave.rs, supersonic_shockwave.rs, stochastic_resonance.rs, game_theoretic_nash.rs, conformal_reversion_filter.rs, trend_runner.rs, renyi_tsallis_entropy.rs, coaxial_breakout.rs, perceptron_gate.rs, proyeccion_espectral.rs, contagion_modulator.rs}`;
- `crates/strategy-core/src/{stat_arb.rs, vecm_arbitrage.rs, maker.rs, momentum_booster.rs, strategy_telemetry.rs, multivariate_coint.rs}`;
- `crates/dark-alpha-engine/src/lib.rs`;
- `crates/risk-engine/src/regime.rs`;
- `src/trailing.rs`, `src/features/` (mod, correlation, ewma, microstructure, omni_strategies, welford) y `src/multi_asset_orchestrator.rs`;
- `crates/god-engine-core/tests/puertas_del_continuo_symmetry_contract.rs` (#713) y `crates/god-engine-core/tests/conformal_direction_symmetry_contract.rs` (#714). C-F3 los revisa junto con el código que certifican, porque certifican cambios de conducta viva (K-06, K-23, K-26 y K-27) que entraron en main sin T-1. Su recibo sigue en C-F10 por la regla genérica `crates/*/tests/`.

**Preguntas**
1. **Independencia.** Los 13 votos espectrales son funciones de un mismo `momentum_z`, y el dominante es el argmax|v| sobre 32 escalas, sin corrección.
2. **Una sola τ.**
   - ρ se publica a `tau_operativa_unificada` (:1749) y luego a `dominant_tau_ms` (:2840, :3883).
   - La vida de la decisión tensorial es `tau_dom`, pero la rama 11 usa `fast_duration_ms`, que sale de `micro_tau` (:5388-5394, :6237), y las ramas 13 y 14 usan `swing_duration_ms` (definida en :6381; rama 13 en :6455 y :6477; rama 14 en :6570 y :6588).
   - Sin τ medida se usa 1 138 000 ms (:7586).
3. **Confianza.**
   - Suelos: 0.50 (:6232), 0.55-0.85 (:6131) y 0.70 + 0.30·coherencia (orchestrator.rs:500).
   - Refuerzos: ML ×1,5 (:1613-1614) y hebbiano ×2 (:4549-4553).
   - Ya está registrado como R-17.
4. **Vetos.**
   - Consejo con supermayoría y un tracker que sólo ve operaciones ejecutadas.
   - Stouffer.
   - Muro calculado con el nivel 0.
   - 1−p para los cortos.
   - VPIN «causal» con umbral 0.75 (:7577).
   - Cada veto debe tener espejo y una medida de EV bloqueado.
5. **Qué modelo opina dónde.**
   - DarkAlpha sólo opina en BTCUSDT (:4722, :4739), y BTC no tiene modelo `_MOTOR`: en `models/` sólo hay `_MOTOR` para ATOM, BNB y NEAR.
   - Eso no basta para decir que DarkAlpha no decide nada. Con la sonda de arranque frío (:7764-7770), BTC puede abrir una sonda tras cada reinicio. En BTC el ensamble sólo tiene a DarkAlpha, y su opinión entra en el compuesto con el 60 % del peso residual (`w_nn`, :5357-5361).
6. **Código sin consumidor.** Maker, `src/trailing.rs`, `src/features/`, `multi_asset_orchestrator`, `momentum_booster`, `strategy_telemetry`, `multivariate_coint`.
7. **Heap en el camino caliente.** Hay `String` y `format!` en cada tick (:4571-4573), `format!` en cada cierre (:4206) y en cada deliberación del consejo (consejo_seniors.rs:431, :623). «< 25 ns»: sin medir.
8. **Tests de #713 y #714.** ¿Prueban la conducta viva que certifican?
   - El de #713 sólo importa `normalized_directional_divergence` (puertas_del_continuo_symmetry_contract.rs:18). No ejecuta `puertas_del_continuo` ni el ML-GATE del núcleo.
   - El de #714 ejecuta el núcleo para K-26, pero sólo comprueba que los dos p-valores son finitos y que en calentamiento los dos lados aceptan (conformal_direction_symmetry_contract.rs:91-101). No compara el p-valor del largo con el del corto. K-27 es un test de fuente (:105-117).
   - El T-1 de aa716edf, que incluye estos cambios, sigue en 16/144 con la misma lista. Eso no prueba la simetría: sólo dice que el fixture no la ejercita.

**Puertas**
- Tras `process_event`, `qo_613_rho_tau` debe ser el IC calculado a `tau_operativa_unificada`.
- Con ruido de entrada, |v_dom| no supera lo esperable del máximo de 32 ruidos y su IC ≈ 0.
- Espejo: invertir todas las entradas invierte el lado y deja igual la confianza.
- Un test con contador de asignaciones por evento.
- Paridad 48D (D-753), golden y T-1.

**Entregable**
- Un «mapa de τ y de confianza» del camino vivo: por factor, unidad, τ, calibración y número de veces que entra en el tamaño. Se cruza con el ledger de Línea E sin editarlo.
- El ledger se usa como índice, no como estado verificado:
  - AGY marcó RESUELTO R-15 en #711 y C-02 y C-W en #712 (docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:135, :42, :34). También marcó RESUELTO K-06 y K-23 en #713 (:85, :86) y «Arreglado» en su tabla de cierre (:205, :206), y «Arreglado» K-26, K-27 y el freno del host en #714 (:207, :208, :214).
  - La revisión de Línea E dice que R-15 «mejora, pero no cierra la causa». Está en §7.3 del ledger del PR #32 (05f58e87, :280-287), que aún no está en aa716edf.
  - Cada RESUELTO o «Arreglado» se comprueba contra el código antes de heredarlo.
- CL-n en zona C y pedidos con test a los demás dueños.
- LC-F3-02 (`atr_1m` ×7,746, :4427) no se arregla en esta fase: pasa al principio de C-F4 junto con el ATR (LC-F6-01).

---

### C-F4 — L4, dinero

**Objetivo**
- Que el tamaño se decida en espacio de riesgo al stop.
- Que cada veto, límite y rechazo tenga unidades correctas, espejo y una prueba de que evita EV negativo.
- Que ningún veto sea absorbente.

**Archivos:**
- `crates/risk-engine/src/{lib.rs, tp_sl.rs, correlation_guard.rs, kelly_envelope.rs, leverage_matrix.rs, drawdown.rs, ruin.rs, capital_regime.rs, kelly.rs, orchestrator.rs, copulas_store.rs, guard.rs, capital_compounder.rs, epigenetic_capital_alloc.rs, epigenetic_fitness_landscape.rs}`;
- `crates/god-engine-core/src/{entry_reservation.rs, quantum_kelly_risk.rs}`;
- `crates/quantum-arena/src/position.rs`;
- `crates/audit-engine/src/{drift_auditor.rs, shadow_ledger.rs}`. El libro sombra pasa aquí desde C-F7, porque lo necesitan D1 en sombra, des-absorber el EV y el paso 3 de §5;
- `config_dir/copulas_manifest.json`.

Bloques que esta fase lee en archivos de otras fases:
- god_engine.rs:1682-1743 (sistema inmune; umbral con d* en :1729-1737), :2354-2434 (breaker de comisiones) y :4023-4151 (envolvente);
- booktick_replay.rs:640-850 (zona Sol: `live_envelope_gate` y su rollback);
- god-engine-core/src/lib.rs:7750-7807 (sonda de arranque frío, ML-GATE y aprobación final) y :4422-4427 (`atr_1m`);
- god-engine-core/src/stateful_engine.rs:850-853, :876-881 y :1024-1029 (ATR).

**Primero, el ATR.**
- El ATR `v_t`, con su trinquete dentro de la vela y sus dos estimadores (LC-F6-01), y `atr_1m` publicado ×7,746 (LC-F3-02) alimentan stops, tamaño y el tope IOC.
- Si se corrigen después de medir aquí «riesgo al stop ±10 %», o el tope IOC en C-F5, esas mediciones dejan de valer.
- Por eso los dos se arreglan al empezar C-F4 (ciclo 13), con sus ids. El recibo de cada archivo sigue en su fase dueña (C-F6 y C-F3).

**Preguntas**
1. **¿Espacio de riesgo?**
   - Hoy Kelly entra como fracción de margen y otra vez dentro del apalancamiento.
   - La envolvente del host sólo cambia `exec_leverage`, con la cantidad ya fijada (god_engine.rs:4093-4151; envio.rs:69-100).
2. **¿Cuántas veces entra la confianza?** (K-18/R-17).
3. **Las 17 razones `REJ_*`** (risk-engine/src/lib.rs:243-1446):
   - Lundberg ÷ L_max (:674);
   - primer toque (:952-974);
   - sonda (:1136, :1188) y prior 0,55 (:1155);
   - `min_ev_mult` (:1170-1175) y `max_fee` 0.035 (:1206);
   - `micro_lev_cap` (:1268-1270) y tope de 2,60 USD (:1323);
   - `risk_normalizer` (:519-528).
4. **Absorbentes.**
   - `peak_capital` no se persiste (:284-286). El pico del host tampoco: es una variable local de la tarea inmune (god_engine.rs:1682).
   - `riesgo_por_operacion` se escribe antes de los vetos del consejo, del ML-GATE y del host. Su único escritor es `evaluate_single_intent` (risk-engine/src/lib.rs:1434-1446), que también corre en el backtest. Sus lectores son risk-engine/src/lib.rs:324 y :646, god_engine.rs:1734 y online_daemon.rs:1131.
   - `DriftRecovery` (god_engine.rs:3825-3827).
5. **Una sola semántica de drawdown y de pico** entre host y riesgo.
   - #714 añadió `.min(d*)` al freno del host sin ningún test del host; su único test es `conformal_direction_symmetry_contract.rs`. Aun así, el ledger lo marca «Arreglado» (LEDGER_DECISION_VIVA_2026-10-10.md:214).
   - La semántica sigue siendo distinta. El host usa `drawdown_maximo(...).min(d*)` (god_engine.rs:1733-1737). El riesgo usa `lerp(dd_max_medido, d*, micro_w).min(d*)` (risk-engine/src/lib.rs:323-341). En régimen micro pleno (`micro_w` = 1) pueden diferir: el riesgo usa d*, y el host min(drawdown_maximo, d*), que es más bajo cuando drawdown_maximo < d*.
6. **Ruina con NaN o ±Inf** (RA-RUIN-F01).
7. **Genes inertes.** `btc/eth_volatility_multiplier` y `kelly_bootstrap_cold`.
   - Retirarlos o conectarlos cambia el vector de 144 genes (`SuperGenotype::DIMENSION`, genome.rs:2006; `validate`, genome_store.rs:248) y los `active.json` de demo y prod.
   - Por eso en esta fase sólo se declaran. Se retiran o se conectan en el ciclo 15, después de GENOME-GATE (LC-F7-16; §8).
8. **Literales anclados a la cuenta de 13 USD.**
9. **La cartera como universo multiactivo.**
   - La candidata entra con la EWMA de riesgo, no con su riesgo real.
   - ρ̄ equicorrelacionada.
   - Cópulas de 2026-08, con ADR-0009 vencida.
   - Ranuras con Δlnτ ≥ 0,80 (position.rs:841).
10. **Dos sondas de arranque frío (D4, LC-F4-19).**
    - Además de `es_fase_sonda` (`trade_count < 5`, risk-engine/src/lib.rs:1136), el núcleo tiene su propia sonda: `arranque_frio_sin_roster` (god-engine-core/src/lib.rs:7764-7770). Se arma sin modelo de roster, con `trade_count == 0` y sin posición abierta en la moneda.
    - `trade_count` no se persiste. Nace en 0 (quantum-arena/src/state.rs:48) y sólo sube al cerrar (god-engine-core/src/lib.rs:4001; reconciliation.rs:524-529).
    - Tras cada reinicio, cada una de las 23 monedas sin `_MOTOR` puede volver a abrir una sonda. El universo es de 26 (bootloader.rs:320-347), y en `models/` sólo hay `_MOTOR` para ATOM, BNB y NEAR.

**Puertas**
- Riesgo al stop realizado = fracción objetivo ±10 % en replay, medido después del arreglo del ATR.
- Continuidad: sin saltos en `kelly_survival_cap_ratio`, en n = 30 ni en 50 USD.
- Espejo.
- Monte Carlo de ruina con pérdidas medidas: la frecuencia observada ≤ la prometida.
- Forense en dos mitades: bloqueada por datos externos. Los tapes viven en el PC del operador y desde la nube Binance responde 403. C-F4 puede cerrarse sin ella, y el recibo la deja abierta.
- `riesgo_por_operacion`: un test en `bt_vivo_parity_audit` con la escritura en el punto común y su reversión cuando se veta la orden.
- Pico: un test de reinicio en el que el pico y el umbral sobreviven.
- D4: una sola sonda abierta en toda la cartera, también tras reiniciar.
- T-1, declarando antes los genes inertes; en esta fase no se retiran ni se conectan (P7).
- Lo que el T-1 no ve. Corre con 1 000 USD (backtest-engine/tests/t1_cobertura_genetica.rs:50-52), así que no ejercita los literales micro: ni el tope de 1,20-2,60 USD (risk-engine/src/lib.rs:1323), ni el techo micro de apalancamiento (:1268-1270), ni el peso micro de `capital_regime`. Para esos cambios la puerta es un fixture de 13 USD, o un caso de 13 USD en `bt_vivo_parity_audit`, que hoy usa 1 000 USD en todos sus casos.
- Las 30 suites de tests de risk-engine (30 en 5338eda1 y en aa716edf).

**Entregable** (CL-n):
- LC-F6-01 y LC-F3-02 (ATR), antes que el resto;
- pico y umbral únicos, con una persistencia mínima del pico definida en esta fase (no espera a `learned_state` de C-F8);
- `riesgo_por_operacion` registrado en un punto común del núcleo y del replay:
  - después del consejo y del ML-GATE (desde god-engine-core/src/lib.rs:7807);
  - con reversión cuando veta el host, y su espejo en `live_envelope_gate` (booktick_replay.rs:673, :830);
  - escribirlo sólo en el host rompería la paridad. `run_backtest_native` (T-1), `booktick_replay`, el examen del demonio y ShadowForest lo dejarían en 0, y `drawdown_maximo` caería al gen. Los cuatro deciden por `process_event` (backtest-engine/src/lib.rs:351; booktick_replay.rs:434, :477, :513; online_daemon.rs:560; random_forest.rs:188);
- ruina con error explícito;
- Lundberg con N/C;
- D4: una sola puerta de sonda para toda la cartera, que unifica la del riesgo y la del núcleo;
- D1 en sombra, como función única para host y replay. El libro sombra sólo registra hasta el paso 2 de §5 (etiquetas); alimentar con él calibrador, consejo y EV va después;
- retirada de `capital_compounder`, `epigenetic_*` y `quantum_kelly_risk`, con acuerdo. Cada retirada cita la prueba de falsación que falló, o «sin consumidor» con su grep, y se lleva sus tests (`risk-engine/tests/compounder_admission_contract.rs` y `risk-engine/tests/epigenetic_allocation_contract.rs`).

---

### C-F5 — L5, ejecución

**Objetivo**
- Que lo que se envía sea lo que se validó.
- Que cada cierre vuelva al aprendizaje con su precio real y su ranura.
- Que el kill-switch pare lo que dice parar.

**Archivos:**
- `src/bin/god_engine.rs`. Bloques:
  - ingesta WS (CL-50…54);
  - reservas :588, :3414;
  - breaker :2354-2434;
  - modelos y entrenador NN :2026-2217;
  - daemon :2098-2126;
  - ShadowForest :2970, :3422, :4890-4923;
  - drift :3762-3827;
  - `close_was_real` :3677-3688;
  - envolvente :4023-4151;
  - envío :4294-4554;
  - IOC :4439;
  - vortex :4267.
- `crates/execution-engine/src/{executor.rs, reconciliation.rs, user_data_stream.rs, ioc_evidence.rs, order_registry.rs, entry_dispatch.rs, trade_accounting.rs, income_evidence.rs, client.rs, execution_evidence.rs, ntp.rs, ws_executor.rs, simulator.rs, shadow.rs, router.rs, quantum_multiplexer.rs, quantum_socket.rs, hot_swap.rs}`;
- `crates/risk-engine/src/envio.rs`;
- `crates/god-engine-core/src/{trailing.rs, reality_physics.rs, slippage_predictor.rs, latency_accelerator.rs, order_flow_aggregator.rs}`;
- `crates/quantum-arena/src/protection_health.rs`;
- `crates/backtest-engine/src/network_jitter.rs`.

**Preguntas**
1. **Paridad validado→enviado.**
   - Apalancamiento: `env_lev.floor()` (god_engine.rs:4094).
   - `boot_lev` usa el margen de otra ranura (god_engine.rs:4039, :4064-4071).
   - El tope IOC de 5 a 35 pb (:4439) no coincide con `tp_sl::roundtrip_friction`. Usa el ATR del arena (:4287-4296), así que se mide después del arreglo del ATR en C-F4.
2. **Cierres.**
   - `close_was_real` no sabe qué ranura cerró (:3677-3688).
   - Los fantasmas toman como salida `entryPrice`, que es 0 en posiciones planas (reconciliation.rs:432-545).
   - El payoff está en USD y con doble EWMA.
   - `break` tras el primer cierre (god-engine-core/src/lib.rs:4332-4333).
3. **Kill-switch (LC-F5-10).**
   - ADR-0015 lo trata como latch y deja abierto que hoy es permanente hasta reiniciar el proceso, sin prueba de runtime del rearme con histéresis (:86-87). Su contexto nombra como escritores el núcleo, el sistema inmune y X-009 (:13-15).
   - En aa716edf hay ocho sitios que lo arman:
     - el núcleo, con capital ≤ 0 (god-engine-core/src/lib.rs:7122);
     - el sistema inmune del host (god_engine.rs:1802);
     - la escalada X-009 (god_engine.rs:4757);
     - el apagado del host, al salir del bucle principal (god_engine.rs:4947-4950). ADR-0015 cita el apagado en sus decisiones (:48), pero no entre los escritores;
     - el demonio de evolución, por pérdida constante (online_daemon.rs:1222, :1235, :1324) y por EWMA(t) < −1,5 (:1343). Con t ≤ −2 el demonio hace rollback (:1220, :1242), pero no arma el latch.
   - El demonio lo arma también en prod. Con la evolución desarmada, el host avisa de que «deriva, kill-switch y rollback siguen activos» (god_engine.rs:2108-2111).
   - Pregunta: ¿cada escritor de `kill_switch_active` figura en ADR-0015 con su criterio?
   - Mientras un estadístico descriptivo del demonio pueda armar el latch, Línea C no recomienda «latch permanente con rearme manual».
   - Ojo: `EMERGENCY_SHUTDOWN.bat` no es un kill-switch (C-F9).
4. **Taker o maker.** ¿Se mide el deslizamiento real por símbolo? `trade_accounting` ya tiene `slippage_bps` (trade_accounting.rs:68).
5. **Trailing.** Escalera discreta de 1,5 / 2,5 / 3,5 / 4,5 ATR de 1 minuto (god-engine-core/src/trailing.rs:193-199).
6. **Sin consumidor.** `router`, `simulator`, `shadow`, `quantum_multiplexer`, `quantum_socket`, `hot_swap`, el stub `ws_executor` y `slippage_predictor`.
7. **R01.** Readmitir el margen final antes del envío.

**Puertas**
- Las 23 suites de tests de execution-engine (23 en 5338eda1 y en aa716edf).
- Test con dos ranuras confirmadas.
- Test de fuente del vortex.
- Test de fuente: todo escritor de `kill_switch_active` figura en ADR-0015 con su criterio. La búsqueda reconoce también la escritura partida en varias líneas (god_engine.rs:4947-4950) y las llamadas a `latch_degradation`.
- Paridad bt↔vivo.
- Lo que el T-1 no ve. El T-1 no ejecuta el host: ni la envolvente, ni la IOC, ni el kill-switch, ni `close_was_real`. Para la IOC, el kill-switch y `close_was_real` la puerta son tests del host con un fixture de 13 USD. `bt_vivo_parity_audit` sólo cubre la envolvente (su espejo `live_envelope_gate`, booktick_replay.rs:673) y necesita un caso de 13 USD (hoy usa 1 000 USD). El recibo dice que el T-1 no cubre estos cambios.
- Nada contra el exchange sin autorización del dueño. Desde la nube, Binance responde 403.

**Entregable.** CL-n de cada hallazgo confirmado, y retirada de módulos muertos con visto bueno.

---
### C-F6 — L6, datos

**Objetivo.** Un estimador por magnitud, features en tiempo físico y un universo y specs medidos en el exchange.

**Archivos:**
- `crates/god-engine-core/src/{stateful_engine.rs, bootloader.rs, liquidation_feed.rs}`;
- `crates/quantum-arena/src/{tick_source.rs, active_universe.rs, symbol_ranker_engine.rs}`;
- `src/{parsers.rs, symbol_manager.rs, orderbook.rs}`;
- `src/bin/{binance_vision_sync.rs, download_history.rs, oi_history_sync.rs, macro_history_sync.rs, parquet_to_bin.rs}`;
- `crates/data-ingest/src/` (cola_ws, liquidation, dynamic_selector, tensor_parser, world_bank, lib);
- `crates/data-pipeline/src/{omni_multiplexer, dynamic_ranker, spot_feed, ws_client, parser, validation, state_db, historical, macro_data, market_context, multiplexer, resilient_stream, api_client, asset_selector, macro_feed, onchain_feed, feature_vm, persistence}.rs`;
- `crates/storage-engine/src/{history_store.rs, raw_zone.rs}`;
- `crates/feature-engine/src/omni_strategies.rs`;
- `crates/execution-engine/src/dynamic_symbols.rs`;
- `crates/backtest-engine/src/{label_evidence.rs, tick_replayer.rs}`;
- `data/oihist/*.csv`.

**Preguntas**
1. **ATR** (LC-F6-01).
   - `v_t` lleva un trinquete dentro de la vela (stateful_engine.rs:851-853).
   - EMA 0.85/0.15 con suelo de 10 pb en vivo, frente a 0.8/0.2 con suelo de 5 pb en el calentamiento (:877-881, :1025-1029).
   - El arreglo se adelanta a C-F4, junto con `atr_1m` ×7.746 (LC-F3-02, god-engine-core/src/lib.rs:4427). Los dos alimentan stops, tamaño y el tope IOC. Si se corrigieran aquí, después de que C-F4 mida el riesgo al stop y C-F5 fije el tope IOC, esas mediciones dejarían de valer. `stateful_engine.rs` conserva aquí su recibo, y C-F4 sólo trabaja sobre el bloque citado.
2. **Muros.** Se calculan con las cantidades del nivel 0 de depth5 (god_engine.rs:3156-3157), que el host guarda como muros con `update_l2_depth` (:3349).
3. **Universo y specs.**
   - 26 símbolos fijos (bootloader.rs:317-350).
   - `max_leverage` por nombre y fees VIP0 (dynamic_ranker.rs:236-251), sin leer `/fapi/v1/leverageBracket`: ningún archivo `.rs` lo nombra.
4. **Ceros estructurales.**
   - 9 dimensiones cross-exchange (omni_multiplexer.rs:183-191), declaradas muertas por contrato (:170-180).
   - `binance_futures` sin escritor (spot_feed.rs:126-136; omni_multiplexer.rs:183).
5. **Tiempo de eventos.** FFT de 64 retornos, MFDFA 10/25/50, RSI14 por tick y EMAs 20/200 por tick (stateful_engine.rs:186-188, :733-737). Las EMAs 9/21/120/720 van sobre la vela interna de 1 minuto (:883-902), que cierra con el primer evento pasados 60 s.
6. **dstream.** El host lo retiró en el ciclo 10 (CL-54, god_engine.rs:1038), que ya está en main con aa716edf. `dns_optimizer.rs:24` todavía lo lista (LC-F8-05).
7. **Etiquetas.** `label_evidence` usa barreras fijas (`BarrierSpec`, label_evidence.rs:94-98), no σ√τ.

R7 de Ronda 7 (datos, telemetría y guardianes; incluye C-02) sigue ABIERTA y sin dueño en la tabla §R7-1 (BARRIDO :1805). Hay que coordinarla en el buzón antes de empezar.

**Puertas**
- ATR sobre una serie de TR conocida, y el mismo estimador en calentamiento y en vivo. Las dos se cierran en C-F4, con el arreglo de LC-F6-01.
- Replay de un tramo sintético con depth conocido. La versión con tape real está bloqueada por datos externos: los tapes viven en el PC del operador.
- Suites de data-pipeline (4) y data-ingest (2).

**Entregable.** CL-n en zona C (ingesta del host) y pedidos con test al resto.

---

### C-F7 — L7, aprendizaje y evolución

**Objetivo**
- Que cada aprendiz use la etiqueta de lo que predice.
- Que haya un solo contrato de promoción con nivel estadístico.
- Que se conserve la paridad entre vivo y evaluación.

**Archivos:**
- `crates/god-engine-core/src/{ensemble.rs, model_reload.rs, outcome_context.rs, recarga_genoma.rs, darwin.rs, fitness_contract.rs}`;
- `crates/quantum-arena/src/{genome.rs, genome_store.rs}`;
- `src/bin/{train_forest.rs, auto_trainer_daemon.rs, evolver.rs, walkforward_evolver.rs, evolution.rs, backtest_windows.rs, train_dark_alpha.rs}`;
- `src/model_manifest.rs`;
- `crates/metacortex-engine/src/{online_learning.rs, compiler_sandbox.rs, evolutionary_templates.rs, hot_swap_controller.rs, fases_autonomous.rs, reminiscence_and_adn.rs, cazador_constantes.rs, epigenoma_store.rs}`;
- `crates/dark-alpha-engine/src/{online_ppo.rs, neuro_plasticity.rs}`;
- `crates/evolution-engine/src/` (todos);
- `crates/backtest-engine/src/{lib.rs, booktick_replay.rs, vectorized.rs, oos_context.rs, metrics.rs}`;
- `crates/storage-engine/src/evolution_ledger.rs`;
- `config_dir/genomes/`, `config_dir/genotypes/`, `config_dir/models_manifest.json`, `models/*.json`;
- `scripts/revalidate_roster.ps1`.

`crates/audit-engine/src/shadow_ledger.rs` pasa a C-F4. Lo necesitan C-F4, para des-absorber el EV y para D1 en sombra, y el paso 3 de §5.

**Zonas.** R7-R6 de Qoder (.r7r6; buzón :8309-8314). Línea E: escritura de modelos y manifiesto. Sol: replay y métricas. GLM: promoción. Línea C: `genome_store`, `darwin` y GENOME-GATE.

**Preguntas**
1. **Promotores.** Hay tres con rigor distinto:
   - el demonio, con DSR y 30 operaciones;
   - ShadowForest, con best-of-9, 15 operaciones, sin multiplicidad, 1 de cada 10 eventos y sin `live_envelope_gate`;
   - `evolver`, con el `process_tick` legado (evolver.rs:430) y capacidad de promover (:674).
2. **Examen.**
   - Las series se replayan en los slots reales del universo.
   - Hay fuga entre entrenamiento y validación.
   - La interpolación es lineal.
   - Las features omni son fabricadas.
   - Un solo τ de 16 s y ≤ 8 monedas (online_daemon.rs:59-66, 430-735).
3. **Multiplicidad.** `cumulative_trials` es monótono (:1888).
4. **Etiquetas.**
   - Bosque: barrera fija de 36/18 pb a 300 s.
   - DarkAlpha: éxito del lado, no dirección del precio.
   - `forest6_prob`: probabilidad de ganar, sin lado.
   - Ensamble: vela de 1 minuto.
   - PPO: estado del momento del cierre.
   - Consejo y calibrador: sólo operaciones ejecutadas.
5. **Vigilantes con nivel.**
   - El demonio arma el latch global `kill_switch_active` por pérdidas constantes (online_daemon.rs:1324) y por una EWMA del estadístico t por debajo de −1,5 (:1335-1347). Esas dos escrituras corren antes de la puerta de armado (:1356), así que valen también en prod. Las de la vigilancia post-promoción (:1222, :1235) sólo corren con una generación vigilada, que se arma después de esa puerta (:1374-1386, :1940-1946). ADR-0015 no nombra ninguna (§1.4).
   - Ville que apuesta a favor.
   - `.forensic_violation` (:1035-1051).
6. **GENOME-GATE** (LC-F7-16). `load_active` y `desde_json` (genome_store.rs:140-182) no llaman a `validate` (:248). Retirar o conectar genes cambia el vector de 144 genes que comprueba `validate` y los `active.json` de demo y prod. Por eso, si C-F4 retira o conecta genes, GENOME-GATE entra antes que C-F4. Si no entra antes, retirar genes espera a C-F7. §8 eligió la segunda opción: en C-F4 los genes inertes sólo se declaran (LC-F4-07), y GENOME-GATE entra en C-F7 (ciclo 15) antes de retirarlos o conectarlos.
7. **Artefactos.**
   - Genomas demo y prod idénticos en la generación 1 «backtest_heritage».
   - Modelos sin metadatos.
   - `revalidate_roster.ps1` borra `_SCALP`, pero el motor usa `_MOTOR`.
8. **T-1.** ¿Mide al juez vivo? No: llama a `run_backtest_native` con 1 000 USD (t1_cobertura_genetica.rs:50-52), sin el host y sin el régimen de 13 USD.

**Puertas**
- Nulo: con genomas aleatorios, ≤ 5 % de promociones falsas en cada promotor.
- Potencia con edge plantado.
- Una posición abierta en la frontera no puntúa en validación.
- Paridad 48D.
- GENOME-GATE rechaza genomas fuera de rango.
- Las 9 suites de evolution-engine.
- T-1.

**Entregable**
- Contrato único de promoción (pedido a GLM y Qoder, con tests).
- GENOME-GATE (CL-n).
- ADR si cambia la doctrina de promoción.

---

### C-F8 — L8, telemetría y operación

**Objetivo.** Que la telemetría no se confunda con control, que el estado aprendido sobreviva a un reinicio y que latencia y memoria estén medidas.

**Archivos:**
- `crates/god-engine-core/src/{orchestrator.rs, direction_diag.rs, ml_coverage.rs}`;
- `crates/quantum-arena/src/state_continuity.rs`;
- `src/{dashboard.rs, bin/dashboard.rs, dark_alpha_router.rs, dark_alpha_sniffer.rs, quantum_arena.rs}`;
- `crates/storage-engine/src/{mmap_bus.rs, lakehouse.rs, ledger.rs, temporal_store.rs}`;
- `crates/os-guardian/src/` (todos);
- `crates/data-pipeline/src/{teleonomia.rs, telemetry_bus.rs, lakehouse_mmap.rs, storage.rs}`;
- `crates/metacortex-engine/src/{shadow_graph_auditor.rs, immune_system.rs}`;
- `crates/audit-engine/src/{trajectory_auditor.rs, telemetry.rs, auditor_interno.rs, behavioral_auditor.rs, cybernetic_resilience.rs, resilience.rs, state_validator.rs}`;
- `crates/telemetry-server/src/` (todos), `crates/telemetry-engine/src/` (todos), `crates/flight-recorder/src/lib.rs`;
- `deployment/`, `docker-compose.yml`, `Dockerfile`, `.dockerignore`;
- `static/`, `dashboard/`;
- `scripts/monitor_demo.ps1`.

**Preguntas**
1. `darwin_approved.store(true)` es incondicional (god_engine.rs:1292).
2. No existen `learned_state` ni `DecisionRecord` en el código. Son pendientes de la hoja de ruta: DecisionRecord en el ciclo 9 y el snapshot del estado aprendido en el ciclo 10 (HOJA_DE_RUTA_CIMIENTOS §6, :415 y :421). La persistencia mínima del pico se adelanta a C-F4 (LC-F4-10), con su test de reinicio, y C-F8 parte de ella.
3. El bus mmap lleva dos esquemas al bosque online.
4. El profiler supone 3 GHz (profiler.rs:123-124) y bajo carga se descartan logs (macros.rs:71-82).
5. Veto por RAM 0.85/0.70 (memory_audit.rs:38, :45; god_engine.rs:3975-3988): ¿umbral medido?
6. `deployment/` es del sistema Python retirado.

**Puertas**
- Benchmark reproducible del camino caliente: tiempo y asignaciones por evento. Hoy sin medir. El medidor se adelanta al ciclo 12 (§8) porque la puerta de rendimiento del protocolo (§7, paso 5) lo necesita. Aquí se le añade el desglose por etapa (núcleo, consejo, riesgo) y por tamaño de cartera.
- Test de ida y vuelta del estado aprendido: guardar, reiniciar y obtener el mismo estado y la misma decisión.
- Test de que `darwin_approved` refleja la auditoría, o retirada.

**Entregable.** Panel ex-post, snapshot del estado aprendido y marca o retirada de los buses sin lector.

---

### C-F9 — L9, herramientas y binarios

**Archivos:**
- `crates/god-engine-core/src/{ml_registry.rs, bin/test_ml.rs, bin/nano_benchmark.rs}`;
- `src/bin/{audit_forensic_backtest.rs, multi_coin_simulator.rs, feature_exporter.rs, spectral_bench.rs, tape_spread_fix.rs, feature_validator.rs, votes_export.rs, income_report.rs, system_health.rs, latency_flow_audit.rs, ml_path_probe.rs, oco_probe.rs, algo_fetch_probe.rs, test_forest.rs, test_forest2.rs, benchmark.rs, config_compiler.rs, graph_server.rs, copulas_manifest.rs, model_manifest.rs}`;
- `crates/quantum-arena/src/{arena_alloc.rs, ring_buffer.rs, zero_alloc_pool.rs, zero_copy_stream.rs, net_multiplexer.rs}`;
- `crates/data-pipeline/src/{chaos_monkey.rs, bypass.rs}`;
- `crates/metacortex-engine/src/lib.rs`;
- `crates/audit-engine/src/{lib.rs, main.rs}`;
- `crates/backtest-engine/src/bin/continuous_evolution_backtest.rs`;
- `crates/graph-architecture/`, `crates/graph-4d/`, `crates/phase-runner/`, con sus `Cargo.toml`. Esos tres `Cargo.toml` sólo tienen esta fase dueña; C-F1 no los toma;
- `scripts/` (audit_*.py, ronda7_dead_census.py, fixtures/*.rs, tests/), salvo los asignados a C-F7, C-F8 y C-F10;
- `launchers/` (15);
- `LAUNCHER.bat`, `LAUNCH_*.bat`, `LAUNCH_GOD_MODE.ps1`;
- `build.ps1`, `build.sh`, `build_loop.ps1`, `run_god_engine.ps1`, `run_sim_5.ps1`.

**Preguntas**
1. ¿El forense que decide si se autoriza operar (`audit_forensic_backtest.rs`) mide el sistema vivo? Usa `FORENSIC_SYMBOL`, con BTCUSDT por defecto (:227), y BTC no tiene `_MOTOR`.
2. Lanzadores:
   - `EMERGENCY_SHUTDOWN.bat` escribe `STOP_TRADING.LOCK` en su directorio de trabajo (:13) y mata `python.exe` (:17). No detiene `god_engine`.
   - El binario ya tiene dos caminos de parada. Con ^C en su consola arma el kill-switch, llama a `flatten_all_positions`, persiste el estado y sale (god_engine.rs:1112-1132, :4942-4976). Con `STOP_TRADING.LOCK`, que busca en su CWD y junto al ejecutable (:5153-5163), arma el latch y aplana, pero el proceso sigue vivo (:1691, :1781-1834). Que el archivo del `.bat` caiga en una de esas rutas está sin verificar.
   - `WANDB_SYNC_ALL.bat` fabrica un Sharpe al azar (:22) e imprime «PRODUCCION AUTORIZADA» (:27).
   - 14 lanzadores apuntan a archivos que no están en el árbol. Diez de `launchers/` ejecutan un script de Python inexistente (`main.py`, `tests/run_backtest_parallel.py`, `dashboard/app.py`, entre otros). `HELP.bat` sólo los nombra, `WANDB_SYNC_ALL.bat` activa un `.venv` que no está versionado, `LAUNCHER.bat` arranca `premium-dashboard` (:86), que tampoco existe, y `LAUNCH_PRODUCTION.bat` copia `.env.template` (:11), que no existe.
   - `LAUNCH_PRODUCTION.bat` anuncia «DINERO REAL ACTIVADO» (:42), carga las claves de `.env` y compila en `target_prod`, pero lanza `god_engine.exe` sin `--force-live` (:46). Con la puerta del host (god_engine.rs:846-875) eso arranca en demo, y en mainnet el ejecutor queda en papel (:1206). Por lectura, no llega a mainnet con un clic. Además copia `.env.template`, que no existe: la plantilla versionada es `template.env` (LC-F9-06).
3. ¿`run_god_engine.ps1` redirige a `logs/demo_vNN.log`, como exige .agents/AGENTS.md:27-29? No: lanza `cargo run` sin redirigir la salida (:12). Además está en UTF-16 e inyecta claves «shadow» que pisan las de `.env`, en esa misma línea (LC-F9-07).
4. `continuous_evolution_backtest` tiene nichos de scalping.
5. Asignadores zero-copy muertos; `net_multiplexer.rs` ni siquiera está declarado.

**Puertas**
- Todos los bins compilan.
- Pasa `scripts/tests`.
- Test de fuente: ningún lanzador apunta a un archivo inexistente.
- Test de fuente: ninguna línea que invoque `god_engine` lleva `--force-live`, sin contar los comentarios (REM) ni los textos que sólo se muestran (echo, title), y ningún lanzador crea `config_dir/MAINNET_ARMED`. Armar mainnet sigue siendo un paso humano explícito; el lanzador que no lo respete se reescribe o se retira.
- Test con un ejecutor simulado: el kill-switch de operador llama a `flatten_all_positions` y el proceso termina. Por lectura, hoy sólo el camino de ^C hace las dos cosas; el de `STOP_TRADING.LOCK` aplana, pero no termina.

**Entregable.** Con acuerdo del dueño:
- lanzadores retirados o reescritos;
- un kill-switch de operador documentado que detenga `god_engine` y aplane posiciones por el binario.

---

### C-F10 — L10, suites, CI y oráculo

**Archivos:**
- `crates/*/tests/` (todos). Dos llegaron con #713 y #714 y certifican cambios de conducta viva sin T-1: `god-engine-core/tests/puertas_del_continuo_symmetry_contract.rs` (#713) y `god-engine-core/tests/conformal_direction_symmetry_contract.rs` (#714). Su recibo es de esta fase, pero C-F3 los lee junto con el código que certifican;
- `crates/omniscient-registry/src/tests.rs`;
- `crates/backtest-engine/src/booktick_causality_contract.rs`;
- `crates/audit-engine/src/forensics.rs`;
- `crates/backtest-engine/tests/{t1_cobertura_genetica.rs, support/t1_measurement.rs}`;
- `.github/workflows/{replay-contracts.yml, unit-suites.yml}`;
- `scripts/quality_gate.ps1`;
- posteriores a aa716edf: `.githooks/pre-commit` (#715) y `scripts/t1_oraculo.sh` (PR #32).

**Zonas.** `unit-suites.yml` es de Línea E; `replay-contracts.yml` de Codex y GLM; el T-1 en paralelo (QS-P) de Línea E. QS-P (§30.7, plan sinc. 05f58e87:1271) y `scripts/t1_oraculo.sh` no están en aa716edf. Al re-anclar este plan, el PR #32 seguía en borrador (head 05f58e87). La API de GitHub lo da después fusionado en main como 03b3f1ca (padres aa716edf y 05f58e87), y ahí el script ya existe. Hasta que el plan se re-ancle a 03b3f1ca o a una revisión posterior, es una dependencia externa de esta fase.

**Preguntas**
1. **¿Qué no corre CI?** Ninguno de los dos workflows ejecuta:
   - 7 de las 9 suites de evolution-engine: `cma_*` (3), `fitness_evidence`, `return_evidence`, `r7_r6_watchdog_evidence` y `shadow_forest_threshold`. De ese crate sólo corren `--lib` y dos suites compiladas con `rustc` (replay-contracts.yml:68-70, :95-105);
   - storage-engine: sus 3 suites (`mmap_admission_contract`, `mmap_open_diagnostics`, `ownership_read_contract`) y sus tests unitarios;
   - telemetry-server, telemetry-engine, omniscient-registry, phase-runner, os-guardian, flight-recorder, graph-architecture y graph-4d;
   - los tests de los binarios del paquete raíz salvo `train_forest`: `feature_exporter.rs` (10), `train_dark_alpha.rs` (8) y `god_engine.rs` (2). `unit-suites.yml:54` sólo corre `--lib --bin train_forest`;
   - el T-1.

   Todo eso sólo se compila, con `check --workspace --all-targets` (replay-contracts.yml:51-52). Un grep en .github/workflows no encuentra esos crates en ningún otro paso.
2. **¿Los tests prueban lo que dicen?** Casos: REV-1 y `forensics.rs`.
3. **¿Cuadran los recuentos citados?**
   - risk-engine tiene 154 `#[test]` en `src/` en 95412dd7, 5338eda1 y aa716edf. En `tests/` tiene 29 archivos en 95412dd7 y 30 en 5338eda1 y en aa716edf; las 30 suites de la puerta de C-F4 son las de aa716edf. Los documentos citan «153/153 tests unitarios y 25/25 suites» (plan sinc. :1055).
   - god-engine-core tiene en `src/` 170 tests en 95412dd7, 178 en 5338eda1 y 180 en aa716edf. Cada cifra incluye un `#[tokio::test]` (bootloader.rs:76): en `#[test]` son 169, 177 y 179.
4. **¿Algún test mide habilidad OOS?** Ninguno en el grupo de señal.
5. **¿main compila en CI?** (LC-F10-06) En aa716edf no queda ningún marcador de conflicto: git grep no encuentra ninguno. El `<<<<<<< HEAD` de #710 estuvo en main desde 759f44ec hasta 0000eb8d (docs/PLAN_MAESTRO_SINCRONIZACION.md:1057 de 0000eb8d). La CI corrió en cada push. «Unit and contract suites» cerró; «Replay contracts» no cerró en ninguno:
   - «Unit and contract suites» dio success en 759f44ec, 0043f889, ab0967e8, 95412dd7, 8f6bcc8a y 0000eb8d.
   - «Replay contracts» falló en 759f44ec, en el paso de marcadores, y quedó cancelled en los cinco push siguientes, porque cada push a main cancela el run anterior (replay-contracts.yml:13-15). El `check --workspace --all-targets` (:51-52) sí terminó en success en ab0967e8, 95412dd7 y 8f6bcc8a. En 759f44ec no llegó a correr, y en 0043f889 y 0000eb8d se canceló. Las suites de contratos que van detrás (desde :53) no llegaron a terminar todas en ningún push de #710 a #714.
   - El paso de marcadores sólo mira `git diff --check HEAD^ HEAD` (:40-44), así que no podía parar los push siguientes.
   - Sobre aa716edf, los dos workflows seguían en curso al re-anclar este plan. Después, la API de GitHub da «Unit and contract suites» en success (run 38097226229). «Replay contracts» quedó cancelled (run 38097226217): el check --all-targets pasó, y la suite de contratos se cortó en el paso 7 cuando el push de 03b3f1ca canceló el run.
6. **Pendientes del ciclo 8** (HOJA_DE_RUTA_CIMIENTOS §6, :406).
   - T-1 nocturno: ninguno de los dos workflows tiene un disparo programado.
   - Toolchain con fecha: la CI ya fija `nightly-2026-06-30` (replay-contracts.yml:45-52; unit-suites.yml:68-74). Sólo falta `rust-toolchain.toml:2`, que va a C-F1. Hecho en el ciclo 11 (CL-56), con su test de fuente.

**Puertas.** Un test-inventario que falle si una suite no la corre ningún workflow y no hay motivo escrito.

**Entregable**
- Inventario test→workflow. Se adelanta al ciclo 11 (§8), porque los tests nuevos de C-F4 a C-F9 caen en suites que hoy no ejecuta ninguna CI (evolution-engine/tests, storage-engine/tests y otras).
  - Hecho en el ciclo 11 (CL-57): `python3 scripts/plan_linea_c.py ci --rev REV --salida TSV` aplica la selección de objetivos de cargo a cada orden `cargo test` y `rustc --test` de los workflows. Resultado en `docs/audit/INVENTARIO_TESTS_CI_2026-10-11.tsv`, sobre la rama del ciclo 11 (main ae6d51a2 más CL-56 y CL-57): 214 objetivos con tests (2 551 `#[test]`), de los que 26 objetivos con 220 tests no los ejecuta ninguna CI. Confirma la lista de la pregunta 1 y añade `continuous_evolution_backtest` y el T-2, que es posterior a aa716edf:
    - evolution-engine: 7 suites, 55 tests; storage-engine: lib y 3 suites, 58 tests; telemetry-server 30, os-guardian 12, omniscient-registry 9 (archivo de C-F1), telemetry-engine 7, phase-runner 7, flight-recorder 5, graph-architecture 5 y graph-4d 4;
    - paquete raíz: `feature_exporter` 10, `train_dark_alpha` 8, `god_engine` 2 (los tests del host) y `continuous_evolution_backtest` 2, que el paquete raíz y backtest-engine compilan desde el mismo archivo;
    - backtest-engine: el T-1 y el T-2 (`t1_cobertura_genetica`, `t2_cobertura_puertas`), fuera a propósito porque son el oráculo largo, y `continuous_evolution_backtest`.
  - Pedido a los dueños, sin tocar sus workflows: Línea E (`unit-suites.yml`) y Codex/GLM (`replay-contracts.yml`) deciden dónde entran esos 24 objetivos, o escriben el motivo de dejarlos fuera. Con eso, `ci --exigir` puede ser la puerta de esta fase.
- Cambios de CI acordados con Línea E y Codex. Entre ellos, que un push a main no cancele el run en curso de `replay-contracts.yml`. #715 (07688cf9), posterior a aa716edf, ya lo hace como `unit-suites.yml:26-31` (replay-contracts.yml:18 en 07688cf9). Falta comprobar en main que un run de replay llega a cerrar. Sigue abierta la otra propuesta: que todo entre por PR con los checks requeridos.

---

### C-F11 — Limpieza: documentación y salida generada

**Archivos:**
- `docs/` salvo `docs/adr/`: 118 archivos en la raíz, 32 en audit y 47 en artifacts (recuentos de aa716edf, iguales a los de 5338eda1);
- `graphify-out/`: 655 archivos generados, el 42 % de los 1 559 versionados;
- informes de la raíz:
  - `ARCHITECTURE.md`, `ATLAS_ANALITICO.md`, `CERTIFICADO_SISTEMA_v1.md`;
  - `FORENSIC_INTELLIGENCE_AUDIT.md`, `INFORME_*.md`;
  - `PLAN_REPARACION_EVOLUCION_SENIOR.md`, `README.md`, `RESULTADO_CERTIFICACION_R2_30D.md`;
  - `STRATEGIES.md`, `WALKTHROUGH_FORENSE_SISTEMICO.md`, `implementation_plan_fee_slayer.md`;
  - `summary_table.md`, `task.md`, `walkthrough.md`;
- `artifacts/live_latency_report.md`, `equity_curve.csv`, `graph_3d_lite.html`, `graph_3d_topology.html`;
- `rustc-ice-2026-09-07T14_42_23-32692.txt`;
- `.agignore`, `.gitattributes`, `.gitignore`, `.freebuff/project-id`.

**Preguntas**
1. ¿Qué afirmaciones de los documentos contradice el código? La lista está en LC-F11-01.
2. ¿`graphify-out` debe versionarse?
3. ¿Sobra el archivo `rustc-ice`?

**Puertas**
- El comprobador de citas path:line de este documento se adelantó a C-F0 (§2).
- Búsqueda en CI de marcadores de conflicto en todo archivo versionado. Hoy el paso de `replay-contracts.yml:40-44` sólo mira el diff `HEAD^ HEAD`. #715 añadió un pre-commit y un paso de `quality_gate.ps1` que también miran sólo el diff.

**Entregable**
- Adendas: nunca se borra evidencia ajena.
- Propuesta de `.gitignore` para `graphify-out`, que deciden AGY o el dueño.
- Tabla de afirmaciones contradichas.
## 3. Teoría → estado → siguiente paso → prueba que debe pasar

En §3 y §4, «lib.rs» sin ruta es crates/god-engine-core/src/lib.rs. Toda cita es de aa716edf salvo que nombre otra revisión.

| # | Teoría (dónde) | Estado | Siguiente paso | Prueba que debe pasar | Fase |
|---|---|---|---|---|---|
| 1 | Fusión espectral de 32 escalas (temporal_spectrum.rs:420-681; pesos :247-257, :608, :1389-1396) | En vivo. El peso depende de qué función lo lee. Sin IC OOS en tape real | Un vector de pesos firmado por el IC prequencial de cada escala, común a todos los lectores | Ruido: IC ≈ 0. OU: IC > 0. Tape real (bloqueada por datos externos): IC OOS mayor que el actual en las dos mitades | C-F2.3 |
| 2 | τ dominante por IC con e-proceso (temporal_spectrum.rs:597-681; evalues.rs) | En vivo. Sin significancia cae al argmax recortado a [30 s, 12 h] | τ por pérdida de pronóstico prequencial (SpectralForecastBank) con test secuencial | PnL neto con τ elegida > τ fija, DSR > 0 en walk-forward purgado. Bloqueada por datos externos | C-F2.3 |
| 3 | Pronóstico prequencial de σ (spectral_tape.rs:746-763; state.rs:303-339) | En vivo. R² OOS de +0,05 a +0,18. Se autoriza por la media de las anclas | HAR-RV por ancla; cada ancla se publica sólo con habilidad propia > 0 | Diebold-Mariano por ancla frente a persistencia. Bloqueada por datos externos | C-F2.3 |
| 4 | Difusión anómala σ(τ)^H para TP/SL (tp_sl.rs:294-411) | En vivo, con τ recortada a las anclas | σ(τ) multiescala del banco espectral | Cobertura empírica de \|r_τ\| ≤ k·σ(τ) para todo τ operado | C-F4 |
| 5 | Primer toque browniano, veto p > 0,88 (tp_sl.rs:469-489; risk-engine/src/lib.rs:952-974) | Mal especificado: la deriva es una coherencia y σ es un rango | μ y σ del pronóstico, una sola p para EV y veto; modo sombra mientras tanto | Brier de P(SL primero) frente a salidas reales (R8-A) | C-F4 |
| 6 | Hurst DFA, bandas 0,45/0,52, MFDFA y duración 2^((H−0,5)/0,02) (math_kernels.rs:520-565; multifractal.rs:372-374; lib.rs:8169-8174, 8491-8492) | En vivo, en tiempo de eventos, sin validar OOS | H continuo desde las funciones de estructura en tiempo físico, con IC | fBm H = 0,7: error < 0,05. Caminata: 0,5 dentro del IC. La duración modulada mejora el EV neto | C-F2.3 |
| 7 | FFT de 64 ticks (spectral.rs; stateful_engine.rs:756-778) | En vivo como feature, con muestreo irregular | Lomb-Scargle o wavelets en tiempo físico | Detecta un ciclo plantado con muestreo irregular | C-F2.3 |
| 8 | Navier-Stokes y Yang-Mills (navier_stokes.rs; yang_mills_gauge.rs:184-188) | En el consejo. Yang-Mills ≈ 0 con retornos asíncronos | Intermitencia multiescala; Yang-Mills con refresh-time y HY, o retirar | «Turbulento» separa el slippage (KS p < 0,01) mejor que el ATR. Varianza real > barajada | C-F2.3 |
| 9 | crash_flux y símplex de régimen (spectral_regime.rs:121; regime.rs:33-180; cálculo vivo en lib.rs:2516-2605) | En vivo. Pesos literales y sigmoides calculadas sólo con BTC | Logística del drawdown futuro; BOCPD/HMM multiactivo calibrado | AUC OOS de caídas > 0,6. Brier de p_crash menor que la climatología. Bloqueada por datos externos | C-F2.3 |
| 10 | BOCPD y W1 (lib.rs:637-644) | Sólo telemetría | BOCPD conjugado sobre retornos normalizados por σ pronosticada | Falsas alarmas < 5 % en ruido. Detección en ≤ N eventos | C-F2.3 |
| 11 | Hawkes univariante (hawkes_bessel.rs:83-84, 87-91, 113-116, 192-207) | En vivo. El ancla λ/μ̂ → 1 + α/β = 1,6 es coherente con un μ̂ que converge a la tasa total observada (:87-91, :169-176), y el test `qo_535_mu_empirico_es_escala_libre` lo afirma (:421-466). μ̂ es una tasa de conteo: los pesos 1 − e^{−dt/60} cancelan el 1/dt (:195-202). Por lectura, dos sesgos bajan μ̂ en ráfagas: el tope `MU_INST_MAX` = 200 (:116, :195) y los dt = 0, que no actualizan μ̂ (:192) aunque el evento entra en λ (:207). Magnitud sin medir | MLE/EM por moneda; luego Hawkes marcado compra/venta | Poisson irregular, con trades en el mismo ms: λ/μ̂ ≈ `STEADY_STATE_RATIO` y `excitacion_hawkes_norm` ≈ 0. Hawkes simulado con n = 0,6: excitación > 0. Residuos reescalados: KS frente a Exp(1) | C-F2.4 |
| 12 | Contagio cruzado (hawkes_cross.rs:62-105) | Cuenta ventanas; toma el máximo sin corregir | Hawkes multivariante por MLE, o entropía de transferencia con surrogados | Pares desfasados circularmente: tasa de líderes ≤ α | C-F2.4 |
| 13 | Lead-lag y modulador de seguidoras (lead_lag.rs:204; contagion_modulator.rs:31) | Líderes fijos; el signo del descuento está por revisar | Hoffmann-Rosenbaum-Yoshida en todos los pares; usar el desfase como señal | Recupera un desfase plantado. IC del voto de la seguidora > 0 | C-F2.4 |
| 14 | Dependencia de cartera: equicorrelación, HY, cópula t, curl de Hodge, Marchenko-Pastur (correlation_guard.rs:184-757; copulas_store.rs; random_matrix.rs) | Candidata con EWMA; cópula de 2026-08; MP sin consumidor | Ledoit-Wolf o RMT a la escala τ, con riesgo real de la candidata; λ rodante con vigencia | El cuantil 95 cubre el stop-out conjunto en replay | C-F4 |
| 15 | Cramér-Lundberg (cramer_lundberg.rs:82-212; risk-engine/src/lib.rs:674) | Se divide por L_max | R_capital = R_nocional·C/N; después modulado por régimen | En replay con caída > m, el veto dispara antes | C-F4 |
| 16 | Consenso de 13 motores (voto_espectral.rs; signal-engine orchestrator.rs:500) | Colineal; argmax; convicción sin calibrar | Pronóstico por escala y selección por habilidad; calibración isotónica | IC de v_dom > z crudo de la mejor escala (p < 0,01). ECE < 0,05 | C-F3 |
| 17 | Habilidad con e-proceso, Bonferroni 8320 y piso 0,15 (skill_motores.rs:38-40) | Distribución de pesos sin medir | FDR online (e-BH, SAFFRON); semivida física por escala | Motor plantado con IC 0,05 despega. Ruido: FDR ≤ α | C-F3 |
| 18 | Consejo de 11 asientos (consejo_seniors.rs:984-1507) | Aprende sólo de ejecutadas; asientos colineales | Meta-etiquetado de aprobadas y vetadas con contrafactual | AUC OOS > 0,55. EV aprobadas − vetadas con IC que excluye 0. Bloqueada por datos externos | C-F3 |
| 19 | Conducta de masas: Kahneman-Tversky, LS ratio, ballenas, spoofing (prospect_theory.rs; microstructure.rs) | En vivo sin sombra; doble conteo | Presión de disposición frente a un precio de referencia estimado del OI; modelo de cancelación con nulo; sombra | Signo estable OOS. Precisión ≥ 0,6 en episodios marcados. Si no, retirar. Bloqueada por datos externos | C-F2.5 |
| 20 | OU sobre la base futuro-spot y «VECM» (stat_arb.rs; vecm_arbitrage.rs:175) | Una sola pata. El VECM es base/ATR | OU por MLE con ADF/KPSS y dos patas; Johansen real o retirar | Vida media estable entre mitades. P&L neto > 0 | C-F3 |
| 21 | Stouffer de momento adverso (lib.rs:5048-5061) | Supone independencia | z = 1ᵀz/√(1ᵀΣ1) con Σ estimada | Tasa de veto en ruido = α | C-F3 |
| 22 | VPIN de cubo único como veto «causal» (math_kernels.rs:150-245; lib.rs:7576-7577) | Regla del tick, no lado real | BV-VPIN con `is_buyer_maker`, renombrado toxicidad | IC > 0 con \|retorno\| futuro, mayor que la persistencia | C-F2.3 |
| 23 | GBDT con triple barrera fija (train_forest.rs:1391-1392, 1082) | En vivo; 1−p para cortos | Meta-etiquetado con las barreras de la orden por τ; un modelo por lado | AUC OOS > 0,55 y EV neto > 0, purgado por τ. Bloqueada por datos externos | C-F7 |
| 24 | Red DarkAlpha (auto_trainer_daemon.rs:110-124) | Etiqueta del lado; sólo opina en BTC | Etiqueta direccional de primer toque en todos los símbolos, o GBDT | Exactitud OOS > base y > bosque. Bloqueada por datos externos | C-F7 |
| 25 | Bosque online → `forest6_prob` (online_random_forest.rs; lib.rs:6785-6809) | Probabilidad sin lado; dos esquemas de features | Boosting por moneda con etiqueta de primer toque, calibrado | Log-loss OOS < entropía de la base. Bloqueada por datos externos | C-F7 |
| 26 | Platt (12 h) y conformal ACI (calibration.rs; conformal.rs) | Aprende sólo de lo ejecutado; cobertura sin medir | Beta o isotónica con edad por dato y contrafactuales; Mondrian por τ | Brier/ECE < Platt. Cobertura ±2 % | C-F2.2 |
| 27 | Ensamble Hedge (ensemble.rs:118-333) | Etiqueta de vela de 1 min; eta fija | AdaHedge con log-loss sobre la etiqueta de primer toque | Regret O(√T) en replay | C-F7 |
| 28 | «PPO», hebbiano y mezclas fijas (online_ppo.rs; perceptron_gate.rs; lib.rs:5288-5362) | Estado del cierre; peso global | Bandido contextual por moneda; stacking por log-loss | Evaluación off-policy doblemente robusta > uniforme, con IC | C-F7 |
| 29 | Kelly por PF, envolvente y matriz de apalancamiento (kelly.rs; kelly_envelope.rs; leverage_matrix.rs) | Espacio de margen; Kelly dos veces; saltos de modo | ½ Kelly bayesiano en riesgo al stop (D1); L como consecuencia del margen | Riesgo realizado = objetivo ±10 %. Crecimiento log OOS ≥ actual (bloqueado por datos externos) | C-F4 |
| 30 | Prior del acierto 0,55 con k0 = 10 (evidence.rs) | Literal | Beta-binomial del pool (James-Stein) | Log-loss prequencial menor | C-F2.2 |
| 31 | Ruina por racha y axioma del 25 % (ruin.rs:46-99) | Sin simular con pérdidas reales | Monte Carlo o CVaR con pérdidas medidas y dependencia | Frecuencia de caer bajo S·C ≤ la prometida | C-F4 |
| 32 | d* = 0,632, kill por EWMA de t y auditor de deriva (drawdown.rs; online_daemon.rs:1305-1347; drift_auditor.rs) | Tres criterios distintos; falsos disparos sin medir | Un e-proceso o CUSUM sobre R ejecutados, con ARL0 calibrado | Falsos disparos ≤ α. Detección en ≤ N cierres | C-F4 |
| 33 | Aptitud ln(K_f/K_0) − λ·DD², λ = 4 ln 2 (fitness.rs:79-103) | Sin validez predictiva medida | Cota inferior del crecimiento log por tiempo, o restricción de CVaR | Correlación de rango (t, t+1) > 0 con p < 0,05 en ≥ 30 rondas. Exige promociones; sustituto en replay más abajo | C-F7 |
| 34 | DSR acumulado y cosecha best-of-9 (selection_stats.rs; random_forest.rs) | Dos rigores; acumulación sin límite | PBO/CSCV o SPA con las pruebas que comparten datos; un contrato | ≤ 5 % de promociones falsas en nulo. Potencia > 50 % | C-F7 |
| 35 | Examen walk-forward del demonio (online_daemon.rs:430-735) | Slots reales, fuga, interpolación lineal | CPCV purgado multiactivo simultáneo, aplanando en la frontera | IC entre aptitud del examen y retorno post-promoción > 0. Exige operar; sustituto en replay más abajo | C-F7 |
| 36 | Ville como vigilante de rollback (ville_e_process.rs; online_daemon.rs:1205, 1388) | Apuesta a media positiva | E-proceso de deriva negativa, o GLR-CUSUM | Rollback falso ≤ α. Potencia > 80 % ante −0,2 R | C-F2.2 |
| 37 | Mutación isótropa frente a CMA-ES (genome.rs:1722-1790; cma_es.rs) | CMA-ES real sin consumidor | CMA-ES o NES con covarianza persistida | Más mejora de aptitud por segundo de CPU | C-F7 |
| 38 | Libro contrafactual de vetos (shadow_ledger.rs) | Sin consumidor | Cablear QS-R4b sólo en sombra. Des-absorber EV, consejo y calibrador después del paso 2 de §5 (etiquetas) | Por fuente: R bloqueado < 0 con IC 95 % | C-F4 |
| 39 | Física de llenado y deslizamiento √t (reality_physics.rs; tp_sl.rs:176-282) | Sin calibrar; dos leyes distintas | Regresión con fills reales | Sesgo no significativo. MAE < 1 pb. Bloqueada por datos externos (fills reales) | C-F5 |
| 40 | Universo y ranking (dynamic_ranker.rs:150-178; bootloader.rs:317-350) | Fijo; el ranking ignora el edge | Selección por edge neto / (fricción + σ a τ) | EV neto OOS > top por volumen. Bloqueada por datos externos | C-F6 |
| 41 | Curvas de genes por horizonte con dos anclas, 30 s y 12 h (horizon_policy.rs:57-64; genome.rs:102-114, 308-355) | En vivo. Cada curva es la ley de potencia que pasa por los valores del gen en las dos anclas (temporal_spectrum.rs:1289-1299), y la τ dominante se recorta a [30 s, 12 h] (temporal_spectrum.rs:678-680) | Lo decide la prueba | Sensibilidad T-1 por tramo de τ | C-F7 |
| 42 | Cuantiles P² adaptativos (adaptive_quantiles.rs:198-241) | En vivo. Fijan los umbrales p80 de OFI y OBI y el p85 holístico (lib.rs:2787, 2797, 4488, 4908-4915). Tasa de disparo sin medir | Lo decide la prueba | Tasa de disparo en el nominal ±2 % | C-F2.2 |
| 43 | Convicción por rama con cota de Wilson (`conviccion_de_rama`, lib.rs:141-150) | En vivo en todas las ramas salvo la 12 (lib.rs:5652-6586). La rama 12 (impulso de flujo, lib.rs:6244-6298) conserva la confianza de `FlowImpulseEngine`, tanh(2·coherencia·decaimiento por entropía) (signal-engine flow_impulse.rs:158), sin cota de Wilson. Su resultado se registra al cerrar (lib.rs:3903-3907), pero nadie lo lee. Si la cota no decide, devuelve el piso por magnitud. Aguas abajo la vuelven a subir el impulso ML de `puertas_del_continuo`, ×(1 + 0,5·ml) hasta 0,99 (lib.rs:1613-1614), y el multiplicador espectral, hasta ×1,35 (lib.rs:7034, 7057) | La convicción entra una sola vez, también en la rama 12; los impulsos posteriores no la suben | Ninguna rama con cota superior < ½ supera `min_confidence` en replay | C-F3 |
| 44 | Descomposición de Helmholtz-Hodge del flujo cruzado (hodge_flow.rs:219-249) | Entra al consejo (lib.rs:5220-5231; consejo_seniors.rs:407). El «flujo» que se descompone es un producto antisimétrico instantáneo de OFI y tanh(100·retorno) de cada par (hodge_flow.rs:245), no un flujo medido entre activos | Pasar el nulo por surrogados de C-F2, o retirar | `curl_share` con datos reales fuera del nulo de surrogados barajados. Si no, retirar. Bloqueada por datos externos | C-F2.3 |
| 45 | Probabilidad de stop del pre-examen del demonio, por puente browniano (online_daemon.rs:1706-1715) | Condiciona al retorno de la barra, con σ de las barras y un piso de 5 pb (:1709). No es la p de la fila 5 | Una sola p de primer toque, la de la fila 5 | Brier de P(stop) frente a los toques del replay, como en la fila 5 | C-F7 |

**Pruebas bloqueadas por datos externos.** Los tapes reales viven en el PC del operador, desde la nube Binance responde 403 y no hay fills reales mientras no se autorice operar.
- Necesitan tape o fills reales: la parte de tape real de #1; #2, #3, #9, #18, #19, #23, #24, #25, #39, #40 y #44; y el crecimiento log OOS de #29. La parte sintética de #1 (ruido y OU) y las pruebas de replay de esas filas sí se pueden cerrar.
- #33 y #35 exigen promociones o retornos post-promoción en vivo, que son imposibles mientras no se autorice operar. Sustitutos en replay, sin promover nada:
  - #33: correlación de rango entre la aptitud de cada genoma en dos ventanas consecutivas y sin solape del mismo replay;
  - #35: IC entre la aptitud del examen y el retorno del mismo genoma en la ventana siguiente del replay.
- Ninguna fase espera a estas pruebas. C-F2.3, C-F2.5, C-F3, C-F4, C-F5, C-F6 y C-F7 pueden cerrarse con sus pruebas sintéticas y de replay, y su recibo deja abiertas las bloqueadas, con ese estado.

## 4. Literales y sesgos

**Método**

1. **Inventario.** Barrido con `rg` de literales numéricos y `const` en el código no de test de los crates del camino vivo. Se produce un TSV con path:línea, valor, símbolo y lector. El recuento total está sin medir.
2. **Clasificación** en cinco clases:
   - **FIS**: constante física o matemática con fuente (√(8/π), ln 2, z = 1,645).
   - **EXC**: regla del exchange (tick, min_notional, brackets, fees). Se lee del API de la cuenta.
   - **MED**: debe medirse. Lleva estimador, error y conducta sin dato, que es «no emitir», nunca un valor inventado.
   - **GEN**: debe ser gen. Va al genoma y necesita sensibilidad T-1 > 0; un gen inerte se retira o se conecta. Retirar o conectar genes cambia el vector de 144 genes que comprueba `validate` (genome_store.rs:248), así que espera a GENOME-GATE (LC-F7-16).
   - **DER**: debe derivarse con una fórmula y un test. Por ejemplo p* = (SL + f)/(TP + SL), o d* de la meta. La política del dueño es un caso de DER y se escribe en un ADR.
3. **Destino.** Un literal sólo queda en el código si es FIS o EXC con fuente, o política escrita en un ADR. Cada cambio tiene su test y pasa T-1. El T-1 corre con 1 000 USD y sin host (t1_cobertura_genetica.rs:50-52), así que no ve los literales del régimen de 13 USD ni los del host. Para esos, la puerta es un fixture de 13 USD y tests del host, o `bt_vivo_parity_audit`.

**Literales principales**

| Dónde | Valor | Efecto | Clase | Fase |
|---|---|---|---|---|
| train_forest.rs:1391-1392, 1082 | TP 0.0036, SL 0.0018, 300 000 ms | Una sola etiqueta del bosque para toda τ y ambos lados | DER | C-F7 |
| risk-engine/src/lib.rs:674 | r_raw / max_exchange_leverage | Lundberg L_max veces más laxo | DER | C-F4 |
| risk-engine/src/lib.rs:972 | 0.88 | Veto de primer toque sin pasar por el EV | DER | C-F4 |
| risk-engine/src/lib.rs:1136, 1149-1156, 1188; lib.rs:7764-7770 | trade_count < 5 en el riesgo y trade_count == 0 en el núcleo; 0.55·TP − 0.45·SL; prior 0.55, k0 10 | Dos sondas optimistas, que vuelven tras cada reinicio porque `trade_count` no se persiste; EV inflado con pocos datos | GEN (D4) + MED | C-F4 |
| ruin.rs:69-82; kelly_envelope.rs:27-29 | axioma 0.25; SURVIVAL_FLOOR 0.05; HORIZON 200; q 0.60 | Hasta el 25 % en riesgo por evento | DER (ADR) | C-F4 |
| risk-engine/src/lib.rs:1323 | (0.25·C).clamp(1.20, 2.60) USD | Límite en dólares anclado a 13 USD | DER | C-F4 |
| risk-engine/src/lib.rs:1075-1081 | base_micro_ratio 0.66/0.62; anchos 0.10/0.30 | En micro, la confianza mínima sube hasta ×1,0645, y la relajan la coherencia y la entropía con anchos fijos | DER | C-F4 |
| risk-engine/src/lib.rs:650, 652 | tope/8 | Sin riesgo medido, la candidata y sus pares cuentan como tope/8 | MED | C-F4 |
| risk-engine/src/lib.rs:1170-1175, 1206, 1268-1270 | 1.05 + 0.20·escasez; 0.035; 5 + 1.5·smoothstep; anclas 5×/50× | Margen sobre el EV y techo de L heredados | DER | C-F4 |
| leverage_matrix.rs:165, 251, 272-281, 318 | 0.40 + 0.60·conf; bono de Hurst [0.5, 2]; ×0.70 en slow | Con H > 0,5 el riesgo se duplica | retirar (D1) | C-F4 |
| kelly.rs:86-94, 114 | modos supervivencia/expansión; piso kelly_clamp_min | Salto de fracción; piso inventado | DER | C-F4 |
| risk-engine/src/lib.rs:521-528, 958 | sl_at_tau(30 s)/sl_at_tau(τ); √(30 000/τ) | Ancla scalp de 30 s | DER | C-F4 |
| state.rs:394-399; risk-engine/src/lib.rs:585, 1099 | ±0.15 por cierre sin decaimiento; clamps 0.60-1.40 y 0.75-1.35 | Tamaño ±40 % según rachas | MED | C-F4 |
| position.rs:731, 841-843, 968 | 3 ranuras; Δ 0.80/0.60; pirámide 0.0028 | Apilado sin riesgo agregado | DER | C-F4 |
| risk orchestrator.rs:178-186, 239 | p_crash ≥ 0.90; 0.25; 0.50; ≥ 0.05 | Veto binario de largos con p calculada sólo con BTC | MED | C-F4 |
| booktick_replay.rs:714, 716-721, 756 | suelo de stop 0.0015; respaldo scalp_sl_base ⊔ 1.5·ATR; τ por defecto 30 000 ms | El replay que usa el examen del demonio (online_daemon.rs:571) mide con stops y τ heredados del scalp. Zona Sol: va como pedido | DER | C-F4 |
| god_engine.rs:4026-4072, 4122, 4140-4149 | z de 1.64 a 0.85; n < 30; C ≤ 50 ⇒ 5×; 0.05·kelly; freno 1.25/0.4; L 1..20 | Envolvente menos conservadora en micro, con escalones | FIS (z) + DER | C-F5 |
| god_engine.rs:4439 | base + 0.20·ATR ∈ [5, 35] pb | La IOC tolera más deslizamiento que la fricción del gate | DER | C-F5 |
| envio.rs:24-31 | tope 20; 0.85/0.80/0.95 | Techo de envío y adaptación del margen con múltiplos fijos | DER (ADR) | C-F5 |
| god-engine-core trailing.rs:44, 185, 193-199, 215 | 1.5/2.5/3.5/4.5 ATR; be 16-21 pb; TP 0.012; SL 5 % | Escalera discreta | DER | C-F5 |
| dynamic_ranker.rs:94, 169, 236-251 | 1e6 USD; 15 %/día; L 125/75/50 por nombre; VIP0 | Techo de L y Lundberg sin datos del exchange | EXC | C-F6 |
| stateful_engine.rs:850-853, 876-881, 1024-1029 | trinquete; 0.85/0.15 con 10 pb frente a 0.8/0.2 con 5 pb | ATR sesgado al alza y distinto en el arranque | DER | C-F4 |
| lib.rs:4422-4427 | suelos 0.0001/0.0005/0.0020 en precio; ×7.746 | ATR horario publicado como de 1 min | DER | C-F4 |
| lib.rs:8169-8174, 8491-8492 | 2^((H−0.5)/0.02); Hurst 0.45/0.52 | Duración hipersensible; cortes de régimen | DER | C-F3 |
| lib.rs:1544-1655, 7733-7793 | boost ×(1 + 0.5·ml) hasta 0.99; veto −0.80; lift (thr−0.5).clamp(0.02, 0.25) | Puerta ML con coeficientes inventados | DER | C-F3 |
| lib.rs:6899-7113, 7815-7830; spectral_regime.rs:121 | marea 0.60/0.40; total (0.65, 1.35); ×(1 − 0.95·flux); pesos 0.35/0.30/0.20/0.15 | Acondicionamiento y margen con pesos fijos | MED | C-F3 |
| lib.rs:3340-3562 | buffers 0.0018-0.003; 0.00145; 0.0028; 720-3600 s; 480 s-86,4 ks | Salidas en fracciones y tiempos fijos | DER | C-F3 |
| lib.rs:3504-3507 | VPIN > 0.85 y OFI adverso; pnl ≤ −max(0.75·SL, 3·fee) | Cierre forzado por toxicidad con un umbral de VPIN fijo | MED + DER | C-F3 |
| lib.rs:6036-6048 | macro ±0.00010; secular ±0.0025 | Las ramas de reversión 7-10 sólo operan dentro de un rango de tendencia fijo. Zona de Línea E: va como pedido | MED | C-F3 |
| lib.rs:5288-5362, 6131, 6232, 4549-4553; signal-engine orchestrator.rs:500 | 0.70/0.30; 0.60/0.40; clamp(0.55, 0.85); clamp(0.50, 0.95); 0.5-2.0; 0.70 + 0.30·coh | Suelos de confianza y mezclas fijas | MED | C-F3 |
| lib.rs:476-507, 6398-6483 | FUSED 0.38/0.22; 0.50 + 0.35·coh; ×50; 0.51 | Confianza fabricada en las ramas 13 y 15 mientras su cota de Wilson no decide (lib.rs:141-150) | DER | C-F3 |
| consejo_seniors.rs:499-500, 548, 698, 914, 984-989, 1494-1507; lib.rs:7577 | ATR 0.0010/0.0040 «scalp 1m»; 0.85; DD 0.95 − 0.10·s; VPIN 0.92; causal 0.75; pesos 0.5-2 con 5 cuentas | Consejo con umbrales fijos | MED | C-F3 |
| lib.rs:7586; stat_arb.rs:10 | 1 138 000 ms (1 138 419,6 en stat_arb.rs) | Horizonte por defecto sin dato: el centro geométrico de las anclas (stat_arb.rs:6-10) | DER (no emitir) | C-F3 |
| skill_motores.rs:38-40 | M/α 8320; piso 0.15; olvido 1/64 | Ponderación inerte | DER | C-F3 |
| hawkes_bessel.rs:83-84, 91, 116, 192-195 | α 0.3, β 0.5; ancla 1.6 = 1 + α/β; tope 1/dt ≤ 200; dt = 0 no actualiza μ̂ | Por lectura, en ráfagas μ̂ baja y λ/μ̂ sube, así que la excitación sube (`excitacion_hawkes_norm` crece con λ/μ̂, :100-107). Magnitud sin medir | MED | C-F2.4 |
| lead_lag.rs:204; hawkes_cross.rs:105; contagion_modulator.rs:31 | BTC 0.60 / ETH 0.40; z > 3; 0.30·m/(m+5) | Líderes fijos; descuento de signo dudoso | MED | C-F2.4 |
| genome.rs:2412-2458 | WR 0.40/0.55; fricción 0.65; fee 0.001; SL 0.005 | `suelo_tp_sl` domina el embudo | MED | C-F7 |
| online_daemon.rs:53, 1215-1231, 1335-1343, 1520, 1641; random_forest.rs:11 | 30; t ≤ −2; −1.5/0.1/25; vol·50; Kelly 0.05-0.50; 15 | Vigilantes y búsqueda sin nivel | DER | C-F7 |
| online_daemon.rs:1130-1142 | riesgo_por_operacion → scalp_sl_base → 0.02 | Sin riesgo medido, el vigilante mide los R con la unidad heredada del scalp | MED | C-F7 |
| online_daemon.rs:1676 frente a :206-208 | 0.6 en el pre-examen; 0.5 en el juez | Dos fronteras distintas entre entrenamiento y validación | DER | C-F7 |
| online_daemon.rs:540-556, 1039-1047 | features omni fabricadas; ML 0.60/0.40 por archivo | Mundo inventado; cambio de umbrales sin armado | retirar | C-F7 |
| fitness.rs:79 | λ = 4 ln 2 | Aversión al riesgo de toda promoción | DER (ADR) | C-F7 |
| god_engine.rs:2986-2989, 3788-3792, 2416; memory_audit.rs:38, 45 | 0.05; 10 observaciones; sombra = 0.95·real; 3 cierres; RAM 0.85/0.70 | Vetos globales sin incertidumbre | MED | C-F4/C-F8 |

**Sesgos sistemáticos que se buscan en todas las fases**
- **Selección.** Calibrador, consejo, Kelly y epigenética aprenden sólo de lo ejecutado.
- **Comparaciones múltiples.**
  - Por exceso de falsos positivos: argmax sobre 32 escalas; máximo de lags con z > 3; best-of-9.
  - Por exceso de corrección: el DSR acumula pruebas sin límite.
- **Estimador de tasa de Hawkes.** μ̂ no es E[1/dt], sino una tasa de conteo (hawkes_bessel.rs:195-202). Los sesgos que muestra el código son el tope de 200/s y los dt = 0 que no actualizan μ̂ (:192-195); los dos bajan μ̂ en ráfagas. Magnitud sin medir.
- **Centrado en BTC.**
  - Régimen calculado sólo con BTC (lib.rs:2516).
  - Líderes BTC/ETH fijos.
  - DarkAlpha sólo en BTC.
  - `coin_id 0` tratado como BTC (lib.rs:1525, 2516).
- **Anclaje a 13 USD.** 1,20-2,60 USD; C ≤ 50 ⇒ 5×.
- **Herencia scalp/swing.**
  - Anclas de 30 s y bandas ATR «scalp 1m».
  - Respaldo `scalp_sl_base` en el vigilante del demonio y en el replay del examen.
  - Nichos de scalping y YAML SCALPING/SWING.
  - Alias que siguen vivos.
- **Etiqueta equivocada.** Lado en vez de dirección; neto en vez de bruto.
- **Supervivencia de modelos.** Sólo los símbolos con `_MOTOR` pasan el ML-GATE. Los demás sólo abren la sonda de arranque frío, que vuelve tras cada reinicio (lib.rs:7764-7770).
- **Asimetría largo/corto.** El recorte por `crash_flux` ya tiene espejo para los cortos (risk orchestrator.rs:171, :175; lib.rs:7828). Sólo el veto binario de caída bloquea largos sin espejo para cortos (risk orchestrator.rs:185-191, D-403).
- **Fuga.** Entrenamiento→validación en el examen. `forensics.rs` escribe `.forensic_violation` (forensics.rs:115), y el demonio lo lee y cambia umbrales (online_daemon.rs:1035-1051).
## 5. Autoadaptación

En §5 y §6, «lib.rs» y «core lib.rs» son crates/god-engine-core/src/lib.rs, y «risk lib.rs» es crates/risk-engine/src/lib.rs. Toda cita es de aa716edf salvo que nombre otra revisión.

**Lazos absorbentes.** Una vez disparados, impiden obtener los datos que los liberarían.

| Lazo | Dónde |
|---|---|
| DriftRecovery: veta todas las entradas del núcleo hasta 10 observaciones válidas seguidas, que sólo llegan con cierres que el propio veto impide | god_engine.rs:2988-2990, :3825-3827; drift_auditor.rs:46-57; lib.rs:2953 |
| Pico no persistido y veto de drawdown sin recuperación dentro de la sesión, en el riesgo y en el host | risk-engine/src/lib.rs:284-286, :336-341; god_engine.rs:1682, :1704-1711 |
| Latch global del kill-switch que arma el demonio por pérdida constante o por una EWMA de t < −1,5, sin rearme automático | online_daemon.rs:1222, :1235, :1324, :1335-1350 |
| DSR con `cumulative_trials` monótono: la evolución se congela | online_daemon.rs:281-283, :1888 |
| Tras un rollback, «intervención manual» | online_daemon.rs:380-385, :1278-1284 |
| Enfriamiento base·2^racha con tope de 12 h. La racha cruda sólo vuelve a 0 con un cierre no perdedor. El tiempo amortiza niveles, así que no bloquea del todo: con la racha alta admite como mucho una entrada cada 12 h | stateful_engine.rs:298-310, :477-499, :521-536 |
| Consejo, calibrador y EV con p calibrada aprenden sólo de ejecutadas | consejo_seniors.rs:1057 y :1390-1399, que se llama al cerrar (lib.rs:4159-4164); calibrador en lib.rs:3983-3990; EV en risk-engine/src/lib.rs:1137-1158 y :1184-1192 |

**Lazos que adaptan con la señal equivocada:**
- DarkAlpha (etiqueta del lado);
- `forest6_prob` (sin lado);
- ensamble (vela de 1 minuto);
- PPO (estado del cierre);
- `riesgo_por_operacion` (cuenta órdenes no enviadas);
- Ville (apuesta a favor);
- bosque online (dos esquemas, claves globales);
- hebbiano global;
- reconciliación (media simple; salida igual a 0);
- payoff de la envolvente (USD, doble EWMA);
- sesgo epigenético sin decaimiento.

**Lazos abiertos.** Miden, pero no gobiernan:
- ganancia epigenética espectral. Se escribe al cerrar (temporal_spectrum.rs:772-793, llamada en lib.rs:3880), pero `fused_score` usa `pesos_observables` (:608, :1408-1424), que no la leen. Entra por `fusion_weight` (:251-256), que usan los lectores de `pesos_espectrales` (:1389-1396), `continuous_energy_density` (temporal_spectrum.rs:899) y lib.rs:4506. El núcleo usa `continuous_energy_density` para arbitrar las dos bandas en el mismo armónico: elige la duración si van del mismo lado (lib.rs:6674-6675) y la intención que pasa, con su lado, si se oponen (:6712-6713). Por esas vías sí llega a decisiones vivas; lo que no gobierna es `fused_score`;
- Lundberg ÷ L_max;
- BOCPD sin actuador;
- LMS online (D-693);
- `drift_real_vs_control_pct`, que sólo se escribe (god_engine.rs:3815-3818);
- umbrales del bosque online que nunca se identifican;
- ledger hot-swap que sólo se escribe;
- trayectoria, sólo telemetría;
- timestamps del registro, que no se actualizan (omniscient-registry/src/lib.rs:65, :73-76).

**Congelados que deberían tener reloj:**
- bosques: ADR-0008, y el script apunta a `_SCALP`;
- cópulas: ADR-0009 vencida (el manifest es de `mes` 2026-08, y ADR-0009:13 dice que ese campo es la vigencia);
- genomas demo y prod idénticos, en la generación 1;
- universo fijo de 26;
- α y β de Hawkes;
- `CouncilParams`;
- símplex de régimen;
- anclas de `capital_regime`;
- dimensiones cross-exchange en cero.

**Orden de cierre.** Cada paso lleva su test.

1. **Liberar los absorbentes que bloquean sin evidencia.**
   - DriftRecovery se libera por el universo de control emparejado o por tiempo.
   - Pico persistido y una sola semántica de drawdown con d*. La persistencia mínima del pico se define en C-F4, con un test de reinicio, y no espera a `learned_state` de C-F8 (LC-F4-10).
   - Rearme documentado del latch. Mientras un estadístico descriptivo del demonio pueda armarlo, Línea C no propone el latch permanente con rearme manual (§1.4, LC-F5-10).
   - El enfriamiento se rearma por evidencia, no sólo ganando.
   - Prueba: con el veto activo y sin posiciones, el sistema vuelve a admitir cuando se cumple el criterio declarado.
2. **Corregir etiquetas y unidades de los aprendices que ya realimentan.** Cerrar lazos con etiquetas malas amplifica el error. Prueba: cada aprendiz se puntúa con la etiqueta de lo que predice, y un test sintético con dirección conocida da exactitud > base.
3. **Cablear el libro contrafactual (QS-R4b)** y alimentar con él calibrador, consejo y EV.
   - El libro (`shadow_ledger.rs`) pasa a C-F4, porque también lo necesitan D1 en sombra y des-absorber el EV.
   - Hasta cerrar el paso 2, el gancho de QS-R4b sólo registra en sombra (§1.3, pedido 3).
   - Prueba: por fuente de veto, la R bloqueada tiene IC; ningún lazo depende sólo de lo ejecutado.
4. **Cerrar los lazos abiertos sólo con evidencia OOS y en sombra primero** (G-TEO punto 5, plan sinc. §30.1 :1129-1132). Aquí entran también los lectores de conducta de masas: LS ratio, funding, OI, liquidaciones, ballenas.
   - Prueba: la decisión con el lazo cerrado mejora el EV neto OOS frente a la sombra.
   - Esa prueba necesita tape real, así que hoy está bloqueada por datos externos: los tapes viven en el PC del operador y desde la nube Binance responde 403. Mientras tanto, el lazo sigue en sombra.
5. **Dar reloj a los congelados.** Revalidación de bosques, cópulas con vigencia comprobada al cargar, Hawkes por MLE, universo por edge neto. Prueba: un artefacto vencido no se carga sin aviso.
6. **Reabrir la evolución** con un solo contrato de promoción. Prueba: ≤ 5 % de promociones falsas en nulo y potencia medida con edge plantado.

## 6. Hallazgos sospechosos para verificar

Ninguno está verificado salvo los marcados «confirmado por lectura», y a ésos todavía les falta el test. Ningún hallazgo se arregla sin un test que falle antes. Las citas son de aa716edf. Esta revisión añade cuatro ids: LC-F4-19, LC-F5-10, LC-F9-06 y LC-F9-07.

**C-F1**
- **LC-F1-01** El registro convierte NaN y ±Inf en 0, cae al valor global de la última moneda y no actualiza el timestamp (omniscient-registry/src/lib.rs:73-76, 244-251, 48, 65).
  - Test RED: NaN debe leerse como desconocido; A no debe devolver el valor de B.
- **LC-F1-02** `banda_paga_friccion` (lib.rs:1496) supone que `coin_id 0` es BTC (lib.rs:1525-1529).
  - Test RED: universo permutado.
- **LC-F1-03** `rust-toolchain.toml:2` sigue con `nightly` sin fecha, mientras la CI ya fija `nightly-2026-06-30` (replay-contracts.yml:45-52; unit-suites.yml:68-74). Confirmado por lectura.
  - Test de fuente: las tres cadenas coinciden.
- **LC-F1-04** No hay garantía de que cada `REJ_*` tenga entrada en `veto_registry`: `buscar()` (veto_registry.rs:382) sólo se usa en sus tests (:636).

**C-F2**
- **LC-F2-01** Pesos espectrales incoherentes entre lectores, y la escala anti-persistente pesa positivo (temporal_spectrum.rs:247-257, 597-640, 1389-1396).
- **LC-F2-02** La habilidad media autoriza todas las anclas de σ (spectral_tape.rs:750-764).
  - Test RED: un ancla con habilidad negativa no debe publicarse.
- **LC-F2-03** μ̂ de Hawkes. Por lectura es una tasa de conteo: los pesos 1−e^{−dt/60} ≈ dt/60 cancelan el 1/dt (hawkes_bessel.rs:195-202). El ancla λ/μ̂ → 1 + α/β = 1,6 es la que documenta el código (:87-91, :169-176) y la que afirma el test `qo_535_mu_empirico_es_escala_libre` (:421-466). El 2,5 sólo valdría si μ fuera la base exógena.
  - Quedan dos sesgos, y los dos bajan μ̂ y suben λ/μ̂ en ráfagas: el tope `MU_INST_MAX` = 200 recorta los dt < 5 ms (:116, :195), y con dt = 0 μ̂ no se actualiza (:192) aunque el evento entra en λ (:207). Magnitud sin medir.
  - Test RED: Poisson irregular, con varios trades en el mismo ms ⇒ λ/μ̂ ≈ `STEADY_STATE_RATIO` y `excitacion_hawkes_norm` ≈ 0. Control: Hawkes simulado con n = 0,6 ⇒ excitación > 0.
- **LC-F2-04** El contagio devuelve None con timestamps repetidos (hawkes_cross.rs:151-155) y compara ventanas distintas por moneda: cada serie son los últimos 256 ticks de su moneda (contagion_publisher.rs:33-43).
- **LC-F2-05** Bonferroni 8320 deja los pesos en el piso 0,15 (skill_motores.rs:38-39, :97).
  - Medición: distribución de pesos en un replay largo.
- **LC-F2-06** Yang-Mills ≈ 0 con retornos asíncronos y pesa hasta 0,20 en el consejo (yang_mills_gauge.rs:184-188; consejo_seniors.rs:462).
- **LC-F2-07** Ville apuesta a media positiva y decide por caída de la e-riqueza ≥ 50 % (ville_e_process.rs:101-119, 212-223; consumidor online_daemon.rs:1205-1209, 1388).
- **LC-F2-08** El LS ratio cuenta dos veces en EnteMercado y el clamp anula la amplificación de prospectos; no está en sombra (lib.rs:7607-7610, :7658-7664; consejo_seniors.rs:581-593, 615; prospect_theory.rs:180-184).

**C-F3**
- **LC-F3-01** ρ(τ) se publica tres veces con dos τ distintas, y el veto de grupo lee la última (lib.rs:1749 frente a :2840 y :3883; risk-engine/src/lib.rs:686). Confirmado por lectura.
  - Dueños: Qoder y Codex (MG05, C-2).
- **LC-F3-02** `atr_1m` ×7.746 y suelos en unidades de precio (lib.rs:4422-4427); el lector es coaxial_breakout.rs:162-165. Confirmado por lectura. Se arregla al empezar C-F4, junto con LC-F6-01 (§2).
- **LC-F3-03** 13 votos colineales y dominante por argmax.
  - Test RED: con ruido puro, la convicción supera el umbral.
- **LC-F3-04** La vida `tau_dom` que fija el orquestador (signal-engine/src/orchestrator.rs:576-587) se ignora en las ramas 11 y 14. La rama 11 usa `fast_duration_ms`, que sale de `micro_tau` (lib.rs:5388-5394, :6237), y la rama 14 usa `swing_duration_ms` (lib.rs:6381, :6566-6592).
- **LC-F3-05** Probabilidad del corto con barreras asimétricas. #713 y #714 sustituyeron 1−p por una proyección a través de la base del modelo: `normalized_directional_divergence` en el ML-GATE (lib.rs:7777-7785) y `p_win_directional` (calibration.rs:352-365, usada en lib.rs:3968, :4301 y :4821-4822). Las dos salen del modelo del largo; ninguna es P(TP del corto antes que SL) con las barreras del corto. Que esto cierre el hallazgo está sin verificar.
- **LC-F3-06** Stouffer con EMAs correlacionadas (lib.rs:5048-5061).
  - Test RED: en ruido, tasa de veto > α.
- **LC-F3-07** Clave global `microstructure_noise_variance` leída para la moneda en curso, con unidades mezcladas. El lector usa la clave global (lib.rs:2131-2135), que `set_reg` escribe para cada moneda junto a la suya (:4401-4407, :4536-4539): gana la última que escribió.
- **LC-F3-08** Peso hebbiano global. Al cerrar, el núcleo escribe `{sym}_hebbian_weight` (lib.rs:4206, :4223) y además la clave global `perceptron_hebbian_weight` (:4224-4226). El perceptrón pide `perceptron_hebbian_weight` escopada por moneda (perceptron_gate.rs:168-176), que nadie escribe, y cae al global, que es el de la última moneda que cerró.
- **LC-F3-09** La duración usa otro Hurst (`features[1]`, lib.rs:6339) con sensibilidad ×2 por ±0,02 (lib.rs:6745-6748, :8169-8174).
- **LC-F3-10** `break` tras el primer cierre: la segunda ranura espera al siguiente evento (lib.rs:4332-4333). Severidad: capital.
- **LC-F3-11** El veto «absoluto» por drawdown del consejo exige DD 0,85-0,95 (consejo_seniors.rs:698).
- **LC-F3-12** Asientos que heredan `intended_direction` (Volatilidad, Riesgo y EnteMercado: consejo_seniors.rs:503, :616, :707; documentado en :1017) y triple comprobación del lado (:1319-1331). Con `intended_direction = 0`, la de AGY no discrimina.
- **LC-F3-13** El descuento a las seguidoras puede tener el signo invertido (contagion_modulator.rs:31; lib.rs:7069-7077).
- **LC-F3-14** El «VECM» es un segundo voto sobre la misma base que StatArb, y los alias scalp/swing siguen vivos (vecm_arbitrage.rs:175; lib.rs:5131-5141; signal-engine/src/orchestrator.rs:308, 317).
- **LC-F3-15** Asignaciones de heap por tick y por cierre (lib.rs:4206, 4571-4573; consejo_seniors.rs:431, 623). Latencia: sin medir.
- **LC-F3-16** DarkAlpha sólo opina en BTCUSDT (lib.rs:4722, :4739), y BTC no tiene `_MOTOR`, así que el roster veta sus entradas. Aun así, decir que no participa en ninguna decisión es demasiado fuerte. Con la sonda de arranque frío (lib.rs:7764-7770), BTC puede abrir una sonda tras cada reinicio, y la opinión de DarkAlpha entra en el compuesto con el 60 % del peso residual (`w_nn`, :5357-5361).
- **LC-F3-17** Sin τ medida se usan 1 138 000 ms (lib.rs:7586).

**C-F4**
- **LC-F4-01** Lundberg ÷ L_max (risk-engine/src/lib.rs:673-674). Confirmado por lectura.
- **LC-F4-02** La envolvente del host sólo mueve el margen; la pérdida al stop no cambia (god_engine.rs:4093-4151; envio.rs:69-100; kelly_envelope.rs:332-340).
- **LC-F4-03** `riesgo_por_operacion` se escribe antes de los vetos del consejo, del ML-GATE y del host. Su único escritor está en `evaluate_single_intent` (risk-engine/src/lib.rs:1434-1446), que corre también en el backtest; el consejo y el ML-GATE vienen después (lib.rs:7669-7807). Lo leen risk-engine/src/lib.rs:324 y :646, god_engine.rs:1734 y online_daemon.rs:1131.
  - El arreglo no puede mover la escritura al host al confirmar el envío. El T-1, `booktick_replay`, el examen del demonio y ShadowForest no pasan por el host: lo dejarían en su valor inicial 0, y `drawdown_maximo` caería al gen.
  - La escritura va a un punto común del núcleo y del replay, después del consejo y del ML-GATE, con reversión si veta el host y su espejo en `live_envelope_gate`. Test en `bt_vivo_parity_audit`.
- **LC-F4-04** Primer toque mal escalado (risk-engine/src/lib.rs:952-974; publica en cada evaluación en :971). En sombra desde CL-100 (decisión del dueño): ya no veta, publica `p_hit_sl_first` y `r15_vetaria`; el modelo de la deriva sigue abierto.
  - Con deriva nula veta todo RR > 7,33.
  - Con SL 200 pb, TP 450 pb, ATR 0,10 %, coherencia −0,02 y τ 30 s, P ≈ 0,90 y veta.
- **LC-F4-05** Salto de la fracción de Kelly en la frontera de modo (kelly.rs:86-95).
- **LC-F4-06** El piso `kelly_clamp_min` se aplica dos veces (kelly.rs:114; risk-engine/src/lib.rs:411-414).
- **LC-F4-07** Genes inertes: `vol_mult` (risk-engine/src/lib.rs:739-767, pasado en :989 a leverage_matrix.rs:84, donde no se usa) y `kelly_bootstrap_cold` (:401-410). En C-F4 sólo se declaran; se retiran o se conectan después de GENOME-GATE (LC-F7-16), porque cambian el vector de 144 genes.
- **LC-F4-08** `risk_normalizer` usa la curva genómica y no el stop real (risk-engine/src/lib.rs:521-528).
- **LC-F4-09** Prior 0,55 y sonda optimista (risk-engine/src/lib.rs:1136, 1149-1157, 1188).
- **LC-F4-10** Dos picos y dos semánticas de drawdown entre host y riesgo. Cerrado (ADR-0017, decisión del dueño «Ambos»): CL-99 (un pico, guardado), CL-106 (freno duro único K(w, l)·r ≤ d*, nivel α en 200 operaciones) y CL-107 (freno de la estrategia, Ville sobre R).
  - #714 añadió `.min(d*)` al host (god_engine.rs:1729-1737) sin ningún test del host y sin el lerp por `micro_w` que aplica el riesgo (risk-engine/src/lib.rs:323-341).
  - El host usa un pico local que nace con el capital del arranque (god_engine.rs:1682, :1704-1711); el riesgo usa `peak_capital` (risk-engine/src/lib.rs:284-286). Ninguno se persiste.
  - El arreglo, en C-F4, incluye una persistencia mínima del pico con un test de reinicio.
- **LC-F4-11** DriftRecovery absorbente (god_engine.rs:2988-2990, 3825-3827; drift_auditor.rs:46-57; lib.rs:2953).
- **LC-F4-12** Sesgo epigenético tipo «hot hand» (state.rs:394-399; risk-engine/src/lib.rs:579-580).
- **LC-F4-13** Cópulas vencidas cargadas sin aviso (copulas_store.rs:32-63; ADR-0009). Manifiesto renovado con septiembre de 2026 (CL-105); el aviso de vencimiento al cargar sigue abierto.
- **LC-F4-14** RA-RUIN-F01: el helper de ruina deja pasar NaN y ±Inf.
- **LC-F4-15** La confianza entra varias veces en el tamaño (K-18/R-17).
- **LC-F4-16** Docs que contradicen su código: ruin.rs:24-29 y correlation_guard.rs:811-833 (fórmula del doc en :816 frente al código en :826-828).
- **LC-F4-17** La candidata entra al veto de grupo con la EWMA y no con su riesgo real (risk-engine/src/lib.rs:612-711; la candidata en :646-650).
- **LC-F4-18** `curl_share` infla ρ (correlation_guard.rs:690-701) aunque el publicador dice «no veta» (contagion_publisher.rs:79-87).
- **LC-F4-19** Dos sondas de arranque frío, las dos por moneda. Confirmado por lectura. Cerrado por CL-98 (D4): una sola puerta de sonda para la cartera, la sonda es la orden mínima.
  - En el riesgo, `es_fase_sonda` (`trade_count < 5`, risk-engine/src/lib.rs:1136). En el núcleo, `arranque_frio_sin_roster` (lib.rs:7764-7770, usada en :7792), que se arma con `trade_count == 0` y sin posición abierta en la moneda.
  - `trade_count` no se persiste: nace en 0 (quantum-arena/src/state.rs:48) y sólo sube al cerrar (lib.rs:4001). Tras cada reinicio, cada una de las 23 monedas sin `_MOTOR` puede volver a abrir una sonda: el universo es de 26 (bootloader.rs:320-347), y `models/` sólo tiene `_MOTOR` de ATOM, BNB y NEAR.
  - Test RED: una sola sonda en toda la cartera, también tras reiniciar (D4).

**C-F5**
- **LC-F5-01** `boot_lev` usa el margen de otra ranura (god_engine.rs:4039, 4064-4071).
- **LC-F5-02** `close_was_real` consume el flag de cualquier ranura (god_engine.rs:3677-3687, 3704-3714).
- **LC-F5-03** Fantasmas: salida = `entryPrice` = 0, el cierre se pierde para el aprendizaje y el win rate se calcula con media simple (reconciliation.rs:432-548).
- **LC-F5-04** Payoff con doble EWMA, en USD, con eps de 1e-3 USD (god_engine.rs:3521-3526; kelly_envelope.rs:209-226).
- **LC-F5-05** El veto de muro usa el nivel 0 (god_engine.rs:3156-3157, 3349).
- **LC-F5-06** Tope IOC de 5 a 35 pb frente a la fricción del gate (god_engine.rs:4439). Usa el ATR del arena, así que se mide después del arreglo del ATR al empezar C-F4.
- **LC-F5-07** `_is_mean_reversion_vortex` sin lector (god_engine.rs:4267).
- **LC-F5-08** Módulos de execution-engine sin consumidor; `ws_executor` es un stub siempre desconectado.
- **LC-F5-09** La meta supone maker; el código es taker (quantum-arena/src/config.rs:208; god_engine.rs:4256-4269).
- **LC-F5-10** Escritores de `kill_switch_active` que ADR-0015 no nombra. Confirmado por lectura.
  - El ADR nombra tres (ADR-0015:13-15): el núcleo (lib.rs:7122), el sistema inmune (god_engine.rs:1802) y X-009 (god_engine.rs:4757).
  - El demonio también arma ese latch, por pérdida constante (online_daemon.rs:1222, :1235, :1324) y por EWMA(t) < −1,5 (:1343). No depende del armado de la evolución (:737-738), así que vale también en prod.
  - El drenaje del apagado (CERT-M4-H02, god_engine.rs:4947-4950) también arma el latch antes de aplanar; ADR-0015 sólo menciona ese aplanado (:48-55), no su escritura del latch. Son ocho puntos de escritura.
  - Test de fuente: todo escritor de `kill_switch_active` figura en ADR-0015 con su criterio.

**C-F6**
- **LC-F6-01** ATR con trinquete y dos estimadores (stateful_engine.rs:850-853, :876-881 y :1024-1029). Se arregla al empezar C-F4, junto con LC-F3-02 (§2); el recibo del archivo sigue en C-F6.
- **LC-F6-02** `max_leverage` por nombre y fees VIP0; riesgo de rechazo -4028 (dynamic_ranker.rs:236-251).
- **LC-F6-03** Spread spot-futuro siempre 0 (spot_feed.rs:126-136; omni_multiplexer.rs:183).
- **LC-F6-04** Features en tiempo de eventos: FFT de 64, MFDFA 10/25/50, RSI14.
- **LC-F6-05** Universo fijo; `active_universe` sólo en tests (bootloader.rs:317-350).
- **LC-F6-06** Barreras fijas en `label_evidence` (label_evidence.rs:95-200).

**C-F7**
- **LC-F7-01** El examen se replaya en los slots reales del universo; la sonda se consume en entrenamiento (online_daemon.rs:430-449, 646-651; symbol_registry.rs:147-157).
  - Medición: operaciones por examen en demo.
- **LC-F7-02** ShadowForest promueve con 15 operaciones, best-of-9, 1 de cada 10 eventos y sin envolvente (random_forest.rs:11, 143-262; god_engine.rs:3422-3423, 4889-4923).
- **LC-F7-03** Fuga entre entrenamiento y validación; monedas congeladas mientras avanza el reloj (online_daemon.rs:670-705).
- **LC-F7-04** Interpolación lineal y features omni fabricadas (online_daemon.rs:507, 538-557, 622-623).
- **LC-F7-05** El bosque online recibe dos esquemas y lee claves globales: el del bus mmap (online_daemon.rs:925-940, que escribe lib.rs:3937-3944), el del registro por moneda (:1013-1025) y el del registro global (:1162-1174), que `set_reg` llena con la última moneda (lib.rs:4401-4403).
- **LC-F7-06** `forest6_prob` no tiene lado y se lee como dirección (online_random_forest.rs:273-276; lib.rs:6785-6808).
- **LC-F7-07** `cumulative_trials` monótono (online_daemon.rs:281-283, 1851, 1888-1903).
- **LC-F7-08** `.forensic_violation` cambia umbrales sin armado, también en prod (online_daemon.rs:1035-1051; forensics.rs:115).
- **LC-F7-09** Cargador de modelos sin hash; el script retira `_SCALP` mientras el motor usa `_MOTOR` (model_reload.rs:32-63; revalidate_roster.ps1:8-9, 50-52).
- **LC-F7-10** `live_envelope_gate` sólo dictamina sin ranura previa y su rollback cierra todas las ranuras (booktick_replay.rs:688-694, 830-848). Zona Sol.
- **LC-F7-11** DarkAlpha etiqueta el éxito del lado como «sube»: el emisor escribe el retorno neto de la operación (lib.rs:4046-4047, :4067) y el entrenador lo etiqueta 1,0 («Long») si supera 0,0005 (auto_trainer_daemon.rs:110-124). Confirmado por lectura.
- **LC-F7-12** El ensamble se puntúa con la vela de 1 min y opiniones viejas (lib.rs:1866-1898; ensemble.rs:238-259).
- **LC-F7-13** PPO con el estado del cierre (lib.rs:4253-4277).
- **LC-F7-14** `evolver` recorre el `process_tick` legado (evolver.rs:430) y puede promover (:674). `walkforward_evolver` no usa `process_tick`: evalúa con el binario `audit_forensic_backtest` (walkforward_evolver.rs:19-22). También puede promover (:385, 396), y a demo y prod con armado (:44-46).
- **LC-F7-15** El tracker del consejo y el calibrador sólo ven operaciones ejecutadas (lib.rs:4159-4164, :3983-3990; consejo_seniors.rs:1057).
- **LC-F7-16** `load_active` no valida (genome_store.rs:140-182, 248). Confirmado por lectura. GENOME-GATE va antes de retirar o conectar genes (§8).
- **LC-F7-17** Genomas demo y prod idénticos en la generación 1 (config_dir/genomes/demo/active.json y prod/active.json): no hay ninguna promoción viva registrada en el árbol.

**C-F8**
- **LC-F8-01** `darwin_approved.store(true)` incondicional (god_engine.rs:1292). Confirmado por lectura.
- **LC-F8-02** Profiler a 3 GHz; no hay benchmarks; latencia sin medir (profiler.rs:123-124).
- **LC-F8-03** Logs descartados bajo carga sin contador (macros.rs:71-82; atomic_telemetry.rs:41-47).
- **LC-F8-04** No hay snapshot del estado aprendido ni `DecisionRecord`.
- **LC-F8-05** `dns_optimizer.rs:24` lista dstream; Prometheus rasca un `:9091` que no existe (deployment/prometheus.yml:7-9).
- **LC-F8-06** `drift_real_vs_control_pct` sin lector (god_engine.rs:3815-3818).

**C-F9**
- **LC-F9-01** `launchers/EMERGENCY_SHUTDOWN.bat` escribe `STOP_TRADING.LOCK` en su directorio de trabajo (:13) y mata `python.exe` (:17). No detiene `god_engine`. Con ese archivo, el binario arma el latch y aplana, pero el proceso sigue vivo (god_engine.rs:1691, :1781-1834). Que el archivo caiga donde el binario lo busca (:5153-5163) está sin verificar. Severidad: capital.
- **LC-F9-02** `launchers/WANDB_SYNC_ALL.bat` fabrica un Sharpe al azar (:22) e imprime «PRODUCCION AUTORIZADA» (:27).
- **LC-F9-03** 14 lanzadores apuntan a archivos que no están en el árbol: diez de `launchers/` ejecutan un script de Python inexistente, y `HELP.bat`, `WANDB_SYNC_ALL.bat`, `LAUNCHER.bat` y `LAUNCH_PRODUCTION.bat` nombran, activan o copian algo que tampoco está. Recuento por lectura en aa716edf; la cifra depende del criterio, y el detalle está en C-F9. Además, `deployment/entrypoint.sh:20-24` ejecuta un `main.py` que no existe.
- **LC-F9-04** Nichos de scalping con anclas literales (continuous_evolution_backtest.rs:341-353, 424-437).
- **LC-F9-05** `net_multiplexer.rs` sin declarar; asignadores zero-copy muertos.
- **LC-F9-06** `LAUNCH_PRODUCTION.bat` anuncia «DINERO REAL ACTIVADO» (:42) y carga las claves de `.env`, pero lanza `god_engine.exe` sin `--force-live` (:46). Con la puerta del host (god_engine.rs:846-875) eso arranca en demo: por lectura, no llega a mainnet con un clic. Además copia `.env.template` (:11), que no existe; la plantilla versionada es `template.env`.
- **LC-F9-07** `run_god_engine.ps1` está en UTF-16, inyecta claves «shadow» que pisan las de `.env` y lanza `cargo run` sin redirigir la salida (:12), cuando .agents/AGENTS.md:27-29 exige `-RedirectStandardOutput logs/demo_vNN.log`.

**C-F10**
- **LC-F10-01** REV-1: `coin_id 999` sale por `REJ_FLAT_COIN` (risk-engine/src/lib.rs:250-251) y nunca llega a `REJ_SPEC` (:490), y el test sólo exige Flat (veto_logic_contracts.rs:134-136).
- **LC-F10-02** Suites que ninguna CI ejecuta. Sólo se compilan, con `check --workspace --all-targets` (replay-contracts.yml:51-52), y un grep en .github/workflows no las encuentra en ningún otro paso:
  - 7 de las 9 suites de evolution-engine; de ese crate sólo corren `--lib` y dos suites compiladas con `rustc` (replay-contracts.yml:68-70, 95-105);
  - storage-engine, con sus 3 suites y sus tests unitarios;
  - telemetry-server, telemetry-engine, omniscient-registry, phase-runner, os-guardian, flight-recorder, graph-architecture y graph-4d;
  - los tests de los binarios fuera de `train_forest` (god_engine.rs, feature_exporter.rs, train_dark_alpha.rs y continuous_evolution_backtest.rs); unit-suites sólo corre `--bin train_forest` (unit-suites.yml:54);
  - el T-1.
- **LC-F10-03** El T-1 mide `run_backtest_native`, no el juez vivo, y está fuera de CI. Llama con BTCUSDT y 1 000 USD (t1_cobertura_genetica.rs:50-52): no ejercita el régimen de 13 USD ni ejecuta el host (envolvente, IOC, kill-switch, `close_was_real`). Su encabezado dice que ejecuta el mismo `GodEngineCore` que producción (:11-13); eso vale para el núcleo, no para el host.
- **LC-F10-04** `forensics.rs:115` está dentro de un `#[test]` (:53-54) con efecto en disco.
- **LC-F10-05** Recuentos de tests citados sin su revisión. El plan sinc. dice «153/153 tests unitarios y 25/25 suites» de risk-engine (:1055).
  - risk-engine: 154 `#[test]` en `src/` en 95412dd7, 5338eda1 y aa716edf; en `tests/`, 29 archivos en 95412dd7 y 30 en 5338eda1 y en aa716edf.
  - god-engine-core: 170, 178 y 180 tests en `src/` en esas tres revisiones. Cada cifra incluye un `#[tokio::test]` (bootloader.rs); en `#[test]` son 169, 177 y 179.
  - Toda cifra citada nombra su revisión.
- **LC-F10-06** En ningún push de #710 a #714 cerró en verde «Replay contracts»; «Unit and contract suites» sí. Confirmado por lectura y por la API de GitHub.
  - El `<<<<<<< HEAD` que dejó #710 estuvo en docs/PLAN_MAESTRO_SINCRONIZACION.md:1057 desde 759f44ec hasta 0000eb8d. En aa716edf ya no queda ningún marcador.
  - «Replay contracts» falló en 759f44ec, en el paso de marcadores, y quedó cancelled en los cinco push siguientes, porque cada push a main cancela el run anterior (replay-contracts.yml:13-15). El `check --workspace --all-targets` (:51-52) terminó en success en ab0967e8, 95412dd7 y 8f6bcc8a; en 759f44ec no llegó a correr, y en 0043f889 y 0000eb8d se canceló. Las suites de contratos que van detrás (desde :53) no terminaron todas en ningún push.
  - El paso de marcadores sólo mira `git diff --check HEAD^ HEAD` (:40-44), así que no podía parar los push siguientes.
  - En aa716edf, consultado el 2026-10-11, «Unit and contract suites» dio success y «Replay contracts» quedó cancelled: el check pasó y la suite de contratos se cortó en el paso 7.
  - #715 (07688cf9), posterior a aa716edf, ya no cancela el run en un push a main (replay-contracts.yml:18 en 07688cf9).

**C-F11**
- **LC-F11-01** Afirmaciones sin respaldo:
  - «< 25 ns, cero heap» (.agents/MEMORIA.md:5, :30, :54): no hay benchmark.
  - «Rough Volatility resuelto»: sólo hay un doc en path_signatures.rs:3.
  - «488 archivos Rust» (plan cuántico :333): hay 526 en aa716edf (524 en 5338eda1).
  - El plan cuántico R6 (:360) cita `evolution-engine/src/{genome_store,darwin}.rs`, que no existen: viven en quantum-arena y en god-engine-core.
  - Invariantes de 55 pb / 2 posiciones / 3 USD / 5× (plan cuántico :64 y Parte II :155-181).
  - «Máximo 2 posiciones»: sin código vivo.
- **LC-F11-02** El identificador C-02 tiene dos significados, y R0–R9 también.
- **LC-F11-03** `graphify-out/` versionado: 655 archivos.
- **LC-F11-04** `rustc-ice-*.txt` versionado.

## 7. Protocolo de cada ciclo

1. **Anunciar en el buzón** (`COORDINACION_CODEX_2026-09-28.md`) antes de ejecutar: fase, archivos, hunks, CL-n reservados y pruebas.
   - Si toca la zona de otro agente, esperar su acuse.
   - No empezar en un archivo con reserva activa ajena.
2. **Rama propia** (`claude/…`) en un worktree aislado (ADR-0007), después de `git fetch` y de revisar los PR abiertos y las ramas ajenas.
   - Si origin/main avanzó desde aa716edf, pasar el comprobador de citas de C-F0 y re-anclar las citas del ciclo antes de editar.
   - Antes de cada edición, volver a hacer grep del símbolo.
   - No matar `god_engine` sin verificar PID y StartTime.
3. **Por hallazgo:**
   - escribir un test que falla en la base y guardar su salida;
   - hacer el arreglo mínimo;
   - comprobar que el test pasa;
   - commit atómico `CL-n: …`.
   - `git status` antes de cada commit y `git add` sólo de los archivos propios; `git add -A` está prohibido.
4. **Revisión adversaria.** Un panel de auditores y dos escépticos por hallazgo. Cada escéptico intenta:
   - reproducir el fallo en main;
   - encontrar un contraejemplo al arreglo;
   - buscar el mismo defecto en el espejo y en otros sitios.
5. **Verificación:** la puerta común completa (§2), con T-1 release ≥ 11,0 % y comparación gen a gen contra la lista de 16/144.
   - Si cambia la conducta viva, el T-1 es obligatorio (plan sinc. §2 :170). Si cambia el tamaño o la ejecución, también la paridad bt↔vivo (:169).
   - El T-1 corre con 1 000 USD y sin host (t1_cobertura_genetica.rs:50-52). Para C-F5 y los literales micro de C-F4, la verificación es un fixture de 13 USD y tests del host, o `bt_vivo_parity_audit`, y el recibo dice que el T-1 no cubre esos cambios.
   - **Rendimiento del camino caliente** (pedido del dueño, 2026-10-11 05:49). El camino caliente es lo que corre por cada evento: `GodEngineCore::process_event` y todo lo que llama, incluida `RiskEngine::evaluate_quantum_order` por intención. Si el cambio toca código de ese camino, el PR informa, en release, en la misma máquina y con la misma cinta sintética fija:
     - asignaciones de heap por evento, con un asignador que las cuenta;
     - ns por evento, mediana y p99, de la base y de la rama.
   - Se acepta si no añade asignaciones por evento y si la mediana de la rama no supera a la de la base en más de lo que difieren dos corridas de la base entre sí. Si empeora, el PR dice qué gana a cambio y el dueño decide.
   - El medidor es `crates/god-engine-core/tests/medidor_camino_caliente.rs` (CL-60). Se corre con `scripts/medir_camino_caliente.sh BASE RAMA`, que compila las dos revisiones con el mismo medidor y el mismo compilador, mide base, rama y base otra vez, y aplica `plan_linea_c.py rendimiento` (código 0 pasa, 1 no pasa, 2 no comparable). Las dos corridas de la base dan el ruido. Se compara con la misma cinta (eventos, monedas y semilla) y sólo en release.
   - `nano_benchmark` no sirve de puerta: mide `process_tick_dual`, no `process_event`, supone 3,5 GHz para pasar ciclos a ns (crates/god-engine-core/src/bin/nano_benchmark.rs:68-72) y no cuenta asignaciones.
6. **PR**, no draft, con:
   - tabla hallazgo → test → commit;
   - T-1, suites y lo que no se probó.

   Se fusiona sólo con CI verde (`replay-contracts` y `unit-suites`), suites locales y T-1. Un run cancelado no cuenta como verde. Tras integrar ramas de otras sesiones: diff contra cada padre y `--all-targets`.
7. **Después del merge:** verificar que el SHA es ancestro de origin/main; entonces borrar o reutilizar la rama propia. Nunca borrar ramas ajenas.
8. **Documentar:**
   - bloque fechado en MEMORIA;
   - fila de Línea C en TABLERO;
   - estado de la fase y recibos por archivo en este documento (OID del blob, preguntas, tests, veredicto: verificado, con hallazgos, retirar o sin consumidor);
   - cierre en el buzón.
9. **Prohibido sin autorización del dueño:** operar, entrenar, promover, y cambiar el golden o el trinquete.

## 8. Orden de los primeros ciclos

Las fases ordenan la revisión. No hay operación autorizada, así que ninguna urgencia de capital obliga a saltarse ese orden. Sólo se adelanta un arreglo si ya está decidido en un ADR o si bloquea la medición de la fase en curso.

- **Ciclo 11.**
  - Hecho: T-1 release de aa716edf, 16/144 con la misma lista de genes inertes que 5338eda1.
  - El PR #29 se fusionó como aa716edf, así que el pedido 4 queda sin objeto (§1.3). Al re-anclar este plan, el PR #32 de Línea E seguía en borrador, con head 05f58e87. La API de GitHub lo da después fusionado en main como 03b3f1ca (2026-10-11, padres aa716edf y 05f58e87), con el mismo árbol que 05f58e87. Su descripción informa T-1 16/144 sobre esa unión; Línea C no lo ha reproducido.
  - MEMORIA y TABLERO de los ciclos 9 y 10.
  - Acuse del pedido 3: en este ciclo, el gancho de QS-R4b sólo registra en sombra. Alimentar con él al calibrador, al consejo o al EV va después del paso 2 de §5.
  - C-F0, con el comprobador de citas de este documento (hecho: §9), y C-F1.
  - Inventario test→workflow de C-F10. Se adelanta porque parte de los tests nuevos, sobre todo de C-F7 a C-F9 (evolution-engine/tests, storage-engine/tests, telemetría y herramientas), caerían en suites que hoy no ejecuta ninguna CI (LC-F10-02).
  - Arreglos ya decididos:
    - `rust-toolchain.toml` = `nightly-2026-06-30`, el mismo que la CI, con un test de fuente que compara las tres cadenas (G0, LC-F1-03). Hecho: CL-56.
    - El pico y el umbral únicos del freno inmune (LC-F4-10) pasan al ciclo 13. §9 mostró que ninguna de las dos semánticas actuales es una prueba de nivel α. Con riesgo medido, los dos frenos leen el gen de producción (0,5) como confianza de la prueba de rachas, es decir α = 0,5, y pueden saltar por azar; en régimen micro el veto la sustituye por d* con c = ½, medido desde el pico móvil (LC-F0-08, LC-F0-09, filas 11 y 35). Igualar una a la otra ahora fijaría una de las dos sin derivar. En C-F4 se deriva una sola prueba con el tamaño real y se lleva al dueño como decisión antes de tocar el freno vivo. La persistencia del pico (LC-F0-10) va con ella.
  - Inventario test→workflow de C-F10: hecho (CL-57, `docs/audit/INVENTARIO_TESTS_CI_2026-10-11.tsv`).
  - Hecho fuera de lo previsto:
    - CL-58: `plan_linea_c.py claves`, claves del registro leídas sin escritor (`docs/audit/CLAVES_SIN_ESCRITOR_2026-10-11.tsv`, 25 candidatas sobre c4e8e285). Las verifica C-F1 y las de estrategias, C-F3.
    - CL-59: #718 (AGY, directo a main) metió Feynman en el orquestador vivo y reinyectaba el dominante espectral con τ inoperable, contra #652/H2. Se adelanta porque restaura un contrato ya decidido. T-1 de 766fb1a4: 16/144, la misma lista inerte.
    - Puerta de rendimiento en §7, paso 5, por el pedido del dueño del 2026-10-11 05:49.
- **Ciclo 12.**
  - Primero el medidor del camino caliente (§7, paso 5): un binario de benchmark que pasa una cinta sintética fija por `process_event`, cuenta asignaciones por evento con un asignador propio y da mediana y p99 de ns por evento. Se adelanta de C-F8 porque sin él la puerta de rendimiento no se puede aplicar. Es sólo medición y no cambia la conducta.
  - Arreglos de C-F1 con su test en rojo y en verde, empezando por los altos de Línea C: LC-F1-01 (carpeta del estado según el modo, defecto de CL-48), LC-F1-03 (gen del stop por símbolo, no por slot) y LC-F1-02 (procedencia del spec). LC-F1-04 y la caída al global (LC-F1-06) se coordinan con AGY, que añadió las lecturas estrictas en #720.
  - C-F2 (sólo lectura y pruebas sintéticas; el T-1 no debe moverse) y C-F3. Los hallazgos de zona ajena salen como pedidos con test que falla.
  - Hecho:
    - CL-60, el medidor (§7, paso 5). Primera lectura bajo wine sobre c4e8e285, con 4 monedas y 200 000 eventos, aún sin genoma ni bosques (la cinta no daba intenciones): mediana de 235 µs por evento, p99 de 465 µs, 21,8 asignaciones por evento y ningún evento sin asignar (trade 30,6, libro 11,0, vela 28,3). Wine infla el tiempo, no las asignaciones; la lectura en Windows nativo queda pendiente del PC del operador.
    - CL-60b. La primera puerta del PR #37 no vale: con un `CARGO_TARGET_DIR` compartido, cargo dio a la rama el binario de la base, y las tres corridas midieron lo mismo. El script toca ahora los fuentes antes de compilar con un target compartido y copia cada ejecutable fuera del target, y la puerta exige un binario por lado y binarios distintos cuando las fuentes difieren.
    - CL-61 cierra LC-F1-01 y CL-62 cierra LC-F1-03 (adenda del ciclo 12 en `docs/audit/HALLAZGOS_C-F1_2026-10-11.md`).
    - LC-F1-02 pasa a C-F6: sacar el refresco de specs del `if changed` movería hoy el techo de apalancamiento de 20 a 25 sin un tramo medido. Va al dueño antes.
    - LC-F1-74 nuevo (ruta de `dynamic_config` distinta en el daemon y en el host), para C-F6.
    - CL-60c: el script guarda cada ejecutable con su revisión y una huella del medidor, el compilador y los argumentos, y lo reutiliza; `MEDIDOR_SOLO_COMPILAR=1` compila sin medir, para medir después con la máquina quieta.
    - Puerta del PR #38 (4bffdd8a frente a 3b5e9050): pasa, 238,5 µs frente a 236,0 µs con 9,7 µs de ruido y 19,424 asignaciones por evento en las dos. El coste de #720 (11cdd14b frente a c4e8e285) no pasa: +15,2 µs por evento (+6,8 %) con 5,9 µs de ruido, por la lectura del reloj en cada escritura del registro, que nada lee en producción. Va a AGY como P1.
    - CL-63 y el perfil del camino caliente (`docs/audit/PERFIL_CAMINO_CALIENTE_2026-10-11.md`): el registro se lleva el 36 % de la CPU y la libm de mingw el 29 % (sólo en gnu); dos tercios de las asignaciones son claves de texto construidas en cada evento. Propuestas P1–P5 con dueño; P2 (pesos espectrales una vez por evento) y P5 (claves por moneda precalculadas) entran en C-F6 con la puerta del medidor.
- **Ciclo 13.** C-F4:
  - primero el ATR (LC-F6-01 y LC-F3-02), antes de medir el riesgo al stop o el tope IOC;
  - absorbentes, unidades de los vetos y RA-RUIN-F01;
  - D4, con una sola puerta de sonda para toda la cartera (LC-F4-19);
  - D1 en sombra, con el libro sombra (`shadow_ledger.rs`) sólo registrando hasta el paso 2 de §5;
  - los genes inertes sólo se declaran: se retiran o se conectan después de GENOME-GATE, en el ciclo 15.
  - La forense en dos mitades queda bloqueada por datos externos; C-F4 puede cerrarse sin ella.
- **Ciclo 14.** C-F5: R01, índice de ranura, vortex, tope IOC, escritores del kill-switch (LC-F5-10) y módulos muertos. El T-1 no ve el host: la puerta son tests del host con un fixture de 13 USD, o `bt_vivo_parity_audit`.
- **Ciclo 15.** C-F6 y C-F7, coordinados con Qoder (R7-R6, R7), GLM, Línea E y Sol. GENOME-GATE (LC-F7-16) entra antes de retirar o conectar genes.
- **Ciclo 16.** C-F8 a C-F11.

## 9. Requisitos de la meta

Rige la meta que reformuló el dueño: maximizar el crecimiento geométrico neto de la cuenta (13 USD en Binance USDⓈ-M, operando como taker), sujeto a límites de riesgo, drawdown, costes y capacidad, y subordinado a la supervivencia del capital. Es la doctrina de ADR-0003:19-22 y de MEMORIA:4098-4100, y sustituye la línea «Meta» de §1.4, que la dejaba en ADR-0016. Como requisito medible se escribe así. La tasa log diaria g = ln(E_fin/E_ini)/días se mide sobre equity valorada: caja más flotante, neta de comisiones taker y funding y sin flujos externos (docs/PLAN_MAESTRO_SINCRONIZACION.md:254-258). La puerta es que su cota inferior al 95 % sea mayor que 0 en cada una de las dos mitades fuera de muestra. El riesgo al stop por operación cumple f ≤ ½·f*(p_LCB, b_net) y f ≤ clamp_ruin (ADR-0016:44-51). Una caída mayor que d*(c; 0,05) = 1 − 0,05^{c/(2 − c)} (ADR-0016:91-97), medida desde una referencia fija por época, refuta el modelo con el que se dimensiona. Aquí c = f_real/f*(p_LCB, b_net) es la fracción de Kelly de la época, con f* > 0. El valor d*(½; 0,05) = 1 − 0,05^{1/3} = 0,632 (piso de 4,79 USD sobre 13 USD, ADR-0016:95-100) es sólo la cota superior (crates/risk-engine/src/drawdown.rs:185-188). Con el techo micro (nota A1) y la referencia de ADR-0016:27-29 (p = 0,40, b = 2,25, f* = 13,3 %) con SL del 1 %, c ≈ 0,075 y d* ≈ 0,11 (cálculo propio). La regla «duplicar cada 3 días» (.agents/AGENTS.md:9-10; ADR-0016:12-16) queda como cota de referencia, g_ref = ln 2/3 = 0,2310 por día (+25,99 % diario compuesto), y se reporta como brecha ĝ/g_ref, no como puerta. Con los techos del régimen micro que hoy impone el código (f ≤ 1,0·SL, nota A1), esa cota exigiría unas 81 operaciones al día a +0,30 R netos con SL del 1 %. Hay dos frecuencias registradas, las dos anteriores a aa716edf: 0,02 al día (ADR-0003:13-15, sin verificar) y de 57 a 90 operaciones por mitad del tape BTC de junio de 2026 (INFORME_FORENSE_MAESTRO.md:6165-6176). El informe no da la duración de cada mitad, así que su frecuencia diaria está sin verificar.

**Notación y abreviaturas**
- ADR-0003 es docs/adr/ADR-0003-doctrina-meta-geometrica.md y ADR-0016 es docs/adr/ADR-0016-riesgo-geometria-y-falsacion-de-la-meta.md. «Plan maestro» es docs/PLAN_MAESTRO_SINCRONIZACION.md, «plan cuántico» es docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md y MEMORIA es .agents/MEMORIA.md. `lib.rs` sin ruta es crates/god-engine-core/src/lib.rs.
- R es la pérdida al stop más la fricción de ida y vuelta, de modo que b_net = (TP − f_rt)/(SL + f_rt) (ADR-0016:48). m es el edge neto medio en R y N son las operaciones ejecutadas por día.
- Crecimiento por operación (ADR-0016:22-24): G(f) = p·ln(1 + f·b) + (1 − p)·ln(1 − f), con f* = p − (1 − p)/b. Por día, g ≈ N·G(f).
- Estados: «comprobado por lectura» quiere decir comprobado leyendo aa716edf, sin ejecutar nada; no es «medido», que se reserva para lo ejecutado o medido sobre datos (§9.4). «Sin medir» incluye «no existe estimador»; «bloqueado por datos externos» quiere decir que hacen falta tapes o fills del operador.
- Las cifras derivadas (tablas, plazos, simulaciones) son cálculo propio a partir de las fórmulas citadas. Ninguna es una medición del repo. Las simulaciones son scripts fuera del repo.

**Lo que exige la cota de referencia** a ½ Kelly, con b = 2,25 y G exacta (cálculo propio):

| Operaciones/día | Edge neto mínimo | p mínima de TP antes que SL |
|---|---|---|
| 4 | +0,61 R | 0,494 |
| 10 | +0,38 R | 0,425 |
| 16 | +0,30 R | 0,400 (el ejemplo de ADR-0016:27-29) |
| 50 | +0,17 R | 0,359 |
| 200 | +0,084 R | 0,333 |

### 9.1 Requisitos

**Crecimiento y edge**

| # | Requisito | Estimador (aa716edf) | Umbral y origen | Prueba que lo refuta | Estado |
|---|---|---|---|---|---|
| 1 | Crecimiento geométrico neto OOS positivo en cada mitad (meta del dueño). | Sólo el CAGR puntual de crates/backtest-engine/src/metrics.rs:147-161, con g = ln(1 + CAGR)/365,25. Existe una serie diaria de equity valorada, caja + flotante − comisión taker de salida (crates/backtest-engine/src/bin/continuous_evolution_backtest.rs:118-151, escrita a CSV en :792-800). No hay cota inferior, bootstrap por bloques ni partición en mitades. Ni la física ni el replay cobran funding; el único cargo está en crates/backtest-engine/src/vectorized.rs:206-210, con tasa sintética, así que el CAGR del replay no es neto de funding. | g_LCB,95 > 0 en cada mitad. Origen: docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:376 (IC por bloques positivo del crecimiento log por operación en las dos mitades), MEMORIA:4098-4100 (EV > 0 en las dos mitades), ADR-0003:19-22 (maximizar, sin número) y z = 1,64 (crates/risk-engine/src/evidence.rs:59). El paso de «por operación» a «por día» es propuesta de Línea C. | La cota superior al 95 % de ĝ es ≤ 0 en una mitad. Certificar g > 0 en una mitad lleva n ≈ z²/((1 − c/2)²·SR_d²) días, con SR_d el Sharpe diario aritmético y c la fracción de Kelly. A ½ Kelly, g = (3/8)·SR_d², luego n ≈ (2/3)·z²/g ≈ 1,79/g (≈ 1,94/g en el modelo binario con p = 0,40 y b = 2,25). Son ≈ 300 días por mitad con g = 0,006/día (ejemplo, sin origen) y potencia del 50 %. Para una potencia del 80 %, (2/3)·(1,64 + 0,84)²/g ≈ 684 días por mitad. | Sin medir: no existe estimador de la cota. Además, bloqueado por datos externos. |
| 2 | Cota de referencia: duplicar cada 72 h. | No hay estimador de G72: el grep de growth_factor_72h, log_growth_72h y G72 en *.rs y *.py da 0. Proxy: el CAGR de la fila 1. | g ≥ ln 2/3 = 0,2310/día. Origen: .agents/AGENTS.md:9, ADR-0016:12-16 y docs/PLAN_MAESTRO_SINCRONIZACION.md:249-251 y :891-892. | La cota superior al 95 % de ĝ queda por debajo de 0,2310. Se reporta como brecha, no veta. | Sin medir: no existe estimador de G72. Además, bloqueado por datos externos. |
| 3 | Edge neto por operación ejecutada, en R. | No existe. El panel mide USD por trade (crates/backtest-engine/src/metrics.rs:116-117). El libro sombra mide R sólo de lo vetado (crates/audit-engine/src/shadow_ledger.rs:118-122) y sólo lo usan tests. | m_LCB > 0 a la N medida. Referencia: +0,30 R a 16/día (ADR-0016:27-29). | La cota superior al 95 % de la media de R OOS es ≤ 0. Plazo ≈ z²·s²/(m̂ − m)² operaciones, con s ≈ 1,6 R para b = 2,25: con m̂ = +0,30 R frente a m = 0, ≈ 76 operaciones con potencia del 50 % y ≈ 173 con el 80 %. En días, n/N̂: con N = 0,02/día (ADR-0003:13-15, sin verificar), 173 operaciones son ≈ 8 650 días; con N = 16/día (ADR-0016:27-29), ≈ 11 días. | Sin medir: no existe estimador. |
| 4 | p_LCB de tocar TP antes que SL por encima del punto muerto. | D1 pide la cota de Jeffreys de P(TP antes que SL) (ADR-0016:47). El gate EV usa `intent.win_probability` si está en (0, 1) y, si no, `win_rate_hierarchical_lcb` con prior 0,55 y k0 = 10, recortado a [0,05; 0,95] (crates/risk-engine/src/lib.rs:1137-1157; crates/risk-engine/src/evidence.rs:134-166). `win_rate_lcb` (evidence.rs:109-125) sólo lo usa el q de ruina (crates/risk-engine/src/ruin.rs:90-99). Las dos cotas parten de `win_rate`, la media posterior, con prior de diseño, de los cierres con PnL neto > 0 (lib.rs:3843, :4001-4006; crates/quantum-arena/src/genome.rs:2428-2442; crates/quantum-arena/src/state.rs:43). | p_LCB > 1/(1 + b_net) (ADR-0016:58-59): 0,3077 con b_net = 2,25. | Con p real de 0,40, 76 cierres dan potencia ≈ 50 % y hacen falta ≈ 170 para el 80 % (LCB de evidence.rs:109-125, binomial exacta, cálculo propio). Refuta: la cota SUPERIOR al 95 % de p queda ≤ 1/(1 + b_net). No superar el punto muerto tras n cierres no refuta el edge. En días, n/N̂: con N = 0,02/día (ADR-0003:13-15, sin verificar), 170 cierres son ≈ 8 500 días; con N = 16/día (ADR-0016:27-29), ≈ 11 días. | Comprobado por lectura: `win_rate` no mide P(TP antes que SL). Incluye las salidas por trailing, decaimiento de alfa, flujo tóxico, zombi y tiempo (lib.rs:3580-3587) y cuenta como perdedor un TP que no cubre la fricción. Sus valores, bloqueados por datos externos. |
| 5 | Frecuencia N de operaciones ejecutadas. | n_trades/span_days (crates/backtest-engine/src/metrics.rs:43-44, :143-144). En el forense, trades entre start_ts y end_ts (src/bin/audit_forensic_backtest.rs:1040-1059). | Dueño: sin número. Referencia: N ≥ g_ref/G(f). ADR-0016:27-29 da ~16/día; ADR-0003:13-15 da ~10/día. | Si N·G(f, m̂) < g_ref, cae la referencia. Con N = 0 en una mitad, g = 0 y la fila 1 no se cumple en ella. | Bloqueado por datos externos. |
| 6 | Riesgo al stop f_R = ½ Kelly sobre p_LCB (D1). | Existe un f_R: Kelly pleno sobre p_LCB y b_LCB, encogido por n/(n + k) y acotado por la racha y por 0,25 (crates/risk-engine/src/kelly_envelope.rs:254-293), con presupuesto N_max = C·f/(SL + coste) en :298-330. El host lo aplica sólo con n ≥ 30, con z = lerp(1,64; 0,85; micro_w) y k = log_lerp(50; 10; micro_w), que valen 0,85 y 10 con 13 USD (src/bin/god_engine.rs:4024-4029, :4093). No es ½ Kelly, y su b sale de una EWMA de PnL en USD (src/bin/god_engine.rs:3522-3526, :3877-3882), no del b_net de la geometría TP/SL. El riesgo realizado: EWMA margen·L·SL/C (crates/risk-engine/src/lib.rs:1434-1446). | f_R = ½·max(0, p_LCB − (1 − p_LCB)/b_net), con f_R ≤ clamp_ruin (ADR-0016:46-51). | Fixture de 13 USD con p_LCB = 0,40, b_net = 2,25 y SL 1 %: D1 pide 6,7 %, y el techo micro deja 1,0 % (nota A1). | Comprobado por lectura: no se cumple. El fixture no se ha ejecutado. |
| 7 | Sin edge medido, tamaño 0; no se infla hasta la mínima. | Por lectura, el margen sube hasta min/L (crates/risk-engine/src/lib.rs:1298-1306) si la mínima cabe en clamp_ruin (:1239-1248). Mientras n < 30, el host usa boot_lev = round(nocional/margen).clamp(1, 10) cuando el slot tiene margen, y 5 sólo sin margen y con cap ≤ 50 (src/bin/god_engine.rs:4064-4077, en especial :4070-4073). Con n ≥ 30, si f ≤ 0 y micro_w > 0, `max_leverage` fija f = clamp_ruin(f_min_ejecutable, q): autoriza la orden mínima sin edge medido (crates/risk-engine/src/kelly_envelope.rs:386-390), contra su propio doc (:335). | f_R = 0 si p_LCB ≤ 1/(1 + b_net) (ADR-0016:56-61). | Test con 13 USD y una moneda sin historial: D1 exige tamaño 0, y por lectura hoy sale la orden mínima. | Comprobado por lectura: no se cumple. El test no se ha ejecutado. |
| 8 | Un solo z en todo dimensionado. | La envolvente usa z = lerp(1,64; 0,85; micro_w), que vale 0,85 con 13 USD (src/bin/god_engine.rs:4026; crates/backtest-engine/src/booktick_replay.rs:728). | z = 1,64 (ADR-0016:47; crates/risk-engine/src/evidence.rs:59). | Test de fuente: cualquier z distinto en un dimensionador. | Comprobado por lectura: no se cumple. |
| 9 | El apalancamiento es consecuencia del margen, no de la convicción. | L multiplica Kelly, convicción, volatilidad, crecimiento y Hurst (crates/risk-engine/src/leverage_matrix.rs:327-336). El factor de crecimiento sube cuando el capital baja (:219, :237). | ADR-0016:54-55; sin número. | Fixture de 13 USD: L cambia con la confianza a margen fijo, o sube tras una pérdida. | Comprobado por lectura: no se cumple. El fixture no se ha ejecutado. |

*Nota A1.* Con 13 USD, micro_w = 1 (crates/risk-engine/src/capital_regime.rs:64, :94-98). El margen se acota a 2,60 USD (crates/risk-engine/src/lib.rs:1323). En la matriz el techo micro es 5 y L ∈ [1, 5] (crates/risk-engine/src/leverage_matrix.rs:261-267, :336), cuantizado hacia abajo en crates/risk-engine/src/lib.rs:1020. L = 6 sólo sale del primer rescate, que redondea hacia abajo un techo de 6,5 (crates/risk-engine/src/lib.rs:1268-1270, :1273-1290). Ese rescate sólo se dispara si margen·L < mínima, es decir con margen < 5/L_matriz (menos de 1 USD a L = 5), y deja el nocional cerca de la mínima. El segundo rescate (:1349-1360) sólo actúa si el margen recortado al tope de [1,20; 2,60] USD (:1323) no alcanza 5,10 USD con el L vigente; sube L a lo sumo a ceil(5,10/1,20) = 5 y conserva `.max(dynamic_leverage)`. El nocional máximo alcanzable es 2,60·5 = 13 USD, luego f ≤ 2,60·5·SL/13 = 1,0·SL, y la orden mínima de 5,10 USD (:1295) da f ≥ 0,39·SL. Si exp(ln 5) redondea a 4,999…, como en glibc, el floor de :1020 deja L = 4 y f ≤ 0,8·SL; en el binario de Windows, sin verificar. El 6,7 % de D1 sólo cabe con SL ≥ 6,7 %. Con f = 1,0 % la cota de referencia pide ≈ 81 operaciones/día a +0,30 R y ≈ 262 a +0,10 R (G exacta con b = 2,25, cálculo propio).

**Supervivencia y drawdown**

| # | Requisito | Estimador (aa716edf) | Umbral y origen | Prueba que lo refuta | Estado |
|---|---|---|---|---|---|
| 10 | El veto de entradas no pasa de d*. | crates/risk-engine/src/lib.rs:311-344, con d* de crates/risk-engine/src/drawdown.rs:193-197. Contrato en crates/risk-engine/tests/qs_d2_d3_geometria_y_drawdown_contract.rs:122-123. | min(lerp(dd_max_medido, d*, micro_w), d*) (ADR-0016:107), con d* = 0,632, c = ½ y α = 0,05 (crates/risk-engine/src/drawdown.rs:170, :173). Con 13 USD el umbral es d* exacto. | Pico de 13 USD: con capital de 4,70 USD debe salir REJ_DRAWDOWN, y con 4,85 USD no. | Comprobado por lectura: se cumple. El contrato no se reejecutó aquí, y el T-1 corre con 1 000 USD (crates/backtest-engine/tests/t1_cobertura_genetica.rs:50-52). |
| 11 | El freno del host usa el mismo umbral, pico y capital que el veto. | src/bin/god_engine.rs:1729-1737 aplica drawdown_maximo(...).min(d*) sin el lerp. Usa min(arena, exchange) (:1696-1703) y un pico local (:1682, :1704-1711). Aplana y deja el latch (:1802-1818). | ADR-0016:114-115; el umbral de la fila 10. | Con el gen de producción global_max_drawdown = 0,5 (config_dir/genomes/prod/active.json:9; α = 0,5 por crates/risk-engine/src/drawdown.rs:48-51) y 13 USD. Sin riesgo medido, drawdown_maximo devuelve el gen (crates/risk-engine/src/drawdown.rs:112-119): el riesgo corta al 63,2 % (piso de 4,79 USD) y el host al 50 % (6,50 USD). Con r = 0,5 % y q = 0,5, k = ln(200/0,5)/ln 2 = 8,64: el riesgo sigue en 0,632 y el host da 1 − 0,995^{8,64} ≈ 0,042. Test: la misma salida en los dos. | Comprobado por lectura: no se cumple, desde el arranque (amplía LC-F4-10). El test no se ha ejecutado. |
| 12 | El piso de 4,79 USD sobrevive a reinicios y el veto tiene salida. | No existe. El pico nace con el capital de arranque (crates/risk-engine/src/lib.rs:227-231), el del host es local (src/bin/god_engine.rs:1682) y ninguno se persiste. | 13·0,05^{1/3} = 4,79 USD; «hasta recuperar o re-medir» (ADR-0016:99-103). | Test de reinicio: con pico de 13 y capital de 5, reiniciar y caer a 4,5 USD debe seguir vetado. Hoy el piso pasa a 1,84 USD. | Sin medir: no existe estimador. |
| 13 | El tope de ruina es coherente con d*. | clamp_ruin = min(f, tope por racha, 0,25) (crates/risk-engine/src/ruin.rs:60-75), con piso 0,05 y horizonte de 200 (crates/risk-engine/src/kelly_envelope.rs:27-29). | Propuesta: (1 − f)^{k(q, α)} ≥ 1 − d*, con k = racha_por_azar(q, α) = ln(H/α)/ln(1/q) (crates/risk-engine/src/drawdown.rs:56-65), al mismo α que d*. Da f ≤ 1 − 0,3684^{1/16,24} = 5,96 % ≈ 6,0 % con q = 0,6 y α = 0,05. Con la racha esperada de crates/risk-engine/src/ruin.rs:48-56 (10,37, sin nivel) saldría 9,2 %, que una racha de 11 pérdidas viola con probabilidad ≈ 25 % en 200 operaciones (cadena de Markov exacta, cálculo propio). El 25 % (crates/risk-engine/src/ruin.rs:74) y el 0,05 (crates/risk-engine/src/kelly_envelope.rs:27) son axiomas sin derivación. | Monte Carlo con la distribución de R medida: P(caer a (1 − d*)·W_ref) > 0,05. Hoy, con q = 0,6, f = 25 % y 4 pérdidas seguidas ya son una caída del 68 %. | Sin medir. Sólo la comparación determinista contra d* puede correr hoy; el Monte Carlo con R medida está bloqueado por datos externos. |
| 14 | Ninguna tolerancia de drawdown del código supera d*. | `check_drawdown_limit` conserva 0,75 y 0,85/0,90 (crates/risk-engine/src/guard.rs:44-46). Sólo la llaman tests. | d* = 0,632; «el literal 0,85 deja de existir» (ADR-0016:110). | Test de fuente sobre crates/risk-engine/src. | Comprobado por lectura: no se cumple, aunque la API no tiene consumidor vivo. |

**Fricción y costes**

| # | Requisito | Estimador (aa716edf) | Umbral y origen | Prueba que lo refuta | Estado |
|---|---|---|---|---|---|
| 15 | Se conoce la comisión taker por pierna y es la misma en todo el código. | La API la pide al arrancar, sólo para el primer símbolo (src/bin/god_engine.rs:1233-1251), y la guarda en la config (:1636-1637). La física usa los literales 0,0002/0,0005 (crates/god-engine-core/src/reality_physics.rs:31-32). La comisión cobrada de verdad se lee de /fapi/v1/income en IncomeTotals (crates/execution-engine/src/income_evidence.rs:385-498, fila 17), sin el nocional de cada fill. | 5 pb por defecto (crates/quantum-arena/src/config.rs:208). B3.11 afirma 4 pb (src/bin/god_engine.rs:1238-1242), sin dato en el repo. | La comisión/nocional de un fill taker real distinta de 0,0005 refuta el defecto; distinta de 0,0004, refuta B3.11. | Bloqueado por datos externos (fills y cuenta). |
| 16 | La fricción de la meta es la de la ruta real, taker. | force_maker = false (src/bin/god_engine.rs:4269). Protección STOP_MARKET y TAKE_PROFIT_MARKET (crates/execution-engine/src/executor.rs:3070, :3091). Gate taker en las dos piernas (crates/risk-engine/src/lib.rs:833-834). | ≥ 2·t_taker de ida y vuelta, más deslizamiento: 10 pb con el defecto de 5 pb (crates/quantum-arena/src/config.rs:208) y 8 pb con los 4 pb de B3.11 (src/bin/god_engine.rs:1238-1242); sin dato real. Los 4 pb maker del plan cuántico (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:178) no tienen fills. | Un fill maker de entrada con menos de 10 000 USD (umbral iceberg, config_dir/genomes/prod/active.json:152) refuta la lectura de la ruta. | Bloqueado por datos externos. |
| 17 | La fricción que cobra el gate no es menor que la realizada. | 2·taker + 2·(piso + ATR·√(lat/60 s)) (crates/risk-engine/src/tp_sl.rs:279-282), en el gate (crates/risk-engine/src/lib.rs:853-854). Lo realizado (comisión y deslizamiento de salida) va a archivos del operador (crates/execution-engine/src/user_data_stream.rs:555-569). IncomeTotals suma comisión, funding y `cost_to_abs_realized_ratio` desde /fapi/v1/income (crates/execution-engine/src/income_evidence.rs:385-498), y lo reporta src/bin/income_report.rs:119-159. En vivo, el FEE-BREAKER suspende de 1 a 12 h las entradas de un símbolo con ≥ 3 cierres, neto < 0 y comisiones mayores que el bruto ganador (src/bin/god_engine.rs:2369-2445); esos umbrales no tienen origen. | Con el genoma prod (config_dir/genomes/prod/active.json:85-86) y 5 pb: 11,4 / 11,6 / 12,0 / 13,4 pb para ATR de 5 / 10 / 20 / 50 pb. | En ≥ 30 ciclos (WF_MIN_TRADES, crates/evolution-engine/src/online_daemon.rs:53, sin derivación), el coste medio realizado supera la f del gate con IC 95 % unilateral. | Bloqueado por datos externos (fills y cuenta). |
| 18 | La fricción del gate incluye el cruce de la horquilla. | No existe en la fricción del gate: roundtrip_friction no tiene término de spread y el EV parte del mid. La física cobra media horquilla al entrar (lib.rs:7867). Hay dos puertas aparte. D-755 exige spread ≤ min(tp_at_tau, σ(τ)) − (maker + taker), con piso de un tick (lib.rs:5427-5454): resta sólo las comisiones (≈ 7 pb), sin latencia ni piso. SeniorEjecucion veta si (spread/2 + 0,5 pb)·(1 + 0,5·(1 − laminar)) > 35 + 65·s (crates/metacortex-engine/src/consejo_seniors.rs:741-747; slip_bps en lib.rs:7518-7523); los literales 0,5, 35 y 65 no tienen origen. | El plan cuántico pide fricción «más el spread» (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:327), sin cifra. | Test sin tapes: con horquilla 0 y con una que pase D-755, la decisión EV es la misma. La cuantía real necesita tapes. | Sin medir: no existe estimador en la fricción del gate. |
| 19 | El deslizamiento de entrada y la latencia del camino caliente caben en el modelo. | No existe para el deslizamiento: el diario de entradas no guarda el precio de decisión (crates/execution-engine/src/trade_accounting.rs:433-441). La latencia del feed se mide por evento, pero sólo se guarda el último valor, sin distribución (src/bin/god_engine.rs:3251-3264). La latencia de orden a fill no tiene estimador: el gate usa los valores del genoma (config_dir/genomes/prod/active.json:85-86) y la física les añade una lognormal con σ = 0,35 (crates/risk-engine/src/tp_sl.rs:195). | Tolerancia IOC de 5 a 35 pb (src/bin/god_engine.rs:4439), sin origen. Gen de latencia de 30,68 ms, sin medición. | Deslizamiento medio de entrada frente al mid de la decisión mayor que piso + latencia en 30 entradas, o p99 de envío a fill mayor que el p99 del modelo. | Sin medir: no existe estimador. Después, bloqueado por datos externos. |
| 20 | La física (replay, T-1 y aptitud) cobra lo mismo que el vivo. | La física trata el cierre por TP como maker (lib.rs:3631); en vivo es TAKE_PROFIT_MARKET, taker (crates/execution-engine/src/executor.rs:3091). | La misma comisión por pierna, como pide D-645 (crates/risk-engine/src/lib.rs:814-834). | Test del núcleo: un TP con t = 5 pb y m = 2 pb debe cobrar 5 pb de salida, y hoy cobra 2. | Comprobado por lectura: no se cumple (T-06, INFORME_ASEGURAMIENTO_TERCERO.md:36). |
| 21 | Todos los consumidores calculan la misma fricción para la misma orden. | Gate y brackets, 11,6 pb. Breakeven y trailing, 16 pb (lib.rs:3286-3287, :3341-3343). D-755, 7 pb (lib.rs:5434-5437). Contabilidad del host y del replay, 7 pb (src/bin/god_engine.rs:3871-3873; crates/backtest-engine/src/booktick_replay.rs:300-310). La física cobra por lado comisión + max(impacto + latencia, piso) (crates/god-engine-core/src/reality_physics.rs:109-111, :246-248), con comisiones literales (:31-32): con ATR de 10 pb y 13 USD, ≈ 5,6 pb por lado y 11,1 pb de ida y vuelta taker, frente a los 11,6 pb de roundtrip_friction (cálculo propio con la latencia estática del gen). | Igual a la del gate (fila 17). | Test con taker, maker, piso, ATR y latencia fijos. La guardia de fuente sólo cuenta llamadas (crates/risk-engine/src/tp_sl.rs:800-819). | Comprobado por lectura: no se cumple. |
| 22 | La puerta EV exige edge ex ante con margen sobre la fricción. | EV = p·TP − (1 − p)·SL frente a f·m, con m = clamp(gen, 1,05 + 0,20·escasez, 1,80) (crates/risk-engine/src/lib.rs:1159-1175, :1184-1196). | Con 13 USD m = 1,25 (política D-641, sin derivación): EV neto > 0,25·f ≈ 2,9 pb, es decir 0,25·f/(SL + f) en R: ≈ 0,026 R con SL del 1 %. | La tasa realizada de TP antes que SL de lo admitido queda por debajo de la p que usó el gate (IC 95 %) en las dos mitades. | Sin medir. |
| 23 | El suelo de viabilidad exige f ≤ 0,65·SL. | min_viable_sl = f/0,65 (crates/quantum-arena/src/genome.rs:2505-2512) y REJ_TP_SL_FLOOR en todo capital (crates/risk-engine/src/lib.rs:912-914). | 0,65 es «política elegida, no una constante física» (crates/quantum-arena/src/genome.rs:2451-2458): sin origen derivado. Con f = 11,6 pb, SL ≥ 17,8 pb. | Una orden admitida con SL < f_r/0,65, con la fricción real f_r de la fila 17. | Bloqueado por datos externos. |
| 24 | El panel de la meta mide los costes. | No existe: ExPostMetrics no tiene campo de costes (crates/backtest-engine/src/metrics.rs:42-68). `fees_est` del replay usa maker + taker (crates/backtest-engine/src/booktick_replay.rs:300-310, :575-579). Fuera del panel, los costes de la cuenta salen de IncomeTotals vía income_report, y el FEE-BREAKER actúa en vivo (fila 17). | Sin origen. | Test sintético: un cierre por SL con t = 5 pb debe dar 10 pb, y `fees_est` da 7 pb. | Sin medir: no existe estimador en el panel. Los costes de la cuenta, bloqueados por datos externos (cuenta). |

**Métricas de evaluación**

| # | Requisito | Estimador (aa716edf) | Umbral y origen | Prueba que lo refuta | Estado |
|---|---|---|---|---|---|
| 25 | Cada corrida reporta el panel ex post. | Se calcula en crates/backtest-engine/src/booktick_replay.rs:619-626. Ningún lector de sus campos ni de `panel_line` (crates/backtest-engine/src/metrics.rs:90) fuera del módulo y su test (grep). | ADR-0003:32-34. | Un binario de evaluación sin `panel_line` en la salida. Hoy lo son todos. | Comprobado por lectura: no se cumple. |
| 26 | Sharpe en la unidad de la meta (R por operación o retorno diario). | El del panel es USD por trade escalado por √(trades/año) (crates/backtest-engine/src/metrics.rs:13-17, :183-190). `shadow_sharpe_ratio` es un t studentizado (crates/evolution-engine/src/online_daemon.rs:1314). | Dueño: sin número. Referencia: 0,237 por operación a 10/día y ½ Kelly (G exacta, b = 2,25, tabla de §9: m = +0,38 R, s = 3,25·√(p(1 − p)) = 1,61 R); 0,248 con la aproximación de difusión de docs/PLAN_MAESTRO_SINCRONIZACION.md:1068-1077. | PSR frente a ese Sharpe < 0,05 sobre R OOS (crates/risk-engine/src/selection_stats.rs:85-92); hoy no está cableado. | Bloqueado por datos externos. Además falta el estimador en la unidad. |
| 27 | MaxDD y CAGR sobre equity valorada. | El replay usa la caja sin flotante (crates/backtest-engine/src/booktick_replay.rs:593-606). El vivo usa min(caja, equity) (src/bin/god_engine.rs:1696-1703). `daily_report` sí construye una serie diaria de equity valorada, caja + flotante − comisión taker de salida (crates/backtest-engine/src/bin/continuous_evolution_backtest.rs:118-151, CSV :792-800). Ni la física ni el replay cobran funding; el único cargo está en crates/backtest-engine/src/vectorized.rs:206-210, con tasa sintética, así que el CAGR del replay no es neto de funding. | Las posiciones abiertas valoradas entran en V (docs/PLAN_MAESTRO_SINCRONIZACION.md:254-256). | Replay que termina con una posición abierta en pérdida: el MaxDD del panel queda por debajo del de la equity. | Comprobado por lectura: no se cumple en el replay. |
| 28 | Sortino, Calmar, CVaR95 y turnover se reportan con su convención. | crates/backtest-engine/src/metrics.rs:163-170, :174, :191-198 y :205-231 (CVaR con ≥ 20 trades). | Sin origen: ADR-0003:19-22 no fija cifras. | Sin umbral no hay refutación; sólo se exige reportarlos. | Comprobado por lectura: no se cumple, porque nadie reporta el panel (fila 25). Sus valores, bloqueados por datos externos. |
| 29 | Capacidad. | No existe un estimador medido. El turnover es sólo un proxy (crates/backtest-engine/src/metrics.rs:25-26). La física modela el impacto como 5 pb·√(Q/1 M USD), con una calibración sin origen (crates/god-engine-core/src/reality_physics.rs:82-88, :231-232). BookDepthSlippagePredictor existe, pero fuera de su módulo sólo lo usa un test (crates/god-engine-core/src/slippage_predictor.rs:13-58; crates/god-engine-core/tests/slippage_continuity_contract.rs:1). | Sin origen. | Deslizamiento realizado frente a nocional y profundidad del libro, por símbolo. | Sin medir: no existe estimador medido. |
| 30 | Correlación entre estrategias. | No existe. crates/risk-engine/src/correlation_guard.rs:184 correlaciona activos, y `TasaAcierto` (lib.rs:80) es tasa de acierto por rama, no PnL. | Sin origen. | Matriz de correlación del R OOS por rama, después de atribuir cada resultado a su rama. | Sin medir: no existe estimador. |
| 31 | Toda promoción pasa el DSR con la multiplicidad acumulada. | crates/risk-engine/src/selection_stats.rs:156-165 y :248, usado en el demonio (crates/evolution-engine/src/online_daemon.rs:1888-1903). Hay un segundo DSR, AntiBiasGovernor::calculate_deflated_sharpe (crates/evolution-engine/src/anti_bias_governor.rs:12-60), con umbral 0,90 sin origen (:58) y sin llamadores (:1). | DSR ≥ 0,95 (crates/risk-engine/src/selection_stats.rs:237-238, «estándar de la industria», sin derivación); docs/PLAN_MAESTRO_SINCRONIZACION.md:1129-1132. | Un promotor sin DSR. Hoy hay cinco (contradicción LC-F0-21). | Comprobado por lectura: no se cumple. |
| 32 | Una sola aptitud en todo promotor. | F = B/o si B ≥ 0 y F = B·o si B < 0, con B = ln(C_f/C_i) − 4·ln2·DD² (crates/god-engine-core/src/fitness_contract.rs:5, :35-55) y el factor OOS o = 1 + max(0, −ln(C_oos,f/C_oos,i)) (crates/evolution-engine/src/fitness.rs:86-104). | crates/evolution-engine/src/fitness.rs:93. La λ se eligió, no se derivó (:76-80). | Un promotor con aptitud propia: src/bin/evolution.rs:386-424. | Comprobado por lectura: no se cumple. |
| 33 | Ningún objetivo ×2 en 3 días entra en una decisión del motor. | Grep de ln2/3, 0,231, 259 200, days_to_double, compound_rate y 2^{1/3} en código, scripts y configuración, sin graphify-out ni docs. | 0 usos en vetos, tamaños o aptitud (ADR-0003:37-38). | Una constante de ese tipo que alimente una decisión. | Comprobado por lectura: se cumple. Quedan restos sin efecto (nota D1). |
| 34 | Cobertura conformal y calibración de p en tape real. | Conformal: `adaptive_error_rate` (crates/god-engine-core/src/conformal.rs:136-142). Platt: no hay estimador de fiabilidad (crates/god-engine-core/src/calibration.rs:107-167 sólo ajusta y aplica). El ensamble sí tiene un estimador de habilidad: el z de la ventaja de Brier frente a la tasa base (crates/god-engine-core/src/ensemble.rs:30-101, :297-305). | α del gen; la tolerancia de ±2 % y «Brier/ECE < Platt» de §3 (fila 26) no tienen origen. | Error conformal fuera de α ± 2 % por activo, o ECE peor que la tasa base fuera de muestra. | Conformal: bloqueado por datos externos. Platt: sin medir, no existe estimador. |

*Nota D1.* Restos sin efecto en decisiones: la variable muerta `_compound_rate_3d` (src/bin/evolution.rs:374-379), las impresiones «Goal: 2.0x» (:627) y «META 100%/3d» (src/bin/backtest_windows.rs:248-250), la telemetría `days_to_double` con 50 operaciones/día fijas (crates/audit-engine/src/telemetry.rs:117-139) y un test aritmético (crates/risk-engine/tests/compounding_audit.rs:60-102). El 259 200 de crates/quantum-arena/src/temporal_spectrum.rs:2375 es la longitud de un test.

**Falsación**

| # | Requisito | Estimador (aa716edf) | Umbral y origen | Prueba que lo refuta | Estado |
|---|---|---|---|---|---|
| 35 | La caída d* es una prueba de nivel α (candidata a de C-F0). | Los dos frenos miden desde el pico móvil (crates/risk-engine/src/lib.rs:311-312; src/bin/god_engine.rs:1704-1711). La fórmula es la de caer alguna vez a x·W₀ (crates/risk-engine/src/drawdown.rs:177-181). | α = 0,05 (crates/risk-engine/src/drawdown.rs:173). d* se calcula con c = f_real/f*(p_LCB, b_net) de la época. Los dos frenos pasan siempre c = ½ (crates/risk-engine/src/lib.rs:336-339; src/bin/god_engine.rs:1729-1732), y 0,632 es sólo la cota superior (crates/risk-engine/src/drawdown.rs:185-188). Con el techo micro (nota A1) y la referencia (p = 0,40, b = 2,25, SL del 1 %), c ≈ 0,075 y d* ≈ 0,11. | Falsas alarmas a ½ Kelly: con el propio modelo, la ganancia log media antes de la primera caída d* desde el pico es (1/α − 1)/3 = 6,33. Simulación propia a g_ref: 6 % a 3 días, 29 % a 10, 70 % a 30 y 98 % a 90; desde W₀, ≈ 5 %. Al tamaño real (c ≈ 0,075), P(caer un 63,2 %) = 0,368^{2/c − 1} < 10⁻⁹, así que el umbral 0,632 casi nunca dispara. Potencia sin edge (p = 1/(1 + b), f = 1 %, deriva log −1,1·10⁻⁴ por operación, aproximación de difusión): una caída de 0,632 tarda de media ≈ 3 300 operaciones desde el pico y ≈ 8 950 desde un W_ref fijo, ≈ 206 o ≈ 560 días a 16/día y siglos a 0,02/día (cálculo propio). | Sin medir: cálculo propio, sin test. Sólo es una prueba al 5 % desde un W_ref fijo por época. |
| 36 | Un e-proceso sobre R ejecutados refuta la meta (candidata b). | No existe. VilleEProcess prueba H0: E[X] ≤ 0 (crates/risk-engine/src/ville_e_process.rs:9-10): certifica edge, no lo refuta. | e-valor ≥ 1/α = 20 (crates/risk-engine/src/ville_e_process.rs:16-18). H0: E[R] ≥ m_meta, con m_meta = 0 (dueño) o +0,30 R (referencia). | Apostar a Y = m_meta − R sin recortar, con λ fijo = 0,12, el argmax de E[ln(1 + λY)] bajo E[R neto] = 0 con b_net = 2,25 y m_meta = +0,30 R (crecimiento de 0,018 por operación): mediana ≈ 130 operaciones (simulación propia). Con λ > 0,225 el proceso pierde riqueza bajo esa alternativa (−0,011 por operación con λ = 0,25). Con el λ adaptativo de VilleEProcess (crates/risk-engine/src/ville_e_process.rs:182-189) acotado a 0,25, la mediana es ≈ 230. Con m = +0,30 R exacto, el rechazo en falso en 2 000 operaciones es ≈ 5 % con λ = 0,12 y ≈ 3,5 % con el adaptativo (cota de Ville: ≤ α). En días, n/N̂: 130 operaciones son ≈ 6 500 días con N = 0,02/día (ADR-0003:13-15, sin verificar) y ≈ 8 días con N = 16/día (ADR-0016:27-29). | Sin medir: no existe estimador. |
| 37 | EV neto OOS > 0 en las dos mitades del forense (candidata c). | Las mitades se lanzan a mano con FORENSIC_START_TICK (src/bin/audit_forensic_backtest.rs:396-409), y FORENSIC_JSON da agregados sin R ni IC (:1040-1059). | EV > 0 en las dos mitades (MEMORIA:4098-4100), con IC por bloques positivo (docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:376). El tamaño de bloque no tiene origen. | EV neto ≤ 0, o un IC que incluye 0, en una de las mitades. | Bloqueado por datos externos. El histórico, de motores anteriores, pierde en las dos mitades (INFORME_FORENSE_MAESTRO.md:6171-6180). |
| 38 | El vigilante Ville post-promoción tiene falsas alarmas ≤ α. | Divide por la EWMA de riesgo y recorta R a [−1, 1] (crates/evolution-engine/src/online_daemon.rs:1130-1143). Degrada si la e-riqueza cae ≥ 50 % (:1205-1209). | El 0,50 no tiene origen: Ville acota sup M_t bajo H0, no su caída. | Con p = 0,40 y +2,25 R/−1,10 R, la media recortada es −0,20 R. Simulación con las reglas del código: rollback en una mediana de 24 observaciones con ese edge, 25 sin edge y 9 con +0,2 R simétrico. | Sin medir: cálculo propio (amplía LC-F2-07). |
| 39 | El kill por EWMA del t descriptivo tiene un ARL0 conocido. | crates/evolution-engine/src/online_daemon.rs:1323-1326 (pérdidas constantes, n ≥ 25) y :1335-1343 (EWMA < −1,50); también corre en prod. | −1,5 sin origen: el código dice que es una política y no un p-valor (:1333-1334). | ARL0 medido sobre retornos sin edge, y disparo con el edge de la referencia. | Sin medir. |
| 40 | La falsación se mide con el mismo freno que actúa en vivo. | El forense dice «Paridad 1:1» (src/bin/audit_forensic_backtest.rs:3, :50). Aplica el veto de entrada d*, porque ejecuta GodEngineCore (:147), que llama a evaluate_quantum_order (lib.rs:7497-7499) con el veto de crates/risk-engine/src/lib.rs:336-343. Le falta sólo el freno inmune del host, con su cierre de posiciones y el latch (src/bin/god_engine.rs:1696-1711, :1802-1818): el grep no encuentra drawdown_maximo, drawdown_de_falsacion ni kill_switch en el forense ni en crates/backtest-engine/src/lib.rs. | La regla de la fila 11. | Forense con riesgo medido y una caída del 6 % desde el pico: el host habría aplanado y el forense sigue operando. | Comprobado por lectura: no se cumple. |

### 9.2 Contradicciones y hallazgos nuevos

Cada contradicción abre un hallazgo en la fase dueña del código o del documento. C-F0 no arregla código.

- **LC-F0-01 (C-F0).** Hay dos doctrinas de la meta aceptadas a la vez. La RULE growth_over_wr (.agents/AGENTS.md:9-10) y ADR-0016:12-18 piden duplicar cada 3 días. ADR-0003:19-22 y ADR-0003:37-38, MEMORIA:4098-4100 y el dueño piden maximizar g sin número fijo. Los dos ADR figuran como «Aceptado» (docs/adr/README.md:7, :20), y ADR-0016 no cita ADR-0003. La convención exige un ADR nuevo que reemplace al anterior con enlace en los dos sentidos (docs/adr/README.md:26-27). §1.4 de este plan también queda superado.
- **LC-F0-02 (C-F0).** Umbrales de acierto y caída sin origen en las reglas. La RULE supone un WR del 60-80 % (.agents/AGENTS.md:10), pero con b = 2,25 un WR del 60 % equivale a +0,95 R, y ADR-0016 trabaja con p = 0,40. El workflow .agent/workflows/geminidetailedworkflow.md:19 pide Sharpe > 2, DD < 1,5 % y WR > 55 %. A ½ Kelly, P(DD ≥ 1,5 %) = 0,985³ ≈ 0,96.
- **LC-F0-03 (C-F0 y C-F10).** Tres cifras de operaciones/día. ADR-0003:13-15 da ~10, que sale de crates/risk-engine/tests/compounding_audit.rs:95-101 suponiendo que todas las operaciones ganan +2,5 %. ADR-0016:27-29 da ~16. El plan cuántico da 89 o 51 con p = 70 % (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:186-194), sin medición. Sólo la de ADR-0016 es una exigencia de edge.
- **LC-F0-04 (C-F0 y C-F11).** Cifras de ADR-0016 y §30.0 sin premisa vigente.
  - ADR-0016:19 y ADR-0016:27 dicen que b = 2,25 es neto de fricción, pero el gen `tp_rr_ratio_btc` = 2,25 es bruto (config_dir/genomes/prod/active.json:35), igual que el RR del plan cuántico (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:175). Con f = 11,6 pb y SL 55 pb, b_net = 1,68: +0,30 R exige p = 0,48 y el punto muerto sube a 0,37.
  - El «~0,22 %» de ADR-0016:31-32 y el 0,216 % de docs/PLAN_MAESTRO_SINCRONIZACION.md:1079-1082 salen del tope de 55 pb que D2 retiró (crates/risk-engine/src/lib.rs:915-923).
  - «Kelly para m = 0,1 R es 10 %» (docs/PLAN_MAESTRO_SINCRONIZACION.md:1091) supone s = 1 R; con b = 2,25 binario es 4,4 %.
- **LC-F0-05 (C-F4).** D1 frente a los techos micro (nota A1): f ≤ 1,0·SL, así que el 6,7 % de D1 sólo cabe con SL ≥ 6,7 %. Además, contra ADR-0016:56-61: el código infla hasta la mínima (crates/risk-engine/src/lib.rs:1298-1306); `max_leverage` autoriza la orden mínima sin edge medido en régimen micro (crates/risk-engine/src/kelly_envelope.rs:386-390), contra su propio doc (:335); y mientras n < 30 el host fija boot_lev = round(nocional/margen).clamp(1, 10), o 5 sin margen y con cap ≤ 50 (src/bin/god_engine.rs:4064-4077).
- **LC-F0-06 (C-F4).** El dimensionado micro usa z = 0,85 (src/bin/god_engine.rs:4026; crates/backtest-engine/src/booktick_replay.rs:728), contra «el mismo z del sistema», 1,64 (ADR-0016:47; crates/risk-engine/src/evidence.rs:59).
- **LC-F0-07 (C-F4).** El apalancamiento depende de la confianza (crates/risk-engine/src/lib.rs:1268-1270; crates/risk-engine/src/leverage_matrix.rs:327-336) y sube cuando el capital baja (crates/risk-engine/src/leverage_matrix.rs:219, :237), contra ADR-0016:54-55 y contra la supervivencia que pide el dueño.
- **LC-F0-08 (C-F4).** El host y el riesgo no comparten semántica, aunque ADR-0016:107, docs/PLAN_MAESTRO_SINCRONIZACION.md:1362 («exactamente la misma cota») y el ledger docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:214 («Arreglado») lo afirman. Con el gen de producción 0,5 (config_dir/genomes/prod/active.json:9), el host salta al 50 % sin riesgo medido y al ≈ 4,2 % con r = 0,5 % y q = 0,5; el riesgo, al 63,2 % en los dos casos (fila 11). La divergencia existe desde el arranque. Amplía LC-F4-10.
- **LC-F0-09 (C-F4).** d* se aplica desde el pico móvil, pero la fórmula es la de caer desde W₀ (crates/risk-engine/src/drawdown.rs:177-181). Como prueba no tiene nivel α (fila 35). Hay que medirla desde un W_ref fijo por época o sustituirla por un e-proceso.
- **LC-F0-10 (C-F4; el rearme del host, C-F5).** El piso de ADR-0016:99-103 no sobrevive a un reinicio. Sin posiciones, el veto es absorbente, y el latch del host sólo se rearma reiniciando (src/bin/god_engine.rs:1802-1818). El reinicio fija el pico en el capital ya caído y borra la falsación. Amplía LC-F4-10.
- **LC-F0-11 (C-F4).** Dos pisos de supervivencia en el mismo crate: clamp_ruin conserva el 5 % tras la racha esperada (crates/risk-engine/src/ruin.rs:19-30; crates/risk-engine/src/kelly_envelope.rs:27) y d* el 36,8 %. ADR-0016:50 llama a clamp_ruin «tope de ruina vigente».
- **LC-F0-12 (C-F4; el plan maestro, C-F11).** `check_drawdown_limit` conserva 0,75 y 0,85 (crates/risk-engine/src/guard.rs:44-45), y docs/PLAN_MAESTRO_SINCRONIZACION.md:1299 sigue describiendo el veto como lerp(…, 0.85, micro_w) con 1,95 USD, contra ADR-0016:110.
- **LC-F0-13 (C-F4).** El tope B3.24 recorta el stop sin acortar τ (crates/risk-engine/src/tp_sl.rs:388-402): con rr_mínimo = 1,5 + f/(0,4·SL) (crates/quantum-arena/src/genome.rs:2358, w = 0,40), si SL > 5·f la orden sale con 0,75·SL_τ + 1,25·f. Es lo que ADR-0016:67-81 descarta.
- **LC-F0-14 (C-F4).** El veto de primer toque usa un 0,88 sin origen (crates/risk-engine/src/lib.rs:972) y, sin deriva, veta todo RR > 7,33, que la banda del gen [1, 10] permite (crates/quantum-arena/src/genome.rs:1826). Amplía LC-F4-04.
- **LC-F0-15 (C-F4).** La exención de sonda con el prior 0,55 (crates/risk-engine/src/lib.rs:1187-1191) nunca rechaza. Con la geometría de compute_tp_sl, TP ≥ 1,5·SL + 2,5·f, así que 0,55·TP − 0,45·SL ≥ 0,375·SL + 1,375·f > f. El gate EV no actúa en las 5 primeras operaciones de cada moneda, y como trade_count no se persiste (crates/quantum-arena/src/state.rs:48), vuelve a pasar tras cada reinicio. Amplía LC-F4-09 y LC-F4-19.
- **LC-F0-16 (C-F4).** El plan cuántico pide que la banda pague la fricción «más el spread» (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:327), y la horquilla no entra en roundtrip_friction (crates/risk-engine/src/tp_sl.rs:279-282), que usan el gate y QO-586. Sí entra en dos puertas aparte, con umbrales sin la fricción completa o sin origen: D-755 (lib.rs:5427-5454) y SeniorEjecucion (crates/metacortex-engine/src/consejo_seniors.rs:741-747) (fila 18).
- **LC-F0-17 (C-F7; ville_e_process.rs, C-F2).** El vigilante post-promoción no puede ver el edge de D1. Recorta R a [−1, 1] (crates/evolution-engine/src/online_daemon.rs:1143; crates/risk-engine/src/ville_e_process.rs:106), y su unidad mezcla fracción de capital con `scalp_sl_base`, que es fracción de precio, y un 0,02 sin origen (crates/evolution-engine/src/online_daemon.rs:1130-1142). Lo que observa son deltas de PnL por moneda, no operaciones (:1085-1092). Amplía LC-F2-07.
- **LC-F0-18 (C-F7).** El panel no tiene lector, contra ADR-0003:32-34 (fila 25). Tampoco tiene costes, capacidad, correlación entre estrategias, estabilidad OOS ni G72 (crates/backtest-engine/src/metrics.rs:42-68), que piden el dueño y docs/PLAN_MAESTRO_SINCRONIZACION.md:889-894.
- **LC-F0-19 (C-F7).** El replay mide MaxDD y CAGR sobre la caja (crates/backtest-engine/src/booktick_replay.rs:593-606), contra docs/PLAN_MAESTRO_SINCRONIZACION.md:254-256. Además, el span del OOS incluye el contexto IS (src/bin/evolution.rs:549-562; RA-OOS-F02, docs/AUDITORIA_RAIZ_REVALIDACION_2026-10-04.md:1112), y en modo libro `c1.or(c2)` deja fuera del panel el segundo cierre de un tick (crates/backtest-engine/src/booktick_replay.rs:551, :573), aunque su PnL llega al capital final (:606). Con qué frecuencia pasa, sin medir.
- **LC-F0-20 (C-F7; el host, C-F5).** `fees_est` y `gross_pnl` del replay (crates/backtest-engine/src/booktick_replay.rs:300-310, :575-579) y `total_fees` del host (src/bin/god_engine.rs:3871-3873, :3888) usan maker + taker, 7 pb, cuando la cuenta paga taker en las dos piernas.
- **LC-F0-21 (C-F7; continuous_evolution_backtest.rs, C-F9).** docs/PLAN_MAESTRO_SINCRONIZACION.md:1129-1132 exige DSR, y promueven sin él: src/bin/evolution.rs:638, src/bin/walkforward_evolver.rs:343-361, la cosecha de ShadowForest (crates/evolution-engine/src/random_forest.rs:252-260), src/bin/evolver.rs:674 y crates/backtest-engine/src/bin/continuous_evolution_backtest.rs:992.
- **LC-F0-22 (C-F7).** src/bin/evolution.rs no usa la aptitud única (crates/evolution-engine/src/fitness.rs:93): su score es ln(crecimiento)·10 000 con penalizaciones de −10 000, −200 000 y −dd·200 000 (src/bin/evolution.rs:386-424). Además imprime «Goal: 2.0x» (src/bin/evolution.rs:618-629), y src/bin/backtest_windows.rs:234-256 compone una tasa diaria aritmética contra «META 100%/3d», contra ADR-0003:37-38.
- **LC-F0-23 (C-F7).** Hay al menos nueve definiciones de «Sharpe» y ninguna está en la unidad de la meta: crates/backtest-engine/src/metrics.rs:189, crates/backtest-engine/src/booktick_replay.rs:615, crates/backtest-engine/src/lib.rs:497-498, src/bin/audit_forensic_backtest.rs:915-925, crates/risk-engine/src/selection_stats.rs:76, crates/evolution-engine/src/online_daemon.rs:1314, crates/audit-engine/src/auditor_interno.rs:126-127, crates/evolution-engine/src/lib.rs:461-476 y crates/evolution-engine/src/polars_evolver.rs:124. Hay además dos deflactados: crates/risk-engine/src/selection_stats.rs:156-165 y crates/evolution-engine/src/anti_bias_governor.rs:12-60 (fila 31). `shadow_sharpe_ratio` guarda un t studentizado (crates/evolution-engine/src/online_daemon.rs:1314; crates/evolution-engine/src/return_evidence.rs:7-8), y el «sharpe IS» de la promoción (src/bin/evolution.rs:643-648) es el del replay, sin anualizar (crates/backtest-engine/src/booktick_replay.rs:607-616).
- **LC-F0-24 (C-F5).** La cosecha de ShadowForest imprime la aptitud logarítmica como «+$X extra» (src/bin/god_engine.rs:4896-4897), y su «OOS» es la misma corrida (crates/evolution-engine/src/random_forest.rs:238-239). Relacionado con LC-F7-02.
- **LC-F0-25 (C-F5).** La física cobra el TP como maker (lib.rs:3631) y en vivo es TAKE_PROFIT_MARKET (crates/execution-engine/src/executor.rs:3091), contra D-645 (crates/risk-engine/src/lib.rs:814-834). T-06 sigue abierto.
- **LC-F0-26 (C-F5; la pregunta 3, C-F0).** Hay tres tasas taker. La física usa siempre 0,0005 (crates/god-engine-core/src/reality_physics.rs:31-32; lib.rs:1145), el gate la de la API (src/bin/god_engine.rs:1233-1251) y B3.11 afirma 0,04 % (:1238-1242). El «≥ 10 pb» de la pregunta 3 de C-F0 sólo vale si la tasa es 5 pb; con 4 pb son 8 pb.
- **LC-F0-27 (C-F3).** «Fricción unificada» (TABLERO.md:115) frente a las seis fricciones de la fila 21, contando la de la física (crates/god-engine-core/src/reality_physics.rs:109-111, :246-248). La guardia de fuente sólo cuenta llamadas (crates/risk-engine/src/tp_sl.rs:800-819).
- **LC-F0-28 (C-F3).** El gen de latencia calibra la media según ADR-0001 (docs/adr/ADR-0001-latencia-lognormal-fills.md:21-22) y crates/risk-engine/src/tp_sl.rs:204-206, y la mediana según el comentario de lib.rs:7879.
- **LC-F0-29 (C-F1).** V-LOGIC-006 declara su umbral «medido», con latencia «muestreada» (crates/risk-engine/src/veto_registry.rs:208-219). El gate usa el gen estático (crates/risk-engine/src/lib.rs:798-802), y no hay fills en el repo.
- **LC-F0-30 (C-F9; la función única de umbral, C-F4).** El forense dice «Paridad 1:1 con Producción» (src/bin/audit_forensic_backtest.rs:3) y no replica el freno del host (fila 40). Sus mitades miden operaciones que en vivo no llegarían a hacerse tras el kill.
- **LC-F0-31 (C-F2).** Hay una segunda CVaR sin llamadores (crates/god-engine-core/src/math_kernels.rs:646-663), con índice por floor, sin mínimo de muestra y 0 con lista vacía, contra ADR-0003:27-28 y crates/backtest-engine/src/metrics.rs:205-231.
- **LC-F0-32 (C-F8).** La telemetría imprime un plazo de duplicación con 50 operaciones/día fijas (crates/audit-engine/src/telemetry.rs:117-139), contra ADR-0003:37-38.
- **LC-F0-33 (C-F11).** Documentos con la meta o los invariantes antiguos:
  - ARCHITECTURE.md:22, :45 y :51 siguen con «+100 % cada 3 días» como única métrica, desautorizados sólo por la nota de :1;
  - la Parte II del plan cuántico (docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:164-178) y su criterio R0 (:337) mantienen 55 pb, 3 USD, 5×, 2 posiciones y 4 pb maker;
  - docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:154 dice que `ex_post_metrics` sólo se llama desde tests, y en aa716edf la llama el replay.

  Amplía LC-F11-01.
- **LC-F0-34 (C-F0).** La «mediana OOS +0.008» de la familia honesta (TABLERO.md:23-24) no declara unidad, ventana ni estimador, así que no se puede comparar con g ni con m. Sin verificar.
- **LC-F0-35 (C-F0 y C-F11).** Después de aa716edf, #717 (AGY, ae6d51a2) añadió la §38 del plan maestro y `docs/PLAN_MAESTRO_EXPONENCIAL_CUANTICO_2026-10-10.md`. Los dos vuelven a fijar «+100 % cada 3 días» como «Meta Financiera Fundamental», con el stop acotado a 55 pb, 5× y 2 posiciones, y derivan 89 operaciones/día con p = 70 % y RR ≥ 2,25. Ese ejemplo supone +0,034 USD por operación con 0,030 USD de riesgo, es decir ≈ +1,12 R, casi cuatro veces la referencia de ADR-0016 (+0,30 R), sin fills ni tape que lo midan. Es una tercera fuente de LC-F0-01 y LC-F0-03, y repite los invariantes que LC-F0-33 ya da por antiguos (el tope de 55 pb lo retiró D2). Se cita sin línea porque no existe en aa716edf.

### 9.3 Lo que no está medido

- **Bloqueado por datos externos.** Los tapes y los fills viven en el PC del operador, y Binance responde 403 desde la nube. Queda bloqueado:
  - g y el edge en R fuera de muestra, p_LCB real y N en tape real;
  - la tasa taker real, la fracción maker, la fricción y el deslizamiento realizados, y los costes de la cuenta (IncomeTotals, fila 17);
  - las dos mitades del forense, los valores del panel y la cobertura conformal sobre tape.
- **Con test posible hoy sin tapes, aún sin ejecutar:**
  - comprobadas por lectura, donde el test confirmaría lo leído: fixtures de dimensionado (filas 6, 7 y 9), host igual a riesgo (11), física igual a vivo (20), una sola fricción (21) y MaxDD con flotante (27);
  - sin medir: pico tras reinicio (12), tope de ruina frente a d* (13: sólo la comparación determinista; el Monte Carlo con R medida está bloqueado por datos externos), spread en la fricción (18), `fees_est` (24), e-proceso refutador (36) y ARL0 de Ville y del kill por EWMA (38 y 39).
- **Sin estimador en el código:** G72, la cota inferior de g, R por operación ejecutada, el f_R de D1 (½ Kelly con b_net; el del host es Kelly pleno con b de una EWMA en USD, fila 6), el deslizamiento de entrada, la latencia de envío a fill, los costes en el panel, una capacidad medida, la correlación entre estrategias, las dos mitades automatizadas, la fiabilidad de Platt y un e-proceso que refute la meta.

### 9.4 Cierre de C-F0

C-F0 se puede cerrar cuando:
- las 40 filas tienen estado, como ahora; las bloqueadas por datos externos, las «sin medir» y las «comprobadas por lectura» se aceptan. «Comprobado por lectura» es un estado propio y distinto de «medido»: describe propiedades del código en aa716edf, no valores de mercado. «Medido» se reserva para lo ejecutado o medido sobre datos, y ninguna fila lo tiene hoy;
- dos escépticos revisan la tabla y las cifras derivadas;
- cada LC-F0-nn queda abierto en su fase dueña;
- el dueño fija la reformulación en un ADR nuevo que reemplace la meta del Contexto de ADR-0016, con enlace en los dos sentidos (docs/adr/README.md:26-27), una adenda a la RULE growth_over_wr y la línea «Meta» de §1.4 al día.

C-F0 no se puede cerrar:
- con una fila sin estado, o con un umbral sin origen escrito como si lo tuviera;
- usando la cota de referencia como puerta;
- llamando «medido» a lo que sólo se leyó en el código.

Cerrar C-F0 no autoriza a operar, entrenar, promover genomas o modelos ni cambiar el golden o el trinquete. El edge fuera de muestra sigue sin medir.
