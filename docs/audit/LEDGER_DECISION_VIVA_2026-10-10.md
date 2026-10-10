# Ledger del camino de decisión viva — QS-R1 (2026-10-10)

> Claude, sesión «elegant» (prefijo QS-n). Base `main` 7387b935 (#709) más la
> rama del PR #30. Contrato que se aplica: G-TEO
> (`docs/PLAN_MAESTRO_SINCRONIZACION.md` §30.1).
>
> Método: tres inventarios de sólo lectura en paralelo (consejo, núcleo,
> riesgo). Después, verificación propia de cada hallazgo marcado
> **VERIFICADO** (lectura del código y, cuando se indica, contrato
> ejecutable). Lo marcado *inventario* es la lectura del agente sin
> re-verificar línea a línea: sirve para priorizar, no para cerrar.
>
> Alcance: el camino que decide una ENTRADA viva. No cubre la gestión de
> posiciones abiertas (trailing, cierres), el demonio de evolución ni el
> replay, salvo donde un factor del camino vivo se alimenta de ellos.
> «Evidencia» significa un test que ejercita el factor. Ningún factor del
> consejo tiene evidencia con datos reales o reproducidos: sólo cargas
> sintéticas. El T-1 recorre el consejo sobre un fixture sintético y mide
> expresividad genética, no rentabilidad.

## 1. Cómo se consume el consejo

`god-engine-core/src/lib.rs:7577-7606`:
`aprobado = consensus.approved && signo(final_signal) == lado de la orden`.
Después viene un gate ML independiente (`lib.rs:7633-7685`). Aguas abajo
nadie consume la MAGNITUD de `final_signal`, el porcentaje de consenso ni la
penalización de override: sólo cuenta la aprobación (*inventario*).

## 2. Hallazgos verificados del consejo

| id | Dónde | Qué pasa | Efecto | Prueba | Dueño propuesto |
|---|---|---|---|---|---|
| **C-22** | `consejo_seniors.rs:1170-1181`, `1297-1301` | `final_signal` promedia TODOS los asientos con señal. Volatilidad (0,9), Riesgo (1,5) y Ente (1,0) votan `intended_direction`. La aprobación exige consenso direccional ≥ 0,35 y signo(`final_signal`) = lado. El comentario 1186-1192 los saca del consenso para no «rubber-stampear», pero siguen decidiendo el signo. | En la rejilla del diagnóstico, **68 de 480 largos aprobados (14 %)** tienen los asientos direccionales netos EN CONTRA. Peor caso: señal direccional −0,53 con `final_signal` +0,56 (OBI −0,9, espectral −0,6, ML 0,30). | `crates/metacortex-engine/tests/qs_r1_c22_direccion_propia_diagnostics.rs` (OPEN, `#[ignore]` con motivo; falla al ejecutarse con `--ignored`) | Claude QS reserva el arreglo mínimo: la aprobación exige que la señal DIRECCIONAL tenga el signo del lado. Cambia conducta ⇒ T-1. |
| **C-W** | `consejo_seniors.rs:669, 713, 919, 1003-1005` | `SEAT_WEIGHT_FLOW` (1,2) está en el asiento Causal, cuya señal es siempre 0: sin efecto. `SEAT_WEIGHT_META` (1,5) está en **Riesgo**, un modulador que vota el lado pedido, lo que amplifica C-22. El asiento Microestructura (flujo) pesa 1,0. `SEAT_WEIGHT_DEVIL` (3,0) va en el Auditor, cuya señal es 0: sin efecto. | Los pesos «de diseño» no recaen sobre los asientos a los que su nombre se refiere. | lectura | dueño del consejo |
| **C-10** | núcleo `lib.rs` ~7558 (#708) y `prospect_theory.rs:129-185` | **RESUELTO en Ola Ω73 (AGY)**: se erradicó el uso de `ml_prob_pure` como entrada de masa; ahora `compute_crowd_net_prospect_pressure` evalúa el sentimiento real de masas (`crowd_ls_ratio` y liquidaciones). Cumple anti-simetría exacta $P(1/LS) = -P(LS)$ y simetría espejo idéntica en `modulation_factor` (verificado con `test_prospect_pressure_mirror_symmetry_in_god_engine` y `test_crowd_prospect_pressure_mirror_symmetry`). | Erradicado el sesgo contra cortos; el consejo evalúa con perfecta simetría espejo. | `qs_r2_simetria_espejo_contract.rs` (2/2 OK) y `prospect_pressure_integration_contract.rs` (3/3 OK) | AGY (CERRADO Ola Ω73) |
| **C-10b** | núcleo `lib.rs` ~7560 | **RESUELTO en Ola Ω73 (AGY)**: se eliminó el `.max(0.0)` en la lectura de registro; ahora evalúa `reg_p.is_finite()` preservando signos negativos de pánico sin clobbering. | Lectura firmada real activa en vivo. | `test_prospect_pressure_mirror_symmetry_in_god_engine` | AGY (CERRADO Ola Ω73) |
| **C-16/18** | `consejo_seniors.rs:385-394, 871` | El consejo sólo recibe P(TP del LARGO antes que SL). En Teleonomía, `ml_edge = ((p − base)·2).clamp(−1, 1)` con base ≈ 0,25 expresa como máximo **−0,5 para un corto** frente a +1,0 para un largo. Además, la base es la del bosque y la probabilidad es la del ensamble bosque⊕NN (*inventario*: `lib.rs:4519` frente a `4703-4717`). | Convicción ML estructuralmente más débil para cortos. | lectura | GLM (Línea B): servir P(TP del CORTO) o un modelo de tres clases; el consejo usa la del lado juzgado |
| **C-13** | `consejo_seniors.rs:698` | Veto de Riesgo si DD > 0,95 − 0,10·s, es decir un umbral entre 0,85 y 0,95. El suelo de supervivencia de 3 USD sobre 13 USD es un DD de 76,9 %. | Veto inalcanzable: el suelo actúa antes. | lectura | dueño del consejo |

## 3. Hallazgos del consejo pendientes de verificación (*inventario*)

- **C-02**: Reynolds y la fracción laminar salen del ÚLTIMO evento, no de la
  EWMA que el motor también calcula (`navier_stokes.rs:189-204`). Un solo
  tick ruidoso baja la confianza del flujo a ×0,40 y sube el slippage
  efectivo ×1,5. Sin evidencia OOS (zona AGY).
- **C-07**: `whale_burst_z` y `spoof_score` sólo se escriben en evento y no
  decaen nunca. Lo tomó la otra sesión Claude (**CL-49**, PR #29); no se
  duplica.
- **C-09**: `macro_staleness_ms = u64::MAX` antes del primer fetch macro (y
  en replay sin historia macro) fija el factor de Ente en ×0,40.
- **C-12**: el asiento «Causal / do-calculus» es VPIN con umbral 0,75 fijado
  en el llamador (`lib.rs:7502`). El nombre no corresponde al cálculo.
- **C-15**: el «slippage» del asiento de ejecución es la mitad del spread
  + 0,5 pb, sin tamaño, profundidad ni latencia; su veto exige spreads
  ≳ 46–69 pb.
- **C-17**: la confianza de Metacognitivo no depende de la fuerza de su
  señal y cae al suelo 0,05 con ≈ 9 % de DD.
- **Información contada varias veces**:
  - OBI: C-03, C-17 y la exención de ruptura de Causal.
  - Espectro: C-04, C-17, C-18.
  - ML: C-10, C-16, C-17, C-18.
  - VPIN: C-12, C-19 y Teleonomía.
  - DD: C-13, C-14, C-17.
  - ATR: C-05, Prospect y Reynolds.
  - No es un fallo en sí, pero cada doble conteo debe estar justificado.

## 4. Núcleo (camino señal → intención)

Orden vivo (*inventario*, `god-engine-core/src/lib.rs`):
1. convicción de rama;
2. `puertas_del_continuo` (una vez por banda);
3. arbitraje de bandas;
4. freno del bosque;
5. escudo macro;
6. condicionamiento espectral;
7. vetos de whiplash, racha, OBI y CVD;
8. ranura;
9. calibrador;
10. `evaluate_quantum_order` (riesgo, l. 7424).

Sólo DESPUÉS van el consejo, el gate ML y `free_cap`. Son vetos sobre una
orden ya dimensionada: ninguno la redimensiona.

### 4.1 Verificados

| id | Dónde | Qué pasa | Efecto | Dueño propuesto |
|---|---|---|---|---|
| **K-06** | `lib.rs:1563-1579` (`puertas_del_continuo`) | Largo: `d = (p − b)·2`; corto: `d = (b − p)·2`, con b ≈ 0,20–0,33 (base del bosque). El veto es `d < −0,80`; la penalización `×(1 + 0,5d)` con suelo 0,20; el impulso `×(1 + 0,5d)` con tope 0,99. | Con b = 0,25, el largo vive en d ∈ [−0,5, +1,5] y el corto en [−1,5, +0,5]. **El modelo no puede vetar nunca un largo**, y veta un corto en cuanto p > b + 0,40. Impulso máximo: largo ×1,75, corto ×1,25. Además el impulso va DESPUÉS de `conviccion_de_rama`: re-infla la convicción de una rama con desventaja demostrada (abierto desde la auditoría del 2026-09-25, punto 2). | GLM (semántica de `p`) + dueño de la puerta (D-743) |
| **K-23** | `lib.rs:7663-7672` (gate ML B3.18) | Largo `p ≥ b + lift_L`, corto `p ≤ b − lift_S`. Con el genoma desplegado (0,62/0,42) los lifts son 0,12 y 0,08, modulados ×[0,7, 1,3] por acuerdo espectral y ×[0,7, 1,3] por Hurst, y recortados a [0,01, 0,30]. | **Los cortos son imposibles si lift_S ≥ b**: con el genoma desplegado cuando b ≤ 0,135, y en los extremos del gen cuando b ≤ 0,30. El comentario dice que H ≈ 0,5 sube el lift; el código es neutro en H = 0,5 (*inventario*). | GLM / dueño del gate |
| **C-22** | consejo (ver §2) | El consejo aprueba con el signo de una media que incluye moduladores que votan el lado pedido. | 14 % de los largos aprobados en la rejilla tienen asientos direccionales netos en contra. | Claude QS (reservado) |

**Raíz común de K-06, K-23, C-10, C-16/18 y K-26.** El sistema sólo sirve
P(TP del LARGO antes que SL). Esa cantidad no es «la probabilidad de que
suba» y no tiene espejo: su neutral es ≈ 1/(1 + RR) ≈ 0,31 con RR 2,25, no
0,5. Cada consumidor que la lee como dirección introduce un sesgo de lado.

El arreglo de raíz es de modelado (Línea B): servir también P(TP del CORTO)
(o un modelo de tres clases), y que cada consumidor use la del lado que juzga
medida contra SU base. Mientras tanto, la transformación mínima coherente es
comparar `p` con `b` en log-odds (`logit p − logit b`). Hace la escala
comparable en los dos sentidos, pero no sustituye a la probabilidad del corto.

### 4.2 Pendientes de verificación (*inventario*)

- **K-02**: la convicción por rama compara el acierto con 0,5 (Wilson).
  - Con RR ≥ 2,25 el punto de equilibrio es ≈ 0,31 más la fricción.
  - Una rama que gana el 40 % con EV > 0 se degrada.
  - Una rama sin historial recibe un suelo de hasta 0,90, más que una
    demostrada (≈ 0,55).
- **K-09**: el freno del bosque lee la clase `was_profitable` del bosque
  online (0/1, sin lado) como «sube». Es asimétrico bajo espejo.
- **K-10/K-12**: la τ se modifica DESPUÉS de la sonda de fricción de QO-586,
  y la paridad se pierde.
  - K-10: ×2 por cada 0,02 de H, con OTRO estimador de Hurst.
  - K-12: media geométrica con la τ resonante.
- **K-12**: el multiplicador espectral llega hasta ×1,35 y puede deshacer el
  freno del bosque. Sus recortes son literales sin calibrar.
- **K-18**: la confianza entra dos veces en el tamaño (calibrador Platt y
  `1 + (conf − min_conf)⁺` del riesgo).
- **K-24**: `free_cap` es un veto binario que nunca redimensiona.
  `SpectralRegimeField::margin_multiplier` no tiene llamador vivo; el
  bloque lo duplica en línea.
- **K-26**: el conformal acepta p (largo) o 1 − p (corto), espejado en 0,5
  cuando el neutral de p es b ≈ 0,25.
- **K-27**: la rama 1 corta admite `micro_trend ≤ +0,00003` y su espejo
  largo exige ≥ 0. Las ramas 9 y 10 usan el literal ±0,24.
- **Doble conteo**:
  - el ML entra ≥ 7 veces (compuesto, D-472, puertas, ramas, B3.18, tres
    asientos del consejo y Prospect);
  - coherencia y marea, 4–5 veces.

## 5. Riesgo (admisión, tamaño, apalancamiento)

### 5.1 Verificados

| id | Dónde | Qué pasa | Efecto | Dueño propuesto |
|---|---|---|---|---|
| **R-15** | `risk-engine/src/lib.rs:937-954`, `tp_sl.rs:469-489` | **RESUELTO en Ola Ω73 (AGY)**: se calibró la deriva dimensionalmente con $\tau_{\text{ratio}} = (30\text{s}/\tau)^{1/2}$ acorde a la física de decorrelación espectral temporal. Previene que brackets amplios en escalas de horas sobre-acumulen deriva constante un-física y veten señales válidas con desacuerdos mínimos, mientras preserva el veto estricto ante mareas genuinamente adversas ($-0.95$). Se publica telemetría viva de `p_hit_sl_first` en el registro omnisciente y se documentó en `V-LOGIC-008`. | Eliminado el castigo artificial a escalas largas; telemetría viva observable en registro. | `test_r7_r2_a1_analytical_first_hitting_gate_contract` (4/4 OK) | AGY (CERRADO Ola Ω73) |
| **R-20** | `lib.rs:1262-1286` | Si el margen de Kelly no alcanza el nocional mínimo, sube el apalancamiento y después **infla el margen** hasta `(MN + 0,1)/L·1,0005`. | Con 13 USD toda orden con k > 0 es la orden mínima: el tamaño es binario y Kelly/confianza no gradúan nada. No es un defecto numérico (el exchange no admite menos), pero significa que en micro el único control es operar o no. La viabilidad (R-19) acota la ruina. | — (hecho estructural) |
| **R-18** | `lib.rs:1112-1172` | Sin historial (n = 0) el gate EV no se evalúa. Con n < 5 se exime si `0,55·TP − 0,45·SL > fricción`, lo que se cumple siempre con RR ≥ 2,25. El gen `ev_fee_multiplier` vale 1,0 en los tres genomas activos, por debajo del suelo 1,05 del clamp, así que no tiene efecto. | Cada moneda opera sus 5 primeras veces con un prior 0,55 no medido: hasta 150 operaciones en 30 monedas sin cota explícita de exposición total en sonda. Es política (D-751b/Ω46), no accidente. | dueño (presupuesto de sonda) |
| **R-03** | `lib.rs:285-314` | Umbral de drawdown = `lerp(dd_max_medido, 0,85, micro_w)`. Con 13 USD, micro_w = 1 y el umbral vale exactamente 0,85. | Permite caer hasta 1,95 USD. **El «suelo de supervivencia de 3 USD (76,9 %)» de los invariantes sagrados no existe en el código**: `SURVIVAL_FLOOR = 0,05` es una fracción dentro del tope de ruina de Kelly. | dueño (§4 decisión 1) |
| **INV-2** | `capital_compounder.rs:37-40` | «Concurrencia máxima 2 posiciones»: `max_concurrent_positions` sólo tiene consumidores en tests. | La concurrencia viva la limitan el margen (tope por orden de 2,60 USD y bruto del orquestador) y las ranuras (3 por moneda), no ese módulo. Los invariantes de la memoria no son todos código. | AGY (documentación) |

### 5.2 Pendientes de verificación (*inventario*)

- **R-09**: el veto de riesgo de grupo usa como riesgo del candidato la EWMA
  de las órdenes PASADAS, no el de esta orden. En micro (r ≈ 0,2 % frente al
  tope 0,25) haría falta del orden de 125 posiciones de la misma apuesta:
  es inerte.
- **R-10**: la cota de Lundberg `ln 20/R` se APRIETA cuando el edge (R)
  crece, y dividir por el apalancamiento máximo del exchange es
  anticonservador. En micro no muerde nunca.
- **R-16**: con Kelly = 0 el apalancamiento sigue siendo > 1.
  - `vol_mult` se calcula y se descarta.
  - El techo micro vale 5,0 en un sitio y 5–6,5 en otro (`lib.rs:1249`).
- **R-17**: el factor micro 0,66/0,62 es un resto del antiguo escalón de
  15 USD.
  - La confianza entra 4 veces en el tamaño.
  - Con el genoma activo (min_conf 0,70), a 13 USD el gate exige ≥ 0,745.
- **R-26/R-27**:
  - el veto de colapso del orquestador sólo actúa sobre largos (D-403 lo
    declara deliberado);
  - el tope por dirección es código muerto;
  - el suelo `.max(0,05)` es inalcanzable.
- **Registro de vetos**:
  - Faltan R-04, R-06, R-07, R-08, la rama IC > 1 de R-10, el bypass micro
    de R-12, R-13, **R-15**, R-16, **R-19**, la inflación de R-20, R-21 y
    el refuerzo por contagio de R-09.
  - Entradas sin código vivo: V-LOGIC-009 (`REJ_SIN_EVIDENCIA` nunca se
    emite), V-TECH-003 y V-RISK-003. V-LOGIC-001 describe otro gate.
  - Descripciones erróneas: V-RISK-001, V-LOGIC-014 y V-LOGIC-015.
- **Cadena de tamaño**:
  1. `k = clamp_ruin(kelly)`;
  2. `× sl(30 s)/sl(τ)` (curva del gen, no el stop real);
  3. `× (1 + (conf − min_conf)⁺)`;
  4. `m = conf·k·C` (MARGEN, no riesgo);
  5. L de la matriz;
  6. rescate de nocional mínimo;
  7. tope de margen por orden;
  8. N = m·L.

  Kelly se aplica en espacio de MARGEN. El riesgo al stop (k·conf·L·SL·C)
  nunca se deriva como objetivo. Ejemplo (C = 13, SL 0,5 %, L 5, conf 0,7,
  k 0,10): la orden se infla a 5,10 USD de nocional y arriesga 0,196 % al
  stop. Kelly en espacio de riesgo habría arriesgado ≈ 7 %.

## 6. Cómo cerrar cada hallazgo

Cada fila se cierra con un contrato que falle antes del arreglo y pase
después, más el T-1 si cambia conducta viva. Un contrato que sólo construye
la carga a mano y llama a la función pura no cierra un hallazgo de cableado
(punto 1 de G-TEO). Los OPEN de este ledger se ejecutan con
`cargo test -p <crate> --test <fichero> -- --ignored`.

## 7. Estado de cierre (2026-10-10, decisiones del dueño en ADR-0016)

El dueño delegó las decisiones de §30.5 del plan («Tú decide»). Quedaron en
`docs/adr/ADR-0016-riesgo-geometria-y-falsacion-de-la-meta.md`, con las
derivaciones.

| id | Estado | Commit / contrato |
|---|---|---|
| **C-22** | Arreglado: la aprobación exige además que la señal direccional neta (asientos con dirección propia) tenga el signo del lado. | QS-C22; `qs_r1_c22_direccion_propia_diagnostics.rs` sin `#[ignore]` (RED 68/480 → GREEN 0). T-1 en el PR. |
| **R-12** | **Corrección de este ledger**: el atajo micro NO admitía un stop bajo el suelo. `compute_tp_sl` ya eleva el stop al suelo; el defecto era otro: con la τ intacta, el stop quedaba más ancho que la dispersión del horizonte. Arreglado: el suelo rechaza en todo capital. | QS-D2; `qs_d2_el_suelo_de_viabilidad_rechaza_en_todo_capital` (RED en micro). |
| **R-13** | Arreglado: se retira el tope micro de 55 pb; la geometría no depende del capital. | QS-D2; `qs_d2_la_geometria_de_la_orden_no_depende_del_capital` (RED: 55 pb frente a 85,25 pb). |
| **R-03** | Arreglado: umbral = min(lerp(dd_max_medido, d*, micro_w), d*) con d* = 1 − 0,05^{1/3} ≈ 0,632. | QS-D3; `qs_d3_drawdown_de_falsacion_derivado` (lib) y `qs_d3_el_veto_de_drawdown_no_pasa_de_la_caida_de_falsacion`. |
| **R-18** | Decidido (D4): una sonda abierta en toda la cartera, orden mínima, ≤ 5 por moneda, sus pérdidas cuentan en D3; el prior 0,55 se mide antes de sustituirlo. | Pendiente (zona núcleo/riesgo, coordinar con la Línea C y el PR #29). |
| **R-20 / cadena de tamaño** | Decidido (D1): ½ Kelly en espacio de RIESGO sobre p_LCB y b neto de fricción; el apalancamiento es consecuencia. | Pendiente (QS-R4: libro contrafactual en sombra antes de activarlo). |
| **Freno del host** | Nuevo: `src/bin/god_engine.rs:1609-1619` usa `drawdown_maximo` sin la cota d*. Hay dos semánticas del cortacircuitos de drawdown. | Petición a la Línea C. |

### 7.1 Verificados después (2026-10-10, tarde): núcleo, ramas de entrada

| id | Dónde | Qué pasa | Arreglo propuesto (lote QS-K, con T-1) |
|---|---|---|---|
| **K-27** | `god-engine-core/src/lib.rs:5821` frente a `:5903` | La rama 1 corta admite `micro_trend <= 0.00003`; su espejo largo (rama 4) exige `>= 0.0`. Deja entrar cortos con microtendencia levemente alcista, contra el propio comentario F8-P9 de la línea anterior. | `micro_trend <= 0.0`, espejo exacto. |
| **K-27b** | `lib.rs:6014`, `:6039` | Las ramas 9 y 10 (rango, estiramiento > 1) exigen `|composite_score| ≥ 0.24` literal; las ramas 7 y 8 del mismo régimen usan `range_thr` (umbral medido × 1,15). 0,24 es el borde inferior de la banda del gen 21 (`tech_threshold`, [0,24; 0,30]): cuando el gen mutaba o el régimen era antipersistente (×1,20), estas dos ramas no lo veían. | `dynamic_tech_thr` (el gen). Se mantiene la relación con 7/8 (`range_thr` = `dynamic_tech_thr` × 1,15): las ramas de estiramiento extremo piden menos score porque ya piden más OBI y más estiramiento. |
| **K-28** | `lib.rs:5967-5974` | Las compuertas de tendencia del régimen de rango usan literales en unidades de precio: `macro_trend ≥ −0,00010` y `secular_trend < 0,0025` (y sus espejos). Es la clase que D-756 eliminó en la rama 2: el mismo literal es un desplome en un símbolo tranquilo y ruido en uno volátil. `z_macro` y `z_secular` ya están calculados en el mismo ámbito (`:4962-4968`). | `z_macro ≥ −Z95` y `z_secular < Z95`, en una función pura `rango_admite_reversion` (espejo exacto por construcción). Se retira `!(h < 0 && s < 0)`, redundante tras `h ≥ 0`. |

Los tres cambian conducta en el núcleo: van juntos en un lote con su T-1 y se
avisan antes en el buzón (zona compartida).

**Resultado (lote QS-K, commit `10ecf0b6`):**
- `god-engine-core --lib`: 173/173, con los tres contratos nuevos
  (`tests_qs_k`): espejo en rejilla, cierre sólo con tendencia
  significativa y guarda de fuente.
- T-1: **PASA 16/144**, con la misma lista y las 145 huellas iguales bit a
  bit a las del árbol del PR #31.
- El fixture no llega a los bordes que QS-K cambia: el T-1 certifica que no
  hay regresión, no ejercita el arreglo. Ejercitarlo pide un fixture con
  rango lateral y microtendencia cerca de cero (pendiente).

### 7.2 Revisión cruzada de Ω69–Ω71 (AGY), 2026-10-10 noche

Sobre `origin/main` 46269a05, leyendo código (sin compilar). REV-1, REV-2 y
REV-5 los verifiqué yo contra el código. El resto es lectura de un
subagente: el propagador está portado línea a línea a un script, sin
ejecutar el Rust. C-10/C-10b (Prospect) ya estaban arriba y no se repiten.

| id | Dónde | Qué pasa | Arreglo propuesto | Dueño |
|---|---|---|---|---|
| **REV-1** HIGH | `risk-engine/tests/veto_logic_contracts.rs:131-137` (V-TECH-002 `spec`) | Usa `coin_id = 999`. `lib.rs:230` sale por `coin_id >= arena.coins.len()` (30 ranuras) con `REJ_FLAT_COIN`; `REJ_SPEC` (`:459`) nunca se alcanza. El test sigue verde si se borra el veto que dice certificar. | `coin_id` dentro del arena sin spec registrado y comprobar la razón (`risk_engine::ultimo_rechazo() == Some(REJ_SPEC)`, QS-R4b, o el delta de `REJECT_COUNTERS_DIR`). | AGY/GLM (registro) |
| **REV-2** HIGH | `veto_logic_contracts.rs:97-110` (V-LOGIC-014 `fee_impact`) | Con `live_taker_fee = 0,50` la fricción de ida y vuelta es ≈ 1,0, el stop viable `f/0,65` ≈ 1,54 y sale antes `REJ_TP_SL_FLOOR` (`lib.rs:887`, también tras D2). `REJ_FEE_IMPACT` (`:1354`) no se alcanza. | Fee que pase el suelo y supere el tope de impacto, y comprobar la razón. Ver REV-11: en 13 USD quizá no exista tal fee. | AGY/GLM |
| **REV-3** MED | `veto_registry.rs:484-525` | `sol_a1_…` sólo busca el texto `fn <nombre>` en el corpus: vale un comentario, un prefijo (`fn foo` casa con `fn foo_bar`) o un test `#[ignore]`, y no mira qué veto alcanza. No puede detectar REV-1 ni REV-2. | Cada entrada del registro declara su `REJ_*`; el test comprueba `ultimo_rechazo()`. | AGY/GLM |
| **REV-4** LOW | `veto_registry.rs:341-349`, test `:112-128` (V-LOGIC-015) | El registro lo describe como orquestador de signal-engine con umbral de gen; el veto real es `PortfolioOrchestrator::allow_trade` en risk-engine con el literal `p_crash ≥ 0,90`. MEMORIA dice «shorts simétricos», pero no hay veto corto espejo (`p_bull ≥ 0,90`) y el caso corto se prueba con `p_bull = 0,01`, que no es el espejo. La contracción continua no se ejercita. | Corregir la ficha; decidir si el espejo existe. | AGY |
| **REV-5** MED | `signal-engine/src/feynman_propagator.rs` (Ω69) | Código muerto: ningún consumidor fuera de tests (`git grep`). `evaluate_for_coin` lee `feynman_coherence`, que sólo escribe un test a mano. El contrato 7 certifica un valor inyectado. | Modo sombra con su contraste OOS antes de cablear, o retirarlo. | AGY |
| **REV-6** MED | `feynman_propagator.rs:199-202` | La fase de la acción es `0,1·L/(1+|z|)`: −0,08 rad con z = 2 y −2,73 rad con z = 10. Un movimiento MÁS fuerte en la misma dirección destruye la coherencia. 16 escalas a +2 y 16 a +10 dan C = 2,65 frente a 32 todas a +2; voto máx. 0,08 frente a 0,76. | La fase no debe depender de |z| así. | AGY |
| **REV-7** MED | `feynman_propagator.rs:210` | Normaliza la coherencia por 32 (C ≤ N_activas por Cauchy-Schwarz). Con las 5 escalas operables alineadas a +2,5: normalizada 0,14, voto máx. 0,117. Misma familia que R7-R4-B-2. | Dividir por el número de escalas activas. | AGY |
| **REV-8** LOW | `feynman_propagator.rs:186` | Una escala no finita se salta, pero la siguiente lee `desplazamientos[k-1]` sin filtrar: NaN se propaga a la suma y la coherencia vale 0 en las 32. El test sólo cubre «todo NaN». | Filtrar también el vecino. | AGY |
| **REV-9** INFO | doc de `feynman_propagator.rs` y MEMORIA Ω69 | La doc dice S = L·Δt_k, pero no hay Δt; la «velocidad» es z_k − z_{k−1} entre escalas, no en el tiempo. MEMORIA llama C = |A|, el código usa |A|²/Σ|ψ|². En ruido N(0,1), C ≈ 0,88, no 0. Los dos tests «físicos» usan vectores constantes. | Corregir la doc. | AGY |
| **REV-10** LOW | `god-engine-core/tests/prospect_pressure_integration_contract.rs` | No construye `GodEngineCore`: no ejercita el cableado de `lib.rs:7558-7573`. Fija `p_crash = 0,05`, cuando la fórmula del núcleo con p_bull 0,62 y liquidación 0,05 da 0,215. «Certificado en el hot path» (MEMORIA Ω70) no lo respalda este test. | Test que pase por el núcleo. | AGY |
| **REV-11** (sospecha) | `risk-engine/src/lib.rs:1185`, `:1354` | En 13 USD: `max_fee_limit` = 0,035; pasar el suelo exige f ≤ 0,65·0,0055 (antes de D2); superar el tope exige lev > 9,8×, y el micro va a 5–6,5×. `REJ_FEE_IMPACT` parece inalcanzable en micro. Falta trazar las ramas de rescate de apalancamiento. **Con D2 (PR #31) el suelo ya no acota f en micro**: un τ largo (stop ancho) paga fricciones mayores, y f·L > 0,035 vuelve a ser alcanzable si el EV lo deja pasar. | Medirlo con `ultimo_rechazo()` en una rejilla (f, τ, capital). | AGY/GLM |

Verificado correcto: función de valor KT (x^0,88; −2,25(−x)^0,88); la
ponderación es Tversky-Kahneman 1992 (MEMORIA dice «Prelec», que es otra);
`modulation_factor` es invariante al espejo; el Lagrangiano y el módulo de
fase son correctos; los tests `flat_coin` e `invalid_input` sí alcanzan su
veto.

### 7.3 Revisión de #710 y #711 (AGY, Ω72/Ω73), 2026-10-10 noche

- **#710** (zero-alloc en `compute_moments` y en el censo del orquestador):
  correcto. Mismos momentos sin el `Vec`. El censo pasa de load+store a
  `fetch_add`, que además cierra una carrera entre hilos. Dejó una línea
  suelta `<<<<<<< HEAD` en el plan: la CI de main se paró en el chequeo de
  marcadores y no compiló nada. La retira el PR de QS-P.
- **#711, C-10/C-10b**: verificado.
  - `compute_crowd_net_prospect_pressure` es antisimétrica en el cociente
    L/S (P(1/LS) = −P(LS)).
  - La lectura del registro conserva el signo.
  - `ls_account_ratio` tiene escritor en el host (`god_engine.rs:2358`).
  - En backtest no hay L/S, la presión vale 0 y el consejo no se modula: el
    T-1 deja de ver el sesgo antiguo contra los cortos.
  - Sigue sin medir si la presión de la masa predice algo (signo,
    contrarian): modo sombra y contraste OOS antes de que module.
- **#711, R-15**: mejora, pero no cierra la causa.
  - Escalar la deriva con `√(30 s/τ)`, acotado a [0,15; 1], quita el
    castigo más fuerte a las escalas largas.
  - La coherencia sigue sin ser una deriva calibrada.
  - Sigue habiendo tres literales sin derivar (30 s, 0,15 y 0,88).
  - Arreglo de fondo: estimar la deriva por escala con el pronóstico
    prequential del banco espectral, o medir el veto con el libro sombra
    (QS-R4a) por su `REJ_*`.
- **Certificación**: ni #710 ni #711 corrieron T-1, y #711 cambia la
  conducta viva (consejo y veto de primer toque). Los certifica el T-1 del
  PR de QS-P/QS-K sobre el árbol candidato, que incluye los dos.
