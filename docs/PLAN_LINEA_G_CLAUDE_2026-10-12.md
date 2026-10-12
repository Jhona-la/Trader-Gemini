# Plan de Línea G — revisión desde la base del HEAD 86218456: vetos, literales, lazos y campo multiactivo continuo (2026-10-12)

> Dueño: Claude, sesión local nueva (prefijo **LG-n**). Rama
> `claude/linea-g-revision-base`, worktree `.claude/worktrees/linea-g`.
> Las otras sesiones Claude conservan su prefijo: Línea C (CL-n, C-F4/ATR),
> Línea E (QS-n), Línea F (LF-n, nube). Alta publicada en el buzón
> (`COORDINACION_CODEX_2026-09-28.md`, 2026-10-12) **antes** de ejecutar.
>
> Este plan **no abre otro censo de archivos**: usa la lista compartida
> `docs/audit/REINICIO_BASE_FASES_C_2026-10-11.tsv`, el formato de recibo de
> `REINICIO_COORDINADO_BASE_2026-10-11.md` §3 y los registros de hallazgos
> existentes (C-F1…F3, `REVISION_PRS_43_55_2026-10-12.md`, LF-1…17,
> `BARRIDO_EXHAUSTIVO_FASES.md`). Añade una lente y cuatro entregables.

## 0. Qué hace la Línea G y por qué ahora

El operador pidió el 2026-10-12 reiniciar la revisión desde la base: «han
cambiado muchas cosas». Desde la última revisión independiente (Línea F,
base `977ba2f8`) entraron a `main` las olas Ω85–Ω88 y los PR #43–#57 de la
Línea C. La Línea G revisa el árbol **tal como está** con cuatro preguntas
que ninguna línea tiene como frente propio:

1. **¿Tiene sentido cada veto, bloqueo, límite y rechazo?** Censo completo
   del camino vivo: origen (derivado / medido / gen / literal / ADR), riesgo
   absorbente y juicio. Un veto que no se puede derivar ni medir es un
   literal disfrazado de prudencia.
2. **¿Qué está fijado a mano?** Literales y sesgos en el camino de decisión.
3. **¿Es de verdad autoadaptativo?** Qué aprende cada lazo, de qué señal, con
   qué olvido, y si puede quedar absorbido.
4. **¿Es un campo multiactivo continuo en τ?** Fugas de estado entre monedas
   y entre escalas; residuos de scalping/swing o de regímenes discretos.

## 1. Estado medido de Git (2026-10-12, al abrir la línea)

- `main` = `origin/main` = `86218456` (Ω88); tras el alta, `2382eb1f`.
- Remotas: `main` y `claude/auditoria-deslizamiento-apalancamiento-sqtc08`
  (ancestro de `main`; worktree activo de la Línea C: no se borra).
- Locales borradas: `cf4-siguiente-local` (0 commits sin equivalente por
  parches en `main`, `git cherry`).
- Locales conservadas (commits sin equivalente en `main`): `cf4-atr-local`
  (CL-66 sin publicar), `claude/cl84-f3702-local` (CL-84 v1, retirada por la
  propia Línea C), `codex/integration-recovery-2026-10-07` (2),
  `qoder/r7r6-aprender-medir` (3, documentos R6).
- Ancestros de `main` con worktree activo (no se pueden borrar sin quitar
  el worktree ajeno): `codex/root-spectral-zero-replay-2026-10-11`,
  `sol/replay-accounting-2026-10-09`.
- 19 worktrees de otros agentes vivos; no se toca ninguno.

## 2. La meta en números (lo que la haría verdad y lo que la refuta)

Meta: duplicar cada 72 h ⇒ crecimiento log `g = ln 2 / 3 = 0,231049` por día.
Con una orden de riesgo `f` (fracción del capital al stop) y resultado `R`
(en múltiplos del stop, **neto de fricción**), el crecimiento log por orden
es `f·μ − f²·E[R²]/2`, máximo en `f* = μ/s²` con `g* = μ²/(2s²)`.

Geometría de la doctrina: RR = 2,25, fricción ida y vuelta ≈ 12 pb (2×5 pb
de comisión taker + deslizamiento). La fricción **en unidades de R** es
`f_R = 12 pb / SL`: con SL = 30 pb vale 0,40 R; con SL = 150 pb, 0,08 R. El
equilibrio es `p* = (1 + f_R)/(RR + 1)`.

| SL (≈ τ) | p* | p supuesto | μ (R neto) | f* (riesgo/orden) | órdenes indep./día a Kelly | a ½ Kelly | al tamaño actual (≈0,2 %) |
|---|---|---|---|---|---|---|---|
| 30 pb | 0,431 | 0,45 | +0,063 | 2,4 % | 309 | 412 | 1 929 |
| 50 pb | 0,382 | 0,40 | +0,060 | 2,4 % | 325 | 434 | 2 010 |
| 50 pb | 0,382 | 0,45 | +0,223 | 8,5 % | 24 | 33 | 526 |
| 150 pb | 0,332 | 0,40 | +0,220 | 8,7 % | 24 | 32 | 531 |
| 150 pb | 0,332 | 0,45 | +0,383 | 14,6 % | 8 | 11 | 304 |

Lectura (consistente con la Línea F §2, que lo calculó sin fricción en R):

- **La fricción decide la escala**: a SL ≤ 30 pb hace falta p ≥ 0,43 sólo
  para no perder. El continuo en τ no es una preferencia estética: la τ
  operable es la que hace `σ(τ)·k ≫ f`. La puerta «banda paga fricción»
  (#586) es la frontera de ese hecho; sus parámetros deben salir de σ(τ) y
  f medidos, no de literales.
- **Con 13 USD el tamaño lo fija el nocional mínimo**: riesgo =
  5,1 USD · SL / 13 USD ⇒ 0,2 % del capital a SL = 50 pb. Llegar a f* = 8,5 %
  exige nocional ≈ 0,085·13/0,005 ≈ 221 USD (apalancamiento ≈ 17×) **y** una
  ventaja medida fuera de muestra. Sin la ventaja, el tamaño correcto es el
  mínimo; sin el tamaño, la ventaja no llega a la meta.
- **Lo que refuta la meta en este sistema**: que la cota inferior de μ
  (neto, fuera de muestra, por τ) sea ≤ 0, o que el número de órdenes
  **independientes** por día (no correlacionadas entre monedas ni entre τ)
  quede por debajo de la columna correspondiente. Ambas cosas se miden; nada
  de lo anterior está medido hoy en datos reales.

## 3. Fases de la Línea G (de la meta al último archivo)

Cada fase tiene escritor, revisor (adversarial), prueba y recibo. Las fases
G0–G6 recorren **el camino vivo** completo; G7 recorre el resto de archivos
de la lista compartida con la lente G. El orden es el del mandato: meta →
conceptos → matemática/estadística/física → algoritmos → código.

| Fase | Qué | Prueba que la cierra | Acople |
|---|---|---|---|
| G0 | Meta y medición: §2; contabilidad de equity neta; qué refuta la meta | Cálculo reproducible; contrato de equity neta del replay | C-F0, R0, LF §2 |
| G1 | Conceptos y contratos: τ de la orden, p*, desconocido ≠ 0, identidad por moneda, unidades | Tests de contrato por concepto | C-F1 |
| G2 | **Vetos, bloqueos, límites, rechazos**: censo completo con origen, alcanzabilidad, absorbencia, redundancia | Censo versionado + test por veto absurdo o absorbente; contrafactual en sombra (QS-R4a) para medir qué veto ahorra o bloquea dinero | R-ledger, `veto_registry.rs`, QS-R4 |
| G3 | **Literales y sesgos** del camino de decisión | Cada literal → derivado / medido / gen con banda / ADR, o fuera | C-F2, CL-58 |
| G4 | **Lazos adaptativos**: señal, olvido, absorbencia (regla Línea F: esperanza neta frente a p*) | Monte Carlo del lazo con ventaja conocida antes de arreglar | LF §4 |
| G5 | **Campo multiactivo y τ**: fugas entre monedas y escalas; escalares por moneda que mezclan τ | Test de dos monedas / dos τ que demuestra la fuga | LF §5, C-F3 |
| G6 | **Teoría**: estado real → siguiente paso → prueba que la falsa | Tabla §7; ninguna teoría entra al camino vivo sin su prueba | ADR-0005 |
| G7 | Resto de archivos (lista compartida), uno por uno con la lente G | Recibo por archivo | REINICIO §3 |
| G8 | Camino caliente y recursos: ns y asignaciones por evento, memoria, disco | Medidor CL-60 antes/después | C-F6, plan §7 de C |
| G9 | Cierre: crecimiento log neto OOS con cota, por τ y por moneda | Examen OOS en dos mitades | C-F0 §9 |

Esta primera ola ejecuta G0–G6 sobre el camino vivo con 18 dimensiones de
lectura (núcleo en 6 tramos, módulos del núcleo ×2, riesgo ×2, señales,
consejo y estrategias, arena ×2, host, ejecución, aprendizaje, datos) y
verificación adversarial de cada hallazgo alto o medio.

<!-- §4 en adelante: se completan al cerrar la ola con los resultados verificados. -->

## 4. Censo de vetos, bloqueos, límites y rechazos (G2, retoma LF-F4)

Datos completos, una fila por veto con archivo, línea, constante, origen,
riesgo absorbente y juicio: [`audit/linea_g/vetos.tsv`](audit/linea_g/vetos.tsv)
(18 dimensiones del camino vivo, base `86218456`; los lectores leyeron 272
archivos distintos y declararon lo no leído en
`audit/linea_g/raw_<dimensión>.json`, campo `coverage`).

| Medida | Valor |
|---|---|
| Vetos, límites, clamps, umbrales y enfriamientos censados | **1 237** |
| Origen literal / derivado / ADR / gen / medido / desconocido | **788 (64 %)** / 218 / 105 / 83 / **38 (3 %)** / 5 |
| Juicio: sólido / cuestionable / sin sentido / muerto / duplicado | 306 / 662 / **129** / **119** / 21 |
| Riesgo absorbente probable / posible | **43** / 59 |

Concentración: `god-engine-core/src/lib.rs` 415 (252 literales, 47 sin
sentido), host `god_engine.rs` 68 (14 absorbentes), `online_daemon.rs` 48 (37
literales), `consejo_seniors.rs` 47 (37 literales), `temporal_spectrum.rs` 38,
`leverage_matrix.rs` 14 (8 sin sentido).

**Lectura.** El sistema no es todavía autoadaptativo en su capa de
protección: dos de cada tres frenos son números escritos a mano y sólo uno de
cada treinta se mide en línea. Los 43 absorbentes son la causa directa de que
el motor «se asfixie» y de que la evolución no pueda corregirlo: un freno que
aprende sólo de lo que deja pasar no recibe nunca el dato que lo liberaría.
Tres familias:

1. **Latches sin rearme por diseño** (ADR-0015: kill-switch, inmune,
   insolvencia, X-009, `REJ_DRAWDOWN`). Son política del dueño; lo que sí es
   defecto es que el pico del drawdown no se persista (un reinicio lo borra) y
   que mida sólo la caja realizada (LC-F4-10, LC-F0-08).
2. **Frenos estadísticos sin olvido** (convicción por rama — **LG-1**,
   envolvente de Kelly — LF-12, aceptación conformal, habilidad de motores,
   ML-GATE sin `_MOTOR`, DSR de promoción). Regla para todos: olvido con la
   semivida del calibrador **y** una sonda acotada al nocional mínimo bajo el
   tope de ruina; sin la sonda, el olvido no basta porque no llegan datos.
3. **Filtros de datos que se autobloquean** (filtro de salto del host —
   **LG-2**, contadores de rate-limit sin edad, veto de deriva que necesita
   las entradas que veta). Un filtro de integridad debe distinguir un dato
   corrupto aislado de un cambio de estado persistente.

**Regla de admisión de un veto (propuesta a todas las líneas).** Un veto
entra o se queda en el camino vivo sólo si cumple las cinco:
(i) su umbral es derivado de la geometría de la orden (p*, σ(τ), fricción),
medido con incertidumbre, o un gen con banda que no contiene regiones
inertes; (ii) se juzga a la τ de la orden; (iii) desconocido ≠ 0 (no relaja ni
vota); (iv) tiene salida: olvido o sonda, o es un latch declarado en un ADR;
(v) tiene dos pruebas, una que rechaza y otra que admite en ambos lados de la
frontera, ejercidas a través del gate real (no de la función suelta).

## 5. Hallazgos de la ola 1 (base `86218456`) y su estado

Método: 18 lectores de sólo lectura (uno por dimensión); cada hallazgo alto o
medio NUEVO pasa por un escéptico que intenta refutarlo, recalcula los números
y busca si ya estaba registrado. Datos: [`hallazgos.tsv`](audit/linea_g/hallazgos.tsv)
(411 hallazgos: 32 HIGH, 195 MED, 184 LOW, con su referencia si ya estaban
registrados), [`candidatos.json`](audit/linea_g/candidatos.json) (67 candidatos
nuevos) y [`verificacion.tsv`](audit/linea_g/verificacion.tsv) (veredicto, zona
dueña, prueba propuesta y arreglo por candidato).

Veredictos de los 67 candidatos: **15 confirmados, 23 plausibles, 27 ya
registrados por otra línea y todavía abiertos, 1 refutado, 1 ya arreglado**
(por LG-3, antes de que el escéptico lo leyera). La mayoría de los HIGH/MED que
encontraron los lectores ya estaban en C-F1…F3, LF-1…17 o el BARRIDO: el
problema del sistema no es que nadie los vea, es que **siguen abiertos**.

### 5.1 Arreglados en esta rama (test que falla antes y pasa después)

| ID | Zona | Defecto | Prueba | Efecto medido |
|---|---|---|---|---|
| **LG-1** | núcleo (`lib.rs`) | Una rama con Wilson hi < p* emitía Long/Short con confianza 0: tapaba la cascada (20→21→22-24, 1-10→11→12, 13→15→14), podía ganar el arbitraje de mismo armónico por energía y quedaba condenada toda la sesión (sin olvido). Ahora `aplanar_sin_conviccion` actúa antes de cada punto de cascada y del arbitraje, y `envejecer_registro_ramas` reduce la muestra a la mitad por cada semivida del calibrador (12 h) conservando la proporción. | `tests_lg1` (3) | T-2 y T-1 sin cambio en el fixture (no hay ramas condenadas en 50 h) |
| **LG-2** | host | El filtro de «flash crash» comparaba con el último tick ACEPTADO: tras un salto real de nivel la moneda quedaba ciega hasta reiniciar. Ahora un pico aislado se descarta y un segundo tick anómalo en el mismo nivel confirma el cambio. | `lg2_un_cambio_de_nivel_no_deja_ciega_a_la_moneda` | — |
| **LG-3** | ejecución (`executor.rs`) | El uso de rate-limit de la última cabecera no caducaba: por encima del headroom bloqueaba para siempre entradas, **salidas, cancelaciones y protecciones**. Ahora caduca con la ventana del exchange (60 s / 10 s). | `tests_lg3` (2) | — |
| **LG-4** | genoma (`genome.rs`) | `tradeable_band_ms` y la reparación RR usaban como dominio la malla representacional (1 ns – 146 años). Los `active.json` de backtest/demo/prod salían de cada carga con TP ×1,68 (RR 2,25 → 3,8; p* 0,35 → 0,235) por una restricción a 156 ns, y `validate` los rechazaba. Ahora el dominio es la banda operable [30 s, 12 h]. Es la causa del síntoma «violan la RR en τ_lo» del ciclo 8 de la Línea C. | `lg4_…_no_se_repara` (RED antes), `tests_lg4` de `genome_store` | **T-1 17/144 → 19/144**: vuelven a ser sensibles los genes 142 y 143 (curva de SL) |

**LG-5** (pedido con prueba OPEN, `lg_calibracion_ev_contract.rs`): Platt y el
conformal aprenden `neto > 0` con cualquier salida y la puerta EV usa esa p como
P(TP antes que SL). Con 15 % TP, 40 % BE y 45 % SL la puerta ve **EV = +0,766 R**
mientras la R neta media realizada es **−0,104 R**: se aprueban y dimensionan
órdenes de esperanza negativa. Dueños: Codex (calibración) y Línea E (D1). Dos
salidas posibles, ambas coherentes: etiqueta de barrera para P(TP primero) con
las salidas intermedias aparte, o que la puerta consuma la R neta esperada
calibrada sobre la puntuación.

### 5.2 Confirmados o plausibles, entregados a su dueño (buzón del 2026-10-12)

| Zona | IDs | Lo más grave |
|---|---|---|
| Línea C | G016, G025, G028, G034, G035, G056 (confirmados); G002, G005, G013, G021, G023, G027, G036, G039, G044, G055, G059, G061 (plausibles) | G025: el suelo de un tick del filtro de horquilla reabre la puerta cuando la fricción supera el recorrido; G034: el veto X-016 «de incoherencia» es un veto de dirección con umbral literal 0,02; G035: el veto CVD F8-P10 mide cambio y no nivel; G056: el umbral de caída de la CUENTA cambia con el nocional mínimo de la moneda candidata |
| Línea C (host, sin reserva activa pero en su dominio) | **G042 (HIGH, plausible)** | El cierre local del núcleo envía un reduce-only por la cantidad de la ranura y después cancela **todas** las piernas OCO del lado: con dos ranuras del mismo lado la que sigue abierta queda sin TP/SL en el exchange, y si el bracket ya cerró la ranura el reduce-only recorta la otra. Requiere identidad de bracket por ranura. |
| Codex | G014/G024 (el conformal exige p > ½: mudo con RR ≥ 2,25), G015/G022 (= LG-5), G063 (Hayashi-Yoshida sin muestra mínima) | |
| Qoder | G064 (confirmado: el dominante se elige entre las 32 escalas y después se veta por τ < 30 s, descartando el veredicto de la banda operable), G004, G011 | |
| AGY | G037 (Hodge desconocido publicado como curl 0 da confianza máxima al asiento de microestructura) | |
| GLM | G048 (la etiqueta de triple barrera del entrenador confunde volatilidad con dirección) | |
| Sol | G050 (el gate de la envolvente del replay dimensiona con la curva; el host, con el stop real) | |
| Línea E | G031 (régimen discreto de Hurst 0,45/0,52 conmuta umbrales y apaga la rama 11) | |
| Libre (próxima ola de la Línea G) | G040 (OCO: reintento con IDs nuevos tras AMBIGUOUS duplica piernas), G046 (fee-breaker de 12 h con 3 cierres, sin comparar contra p*), G049 (el examen walk-forward del demonio sólo puede juzgar τ cortas) | |

Refutado (no reabrir): G030. El respaldo «sin libro» de las ramas 22-24 es
inalcanzable desde `process_event` (las cantidades efectivas nunca son 0); el
mismo argumento refuta la alcanzabilidad que supone LC-F1-19.

## 6. Lazos adaptativos (G4): el defecto es uno solo

Datos: [`lazos.tsv`](audit/linea_g/lazos.tsv). De **211** cosas que aprenden:
**64** cerradas y vivas, **59 con la señal equivocada**, **53** abiertas (nadie
consume lo aprendido), **19** literales que no aprenden y **16 absorbentes**.

Las 59 con señal equivocada comparten la misma raíz: cada lazo inventa su
etiqueta (signo del PnL neto, acierto direccional contra ½, vela de 1 min, signo
del mid, racha con 5 pb brutos) y casi ninguno sabe qué geometría tenía la
orden. Por eso el sistema todavía no es autoadaptativo: sus órganos aprenden
cosas distintas de la misma operación y se contradicen.

**Propuesta (G4, para todas las líneas): un único resultado de orden.** Al
abrir se congela y al cerrar se completa un registro por orden:
`{moneda, rama, τ_orden, score_entrada, p*_entrada = (SL+f)/(TP+SL), TP, SL,
f, etiqueta_barrera: Option<bool>, R_neto = pnl_neto / riesgo_al_SL,
motivo_salida, edad}`. Cada lazo consume la estadística que corresponde a su
decisión, siempre la misma para la misma pregunta:

- **¿Esta fuente tiene ventaja?** (convicción por rama, pesos del consejo,
  Hedge, skill de motores, e-procesos): media de `R_neto` frente a 0, o
  `etiqueta_barrera − p*_entrada`, con e-proceso y olvido. Nunca acierto
  contra ½.
- **¿Qué probabilidad usa la puerta EV?** P(TP primero) con
  `etiqueta_barrera`, o directamente E[R_neto | score] (LG-5).
- **¿Cuánto arriesgar?** Kelly sobre la cota inferior de E[R_neto] y su
  varianza, por banda de τ.
- **¿Cuándo enfriar?** Rachas de `R_neto < 0` por banda de ln τ (LF-5).

Todos con la misma semivida (la del calibrador) y, si su veto puede cortar su
propio flujo de datos, con una sonda acotada al nocional mínimo bajo el tope
de ruina.

## 7. Teoría: siguiente paso de cada pieza y la prueba que lo falsa

Rige ADR-0005: una teoría entra al camino vivo sólo si su contribución
marginal **fuera de muestra** (log-loss, QLIKE o crecimiento log neto, según
el caso) es positiva en las dos mitades, con multiplicidad controlada. La
tabla de la Línea F (§7 de su plan: bosque → boosting con primer toque,
calibración condicional, τ por e-valor, primer toque OU/saltos, Kelly en
espacio de riesgo, riesgo de cartera monetario, evolución con DSR, conducta
de la multitud como observable, y el triage de las analogías «del milenio»)
**sigue vigente y no se repite**. La Línea G añade las piezas que esa tabla no
cubre, con el mismo formato:

| Pieza viva | Siguiente paso (el «XGBoost» de esa pieza) | Prueba que lo falsa antes de cablear |
|---|---|---|
| Hawkes univariante con μ̂ por EWMA | Hawkes **multivariante marcado** entre monedas (excitación cruzada, núcleo de ley de potencias), ajustado por EM en línea | Log-verosimilitud OOS mejor que Poisson y que el univariante; test de reescalado temporal (KS) de los residuos |
| σ(τ) pronosticada (RLS de `spectral_tape`, anclas) | HAR-RV en log-varianza con estimador realizado robusto a ruido (pre-promediado); volatilidad rugosa sólo si se mide H_vol < 0,5 | QLIKE OOS por τ frente a persistencia y a la EWMA actual |
| Consenso por IC prequential (pesos por motor×escala) | Agregación en línea por **pérdida propia** (Hedge / gradiente exponenciado sobre log-loss de la probabilidad de primer toque), con regret acotado | Regret OOS frente al mejor experto fijo y frente a pesos iguales |
| Significancia (Ville por celda, Bonferroni M/α) | e-valores con **e-BH** para la familia motor×escala×moneda (controla FDR anytime-valid, más potencia que Bonferroni) | Tasa de falsos descubrimientos en ruido barajado ≤ α con potencia mayor que la actual |
| Símplex de régimen (sigmoides de z_BTC y Hurst) + BOCPD sobre W₁ | Modelo de espacio de estados (HMM gaussiano en log-retornos y log-volumen por τ) con detección de cambio sobre la **escala**, no sobre W₁ (F2.3-08b) | Log-verosimilitud predictiva OOS; detección de un cambio de volatilidad ×5 en > 90 % de caminos con falsas alarmas ≤ α |
| Lead-lag (correlación cruzada con lags físicos) | Transferencia de información con nulo por permutación de bloques, a la τ de la orden; Granger multivariante con regularización | IC OOS del líder sobre el seguidor > 0 con IC por bloques; desaparece en surrogados de fase |
| Campo multiactivo × τ | **Coherencia espectral cruzada** (wavelet MODWT) entre monedas por escala, con significancia por surrogados (IAAFT); la «misma apuesta» se juzga a la τ de la orden | Dos monedas con ρ₄ₕ alta y ρ_tick baja se agrupan a 4 h y no a 30 s; en surrogados la coherencia no es significativa |
| Elección IOC / maker (B3.29, CL-14) | Ejecución óptima con posición en cola (Avellaneda–Stoikov / modelo de cola de Cont–de Larrard) usando el libro L2 vivo | Deslizamiento realizado vs predicho por decil; la ruta maker sólo se habilita si el coste esperado medido (incluida la selección adversa) es menor |
| Conformal (ACI por moneda) | Conformal **condicional a τ** con intercambiabilidad ponderada por edad (covariate shift) | Cobertura empírica ±2 pp del nominal por τ y moneda, en las dos mitades |
| Evolución (GA sobre 144 genes) | CMA-ES / búsqueda bayesiana sobre el subespacio de genes **sensibles** (T-1), con presupuesto de pruebas contabilizado en el DSR | El genoma promovido supera al incumbente OOS con e-valor ≥ 1/α; los genes inertes no se mutan (gasto de multiplicidad) |

**Sobre «pensar como un universo continuo»**: el objeto matemático correcto
es un campo `X(i, τ, t)` (moneda × escala × tiempo) con densidad espectral
cruzada `S_ij(ω)` y coherencia por escala. Las decisiones son funcionales de
ese campo: la dirección sale del consenso por escala, la τ de la orden de la
escala con habilidad demostrada que **paga la fricción** (§2), y el tamaño del
riesgo de cartera **a esa τ** (no de la correlación de los últimos ticks). Los
escalares por moneda que mezclan τ distintas (enfriamiento, racha, epigenética,
riesgo por operación: LF-5, LF-7, LF-11) son exactamente las rupturas de ese
campo y son prioridad en G5.

## 8. Recursos y compilación (lo que funcionó en el PC del operador)

- **Worktree anidado ⇒ `rustflags` duplicados.** Un worktree dentro del
  checkout (`.claude/worktrees/…`) hereda dos `.cargo/config.toml` (el suyo y
  el del padre) y cargo **concatena** los `rustflags`: la huella cambia, se
  recompilan todas las dependencias externas y rustc llegó a caerse con
  `STATUS_STACK_BUFFER_OVERRUN`. Solución: `export RUSTFLAGS="-C
  target-cpu=native -C opt-level=3"` (la variable sustituye a la
  configuración en vez de sumarse).
- Con esa variable se puede usar el `target` del checkout principal sin
  abrir otro (disco: 11–21 GB libres en esta sesión): las dependencias
  externas se reutilizan y los crates del workspace compilados desde el
  worktree obtienen huellas propias (no pisan los artefactos de los demás).
- Auditoría por subagentes en **tandas de 3** (regla de la memoria): 18
  lectores y 13 escépticos, 9,4 M tokens en total, sin tocar el límite de
  sesión y sin errores. Los escépticos leen los candidatos de un JSON del
  repositorio (el guion sólo recibe los ids).

## 9. Protocolo y siguientes fases

- La Línea G sólo arregla en archivos sin reserva activa o con el dueño
  avisado, con test que falla antes, golden, T-2 y T-1. Todo lo demás va al
  dueño con su prueba.
- **Ola 2 (siguiente)**: G040, G046 y G049 (zona libre); el contrato único de
  resultado de orden (§6) como propuesta con prueba de Monte Carlo antes de
  cablear; la regla de admisión de vetos (§4) sobre los 43 absorbentes.
- **G7**: cada lector declaró lo que no leyó (`raw_<dimensión>.json`,
  `coverage.unread`). Esa lista, unida a `REINICIO_BASE_FASES_C_2026-10-11.tsv`,
  es la cola archivo por archivo de la ola 3.
- Recibo de esta ola: rama `claude/linea-g-revision-base`, commits LG-1…LG-5,
  contrato de `genome_store` y merge con `main` (CL-87…CL-96); números en
  §5.1 y en el buzón.
