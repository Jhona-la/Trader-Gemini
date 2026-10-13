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

### 4.1 Triaje de los 43 vetos absorbentes (ola 2)

Cada absorbente cae en una de cuatro familias; la regla de admisión de arriba
dice qué le falta a cada una.

| Familia | Vetos (archivo:línea) | Qué le falta | Estado / dueño |
|---|---|---|---|
| **A. Latch por diseño** (ADR-0015/0016) | kill-switch de entradas (`lib.rs:3156`, `executor.rs:1271`), latch inmune (`god_engine.rs:1834`), insolvencia (`:1742`), X-009 (`:4756`), límite de drawdown del inmune (`:1737`), latencia del inmune (`:1751`, `:1779`), `REJ_DRAWDOWN` (`risk lib.rs:342`), d* y `ALFA_FALSACION` (`drawdown.rs:170/196`), kills del demonio (`online_daemon.rs:1253/1346/1365`) | Nada como política; sí defectos laterales: pico del drawdown no persistido y sólo caja realizada (LC-F4-10), doble lectura del gen (LC-F1-30), kills del demonio por literales sin ARL₀ | Línea C (CL-99 «pico» en curso); demonio: dueño a confirmar |
| **B. Freno estadístico sin olvido o sin sonda** | convicción por rama (`lib.rs:181`, `:6555`), p* = ½ en ramas 1-10 (`:195`), conformal singleton (`conformal.rs:175`), habilidad del ensamble (`ensemble.rs:106`), skill de motores (`skill_motores.rs:100`), DSR de promoción (`darwin.rs:669`, `online_daemon.rs:1917`), veto de grupo (`espectral_multiactivo.rs:69`), envolvente de Kelly (`kelly_envelope.rs:261/282/387`, `envio.rs:76`, `god_engine.rs:4153`), ML-GATE sin `_MOTOR` (`lib.rs:8149`), epigenética (`state.rs:433/434`), sonda por moneda (`lib.rs:8121`) | Olvido con la semivida del calibrador **y** sonda acotada al nocional mínimo; puntuar por R neto frente a p* (LG-9) | **LG-1** cerró la convicción por rama; **CL-98** (Línea C) cerró la sonda única; el resto con su dueño (LF-12 Línea C+E, conformal Codex, skill Qoder, ensamble GLM, epigenética Línea C/AGY) |
| **C. Filtro de datos que se autobloquea** | filtro de salto (`god_engine.rs:3369`), headroom de rate-limit (`executor.rs:1320`), umbrales de rate-limit como genes (`god_engine.rs:4781`), veto de deriva (`god_engine.rs:2986/2989/3976`, `:3752`), EWMA del OBI de trades (`microstructure.rs:52`) | Distinguir dato corrupto aislado de cambio persistente; caducidad; los límites del exchange no son genes | **LG-2** y **LG-3** cerrados; veto de deriva = centinela de contabilidad que actúa como latch cuando no hay posiciones (Codex, XXXVI); genes de rate-limit congelados (**LG-18**); OBI de trades → GLM (exige reentrenar) |
| **D. Enrutado a una capacidad inexistente** | ruta Iceberg (`entry_dispatch.rs:59`, `god_engine.rs:4427`) | La ruta sólo debe elegirse si el adaptador la soporta | **LG-10** cerrado (techo oculto de 10 000 USD de nocional en prod); el gen se conserva (vuelve a tener efecto con iceberg nativo) |

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

## 10. Ola 2 (2026-10-12, base `be8fa6ad`): lectura de todos los archivos restantes y diez arreglos

### 10.1 G7 — cada archivo Rust del repositorio leído al menos una vez

La ola 1 leyó 191 de los 627 `.rs` versionados. La ola 2 asignó los 436
restantes a 10 lectores de sólo lectura (6 dimensiones de código, ≈35 000
líneas; 4 de pruebas, ≈42 000 líneas; listas en
[`ola2_dims.json`](audit/linea_g/ola2_dims.json)). **Los 436 tienen veredicto**
en [`g7_archivos.tsv`](audit/linea_g/g7_archivos.tsv); hallazgos en
[`g7_hallazgos.tsv`](audit/linea_g/g7_hallazgos.tsv) (92: 2 HIGH, 35 MED, 55
LOW) y vetos nuevos en [`g7_vetos.tsv`](audit/linea_g/g7_vetos.tsv). Entre las
dos olas, **los 627 archivos Rust han pasado por un lector**; los que no son
Rust (documentos, datos, generados) siguen la lista compartida de REINICIO.

**Honestidad de las pruebas** (≈283 archivos de prueba leídos): 252 honestas,
51 mixtas, **35 vacuas**, **22 que prueban una copia muerta**, **16 que fijan
un defecto en verde**, 13 frágiles, 1 `#[ignore]` rancio. Ejemplos: las
pruebas de nivel H0 de Ville no pueden fallar (con ±0,04 y 150 pasos la
riqueza máxima posible es 19,5 < 1/α); `r7_r2_chaos_simplex_contracts_margin_smoothly`
pasa sin el término que dice probar; el contrato R7-R6 certifica en una copia
local el recorte ±1 R que LF-3 retiró; `hawkes_chain_integration` certifica
como «roles correctos» un artefacto de relojes periódicos; la prueba de
F3.3-01 se dio por activada mientras 18 ramas vivas siguen con p* = ½. Es la
duodécima confirmación del patrón «contrato verde con física muerta»: un
verde no prueba nada si la prueba no puede fallar.

### 10.2 Arreglados en la ola 2 (test que falla antes y pasa después)

| ID | Zona | Defecto | Prueba |
|---|---|---|---|
| **LG-6** (G040) | ejecución | El reintento OCO con id nuevo (`_SLR/_TPR`) tras un AMBIGUOUS podía dejar dos SL o dos TP; la reversión parcial se tragaba las cancelaciones fallidas. Ahora mismo `clientAlgoId` (el -4116 confirma la pierna) y las cancelaciones fallidas quedan en «pendientes» y ensucian la protección. | `tests_lg6` (2) |
| **LG-7** (G049a) | evolución | El examen walk-forward del demonio censuraba lo abierto al final de cada tramo (no restaba aptitud ni drawdown). Ahora se valora a mercado con la fricción de salida taker. | `tests_lg7` |
| **LG-8** (G046) | host | El fee-breaker suspendía hasta 12 h con 3 registros REALIZED_PNL (llenados) — ≈22 % de falsas suspensiones con esperanza positiva. Ahora agrupa por cierre, exige la cola binomial P(X ≤ ganadoras \| n, p*) ≤ α y la suspensión caduca con su evidencia. | Monte Carlo ≤ α |
| **LG-9** (G4) | núcleo, **sin cablear** | Contrato único de resultado de orden y evidencia de ventaja por R neto (e-procesos de Ville). Wilson contra ½ condena 200/200 fuentes rentables (p = 0,35); la R neta 0/200. | `resultado_orden::tests` (4) |
| **LG-10** | host/ejecución | Todo nocional sobre el gen `iceberg_volume_threshold` se enrutaba a Iceberg y el despachador lo rechazaba siempre (Binance sin iceberg nativo): techo oculto de 10 000 USD de nocional en prod, 100–2 000 USD con genomas aleatorios. El replay nunca lo modelaba (divergencia backtest↔vivo). | `lg10_…` |
| **LG-11** | host/telemetría | El host nunca inicializaba la cola de logs: 161 `telemetry_log!` hacían `println!` síncrono en el bucle vivo (QuickEdit de Windows lo congela); la cola reservaba ~520 MB y su hilo giraba. Ahora cola de 8,5 MB, consumo por lotes, sueño si está vacía, vaciado antes de salir. | `tests_lg11` (2) |
| **LG-12** | ejecución/host | Los límites de peticiones del exchange salían de genes (mutaban hasta 5 000 de peso con un real de 2 400: el freno local no actuaba antes de un 418). Ahora son las constantes documentadas de Binance. | `tests_lg12` (2) |
| **LG-13** | host/os-guardian | `EmptyWorkingSet` sobre el proceso vivo cada hora, siempre (fallos de página en el camino caliente). Ahora sólo con presión medida (memoria disponible < working set). | `tests_lg13` |
| **LG-14** | os-guardian | El tope duro del JobObject sumaba el commit del entrenador hijo y el veto medía otra magnitud (working set): el motor podía abortar con posiciones abiertas sin veto. Ahora tope por proceso y el veto mide el commit. | `tests_lg14` (2) |
| **LG-15** | herramientas | `evolver`/`evolution` leían los tapes desde el byte 0 ignorando la cabecera de 8 B: optimizaban sobre registros desplazados y el evolver (lanzado por `LAUNCHER.bat`) promovía ese genoma. | `tests_lg15` |

### 10.3 Entregas de la ola 2 por dueño

- **Línea C**: `prueba_reversion.rs` (CL-97): la varianza HAC centra el
  retardo 1 con la masa de todas las filas, no la de los pares (t se infla
  ×1,7 con la base derivada y huecos de spot), y los nodos 21-22 nunca maduran
  porque `reset_engines` los reinicia en cada reconexión (≤ 24 h): StatArb
  queda mudo para τ* ≥ 73 min. **G042**: diseño propuesto — guardar el
  `base_id` del bracket por ranura y cancelar sólo `{base}_SL`/`{base}_TP` (con
  LG-6 ya no hay variantes), y omitir el reduce-only si el bracket de esa
  ranura ya llenó.
- **AGY** (contratos Ω): las pruebas vacuas o de copia muerta de §10.1
  (Ville H0, chaos simplex, multislot/niche3/gauge, R7-R6 watchdog,
  Yang-Mills, TTL multiactivo).
- **Codex**: el veto de deriva es un centinela de contabilidad que actúa como
  latch cuando no hay posiciones (necesita 10 cierres y veta las entradas que
  los producirían); `model_manifest`: la frescura del holdout se decide por el
  sha256 del archivo (un tape re-exportado cuenta como holdout nuevo).
- **Sol**: el contrato de causalidad del replay no puede ver un lookahead de
  una fila; `continuous_evolution_backtest` adopta cada día el mejor de hasta
  1 000 mutantes sin corrección por multiplicidad.
- **GLM**: `auto_trainer_daemon` reescribe la red viva de BTC con un gate de
  BCE sobre una etiqueta sin lado; `config_compiler` promueve a demo y backtest
  sin evaluar; `walkforward_evolver` evoluciona sobre un tape sintético sin
  cabecera.
- **Zona libre (ola 3 de la Línea G)**: el evolver promueve con aptitud
  in-sample y presta el bosque de BTC a todas las monedas; `oi_history_sync`
  guarda 500 h de las 720 que pide; congelar los genes que ya no tienen efecto
  (iceberg, rate-limit, los 7 de R7-R1-2); el mínimo de 30 cierres del examen
  del demonio (G049b); los 6 `println!` síncronos del núcleo.

## 11. Ola 3 (2026-10-12, base `76bcdaaf`): la zona libre que dejó la ola 2

### 11.1 Arreglados (test que falla antes y pasa después)

| ID | Zona | Defecto | Prueba |
|---|---|---|---|
| **LG-16** | herramientas (`oi_history_sync`) | La paginación hacia atrás tomaba como cursor la fila MÁS NUEVA de la página (el endpoint devuelve orden ascendente): la segunda página repetía 499 filas y añadía una. Se guardaban 501 h de las ~720 que conserva `openInterestHist`, y cada ejecución perdía para siempre ≈9 días del histórico que entrena `{SYM}_OI`. Un fallo de red en una página posterior descartaba la serie entera. | `tests_lg16` (3; RED 501 de 721) |
| **LG-17 / LG-17b** | herramientas (`evolver`, lo arranca `LAUNCHER.bat`) | Promovía con la aptitud de la propia selección (16 × 6 sobre el mismo flujo) y con un OR que aceptaba capital final > inicial aunque la aptitud fuera INVIABLE. Evaluaba en otro mundo: bosque de BTC prestado a todas las monedas, 34 features de microestructura pisando el omni macro, red aleatoria votando sin DarkAlpha, fricción cobrada dos veces, arena construido dentro de un worker de rayon, universo sin publicar (`update_market_data` era un no-op) y, con `TG_GENOME_ENV` ausente o desconocida, promoción al silo `__UNSET__`. Ahora: selección en el 70 % cronológico inicial y juicio del campeón y del incumbente en el 30 % final con `fitness::compute` (≥ 30 operaciones) y el margen común; guarda de entorno del almacén; universo publicado; sin préstamo de bosques; omni neutro; NN ausente = no opina. **No promueve** mientras su evaluador no sea el canónico (`process_tick` legacy, sin envolvente del host): explora e informa. | `tests_lg17` (7; RED en el cableado), humo del binario con entorno ausente, desconocido y válido |
| **LG-18 / LG-18b** | genoma | Nueve genes sin lector (los 7 de R7-R1-2 y los 2 límites de peticiones que LG-12 dejó sin efecto) seguían mutando: dimensiones que derivan sin presión y suman ensayos al recuento de selección. Se congelan como G0-4. La prueba D-650 de simetría del hot-swap se apoyaba en la mutación para distinguirlos: ahora los fija a mano (verificado por mutación: quitando un `.store`, falla en el gen 133). El gen de Iceberg NO se congela: con un adaptador con iceberg nativo vuelve a tener efecto. | `tests_lg18` (RED: los 9 mutaban), D-650 por mutación |
| **LG-19 / LG-19b** | núcleo/telemetría | Los 4 `println!` de producción del núcleo (OPEN/CLOSE TRACE, ML-DIAG) pasan a la cola de LG-11; sin cola siguen saliendo por stdout (T-1 y forense iguales). El buffer cortaba a 512 B dentro de un carácter multibyte («UTF8_ERR») y, tras el corte, las piezas siguientes rellenaban los bytes libres (la línea dejaba de ser prefijo del original). | `tests_lg19` (3; RED por mutación) |

**Revisión adversarial de la ola.** Tres revisores (corrección, paridad,
regresión) leyeron el diff: LG-16 y LG-18 sin defectos; LG-17 sobreafirmaba la
paridad (de ahí LG-17b); LG-19 tenía el corte a medias (LG-19b); LG-18 le
quitaba cobertura a D-650 (LG-18b).

**Lo que los tapes del evolver dicen ahora.** `data/*_ticks.bin` (agosto) no
tienen cabecera: origen desconocido, que D-691 trata como sintético. Aunque el
evaluador fuese canónico, con ellos no promovería.

**LAUNCHER.bat (decisión del operador).** El lanzador no define
`TG_GENOME_ENV`: con LG-17 el «Cerebro Evolutivo» sale con error al arrancar
en una consola minimizada que se cierra sola. Antes corría y promovía al silo
`__UNSET__`, que nadie lee: no se pierde nada, pero el operador debe saber que
hoy el evolver no corre desde el lanzador. Para que vuelva a tener sentido
hacen falta tapes reales y el replay canónico multiactivo.

### 11.2 Lo que sigue abierto

- **Replay canónico multiactivo** (pedido a **Sol**): generalizar
  `booktick_replay` a N monedas (process_event, envolvente por moneda,
  universo, omni con historia). Con él, el evolver pasa a evaluar con el
  mismo motor que opera y puede volver a promover.
- **G049b** (examen del demonio: 30 cierres en ≤ 8 tramos de 53 min sólo deja
  juzgar τ cortas). Con LG-7 lo abierto ya se valora a mercado; el arreglo de
  fondo es medir la evidencia en barras de mercado del examen (no en cierres)
  o alargar la ventana con la τ del genoma. Diseño del `online_daemon`:
  ofrecido a su dueño con la prueba de la ola 2.
- **Cablear LG-9** (resultado de orden por R neto) en la convicción por rama,
  los pesos del consejo, el skill de motores y la calibración: decisión de sus
  dueños, con su T-1.
- **G042** (cierre local que purga las piernas OCO del lado): sin acuse de la
  Línea C; diseño en §10.3.
- **EWMA del OBI de trades** (`microstructure.rs:41`, R7-R2-D-4): la delta de
  volumen acumula de por vida y entra al vector del bosque; cambiarla exige
  reentrenar (**GLM**).

## 12. Ola 4 (2026-10-12, base `943c85b3`): revisión desde la base de lo que cambiaron las otras líneas

### 12.1 Método

Desde la base de la ola 1 (`86218456`) entraron en `main` 22 commits de otras
líneas: Línea C de CL-87 a CL-98b (z de la base, prueba de reversión por nodo,
σ̂60, caché del piso, τ y sonda una vez por banda, sonda única, orden mínima al
lote) y AGY Ω88/Ω89 (fixture conformal, convicción contrarian del Ente, doble
conteo de masas). Unos 40 `.rs`. Tres lectores de sólo lectura con lentes
distintas (D1 matemática y estadística de la base y StatArb; D2 riesgo, sonda,
consejo y núcleo; D3 honestidad de las pruebas nuevas; 53 archivos leídos
completos) y un verificador escéptico por lente que intentó refutar cada
hallazgo. Tabla completa con el veredicto en
[`ola4_delta.tsv`](audit/linea_g/ola4_delta.tsv).

### 12.2 Resultado

**25 hallazgos: 23 confirmados o plausibles, 2 refutados** (las
dos pruebas F3.7-02 de 20 min que pasan por inmadurez, porque su dueño lo
documenta como abstención y no como potencia). Por dueño:

| Dueño | Hallazgos | Los que nadie había registrado |
|---|---|---|
| **Línea C** | D1-01…06, D1-09, D2-01…03, D2-07, D3-01, D3-03, D3-05…09 | **D2-01** (MED): con CL-98b la orden mínima depende de stepSize/minQty, pero en el examen del demonio el spec es el del símbolo vivo de la ranura y el precio el sintético de 100 USD, así que en lotes gruesos la orden mínima del examen no es la real. **D2-02** (MED): la sonda única es de cartera y el examen reproduce las monedas en serie; una sonda abierta al final del tramo de la moneda k ya no recibe eventos que la cierren y bloquea toda sonda posterior. **D1-05** (LOW): la retención de `base_z` es el literal de 30 s del TTL; en el nodo 17 son el 83 % de su masa y entre 30 y 38 s el nodo sigue caliente. **D2-07** (LOW): `sonda::admite_sonda` no tiene llamadores; el núcleo compone la puerta con `trade_count == 0` y el riesgo con `< CIERRES_DE_SONDA`. **D3-01** (MED): `cl97_un_spot_sin_refrescar_no_forma_pares` no puede fallar (en 25 s ningún nodo madura). **D3-03**: `cf3_sonda_respeta_d4` sigue `#[ignore]` con la premisa vieja aunque CL-98 cerró D4. |
| **AGY** | D2-04, D2-05, D2-06, D3-02 | **D2-04**: nadie escribe `prospect_pressure`; tras Ω89 el núcleo la manda a 0 siempre, así que el impulso contrarian de Ω88 y el detector de Ω89 nunca corren en vivo (sus pruebas los ejercitan con una presión que el núcleo ya no produce). |
| **Dueño del `online_daemon`** (Línea C, fase C-F7) | D1-08 | Las features de entrenamiento del bosque sombra se leen en el muestreo tras el cierre y las de servicio cada 500 ms en vivo: desfase entrenamiento/servicio además del de claves (LC-F1-08). |

Los ya registrados (G053/D1-01 la hipótesis que paga: la base revierte pero el
voto apuesta por el futuro sin medir qué parte lleva F; D1-04 el centrado HAC,
que entregué en la ola 2; D1-03/D3-09 el doble voto StatArb + VECM; D3-06 el
spot caducado puesto a 0) se confirman con números nuevos.

**El examen del demonio acumula ya cinco defectos** (G049, G049b, D2-01, D2-02,
D1-08) y es el que promueve genomas en vivo: la autoevolución del sistema
depende de que juzgue bien. Está en la fase C-F7 de la Línea C. Si no lo van a
abordar pronto, la Línea G lo toma en la ola 5.

### 12.3 Arreglado en la ola 4

| ID | Zona | Defecto | Prueba |
|---|---|---|---|
| **LG-20** | telemetría | `forensic_audit.db` era una ruta común a demo y prod, guardaba lo no finito como 0 y el host le reenviaba el mismo genoma (~5 KB) cada 5 000 mensajes: ~15 MB/día sin lector. Ahora va por entorno (CL-48), lo desconocido es NULL y un genoma sólo se guarda cuando cambia. | `tests_lg20` (3, RED antes) |

### 12.4 Revisión de Ω90 (AGY, documentación)

El plan maestro integral de AGY repite como «invariantes sagrados» un tope de 2
posiciones y un suelo de 3 USD que el ledger QS-R1 ya mostró que no existen en
el código, y cita como vigente la huella del T-2 `998a7de2…` cuando la sonda
única la cambió a `a7c2da76…` (68 operaciones, PnL −0,164).

### 12.5 La meta con lo medido hoy

- T-2 (fixture lateral, 13 USD): 68 operaciones, acierto 42,6 %, PnL −0,164 USD.
  El fixture no tiene ventaja y el sistema pierde lo que cuesta operarlo:
  correcto, pero no dice nada de la meta.
- T-1: 19/144 genes cambian algo en el fixture; 125 no. La búsqueda evolutiva
  gasta casi todo su presupuesto (y su recuento de multiplicidad) en
  dimensiones sin efecto observable.
- Ventaja fuera de muestra: sin medir en ningún sitio con un procedimiento
  válido. La meta de ×2 en 3 días sigue sin evidencia a favor ni en contra.
  Los tres requisitos para medirla, por orden: un examen del demonio que juzgue
  bien (C-F7), un replay canónico multiactivo (Sol) y un criterio de evidencia
  por R neta frente a p* (LG-9 / CL-107).

## 13. Ola 5 (2026-10-12, base `59baf0c2`): genes muertos y revisión previa al merge de la Línea C

### 13.1 LG-21 — censo de los 144 genes

Censo estático ([`genes_censo.tsv`](audit/linea_g/genes_censo.tsv)): para cada
gen, quién lo lee fuera de su ciclo de vida (to/from_vector, default, random,
mutate, bounds, apply/current_from_arena), y si es sensible en el T-1.

| Estado | Dimensiones |
|---|---|
| Vivas (incluidas las 4 de las curvas TP/SL) | 95 |
| Sin lector, congeladas por **LG-21** | 35 |
| Sin lector, congeladas por LG-18 | 9 |
| Congelada por G0-4 (`capital_split_scalp`) | 1 |
| Vistas re-derivadas de las curvas (X-005) | 4 |

Los 35 de LG-21: los 5 pesos y los 2 recortes del PPO, los 7 de cooldown y
confianza macro, los 2 de parada dura, los 2 de arranque de Kelly, sinergia
(2), spot (2), recortes del ML (2), operadores tensoriales (2), fraccional (2),
régimen (2), BFT, memoria temporal, OBI de respaldo, velocidad VECM y la
normalización de volumen de Hawkes. Ninguno sensible en el T-1. Seguían
mutando, así que genomas que sólo diferían en ellos se juzgaban y promovían
como candidatos nuevos sin cambio de conducta. **Congelar no retira**: el
vector sigue teniendo 144, y conectar o retirar cada uno es decisión de su
dueño tras GENOME-GATE (C-F7 de la Línea C). La prueba D-650 de simetría del
hot-swap los fija ahora dentro de sus cotas (verificado por mutación).

### 13.2 Revisión previa al merge de la tanda de la Línea C (CL-101…CL-108)

Tres lentes y un verificador por lente sobre `origin/main..claude/tanda-cf4`
(51 archivos leídos), antes de que la Línea C fusione. Tabla:
[`ola5_tanda_cf4.tsv`](audit/linea_g/ola5_tanda_cf4.tsv). **26 hallazgos: 4
MED, 21 LOW, 1 refutado** (el freno de la estrategia es absorbente por decisión
del dueño, ADR-0017).

- **T1-02** (MED, CL-106): el umbral del freno duro es `K·r` con `r` = riesgo
  al stop sobre el capital ACTUAL, pero la caída se mide desde el PICO. Con
  nocional fijo (la orden mínima, el régimen de 13 USD) el umbral se aleja al
  caer: con stop 1 % dispara tras ≈49,5 stops en vez de 40, y con stop 2 % el
  umbral fino no se alcanza nunca y rige d* = 63,2 %. Arreglo propuesto: medir
  la caída en stops con el riesgo en USD (pico − capital)/s̄ ≥ K.
- **T2-1** (MED, CL-108): un archivo de estado corrupto, truncado o de otra
  versión se trata como ausente: el host arranca en cero sin decirlo, con el
  freno de la estrategia suelto.
- **T2-2** (MED): se restaura sólo parte de lo que aprende cada cierre (el
  sesgo epigenético y el estimador de Cramér-Lundberg no vuelven).
- **T2-3** (MED): el PnL realizado de sesiones anteriores entra en un
  acumulador que se lee como PnL de la sesión; la deriva contra el control del
  bosque sombra sale falsa.
- Entre los LOW, **T1-01**: la escala del proceso e sale de la geometría media
  (EWMA) y no de la de la posición que cierra; con órdenes de RR distinto
  conviviendo, el recorte a −1 de las ganancias grandes rompe la
  supermartingala bajo H0. **T1-08**: el Monte Carlo del cuantil del freno duro
  (4·10⁶ sorteos) corre con un mutex global en el camino de cada intención
  nueva. **T3-03**: el ADR-0009 renovado dice lo contrario de lo que hace el
  código con los pares de SHIB.

### 13.3 Lo que no se hizo

Las pruebas OPEN del examen del demonio (D2-01, D2-02) no caben sin tocar
`online_daemon.rs` (las funciones del examen son privadas). Los escenarios con
números están en la tabla de la ola 4; la decisión de C-F7 sigue con la Línea
C.

### 13.4 Línea H

Una tercera sesión Claude abrió la Línea H (`docs/PLAN_LINEA_H_CLAUDE_2026-10-12.md`)
con lo que §12.5 dejó escrito: un T-3 con una escalera de fixtures de ventaja
CONOCIDA, para medir si la cadena compone o mata una ventaja real y qué puerta
la mata. Es el instrumento que faltaba para hablar de la meta con evidencia.

## 14. Ola 6 (2026-10-12, base `15b6cc6f`): revisión del T-3 de la Línea H y el techo de fricción

### 14.1 Réplica

`t3_valle_de_capital` en release, en mi worktree: las cifras coinciden al
céntimo con las de la Línea H (el replay es determinista). 13/20/26 USD: 89
cierres y +10,8/+7,7/+6,0 %; **35/50/65 USD: 0 cierres, 6 914 rechazos
`fee_impact`**; 100 USD: 1 cierre; 215/500/1 000 USD: 89 cierres y
+3,9/+3,0/+2,0 %. Con las mismas 89 decisiones el porcentaje cae al crecer el
capital: el tamaño no escala con la cuenta.

### 14.2 Revisión (tres lentes y verificación; tabla en [`ola6_t3_linea_h.tsv`](audit/linea_g/ola6_t3_linea_h.tsv))

**31 hallazgos, los 31 confirmados (3 HIGH, 10 MED, 18 LOW).**

- **El instrumento (H1-01, HIGH)**: el estado oculto no está oculto. El ruido
  es uniforme y acotado, así que una vela con |r| > h − d revela el estado con
  certeza (canal de borrado: la columna «lectura» mide esa probabilidad); y
  mechas y volumen salen del mismo |u| que el ruido del retorno, de modo que
  (r, volumen) decodifica el estado en cada vela. El «techo» del oráculo no
  acota al motor: es un lector causal del OHLCV que el motor también recibe.
  Las ganancias en peldaños con oráculo negativo (a = 0,6–0,8) no pueden
  atribuirse a la cadena de Markov declarada. Comprobé aparte el look-ahead
  que añadió el verificador (`omni[34/35/43]` con el cierre de la vela en
  curso en `run_backtest_native`): ningún consumidor lee esos tres índices, así
  que es inerte y **no afecta al golden, al T-1 ni al T-2**.
- **El pago (H1-02)**: `b_real` es un cociente en USD; con las mismas 89
  decisiones baja de 1,40 a 1,04 sólo porque el nocional covaría con el
  resultado. No es el pago: hay que publicarlo en R por cierre.
- **LH-04 matizado (H1-03/H2-03/H2-07)**: «todas las órdenes van a la orden
  mínima» sólo está respaldado con 13 USD (micro_w = 1); con 20 y 26 USD ya hay
  órdenes mayores (PnL distintos con las mismas decisiones).
- **LH-02 confirmado y precisado (H2-01/H2-05)**: el apagón empieza hacia
  27,5 USD (no 15) y lo produce `f·L > lerp(1 %, 3,5 %, w)` tras el rescate de
  apalancamiento. **H2-02**: además es absorbente por moneda (el margen de
  exploración `kelly_cold` rige con PF desconocido o ≤ 1).
- **LH-03 (H3V-01/H3V-02/H3V-03)**: el host vivo hace lo mismo que el replay
  (≥ 50 USD, no sólo 1 000); el b de la envolvente se suaviza dos veces y en el
  cierre 30 el primer cierre pesa el 77–88 %; el umbral real es b > 1,27.
- **LH-05/LH-07 con la ley correcta**: la constante de la EMA21 es 10,5 velas
  (no 21; con d = 0,2 % el piso se cruza en la vela 19); el retroceso del
  enfriamiento es geométrico de razón √2 (1, 1, 2, 3, 4, 7, 8, 15… bases), no
  k·base.
- Menores del instrumento: fricción del techo 10 pb frente a los 13,3–14 pb que
  paga el motor; el techo resuelve el bracket con las mechas completas y el
  puente del motor sólo recorre 1/3 y 1/2 del camino; lo desconocido como 0;
  `REJECT_COUNTERS` global con pruebas en paralelo (usar `--test-threads=1`);
  la intervención «de un gen» cambia dos.

### 14.3 El techo de fricción, derivado ([`ola6_derivacion_friccion.md`](audit/linea_g/ola6_derivacion_friccion.md))

`fee_impact = f·L = f·N/M` es la fricción por unidad de **margen**. Con un
nocional dado, L sólo reparte ese nocional entre margen y préstamo: no cambia
la comisión en USD, ni el riesgo al stop, ni la fricción en R, ni la fricción
por unidad de capital. El veto castiga un artefacto contable y, leído en
espacio de riesgo, es un tope de tamaño que no mira p (H2-04).

Lo que decide si una orden paga su fricción sale del crecimiento log por
operación, `g(r) = p·ln(1 + r(RR − f_R)) + (1 − p)·ln(1 − r(1 + f_R))`, con
`f_R = f/SL`:

- `g'(0) = p·RR − (1 − p) − f_R`: la fricción entra con coeficiente 1.
- Viable ⇔ `p_LCB > p* = (1 + f_R)/(1 + RR) = (SL + f)/(TP + SL)` ⇔
  `f < p_LCB·(TP + SL) − SL`. Ése es el techo de fricción: en R, sin L, sin
  margen y sin capital, y es la misma p* de LG-9 y CL-107.
- Tamaño: `r* = p_LCB/(1 + f_R) − (1 − p_LCB)/(RR − f_R)` (½ por política),
  `N = r·C/SL`.

Con la geometría de referencia (SL 0,5 %, RR 2,25, f ≈ 13,3 pb): p* = 0,390;
en el peldaño fuerte, con la p del oráculo, el techo es f < 57 pb, y con el
acierto realizado y su cota (p_LCB ≈ 0,568) es f < 42 pb. La propuesta a la
Línea C: retirar `fee_impact` en favor de la puerta EV con p_LCB (que hoy usa
una p puntual y un multiplicador de fricción con literales, H2-09).

## 15. Ola 7 (2026-10-12, base `0c2906fb`): la escala rápida de adaptación y su límite de información

### 15.1 LG-22: detector de pérdida de ventaja (núcleo, sin cablear)

La ola 2 de la Línea H mide que ningún lazo desaprende: la p de la puerta EV
es un Platt sobre los últimos 512 cierres que no envejece mientras se opera
(LH-12). En E4 (ventaja → antipersistencia), con el #62, el motor hace 41
cierres al 31,7 % (×3,4 los del mismo tramo en frío), y sólo el freno de la
estrategia (CL-107) lo para, al final del tramo. Su regla: hace falta una
escala rápida por rama y escala, sobre la R neta frente a p*.

`DetectorPerdidaDeVentaja` (`god-engine-core/src/resultado_orden.rs`, junto
a LG-9) es esa escala, construida como e-detector (Shin–Ramdas–Rinaldo,
NEJSDS 2023) para que sus falsas alarmas tengan nivel:

- **H0**, por cierre y en orden de cierre: E[R_t | cierres previos,
  geometría] ≥ μ. Con μ = 0 (`nuevo`) la fuente paga su fricción
  (p ≥ p* = (SL + f)/(TP + SL)). Con `con_referencia(μ)`, conserva una
  ventaja declarada de μ R.
- **e-valor** del cierre: `e_t = 1 − λ·(r_t − μ)`, con `λ = φ/(r_max_t − μ)` y
  `r_max_t = (TP − f)/SL` de esa orden, conocida al abrir (λ predecible).
  Bajo H0, E[e_t] ≤ 1. Vale mientras `r − μ ≤ (r_max − μ)/φ_max`
  (2,5 veces con φ_max = 0,4): por encima el e-valor se recortaría a 0, y
  recortar las ganancias rompe la garantía. Con dos barreras no ocurre;
  esos cierres se cuentan en `fuera_de_garantia`.
- **e-CUSUM** por fracción φ ∈ {0,05; 0,1; 0,2; 0,4}:
  `M_t = e_t·max(1, M_{t−1})`, `M_0 = 0`. Está acotado por el de
  Shiryaev–Roberts, y la media de las cuatro fracciones es un e-detector:
  alarmar en `M ≥ ARL` da un tiempo medio hasta la falsa alarma ≥ ARL cierres
  para toda fuente que cumpla H0, sin fijar n. El suelo `max(1, ·)` olvida la
  historia favorable: una racha ganadora larga no retrasa la alarma.
- **`retardo_aproximado(rr, f_r, μ, E[R]_después, ARL)`** = ln(K·ARL)/g, una
  aproximación pesimista (×1,3–1,6 sobre lo simulado en cambios pequeños).
- **Por fuente y reiniciable** tras consumir la alarma. El freno de CL-107
  falsa la estrategia entera con H0 = +0,30 R y queda echado. Quién consume la
  alarma es decisión de sus dueños: Línea E / Codex (calibración) y Qoder /
  Línea C (núcleo).

**Pruebas** (11 en `resultado_orden`). Las tres mutaciones que la revisión
dejó pasar en la primera versión ahora se detectan: quitar el suelo, desplazar
la referencia y agregar con el máximo.

| Prueba | Resultado |
|---|---|
| E[e] del primer cierre en la frontera (RR 2,25 y 5; μ = 0 y 0,3) | 1 exacto (1e-12); < 1 con ventaja |
| ARL bajo H0 en la frontera, geometrías mezcladas | 3 813 cierres (cota 50) |
| +0,2 R durante 600 cierres y luego −0,3 R (ARL 500, sin reinicio en el cambio) | 0/200 falsas alarmas; retardo 303 (aproximación 421) |
| E4 (−0,24 R): μ = 0 y ARL 500 / μ = +0,3 R y ARL 20 | 480 / 56 cierres |
| Inversión a −1 R | 45,6 cierres |
| E[R] = 0: ventana de 512 negativa / detector (ARL 50) en alarma | 46 % del tiempo / 0,03 % de los cierres |
| Suelo: tras 50 ganancias y una pérdida | el estadístico exacto de una sola pérdida |

### 15.2 Lo que mide: la velocidad la fija la H0, y la escala del régimen no cabe en los cierres

Corrijo lo que prometía la reserva («alarma en pocos cierres»). La R por
cierre es dispersa (σ ≈ 1,58 R con RR 2,25): cada cierre informa poco y el
retardo depende sobre todo de la referencia de la H0. Cierres hasta la
alarma, simulados (RR 2,25, f = 0,27 R; 300–600 caminos):

| H0 | ARL | −0,2 R | −0,24 R (E4) | −0,3 R | −1 R |
|---|---|---|---|---|---|
| μ = 0 (paga su fricción) | 500 | ≈ 670 | ≈ 470 | ≈ 310 | ≈ 46 |
| μ = 0 | 20 | ≈ 220 | ≈ 176 | ≈ 127 | ≈ 25 |
| μ = +0,3 R (la de CL-107) | 500 | ≈ 131 | ≈ 112 | ≈ 94 | ≈ 31 |
| μ = +0,3 R | 20 | ≈ 63 | ≈ 55 | ≈ 48 | ≈ 17 |

- **Con μ = 0**, ninguna prueba con nivel sobre la R de los cierres ve el E4
  (41 cierres) a tiempo.
- **Con la ventaja declarada como μ** se ve en decenas de cierres. El precio:
  también alarma ante fuentes que siguen pagando su fricción con menos
  ventaja (0 ≤ E[R] < μ). Es lo que hace CL-107, y por eso es la primera
  puerta que reacciona en E4, al final del tramo.
- **La escala del régimen (10–20 velas, la que pide la Línea H) no cabe en
  los cierres**: el E4 cierra una vez cada ≈ 70 velas. Tampoco cabe en la
  predicción por vela de una sola moneda. Un acierto direccional que cae a
  0,317 aporta KL ≈ 0,069 nats por vela frente a 0,5 y ≈ 0,110 frente a una
  ventaja declarada de 0,55; eso da ≈ 40–110 velas (estimación con velas
  independientes). Si el cambio de estructura es común a las 13 monedas y se
  agrega la evidencia, la información por vela se multiplica por 13:
  ≈ 3–9 velas.
- **Regla que sale de aquí**: la escala rápida tiene que agregar información
  por hora, no reinterpretar los mismos cierres. Las fuentes son la predicción
  frente a lo realizado en cada vela, la agregación entre monedas y escalas, y
  el libro sombra de las intenciones vetadas
  (`audit_engine::shadow_ledger::LibroSombra`, QS-R4a, sin gancho en el
  núcleo; 929 rechazos en 50 h en frío, ≈ 18 por hora). LG-22 sirve tal cual
  sobre cualquiera de esas series (el ARL se mide entonces en sus unidades) y,
  sobre los cierres, para las inversiones y como prueba con nivel por fuente.

### 15.3 Revisión de Ω92 (AGY)

Leí el diff de `0b18e589` (`consejo_seniors.rs`) contra el riesgo vivo.

- **F3.4-01, correcto.** `θ_i = Σ s_i·r / Σ |r|` acredita a cada asiento por
  la esperanza de su apuesta direccional, no por acierto contra ½. Con |r|
  constante se reduce exactamente al acierto, y con RR > 1 sube un asiento
  que respalda una fuente rentable con acierto < 50 %. Es la regla que pidió
  LF-14. Queda la selección: sólo aprende de lo ejecutado, igual que antes.
- **F3.4-02, correcto.** El Auditor deja de co-vetar el riesgo causal que ya
  veta `SeniorCausal`, y la supermayoría vuelve a poder anular un único veto.
- **F3.4-06 sólo decide sin riesgo medido.**
  - El umbral nuevo es `d*·(0,95 − 0,10·s) ∈ [0,537; 0,600]`, sobre la misma
    caída que lee el riesgo (mismo pico, `lib.rs` ≈ 8196).
  - El freno duro del riesgo (CL-106) es `min(K·r, d*)` con K ≈ 31–33 stops:
    ≈ 7,8 % de la cuenta con stop del 0,5 % y ≈ 15 % con stop del 1 %. Con
    riesgo medido, el riesgo rechaza muy por debajo de 0,54 y el veto del
    consejo nunca es el que decide. Sólo manda en frío, cuando el freno es
    d* = 0,632, en la franja [0,537; 0,632).
  - Repite el dato: la caída queda vetada por dos componentes, contra el «un
    dato, un veto» que el mismo commit aplica al Auditor.
  - El literal `1 − 0,05^(1/3)` duplica
    `risk_engine::drawdown::drawdown_de_falsacion`. Metacortex no depende de
    risk-engine: si el riesgo cambia c o α, el consejo no se entera.
- **Propuesta (AGY, consejo)**: pasar en el payload el umbral vivo del riesgo
  (`umbral_freno_duro_de`) y que el consejo module la convicción relativa a él
  sin vetar, o retirar el veto de caída del consejo y dejar su modulador por
  racha. Cualquiera de las dos deja un solo veto sobre la caída.

### 15.4 Revisión adversarial de LG-22 (tres lentes: estadística, código y pruebas, afirmaciones)

**Sin HIGH.** El núcleo estadístico se confirmó: e-valor válido, cota de
Shiryaev–Roberts, media de e-detectores, versión logarítmica equivalente
(también con e = 0) y la inversa de p exacta. Lo corregido:

- **Prueba del retardo (MED)**: reiniciaba el detector en el instante del
  cambio, un dato que el detector real no tiene, y así no veía el suelo del
  CUSUM. Ahora sólo se reinicia tras una alarma. Sin suelo, el retardo pasa
  de ≈ 300 a ≈ 1 300 y la prueba falla.
- **Cifras de E4 (MED)**: eran las anteriores al #62. Ahora son 41 cierres,
  acierto 0,317 y 31 rechazos `estrategia_falsada` al final del tramo. La
  comparación con CL-107 era incompleta: la diferencia está en la H0 (+0,30 R
  frente a E[R] ≥ 0). De ahí sale la referencia μ.
- **La prueba de ARL no tenía potencia (LOW)**: el ARL real es ≈ 78 veces la
  cota. Se añade la igualdad exacta E[e] = 1 en la frontera.
- **«≈ 1 000 cierres» venía de la aproximación (LOW)**: lo simulado es
  ≈ 670. Las cifras de la tabla son ahora simuladas.
- **El recorte `max(0)` limita la garantía (LOW)**: vale con
  `r − μ ≤ (r_max − μ)/φ_max`. Está documentado y se cuenta en
  `fuera_de_garantia`.
- **`alarma(1.0)` con 0 cierres (LOW)**: ahora `M_0 = 0`.
- **La demostración de la ventana no ejecutaba el detector (LOW)**: ahora lo
  ejecuta sobre la misma serie.
- **Contadores vitalicios (INFO)**: `cierres_totales` es el denominador de
  `fuera_de_cota` y `fuera_de_garantia`.
- **Filtración por orden de cierre con posiciones solapadas (INFO)**: queda
  documentada. En simulación, el ARL no cambia de forma material.
