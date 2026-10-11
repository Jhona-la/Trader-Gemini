# Plan de Línea F — revisión independiente de composición, lazos adaptativos y campo multiactivo (2026-10-11)

> Dueño: Claude (sesión cloud «laughing-fermi»), prefijo **LF-n**, rama
> `claude/laughing-fermi-xoqnb7`. Base de este plan: `origin/main`
> `977ba2f8` (Ω84, #722). Las citas `ruta:línea` son de esa revisión:
> re-grep del símbolo antes de editar.
>
> Este documento **no crea otro censo ni otro recorrido**. Se acopla a los
> que ya existen y les añade un frente que nadie tenía asignado:
>
> - recorrido R0–R9 (Qoder/AGY y Codex/Sol):
>   [PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md](PLAN_REVISION_ARCHIVO_POR_ARCHIVO_2026-10-07.md);
> - fases C-F0…C-F11 y lista de recibos por archivo (Línea C, Claude):
>   [PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md](PLAN_LINEA_C_DESDE_LA_BASE_2026-10-11.md),
>   [audit/REINICIO_BASE_FASES_C_2026-10-11.tsv](audit/REINICIO_BASE_FASES_C_2026-10-11.tsv);
> - reinicio coordinado y protocolo de recibo (Codex SI):
>   [audit/REINICIO_COORDINADO_BASE_2026-10-11.md](audit/REINICIO_COORDINADO_BASE_2026-10-11.md);
> - plan rector (AGY, Ω81):
>   [PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md](PLAN_MAESTRO_REVISION_BASE_UNIVERSO_ESPECTRAL_2026-10-11.md);
> - contratos entre agentes: [PLAN_MAESTRO_SINCRONIZACION.md](PLAN_MAESTRO_SINCRONIZACION.md) §45.

## 0. Qué hace la Línea F y por qué hacía falta

Desde el 9 de octubre, AGY empuja olas directamente a `main` (Ω54…Ω84) sin
PR ni revisión independiente, y cada una cambia el camino vivo. La Línea C
midió el coste de #720 (+15,2 µs por evento) y por eso nació #722; pero
**nadie revisa la semántica de lo que entra sin PR**. Los barridos de
Qoder (R6, R7) y de Línea C (C-F1) encontraron, cada vez, que los arreglos
de una ola cargaban un defecto nuevo («todo fix carga bug», cuatro rondas
seguidas). La Línea F toma tres frentes sin dueño:

| Frente | Pregunta | Salida |
|---|---|---|
| F-A. Revisión de composición | ¿Lo que entró a `main` sin PR cambia conducta que su mensaje no declara? | Hallazgos LF-n con test que falla antes del arreglo; arreglo sólo si el archivo no está reservado, si no, pedido al dueño. |
| F-B. Lazos adaptativos | ¿Qué «aprendizaje» está realmente cerrado en vivo, cuál está abierto (sólo telemetría), cuál es absorbente y cuál es un literal disfrazado? | Mapa §4 con escritor, lector, reloj y clasificación por lazo. |
| F-C. Campo multiactivo y continuo en τ | ¿Dónde el código vivo colapsa (moneda, τ) a un escalar por moneda o global y se cruzan señales entre activos o escalas? | Hallazgos §5 con escenario numérico. |

La Línea F **no posee archivos de producción**: revisa los de todos. Edita
sólo (a) sus documentos, (b) tests nuevos propios y (c) arreglos mínimos de
regresiones introducidas por commits sin PR cuando el archivo no tiene
reserva activa en el buzón; en todo otro caso entrega reproducción al dueño.

## 1. Estado medido de Git (2026-10-11, al abrir la línea)

| Ref | SHA | Estado |
|---|---|---|
| `origin/main` | `977ba2f8` | Ω84 (#722) de AGY, empujado sin PR. |
| `claude/auditoria-deslizamiento-apalancamiento-sqtc08` | `ab2f4671` (avanza) | Línea C, PR #40 abierto (C-F2). **Activa: no tocar.** |
| `claude/elegant-euler-mmtht4` | `f1a70192` | Línea E. Ancestro de `main` (0 commits propios). **Se intentó borrar del remoto: el proxy git de la nube corta el `push --delete`** (mismo bloqueo que registró Línea C). Queda para el dueño o una sesión local: `git push origin --delete claude/elegant-euler-mmtht4`. |
| `claude/laughing-fermi-xoqnb7` | igual a `main` al abrir | Esta línea. Alguien la había borrado del remoto; se re-publica con este plan. |

PR abiertos: sólo #40 (Línea C). Lo que AGY empuja sin PR no tiene CI
previa al merge: la CI de push a `main` es la única red.

## 2. La meta, en números que pueden refutarla

Se adopta la definición de Codex SI (REINICIO §2): sobre patrimonio neto
MTM, \(G_{72}(t)=\log V(t+72h)/V(t)\) y la referencia \(\log 2\) por
ventana, es decir **0,231049 de crecimiento log por día (≡ 25,99 %
diario compuesto)**. La pregunta útil no es «¿se puede?» sino «¿qué
ventaja y cuántas operaciones independientes por día exige?».

Para una apuesta binaria con relación R:R = b = 2,25 (el piso de RR que
usan los planes), probabilidad de acierto p y fracción del capital
arriesgada al stop f, el crecimiento log por operación es
\(g(f)=p\ln(1+fb)+(1-p)\ln(1-f)\), máximo en \(f^*=p-(1-p)/b\) (Kelly).
Operaciones independientes por día necesarias = 0,231049 / g:

| Ventaja neta por operación (en R) | p que implica | f* (Kelly pleno) | Kelly pleno | ½ Kelly | ¼ Kelly | Riesgo actual ≈ 0,196 % (1 orden mínima, SL 0,5 %) |
|---|---|---|---|---|---|---|
| +0,02 R | 0,314 | 0,89 % | 2 609 /día | 3 474 | 5 949 | 6 632 |
| +0,05 R | 0,323 | 2,22 % | 420 | 558 | 954 | 2 469 |
| +0,10 R | 0,338 | 4,44 % | 106 | 140 | 239 | 1 207 |
| +0,20 R | 0,369 | 8,89 % | 27 | 36 | 60 | 597 |
| +0,4625 R (p = 0,45, el supuesto de los planes Ω) | 0,450 | 20,6 % | 5,2 | 6,8 | 11,4 | 256 |

Lectura (calculado, no medido; independencia supuesta, y con correlación
entre operaciones el número crece):

1. Con el tamaño actual (riesgo ≈ 0,2 % por operación: orden mínima de
   5,10 USD y stop de 0,5 % sobre 13 USD) la meta exige cientos o miles de
   operaciones independientes al día **aun con ventajas que nadie ha
   medido**. Coincide con QS-R0 de Línea E: la palanca es el riesgo por
   operación, y sólo es segura con ventaja OOS medida.
2. El supuesto p = 0,45 con RR 2,25 de los planes Ω equivale a +0,46 R
   por operación: una ventaja excepcional que **ningún fixture ni tape ha
   mostrado** (el T-2 mide acierto real 0,20 y p calibrada mediana 0,377;
   ledger §7.5). No debe usarse como dato.
3. Por tanto el orden de trabajo para la meta es: (i) medir ventaja OOS
   neta con etiquetas honestas y nulos; (ii) dimensionar en espacio de
   riesgo sobre la cota inferior de esa ventaja (D1 de ADR-0016); (iii)
   sólo entonces subir frecuencia/activos. Ninguna teoría nueva sustituye
   (i).

Falsadores: G₇₂ por ventana con su distribución (no la media), fracción de
ventanas con G₇₂ ≥ log 2, drawdown e insolvencia; ventaja OOS con
intervalo en las dos mitades del examen; independencia de operaciones
(correlación de resultados por moneda y por τ).

## 3. Fases de la Línea F y su acople con R / C-F

Cada fase termina con: hallazgos LF-n en §6, pruebas ejecutadas con su SHA
y comando, entrada en el buzón y en MEMORIA, y PR propio.

| Fase LF | Contenido | Acople | Estado |
|---|---|---|---|
| LF-F0 | Git, ramas, PRs, recibo de qué llegó a `main` | R9 / C-F10 | Hecha (§1). |
| LF-F1 | Revisión de composición de Ω82–Ω84 (registro, reloj, getters estrictos, `sym`) | R1 / C-F1 (contratos) | Esta ola (§6). |
| LF-F2 | Mapa de lazos adaptativos (F-B) | R6 / C-F7 y §5 de Línea C | Esta ola (§4). |
| LF-F3 | Campo multiactivo y τ (F-C) | R1/R5 / C-F3, C-F4 | Esta ola (§5). |
| LF-F4 | Vetos: alcanzabilidad, frontera, recuperación desde cero posiciones, precedencia | R5 / C-F3–F5, protocolo REINICIO §6 | Siguiente ola. Por cada veto del registro y cada `return`/`clamp` del gate: test de admisión y de rechazo que llegue al gate, ambos lados de la frontera, y prueba de que no es absorbente. |
| LF-F5 | Revisión de cada ola nueva que entre sin PR, en las 24 h siguientes | transversal | Permanente. |

Recorrido archivo por archivo: la Línea F usa la lista
[REINICIO_BASE_FASES_C_2026-10-11.tsv](audit/REINICIO_BASE_FASES_C_2026-10-11.tsv)
(1 584 rutas, una fase dueña por ruta). No crea otra. Su cola propia son
los archivos tocados por commits sin PR y sus consumidores directos; cada
revisión deja recibo con el formato de REINICIO §3 (ruta, blob, rangos
leídos, contrato, productores, consumidores, hallazgos, pruebas, límites).
Revisar una función no cierra un archivo.

## 4. Lazos adaptativos (F-B)

«Autoadaptativo» se define operativamente: **observación → actualización →
consumo en una decisión viva**, con reloj causal, y con salida de los
estados absorbentes sin intervención humana. Cada lazo se clasifica:

- **CERRADO-VIVO**: escritor y lector vivo verificados con `ruta:línea`;
- **ABIERTO**: se actualiza pero ninguna decisión viva lo consume (o el
  lector existe y el escritor está muerto);
- **ABSORBENTE**: hay un estado del que el lazo no sale solo (p. ej. un
  gate que bloquea las operaciones que darían la evidencia para reabrirlo);
- **LITERAL**: lo «adaptativo» lo decide una constante en el camino vivo.

El mapa medido está en §4.1 (se completa con la revisión de esta ola).

### 4.1 Mapa (977ba2f8, por lectura de código; «verif.» = la Línea F releyó las líneas)

| Lazo | Lector vivo | Reloj | Clase | Nota |
|---|---|---|---|---|
| Calibrador Platt por moneda + global | `win_probability` → veto EV (REJ_EV) | cierre; olvido 12 h por tiempo de evento | CERRADO-VIVO, lento | Sólo aprende de lo ejecutado (LC-F7-15, XLIV-5). |
| `conformal_by_coin` | `conformal_accept_*` → voto del filtro de reversión | cierres; ventana 200, sin olvido | CERRADO-VIVO débil | 30 cierres por moneda antes de actuar. |
| Epigenética: umbral | gate de confianza ×[0,75; 1,35] | cierre; τ de decaimiento 900 000 ms literal; reloj sólo con eventos de depth | CERRADO-VIVO / LITERAL | En cintas sin depth el decaimiento no corre. |
| Epigenética: sesgo de tamaño | `effective_confidence` → tamaño | cierre; sin olvido | ABSORBENTE parcial | Escalar por moneda dominado por cierres rápidos (LF-7). |
| `online_learner` (Kalman TD) | ninguno (diagnóstico) | cierre | ABIERTO | D-693. |
| Pesos Hedge del ensamble | `combined()` | vela / cierre | ABIERTO en la práctica (LF-13) | Ninguna moneda tiene dos modelos: los pesos no cambian `ml_prob`. |
| Convicción por rama (`TasaAcierto`) | confianza → gate y tamaño | cierre; sin reloj ni persistencia | CERRADO-VIVO, referencia equivocada (LF-14, K-02) | Cota de Wilson contra 0,70 de confianza: una rama con 40/100 (+0,30 R) queda vetada. |
| Freno del bosque | rama/bosque | cierre | CERRADO-VIVO | LC-F1-18. |
| Pesos por habilidad `SkillMotores` (#626/#661) | consenso espectral → orquestador (#624) | maduración de bloques; e-proceso sin olvido | LITERAL en la práctica | Umbral de familia 8 320 ≈ 820 maduraciones: días de uptime; todos en el piso 0,15 (LC-F2-05). |
| τ por habilidad (#594) | `dominant_tau_ms` | maduración de bloques | casi siempre respaldo (argmax de energía) | R7-R2-C-1. |
| Bosques `_MOTOR` | inferencia | archivos en disco | CONGELADO | Sólo entrenamiento offline (ADR-0008). |
| Darwin `evolve_online` | — | 60 s pared | ABIERTO (apagado por dos variables) | — |
| Promoción del demonio + DSR | `apply_to_arena` | 60/180 s; sólo armado | ABSORBENTE (`cumulative_trials` sólo crece) | LC-F7-07. |
| Rollback / Ville del demonio | rollback del genoma | muestreo 500 ms de PnL realizado | **ABIERTO en prod y sesgado al armarse → corregido (LF-3, LF-4)** | verif. |
| Umbrales del bosque online | umbrales ML; freno del núcleo | 500 ms | CERRADO en demo | LC-F7-05/06. |
| Envolvente de Kelly | apalancamiento del envío; veto del host | cierre; Beta sin olvido, persistida | **ABSORBENTE fuera de la banda micro (LF-12)** | verif. |
| `riesgo_por_operacion` | veto de drawdown, grupo, demonio | cada orden validada (no enviada) | CERRADO-VIVO, etiqueta mal especificada | LC-F4-03; además es global (LF-11). |
| Drawdown / latch | toda entrada | — | ABSORBENTE (sin rearme automático) | ADR-0015, LC-F4-10. |
| PPO | `micro_score` | cierre; una política global | CERRADO-VIVO, recompensa escalada por τ | LC-F7-13, LF-9. |
| Pesos de asientos del consejo | `compute_weights` | cierre; un tracker global | **CERRADO-VIVO por tasa de acierto (LF-14)** | verif. |
| Prospect Theory | asiento del consejo | — | LITERAL por diseño | λ = 2,25, α = β = 0,88. |

**Patrón común (el hallazgo central de esta ola).** Los lazos que sí
deciden puntúan **acierto direccional contra 0,5**, mientras la doctrina
opera con RR ≥ 2,25, cuyo punto de equilibrio es p* = 1/(1+RR) ≈ 0,31.
Con 35 % de acierto y esperanza +0,14 R por operación, esos lazos
aprenden que el sistema «se equivoca» y frenan precisamente las
operaciones que ganan dinero: convicción por rama (K-02), pesos del
consejo (LF-14) y el vigilante de rollback (LF-3, ya corregido). Arreglo
único propuesto: **toda asignación de crédito se puntúa por esperanza neta
frente al punto de equilibrio de la geometría de la orden** (o por log-loss
de una probabilidad calibrada), nunca por acierto contra 0,5.

## 5. Campo multiactivo continuo en τ (F-C)

Doctrina (ADR-0014): no hay scalping/swing; cada posición vive en su τ
continuo (`Position::entry_tau_ms`), hasta 3 ranuras por moneda separadas
en ln τ; el mercado es un campo \(\mathcal S(\text{instrumento}, \tau, t)\)
con incertidumbre. Las pruebas transversales que la Línea F exige a todo
cambio del camino vivo:

1. **Permutación de activos**: permutar monedas junto con sus specs,
   modelos y estado deja las decisiones permutadas igual.
2. **Aislamiento**: escribir el estado de la moneda A no cambia la
   decisión de B salvo dependencias declaradas (correlación, margen común).
3. **Aislamiento entre τ**: un cierre a τ = 4 h no cambia la admisión de
   un candidato a τ = 30 s de la misma moneda salvo por margen/riesgo
   común declarado.
4. **Espejo largo/corto** con costes simétricos.
5. **Invariancia a la cadencia** de consulta y a eventos tardíos,
   duplicados o con igual timestamp.

Hallazgos de esta ola: §6, LF-F3.

## 6. Hallazgos de esta ola (base 977ba2f8)

Método: tres revisores en paralelo (composición de Ω82–Ω84, lazos
adaptativos, multiactivo/τ), sólo lectura; la Línea F re-verificó contra el
código cada hallazgo de severidad alta antes de registrarlo. «Arreglado» =
test que falla antes del cambio y pasa después, en esta rama.

| ID | Sev. | Hallazgo | Evidencia | Estado / dueño |
|---|---|---|---|---|
| LF-1 | MED | Ω84 cambió el respaldo de identidad del slot de `""` a `"BTCUSDT"`: un slot sin universo ni spec se convierte en BTC (hereda `BTCUSDT_MOTOR`, escribe claves `BTCUSDT_*`, alimenta el líder del lead-lag con su propio flujo). Contradice ADR-0011 y B3.18b. Latente en vivo; alcanzable en `run_booktick_replay` (no registra identidades a propósito) y `votes_export`. | `god-engine-core/src/lib.rs:3104-3105`; `symbol_registry.rs:231` (specs por defecto vacíos) | **Arreglado** (respaldo `""`; conserva la unificación de identidad de Ω84). Amplía LC-F1-32 (Línea C). |
| LF-1b | INFO | Efecto no declarado de Ω84: al unificar la identidad, el aprendizaje al cierre (Hedge, rama, bosque, `record_trade_outcome`) empezó a funcionar en replays sin identidad (antes el sello de entrada y el filtro del cierre comparaban cadenas distintas). Los datasets de `votes_export` antes y después de Ω84 difieren. T-1/T-2 no lo ven (registran BTCUSDT en el slot 0). | `lib.rs:3823,3830` vs `:8067,8092,8130` | Registrado; avisar a GLM (dataset L2). |
| LF-2 | MED (latente bajo Ville) | El arreglo C-2 de AGY (IC de grupo `qo_613_rho_tau` a la τ operativa unificada) lo sobrescribe en cada evento la publicación MG05 de Codex dentro de `process_tick_dual`, con el nodo crudo. Y el nodo más cercano se elegía en distancia lineal sobre una malla geométrica (40 s → nodo de 17,2 s, fuera de la banda). | `lib.rs:1777-1782` vs `:2924-2930`; `evidence_publication.rs` | **Arreglado** (las dos rutas usan `tau_operativa_unificada`; nodo en ln τ). Sigue abierto: el veto debería usar la τ de la **orden** candidata (Codex D / Línea C). |
| LF-3 | HIGH | El vigilante de rollback del demonio no era una prueba: revertía cuando un e-proceso a favor del genoma caía 50 % desde su pico, y recortaba cada TP a +1 R. Monte Carlo (400 trayectorias i.i.d., RR 2,25): revierte el 100 % de los genomas con +0,14 R y +0,30 R por operación. | `online_daemon.rs:1143-1144, 1208`; `ville_e_process.rs:106` | **Arreglado**: `VigilanciaDegradacion` apuesta en contra (H0: E[X] ≥ 0 sobre [−1, RR máx]) y revierte sólo si la riqueza supera 1/α. Falsa reversión medida ≤ 0,0425 con esperanza 0; detecta −0,67 R en ~27 operaciones. |
| LF-4 | HIGH | La vigilancia de rollback se armaba detrás del `return` del gate de evolución en vivo: en prod (desarmado por defecto) nunca se armaba, contra el doc de `live_evolution_armed` y el log del host. **Arreglar LF-4 sin LF-3 habría encendido en prod un vigilante que revierte genomas rentables.** | `online_daemon.rs:1356` vs `:1375-1392` | **Arreglado** junto con LF-3 (el armado precede al gate). |
| LF-5 | HIGH | Enfriamiento, racha de pérdidas y anti-whiplash son un escalar por moneda atado a la τ de la última ranura que cerró: un cierre a 4 h bloquea durante 4 h (480× su τ) a todo candidato de 30 s de esa moneda, aun tras ganar; tres pérdidas a 30 s dejan entrar un candidato de 4 h a los 4 min; una ganancia a 4 h borra la racha de tres pérdidas rápidas. Contradice el propio comentario D-622 (`lib.rs:6677-6679`). | `lib.rs:5561-5572, 3909-3912, 7210, 7256`; `stateful_engine.rs:282-311, 521-534` | Abierto → **Línea C** (C-F3). Propuesta: enfriamiento por banda de ln τ (las ranuras ya están separadas Δln τ ≥ 0,8); cambia conducta viva: T-1. |
| LF-6 | MED | El filtro de spread se juzga a la τ dominante cruda y lo comparten candidatos de toda τ: con τ_dom = 30 s veta una intención de 4 h con 2 pb de spread; con τ_dom = 4,9 h deja pasar un candidato de 30 s con spread varias veces su recorrido. | `lib.rs:5522-5531` | Abierto → Línea C (relacionado LC-F1-14). |
| LF-7 | MED | Sesgo y umbral epigenéticos: un escalar por moneda, dominado por cierres rápidos (una pérdida a 30 s mueve el sesgo ~19× más que una a 4 h), que dimensiona órdenes de toda τ; ya existe la versión por τ (`apply_epigenetic_outcome`) pero el riesgo lee el escalar. | `state.rs:422-432`; `risk-engine/src/lib.rs:579-584, 1087-1094` | Abierto → Línea C / AGY. |
| LF-8 | MED | La pertenencia a «la misma apuesta» usa correlación de los últimos 512 ticks (segundos) e ignora la τ de la orden: un ETH a 4 h con ρ₄ₕ ≈ 0,85 frente a un BTC abierto queda fuera del grupo si ρ_tick ≈ 0,4. | `correlation_guard.rs:593, 631-634, 657` | Abierto → Codex (Línea D). |
| LF-9 | LOW-MED | Una sola política PPO para todas las monedas y τ, con recompensa no normalizada por σ(τ). | `lib.rs:4239, 4356, 5371`; `online_ppo.rs:101-115` | Conocido (plan C fila 28); detalle nuevo. |
| LF-10 | LOW | Dedup núcleo↔bracket por símbolo y 5 s: dos ranuras del mismo lado que cierran en 5 s pierden el segundo cierre de la envolvente. El arreglo propuesto en F3-C5 (símbolo+lado) no lo cubre. | `god_engine.rs:2918, 3508-3513, 3858-3863` | Extiende F3-C5 → Línea C. |
| LF-11 | LOW | El veto de grupo usa la EWMA global `riesgo_por_operacion` como riesgo del candidato: tras órdenes a 4 h (SL ~1,5 %) un candidato a 30 s (SL ~0,3 %) se cobra ~5×. | `risk-engine/src/lib.rs:646-650, 1433-1442` | Abierto → Línea C. |
| LF-12 | HIGH | La envolvente de Kelly bloquea la cuenta para siempre al salir de la banda micro (capital ≥ 10× nocional mínimo ≈ 50 USD): con ≥ 30 operaciones y Kelly conservador ≤ 0 el envío queda en 0, no se registran operaciones, el posterior no cambia (sin olvido, persistido). Un historial rentable (35 %, RR 2,25) da Kelly conservador −0,12 con n = 60 y −0,036 con n = 200. | `kelly_envelope.rs:387-393`; `god_engine.rs:4149-4151, 4324-4330` | Abierto → Línea C + E (D1/D4 de ADR-0016). |
| LF-13 | MED | Los pesos Hedge del ensamble no afectan ninguna decisión: ninguna moneda tiene dos modelos (DarkAlpha sólo en BTCUSDT; sólo hay `_MOTOR` de ATOM, BNB y NEAR), y `combine()` normaliza sobre los que opinan. | `ensemble.rs:198-235`; `lib.rs:4818-4846` | Abierto → GLM (B). |
| LF-14 | HIGH (diseño) | Los pesos de los asientos del consejo se calculan por acierto direccional contra el signo del retorno: con RR 2,25 y 35 % de acierto, los asientos que apoyaron las operaciones rentables puntúan 0,35 y bajan; los que disentían puntúan 0,65 y suben. Mismo defecto que K-02 (convicción por rama). | `consejo_seniors.rs:1390-1399, 1465-1504` | Abierto → Qoder / AGY (metacortex) con la regla de §4.1. |
| LF-15 | LOW-MED | Dos familias de lectores del registro discrepan ante NaN desde Ω82: `get_scoped_parameter`/`get_scoped` devuelven la primera clave existente aunque sea NaN; `get_scoped_val_or` la salta y cae a la moneda o al global. | `omniscient-registry/src/lib.rs:350-379, 382-407` | Abierto → AGY (dueño del registro). |
| LF-16 | LOW | El reloj de eventos del registro (Ω84) es un `store` sin máximo (puede retroceder) y sella también escrituras de otros hilos (spot, OI, reconciliación) con la última hora del núcleo. Hoy ningún lector consulta los sellos: la «frescura» prometida en Ω82/Ω84 no se usa. | `omniscient-registry/src/lib.rs:162-165`; `spot_feed.rs:119-121` | Refina LC-F1-35. |
| LF-17 | LOW | Documentación: `evidence_publication.rs` decía que el registro normaliza no finitos a cero (falso desde Ω82; corregido aquí). Ω84 anuncia «cero heap» pero `sym` sigue siendo un `String` clonado por tick; `with_symbol` no tiene llamadores. | `evidence_publication.rs:3-5`; `symbol_registry.rs:140, 146-158` | Doc corregido; resto → AGY. |

**Ω85 (#723, `369eb34f`, empujado a `main` durante esta ola) — revisado,
sin defecto de conducta.** Los búferes de Yang-Mills/Hodge pasan de clonar
`Vec` a arreglos en pila de 32 (MAX_COINS = 30): mismas entradas, misma
caducidad a 10 s. `FlowImpulse` y `RenyiTsallis` leen con
`get_scoped_val_strict` las mismas claves (`{SYM}_k`, después `c{id}:k`); la
única diferencia es que una clave de símbolo NaN ahora cae a la de la moneda
en vez de anular el voto (matiz de LF-15; ambas las escribe `set_reg` a la
vez, así que no cambia el voto en la práctica). La cifra «76 % de
asignaciones» no está medida con el medidor de Línea C.

Refutado por los revisores (no reabrir): el símplex de régimen BTC no
depende de `sym`; el formato de clave de los lectores estrictos de Ω83 es
`c{id}:` (igual que `set_for_coin`); `get_global_suffixed` construye la misma
clave que el `format!` anterior; ningún par de núcleos comparte registro;
`online_learner` no entra en `ml_prob`; la reserva de margen entre monedas
está serializada en un hilo; la simetría largo/corto de las puertas del
continuo se mantiene salvo los casos ya archivados.

## 7. Teoría: estado, siguiente paso y prueba que la falsa

Se aplica ADR-0005 (existe / parcial / candidata / especulativa; la
medición en datos reales es el portón). La tabla completa por dominio está
en §3 del plan de Línea C y en la Parte IV del plan rector de AGY; aquí
sólo se fija **el orden** y se corrige lo que esos planes presentan como
hecho sin serlo.

| Pieza viva | Siguiente paso | Prueba que debe pasar antes de cablear | Dueño propuesto |
|---|---|---|---|
| Bosque aleatorio por moneda (`NanoForest`) | Gradient boosting por moneda con etiqueta de **primer toque** TP/SL del lado y log-loss OOS; después calibración (isotónica/Platt con edad por dato). | Log-loss y Brier OOS mejores que el bosque y que la base, en las dos mitades, con el mismo presupuesto de datos; paridad 48D train/serve. | GLM (B) |
| Calibración por moneda | Calibración condicional a (moneda, τ) con olvido por tiempo y corrección del sesgo de selección (sólo aprende de lo ejecutado). | Contrafactual del libro sombra: la p de las intenciones vetadas no difiere de la de las ejecutadas a igual score. | Línea E (QS-R4) |
| Espectro temporal y τ* por habilidad (#594, #661 Ville) | e-valor de familia también para la τ de la **orden** (no sólo telemetría); incertidumbre de τ* propagada al tamaño. | En ruido barajado, la τ elegida no tiene habilidad y el tamaño cae a la sonda. | Qoder (A) |
| Primer toque analítico (BM con deriva, `tp_sl.rs`) | OU / difusión con saltos (Hawkes en la intensidad) para el primer toque; volatilidad rugosa sólo si H < 0,5 se mide. | Frecuencia empírica de TP antes de SL en tapes vs la fórmula, por decil de score. | Codex (R8-A) |
| Kelly / ruina (Cramér-Lundberg, LCB) | Dimensionar en espacio de riesgo: riesgo = fracción de Kelly sobre la cota inferior de la ventaja, nocional = riesgo / SL, apalancamiento = consecuencia (D1). | Con ventaja 0 el tamaño es la sonda mínima; con ventaja medida, crecimiento log OOS > 0 en cada mitad. | Línea C + E |
| Dependencia entre activos (correlación, RMT, Hodge del contagio) | Riesgo de cartera monetario ponderado por posición, τ y stop, con reserva atómica de margen. | Dos admisiones concurrentes nunca gastan el mismo margen; correlación desconocida = adversa. | Codex (D) |
| Evolución (Darwin, demonio, Ville) | Multiplicidad acumulada entre reinicios; promoción sólo con DSR/e-valor OOS; rollback al linaje correcto. | Registro de consultas al holdout; un genoma promovido sobrevive al reinicio y el rollback restaura el padre exacto. | GLM / Codex |
| Conducta de la multitud (L/S ratio, liquidaciones, Prospect Theory) | Tratarlo como **observable** con contribución marginal OOS, no como «lectura de pensamientos». | Ablation: quitar el asiento no empeora el log-loss OOS ⇒ se retira. | Qoder |

**Teorías «del milenio» y analogías físicas — triage honesto:**

- *Navier–Stokes*: el «número de Reynolds del libro» es una razón de
  magnitudes de microestructura con nombre físico; no resuelve ninguna
  ecuación. Se conserva sólo si su contribución marginal OOS es positiva.
- *Yang–Mills*: con todos los pares cotizados contra USDT, la holonomía de
  log-precios con β = 1 es **idénticamente cero** (los ciclos telescopian;
  R6-C2). Con β estimado es un residuo de cointegración, no un campo gauge.
  Útil sólo como estadístico de cointegración con su nulo.
- *Hodge*: la descomposición de Helmholtz–Hodge de un flujo dirigido entre
  activos sí es matemática aplicable (gradiente vs rotacional en un grafo);
  su valor depende de que el flujo de entrada no sea un gradiente por
  construcción (R6-A1).
- *Riemann, P vs NP, Birch–Swinnerton-Dyer, Poincaré*: sin observable de
  mercado identificable. Se registran como especulativos y no se cablean.
- *Cuántica*: todo lo que hay es cálculo clásico con amplitudes complejas
  («inspiración cuántica»). No acredita ventaja cuántica; se evalúa como
  cualquier otro estimador.

## 8. Recursos y compilación en la nube (lo que funcionó hoy)

- `apt-get install gcc-mingw-w64-x86-64 wine64` funciona a través del proxy
  de la sesión (≈ 3 min). Después `rustup target add x86_64-pc-windows-gnu`.
- Variables: `RUSTFLAGS="-C target-cpu=x86-64-v2 -C opt-level=3"` (el
  contenedor puede reanudarse en otra CPU; `native` dejó binarios con
  AVX-512 FP16), `CARGO_TARGET_DIR` propio por worktree,
  `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=wine`, `WINEDEBUG=-all`.
- `os-guardian` es sólo Windows: `cargo check` nativo en Linux falla en
  `main` también; no es regresión.
- Una sola cadena Cargo pesada a la vez (4 CPU, 15 GB).
- **Disco**: la cuota de la sesión es ~30 GB. Las suites de test en debug con
  símbolos la agotan (cada ejecutable de test pesa cientos de MB; el enlazador
  falló con «No space left on device»). Compilar los tests con
  `CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0` y borrar
  `<target>/x86_64-pc-windows-gnu/debug/incremental` si hace falta.

## 9. Protocolo con los demás agentes

- Antes de editar: `git fetch`, buzón (`COORDINACION_CODEX_2026-09-28.md`)
  y TABLERO; re-grep de cada ancla.
- Stage sólo de archivos propios (nunca `git add -A`); diff contra cada
  padre tras cada merge; `cargo check --workspace --all-targets`.
- Cambios de conducta viva entran por PR con CI y, si tocan el camino del
  fixture, con T-1 (≥ 11,0 % gen a gen) adjunto. La Línea F pide lo mismo a
  las olas Ω: **toda ola que cambie el camino vivo entra por PR**, para que
  la CI y una revisión independiente corran antes del merge.
