# Contrato de conceptos de la Línea C (fase C-F1)

Este documento fija las reglas que rigen desde ahora para siete conceptos que cruzan todo el motor: unidades, identidad, desconocido, horizonte continuo, espejo, versión y construcción, y vetos. Cada apartado dice la regla, de dónde sale y dónde se cumple o se incumple hoy. Los incumplimientos remiten a las entradas LC-F1-nn de `docs/audit/HALLAZGOS_C-F1_2026-10-11.md`, que traen el test que falla hoy. No se incluye ningún hallazgo refutado.

Base: main en ae6d51a2. Origen: los conceptos de los siete revisores de la fase, corregidos por los escépticos y comprobados leyendo el código en esa base. Nada se compiló ni se ejecutó.

**Estado en `main` c4e8e285 (añadido al publicar).** Tras ae6d51a2 entraron #718, #719 y #720 (AGY, directos a `main`), y el PR #37 añade CL-58 y CL-59. Las citas de este documento siguen ancladas a ae6d51a2: en `lib.rs`, las líneas posteriores a la 1072 se desplazan 7 en c4e8e285, y las posteriores a la 2427, 65. Lo que cambia en el apartado 3 («desconocido»): #720 deja de convertir NaN y ±Inf en 0 en el registro y renueva el sello de tiempo en cada escritura (LC-F1-34 y LC-F1-35, ver la adenda de los hallazgos). La caída al global de LC-F1-06 sigue abierta.

Cómo comprobar las citas:

```
python3 scripts/plan_linea_c.py citas <archivo> --rev ae6d51a2 --alias lib.rs=crates/god-engine-core/src/lib.rs --alias god_engine.rs=src/bin/god_engine.rs
```

El script no está en ae6d51a2: viene de la rama de Línea C, y se ejecuta con `--repo` apuntando a un checkout de ae6d51a2. Comprueba que el archivo exista y que la línea esté en rango; que la línea diga lo afirmado lo comprobaron los revisores y los escépticos.

Convenciones:
- `lib.rs` sin ruta es crates/god-engine-core/src/lib.rs (el núcleo) y `god_engine.rs` sin ruta es src/bin/god_engine.rs (el host). Las demás rutas son completas desde la raíz.
- Un `:NNN` suelto se refiere al último archivo citado en la misma línea.
- «Regla» obliga a todo código nuevo y a todo arreglo desde hoy. Lo que hoy la incumple se arregla en la fase dueña del hallazgo, no de golpe.
- Las zonas ajenas no se tocan desde la Línea C: Codex (correlation_guard.rs, random_matrix.rs, latch_degradation), Qoder/GLM (Hawkes y flow_excitation_confluence.rs) y Línea E (ramas 1 a 10 del núcleo, .github/workflows/unit-suites.yml, model_manifest.rs y la escritura de modelos de train_forest). Si una regla las afecta, va como pedido a su dueño.

---

## 1. Unidades

**Regla.**
1. Toda magnitud que cruza una frontera (campo de QuantumConfig, CoinArena o GlobalArena, gen, clave del registro, argumento de una función pública o variable de entorno) declara en su doc una unidad de este vocabulario cerrado: fracción de precio (0,001 = 10 pb); puntos básicos, sólo para mostrar o registrar y nunca como entrada de una fórmula; USD, diciendo si es nocional, margen o capital; fracción de capital; unidades de precio; cantidad base; milisegundos; eventos; probabilidad; z-score; razón adimensional con su numerador y su denominador; nivel de confianza 1 − α.
2. Toda dispersión declara su tipo estadístico y su horizonte. El ATR de 1 minuto es un rango medio en unidades de precio, y su cociente con el precio es una fracción de precio. σ es una desviación típica por paso τ. La conversión entre rango y σ es el factor √(8/π) de crates/risk-engine/src/tp_sl.rs:138-142. Una misma función no puede tratar el mismo argumento como rango en un sitio y como σ en otro sin declararlo.
3. Cada gen declara su unidad y la variable concreta del consumidor con la que se compara. Su banda, su baseline y su rango aleatorio caen dentro de lo que esa variable puede valer. Un gen tiene una sola lectura en todos sus consumidores. No se derivan milisegundos, USD, fracciones de salto ni probabilidades a partir de la comisión adimensional.
4. Lo desconocido se modela con Option o con un único centinela documentado por campo. Nunca con 0 USD, con 0 de una magnitud medible ni con el gen reinterpretado en otra unidad. El apartado 3 da el detalle.
5. La memoria de toda EWMA se expresa en milisegundos físicos. Si alguna se mide en eventos, lo declara y lo justifica.
6. Las claves del registro y los nombres nuevos llevan sufijo de unidad: _ms, _usd, _frac, _bps, _qty, _z, _prob o _ratio. El sufijo _pct queda reservado, porque hoy significa fracción en varios sitios.
7. Toda conversión entre unidades vive en una función con nombre, con su pendiente documentada y un test de punto fijo.
8. El contrato mantiene una tabla de los campos de CoinArena y GlobalArena con unidad, escritor y lector. Una guarda de fuentes rechaza cualquier campo nuevo sin unidad declarada.

**Origen.** Concepto del revisor de unidades; D-754c para el rango del ATR (crates/risk-engine/src/tp_sl.rs:138-142 y :345-354); #660 para la EWMA en tiempo físico (crates/quantum-arena/src/state.rs:730-772); D-745 para el centinela de la τ dominante (crates/quantum-arena/src/state.rs:203-209).

**Se cumple hoy.**
- dominant_tau_ms declara milisegundos y su centinela «0 = el espectro aún no ha arrancado» (crates/quantum-arena/src/state.rs:203-209).
- riesgo_por_operacion declara fracción del capital y su centinela «0 = todavía no se ha dimensionado ninguna orden» (crates/quantum-arena/src/state.rs:515-520).
- El decaimiento del CVD y del OBI agregados vive en tiempo físico, con τ = 60 s (crates/quantum-arena/src/state.rs:730-772).
- La conversión de ATR a σ del núcleo es una función con nombre (crates/god-engine-core/src/diffusion.rs:57-63).

**Se incumple hoy.**
- live_maker_fee y live_taker_fee guardan una fracción con un comentario que dice «%» (crates/quantum-arena/src/config.rs:207-208).
- get_atr_pct se documenta como porcentaje y devuelve una fracción (crates/god-engine-core/src/stateful_engine.rs:1176-1182).
- Hallazgos: LC-F1-20 (cvd_veto_threshold con banda fuera del ratio), LC-F1-21 (dynamic_atr_min inerte), LC-F1-22 (pendiente del sesgo spot), LC-F1-30 (drawdown con doble lectura del gen), LC-F1-31 (EWMA en eventos), LC-F1-41 (R-múltiplos del daemon), LC-F1-42 (new_baseline fuera de banda), LC-F1-47 (capital 0 USD), LC-F1-48 (anclas del OI), LC-F1-53 (doc de la intermitencia), LC-F1-55 (ATR como rango y como σ) y LC-F1-56 (muros L2).

**Tabla inicial de campos.** Se completa en C-F1; estas filas ya están comprobadas.

| Campo | Unidad | Cita | Estado |
|---|---|---|---|
| current_atr | unidades de precio (rango medio de 1 min) | crates/quantum-arena/src/state.rs:196 | el doc no declara la unidad |
| sigma_forecast | fracción de precio | crates/quantum-arena/src/state.rs:197-202 | cumple |
| dominant_tau_ms | ms; 0 = desconocida | crates/quantum-arena/src/state.rs:203-209 | cumple |
| riesgo_por_operacion | fracción de capital; 0 = sin medir | crates/quantum-arena/src/state.rs:515-520 | cumple |
| l2_bid_wall, l2_ask_wall | cantidad base del mejor nivel, según los escritores | crates/quantum-arena/src/state.rs:277-279 | el doc dice volumen acumulado (LC-F1-56) |
| open_interest_norm | log-normalizado con anclas en USD | crates/quantum-arena/src/state.rs:237-241 | el código ancla en otra cifra (LC-F1-48) |
| spectral_intermittency | adimensional en [0, 1] | crates/quantum-arena/src/state.rs:231-232 | doc con la fórmula retirada (LC-F1-53) |
| last_ws_latency_ms | ms; nace en 0 | crates/quantum-arena/src/state.rs:523 | 0 no se distingue de «sin medir» (LC-F1-62) |
| live_maker_fee, live_taker_fee | fracción | crates/quantum-arena/src/config.rs:207-208 | el comentario dice «%» |

---

## 2. Identidad

**Regla.**
1. La identidad de un activo es su símbolo canónico, en MAYÚSCULAS ASCII, obtenido con una única función, simbolo_canonico (crates/quantum-arena/src/symbols.rs:10-12). Se aplica en todas las fronteras: universo, registro de specs, búsquedas, claves {SYM}_* del registro, nombres de modelos, peticiones y confirmaciones REST y archivos persistidos. Un universo o un registro con símbolos repetidos es inválido.
2. coin_id es sólo un índice de enrutamiento local al proceso, inmutable mientras vive el proceso. No tiene semántica: queda prohibido decidir por posición (coin_id == 0, coin_id == 1, coins[0]).
3. Los papeles especiales (líderes BTC y ETH del lead-lag, fuente del régimen global, genes con sufijo _btc) se definen por igualdad exacta con constantes de símbolo canónico, nunca por prefijo ni por posición. Si el símbolo de un papel no está en el universo, la magnitud es desconocida: no se publica, no vota y no se calcula con otra moneda. Cada gen _btc declara si se aplica sólo a BTCUSDT o a todas las monedas.
4. Desconocido no es un valor. Un slot sin símbolo, un símbolo sin spec medido o un símbolo ilegible no se sustituyen por "BTCUSDT", por "" ni por el slot 0: el consumidor se abstiene con un código de rechazo. Un spec lleva procedencia (medido de exchangeInfo o declarado), y sólo uno medido autoriza a dimensionar. El refresco desde exchangeInfo no depende de que cambie la propuesta de universo.
5. Hay una sola resolución de identidad por evento: el núcleo resuelve símbolo y spec una vez por tick y los reutiliza. try_symbol, try_spec, try_index y get_coin_id son el mismo mapa, también para un slot sin identidad, en el que los cuatro devuelven None.
6. Los datos por moneda sólo se leen por moneda (detalle en el apartado 3).
7. Paridad entre el vivo y los medidores: el forense, el entrenador, el examen walk-forward del demonio y el backtest reproducen un símbolo en un slot cuya identidad resuelve a ese símbolo, y toda decisión específica de un activo da el mismo resultado para el mismo símbolo en cualquier slot.
8. Ciclo de vida del universo: la propuesta de rotación se persiste en una sola ruta, la del estado (apartado 6), y el arranque la lee. Si el universo va a ser la lista literal del bootloader, hay que enmendar ADR-0011.
9. Identidad de cantidades y órdenes: toda comisión lleva su activo, y un ack sin orderId, clientOrderId o symbol, o que pertenece a otra orden, es una respuesta ambigua, no una conocida.
10. El mapeo entre símbolos de futuros y de spot, y los prefijos multiplicadores (1000…), se declaran de forma explícita.

**Origen.** ADR-0011, decisiones 1 y 2 y su consecuencia «try_symbol, try_spec y try_index son el mismo mapa» (docs/adr/ADR-0011-identidad-canonica-del-simbolo.md:21-35); D-725, un solo espacio de índice (crates/quantum-arena/src/symbol_registry.rs:106-119); concepto del revisor de identidad.

**Se cumple hoy.**
- El universo se canoniza al publicarse (crates/quantum-arena/src/symbols.rs:19-22), y el contrato de identidad pasa con los productores reales (crates/quantum-arena/tests/identidad_simbolo_contract.rs:14-34).
- MOD2/7-005 lee el peso hebbiano del núcleo por símbolo (lib.rs:4563-4575).
- merge_universe_with_hysteresis canoniza y evita duplicados (src/symbol_manager.rs:258-313).

**Se incumple hoy.** LC-F1-02 (specs inventados presentados como oficiales), LC-F1-03 (el slot 0 recibe el gen del stop de BTC), LC-F1-06 (caída al global, causa raíz), LC-F1-09 (la propuesta de universo nunca se aplica), LC-F1-11 (régimen global del slot 0), LC-F1-12 (examen walk-forward), LC-F1-13 (backtest nativo), LC-F1-32 (slot sin identidad pasa a BTCUSDT), LC-F1-33 (ack sin identidad), LC-F1-45 (líderes por prefijo), LC-F1-49 (lista negra del escáner), LC-F1-57 (registro sin canonizar), LC-F1-58 (votantes del slot 0), LC-F1-59 (censo del slot 0), LC-F1-60 (comisiones en activos distintos) y LC-F1-61 (suposiciones inertes de posición).

---

## 3. Desconocido

**Regla.**
1. Una lectura del registro tiene tres partes: valor, estado y sello. El estado es Medido, Desconocido o Inválido. El sello es el tiempo del evento que produjo el dato (reloj del exchange, en ms), que pasa el escritor; no el reloj de pared, que rompe la paridad entre backtest y vivo.
2. Una escritura no finita no se convierte en 0: queda como Desconocido o Inválido, o se rechaza y se cuenta. La lectura devuelve Option o estado, nunca un número inventado.
3. El ámbito es propiedad de la clave y se declara una sola vez: global de verdad (genes, configuración, observables del universo, estado del host), por moneda (mediciones de un activo) o por par o grupo. Una clave por moneda nunca cae al global: su ausencia es Desconocido. La cadena símbolo, moneda y global sólo vale para las claves declaradas «global con anulación por moneda», que son los genes. Un publicador por moneda no escribe la clave global.
4. Frescura: cada lectura en un camino de decisión declara su edad máxima, en unidades de su τ o de la cadencia del publicador, y pasada esa edad es Desconocido. Un publicador que en un ciclo no puede medir publica Desconocido de forma explícita, en vez de salir sin escribir.
5. Propagación: un voto con alguna entrada desconocida se abstiene y no suma peso activo al consenso. Una guarda o un veto con una entrada desconocida falla cerrado. Un aprendiz con features desconocidas o de otra moneda descarta la muestra. Un valor neutro (0,5, 1,0 o 0) sólo sustituye a lo desconocido si el contrato del lector lo declara, y nunca dentro del registro.
6. La salud del feed tiene tres estados por símbolo: Desconocido hasta el primer evento vivo; Vivo, con la edad del último evento; y Estancado, por timeout, error, cierre o fallo de reconexión. «Sano» exige evidencia. Los replays y los tests declaran el feed vivo de forma explícita.
7. Enmienda necesaria: ADR-0014, principio 4, manda que todo publicador condicionado escriba 0.0 en su rama else (docs/adr/ADR-0014-doctrina-continuo-espectral.md:30-33). Su fin (que el lector no consuma un valor viejo) se conserva, pero el medio cambia: el else publica Desconocido, no 0.

**Origen.** R6-B15, que diseña una t½ infinita para que la guarda falle cerrada (crates/strategy-core/src/vecm_arbitrage.rs:287-293); MOD2/7-005, que ya separó un lector por símbolo (lib.rs:4563-4575); D-742, «un espectro sin observar no proyecta» (crates/signal-engine/src/proyeccion_espectral.rs:17-19); concepto del revisor de desconocido.

**Se cumple hoy.**
- FlowExcitationConfluenceEngine se abstiene si no hay base medida (CL-15, crates/signal-engine/src/flow_excitation_confluence.rs:201-203). Es zona de Qoder/GLM; su lectura con caída al global se coordina con LC-F1-06.
- El campo dominant_tau_ms usa un centinela documentado (crates/quantum-arena/src/state.rs:203-209).

**Se incumple hoy.**
- El registro guarda 0 para todo valor no finito (crates/omniscient-registry/src/lib.rs:53-57) y la cadena de ámbito cae al global (crates/omniscient-registry/src/lib.rs:261-279).
- Hallazgos: LC-F1-04 (peso hebbiano de otra moneda), LC-F1-06 (causa raíz del ámbito), LC-F1-07 (StatArb de otra moneda), LC-F1-08 (bosque sombra con features globales), LC-F1-32 (slot sin identidad), LC-F1-34 (±Inf y NaN pasan a 0), LC-F1-35 (sello que no se renueva), LC-F1-44 (sombra de resonancia), LC-F1-46 (proyección espectral), LC-F1-47 (capital 0 USD), LC-F1-48 (OI de precio desconocido) y LC-F1-62 (salud del feed).

---

## 4. Horizonte continuo

**Regla.**
- (a) Horizonte de la posición. τ_pos es ValidatedOrder.tau_ms, que el núcleo guarda en Position::entry_tau_ms (crates/risk-engine/src/lib.rs:41-48). Es inmutable durante la vida de la posición. Toda decisión sobre una intención o una posición (puertas, veto espectral, geometría TP/SL, break-even, trailing, caducidades, Kelly, apalancamiento del host y enfriamiento) es función de su τ. El estado del mercado se lee en escalas definidas en relación con esa τ, nunca en centros absolutos.
- (b) τ desconocida. «Sin τ» es un estado explícito (Option o un centinela con nombre), no un 0 tratado como número. Tiene una sola política, compartida por el host, el núcleo y el replay, y fija durante la vida de la posición. Ninguna etiqueta convierte τ = 0 en una banda.
- (c) τ dominante. Hay una sola definición publicada, y el nodo crudo de la malla no se usa como horizonte de una decisión sin interpolar. operating_tau_ms deja de llamarse «fuente única» o pasa a recibir la τ medida.
- (d) Continuo, no regímenes. Ninguna decisión salta al cruzar un umbral fijo de τ, de capital o de una probabilidad de régimen, salvo que el contrato la declare veto duro con su derivación y su test. Las clasificaciones en bandas son sólo telemetría, con una partición canónica. Las saturaciones de ventanas temporales se declaran como política con su derivación.
- (e) Kelly. La fracción que dimensiona una orden se evalúa en la τ de esa orden en el momento de evaluarla. Lo que se guarda en el cierre es evidencia (W, PF y n), no una fracción ligada a otra τ. La transición entre supervivencia y expansión es continua en el capital.
- (f) Nombres heredados. Los genes scalp_* y swing_* se documentan como «valor en el ancla de 30 s o de 12 h»; los inertes se retiran o se marcan como tales. Las API con bandas y sin lector de producción se retiran o se documentan como compatibilidad que delega en la superficie continua, para que ningún merge vuelva a cablearlas (precedente: CL-22).

**Origen.** D-745, una posición y un horizonte (crates/risk-engine/src/lib.rs:41-48); C-2, horizonte unificado del tick (crates/quantum-arena/src/temporal_spectrum.rs:1192-1208); D-641, régimen de capital continuo (crates/risk-engine/src/capital_regime.rs:1-20); concepto del revisor de horizonte.

**Se cumple hoy.**
- El régimen de capital es continuo (crates/risk-engine/src/capital_regime.rs:1-20).
- El riesgo resuelve el horizonte de una intención sin duración con la τ medida y sólo cae al gen en frío (crates/risk-engine/src/lib.rs:1482-1490).

**Se incumple hoy.**
- Hay tres políticas para la τ desconocida: el host recorta τ = 0 al ancla de 30 s (god_engine.rs:68), el núcleo usa la τ dominante viva (lib.rs:3229-3242) y el replay usa 30 s (crates/backtest-engine/src/booktick_replay.rs:753-756).
- El doc de PositionHorizon dice que entry_tau_ms es «la τ dominante del espectro temporal en el instante de la entrada» (crates/quantum-arena/src/position.rs:14-16), mientras D-745 la define como la τ con la que se dimensionó la orden (crates/risk-engine/src/lib.rs:41-48). Hay que alinear el doc.
- Hallazgos: LC-F1-05 (X-016 con tres bandas fijas), LC-F1-14 (dos «τ dominante»), LC-F1-15 (posición adoptada sin τ), LC-F1-16 (salto de Kelly), LC-F1-17 (apalancamiento de arranque), LC-F1-36 (Kelly del cierre anterior), LC-F1-63 (bandas Micro, Meso y Macro), LC-F1-64 (alias del orquestador) y LC-F1-65 (fuente única del horizonte).

---

## 5. Espejo

**Regla.**
1. Qué es p. El bosque _MOTOR estima p = P(TP del largo antes que su SL | x), con la geometría de su etiqueta (TP 0,36 %, SL 0,18 % y timeouts descartados, src/bin/train_forest.rs:1391-1392 y :1689-1708), y su base b. Es una probabilidad del largo.
2. Qué es la probabilidad del corto. Con la misma geometría, el corto gana si el precio cae TP antes de subir SL. Con TP ≠ SL no es 1 − p, que es P(SL del largo primero). Bajo el modelo de primer toque del repositorio (crates/risk-engine/src/tp_sl.rs:462-489), P_corto(μ) = P_largo(−μ). El contrato elige una de dos vías y la escribe: (a) etiquetar y validar el evento corto con su propia barrera, o (b) declarar el modelo de trayectoria y derivar el espejo con la geometría TP/SL como entrada. Hasta entonces ningún documento llama «calibrada» a la p del corto ni promete «idéntico rigor».
3. SignalIntent::win_probability significa P(ganar neto | lado y geometría de esta orden). «Desconocido» es un estado distinguible (Option, o NaN rechazado de forma explícita), con una sola lectura en todos sus consumidores. El doc del tipo dice si la selección lo lee.
4. Evidencia por lado. Mezclar largos y cortos en un mismo calibrador o registro de acierto es válido sólo si se documenta que los lados son intercambiables; separarlos exige medir antes. Cada registro de acierto se compara con la tasa base de su propia etiqueta, no con ½.
5. Cada clave publicada declara su magnitud y su lado. Una P(la operación gana) no se lee como P(sube) ni se complementa para el otro lado.
6. Regla de espejo para puertas, umbrales y opiniones: G(Largo, x) = G(Corto, M(x)), donde M refleja el precio (OBI, CVD, marea, deriva, stretch y z cambian de signo, y p_largo pasa a p_corto según el punto 2). Cada puerta lleva su test espejo, y toda transformación de p en opinión o en umbral usa la misma normalización. Las exenciones unilaterales se declaran como política del dueño, con su justificación y su contrato.
7. Igualdad de rigor: la misma probabilidad mínima de ganar en el umbral de cada lado, medida con el modelo del punto 2. No basta con la misma divergencia normalizada.

**Origen.** K-26 y la normalización por semi-intervalos (crates/god-engine-core/src/calibration.rs:321-322); CERT-M2-C04, que quiere eliminar el sesgo estructural contra el corto (crates/metacortex-engine/src/consejo_seniors.rs:382-384); concepto del revisor de espejo.

**Se cumple hoy.**
- El veto CVD es espejo exacto entre los dos lados (lib.rs:1663-1667).
- La rama 13 escala los umbrales del bosque de forma simétrica sobre los semi-intervalos [0, base] y [base, 1] (lib.rs:6392-6396).
- El respaldo discreto del orquestador que sólo protege a los largos es una exención declarada (crates/risk-engine/src/orchestrator.rs:191-198) y fijada por un contrato de Codex (crates/risk-engine/tests/nonfinite_admission_contract.rs:175-190). Cumple el punto 6.

**Sin verificar en esta fase.** Los valores reales de ml_model_base en los modelos promovidos y la validez empírica del modelo browniano en los tapes. Los tests de los hallazgos usan ese modelo como oráculo declarado.

**Se incumple hoy.** LC-F1-18 (el freno del bosque lee P(ganar) como P(subida)), LC-F1-19 (rama 20 con el gen del largo), LC-F1-37 (contrato de win_probability), LC-F1-38 (puerta S-6), LC-F1-39 («el corto consume 1−p»), LC-F1-43 (p_win_directional), LC-F1-66 (opiniones ML del consejo), LC-F1-67 (umbral corto del bosque sombra) y LC-F1-68 (deriva del auditor sombra).

---

## 6. Versión y construcción

**Regla.**
1. Binario certificado. Un solo triple de producción, el del operador, x86_64-pc-windows-msvc (rustc-ice-2026-09-07T14_42_23-32692.txt:40), y una línea base de CPU explícita (x86-64-v2 o v3) versionada en .cargo/config.toml, nunca target-cpu=native. Los mismos rustflags para la CI, la nube, el T-1 y los lanzadores; ningún lanzador redefine RUSTFLAGS. Toda medición que certifique algo (T-1, goldens, forense, walk-forward) imprime la versión de rustc, el triple, los rustflags y los hashes del árbol git y de Cargo.lock. Una medición en el triple gnu bajo wine no certifica el binario msvc mientras no se demuestre que las huellas son iguales bit a bit en ambos triples.
2. Perfiles. El release usa panic = "abort" (Cargo.toml:185): ningún código de producción cuenta con el desenrollado de un pánico. overflow-checks y debug-assertions se declaran de forma explícita en el perfil release. Ningún contrato vive sólo en un debug_assert, y la aritmética entera que deba envolver usa wrapping_*.
3. Ediciones y targets. La edición se declara por crate. Ningún fuente se compila en dos paquetes, y los nombres de binario son únicos en el workspace.
4. Dependencias. Cargo.lock es la autoridad (--locked en toda la CI), sin dependencias git. La reproducibilidad sembrada (StdRng en crates/quantum-arena/src/genome.rs:1734 y en src/bin/train_forest.rs:1921) queda atada a la versión de rand del lock, y cambiarla obliga a recertificar. Las features no cambian la conducta según qué crates entren en la compilación: todo reqwest 0.11 lleva default-features = false y el backend TLS explícito. Todo artefacto se valida tras releerlo con el mismo parser.
5. Entorno. Tres conceptos ortogonales, cada uno leído por una sola función de la Línea C con un único parser booleano: (a) el venue, testnet o mainnet, cuya ausencia es un error o tiene el mismo valor por defecto en todos los binarios, y en el que "1" nunca significa mainnet; (b) el modo de operación, papel o real, que deciden --force-live, MAINNET_ARMED y STOP_TRADING.LOCK (god_engine.rs:857-879); y (c) la carpeta del estado aprendido, demo o prod, que se deriva del modo y no del venue: el papel nunca escribe en prod. Todas las rutas de estado cuelgan de data_root(), que honra TGM_DATA_DIR, y live_fees.json también va por entorno.
6. Secretos y plantilla. Un único proveedor de credenciales con nombres canónicos y precedencia documentada. La clave y el secreto salen del mismo par de variables, se recortan los espacios, y los rellenos y los vacíos cuentan como ausentes. template.env enumera exactamente las variables que se leen, declara la gramática de dotenvy y no contiene centinelas que los lectores conviertan en números: un capital ausente o ≤ 0 es un error, no 0, 5 o 13 según el binario. Lanzadores y herramientas leen el .env con el mismo parser.

**Origen.** D-663, que declara que target-cpu=native rompe la reproducibilidad numérica (INFORME_FORENSE_MAESTRO.md:5136); CL-48, separación de la carpeta del estado por entorno (crates/quantum-arena/src/paths.rs:27-32); D-07, raíz de datos única (src/env_manager.rs:9-18); concepto del revisor de construcción.

**Se cumple hoy.**
- Cargo.lock no tiene dependencias git.
- genome_store valida el artefacto tras releerlo (crates/quantum-arena/src/genome_store.rs:654-656).
- Los timestamps revisados usan i64 o checked_add (crates/backtest-engine/src/label_evidence.rs:170-173).

**Se incumple hoy.**
- target-cpu=native en .cargo/config.toml:2 (LC-F1-29).
- El predicado de la carpeta del estado mira variables de venue (crates/quantum-arena/src/paths.rs:21-25) (LC-F1-01).
- Hallazgos: LC-F1-01 (el papel escribe en data/prod), LC-F1-10 (el gestor de símbolos usa is_demo_env como venue), LC-F1-29 (target-cpu=native), LC-F1-40 (venue por defecto distinto por binario), LC-F1-47 (INITIAL_CAPITAL), LC-F1-52 (template.env), LC-F1-54 (TGM_DATA_DIR y live_fees.json), LC-F1-71 (binario doble), LC-F1-72 (credenciales) y LC-F1-73 (TLS de reqwest).

---

## 7. Vetos

**Regla.**
1. Identidad. Cada punto donde el sistema se niega a actuar es un veto con id estable y con un código de máquina. En risk-engine el código es el nombre de la constante REJ_* (crates/risk-engine/src/lib.rs:94-115); en el envío, la variante de VetoEnvio (crates/risk-engine/src/envio.rs:34-42); las puertas del núcleo y del host tienen un id de puerta. EntradaVeto lleva un campo `codigo`. La relación entre las ranuras REJ_* que devuelve rej() y las entradas activas es biyectiva, y una ranura definida que nunca se devuelve sólo existe como retirada, con su linaje.
2. Clase. RiesgoDuro queda definido como en crates/risk-engine/src/veto_registry.rs:20-21 (ruina, drawdown, margen y kill-switch). Un dato desconocido (NaN o fuera de dominio) va a REJ_INVALID_INPUT, no a una ranura de decisión.
3. Umbral. La ficha guarda la expresión o el valor del umbral, no sólo su fuente. Un umbral de gen nombra el campo exacto del genoma; uno medido nombra su estimador; uno literal da su valor y su porqué. La ficha guarda también la ubicación del veto. Un test de fuentes verifica que los genes citados existen.
4. Dueño y fecha. Son dos campos separados. La fecha es ISO, obligatoria, y corresponde al último cambio de semántica.
5. Test. El test pineado observa el rechazo: ultimo_rechazo() == Some(REJ_X), o el contador de esa ranura, en el caso positivo, y admisión u otro código en el negativo. Un test de diagnóstico sin aserciones, un #[ignore] o un test de una función auxiliar no certifican un veto. Para riesgo duro activo es obligatorio.
6. Deriva. La regla de mismo commit de ADR-0004 se hace cumplir con el test de fuentes siguiente, no con disciplina.
7. Alcance. El registro declara qué familias cubre y cuáles excluye, con su motivo. Las tablas de rechazo de zonas ajenas (Codex: correlation_guard y latch_degradation; Qoder/GLM: Hawkes y flow_excitation; Línea E: ramas 1 a 10) se registran con su dueño como responsable; la Línea C no las reescribe.
8. Uso en ejecución (recomendado, no obligatorio). Que reject_report y el libro contrafactual muestren el id del registro junto al nombre del REJ. Hoy el registro es una tabla de auditoría estática por diseño (ADR-0004), y eso no es un defecto.
9. El censo «medido ≥ literal» (crates/risk-engine/src/veto_registry.rs:645-656) es una política, no una verdad. Corregir una ficha para que diga lo que hace el código nunca queda bloqueado por ese recuento.

**Test de fuentes que fija la regla.** `rej_registro_biyectivo_con_el_codigo`, en crates/risk-engine/tests/veto_registry_source_contract.rs.
- Entrada: include_str!("../src/lib.rs") cortado antes de #[cfg(test)]. El conjunto D sale de las constantes `pub const REJ_*: usize`, y el conjunto R de las llamadas `rej(REJ_*)`.
- Aserciones:
  - (a) Cada código de D es el `codigo` de al menos una entrada, y todas sus entradas comparten clase.
  - (b) Cada `codigo` de una entrada pertenece a D, a las variantes de VetoEnvio o a una lista declarada de puertas del núcleo.
  - (c) Una entrada activa con código c exige que c esté en R, y un c de D que no está en R exige que todas sus entradas estén retiradas.
  - (d) El cuerpo de la función del test pineado contiene c.
- Hoy: rojo. Falla (a) porque el campo no existe. Con el emparejamiento por nombre, (c) falla por LC-F1-23 y (d) por LC-F1-25.
- Tras el arreglo: verde.

**Origen.** ADR-0004: el registro como código, riesgo duro activo con test y mantenimiento en el mismo commit (docs/adr/ADR-0004-veto-registry-como-codigo.md:12-13, :19 y :20-24); REV-3, que pide que cada entrada declare su REJ_* y que el test compruebe ultimo_rechazo() (docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:248); concepto del revisor de vetos.

**Se cumple hoy.**
- REV-1 y REV-2 se cerraron con aserciones exactas sobre ultimo_rechazo() (docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:246-247), visibles en crates/risk-engine/tests/veto_logic_contracts.rs:125 y :197.
- El registro declara su alcance inicial: las compuertas nombradas del risk-engine y los vetos estructurales medidos (crates/risk-engine/src/veto_registry.rs:60-63).

**Se incumple hoy.** LC-F1-23 (tres REJ muertos dados por activos), LC-F1-24 (REJ_VIABILIDAD sin entrada propia), LC-F1-25 (los tests no observan su REJ), LC-F1-26 (V-RISK-001 describe otro rechazo), LC-F1-27 (V-RISK-005 con el umbral 0,85), LC-F1-28 (genes inexistentes citados), LC-F1-50 (V-LOGIC-005 «no absorbente»), LC-F1-51 (dientes nominales), LC-F1-69 (fichas sin fecha) y LC-F1-70 (FUENTES_CRUZADAS_CONTRATOS).
