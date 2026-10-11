# Hallazgos de la fase C-F1 (Línea C)

Base: main en ae6d51a2. Origen: siete revisores, uno por concepto (unidades, identidad, desconocido, horizonte continuo, espejo, vetos, versión y construcción), y dos escépticos por hallazgo. Aquí sólo figuran los hallazgos **confirmados** (ningún escéptico los refuta) y los **disputados** (un escéptico los refuta y el otro no). En cada disputado se dice qué objetó el escéptico. Los refutados van al final, uno por línea. Se aplican las correcciones de los escépticos: citas desplazadas, cifras, alcance, tests y severidad.

Convenciones:
- `lib.rs` sin ruta es crates/god-engine-core/src/lib.rs (el núcleo) y `god_engine.rs` sin ruta es src/bin/god_engine.rs (el host). Las demás rutas son completas desde la raíz.
- Un `:NNN` suelto se refiere al último archivo citado en la misma línea.
- Las citas se comprueban con `python3 scripts/plan_linea_c.py citas <archivo> --rev ae6d51a2 --alias lib.rs=crates/god-engine-core/src/lib.rs --alias god_engine.rs=src/bin/god_engine.rs`. El script no está en ae6d51a2: viene de la rama de Línea C y se ejecuta con `--repo` apuntando a un checkout de ae6d51a2. El script comprueba que el archivo y la línea existan; que la línea diga lo afirmado lo comprobaron los revisores y los escépticos leyendo el archivo.
- Fase dueña: C-F1 si el arreglo es del concepto (contrato, registro, ficha, doc de un tipo); si no, la fase del código según el plan: C-F2 calibración y espectro, C-F3 decisión viva, C-F4 dinero, C-F5 ejecución, C-F6 datos, C-F7 aprendizaje, C-F9 herramientas.
- Cada test «falla hoy» por lectura del código en ae6d51a2. Ninguno se ejecutó en esta fase (no se compiló).

**Adenda: estado en `main` c4e8e285 (añadida al publicar, 2026-10-11).** Este documento está anclado a ae6d51a2. Después entraron #718, #719 y #720 (AGY, directos a `main`), y el PR #37 de Línea C (CL-58, CL-59). En `lib.rs`, las líneas posteriores a la 1072 se desplazan 7 en c4e8e285, y las posteriores a la 2427, 65. Cambia el estado de estos hallazgos:
- **LC-F1-34: cerrado en `main` por #720**, comprobado leyendo c4e8e285. El registro ya no convierte NaN ni ±Inf en 0, y los lectores con respaldo saltan el valor no finito. La t½ = +Inf de StatArb llega cruda y el veto `t½ ≤ 2τ*` vuelve a saltar. Falta el test (1) de StatArb: #720 sólo añadió los tests del propio registro.
- **LC-F1-35: cerrado en parte por #720.** `set_value` renueva el sello en cada escritura, pero con el reloj de pared, que no sirve en replay. Ningún lector lo consulta todavía. El pedido (2) a Qoder/GLM sobre el publicador de contagio sigue abierto. Además, lee el reloj en cada escritura, y `process_event` hace entre unas 135 y 280 escrituras por evento (sin medir; puerta de rendimiento del plan, §7, paso 5).
- **LC-F1-06: sigue abierto.** #720 añadió lecturas estrictas sin caída al global (`get_for_coin_strict`, `get_scoped_strict` y sus variantes `_or`), pero ningún lector las usa todavía. Ahora también un NaN de la moneda cae al global, que es el de la última moneda que escribió. Siguen abiertos con él LC-F1-04, LC-F1-07, LC-F1-08, LC-F1-44, LC-F1-46 y LC-F1-58.
- Relacionado, fuera de esta fase: CL-58 (`plan_linea_c.py claves`) lista 25 claves del registro leídas sin escritor sobre c4e8e285 (`docs/audit/CLAVES_SIN_ESCRITOR_2026-10-11.tsv`). CL-59 corrige la reinyección del dominante espectral que #718 introdujo en Feynman, contra #652/H2.

Recuento: 73 hallazgos, 42 confirmados y 31 disputados. Por severidad: 4 altos, 36 medios (25 confirmados y 11 disputados) y 33 bajos (13 confirmados y 20 disputados).

| LC-F1 | Hallazgo | Severidad | Estado | Fase dueña |
|---|---|---|---|---|
| 01 | El papel sin banderas escribe el estado aprendido en data/prod | alta | confirmado | C-F1 |
| 02 | Specs inventados servidos como oficiales y casi nunca refrescados | alta | confirmado | C-F1 y C-F6 |
| 03 | El slot 0 recibe el gen del stop de BTC sea cual sea su símbolo | alta | confirmado | C-F4 y C-F3 |
| 04 | El perceptrón vota con el peso hebbiano de otra moneda | alta | confirmado | C-F3 |
| 05 | X-016 veta con tres bandas fijas y escalones | media | confirmado | C-F3 |
| 06 | Las lecturas por moneda caen al global (causa raíz) | media | confirmado | C-F1 |
| 07 | StatArb hereda la z y la vida media de otra moneda | media | confirmado | C-F3 |
| 08 | El bosque sombra aprende con features globales | media | confirmado | C-F7 |
| 09 | La propuesta de universo nunca se aplica al reiniciar | media | confirmado | C-F6 |
| 10 | El gestor de símbolos usa is_demo_env como venue | media | confirmado | C-F6 |
| 11 | El régimen global sale del slot 0 | media | confirmado | C-F3 |
| 12 | El examen walk-forward cruza identidades | media | confirmado | C-F7 |
| 13 | El backtest nativo cae al slot 0 de otro símbolo | media | confirmado | C-F7 |
| 14 | Dos «τ dominante» con el mismo nombre | media | confirmado | C-F3 |
| 15 | Una posición adoptada sin τ vive con dos horizontes | media | confirmado | C-F5 y C-F3 |
| 16 | Kelly salta en el umbral de supervivencia | media | confirmado | C-F4 |
| 17 | Apalancamiento de arranque con el margen de otra ranura | media | confirmado | C-F5 |
| 18 | El freno del bosque lee P(ganar) como P(subida) | media | confirmado | C-F3 y C-F7 |
| 19 | Rama 20: el corto usa el gen del largo | media | confirmado | C-F3 |
| 20 | cvd_veto_threshold con banda fuera del ratio | media | confirmado | C-F3 |
| 21 | dynamic_atr_min inerte al nacer (V-010) | media | confirmado | C-F3 y C-F7 |
| 22 | Sesgo spot 20 veces más empinado que su contrato | media | confirmado | C-F3 |
| 23 | Tres REJ muertos dados por activos | media | confirmado | C-F1 |
| 24 | REJ_VIABILIDAD sin entrada propia | media | confirmado | C-F1 |
| 25 | Los tests de vetos no observan su REJ (REV-3) | media | confirmado | C-F1 |
| 26 | V-RISK-001 describe otro rechazo | media | confirmado | C-F1 |
| 27 | V-RISK-005 conserva el umbral 0,85 | media | confirmado | C-F1 |
| 28 | V-LOGIC-013 y V-LOGIC-014 citan genes que no existen | media | confirmado | C-F1 |
| 29 | target-cpu=native en la configuración del repo (D-663) | media | confirmado | C-F1 |
| 30 | Drawdown con doble lectura del gen | media | disputado | C-F4 |
| 31 | EWMA de CVD y campo espectral con memoria en eventos | media | disputado | C-F3 |
| 32 | Un slot sin identidad pasa a BTCUSDT | media | disputado | C-F3 |
| 33 | Un ack 2xx sin identidad se acepta | media | disputado | C-F1 y C-F5 |
| 34 | El registro convierte ±Inf y NaN en 0 | media | disputado | C-F1 |
| 35 | El sello de tiempo del registro no se renueva | media | disputado | C-F1 |
| 36 | El Kelly de una orden sale del cierre anterior | media | disputado | C-F4 |
| 37 | win_probability contradice a sus lectores | media | disputado | C-F1 |
| 38 | La puerta S-6 no exige la misma P(ganar) a los dos lados | media | disputado | C-F3 |
| 39 | train_forest documenta «el corto consume 1−p» | media | disputado | C-F7 |
| 40 | El venue por defecto cambia según el binario | media | disputado | C-F1 |
| 41 | El daemon calcula R-múltiplos con una unidad equivocada | baja | confirmado | C-F7 |
| 42 | new_baseline nace fuera de sus bandas con la comisión viva | baja | confirmado | C-F7 |
| 43 | p_win_directional no es P(ganar) del corto con barreras asimétricas | baja | confirmado | C-F2 |
| 44 | La sombra de resonancia usa la varianza de otra moneda | baja | confirmado | C-F3 |
| 45 | Líderes BTC y ETH reconocidos por prefijo | baja | confirmado | C-F3 |
| 46 | La proyección espectral vota con el espectro de otra moneda | baja | confirmado | C-F3 |
| 47 | INITIAL_CAPITAL=0.0 se lee como 0 USD | baja | confirmado | C-F1 y C-F9 |
| 48 | El OI normalizado ancla en $100k y no en $100M | baja | confirmado | C-F5 |
| 49 | La lista negra del escáner también actúa en mainnet | baja | confirmado | C-F6 |
| 50 | V-LOGIC-005 llama «no absorbente» a un latch permanente | baja | confirmado | C-F1 |
| 51 | Los dientes de cobertura del registro son nominales | baja | confirmado | C-F1 |
| 52 | template.env con rellenos que nadie reconoce | baja | confirmado | C-F1 |
| 53 | El doc de la intermitencia conserva la fórmula K41 | baja | confirmado | C-F1 |
| 54 | TGM_DATA_DIR ignorado y live_fees.json fuera de la separación | baja | disputado | C-F1 |
| 55 | El ATR se lee como rango y como σ | baja | disputado | C-F4 |
| 56 | Los muros L2 no son volumen acumulado | baja | disputado | C-F1 |
| 57 | El registro de specs no canoniza ni deduplica | baja | disputado | C-F1 |
| 58 | Los votantes del slot 0 leen el global | baja | disputado | C-F3 |
| 59 | El censo espectral depende del slot 0 | baja | disputado | C-F3 |
| 60 | total_commission mezcla activos | baja | disputado | C-F5 |
| 61 | Suposiciones inertes de slot 0 = BTC | baja | disputado | C-F4, C-F6 y C-F3 |
| 62 | La salud del feed arranca «sana» | baja | disputado | C-F1 y C-F5 |
| 63 | Bandas Micro/Meso/Macro incompatibles | baja | disputado | C-F3 y C-F4 |
| 64 | Alias del orquestador que la cabecera da por retirados | baja | disputado | C-F3 |
| 65 | «Fuente única» del horizonte que no lo es | baja | disputado | C-F1 y C-F4 |
| 66 | Opiniones ML del consejo sin espejo con base baja | baja | disputado | C-F3 |
| 67 | Umbral corto del bosque sombra con 1 − P(ganar) | baja | disputado | C-F7 |
| 68 | La deriva del auditor sombra ignora el lado | baja | disputado | C-F3 |
| 69 | Tres fichas de veto sin fecha | baja | disputado | C-F1 |
| 70 | FUENTES_CRUZADAS_CONTRATOS no coincide con la macro | baja | disputado | C-F1 |
| 71 | Un mismo fuente compilado como dos binarios | baja | disputado | C-F1 |
| 72 | Credenciales con precedencias distintas | baja | disputado | C-F1 y C-F5 |
| 73 | El TLS del cliente de órdenes depende de otro manifiesto | baja | disputado | C-F1 y C-F5 |

---

## Severidad alta

### LC-F1-01. El papel sin banderas escribe el estado aprendido en data/prod

Alta. Confirmado. Panel: versión y construcción, n.º 1. Fase dueña: C-F1 (concepto «carpeta del estado» y su función). Zona: Línea C.

**Afirmación.** La carpeta del estado la decide is_demo_env, que sólo mira SHADOW_MODE, USE_TESTNET y BINANCE_USE_DEMO (crates/quantum-arena/src/paths.rs:21-25), y env_data_path_for forma data/{demo|prod} con ese predicado (crates/quantum-arena/src/paths.rs:43-45). El modo de operación lo decide otra regla: sin --force-live, is_demo_mode vale true (god_engine.rs:857-875), y el linaje del genoma usa una tercera, TG_GENOME_ENV=demo fijado para el papel (god_engine.rs:881). La envolvente Kelly se guarda con env_data_path (god_engine.rs:473), se persiste tras los cierres sin mirar el modo (god_engine.rs:3876-3885, y también :3530 y :4972) y se restaura al arrancar (god_engine.rs:2887). CL-48 dice que separa justo esto (crates/quantum-arena/src/paths.rs:27-32).

**Impacto.** Un arranque sin variables y sin --force-live opera en papel, pero el posterior de Jeffreys de la envolvente Kelly se aprende de cierres simulados en data/prod y se restaura en la siguiente sesión armada, donde dimensiona órdenes con dinero real. El genoma sí queda separado por modo, así que conviven dos reglas de separación que se contradicen. Sólo está verificada la vía Kelly; que el fee-breaker y los diarios del papel escriban en prod es plausible, pero no se comprobó.

**Test que falla hoy.** `cl48_papel_sin_banderas_escribe_en_demo` (crates/quantum-arena/tests, serializado con un Mutex de entorno). Entrada: SHADOW_MODE, USE_TESTNET y BINANCE_USE_DEMO ausentes y modo papel. Esperado: la ruta del estado para «kelly_envelope.json» es data/demo/kelly_envelope.json. Actual: la única regla es env_data_path_for(is_demo_env(), "kelly_envelope.json") y devuelve data/prod/kelly_envelope.json. Corrección de un escéptico: el arreglo no puede hacer que is_demo_env siga al modo, porque ese predicado también elige el venue (src/symbol_manager.rs:151; god_engine.rs:2467). Hay que separar una función carpeta_de_estado(modo) de otra venue_es_testnet(), y el test se escribe sobre la primera.

### LC-F1-02. Specs inventados servidos como oficiales y casi nunca refrescados

Alta. Confirmado. Panel: identidad, n.º 2. Fase dueña: C-F1 (procedencia del spec); el refresco desde exchangeInfo es de C-F6. Zona: Línea C.

**Afirmación.** get_official_binance_spec es una tabla literal de 16 brazos (17 nombres, porque MATICUSDT y POLUSDT comparten uno en crates/quantum-arena/src/symbol_registry.rs:184) con un caso por defecto inventado, `_ => (0.01, 0.001, 0.01)` (crates/quantum-arena/src/symbol_registry.rs:169-187). Fija min_notional 5,0 comentado como «Official» y max_leverage 20 para todos (crates/quantum-arena/src/symbol_registry.rs:193-194). El host registra con ella todo el universo al arrancar (god_engine.rs:928-932). El único refresco con exchangeInfo está dentro de `if changed && !top_symbols.is_empty()` (src/symbol_manager.rs:193), que llama a update_registry en src/symbol_manager.rs:212. Sin archivo data/dynamic_config.*, la propuesta se compara consigo misma (src/symbol_manager.rs:165-177) y nunca refresca; con un archivo antiguo de otra lista refresca una vez y deja de hacerlo en cuanto coincide. El riesgo afirma que el mínimo viene de exchangeInfo (crates/risk-engine/src/lib.rs:480-492). El ejecutor acepta el spec del registro como filtro real si falla exchangeInfo (crates/execution-engine/src/executor.rs:1676-1688), aunque D-631 prohíbe usar pasos ajenos al símbolo (crates/execution-engine/src/executor.rs:1655-1661). El forense y el entrenador registran además max_leverage 125 (src/bin/audit_forensic_backtest.rs:281; src/bin/train_forest.rs:1300) frente al 20 del vivo. La tabla del forense contradice a la del registro en ETHUSDT, SOLUSDT y NEARUSDT (src/bin/audit_forensic_backtest.rs:260-265 frente a crates/quantum-arena/src/symbol_registry.rs:171-180), así que al menos una está mal.

**Impacto.** El riesgo decide la viabilidad, el régimen micro y el techo de apalancamiento con un spec no medido que se presenta como oficial. Es el «desconocido convertido en número» que D-631 prohibió en el ejecutor. Desde el repo no se pueden comprobar los filtros reales de Binance: el defecto es la procedencia, no un valor concreto.

**Test que falla hoy.** (1) `g1_spec_sin_evidencia_no_se_inventa` (crates/quantum-arena/tests, binario propio): update_dynamic_universe(["IDSPUSDT"]) y update_registry(vec![get_official_binance_spec("IDSPUSDT")]). Esperado: try_spec(0) es None, o el spec queda marcado como no medido y evaluate_quantum_order rechaza con REJ_SPEC. Actual: Some con tick 0,001, paso 0,01 y min_notional 5,0. Si el arreglo consiste en pedir exchangeInfo al arrancar, el test debe ejercitar la función de arranque. (2) Tras extraer a una función pura la decisión del tick del demonio: con la propuesta igual a la persistida, o sin archivo, el plan del tick incluye el refresco de specs. Actual: el refresco vive dentro de `if changed` y no ocurre.

### LC-F1-03. El slot 0 recibe el gen del stop de BTC sea cual sea su símbolo

Alta. Confirmado. Panel: identidad, n.º 1. Fase dueña: C-F4 (gate de riesgo) y C-F3 (sonda del núcleo). Zona: Línea C.

**Afirmación.** La sonda de banda operable del núcleo elige sl_atr_mult_btc si coin_id == 0 y sl_atr_multiplier en otro caso (lib.rs:1547-1551), y el gate de riesgo hace lo mismo (crates/risk-engine/src/lib.rs:891-895). Nada comprueba que el slot 0 sea BTCUSDT; en vivo lo es sólo porque la lista literal del bootloader pone btcusdt primero (crates/god-engine-core/src/bootloader.rs:321), con atomusdt en la posición 13 (crates/god-engine-core/src/bootloader.rs:334). El forense y el entrenador ponen el símbolo que miden en el slot 0 (src/bin/audit_forensic_backtest.rs:718-719; src/bin/train_forest.rs:1477-1484). El otro gen _btc de la geometría, tp_rr_ratio_btc, se aplica a todas las monedas (lib.rs:1561 y crates/risk-engine/src/lib.rs:902), así que el sufijo no tiene un significado único. Se incumple el contrato del propio forense (src/bin/audit_forensic_backtest.rs:253-255) y ADR-0011, que fija un mapa slot↔símbolo inmutable en el que try_symbol, try_spec y try_index son el mismo mapa (docs/adr/ADR-0011-identidad-canonica-del-simbolo.md:21-35).

**Impacto.** Con los valores por defecto (sl_atr_mult_btc 1,0 en crates/quantum-arena/src/genome.rs:710 y sl_atr_multiplier φ/2 ≈ 0,809 en crates/quantum-arena/src/genome.rs:780), el stop del forense para ATOMUSDT es unas 1,24 veces el del vivo. La evolución ajusta el gen 17 con la serie que cae en el slot 0, que en el examen del demonio es la más larga (crates/evolution-engine/src/online_daemon.rs:646-651), mientras el vivo opera las alts con el gen 82. El instrumento de medida no mide la geometría que opera el vivo.

**Test que falla hoy.** `g1_el_stop_no_depende_del_slot` (crates/risk-engine/tests, binario propio, preparado como crates/risk-engine/tests/geometry_hurst_contract.rs). Entrada: dos specs idénticos IDSLAUSDT e IDSLBUSDT; precio 100, ATR 0,2 y Hurst 0,5 en coins[0] y coins[1]; sl_atr_mult_btc 3,0, sl_atr_multiplier 1,5 y tp_rr_ratio_btc 3,0; la misma intención larga con τ = 1 h evaluada con evaluate_quantum_order(0, …) y evaluate_quantum_order(1, …). Esperado: la misma distancia relativa del stop en los dos slots. Actual: la del slot 0 es unas 2 veces la del slot 1. Corrección de los escépticos: hay que comprobar que las dos órdenes salen validadas, o comparar el sl_pct de la geometría del gate, porque la intención puede caer antes en EV, geometría o nocional.

### LC-F1-04. El perceptrón vota con el peso hebbiano de otra moneda

Alta. Confirmado. Panel: desconocido, n.º 1. Fase dueña: C-F3. Zona: Línea C (lector de signal-engine y escritor del cierre del núcleo; si el escritor cae en la reserva de Línea E, va como pedido a Línea E). Causa raíz común: LC-F1-06.

**Afirmación.** PerceptronGateEngine lee perceptron_hebbian_weight con get_scoped_parameter y, si falta, usa 1,0 (crates/signal-engine/src/perceptron_gate.rs:168-176). La resolución busca la clave del símbolo, luego la de la moneda y por último la global (crates/omniscient-registry/src/lib.rs:261-279). Nadie escribe la clave por símbolo ni por moneda. El único escritor fuera de los tests es el cierre del núcleo, que escribe la clave global con el peso de la moneda que acaba de cerrar y siembra el peso de un símbolo sin historia desde ese global (lib.rs:4227-4248). Esto contradice D-190 (lib.rs:4227) y MOD2/7-005, que sólo se aplicó al lector hebbian_mult (lib.rs:4563-4575). La auditoría IV ya lo había señalado (docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_IV_2026-09-24.md:287).

**Impacto.** Con un OFI de 0,6, el voto vale ≈ 0,136 si la última moneda que cerró dejó un peso de 0,25 y ≈ 0,739 si dejó 2,0: un factor de 5,4 que decide otra moneda. Corrección de los escépticos: el voto escalar sólo fija la convicción cuando el consenso espectral no está activo; con él activo, la convicción neta pasa a v_dom·(0,70 + 0,30·coherencia) (crates/signal-engine/src/orchestrator.rs:384 y :526-527) y el voto espectral del perceptrón (lib.rs:2296) no usa el peso. Por otro lado el daño es mayor de lo descrito: como el peso por símbolo nace del global de otra moneda en el primer cierre (lib.rs:4228-4239), el lector hebbian_mult que MOD2/7-005 corrigió también hereda un peso contaminado.

**Test que falla hoy.** `perceptron_no_usa_peso_hebbiano_de_otra_moneda` (tests de crates/signal-engine/src/perceptron_gate.rs). Entrada: set_for_coin(1, "order_flow_imbalance", 0.6) y set_scoped("ETHUSDT", "order_flow_imbalance", 0.6); caso A con set("perceptron_hebbian_weight", 0.25) y caso B con 2.0, que simulan lo que deja el cierre de otra moneda. Esperado: evaluate_for_coin(1, "ETHUSDT") da lo mismo en A y en B, igual a infer(0.6, 1.0) ≈ 0,157. Actual: ≈ 0,136 en A y ≈ 0,739 en B. Complemento pedido por un escéptico: tras un cierre de la moneda 0 con peso global 0,25, el primer cierre de ETHUSDT debe sembrar su peso en el neutro 1,0; hoy lo siembra en 0,25 (lib.rs:4229-4238).

---

## Severidad media

### LC-F1-05. X-016 veta con tres bandas fijas y escalones

Media (el revisor la puso alta; los escépticos la bajan porque la frecuencia del veto no está medida). Confirmado. Panel: horizonte, n.º 1. Fase dueña: C-F3. Zona: Línea C (bloque posterior a la fusión, fuera de las ramas 1 a 10).

**Afirmación.** El acondicionamiento espectral de cada candidato no plano (lib.rs:6937-6938) calcula la marea como (swing·0,60 + secular·0,40)·signo y el táctico como tactical_score·signo (lib.rs:6942-6943). Son proyecciones con centros fijos de 60 s, 4 h y 30 d (crates/quantum-arena/src/temporal_spectrum.rs:1147-1163). La persistencia se lee en spec.dominant_tau_ms (lib.rs:6945-6946). La decisión es binaria (lib.rs:6965-6966): la señal pasa a Flat (lib.rs:7045-7046) salvo que marea ≥ −0,05 y táctico ≥ 0, o marea > 0,02, táctico < 0 y wave_phase_ok. El comentario del bloque dice que, «en lugar de un veto binario booleano rígido», el tensor «MODULA LA CONVICCIÓN», con umbral 0,05 (lib.rs:6950-6953); el código usa 0,02 y anula la señal. También contradice crates/quantum-arena/src/temporal_spectrum.rs:105-107 («ningún código decide por pertenecer a una banda»). Corrección de los escépticos: el bloque sí lee expected_duration_ms, pero después de decidir, para reescribirlo como media geométrica entre la τ de la intención y continuous_resonant_tau_ms, recortada a [30 s, 12 h] (lib.rs:7104-7124). Con D-745 (crates/risk-engine/src/lib.rs:41-48) sólo hay contradicción por extensión, porque D-745 enumera geometría, trailing, caducidad, Kelly del cierre y apalancamiento, no puertas de entrada.

**Impacto.** La puerta decide sobre todas las entradas no planas. Con la marea en [−0,05; 0,02], pasar el táctico de 0 a −0,001 convierte una intención de confianza ≈ 0,70 en Flat. La frecuencia real del veto no está medida.

**Test que falla hoy.** Extraer el bloque a una función pura `veredicto_x016(macro_tide, tactical_align, wave_phase_ok, coherence, entropy, persist, trending, conf) -> f64` sin cambiar la conducta y sin el modulador de contagio (lib.rs:7092-7100), que se aplica después. Test `x016_es_continuo_en_la_alineacion_tactica`: largo, conf 0,70, macro_tide 0,0, wave_phase_ok true, coherence 0,15, entropy 0,65, persist 0, trending false; tactical_align +1e-6 frente a −1e-6. Esperado: |Δconf| ≤ 1e-3. Actual: 0,70 frente a 0,0 (Flat por lib.rs:6965-6967 y :7045-7046). El segundo test del revisor (que el veto lea la τ de la intención) se retira a petición de un escéptico, porque no hay contrato vigente que lo respalde; pasa a ser una regla del contrato de horizonte continuo.

### LC-F1-06. Las lecturas por moneda caen al global (causa raíz)

Media. Confirmado. Panel: identidad, n.º 6. Fase dueña: C-F1 (semántica de ámbito del registro). Zona: Línea C (núcleo y omniscient-registry). El arreglo debe coordinarse con los lectores que hoy dependen del global en crates/signal-engine/src/flow_excitation_confluence.rs:192-203 (zona de Qoder/GLM).

**Afirmación.** set_reg escribe cada dato por moneda también en la clave global y en la del símbolo (lib.rs:4423-4429). get_for_coin_or y get_scoped_value_or recurren a la clave global si falta la de la moneda (crates/omniscient-registry/src/lib.rs:244-250 y :226-231), y get_scoped_parameter hace lo mismo (crates/omniscient-registry/src/lib.rs:255-279). Si falta el dato de la moneda X, se lee el de la última moneda que escribió. El propio código lo reconoce (crates/signal-engine/src/renyi_tsallis_entropy.rs:233-238), y MOD2/7-009 lo corrige sólo en dos votantes. Es la causa raíz de LC-F1-04, LC-F1-07, LC-F1-08, LC-F1-44, LC-F1-46 y LC-F1-58.

**Impacto.** Todo lector por moneda de una clave escrita por set_reg recibe datos de otra moneda en lugar de «desconocido». Corrección de un escéptico: el ejemplo del revisor (el cierre de una posición adoptada) es inexacto, porque esa posición nace con ml_prediction 0 (crates/execution-engine/src/reconciliation.rs:115-124) y sólo el lector de lib.rs:4312 la alcanza, en el primer tick. Las vías vivas fuertes son StatArb (crates/strategy-core/src/stat_arb.rs:388) y la proyección espectral (crates/signal-engine/src/proyeccion_espectral.rs:44 y :66).

**Test que falla hoy.** `g1_lectura_por_moneda_no_devuelve_otra_moneda` (crates/god-engine-core/tests, binario propio): universo ["IDGLAUSDT","IDGLBUSDT"] con specs, bosque IDGLAUSDT_MOTOR con base 0,30 y un process_tick_dual(0, …). Esperado: registry.get_for_coin_or(1, "ml_model_base", NaN) es NaN. Actual: 0,30. Variante más directa que propone un escéptico: ProyeccionEspectralEngine con sólo las claves globales espectral_masa_resuelta, espectral_senal_proyectada y espectral_concentracion devuelve 0,0 para una moneda sin claves propias; hoy vota con los datos globales.

### LC-F1-07. StatArb hereda la z y la vida media de otra moneda

Media. Confirmado. Panel: desconocido, n.º 2. Fase dueña: C-F3. Zona: Línea C (consumidor en strategy-core). Causa raíz: LC-F1-06.

**Afirmación.** El lector promete que una clave ausente lleva al respaldo bit a bit con vecm_zscore (crates/strategy-core/src/stat_arb.rs:378-379), pero get_scoped_parameter (crates/strategy-core/src/stat_arb.rs:387-389) cae al global (crates/omniscient-registry/src/lib.rs:279). El núcleo sólo publica statarb_* cuando la SDE de la moneda está madura (lib.rs:5194-5206; crates/strategy-core/src/stat_arb.rs:117-121), y dominant_tau_ms se publica siempre (lib.rs:4894-4899). Una moneda B sin madurez lee la z y la t½ de la última moneda A que publicó, pasa la guarda con su propio τ* (crates/strategy-core/src/stat_arb.rs:405-408) y vota reversión con la z de A.

**Impacto.** Con z = 2,5 el voto vale −tanh(1/1,5) ≈ −0,583, casi pleno en el consenso tensorial. Corrección de los escépticos: la SDE madura con 10 actualizaciones (crates/strategy-core/src/vecm_arbitrage.rs:529-533) y el spot también llega por WebSocket (crates/data-pipeline/src/spot_feed.rs:174), así que en una moneda con spot el calentamiento dura segundos. El caso persistente es un futuro sin par spot del mismo nombre, porque el feed empareja por nombre idéntico (crates/data-pipeline/src/spot_feed.rs:150-154) y el ranker admite bases como 1000PEPEUSDT (crates/data-pipeline/src/dynamic_ranker.rs:290). No se verificó que haya un símbolo así en el universo vivo. Con el consenso espectral activo, el voto escalar no fija la convicción (crates/signal-engine/src/orchestrator.rs:526-527).

**Test que falla hoy.** `statarb_moneda_sin_fisica_propia_no_hereda_z_global` (tests de crates/strategy-core/src/stat_arb.rs). Moneda 0 madura, escrita como lo hace set_reg (set, set_for_coin(0, …) y set_scoped("BTCUSDT", …)) con statarb_ou_zscore 2,5, statarb_half_life_ms 60 000 y dominant_tau_ms 60 000. Moneda 1 sólo con lo suyo: dominant_tau_ms 60 000 y vecm_zscore 0,0, por moneda y por símbolo. Esperado: StatArbEngine::new(30, 1.5).evaluate_for_coin(1, "ETHUSDT") == 0,0. Actual: ≈ −0,583.

### LC-F1-08. El bosque sombra aprende con features globales y el retorno de la moneda que cierra

Media. Confirmado. Panel: desconocido, n.º 6. Fase dueña: C-F7. Zona: evolution-engine (consumidor del registro). Causa raíz: LC-F1-06.

**Afirmación.** Al registrar el cierre de una moneda, el daemon lee orderbook_imbalance, price_acceleration, atr_pct y price_velocity con get_value_or (crates/evolution-engine/src/online_daemon.rs:1162-1173), que sólo consulta la clave global (crates/omniscient-registry/src/lib.rs:198-200), y los junta con el Hurst, el spread y el retorno de la moneda que cierra (crates/evolution-engine/src/online_daemon.rs:1174). El comentario dice que lee el estado de esa moneda (crates/evolution-engine/src/online_daemon.rs:1148-1153). La predicción del mismo bosque usa get_for_coin_or (crates/evolution-engine/src/online_daemon.rs:1012-1028): entrena y sirve con features distintas.

**Impacto.** Con N monedas parejas, la fracción de observaciones mal emparejadas ronda (N−1)/N. El bosque gobierna el freno de confianza por moneda (lib.rs:6808-6833) y, con la evolución viva armada, los umbrales ML del genoma (crates/evolution-engine/src/online_daemon.rs:978-992). Corrección de los escépticos: el freno se autolimita, porque sólo actúa si la cota de acierto supera 0,5 (lib.rs:6820-6824). Las features se leen cuando sample_realized_returns detecta el delta de PnL en el bucle de unos 500 ms (crates/evolution-engine/src/online_daemon.rs:1054-1055), no en el instante exacto del cierre.

**Test que falla hoy.** `features_de_cierre_son_de_la_moneda_que_cierra` (tests de crates/evolution-engine/src/online_daemon.rs): arena con 2 monedas y unified_capital > 0; set_for_coin(0, "orderbook_imbalance", 0.8) y set("orderbook_imbalance", -0.5); el pnl_realized de la moneda 0 cambia entre dos llamadas a sample_realized_returns (la primera fija la línea base). Esperado: la última observación del bosque tiene obi == 0,8. Actual: −0,5.

### LC-F1-09. La propuesta de universo de ADR-0011 nunca se aplica al reiniciar

Media. Confirmado. Panel: identidad, n.º 8. Fase dueña: C-F6. Zona: Línea C (host y src/symbol_manager.rs).

**Afirmación.** ADR-0011, decisión 2, dice que «la propuesta se aplica al reiniciar» (docs/adr/ADR-0011-identidad-canonica-del-simbolo.md:24-28). El arranque toma siempre la lista literal de 26 símbolos del bootloader, que además ignora _is_demo (crates/god-engine-core/src/bootloader.rs:317-351). La publica (god_engine.rs:913) y sobrescribe con ella la configuración leída (god_engine.rs:954). El host lee data/{demo|prod}/dynamic_config.* (god_engine.rs:938-939, que resuelve crates/quantum-arena/src/paths.rs:43-44), y el demonio lee y escribe los literales data/dynamic_config.* (src/symbol_manager.rs:165, :171 y :203-207). El hot-reload vigila la ruta del entorno (god_engine.rs:1316-1328), que el demonio no toca.

**Impacto.** La rotación no existe ni en caliente ni en frío: el universo vivo es siempre la lista del bootloader. Al arreglarlo se activarán LC-F1-45 (líderes por prefijo) y LC-F1-49 (lista negra del escáner), porque la propuesta pasará a gobernar el universo.

**Test que falla hoy.** `cl38_la_propuesta_persistida_es_la_que_lee_el_arranque`. (a) Guarda sobre la parte de producción de src/symbol_manager.rs: no contiene el literal `"data/dynamic_config` sino EnvManager::data_path. Actual: aparece en src/symbol_manager.rs:165, :171 y :203-207, y también en el demonio sin llamadores (src/symbol_manager.rs:55, :61 y :94-98). (b) Con dynamic_config.json persistido en la ruta del entorno con ["IDPRAUSDT","IDPRBUSDT"], la función de arranque devuelve ese universo. Actual: execute_phase_1_and_2(false) devuelve siempre los 26 literales.

### LC-F1-10. El gestor de símbolos usa is_demo_env como selector de venue

Media. Confirmado. Panel: versión y construcción, n.º 2. Fase dueña: C-F6, con la función de venue que fija C-F1. Zona: Línea C.

**Afirmación.** EnvManager::is_demo_env está documentado para resolver rutas (src/env_manager.rs:4-5). El demonio vivo lo usa como is_testnet (src/symbol_manager.rs:151) y lo pasa a fetch_dynamic_universe (src/symbol_manager.rs:155), que elige la base de testnet y pone min_vol = 0 (crates/data-pipeline/src/dynamic_ranker.rs:68-72 y :94). Con SHADOW_MODE=true y USE_TESTNET=false, como fija LAUNCH_SHADOW_MAINNET.bat:22-25, el WebSocket y las órdenes van a mainnet porque sólo miran USE_TESTNET (god_engine.rs:1029-1033 y :1172-1186). El sondeo de open interest elige su URL con el mismo predicado (god_engine.rs:2467-2472). Corrección de un escéptico: src/symbol_manager.rs:19, :29 y :105 pertenecen a evolve_symbols_daemon, que no tiene llamadores; la ruta viva es la de las líneas 151, 155 y 212.

**Impacto.** En shadow mainnet, que existe para medir mainnet sin riesgo, el universo y el OI de las features salen de testnet. Los filtros de testnet sólo llegan al registro si update_registry corre, lo que exige `changed` (src/symbol_manager.rs:193) y depende de un archivo heredado en data/ (lo escriben src/bin/evolution.rs:696 y src/bin/config_compiler.rs:26).

**Test que falla hoy.** `symbol_manager_shadow_pide_universo_a_mainnet` (test del crate raíz, serializado): SHADOW_MODE=true y USE_TESTNET ausente. Esperado: el valor que se pasa como is_testnet a fetch_dynamic_universe (src/symbol_manager.rs:151-155) es false, igual que el venue de god_engine.rs:1172-1176. Actual: true, y la consulta va a https://testnet.binancefuture.com. Se formula sobre el valor que hoy se pasa, como pide un escéptico, para que quede en rojo sin código nuevo.

### LC-F1-11. El régimen global sale del slot 0, no de BTC

Media. Confirmado. Panel: identidad, n.º 5. Ya figura en docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:26 y :88. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** D-404 dice que el régimen global se basa en BTC (lib.rs:2539), pero el código usa `if coin_id == 0` y feature_engines[0] (lib.rs:2538-2540), y publica regime_p_* y market_regime_p_* sin comprobar que el slot 0 sea BTCUSDT (lib.rs:2615-2633). Los consumidores son vivos: el orquestador contrae el margen por p_crash, p_bull y p_chaos (crates/risk-engine/src/orchestrator.rs:158-160) y el riesgo lee market_regime (crates/risk-engine/src/lib.rs:1386).

**Impacto.** En el forense de ATOMUSDT, que está en el slot 0, los vetos de cartera reaccionan a ATOM; en vivo reaccionan a BTC. Si BTC falta, el régimen se publica igual en lugar de quedar desconocido.

**Test que falla hoy.** `g1_regimen_global_no_sale_de_un_slot_sin_btc` (crates/god-engine-core/tests, binario propio): universo ["IDRGUSDT"] con spec y varios process_event de depth con precio creciente. Esperado: el régimen no sale de IDRGUSDT; vale cualquiera de las dos variantes del arreglo, clave market_regime_p_range ausente o régimen neutral publicado, como propone la hoja de ruta. Actual: Some, calculado con IDRGUSDT.

### LC-F1-12. El examen walk-forward del demonio cruza identidades

Media. Confirmado. Panel: identidad, n.º 4. Fase dueña: C-F7. Zona: evolution-engine.

**Afirmación.** El doc promete specs sintéticos WFD{i} (crates/evolution-engine/src/online_daemon.rs:429), pero con el universo cargado try_index sólo busca en el universo (crates/quantum-arena/src/symbol_registry.rs:147-152) y la función devuelve 0..n (crates/evolution-engine/src/online_daemon.rs:448), que son slots de monedas reales. Las series se ordenan por longitud (crates/evolution-engine/src/online_daemon.rs:646) y se asignan a esos slots (:651). La serie de la moneda k se juzga en el slot i con el bosque, el tick y el gen del stop del símbolo i (crates/evolution-engine/src/online_daemon.rs:494 y :560; lib.rs:4593-4595). El examen inserta además símbolos WFD en el registro del proceso vivo (crates/evolution-engine/src/online_daemon.rs:446).

**Impacto.** La selección de genomas en vivo se juzga con identidades cruzadas. Sólo decide promociones con la evolución viva armada (god_engine.rs:2108); en prod está desarmada por defecto.

**Test que falla hoy.** Corregido por los escépticos: el arreglo natural es conservar la identidad, no inventar slots WFD, que chocarían con ADR-0011 y caerían en el respaldo de LC-F1-32. `examen_wf_juzga_cada_serie_en_su_slot` (módulo de tests de crates/evolution-engine/src/online_daemon.rs con un mutex local, porque UNIVERSE_TEST_MUTEX es #[cfg(test)] de quantum-arena): universo ["IDWFAUSDT","IDWFBUSDT"] y una serie más larga para la moneda 1. Esperado: la serie de la moneda 1 se reproduce en el slot 1. Actual: wf_real_coin_ids devuelve [0, 1] y la serie más larga, que es la de la moneda 1, va al slot 0. Hay que corregir también el doc de la línea 429.

### LC-F1-13. El backtest nativo cae al slot 0 de otro símbolo

Media. Confirmado. Panel: identidad, n.º 9. Fase dueña: C-F7. Zona: backtest-engine.

**Afirmación.** asegurar_spec_nativo y run_backtest_native resuelven el slot con try_index(symbol).unwrap_or(0) (crates/backtest-engine/src/lib.rs:31 y :60) y sólo registran el spec si el slot 0 está vacío (crates/backtest-engine/src/lib.rs:32). run_backtest_native calcula el slot antes de asegurar el spec (crates/backtest-engine/src/lib.rs:60-61). Por FFI, un símbolo que no es UTF-8 válido se convierte en "BTCUSDT" (crates/backtest-engine/src/lib.rs:586 y :673).

**Impacto.** Es de evidencia, no de órdenes vivas: las exportaciones FFI no tienen llamadores en el repo y polars_evolver no tiene consumidor (crates/evolution-engine/src/polars_evolver.rs:1). Un escéptico añade que golden_backtest_determinism no fija el entorno como dice, porque update_registry(Vec::new()) sobre un registro no vacío no hace nada (crates/backtest-engine/src/lib.rs:944; crates/quantum-arena/src/symbol_registry.rs:84-103).

**Test que falla hoy.** `g1_backtest_de_un_segundo_simbolo_no_hereda_el_slot_0` (crates/backtest-engine/tests, binario propio): universo vacío, update_registry con el spec IDBTAUSDT (tick 0,1) y asegurar_spec_nativo("IDBTBUSDT"). Esperado: try_index("IDBTBUSDT") es Some. Actual: None, y el backtest correría en el slot 0, que es IDBTAUSDT.

### LC-F1-14. Dos «τ dominante» con el mismo nombre

Media. Confirmado. Panel: horizonte, n.º 5. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** C-2 promete que el veto de grupo, el dimensionador de TP/SL y las ramas hablen del mismo horizonte (crates/quantum-arena/src/temporal_spectrum.rs:1192-1208). El núcleo publica al arena tau_operativa_unificada() (lib.rs:2005-2012), pero en el registro, bajo la clave "dominant_tau_ms", publica el campo crudo spec.dominant_tau_ms (lib.rs:4894-4899). Ese campo es un nodo de la malla ×4 recortado a [30 s, 12 h] (crates/quantum-arena/src/temporal_spectrum.rs:664-680). Con el nodo crudo deciden el enfriamiento (lib.rs:5503-5510), la persistencia de X-016 (lib.rs:6945), la banda de Kelly del cierre (lib.rs:4126), el consejo (lib.rs:7570) y StatArb (crates/strategy-core/src/stat_arb.rs:402). Corrección de los escépticos: el risk-engine tampoco dimensiona con la unificada en el camino vivo. Toda intención que pasa X-016 sale con expected_duration_ms > 0 (lib.rs:7120-7124) y horizon_tau_ms_coin usa ese valor primero (crates/risk-engine/src/lib.rs:1494-1496). La τ que dimensiona es la media geométrica de lib.rs:7104-7119. Conviven tres valores: el nodo crudo, el centroide y la media geométrica.

**Impacto.** Sin habilidad medida, algo habitual en el arranque, la puerta, el enfriamiento y las estrategias razonan en un nodo que salta por factores de 4 y que no es el horizonte de la orden. Es la clase de disparidad que D-745 vino a cerrar.

**Test que falla hoy.** `la_tau_dominante_publicada_es_una_sola` (god-engine-core): sembrar temporal_spectrum[0] sin habilidad medida y con la energía repartida entre nodos, de modo que el argmax y el centroide difieran (con toda la energía en un nodo coinciden). Las entradas no deben estar bloqueadas, porque la publicación (lib.rs:4899) va después del retorno de lib.rs:4888-4890. Tras un process_event se compara registry.get_for_coin(0, "dominant_tau_ms") con arena.coins[0].dominant_tau_ms. Esperado: iguales. Actual: el nodo crudo frente al centroide.

### LC-F1-15. Una posición adoptada sin τ vive con dos horizontes

Media. Confirmado. Panel: horizonte, n.º 4. Fase dueña: C-F5 (host y adopción) y C-F3 (gestión del núcleo). Zona: Línea C.

**Afirmación.** La adopción al arrancar abre la ranura con TP = SL = 0 (crates/execution-engine/src/reconciliation.rs:107-127), y la cadena open_with_horizon, open_with_full_meta y open_with_fee termina con entry_tau_ms = 0 (crates/quantum-arena/src/position.rs:161-216 y :333-352). El host sólo escribe la τ si el diario la recuerda (god_engine.rs:1969-1972) y protege la posición en el exchange con el ancla rápida de 30 s (god_engine.rs:48-52 y :68). El núcleo gestiona la misma posición con spec.dominant_tau_ms, que cambia tick a tick (lib.rs:3229-3242). Con ella calcula el stop de respaldo (lib.rs:3248-3250), que manda porque TP = SL = 0 (lib.rs:3287-3297), y todas las caducidades (lib.rs:3534-3541 y :3582-3584). Esto contradice D-745 (crates/risk-engine/src/lib.rs:41-48). El comentario «ancla rápida = mínimo de la curva por construcción» (god_engine.rs:1993-1994) es falso con una pendiente del SL negativa, que el genoma permite (crates/quantum-arena/src/genome.rs:2112).

**Impacto.** Tras un reinicio sin diario, el stop del núcleo y el del exchange difieren, y el del núcleo se mueve con el espectro. Si la dominante salta de 68,7 s a 4,89 h, la vida absoluta de la posición pasa de 50 min a 24 h (el techo de 86 400 000 ms) sin que cambie nada de la posición. Sólo afecta a posiciones adoptadas sin τ en el diario; no se midió cuántas hay.

**Test que falla hoy.** Un escéptico advierte que process_event recalcula dominant_tau_ms en cada evento (crates/quantum-arena/src/temporal_spectrum.rs:676-680), así que asignarla a mano puede quedar pisado. Por eso: extraer la resolución de la τ de gestión a una función pura `tau_de_gestion(entry_tau_ms, dominante)` y añadir `posicion_adoptada_sin_tau_tiene_un_solo_horizonte`. Entrada: entry_tau_ms 0 con dominante 30 000 y con 43 200 000. Esperado: el mismo valor en los dos casos, el que usa genome_protection_prices(…, 0) en el exchange (30 s). Actual: 30 000 frente a 43 200 000. Versión de integración del revisor: genoma con sl_at_tau(30 s) = 0,3 % y sl_at_tau(12 h) = 1,5 %, ATR pequeño y precio de 100 a 99,2; hoy el motor con dominante de 30 s cierra por stop y el de 12 h no.

### LC-F1-16. Kelly salta en el umbral de supervivencia

Media. Confirmado. Panel: horizonte, n.º 2. Fase dueña: C-F4. Zona: Línea C.

**Afirmación.** calculate_kelly_fraction elige entre dos fórmulas con un if sobre capital_ratio < kelly_survival_cap_ratio (crates/risk-engine/src/kelly.rs:86-95), y las dos ramas no coinciden en la frontera. El comentario del bloque habla de «regímenes continuos» que protegen «de forma continua» (crates/risk-engine/src/kelly.rs:74-80), y D-641 prohíbe que la conducta cambie de golpe al cruzar un umbral de capital (crates/risk-engine/src/capital_regime.rs:1-20). El umbral por defecto es φ (crates/quantum-arena/src/genome.rs:808). El núcleo guarda la fracción (lib.rs:4143-4157) y el riesgo la usa en la orden siguiente (crates/risk-engine/src/lib.rs:387-390 y :411-415).

**Impacto.** Con los parámetros del test, la fracción se multiplica por ≈ 2,07 al cruzar 13·φ ≈ 21,03 USD. Correcciones de los escépticos: el tamaño y el signo del salto dependen de wr, kelly_at_tau(τ), s y kelly_expansion_mult (con strategy_base 0,75 y wr 0,6 el salto es hacia abajo, de 0,572 a 0,475). base_capital no es fijo, porque el host lo sobrescribe con el saldo real al pasar a mainnet (god_engine.rs:3650-3652). El núcleo pasa la cota inferior del PF (lib.rs:4137-4141). Lo verificable es la discontinuidad, no un factor fijo.

**Test que falla hoy.** `kelly_es_continuo_en_el_umbral_de_supervivencia` (risk-engine): φ = (1+√5)/2; a = calculate_kelly_fraction(0.6, 2.0, 13·φ − 1e-6, 13.0, φ, φ, 0.01, 1.0, 0.30, 0.5) y b igual con 13·φ + 1e-6. Esperado: |a − b| < 1e-3. Actual: a ≈ 0,06869 y b ≈ 0,14250 (calculado a mano; el tope de ruina no corta).

### LC-F1-17. Apalancamiento de arranque con el margen de otra ranura

Media. Confirmado. Panel: horizonte, n.º 8. Fase dueña: C-F5. Zona: Línea C (host).

**Afirmación.** En el bootstrap (envelope_n < 30, god_engine.rs:4064), el apalancamiento enviado es round(nocional / margen) (god_engine.rs:4070-4076), con el margen de la primera ranura abierta de la moneda (god_engine.rs:4039), en orden [0, 1, 2] (crates/quantum-arena/src/position.rs:761-763). El mismo bloque toma la τ de la ranura reservada (god_engine.rs:4053-4062) y su comentario CL-4 dice que no se debe leer otra ranura (god_engine.rs:4103-4105). Sin margen conocido, el respaldo salta en cap_now = 50 USD (god_engine.rs:4042 y :4066-4072), en contra de D-641 (crates/risk-engine/src/capital_regime.rs:1-20).

**Impacto.** Con dos posiciones en la misma moneda, la orden nueva sale con el apalancamiento derivado del margen de la otra: ranura 0 con margen 2,0 y ranura nueva con nocional 5,05 y margen 1,01 dan 3× en lugar de 5×. Correcciones de los escépticos: por el techo de CL-41 (crates/risk-engine/src/envio.rs:85-86; god_engine.rs:4298-4301) el defecto sólo puede bajar el apalancamiento, nunca superar el validado. El escalón de 50 USD es casi inalcanzable, porque cuando el host calcula boot_lev el núcleo ya abrió la ranura reservada (god_engine.rs:3406-3414). Que envelope_n < 30 sea todo el régimen vivo actual no se puede comprobar desde el código.

**Test que falla hoy.** Extraer `apalancamiento_de_arranque(arena, reserva, notional, cap_now) -> u32` sin cambiar la conducta y añadir `bootstrap_usa_el_margen_de_la_ranura_reservada`: ranura 0 abierta con margen 2,0, ranura 1 reservada y abierta con margen 1,01, nocional 5,05 y capital 13. Esperado: 5. Actual: 3. El test del escalón de 50 USD queda como higiene, porque ejercita un camino casi inalcanzable.

### LC-F1-18. El freno del bosque lee P(ganar) como P(subida)

Media. Confirmado. Panel: espejo, n.º 1. Fase dueña: C-F3 (lectura en el núcleo) y C-F7 (publicación y registro de acierto). Zona: Línea C (núcleo fuera de las ramas 1 a 10, y evolution-engine).

**Afirmación.** El núcleo interpreta forest6_prob como voto direccional (lib.rs:6808-6818) y aplica f6 < 1 − c a los largos y f6 > c a los cortos, con recorte 2·(1 − c) (lib.rs:6824-6825 y :6829). El daemon entrena con was_profitable = net_pnl_pct > 0 (crates/evolution-engine/src/online_random_forest.rs:82) y publica 0,6·sigmoide + 0,4·clase (crates/evolution-engine/src/online_random_forest.rs:273-276) sin feature de lado al servir (crates/evolution-engine/src/online_daemon.rs:1013-1027). Uno de sus caminos de entrenamiento sí codifica el lado en la sexta feature (crates/evolution-engine/src/online_daemon.rs:938), así que además hay una discrepancia entre entrenamiento y servicio. El registro de acierto compara el voto con etiqueta_barrera (lib.rs:3938-3940), que en un corto sólo da Some(false) o None (lib.rs:192-198), y lo mide contra ½.

**Impacto.** Cuando el daemon publica forest6_prob, un corto que el bosque considera ganador pierde confianza (×2·(1 − c)). Un votante constante bajista lleva la cota de acierto por encima de ½, y entonces el freno recorta largos sin evidencia. Contribuye también que la etiqueta larga tiene una base de ≈ 1/3 con RR 2 y se compara con ½. No se comprobó si el daemon corre en producción; sin él, la clave vale 0,5 y no hay voto.

**Test que falla hoy.** (1) `freno_bosque_no_recorta_corto_que_predice_ganador`: forest6_prob 0,9, cota 0,70 y un Short con confianza 0,80. Esperado: 0,80. Actual: 0,48. (2) `registro_bosque_no_premia_al_votante_constante_bajista`, llevado por una función extraída `observacion_bosque(voto, is_long, reason_code) -> Option<bool>` (un escéptico señala que comparar a mano sobre TasaAcierto no detectaría el arreglo y que un unwrap entraría en pánico si el corto pasa a None): 20 largos TP, 40 largos SL y 60 cortos TP, siempre con voto «baja». Esperado: la cota de Wilson no supera ½. Actual: 100 aciertos de 120 y cota ≈ 0,757.

### LC-F1-19. Rama 20: el corto usa el gen del largo

Media. Confirmado. Panel: espejo, n.º 5. Fase dueña: C-F3. Zona: Línea C (rama 20, fuera de las ramas 1 a 10).

**Afirmación.** La rama ML del respaldo sin libro descarta el gen corto (lib.rs:5655), toma el lift sólo del gen largo (lib.rs:5659) y aplica base ± lift en unidades absolutas (lib.rs:5669 y :5680), con techos de convicción asimétricos (lib.rs:5671 y :5682). Contradice el comentario B3.36 («los MISMOS genes», lib.rs:5647-5654) y K-23 (crates/god-engine-core/src/calibration.rs:325-326). La rama 13 sí escala por semi-intervalos (lib.rs:6392-6396).

**Impacto.** El corto es imposible si lift ≥ base, es decir, con un gen largo ≥ 0,5 + base. Con los genes por defecto (lift 0,0698) y base 0,18 es alcanzable (p < 0,1102), pero exige una divergencia normalizada 4,55 veces la del largo. Sólo ocurre con book_absent, fast_intent plano y atr_pct > tick_pct (lib.rs:5646): sobre todo en cintas sin libro (backtest y evolución). La puerta S-6 posterior sí lee el gen 23 en todo corto (lib.rs:7755-7816).

**Test que falla hoy.** Extraer la decisión de la rama 20 a una función pura (p, base, umbral_largo, umbral_corto) y añadir `fallback_sin_libro_emite_corto_espejo`: base 0,18, umbrales 0,75 y 0,25. Con p = 0,909 (div +0,889) sale Long. Esperado con p = 0,02 (div −0,889): Short. Actual: Flat, porque 0,02 < 0,18 − 0,25 es falso.

### LC-F1-20. cvd_veto_threshold tiene una banda fuera del rango del ratio

Media. Confirmado. Panel: unidades, n.º 3. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** puertas_del_continuo compara cvd_ratio = (buy − sell)/(buy + sell), que vive en [−1, 1] (lib.rs:1641-1648), con el gen cvd_veto_threshold (lib.rs:1651) en lib.rs:1663-1667. La banda del gen es [0,1; 2,0] (crates/quantum-arena/src/genome.rs:3053 y :3207), new_random muestrea 0,1..1,5 (crates/quantum-arena/src/genome.rs:952) y la mutación llega a 2,0 (crates/quantum-arena/src/genome.rs:1874). Con gen ≥ 1 el veto no puede dispararse. Es la misma clase de defecto que D-749 corrigió para el muro (crates/god-engine-core/src/calibration.rs:378-391).

**Impacto.** Hoy el genoma vivo lleva 0,5108 (config_dir/genomes/demo/active.json:73) y el veto está armado. El defecto afecta a genomas aleatorios o evolucionados con gen ≥ 1: un 35,7 % de new_random y el 52,6 % superior de la banda.

**Test que falla hoy.** `cvd_veto_con_gen_en_su_banda_puede_vetar`, con el patrón de los tests de lib.rs:8716-8757: cvd_veto_threshold 1,5, agg_buy_vol 0, agg_sell_vol 100 (ratio −1), muros a 0 y core.puertas_del_continuo(0, intencion_larga(), true, 0.50, 0.60, 0.5, 0.0). Esperado: Flat. Actual: Long.

### LC-F1-21. dynamic_atr_min es inerte al nacer

Media. Confirmado. Panel: unidades, n.º 1. Duplica V-010 (docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:95 y :259). Fase dueña: C-F3 (gate) y C-F7 (banda y muestreo del gen). Zona: Línea C.

**Afirmación.** El baseline vale 1e-6 con el comentario de un ATR por tick (crates/quantum-arena/src/genome.rs:720), new_random muestrea 1e-7..5e-6 (crates/quantum-arena/src/genome.rs:923) y la banda es [1e-7; 0,01] (crates/quantum-arena/src/genome.rs:3024 y :3178). El gate hace atr_ok = atr_pct > dynamic_atr_min (lib.rs:5513) con atr_pct = get_atr_pct() (lib.rs:3195), que devuelve v_t/last_price aunque su doc diga «percentage» (crates/god-engine-core/src/stateful_engine.rs:1176-1182). v_t tiene un suelo de 0,0010·precio en ticks (crates/god-engine-core/src/stateful_engine.rs:850-853 y :876-881) y de 0,0005·close en klines (crates/god-engine-core/src/stateful_engine.rs:1024-1028). Otros lectores usan escalas distintas: el suelo de gestión .max(0.001) (lib.rs:3196-3202), el máximo con 0,0005 del daemon sobre una σ de barras de 16 s (crates/evolution-engine/src/online_daemon.rs:1452) y la mutación de ±0,0002 recortada a [1e-4; 0,01] del binario evolution (src/bin/evolution.rs:279 y :318). El genoma versionado lleva 1,497e-4 (config_dir/genomes/demo/active.json:44). Corrección de un escéptico: el reinicio a 0,0012 de crates/evolution-engine/src/lib.rs:647 es código muerto y no cuenta como lector.

**Impacto.** El gate de volatilidad mínima no filtra nada con el baseline, con new_random ni con el genoma versionado; la decisión viva afectada es viable_para_entrar (lib.rs:5514). Corrección de los escépticos: la mutación es aditiva sobre el rango completo (crates/quantum-arena/src/genome.rs:1786-1790 y :1837) y en pocos pasos puede subir el gen por encima de 5e-4, donde el gate sí veta. Lo inerte son los puntos de partida, no la dimensión entera.

**Test que falla hoy.** `dynamic_atr_min_vive_en_la_unidad_del_atr_de_1m`: StatefulEngine con precio constante 100 durante 10 minutos; v_t parte de 0,005·precio (crates/god-engine-core/src/stateful_engine.rs:838-840) y decae ×0,85 por minuto, así que atr_pct ≈ 0,0018 (no el suelo de 0,0010). Se evalúa atr_ok = atr_pct > gen con el baseline y con 1 000 genomas de new_random. Esperado: algún valor de partida da atr_ok == false. Actual: true en todos, porque cualquier gen ≤ 5e-6 queda tres órdenes de magnitud por debajo.

### LC-F1-22. El sesgo spot es 20 veces más empinado que su contrato

Media. Confirmado. Panel: unidades, n.º 4. Fase dueña: C-F3. Zona: Línea C (el ml_prob compuesto también lo leen las ramas 1 a 10, que son de Línea E).

**Afirmación.** El comentario MOD2/7-028 declara «10 bps ⇒ ±0.05» y que el sesgo escala con la magnitud real del spread (lib.rs:4372-4379). El código hace spot_bias = (spread_bps/10).clamp(−0,05; 0,05) (lib.rs:4384-4385): pendiente de 0,1 por pb frente a 0,005 declarada, 20 veces más, y saturación en 0,5 pb. compose_ml_prob suma el sesgo sin escala (crates/god-engine-core/src/calibration.rs:285-289) y el resultado (lib.rs:4808) entra en puertas_del_continuo (lib.rs:6337-6345). El feed spot está cableado en vivo (god_engine.rs:1858). Corrección de los escépticos: MOD2/7-029 nombra el gate B3.18, y ese gate sí lee ml_prob_pure (lib.rs:7731-7753), así que no se viola la letra del contrato. Lo que ocurre es que el riesgo que quiso evitar reaparece en el veto F-009 de D-743 (ml_directional < −0,80, lib.rs:1631), que el sesgo puede cruzar sin opinión del modelo.

**Impacto.** Una diferencia de medio punto básico entre spot y futuro, ruido habitual de microestructura, mueve 5 puntos de probabilidad en la ponderación F-009. El paso de pb a probabilidad no tiene una conversión declarada.

**Test que falla hoy.** `sesgo_spot_sigue_su_contrato` (extrayendo `sesgo_spot(spot_mid, mid)`): mid 100 con spot 100,01 (1 pb) y 100,05 (5 pb). Esperado: 0,005 y 0,025. Actual: 0,05 y 0,05. El caso de guardia sobre el argumento de puertas_del_continuo se retira, porque pide un cambio de contrato y no repara uno vigente.

### LC-F1-23. Tres códigos REJ muertos figuran como vetos activos

Media (el revisor la puso alta; los escépticos la bajan porque no cambia ninguna orden). Confirmado. Panel: vetos, n.º 1. Ya figura en docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:166-167. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** REJ_MARGIN_INSUFFICIENT (7), REJ_OTROS (9) y REJ_SIN_EVIDENCIA (14) están definidos (crates/risk-engine/src/lib.rs:102, :104 y :113), y el registro los da por activos en V-RISK-003, V-TECH-003 y V-LOGIC-009 (crates/risk-engine/src/veto_registry.rs:89-100, :247-258 y :355-366). Ninguna llamada a rej (crates/risk-engine/src/lib.rs:161) los devuelve: lib.rs del risk-engine devuelve 14 códigos distintos de los 17 definidos. El commit f9ca4284 (Ω46) quitó la última llamada con REJ_SIN_EVIDENCIA sin tocar el registro, y el comentario de crates/risk-engine/src/lib.rs:1119-1121 todavía promete rechazar «POR FALTA DE EVIDENCIA». Se incumple la regla de mismo commit (docs/adr/ADR-0004-veto-registry-como-codigo.md:23-24).

**Impacto.** El registro presenta como compuertas vivas tres ranuras muertas. Correcciones de los escépticos: reject_report no muestra ceros, sino que oculta esas ranuras, porque filtra contadores > 0 (crates/risk-engine/src/lib.rs:196). La protección de margen sí existe: el veto del 95 % del margen libre (crates/risk-engine/src/envio.rs:96-98), que usan el host (god_engine.rs:4354-4378) y el replay (crates/backtest-engine/src/booktick_replay.rs:786), y el presupuesto del orquestador (crates/risk-engine/src/orchestrator.rs:241-268), que rechaza por REJ_ORCHESTRATOR. Lo muerto es la ranura 7 y su enlace con V-RISK-003.

**Test que falla hoy.** `registro_activo_implica_llamada_rej` (nuevo, crates/risk-engine/tests/veto_registry_source_contract.rs): el texto de include_str!("../src/lib.rs") cortado antes de #[cfg(test)] (crates/risk-engine/src/lib.rs:1514); D son las constantes `pub const (REJ_[A-Z_]+): usize` y R los argumentos de `rej(REJ_…)`. Para cada REJECT_NAMES[i], si la entrada que lo cubre está Activa, la constante i debe estar en R. Esperado: verde. Actual: rojo en i = 7, 9 y 14. El arreglo marca las tres entradas como Retirado con su linaje, o retira las ranuras, y registra VetoEnvio::Margen con su propia ubicación.

### LC-F1-24. El veto de ruina REJ_VIABILIDAD no tiene entrada propia

Media (bajada desde alta: el veto funciona y el defecto está en su ficha). Confirmado. Panel: vetos, n.º 2. Ya figura en docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:167. Fase dueña: C-F1. Zona: Línea C (la ficha D-755 cita el núcleo; si se reubica en lib.rs, avisar a Línea E).

**Afirmación.** REJ_VIABILIDAD (D-750, crates/risk-engine/src/lib.rs:108-110) rechaza cuando la orden mínima del símbolo, con el stop real, arriesga más que el tope de ruina (crates/risk-engine/src/lib.rs:1225-1247; crates/risk-engine/src/capital_regime.rs:241-246). Según la definición del propio registro es riesgo duro (crates/risk-engine/src/veto_registry.rs:20-21). El registro no lo menciona: el test de cobertura lo empareja por la subcadena «viabilidad» (crates/risk-engine/src/veto_registry.rs:671-673) con V-LOGIC-001, de clase Logica (crates/risk-engine/src/veto_registry.rs:137-148), que describe D-755, el filtro de horquilla del núcleo (lib.rs:5422-5470). El test que pinea V-LOGIC-001 no tiene asserts (crates/backtest-engine/tests/t1_cobertura_genetica.rs:89-117).

**Impacto.** Un veto de ruina figura como lógica que «se revisa periódicamente», y podría relajarse sin la decisión que exige el riesgo duro. orden_viable tiene pruebas unitarias (crates/risk-engine/src/capital_regime.rs:361-365), pero ningún contrato observa REJ_VIABILIDAD a través de evaluate_quantum_order.

**Test que falla hoy.** `rej_viabilidad_es_riesgo_duro_d750`: la entrada que cubre REJECT_NAMES[REJ_VIABILIDAD] tiene clase RiesgoDuro, su causa contiene "D-750" y el cuerpo de su test contiene "REJ_VIABILIDAD". Actual: devuelve V-LOGIC-001, de clase Logica, con causa D-755 y un test sin asserts.

### LC-F1-25. Los tests que pinean los vetos no observan su código de rechazo (REV-3 abierto)

Media. Confirmado. Panel: vetos, n.º 5. Fase dueña: C-F1. Zona: Línea C (REV-3 tenía como dueños a AGY/GLM, docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:248).

**Afirmación.** EntradaVeto no tiene campo de código (crates/risk-engine/src/veto_registry.rs:40-58), y el diente sólo busca el texto `fn nombre(` en el corpus (crates/risk-engine/src/veto_registry.rs:496-498). .agents/MEMORIA.md:8 afirma que el registro tiene «constantes REJ_*», pero el único REJ_ del archivo está en un comentario (crates/risk-engine/src/veto_registry.rs:61). Corrección de los escépticos: ninguno de los ocho riesgos duros activos pinea un test que observe su REJ_*, no sólo cuatro. V-RISK-001 apunta a una función de correlación (crates/risk-engine/tests/correlation_admission_contract.rs:402-421). V-RISK-002 apunta a una función pura (crates/risk-engine/tests/correlation_admission_contract.rs:381-397). V-RISK-003 compara dos corridas del replay (crates/backtest-engine/src/booktick_replay.rs:1129-1151). V-RISK-004 sólo mira las admitidas (crates/risk-engine/tests/leverage_admission_contract.rs:133-156). V-RISK-005 prueba una fórmula de q (crates/risk-engine/src/drawdown.rs:251-256). V-RISK-006 prueba una función pura (crates/risk-engine/src/correlation_guard.rs:1280-1299). V-LOGIC-005 sólo lee el fuente. V-LOGIC-006 y V-LOGIC-013 apuntan a una prueba de roundtrip_friction (crates/risk-engine/src/tp_sl.rs:825-842). REJ_EV no tiene ningún contrato de comportamiento.

**Impacto.** La regla «riesgo duro activo sin test es fallo de suite» (docs/adr/ADR-0004-veto-registry-como-codigo.md:20) se cumple de forma nominal.

**Test que falla hoy.** `cada_entrada_con_rej_pinea_un_test_que_lo_observa`, con un campo nuevo `codigo: Option<&'static str>`: para cada entrada con codigo = Some(c), el cuerpo de `fn <test>(` contiene c y además ultimo_rechazo() o REJECT_COUNTERS. Hoy no compila porque el campo no existe; con el emparejamiento actual por nombre falla en V-RISK-001 a V-RISK-006, V-LOGIC-005, V-LOGIC-006 y V-LOGIC-013. Para repinear V-RISK-005 al contrato qs_d3 hay que añadir crates/risk-engine/tests/qs_d2_d3_geometria_y_drawdown_contract.rs a corpus_contratos! (crates/risk-engine/src/veto_registry.rs:454-471).

### LC-F1-26. V-RISK-001 describe un rechazo que el código no hace

Media. Confirmado. Panel: vetos, n.º 3. Ya figura en docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:321 y docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:168. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** La ficha atribuye exposure0 a un riesgo_por_operacion en 0 o inválido (crates/risk-engine/src/veto_registry.rs:68-70). Con ese riesgo inválido, el código usa tope/8 y no rechaza (crates/risk-engine/src/lib.rs:646-651). REJ_EXPOSURE_ZERO salta cuando raw_exposure = dir·conf·kelly·capital vale 0 (crates/risk-engine/src/lib.rs:586-592). Como la confianza está acotada a [0,05; 0,98] (crates/risk-engine/src/lib.rs:585) y el capital > 0 se comprueba antes (crates/risk-engine/src/lib.rs:470), eso exige un Kelly nulo o negativo, que micro_kelly.max(0.0) deja en 0 (crates/risk-engine/src/lib.rs:574-577). El test pineado prueba una función de correlation_guard (crates/risk-engine/tests/correlation_admission_contract.rs:402-421).

**Impacto.** Quien lea el registro buscará la causa de los rechazos exposure0 en la EWMA de riesgo. Corrección de un escéptico: kelly_clamp_min está acotado a [0,001; 0,1] (crates/quantum-arena/src/genome.rs:928 y :1842), así que en vivo un Kelly nulo por falta de edge no se da. La ranura salta sobre todo cuando kelly_fraction o kelly_bootstrap_cold no son finitos, porque clamp_ruin devuelve 0 para un no finito (crates/risk-engine/src/ruin.rs:67-70). Un dato desconocido se cuenta como exposure0 y no como entrada inválida; la ficha corregida debe decirlo.

**Test que falla hoy.** `v_risk_001_su_test_observa_rej_exposure_zero` (de fuentes): el cuerpo de la función pineada por buscar("V-RISK-001").test contiene "REJ_EXPOSURE_ZERO". Actual: no lo contiene. Complemento de conducta, deducido por lectura y sin ejecutar: con el fixture de crates/risk-engine/tests/veto_logic_contracts.rs:15-55 y un kelly_fraction no finito, la regla del contrato pide ultimo_rechazo() == Some(REJ_INVALID_INPUT); hoy sería REJ_EXPOSURE_ZERO.

### LC-F1-27. V-RISK-005 conserva el umbral 0,85 que QS-D3 retiró

Media. Confirmado. Panel: vetos, n.º 4. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** La ficha dice que el umbral es lerp(dd_max_medido, 0.85, micro_w) (crates/risk-engine/src/veto_registry.rs:116). Desde QS-D3 el código usa lerp(dd_max_medido, d*, micro_w).min(d*) (crates/risk-engine/src/lib.rs:328-343, en concreto :336-341), con d* = 1 − 0,05^(1/3) ≈ 0,632 (crates/risk-engine/src/drawdown.rs:193-197, constantes en :170 y :173). El commit 249d2e88 no tocó el registro, y el comentario del código también quedó obsoleto (crates/risk-engine/src/lib.rs:291-306). El test pineado (crates/risk-engine/src/drawdown.rs:251-256) sólo prueba q_perdida_cartera. Sí observan REJ_DRAWDOWN el contrato qs_d3 (crates/risk-engine/tests/qs_d2_d3_geometria_y_drawdown_contract.rs:123-156) y crates/risk-engine/tests/drawdown_portfolio_contract.rs:9-12.

**Impacto.** La ficha de un veto de riesgo duro da un umbral que dejaría caer de 13 a 1,95 USD, cuando el código corta en 4,79 USD. Incumple la regla de mismo commit (docs/adr/ADR-0004-veto-registry-como-codigo.md:23-24), y el diente que promete el módulo (crates/risk-engine/src/veto_registry.rs:10-11) no lo detectó. El veto vivo es correcto; la deriva está en el registro.

**Test que falla hoy.** `v_risk_005_cita_el_umbral_vigente`: la causa de buscar("V-RISK-005") no contiene "0.85"; si el lib.rs del risk-engine contiene drawdown_de_falsacion, la causa o la fuente del umbral lo nombran (o dicen "d*"); y el cuerpo del test pineado contiene "REJ_DRAWDOWN". Actual: falla en las tres aserciones.

### LC-F1-28. V-LOGIC-013 y V-LOGIC-014 citan genes que no existen

Media. Confirmado. Panel: vetos, n.º 7. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** La fuente del umbral de V-LOGIC-013 dice «gen (gate_margin)» (crates/risk-engine/src/veto_registry.rs:323) y sus datos no citan la probabilidad (crates/risk-engine/src/veto_registry.rs:324). gate_margin sólo es un argumento de línea de órdenes de train_forest (src/bin/train_forest.rs:2085-2092 y :2122). El gen real es ev_fee_multiplier (crates/quantum-arena/src/config.rs:120; crates/quantum-arena/src/genome.rs:213), acotado a [1,05 + 0,20·escasez; 1,80] (crates/risk-engine/src/lib.rs:1170-1175), y la decisión depende de p_ganar (crates/risk-engine/src/lib.rs:1137-1157) y de los literales de sonda 0,55 y 0,45 (crates/risk-engine/src/lib.rs:1184-1194). Un escéptico añade V-LOGIC-014: cita «gen (max_acceptable_fee_pct)» (crates/risk-engine/src/veto_registry.rs:335), que es una variable local (crates/risk-engine/src/lib.rs:1198); el gen es max_fee_pct (crates/quantum-arena/src/genome.rs:166).

**Impacto.** Quien busque el gen para calibrar el veto EV no lo encontrará, y la dependencia del calibrador, causa del posible estado absorbente, no consta en la ficha.

**Test que falla hoy.** `los_genes_citados_existen`: para cada entrada cuya fuente del umbral contenga «gen (X)» con X un identificador (se excluye el texto libre de crates/risk-engine/src/veto_registry.rs:263), el genoma contiene `pub X:`. Actual: falla con gate_margin (V-LOGIC-013) y con max_acceptable_fee_pct (V-LOGIC-014).

### LC-F1-29. target-cpu=native en la configuración del repo (D-663)

Media. Confirmado. Panel: versión y construcción, n.º 5. No es nuevo: es D-663 (INFORME_FORENSE_MAESTRO.md:5136), abierto en docs/HOJA_DE_RUTA_CIMIENTOS_2026-10-01.md:129. Fase dueña: C-F1 (contrato de construcción); los scripts son de C-F9 y los flags de .github/workflows/unit-suites.yml, de Línea E (va como pedido). Zona: Línea C.

**Afirmación.** .cargo/config.toml:2 fija target-cpu=native en toda compilación. El oráculo T-1 documenta que native emitió AVX512-FP16 en una VM sin esa extensión y lo sustituye por x86-64-v2 con el triple gnu (scripts/t1_oraculo.sh:41-44), pero sólo cuando corre en Linux. build.ps1:5, build.sh:9, LAUNCHER.bat:53 y LAUNCH_SHADOW_MAINNET.bat:10 exportan RUSTFLAGS con native, que reemplaza a build.rustflags. LAUNCH_PRODUCTION.bat:33 compila con el native de config.toml. La CI corre sin RUSTFLAGS (.github/workflows/unit-suites.yml:55-62).

**Impacto.** Un binario o una caché compilados en una máquina pueden abortar con instrucción ilegal en otra, y ya ocurrió. «T-1 pasa» no está atado al binario que opera con dinero. El triple msvc de producción es una inferencia del archivo rustc-ice, no una declaración del build de despliegue.

**Test que falla hoy.** `g0_rustflags_del_repo_portables` (de fuentes, crate raíz): .cargo/config.toml y los lanzadores (build.ps1, build.sh, LAUNCHER.bat, LAUNCH_SHADOW_MAINNET.bat) no contienen "target-cpu=native" y usan la misma línea base de CPU que scripts/t1_oraculo.sh. Actual: falla en .cargo/config.toml:2. Corrección de un escéptico: t1_oraculo.sh usa x86-64-v2 y D-663 prescribe x86-64-v3 (INFORME_FORENSE_MAESTRO.md:5148); hay que fijar una y alinear las dos.

### LC-F1-30. El gen de drawdown tiene dos lecturas

Media (el revisor la puso alta; se baja porque con 13 USD el veto del risk-engine queda igual con o sin riesgo medido). Disputado. Panel: unidades, n.º 2. Fase dueña: C-F4. Zona: Línea C.

**Afirmación.** La cabecera del módulo dice que el gen tiene «una sola lectura», la confianza, α = 1 − gen (crates/risk-engine/src/drawdown.rs:32-36). drawdown_maximo devuelve gen_confianza.clamp(0,02; 0,95) cuando drawdown_compatible da None, lo que pasa con riesgo_por_operacion ≤ 0 (crates/risk-engine/src/drawdown.rs:82-84 y :112-120). Ese riesgo nace en 0 (crates/quantum-arena/src/state.rs:611) y sólo se escribe al validar una orden (crates/risk-engine/src/lib.rs:1434-1444). Con q = 0,5, r = 0,005 y gen 0,95, la lectura medida da 0,0582; la de respaldo da 0,95, que el host recorta a 0,6316 con .min(d_falsacion) (god_engine.rs:1733-1737; crates/risk-engine/src/drawdown.rs:193-196).

**Impacto.** Tras un reinicio, el cortacircuitos del host admite hasta un 63 % de caída en lugar de un 6 %. Corrección de un escéptico: en el veto del risk-engine con 13 USD, micro_w = 1 (crates/risk-engine/src/capital_regime.rs:94-98) y el umbral queda en 0,6316 con o sin riesgo medido (crates/risk-engine/src/lib.rs:340-341).

**Objeción del escéptico.** La doble lectura es la decisión documentada D-744b: el doc de drawdown_maximo dice que, sin riesgo medido, rige el gen leído como fracción de caída (crates/risk-engine/src/drawdown.rs:98-111), y un test la fija (crates/risk-engine/src/drawdown.rs:229-231). El respaldo de 0,632 es la cota que QS-D3 declara admisible. Lo verificable es que la cabecera contradice esa excepción; cambiar la política es decisión del dueño de D-744b y QS-D3.

**Test que falla hoy.** `drawdown_tiene_una_sola_semantica_documentada` (guarda de fuentes): la cabecera de drawdown.rs no afirma «una sola lectura» mientras exista el respaldo de crates/risk-engine/src/drawdown.rs:112-120; o bien drawdown_maximo(0.0, 0.5, g) devuelve un estado desconocido que obligue al llamador a decidir. Actual: la cabecera lo afirma y drawdown_maximo(0.0, 0.5, 0.95) == 0,95. Si se cambia la política, el test d744b debe cambiar en el mismo commit.

### LC-F1-31. Las EWMA del z de CVD y del campo espectral tienen memoria en eventos

Media. Disputado. Panel: unidades, n.º 6. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** La doctrina #660 del mismo struct pide memoria en tiempo físico (crates/quantum-arena/src/state.rs:253-256; el flujo agregado usa τ = 60 s en crates/quantum-arena/src/state.rs:730-772). La EWMA del z de CVD usa ALPHA_CVD = 0,002 por evento (lib.rs:5106-5134, constante en lib.rs:5117) y el campo espectral usa ALPHA_CAMPO = 0,0033 por intención (lib.rs:6973-6984). Además sólo se actualizan en eventos que pasan la guarda de entradas bloqueadas (lib.rs:4888-4890).

**Impacto.** La misma magnitud tiene memoria de minutos en calma y de segundos en un pico, y la paridad entre vivo y backtest depende de la tasa de eventos del tape.

**Objeción del escéptico.** La memoria en eventos está declarada en el propio sitio («memoria ~500 eventos», lib.rs:5107; «α≈1/300 intents», lib.rs:6976), así que es falso que nadie la declare, y ADR-0014 no enuncia una regla de memoria física entre sus seis principios. La paridad entre vivo y backtest es una preocupación legítima con precedente en D-615b (crates/god-engine-core/src/stateful_engine.rs:851-866), pero es decisión de diseño del dueño.

**Test que falla hoy.** `memoria_del_cvd_z_en_tiempo_fisico`: 120 s del mismo mercado sintético servidos a 1 evento/s y a 10 eventos/s, con entradas no bloqueadas. Esperado: el mismo cvd_ewma_peso en los dos casos. Actual: 1 − 0,998^120 = 0,2136 frente a 1 − 0,998^1200 = 0,9095.

### LC-F1-32. Un slot sin identidad se convierte en BTCUSDT dentro del núcleo

Media. Disputado. Panel: identidad, n.º 3. Fase dueña: C-F3. Zona: Línea C (bloque ML y consenso fuera de las ramas 1 a 10; confirmar con Línea E).

**Afirmación.** El comentario B3.18b dice «SIN FALLBACK a BTC» (lib.rs:4587). Si try_symbol no resuelve, el núcleo usa "BTCUSDT" para el bosque _MOTOR (lib.rs:4593-4595), el predictor _VOL (lib.rs:4622) y el NN sólo-BTC (lib.rs:4761). Si try_spec no resuelve, consulta el consenso con "BTCUSDT" (lib.rs:5355-5357 y :5364), y los votantes leen primero las claves del símbolo (crates/signal-engine/src/flow_impulse.rs:305; crates/signal-engine/src/renyi_tsallis_entropy.rs:247). Dentro del mismo process_tick_dual el slot recibe tres identidades: "" (lib.rs:3036-3038), try_symbol con BTC por defecto (lib.rs:4593) y try_spec con BTC por defecto (lib.rs:5355). ADR-0011 dice que try_symbol, try_spec y try_index son el mismo mapa (docs/adr/ADR-0011-identidad-canonica-del-simbolo.md:32-34).

**Impacto.** Es latente en vivo, porque el host registra un spec por símbolo del universo (god_engine.rs:928-932). Es alcanzable en binarios o tests cuyo universo sea más corto que los slots usados; entonces la moneda hereda el bosque, la base del modelo, el NN y el libro de BTC.

**Objeción del escéptico.** No llega a ninguna decisión viva ni a ningún binario de evidencia: es endurecimiento. Hay que vigilarlo al arreglar LC-F1-12 (si se crean slots fuera del universo) o LC-F1-02 (si try_spec devuelve None para specs no medidos), porque esos respaldos pasarían a ser alcanzables. La forma natural del arreglo es abortar el evento.

**Test que falla hoy.** `g1_slot_sin_identidad_no_hereda_el_bosque_de_btc` (crates/god-engine-core/tests): universo ["IDFBUSDT"] con su spec; NanoForest::store_global("BTCUSDT_MOTOR", bosque con base 0,30); swing_nn y scalp_forest a None; core.process_tick_dual(1, 100.0, 100.01, 5.0, 5.0, 10_000, &[0.0; 54], true, false). Esperado: get_for_coin_or(1, "ml_model_base", NaN) distinto de 0,30 (0,5 sin modelo, o NaN si se rechaza el evento). Actual: 0,30, la base del bosque de BTC.

### LC-F1-33. Un ack 2xx sin identidad se acepta como respuesta conocida

Media. Disputado. Panel: identidad, n.º 10. Deuda OPEN ya registrada en crates/execution-engine/tests/execution_open_diagnostics.rs:10-14. Fase dueña: C-F1 (identidad de una orden) y C-F5 (ejecutor). Zona: Línea C.

**Afirmación.** Todos los campos de OrderAck llevan default (crates/execution-engine/src/order_types.rs:86-115) y parse_order_body es un from_str directo (crates/execution-engine/src/order_types.rs:189-192), así que `{}` es un ack válido. client.rs llama a Ok(ack) «respuesta conocida» (crates/execution-engine/src/client.rs:305-312). El ejecutor lo aplica al registro bajo la clave "" y omite el aviso si status está vacío (crates/execution-engine/src/executor.rs:2140-2149; crates/execution-engine/src/order_registry.rs:296-303). Nada comprueba clientOrderId ni symbol.

**Impacto.** Un cuerpo truncado que sigue siendo JSON válido, o el ack de otra orden, cuenta como éxito sin activar AMBIGUOUS. No se verificó si ORDER_TRADE_UPDATE corrige después la entrada real, así que no se afirma que la orden quede sin seguimiento. El punto con más consecuencias es el aplanado, que cuenta un ack sin identidad como cierre (crates/execution-engine/src/executor.rs:900-905).

**Objeción del escéptico.** Ya está registrado como deuda OPEN con un testigo que fija la conducta actual, y no tiene un disparador realista.

**Test que falla hoy.** `ack_sin_identidad_es_ambiguo`: parse_order_body("{}"). Esperado: Err, que execute_order_payload_typed convierte en AMBIGUOUS; variante: un ack con un clientOrderId distinto del enviado también es AMBIGUOUS. Actual: Ok con order_id 0 y symbol "". El arreglo invierte el testigo OPEN en lugar de añadir un test paralelo.

### LC-F1-34. El registro convierte ±Inf y NaN en 0

Media. Disputado. Panel: desconocido, n.º 3. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** Parameter::new, set_value y restore_from_snapshot guardan 0,0 para cualquier valor no finito (crates/omniscient-registry/src/lib.rs:53-57, :74 y :346), y un test lo consagra (crates/omniscient-registry/src/lib.rs:443-446). R6-B15 diseña t½ = +∞ con θ ≤ 1e-4 para que la guarda «rechace de forma fail-closed» (crates/strategy-core/src/vecm_arbitrage.rs:287-293 y :569-578). El núcleo publica esa t½ (lib.rs:5202 y :5205), el registro guarda 0 y la guarda !(t½ ≤ 2τ*) (crates/strategy-core/src/stat_arb.rs:405-408) no veta. La rama de «clave venenosa» (crates/strategy-core/src/stat_arb.rs:411-412) es inalcanzable. Los publicadores tienen que usar máscaras numéricas (crates/god-engine-core/src/evidence_publication.rs:3-5 y :52-55).

**Impacto.** Con θ degenerada la guarda deja pasar la serie que no revierte. Hoy el voto suele quedar en 0, porque con la varianza de reserva 1,0 (crates/strategy-core/src/vecm_arbitrage.rs:297-303) la z es diminuta; el daño es latente. Un escéptico precisa que la máscara −1 de IC, aunque indistinguible de un IC válido, sólo endurece la puerta actual.

**Objeción del escéptico.** La normalización a 0 es un diseño documentado (FIX #1540, crates/omniscient-registry/src/lib.rs:345-346) sobre el que se construye la disciplina de publicación. Propone reformularlo, con severidad baja, contra el publicador del núcleo (lib.rs:5202-5205): con el motor llevado a θ ≤ 1e-4 tras 10 actualizaciones, la t½ publicada debería superar 2·τ*; hoy vale 0.

**Test que falla hoy.** (1) `statarb_half_life_infinita_publicada_veta` (tests de crates/strategy-core/src/stat_arb.rs): reg.set("statarb_ou_zscore", 2.5), reg.set("statarb_half_life_ms", f64::INFINITY) y reg.set("dominant_tau_ms", 60_000.0). Esperado: evaluate() == 0,0. Actual: ≈ −0,583. (2) `set_no_finito_no_se_vuelve_cero`: r.set("k", f64::INFINITY). Esperado: get_value_fast("k") distinto de Some(0.0). Actual: Some(0.0). El test de crates/omniscient-registry/src/lib.rs:443-446 cambia con el arreglo.

### LC-F1-35. El sello de tiempo del registro se fija al crear la clave y no se renueva

Media. Disputado. Panel: desconocido, n.º 4. Fase dueña: C-F1 (registro), con pedidos a Qoder/GLM (publicador de contagio, crates/god-engine-core/src/contagion_publisher.rs) y a Codex (lector en crates/risk-engine/src/correlation_guard.rs). Zona: Línea C para el registro.

**Afirmación.** Parameter.timestamp es un i64 no atómico (crates/omniscient-registry/src/lib.rs:48) que sólo se asigna en new con el reloj de pared (crates/omniscient-registry/src/lib.rs:65). set_value no lo toca (crates/omniscient-registry/src/lib.rs:73-76), y set y register_or_update reutilizan el Parameter (crates/omniscient-registry/src/lib.rs:162-178). Su único lector es take_snapshot (crates/omniscient-registry/src/lib.rs:335), sin llamadores fuera del crate. publish_contagion_roles sale sin escribir si quedan menos de 2 monedas o si la matriz falla (crates/god-engine-core/src/contagion_publisher.rs:47-49 y :54-64), y sólo escribe las monedas filtradas (crates/god-engine-core/src/contagion_publisher.rs:67-77). El rol viejo sigue recortando confianza (lib.rs:7092-7100) y el curl viejo sigue inflando la ρ sistémica (crates/risk-engine/src/correlation_guard.rs:696).

**Impacto.** Un rol de seguidor medido hace horas sigue descontando la confianza de una moneda. Cada consumidor reinventa la caducidad por su cuenta (lib.rs:1777-1800, :5188 y :5212; crates/god-engine-core/src/entes_consejo.rs:3-9), y el reloj de pared tampoco serviría en replay.

**Objeción del escéptico.** Ningún contrato presenta ese campo como sello de frescura. La edad en el registro es una mejora de diseño; si se presenta de nuevo, debe ir como pedido a Qoder/GLM con un test sobre el publicador.

**Test que falla hoy.** (1) `set_renueva_el_sello_de_tiempo`: r.set("k", 1.0), leer t0, dormir 5 ms, r.set("k", 2.0) y leer t1. Esperado: t1 > t0, o el sello de evento que pase el escritor. Actual: t1 == t0. (2) Pedido a Qoder/GLM: una primera llamada a publish_contagion_roles con dos monedas calientes escribe un rol r ≠ 0 en c0; una segunda con la matriz en None (por ejemplo, con un timestamp repetido en una serie). Esperado: c0:hawkes_contagion_net_role vuelve a la ausencia declarada. Actual: conserva r.

### LC-F1-36. El Kelly que dimensiona una orden sale del cierre anterior

Media. Disputado. Panel: horizonte, n.º 3. Fase dueña: C-F4. Zona: Línea C.

**Afirmación.** El único consumidor vivo de kelly_at_tau en el núcleo está en el cierre (lib.rs:4114-4117). Se evalúa en la τ de la operación que se cierra (lib.rs:3229-3242), con la persistencia de spec.dominant_tau_ms de ese instante (lib.rs:4121-4130). El resultado se guarda como escalar por moneda (lib.rs:4143-4157), y el riesgo lo usa para la orden siguiente, sea cual sea su τ (crates/risk-engine/src/lib.rs:387-390 y :411-415), corregido sólo por risk_normalizer (crates/risk-engine/src/lib.rs:527-534).

**Impacto.** Con los anclas por defecto, W = 0,6, PF = 2 y capital igual a la base, la fracción guardada vale 0,060 tras un cierre de 30 s y 0,108 tras uno de 12 h (×1,8). Corrección de un escéptico: con persistencia −0,8 la cifra es 0,048, no 0,054.

**Objeción del escéptico.** D-745 (crates/risk-engine/src/lib.rs:41-48) pide que «el Kelly del cierre» razone sobre el horizonte de la posición que se cierra, y eso es lo que hace el código. No hay violación literal: hay una suposición oculta (una curva que depende de τ se reutiliza para otra τ), que debe plantearse como propuesta de diseño.

**Test que falla hoy.** `kelly_de_la_orden_no_depende_de_la_tau_del_cierre_anterior` (risk-engine): dos GlobalArena idénticas (spec registrado, capital 1 000 USD, trade_count 10, win_rate 0,6, PF ≈ 2) que sólo difieren en metrics.kelly_fraction, 0,060 frente a 0,108; la misma intención larga con confianza 0,8 y expected_duration_ms 43 200 000. Esperado: el mismo volume_usd. Actual: proporcional a 0,060 frente a 0,108 antes de topes. Un escéptico pide fijar capital y base para que el núcleo esté en modo supervivencia.

### LC-F1-37. win_probability: el contrato de types.rs contradice a sus lectores

Media. Disputado. Panel: espejo, n.º 2. Fase dueña: C-F1 (contrato del tipo). Zona: Línea C.

**Afirmación.** types.rs dice que 0,0 significa «sin calibrar», que el consumidor usa confidence y que la selección nunca lee el campo (crates/strategy-core/src/types.rs:33-36). El núcleo adjunta siempre calibrate_at (lib.rs:7511-7519), que con calibradores vacíos es la identidad (crates/god-engine-core/src/calibration.rs:19-21 y :130-136). La puerta EV, que es de selección, lee el campo (crates/risk-engine/src/lib.rs:1137-1157) y rechaza con REJ_EV (crates/risk-engine/src/lib.rs:1180-1194). El 0,0 cae a confidence en el apalancamiento (crates/risk-engine/src/leverage_matrix.rs:116-120) pero no en la puerta EV. El comentario del núcleo «sólo alimenta el Kelly» (lib.rs:7497-7503) también quedó obsoleto desde D-751 (crates/risk-engine/src/lib.rs:1105-1121).

**Impacto.** El documento del tipo induce a error a cualquier consumidor nuevo. Corrección de un escéptico: en arranque frío total (trade_count 0) la rama None no vetaría nada, así que la puntuación cruda no hace la puerta más permisiva. El caso con consecuencias es otro: tras un periodo sin datos, calibrate_at mezcla hacia la puntuación cruda con semivida de 12 h (crates/god-engine-core/src/calibration.rs:138-141), y con trade_count ≥ 5 la puerta EV aplica p ≈ puntuación sin el respaldo de sonda.

**Objeción del escéptico.** La identidad en frío es diseño declarado (D-619) y la mezcla es XLIV-5 (crates/god-engine-core/src/calibration.rs:114-129). Sólo queda la contradicción documental, y el test del revisor no aplica.

**Test que falla hoy.** `win_probability_doc_coincide_con_sus_lectores` (guarda de fuentes): crates/strategy-core/src/types.rs no afirma «La selección nunca la lee» mientras el riesgo rechace con REJ_EV según win_probability, y el centinela de «sin calibrar» es el mismo en leverage_matrix y en la puerta EV. Actual: falla en las dos aserciones.

### LC-F1-38. La puerta S-6 no exige la misma probabilidad de ganar a los dos lados

Media. Disputado. Panel: espejo, n.º 4. Fase dueña: C-F3. Zona: Línea C (puerta de entrada fuera de las ramas 1 a 10). Misma causa que LC-F1-43.

**Afirmación.** La puerta S-6 exige a los dos lados la misma divergencia normalizada (lib.rs:7800-7816), con lifts por lado (lib.rs:7759-7760) y genes por defecto 0,5698 y 0,4302 (crates/quantum-arena/src/genome.rs:615 y :715-716). Su comentario promete «idéntico rigor estadístico relativo» (lib.rs:7797-7799). Con la etiqueta de train_forest (TP 0,36 % y SL 0,18 %, src/bin/train_forest.rs:1391-1392) y el modelo de primer toque del repo (crates/risk-engine/src/tp_sl.rs:469-489), el largo necesita P(ganar) 0,426 en su umbral y el corto pasa con 0,382.

**Impacto.** Con la geometría browniana, la selección queda sesgada hacia el corto.

**Objeción del escéptico.** Con el mapa del propio código (p_win_directional), los dos umbrales dan la misma P(ganar), 0,4264, así que la puerta es simétrica en el modelo lineal del repo y «relativo» se cumple al pie de la letra. La asimetría sólo aparece con el mapa browniano. Al corregir LC-F1-43 la puerta debe heredar el arreglo, sin parchearse aparte.

**Test que falla hoy.** `puerta_s6_exige_la_misma_probabilidad_de_ganar_a_ambos_lados`: base 1/3, genes por defecto (required_edge 0,1396 con agree 0 y H 0,5), un Short con ml_now 0,27 (div −0,19) y el oráculo de primer toque con tp 0,0036, sl 0,0018 y σ 0,0018. P_corto(0,27) ≈ 0,401 < 0,4264. Esperado: el corto se rechaza. Actual: se admite (0,19 ≥ 0,1396).

### LC-F1-39. train_forest documenta que el corto «consume 1−p»

Media. Disputado. Panel: espejo, n.º 7. Fase dueña: C-F7, coordinado con Línea E (escritura de modelos de train_forest). Zona: Línea C para el etiquetado.

**Afirmación.** El comentario de la triple barrera dice que el corto consume 1 − p al servir y que P(SL largo primero) es un «proxy honesto» de la hipótesis corta (src/bin/train_forest.rs:1684-1688). Con TP 0,36 % y SL 0,18 % (src/bin/train_forest.rs:1391-1392), «SL largo primero» vale 2/3 en un mercado neutro y «gana el corto» vale 1/3: no es el mismo evento. El servicio ya no usa 1 − p (crates/god-engine-core/src/calibration.rs:356-357). El entrenador sólo etiqueta el evento largo (src/bin/train_forest.rs:1689-1708), y el gate de promoción sólo certifica sobre él (src/bin/train_forest.rs:2129).

**Impacto.** Documental y de certificación: el modelo promovido no tiene evidencia fuera de muestra sobre el evento que hace ganar a un corto.

**Objeción del escéptico.** El comentario dice «proxy», es decir, reconoce que no es el mismo evento, y ningún consumidor vivo usa 1 − p desde K-26. Basta con quitar «consume 1−p en serve»; una etiqueta corta toca la zona de Línea E.

**Test que falla hoy.** `train_forest_no_documenta_1_menos_p` (guarda de fuentes): src/bin/train_forest.rs no contiene «consume 1−p». Actual: aparece en src/bin/train_forest.rs:1684-1688. El test de diseño del revisor (una función de etiqueta con el lado como argumento) queda como pedido a Línea E.

### LC-F1-40. El venue por defecto y el formato del booleano cambian según el binario

Media. Disputado. Panel: versión y construcción, n.º 3. Fase dueña: C-F1 (función única de venue). Zona: Línea C.

**Afirmación.** El host lee USE_TESTNET con unwrap_or_default() == "true" (god_engine.rs:1029-1033 y :1172-1176), así que su ausencia es mainnet. El forense usa unwrap_or("true") (src/bin/audit_forensic_backtest.rs:61-65), así que su ausencia es testnet. income_report y system_health miran testnet salvo --live (src/bin/income_report.rs:43; src/bin/system_health.rs:19), y benchmark usa BINANCE_IS_TESTNET (src/bin/benchmark.rs:47-50). Corrección de los escépticos: son 16 lecturas de env::var("USE_TESTNET") en 10 archivos, más la lista de crates/quantum-arena/src/paths.rs:22. Todos aceptan "TRUE"; sólo "1" falla, aunque otras banderas del proyecto aceptan "1" (crates/evolution-engine/src/online_daemon.rs:740-742; crates/god-engine-core/src/darwin.rs:671-673). template.env no declara USE_TESTNET (template.env:1-8).

**Impacto.** USE_TESTNET=1 con claves de los dos venues apunta a mainnet sin aviso. system_health atenúa la confusión, porque imprime el entorno en su cabecera (src/bin/system_health.rs:34-41).

**Objeción del escéptico.** system_health e income_report documentan su defecto testnet y la bandera --live (src/bin/system_health.rs:10-11; src/bin/income_report.rs:5-7), y BEST_WS_ENDPOINT (god_engine.rs:1026) es una sobrescritura explícita del operador. El núcleo defendible es estrecho: "1" se interpreta como mainnet sin avisar de un valor no reconocido.

**Test que falla hoy.** `g0_venue_tiene_un_solo_lector` (de fuentes, crate raíz): apariciones de env::var("USE_TESTNET") fuera de la función canónica. Esperado: 0. Actual: 16. Tabla para la función canónica venue_desde_env: "1", "TRUE" y "true" dan testnet, y un valor no reconocido da error. Hoy la expresión de god_engine.rs:1172-1176 convierte "1" en mainnet.

---

## Severidad baja

### LC-F1-41. El daemon calcula R-múltiplos con una unidad equivocada

Baja. Confirmado. Panel: unidades, n.º 5 (los dos escépticos bajan la severidad de media a baja). Fase dueña: C-F7. Zona: Línea C.

**Afirmación.** R7-R6-IG-1 convierte el retorno sobre el capital (crates/evolution-engine/src/online_daemon.rs:420-427 y :1109) en R-múltiplos dividiéndolo por riesgo_por_operacion (crates/evolution-engine/src/online_daemon.rs:1125-1144), que es una fracción del capital (crates/quantum-arena/src/state.rs:515-520) que el riesgo calcula como volumen·L·SL/capital (crates/risk-engine/src/lib.rs:1434-1445). Si no está medido, cae a config.scalp_sl_base, que es la distancia del stop en fracción de precio, sin margen ni apalancamiento, y después a un literal 0,02 sin unidad declarada. El cociente se recorta a [−1, 1] antes de alimentar el e-proceso de Ville, así que la unidad equivocada no se nota en el rango.

**Impacto.** Corrección de los escépticos: el respaldo sólo actúa en los cierres que ocurren entre un reinicio y la primera orden validada del proceso, porque riesgo_por_operacion pasa a ser positivo con la primera orden (crates/risk-engine/src/lib.rs:1434-1445) y actualizar_riesgo_ewma ignora los valores no positivos (crates/risk-engine/src/drawdown.rs:156-165). Además el e-proceso sólo existe tras una promoción en el mismo proceso (crates/evolution-engine/src/online_daemon.rs:855-856). En esa ventana, por ejemplo con posiciones adoptadas al arrancar y una promoción temprana, una pérdida al stop se observa como −0,5 R en lugar de −1 R. Un factor positivo no cambia el signo de la evidencia de Ville: cambia su potencia y el recorte en ±1.

**Test que falla hoy.** `unidad_de_riesgo_no_mezcla_precio_y_capital`, tras extraer unidad_de_riesgo(riesgo_medido, scalp_sl_base). Entrada: riesgo_por_operacion 0, scalp_sl_base 0,002 y una pérdida exactamente al stop con capital 13 USD, margen 1,3 USD, L = 5 y SL = 0,002, es decir pnl −0,013 USD y ret −0,001. Esperado: None (riesgo desconocido, no se observa) o −1,0 R si la unidad se reconstruye como margen·L·SL/capital = 0,001. Actual: −0,5.

### LC-F1-42. new_baseline nace fuera de sus bandas con la comisión viva

Baja. Confirmado. Panel: unidades, n.º 9 (los escépticos bajan la severidad a baja). Fase dueña: C-F7. Zona: Línea C.

**Afirmación.** new_baseline se presenta como derivación con «Cero Hardcoding Heurístico» (crates/quantum-arena/src/genome.rs:614) y obtiene magnitudes con unidad multiplicando la comisión adimensional: scalp_obi_threshold, leverage_cap, regime_duration_ms, flash_crash_jump_pct, maker_only_capital_threshold, hawkes_volume_norm y base_duration_ms (crates/quantum-arena/src/genome.rs:657, :730, :735, :751, :806, :828 y :829). N-06 exige que el baseline sea estable bajo round-trip y pase el gate de promoción (crates/quantum-arena/src/genome_store.rs:788-821), pero sólo se prueba con taker 0,0005. El host lo construye con la comisión viva (god_engine.rs:1264), y su propio comentario dice que la cuenta paga 0,04 % (god_engine.rs:1238-1242). Con taker 0,0004, base_duration_ms vale 60/0,0004 = 150 000 ms, por encima de su cota superior de 120 000 (crates/quantum-arena/src/genome.rs:3109 y :3263 son la cota inferior y la superior).

**Impacto.** Corrección de los escépticos: load_or_baseline devuelve new_baseline sin pasar por validate ni por from_vector (crates/quantum-arena/src/genome.rs:604), así que el motor operaría con 150 000 ms fuera de banda. El recorte (crates/quantum-arena/src/genome.rs:2886) sólo ocurre si el genoma pasa por la evolución, y el rechazo de validate (crates/quantum-arena/src/genome_store.rs:248-268) sólo al intentar promoverlo. El caso vivo es estrecho, porque el baseline sólo se usa sin envelope, espejo legacy, quantum_champion.json ni genesis_genome.json (crates/quantum-arena/src/genome.rs:566-604). flash_crash_jump_pct = 0,12 está dentro de su banda y el «15 %» es sólo un comentario del host (god_engine.rs:3357), así que sale del test. Derivar unidades de la comisión es una crítica de diseño; el defecto verificable es la violación de N-06.

**Test que falla hoy.** `baseline_con_comision_viva_nace_dentro_de_sus_bandas`. Entrada: SuperGenotype::new_baseline(0.0002, 0.0004); to_vector() comparado con get_lower_bounds() y get_upper_bounds(). Esperado: todos los genes dentro de su banda, como N-06 exige con 0,0005. Actual: el índice 112 (base_duration_ms) vale 150 000 > 120 000.

### LC-F1-43. p_win_directional no es la probabilidad de ganar del corto con barreras asimétricas

Baja. Confirmado. Panel: espejo, n.º 3 (un escéptico baja la severidad de media a baja). Fase dueña: C-F2. Zona: Línea C. Misma causa que LC-F1-38.

**Afirmación.** El doc de p_win_directional promete «la probabilidad calibrada de que la orden gane, según su dirección» (crates/god-engine-core/src/calibration.rs:344-346). Para el corto aplica un reflejo lineal por semi-intervalos alrededor de la base (crates/god-engine-core/src/calibration.rs:364-374), que sólo coincide con 1 − p cuando la base es 0,5. Bajo el modelo de primer toque que el propio repositorio usa (crates/risk-engine/src/tp_sl.rs:469-489), P_corto(μ) = P_largo(−μ), y el reflejo sólo coincide con ese valor en p ∈ {0, base, 1}. Los lectores existen: la aceptación conformal del corto (lib.rs:4843-4854), la actualización conformal del cierre (lib.rs:3990-3994) y el td_error del aprendiz online (lib.rs:4323-4325). conformal_accept_short apaga los votos cortos de ConformalReversionFilterEngine (crates/signal-engine/src/conformal_reversion_filter.rs:195-196).

**Impacto.** El conformal del corto juzga una probabilidad inflada (+0,11 en el ejemplo) y aprende de ella al cierre. El consumidor directo es un filtro de una sola estrategia del tensor y dos lazos de aprendizaje, por eso la severidad es baja. Corrección de los escépticos: el «valor verdadero» depende del modelo. El hallazgo vale bajo el modelo browniano de horizonte infinito con base = sl/(tp+sl); con la etiqueta real de train_forest la base viva no es 1/3, así que el arreglo debe recibir la geometría TP/SL como entrada en lugar de deducirla de la base. El contrato existente (crates/god-engine-core/tests/conformal_direction_symmetry_contract.rs:34-71) sólo fija los puntos ancla, y el arreglo debe seguir cumpliéndolo.

**Test que falla hoy.** `p_win_directional_coincide_con_primer_toque_browniano` (tests de calibration.rs). Entrada: tp 0,0036, sl 0,0018, σ 0,0018 y μ −0,0004; p_largo = 1 − probabilidad_tocar_sl_antes_de_tp(tp, sl, μ, σ) ≈ 0,2003; base = 1 − probabilidad_tocar_sl_antes_de_tp(tp, sl, 0, σ) = 1/3. Esperado: p_win_directional(false, p_largo, base) ≈ 1 − probabilidad_tocar_sl_antes_de_tp(tp, sl, +0,0004, σ) ≈ 0,4873 (±1e-3). Actual: 0,5994.

### LC-F1-44. La sombra de resonancia de una moneda usa la varianza de ruido de otra

Baja. Confirmado. Panel: desconocido, n.º 5 (un escéptico baja la severidad de media a baja). Fase dueña: C-F3. Zona: sombras espectrales del núcleo; según lo leído están fuera de las ramas 1 a 10, y Línea E debe confirmarlo. Misma causa raíz que LC-F1-06.

**Afirmación.** La sombra de resonancia estocástica lee get_value_or("microstructure_noise_variance", 0.05), la clave GLOBAL (lib.rs:2153-2157), aunque su comentario #660 dice que lee la clave que el núcleo escribe al tick para no quedar «sorda a la volatilidad real» (lib.rs:2147-2152). set_reg publica esa clave por moneda y por símbolo además del global (lib.rs:4558-4561 y :4423-4428), así que el global es el valor de la última moneda que pasó por process_tick_dual. El motor escalar equivalente sí lee por ámbito (crates/signal-engine/src/stochastic_resonance.rs:140). voto_res entra en votos_espectrales (lib.rs:2427-2440).

**Impacto.** Con monedas intercaladas, la ganancia de resonancia de una moneda se calcula con la σ de otra. Corrección de un escéptico: el efecto es pequeño, porque los desplazamientos son momentum_z recortados a ±10 (lib.rs:2048-2054) y la ganancia sech sólo cambia de forma apreciable con |x| ≲ 0,3; con |x| ≈ 1 la diferencia es de unos 2 %. El mismo escéptico señala un defecto mayor en el mismo lector: compara una fracción de ATR con una z, el desajuste que #650 corrigió para el motor de choque (lib.rs:2124-2129), y, con una sola moneda, lee el valor del evento anterior, porque corre antes de process_tick_dual (lib.rs:2740 frente a :4557).

**Test que falla hoy.** `sombra_resonancia_invariante_al_ruido_global` (crates/god-engine-core/tests): dos núcleos con arenas propias reciben la misma secuencia de eventos de la moneda 0; en el segundo, antes de cada process_event, se hace arena.registry.set("microstructure_noise_variance", 0.01), lo que dejaría otra moneda. Esperado: el valor final de c0:sombra_res_consenso igual en los dos. Actual: difiere. Alternativa mínima, con el patrón de contrato de fuente del repo (lib.rs:9325-9329): el fuente no contiene `get_value_or("microstructure_noise_variance"`.

### LC-F1-45. Los líderes BTC y ETH se reconocen por prefijo del símbolo

Baja. Confirmado. Panel: identidad, n.º 7 (un escéptico baja la severidad a baja mientras LC-F1-09 siga abierto). Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** El núcleo toma el símbolo de try_spec con "" por defecto y decide el líder con starts_with("BTC") y starts_with("ETH") (lib.rs:3036-3040). lead_lag tiene un único buffer por líder (crates/feature-engine/src/lead_lag.rs:73-91), y momentum_lider devuelve 0,6·OFI_btc + 0,4·OFI_eth con la última muestra (crates/feature-engine/src/lead_lag.rs:197-204). El núcleo publica lead_lag_divergence y leader_momentum (lib.rs:4458-4459). Cualquier símbolo con esos prefijos (BTCDOMUSDT, ETHFIUSDT, ETHWUSDT) escribe en el buffer del líder, y el escáner los admite (crates/data-pipeline/src/dynamic_ranker.rs:119-130). Un slot sin spec pierde su papel de líder sin aviso.

**Impacto.** Hoy es latente: la lista del bootloader no tiene otros símbolos con esos prefijos (crates/god-engine-core/src/bootloader.rs:320-347). Cuando la propuesta de universo se aplique al reiniciar (LC-F1-09, como exige ADR-0011), ETHFIUSDT junto a ETHUSDT mezclaría dos instrumentos en el buffer del líder ETH, y todas las alts recibirían leader_momentum y lead_lag_divergence falsos. El arreglo debe ir antes o junto con el de LC-F1-09. La vía del forense es poco realista, porque sólo hay modelos para ATOM, BNB y NEAR.

**Test que falla hoy.** `g1_lider_por_identidad_exacta` (crates/god-engine-core/tests, binario propio): universo ["BTCDOMUSDT"] con spec; dos core.process_tick_dual(0, …) a 10 000 y 10 500 ms, con la cantidad del bid pasando de 5,0 a 9,0 para que la OFI no sea cero. Esperado: registry.get_for_coin_or(0, "leader_momentum", NaN) == 0,0, porque no hay BTCUSDT. Actual: 0,6·OFI, distinto de cero.

### LC-F1-46. La proyección espectral vota con el espectro de otra moneda en el primer evento

Baja. Confirmado. Panel: desconocido, n.º 7. Fase dueña: C-F3. Zona: Línea C. Misma causa raíz que LC-F1-06.

**Afirmación.** El motor declara que las proyecciones son por moneda, que una lectura global «mezclaría símbolos» y que un espectro sin observar no proyecta (D-742) (crates/signal-engine/src/proyeccion_espectral.rs:17-19 y :59-60). Pero lee con get_scoped_parameter (crates/signal-engine/src/proyeccion_espectral.rs:41-46), que cae al global (crates/omniscient-registry/src/lib.rs:279). El núcleo sólo publica espectral_* si sum_w > 0 (lib.rs:4543). En el primer evento de una moneda el tiempo transcurrido es 0, la masa es 0 y todos los pesos son 0 (crates/quantum-arena/src/temporal_spectrum.rs:1409-1416), y sum_w también es 0 mientras ewma_dev_vol ≤ 1e-12 en todas las escalas (crates/quantum-arena/src/temporal_spectrum.rs:1418-1422). La moneda lee entonces el global que dejó otra.

**Impacto.** En el primer evento de trade de cada moneda, y mientras dure esa ventana, el voto espectral vale señal × concentración de otra moneda; puede llegar a 1,0, porque la señal está en [−1, 1] y la concentración en [0, 1]. Un evento de depth sin trade no llega, porque el return de entries_blocked ocurre antes (lib.rs:4888-4890). No está demostrado que llegue a una orden. Los tests del módulo escriben sólo el global y leen la moneda 0 (crates/signal-engine/src/proyeccion_espectral.rs:96, :107 y :127): dependen de la caída al global y cambiarán con el arreglo.

**Test que falla hoy.** `proyeccion_sin_espectro_propio_se_abstiene` (tests de crates/signal-engine/src/proyeccion_espectral.rs): set global y set_for_coin(0, …) con espectral_masa_resuelta 0,9, espectral_senal_proyectada 0,9 y espectral_concentracion 0,9. Esperado: evaluate_for_coin(1, "ETHUSDT") == 0,0. Actual: 0,81.

### LC-F1-47. INITIAL_CAPITAL="0.0" se lee como 0 USD y no como «dinámico»

Baja. Confirmado. Panel: versión y construcción, n.º 4 (confirmado), fusionado con unidades, n.º 10 (disputado). Los escépticos bajan la severidad a baja. Fase dueña: C-F1 (resolvedor del capital) y C-F9 (herramientas). Zona: Línea C.

**Afirmación.** La plantilla dice que INITIAL_CAPITAL="0.0" «activa la extracción dinámica automática desde la API» (template.env:4), y README.md:45 manda copiarla a .env. Ningún lector lo implementa: god_engine no lee la variable y toma el capital del balance (god_engine.rs:1213-1228). El forense carga .env con dotenvy (src/bin/audit_forensic_backtest.rs:59), parsea 0,0 (src/bin/audit_forensic_backtest.rs:120-123), imprime «Capital Base: $0.0000» (:125), construye el arena con 0 (:141) y emite FORENSIC_JSON (:1041-1046). walkforward_evolver lanza el forense sin fijar INITIAL_CAPITAL (src/bin/walkforward_evolver.rs:119-124). Los lectores no coinciden entre sí: las funciones FFI de backtest-engine toman el 0 (crates/backtest-engine/src/lib.rs:588-591) y run_vectorized_hybrid devuelve Ok con ceros (crates/backtest-engine/src/vectorized.rs:15-21), mientras evolver y benchmark convierten cualquier valor ≤ 0 en 13 (src/bin/evolver.rs:42-49; src/bin/benchmark.rs:19-23) y train_forest en 5 (src/bin/train_forest.rs:1127-1132). TRADING_SYMBOL (template.env:3) no tiene ningún lector.

**Impacto.** Con un .env copiado de la plantilla, el forense mide un sistema sin capital y devuelve ROI 0 sin error. Corrección de los escépticos: el walk-forward no promueve a ciegas, sino que falla cerrado con «ningún finalista es viable» (src/bin/walkforward_evolver.rs:338-366), aunque la causa queda oculta tras un motivo genérico. No llega a ninguna decisión viva. run_vectorized_hybrid y polars_evolver no tienen uso en producción (crates/evolution-engine/src/polars_evolver.rs:1). Con LAUNCHER.bat el valor conserva comillas y comentario, el parse falla y los lectores con respaldo usan 13 (LAUNCHER.bat:76-81). Que el forense haga «cero operaciones» no se comprobó.

**Test que falla hoy.** `capital_desconocido_no_es_cero_usd`, tras extraer una función común capital_inicial_desde_env a la que deleguen el forense y los demás lectores. Entrada: Some("0.0"). Esperado: Err («0.0 es el centinela dinámico; el capital es desconocido»). Actual: el forense obtiene 0,0 (src/bin/audit_forensic_backtest.rs:120-123), y los demás 13 o 5. Variante de integración: audit_forensic_backtest con INITIAL_CAPITAL=0.0 sobre un tape mínimo debe salir con código distinto de 0 y sin FORENSIC_JSON; hoy imprime FORENSIC_JSON con initial_capital igual a 0.000000.

### LC-F1-48. El OI normalizado ancla en $100k y no en $100M

Baja. Confirmado. Panel: unidades, n.º 7. Fase dueña: C-F5. Zona: Línea C.

**Afirmación.** El comentario CERT-M1-M02 dice que el OI en USD se normaliza en escala log «[$100M … $10B]» (god_engine.rs:2533-2534), y el doc del campo dice «normalizado log contra $100M» (crates/quantum-arena/src/state.rs:237-241). El código usa (ln(oi_usd) − 11,5129)/(23,0259 − 11,5129) (god_engine.rs:2543-2545). ln(1e5) = 11,5129, así que el ancla inferior real es $100k; para $100M (ln ≈ 18,42) el valor es 0,60 y no 0. Los números ln del propio comentario («ln 11.51→23.03») coinciden con el código y no con «$100M», así que el comentario se contradice a sí mismo.

**Impacto.** El consejo lee el campo (crates/metacortex-engine/src/consejo_seniors.rs:535-539) y lo convierte en p_oi ∈ [0,80, 1] (crates/metacortex-engine/src/consejo_seniors.rs:566-571): para $100M sale ≈ 0,90 en lugar de 1,0. El impacto es acotado. Corrección de los escépticos: con precio desconocido el campo queda en 0, que el consejo trata como «sin modulación» (p_oi = 1), así que esa parte no causa daño y sale del test.

**Test que falla hoy.** `oi_norm_respeta_sus_anclas`, tras extraer normalizar_oi_usd(oi_usd). Entrada: 1e8 USD y 1e10 USD. Esperado: 0 y 1. Actual: 0,60 y 1.

### LC-F1-49. La lista negra de testnet del escáner también actúa en mainnet

Baja. Confirmado. Panel: identidad, n.º 15. Fase dueña: C-F6. Zona: Línea C (data-pipeline).

**Afirmación.** La lista testnet_blacklist se describe como basura de testnet (crates/data-pipeline/src/dynamic_ranker.rs:106-118), pero se aplica sin mirar is_testnet (crates/data-pipeline/src/dynamic_ranker.rs:129), a diferencia de min_vol (:94). Por eso en mainnet excluye símbolos reales como OPUSDT, que ocupa el slot 21 del bootloader (crates/god-engine-core/src/bootloader.rs:342). El apalancamiento máximo del spec sale de una lista de nombres (crates/data-pipeline/src/dynamic_ranker.rs:236-240) y llega al riesgo como techo del exchange (crates/risk-engine/src/lib.rs:492). Los filtros ilegibles caen en valores por defecto (crates/data-pipeline/src/dynamic_ranker.rs:205-208).

**Impacto.** OPUSDT nunca recibe un spec de exchangeInfo aunque el demonio refresque, y conserva el inventado (LC-F1-02). Hoy el impacto es pequeño porque el refresco casi nunca ocurre; crece al arreglar LC-F1-02 y LC-F1-09. En demo la exclusión es la intención declarada, así que el defecto es sólo de producción.

**Test que falla hoy.** Tras extraer la regla a is_sane_symbol(s, is_testnet): is_sane_symbol("OPUSDT", false) == true. Actual: la regla equivalente (crates/data-pipeline/src/dynamic_ranker.rs:119-130) rechaza OPUSDT en cualquier entorno.

### LC-F1-50. V-LOGIC-005 llama «no absorbente» a un latch permanente

Baja. Confirmado. Panel: vetos, n.º 8 (un escéptico baja la severidad de media a baja). Fase dueña: C-F1. Zona: Línea C para la ficha; el disparador latch_degradation es de Codex.

**Afirmación.** El contrato sol_a1_kill_switch_tiene_contrato_y_no_es_absorbente sólo comprueba que el núcleo contiene los textos kill_switch_active.load y kill_switch_active.store (crates/risk-engine/src/veto_registry.rs:540-553), y su propio doc admite que el latch «congela la operación para siempre (estado absorbente)» (crates/risk-engine/src/veto_registry.rs:531-533). Ningún archivo escribe false en el latch: las escrituras son store(true) en el núcleo (lib.rs:7145) y en el host (god_engine.rs:1802, :4757 y :4946-4950), y latch_degradation sólo lo activa (crates/evolution-engine/src/return_evidence.rs:148-153), con llamadas en crates/evolution-engine/src/online_daemon.rs:1222, :1235, :1324 y :1343. La deuda habla de la «prueba de runtime del rearme (histeresis)» como si el rearme existiera (crates/risk-engine/src/veto_registry.rs:200-205), y ADR-0015 hace lo mismo (docs/adr/ADR-0015-semantica-del-kill-switch.md:86-87). fuente_umbral dice «literal» (crates/risk-engine/src/veto_registry.rs:193), pero el disparo por drawdown usa drawdown_maximo(...).min(d*) (god_engine.rs:1729-1737) y el de latencia un gen (god_engine.rs:1747-1751).

**Impacto.** La ficha del veto más peligroso afirma una propiedad que el código no tiene. ADR-0015 ya marca la permanencia como «Abierto», así que el comportamiento es conocido: el defecto es que el registro y el nombre del test dicen lo contrario. No cambia ninguna decisión viva. Los disparos del daemon (zona de Codex) no constan en los datos de la ficha.

**Test que falla hoy.** `v_logic_005_declara_latch_permanente_si_no_hay_rearme`: sobre corpus_contratos!() más include_str! de src/bin/god_engine.rs, si ningún archivo contiene "kill_switch_active.store(false", la deuda de buscar("V-LOGIC-005") debe contener «permanente» y el nombre del test no debe contener «no_es_absorbente». Actual: falla en las dos aserciones. Pasa tras renombrar el test, corregir la deuda y cambiar fuente_umbral a «medido (drawdown_maximo·d*) + gen (latencia)». Pedido a Codex: documentar latch_degradation como disparador en los datos de la ficha.

### LC-F1-51. Los dientes de cobertura del registro de vetos son nominales

Baja. Confirmado. Panel: vetos, n.º 10. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** El módulo promete que sus tests detectan la deriva entre el registro y el código (crates/risk-engine/src/veto_registry.rs:9-11). La cobertura directa acepta cualquier entrada cuyo nombre o datos contengan el nombre de la ranura (crates/risk-engine/src/veto_registry.rs:671-673): «spec» casa con seis entradas (V-RISK-002, V-RISK-004, V-LOGIC-008, V-TECH-002, V-LOGIC-015 y V-TECH-003, porque «espectral» y «específica» contienen «spec»). La cobertura inversa acepta cualquier entrada cuyos datos digan «core» o «host» (:693-697). Ninguna de las dos mira si la ranura se devuelve en algún sitio. retirados_conservan_linaje promete revisar el código fuente (:636-637), pero sólo lee el registro dentro de un if let que pasa en silencio si falta la entrada (:638-640).

**Impacto.** Los tests de cobertura dan una falsa garantía: borrar V-TECH-002 deja el test en verde. Corrección de los escépticos: la laxitud del emparejamiento está declarada en el propio test (crates/risk-engine/src/veto_registry.rs:668-670), así que el defecto es la falsa garantía y el comentario incumplido. Que el registro no tenga consumidor en ejecución es su diseño (ADR-0004) y sale del impacto. buscar() sí comprueba el estado de una entrada contra el registro, pero no consulta el código fuente como promete.

**Test que falla hoy.** `cobertura_detecta_entrada_borrada` (mutación): copia de REGISTRO_VETOS sin V-TECH-002 y la cobertura extraída a una función pura cubre(nombre, registro) -> usize. Esperado: 0 entradas para «spec». Actual: 5. Pasa tras emparejar exactamente por un campo de código (véase el contrato, apartado de vetos). Además, retirados_conservan_linaje debe usar buscar(...).expect(...) para V-LOGIC-003 y V-LOGIC-004.

### LC-F1-52. template.env con rellenos que nadie reconoce como ausentes

Baja. Confirmado. Panel: versión y construcción, n.º 7. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** Los valores de relleno de la plantilla (template.env:1-2 y :7-8) no están vacíos, y los lectores sólo comprueban que no lo estén: god_engine sólo avisa con la clave vacía (god_engine.rs:1168-1170) y TelegramBot::new acepta cualquier token no vacío (crates/telemetry-server/src/telegram_bot.rs:13-17). TRADING_SYMBOL (template.env:3) no tiene lector. El formato con comillas y comentario en línea sólo lo entiende dotenvy: los parsers propios de income_report y system_health hacen split_once('=') y trim sin quitar comillas (src/bin/income_report.rs:48-55; src/bin/system_health.rs:22-29), y el for /f de LAUNCH_PRODUCTION.bat conserva comillas y comentario (LAUNCH_PRODUCTION.bat:19-21). Como dotenvy no sobrescribe variables existentes (god_engine.rs:784 y el comentario de :824-826), god_engine recibe entonces las claves entre comillas. LAUNCH_PRODUCTION.bat copia .env.template, que no existe en el repositorio (LAUNCH_PRODUCTION.bat:11), y compara con un relleno distinto del de la plantilla (:23).

**Impacto.** Falla cerrado. Corrección de los escépticos: LAUNCH_PRODUCTION.bat no pasa --force-live (LAUNCH_PRODUCTION.bat:46), así que corre en papel, y con una clave inválida el bucle de balance registra el error de la API cada 5 s (god_engine.rs:1222-1224); lo que nunca dice es que la clave sea un relleno. Un escéptico dejó sin comprobar que las alertas fallen en silencio; aquí se comprobó: send_message devuelve Err (crates/telemetry-server/src/telegram_bot.rs:44-55) y el llamador lo descarta con `let _ =` (crates/telemetry-server/src/lib.rs:157). El operador cree elegir símbolos con TRADING_SYMBOL y no tiene efecto.

**Test que falla hoy.** `g0_template_env_coherente` (crate raíz): (1) toda clave de template.env tiene al menos un lector en src/ o crates/; hoy TRADING_SYMBOL tiene 0. (2) El parser de income_report, extraído a una función, y dotenvy::from_path producen el mismo mapa a partir de template.env; hoy BINANCE_API_KEY vale "\"TU_LLAVE_PUBLICA_AQUI\"" en uno y "TU_LLAVE_PUBLICA_AQUI" en el otro. (3) El proveedor de credenciales clasifica "TU_LLAVE_PUBLICA_AQUI" como ausente; hoy sólo se comprueba que esté vacía.

### LC-F1-53. El doc de la intermitencia conserva la fórmula K41

Baja. Confirmado. Panel: unidades, n.º 12. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** El doc del campo dice «Intermitencia χ = (1−ζ3)⁺» respecto a K41 (crates/quantum-arena/src/state.rs:231-232), y el comentario del núcleo lo repite (lib.rs:4921-4924). El cálculo vigente desde XLIV-6 es χ = ((3/2)ζ₂ − ζ₃)⁺ recortado a [0, 1] (crates/quantum-arena/src/temporal_spectrum.rs:1345-1346 y :1374), con su justificación en :1360-1367.

**Impacto.** Desajuste de contrato, no de cálculo. Corrección de los escépticos: con la fórmula vigente χ = 0 para cualquier ζ(p) lineal, sea Brown o K41, y el rango sigue siendo [0, 1], así que la frase «χ=0 (K41) no cambia nada» (lib.rs:4924) sigue siendo cierta. Lo que está mal es la fórmula y la referencia que declara el doc; quien cablee un consumidor leyéndolo interpretará la magnitud con otra fórmula.

**Test que falla hoy.** `doc_intermitencia_coincide_con_la_formula` (guarda de fuentes): crates/quantum-arena/src/state.rs y crates/god-engine-core/src/lib.rs no contienen «(1−ζ3)⁺». Actual: aparece en crates/quantum-arena/src/state.rs:231 y en lib.rs:4921.

### LC-F1-54. TGM_DATA_DIR ignorado y live_fees.json fuera de la separación por entorno

Baja. Disputado. Panel: identidad, n.º 14, fusionado con versión y construcción, n.º 6 (propuesto como media). Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** D-07 dice que todos los paths de datos deben usar data_root(), que honra TGM_DATA_DIR, para no depender del directorio de trabajo (src/env_manager.rs:9-18; crates/quantum-arena/src/paths.rs:1-2 y :5-10). Pero EnvManager::data_path delega en env_data_path (src/env_manager.rs:49-53), y env_data_path_for escribe "data/" a mano (crates/quantum-arena/src/paths.rs:43-45). god_engine resuelve por esa vía dynamic_config (god_engine.rs:938-939), state.redb (:1354) y evolution_ledger.redb (:2097). live_fees.json se escribe con el literal "data/live_fees.json" (god_engine.rs:1247-1250) y se lee con el mismo literal (crates/execution-engine/src/simulator.rs:12-13), común a testnet y mainnet.

**Impacto.** Con TGM_DATA_DIR definido, el estado por entorno se escribiría relativo al directorio de trabajo, lejos de la raíz configurada. Es una trampa latente: en todas las configuraciones versionadas el comportamiento actual coincide con el esperado.

**Objeción del escéptico.** Ningún lanzador, script, Dockerfile ni docker-compose define TGM_DATA_DIR, y EnvManager::data_path nunca la honró (antes de CL-48 usaba resolve_path("data")). CL-48 fijó la regla literal con un test (crates/quantum-arena/src/paths.rs:51-55). El único lector de live_fees.json es SimulatedExecutor, sin uso fuera de su archivo, así que no hay paridad B3.11 que contaminar. D-07 ya se incumple en muchas rutas literales "data/...", y arreglar sólo data_path repartiría el estado entre dos raíces. Si se arregla, el cambio va en env_data_path (no en la función pura con su test CL-48) o en el doc D-07. El escéptico señala que el defecto concreto de la zona es el de dynamic_config, que recoge LC-F1-09.

**Test que falla hoy.** `d07_estado_respeta_tgm_data_dir` (crates/quantum-arena/tests, serializado): TGM_DATA_DIR=/tmp/tg sin variables de demo. Esperado: env_data_path("kelly_envelope.json") empieza por "/tmp/tg/". Actual: "data/prod/kelly_envelope.json". Segundo test, de fuentes: god_engine.rs y crates/execution-engine/src/simulator.rs no contienen el literal "data/live_fees.json". Actual: aparece en god_engine.rs:1248 y en crates/execution-engine/src/simulator.rs:13.

### LC-F1-55. El mismo ATR se lee como rango en el stop y como σ en el deslizamiento

Baja. Disputado. Panel: unidades, n.º 8. Fase dueña: C-F4. Zona: Línea C.

**Afirmación.** D-754c declara que el ATR de 1 minuto es un rango medio de √(8/π)·σ y multiplica el pronóstico σ por RANGO_PARKINSON (crates/risk-engine/src/tp_sl.rs:138-142 y :345-354). D-747 usa el mismo atr_ratio como σ del paso de 60 s y lo escala por √(lat/60000) (crates/risk-engine/src/tp_sl.rs:144-175 y :176-190), y un test fija que con 60 s de latencia el deslizamiento es el ATR (:788). El núcleo trata el ATR como rango (crates/god-engine-core/src/diffusion.rs:57-63). roundtrip_friction consume ese deslizamiento (crates/risk-engine/src/tp_sl.rs:279-282), y la física lo llama en crates/god-engine-core/src/reality_physics.rs:106-107. Las dos lecturas difieren en √(8/π) ≈ 1,596.

**Impacto.** El término de latencia de la fricción se cobra con una dispersión que el mismo módulo declara en otra escala, hasta 1,6 veces.

**Objeción del escéptico.** Usar atr_ratio como dispersión de 60 s es el diseño documentado de D-747, fijado a propósito por un test (crates/risk-engine/src/tp_sl.rs:786-788), y su contrato es la paridad entre gate, host y física, que se cumple. D-754c convierte el σ porque el gen sl_atr_multiplier está en múltiplos del rango, no porque obligue a reconvertir la fricción. El valor esperado del revisor impone un modelo: el desplazamiento absoluto esperado sería √(2/π)·σ, más o menos la mitad del ATR. El impacto es despreciable: con 13,6 ms y ATR 0,2 % el término vale unos 0,30 pb por lado (crates/god-engine-core/src/reality_physics.rs:99-101), y el factor cambia de 0,1 a 0,2 pb frente a 4 pb de comisión. Propone aclarar el doc de D-747 sin cambiar el cálculo.

**Test que falla hoy.** Guarda de fuentes `d747_declara_la_escala_del_atr`: el doc de latency_slippage_pct no llama σ(τ_ref) a atr_ratio sin declarar que es el rango de D-754c. Actual: lo hace (crates/risk-engine/src/tp_sl.rs:166-170). La alternativa numérica del revisor, latency_slippage_pct(0.002, 60000.0) ≈ 0,001253 frente a 0,002 hoy, impone un modelo y queda como decisión de diseño.

### LC-F1-56. Los muros L2 se documentan como volumen acumulado y guardan el mejor nivel

Baja. Disputado. Panel: unidades, n.º 11. Fase dueña: C-F1 (doc del campo). Zona: Línea C.

**Afirmación.** El doc define l2_bid_wall y l2_ask_wall como «Volumen acumulado en bids/asks» (crates/quantum-arena/src/state.rs:277-279). El host toma el primer nivel (god_engine.rs:3156-3157), asigna dbq = bq (:3185) y llama a update_l2_depth (:3349). El entrenador (src/bin/train_forest.rs:1476) y el forense (src/bin/audit_forensic_backtest.rs:462 y :667) escriben también cantidades de un nivel. Sólo BinanceStreamer suma niveles (crates/data-pipeline/src/parser.rs:178 y :234), y no se usa fuera de su crate. La puerta D-749 lee estos campos (lib.rs:1649-1650).

**Impacto.** El veto de muro y los rasgos derivados leen el desequilibrio del nivel 1 bajo el nombre de profundidad L2, sin unidad declarada.

**Objeción del escéptico.** Los tres escritores guardan coherentemente la cantidad del mejor nivel, los tapes sólo traen bq/aq y la paridad D-753 impide sumar cinco niveles en vivo sin rehacerlos. El gen de D-749 compara una razón entre dos magnitudes medidas siempre igual. El arreglo razonable es corregir el doc, y con eso el test del revisor (esperado 15) seguiría fallando.

**Test que falla hoy.** Guarda de fuentes `muro_l2_doc_coincide_con_el_escritor`: crates/quantum-arena/src/state.rs no dice «Volumen acumulado» para l2_bid_wall y l2_ask_wall mientras el host escriba el primer nivel. Actual: lo dice en crates/quantum-arena/src/state.rs:278-279.

### LC-F1-57. El registro de specs no canoniza ni deduplica

Baja. Disputado. Panel: identidad, n.º 11. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** ADR-0011 fija una grafía canónica en mayúsculas ASCII mediante simbolo_canonico (docs/adr/ADR-0011-identidad-canonica-del-simbolo.md:21-28; crates/quantum-arena/src/symbols.rs:10-12). update_registry guarda el símbolo tal como llega y no deduplica en la primera carga (crates/quantum-arena/src/symbol_registry.rs:86-89 y :98). try_index y get_official_binance_spec usan to_uppercase de Unicode (crates/quantum-arena/src/symbol_registry.rs:149-151 y :168). update_dynamic_universe canoniza pero tampoco deduplica (crates/quantum-arena/src/symbols.rs:19-22). try_spec devuelve el símbolo en su grafía original, que el núcleo usa para las claves con ámbito (lib.rs:3036).

**Impacto.** Un spec en minúsculas haría que try_spec(i).symbol difiera de try_symbol(i) y que las claves {SYM}_* se escriban en otra grafía. Un universo con un símbolo repetido deja un slot al que no se llega por nombre.

**Objeción del escéptico.** Ningún productor viola el invariante, ni hoy ni tras los otros arreglos: get_official_binance_spec pone mayúsculas, el forense y el entrenador también, el bootloader no tiene duplicados y merge_universe_with_hysteresis canoniza y evita duplicados (src/symbol_manager.rs:258-313). El contrato de identidad pasa con los productores reales (crates/quantum-arena/tests/identidad_simbolo_contract.rs:14-34). Es endurecimiento defensivo.

**Test que falla hoy.** `g1_registro_canoniza_y_deduplica` (crates/quantum-arena/tests, binario propio): con registro y universo vacíos, update_registry(vec![spec("idrcusdt"), spec("IDRCUSDT")]). Esperado: try_symbol(0) == Some("IDRCUSDT") y try_spec(1) es None. Actual: try_symbol(0) == Some("idrcusdt") y try_spec(1) es Some (duplicado).

### LC-F1-58. Los votantes de entropía y de impulso dejan al slot 0 leer el global

Baja. Disputado. Panel: identidad, n.º 12. Fase dueña: C-F3. Zona: Línea C (signal-engine). Misma causa raíz que LC-F1-06.

**Afirmación.** Los dos votantes neutralizan una moneda sin datos propios para no leer «la ÚLTIMA moneda que escribió el global», salvo el slot 0, que justifican como «BTC, escritor convencional del global» (crates/signal-engine/src/renyi_tsallis_entropy.rs:233-241 y :255; crates/signal-engine/src/flow_impulse.rs:291-299 y :317). El núcleo escribe el global desde todas las monedas con set_reg (lib.rs:4423-4425), así que la premisa es falsa, y ningún contrato dice que el slot 0 sea BTC. evaluate() rellena lo desconocido con valores fijos (crates/signal-engine/src/renyi_tsallis_entropy.rs:216-229; crates/signal-engine/src/flow_impulse.rs:286). Los tests fijan la excepción (crates/signal-engine/src/renyi_tsallis_entropy.rs:348-350; crates/signal-engine/src/flow_impulse.rs:435-437).

**Impacto.** En el arranque del slot 0, o con un símbolo mal resuelto, el voto del slot 0 usaría el libro de otra moneda.

**Objeción del escéptico.** Es una decisión documentada (MOD2/7-009) y fijada por tests, e inalcanzable en el camino del núcleo: las cuatro claves se publican sin condición con set_reg, que siempre escribe c{id}:clave aunque el símbolo esté vacío, y la consulta del consenso ocurre después en el mismo tick. Los envoltorios legados con (0, "BTCUSDT") no tienen llamadores de producción. Sólo queda corregir el comentario como higiene.

**Test que falla hoy.** `mod2_7_009_slot_0_sin_datos_vota_neutral`: registro sólo con claves globales (tsallis_q_entropy 0,2 y order_book_imbalance −0,9) y evaluate_for_coin(0, "IDX0USDT") en los dos motores. Esperado: 0,0. Actual: menor que 0. Corrección de un escéptico: para FlowImpulseEngine hay que sembrar además hawkes_intensity 2,0 y order_flow_imbalance −0,8, como el test existente (crates/signal-engine/src/flow_impulse.rs:422-425), porque con hawkes en su valor por defecto el voto ya da 0. Hay que retirar las aserciones que fijan la excepción.

### LC-F1-59. El censo espectral depende del slot 0

Baja. Disputado. Panel: identidad, n.º 13. Fase dueña: C-F3. Zona: Línea C (signal-engine).

**Afirmación.** decisiones_espectrales cuenta las decisiones de cualquier moneda (crates/signal-engine/src/orchestrator.rs:384-387), pero el censo sólo se publica cuando evalúa el slot 0, y el denominador sólo cuenta los consensos del slot 0 (crates/signal-engine/src/orchestrator.rs:425-427). La fracción se calcula y se publica en :453-459.

**Impacto.** Con N monedas la fracción tiende a N veces la real y puede superar 1. Si el slot 0 no recibe eventos, el censo no se publica.

**Objeción del escéptico.** Es telemetría pura: qo_624_fraccion_espectral y qo_624_decisiones_espectrales no tienen lectores, sólo se escriben. No llega a ninguna orden, tamaño ni veto.

**Test que falla hoy.** `qo_624_fraccion_no_supera_uno`: publicar consenso_espectral_dominante 0,5 y consenso_espectral_tau 60 000 para la moneda 1, evaluar la moneda 1 diez veces y la moneda 0 una vez. Esperado: qo_624_fraccion_espectral ≤ 1. Actual: 10.

### LC-F1-60. total_commission suma comisiones en activos distintos

Baja. Disputado. Panel: identidad, n.º 17. Fase dueña: C-F5. Zona: Línea C.

**Afirmación.** OrderAck::total_commission suma el campo commission de todos los fills sin mirar commission_asset (crates/execution-engine/src/order_types.rs:141-153 y :51-52), y el test del módulo suma un fill en BNB (:233 y :240). Corrección de un escéptico: order_registry tiene el mismo defecto. apply_ack suma f.commission de todos los fills con abs() (crates/execution-engine/src/order_registry.rs:342-357), el camino WS suma u.commission sin mirar el activo (:421-431) y recompute_total_commission las combina (:124-126).

**Impacto.** La API pública devuelve una suma sin unidad.

**Objeción del escéptico.** Ni OrderAck::total_commission ni TrackedOrder.total_commission tienen lectores de producción, sólo tests (crates/execution-engine/src/reconciliation.rs:769-778 está dentro de mod tests). El defecto ya está registrado como deuda OPEN con su testigo, open_debt_commissions_mix_currencies_and_discard_rebates (crates/execution-engine/tests/execution_open_diagnostics.rs:17-41). No llega a ninguna decisión viva. Si se arregla, hay que invertir ese testigo y cubrir los dos sitios.

**Test que falla hoy.** `total_commission_no_mezcla_activos`: ack con un fill de 0,0002 BNB y otro de 0,01 USDT. Esperado: error o desglose por activo. Actual: 0,0102. El testigo OPEN existente ya lo fija.

### LC-F1-61. Quedan suposiciones inertes de slot 0 = BTC y slot 1 = ETH

Baja. Disputado. Panel: identidad, n.º 18. Fase dueña: C-F4, C-F6 y C-F3. Zona: Línea C.

**Afirmación.** Quedan cuatro identidades por posición sin efecto hoy. (a) vol_mult del riesgo usa coin_id == 0 como BTC, coin_id == 1 como ETH y coins[0] como referencia (crates/risk-engine/src/lib.rs:739, :744 y :754), pero sólo se pasa a calculate_dynamic_leverage (:989), que lo descarta (crates/risk-engine/src/leverage_matrix.rs:84). (b) spot_feed actualiza el spot de referencia con coin_id == 0 || symbol == "BTCUSDT" (crates/data-pipeline/src/spot_feed.rs:123-136), y ese valor no tiene efecto. (c) SymbolRankerEngine::run_daemon reescribe el universo vivo (crates/quantum-arena/src/symbol_ranker_engine.rs:39-47), cosa que ADR-0011 prohíbe, pero no tiene llamadores. (d) Los envoltorios legados del orquestador evalúan (0, "BTCUSDT") (crates/signal-engine/src/orchestrator.rs:332, :626 y :641), sin llamadores de producción.

**Impacto.** Ninguno medible hoy. Cada pieza es una trampa para quien la reactive.

**Objeción del escéptico.** El propio hallazgo admite que no hay efecto, y el caso vivo de identidad por posición ya está cubierto por LC-F1-03 y LC-F1-11. La guarda léxica del revisor prohibiría usos legítimos de índices y, aplicada a todo el núcleo, saltaría con los usos que sí tienen efecto y podría tocar las ramas 1 a 10 de Línea E. Puede quedar como lista de vigilancia.

**Test que falla hoy.** `g1_sin_identidad_por_posicion`, acotada a las cuatro piezas: la función de vol_mult de crates/risk-engine/src/lib.rs, crates/data-pipeline/src/spot_feed.rs y los envoltorios de crates/signal-engine/src/orchestrator.rs no usan `coin_id == 0`, `coin_id == 1` ni `coins[0]` como identidad. Actual: aparecen en crates/risk-engine/src/lib.rs:739, :744 y :754.

### LC-F1-62. La salud del feed arranca «sana» y no marca las caídas por error

Baja. Disputado. Panel: desconocido, n.º 8. Fase dueña: C-F1 (estado de tres valores) y C-F5 (reconexión del host). Zona: Línea C.

**Afirmación.** WATCHDOG_STALLED nace en false (crates/quantum-arena/src/feed_health.rs:15) y last_ws_latency_ms nace en 0 (crates/quantum-arena/src/state.rs:614), así que antes del primer evento el feed se declara no estancado. Sólo el timeout del watchdog llama a stall() (god_engine.rs:5110-5119). Los fallos de resolución (god_engine.rs:5039-5041), TCP (:5056-5059) y TLS (:5131-5134), y el error o cierre del stream (:5102-5109), reconectan sin marcarlo. Cualquier evento limpia el flag (god_engine.rs:3260-3265) en un único stream combinado (:1080-1081).

**Impacto.** El núcleo no evalúa entradas sin eventos. Corrección de un escéptico: el flag sí cambia una decisión combinado con la latencia, porque con stall y latencia mayor que la mitad del umbral el inmune da strike (god_engine.rs:1763-1767). Una caída larga por error, que no se marca, puede dar menos strikes que una por timeout.

**Objeción del escéptico.** El comportamiento coincide con el contrato documentado del puente: el watchdog marca y cualquier evento vivo limpia (crates/quantum-arena/src/feed_health.rs:1-11), y un solo átomo para todos los símbolos está declarado (:9-10). Nacer en «estancado» bloquearía las entradas en todo replay, backtest y test del núcleo, porque entries_blocked incluye is_stalled() (lib.rs:2973-2978). Esos arneses dependen hoy del false inicial (crates/backtest-engine/tests/t1_cobertura_genetica.rs:62), y train_forest llama a stall() a propósito (src/bin/train_forest.rs:1332). Lo defendible es la parte estrecha: las reconexiones por error no llaman a stall().

**Test que falla hoy.** El que proponen los escépticos: extraer la lógica de reconexión del host a una función con su test, `error_de_conexion_marca_el_feed`. Entrada: un error de conexión (DNS, TCP, TLS o cierre del stream). Esperado: is_stalled() == true. Actual: queda en false, porque esas ramas no llaman a stall(). El test del revisor (is_stalled() en un binario propio sin eventos) también falla hoy, pero su arreglo exige que todos los arneses declaren el feed vivo.

### LC-F1-63. Bandas Micro, Meso y Macro con tres particiones incompatibles

Baja. Disputado. Panel: horizonte, n.º 6. Fase dueña: C-F3 y C-F4. Zona: Línea C.

**Afirmación.** stateful_engine parte en menos de 60 s, menos de 30 min y el resto (crates/god-engine-core/src/stateful_engine.rs:269-278), y position.rs en hasta 60 s, hasta 15 min y el resto (crates/quantum-arena/src/position.rs:651-670 y :798-815). spectral_regime_name llama «MicroScalp» a τ = 0, que significa desconocida (crates/quantum-arena/src/position.rs:661-663), mientras is_micro_scalp exige τ > 0 (:674-677). Los arrays por banda (crates/god-engine-core/src/stateful_engine.rs:118-120) se escriben en cada cierre, pero sólo los leen can_open_at_tau y sus variantes (crates/god-engine-core/src/stateful_engine.rs:321-391), sin llamadores de producción. can_open_at_tau conserva el salto en 60 s, contra la regla continua de temporal_spectrum (crates/quantum-arena/src/temporal_spectrum.rs:105-107).

**Impacto.** Sin efecto vivo: estado muerto que se escribe en cada cierre, tres definiciones de «micro» y una etiqueta que convierte «sin τ» en «MicroScalp». Corrección de un escéptico: record_trade_outcome también actualiza fastband_loss_streak y last_exit_*, que can_open_position_ms sí lee en vivo (crates/god-engine-core/src/stateful_engine.rs:521-534); lo muerto son los arrays por banda. spectral_regime_name tampoco tiene llamadores de producción, así que la etiqueta no se emite ni en telemetría.

**Objeción del escéptico.** El escalón ya está registrado como limitación abierta (crates/god-engine-core/tests/stateful_open_diagnostics.rs:5-13), así que no es un hallazgo nuevo, y ninguna orden, tamaño ni veto depende de esto. Si se mantiene, va como limpieza: retirar la API muerta y los arrays sin lector, o corregir la etiqueta de τ = 0.

**Test que falla hoy.** `tau_desconocida_no_es_micro`: posición abierta con open_with_fee (τ = 0). Esperado: spectral_regime_name() distinto de "MicroScalp" (por ejemplo "SinTau"). Actual: "MicroScalp". Y `can_open_at_tau_es_continua_en_tau`, con StatefulEngine::new(), record_trade_outcome(59_999.0, true, true, 1, 1000), record_trade_outcome(3_600_000.0, false, true, 2, 1100) y current_ts 20 000. Esperado: can_open_at_tau(59_999.0, 10_000) == can_open_at_tau(60_000.0, 10_000), o la API retirada. Actual: false frente a true, como fija open_cooldown_has_a_hard_band_boundary.

### LC-F1-64. Alias del orquestador que la cabecera da por retirados

Baja. Disputado. Panel: horizonte, n.º 7. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** La cabecera U-ERR-2 da por retiradas evaluate_horizon_consensus, los alias de scalp y swing, evaluate_dual_consensus[_for_coin] y evaluate_consensus[_for_coin], y dice «Queda UNA superficie» (crates/signal-engine/src/orchestrator.rs:16-38). Todas existen (crates/signal-engine/src/orchestrator.rs:118, :308, :317, :625 y :640). Tras un return anticipado (:116-121), evaluate_horizon_consensus conserva un camino inalcanzable que calcula una vida de 1× a 10× base_duration_ms (:228-235) y lee el umbral descartado explosive_confidence_threshold (:215-219), porque TradeHorizon sólo tiene Continuous (crates/strategy-core/src/types.rs:14-18).

**Impacto.** Sin efecto vivo. El doc engaña al lector y mantiene dormida una segunda regla de decisión, que reviviría si TradeHorizon volviera a tener otra variante.

**Objeción del escéptico.** La reintroducción fue deliberada y está documentada como compatibilidad, y un test de contrato llama a evaluate_consensus_for_coin y a evaluate_dual_consensus_for_coin y exige una sola evaluación (crates/signal-engine/tests/single_consensus_contract.rs:46 y :54). Un test que prohíba su existencia rompería ese contrato. Basta con actualizar la cabecera.

**Test que falla hoy.** Guarda de fuentes `u_err_2_cabecera_coincide_con_la_superficie`: si orchestrator.rs, antes de #[cfg(test)], contiene `fn evaluate_horizon_consensus` o `fn evaluate_consensus(`, la cabecera no puede decir «Queda UNA superficie» sin nombrar los alias de compatibilidad. Actual: falla, porque se dan las dos cosas (crates/signal-engine/src/orchestrator.rs:37 y :118).

### LC-F1-65. «Fuente única» del horizonte que no lo es

Baja. Disputado. Panel: horizonte, n.º 9. Fase dueña: C-F1 (doc) y C-F4 (respaldo de leverage_matrix). Zona: Línea C.

**Afirmación.** El doc de TradeHorizon dice que el eje temporal vive en expected_duration_ms + temporal_scale (crates/strategy-core/src/types.rs:9-13), y operating_tau_ms se declara «FUENTE ÚNICA» con esas dos entradas (crates/quantum-arena/src/temporal_spectrum.rs:172-180). Pero horizon_tau_ms_coin intercala entre ellas la τ dominante publicada de la moneda (crates/risk-engine/src/lib.rs:1492-1511). Corrección de un escéptico: la divergencia no es sólo documental, porque leverage_matrix usa operating_tau_ms, con dos eslabones, como respaldo del stop para la misma intención (crates/risk-engine/src/leverage_matrix.rs:189-196).

**Impacto.** Quien implemente desde el doc reproduce un horizonte distinto del que dimensiona la orden. En el camino vivo del núcleo las intenciones llegan con expected_duration_ms > 0 (lib.rs:7120-7124), así que el eslabón intermedio sólo actúa con duración 0.

**Objeción del escéptico.** D-745 documenta la conducta real en horizon_tau_ms_coin (crates/risk-engine/src/lib.rs:1482-1490); el doc del tipo y la etiqueta quedaron desfasados sin que nadie dimensione mal. El test del revisor sólo pasaría cambiando la firma de operating_tau_ms, no corrigiendo el doc. Si hay un defecto con impacto, es el respaldo de leverage_matrix.

**Test que falla hoy.** `la_fuente_unica_del_horizonte_coincide_con_el_dimensionado` (tests de crates/risk-engine/src/lib.rs): arena con coins[0].dominant_tau_ms 600 000 y temporal_scale 0,5; intención con expected_duration_ms 0 y sin sl_operado. Esperado: la τ del respaldo de leverage_matrix es la de horizon_tau_ms_coin. Actual: 600 000 frente a unos 1 138 420 (operating_tau_ms con escala 0,5).

### LC-F1-66. Opiniones ML del consejo sin espejo con base baja

Baja. Disputado. Panel: espejo, n.º 8. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** ml_opinion centra en la base, según CERT-M2-C04 (crates/metacortex-engine/src/consejo_seniors.rs:382-384), pero su banda muerta de 0,05 y su escala de 0,20 son absolutas (crates/metacortex-engine/src/consejo_seniors.rs:385-394). Con base 0,18, el extremo bajista (p = 0) da −0,867 y el alcista llega a +1 con p = 0,38. Teleonomía usa (p − base)·2 (:871), cuyo mínimo es −2·base. El doc de ml_opinion sigue describiendo un centrado en 0,5 (:378-380). La usan SeniorML y SeniorMetacognitivo (:778 y :811).

**Impacto.** Ante la misma convicción normalizada, esos asientos aportan menos al consenso corto que al largo, y Teleonomía resta utilidad al corto. Corrección de un escéptico: la contradicción es con el propósito de CERT-M2-C04 (eliminar el sesgo estructural contra el corto), no con la «simetría de rango» de crates/god-engine-core/src/calibration.rs:321-322, que es el contrato de otra función; con base ≥ 0,20, ml_opinion sí llega a −1.

**Objeción del escéptico.** El contrato documentado del consejo es una distancia absoluta alrededor de la base, simétrica en puntos de probabilidad. El tope sólo pesa con base < 0,20, en el extremo bajo del rango medido por CL-21. Imponer la métrica normalizada es una preferencia de diseño.

**Test que falla hoy.** `ml_opinion_es_espejo_en_semiintervalos` (tests de consejo_seniors.rs): ml_opinion(0.0, 0.18) == −1,0, hoy −0,867; ml_opinion(0.09, 0.18) == −ml_opinion(0.59, 0.18), hoy −0,267 frente a 1,0. Teleonomía con base 0,30, fused 0, persistence 0, Hurst 0,5, slippage 0, do_calculus 0 y wr 0,5: signal(ml 0,15) == −signal(ml 0,65); hoy −0,12 frente a +0,28.

### LC-F1-67. Umbral corto del bosque sombra con 1 − P(ganar)

Baja. Disputado. Panel: espejo, n.º 10. Fase dueña: C-F7. Zona: Línea C (evolution-engine).

**Afirmación.** En la rejilla de umbrales, el «corto» se identifica por el signo de obs.obi (crates/evolution-engine/src/online_random_forest.rs:205), que en el camino con features reales es el OBI del libro y no el lado (:73-83). Su PnL se cuenta cuando prob_win < 1 − th (:208), es decir, cuando el bosque predice que la operación pierde, aunque was_profitable ya es la ganancia del propio corto (:82). El resultado 1 − th (:222) se escribiría en ml_threshold_short (crates/evolution-engine/src/online_daemon.rs:982-991), que gobierna otra magnitud. Contradice el comentario CERT-M8-C02, según el cual sólo cuentan las observaciones cuya probabilidad supera th (crates/evolution-engine/src/online_random_forest.rs:186-192).

**Impacto.** Latente: con predicciones de clase 0/1 las curvas son planas y no se publica nada (crates/evolution-engine/src/online_random_forest.rs:200-204 y :226-233). Si un clasificador con probabilidades sustituye a smartcore, el daemon escribiría un umbral elegido con las operaciones cortas que el bosque predijo perdedoras.

**Objeción del escéptico.** No llega a ninguna decisión viva (CL-18 impide publicar y el daemon sólo escribe con la evolución armada), y el test exige probar un camino que hoy no existe. Merece una nota para el dueño de evolution-engine.

**Test que falla hoy.** `umbral_corto_usa_p_ganar_del_corto`, tras extraer la rejilla a una función pura con predicciones continuas: 50 observaciones cortas con prob_win 0,9 y pnl +0,01, y 50 con prob_win 0,1 y pnl −0,01. Esperado: con th 0,6 la curva corta suma +0,5. Actual: −0,5.

### LC-F1-68. La deriva del auditor sombra ignora el lado

Baja. Disputado. Panel: espejo, n.º 11. Fase dueña: C-F3. Zona: Línea C.

**Afirmación.** La deriva de PnL resta un retorno esperado (ml_at_entry − 0,5)·0,02 que no depende del lado y que centra en 0,5, no en la base (lib.rs:4211), mientras realized_ret es el retorno neto de la propia posición (lib.rs:4173). Un corto acertado con ml bajo se registra como sorpresa positiva. Corrección de un escéptico: el centrado en 0,5 también sesga los largos.

**Impacto.** Telemetría: la alarma salta con pnl_drift < −0,5 (crates/metacortex-engine/src/shadow_graph_auditor.rs:81), un umbral en fracción del nocional prácticamente inalcanzable.

**Objeción del escéptico.** Es telemetría sin consumidor de decisión: evaluate_system_drift sólo tiene el envoltorio de lib.rs:1295-1296, que nadie llama, y sus tests. Habría que corregirla si algún día se conecta el reentrenamiento por deriva.

**Test que falla hoy.** `pnl_drift_espejo`, tras extraer la fórmula a una función pura: un largo con ml 0,80 y retorno +0,006, y un corto con ml 0,20 y retorno +0,006. Esperado: la misma deriva (0,0). Actual: 0,0 para el largo y +0,012 para el corto.

### LC-F1-69. Tres fichas de veto sin fecha

Baja. Disputado. Panel: vetos, n.º 11. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** EntradaVeto no tiene campo de fecha; su doc dice que va en responsable (crates/risk-engine/src/veto_registry.rs:50-51), y ADR-0004 la exige (docs/adr/ADR-0004-veto-registry-como-codigo.md:12-13 y :19). V-LOGIC-007, V-LOGIC-009 y V-LOGIC-010 no llevan fecha AAAA-MM-DD (crates/risk-engine/src/veto_registry.rs:226, :253 y :265), y el test de completitud sólo exige que responsable no esté vacío (:571-574).

**Impacto.** Higiene documental de la gobernanza.

**Objeción del escéptico.** No afecta a ninguna decisión, y la fecha se puede reconstruir con git: la certificación de la ola Ω58 es el commit e96afb0f, del 2026-10-09, posterior a Ω46 (f9ca4284, del mismo día). Basta con completar la fecha en las tres entradas.

**Test que falla hoy.** `toda_entrada_tiene_fecha_iso`: para cada entrada, responsable contiene una fecha que casa con \d{4}-\d{2}-\d{2}, o existe un campo fecha validado. Actual: falla en V-LOGIC-007.

### LC-F1-70. FUENTES_CRUZADAS_CONTRATOS no coincide con la macro

Baja. Disputado. Panel: vetos, n.º 12. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** El doc dice que la lista «documenta el MISMO conjunto» que corpus_contratos! (crates/risk-engine/src/veto_registry.rs:423-426). La lista tiene 17 rutas (crates/risk-engine/src/veto_registry.rs:428-444) y la macro 18, porque incluye veto_registry.rs (:454-471, con el propio archivo en :468). La lista mezcla bases: 7 rutas que empiezan por ../ sólo existen relativas a src/ (:437-442 y :444), y 10 sólo relativas a la raíz del crate (:428-436 y :443). El doc de la macro también dice que concatena las fuentes declaradas (:447-449). Ningún código usa la constante.

**Impacto.** Quien añada un contrato siguiendo la lista puede escribir una ruta que no resuelve, o creer que el corpus no incluye el propio registro.

**Objeción del escéptico.** Es una constante muerta, sin lectores; la discrepancia no llega a ninguna decisión ni a ningún test. Es documentación que se corrige de paso.

**Test que falla hoy.** `fuentes_cruzadas_resuelven_y_coinciden`: para cada ruta p de la lista, Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(p).exists(), y la longitud es 18. Actual: falla con "src/lib.rs", que resuelve a src/src/lib.rs, y con la longitud 17.

### LC-F1-71. Un mismo fuente compilado como dos binarios con ediciones distintas

Baja. Disputado. Panel: versión y construcción, n.º 8. Fase dueña: C-F1. Zona: Línea C.

**Afirmación.** El paquete raíz, de edición 2021 y con autobins = false (Cargo.toml:34 y :36), declara un [[bin]] continuous_evolution_backtest cuyo path apunta dentro de crates/backtest-engine (Cargo.toml:157-159). backtest-engine, de edición 2024 y sin autobins = false (crates/backtest-engine/Cargo.toml:1-4), descubre el mismo src/bin/continuous_evolution_backtest.rs como binario propio. build.ps1 y build.sh compilan con --workspace (build.ps1:8; build.sh:12), y salen dos destinos con el mismo nombre de archivo.

**Impacto.** El ejecutable con ese nombre es el último que se enlazó, y cargo avisa de la colisión. No se encontraron construcciones sensibles a la edición en ese archivo, así que no se afirma una diferencia de conducta.

**Objeción del escéptico.** Los dos salen del mismo fuente, cargo sólo avisa, es una herramienta fuera de la ruta de órdenes y ningún contrato se contradice. Es higiene del build, sin severidad.

**Test que falla hoy.** `g0_nombres_de_binario_unicos` (crate raíz): cargo metadata --format-version 1 --no-deps y los targets bin de todos los paquetes. Esperado: nombres únicos. Actual: continuous_evolution_backtest aparece en trader-gemini-v5 (Cargo.toml:30) y en backtest-engine.

### LC-F1-72. Credenciales con precedencias distintas

Baja. Disputado. Panel: versión y construcción, n.º 9. Fase dueña: C-F1 (proveedor único) y C-F5. Zona: Línea C.

**Afirmación.** god_engine prefiere MAINNET_API_KEY y TESTNET_API_KEY y sólo recurre a BINANCE_* si la clave está vacía, y toma el secreto del mismo prefijo que la clave (god_engine.rs:1152-1166). Con MAINNET_API_KEY definida y sin MAINNET_SECRET_KEY, el resultado es una clave sin secreto. El nombre que el repositorio declara canónico es BINANCE_API_KEY (crates/data-pipeline/src/api_client.rs:21-22; template.env:1), y el aviso del host empuja a definir MAINNET_API_KEY (god_engine.rs:1168-1170). El forense prefiere BINANCE_DEMO_* en testnet y BINANCE_API_KEY en mainnet (src/bin/audit_forensic_backtest.rs:66-83). EnvManager::get_var, que recorta espacios, sólo se usa en su test (src/env_manager.rs:63-70).

**Impacto.** Con dos cuentas configuradas con nombres distintos, el motor opera en una mientras las herramientas informan de otra. Con un par incompleto, el motor firma con un secreto vacío. Falla cerrado, porque el bucle de balance registra el error de la API en cada reintento (god_engine.rs:1222-1224), pero no diagnostica el par incompleto.

**Objeción del escéptico.** resolve_credentials_from_lookup, el proveedor que el revisor toma como referencia, es código muerto: sólo lo llaman funciones de su archivo sin llamadores externos (crates/data-pipeline/src/api_client.rs:94 y :166). Además mezcla pares por sí mismo, porque resuelve clave y secreto por separado, y en testnet prefiere BINANCE_TESTNET_* antes que BINANCE_DEMO_*, al revés que el forense. El defecto defendible es más estrecho: la regla canónica del motor.

**Test que falla hoy.** `g0_credenciales_regla_canonica`, sobre la función que usa god_engine, extraída. Entrada: MAINNET_API_KEY "A", BINANCE_API_KEY "B", BINANCE_SECRET_KEY "sb" y ningún MAINNET_SECRET_KEY. Esperado: Err («par MAINNET_* incompleto») o ("B", "sb"), nunca una clave sin su secreto. Actual: ("A", "").

### LC-F1-73. El TLS del cliente de órdenes lo decide el manifiesto de otro crate

Baja. Disputado. Panel: versión y construcción, n.º 10. Fase dueña: C-F1 y C-F5. Zona: Línea C; la matriz de CI es de Línea E.

**Afirmación.** god-engine-core declara reqwest 0.11 sin default-features = false (crates/god-engine-core/Cargo.toml:25), mientras los demás manifiestos con reqwest 0.11 lo desactivan (Cargo.toml:75; crates/execution-engine/Cargo.toml:13; crates/quantum-arena/Cargo.toml:14; crates/telemetry-server/Cargo.toml:16; crates/backtest-engine/Cargo.toml:11). Cargo.lock incluye native-tls, hyper-tls y tokio-native-tls para reqwest 0.11.27. El cliente REST del ejecutor se construye con hickory_dns(true) sin use_rustls_tls (crates/execution-engine/src/client.rs:136-144), a diferencia de QuantumSocketPool (crates/execution-engine/src/quantum_socket.rs:38-39). execution-engine no depende de god-engine-core, así que `cargo test -p execution-engine` no unifica default-tls y el binario sí. Que reqwest elija native-tls por defecto con default-tls activo depende de su tls.rs, que está en el registro de cargo, fuera del repositorio, y no se verificó en ae6d51a2.

**Impacto.** La pila TLS del camino de órdenes la decide el manifiesto de otro crate. No cambia ningún número. La matriz de CI no incluye execution-engine (.github/workflows/unit-suites.yml:45-54), así que tampoco se ejecuta ese cliente con ningún backend.

**Objeción del escéptico.** No contradice ningún contrato: execution-engine ya usa native-tls para el WebSocket (crates/execution-engine/Cargo.toml:18), las pruebas unitarias no ejercitan TLS y no hay ninguna orden, tamaño ni veto afectado. El test de manifiestos impone una preferencia; como mucho, hay que fijar el backend en BinanceClient.

**Test que falla hoy.** `g0_reqwest_011_sin_tls_implicito` (manifiestos, crate raíz): toda declaración de reqwest 0.11 del workspace lleva default-features = false. Actual: falla en crates/god-engine-core/Cargo.toml:25.

---

## Refutados

- Identidad, n.º 16 (la confirmación de apalancamiento compara bytes): falla cerrado; si acaso, hay que canonizar en el emisor (crates/execution-engine/src/executor.rs:3576-3577).
- Desconocido, n.º 9 (detect_collisions informa cero colisiones sin medir): su único llamador consulta un registro propio vacío (crates/audit-engine/src/main.rs:16 y :75); la colisión real ya está en LC-F1-06.
- Espejo, n.º 6 (calibrador, win rate y registros de rama mezclan lados): ningún contrato exige separarlos; es una decisión de diseño que antes habría que medir.
- Espejo, n.º 9 (el veto discreto de respaldo sólo protege a los largos): está declarado (crates/risk-engine/src/orchestrator.rs:191-198), lo fija crates/risk-engine/tests/nonfinite_admission_contract.rs:175-190 y es inalcanzable con p_crash publicado.
- Vetos, n.º 6 (el prefijo V-LOGIC no casa con la clase RiesgoDuro): los ids son estables y ningún contrato obliga a que prefijo y clase coincidan.
- Vetos, n.º 9 (rechazos tipados fuera del registro): V-RISK-003 ya describe el guard del 95 % (crates/risk-engine/src/envio.rs:96-98), y la aduana de datos queda fuera del alcance del registro (crates/risk-engine/src/veto_registry.rs:60-63).
