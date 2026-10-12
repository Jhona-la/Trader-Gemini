# Hallazgos de la fase C-F3 (Línea C): la decisión viva, del evento a la orden

Base: `main` en 1d1f8bdd. Ocho bloques; en cada uno, un auditor siguió la intención desde el evento hasta la orden y dos escépticos (lente «refutar» y lente «completitud») verificaron cada hallazgo por lectura. Fase de sólo lectura: nadie compiló ni ejecutó nada. Las predicciones de fallo son aritmética cerrada o estimación por lectura, y cada hallazgo lo indica. Las pruebas propuestas están en `docs/audit/PRUEBAS_C-F3_borrador_2026-10-11.json`, sin escribir ni ejecutar todavía.

Convenciones:

- Bloques: F3.1 independencia y consenso espectral, F3.2 una sola τ, F3.3 confianza, F3.4 vetos, F3.5 modelos, F3.6 motores de señal, F3.7 strategy-core, código muerto y montículo, F3.8 tests de #713/#714 y ledger.
- `lib.rs` sin ruta es crates/god-engine-core/src/lib.rs y `god_engine.rs` es src/bin/god_engine.rs. Las citas son de 1d1f8bdd.
- Severidad: alta cambia decisiones vivas de dinero o de veto de forma errónea; media es un sesgo medible o una teoría mal especificada con consumidor vivo; baja no tiene consumidor vivo, o es documentación falsa o cosmética.
- «Dueño» es quien arregla. Las zonas ajenas reciben un pedido con su test en rojo; Línea C no toca su código.
- El mapa de factores (unidad, τ, calibración, cuántas veces entra en el tamaño, espejo), la verificación del ledger, las respuestas del plan, los literales, las retiradas y los faltantes de los escépticos están en `docs/audit/MAPA_C-F3_2026-10-11.md`.

Recuento: 62 hallazgos, 7 altos, 41 medios y 14 bajos. Los dos escépticos confirman 58; F3.2-04 queda retirado por duplicar LC-F1-36, y tres quedan disputados (detalle en cada uno). Los escépticos añadieron además 115 faltantes, recogidos en el mapa.

| Id | Hallazgo | Severidad | Estado | Dueño |
|---|---|---|---|---|
| F3.1-01 | El veredicto espectral decide en ruido puro: es una sola fuente votada 13 veces, con un argmax sin nulo y una coherencia que no es independiente | alta | confirmado | Qoder (espectro y consenso, PLAN_LINEA_C_DESDE_LA_BASE §1.2 :54-61). Línea C entrega el pe |
| F3.1-02 | El dominante puede salir de una escala que no ha visto su τ (hasta ~146 años); allí z es el retorno desde el arranque del proceso. Amplía la nota sin id del borrador C-F2 (FICHAS :983) | alta | confirmado | Qoder (espectro y consenso). |
| F3.1-03 | El consenso escalar no depende de la magnitud de los votos: un único voto de 2·10⁻⁹ produce Long con confianza 0,70 | alta | confirmado | Qoder (consenso). |
| F3.2-01 | QO-586 y el co-despacho juzgan una τ que la orden no lleva | alta | cerrado (CL-74) | Línea C: puertas_del_continuo y X-016 en lib.rs, fuera de las ramas 1-10. |
| F3.3-01 | conviccion_de_rama no es monótona en la evidencia y compara con ½ en vez de con el equilibrio de la geometría | alta | cerrado (CL-75) | Línea C: la función y las ramas 11, 13 y 15. Pedido a Línea E para los llamadores de las r |
| F3.3-02 | Una desventaja demostrada (hi < ½) se reinfla por encima del gate: el doc de D-752 es falso | alta | cerrado (CL-75) | Línea C (puertas, modulación, superposición); el gate corresponde a C-F4. |
| F3.6-01 | El solitón y el choque votan siempre LARGO en el consenso escalar: su entrada de «velocidad» es |Δmid| sin signo | alta | cerrado (CL-73) | Línea C (motores de signal-engine, CL-n). Si el arreglo cambia el escritor compartido dir_ |
| F3.1-04 | El TTL del veredicto espectral no tiene techo: con τ_dom ≥ 19,5 h el dominante sobrevive a la vida que fija el orquestador, hasta ~146 años | media | confirmado | Qoder (consenso; skill_motores). |
| F3.1-05 | La sombra espectral del solitón rompe el espejo: con OFI +0,5 la pendiente es 0,5 y con −0,5 es 1,0. El comentario «ESPEJO EXACTO» es falso y un test lo deja fijado | media | confirmado | Qoder (sombra espectral del consenso). El arreglo debe invertir la última aserción de h2_1 |
| F3.1-06 | La concentración espectral usa N = 32 fijo y mide falta de observación, no estructura: en ruido blanco vale ≈ 0,9 y no amortigua el voto de la proyección | media | confirmado | Qoder (espectro). |
| F3.1-07 | Un sesgo spot «simétrico» mueve nn_score 2,3 veces más hacia abajo que hacia arriba cuando la base del bosque es 0,30 | media | confirmado | Codex (calibración: compose_ml_prob), junto con el autor de CL-13. El punto de aplicación  |
| F3.2-02 | El veto de grupo lee el ρ de una escala que no es la de la orden; la puerta del plan está mal especificada | media | confirmado | Línea C: publicación del núcleo y lector del risk-engine. |
| F3.2-03 | R-15 sigue penalizando las τ largas por encima de 22 minutos | media | confirmado | Línea C (compuerta de primer toque del risk-engine). El autor anterior es AGY (Ola Ω73) y  |
| F3.2-04 | El Kelly de la moneda lleva la τ de la operación cerrada a la siguiente orden | media | refutado (duplicado) | Línea C |
| F3.2-05 | La τ de la señal no es la τ de la intención en las ramas 11, 12 y 13 | media | confirmado | Línea C (ramas 11-13 del núcleo y orchestrator). La rama 12 vive en signal-engine. |
| F3.2-06 | Una posición sin τ de decisión entrena el espectro, el enfriamiento y el Kelly con el nodo crudo | media | confirmado | Línea C (camino de cierre). La adopción corresponde a C-F5. |
| F3.3-03 | La rama 11 mide su convicción a τ_dom pero se dimensiona a la micro τ: expected_lifetime_ms del orquestador no tiene lector | media | confirmado | Línea C. |
| F3.3-04 | El sesgo spot rompe el espejo de la puerta ML: el corto recibe hasta 4,5 veces más impulso que el largo | media | confirmado | Línea C. |
| F3.3-05 | La coherencia inter-espectral se recorta en 0: un espectro en contra da la misma confianza tensorial que uno neutro | media | confirmado | Línea C (orchestrator.rs). |
| F3.3-06 | El peso hebbiano aprende la tasa de acierto (signo del PnL), no el EV, y multiplica el compuesto ×[0,5; 2] | media | cerrado (CL-76) | Línea C (regla y lector). Los consumidores de las ramas 1-10 son de Línea E. |
| F3.3-07 | La rama 13 cuenta dos veces la misma lectura del ML (piso y puerta) | media | confirmado | Línea C. |
| F3.3-08 | La puntuación cruda dimensiona tres veces y la calibrada una | media | confirmado | Línea C, bloque C-F4 (riesgo). |
| F3.3-09 | La rama 12 no pasa por conviccion_de_rama: su historial se escribe y nunca se lee | media | confirmado | Línea C. |
| F3.4-01 | Los pesos de los asientos del consejo premian la tasa de acierto de las operaciones ejecutadas, no la esperanza | media | confirmado | Línea C (consejo, C-F3) |
| F3.4-02 | Un mismo dato (VPIN) produce dos vetos y vuelve imposible el override por encima de 0,92 | media | confirmado | Línea C (consejo) |
| F3.4-03 | El escudo de momento adverso no está al 95 %: une cuatro pruebas y un Stouffer sobre ejes correlacionados | media | confirmado | Línea C: la definición del escudo está fuera de las ramas 1-10. Se pide a la Línea E la mi |
| F3.4-04 | El veto de muro es el OBI de nivel 0 con otro umbral, y sólo existe en vivo | media | confirmado | Línea C (puertas_del_continuo y harness de replay). La banda de new_random es una petición |
| F3.4-05 | La sonda de arranque frío no produce la evidencia que exige el ML-GATE, es por moneda y se repite tras cada reinicio | media | confirmado | Línea C (ML-GATE del núcleo), con decisión del dueño sobre el presupuesto de sonda (R-18 D |
| F3.5-01 | NEAR cuenta como cubierta, pero su bosque y su predictor _VOL no cargan; el freno VOL no tiene ningún modelo válido | media | confirmado | Línea C: que la cobertura signifique cargabilidad, usando el veredicto de ml_registry. Ped |
| F3.5-02 | El bosque opina con el macro de semilla, fuera de su soporte de entrenamiento; la caducidad del macro no se consulta | media | confirmado | Línea C (núcleo): no enviar el bosque al ensamble si macro_staleness_ms supera la ventana  |
| F3.5-03 | DarkAlpha se entrena con la entrada cruda y se sirve en z-scores; su compuerta de promoción compara pérdidas medidas en coordenadas distintas | media | confirmado | Línea C (núcleo): rechazar en la recarga y en la carga inicial una red sin Scaler ajustado |
| F3.5-04 | El sesgo spot no tiene espejo cuando la base del bosque no es 0,5: el mismo diferencial empuja (1 − b)/b veces más hacia el corto | media | confirmado | Línea C: aplicar el sesgo en log-odds (logit p + κ·s) o después de la divergencia, con el  |
| F3.5-05 | El ensamble bosque⊕DarkAlpha promedia probabilidades de eventos distintos y todos los consumidores lo miden contra la base del bosque | media | confirmado | Línea C (ensamble): no mezclar modelos con etiquetas distintas. Hay que publicar la base d |
| F3.6-02 | La sombra espectral del solitón rompe el espejo: su pendiente depende del signo del OFI | media | confirmado | Línea C (sombra en lib.rs:2088-2112 y voto_espectral del solitón). El test H2-10 es de GLM |
| F3.6-03 | El votante Rényi-Tsallis es una función de OBI: la «entropía» publicada es h(|OBI|) y Rényi nunca se calcula en vivo | media | confirmado | Línea C |
| F3.6-04 | El coaxial escalar vale 0 por álgebra en precios ≥ 0,2236 USD y fabrica votos de ±0,19 a ±0,22 en monedas más baratas | media | cerrado (CL-66) | Línea C para el lector coaxial. Los suelos van con LC-F3-02 al principio de C-F4. |
| F3.6-05 | Las teorías nombradas no están implementadas: los motores son envolturas de OBI, OFI y la desviación de posición | media | confirmado | Línea C |
| F3.6-06 | El peso «hebbiano» premia la tasa de acierto en vez de la esperanza, y entra dos veces en la decisión | media | cerrado en la regla (CL-76) | Línea C para update_weight. El consumo de hebbian_mult en el compuesto de las ramas 1–10 e |
| F3.6-07 | La vida de la rama 12 sale del gen base_duration_ms, no de una τ medida, y pesa la mitad en log tras la mezcla | media | confirmado | Línea C |
| F3.6-08 | En TrendRunner la volatilidad sube la convicción (vol_boost), en contra de D-746 | media | cerrado (CL-77) | Línea C |
| F3.7-01 | `vecm_zscore` es el nivel de la base entre el ATR, no un z: una prima constante vota reversión de forma permanente | media | cerrado con spot (CL-78) | Línea C (escritor en lib.rs y lectores de strategy-core). El lector conformal se ve afecta |
| F3.7-02 | La cointegración se supone: la guarda t½ ≤ 2τ* deja pasar caminatas aleatorias independientes | media | confirmado | Línea C |
| F3.7-03 | El TTL anti-staleness del spot está muerto en producción: un spot congelado sigue votando | media | cerrado (CL-79) | Línea C (núcleo). data-pipeline debe publicar la edad del spot. |
| F3.7-04 | StatArb sigue heredando z y t½ de otra moneda por la caída al global; las API estrictas no se usan | media | confirmado | Línea C (lectores de strategy-core y signal-engine) |
| F3.7-05 | El mismo escalar entra en hasta tres votos del consenso | media | disputado | Línea C |
| F3.7-06 | Régimen global: sólo con eventos de libro del slot 0, en reloj de eventos, sin τ en `allow_trade`, sin calibrar y con un respaldo inalcanzable | media | confirmado | Línea C |
| F3.8-01 | El sesgo spot aditivo rompe el espejo de K-06/K-26 cuando la base no es 0,5 | media | confirmado | Línea C (lib.rs fuera de las ramas 1-10 y calibration.rs) |
| F3.8-02 | Las puertas K-06 (y el gate K-23) no son espejo bajo el modelo de primer toque del propio repo: favorecen a los cortos | media | confirmado | Línea C |
| F3.3-10 | El colchón de free_cap es un veto binario silencioso, duplicado en línea, y su test de espejo prueba una copia | baja | confirmado | Línea C. |
| F3.3-11 | Documentación falsa y código muerto en la ruta de la confianza | baja | confirmado | Línea C. |
| F3.4-06 | El veto «absoluto» de drawdown del consejo es inalcanzable: el risk-engine veta antes en d* ≈ 0,632 | baja | confirmado | Línea C (consejo) |
| F3.4-07 | El consejo no está en el registro de vetos: V-LOGIC-007 lleva su nombre pero pinea REJ_CONFIDENCE del risk-engine | baja | disputado | Línea C (entradas del registro), coordinando con quien mantenga risk-engine |
| F3.4-08 | El test de integración de K-26 corre con base 0,5, no con la 0,25 que declara | baja | confirmado | Línea C (revisión en C-F3, recibo en C-F10), con aviso a AGY como autor de #714 |
| F3.5-06 | El inventario de modelos versionado no coincide con el manifiesto, y la documentación del registro es falsa | baja | confirmado | Línea C (documentación y heartbeat); pedido a Línea E: decidir si models/ se versiona ente |
| F3.6-09 | Documentación falsa y API pública sin consumidor en los motores | baja | confirmado | Línea C |
| F3.7-07 | El núcleo calcula una cotización maker en cada tick y la descarta | baja | confirmado | Línea C |
| F3.7-08 | Heap por evento, E/S síncrona de ficheros por cierre y justificaciones sin lector por deliberación | baja | confirmado | Línea C propone; cada dueño de archivo decide (metacortex-engine para el consejo). |
| F3.7-09 | Publicaciones sin lector: las 8 claves `market_regime_*` y `statarb_beta` | baja | confirmado | Línea C |
| F3.7-10 | Tres τ* de respaldo distintas, y la doc R6-B13 («único fallback») es falsa | baja | disputado | Línea C |
| F3.8-03 | Certificaciones del ledger que no ejecutan el código que certifican (#713, #714, REV-10, R-15) | baja | confirmado | AGY, autor de #713, #714 y REV-10 (petición con test). Las filas del ledger son de Línea E |
| F3.8-04 | La misma base del modelo pasa por cuatro guardas distintas según el camino | baja | confirmado | Línea C |
| F3.8-05 | hurst_mod del gate ML va con signo, aunque su comentario dice |H − 0,5| | baja | confirmado | Línea C |

## Severidad alta

### F3.1-01. El veredicto espectral decide en ruido puro: es una sola fuente votada 13 veces, con un argmax sin nulo y una coherencia que no es independiente

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/god-engine-core/src/lib.rs:2052-2059, :2489-2503, :2538-2590; crates/signal-engine/src/voto_espectral.rs:73-84, :101-118, :149-172; crates/signal-engine/src/orchestrator.rs:506-530, :549-560, :595-621
- Causa: Los 13 motores son funciones impares del mismo x_k = momentum_z_k (lib.rs:2060-2423). El signo de cada voto es ±signo(x_k), con el oscilador como único contrario, así que el consenso por escala reescala una sola fuente en lugar de agregar varias. El dominante es el argmax de |consenso_k| entre las escalas que superan la resolución (hasta 17 a 1 evento/s), sin corrección por selección (voto_espectral.rs:73-84). El comentario H4 llama «modulación INDEPENDIENTE» a la coherencia (orchestrator.rs:506-513), pero es media_banda_activa del MISMO vector, con la escala dominante incluida (lib.rs:2582-2590). Con una sola escala activa vale 1. La decisión sólo exige |net| > cutoff_floor = (min_confidence − 0,50)·1,0 (orchestrator.rs:560). Con el genoma base (genome.rs:712, 0,55·1,018) eso es ≈ 0,06, es decir |v_dom| > ≈ 0,086. Es un umbral sacado del gen, no del nulo de |v_dom| en ruido. Estimación por lectura, sin medir: a |x| ≈ 1,25, la desviación típica de z en ruido con este denominador (media|dev|), el voto medio de los 13 ronda 0,2-0,3.
- Consecuencia viva: Con espectro activo el núcleo recibe un tensor Long o Short casi en cada evento de un mercado sin deriva. Peso en el compuesto: tensor_boost entra con w_tensor = 0,40·(1 − 0,385) = 0,246 (lib.rs:5428-5446; genome.rs:732), multiplicado por un factor hebbiano de hasta ×2 (lib.rs:4633-4637). Con v_dom = 0,3 el compuesto se desplaza ≈ ±0,05-0,15. Usos del compuesto: - decide el veto por signo de puertas_del_continuo (lib.rs:1610-1611); - fija los umbrales de las ramas (lib.rs:5972-6206); - alimenta los cuantiles de esos umbrales (lib.rs:5469). Rama 11: dispara si |net| ≥ 0,56 (lib.rs:6216-6221), lo que se alcanza con |v_dom| entre 0,56 y 0,80. La frecuencia viva está sin medir. qo_652_fraccion_sobre_corte y qo_624_fraccion_espectral (orchestrator.rs:452-472) la darían en un log de demo.
- Prueba que falla hoy: cf3_en_ruido_el_veredicto_espectral_no_decide (crates/god-engine-core/tests/cf3_consenso_espectral.rs). Caminata aleatoria de 7 200 eventos de profundidad a 1 evento/s. Entre las evaluaciones con dominante activo (v ≠ 0 y τ ≥ 30 s), la fracción de decisiones no Flat debe ser ≤ 0,05. Predicción: falla (alta confianza, no medido).
- Dueño: Qoder (espectro y consenso, PLAN_LINEA_C_DESDE_LA_BASE §1.2 :54-61). Línea C entrega el pedido con el test.
- Relación con hallazgos anteriores: LC-F3-03
- Escépticos: confirmado.
  - Corrección (refutar): 1. El oscilador no es el único contrario. Conformal vota −tanh(z_k/2) cuando la escala vecina más lenta se opone a z_k (conformal_reversion_filter.rs:140 y :146). La fuente sigue siendo una sola, el recorrido del precio, pero la cuenta «12 a favor y 1 en contra» es falsa. 2. Hawkes, flujo y confluencia se abstienen sin excitación (hawkes_bessel.rs:145; flow_impulse.rs:82; flow_excitation_confluence.rs:76-79). La tendencia se abstiene con H ≤ 0,5 (trend_runner.rs:117-121), y coaxial sin compresión. Aun así todos cuentan en Σw (voto_espectral.rs:157-164). Por tanto, para el mismo x, |v_dom| cambia con el estado no direccional (intensidad de trades, Hurst): un corte fijo no es un umbral fijo sobre x. 3. Los números de la consecuencia son del genoma base. El genoma activo de 1d1f8bdd (config_dir/genomes/{prod,demo,backtest}/active.json) tiene min_confidence_btc 0,70 y weight_obi 0,9147, que se recorta a 0,8 en lib.rs:5440. Con él: - cutoff_floor = 0,20, así que hace falta |v_dom| > 0,20-0,29; - tensor_cutoff de la rama 11 = 0,70, y por la vía espectral hace falta |v_dom| ≥ 0,70-1,0; - w_tensor = 0,08 y w_nn = 0,12. El empuje en el compuesto es unas 3 veces menor que el descrito. Mantengo severidad alta por mecanismo: decide la dirección de tensor_boost en ruido. 4. «Casi en cada evento» debe decir «en cada evento de trade». 5. Los instrumentos propuestos para medir la frecuencia viva no sirven tal como están: - qo_624_fraccion_espectral está roto con más de una moneda (LC-F1-59); - qo_652 cuenta todas las evaluaciones, incluidas las que no tienen dominante y las de τ < 30 s (orchestrator.rs:591-593). Hay que arreglarlos antes de leer un log de demo.
  - Corrección (completitud): Hay que corregir la consecuencia viva y el instrumento de medida. 1. La rama 11 no dispara sólo con |net| ≥ 0,56. Además exige las 9 condiciones de lib.rs:6274-6312, entre ellas: - composite_score ≥ max(0,9·tech_threshold, p85|composite|), con piso_score_medido en lib.rs:4999 y uso en :6259; - OBI > max(0,85·umbral, p80|OBI|). El daño principal entra por el compuesto: el veto de signo D-472 (lib.rs:1610-1611), el cortafuegos de racha D-499 (lib.rs:7349-7369), las excepciones D-624 (lib.rs:6949-6951, :6977-6979) y el cuantil p85|composite|, que fija los umbrales de las ramas 1-10 de Línea E (lib.rs:5963, :6047). El ruido del tensor sube los umbrales de Línea E. 2. Falta el mecanismo que lleva el argmax a |x| grande. El único voto contrario es el oscilador, −(x+0,4x³)·e^{−0,5x²} (quantum_oscillator.rs:30-56, :102-114). Usa k, λ y α literales sin escritor de producción (lib.rs:2063-2070; rg sólo encuentra tests). Vale −0,93 en x = 1,25 y −0,05 en x = 3,5, mientras los votos de momento saturan. A eso se suma la tendencia, que en ruido vota ≈ 0,48·tanh(x) por el centro nulo 0,521 del DFA (F2.3b-01; trend_runner.rs:117-118). 3. Las telemetrías qo_624 y qo_652 no sirven para medir la frecuencia: - qo_624 es LC-F1-59. - qo_652_vdom_sobre_corte cuenta también los dominantes con τ inoperable. - vdom_evaluados cuenta todas las evaluaciones, también las escalares, y suma todas las monedas aunque sólo se publique desde la moneda 0 (orchestrator.rs:425, :591-593). El instrumento correcto es core.diag_dir.tensor_signal y contribution_sum (campo pub en lib.rs:940; direction_diag.rs:143-167), por núcleo y moneda.

### F3.1-02. El dominante puede salir de una escala que no ha visto su τ (hasta ~146 años); allí z es el retorno desde el arranque del proceso. Amplía la nota sin id del borrador C-F2 (FICHAS :983)

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/quantum-arena/src/temporal_spectrum.rs:548-563; crates/signal-engine/src/skill_motores.rs:283-302; crates/signal-engine/src/voto_espectral.rs:149-172; crates/god-engine-core/src/lib.rs:2531-2570; crates/signal-engine/src/orchestrator.rs:382-384, :576-587
- Causa: Cuando τ ≫ transcurrido, α ≈ dt/τ y la EWMA se queda en p0. Entonces ewma_dev_vol = raw/masa ≈ media|p − p0|/p0, y z ≈ (p − p0)/media|p − p0| (temporal_spectrum.rs:548-563): mide el precio frente al de arranque y vale casi lo mismo en todas esas escalas. D-742 aplica la masa en la fusión (pesos_observables, temporal_spectrum.rs:1408-1424), pero el consenso de 13 motores no la usa: - aplicar_gate_observabilidad sólo corta por DEBAJO de la resolución (skill_motores.rs:283-302); - consenso_por_escala no pondera por masa (voto_espectral.rs:149-172); - los bloques de esas escalas no maduran nunca, así que la habilidad tampoco puede descontarlas y pesan el piso 0,15 como todas. Por último, el orquestador sólo exige τ ≥ 30 s (orchestrator.rs:382-384) y recorta la vida a 12 h (:576-587).
- Consecuencia viva: La dirección del tensor puede depender de cuándo se reinició el proceso. Las ~10 escalas no observadas comparten casi el mismo voto. Cuando el dominante es una de ellas, también inflan la media de banda y, con ella, la coherencia (lib.rs:2582-2590), que lleva el factor hacia 1,0. Alimenta F3.1-01 y F3.1-04. Frecuencia sin medir.
- Prueba que falla hoy: cf3_el_dominante_no_sale_de_una_escala_no_observada (crates/god-engine-core/tests/cf3_consenso_espectral.rs). Misma caminata y 30 min de calentamiento; después, en todo evento con v ≠ 0, consenso_espectral_tau ≤ tiempo transcurrido. Predicción: falla (alta confianza, no medido).
- Dueño: Qoder (espectro y consenso).
- Escépticos: confirmado.
  - Corrección (refutar): 1. Duplica casi entera la nota sin id del final de FICHAS_C-F2 («El consenso de 13 motores tampoco excluye las escalas que no han visto su τ», con el test cf2_espectro_no_elige_escalas_no_observadas). Hoy está publicada en wt-c12/docs/audit/FICHAS_C-F2_2026-10-11.md con el mismo contenido. El mecanismo de z como retorno desde el arranque ya está en F2.6-02. El campo «duplica» no puede quedar vacío: debe citar esa nota y F2.6-02. 2. Lo que sí añade: la inflación de la coherencia y el enlace con el TTL (F3.1-04). 3. Añado una trampa para el arreglo. consenso_por_escala normaliza por Σw en cada escala (voto_espectral.rs:157-168), así que multiplicar por la masa todos los pesos de una escala se cancela y no cambia nada. El arreglo D-742 tiene que ir en la selección del dominante (argmax de masa·|c_k|, o exigir τ ≤ transcurrido, que equivale a masa ≥ 1 − e⁻¹), no en la matriz de pesos. 4. Severidad: propongo media. Su efecto vivo está contenido en F3.1-01, y con el genoma activo w_tensor = 0,08.
  - Corrección (completitud): Añadir la causa que hace probable que ganen el argmax: el voto contrario del oscilador se apaga con |x| grande y los de momento saturan (ver la corrección de F3.1-01). Añadir también la dependencia del ritmo de eventos. resolucion_efectiva_ms usa TODOS los eventos (temporal_spectrum.rs:1430-1437). En vivo, con depth cada 100 ms más trades, entran escalas desde ~268 ms (k = 14). En un replay de un evento por segundo entran desde k = 15. El conjunto de escalas que pueden ser dominantes cambia con la mezcla de eventos, no con el tiempo físico. Severidad alta, de acuerdo.

### F3.1-03. El consenso escalar no depende de la magnitud de los votos: un único voto de 2·10⁻⁹ produce Long con confianza 0,70

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/signal-engine/src/orchestrator.rs:398-422, :477-500, :595-621
- Causa: net = (Σo⁺ − Σ|o⁻|)/Σ|o|·(0,70 + 0,30·eff) (orchestrator.rs:489-500). El primer factor sólo mira signos y el segundo tiene un piso de 0,70. eff = (Σ|o|/19)·1,4 es una magnitud media diluida entre las 19 registradas. La guarda sólo corta si Σ|o| == 0 exactamente (:477). Varios votantes leen el mismo dato: OFI en 9, OBI en 6, price_velocity en 5 y la cointegración en 3. Que coincidan en signo es contar el mismo dato varias veces y no aporta evidencia, pero aun así lleva net a ≥ 0,70.
- Consecuencia viva: Sin veredicto espectral activo, dos o tres votantes débiles que comparten el signo del OFI bastan para un tensor de 0,70: - supera el corte de la rama 11 (0,56 con el genoma base, lib.rs:6216-6221); - suma 0,246·0,70 ≈ 0,17 al compuesto (lib.rs:5428-5446), y más si el factor hebbiano es > 1. Misma raíz que dos casos ya vistos: CL-59 (Feynman reinyectaba v_dom) y F2.6-01 (Navier). Frecuencia viva sin medir.
- Prueba que falla hoy: cf3_un_voto_infinitesimal_no_compra_conviccion (crates/signal-engine/tests/cf3_consenso_escalar.rs). - Una estrategia fija de 2e-9 debe dar Flat o una confianza < 1e-6. - Los votos (0,002, 0,002) deben dar como mucho el 10 % de la confianza de (0,9, 0,9). Predicción: falla con certeza. Hoy dan 0,70, 0,70 y 0,997.
- Dueño: Qoder (consenso).
- Escépticos: confirmado.
  - Corrección (refutar): 1. Precisar: la fórmula es invariante a una escala común de los votos, salvo el factor [0,70; 1]. Cuando hay signos mezclados, las magnitudes sí pesan a través de |o_i|. 2. Agravante con el genoma activo: tensor_cutoff = clamp(0,70; 0,55; 0,85) = 0,70 (lib.rs:6216), y la puerta es «≥» (lib.rs:6220). Cualquier ensamble escalar unánime, por débil que sea, da 0,70 + 0,30·eff > 0,70 y pasa la puerta de la rama 11. Por la vía espectral, en cambio, hace falta |v_dom| ≥ 0,70. En vivo, la rama 11 es alcanzable sobre todo por este camino defectuoso.
  - Corrección (completitud): Hay que matizar la consecuencia en la rama 11, porque además del corte necesita las 9 condiciones (lib.rs:6274-6312). Falta además la contracara del mismo interruptor: un dominante espectral infinitesimal SILENCIA a un ensamble escalar fuerte. Va como faltante, porque es otra prueba y otra cara del defecto.

### F3.2-01. QO-586 y el co-despacho juzgan una τ que la orden no lleva

- Bloque: F3.2, Una sola τ.
- Ubicación: lib.rs:1587-1607 (QO-586; llamado en :6399 y :6691); lib.rs:6716-6745 y :262-268 (co-despacho); lib.rs:6830-6833 y :8254-8267 (Hurst); lib.rs:7166-7186 (media geométrica, :7181); risk-engine/src/lib.rs:909; quantum-arena/src/position.rs:841-890
- Causa: Las puertas se evalúan sobre expected_duration_ms de fast_intent y slow_intent: - micro τ ≤ centroide (temporal_spectrum.rs:1211-1238); - macro τ ≥ centroide (:1241-1267). Después, la modulación de Hurst (×[0,5; 2]) y la media geométrica 0,5/0,5 con el centroide reescriben la τ: la micro sube y la macro baja. Con H > 0, σ(τ) crece con τ, y el suelo de fricción es monótono. Por eso: - QO-586 sobre-veta la banda rápida (juzga una τ menor que la de la orden). - QO-586 sub-veta la banda lenta (juzga una τ mayor). Lo que deja pasar muere en REJ_TP_SL_FLOOR del riesgo, que es justo el desperdicio que QO-586 venía a evitar. En el co-despacho, Δ ln τ se mide antes de la mezcla. La mezcla con el mismo centroide lo divide por 2, y la ranura compara ese Δ/2 contra 0,80, así que ve colisión.
- Consecuencia viva: Se vetan entradas rápidas que serían viables a su τ real (qo_586_tau_inoperable), y se aprueban lentas que el riesgo rechaza después. Un segundo candidato de otra banda muere en la ranura, como #593 dice haber cerrado. Corrige F2.3b-01: el ×2 de Hurst no llega a banda_paga_friccion. No hay magnitud medida.
- Prueba que falla hoy: cf3_la_tau_que_juzga_qo586_es_la_tau_de_la_orden (crates/god-engine-core/tests/cf3_una_sola_tau.rs). Recorre la cinta T-1 por process_event. Cada vez que se abre una ranura, su entry_tau_ms debe coincidir (con error relativo de 1e-6) con la τ que juzgaron las puertas (last_fast_intent[0] o last_slow_intent[0].expected_duration_ms, guardados en lib.rs:6702-6706, antes de la modulación y de la mezcla). Hoy, la τ de la orden es √(τ_banda·H-mod·τ_c).
  - Cerrado por CL-74 (2026-10-11): `GodEngineCore::tau_de_la_orden` aplica la modulación de Hurst (D-609) y la media geométrica con el centroide resonante (#565) ANTES de `puertas_del_continuo`, con la misma fórmula (`tau_hacia_el_centroide`), y se retiran las dos reescrituras posteriores. QO-586, el co-despacho y la ranura juzgan ya la τ con la que vive la orden. Con la cinta T-1, la prueba abre dos órdenes y las dos viven a la τ juzgada (antes, una y a otra τ). Mueve la T-2, como se esperaba de una puerta: de 5a0b7b940fb4809b (37 operaciones, PnL −0,348, acierto 32 %, 3 205 vetos EV) a 3c243afa79d0e3e0 (62 operaciones, PnL −0,259, acierto 39 %, 1 302 vetos EV). No medí el reparto por banda. Lo esperable por el hallazgo es que la banda rápida deje de quedar sobre-vetada y que la lenta muera antes, en QO-586, en vez de en el riesgo. El golden no cambia.
- Dueño: Línea C: puertas_del_continuo y X-016 en lib.rs, fuera de las ramas 1-10.
- Relación con hallazgos anteriores: K-10/K-12 (ledger de Línea E, no verificado); corrige F2.3b-01
- Escépticos: confirmado.
  - Corrección (refutar): 1. Colisión del co-despacho. En el mismo evento no ocurre: en cuanto el primer candidato abre, el bucle hace break (lib.rs:8165-8167, «The legacy API returns ONE proposal»), y el segundo no llega a find_resonant_slot. La división de Δln τ por 2 actúa entre eventos: el candidato de la otra banda en un evento posterior se compara (lib.rs:7509-7520) con la entry_tau, ya mezclada, de la posición abierta antes. Además, el apilado en la misma dirección lo frena QO-588 (misma_direccion_sin_asegurar, lib.rs:7548-7552). 2. Sentido del sesgo. Sobre-vetar la banda rápida y sub-vetar la lenta es lo habitual, pero no siempre: - con factor de Hurst 0,5 y τ_c < 2·τ_f, la τ de la orden queda por debajo de la τ rápida; - con factor 2 y τ_s < 2·τ_c, queda por encima de la τ lenta. 3. Severidad alta, con la magnitud sin medir. Se puede medir en la cinta T-1 contando los qo_586_tau_inoperable que pagarían a la τ de la orden (ver la prueba corregida).
  - Corrección (completitud): 1) El mecanismo del co-despacho es otro: - El bucle de candidatos devuelve como mucho una orden por evento: `if new_order.is_some() { break; }` (lib.rs:8167, «The legacy API returns ONE proposal»). - La banda rápida va siempre primero (candidates[0] = fast_intent, lib.rs:6744-6745). - QO-588 exige que la posición previa de la misma dirección gane al menos 28 pb (position.rs:951-974, llamada en lib.rs:7548-7552), y en el mismo evento la recién abierta no puede cumplirlo. Por eso un segundo candidato de la misma dirección nunca se evalúa en el evento en que el primero abrió. La reducción de Δln a la mitad sólo actúa contra posiciones abiertas en eventos anteriores. #593 sigue abierto, pero por estas tres causas, no por la ranura sola. 2) Sobre «corrige F2.3b-01»: sólo vale para la duración. H sí llega a QO-586, porque es el exponente de σ(τ) en la sonda (lib.rs:1559-1566 → tp_sl.rs:294-318). 3) Una incertidumbre del test queda resuelta: la sonda de arranque frío es la misma orden de la banda y no lleva τ propia (lib.rs:7835-7853). 4) La severidad alta sólo se sostiene por el sobre-veto de la banda rápida, y no está medida. QO-586 publica únicamente la última τ vetada (qo_586_tau_inoperable): no lleva contador ni mide el EV que bloquea. La parte lenta es desperdicio, no dinero: lo que deja pasar lo rechaza REJ_TP_SL_FLOOR.

### F3.3-01. conviccion_de_rama no es monótona en la evidencia y compara con ½ en vez de con el equilibrio de la geometría

- Bloque: F3.3, Confianza.
- Ubicación: crates/god-engine-core/src/lib.rs:141-150. Llamadas: lib.rs:6320, :6536, :6558, :6592 (Línea C); lib.rs:5987-6205 y :5736-5937 (Línea E).
- Causa: Hay tres estados con un umbral fijo de ½: - si lo > ½, devuelve lo; - si hi < ½, devuelve hi; - si no, devuelve piso_por_magnitud, que llega a 0,95 en las ramas 11 y 15 y a 0,90 con sig_conf. Una rama recién nacida recibe más confianza que una con ventaja demostrada. Y ½ no es el equilibrio: con TP = 2,25·SL y fricción ≈ 0,2·SL, p* = 1,2/3,25 ≈ 0,369.
- Consecuencia viva: Cada cierre ganador que lleva la cota de Wilson por encima de ½ puede bajar la confianza: con 21/30 cae de 0,75 a 0,521, por debajo del gate de 0,745. La rama queda vetada justo cuando demuestra ventaja. Al revés, una rama con WR 40-45 % y EV > 0 se condena (hi < ½) a n = 91 o n = 370 y pierde el tamaño.
- Prueba que falla hoy: T1 cf3_conviccion_monotona_en_la_evidencia (21/30 da 0,521 frente a 0,75 con 16/30) y T2 cf3_conviccion_no_condena_una_rama_sobre_su_equilibrio (400/180: lo 0,402 > p* 0,369 y la función devuelve 0,499).
  - Cerrado por CL-75 (2026-10-11) en la función y en las ramas 11, 13, 14 y 15. `conviccion_de_rama_en(registro, piso, p*)` compara el intervalo de Wilson con el equilibrio de la orden, p* = (SL + f)/(TP + SL) (`p_equilibrio`; es el umbral del veto EV del riesgo con multiplicador 1). Devuelve 0 si hi < p*, máx(lo, piso) si lo > p* y el piso en otro caso, así que es no decreciente en los aciertos (`cf3_conviccion_monotona_en_toda_la_rejilla`: n ≤ 200, tres pisos y cuatro p*). El núcleo calcula p* con la misma sonda de QO-586, a la τ de la orden (`p_equilibrio_de_la_orden`). `conviccion_de_rama(registro, piso)` usa p* = ½ para los llamadores de las ramas 1-10 y 20-23 (Línea E). Su pedido: pasar p* con `p_equilibrio_de_la_orden`. La T-2 pasa de 3c243afa79d0e3e0 (62 operaciones, PnL −0,259, acierto 39 %, 1 302 vetos EV) a a8d60390ff2f0f74 (68 operaciones, PnL −0,118, acierto 43 %, 899 vetos EV). El fixture no tiene ventaja, así que no se lee como rentabilidad.
- Dueño: Línea C: la función y las ramas 11, 13 y 15. Pedido a Línea E para los llamadores de las ramas 1-10 y 20-23.
- Relación con hallazgos anteriores: K-02 (ledger, pendiente). Añade dos pruebas ejecutables, el cálculo de n para condenar una rama rentable y que el test interno (lib.rs:9046-9048) sólo cubre la zona lo > ½.
- Escépticos: confirmado.
  - Corrección (refutar): Severidad media. Hay que decir que añade a K-02 las pruebas T1 y T2 y la observación de que el test interno (lib.rs:9046-9048) sólo cubre lo > ½. Precisiones: - Con frecuencias exactas, la primera n que condena una rama es 100 si gana el 40 % y 400 si gana el 45 %. Las cifras 91 y 370 son aproximaciones continuas. - p* ≈ 0,369 supone una fricción de 0,2·SL, y eso no está verificado. p* sólo queda por debajo de ½ si f/SL < 0,625. Como suelo_tp_sl admite f/SL hasta 0,65, en ese extremo ½ es laxo, no estricto. El criterio correcto es p* de cada orden, no «½ es demasiado alto».
  - Corrección (completitud): Severidad alta, pero condicionada. rama_registro no se persiste: se inicializa en lib.rs:1229 y su único escritor es lib.rs:3989. n se reinicia en cada proceso, así que el efecto vivo exige entre unos 30 y 90 cierres por rama y por moneda dentro de una misma sesión, y esa frecuencia no está medida. El mismo ½ (en lugar de p*) aparece en otros dos sitios, que corresponden a C-F4: - leverage_matrix.rs:135-139, con (prob_win − 0,5)·2, que sólo pesa en la τ lenta (:317-319); - el prior 0,55/0,45 de la sonda (risk lib.rs:1185). La prueba T2 debe reformularse (ver «pruebas»).

### F3.3-02. Una desventaja demostrada (hi < ½) se reinfla por encima del gate: el doc de D-752 es falso

- Bloque: F3.3, Confianza.
- Ubicación: Doc: lib.rs:131-133. Refuerzos: lib.rs:1639-1640 (ML ×1,5), lib.rs:7119 y :7141 (espectral ×1,35), lib.rs:6752-6753. Gate: crates/risk-engine/src/lib.rs:1092-1100.
- Causa: Tras conviccion_de_rama se aplican multiplicadores sin estado: - puerta ML (1+0,5·d) hasta ×1,5; - modulación espectral hasta ×1,35, con techo 0,98. El gate del riesgo compara con min_conf·epi_thresh (recortado a [0,75; 1,35]), no con ½.
- Consecuencia viva: Una rama con 80/200 aciertos (hi 0,469) sale con 0,704 sólo con el ML y con 0,95 con ML y espectro. Supera min_conf 0,70 y, tras 12 h sin operar, Platt devuelve casi la misma puntuación como p y pasa el EV. Una rama con 50/200 (hi 0,314, EV < 0 con p* 0,369) llega a 0,636, y abre si epi_thresh ≤ 0,909.
- Prueba que falla hoy: T3 cf3_desventaja_demostrada_no_se_reinfla: confianza_modulada(0,469; 1,35) = 0,633 ≥ ½. La parte del ML necesita extraer el impulso de puertas_del_continuo (privada) a una función pública.
  - Cerrado por CL-75 (2026-10-11): una desventaja demostrada (hi < p*) da convicción 0, así que ningún multiplicador posterior la reinfla (×1,10…×1,35 dan 0). Afecta también a las ramas 1-10 a través del envoltorio con p* = ½.
- Dueño: Línea C (puertas, modulación, superposición); el gate corresponde a C-F4.
- Relación con hallazgos anteriores: K-12 y el plan §3 #43 (re-inflado). Añade que el doc de lib.rs:131-133 es falso, la cuenta 0,469 → 0,704 → 0,95 y el caso con EV < 0.
- Escépticos: confirmado.
  - Corrección (refutar): Severidad media en vivo. Dejar como aportación propia la falsedad del doc, la cuenta presentada como cota superior y el caso 50/200 (hi 0,3143 → 0,636 como máximo). Citar el §3 fila 43 como origen.
  - Corrección (completitud): La cadena del auditor omite dos factores vivos que se aplican entre la puerta y Platt: - el freno del bosque (lib.rs:6891-6893), que puede recortar; - el modulador de contagio, ×[0,70; 1] (lib.rs:7155-7162, F2.4-08). Con ellos, el resultado depende del estado; la reinflación sigue siendo posible, pero no es determinista. El rango de epi_thresh también está mal: el escritor lo acota a [0,80; 1,25] (crates/quantum-arena/src/state.rs:410-413), y el lector, que recorta a [0,75; 1,35], no llega a ver 0,75. El gate mínimo efectivo es 0,56 (0,70 × 0,80) o 0,596 en el régimen micro, no 0,525 ni 0,559. El paso al EV se apoya en LC-F1-37: Platt es la identidad en frío y tras cada reinicio, porque los calibradores no se persisten (se crean en lib.rs:1202-1208 y no hay cargador). Hay que citarlo.

### F3.6-01. El solitón y el choque votan siempre LARGO en el consenso escalar: su entrada de «velocidad» es |Δmid| sin signo

- Bloque: F3.6, Motores de señal.
- Ubicación: soliton_wave.rs:149 (lee price_velocity), :34-38 (|amplitud|), :204-218 (salida); supersonic_shockwave.rs:129-144, :44-55, :224-233; escritor stateful_engine.rs:738, :784-788 y lib.rs:4503-4504; registro en lib.rs:1014-1079; consumidores orchestrator.rs:337-614, lib.rs:5424-5446, ramas 11 (lib.rs:6216-6330) y 14 (lib.rs:6642-6678)
- Causa: F2.6-01 dejó sin leer cómo usa el signo cada lector de price_velocity. Lo leí: - Solitón: toma |OFI| como amplitud y la multiplica por tanh(norm_vel·1e4), con norm_vel = dir_v/mid·10 ≥ 0. Su salida es ≥ 0. - Choque: speed = dir_v ≥ 0, el salto está en [0,1] y vale 0 con M ≤ 1, y tanh(speed/sound/2) ≥ 0. Las dos salidas son funciones pares del mercado reflejado y nunca negativas. Los respaldos a price_velocity de coaxial_breakout.rs:214 y trend_runner.rs:214 nunca se alcanzan, porque order_flow_direction (lib.rs:4512) y trend_direction (lib.rs:5148-5160) se escriben siempre.
- Consecuencia viva: Cuando decide el consenso escalar (arranque, espectro plano o τ_dom < 30 s; orchestrator.rs:375-384): - el solitón suma ≈ +|OFI|, por ejemplo +0,30 con OFI = −0,30, en contra del flujo vendedor; - el choque suma hasta +0,76 en cualquier salto, también en una caída. Como net = Σv/Σ|v|, eso quita convicción a los cortos y se la da a los largos, y llega a tensor_boost, al compuesto y a las ramas 11 y 14. La fracción de eventos en respaldo escalar no está medida; la miden los contadores del censo #710.
- Prueba que falla hoy: cf3_6_espejo_soliton_escalar y cf3_6_espejo_choque_escalar (crates/signal-engine/tests/cf3_6_espejo_motores.rs). El solitón da v(+) = +0,3922 y v(−) = +0,3921; el choque da +0,5255 en los dos lados. Los dos fallan hoy.
  - Cerrado por CL-73 (2026-10-11): el productor `dir_velocity` (stateful_engine.rs) promedia el paso del mid con su signo; la rapidez sin signo sigue en `last_inst_v` y en `a_t`. El solitón y el choque no cambian de código: con la velocidad firmada son impares. Las dos pruebas se activaron con el registro espejo del escritor nuevo (`price_velocity` = ±0,5): solitón ±0,3922 y choque ±0,5255. Golden y T-2 idénticos; T-1 en el PR.
- Dueño: Línea C (motores de signal-engine, CL-n). Si el arreglo cambia el escritor compartido dir_velocity (stateful_engine.rs:788), avisar a quien lo lea en otras zonas.
- Relación con hallazgos anteriores: F2.6-01 (raíz). Este hallazgo cierra el faltante de lectores que dejó C-F2.
- Escépticos: confirmado.
  - Corrección (refutar): Hay cuatro correcciones. 1. Magnitud del solitón: depende del tamaño relativo del tick. El argumento es 1e5·dir_v/mid, así que el factor satura con unos 0,3 pb por evento. En el roster con modelo (models/ sólo tiene ATOM, BNB y NEAR, con ticks de 0,2 a 2 pb) el voto queda ≈ |OFI|·sech(|OFI|·pos). En BTC (0,017 pb por tick) el voto es bastante menor. El «+0,30 con OFI = −0,30» es el caso saturado. 2. Severidad del choque: M > 1 exige que la EWMA por evento de |Δmid| supere máx(spread, v_t), donde v_t ≥ 10 pb del precio (stateful_engine.rs:880). Con un peso de 0,15, eso pide un salto de más de ~6,7 ATR de 1 min en un solo evento. El voto del choque es 0 casi siempre y sólo vota largo en crashes. Lo propongo como alta para el solitón y media para el choque. La fracción la mide censo_no_cero_SupersonicShockwaveEngine. 3. Antecedente sin citar: FMT-001 (docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_2026-09-24.md:67-69). C-F2 asignó el productor dir_velocity al dueño de stateful_engine: el arreglo raíz va con F2.6-01 y no se puede hacer sólo en los motores. 4. Las citas del choque se han movido: la lectura de order_flow_velocity está en :136 y la salida en :233.
  - Corrección (completitud): Hay que añadir la prueba de espejo a nivel de consenso (véase el faltante 1), porque la de motor a motor no muestra el efecto en el umbral. También hay que citar la interacción con el piso escala-invariante de 0,70 (ensemble_characterization.rs:37).

## Severidad media

### F3.1-04. El TTL del veredicto espectral no tiene techo: con τ_dom ≥ 19,5 h el dominante sobrevive a la vida que fija el orquestador, hasta ~146 años

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/signal-engine/src/skill_motores.rs:262-274; crates/god-engine-core/src/lib.rs:1785-1804, :2564-2569; crates/signal-engine/src/orchestrator.rs:576-587
- Causa: ttl_consenso_ms = max(τ_dom, 30 s), sin acotarlo por TAU_ANCHOR_SLOW_MS (skill_motores.rs:268-274). La vida, en cambio, es min(τ_dom, 12 h) (orchestrator.rs:576-587): los dos usan τ distintas. La caducidad compara con consenso_tau_ms, la τ cruda (lib.rs:1785-1804). El test qo_648_ttl_consenso sólo prueba hasta 43 200 000 ms (skill_motores.rs:477-483).
- Consecuencia viva: Si se cae el stream de depth y siguen llegando trades, un dominante con k ≥ 23 (7,0·10⁷ ms ≈ 19,5 h) sigue dirigiendo el tensor en cada trade (orchestrator.rs:365-384) más allá de 12 h; con k = 31, indefinidamente. Hoy sólo llega a esas escalas un dominante no observado (F3.1-02), así que el daño depende de ese hallazgo.
- Prueba que falla hoy: cf3_ttl_del_veredicto_no_supera_la_vida_que_fija_el_orquestador (crates/signal-engine/tests/cf3_ttl_consenso.rs). Para cada τ de SPECTRUM_SCALES_MS, ttl ≤ max(min(τ, 12 h), 30 s). Predicción: falla con certeza en k = 23.
- Dueño: Qoder (consenso; skill_motores).
- Escépticos: confirmado.
  - Corrección (refutar): Su daño vivo exige dos condiciones a la vez: - que se caiga sólo el stream de depth mientras siguen los trades (el orquestador sólo se evalúa en trades); - que el dominante esté en k ≥ 23, lo que hoy sólo ocurre por F3.1-02. Mantengo media como cota superior; si F3.1-02 se arregla primero, baja. Para el arreglo, usar la misma τ acotada que la vida, min(τ_dom, TAU_ANCHOR_SLOW_MS), y no otro literal.
  - Corrección (completitud): Bajar la severidad a baja. Hacen falta tres cosas a la vez: - que el depth lleve caído más de 12 h mientras siguen llegando trades; - que el dominante esté en k ≥ 23; - que no haya reconexión del host (no verifiqué si el watchdog WS reconecta por stream). Si se arregla F3.1-02, k = 23 sólo puede ser dominante tras 19,5 h de proceso. La prueba sigue siendo válida como propiedad de «una sola τ».

### F3.1-05. La sombra espectral del solitón rompe el espejo: con OFI +0,5 la pendiente es 0,5 y con −0,5 es 1,0. El comentario «ESPEJO EXACTO» es falso y un test lo deja fijado

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/god-engine-core/src/lib.rs:2087-2124; crates/signal-engine/src/soliton_wave.rs:27-37, :81-97; crates/god-engine-core/tests/sombras_espectrales_telemetria_contract.rs:131-137
- Causa: lib.rs:2097-2109 pasa como amplitud el OFI con su signo. voto_espectral la convierte en 1,0 si es ≤ 0 y la usa tal cual si es > 0 (soliton_wave.rs:81-86). El motor vivo usa |amplitud| (compute_soliton_amplitude, soliton_wave.rs:33-37), así que no depende del signo del OFI. El test h2_10 (sombras_espectrales_telemetria_contract.rs:131-137) afirma que amp ≤ 0 → 1,0 «preserva la semántica del vivo», y no es así.
- Consecuencia viva: El solitón es uno de los 13 votos del consenso que consume el orquestador. Con flujo comprador moderado (0 < OFI < 1) su voto pesa menos que con flujo vendedor de la misma magnitud, sobre el mismo camino de precios. Es un sesgo de magnitud que depende del lado del flujo, sin medir.
- Prueba que falla hoy: cf3_el_soliton_espectral_es_par_en_el_signo_del_ofi (crates/god-engine-core/tests/cf3_consenso_espectral.rs). - Parte pura: voto_espectral(x, 0,5) frente a voto_espectral(x, −0,5), usando el motor vivo como referencia de paridad. - Parte integrada: dos núcleos sobre la misma caminata, con OFI +0,5 y −0,5. Predicción: falla con certeza en la parte pura.
- Dueño: Qoder (sombra espectral del consenso). El arreglo debe invertir la última aserción de h2_10.
- Escépticos: confirmado.
  - Corrección (refutar): 1. El sesgo es direccional, no sólo de magnitud. El OFI es una EWMA de ofi_norm recortado a [−10; 10] (feature-engine/src/microstructure.rs:171 y :188-190), no a [−1, 1] como dice el comentario de lib.rs:2105-2107. - Con 0 < OFI < 1 (lo típico), la pendiente es OFI < 1. - Con OFI ≤ 0, la pendiente es 1. Como el flujo y el desplazamiento del precio están correlacionados en positivo, en subidas con flujo comprador el solitón vota débil, y en bajadas con flujo vendedor vota fuerte. El resultado es un sesgo corto en |consenso|. 2. Con OFI > 1 la asimetría se invierte. 3. El voto que se abstiene (OFI ≈ 0⁺, pendiente ≈ 0) sigue contando en Σw y diluye el consenso.
  - Corrección (completitud): No es sólo un sesgo de magnitud: tiene dirección. - Con flujo que confirma la tendencia, el voto largo (OFI ∈ (0,1)) queda atenuado a tanh(OFI·x). - El corto con OFI < 0 vota tanh(x) entero. El resultado es un sesgo corto sistemático en la magnitud del consenso. El arreglo con |OFI| abre además una discontinuidad en 0: con OFI = 1e-3 la amplitud vale 1e-3 y el voto ≈ 0, mientras que con OFI = 0 la amplitud vale 1,0. El vivo también vale 0 en 0, pero por otro camino. El pedido a Qoder debe fijar la continuidad en 0, no sólo la paridad.

### F3.1-06. La concentración espectral usa N = 32 fijo y mide falta de observación, no estructura: en ruido blanco vale ≈ 0,9 y no amortigua el voto de la proyección

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/god-engine-core/src/lib.rs:4586-4618; crates/signal-engine/src/proyeccion_espectral.rs:10-21, :65-82; crates/quantum-arena/src/temporal_spectrum.rs:1389-1424
- Causa: concentración = 1 − (Σw)²/(32·Σw²) (lib.rs:4610-4611), con w = pesos_espectrales = masa × resolución / vol. En una caminata aleatoria: - 1/vol cae ≈ ×½ por cada escala observada (vol ∝ √τ, con razón 4 entre escalas); - la masa cae ×¼ por cada escala no observada; - la resolución corta las escalas de abajo. Por eso la razón de participación queda en ≈ 3-4 de 32 (estimación por lectura) y la concentración ≈ 0,88-0,9, aunque el precio sea ruido. El doc promete «0 = masa difusa (ruido blanco)» (proyeccion_espectral.rs:12-13).
- Consecuencia viva: ProyeccionEspectral vota senal·conc (proyeccion_espectral.rs:81). En ruido el amortiguador casi no actúa y el voto queda en ≈ 0,9 veces la media ponderada de tanh(momentum_z), el mismo dato que el veredicto espectral. Es uno de los 19 votantes escalares. Por F3.1-03 su magnitud pesa poco, pero su signo sí cuenta.
- Prueba que falla hoy: cf3_la_concentracion_espectral_de_un_ruido_es_baja (crates/god-engine-core/tests/cf3_consenso_espectral.rs). Tras 3 600 eventos de caminata, espectral_concentracion ≤ 0,2. Predicción: falla casi con certeza; con una razón de participación ≤ ~4, la concentración es ≥ 0,87.
- Dueño: Qoder (espectro).
- Escépticos: confirmado.
  - Corrección (refutar): 1. La causa está mal descrita. pesos_espectrales NO es masa × resolución / vol. Es fusion_weight × observable (temporal_spectrum.rs:1389-1396), y fusion_weight = clamp(2·|persistencia|·ganancia; 0,02; 3) (temporal_spectrum.rs:247-257) no contiene 1/vol. El 1/vol está en el segundo vector de pesos_observables, el de la fusión, no aquí. Lo que fija la concentración alta es N = 32 frente a unas 6-10 escalas observables, más los pesos aleatorios por |persistencia|. Hay que corregir también la fila «Proyección espectral» del mapa. 2. La consecuencia viva es dudosa. El voto se abstiene si espectral_masa_resuelta < 0,25 (proyeccion_espectral.rs:29 y :66-68), y masa = Σ fw·obs / Σ fw. Las 15 escalas por debajo de la resolución tienen fw ≈ 0,29 (persistencia de retornos por evento) y observable ≈ 0. Por lectura estimo masa ≈ 0,27 a 1 evento/s y 1 h, en el borde. En ruido el voto puede abstenerse a menudo; no está medido. 3. Severidad: baja-media. La concentración nunca cambia el signo del voto, y por F3.1-03 en modo escalar casi sólo cuentan los signos.
  - Corrección (completitud): Causa correcta: - Las escalas sin bloque maduro pesan el piso 0,02 × masa. - Las resueltas y maduras pesan |persistencia|, que en ruido es aleatoria. - Así, la masa recae en unas 6 escalas (k = 15..20 tras 1 h a 1 ev/s), y 1 − PR/32 ≥ 0,81 aunque esas 6 pesen igual. Se añade que fusion_weight ignora el signo de la persistencia: las escalas que revierten pesan como las persistentes y votan momento en espectral_senal_proyectada. Es la misma clase que F2.3-03, extendida a la proyección (lib.rs:4586-4618).

### F3.1-07. Un sesgo spot «simétrico» mueve nn_score 2,3 veces más hacia abajo que hacia arriba cuando la base del bosque es 0,30

- Bloque: F3.1, Independencia de las fuentes y consenso espectral.
- Ubicación: crates/god-engine-core/src/lib.rs:4442-4447, :4867, :4937-4938; crates/god-engine-core/src/calibration.rs:285-289; crates/god-engine-core/src/stateful_engine.rs:572-584
- Causa: spot_bias ∈ [−0,05, +0,05] se suma a la probabilidad (compose_ml_prob, calibration.rs:285-289) ANTES del mapeo de CL-13. Ese mapeo normaliza cada lado por su propio recorrido: d/(1 − b) hacia arriba y d/b hacia abajo (stateful_engine.rs:572-584). | Base b | +0,05 | −0,05 | |---|---|---| | 0,30 | +0,071 | −0,167 | | 0,18 | +0,061 | −0,278 | Sólo con b = 0,5 el efecto es simétrico.
- Consecuencia viva: Con w_nn = 0,369 (genoma base), una base spot-futuros de ±10 pb desplaza el compuesto +0,026 o −0,062. Es un sesgo corto en el compuesto, cuyo signo vetan las puertas (lib.rs:1610-1611). Sólo afecta a los símbolos con bosque cargado (base 0,18-0,30 según CL-21). LC-F1-22 trata la pendiente del sesgo; esto es su asimetría tras CL-13.
- Prueba que falla hoy: cf3_un_sesgo_spot_simetrico_mueve_nn_score_simetricamente (crates/god-engine-core/tests/cf3_sesgo_spot_espejo.rs). Predicción: falla con certeza en b = 0,30.
- Dueño: Codex (calibración: compose_ml_prob), junto con el autor de CL-13. El punto de aplicación está en lib.rs, fuera de las ramas 1-10.
- Relación con hallazgos anteriores: LC-F1-22
- Escépticos: confirmado.
  - Corrección (refutar): 1. Falta un segundo consumidor vivo con la misma asimetría: F-009 en puertas_del_continuo. Usa normalized_directional_divergence (calibration.rs:331-342; lib.rs:1628-1640), el mismo mapeo por semi-intervalos. Con b = 0,30, un sesgo de −0,05 penaliza un largo ×0,917, y uno de +0,05 penaliza un corto ×0,964. 2. El defecto es más general que el sesgo spot. Para cualquier perturbación simétrica de p, y en particular para un bosque calibrado con E[p] = b < 0,5, sale E[nn_score] = E[(p − b)⁺]·(1/(1 − b) − 1/b) < 0. Es un sesgo corto residual del propio mapeo de CL-13. 3. Justificación de la propiedad: el EV del largo es lineal en p, así que ±δ en p mueve el EV de forma simétrica. 4. Peso vivo con el genoma activo: w_nn = 0,12, no 0,369. Un sesgo de ±0,05 desplaza el compuesto +0,009 o −0,020 (b = 0,30).
  - Corrección (completitud): La consecuencia está subestimada: el sesgo no es ocasional. spot_bias = clamp(spread_bps/10, ±0,05) (lib.rs:4444-4447) satura con |base| ≥ 0,5 pb (LC-F1-22), así que con cualquier base distinta de 0 empuja el compuesto casi siempre: −0,062 con base negativa y +0,026 con positiva, con b = 0,30 y w_nn = 0,369. La asimetría es la de CL-13 aplicada a una perturbación aditiva en p. El arreglo es sumar el sesgo en espacio de lift, o como término propio del compuesto, no tocar CL-13.

### F3.2-02. El veto de grupo lee el ρ de una escala que no es la de la orden; la puerta del plan está mal especificada

- Bloque: F3.2, Una sola τ.
- Ubicación: risk-engine/src/lib.rs:676-704 y :871; lib.rs:1775-1780, :2924-2929 y :3967-3972; god-engine-core/src/evidence_publication.rs:56-77
- Causa: La clave «qo_613_rho_tau» no lleva τ. El núcleo la publica a la τ espectral (la unificada o la cruda). El lector la aplica a cualquier orden sin compararla con tau_for_sizing, que es la media geométrica de la τ de rama con el centroide. Aunque se arregle F2.3-02 y se publique una sola vez a la τ unificada, la orden se dimensiona en otro nodo ×4 casi siempre, porque el valor mezclado cae entre la micro y el centroide.
- Consecuencia viva: El veto de grupo aprieta con un ρ maduro de una escala que no es la de la orden. Y no aprieta cuando la escala de la orden tiene ρ alto pero la publicada no lo tiene. El sesgo no está medido.
- Prueba que falla hoy: cf3_rho_del_veto_de_grupo_es_el_de_la_tau_de_la_orden (crates/god-engine-core/tests/cf3_rho_tau_orden.rs). El multiactivo está maduro en la escala A = 19 (IC 1) y no en B = 20 (None). Se publica a SPECTRUM_SCALES_MS[A] y se evalúa una orden con expected_duration_ms = SPECTRUM_SCALES_MS[B]. Esperado: qo_613_aprietes = 0. Hoy: 1. Control: una orden a la τ de A da 1, y pasa.
- Dueño: Línea C: publicación del núcleo y lector del risk-engine.
- Escépticos: confirmado.
  - Corrección (refutar): - «Casi siempre en otro nodo ×4» no está verificado: depende de τ_banda/τ_c. Hay que suavizarlo. - Hay que citar LC-F3-01 del plan (PLAN:1065), que es la mitad de publicación, junto con F2.3-02. - El arreglo choca con un test vivo. mg05_real_risk_reader_stops_tightening_on_cold_b_and_resumes_on_a (evidence_expiry_contract.rs:386-411) aprieta una orden de 60 s con el IC de τ_A = 274,9 s: certifica hoy la conducta que el hallazgo llama defecto, y habrá que reescribirlo con la orden a τ_A. - El lector por escala debe usar distancia en ln τ, no la distancia absoluta de F2.3-02.
  - Corrección (completitud): No hay cambio de fondo. Una precisión para el caso multiactivo: aunque el IC se publicara por escala, la base ρ_escalar con la que se compara sale de dependency_exposure. Esa función no tiene parámetro τ y mide ρ sobre 512 ticks del anillo (correlation_guard.rs:48, :586-593, :631). La comparación seguiría mezclando escalas; lo detallo en los faltantes.

### F3.2-03. R-15 sigue penalizando las τ largas por encima de 22 minutos

- Bloque: F3.2, Una sola τ.
- Ubicación: risk-engine/src/lib.rs:950-972 (recorte en :955, veto en :969); risk-engine/src/tp_sl.rs:469-489
- Causa: tau_ratio = √(30 s/τ) recortado a [0,15; 1]. Con brackets ∝ √τ (la ley browniana de la geometría con H = 0,5), θ·tp = 2μ·tp/σ² es constante sólo mientras √(30 s/τ) ≥ 0,15, es decir, con τ ≤ 1 333 s. Por encima vuelve la acumulación ∝ √τ que R-15 decía retirar. Además, 30 s es un ancla y no una decorrelación medida de la coherencia.
- Consecuencia viva: Con un desacuerdo espectral leve (−0,3), una orden a 4-12 h se veta con REJ_TARGET_GEOMETRY y la misma orden a 5-20 min pasa: P = 0,762 hasta 1 333 s, 0,883 a 4 h y 0,953 a 12 h, frente al umbral de 0,88. Es lo que el ledger declara eliminado.
- Prueba que falla hoy: cf3_r15_primer_toque_invariante_en_tau_con_brackets_brownianos (crates/risk-engine/tests/cf3_r15_tau.rs). Se lee p_hit_sl_first del registro a τ = 300 s, 1 200 s, 4 h y 12 h, con sl = 0,5·ATR·√(τ/60 s) y tp = 2,25·sl. Esperado: P idéntica a la de 300 s (≈ 0,762), con |Δ| < 1e-9. Hoy da 0,883 y 0,953 a 4 h y 12 h.
- Dueño: Línea C (compuerta de primer toque del risk-engine). El autor anterior es AGY (Ola Ω73) y conviene avisarle.
- Relación con hallazgos anteriores: R-15 (ledger de Línea E, declarado RESUELTO)
- Escépticos: confirmado.
  - Corrección (refutar): - Dueño: según el plan es C-F4 (§3 fila 5, «mal especificado: la deriva es una coherencia y σ es un rango»; §2, PLAN:97, LC-F4-04, R-15 a sombra). - «Duplica» debe citar §3 fila 5, LC-F4-04 y el ledger §7.3 (:281-287), que ya lista los literales 30 s, 0,15 y 0,88. Lo nuevo es el umbral de 1 333 s y el test determinista. - La invariancia esperada es la promesa del propio R-15 (ledger :135, «Eliminado el castigo artificial a escalas largas»), no una verdad física. - Con H ≠ 0,5 los brackets van como τ^H y el factor √(30 s/τ) tampoco es invariante por debajo de 1 333 s.
  - Corrección (completitud): Tres precisiones sobre el ledger y el modelo: - Ningún test certifica τ largas. test_r7_r2_a1 (crates/risk-engine/tests/r7_r2_ruin_chaos_contract.rs:119-200) usa sólo τ = 60 s y mira el contador REJ_TARGET_GEOMETRY, que también suben los rechazos de geometría anteriores. - La tabla de cierre del ledger (:204) cita prospect_pressure_integration_contract.rs, que no lee p_hit_sl_first. Un grep sólo lo encuentra en risk-engine/src/lib.rs:968. - La cita del ledger (:135, «lib.rs:937-954») está desfasada: hoy es :950-972. Además, la σ del primer toque es el ATR de 1 min (:964), mientras el TP y el SL usan σ(τ). La invariancia sólo es exacta con H = 0,5.

### F3.2-04. El Kelly de la moneda lleva la τ de la operación cerrada a la siguiente orden

- Bloque: F3.2, Una sola τ.
- Ubicación: lib.rs:4179 y :4205-4218; risk-engine/src/lib.rs:387-411, :521-528 y :576-586; risk-engine/src/leverage_matrix.rs:327
- Causa: strategy_base = kelly_at_tau(tau_trade_ms) entra a calculate_kelly_fraction y se guarda en metrics.kelly_fraction, un escalar por moneda sin τ. El riesgo lo lee como raw_kelly para una orden con otra τ y le aplica además el normalizador sl(30 s)/sl(τ nueva). Resultado: la curva de Kelly se evalúa a la τ anterior y la normalización de riesgo a la τ nueva. Con una curva genómica creciente (b > 0), una operación larga cerrada infla el tamaño de la siguiente operación corta, y al revés.
- Consecuencia viva: El tamaño depende del horizonte de la operación anterior. Kelly entra dos veces (raw_exposure y capital_factor del apalancamiento), así que el sesgo pasa también al apalancamiento. Predicción para el escenario del test, sin ejecutar: unos 0,060 frente a 0,109.
- Prueba que falla hoy: cf3_kelly_de_la_moneda_no_depende_de_la_tau_de_la_operacion_cerrada (crates/god-engine-core/tests/cf3_tau_aprendizaje.rs). Dos núcleos iguales con la curva de Kelly 0,25 a 60 s y 0,6 a 4 h. Uno cierra una posición con entry_tau de 60 000 ms y el otro con 14 400 000 ms, los dos a 101,1. Esperado: el mismo kelly_fraction (|Δ| < 1e-12). Hoy son distintos.
- Dueño: Línea C
- Relación con hallazgos anteriores: LC-F1-14 (lo amplía: allí sólo aparece spectral_conf con el nodo crudo)
- Escépticos: refutado (duplicado).
  - Lente refutar, refutado: No es un hallazgo nuevo: es LC-F1-36, «El Kelly que dimensiona una orden sale del cierre anterior» (HALLAZGOS_C-F1:474-482; C-F4, disputado), con las mismas cifras (0,060 frente a 0,108). No es LC-F1-14, que sólo trata spectral_conf. Además, la consecuencia «Kelly entra dos veces y el sesgo pasa al apalancamiento» es falsa. - leverage_matrix calcula su propio Kelly con kelly_from_pf(prob_win, pf) (leverage_matrix.rs:116-132), y capital_factor (:327) no lee metrics.kelly_fraction. - El único lector vivo de kelly_fraction es risk-engine/src/lib.rs:387-391 (rg kelly_fraction). - strategy_base sólo entra como survival_coeff = clamp(0,20; 0,75) en modo supervivencia (kelly.rs:85-90). En expansión no influye. - Al consumir, el clamp [kelly_clamp_min, kelly_clamp_max] (risk-engine/src/lib.rs:411-415) puede borrar la diferencia.
  - Corrección (refutar): Hay que retirarlo de F3.2. El test de núcleo puede quedar como testigo de integración de LC-F1-36 en C-F4, con la precondición de modo supervivencia explícita y sin la frase sobre el apalancamiento. El objetor de LC-F1-36 ya señaló que D-745 se cumple literalmente: el Kelly del cierre usa la τ de entrada.
  - Corrección (completitud): El alcance es más estrecho de lo que se dice. strategy_base sólo entra como survival_coeff, recortado a [0,20; 0,75], en el modo de supervivencia, es decir, cuando capital_ratio < kelly_survival_cap_ratio (risk-engine/src/kelly.rs:81-89). En el modo de expansión (:90-95) la τ de la operación cerrada no influye. La severidad media se mantiene, pero sólo para ese modo. La predicción del test se sostiene: base_capital vale el capital inicial (quantum-arena/src/config.rs:205), el cociente es ≈ 1 < 10 y el sistema está en supervivencia.

### F3.2-05. La τ de la señal no es la τ de la intención en las ramas 11, 12 y 13

- Bloque: F3.2, Una sola τ.
- Ubicación: lib.rs:2551-2557 y :6300-6323 (rama 11); signal-engine/src/flow_impulse.rs:203-213 (rama 12); lib.rs:6460-6466, :6540 y :6562 (rama 13); signal-engine/src/orchestrator.rs:561-587; execution-engine/src/router.rs:74-77
- Causa: Cada una de las tres ramas emite a una τ distinta de la escala en que se tomó su decisión: - Rama 11: decide la dirección con el consenso a τ_k, un nodo ×4 dentro de la banda, y viaja a micro τ ≤ 30 min. El contrato #624 («vive a la escala que habló») se escribe en expected_lifetime_ms, cuyo único lector es QuantumOrderRouter, sin consumidor vivo. - Rama 12: su τ es el gen base_duration_ms, de al menos 15 s, y no tiene relación con el espectro. - Rama 13: la dirección sale de EMAs 9/21 sobre velas de 1 min (unos 9-21 min), pero viaja con la macro τ, que va del centroide a 12 h.
- Consecuencia viva: El TP/SL y la vida de la posición se fijan a una escala en la que la señal no ha medido habilidad. La convicción de rama (Wilson) mezcla resultados de τ distintas.
- Prueba que falla hoy: cf3_la_rama_11_vive_a_la_escala_que_hablo (crates/god-engine-core/tests/cf3_una_sola_tau.rs). Cuando last_fast_intent[0] es de la rama 11 (volume_flow_rate 11), consenso_espectral_dominante ≠ 0 y consenso_espectral_tau ≥ 30 s, debe cumplirse expected_duration_ms = min(τ_k, 12 h). Hoy da fast_duration_ms.
  - Nota de CL-74: tras CL-74 la rama 11 sigue viajando a la micro τ (128/128 intenciones), y la τ de la orden es ahora √(H(τ)·τ_c), con τ_c el centroide resonante. Para cerrar F3.2-05 hay que decidir antes si la convergencia hacia el centroide (#565) se aplica también a la rama 11: el contrato #624 dice que la posición vive a la escala que habló.
- Dueño: Línea C (ramas 11-13 del núcleo y orchestrator). La rama 12 vive en signal-engine.
- Escépticos: confirmado.
  - Corrección (refutar): - La parte de la rama 11 ya es LC-F3-04 del plan (PLAN:1070); hay que citarla. Lo nuevo son las ramas 12 y 13. - La rama 14 que cita LC-F3-04 está contenida (CONSENSUS_BRANCH_ENABLED = false, lib.rs:8603). - Tras la media geométrica el desajuste se reduce a la mitad en ln τ, pero no desaparece. La severidad media es razonable.
  - Corrección (completitud): Añado dos cosas: - En la fusión constructiva, la τ la decide la densidad de energía (lib.rs:6764-6768) y la etiqueta de rama la decide la mayor confianza (etiqueta_fusion_constructiva, :6774-6779). Una rama puede quedarse con el resultado de una operación hecha a la τ de la otra banda. El Wilson de rama mezcla τ también por este camino. - La rama 12: el genoma activo lleva base_duration_ms = 115 059 ms (config_dir/genomes/demo/active.json:129), pero la banda del gen es [10 000; 120 000] (genome.rs:1935). Con el mínimo de 15 s, QO-586 puede juzgar una τ por debajo de TAU_ANCHOR_FAST_MS que la mezcla nunca deja usar (:7182-7184).

### F3.2-06. Una posición sin τ de decisión entrena el espectro, el enfriamiento y el Kelly con el nodo crudo

- Bloque: F3.2, Una sola τ.
- Ubicación: lib.rs:3291-3304, :3916-3922, :3960-3964 y :4179; contraste en lib.rs:3974-3983
- Causa: tau_trade_ms vale entry_tau_ms si es mayor que 0. Si no, toma spec.dominant_tau_ms del cierre o el gen. Con esa τ se ejecutan: - record_trade_outcome (enfriamiento); - apply_spectral_epigenetic_feedback_with_time; - apply_epigenetic_outcome (ganancia de las 32 escalas, con núcleo gaussiano σ = 0,8 en ln τ); - kelly_at_tau. Sólo los modelos exigen la evidencia congelada en la apertura (learning_evidence).
- Consecuencia viva: Las posiciones adoptadas tras un reinicio sin diario (entry_tau 0, LC-F1-15) premian o castigan escalas que no decidieron la entrada. Con el espectro frío, la escala es la de 30 000 ms. No se ha medido con qué frecuencia ocurre.
- Prueba que falla hoy: cf3_posicion_sin_tau_de_decision_no_entrena_el_espectro (crates/god-engine-core/tests/cf3_tau_aprendizaje.rs). Posición manual con entry_tau 0, cerrada a 97 en IsolatedSimulation. Esperado: epigenetic_gain de las 32 escalas sin cambios. Hoy, las escalas cercanas al nodo crudo bajan.
- Dueño: Línea C (camino de cierre). La adopción corresponde a C-F5.
- Relación con hallazgos anteriores: LC-F1-15 (lo amplía: allí sólo se trata la gestión)
- Escépticos: confirmado.
  - Corrección (refutar): Ninguna de fondo. Precisión: «con el espectro frío la escala es la de 30 000 ms» sólo vale después del primer update; antes, el campo vale 0 y cae al gen.
  - Corrección (completitud): Tres precisiones: - Con un espectro sin actualizar, dominant_tau_ms vale 0 (temporal_spectrum.rs:375). La τ cae entonces al gen temporal_scale (lib.rs:3297-3303), no a 30 000 ms. - Con entry_tau = 0, tau_ultimo_cierre_ms no se actualiza (lib.rs:3905-3907). El siguiente enfriamiento conserva la τ de la posición anterior. - Lo más relevante es que ese aprendizaje mueve epigenetic_gain, y de ahí fusion_weight, pesos_espectrales, el centroide y las τ micro y macro. Por esa vía cambia la τ de las órdenes siguientes (faltante 6).

### F3.3-03. La rama 11 mide su convicción a τ_dom pero se dimensiona a la micro τ: expected_lifetime_ms del orquestador no tiene lector

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:6318-6323 (expected_duration_ms: fast_duration_ms) y lib.rs:5472-5478. orchestrator.rs:576-587.
- Causa: El orquestador devuelve la vida de la posición a la escala que habló (#624: tau_dom recortado a 12 h). El núcleo descarta ese campo (un grep de 'expected_lifetime_ms' en lib.rs no da resultados) y pone fast_duration_ms = micro_resonant_tau recortado a [30 s; 30 min].
- Consecuencia viva: Un voto dominante a 1 h abre con una τ de 30 s a 30 min. TP, SL, risk_normalizer y la puerta de banda QO-586 se calculan a una escala en la que la señal no habló, lo que rompe el contrato #624/H2 descrito en orchestrator.rs:375-384.
- Prueba que falla hoy: T10 cf3_rama_11_vive_a_la_escala_que_hablo (test de fuente: lib.rs no contiene 'tensor_cont.expected_lifetime_ms').
- Dueño: Línea C.
- Relación con hallazgos anteriores: En los documentos leídos (plan C-F3, C-F1, borrador C-F2 y ledger) no encontré un id previo para este emparejamiento de τ.
- Escépticos: confirmado.
  - Corrección (refutar): Marcar como duplicado de LC-F3-04. Lo que añade es que el único lector, router.rs:74, no es vivo, y que la τ final la deciden K-10, K-12 y LC-F3-09, no la rama.
  - Corrección (completitud): Arreglar sólo la rama 11 no consigue «una sola τ». Después de la puerta QO-586 (lib.rs:1599-1608, que mide con la τ que lleve la intención en ese momento), la τ se reescribe dos veces: - hurst_duration_modulation (lib.rs:6830-6833); - la media geométrica con la τ resonante (lib.rs:7166-7186; K-10 y K-12). El arreglo tiene que coordinarse con esos dos y con la observación de LC-F1 sobre las tres τ que conviven. La prueba de fuente T9 es frágil (ver «pruebas»).

### F3.3-04. El sesgo spot rompe el espejo de la puerta ML: el corto recibe hasta 4,5 veces más impulso que el largo

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:4442-4447 (spot_bias) y lib.rs:4870. calibration.rs:285-289 (compose_ml_prob). lib.rs:1629-1640.
- Causa: compose_ml_prob suma s en unidades de probabilidad: (p+s).clamp(0,1). La puerta normaliza la divergencia por (1−b) arriba y por b abajo. Para el mismo |s|, d_largo = s/(1−b) y d_corto = s/b.
- Consecuencia viva: Con b = 0,18 y s = ±0,05 (tope del sesgo), el largo recibe ×1,0305 y el corto ×1,1389. Con b = 0,215 (ATOM) son ×1,0318 y ×1,1163. El sesgo spot favorece sistemáticamente a los cortos en la confianza que llega al tamaño.
- Prueba que falla hoy: T5 cf3_sesgo_spot_es_espejo_en_la_puerta_ml: ndd(compose(b,s),b) + ndd(compose(b,−s),b) ≠ 0 para b ∈ {0,18; 0,215; 0,30}.
- Dueño: Línea C.
- Relación con hallazgos anteriores: LC-F1-22 (magnitud del sesgo spot). Añade la asimetría de espejo en la puerta, que LC-F1-22 no cubre, y que el test de K-06 no la detecta.
- Escépticos: confirmado.
  - Corrección (refutar): Formularlo así: para el mismo |s|, el impulso del corto es (1−b)/b veces el del largo (4,56 con b = 0,18). Que el sesgo neto vaya hacia los cortos queda condicionado a la distribución medida de la base.
  - Corrección (completitud): Es un caso particular de la «raíz común» que el ledger registra en §4.1 (LEDGER_DECISION_VIVA:93-104): el bosque sirve P(TP del LARGO antes que el SL), que no tiene espejo. Aun con s = 0, la simetría de la puerta ML es sólo formal en el espacio ndd. Por tanto, la respuesta del auditor «el espejo se cumple en la puerta ML cuando el sesgo spot es nulo» es demasiado fuerte. Hay que citar §4.1. T5 es una condición necesaria del espejo, no suficiente. Mantengo la severidad media.

### F3.3-05. La coherencia inter-espectral se recorta en 0: un espectro en contra da la misma confianza tensorial que uno neutro

- Bloque: F3.3, Confianza.
- Ubicación: crates/signal-engine/src/orchestrator.rs:519-530. Consumo: lib.rs:6216-6221, :6314-6320 y :5428-5446.
- Causa: coherencia_inter = (media/v_dom).clamp(0,1). Una media de signo opuesto da 0, igual que una media nula, y el suelo 0,70·|v_dom| se mantiene.
- Consecuencia viva: Si la banda contradice a la escala dominante (v_dom 0,8, media −0,8), la confianza es 0,56, idéntica al caso neutro. Esa cifra alimenta la rama 11 y tensor_boost en el compuesto, y con él los pisos de las ramas 1-10.
- Prueba que falla hoy: T6 cf3_espectro_opuesto_baja_la_confianza_tensorial: hoy da 0,56 en ambos casos.
- Dueño: Línea C (orchestrator.rs).
- Relación con hallazgos anteriores: Relacionado con F2.3-04 (dos coherencias) y con FICHAS:974 (dilución del camino escalar), pero distinto: ninguno trata la coherencia negativa recortada.
- Escépticos: confirmado.
  - Corrección (refutar): Añadir que el doc y el código discrepan. Si alguien «arreglara» el código para que siguiera el doc (|media|), una banda en contra daría convicción plena, que es peor.

### F3.3-06. El peso hebbiano aprende la tasa de acierto (signo del PnL), no el EV, y multiplica el compuesto ×[0,5; 2]

- Bloque: F3.3, Confianza.
- Ubicación: crates/signal-engine/src/perceptron_gate.rs:85-110. Escritura: lib.rs:4289-4310. Lectura: lib.rs:4633-4637. Uso: lib.rs:5446.
- Causa: update_weight suma o resta lr según el signo de net_trade_pnl, sin magnitud. El escritor recorta a [0,25; 2] y el lector a [0,5; 2].
- Consecuencia viva: Con la geometría activa (RR 2,25), una moneda con WR 40 % y EV +0,30 R baja a ≈ 0,29, que el lector recorta a 0,5: el compuesto se reduce a la mitad. Una con WR 60 % y EV −0,10 R sube a 2,0. Afecta a los pisos sig_conf(composite) de las ramas 1-10, al gate tensorial de la rama 11 (lib.rs:6287 y :6298), al escudo macro y al cortafuegos de rachas (lib.rs:7355 y :7365).
- Prueba que falla hoy: T4 cf3_hebbiano_sigue_el_edge_y_no_la_tasa_de_acierto: hoy da ≈ 0,29 y 2,0.
  - Cerrado por CL-76 (2026-10-11). `update_weight` es un gradiente exponenciado sobre el resultado neto en unidades de riesgo: w ← clamp(w·exp(lr·R), 0,25, 2), con R = PnL neto / nocional / SL inicial de la posición (`sl_price`, que el trailing no mueve). ln w deriva lr·E[R] por operación, así que sube con EV > 0 y baja con EV < 0 sea cual sea el WR. Con la prueba: EV +0,30 R (WR 40 %) da w = 2,0 y EV −0,10 R (WR 60 %) da 0,26 (antes 0,29 y 2,0). No cambia la tasa de aprendizaje ni los límites del escritor [0,25; 2] y del lector [0,5; 2]. Siguen abiertos: la siembra desde la clave global (LC-F1-04) y la misma regla de signo en la ganancia epigenética espectral (`apply_epigenetic_outcome`, temporal_spectrum.rs:772-793, zona de Qoder: pedido en el buzón).
  - T-2 con CL-76 (aviso al dueño del fixture): de a8d60390ff2f0f74 (68 operaciones, PnL −0,118, acierto 43 %, 899 vetos EV) a 998a7de2cfba2519 (131 operaciones, PnL +0,931, acierto 56 %, 898 vetos EV). Golden sin cambios. El T-2 no mide rentabilidad: el movimiento sólo dice que la regla del peso cambia qué intenciones pasan en el fixture, no que haya ventaja.
- Dueño: Línea C (regla y lector). Los consumidores de las ramas 1-10 son de Línea E.
- Relación con hallazgos anteriores: LC-F1-04 (contaminación entre monedas por la clave global). Añade la regla de aprendizaje y los límites incoherentes entre escritor y lector.
- Escépticos: confirmado.
  - Corrección (refutar): Severidad media. Precisar que el daño cierto está en los pisos sig_conf y en la escala absoluta del compuesto, y que en los gates relativos al cuantil se atenúa.
  - Corrección (completitud): La misma regla de aprender del signo y no del EV está también en la ganancia epigenética espectral (temporal_spectrum.rs:772-793), que el mapa no tiene (faltante 1). Además, su premio de 0,06 frente al castigo de 0,10 fija el equilibrio en una tasa de acierto de alrededor del 61-62 %. Conviene tratar las dos reglas juntas.

### F3.3-07. La rama 13 cuenta dos veces la misma lectura del ML (piso y puerta)

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:6529-6536 y :6551-6558 (piso con ndd(swing_nn_pred)). lib.rs:1629-1640 aplicado en lib.rs:6691-6699.
- Causa: El piso de la rama 13 usa max(|macd|·H·50, ndd⁺(swing_nn_pred, base)). Luego slow_intent pasa por puertas_del_continuo, que vuelve a multiplicar por (1+0,5·ndd(ml_prob, base)). swing_nn_pred y ml_prob vienen del mismo ensamble (lib.rs:4855 y :6429-6434), y el gate propio de la rama (lib.rs:6525) y el ML-GATE lo usan otra vez.
- Consecuencia viva: Con ndd 0,8: piso sig_conf(tanh 0,8) = 0,766, después ×1,4 = 1,07, que queda en el techo de 0,99. Una sola lectura del ML basta para saturar la confianza de la rama.
- Prueba que falla hoy: T11 cf3_rama_13_cuenta_el_ml_una_sola_vez (test de fuente: hoy 'normalized_directional_divergence(swing_nn_pred' aparece 2 veces). La prueba de propiedad exige extraer el impulso de la puerta a una función pública.
- Dueño: Línea C.
- Relación con hallazgos anteriores: Ledger §4 «Doble conteo» (ML ≥ 7). Añade el caso concreto dentro de una sola rama y la saturación a 0,99.
- Escépticos: confirmado.
  - Corrección (refutar): Sin cambio de severidad. Citar la fila 107 del §4 del plan.
  - Corrección (completitud): El conteo es mayor que el del auditor: - Cuando las dos bandas coinciden, las dos llevan ya el factor ML de la puerta, y confianza_superpuesta suma además 0,10·lo (lib.rs:6752-6753). Es la tercera entrada del mismo ML en una sola intención. - La rama 20 (ML re-centrado, lib.rs:5730-5752) también fija su convicción en 0,5 + |p − b| y luego pasa por la puerta. Esa rama sólo corre sin libro (lib.rs:5381 y :5708), así que su peso vivo es bajo.

### F3.3-08. La puntuación cruda dimensiona tres veces y la calibrada una

- Bloque: F3.3, Confianza.
- Ubicación: risk-engine/src/lib.rs:564-577, :584-586 y :1265-1267. leverage_matrix.rs:116-125, :165 y :315-318.
- Causa: El tamaño multiplica: - conf·epi_bias en el margen; - 1+(conf−min)⁺ en el Kelly micro; - 0,40+0,60·conf en el apalancamiento; - y aplica un techo por conf_t. La calibrada sólo entra con un 30 % en prob_win del Kelly del apalancamiento.
- Consecuencia viva: Con la misma p calibrada, una confianza de 0,95 frente a 0,80 da ≈ 1,19 × 1,10 ≈ 1,31 veces más nocional fuera del régimen micro. Las cifras puestas por la cadena de literales (F3.3-02) se convierten en tamaño.
- Prueba que falla hoy: T8 cf3_la_puntuacion_cruda_no_dimensiona_por_su_cuenta: misma win_probability 0,9 con conf 0,80 y 0,95 a capital 1000.
- Dueño: Línea C, bloque C-F4 (riesgo).
- Relación con hallazgos anteriores: R-17 y K-18. Añade la cuenta exacta (3 crudas + 1 calibrada + techo) y la prueba.
- Escépticos: confirmado.
  - Corrección (refutar): Dejarlo en C-F4 como recuento verificado. Añadir el caso del calibrador frío y decir que 1,31 es una cota.
  - Corrección (completitud): La calibrada es Platt(cruda) (lib.rs:7566-7581), así que la cruda llega cuatro veces. Además, en prob_win la calibrada sólo pesa un 30 % frente al win_rate histórico de la moneda, que mezcla todas las ramas (leverage_matrix.rs:121-122). El apalancamiento se cuantiza en enteros y tiene techos, y eso puede enmascarar parte del 1,31 en la orden final: T8 necesita un control (ver «pruebas»). El dueño es C-F4.

### F3.3-09. La rama 12 no pasa por conviccion_de_rama: su historial se escribe y nunca se lee

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:6329-6383 (etiqueta 12.0 en lib.rs:6379). Escritura: lib.rs:3986-3991.
- Causa: fast_intent = impulso_intent con su confianza tanh(coh·decay·2) intacta. Un grep de 'registro_ramas\[' sólo da 1, 2, 4, 5, 7-11, 13, 14 y 15.
- Consecuencia viva: Una rama 12 con desventaja demostrada sigue emitiendo su magnitud, que puede pasar el gate. D-752 no la cubre.
- Prueba que falla hoy: T9 cf3_rama_12_lee_su_historial (test de fuente: 'registro_ramas[12]' no aparece).
- Dueño: Línea C.
- Relación con hallazgos anteriores: MEMORIA (PR #5, hallazgo 2) decía que las ramas 11-14 no pasaban por D-752. Hoy 11, 13, 14 y 15 sí pasan; la 12 sigue fuera.
- Escépticos: confirmado.
  - Corrección (refutar): Marcar como duplicado del §3 fila 43; lo único que aporta es la prueba T9.

### F3.4-01. Los pesos de los asientos del consejo premian la tasa de acierto de las operaciones ejecutadas, no la esperanza

- Bloque: F3.4, Vetos.
- Ubicación: consejo_seniors.rs:1445-1484 (record_outcome), :1487-1513 (compute_weights), :1112-1119 y :1152-1157 (aplicación y safe_wr), :1387-1399; lib.rs:4236-4250 (alimentación), :7754-7758 (deliberar_traced con None), :8066-8075 (evidencia guardada al abrir)
- Causa: was_correct compara el signo de agreement = señal·lado con el signo del retorno neto (:1474-1475), y el peso es acierto/acierto medio, recortado a [0,5, 2]. La magnitud del retorno se descarta. Sólo se registran operaciones que pasaron el consejo, el ML-GATE y el riesgo, así que lo vetado nunca se puntúa. Con RR > 1 y acierto < ½, que es la geometría que pide la meta de crecimiento, el asiento que respalda una estrategia de EV > 0 pierde peso frente al que se opone. El doc de record_trade_outcome lo llama «attribution on selected trades, NOT counterfactual», pero el peso decide el consenso vivo. Además, safe_wr encoge el wr de la moneda con el n global del tracker.
- Consecuencia viva: El multiplicador escala peso·confianza en final_signal, en el consenso direccional y en directional_signal. Un asiento contrario a una estrategia rentable de RR alto puede ganar hasta ×2 de peso, bajar el consenso del lado por debajo de 0,35 o impedir la supermayoría de 0,80 del override. Es el mismo defecto que F2.2-02 (Ville mide el acierto contra ½ en lugar del EV en R), pero en otro consumidor.
- Prueba que falla hoy: cf3_tracker_premia_ev_no_acierto (metacortex-engine). Se registran 100 resultados con el asiento 0 en +1 y el asiento 1 en −1: 35 de +0,02 y 65 de −0,01, con EV = +0,0005. Se exige w0 ≥ w1. Hoy w0 = 0,70 y w1 = 1,30. El espejo con EV < 0 (65 de +0,005 y 35 de −0,015) también sale invertido.
- Dueño: Línea C (consejo, C-F3)
- Escépticos: confirmado.
  - Corrección (completitud): Añadir a la prueba el espejo del corto por la API viva, record_trade_outcome(&señales, ret, is_long = false), consejo_seniors.rs:1390. Así se cubre la inversión agreement = señal·lado y no sólo el tracker en bruto, que se salta la máscara de moduladores.

### F3.4-02. Un mismo dato (VPIN) produce dos vetos y vuelve imposible el override por encima de 0,92

- Bloque: F3.4, Vetos.
- Ubicación: consejo_seniors.rs:652-664 (Causal), :905-914 (Auditor), :1262-1291 (quórum); lib.rs:7598-7601, :7661-7662
- Causa: do_calculus_risk = vpin_risk alimenta dos asientos: - Causal veta por encima de 0,75, y un veto único admite override con supermayoría. - El Auditor veta por encima de 0,92. Con VPIN > 0,92 hay dos vetos, y eso cae en la rama irrevocable. Es el mismo acoplamiento que XLI·D7 retiró para el DD. El comentario del propio Auditor (:905-913) dice que «el DD cuenta una sola vez», y aquí el VPIN cuenta dos. La escalera 0,75 / 0,88 / 0,92 no tiene derivación.
- Consecuencia viva: En la banda de VPIN > 0,92 el veto es absoluto por duplicación, no por decisión. Con el lado por la regla del tick (F2.3b-02), la distribución del VPIN vivo no está medida y no se sabe con qué frecuencia cae en esa banda. No repite F2.3b-02 ni la ficha de umbrales VPIN de C-F2.
- Prueba que falla hoy: cf3_vpin_un_dato_un_veto. Payload: largo, OBI 0,2 (sin bypass), fused 0,8, persistence 0,8, ml 0,6, base 0,3, Hurst 0,7, τ 3 600 000 ms (s ≈ 0,658), slippage 0, wr 0,5. - Con VPIN 0,90 se aprueba: un solo veto, consenso 1,0 y |final| ≈ 0,89 ≥ 0,28. - Con VPIN 0,93 se rechaza. assert_eq!(approved(0,93), approved(0,90)) falla.
- Dueño: Línea C (consejo)
- Escépticos: confirmado.
  - Corrección (completitud): Hay dos matices. (1) El valor esperado de la prueba (approved(0,93) == approved(0,90)) es una política: «un dato, un veto». No es una propiedad derivada. Si el dueño quiere un techo absoluto de VPIN, la corrección es expresarlo como una rama explícita, igual que Riesgo y Ente (:1272-1281), y no por duplicación de asientos. La severidad es media sólo si ese techo no es intencionado; el comentario del Auditor (:905-913) sugiere que no lo es. (2) Por la regla de FICHAS_C-F2:872, la rejilla debe repetirse con deliberar_with_weights (consejo_seniors.rs:1069) y multiplicadores de Meta en {1, 2}. El resultado no cambia, porque todos los direccionales son positivos y el consenso sigue en 1,0.

### F3.4-03. El escudo de momento adverso no está al 95 %: une cuatro pruebas y un Stouffer sobre ejes correlacionados

- Bloque: F3.4, Vetos.
- Ubicación: lib.rs:5077 (comentario), :5080-5146 (ejes y regla). Consumidores: :5975 y :6059 (ramas 1 y 4), :6283 y :6294 (rama tensor). diffusion.rs:57, :67-102
- Causa: Σz/√k supone ejes independientes. z_macro (EMA 9-21), z_higher (p − EMA120) y z_secular (p − EMA720) salen de la misma caminata: ρ = 0,796 / 0,700 / 0,391, Var(S) = 1,53 y P(S > Z95) = 0,057 por lado. Además, la regla «algún eje > Z95 o S > Z95» es una unión sin corrección de multiplicidad. Resultados: - Monte Carlo con normales correlacionadas (2·10⁶): 0,086 por lado. - Caminata de 4·10⁵ barras con las funciones del repo: 0,085. - Con z_micro = 0, es decir sin σ de tick: 0,045. - Nominal: 0,025.
- Consecuencia viva: En ausencia de momento, el escudo bloquea entradas de las ramas 1, 4 y tensor unas 3,4 veces más de lo nominal. La tasa viva real no está medida: el suelo de 10 pb y el trinquete del ATR (F2.2-07) desinflan las z. Respecto a la semilla, se añade la tasa medida, la parte de la unión y la prueba.
- Prueba que falla hoy: cf3_stouffer_tasa_en_ruido (god-engine-core). Una caminata log-normal con σ = 1e-3 por barra, ATR = σ·PARKINSON, las EMAs 9/21/120/720 y las z de diffusion. Exige una tasa de bloqueo ≤ 0,05 por lado; hoy da ≈ 0,086.
- Dueño: Línea C: la definición del escudo está fuera de las ramas 1-10. Se pide a la Línea E la misma prueba para las ramas 1 y 4, que lo consumen.
- Relación con hallazgos anteriores: LC-F3-06
- Escépticos: confirmado.
  - Corrección (completitud): La atribución está sobredimensionada. Con los cuatro ejes INDEPENDIENTES, la misma regla bloquea 0,080 por lado (Monte Carlo de 4·10⁶). Es decir, unas 0,074 de las 0,086 vienen de la unión, que el código declara a propósito: «UNO SOLO que grite (z > Z95) o VARIOS que susurren», lib.rs:5128-5132. La correlación entre EMAs sólo añade ≈ 0,006 a la tasa compuesta. Donde la correlación sí pesa es en el término de Stouffer aislado: 0,057 frente a 0,025. El comentario de «significación al 95 %» de lib.rs:5077 es de D-756 y se refiere a trend_direction (z_higher_dir). No es un contrato del escudo. El hallazgo debe separar dos cosas: - (a) Stouffer mal especificado sobre ejes correlacionados (media). El denominador debe ser √(1ᵀΣ1), con Σ analítica de diffusion.rs. - (b) Unión sin control de multiplicidad. Es una decisión de diseño que pide un contrato declarado (FWER por lado), no un error numérico. Ya existe como semilla LC-F3-06 (plan:1072).
  - Corrección (completitud): La tasa sólo vale en régimen estacionario. - Tras reset_engines, o en un arranque sin las klines de calentamiento, las EMAs de vela valen 0. get_macro_trend cae a las EMAs de tick (stateful_engine.rs:1232-1240) tipificadas como si fueran de 1 min, y con v_t = 0 el ATR vale 0, así que ema_*_z devuelve 0 (diffusion.rs:85-102). Durante unas 2 h, que es lo que tarda EMA120, el escudo es otra prueba que descansa casi sólo en el eje micro. - higher y secular usan last_price dentro de la vela frente a una EMA cerrada en la vela anterior (stateful_engine.rs:1245-1258). Es un efecto pequeño, pero no es la caminata de barras que simula la prueba. - La prueba replica la regla en vez de llamarla (ver «pruebas»).

### F3.4-04. El veto de muro es el OBI de nivel 0 con otro umbral, y sólo existe en vivo

- Bloque: F3.4, Vetos.
- Ubicación: calibration.rs:393-414; lib.rs:1653-1675; god_engine.rs:3156-3157, :3185, :3196, :3349; state.rs:278-279, :792-805; backtest-engine/src/booktick_replay.rs y lib.rs:48 (no escriben los muros); genome.rs:953, :1875, :3054, :3208
- Causa: El host guarda como muro la cantidad del mejor nivel y calcula depth_obi de ese mismo nivel. muro_en_contra(b, a, r, largo) equivale a (b−a)/(b+a) ≤ (1−r)/(1+r): con el gen activo 5,324 eso es OBI ≤ −0,684. Es el mismo dato que ya usan Micro, Meta y el bypass de Causal. Los únicos escritores de update_l2_depth son god_engine.rs:3349, audit_forensic_backtest.rs, train_forest.rs:1476 y ws_client.rs:459. booktick_replay y run_backtest_native (T-1 y el examen WF del daemon) no lo escriben. En el replay, en_contra = 0, el veto no dispara y el gen 57 es inerte para la evolución. Además, new_random sortea 0,5..5,0, fuera de la banda [5, 50], y con un gen ≤ 1 el fallback es 2,0 (OBI ≤ −0,333). No verifiqué si un genoma creado con new_random llega a vivo sin pasar por from_vector, que sí recorta a la banda.
- Consecuencia viva: En vivo veta largos con OBI de nivel 0 ≤ −0,684 y cortos con OBI ≥ +0,684. El replay que decide las promociones no ve ese veto, así que el genoma se selecciona sin él. Se relaciona con LC-F1-56 (doc de los muros como «volumen acumulado»); aquí se añaden la identidad y la divergencia entre vivo y replay.
- Prueba que falla hoy: cf3_replay_escribe_los_muros (backtest-engine). 300 ReplayTick con bid_qty 1 y ask_qty 99. Con el observer se leen l2_bid_wall y l2_ask_wall, y se exige (1, 99) como en vivo. Hoy sale (0, 0). La identidad se documenta con cf3_muro_es_obi_nivel0, que pasa.
- Dueño: Línea C (puertas_del_continuo y harness de replay). La banda de new_random es una petición a quien mantenga genome.rs.
- Escépticos: confirmado.
  - Corrección (completitud): El punto que el auditor dejó sin verificar queda resuelto: SuperGenotype::new_random (genome.rs:851, dentro de impl SuperGenotype en :273) no tiene ningún llamador. `rg 'new_random\(\)'` sólo encuentra Genotype::new_random de darwin.rs y polars_evolver.rs, que es otro tipo. El sorteo 0,5..5,0 (genome.rs:953) fuera de la banda [5, 50] (:1875, :3054, :3208) no tiene consumidor vivo: es baja, no parte de la severidad media.
  - Corrección (completitud): Ampliar el título a «paridad vivo/replay del payload del consejo y de los vetos del núcleo», con la lista de campos que sólo existen en vivo. Con tapes de libro las liquidaciones y los tamaños de trade no se pueden reconstruir: el siguiente paso es una decisión de política (grabar esos flujos o declarar esos vetos «sólo vivo» y no seleccionar genes con ellos), no sólo un escritor en el replay.

### F3.4-05. La sonda de arranque frío no produce la evidencia que exige el ML-GATE, es por moneda y se repite tras cada reinicio

- Bloque: F3.4, Vetos.
- Ubicación: lib.rs:7835-7855 (comentario y predicado), :7877-7878, :4665 (has_roster_model = active_forest.is_some()), :4085 (único incremento de trade_count); god_engine.rs:27-46 (los modelos sólo se cargan desde fichero); state_continuity.rs:26-32
- Causa: El gate pide has_roster_model, y eso sólo lo crea un fichero {SYM}_MOTOR entrenado offline. Lo único que produce la sonda es un cierre que sube trade_count. Por eso el comentario de :7838-7839 («la primera orden ES la sonda que genera la evidencia que el gate exige») no se cumple. Tras el primer cierre, la moneda queda vetada hasta el próximo reinicio. Al reiniciar sin posición abierta, trade_count vuelve a 0 y hay otra sonda. El límite es is_any_open() de la propia moneda, no de la cartera: con 27 de 30 monedas sin modelo, según el comentario, caben 27 sondas simultáneas. Eso contradice la decisión D4.
- Consecuencia viva: El número de sondas vivas depende de los reinicios, no de la evidencia, y el resultado de la sonda nunca cambia el veto. Se relaciona con LC-F3-16 (BTC sondea tras cada reinicio) y con XLIV-7.
- Prueba que falla hoy: cf3_sonda_respeta_d4. Requiere extraer, sin cambiar la conducta, el predicado de lib.rs:7849-7855 a una función pura sonda_permitida(has_roster, cierres_moneda, abierta_en_moneda, sondas_abiertas_en_cartera). Hoy no compila. Extraída con la semántica actual, falla: ignora el conteo de cartera, así que sonda_permitida(false, 0, false, 1) da true.
- Dueño: Línea C (ML-GATE del núcleo), con decisión del dueño sobre el presupuesto de sonda (R-18 D4). Coordinar con el gate EV del risk-engine (risk lib.rs:1132, C-F4).
- Escépticos: confirmado.
  - Corrección (completitud): La cita «lib.rs:4085, único incremento de trade_count» es falsa. execution-engine/src/reconciliation.rs:524-529 también lo incrementa, al limpiar posiciones fantasma, y el plan ya lo dice (PLAN:461). Ese camino además actualiza el wr con una media simple que repite el defecto D-680. Ya está registrado como LC-F5-03 (PLAN:1120), y según esa semilla la salida = entryPrice = 0 impide entrar en la rama `exit_price > 0.0` (reconciliation.rs:502). No lo verifiqué contra el exchange. Nada de esto restaura el conteo tras un reinicio, así que la conclusión se mantiene. Además, la ruta de cartera de la API (quantum-arena/src/state_continuity.rs) no es de god-engine-core, como sugiere la ubicación del hallazgo.
  - Corrección (completitud): Matizar dos cosas. (1) «27 sondas simultáneas» es una cota superior: no verifiqué el tope de margen, la guarda de correlación ni los slots, que pueden limitarla. (2) Según el plan §1.2, la «sonda única de cartera (D4)» la posee Línea C. D4 ya está decidida (LEDGER:212), así que el arreglo no espera al dueño: sólo hay que coordinarlo con el gate EV del risk-engine (C-F4).

### F3.5-01. NEAR cuenta como cubierta, pero su bosque y su predictor _VOL no cargan; el freno VOL no tiene ningún modelo válido

- Bloque: F3.5, Modelos.
- Ubicación: models/NEARUSDT_MOTOR.json; models/NEARUSDT_VOL.json; crates/god-engine-core/src/ml_coverage.rs:57-97; crates/god-engine-core/src/ml_inference.rs:73-166; src/bin/god_engine.rs:27-46, 1294-1334, 4122-4150
- Causa: Los dos artefactos de NEAR guardan los índices de los hijos relativos a su árbol: en el árbol 1, que empieza en el nodo 61, ese nodo tiene left = 1. Hay 722 hijos fuera de rango en _MOTOR y 285 en _VOL. El cargador exige índices globales dentro de [start, end) (ml_inference.rs:119-131, error «split has missing or cross-tree child»), y el manifiesto del operador da el mismo veredicto. roster_coverage decide la cobertura por la sola presencia de `{SYM}_MOTOR.json`, sin validar (ml_coverage.rs:57-97). Por eso la línea ROSTER-COBERTURA cuenta a NEAR como cubierta mientras el núcleo no tiene bosque y NEAR sólo abre su sonda (lib.rs:4665, :7849-7877). El host reintenta y registra el rechazo en cada sondeo de 10 s (god_engine.rs:27-46). El VOL-BRAKE (god_engine.rs:4122-4150) no tiene ningún _VOL válido, ni en el repo ni en el manifiesto: vol_forecast_base vale 0 y el freno vale 1 siempre. El commit 1bc1efeb ya evaluaba con índices globales, así que estos artefactos nunca se sirvieron bien.
- Consecuencia viva: El plan y el operador creen que NEAR decide con modelo, pero en esta revisión 24 de las 26 monedas del universo operan sin modelo. El freno de volatilidad pronosticada está muerto en todas las máquinas conocidas.
- Prueba que falla hoy: cf3_cobertura_del_roster_coincide_con_el_cargador: roster_coverage sobre models/ pone NEARUSDT en covered, y escanear_models da valido_estructuralmente = Some(false) para NEARUSDT_MOTOR.
- Dueño: Línea C: que la cobertura signifique cargabilidad, usando el veredicto de ml_registry. Pedido a Línea E: reexportar NEAR con índices globales o retirar los dos artefactos y su entrada del manifiesto.
- Escépticos: confirmado.
  - Corrección (refutar): Severidad: propongo baja para la parte de Línea C. - Que NEAR opere sin modelo es la consecuencia CORRECTA de rechazar un artefacto roto. Lo erróneo es sólo la telemetría ROSTER-COBERTURA (una línea al arranque) y la creencia del operador. - El VOL-BRAKE vale 1 por ausencia de artefacto, no por un defecto del freno. - La reexportación pertenece a Línea E. Corregir los recuentos (nodos de split) y el commit de origen (12722b13). Faltante relacionado: el mismo vigilante intenta cargar DarkAlpha_BTCUSDT.json como NanoForest cada 10 s en la máquina del operador, porque escanea todo *.json (model_reload.rs:41-46).
  - Corrección (completitud): Hay que añadir que la sonda de NEAR, que existe porque el modelo no carga, se exime del veto EV con el prior literal 0,55 (risk-engine/src/lib.rs:1184-1194; ver faltantes). No verifiqué que ModelEntry.key valga exactamente "NEARUSDT_MOTOR" para el .json (ml_registry.rs:35-71). Conviene que el test imprima manifiesto.entries si no encuentra la clave.

### F3.5-02. El bosque opina con el macro de semilla, fuera de su soporte de entrenamiento; la caducidad del macro no se consulta

- Bloque: F3.5, Modelos.
- Ubicación: crates/god-engine-core/src/lib.rs:4762-4793; crates/god-engine-core/src/ml_inference.rs:35, 391-408; crates/data-pipeline/src/omni_multiplexer.rs:110-113, 248-255; src/bin/god_engine.rs:3377-3380; src/bin/train_forest.rs:1403-1455, 1501-1506
- Causa: **Entrenamiento.** El entrenador sólo emite muestras con macro vigente: macro_vigente en train_forest.rs:1403-1452 y el descarte en :1501-1506. **Servicio.** - OmniState arranca con las semillas DXY 104, SP500 5 100, NASDAQ 18 000 y VIX 15 (omni_multiplexer.rs:110-113) y las sirve como dato hasta el primer éxito de Yahoo. - macro_ml_features sólo neutraliza x ≤ 0 o no finito (ml_inference.rs:391-398). Con las semillas, SP500 vale 0,2 y NASDAQ 0,0 en unidades del bosque. - Los tres bosques versionados parten SP500 con umbrales de 4,53 a 5,28 y NASDAQ con umbrales de 3,58 a 3,99. Con las semillas, todos los árboles toman siempre la rama baja. - El host calcula macro_staleness_ms (god_engine.rs:3379-3380; u64::MAX si nunca hubo éxito, omni_multiplexer.rs:248-255), pero la inferencia no lo lee. Su único lector es el consejo (lib.rs:7709-7712). **Contratos y comentarios.** - No se cumple el contrato de ml_inference.rs:35 («una entrada insuficiente produce ausencia de predicción»). - El comentario de lib.rs:4772-4774 («El trainer descarta esas muestras») es falso para el NaN: train_forest.rs:1543-1549 lo sanea igual. El backtest también fija niveles de semilla (backtest-engine/src/lib.rs:213). No pude verificar qué rango macro cubre el mes de entrenamiento.
- Consecuencia viva: Con el feed macro caído, ATOM y BNB deciden el ML-GATE, las puertas y las ramas con un bosque evaluado fuera de soporte. Pasa en la nube (Yahoo bloqueado), en cualquier arranque antes del primer éxito y siempre en el T-1 y el backtest.
- Prueba que falla hoy: cf3_bosque_no_opina_con_macro_caducado. El control con macro fresco pasa (0,119 frente a 0,881). Con macro_staleness_ms = u64::MAX las dos salidas siguen siendo 0,119 y 0,881 en lugar de coincidir.
- Dueño: Línea C (núcleo): no enviar el bosque al ensamble si macro_staleness_ms supera la ventana de vigencia del entrenador. Pedido a Línea E: que el backtest marque el macro como caducado en lugar de fijar niveles.
- Escépticos: confirmado.
  - Corrección (refutar): Hay que reformular la consecuencia y el remedio: - El defecto es servir semillas inventadas como dato, no la caducidad en general. - Remedio: frescura POR SERIE, o semillas NaN con ausencia explícita. Sin las cuatro series observadas al menos una vez, el bosque no opina. - Que no dependa de macro_staleness_ms global, que hoy se satisface con PAXG. - Quitar el T-1 de la consecuencia. Precisar que el backtest sintético marca el macro como fresco. - Corregir el comentario de lib.rs:4772-4774. - Señalar a Qoder/data-pipeline (no es zona de Línea C) que `updated` mezcla PAXG con el macro.
  - Corrección (completitud): Faltan dos cosas en el hallazgo. 1. **El backtest declara el macro fresco.** backtest-engine/src/lib.rs:348 publica macro_staleness_ms = 0,0 mientras sirve niveles fijos (DXY 104,2, SP500 5 120 y NASDAQ 18 100 en backtest-engine/src/lib.rs:213-223). Un arreglo que mire la caducidad no cambia nada en el backtest si Línea E no cambia esa línea. booktick_replay.rs:416-417 sí publica la caducidad real. Las FICHAS C-F2 dan R6-A6 por «parcialmente cerrada» justo por esa línea 348, que en realidad miente. 2. **El arreglo propuesto, aplicado tal cual, fabrica un lift largo.** Si el bosque deja de enviar al ensamble pero has_roster_model sigue verdadero, combined() es None. Entonces base_ml_prob = 0,5 (lib.rs:4855-4859) se mide contra la base del bosque: en ATOM, (0,5 − 0,215)/0,785 = +0,36. Ese valor abre los largos en el ML-GATE (ml_now = ml_prob_pure, lib.rs:7810-7816) y los empuja ×1,18 en las puertas. Sin macro vigente hay que hacer que ml_prob = ml_model_base, o que la moneda cuente como sin modelo.

### F3.5-03. DarkAlpha se entrena con la entrada cruda y se sirve en z-scores; su compuerta de promoción compara pérdidas medidas en coordenadas distintas

- Bloque: F3.5, Modelos.
- Ubicación: src/bin/auto_trainer_daemon.rs:66-73, 121-124, 147, 160-176, 213-217, 376-377, 396-401; crates/dark-alpha-engine/src/lib.rs:682-697, 761-840, 1030-1037; crates/god-engine-core/src/lib.rs:1718-1729; src/bin/god_engine.rs:2026-2052, 2923-2935
- Causa: **Entrenamiento y promoción.** - El demonio parte de DarkAlphaEngine::new(54,64,32), con scaler None y sin congelar (dark-alpha-engine/src/lib.rs:682-697). - Como scaler_clone es None, entrena con x crudas (auto_trainer_daemon.rs:213-217). - La pérdida de validación usa eng.predict (:160-176). Sin scaler y sin congelar, predict actualiza los Welford y evalúa en z (dark-alpha-engine/src/lib.rs:794-840). - La compuerta val_after >= val_before (:376-377) compara así dos BCE medidas con normalizadores en estados distintos. - El artefacto se guarda sin scaler y con freeze_normalizers = false (:396-401). **Servicio.** - En la recarga en caliente, el núcleo acepta cualquier red con capas válidas y la congela (lib.rs:1720-1729). Después sirve con los Welford globales que dejó la validación (en z) o devuelve None. - El host carga la red al arranque sin congelarla ni sanearla (god_engine.rs:2923-2935). Durante unos 10 s, hasta que el vigilante (last_mtime = EPOCH, :2026-2052) envía la copia congelada, cada predicción muta los normalizadores. **Otros.** - La etiqueta no tiene lado (:121-124; ya en la fila 24 del plan). - El dataset de BTC sólo crece con cierres no adoptados (lib.rs:4117-4160). - train_dark_alpha.rs sí ajusta un Scaler y congela (:120-146, :406), pero escribe un _CANDIDATE. No puedo verificar con qué herramienta se produjo el DarkAlpha_BTCUSDT.json del operador.
- Consecuencia viva: En el PC del operador, si el artefacto de BTC viene del demonio, el voto NN de BTC aplica a entradas en una escala distinta de la del ajuste y se mezcla con el bosque en el ensamble (F3.5-05). En el repo es inerte: no hay DarkAlpha_BTCUSDT.json.
- Prueba que falla hoy: cf3_nucleo_no_sirve_red_sin_sistema_de_coordenadas: el núcleo instala por model_rx una red sin scaler, con normalizadores poblados por predict sin congelar, y swing_nn queda Some con scaler None.
- Dueño: Línea C (núcleo): rechazar en la recarga y en la carga inicial una red sin Scaler ajustado. Pedido al dueño del demonio (el plan no lo asigna; propongo Línea E): entrenar con fit() o con un Scaler ajustado y comparar las BCE en el mismo sistema de coordenadas.
- Escépticos: confirmado.
  - Corrección (refutar): Añadir que god_engine.rs:2924-2933 reemplaza la red congelada por el núcleo con una sin congelar. Exigir un Scaler es UNA salida, no la única. Valdría también que el demonio entrene en las z de normalizadores ya ajustados y congelados. El dueño debe elegir; el test debería fijar la propiedad (mismo sistema de coordenadas), no la implementación.
  - Corrección (completitud): La consecuencia viva sigue siendo condicional. No verifiqué que el demonio corra en el PC del operador ni de dónde viene su DarkAlpha_BTCUSDT.json. La severidad media es correcta sólo si ese artefacto existe.

### F3.5-04. El sesgo spot no tiene espejo cuando la base del bosque no es 0,5: el mismo diferencial empuja (1 − b)/b veces más hacia el corto

- Bloque: F3.5, Modelos.
- Ubicación: crates/god-engine-core/src/lib.rs:4442-4448, 4870, 4937, 1624-1641, 4903-4916; crates/god-engine-core/src/calibration.rs:285-289, 331-342; crates/god-engine-core/src/stateful_engine.rs:572-584
- Causa: compose_ml_prob suma spot_bias en unidades absolutas de probabilidad (calibration.rs:285-289). Después, nn_score (stateful_engine.rs:572-584) y las puertas (normalized_directional_divergence, calibration.rs:331-342, usada en lib.rs:1629) dividen por semiintervalos distintos: (1 − b) por encima de la base y b por debajo. Con b = 0,20 y |s| = 0,05: - nn_score da +0,0625 frente a −0,25; - las puertas dan ×1,031 al largo frente a ×1,125 al corto. La influencia de un dato de microestructura depende así de la base del modelo. Afecta hoy a ATOM (b = 0,215) y BNB (b = 0,181) en el compuesto, en las puertas, en la rama 13 (lib.rs:6429-6433) y en el conformal (lib.rs:4903-4916). El ML-GATE no se ve afectado porque usa ml_prob_pure. Este hallazgo extiende LC-F1-22 (pendiente literal del sesgo) con la falta de espejo.
- Consecuencia viva: Sesgo corto medible en ATOM y BNB, proporcional a |spot_bias|. En las monedas sin modelo (b = 0,5) el espejo se conserva.
- Prueba que falla hoy: cf3_sesgo_spot_es_impar_con_base_honesta: con b = 0,181 y s = 0,01, nn_score da +0,0122 frente a −0,0552 (la suma es −0,043, no 0).
- Dueño: Línea C: aplicar el sesgo en log-odds (logit p + κ·s) o después de la divergencia, con el mismo peso a cada lado.
- Relación con hallazgos anteriores: LC-F1-22 (parcial)
- Escépticos: confirmado.
  - Corrección (refutar): Quitar la rama 13 (con modelo) de la ubicación y de la consecuencia; los afectados vivos son el compuesto (nn_score), las dos puertas, el conformal y FlowExcitation. Corregir el remedio: sumar κ·s a la divergencia normalizada (o a nn_score y ml_directional), no en log-odds. Mantener «duplica LC-F1-22 (parcial)» y anotar la relación con LC-F1-43 (coordenada de reflexión).
  - Corrección (completitud): Hay tres correcciones. 1. **La rama 13 NO está afectada en monedas con modelo.** swing_nn_pred es ens.combined().clamp(0,01, 0,99), la versión pura sin spot (lib.rs:6429-6433). Sólo cae a coin.ml_prob, con spot, cuando no hay modelo, y entonces b = 0,5 y el espejo se conserva. Hay que quitarla de la lista de afectados. 2. **El arreglo «en log-odds» no pasa la prueba propuesta.** Con logit p = logit b ± κs, b = 0,181 y κs = 0,3, la divergencia normalizada da +0,0595 frente a −0,2227. Sólo pasa la otra opción: sumar el sesgo DESPUÉS de la divergencia y con el mismo peso a cada lado. Prueba y arreglo tienen que elegir la misma escala. 3. **La raíz es K-06 más un sesgo en unidades absolutas.** La normalización por semiintervalos amplifica cualquier perturbación absoluta de p por 1/b frente a 1/(1 − b). Conviene citarlo junto a LC-F1-43, no sólo a LC-F1-22.

### F3.5-05. El ensamble bosque⊕DarkAlpha promedia probabilidades de eventos distintos y todos los consumidores lo miden contra la base del bosque

- Bloque: F3.5, Modelos.
- Ubicación: crates/god-engine-core/src/ensemble.rs:157-258; crates/god-engine-core/src/lib.rs:4643, 4671-4675, 4806-4886, 1892-1928, 7461-7465, 7808-7816
- Causa: **Mezcla de eventos.** En BTC con los dos modelos, combined() es la media ponderada de dos probabilidades de eventos distintos: - la del bosque, P(TP largo primero en 300 s), con base ≈ 0,156 según el manifiesto; - la de DarkAlpha, P(retorno neto > 5 pb), sin lado y con otra base. Pero ml_model_base es sólo la del bosque (lib.rs:4671-4675, :4886). Con un bosque sin información en 0,20 y una red constante en 0,45, ml_prob vale 0,325: un lift de +0,156 sacado de la nada. Las puertas, nn_score, FlowExcitation y el consejo lo leen como apoyo al largo. **Calificación del Hedge.** - Califica cada vela de 1 min contra ±sl_at_tau(30 s) (lib.rs:1892-1928), un tercer evento que favorece al modelo con la base más cercana a 0,5. - Las predicciones sólo se reinician en velas decisivas (ensemble.rs:256-258), así que tras velas neutras se califica una opinión de varios minutos antes. - nn_penalty usa la z de habilidad del ensamble combinado y sólo castiga al NN (ensemble.rs:201-208). **Uso y comentarios.** - neural_skill decide qué p mira el ML-GATE (lib.rs:7461-7465, :7810-7816), aunque su comentario D-696 lo declara telemetría. - En el resto del roster el ensamble es la identidad (un solo participante), así que el comentario de lib.rs:4643 («ambos modelos opinan») es falso. Amplía C-16/18 del ledger con el mecanismo y su tamaño.
- Consecuencia viva: Inerte en el repo, que no tiene DarkAlpha. En el PC del operador fabrica un sesgo largo en BTC en todos los consumidores de ml_prob; el ML-GATE se libra cuando neural_skill es falso.
- Prueba que falla hoy: cf3_dos_modelos_sin_informacion_dejan_neutro_el_canal_ml: ml_prob_motor vale 0,325 y ml_model_base 0,20.
- Dueño: Línea C (ensamble): no mezclar modelos con etiquetas distintas. Hay que publicar la base de la mezcla o dejar el NN en sombra hasta que comparta etiqueta.
- Relación con hallazgos anteriores: C-16/18 (ledger, parcial)
- Escépticos: confirmado.
  - Corrección (refutar): Marcar «duplica: C-16/18 (ledger:37), parcial» y la magnitud en vivo como no verificada (depende de la red del operador). La afirmación «el Hedge favorece al modelo de base cercana a 0,5» es plausible por el Brier frente a una etiqueta simétrica, pero no está medida.
  - Corrección (completitud): Falta en el mecanismo el segundo canal de aprendizaje del Hedge: update_with_trade_snapshot, al cierre (lib.rs:3974-3983; ensemble.rs:277-296). Califica contra una cuarta etiqueta (el signo del mid, direccion_realizada). Usa eta_trade = 0,25·(|pnl|/0,001).clamp(0,5, 3).clamp(0,08, 0,75), sin olvido, y sólo sobre operaciones ejecutadas. Así, el peso relativo bosque/NN en BTC aprende de la magnitud del PnL (ver faltantes).

### F3.6-02. La sombra espectral del solitón rompe el espejo: su pendiente depende del signo del OFI

- Bloque: F3.6, Motores de señal.
- Ubicación: soliton_wave.rs:81-98 (a = amplitud si es > 0, si no 1,0); lib.rs:2088-2112 (pasa el OFI firmado y lo declara «ESPEJO EXACTO»); god-engine-core/tests/sombras_espectrales_telemetria_contract.rs:133-138
- Causa: Con flujo comprador (OFI = +0,2) la pendiente es 0,2 y el voto por escala es tanh(0,2·x). En el mercado reflejado (OFI = −0,2) la amplitud se sustituye por 1,0 y el voto es tanh(−x). La respuesta es grande con OFI ≤ 0 y pequeña con OFI > 0, sea cual sea el signo de x. Si momentum y flujo coinciden, que es lo habitual, eso da un sesgo CORTO en magnitud. El vivo escalar usa |OFI|, así que la sombra tampoco tiene «la misma semántica». El test H2-10 fija ese defecto como contrato.
- Consecuencia viva: consenso_espectral_dominante y consenso_espectral_media (lib.rs:2546-2560) llevan un término sesgado de unos 0,04 por escala (1/13 del consenso con el piso de 0,15). v_dom sustituye al net escalar cuando τ_dom ≥ 30 s (orchestrator.rs:516-528).
- Prueba que falla hoy: cf3_6_espejo_espectral_soliton_con_el_ofi_del_vivo: en la escala 0 da 0,1974 + (−0,7616) = −0,564, cuando debería ser 0. Falla hoy.
- Dueño: Línea C (sombra en lib.rs:2088-2112 y voto_espectral del solitón). El test H2-10 es de GLM: petición de sustituir su aserción de :133-138 por la antisimetría.
- Escépticos: confirmado.
  - Corrección (refutar): Hay que añadir lo que el auditor no vio: la sustitución hace el voto DISCONTINUO en OFI = 0. Con OFI = +1e-9 la pendiente es 1e-3 (por el recorte) y el voto ≈ 0. Con OFI = −1e-9 la pendiente es 1,0 y el voto es tanh(x). Es un salto de hasta 0,76 por escala a un lado y otro del flujo nulo. El sesgo exacto por escala depende de los pesos de SkillMotores después de la puerta de observabilidad; el «≈ 0,04» supone pesos iguales entre 13 motores.
  - Corrección (completitud): Hay que anotar el retraso de un evento y la caída al global en el arranque. Ninguna de las dos cosas cambia la severidad media.

### F3.6-03. El votante Rényi-Tsallis es una función de OBI: la «entropía» publicada es h(|OBI|) y Rényi nunca se calcula en vivo

- Bloque: F3.6, Motores de señal.
- Ubicación: lib.rs:4451-4473 (p_bid = (1 + OBI)/2; tsallis_ent), :4523; renyi_tsallis_entropy.rs:142-165 (vote), :242-258; calculate_renyi_entropy (:90) sólo en tests (:276-315)
- Causa: p_bid = bid_qty/(bid_qty + ask_qty) = (1 + OBI)/2, así que la «entropía» es una función determinista de |OBI| en el mismo evento, sin ventana ni historia. El voto queda (1 − h)·OBI·rampa(h)·rampa(|OBI|) = g(OBI). Sólo se activa con |OBI| > 0,655 (h < 0,60) y es pleno con |OBI| ≥ 0,79. Con él, OBI entra en el consenso escalar por cuatro votantes: Nash, la mitad de la resonancia, FlowImpulse y éste. Eso eleva eff = 1,4·Σ|v|/19 sin aportar una fuente nueva.
- Consecuencia viva: Con libros muy desequilibrados (|OBI| > 0,66), el consenso escalar cuenta OBI otra vez, hasta ±0,71 con |OBI| = 0,9. El resultado es un sesgo hacia el lado del libro sin información de entropía.
- Prueba que falla hoy: cf3_6_entropia_publicada_no_es_funcion_de_obi (crates/god-engine-core/tests/cf3_6_entropia_no_es_obi.rs). La desviación máxima |T − h(OBI)| es 0 por álgebra en todos los libros y trayectorias. Falla hoy.
- Dueño: Línea C
- Relación con hallazgos anteriores: Relacionado con LC-F3-03 (colinealidad espectral). Este hallazgo es el camino escalar.
- Escépticos: confirmado.

### F3.6-04. El coaxial escalar vale 0 por álgebra en precios ≥ 0,2236 USD y fabrica votos de ±0,19 a ±0,22 en monedas más baratas

- Bloque: F3.6, Motores de señal.
- Ubicación: lib.rs:4475 (micro_v = v_t/mid), :4506-4512; coaxial_breakout.rs:150-188, :230-233; stateful_engine.rs:875-881 (v_t ≥ 10 pb del precio), :1177-1183 (atr_pct = v_t/last_price)
- Causa: Con last_price = mid, las tres «ATR» son la misma v_t (la ATR de 1 min) con suelos en dólares: atr_1s = máx(v_t, 1e-4), atr_5s = máx(v_t, 5e-4) y atr_1m = máx(7,746·v_t, 0,002). - comp_1s = máx(1 − √5·máx(v_t, 1e-4)/máx(v_t, 5e-4), 0) vale 0 en cuanto v_t ≥ 2,236e-4 USD. Con el suelo de 10 pb, eso cubre todo precio ≥ 0,2236 USD. - Por debajo, los suelos fabrican compresión. Con v_t < 1e-4 USD el voto es ±tanh(1)·0,288 = ±0,219, con el signo de sign(bid_qty − ask_qty).
- Consecuencia viva: En monedas por debajo de 0,2236 USD con la ATR cerca del suelo entra un tercer voto con el lado del libro, además de Nash y Rényi. No verifiqué qué monedas del universo vivo están en ese rango.
- Prueba que falla hoy: cf3_6_coaxial_invariante_a_la_unidad_de_precio: con el mismo ATR relativo de 20 pb, el voto es 0 a 60 000 USD y 0,1853 a 0,06 USD. Falla hoy.
- Dueño: Línea C para el lector coaxial. Los suelos van con LC-F3-02 al principio de C-F4.
- Relación con hallazgos anteriores: LC-F3-02 (literales) y MEMORIA «B3 Coaxial vota 0». Lo nuevo es el mecanismo y el voto fabricado.
- Escépticos: confirmado.
  - Corrección (refutar): Bajaría la severidad a baja. Los tres símbolos con modelo en models/ (ATOMUSDT, BNBUSDT, NEARUSDT) cotizan muy por encima de 0,2236 USD, así que el coaxial es idénticamente 0 en ellos. Los símbolos por debajo de ese precio no tienen _MOTOR. Según MEMORIA, la puerta de roster veta sus entradas; no lo verifiqué en el código de la puerta (ml_coverage.rs sólo clasifica). Durante el calentamiento con process_kline el suelo de v_t es de 5 pb (stateful_engine.rs:1028), y el umbral sube a ≈ 0,447 USD; es un matiz menor. El daño vivo de hoy es sólo diluir eff con un votante muerto. El voto fabricado es un riesgo latente si entra un símbolo barato.
  - Corrección (completitud): La prueba debe ejercitar el escritor; véase cf3_6_coaxial_invariante_a_la_unidad_de_precio en «pruebas».

### F3.6-05. Las teorías nombradas no están implementadas: los motores son envolturas de OBI, OFI y la desviación de posición

- Bloque: F3.6, Motores de señal.
- Ubicación: stochastic_resonance.rs:40-63 con lib.rs:4620-4623; game_theoretic_nash.rs:53-75, :140-210; supersonic_shockwave.rs:44-55, :167; soliton_wave.rs:27-56; quantum_oscillator.rs:30, :132-225; trend_runner.rs:37-98, :236-244
- Causa: Lo que hace cada motor: - Resonancia: ganancia 1 + sech(ln(|s|/σ)), sin umbral ni dinámica. Con σ² = 0,1·atr_pct (σ ≈ 0,014, en unidades de ATR frente a una señal adimensional) vale ≈ 1,05, y el voto ≈ ½OFI + ½OBI. - Nash: tanh(OBI·(1 − mín(cvpin, 0,9))), sin juego. - Choque: mide saltos por evento, no un número de Mach. El doc «precio/s» (:167) es falso. - Oscilador: reversión a la EMA21 de 1 min con fuerza de Duffing y confinamiento gaussiano. - TrendRunner: usa un TP como convicción. El consenso cuenta como acuerdo lo que es la misma entrada repetida.
- Consecuencia viva: El ensamble escalar de 19 votantes tiene cuatro fuentes con signo. Los nombres hacen pensar en mediciones independientes que no existen, y eff y net suben por repetición.
- Prueba que falla hoy: cf3_6_resonancia_estocastica_tiene_maximo_interior. La correlación de la salida con la señal limpia es 0,951, 0,832, 0,587, 0,342, 0,182 y 0,097 para σ de 0,0025 a 0,08: el máximo está en el borde, no en el interior. Falla hoy.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): El número de la ganancia está sobrestimado como valor típico. Con σ ≈ 0,0141 (atr_pct = 0,002), la ganancia vale 1,05 con |s| = 0,55, ≈ 1,28 con |s| = 0,1 y 2 con |s| = σ. Decir «≈ 1,05 en el régimen vivo» sólo vale para señales del orden de 0,5. La conclusión no cambia: la salida es ≈ ½OFI + ½OBI con una ganancia modesta.
  - Corrección (completitud): Hay que reescribir el resumen: «los diez motores de F3.6 aportan cuatro fuentes con signo». El recuento del ensamble de 19 queda pendiente de los otros bloques. También hay que quitar el «≈ 1,05» genérico. La prueba de resonancia necesita la corrección indicada en «pruebas».

### F3.6-06. El peso «hebbiano» premia la tasa de acierto en vez de la esperanza, y entra dos veces en la decisión

- Bloque: F3.6, Motores de señal.
- Ubicación: perceptron_gate.rs:85-111; lib.rs:4289-4310 (escritura al cierre, también en la clave global), :4633-4637 (hebbian_mult = clamp(w, 0,5, 2)), :5446-5453; perceptron_gate.rs:168-176
- Causa: La regla no es de Hebb: no tiene término presináptico, así que no sabe si el perceptrón votó ni en qué lado, y del PnL sólo mira el signo, no el tamaño. Con un WR del 60 % y pago 1:2 (EV de −0,2 por operación) el peso sube hasta el tope. Ese w multiplica el compuesto entero, y además abre la compuerta del perceptrón, cuyo voto ya va dentro de tensor_boost.
- Consecuencia viva: Contradice RULE[growth_over_wr]. Con muchas ganancias pequeñas y pérdidas grandes, el núcleo multiplica el compuesto hasta ×1,92 (lib.rs:5446) y sube sig_conf en las ramas 1–10. El umbral p85 medido de |compuesto| (lib.rs:4994, :5469) se adapta con retraso y sólo lo compensa en parte; no lo medí.
- Prueba que falla hoy: cf3_6_peso_hebb_no_premia_acierto_con_esperanza_negativa: w = 1,917 tras 200 cierres con EV < 0. Falla hoy.
  - Regla cerrada por CL-76 (2026-10-11): con EV −0,2 R el peso baja al suelo 0,25 (antes 1,917). Ver F3.3-06. Siguen abiertos: la falta de término presináptico (la regla no sabe si el perceptrón votó ni en qué lado), la compuerta que lee la clave global de la última moneda (LC-F1-04) y el consumo de hebbian_mult en el compuesto de las ramas 1-10 (Línea E).
- Dueño: Línea C para update_weight. El consumo de hebbian_mult en el compuesto de las ramas 1–10 es de Línea E: petición con este test.
- Relación con hallazgos anteriores: Parcial con LC-F3-08 (clave global) y con el plan P3 / R-17 (refuerzo ×2). Lo nuevo es que premia el WR con EV negativo y que entra dos veces.
- Escépticos: confirmado.
  - Corrección (refutar): «Entra dos veces con el mismo w» no es exacto. - El perceptrón lee perceptron_hebbian_weight, que sólo existe como clave GLOBAL (lib.rs:4308-4310). Es el peso de la última moneda que cerró, no el del símbolo. Eso ya es LC-F1-04 (alta), que el auditor no cita; sólo cita LC-F3-08. - Esa segunda entrada sólo mueve la compuerta (piso 0,15 si |OFI|·w < 0,5) de un votante de 19, y sólo en el respaldo escalar: su efecto es marginal. - Lo nuevo y vigente es que la regla premia el WR con EV negativo en hebbian_mult (×1,92 sobre el compuesto). - El primer cierre de un símbolo siembra su peso desde el global (lib.rs:4290-4300): es LC-F1-04, ya publicado.
  - Corrección (completitud): Hay que decir «w_moneda en el compuesto y w_última_moneda_cerrada en la compuerta» y añadir la siembra desde el global (lib.rs:4290-4299) como ampliación de LC-F3-08.

### F3.6-07. La vida de la rama 12 sale del gen base_duration_ms, no de una τ medida, y pesa la mitad en log tras la mezcla

- Bloque: F3.6, Motores de señal.
- Ubicación: flow_impulse.rs:203-213; lib.rs:6337-6380 (rama 12), :7164-7186 (media geométrica con resonant_tau y suelos de 30 000); orchestrator.rs:575-586
- Causa: evaluate_flow_impulse no lee ninguna τ del mercado: la duración es un gen evolutivo con suelo de 15 000. Aguas abajo, continuous_tau = exp(½·ln máx(gen, 30 000) + ½·ln máx(τ_res, 30 000)), así que el gen fija la mitad del logaritmo del horizonte. El suelo de 15 000 no tiene efecto. Sin dominante espectral, el consenso escalar también toma su vida del gen: gen·(1 + 9·conf).
- Consecuencia viva: La τ con que se dimensiona la entrada de la rama 12 depende de un gen y no de la escala que habló. Según MEMORIA (D-745), esa τ pasa a ValidatedOrder::tau_ms y Position::entry_tau_ms; no lo verifiqué en este bloque.
- Prueba que falla hoy: cf3_6_rama12_la_vida_no_es_un_gen: con el mismo mercado y la misma τ publicada, la vida es 15 000 ms con el gen en 10 000 y 100 000 ms con el gen en 100 000. Falla hoy.
- Dueño: Línea C
- Relación con hallazgos anteriores: Relacionado con LC-F3-04 (ramas 11 y 14). La rama 12 y la vida escalar no están en él.
- Escépticos: confirmado.
  - Corrección (refutar): Hay dos correcciones. 1. La cadena tiene dos pasos que el auditor omite: la fusión de bandas, que con la misma dirección y |Δln τ| < umbral elige la duración de la banda de mayor energía (lib.rs:6763-6767), y hurst_duration_modulation ×[0,5; 2] (lib.rs:6830, :8254-8266), ambos antes de la media geométrica. El gen sólo fija la mitad del log cuando gana la banda rápida, y aun así lo hace tras el factor de Hurst. 2. La parte de la «vida del consenso escalar» (orchestrator.rs:575-586) se REFUTA como consecuencia viva. TensorDecision.expected_lifetime_ms no tiene consumidor vivo: - lib.rs sólo lee tensor_cont.signal y net_confidence (grep: :5428-5455, :6218-6322, :6642-6673); - el único lector es QuantumOrderRouter (execution-engine/src/router.rs:74), que sólo aparece en `pub use` (execution-engine/src/lib.rs:32) y no se instancia en ningún sitio; - el host usa DarkAlphaRouter (god_engine.rs:1286). Esto también corrige la premisa de LC-F3-04: la vida tau_dom del orquestador no la usa nadie, no sólo las ramas 11 y 14. Esa parte es de severidad baja.
  - Corrección (completitud): Hay que retirar orchestrator.rs:575-586 del hallazgo, o reclasificarlo como baja (cálculo sin consumidor). Afecta también a LC-F3-04: la τ_dom que «fija el orquestador» no la consume nadie en vivo, no sólo las ramas 11 y 14. El arreglo de P2 debe llevar τ_dom por SignalIntent::expected_duration_ms, o retirar el campo junto con QuantumOrderRouter (plan §3, fila 7).

### F3.6-08. En TrendRunner la volatilidad sube la convicción (vol_boost), en contra de D-746

- Bloque: F3.6, Motores de señal.
- Ubicación: trend_runner.rs:71-72 (vol_boost = atr·expansión·4), :236-244 (voto = tanh(dir)·h_weight·tanh(10·tp))
- Causa: calculate_expanded_tp calcula un tamaño de TP y el motor lo usa como convicción. Con el mismo Hurst y el mismo VPIN, pasar de una ATR del 0,1 % a una del 1 % multiplica el voto por 1,23.
- Consecuencia viva: En el respaldo escalar, las monedas más volátiles reciben más convicción de tendencia sin que su probabilidad de continuación sea mayor. Su peso en net es pequeño (|v| ≤ 0,76).
- Prueba que falla hoy: cf3_6_trend_runner_la_volatilidad_no_es_conviccion: el voto pasa de 0,2675 a 0,3288. Falla hoy.
  - Cerrado por CL-77 (2026-10-11). La convicción del motor es `conviccion_de_expansion(hurst, vpin)` = tanh(10·tp) con el TP de expansión sin `vol_boost`: no lee el ATR, es no decreciente en Hurst y no creciente en VPIN (`cl77_la_conviccion_no_lee_el_atr_y_es_monotona`, rejilla de ATR 0,01 %–5 %). Vale para el voto escalar y para el espectral, que deja de recibir el ATR. `calculate_expanded_tp` no cambia: que el objetivo crezca con la volatilidad es legítimo.
  - T-2 y golden idénticos a CL-76: el fixture no recorre este voto.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): La consecuencia es pequeña: sólo actúa en el respaldo escalar, el factor es ×1,23 y el signo no cambia. La dejaría en baja-media. D-746 se refería al freno de apalancamiento; aquí se aplica por analogía, no por contrato.

### F3.7-01. `vecm_zscore` es el nivel de la base entre el ATR, no un z: una prima constante vota reversión de forma permanente

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:5215-5225; lectores crates/strategy-core/src/vecm_arbitrage.rs:175-203, crates/strategy-core/src/stat_arb.rs:413-418, crates/signal-engine/src/conformal_reversion_filter.rs:183
- Causa: La fórmula es (F − S)/(F·max(ATR%, 0,0005)) recortada a ±3. No resta la media de la base ni divide por la σ de la base: divide por el ATR del futuro. La base futuro-spot lleva una prima de financiación de varios pb que no revierte a 0. Con el ATR% en su suelo de 10 pb, una prima de 15 pb ya supera el umbral 1,5, y una de 30 pb satura en 3: VECM vota −1 en todos los eventos. Sin spot, el respaldo es (mid − EMA lenta)/ATR, que mide momento y se lee como reversión.
- Consecuencia viva: Con spot vivo, VECM y conformal dan un voto sistemático contra el lado de la prima (corto con contango), sea cual sea la dinámica. Ese voto se suma al consenso y a `tensor_boost`.
- Prueba que falla hoy: cf3_vecm_base_constante_no_vota_reversion (crates/god-engine-core/tests/cf3_vecm_base_constante.rs): base constante de +30 pb durante 1 800 ticks de 1 s. Espera |vecm_zscore| < 0,5 y voto VECM igual a 0. Predicción: z = 3 (o ≥ 0,6 si el ATR no ha bajado a su suelo); falla.
  - Cerrado por CL-78 (2026-10-11) en el camino con spot. `vecm_zscore` es el z de la base b = (F − S)/F contra su propia media y dispersión (`god_engine_core::base_z`): promedios temporales con olvido exponencial por nodo de la malla (τ_17 a τ_23, de 17 s a 19,5 h), que pesan el tiempo y no los eventos, leídos a la τ dominante de la moneda (la de la SDE OU de la base) con interpolación en ln τ; calentamiento de una τ por nodo; piso de σ = un tick relativo (CL-69); causal (la observación nueva contra el pasado). Por nodo y no con una sola EWMA de la τ vigente: la τ dominante salta entre nodos (68,7 s ↔ 274,9 s en la prueba) y la EWMA de memoria variable perdía la historia y se abstenía (679/900 lecturas antes de este diseño). Medido en la prueba: z medio −0,004, voto medio −0,002; 900/900 lecturas activas con desviación típica 1,005 (`cl78_en_equilibrio_el_z_tiene_dispersion_unidad`); una dislocación de ~5σ da z = 3 y voto −1 (`cl78_una_dislocacion_de_la_base_vota_reversion`). Coste: una sola `exp_m1` por evento con spot (las demás por cuadrados, porque la malla es 4^k; error relativo < 1e-12 frente a la exponencial directa). Sigue abierto, con F3.7-05: sin spot, `vecm_zscore` es (mid − EMA lenta)/ATR, la misma expresión que `quantum_position_deviation`.
- Dueño: Línea C (escritor en lib.rs y lectores de strategy-core). El lector conformal se ve afectado igual.
- Relación con hallazgos anteriores: Amplía LC-F3-14 y la fila 20 de §3 del plan (lo nuevo es el sesgo por la prima media). `cointegration_zscore` ya es F2.4-11.
- Escépticos: confirmado.
  - Corrección (refutar): Precisar la cadencia: se publica en eventos con entradas permitidas (trades y velas), no en cada tick. En la prueba, tras 30 min de ruido ínfimo atr_pct ≈ 0,001 + 0,004·0,85^30 ≈ 0,00103, así que z ≈ 0,00299/0,00103 ≈ 2,9, no 3 recortado. La prueba falla igual. Rebajar 'de forma permanente' a 'mientras la prima supere unos 1,5·ATR%', y añadir el caso de demo (spot de mainnet frente a futuro de testnet) como consecuencia sin verificar.
  - Corrección (completitud): En la causa, cambiar «suelo de 10 pb» por «suelo de 5 pb en lib.rs:5217». En la consecuencia viva, añadir: «sólo cuando consenso_espectral_dominante está inactivo o tiene τ < 30 s (orchestrator.rs:382-384). En ese caso entra en el compuesto, en la rama 11 y en la rama 14, y un único voto VECM basta para dar |net_confidence| ≈ 0,70 (orchestrator.rs:488-503)». Añadir también que, sin spot, vecm_zscore es idéntico a quantum_position_deviation (faltante 3).

### F3.7-02. La cointegración se supone: la guarda t½ ≤ 2τ* deja pasar caminatas aleatorias independientes

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/strategy-core/src/stat_arb.rs:53, :91, :119, :151-168, :269-348, :398-408; crates/strategy-core/src/vecm_arbitrage.rs:518-613
- Causa: No hay contraste con un nulo de raíz unitaria (Johansen, Engle-Granger, ADF o KPSS): el grep da 0. La β por RLS con λ = 0,998 por evento sigue el cociente ln F/ln S y deja un residuo paso-alto que revierte por construcción. El WLS de θ, con memoria de 300 s, tiene el sesgo de Dickey-Fuller (θ̂ > 0 aun con θ = 0). El prior θ = 0,1 con suavizado 0,05 por evento da t½ ≈ 7 s justo cuando ya se publica z (madurez 10). Además, θ se suaviza en reloj de eventos aunque el WLS pesa en tiempo físico.
- Consecuencia viva: En vivo el par es el futuro y el spot del mismo activo, cointegrados en la práctica, así que el daño es de validación: la puerta no certifica nada, y la z de un residuo filtrado no mide la dislocación económica de la base. Si mañana el par cambia, la puerta tampoco lo detendría.
- Prueba que falla hoy: cf3_guarda_t_media_no_deja_pasar_caminatas_independientes (crates/strategy-core/tests/cf3_guarda_t_media.rs): 40 pares de caminatas independientes, 6 000 pasos de 200 ms, τ = 600 s, con la configuración viva. Espera que pasen la guarda ≤ 10 %. Predicción: ≈ 40/40.
- Dueño: Línea C
- Relación con hallazgos anteriores: Fila 20 de §3 del plan (cointegración supuesta); relacionado con LC-F3-14.
- Escépticos: confirmado.
  - Corrección (refutar): Añadir que en vivo la SDE sólo avanza en eventos con entradas permitidas (lib.rs:2793 y :4950), así que su reloj de eventos es el de trades y no el de ticks. Con τ* = 30 s la guarda exige t½ ≤ 60 s, y la memoria del RLS en tiempo físico (500 eventos) depende de la tasa de trades: la aceptación de una caminata depende de la actividad del mercado, no de la cointegración.
  - Corrección (completitud): Parametrizar el test en τ ∈ {30 s, 600 s} y en un periodo de muestreo ∈ {100 ms, 1 s}, y dar la predicción por variante: probable fallo a 600 s e incierta a 30 s con muestreo síncrono. Añadir la variante con spot refrescado cada 1 s (el régimen de sólo REST, spot_feed.rs:23 y :207), en la que predigo t½ < 1 s y que pase la guarda.

### F3.7-03. El TTL anti-staleness del spot está muerto en producción: un spot congelado sigue votando

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:5250-5263; crates/data-pipeline/src/spot_feed.rs:110-115; test crates/god-engine-core/tests/statarb_live_physics_contract.rs:140-174
- Causa: El único escritor del spot (`apply_spot_tick`, el único llamador de `update_spot_data`) descarta bid/ask ≤ 0, así que cuando cae el stream la arena conserva el último spot > 0. El núcleo llama a `update_with_clock` en cada tick con el `event_time_ms` del futuro (lib.rs:5252-5254), con lo que `last_ou_ts_ms` avanza en cada tick y la edad medida nunca supera 30 s. El test vigente pasa sólo porque pone el spot a 0 (statarb_live_physics_contract.rs:164), cosa que producción no hace. No hay marca de tiempo del spot en la arena.
- Consecuencia viva: Con el stream spot caído, la SDE sigue comparando F_t con un S fijo: el «spread» pasa a ser el precio del futuro, y StatArb vota contra su movimiento sin límite de edad. spot_feed.rs:27 y :30 usan siempre mainnet; en demo no verifiqué cómo se emparejan.
- Prueba que falla hoy: cf3_spot_congelado_mas_alla_del_ttl_publica_abstencion (crates/god-engine-core/tests/cf3_spot_congelado_ttl.rs): igual que el test vigente, pero sin poner el spot a 0. Tras +31 s espera Some(0.0). Predicción: z ≠ 0; falla.
  - Cerrado por CL-79 (2026-10-11). La arena guarda la marca de llegada del spot (`CoinArena::spot_ts_ms`), escrita por `update_spot_data` con el reloj de eventos del núcleo (`registry.get_event_time()`, la marca del último evento del futuro procesado): el bookTicker de spot no trae la hora del exchange, y así el TTL se mide con el mismo reloj en vivo y en los replays. El núcleo calcula `spot_fresco` (spot > 0 y edad ≤ 30 s) y lo usan los tres consumidores: la SDE OU sólo se alimenta con spot fresco y se abstiene si no lo es; el z de la base (CL-78) vale 0 con spot caducado; y el sesgo de spot sobre el ML (`spot_bias`, MOD2/7-028) tampoco lee un spot caducado. La frescura es por llegada, no por cambio de valor: un spot repetido con el mismo precio (el sondeo REST en un mercado quieto) sigue fresco (`cl79_spot_repetido_sigue_fresco`). Pendiente fuera de Línea C: el demonio de evolución (`online_daemon.rs:170, :1006, :1158`) también lee el spot sin su edad (aviso en el buzón).
- Dueño: Línea C (núcleo). data-pipeline debe publicar la edad del spot.
- Escépticos: confirmado.
  - Corrección (refutar): Cambiar 'en cada tick' por 'en cada evento con entradas permitidas' (lib.rs:4950 corta los eventos de libro). Mencionar el sondeo REST de 1 s como mitigación de la probabilidad, no del defecto.
  - Corrección (completitud): Añadir a la consecuencia viva que la edad del spot hace falta tanto para el TTL como para no confundir el retraso del refresco con una dislocación. El arreglo es el mismo: data-pipeline publica la edad del spot (o su ts) y el núcleo la usa.

### F3.7-04. StatArb sigue heredando z y t½ de otra moneda por la caída al global; las API estrictas no se usan

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/strategy-core/src/stat_arb.rs:388-418; crates/god-engine-core/src/lib.rs:4485-4491; crates/omniscient-registry/src/lib.rs:304-330
- Causa: `set_reg` escribe la clave global además de la escopada. El lector usa `get_scoped_parameter`, que cae al global cuando faltan las claves del símbolo y de la moneda. Ningún lector vivo usa `get_scoped_strict`.
- Consecuencia viva: Lo que se añade: con dos monedas, la que no tiene SDE madura vota con la z y la t½ de la última moneda que escribió, pero la guarda la compara con SU τ*. Lo mismo vale para VECM y conformal con `vecm_zscore` en el arranque de cada moneda.
- Prueba que falla hoy: Los tests (1)/(2) de LC-F1-07, sin cambios. No añado otro.
- Dueño: Línea C (lectores de strategy-core y signal-engine)
- Relación con hallazgos anteriores: LC-F1-07 (causa raíz LC-F1-06)
- Escépticos: confirmado.
  - Corrección (refutar): Marcarlo como duplicado de LC-F1-07 sin prueba nueva. Mover la interacción con F3.7-05 (StatArb de una moneda sin spot lee la z global de otra en lugar de vecm_zscore) a la nota de LC-F1-07.

### F3.7-05. El mismo escalar entra en hasta tres votos del consenso

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:1056-1067; crates/strategy-core/src/vecm_arbitrage.rs:175-203; crates/strategy-core/src/stat_arb.rs:413-418; crates/signal-engine/src/conformal_reversion_filter.rs:183
- Causa: VECM siempre, StatArb cuando falta su clave propia y el filtro conformal leen `vecm_zscore`. Con |z| > 1,5, StatArb y VECM votan el mismo signo por construcción (tanh frente a rampa).
- Consecuencia viva: El consenso cuenta como dos o tres opiniones independientes lo que es una sola variable, y eso infla la confianza del compuesto.
- Prueba que falla hoy: cf3_dos_votos_del_consenso_no_son_la_misma_variable (crates/strategy-core/tests/cf3_votos_misma_variable.rs): en un registro que sólo tiene `vecm_zscore` ∈ [−3, 3], StatArb y VECM coinciden en signo en todos los casos activos. Determinista; falla.
- Dueño: Línea C
- Relación con hallazgos anteriores: LC-F3-14
- Escépticos: disputado.
  - Corrección (refutar): Formular como 'VECM y conformal siempre; StatArb sólo mientras no exista ninguna clave statarb_ou_zscore (ni propia ni global)'. Es duplicado de LC-F3-14 (plan:1081), y lo nuevo es que conformal también lo lee.
  - Lente completitud, dudoso: El test es determinista y falla, pero ejercita un camino que en vivo casi no se recorre. StatArb sólo lee vecm_zscore si `statarb_ou_zscore` falta en el símbolo, en la moneda Y en el global (stat_arb.rs:387-418). Como set_reg escribe el global (lib.rs:4486), basta con que UNA moneda madure su SDE (10 pares, unos 10 s con spot) para que el respaldo deje de usarse en todas. La duplicación que sí está viva es otra, y el auditor no la vio. Sin par spot (o con el spot sin publicar), vecm_basis_z sale por la rama `kline_ema_slow` (lib.rs:5219-5222), que es la MISMA expresión que `quantum_position_deviation` (lib.rs:5226-5231). Ese valor lo leen QuantumOscillatorEngine (quantum_oscillator.rs:147-153, voto (−k·x − λ·x³)·e^{−αx²}, de signo −sign(x)) y, como respaldo, SolitonWaveEngine (soliton_wave.rs:165-171). Con spot, VECM y conformal leen la misma base. Conformal sólo vota cuando la tendencia se opone a z (conformal_reversion_filter.rs:166-169), así que su redundancia es parcial.
  - Corrección (completitud): Reescribir la consecuencia: «con spot, VECM y conformal (condicionado a la tendencia) leen la misma base; sin spot, VECM, conformal y el oscilador cuántico leen el mismo escalar, (mid − EMA lenta)/ATR». Sustituir la prueba por la del faltante 3.

### F3.7-06. Régimen global: sólo con eventos de libro del slot 0, en reloj de eventos, sin τ en `allow_trade`, sin calibrar y con un respaldo inalcanzable

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:2600-2703; crates/risk-engine/src/orchestrator.rs:100-112, :158-200; crates/god-engine-core/src/stateful_engine.rs:186-188
- Causa: El símplex se recalcula sólo en eventos de libro del slot 0. La tendencia usa EMAs de 20/200 eventos, así que su escala física depende de la tasa de eventos. `allow_trade` no recibe la τ de la orden. Las sigmoides tienen anchos literales y no hay calibración. El respaldo discreto exige p_crash ≤ 0, y eso no ocurre tras el primer cálculo.
- Consecuencia viva: El veto duro y la contracción de margen de todas las monedas dependen de la tasa de eventos de libro del slot 0 y de unas pseudoprobabilidades sin calibrar, con el mismo umbral para cualquier τ.
- Prueba que falla hoy: cf3_regimen_no_depende_de_la_tasa_de_eventos (crates/god-engine-core/tests/cf3_regimen_tasa_eventos.rs): la misma trayectoria de precio con y sin un libro duplicado a +500 ms; espera |Δp_bull| < 0,05. Predicción: incierta.
- Dueño: Línea C
- Relación con hallazgos anteriores: FA-7 (C-F2) y LC-F1-11. Lo que se añade: no recibe τ, el respaldo es inalcanzable y las 8 claves no tienen lector.
- Escépticos: confirmado.
  - Corrección (refutar): Corregir la cota (p_crash ≥ ≈ 3,7e-44/suma, estrictamente > 0) y decir que el respaldo es inalcanzable SIEMPRE en vivo, también antes del primer cálculo. La prueba propuesta no demuestra el defecto (ver pruebas).
  - Corrección (completitud): Añadir a la causa: «sin marca de tiempo; si el slot 0 deja de recibir libro, el veto 0,90 y la contracción deciden para todas las monedas con el último valor». Corregir la prueba como se indica abajo.

### F3.8-01. El sesgo spot aditivo rompe el espejo de K-06/K-26 cuando la base no es 0,5

- Bloque: F3.8, Tests de #713/#714 y ledger de decisión viva.
- Ubicación: crates/god-engine-core/src/lib.rs:4440-4447 (spot_bias); calibration.rs:285-289 (compose_ml_prob); consumidores lib.rs:1629 (puertas), :4905-4906 (conformal), :4937 (nn_score) y :4870 (ml_prob con sesgo)
- Causa: El sesgo se suma en espacio de probabilidad: p + s, con s = ±0,05, saturado desde 0,5 pb de basis. Después se mide con la divergencia normalizada, que divide por (1 − b) hacia arriba y por b hacia abajo. Con b = 0,25 el mismo |s| vale d = +0,067 a favor del largo y d = −0,20 a favor del corto, así que el corto recibe 3 veces el apoyo. La simetría que declaran K-06/K-26 (calibration.rs:345-357) sólo se cumple con b = 0,5. El gate ML se salva porque lee la versión pura (lib.rs:4869 y :7810-7816).
- Consecuencia viva: El host alimenta el spot en vivo (god_engine.rs:1858 → spot_feed.rs:141/221 → state.rs:715), y cualquier |basis| ≥ 0,5 pb satura el sesgo. Con b ≈ 0,25: - En las puertas, el largo con +5 pb sube su confianza ×1,033 y el corto espejo con −5 pb sube ×1,10. - El p_win del conformal queda en 0,30 frente a 0,40. - nn_score aporta +0,067 frente a −0,20 al compuesto. Es un sesgo medible a favor de los cortos en la confianza y en el tamaño. Backtest, T-1 y T-2 no tienen spot, así que no lo ven.
- Prueba que falla hoy: cf3_espejo_del_sesgo_spot_con_base_honesta, en crates/god-engine-core/tests/cf3_espejo_conformal_contract.rs. Monta dos núcleos con un bosque de sólo sesgo de base 0,25 y el calibrador caliente, y compara basis +5 pb con −5 pb. Hoy d_A + d_B = −0,133 (esperado 0 ± 1e-6), y el p-valor del largo en A (0,3171) difiere del corto en B (0,4146).
- Dueño: Línea C (lib.rs fuera de las ramas 1-10 y calibration.rs)
- Relación con hallazgos anteriores: LC-F1-22. Lo amplío: LC-F1-22 mide la pendiente ×20 y el cruce del veto F-009; aquí se añade la ruptura del espejo con base honesta y su paso al conformal y a nn_score.
- Escépticos: confirmado.
  - Corrección (refutar): Presentarlo junto con F3.8-02 como una sola causa raíz: la divergencia normalizada y p_win_directional no son consistentes en espejo a primer orden para b ≠ 0,5. Mantener este test como la prueba de espejo sin modelo. Corregir la cita del escritor a spot_feed.rs:115. Lo que añade sobre LC-F1-22 es el espejo; la pendiente ×20 ya está allí.
  - Corrección (completitud): Desde la completitud, la consecuencia viva está infradimensionada en dos puntos. (1) La confianza que sale de las puertas va íntegra al risk-engine: - veto REJ_CONFIDENCE (risk-engine/src/lib.rs:1092-1099); - convicción del micro-Kelly (:565-567); - exposición (:584-586); - tope de apalancamiento conf_t (:1265); - Platt → win_probability → veto REJ_EV (lib.rs:7566-7581 y risk-engine/src/lib.rs:1134-1189). Con Platt frío (a = 1, b = 0, calibration.rs:87-93) la p que juzga el EV es la confianza misma. El sesgo a favor del corto cruza, por tanto, dos vetos vivos, no sólo el tamaño. (2) La ml_prob con sesgo también se publica como ml_prob_motor (lib.rs:4877). Ese valor lo lee FlowExcitationConfluence (flow_excitation_confluence.rs:170, con lift absoluto en :192-205; zona Qoder), un camino que el auditor no cuenta. El criterio de aceptación d_A + d_B = 0 da por hecho que el espejo correcto es la divergencia normalizada. Si se arregla aplicando el sesgo en log-odds, como sugiere la tabla de literales, la prueba seguiría roja. F3.8-01 y F3.8-02 tienen que compartir una sola definición de espejo antes de redactar ninguna de las dos pruebas.

### F3.8-02. Las puertas K-06 (y el gate K-23) no son espejo bajo el modelo de primer toque del propio repo: favorecen a los cortos

- Bloque: F3.8, Tests de #713/#714 y ledger de decisión viva.
- Ubicación: crates/god-engine-core/src/lib.rs:1624-1641 (puertas_del_continuo), :7862-7879 (gate ML) y calibration.rs:331-343. Modelo de referencia: crates/risk-engine/src/tp_sl.rs:469-489. Etiqueta: src/bin/train_forest.rs:1391-1392.
- Causa: p es P(TP del largo antes que SL) con TP = 2·SL. Bajo browniano con deriva, el corto con deriva −μ tiene las mismas probabilidades que el largo con +μ, así que su p espejo es p_largo(−μ). Eso no es la reflexión lineal de p_largo(μ) alrededor de la base. Ejemplo con b = 1/3: μ = 4e-4 da d_largo = 0,231 y d_corto espejo = 0,399. En log-odds el par está casi equilibrado (+0,642 frente a −0,691). Además conviven dos definiciones vivas de espejo: R-15 refleja la deriva exactamente (risk-engine/src/lib.rs:956-960), mientras K-06/K-23/K-26 reflejan la probabilidad linealmente.
- Consecuencia viva: Toda intención rápida o lenta pasa por las puertas (lib.rs:6399, :6691). Con base honesta < 0,5: - Un corto con el modelo a favor recibe hasta 1,7 veces el impulso de confianza del largo espejo. - El veto de contradicción fuerte salta para el largo con una deriva adversa de 1,07e-3; el corto necesita 1,80e-3. El sesgo hacia los cortos afecta a sig_conf y al tamaño (confianza × (1 + 0,5·d)).
- Prueba que falla hoy: cf3_puertas_espejo_bajo_primer_toque (mod tests_cf3_puertas_espejo en lib.rs). Usa tp 0,0036, sl 0,0018, σ 0,0018 y base p_largo(0) = 1/3, y para μ ∈ {1e-4, 4e-4, −1,1e-3} compara puertas(Largo, p_largo(μ)) con puertas(Corto, p_largo(−μ)). Hoy: - μ = 1e-4: confianza 0,8226 frente a 0,8435. - μ = 4e-4: 0,8924 frente a 0,9596. - μ = −1,1e-3: el largo se veta (d = −0,81) y el corto espejo sobrevive con ×0,707 (d = −0,586).
- Dueño: Línea C
- Relación con hallazgos anteriores: LC-F1-38 y LC-F1-43. Los amplío: LC-F1-43 dice que p_win_directional no es P(gana corto) en el conformal, y LC-F1-38 discute el gate S-6. Aquí se cuantifica en las puertas, que deciden cada intención, y se muestra que el veto es asimétrico.
- Escépticos: confirmado.
  - Corrección (refutar): Fusionarlo con F3.8-01 y anotarlo como ampliación de LC-F1-43/LC-F1-38, no como causa nueva. El arreglo es único: un mapa de espejo que reciba la geometría TP/SL, o trabajar en log-odds o en deriva. Las puertas, el gate, el conformal y nn_score deben heredarlo, no parchearse por separado. Añadir el argumento de primer orden, que cierra la disputa de LC-F1-38.
  - Corrección (completitud): Hay que añadir tres cosas. (a) La asimetría no se queda en «sig_conf y tamaño». Llega a REJ_CONFIDENCE (risk-engine/src/lib.rs:1098) y, vía Platt en frío, a REJ_EV (:1134-1189). Con base < 0,5 el veto de confianza deja pasar más cortos que largos espejo. Si se mide con el contrafactual, la severidad podría subir a alta. (b) En la decisión conviven TRES convenciones de espejo, no dos: - deriva exacta en R-15 (risk-engine/src/lib.rs:956-960); - reflexión normalizada en K-06, K-23 y K-26; - distancia absoluta en probabilidad en el consejo: ml_opinion (consejo_seniors.rs:385-394), Metacognitivo (:811) y Teleonomía (p − b)·2 (:871). Es LC-F1-66 y la fila abierta C-16/18 (LEDGER:37). FlowExcitation añade un lift absoluto (flow_excitation_confluence.rs:192-205). (c) Ninguna de esas convenciones coincide con el espejo de primer toque. El siguiente paso no es parchear cada camino, sino una función única p_corto(p) = p_largo(−p_largo⁻¹(p)), o un segundo modelo con la etiqueta del corto (es lo que pide GLM en C-16/18). Todos los caminos tendrían que llamarla.

## Severidad baja

### F3.3-10. El colchón de free_cap es un veto binario silencioso, duplicado en línea, y su test de espejo prueba una copia

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:7899-7915 y :7938 (sin else hasta lib.rs:8162). quantum-arena/src/spectral_regime.rs:164-190. god-engine-core/tests/cushion_symmetry_contract.rs:13-23.
- Causa: Si margin_req > free_cap, la orden desaparece sin código REJ ni contador. La fórmula (1−0,95·flux).clamp(0,05; 1) con marea adversa es idéntica a la de SpectralRegimeField::margin_multiplier, que no tiene llamador vivo. El test re-implementa el cierre.
- Consecuencia viva: No hay telemetría de cuántas entradas veta. Un cambio en el código vivo no rompe el test.
- Prueba que falla hoy: T12 cf3_colchon_usa_la_funcion_con_contrato (test de fuente: lib.rs no llama a margin_multiplier).
- Dueño: Línea C.
- Relación con hallazgos anteriores: K-24 y F2.3-05 (crash_flux en ruido). Añade que no hay contador y que el test existente es tautológico.
- Escépticos: confirmado.
  - Corrección (refutar): Citar FICHAS C-F2 :400 y escoger una sola dirección. Lo nuevo es el test tautológico y la falta de contador.
  - Corrección (completitud): Hay que añadir dos cosas: - Los vetos de puertas_del_continuo tampoco tienen contador ni EV (faltante 5). - La marea del colchón (global_coherence) y crash_flux dependen de la ganancia epigenética espectral (faltante 1). La severidad baja es razonable para el colchón aislado.

### F3.3-11. Documentación falsa y código muerto en la ruta de la confianza

- Bloque: F3.3, Confianza.
- Ubicación: lib.rs:152-153 (sig_conf «Bayesiana calibrada»), lib.rs:131-133 (doc de conviccion_de_rama), lib.rs:7490-7504 (ml_align_desalineado con cuerpo vacío y comentario «cuenta»), lib.rs:7133-7138 (let _ = p_transition), lib.rs:7198-7203 (_global_leverage).
- Causa: Quedan restos de olas anteriores. El comentario «Sólo diagnóstico: cuenta la desalineación» describe un if sin cuerpo.
- Consecuencia viva: No tiene consumidor vivo: induce a error en las auditorías y gasta cálculo en la ruta caliente.
- Prueba que falla hoy: T3 refuta el doc de lib.rs:131-133. Para el resto basta un grep: 'ml_align_desalineado' sólo aparece en lib.rs:7490 y :7501.
- Dueño: Línea C.
- Escépticos: confirmado.
  - Corrección (refutar): Quitar p_transition de lo nuevo, o citarlo como duplicado.
  - Corrección (completitud): Hay que añadir otro resto. En el host, is_high_confidence (god_engine.rs:4158-4161) se calcula con ml_prob > 0,80 o < 0,20, valores absolutos que ignoran ml_model_base, y sólo sirve para una etiqueta de log (god_engine.rs:4571). Es cosmético, pero desmiente que el grep de 'confidence' en el host no dé coincidencias.

### F3.4-06. El veto «absoluto» de drawdown del consejo es inalcanzable: el risk-engine veta antes en d* ≈ 0,632

- Bloque: F3.4, Vetos.
- Ubicación: consejo_seniors.rs:698-700, :1272-1281; lib.rs:7581-7596; risk-engine lib.rs:309-344; risk-engine drawdown.rs:170-193
- Causa: El umbral del consejo es 0,85-0,95. El risk-engine rechaza antes, con un umbral = lerp(dd_max_medido, d*, micro_w).min(d*), donde d* = drawdown_de_falsacion(0,5, 0,05) ≈ 0,632, y lo hace con el mismo unified_capital y el mismo peak_capital. Ningún DD que llegue al consejo supera d*.
- Consecuencia viva: Ninguna: la rama absoluta de Riesgo es código muerto. Se añade la causa exacta (d* del risk-engine, no sólo el suelo de 3 USD que cita C-13) y la prueba.
- Prueba que falla hoy: cf3_veto_dd_consejo_alcanzable. Con dd = 0,999·d* y τ ∈ {30 s, 10 min, 1 h, 12 h}, ninguna deliberación da vetoed_by == Some(Riesgo).
- Dueño: Línea C (consejo)
- Relación con hallazgos anteriores: LC-F3-11
- Escépticos: confirmado.
  - Corrección (completitud): La prueba propuesta puede pasar en falso. integrity_failure devuelve vetoed_by = Some(SeniorRole::Riesgo) (consejo_seniors.rs:334-347), así que cualquier payload fuera de dominio «alcanza» el veto de DD. Hay que mirar la opinión del asiento: deliberar_traced(&p, 0.5, None).opinions.iter().any(|o| o.role == SeniorRole::Riesgo && o.is_veto).

### F3.4-07. El consejo no está en el registro de vetos: V-LOGIC-007 lleva su nombre pero pinea REJ_CONFIDENCE del risk-engine

- Bloque: F3.4, Vetos.
- Ubicación: risk-engine/src/veto_registry.rs:221-231, :64-379, :665-706; risk-engine/tests/veto_logic_contracts.rs:57
- Causa: El test pineado sólo llama a RiskEngine::evaluate_quantum_order y comprueba ultimo_rechazo() == REJ_CONFIDENCE; nunca ejecuta deliberar. No tienen entrada: - ninguno de los seis vetos de asiento; - Stouffer, el muro, el CVD, el veto ML −0,80 y D-472; - el conformal y la salida tóxica. La cobertura sólo exige los REJ nombrados. Además, la tarea situaba el registro en god-engine-core y está en risk-engine.
- Consecuencia viva: Sin efecto directo en las decisiones. El registro certifica la «confianza del consejo» sin ejecutarla y deja fuera los vetos vivos del bloque.
- Prueba que falla hoy: cf3_v_logic_007_pinea_el_consejo. Lee el cuerpo del test pineado con include_str! y exige que contenga «deliberar». Hoy no lo contiene.
- Dueño: Línea C (entradas del registro), coordinando con quien mantenga risk-engine
- Escépticos: disputado.
  - Lente completitud, dudoso: Es cierto que ningún veto de asiento del consejo, ni Stouffer, muro, CVD, el veto ML −0,80 o D-472, tiene entrada en el registro. Pero la lectura de V-LOGIC-007 está mal. Su causa es «fusión espectral/council por debajo del umbral del gen» y sus datos son «council_fused, thresholds del genoma activo» (veto_registry.rs:221-231): describe REJ_CONFIDENCE del risk-engine, y su test lo pinea correctamente (veto_logic_contracts.rs:57-70). No es un certificado falso del consejo, sólo un nombre engañoso. La cabecera del registro (veto_registry.rs:60-63) declara la deuda: «Las puertas internas del consejo/ramas se añaden por ola». El hallazgo queda en deuda declarada más un renombrado (baja).
  - Corrección (completitud): Hay que reformularlo como «el registro no cubre los vetos del consejo ni los del núcleo fuera del risk-engine (deuda declarada en veto_registry.rs:60-63); V-LOGIC-007 debería llamarse “confianza de la intención (REJ_CONFIDENCE)”». La prueba debe exigir entradas para los vetos de asiento, no que el test de V-LOGIC-007 ejecute deliberar.
  - Lente completitud, dudoso: La premisa está mal leída. V-LOGIC-007 tiene como causa «fusión espectral/council por debajo del umbral del gen» y como datos «council_fused» (veto_registry.rs:221-231): describe REJ_CONFIDENCE del risk-engine, y su test (veto_logic_contracts.rs:57-72) es coherente con esa causa. El nombre «confianza del consejo» confunde, pero no certifica a deliberar. El defecto real es otro: los vetos de consejo_seniors no tienen entrada. El propio registro lo declara pendiente («Las puertas internas del consejo/ramas se añaden por ola», veto_registry.rs:60-63). Está relacionado con LC-F1-51 (cobertura nominal) y con el hallazgo de C-F1 de la fecha ausente en V-LOGIC-007.
  - Corrección (completitud): Reformular el hallazgo: «los seis vetos de asiento y la aprobación del consejo no tienen entrada (deuda declarada en veto_registry.rs:60-63), y V-LOGIC-007 lleva un nombre engañoso». Hay que sustituir la prueba: la de hoy empujaría a reescribir el test de REJ_CONFIDENCE (ver «pruebas»).

### F3.4-08. El test de integración de K-26 corre con base 0,5, no con la 0,25 que declara

- Bloque: F3.4, Vetos.
- Ubicación: god-engine-core/tests/conformal_direction_symmetry_contract.rs:75-108; lib.rs:4665-4675, :4886, :4890-4916
- Causa: El fixture escribe ml_model_base = 0,25 por moneda. El núcleo, sin bosque, calcula una base de 0,5 (unwrap_or(0.5)) y la publica con set_reg en lib.rs:4886, antes del bloque conformal del mismo tick. El test sólo comprueba que los valores sean finitos y la aceptación durante el calentamiento, que acepta siempre. Por eso no distingue base 0,25 de 0,5, ni comprueba la simetría que dice probar.
- Consecuencia viva: Sin efecto en las decisiones. Es un certificado falso de K-26 que se suma a la pregunta 8 del plan. El arreglo es del test: inyectar un bosque con base 0,25, o probar el bloque conformal como función pura.
- Prueba que falla hoy: cf3_k26_corre_con_base_025. Después de process_tick_dual exige get_for_coin_or(0, «ml_model_base») == 0,25; hoy da 0,5.
- Dueño: Línea C (revisión en C-F3, recibo en C-F10), con aviso a AGY como autor de #714
- Escépticos: confirmado.
  - Corrección (completitud): El arreglo es del test de #714, no del núcleo (ver la prueba cf3_k26_corre_con_base_025 en «pruebas»).

### F3.5-06. El inventario de modelos versionado no coincide con el manifiesto, y la documentación del registro es falsa

- Bloque: F3.5, Modelos.
- Ubicación: models/; config_dir/models_manifest.json; crates/god-engine-core/src/ml_registry.rs:4-5; crates/god-engine-core/src/lib.rs:4922
- Causa: models/ está en .gitignore, pero se versionan 4 artefactos, de los que 2 son válidos. El manifiesto (último commit 800ddd42, generado el 2026-10-02) describe otra máquina: 7 _MOTOR válidos, 10 FDUSD inertes y 3 inválidos. ml_registry.rs:4-5 dice que un clon fresco mide el piso sin modelos, y no es así. El heartbeat de lib.rs:4922 lista NEAR, ATOM y BNB a mano. Ningún consumidor vivo lee el manifiesto: su único lector es src/bin/model_manifest.rs.
- Consecuencia viva: La respuesta a «qué modelo decide» depende de la máquina. El T-1 y el forense en la nube miden un sistema con 2 modelos, mientras el operador opera con 7.
- Prueba que falla hoy: cf3_models_versionado_coincide_con_manifiesto: el manifiesto declara ADAUSDT_MOTOR.json válido y no inerte, pero no existe en models/.
- Dueño: Línea C (documentación y heartbeat); pedido a Línea E: decidir si models/ se versiona entero o no se versiona, y regenerar el manifiesto en consecuencia.
- Escépticos: confirmado.
  - Corrección (refutar): Quitar el T-1 de la consecuencia. Lo que difiere entre máquinas es el forense, la demo y la telemetría de cobertura.
  - Corrección (completitud): La severidad baja es correcta. La prueba codifica una política (versionar models/ entero) que decide Línea E: debe ir como pedido, no como test de Línea C.

### F3.6-09. Documentación falsa y API pública sin consumidor en los motores

- Bloque: F3.6, Motores de señal.
- Ubicación: perceptron_gate.rs:50-58, :66-75 («banda operativa 8..23»: k = 8 son 65 µs, y la banda operable es 18..22 según temporal_spectrum.rs:108-129); supersonic_shockwave.rs:167 («precio/s») y :72 (carácter «超» en el doc); soliton_wave.rs:212-216 («1 bp/s ≈ tanh(1)»: la entrada va por evento, no tiene signo, y 1 pb da tanh(10)); funciones que sólo usan los tests (véase retiradas)
- Causa: Los comentarios de las reparaciones no describen el código vivo, y la API pública muerta da a entender que la teoría está implementada.
- Consecuencia viva: Ninguna directa: desde CL-32 las escalas por debajo de la resolución ya no votan (lib.rs:2525-2528).
- Prueba que falla hoy: cf3_6_perceptron_banda_espectral_es_la_operable: el voto en la escala de 65 µs vale lo mismo que en la de 18 min, porque las dos dan infer(1, 1). Falla hoy.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): Las líneas se han movido: el carácter «超» está en supersonic_shockwave.rs:71 y el doc «precio/s» en :168.

### F3.7-07. El núcleo calcula una cotización maker en cada tick y la descarta

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:8170-8209, :2803; crates/backtest-engine/src/bin/continuous_evolution_backtest.rs:607
- Causa: `generate_quote` corre cuando la latencia es ≤ 25 ms (recorre los slots para el inventario y lee 4 genes), y los dos llamadores ignoran el tercer valor devuelto.
- Consecuencia viva: CPU por tick sin ninguna decisión. Los genes maker_spread_pct y maker_obi_threshold no gobiernan nada.
- Prueba que falla hoy: cf3_el_nucleo_no_calcula_cotizacion_sin_consumidor (crates/god-engine-core/tests/cf3_maker_sin_consumidor.rs): espera que `process_tick_dual` devuelva None en el tercer valor. Predicción: Some; falla.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): Completar la lista de llamadores (wrapper process_tick → evolver.rs:430 y darwin.rs:368; backtest :695; nano_benchmark) y cambiar 'cada tick' por 'cada evento no bloqueado (trades y velas)'. Al retirarlo también cambia la firma de process_tick (lib.rs:8218).
  - Corrección (completitud): Cambiar a «en el núcleo vivo no gobiernan nada; maker_spread_pct sigue en el simulador de backtest-engine/src/lib.rs:311-312 y tensor_poly_a en flow_impulse.rs:154. La retirada exige el oráculo T-1».

### F3.7-08. Heap por evento, E/S síncrona de ficheros por cierre y justificaciones sin lector por deliberación

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:1978, :3098, :4655-4723, :5275-5296, :5417, :5511 (por evento); :4045, :4133-4156, :4253-4269, :4290, :4378 (por cierre); crates/metacortex-engine/src/consejo_seniors.rs:431-921, :1124-1174, :1340-1361 (por deliberación)
- Causa: Claves de texto y nombres de símbolo construidos en cada evento. En cada cierre, apertura y escritura de un CSV con 55 `format!` en el hilo del núcleo, y un `TraumaRecord` con E/S. En cada deliberación, 11-12 `format!` y clones de opiniones que el núcleo no lee (sólo lee approved, final_signal y signals).
- Consecuencia viva: Latencia: la E/S síncrona cae justo en el cierre, que es cuando la latencia importa. No cambia decisiones.
- Prueba que falla hoy: cf3_evento_de_libro_estacionario_no_asigna (crates/god-engine-core/tests/cf3_libro_sin_asignar.rs): con un asignador contador, espera < 1 asignación por evento de libro tras el calentamiento. Predicción: ≥ 10 (el perfil midió 11,0); falla.
- Dueño: Línea C propone; cada dueño de archivo decide (metacortex-engine para el consejo).
- Relación con hallazgos anteriores: LC-F3-15 y perfil P5. Lo que se añade: el inventario por cierre y por deliberación, y la E/S síncrona.
- Escépticos: confirmado.
  - Corrección (refutar): Reclasificar :5275-5296, :5417 y :5511 como 'por evento con entradas permitidas (trade o vela)'. Añadir el stat de Path::exists por cierre y la E/S de record_trauma. La prueba sigue fallando con ≥ 6 asignaciones por evento de libro.

### F3.7-09. Publicaciones sin lector: las 8 claves `market_regime_*` y `statarb_beta`

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/god-engine-core/src/lib.rs:2684-2694, :5268
- Causa: Se escriben en cada evento de libro del slot 0 (8 claves globales) y en cada tick con SDE madura (`statarb_beta`, 3 claves por `set_reg`), y no hay lector fuera de tests. Con #720 cada escritura cuesta unos 73 ns más (perfil §5b).
- Consecuencia viva: Coste de CPU y ruido en el registro. Ninguna decisión.
- Prueba que falla hoy: Guarda léxica: toda clave publicada por lib.rs debe tener un lector fuera de /tests/. Hoy falla para estas 9 claves (`rg '"market_regime_' --type rust` sólo aparece en cushion_symmetry_contract.rs, que las escribe él mismo).
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): Excluir scan_all de audit-engine como 'lector'. statarb_beta se escribe por evento no bloqueado con SDE madura, no por tick.
  - Corrección (completitud): Ubicación: lib.rs:2687-2696, no :2684-2694.

### F3.7-10. Tres τ* de respaldo distintas, y la doc R6-B13 («único fallback») es falsa

- Bloque: F3.7, strategy-core, código muerto y montículo.
- Ubicación: crates/strategy-core/src/stat_arb.rs:10, :292-297; crates/god-engine-core/src/lib.rs:4956-4960 (30 000), :7667-7671 (1 138 000)
- Causa: `DEFAULT_DOMINANT_TAU_MS` = 1 138 419,6 sólo se usa en strategy-core. El núcleo usa 30 000 y 1 138 000 como respaldo de `temporal_spectrum.get(coin_id)`.
- Consecuencia viva: Prácticamente ninguna: los respaldos de lib.rs sólo actúan si `coin_id` está fuera de `temporal_spectrum`, y el lector casi nunca usa el suyo porque el núcleo publica `dominant_tau_ms` en cada tick. Es documentación falsa.
- Prueba que falla hoy: Guarda léxica: los tres sitios usan la misma constante. Hoy falla.
- Dueño: Línea C
- Relación con hallazgos anteriores: Relacionado con LC-F3-17.
- Escépticos: disputado.
  - Lente refutar, dudoso: El hecho es cierto: hay tres respaldos distintos (stat_arb.rs:10 = 1 138 419,6; lib.rs:4960 = 30 000; lib.rs:7671 = 1 138 000). Pero la doc de stat_arb.rs:6-9 dice 'Único valor para TODOS los caminos del motor' y lo compara con 'antes 30.0 s en update_with_clock vs 1138.0 s en evaluate_for_coin'. 'Motor' se refiere a StatArbEngine, y dentro de él el respaldo sí está unificado (stat_arb.rs:292-297 y :402). Calificar la doc de falsa es discutible. Sin consumidor que cambie decisiones, salvo coin_id fuera de temporal_spectrum.
  - Corrección (refutar): Reformular como 'tres constantes de respaldo para la misma τ* en dos crates (LC-F3-17)' y retirar 'doc falsa'. Severidad baja.

### F3.8-03. Certificaciones del ledger que no ejecutan el código que certifican (#713, #714, REV-10, R-15)

- Bloque: F3.8, Tests de #713/#714 y ledger de decisión viva.
- Ubicación: crates/god-engine-core/tests/puertas_del_continuo_symmetry_contract.rs:18 y :113-119; tests/conformal_direction_symmetry_contract.rs:75-110 y :112-125; tests/prospect_pressure_integration_contract.rs:151-197; docs/audit/LEDGER_DECISION_VIVA_2026-10-10.md:205-209 y :255
- Causa: - #713 prueba una función pura que ya tiene test en el crate (calibration.rs:724-763), con un bucle tautológico. - #714 escribe una base que el núcleo pisa (lib.rs:4886) y evalúa con el calibrador frío (conformal.rs:151-153). - La guarda de fuente K-27 de #714 se cumple con el texto antiguo. - REV-10 recalcula Prospect fuera del núcleo. - El ledger cita para R-15 un archivo que no tiene pruebas de R-15.
- Consecuencia viva: No cambia ninguna decisión en vivo. Pero el ledger declara cerrados K-06, K-23, K-26, R-15 y REV-10 con pruebas que no detectarían una regresión. El T-1 no ve las puertas (LEDGER §7.4) y el T-2 no ve ni el gate ML ni el conformal (LEDGER:408-415), así que hoy nada protege esos cambios.
- Prueba que falla hoy: cf3_fixture_714_no_alcanza_la_base_que_declara, en tests/cf3_espejo_conformal_contract.rs. Reproduce el fixture literal de #714 y comprueba dos cosas tras el tick. Hoy falla en ambas: - Que el registro conserva ml_model_base = 0,25: vale 0,5. - Que conformal_p_value < 1: vale 1,0. Mutación de comprobación: cambiar lib.rs:4906 por `1.0 - ml_prob` deja verdes los 3 tests de #714. Cambiar lib.rs:1629 por `(ml_prob - ml_base) * 2.0` deja verdes los 4 tests de #713. Esto es razonamiento sobre el código; sin cargo no lo ejecuté.
- Dueño: AGY, autor de #713, #714 y REV-10 (petición con test). Las filas del ledger son de Línea E.
- Escépticos: confirmado.
  - Corrección (refutar): Corregir las citas del ledger, desplazadas una línea en la tabla §7: C-22 :200, C-W :201, C-02 :202, C-10 :203, R-15 :204, K-06 :205, K-23 :206, K-26 :207, K-27 :208, R-12 :209, R-13 :210, R-03 :211. El auditor da :201-:212; el freno :214 y los REV :246-:257 sí son correctos. Donde dice «LEDGER:205-209» debe decir :204-:208. La fila del T-2 «15 → 16» está en LEDGER:392, no en :404. Lo de 0 vetos del gate ML está en :413-414.
  - Corrección (completitud): El auditor cita REV-5 como «orchestrator.rs:365-387 y :518-527» sin decir el crate. Es crates/signal-engine/src/orchestrator.rs; risk-engine/src/orchestrator.rs es el de REV-4. Las líneas son correctas en signal-engine. Hay que añadir que la mutación de lib.rs:4906 tampoco la detectaría el T-2: según LEDGER:408-415, #714 no mueve las huellas.

### F3.8-04. La misma base del modelo pasa por cuatro guardas distintas según el camino

- Bloque: F3.8, Tests de #713/#714 y ledger de decisión viva.
- Ubicación: lib.rs:1624-1628 (puertas: (0,05; 0,95) abierto, si no 0,5); :7862-7865 (gate: sin guarda, sólo el clamp [0,01; 0,99] de calibration.rs:337); stateful_engine.rs:576-580 (nn_score: (0, 1), si no 0,5); lib.rs:4040-4050 (cierre: registro en (0,05; 0,95), si no el bosque). Productor: lib.rs:4671-4675.
- Causa: El núcleo ya recorta la base a [0,05; 0,95] (lib.rs:4675), pero las puertas aplican un intervalo abierto. Con un bosque de base ≤ 0,05 o ≥ 0,95, la base publicada vale exactamente 0,05 o 0,95. En ese caso las puertas caen a 0,5, mientras el gate, el conformal y el cierre usan 0,05. Además, el cierre K-26 lee la base en el momento del cierre, no la que había al entrar.
- Consecuencia viva: Sólo aparece con bosques de base extrema; las bases honestas medidas están en 0,23-0,30. Es la misma decisión leída con dos bases distintas según el camino. La fuente única debería ser una fn base_honesta().
- Prueba que falla hoy: cf3_puertas_y_gate_ml_leen_la_misma_base (en el crate), con base 0,05 y p = 0,04. El gate mide d = −0,2, una penalización. puertas_del_continuo devuelve Flat porque mide d = −0,92 contra 0,5. Esperado: Long con confianza 0,80·0,90.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): Una fn única de base honesta que todos los caminos llamen. Al cierre, guardar la base de la ENTRADA en la posición en vez de releer el registro (lib.rs:4040).
  - Corrección (completitud): La severidad baja es correcta: las bases medidas están entre 0,18 y 0,30. Se puede añadir que nn_score usa una cuarta guarda, (0, 1) → 0,5 (stateful_engine.rs:576-580). La función única de base honesta debería sustituir las cuatro.

### F3.8-05. hurst_mod del gate ML va con signo, aunque su comentario dice |H − 0,5|

- Bloque: F3.8, Tests de #713/#714 y ledger de decisión viva.
- Ubicación: crates/god-engine-core/src/lib.rs:7829-7834
- Causa: El comentario describe una modulación por |H − 0,50| en la que el browniano eleva el lift. El código es `1 − 0,40·(H − 0,5)` recortado: H = 0,1 da 1,16; H = 0,3, 1,08; H = 0,5, 1,0; H = 0,7, 0,92; H = 0,9, 0,84. Así, la antipersistencia exige más que el browniano, y un mismo |H − 0,5| da exigencias opuestas.
- Consecuencia viva: Actúa igual sobre los dos lados, así que no rompe el espejo. Es documentación falsa sobre una modulación viva del gate, que el T-2 no ejercita (0 vetos).
- Prueba que falla hoy: Primero hay que extraer, sin cambiar la conducta, `fn modulacion_lift_por_hurst(h: f64) -> f64` con la expresión de lib.rs:7832. Después, el contrato que declara el comentario falla: modulacion(0,3) == modulacion(0,7) da 1,08 frente a 0,92, y el máximo no está en H = 0,5.
- Dueño: Línea C
- Escépticos: confirmado.
  - Corrección (refutar): Severidad baja. Decidir la semántica (con signo o |H − 0,5|) a partir de la habilidad medida del bosque frente a H, corregir el comentario y retirar el clamp exterior muerto.
  - Corrección (completitud): No cambia nada desde esta lente. Además, hurst_val es el DFA por ticks (reloj de eventos). El caso de tiempo físico frente a eventos tampoco está cubierto en esta modulación.


## Pruebas de solución conocida (CL-72)

Las 62 propuestas de `PRUEBAS_C-F3_borrador_2026-10-11.json` están escritas, compiladas y ejecutadas en el PC del operador (Windows, x86_64-pc-windows-msvc, nightly-2026-06-30) sobre main 8f572f7c, que ya incluye CL-66…CL-71 y #721-#725. Son 71 funciones de test: 13 activas que pasan y 58 OPEN (`#[ignore = "OPEN F3.x-nn (dueño): causa"]`). Con `--include-ignored`, las 58 fallan en su aserción principal por la causa de su hallazgo; ninguna falla en una premisa. Sin ese flag, todos los archivos dan `ok`. Ninguna tarda más de 15 s en debug.

Resultado por prueba (archivo, hallazgo, dueño, estado, mensaje observado y notas): `docs/audit/PRUEBAS_C-F3_2026-10-11.tsv`.

| Crate | Activas (pasan) | OPEN (fallan por su causa) |
|---|---|---|
| god-engine-core (incluye `mod tests_cf3_puertas_espejo` en lib.rs) | 3 | 38 |
| signal-engine | 9 | 12 |
| risk-engine | 1 | 3 |
| metacortex-engine | 0 | 2 |
| strategy-core | 0 | 2 |
| backtest-engine | 0 | 1 |
| **Total** | **13** | **58** |

Lo que cambió respecto al borrador:

- **F3.6-04 cerrado por CL-66.** El núcleo ya no publica `atr_1s/5s/1m` y el coaxial se abstiene sin ellas. `cf3_6_coaxial_invariante_a_la_unidad_de_precio` queda activa con un control: con ATR por escala proporcionales al precio, el lector da el mismo voto no nulo (0,735754519) a 60 000 y a 0,06 USD.
- **F3.4-07: la prueba del borrador se descarta** porque su verdad es errónea. V-LOGIC-007 describe `REJ_CONFIDENCE` del riesgo y su test lo pinea bien. La sustituyen dos pruebas: una activa que fija esa lectura y una OPEN que exige una entrada del registro para cada veto de asiento del consejo (6 de 6 sin entrada).
- **F3.7-08, magnitud rebajada.** Un evento de libro asigna una vez, no 6-11: el `String` que crea `try_symbol(coin_id).unwrap_or_default()`. Los `format!`, los clones y `try_spec` ya no asignan tras las búsquedas sin asignación de AGY.
- **Rehechas porque la versión del borrador no podía fallar por la causa:**
  - VECM con base constante: el futuro alterno ±1e-5 anulaba los retornos de 60 s. Ahora el spot es un paseo aleatorio, el futuro lleva una prima fija y la verdad es de sesgo en media.
  - Régimen y tasa de eventos: con Hurst neutro la tolerancia absoluta nunca fallaba. Ahora una deriva de 5e-6 pone z en la zona sensible de la sigmoide y se compara la razón de medias (±10 %).
- **Adaptadas a la API viva:**
  - F3.2-01: eventos trade + libro, y el bosque direccional del oráculo T-1.
  - F3.2-05: el veredicto espectral se fija en el registro.
  - F3.6-01: el registro espejo ya no publica las tres ATR.
  - F3.4-05: réplica literal del predicado de la sonda, con comprobación del texto en lib.rs.
  - F3.4-06: el veto se lee en el asiento Riesgo vía `deliberar_traced`.
- **F3.2-04 (retirado por duplicar LC-F1-36)** queda como testigo de LC-F1-36. F3.3-11, F3.7-04, F3.7-09, F3.7-10 y F3.8-05 no tenían prueba en el borrador y siguen sin ella.

**Zonas.** El dueño de cada prueba OPEN es el del hallazgo: 48 Línea C, 6 Qoder (F3.1-01…06), 1 Codex (F3.1-07), 1 AGY (F3.8-03), 1 Línea E (F3.5-06) y 1 que decide cada dueño de archivo (F3.7-08). Algunos hallazgos asignados aquí a Línea C caen en zonas que el plan (§1.2) da a otros: el consejo (F3.4-01, -02 y -06, AGY), la coherencia del orquestador (F3.3-05, Qoder) y `calibration.rs` (F3.5-04 y F3.8-01, Codex). Línea C confirma la zona con su dueño antes de arreglarlos.
