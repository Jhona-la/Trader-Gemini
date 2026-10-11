# Hallazgos de la fase C-F2 (Línea C): teoría por dominio

Base: `main` en c4e8e285. Siete dominios, cada uno con un auditor que escribió una ficha por archivo y diseñó pruebas de solución conocida, y dos escépticos (lente «refutar» y lente «completitud»). Fase de sólo lectura: los auditores no compilaron ni ejecutaron nada. Las pruebas se escribieron y ejecutaron después (§ «Pruebas de solución conocida»).

Convenciones:

- Los ids son por dominio: F2.1 riesgo y cartera, F2.2 estadística y calibración, F2.3 espectro y régimen, F2.3b física y fractales, F2.4 procesos puntuales y contagio, F2.5 conducta de masas, F2.6 «cuántico».
- `lib.rs` sin ruta es crates/god-engine-core/src/lib.rs (el núcleo) y `god_engine.rs` sin ruta es src/bin/god_engine.rs (el host). Las citas son de c4e8e285. Entre c4e8e285 y `main` 1d1f8bdd el código de producción sólo cambia en `lib.rs` y `risk-engine/src/lib.rs` (CL-62, siete líneas cada uno), `state.rs`, `paths.rs` y `symbol_registry.rs` (CL-61 y CL-62) y `feynman_propagator.rs` (CL-59). Los auditores ya tuvieron en cuenta CL-59 en los hallazgos de Feynman.
- Severidad: alta cambia decisiones vivas de dinero o de veto de forma errónea; media es un sesgo medible o una teoría mal especificada con consumidor vivo; baja no tiene consumidor vivo, o es documentación falsa o cosmética.
- «Dueño» es quien arregla. Las zonas ajenas (Codex, Qoder/GLM, AGY, Línea E) reciben un pedido con su test en rojo; Línea C no toca su código.
- Las fichas por archivo, las respuestas a las preguntas del plan, los literales, las retiradas y lo que señaló el escéptico de completitud están en `docs/audit/FICHAS_C-F2_2026-10-11.md`.

Recuento: 67 hallazgos, 6 altos, 32 medios y 29 bajos. Los dos escépticos confirman 66; uno (F2.5-01) queda reformulado. La revisión cruzada del PC (CL-65) añade dos sub-hallazgos con prueba propia, F2.2-03b y F2.3-08b (baja), descritos dentro de F2.2-03 y F2.3-08, y una segunda causa en F2.2-10.

| Id | Hallazgo | Severidad | Estado | Dueño |
|---|---|---|---|---|
| F2.2-01 | El vigilante de rollback apuesta a favor del edge y dispara por caída de evidencia: revierte genomas rentables | alta | confirmado | Línea C, C-F2.2 (online_daemon.rs y ville_e_process.rs) |
| F2.3-01 | La τ operativa unificada cambia de escala según el signo de un IC sin validar | alta | confirmado | Línea C (C-F3, τ dominante) |
| F2.3b-01 | El DFA clasifica un paseo aleatorio como persistente en torno a la mitad de las veces y la duración se duplica | alta | confirmado | Línea C: el estimador y hurst_duration_modulation (función libre en lib.rs:8257) junto con |
| F2.3b-02 | El VPIN vivo decide el lado del agresor por la regla del tick sobre el mid e ignora is_buyer_maker | alta | confirmado | Línea C (núcleo, StatefulEngine) |
| F2.4-01 | Los fills del mismo milisegundo inflan el ancla de Hawkes a 1 + 0,6·m | alta | confirmado | Qoder/GLM (hawkes_bessel.rs). Línea C sólo lo pide, con el test. |
| F2.6-01 | El voto escalar vivo de NavierStokesManifoldEngine vota largo en cualquier mercado turbulento porque su «velocidad» no tiene signo | alta | confirmado | AGY (consumidor NavierStokesManifoldEngine, #702/#718). El productor sin signo (FMT-001, s |
| F2.1-01 | El veto Lundberg aprieta más cuanto mejor es el edge: compara el riesgo al stop con un margen de caída que escala con la exposición | media | confirmado | C-F4 (Línea C), para el consumidor en risk-engine/src/lib.rs:670-712. Pedido a Codex para  |
| F2.1-02 | R̂ y su LCB suponen cierres iid: con cierres dependientes, R sale inflado por la longitud del bloque y la LCB pierde cobertura | media | confirmado | Línea C (risk-engine/src/cramer_lundberg.rs). El test se escribe en C-F2.1 y el arreglo va |
| F2.1-03 | El curl de Hodge del campo de contagio no mide circulación: cuenta los pares censurados como flujo cero medido y usa z-scores no aditivos | media | confirmado | Línea C, para hodge.rs (grafo observado y nulo). Pedidos a Codex (consumo en correlation_g |
| F2.1-04 | El curl es un escalar global: un ciclo entre monedas ajenas infla la ρ de cualquier grupo | media | confirmado | Codex (lector en correlation_guard.rs) y Qoder/GLM (publicador, como en LC-F1-35). |
| F2.2-02 | El recorte a ±1 R hace que Ville mida la tasa de acierto contra ½ en lugar de la esperanza en R | media | confirmado | Línea C, C-F2.2 |
| F2.2-03 | La cota inferior del profit factor cubre ~90 %, 86 % y 80 %, no el 95 % que declara | media | confirmado | Línea C, C-F2.2 |
| F2.2-04 | Con λ fijo de 0,10, el e-proceso de habilidad no despega con IC < 0,079; con M/α = 8320 los pesos quedan en el piso | media | confirmado | Línea C, C-F2.2 (evalues.rs y skill_motores.rs) |
| F2.2-05 | La cobertura conformal sólo se cuenta en ejecutadas, no se publica y se aparta del nominal en los candidatos | media | confirmado | Línea C, C-F2.2 |
| F2.2-06 | Platt aprende sólo de lo ejecutado y extrapola su tasa a puntuaciones nunca ejecutadas | media | confirmado | Línea C, C-F2.2 |
| F2.2-07 | El ATR vivo, con piso de 10 pb y trinquete, viola el supuesto σ = ATR/√(8/π) de diffusion | media | confirmado | Línea C, C-F4 (ATR, plan §4); C-F2.2 aporta la prueba |
| F2.2-08 | El piso «p80 medido» de P² no es un p80 y no olvida | media | confirmado | Línea C, C-F2.2 |
| F2.2-10 | El DSR del demonio casi no tiene potencia y la pierde a cada ronda | media | confirmado | Línea C, C-F7 (plan §3 fila 34) |
| F2.3-02 | qo_613_rho_tau se publica dos veces por evento, gana la τ cruda y el mapeo τ→escala usa distancia absoluta | media | confirmado | Línea C (C-F3) |
| F2.3-03 | La fusión ignora el signo de la habilidad: las escalas que revierten votan momento | media | confirmado | Línea C (C-F3, fusión) |
| F2.3-04 | Dos vectores de pesos y dos cantidades distintas con el mismo nombre spectral_coherence | media | confirmado | Línea C (C-F3) |
| F2.3-05 | crash_flux es alto en ruido puro y recorta capital y alimenta vetos | media | confirmado | Línea C (C-F2.3, régimen) |
| F2.3-06 | La σ pronosticada es la mediana (sesgo de Jensen) y se usa como media en TP/SL | media | confirmado | Línea C (C-F2.3, fila 3) |
| F2.3-07 | El IC cruzado se atenúa por el desfase de bloques entre monedas | media | confirmado | Línea C (C-F2.3) |
| F2.3b-03 | La entropía viva mide la proporción de eventos sin cambio de mid, no la forma de los retornos | media | confirmado | Línea C (núcleo). FlowImpulseEngine es signal-engine (no verifiqué su zona). |
| F2.3b-04 | El FFT de 64 eventos no localiza un ciclo físico: el bin depende de la densidad de eventos | media | confirmado | Línea C (espectro). El versionado del vector ML, con Línea E (model_manifest). |
| F2.3b-05 | El Reynolds de Navier-Stokes es proporcional a 1/√dt: mide cadencia | media | confirmado | AGY (pedido con test; Línea C no lo arregla) |
| F2.3b-06 | El curl_share de Hodge es una función cerrada de momentos transversales y vale unos 0,93 sin ninguna circulación | media | confirmado | AGY por historia (Ω37). No verifiqué la reserva de zonas; va como pedido con test. |
| F2.3b-07 | La curvatura de Yang-Mills es idénticamente cero salvo por la deriva de β (gauge puro) | media | confirmado | AGY (pedido con test) |
| F2.4-02 | El contagio se mide sobre el reloj de @depth5, no sobre la actividad de mercado | media | confirmado | Qoder/GLM (hawkes_cross.rs, contagion_publisher.rs); Codex para el lector en correlation_g |
| F2.4-03 | La intensidad viva incluye el propio evento (lectura de Palm), contra la fórmula declarada con t_i < t | media | confirmado | Qoder/GLM (hawkes_bessel.rs); el núcleo, para el orden de registro y lectura. |
| F2.4-04 | El exceso λ/μ̂ − 1,6 no se normaliza por su dispersión nula, que depende del ritmo | media | confirmado | Qoder/GLM (hawkes_bessel.rs, flow_excitation_confluence.rs); núcleo (flow_impulse). |
| F2.4-05 | El tope MU_INST_MAX = 200/s y la siembra MU_SEED_MAX = 50 hunden μ̂ en las ráfagas | media | confirmado | Qoder/GLM (hawkes_bessel.rs). |
| F2.4-06 | La teoría nombrada no se estima: α y β son literales y n = 0,6 se publica como si fuera una medida | media | confirmado | Qoder/GLM (hawkes_bessel.rs); Línea C para la clave publicada en lib.rs. |
| F2.4-07 | Lead-lag declara líder en el 82 % de los pares independientes, frente al 5 % nominal | media | confirmado | Línea C (lead_lag.rs no está en zona ajena); el cableado del slot 3 del PPO es del núcleo. |
| F2.4-08 | El modulador de contagio descuenta hasta un 30 % sin dirección y sin evidencia OOS | media | confirmado | Qoder/GLM (cadena de contagio); el núcleo, para el lector en lib.rs:7157-7165. |
| F2.5-01 | La convicción del Ente sólo decide en dos sitios: su veto de cascada y el suelo del override de un veto único | media | reformulado | Línea C (C-F3, consejo). Decidir entre sombra declarada (G-TEO 5) o retirada. |
| F2.5-09 | La EMA del OFI es por evento y cada trade con el libro quieto le mete un 0: la lectura cae con la actividad | media | confirmado | Línea C (alimentación del núcleo y salidas, fuera de las ramas 1-10) |
| F2.1-05 | Documentación falsa en Hodge y Lundberg | baja | confirmado | Línea C |
| F2.1-06 | Marchenko-Pastur sin consumidor; effective_bets mal nombrado; el veredicto del borde no tiene tamaño del 5 % y un test es vacuo | baja | confirmado | Codex (pedido) |
| F2.2-09 | La cota jerárquica no es monótona, con n = 0 devuelve la media del prior y en vivo es inalcanzable | baja | confirmado | Línea C, C-F2.2 |
| F2.2-11 | correlation.rs no tiene consumidor, y su sesgo de R7-R2-E-9 se cancela en el cociente | baja | confirmado | Línea C, C-F2.2 (retirada) |
| F2.3-08 | BOCPD con hazard por evento, datos W1 solapados y predictiva sin varianza posterior | baja | confirmado | Línea C (C-F2.3, fila 10) |
| F2.3-09 | Documentación falsa o desfasada en el dominio | baja | confirmado | Línea C |
| F2.3-10 | Cálculo muerto en el camino caliente: pronóstico de volumen y campos sólo de escritura | baja | confirmado | Línea C |
| F2.3b-08 | La vela interna del Hurst se re-ancla en el evento de cierre y comprime los huecos en un solo retorno | baja | confirmado | Línea C |
| F2.3b-09 | El «Hurst» multifractal está encogido hacia 0,5 y sesgado: con H = 0,7 da unos 0,62-0,64 | baja | confirmado | Línea C |
| F2.3b-10 | cache_generation cambia en cada consulta y la deduplicación de D0 nunca actúa | baja | confirmado | Línea C |
| F2.3b-11 | La cadena Kalman → TensorRing se calcula en cada evento sin lector y con unidades mezcladas | baja | confirmado | Línea C |
| F2.3b-12 | hurst_escala_continua → hurst_scale_matched se escribe y nadie lo lee | baja | confirmado | Línea C |
| F2.4-09 | Un solo empate de timestamps anula toda la matriz de contagio | baja | confirmado | Qoder/GLM. |
| F2.4-10 | HawkesProcessEngine (feature-engine) se escribe en cada evento y nadie lo lee | baja | confirmado | Qoder/GLM. |
| F2.4-11 | cointegration_zscore publica el momentum del líder bajo un nombre de cointegración | baja | confirmado | Línea C (núcleo). |
| F2.4-12 | Cópula t: dominio documentado (0,1) frente a [0,1] aceptado, τ-a con empates y colas simétricas | baja | confirmado | Línea C (copulas.rs); el lector del guard es de Codex. |
| F2.5-02 | El LS ratio entra dos veces en la convicción del Ente, y la severidad de liquidaciones tres veces | baja | confirmado | Línea C (lib.rs:7727-7753) y pedido con test a AGY (Prospect) |
| F2.5-03 | El impulso contrarian de Prospect [1,0; 1,30] queda anulado por el recorte de entity_factor y por el contrato [0,1] de la confianza; el test existente lo enmascara | baja | confirmado | AGY (Prospect, pedido con test) y Línea C (recorte del asiento) |
| F2.5-04 | El poller de LS y taker consulta 30 símbolos arbitrarios de todo el mercado, no el roster | baja | confirmado | Línea C (data-pipeline no figura entre las zonas ajenas) |
| F2.5-05 | El «sentimiento de la multitud minorista» se alimenta con el LS de top traders | baja | confirmado | Línea C |
| F2.5-06 | La presión de Prospect de la masa no tiene punto de referencia: es una transformación de ln LS, en unidades arbitrarias | baja | confirmado | AGY (pedido con test) |
| F2.5-07 | SpoofingDetector: ventana igual a la cadencia de @depth5, no separa cancelación de ejecución ni de traslación, y su doc dice «desconectado» | baja | confirmado | Línea C |
| F2.5-08 | El detector de ballenas (z ≥ 2,5 sobre nocional crudo) no está calibrado y las propias ballenas lo enmascaran | baja | confirmado | Línea C |
| F2.5-10 | Tests que dicen certificar Prospect «a través de GodEngineCore» sin observar el camino vivo | baja | confirmado | AGY (tests de Prospect) |
| F2.6-02 | feynman_coherence cuenta escalas no observadas y no filtra el ruido: en una caminata aleatoria supera el corte 0,20 con frecuencia | baja | confirmado | AGY |
| F2.6-03 | Feynman: la «integral de camino» no tiene caminos ni un ħ efectivo, y la documentación describe constantes y unidades que el código no tiene | baja | confirmado | AGY |
| F2.6-04 | Sombra de NavierStokesManifold: sin lector, ν sin escritor, «flujo de cascada» telescópico y «energía» autonormalizada que no ve la energía inyectada | baja | confirmado | AGY |
| F2.6-05 | QuantumTensorStore no tiene consumidor y su «Lyapunov» no es un exponente de Lyapunov | baja | confirmado | Línea C (feature-engine, fuera de zonas ajenas); retirada |
| F2.6-06 | SimdNeuralNet se construye y reinicia en cada StatefulEngine pero nunca se evalúa ni se entrena | baja | confirmado | Línea C (feature-engine y stateful_engine.rs, que no es la reserva de Línea E en lib.rs);  |

## Severidad alta

### F2.2-01. El vigilante de rollback apuesta a favor del edge y dispara por caída de evidencia: revierte genomas rentables

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/evolution-engine/src/online_daemon.rs:1205-1209, :1241-1290, :1388 y :1948; crates/risk-engine/src/ville_e_process.rs:181-189 y :202-223
- Causa: VilleEProcess usa siempre λ ≥ λ_min = 0,05 > 0, así que su H0 es «sin edge» (E[X] ≤ 0). La desigualdad de Ville sólo acota P(sup M ≥ 1/α) bajo ese nulo. No acota la caída del 50 % desde el pico (is_evidence_decayed(0.50)) ni la condición M < 1 tras un pico ≥ 2 (is_exhausted). λ sube a 0,5 en cuanto la media corrida es alta y la varianza estimada baja, y entonces una sola pérdida de −1 R parte e por la mitad. La rama t ≤ −2 también pierde el nivel por mirarse en cada ciclo: con 0 R dispara en el 17,9 % de los caminos en 300 operaciones, frente al 2,3 % nominal.
- Consecuencia viva: En demo, donde el vigilante está armado por defecto (online_daemon.rs:760-770), toda promoción se revierte en unos 20 cierres sea cual sea su edge: apply_to_arena restaura el padre (:1265) y el ledger registra rollbacks falsos. La evolución en vivo no puede retener ningún genoma nuevo. En prod sólo ocurre con TG_LIVE_GENOME_EVOLUTION_ARMED=1.
- Prueba que falla hoy: cf2_vigilante_no_revierte_un_genoma_rentable_testigo: la secuencia [+1, +1, +1, −1, −1] suma +1 R. En el paso 4, λ = 0,5 y e = 0,579; en el paso 5, drawdown = 0,75 y el predicado vivo da true. cf2_vigilante_nivel_sin_degradacion: con +0,10 R, +0,35 R y 0 R hay rollback en el 100 % de 500 caminos (media de 16 a 23 operaciones), cuando lo esperado es ≤ 5 %.
- Dueño: Línea C, C-F2.2 (online_daemon.rs y ville_e_process.rs)
- Fila de §3 del plan: 36
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La severidad alta sólo se sostiene para demo, donde el vigilante está armado por defecto pero con capital de papel. En prod el demonio retorna en online_daemon.rs:1356 antes de armar la vigilancia, así que no toca dinero real salvo con TG_LIVE_GENOME_EVOLUTION_ARMED=1; por la regla estricta de «dinero vivo» sería media. El arreglo de la fila 36 debe cubrir también la regla t ≤ −2 con miradas repetidas (online_daemon.rs:1215-1221), que está en el mismo OR (:1241).
  - Corrección (completitud): La severidad alta sólo vale en demo, armado por defecto (online_daemon.rs:760-770). En prod está desarmado salvo TG_LIVE_GENOME_EVOLUTION_ARMED=1, y ahí sería media. Además falta la consecuencia de segundo orden: tras un rollback falso el vigilante se rearma sobre el genoma restaurado y oscila entre padre e hijo (faltante FA-2). El nivel del criterio t ≤ −2 (:1220) y del kill por EWMA < −1,5 (:1343) es el mismo problema de la fila 32.

### F2.3-01. La τ operativa unificada cambia de escala según el signo de un IC sin validar

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/temporal_spectrum.rs:1201-1208 (doc :1193-1199); habilidad_en :382-403; habilidad_medida :264-278
- Causa: La doc dice que manda la escala «de habilidad demostrada por Ville (IC > 0 validado)», pero el código sólo comprueba habilidad_en(dominant) > 0, y eso es un IC con n ≥ 30 sin e-proceso. Cuando ninguna escala pasa significativo_familia(5), dominant_tau_ms es el respaldo argmax |w·s| (≈30 s, :617-626). habilidad_en(30 s) lee el IC nulo de la escala 17, cuyo signo es aleatorio (SE ≈ 0,09 con EWMA 1/64). Resultado: la τ operativa salta entre ≈30 s y el centroide (minutos) por ruido.
- Consecuencia viva: arena dominant_tau_ms (lib.rs:2015-2019) → horizon_tau_ms_coin (risk-engine/src/lib.rs:1492-1503) fija el horizonte de dimensionamiento y de TP/SL cuando la intención no trae duración. La rama 15 usa tau_star como expected_duration (lib.rs:6579-6600). Un factor de 4 a 20 en τ cambia σ(τ) en un factor de 2 a 4,5.
- Prueba que falla hoy: cf2_rw_sin_habilidad_validada_la_tau_unificada_es_el_centroide: 20 semillas de caminata aleatoria, 14 400 eventos cada una. Exige ≤ 3 semillas donde tau_operativa_unificada() ≠ continuous_resonant_tau_ms(), porque con α = 0,05 P(X ≥ 4) ≈ 0,016. Hoy se esperan ≈ 10/20 (signo de un IC nulo; P(≤ 3 | p = 0,5) ≈ 0,001).
- Dueño: Línea C (C-F3, τ dominante)
- Fila de §3 del plan: 2
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Severidad media, no alta. Causa: el respaldo es el argmax en banda (τ18 ≈ 68,7 s, escala 18), no 30 s ni la escala 17. Consecuencia: cambia la duración de la rama 15, y tras la media geométrica de lib.rs:7184 el horizonte se mueve en un factor √(τ_nodo/τ_centroide), no de 4 a 20; el camino del arena al risk-engine está dormido para las intenciones del núcleo (LC-F1-14/65).
  - Corrección (completitud): Severidad alta, que mantengo por el veto QO-586. Cita corregida del respaldo: temporal_spectrum.rs:620-625 y :671-680 (argmax en banda, escala 18). Consecuencia viva: lib.rs:6579-6604 → lib.rs:1600-1608 y lib.rs:7170-7190, no risk-engine/src/lib.rs:1492-1503.

### F2.3b-01. El DFA clasifica un paseo aleatorio como persistente en torno a la mitad de las veces y la duración se duplica

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/hurst_dfa.rs:222-301; crates/god-engine-core/src/math_kernels.rs:549-551; crates/god-engine-core/src/lib.rs:8579-8580, :5545-5546, :8257-8270, :6833-6836
- Causa: Con ruido blanco, el DFA1 sobre las escalas {4..256} tiene un sesgo positivo de muestra finita: E[F²(s)] = (s²−4)/(15s) da una pendiente esperada de 0,526, unos 0,521 tras la corrección de Jensen. Su SD por realización es de unos 0,03-0,04 a N = 1024 (a confirmar por Monte Carlo). La banda «persistente» (H > 0,52) cae justo en la media nula. La modulación 2^((H−0,5)/0,02) usa una escala menor que la SD del estimador, de modo que en la mediana nula da 2^1,05, que se recorta a ×2. Ni el estimador ni sus consumidores tienen IC.
- Consecuencia viva: En un mercado sin memoria, la mitad de las evaluaciones con Hurst maduro duplican expected_duration_ms (lib.rs:6833). El efecto pasa a τ (lib.rs:7169-7189, :7512, :8010), al TP/SL y a banda_paga_friccion (lib.rs:1601). Además, las ramas que exigen is_persistent (lib.rs:5834-5883) se habilitan con ruido, y trend_runner suma tanh(0,084) (trend_runner.rs:60). El efecto en PnL no está medido.
- Prueba que falla hoy: - cf2_paseo_aleatorio_no_se_clasifica_persistente: 200 paseos de 1024 retornos; esperado ≤ 10 % persistentes, predigo ≈ 50 %. - cf2_duracion_no_se_modula_bajo_paseo_aleatorio: factor mediano esperado en [0,9; 1,1], predigo ≈ 2. Ambas en crates/god-engine-core/tests/cf2_hurst_bandas_solucion_conocida.rs. - cf2_dfa_paseo_aleatorio_centrado_en_medio (crates/feature-engine/tests/cf2_hurst_solucion_conocida.rs): media ≈ 0,521 frente a 0,5 ± 4·SE, con SE ≈ 0,0025.
- Dueño: Línea C: el estimador y hurst_duration_modulation (función libre en lib.rs:8257) junto con su llamada. El uso de las bandas en lib.rs:5545-5901 va como pedido a Línea E si cae dentro de las ramas 1-10; no verifiqué el límite exacto.
- Fila de §3 del plan: 6
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Severidad alta, mantenida. Hay que corregir el campo «duplica»: amplía R7-R2-E-2 (BARRIDO_EXHAUSTIVO_FASES.md:2375-2384, sin nulo medido entre ±0,02 y ±0,14), que sigue vigente. También la magnitud: el ×2 de expected_duration_ms pasa después por la media geométrica con la τ resonante (lib.rs:7167-7186), así que en ese camino la τ final cambia unas √2 veces, no 2.
  - Corrección (completitud): Hay que añadir el canal directo que falta en la consecuencia viva. Sin pronóstico σ con habilidad (tp_sl.rs:352-355), el H del DFA es el exponente de σ(τ) = atr·(τ/60 s)^H (tp_sl.rs:294-306 y :314-318; risk-engine/src/lib.rs:873-890). - Sólo el sesgo de +0,021 infla el stop y el TP ×1,09 a 1 h y ×1,15 a 12 h. - Con la SD supuesta de 0,035 (sin medir), la banda del 95 % por realización es ×[0,75; 1,32] a 1 h y ×[0,64; 1,57] a 12 h. Faltan dos consumidores: - SeniorTeleonomia: predictability = 2|H−0,5| (consejo_seniors.rs:867; payload en lib.rs:4453 → :7657). - trend_runner: h_weight = tanh((H−0,5)/0,04), que vale unos 0,48 en el centro nulo (signal-engine/src/trend_runner.rs:236-244). Solapes: - Con LC-F3-09 y con la fila de §4 «lib.rs:8169-8174, 8491-8492 (DER, C-F3)». Lo nuevo es el centro nulo en 0,521 y la tasa de falsos persistentes. - En c4e8e285, features[1] (stateful_engine.rs:1079 y :1095) es el mismo RecursiveHurst que coin.hurst_exponent (lib.rs:3149-3154). El «otro Hurst» de LC-F3-09 ya no aplica; sólo cambia el instante de lectura. Se mantiene la severidad alta.

### F2.3b-02. El VPIN vivo decide el lado del agresor por la regla del tick sobre el mid e ignora is_buyer_maker

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:805-829; crates/god-engine-core/src/lib.rs:550-565, :1706, :2729, :3088
- Causa: process_tick no recibe el lado. El lado sale de la dirección del último cambio del mid, incluidos los cambios de eventos de profundidad con volumen 0. Los trades se procesan con el mid del último libro (libro_efectivo), así que casi todos son ticks planos que heredan el lado del último movimiento del libro. El is_buyer_maker real llega a process_event (lib.rs:1706) y se usa en update_trade_flow (lib.rs:2729), pero no llega al VPIN.
- Consecuencia viva: El VPIN es do_calculus_risk (lib.rs:7601-7604, :7664) y llega a: - SeniorCausal, que veta por encima de 0,75 salvo el bypass de breakout alineado < 0,88 (consejo_seniors.rs:652-664); - el Auditor, que veta por encima de 0,92 (:914); - la penalización 1 − 0,30·VPIN (:872); - la salida tóxica (> 0,85 con OFI adverso, lib.rs:3584-3594); - la feature [7] del bosque (stateful_engine.rs:1099-1101).
- Prueba que falla hoy: cf2_vpin_vivo_ignora_el_agresor (crates/god-engine-core/tests/cf2_vpin_entropia_solucion_conocida.rs): un depth al alza seguido de 4000 trades al mismo mid con agresor real equilibrado. Predigo un VPIN vivo de 1,0 exacto; el control con el agresor real da unos 0,04, y la aserción pide < 0,75.
- Dueño: Línea C (núcleo, StatefulEngine)
- Fila de §3 del plan: 22
- Relación con hallazgos anteriores: Ya está nombrado en la fila 22 del plan y en MEMORIA.md (2026-09-25, «Hallazgos NO corregidos», n.º 3), sin id LC-F1/R7. Aporto el test que falla y la cuantificación.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La frecuencia de vetos falsos en vivo no está medida, así que la severidad alta descansa en el mecanismo. Hay que añadir que el comentario D-110 llama «Lee-Ready» a una regla del tick sobre mids (stateful_engine.rs:803), lo cual es falso. La feature [7] no es el VPIN: es el desequilibrio con signo (B−S)/(B+S) (stateful_engine.rs:1099-1101), que entra al bosque por get_universal_features (stateful_engine.rs:1119-1141 → lib.rs:4644 → vin[..34]). El lado está igual de mal ahí.
  - Corrección (completitud): La lista de lectores está equivocada: la clave cvpin sí tiene lectores vivos. - trend_runner.rs:183-189 la lee como clave primaria; vpin y order_flow_vpin son sólo respaldo. - game_theoretic_nash.rs:194-203 la usa como presión adversarial, porque game_theory_adversarial_pressure no tiene escritor (R7-R3-E-3). La salida es tanh(OFI·(1−VPIN)) (:52-82). - Las sombras espectrales de lib.rs:2244-2256 y :2378-2382. Con el VPIN saturado en 1, el voto de Nash cae ×0,1 y la expansión del TP del trend_runner queda en el piso de 0,2 (trend_runner.rs:64). Esto amplía la consecuencia viva. El arreglo del lado tiene que re-derivar en el mismo cambio los umbrales 0,75, 0,85, 0,88 y 0,92. Con el lado real y K = 100-200, el nulo tiene media 0,04-0,056 y SD 0,03-0,043, así que esos umbrales quedan a más de 10 σ y el veto enmudecería. Es un cambio de conducta viva y debe ir primero en sombra.

### F2.4-01. Los fills del mismo milisegundo inflan el ancla de Hawkes a 1 + 0,6·m

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/hawkes_bessel.rs:192 y :207; consumida en crates/god-engine-core/src/lib.rs:2726 → :4484 → :4538
- Causa: record_event empuja SIEMPRE el evento a la cola que suma λ (:207), pero sólo actualiza μ̂ si ts > last_ts (:192). El host toma el campo E de @trade (src/parsers.rs:117-142; event_time en src/bin/god_engine.rs:3131-3133), y un barrido de varios niveles publica varios mensajes con el mismo E. Con m fills por timestamp distinto, E[λ]/μ̂ = 1 + (α/β)·m: vale 2,2 con m = 2 y da una excitación media de 0,35 en un flujo sin excitación alguna.
- Consecuencia viva: La puerta de confluencia (flow_excitation_confluence.rs:224-235, umbral 1,6 + (gen − 0,5)·2, que es 1,7 con el gen por defecto 0,55) queda siempre abierta en los símbolos con lotes, y vuelve la tautología #535. También sesga la rama 12 (flow_impulse, lib.rs:6337-6352), el voto del tensor (hawkes_bessel.rs:371 vía lib.rs:5427-5429) y el slot 2 del PPO (lib.rs:5351-5352, con peso 0,70 en micro_score). La severidad es alta condicionada: el mecanismo es determinista, pero m no se midió porque no hay tapes en el repo. El replay mete un trade por fila sin multiplicidad (booktick_replay.rs:429-450), así que backtest y vivo no miden la misma cantidad (G-TEO §30.1, punto 6).
- Prueba que falla hoy: cf2_hawkes_lotes_mismo_ms_fase_uniforme_ancla_1_6 (crates/signal-engine/tests/cf2_hawkes_solucion_conocida.rs): 5 lotes/s con K = 1 + Poisson(1) fills en el mismo ms; espera 1,6 ± 0,05 y una excitación |e| ≤ 0,03; hoy da ≈ 2,2 y ≈ 0,35.
- Dueño: Qoder/GLM (hawkes_bessel.rs). Línea C sólo lo pide, con el test.
- Fila de §3 del plan: 11 (el plan §3 lo nombra sin medirlo)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Severidad: media confirmada, y alta sólo si se mide m. Con m ≥ 1,17, la media supera ya el umbral 1,7 del gen por defecto. m no está medido porque no hay tapes en el repo. Matiz: los timestamps vivos van en ms, así que dos trades independientes en el mismo ms también producen dt = 0 (efecto pequeño, m ≈ 1 + r·0,0005). Que el replay vaya «sin multiplicidad» no está verificado: booktick_replay pasa t.ts_ms por fila (crates/backtest-engine/src/booktick_replay.rs:436-449), y un tape con varias filas en el mismo ms reproduce el defecto.
  - Corrección (completitud): En la consecuencia viva, describir el lector de confluencia como una rampa (:270-277) y medir el efecto como E[w_hawkes], no como P(ratio > 1,7). El siguiente paso tiene que fijar con literatura cómo se tratan los empates. Lallouache y Challet (2016, Quant. Finance 16) muestran que la resolución del timestamp y el jitter sesgan el n ajustado. Recomiendo colapsar en un evento marcado con su multiplicidad, porque una orden agresora es un solo evento.

### F2.6-01. El voto escalar vivo de NavierStokesManifoldEngine vota largo en cualquier mercado turbulento porque su «velocidad» no tiene signo

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/signal-engine/src/navier_stokes_manifold.rs:202-218 (consumidor); productor en crates/god-engine-core/src/stateful_engine.rs:739 y :788, publicado en crates/god-engine-core/src/lib.rs:4495 y :4506; registrado en el orquestador vivo en lib.rs:1073-1075 y evaluado en crates/signal-engine/src/orchestrator.rs:398-421
- Causa: price_velocity es EWMA(0,15) de |Δmid| en unidades de precio: siempre ≥ 0 (FMT-001). El voto vale laminar·OFI + (1 − laminar)·tanh(100·price_velocity): con laminar → 0 vota +tanh(100·v) tanto si el precio sube como si baja. Además, el 100 literal hace que el término sature en BTC (|Δp| de dólares) y sea ≈ 0 en monedas de precio < 0,01, así que la conducta depende del nivel de precio. Incumple G-TEO §30.1.3 (simetría).
- Consecuencia viva: En la rama escalar del consenso (sin veredicto espectral activo: arranque, espectro plano, τ_dom < 30 s o TTL vencido; orchestrator.rs:384, :477-498), un voto de Navier de +0,9999 con los demás votantes en 0 produce TensorDecision Long con confianza ≈ 0,72. Es la misma aritmética que CL-59 midió para Feynman (0,715). Esa decisión entra en composite_score (lib.rs:5446-5449) y en la rama tensorial (lib.rs:6221-6323). No he medido con qué frecuencia el voto es decisivo; la telemetría qo_624_fraccion_espectral daría la fracción de ticks en rama escalar.
- Prueba que falla hoy: cf2_6_navier_voto_escalar_es_antisimetrico_con_el_mercado_espejado (crates/god-engine-core/tests/cf2_navier_velocidad_espejo.rs): con caminos espejados (±5 céntimos por evento), laminar = 0 y OFI = 0, el voto da ≈ +0,9999 en los dos.
- Dueño: AGY (consumidor NavierStokesManifoldEngine, #702/#718). El productor sin signo (FMT-001, stateful_engine.rs) es del núcleo: se pide como dato al dueño de stateful_engine, no se arregla desde Línea C.
- Fila de §3 del plan: ninguna (F2.6 no tiene fila; la fila 8 es C-F2.3)
- Relación con hallazgos anteriores: FMT-001 (docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_2026-09-24.md:67-69; «persiste» en docs/AUDITORIA_FUNDAMENTOS_CIENTIFICOS_XXIV_2026-09-25.md:194) para el productor. El consumidor vivo es nuevo (#718) y no figura en LC-F1.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Mantengo la severidad alta por mecanismo, con tres precisiones. (1) El 0,72 sale de un escenario construido («los demás en 0»). Navier sola no abre una posición: la rama 11 exige además composite ≥ tensor_tech_thr y effective_obi_long > range_obi (lib.rs:6283-6293). El daño es un sesgo largo sistemático en tensor_boost y composite_score, que también leen las ramas de lib.rs:5975-6209. (2) La magnitud depende del nivel de precio: el término satura en BTC, ETH, SOL o BNB y vale ≈ 0 en monedas con |Δp| ~ 1e-4. La fracción de turbulencia también depende del Re de feature-engine/navier_stokes.rs:196-207 (τ_relax = 0,2 s; un tick por cada 0,2 s da Re ≈ 1, laminar 0,5). Nada de esto está medido. (3) Navier no es el único lector vivo de este price_velocity sin signo: también lo leen soliton_wave.rs:149, supersonic_shockwave.rs:144, coaxial_breakout.rs:214 y trend_runner.rs:214 en el mismo orquestador, y el bosque del daemon en online_daemon.rs:1023 y :1172. No leí cómo usa el signo cada uno.
  - Corrección (completitud): Mantener «alta», pero acotarla: el voto largo sistemático afecta a las monedas en que |Δmid| por evento ≳ 0,01 unidades de precio, y su frecuencia está sin medir hasta hacer la ablación del faltante 6. Remedio: NO cablear tal cual NavierStokesReynoldsEngine::velocity, que es USD/s y depende del nivel de precio (ver faltante 4).

## Severidad media

### F2.1-01. El veto Lundberg aprieta más cuanto mejor es el edge: compara el riesgo al stop con un margen de caída que escala con la exposición

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/risk-engine/src/correlation_guard.rs:546-563 (función pura, zona Codex); crates/risk-engine/src/lib.rs:670-712 (llamador); test que fija la conducta: correlation_guard.rs:1280-1299
- Causa: m = ln(1/ε)/R_capital es la caída acumulada que el paseo alcanza con probabilidad ≤ ε, y decrece al crecer R. El veto exige riesgo_grupo ≤ min(tope_streak, m), así que con más R (menos ruina) el margen es menor. Usada como tope de ruina, la cota de Lundberg limita la exposición: P(caída ≥ D) ≤ e^{−(R_n/f)·D} ≤ ε ⇔ f ≤ R_n·D/ln(1/ε), que crece con R. Con la conversión correcta R_capital = R_n·C/N, riesgo_grupo = f·SL·g y m = f·ln(1/ε)/R_n, así que f se cancela. Con ÷L_max (LC-F4-01) el veto queda relajado L_max/f veces. La propia ficha V-RISK-006 (veto_registry.rs:128) reconoce que el min mezcla horizontes y pide diseño.
- Consecuencia viva: Hoy el veto es casi inerte. Con L_max = 20 sólo dispara si R_lcb > 20·ln 20/riesgo_grupo (> 240 con el grupo en el tope de 0,25). Cuando dispara, veta las monedas con más edge medido. Si C-F4 sólo cambia el factor a C/N, como dice el plan, el veto pasará a disparar cuando R_n·SL·g > ln 20 ≈ 3, sin mirar el tamaño, y seguirá vetando primero los mejores edges.
- Prueba que falla hoy: cf2_lundberg_veto_monotono_en_r (crates/risk-engine/tests/cf2_riesgo_solucion_conocida.rs), con riesgos [0,04; 0,04; 0,04], ρ 0,2, tope 0,10 y ε 0,05. Exige que una R mayor nunca vete lo que una menor admite. Hoy R = 15 admite y R = 40 veta, que es justo lo que fija ola9_veto_por_riesgo_cramer_lundberg_bounds.
- Dueño: C-F4 (Línea C), para el consumidor en risk-engine/src/lib.rs:670-712. Pedido a Codex para la función pura de correlation_guard.rs:546-563 y su test ola9.
- Fila de §3 del plan: 15 (fase dueña C-F4)
- Relación con hallazgos anteriores: Relacionado con LC-F4-01 (factor ÷L_max), pero no lo duplica: LC-F4-01 trata el factor y esto, la forma de la comparación.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): (1) L_max no es siempre 20. El host arranca con get_official_binance_spec (L = 20, god_engine.rs:930-932; symbol_registry.rs:194), pero symbol_manager refresca los specs con dynamic_ranker (src/symbol_manager.rs:28-29, :105 y :212), que fija L = 125/75/50 según el nombre (crates/data-pipeline/src/dynamic_ranker.rs:236-240). Tras el primer refresco, el relajamiento es L/f ≈ 143-357× con f = 0,35, y el disparo con el grupo en 0,25 exige R_lcb > 599-1 498, no > 240. (2) La cifra «f típico 0,35» no tiene cita: es ilustrativa. (3) La prueba de aceptación de la fila 15 del plan (§3, PLAN_LINEA_C:846, «con caída > m el veto dispara antes») está mal planteada mientras la comparación conserve esta monotonía.
  - Corrección (completitud): Añadir el incumplimiento de G-TEO (2) y (5). Al arreglarlo hay que actualizar ola9 y su referencia en veto_registry.rs:134 y :402. Ojo: la prueba de §3, fila 15 («en replay con caída > m, el veto dispara antes») la aprobaría también el veto invertido actual (ver faltantes).

### F2.1-02. R̂ y su LCB suponen cierres iid: con cierres dependientes, R sale inflado por la longitud del bloque y la LCB pierde cobertura

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/risk-engine/src/cramer_lundberg.rs:82-202 (estimador y SE); alimentación en crates/god-engine-core/src/lib.rs:3950-3960
- Causa: g(R) promedia e^{−R·y_i} cierre a cierre, y el SE sándwich (:161-186) supone independencia. Si los cierres llegan en bloques de b iguales, los incrementos del paseo son ±b·a y el R verdadero es R_iid/b. Pasa en vivo porque hay hasta MAX_SPECTRAL_SLOTS = 3 ranuras por moneda (quantum-arena/src/position.rs:731) y cierran en eventos consecutivos tras el mismo movimiento (god-engine-core/src/lib.rs:4419-4421, LC-F3-10). El estimador devuelve ≈ R_iid, con el SE de n cierres en lugar del de n/b bloques. Además, el anillo de 256 cierres (:42) mezcla τ de 30 s a 12 h sin tiempo físico.
- Consecuencia viva: El R_lcb publicado en c{id}:lundberg_r_nocional sale demasiado alto y el margen m demasiado pequeño. Por la inversión de F2.1-01, eso produce más vetos REJ_CORRELATION, no menos. La magnitud en vivo está sin medir (necesita tapes).
- Prueba que falla hoy: cf2_lundberg_cierres_dependientes_inflan_r: b = 3, 85 bloques, p = 0,55, a = 1 %, R verdadero = 6,69, 2 000 réplicas. Exige una cobertura de la LCB ≥ 0,920; por cálculo binomial predigo ≈ 0,65.
- Dueño: Línea C (risk-engine/src/cramer_lundberg.rs). El test se escribe en C-F2.1 y el arreglo va en C-F4, porque C-F2 no cambia conducta viva.
- Fila de §3 del plan: 15
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): b = 3 cierres idénticos es la cota extrema del efecto, no el caso vivo. Las tres ranuras tienen entrada, τ y brackets distintos. El break de lib.rs:4419-4421 sólo demuestra que las ranuras cierran en eventos distintos, no que sus retornos estén correlados. La magnitud sigue sin medir (requiere tapes), y la frase «cierran tras el mismo movimiento» es una hipótesis, no algo verificado.
  - Corrección (completitud): Generalizar el test a dependencia parcial, que es el caso realista. Bloques gaussianos equicorrelacionados (μ, σ, correlación intrabloque ρ_b) dan una solución cerrada: R_verdadero = 2μ/(σ²(1+(b−1)ρ_b)). Con μ = 0,001, σ = 0,005, b = 3 y ρ_b = 0,5, R = 40 frente a R_iid = 80, y predigo una cobertura de la LCB ≈ 0,5.

### F2.1-03. El curl de Hodge del campo de contagio no mide circulación: cuenta los pares censurados como flujo cero medido y usa z-scores no aditivos

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/risk-engine/src/hodge.rs:76-119; crates/god-engine-core/src/contagion_publisher.rs:52-88; consumo en crates/risk-engine/src/correlation_guard.rs:690-701
- Causa: (a) hodge_curl_share proyecta sobre el grafo completo de los nodos con algún flujo (K_r) y trata como flujo cero medido las aristas que hawkes_cross dejó en 0 por z < 3 (hawkes_cross.rs:105, :164-167). El propio productor declara esos ceros «ausencia de afirmación» (:122-123). Una cadena A→B→C con A→C censurada da 2/3, y en el grafo observado, que es un árbol, el curl es 0. (b) El flujo son z-scores, que escalan con √n y la tasa base y no son aditivos a lo largo de un camino. En la cascada perfecta del fixture xlvic (hodge_contagion_contract.rs:19-34), z_L→S1 = z_S1→S2 ≈ 42,3 y z_L→S2 ≈ 24,4, y el curl sale ≈ 0,289 en lugar de 0 (cálculo a mano). (c) Con flujo sin estructura, el valor nulo es 1 − 2/r. Ningún test mide si el curl predice co-movimiento.
- Consecuencia viva: dependency_exposure (llamado en risk-engine/src/lib.rs:612-619) infla la ρ del grupo con ρ + (1−ρ)·curl² (correlation_guard.rs:701). Con curl 0,289, ρ sube 0,084·(1−ρ); con curl 0,75, 0,56·(1−ρ). Eso da más vetos REJ_CORRELATION sin evidencia. La magnitud en vivo está sin medir.
- Prueba que falla hoy: cf2_hodge_cadena_censurada_no_es_ciclo (espera 0 y hoy da 2/3) y cf2_hodge_cascada_jerarquica_del_kernel_vivo (espera < 0,05 y hoy da ≈ 0,289). Las dos están en la lista de pruebas.
- Dueño: Línea C, para hodge.rs (grafo observado y nulo). Pedidos a Codex (consumo en correlation_guard.rs:690-701) y a Qoder/GLM (los z como flujo en hawkes_cross.rs).
- Fila de §3 del plan: 14 (fase dueña C-F4)
- Relación con hallazgos anteriores: Amplía LC-F4-18 (infla ρ aunque el publicador dice «no veta»). La caducidad del valor es LC-F1-35.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): (c) «Con flujo sin estructura el valor nulo es 1 − 2/r» sólo vale para un flujo denso e isótropo sobre K_r. El campo vivo son z ≥ 3 o 0, antisimetrizados, con muchas aristas censuradas, y su nulo no es Beta(7/2, 21/2): hay que simularlo con surrogados. Un posible neutralizador aguas arriba, sin medir: contagion_matrix devuelve None si alguna serie no es estrictamente creciente (hawkes_cross.rs:151-156), y los ticks entran con event_time_ms sin deduplicar (god-engine-core/src/lib.rs:2594-2601 → quantum-arena/src/state.rs:668-688 y :341-345). Si una moneda activa tiene dos eventos en el mismo ms dentro de sus últimos 256 ticks, el publicador no escribe y el curl publicado antes sigue vigente (LC-F1-35). La frecuencia viva del inflado puede ser baja o puede ser la de un valor viejo. Por último, la afirmación «ninguna ficha R7 trata la censura» es demasiado fuerte: R7-R3-A-1 (BARRIDO:2824-2840) es el mismo fenómeno a nivel de nodo (la ausencia fabrica vorticidad) en hodge_flow.rs, y esto es su análogo a nivel de arista. Hay que citarla como familia.
  - Corrección (completitud): Añadir la ceguera ante 2-ciclos y la reutilización de F2-C5. El siguiente paso propuesto (HodgeRank ponderado sobre n_ij − n_ji) hereda el defecto; ver la ficha de hodge.rs.

### F2.1-04. El curl es un escalar global: un ciclo entre monedas ajenas infla la ρ de cualquier grupo

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/god-engine-core/src/contagion_publisher.rs:84-88 (registry.set sin moneda); crates/risk-engine/src/correlation_guard.rs:694-701 (get_value_fast global)
- Causa: hodge_curl_share se calcula sobre todas las monedas activas y se publica sin moneda. dependency_exposure lo aplica a la ρ del grupo misma-apuesta de la candidata aunque ninguno de sus miembros esté en el ciclo. La firma del registro no permite saber quién participa.
- Consecuencia viva: La ρ y el riesgo de grupo salen inflados en grupos ajenos al ciclo, y eso puede vetar órdenes de monedas que no forman parte de ninguna cámara de eco.
- Prueba que falla hoy: Test de integración en god-engine-core/tests, con el fixture y los helpers de risk-engine/tests/correlation_admission_contract.rs:89-178 (fixture, open, pair_ticks): monedas 0 y 1 con ticks correlacionados y open(1, 0, true). Primero se mide same_bet_rho_efectivo de la candidata 0. Después se publica el curl de un 3-ciclo plantado sólo entre las monedas 2, 3 y 4, con las series de xlvic_ciclo_hawkes_real_da_curl_alto (god-engine-core/tests/hodge_contagion_contract.rs:44-65), vía publish_contagion_roles o, en su defecto, poniendo la clave global como lo haría el publicador. Esperado: la ρ del grupo {0, 1} no cambia. Hoy: base + (1−base)·curl², con curl ≈ 1. Es una propuesta, sin ejecutar.
- Dueño: Codex (lector en correlation_guard.rs) y Qoder/GLM (publicador, como en LC-F1-35).
- Fila de §3 del plan: 14
- Relación con hallazgos anteriores: No duplica: LC-F1-35 es la caducidad, y R7-R3-D-1 es el curl por moneda de hodge_flow en el host.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La prueba propuesta está en prosa, no en código. Los helpers fixture, open y pair_ticks son funciones privadas de risk-engine/tests/correlation_admission_contract.rs:89-178 y no se pueden importar desde god-engine-core/tests: hay que copiarlos. Para publicar con publish_contagion_roles (pub fn, contagion_publisher.rs:21), las monedas 2-4 necesitan spectral_fisher ≥ 0 (:29-31), al menos 20 ticks (:33-36) y timestamps estrictamente crecientes (hawkes_cross.rs:151-156). Si no, la matriz da None y la prueba pasaría por vacuidad.
  - Corrección (completitud): Bastaría con añadir a agy_aud_p06 el oráculo de pertenencia (el ciclo está fuera del grupo ⇒ ρ sin cambio). La versión con publicador exige registrar los símbolos 2-4.

### F2.2-02. El recorte a ±1 R hace que Ville mida la tasa de acierto contra ½ en lugar de la esperanza en R

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/risk-engine/src/ville_e_process.rs:106; crates/evolution-engine/src/online_daemon.rs:1143
- Causa: Con TP a k·R (k > 1) y SL a −1 R, el recorte convierte cada ganancia en +1, de modo que la media observada es 2p − 1 y no p·k − (1−p). Para p ∈ (1/(1+k), ½) el signo se invierte. La positividad de 1 + λx sólo exige x > −1/λ_max = −2, así que el recorte superior sobra.
- Consecuencia viva: Los genomas con RR > 1 y acierto < 50 %, que son los que favorece la meta de crecimiento (geometría típica TP 1,125 % y SL 0,5 %), se ven como degradados por la evidencia de Ville. Agrava F2.2-01. La causa es distinta de la unidad de R de LC-F1-41.
- Prueba que falla hoy: cf2_ville_certifica_edge_con_rr_2: +2 R con p = 0,45 y −1 R da E[X] = +0,35 R. Lo esperado es certificar (pico ≥ 20) en al menos el 80 % de 200 caminos de 2 000 operaciones. Con el recorte vivo certifica el 1 %; recortando sólo por abajo, el 100 %.
- Dueño: Línea C, C-F2.2
- Fila de §3 del plan: 36
- Relación con hallazgos anteriores: R7-R2-A-4 (con la dirección corregida: con RR > 1 el recorte impide certificar, no lo facilita)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La dirección de R7-R2-A-4 depende de la geometría. Las pérdidas más allá de −1 R (deslizamiento o huecos) recortadas a −1 facilitan la certificación; las ganancias más allá de +1 R la dificultan. Con RR > 1 domina lo segundo, como dice el auditor. R7-R2-A-4 cita además la normalización vieja `/0,02`; hoy la unidad es riesgo_por_operacion (online_daemon.rs:1130-1141, R7-R6-IG-1). En vivo el efecto llega por los predicados de degradación (la media negativa activa la condición running_mean ≤ 0 de is_exhausted, :222), porque la certificación no tiene consumidor.
  - Corrección (completitud): El recorte se aplica dos veces: primero en online_daemon.rs:1143, (ret/unit_risk).clamp(−1,1), y luego en ville_e_process.rs:106. Quitar sólo el de :106 no cambia el camino vivo, así que la prueba cf2_ville_certifica_edge_con_rr_2 debe pasar también por la normalización del demonio, extraída a una función. Para la positividad basta x > −1/λ_max = −2, y :116 ya pone un suelo de 1e−12. Exigir x ≥ −1 es más estricto de lo necesario y borra las pérdidas por gap más allá del stop (R6-B10, BARRIDO:1517-1521). Por último, la fuente de la unidad R es LC-F1-41.

### F2.2-03. La cota inferior del profit factor cubre ~90 %, 86 % y 80 %, no el 95 % que declara

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/risk-engine/src/evidence.rs:88 (doc en :34-49)
- Causa: Usa SE[ln PF] = √(2/n). Por el método delta, Var(ln PF) = [(cv_w²+1−p)/p + (cv_l²+p)/(1−p) + 2]/n. Falta el +2 de la covarianza negativa entre ln W y ln L (cada ganada es una perdida menos) y faltan los términos de p. Eso da 4/n con ±1 R y p = ½, 4,5/n con 2:1 y p = ⅓, y 8/n con magnitudes exponenciales.
- Consecuencia viva: Con PF real 1, se declara edge (LCB > 1) en el 10-20 % de los casos en lugar del 5 %. Eso saca a la moneda de la exploración en kelly_cold (risk lib.rs:401-406) y alimenta un kelly_fraction optimista (core lib.rs:4203-4221).
- Prueba que falla hoy: cf2_pf_lcb_cubre_al_95_con_pf_real_1, por enumeración exacta con n = 100: cobertura 0,9033 (1:1) y 0,8630 (2:1), cuando se espera ≥ 0,94. cf2_pf_lcb_cubre_con_magnitudes_exponenciales: ≈ 0,80, cuando se espera ≥ 0,93.
  - Actualización (revisión cruzada, CL-65): se registra como F2.2-03b la sub-causa FA-1 del escéptico de completitud: la cota mirada en cada cierre no es anytime-valid (cobertura secuencial ≈ 0,54 y ≈ 0,57 por programación dinámica exacta). La prueba es cf2_pf_lcb_cobertura_secuencial_en_cada_cierre y no se cierra con el arreglo del SE, sólo con una cota válida bajo parada opcional.
- Dueño: Línea C, C-F2.2
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La cadena de consumidor está mal citada. kelly_cold (risk lib.rs:401-406) sólo se alcanza con trades_n == 0, y entonces el PF ya es PF_DESCONOCIDO por definición (evidence.rs:78-80), o con raw_kelly ≤ 0 (risk lib.rs:411-415). El camino efectivo es core lib.rs:4203-4221 → kelly.rs:60-72: con PF ≤ 1 se aplica la rampa de exploración y con PF > 1 se aplica W(1−1/PF). De ahí sale metrics.kelly_fraction → raw_kelly (risk lib.rs:389-415).
  - Corrección (completitud): Hay dos agravantes que el auditor no recoge. (1) La cota se consulta tras cada cierre (risk lib.rs:363-376 y :401-406; core lib.rs:4203). Aunque la varianza fuera correcta, una z fija no da cobertura uniforme en el tiempo: por la ley del logaritmo iterado, con PF = 1 el cruce llega a ocurrir con probabilidad 1 (faltante FA-1). (2) W y L son sumas en USD. Si el capital crece, como persigue la meta, las operaciones recientes pesan más, el n efectivo baja y el error típico real crece aún más.

### F2.2-04. Con λ fijo de 0,10, el e-proceso de habilidad no despega con IC < 0,079; con M/α = 8320 los pesos quedan en el piso

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/quantum-arena/src/evalues.rs:43, :90-100 y :122-125; crates/signal-engine/src/skill_motores.rs:37-39, :94-103 y :223-232; crates/god-engine-core/src/lib.rs:2522-2525; crates/quantum-arena/src/temporal_spectrum.rs:663
- Causa: Con λ fijo, E[ln factor] < 0 para P(acuerdo) < 0,5251: el capital tiende a 0 con cualquier umbral. Por encima de ese valor, el umbral de familia exige 9,03 nats. Un e-proceso mixto o con λ predecible tendría deriva positiva para todo p > ½.
- Consecuencia viva: Con IC realistas, los pesos de la composición del consenso son el piso uniforme de 0,15, así que la ponderación por habilidad está inerte. El τ* por habilidad cae al argmax de energía, que es el dominant_tau_ms anclado en 30 s que MEMORIA deja abierto. El veto de grupo multiactivo, con umbral 43 500, no se activa nunca.
- Prueba que falla hoy: cf2_eproceso_detecta_ic_005_en_la_familia_de_416: señal y retorno gaussianos con ρ = 0,05. Lo esperado es una potencia ≥ 0,5 en 20 000 bloques; hoy es ≈ 0,025.
- Dueño: Línea C, C-F2.2 (evalues.rs y skill_motores.rs)
- Relación con hallazgos anteriores: R7-R2-C-1 (parcial: M = 5 ya corregido, la causa λ sigue); R7-R2-C-5 (M = 2 175)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): (1) «El veto de grupo no se activa nunca» es falso para pares muy correlacionados. espectral_multiactivo.rs:95-98 apuesta sobre sign(ra·rb) con la señal del IC vigente. Con ρ = 0,6, p = ½ + asin(0,6)/π = 0,705 y la deriva es +0,036 nats por bloque, así que ln 43 500 = 10,7 nats se alcanza en unos 300 bloques. Sólo los pares con |ρ| < 0,079 no lo activan nunca. (2) El λ fijo sólo es la causa única con IC < 0,079. Con λ adaptativo ideal e IC 0,05, la tasa es T·(2p−1)²/2 ≈ 5,1·10⁻⁴ nats por bloque: 9,03 nats exigen ≈ 1,8·10⁴ bloques con λ oráculo y ≈ 2,7·10⁴ con mezcla (regret ≈ ½ ln T). La M = 416 de Bonferroni es causa conjunta; el siguiente paso necesita además e-BH o reducir M (plan, fila 17).
  - Corrección (completitud): El siguiente paso es insuficiente. Un e-proceso mixto que apueste sólo al signo tiene como crecimiento óptimo KL(0,5159‖½) = 5,07·10⁻⁴ nats por bloque. Necesita unos 17 800 bloques para ln 8320 con λ oráculo, y unos 25 000-30 000 con mezcla (regret ≈ ½ ln n). Apostando sobre el producto v·r estandarizado y acotado, el crecimiento es ≈ ρ²/2 = 1,25·10⁻³ y bastan unos 7 000-11 000 bloques. Antes de dar más potencia hay que corregir el nulo, porque hoy la deriva cuenta como habilidad (FA-4), y el sesgo del reloj físico en τ* (FA-5). Faltan las filas §3 2 (C-F2.3) y 17 (C-F3), que la ficha deja vacías. Dueño: espectro y consenso son de Qoder (PLAN_LINEA_C:61), así que es un pedido con test, no un arreglo de Línea C.

### F2.2-05. La cobertura conformal sólo se cuenta en ejecutadas, no se publica y se aparta del nominal en los candidatos

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/god-engine-core/src/conformal.rs:136-142 y :182-203; crates/god-engine-core/src/lib.rs:4055-4060 y :4910-4919
- Causa: update sólo recibe cierres ejecutados con ml_at_entry > 0, mientras que accepts() se evalúa en todos los candidatos. Así se rompe la intercambiabilidad entre la población de calibración y la de decisión. Además adaptive_error_rate() no tiene lector.
- Consecuencia viva: conformal_accept_long/short vetan votos del ConformalReversionFilterEngine (conformal_reversion_filter.rs:148-151) a un nivel real distinto del gen conformal_alpha. La evolución ajusta un gen cuyo efecto no es el que dice su nombre.
- Prueba que falla hoy: cf2_conformal_cobertura_en_candidatos_con_seleccion: si sólo se ejecuta p < 0,6, el error en candidatos es 0,165 cuando se espera 0,10 ± 0,02, mientras el error en ejecutadas es 0,100. cf2_conformal_riesgo_entre_aceptadas: la pérdida entre aceptadas es 0,216 frente a α = 0,10.
- Dueño: Línea C, C-F2.2
- Fila de §3 del plan: 26
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La severidad debería bajar a baja. (a) El propio doc ya declara que no certifica cobertura bajo selección (conformal.rs:63-69 y :164-166). (b) El consumidor es el voto de un solo motor (conformal_reversion_filter.rs:148-151, registrado en lib.rs:1048 y :2332); es la misma razón por la que LC-F1-43 quedó en baja. (c) Los calibradores por moneda viven en memoria (conformal.rs no deriva Serialize) y aceptan todo hasta 30 cierres ejecutados por moneda (conformal.rs:91 y :170-172). Con 13 USD el veto pasa casi siempre en calentamiento. Además, adaptive_error_rate sí aparece fuera de conformal.rs, en tests (conformal_wiring_contract.rs:145-240 y conformal.rs:314); lo correcto es «sin lector vivo».
  - Corrección (completitud): La población de decisión viva no son las intenciones. Son todos los eventos por moneda, evaluados para los dos lados antes del interlock (lib.rs:4894-4919, comentario CF-01), así que está todavía más lejos de la de calibración. El siguiente paso propuesto (Mondrian y medir la cobertura) no controla el riesgo entre aceptadas, que es el uso real de accepts() como veto. Los sucesores con respaldo son tres: selección conformal con p-valores conformales y BH (Jin y Candès 2023, JMLR 24), que controla el FDR entre seleccionadas; control de riesgo conformal (Angelopoulos et al., ICLR 2024); y conformal ponderado por la propensión de ejecución (Tibshirani, Barber, Candès y Ramdas 2019). Dueño: conformal es zona de Codex (PLAN_LINEA_C:59 y :316). El auditor la asigna a Línea C.

### F2.2-06. Platt aprende sólo de lo ejecutado y extrapola su tasa a puntuaciones nunca ejecutadas

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/god-engine-core/src/calibration.rs:130-165; crates/god-engine-core/src/lib.rs:4067-4077 y :7576-7584; crates/risk-engine/src/lib.rs:1137-1191
- Causa: Las puertas seleccionan qué se ejecuta. Un mapa logístico global con pendiente ≥ 0 ajustado sobre datos planos lleva a a 0 y produce una p constante. El olvido de 12 h sólo devuelve la p a la puntuación cruda; no hay contrafactuales.
- Consecuencia viva: La puerta EV (REJ_EV) y el Kelly leen una p que no depende de la puntuación en las zonas sin datos. Una intención de puntuación baja hereda la tasa de las altas y puede pasar con EV real negativo. Está relacionado con LC-F1-37 sin duplicarlo.
- Prueba que falla hoy: cf2_platt_no_extrapola_fuera_de_lo_ejecutado: se ejecuta s ∈ [0,7; 0,95] con tasa 0,60 y la verdad es p(0,5) = 0,30. calibrate_at(0,5) devuelve ≈ 0,593.
- Dueño: Línea C, C-F2.2
- Fila de §3 del plan: 26
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La consecuencia está sobredimensionada. La puerta REJ_CONFIDENCE (risk lib.rs:1095-1103) veta antes de la puerta EV toda puntuación inferior a min_required_confidence. Así, las puntuaciones que llegan al EV y las ejecutadas comparten cota inferior, y la extrapolación sólo ocurre en la franja entre ese umbral (que se mueve con la epigenética y la escasez, :1088-1100) y la menor puntuación ejecutada. Su magnitud está sin medir. Los calibradores, además, no se persisten: tras cada reinicio vuelven a la identidad (p = s), lo que ya cubre LC-F1-37. La verdad p(0,5) = 0,30 de la prueba no se puede identificar con datos ejecutados en [0,7; 0,95].
  - Corrección (completitud): Hay tres huecos. (1) Falta el espejo largo/corto: hay calibradores por moneda y uno global, ninguno por lado ni por rama de origen de la puntuación (lib.rs:7576-7584), lo que contradice G-TEO §30.1(3) (FA-7). (2) Los contrafactuales del siguiente paso dependen de cablear shadow_ledger (fila 38, C-F4, hoy sin consumidor), una dependencia que el auditor no declara. (3) Ningún calibrador puede pasar cf2_platt_no_extrapola sin datos contrafactuales: la prueba es la puerta del sucesor, no un fallo de Platt. Faltan Venn-Abers (Vovk y Petej 2014) y Beta (Kull et al. 2017) como sucesores con garantía. Dueño: calibración es zona de Codex (PLAN_LINEA_C:59).

### F2.2-07. El ATR vivo, con piso de 10 pb y trinquete, viola el supuesto σ = ATR/√(8/π) de diffusion

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:849-853 y :876-881; crates/god-engine-core/src/diffusion.rs:57-63; consumidores en crates/god-engine-core/src/lib.rs:5030-5149
- Causa: El ATR tiene un piso de 0,0010·precio y un trinquete al alza dentro de la vela. Con σ de 1 minuto < 6,27 pb manda el piso. Además, el rango de K ticks es discreto (sesgo 2·0,5826/√K).
- Consecuencia viva: En mercados tranquilos los z de tendencia y piso_tendencia (core lib.rs:5030-5036) se calculan con una σ inflada, así que los z quedan subestimados y el escudo macro y los vetos de tendencia responden al piso y no al mercado.
- Prueba que falla hoy: cf2_atr_vivo_da_la_sigma_de_un_minuto: σ = 4 pb por minuto, 60 ticks por minuto y 600 minutos. Lo esperado es σ_ATR/σ ≈ 0,906 (entre 0,80 y 1,10); hoy da ≥ 1,566.
  - Cerrado por CL-71 (C-F4, 2026-10-11): el ATR de las decisiones es el rango equivalente de la σ de 60 s (sigma_minuto, varianza realizada de r60 sobre una rejilla de 1 s), sin trinquete ni suelo de 10 pb; el piso es el tick (CL-69). Las dos pruebas se activaron: σ_ATR/σ = 1,039 (4 pb/min) y 0,968 (30 pb/min). El nulo FA-9 está en cf4_atr_sigma_minuto: |z| > 1,96 en el 4,4 % de los minutos de un paseo aleatorio (antes ≈ 0,2 %). El rasgo del modelo conserva el algoritmo viejo hasta reentrenar (CL-68).
- Dueño: Línea C, C-F4 (ATR, plan §4); C-F2.2 aporta la prueba
- Relación con hallazgos anteriores: plan §4 (stateful_engine.rs:850-853 y 876-881, C-F4); relacionado con LC-F1-55
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Quitar sólo el piso no bastaría para la banda [0,80; 1,10]. El trinquete intra-vela con la EMA 0,85/0,15 lleva v al equilibrio v ≈ μ + 5,67·E[(TR − v)⁺]. Con CV(TR) ≈ 0,30 sale v ≈ 1,2·E[TR] y una razón ≈ 1,1; lo estimé a mano, sin simular. Hacen falta los dos arreglos, y el plan §4 los lista juntos.
  - Corrección (completitud): La prueba con σ = 4 pb sólo ejercita el piso. El trinquete intra-vela (stateful_engine.rs:849-853: v_t = tr cuando tr > v_t) sesga v_t hacia arriba también sin piso: es una EMA que sube entera y baja al 15 %. Falta una prueba complementaria con σ = 30 pb/min, sin piso, para aislarlo. Mi predicción es que falla con una razón ≈ 1,2-1,3, sin medir. También falta el nulo extremo a extremo de los z de tendencia (FA-9).

### F2.2-08. El piso «p80 medido» de P² no es un p80 y no olvida

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/quantum-arena/src/adaptive_quantiles.rs:191-241; crates/god-engine-core/src/lib.rs:2866-2892, :4985-5003 y :5472
- Causa: El umbral multiplica el cuantil por (1 + 0,5χ), y el P² es acumulativo de por vida.
- Consecuencia viva: Los umbrales de OBI/OFI y el piso del score analítico disparan a tasas que nadie fija: 2,5 veces más «eventos» en un régimen volátil nuevo y un 37 % menos con χ > 0.
- Prueba que falla hoy: cf2_umbral_obi_con_intermitencia_conserva_el_p80: con χ = 0,4 la tasa es 0,125, cuando se espera 0,20 ± 0,02. cf2_p2_tras_cambio_de_regimen_vuelve_al_nominal: con σ ×2 la tasa es ≈ 0,49. Con datos iid y sin multiplicador pasa (0,2007).
  - Actualización (revisión cruzada, CL-65): la prueba se llama ahora cf2_umbral_obi_con_intermitencia_sigue_la_cola_analitica y es ACTIVA: fija la cola analítica del diseño vivo, 2(1 − Φ(1,2·1,2816)) = 0,1241 (no un p80), y es una regresión documental (replica el multiplicador; el piso vivo es privado). La OPEN de este hallazgo es cf2_p2_tras_cambio_de_regimen_vuelve_al_nominal; su ventana [10 000, 20 000] se recalcula con la memoria que fije el arreglo.
- Dueño: Línea C, C-F2.2
- Fila de §3 del plan: 42
- Relación con hallazgos anteriores: R7-R2-C-9
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): No es nuevo: es R7-R2-C-9, re-anclada. El multiplicador es diseño explícito (lib.rs:4986-4990, Ola XLI·C2: «un evento del percentil 80/85 exige más cuando el régimen es intermitente»). El defecto es la etiqueta «p80 medido» y que la tasa no está declarada. Lo sustantivo es el estimador acumulativo, que su doc ya admite (adaptive_quantiles.rs:186-188). El umbral vivo es además max(curva del gen, piso) (lib.rs:2873-2892), así que la tasa del piso sólo gobierna cuando supera al gen.
  - Corrección (completitud): La parte de χ es diseño documentado («endurece», lib.rs:4986-4989), y su defecto es la etiqueta «p80 medido», ya recogido en R7-R2-C-9; no es una tasa errónea. Además, la tasa viva es la de max(curva del gen, piso) (lib.rs:2873-2892), así que medir sólo el piso sobreestima el impacto. Hay también un sesgo de muestra que el auditor no ve: el P² sólo se actualiza en eventos con entradas no bloqueadas, porque el return de lib.rs:4953-4955 precede a :5472. Por tanto excluye las tormentas de latencia. Con todo, la severidad media sólo se sostiene para el cambio de régimen.

### F2.2-10. El DSR del demonio casi no tiene potencia y la pierde a cada ronda

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/risk-engine/src/selection_stats.rs:139-166 y :248-270; crates/evolution-engine/src/online_daemon.rs:1888-1903; crates/god-engine-core/src/darwin.rs:650-657
- Causa: El listón E[max SR] crece como √(2 ln N), y N acumula sin límite las pruebas de todas las rondas (≈ 2 001 por ronda), mientras n, el número de barras del examen, no crece.
- Consecuencia viva: Salvo edges enormes, ninguna promoción del demonio pasa el DSR, así que la evolución en vivo también queda cerrada por estadística, además de por F2.2-01.
- Prueba que falla hoy: cf2_dsr_nivel_y_potencia_con_100_pruebas: con SR 0,2 por observación, n = 250 y N = 100, la potencia es ≈ 0,137, cuando se pide > 0,5. El nivel es 0 y pasa. Por cálculo, con N = 20 010 la potencia cae a ≈ 1 %.
  - Actualización (revisión cruzada, CL-65): la prueba ya no usa SR 0,2 ni pide potencia > 0,5. Planta Sharpe 0,25 con n = 250 y compara con el Šidák exacto ((1−α)^{1/N}), con N = 100 y N = 20 010. Mide una SEGUNDA causa, que se añade a este hallazgo: el umbral del DSR, E[max] + 1,645·σ_SR, deflacta dos veces frente al cuantil (1−α)^{1/N} del máximo (nivel ≈ 0 y potencia ≈ 0,40 frente a ≈ 0,74 del Šidák con N = 100). La acumulación de N entre rondas (causa 1) es del cableado de C-F7 y no se prueba aquí.
- Dueño: Línea C, C-F7 (plan §3 fila 34)
- Fila de §3 del plan: 34
- Relación con hallazgos anteriores: R7-R2-A-9; plan §3 fila 34
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): «Duplica R7-R2-A-9» está mal: ese testigo va en la dirección contraria (una duplicación 10× pasa el DSR, que es laxitud ante la dependencia). El antecedente correcto es el plan §4 («Por exceso de corrección: el DSR acumula pruebas sin límite») y la fila 34. El criterio «potencia > 0,5» con SR 0,2, n = 250 y N = 100 es inalcanzable para cualquier procedimiento con FWER del 5 %: un Šidák (z ≈ 3,28) tiene potencia ≈ Φ(3,16 − 3,28) ≈ 0,45. Ver la corrección de la prueba.
  - Corrección (completitud): La prueba con N = 100 mide la potencia intrínseca del DSR, no el defecto. Con SR 0,2, n = 250 y N = 100, un DSR correcto también da ≈ 0,16 analítico: SR* = 2,53·0,0634 = 0,160 y hace falta SR ≥ 0,264. El defecto real es triple: la acumulación de N entre rondas que reutilizan los mismos datos de examen, la falta de N efectivo (dependencia entre candidatos casi idénticos) y la selección por aptitud y no por Sharpe (online_daemon.rs:1840-1848), que no es el supuesto de Gumbel. Dueño: el demonio es tubería de promoción de GLM (PLAN_LINEA_C:63); darwin.rs sí es de Línea C.

### F2.3-02. qo_613_rho_tau se publica dos veces por evento, gana la τ cruda y el mapeo τ→escala usa distancia absoluta

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/god-engine-core/src/lib.rs:1777-1782 frente a lib.rs:2919-2932 (process_event → process_tick_dual en :2806); crates/god-engine-core/src/evidence_publication.rs:65-71
- Causa: El arreglo C-2 publica el IC cruzado en tau_operativa_unificada, pero en el mismo evento process_tick_dual vuelve a publicar con spec.dominant_tau_ms y lo sobrescribe, así que el lector siempre ve la τ cruda. Además publicar_coherencia elige la escala por |τ_k − τ| en lugar de |ln τ_k − ln τ|. Para τ entre 2·4^k y 2,5·4^k (×10⁻⁶ ms) elige la escala inferior. F2-B2 sólo corrigió esto en habilidad_en.
- Consecuencia viva: risk-engine/src/lib.rs:676-704 (lectura en :686) endurece el veto de grupo con el IC de otra escala distinta de la del horizonte operativo.
- Prueba que falla hoy: Hay dos pruebas propuestas. Unitaria: publicar_coherencia con τ = 2,2·4^18·10⁻⁶ ms debe consultar la escala 19, la más cercana en ln (ln 2,2 = 0,79 > ln(4/2,2) = 0,60); hoy consulta la 18. Integración: con GodEngineCore y un espectro en respaldo (caminata aleatoria), tras process_event el valor de qo_613_rho_tau debe ser coherencia_media_con_todas(coin, escala(tau_operativa_unificada)); hoy es el de escala(dominant_tau_ms).
- Dueño: Línea C (C-F3)
- Fila de §3 del plan: 2
- Relación con hallazgos anteriores: Relacionado con LC-F1-14 y R7-R2-C-2 (dos τ dominantes); la sobrescritura por el dual es nueva en c4e8e285
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): No es nueva en c4e8e285: en aa716edf ya estaban las dos publicaciones (aa716edf lib.rs:1753 con la unificada y :2844 con dominant_tau_ms). Es el residuo de R7-R2-C-2 que dejó el arreglo C-2. Falta un tercer sitio que publica con dominant_tau_ms: el feedback de cierre (lib.rs:3970-3975). La parte de distancia absoluta hoy está latente: el valor que sobrevive es siempre un nodo (τ18…τ22, o los recortes de 30 s y 12 h), y con un nodo los dos métricos coinciden salvo en el recorte de 12 h (absoluta → 22, ln → 23). Pasará a ser viva cuando se arregle la sobrescritura y gane la τ continua. Severidad media se mantiene.
  - Corrección (completitud): Añadir lib.rs:3967-3974 (tercera publicación). Severidad media.

### F2.3-03. La fusión ignora el signo de la habilidad: las escalas que revierten votan momento

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/temporal_spectrum.rs:247-257 (|p|), :597-616 (fusión con pesos_observables sin p ni IC), :649-670 (selección sólo con IC > 0); crates/quantum-arena/src/evalues.rs (apuesta sólo a s·r > 0)
- Causa: La señal de cada escala es momento. En un proceso con reversión su IC es negativo y conocido (OU con vida media de 30 s: ≈ −0,47 a 69 s, ≈ −0,62 a 275 s). La fusión la suma con signo positivo, |p| aumenta el peso de esas escalas en la masa, y ni la selección de τ ni el e-proceso consideran IC < 0.
- Consecuencia viva: fused_score llega a la rama 15 (lib.rs:6576-6600), al consejo (lib.rs:7634) y a fused_aligned (lib.rs:3611). spectral_coherence(is_long) llega a X-016 (lib.rs:7006). En un régimen de reversión el sistema apuesta contra su propia evidencia.
- Prueba que falla hoy: cf2_ou_la_fusion_firmada_predice_el_bloque_siguiente: IC de fused_score contra el retorno de 69 s > 3/√m. Hoy se espera negativo. También cf2_ou_tau_dominante_sigue_la_habilidad_absoluta (hoy 0 aciertos). cf2_ou_ic_de_la_senal_de_momento_es_negativo documenta el valor conocido (pasa).
- Dueño: Línea C (C-F3, fusión)
- Fila de §3 del plan: 1
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Matices. La limitación está documentada en el propio código (temporal_spectrum.rs:594-596: «su IC es positivo en BTC y negativo en SOL; eso lo decide quien lo consuma»). En la rama 15 la amortiguan conviccion_de_rama (lib.rs:141-150), que con la cota superior de Wilson < ½ devuelve una confianza < ½, y la exigencia de continuación para el umbral moderado (lib.rs:512-519). Ni el consejo (lib.rs:7634) ni fused_aligned (lib.rs:3611) están amortiguados. Faltan lectores de la masa con |p|: micro_resonant_tau_ms, que fija la duración de la banda rápida (lib.rs:5474-5479), y el arbitraje por continuous_energy_density (lib.rs:6758-6766 y :6796-6804). Severidad media.
  - Corrección (completitud): Severidad media. Siguiente paso: RLS/ridge conjunto por horizonte (ver faltante FA-5 para el horizonte).

### F2.3-04. Dos vectores de pesos y dos cantidades distintas con el mismo nombre spectral_coherence

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/temporal_spectrum.rs:1079-1102 (pesos_observables) frente a :975-1072 (pesos_espectrales, global_coherence); doc :242-245; crates/god-engine-core/src/lib.rs:1810 y :6995-7110
- Causa: TemporalSpectrum::spectral_coherence pondera con observable/vol, y arena.spectral_coherence (escrita desde field.global_coherence) pondera además con 2|p|·gain. X-016 combina en una sola decisión la versión V_obs (lib.rs:7006) con la marea macro y el campo, que son V_esp (lib.rs:7007-7009). El risk-engine lee la versión V_esp creyendo leer la coherencia del consejo. El comentario de :1080-1084 («una sola verdad») y la doc de :242-245 son falsos.
- Consecuencia viva: crash_flux, free_cap (lib.rs:7905-7921) y risk-engine/src/lib.rs:957, :1061 y :1069 juzgan con una «coherencia» numéricamente distinta de la del consejo y X-016.
- Prueba que falla hoy: cf2_un_solo_vector_de_pesos_para_fusion_y_campo: |spec.spectral_coherence(true) − spec.spectral_field(true).global_coherence| < 1e-9 en todos los eventos tras el calentamiento. Hoy difieren en todos los eventos con |p| ≠ ½.
  - Actualización (revisión cruzada, CL-65): la prueba se llama cf2_contrato_peso_unico_fila1. Es un contrato de diseño: se pone verde si el arreglo unifica los pesos; si el arreglo es renombrar, se borra.
- Dueño: Línea C (C-F3)
- Fila de §3 del plan: 1
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Severidad baja: es colisión de nombres y una decisión de diseño pendiente (siguiente paso de la fila 1), no una decisión viva errónea demostrada. Hay que añadir la rama 15, que mezcla fused (V_obs) con field.global_coherence (V_esp) en confluencia_resonante (lib.rs:505-529, llamada en :6588-6594).
  - Corrección (completitud): Causa: «V_obs = observable/σ_dev, V_esp = max(2|p|·gain, 0,02)·observable, sin 1/σ». Severidad media.

### F2.3-05. crash_flux es alto en ruido puro y recorta capital y alimenta vetos

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/spectral_regime.rs:80-127; test engañoso en spectral_regime.rs:212-217
- Causa: Se suman cuatro efectos. (a) La deriva = Δln τ*·3,6e6/elapsed se recorta a ±1 y satura con |Δln τ*| ≥ 0,0167 por minuto, así que en ruido accel_fast ≈ 1 la mitad del tiempo (0,35·0,5 ≈ 0,17). (b) 1 − entropía tiene un piso de 1 − ln k/ln 32 (≈ 0,34 con k = 10), que aporta ≈ 0,07. (c) |marea| > 0 en ruido. (d) micro_anti lee hurst_at(15 s) = (p+1)/2, que en datos reales baja por el rebote bid-ask. El test que dice probar el ruido usa un espectro frío sin energía y da 0 por construcción. Viola los puntos 4 y 5 de G-TEO §30.1.
- Consecuencia viva: lib.rs:7905-7921 multiplica free_cap por (1 − 0,95·flux) cuando la marea es adversa: ≈ 30 % menos capital en ruido para el lado contrario a la marea. orchestrator.rs:164-183 suma 0,25·max(crash_flux) a la presión direccional, que contribuye a p_crash/p_bull ≥ 0,90 (veto, :186-202).
- Prueba que falla hoy: cf2_rw_crash_flux_es_bajo_en_ruido: caminata aleatoria de 21 600 eventos. Exige media de crash_flux < 0,10 y fracción con |dominant_drift| ≥ 0,99 < 0,10, que es la falsación que el propio módulo declara en :21-23. Hoy se espera una media ≈ 0,3.
- Dueño: Línea C (C-F2.3, régimen)
- Fila de §3 del plan: 9
- Relación con hallazgos anteriores: R7-R2-C-3 (sólo la parte de saturación de la deriva)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La consecuencia en el orquestador está mal descrita. crash_flux entra en directional_pressure = 0,25·max(directional_max, systemic_pressure) (orchestrator.rs:164-183), que es un máximo y no una suma, y esa presión resta al exposure_limit (:245-249). No interviene en los vetos p_crash ≥ 0,90 ni p_bull ≥ 0,90 (:185-202), que leen sólo el símplex. El recorte de free_cap está en lib.rs:7908-7918. La cuantificación de free_cap y la saturación de la deriva ya están en R7-R2-C-3.
  - Corrección (completitud): Consecuencia: «resta hasta 0,25 del límite de exposición (orchestrator.rs:176-183, :244-249)», no «contribuye al veto p_crash/p_bull ≥ 0,90». Severidad media.

### F2.3-06. La σ pronosticada es la mediana (sesgo de Jensen) y se usa como media en TP/SL

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/spectral_tape.rs:713-721 (sigma_at), :488-512 (objetivo en ln)
- Causa: El RLS pronostica E[ln RV], y √exp(·) da la mediana. Con n incrementos gaussianos por ventana el sesgo exacto es E[ln σ̂ − ln σ] = ½(ψ(n/2) + ln 2 − ln n): −0,135 con n = 4 y −0,032 con n = 16.
- Consecuencia viva: tp_sl.rs:352-355 (vía risk-engine/src/lib.rs:900 y sigma_forecast_at): stops y TP un 10-13 % más estrechos en monedas de poca actividad cuando habilidad_volatilidad > 0.
- Prueba que falla hoy: cf2_sigma_pronosticada_es_insesgada_con_cuatro_trades_por_ventana: |sesgo log| < 0,05 en el ancla 17 con 4 trades por ventana. Hoy se espera entre −0,11 y −0,135, según cuente o no el incremento del borde.
- Dueño: Línea C (C-F2.3, fila 3)
- Fila de §3 del plan: 3
- Relación con hallazgos anteriores: R7-R2-C-8
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Duplica R7-R2-C-8, que sigue vigente; hay que presentarlo como su cuantificación, no como hallazgo nuevo. El rango «−0,11 a −0,135» se reduce a −0,135, porque la ventana cuenta 4 incrementos y no 5. El sesgo baja a −0,032 con 16 incrementos, así que pesa en las monedas poco activas. Severidad media.
  - Corrección (completitud): Añadir el consumidor lib.rs:1525-1570 y :1600-1608. Severidad media.

### F2.3-07. El IC cruzado se atenúa por el desfase de bloques entre monedas

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/espectral_multiactivo.rs:195-206 (tolerancia 0,5·τ) y :94-119
- Causa: El código empareja bloques cuyos cierres distan hasta 0,5·τ, pero la covarianza esperada de dos bloques desplazados un gap es ρ·(τ − gap)/τ. Sin corrección, el IC subestima |ρ| hasta en un 50 %.
- Consecuencia viva: qo_613_rho_tau subestima la dependencia y el veto de grupo (risk-engine/src/lib.rs:676-704) se endurece menos de lo que debe. La ruta se activa poco por R7-R2-C-5.
- Prueba que falla hoy: cf2_ic_cruzado_estima_rho_a_la_escala: el control alineado da ≈ 0,6 (pasa); con un desfase de 28 s se espera 0,6·41/69 ≈ 0,357 frente al contrato |IC − 0,6| < 0,10 (falla).
- Dueño: Línea C (C-F2.3)
- Fila de §3 del plan: 1
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay que añadir dos cosas. Con fases aleatorias el gap es uniforme en [0, 0,5τ], así que la atenuación media es 0,75. Y el e-proceso del par (:94-103) también se ralentiza: el acierto de signo baja de 0,705 a 0,616, y el crecimiento log por par, de ≈ 0,036 a ≈ 0,018, con lo que hacen falta ≈ 2× pares para cruzar 43 500. El consumidor sólo aprieta (risk-engine/src/lib.rs:686-704), así que el efecto es apretar menos, y esa ruta se activa poco (R7-R2-C-5). Severidad media, en el límite con baja.
  - Corrección (completitud): Severidad media. Añadir una variante de la prueba con fases aleatorias.

### F2.3b-03. La entropía viva mide la proporción de eventos sin cambio de mid, no la forma de los retornos

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:742-745; crates/god-engine-core/src/math_kernels.rs:420-462
- Causa: Se alimenta en cada evento con el retorno del mid. Los eventos sin cambio (trades al mid del libro, depth sin cambio de BBO) son ceros exactos que caen en uno o dos bins centrales, y la memoria es de unos 1000 eventos, no de un tiempo.
- Consecuencia viva: Llega a la feature [8] del bosque (stateful_engine.rs:1104 → lib.rs:4643) y a FlowImpulseEngine::evaluate_flow_impulse (lib.rs:6343-6349): la misma moneda cambia de «entropía» al cambiar la actividad del libro.
- Prueba que falla hoy: cf2_entropia_invariante_a_eventos_sin_cambio: la misma difusión de 20 000 movimientos, con 0 % y con 80 % de eventos sin cambio. Predigo 0,9685 frente a un valor entre 0,37 y 0,59 (según caigan los ceros en uno o en dos bins), con tolerancia de 0,05.
- Dueño: Línea C (núcleo). FlowImpulseEngine es signal-engine (no verifiqué su zona).
- Relación con hallazgos anteriores: Es de la misma familia que LC-F1-31 (memoria en eventos), pero no es el mismo hallazgo.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Ninguna. La relación con LC-F1-31 está bien declarada.
  - Corrección (completitud): Hay que enlazarlo con la fila 43 de §3. FlowImpulseEngine usa la entropía en la confianza, tanh(1 − 10·poly_a·H), y en el umbral z, 1 + 100·poly_b·H (signal-engine/src/flow_impulse.rs:152-160). La rama 12 gana confianza y baja su umbral cuando crece la proporción de eventos sin cambio de mid. La prueba propuesta especifica el reloj; no es una solución conocida del estimador, porque la distribución por evento sí cambia de verdad. Se mantiene media.

### F2.3b-04. El FFT de 64 eventos no localiza un ciclo físico: el bin depende de la densidad de eventos

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/spectral.rs:1-6, :57-121; crates/god-engine-core/src/stateful_engine.rs:756-771
- Causa: No tiene eje temporal: el bin son ciclos por evento y la ventana dura lo que tarden en llegar 64 eventos.
- Consecuencia viva: Tres features del vector ML (bin, potencia y centroide; lib.rs:4694, :4733, :4769) cambian con la actividad del libro. Train y serve sólo coinciden si el tape tiene la misma mezcla de eventos.
- Prueba que falla hoy: cf2_ciclo_muestreo_irregular_bin_4: un ciclo de 18 s observado 4 veces en 72 s, con ráfagas de 16 eventos a 10 ms y tramos de 16 a 2,24 s. Predigo el bin 8 frente al bin 4 esperado; el control uniforme da 4.
- Dueño: Línea C (espectro). El versionado del vector ML, con Línea E (model_manifest).
- Fila de §3 del plan: 7
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Severidad media, aceptable: a diferencia de F2.3b-09, el bin no es una transformación monótona de una magnitud física, y la mezcla de eventos del tape cambia su significado.
  - Corrección (completitud): El siguiente paso de las «firmas con canal temporal» no vale: una firma de nivel 2 describe la geometría y el orden del camino, no estima frecuencias. Los sucesores correctos son dos: - Lomb-Scargle (Lomb 1976; Scargle 1982) sobre pares (t, r), con probabilidad de falsa alarma (Baluev 2008); - el banco de 32 escalas en tiempo físico que ya existe (quantum-arena/src/temporal_spectrum.rs). Cualquiera de los dos exige versionar el vector con Línea E. Falta además un hecho: el calentamiento REST mete retornos de 1 minuto en el mismo anillo de eventos (stateful_engine.rs:1000-1003).

### F2.3b-05. El Reynolds de Navier-Stokes es proporcional a 1/√dt: mide cadencia

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/navier_stokes.rs:134-145, :160-199; crates/god-engine-core/src/lib.rs:5322-5333; crates/metacortex-engine/src/consejo_seniors.rs:414-420, :738-746
- Causa: u = Δmid/dt con el dt entre eventos. Para un mid difusivo |Δmid| ∝ √dt, así que Re ∝ dt^{−1/2}. Además la EWMA es por evento (α = 0,05), y los trades al mid del libro dan Re = 0.
- Consecuencia viva: Con la misma difusión, laminar_share vale unos 0,09 a 10 ms y unos 0,50 a 100 ms. Por eso hydro_factor pasa de 0,45 a 0,70 (consejo_seniors.rs:416), y el deslizamiento efectivo de ×1,45 a ×1,25 (:740), que alimenta el veto de ejecución (:745-746).
- Prueba que falla hoy: cf2_re_invariante_a_la_cadencia (crates/feature-engine/tests/cf2_navier_stokes_cadencia.rs): la misma σ = 0,02 USD/√s con dt de 10 ms y de 100 ms. Predigo un cociente de √10 ≈ 3,16 frente a 1 ± 0,10.
- Dueño: AGY (pedido con test; Línea C no lo arregla)
- Fila de §3 del plan: 8
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay que sustituir «por evento» por «por trade o kline no bloqueado» y retirar la frase del pico por depth. Las anclas correctas son hydro_factor en consejo_seniors.rs:423 (no :416) y el deslizamiento en :741 (no :740).
  - Corrección (completitud): El sucesor que nombra la fila 8 (intermitencia multiescala) ya existe y está vivo: χ = (1,5·ζ₂ − ζ₃)⁺ (temporal_spectrum.rs:1368-1377), publicado como spectral_intermittency (lib.rs:1848-1852 y :2973-2977). Lo leen intermittency_mult_de (lib.rs:2866-2871) y lib.rs:4992. El pedido a AGY debe exigir que NS demuestre valor incremental sobre χ en tiempo físico, o que se retire.

### F2.3b-06. El curl_share de Hodge es una función cerrada de momentos transversales y vale unos 0,93 sin ninguna circulación

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/hodge_flow.rs:219-249, :79-156; crates/god-engine-core/src/lib.rs:5296-5320; crates/metacortex-engine/src/consejo_seniors.rs:405-411
- Causa: F = ½(abᵀ − baᵀ), con a = OFI y b = tanh(100r), es de rango 2. Por eso curl_share = C/(C+G), con C = ¼(|a′|²|b′|² − (a′·b′)²) y G = (n/4)|b̄a′ − āb′|². Con medias pequeñas da unos (n−1)/(n+1). No hay retardo ni ciclo: sólo momentos de corte transversal de retornos de un evento, que además son asíncronos.
- Consecuencia viva: laminar_factor ≈ 1 − 0,70·0,93 ≈ 0,35 con 30 monedas, o 0,53 con 5: SeniorMicroestructura pierde entre el 47 % y el 65 % de su confianza sin evidencia (consejo_seniors.rs:411).
- Prueba que falla hoy: cf2_hodge_sin_circulacion_no_es_vortice: con OFI y retornos independientes y n = 30, el doc (hodge_flow.rs:29-30) promete curl bajo sin vórtice, y la aserción pide < 0,5. Predigo una media de unos 0,93. La identidad se demuestra en cf2_hodge_curl_share_es_funcion_de_momentos_transversales.
  - Actualización (revisión cruzada, CL-65): esa prueba no existe. La ley nula está en la activa cf2_hodge_ruido_independiente_da_curl_cercano_a_uno (curl ~ Beta((n−2)/2, 1), media 14/15 con n = 30; observado 0,9314) y en la identidad cf2_hodge_curl_share_es_funcion_de_momentos_transversales. Este hallazgo no tiene prueba roja: la falsación pendiente está en el consumidor (SeniorMicroestructura, laminar_factor). Sobre el dueño, el encabezado del archivo dice «AGY por historia» y el escéptico de completitud dice Línea C (hodge_flow.rs no está en ninguna zona ajena). Rige Línea C.
- Dueño: AGY por historia (Ω37). No verifiqué la reserva de zonas; va como pedido con test.
- Fila de §3 del plan: 44
- Relación con hallazgos anteriores: Está relacionado con R7-R3-A-2 (el test del vórtice no discrimina), pero no es el mismo hallazgo.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): El ancla correcta de laminar_factor es consejo_seniors.rs:413 (no :411). Como en NS, el cálculo solo corre en trades y klines (lib.rs:4953 frente a :5296). Los valores 0,93 y 0,35 son del nulo gaussiano i.i.d. con n=30; la distribución viva no está medida.
  - Corrección (completitud): Dueño: hodge_flow.rs no figura en ninguna zona ajena de la lista (Codex, Qoder/GLM, AGY = Feynman, NS, YM, Prospect y registro, Línea E). El arreglo o la retirada son de Línea C, no un pedido a AGY. Nulo: barajar etiquetas de moneda es trivial, pero el surrogado que desacopla (permutar los retornos respecto a la OFI entre monedas) no lo es. Muestra que la única información del estadístico es la correlación transversal entre OFI y retorno, más sus medias. Sucesor realizable: HodgeRank (Jiang, Lim, Yao y Ye 2011) sobre una matriz antisimétrica de lead-lag de Hoffmann-Rosenbaum-Yoshida (lead_lag.rs, F2.4), con nulo por desplazamiento circular. Junto con NS, la confianza de SeniorMicroestructura queda en |OBI|·0,35·(0,45-0,70), es decir, entre 0,16 y 0,25 veces |OBI| (consejo_seniors.rs:407-424).

### F2.3b-07. La curvatura de Yang-Mills es idénticamente cero salvo por la deriva de β (gauge puro)

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/strategy-core/src/yang_mills_gauge.rs:148-190; crates/god-engine-core/src/lib.rs:5274-5294; crates/metacortex-engine/src/consejo_seniors.rs:452-463; src/bin/god_engine.rs:4265-4268
- Causa: A_ij = r_i − β_ij r_j con β = 1 es una conexión exacta, así que F_ijk ≡ 0. El LMS mueve β a un ritmo de unos 5e-9 por evento. Con numerario común USDT no existe holonomía observable, y con retornos asíncronos F ≈ r_c(1−β).
- Consecuencia viva: Ninguna efectiva: w_ym ≈ 1e-13 en SeniorSeriesTemporales y la variable del host es muerta. Su coste es de O(n³) por evento.
- Prueba que falla hoy: cf2_ym_dislocacion_triangular_plantada_se_detecta (crates/strategy-core/tests/cf2_yang_mills_nulo.rs): una dislocación AR(1) plantada en el cruce, con Var(Δu) = 2e-7 por paso. La aserción pide una acción media > 1e-9 y predigo unos 1e-14.
- Dueño: AGY (pedido con test)
- Fila de §3 del plan: 8
- Relación con hallazgos anteriores: Complementa R7-R3-B-2, R7-R3-E-4 y R7-R3-D-3, que siguen vigentes. La nulidad idéntica es nueva.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Añadir el consumidor del orquestador. Retirar YM no es neutro: cambia el denominador all.len() de avg_conviction cuando decide el ensamble escalar, y puede mover T-1. Los tests existentes (crates/strategy-core/tests/yang_mills_gauge_contract.rs:63-88 y :105-144) solo obtienen acción no nula con with_beta≠1, una configuración que no existe en vivo.
  - Corrección (completitud): Duplica en parte LC-F2-06 del plan («Yang-Mills ≈ 0 con retornos asíncronos», PLAN:1060). Lo nuevo es la nulidad idéntica con β = 1. El test vigente yang_mills_contrato_corriente_restauradora_tras_dislocacion (strategy-core/tests/yang_mills_gauge_contract.rs:64-88) sólo pasa porque with_beta planta β ≠ 1, y producción nunca lo usa (lib.rs:1070 y :1184). Certifica r·(1−β), no una dislocación. La severidad media sólo se sostiene por letra (consumidor vivo); el efecto vivo es nulo.

### F2.4-02. El contagio se mide sobre el reloj de @depth5, no sobre la actividad de mercado

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/god-engine-core/src/lib.rs:2594 (tick_ring sólo dentro de if is_depth, :1967) → crates/god-engine-core/src/contagion_publisher.rs:33-52 → crates/feature-engine/src/hawkes_cross.rs:62-107
- Causa: tick_ring se alimenta sólo con eventos de profundidad, que llegan con una cadencia casi periódica: 250 ms según la documentación de Binance para @depth5 sin sufijo, suscrito en god_engine.rs:994-998; no se verificó con tape. cross_excitation compara la fracción de aciertos en (t, t+lag] con base = densidad·lag. Con 4 eventos/s, la base es ≥ 1 para todo lag ≥ 500 ms (se recorta a 1 − 1e−9), así que sólo puede puntuar el lag de 200 ms. Ahí, dos relojes periódicos independientes con un desfase estable entre 1 y 199 ms dan rate = 1 frente a una base de 0,80: z = 7,92 en un sentido y 0 en el otro. El nulo binomial supone llegadas sin estructura, y un reloj periódico no lo cumple.
- Consecuencia viva: Con N = 2, el seguidor recibe net ≈ −7,9. modulate_by_net_role recorta su confianza un 18 % (lib.rs:7157-7165), y el guard aplica amplificar_por_contagio con boost = min((7,9 − 3)/10; 0,5) = 0,49 (correlation_guard.rs:638-640, :821-833): un par con r = 0 pasa a 0,49 en la admisión. Con N monedas, la dispersión del rol baja como ≈ 7,92·√(0,4/(N−1)) (1,67 con N = 10), pero cada par con fase estable conserva su celda. La severidad es media condicionada y pasa a alta si el tape confirma una cadencia periódica con fase estable entre símbolos. Los tests existentes (hawkes_cross.rs:222-279, tests/hawkes_chain_integration.rs:17-96) usan rejillas regulares desplazadas como contagio y certifican el artefacto.
- Prueba que falla hoy: cf2_contagio_relojes_periodicos_independientes_no_son_contagio (crates/feature-engine/tests/cf2_dependencia_solucion_conocida.rs): dos relojes de 250 ms desfasados 30 ms, con jitter de 0 a 9 ms; espera m < 3 y |net| < 1; hoy m[0][1] ≈ 7,9 y los roles valen ±7,9.
- Dueño: Qoder/GLM (hawkes_cross.rs, contagion_publisher.rs); Codex para el lector en correlation_guard.rs.
- Fila de §3 del plan: 12
- Relación con hallazgos anteriores: Relacionado con F2-C5 (máximo de z sin corrección) y con Codex §23–25; no duplica ninguna ficha LC-F1.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Media condicionada, como dice el auditor. Añado el escenario contrario, también sin verificar: si Binance sella el @depth5 de todos los símbolos con el mismo tick de servidor (fase común con signo aleatorio), el rate cae a ≈ 0,5 frente a la base de 0,8. La matriz sale entonces de ceros y el modulador queda inerte. En los dos casos el estimador no mide contagio de actividad. Tampoco existe contraparte en el replay (ver faltantes).
  - Corrección (completitud): Añadir a la consecuencia viva el curl ≈ 0,65 con tres relojes, que suma (1 − ρ)·0,43 a la ρ sistémica de toda admisión. Hay que decir también que B1 certifica la fórmula y no el cableado (G-TEO §30.1, punto 1): falta un test de cableado, que propongo en los faltantes. La severidad media queda condicionada y sube a alta si el tape confirma fases estables entre símbolos.

### F2.4-03. La intensidad viva incluye el propio evento (lectura de Palm), contra la fórmula declarada con t_i < t

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/hawkes_bessel.rs:232 (dt >= 0.0) frente a la documentación en :14; lib.rs:2726 → :4484-4486; sombra en lib.rs:2214-2223
- Causa: El núcleo registra el trade (lib.rs:2726) y después lee intensity_ratio en el mismo event_time (lib.rs:4484). Las decisiones sólo se toman en trades (lib.rs:2796). Como intensity incluye dt = 0, cada lectura viva suma el α del evento que la provoca: E[λ/μ̂ | evento] = 1,6 + α/μ̂, que vale 1,89 a 1/s, 2,2 a 0,5/s y 1,63 a 10/s. La intensidad predecible de la teoría es λ(t⁻). Además, la sombra (lib.rs:2214-2223) lee ANTES del registro y sólo en eventos de depth, así que sombra y vivo miden cantidades distintas.
- Consecuencia viva: El sesgo depende del ritmo: los símbolos lentos parecen excitados siempre (excitación ≈ 0,18 a 1/s sin cascada), lo que favorece a los ilíquidos en la puerta de confluencia, en el voto del tensor y en el slot 2 del PPO. Viaja por el mismo canal que F2.4-01 y se suma a él.
- Prueba que falla hoy: cf2_hawkes_poisson_1hz_lectura_en_evento_excitacion_nula: Poisson a 1/s leído en cada evento; espera 1,6 ± 0,05 y una excitación |e| ≤ 0,03; hoy da ≈ 1,89 y ≈ 0,18.
- Dueño: Qoder/GLM (hawkes_bessel.rs); el núcleo, para el orden de registro y lectura.
- Fila de §3 del plan: 11
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay que corregir la premisa «las decisiones sólo se toman en trades». allow_entries = !latency_panic && (!is_depth || is_trade) (lib.rs:2796) vale true también en los eventos de kline (is_trade = false, is_depth = false), y el host los pasa a process_event (src/bin/god_engine.rs:3138-3145 y :3406-3410). En esos instantes la lectura no es de Palm. El sesgo de la decisión viva es entonces una mezcla ponderada por r_trade/(r_trade + r_kline). Se diluye justo en los símbolos lentos, donde el término α/μ̂ es mayor. La cadencia de @kline_1m (250 ms según la documentación de Binance) no la verifiqué en el repo. Severidad media, sin cambio. El test A4 sigue siendo válido como contrato de la lectura en el trade.
  - Corrección (completitud): Reescribir la consecuencia viva: el sesgo 1,6 + α/μ̂ sólo afecta a las decisiones tomadas en eventos de trade. La lectura viva es una mezcla: Palm en los trades, fase casi uniforme en las klines (≈ 4/s) y el evento de cierre para el PPO. En una moneda de 0,5 trades/s, ≈ 89 % de los eventos de decisión son klines. La severidad media se mantiene. Los tests A4 y A7 deben reproducir esa mezcla (ver los faltantes).

### F2.4-04. El exceso λ/μ̂ − 1,6 no se normaliza por su dispersión nula, que depende del ritmo

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/hawkes_bessel.rs:99-107 y :260-263; crates/signal-engine/src/flow_impulse.rs:126-186; flow_excitation_confluence.rs:224-235
- Causa: Bajo el nulo, Var(λ)/μ̂² ≈ α²/(2βr), así que la dispersión es ≈ α/√(2βr): 0,42 a 0,5/s y 0,055 a 30/s. Un mismo umbral absoluto es entonces una prueba con tamaño distinto en cada moneda. P(R > 2,0) es 15,5 % a 0,5/s y ≈ 0 % a 30/s; P(R > 1,7) es 35 % frente a 15 %. La «escala libre» de la documentación (:260-263) sólo vale para la media, y flow_impulse usa una sola desviación genómica para todas las monedas.
- Consecuencia viva: Falsas cascadas en los ilíquidos: la puerta de confluencia, la z de flow_impulse y el voto del tensor se activan por ruido de muestreo en las monedas lentas.
- Prueba que falla hoy: cf2_hawkes_tasa_de_falsa_cascada_no_depende_del_ritmo: Poisson a 0,5/s frente a 30/s, con umbral 2,0; espera |Δ fracción| ≤ 0,06; hoy da ≈ 0,155.
- Dueño: Qoder/GLM (hawkes_bessel.rs, flow_excitation_confluence.rs); núcleo (flow_impulse).
- Fila de §3 del plan: 11
- Escépticos: confirmado por los dos escépticos.
  - Corrección (completitud): El nulo de Poisson es el correcto para «sin excitación». A7 debe medir la puerta viva, es decir, la rampa de 1,7 con ancho 0,51 de flow_excitation_confluence.rs:270-277, y no un umbral 2,0 inventado. El z-gate de flow_impulse (:162 y :181-183) necesita su propio nulo por ritmo.

### F2.4-05. El tope MU_INST_MAX = 200/s y la siembra MU_SEED_MAX = 50 hunden μ̂ en las ráfagas

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/hawkes_bessel.rs:116-117, :195-199
- Causa: inst = clamp(1/dt, 0,01, 200) recorta cada intervalo de menos de 5 ms. Con Poisson a tasa r, la fracción perdida es L(r) = 1 − (1 − e^{−ra})/(ra), con a = 5 ms: 2,5 % a 10/s, 7,1 % a 30/s y 21 % a 100/s. El cociente sube a 1,615, 1,646 y 1,76. Justo en las ráfagas, cuando más importa, el sesgo crece.
- Consecuencia viva: En las ráfagas la excitación se infla por construcción (≈ +0,10 en el tanh a 100/s), por encima de la excitación real. Afecta a los mismos consumidores que F2.4-01.
- Prueba que falla hoy: cf2_hawkes_poisson_100hz_fase_uniforme_ancla_1_6: Poisson a 100/s; espera 1,6 ± 0,03; hoy da ≈ 1,76.
- Dueño: Qoder/GLM (hawkes_bessel.rs).
- Fila de §3 del plan: 11
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): El título y la causa mezclan dos cosas. MU_SEED_MAX sólo actúa en el segundo evento de la vida del motor (n_seen == 1, crates/signal-engine/src/hawkes_bessel.rs:196-199): es un sesgo de arranque, no de ráfaga. Debe decir «el tope MU_INST_MAX = 200/s (:116, :195) hunde μ̂ en las ráfagas». Matiz vivo: con timestamps en ms, el tope actúa sobre dt ∈ {1, 2, 3, 4} ms, y los dt < 1 ms se convierten en dt = 0, que es F2.4-01. El test A2 usa tiempos continuos y aísla bien el tope.

### F2.4-06. La teoría nombrada no se estima: α y β son literales y n = 0,6 se publica como si fuera una medida

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/hawkes_bessel.rs:82-91, :246-252; crates/god-engine-core/src/lib.rs:4484-4486 (hawkes_eta_real), :4539 ("hawkes_branching")
- Causa: α = 0,3 y β = 0,5 son constantes; branching_ratio devuelve siempre 0,6, y el núcleo lo llama hawkes_eta_real y lo publica. λ/μ̂ = 1 + (α/β)·r̂_2s/r̂_60s es un cociente de dos tasas EWMA (de 2 s y de 60 s), no un estimador de n ni de la intensidad condicional: en fase uniforme no distingue n = 0,6 de n = 0 (1,599 frente a 1,601 en el espejo).
- Consecuencia viva: La fila 11 no puede pasar G-TEO §30.1, punto 5 (evidencia OOS contra un nulo). "hawkes_branching" no tiene lector, pero engaña a cualquiera que lo lea en el futuro. Por sí mismo no cambia decisiones.
- Prueba que falla hoy: Ningún test falla hoy: cf2_hawkes_fase_uniforme_no_distingue_n PASA y documenta la ceguera. La prueba que fallaría, un KS de los residuos reescalados Λ(t_i) − Λ(t_{i−1}) ~ Exp(1) frente a un Hawkes plantado con n ≠ 0,6, no se puede escribir porque no existe estimador de (μ, α, β).
- Dueño: Qoder/GLM (hawkes_bessel.rs); Línea C para la clave publicada en lib.rs.
- Fila de §3 del plan: 11
- Escépticos: confirmado por los dos escépticos.
  - Corrección (completitud): Prueba que falla hoy: cf2_hawkes_reescalado_temporal_con_la_lambda_del_motor; el código está en los faltantes. Espera una media de 1 ± 0,05 (≈ 5 errores estándar con n ≈ 10 500) y obtiene ≈ 1,6. Hay que añadir al siguiente paso que, una vez estimado (μ, α, β), la lectura pase a ser λ*(t⁻)/r̂ con ancla 1 y no λ/μ̂ con ancla 1,6.

### F2.4-07. Lead-lag declara líder en el 82 % de los pares independientes, frente al 5 % nominal

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/feature-engine/src/lead_lag.rs:50-58, :117-193, :316; crates/god-engine-core/src/lib.rs:3104-3125, :5351-5377
- Causa: Calcula Pearson sobre NIVELES de un OFI que es una EWMA con α = 0,1 (microstructure.rs:190), con autocorrelación 0,9 por paso. Por el efecto Granger–Newbold, |ρ| se infla, mientras el umbral 2/√(n+2) (con piso 0,25) supone muestras independientes. Además se toma el máximo sobre 5 lags sin corrección. Hay un efecto secundario: la extrapolación congelada de interp_lineal (:316) contradice la documentación (:123-124, «exige muestras futuras») y deja el lag de 10 s sin muestras propias. En el espejo: 0,88 con extrapolación congelada, 0,82 sin ella y 0,19 con ruido blanco.
- Consecuencia viva: predict_altcoin_impulse_con_reloj devuelve div = tanh(0,8·clamp(exceso, ±3))·ρ (:243-244) con un ρ espurio. Esa divergencia entra en el slot 3 del PPO (lib.rs:5351-5377), que pesa 0,70 en micro_score. Las claves "lead_lag_divergence" y "leader_momentum" no tienen lector (lib.rs:4523-4524).
- Prueba que falla hoy: cf2_lead_lag_nulo_tasa_de_falsos_lideres: 400 réplicas de AR(1) independientes con φ = 0,9; espera una tasa de lag > 0 ≤ 0,083; hoy da ≈ 0,82.
- Dueño: Línea C (lead_lag.rs no está en zona ajena); el cableado del slot 3 del PPO es del núcleo.
- Fila de §3 del plan: 13
- Relación con hallazgos anteriores: Reabre R7-R2-E-3 (arreglada en parte por Ω50); R7-R2-E-4 sigue vigente.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay una incoherencia interna. La causa dice «0,88 con extrapolación congelada, 0,82 sin ella», y el código actual congela. Si el espejo es correcto, «hoy» sale ≈ 0,88, no ≈ 0,82; las dos cifras fallan el umbral de 0,083. El umbral falla también con ruido blanco (≈ 0,19, por el máximo sobre los lags), así que el arreglo tiene que corregir la autocorrelación y la multiplicidad a la vez. Los tests existentes tienen un matiz: qo_658_lead_lag_con_lags_y_reloj (lead_lag.rs:334-351) sí planta un desfase de 1 s, pero sólo exige lag > 0 con un seno determinista, así que no es una prueba de recuperación. Falta además omega11_lead_lag_eth_sin_autoreferencia (:399-418). En vivo hay un agravante que el auditor no vio (ventana en muestras): ver faltantes.
  - Corrección (completitud): Añadir los faltantes de factor común contemporáneo y de ventana en muestras. La ficha R7-R2-E-3 tiene un «Extra» que dice que lag ≤ 0 es inalcanzable; eso quedó OBSOLETO en c4e8e285, porque lag_optimo arranca en (0, 0) (:184) y devuelve 0 cuando nada pasa el umbral.

### F2.4-08. El modulador de contagio descuenta hasta un 30 % sin dirección y sin evidencia OOS

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/signal-engine/src/contagion_modulator.rs:26-33; crates/god-engine-core/src/lib.rs:7157-7165
- Causa: El descuento es 0,30·m/(m + 5) para net_role < 0, con m = −net_role recortado a 50. No depende de la dirección de la señal ni del movimiento del líder. net_role es un z de coincidencia de actividad, sin signo de precio, así que no contiene información direccional que justifique castigar a los seguidores. No hay medición OOS contra un nulo: G-TEO §30.1, punto 5, exige que vaya en sombra con factor 1.
- Consecuencia viva: Combinado con F2.4-02, recorta la confianza de cualquier moneda cuyo reloj de depth vaya por detrás del de otra, sin relación con el riesgo de contagio real.
- Prueba que falla hoy: No falla ningún test numérico hoy. Contrato propuesto: mientras no exista registro OOS del efecto, modulate_by_net_role(0,8; −7,9) debe devolver 0,8; hoy devuelve ≈ 0,653 (descuento del 18,4 %).
- Dueño: Qoder/GLM (cadena de contagio); el núcleo, para el lector en lib.rs:7157-7165.
- Fila de §3 del plan: 13 (el plan deja el signo «por revisar»)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (completitud): Además del factor 1 en sombra, el sucesor debe modular por la celda m[líder][seguidora] y por el movimiento reciente del líder, con su IC medido.

### F2.5-01. La convicción del Ente sólo decide en dos sitios: su veto de cascada y el suelo del override de un veto único

- Dominio: F2.5, Conducta humana y de masas.
- Título del auditor, refutado tal como estaba: «La convicción continua del asiento EnteMercado no llega a ninguna decisión viva; sólo decide su veto de cascada». Los dos escépticos muestran que la convicción del Ente sí cambia el override de un veto único, porque los asientos direccionales no votan ±1.
- Ubicación: crates/metacortex-engine/src/consejo_seniors.rs:615-622, :1171-1184, :1194-1205, :1281-1287, :1310-1331; crates/god-engine-core/src/lib.rs:7757-7782
- Causa: Tras MOD2/7-006 y QS-C22, la aprobación exige consenso y directional_signal de los asientos direccionales, que excluyen al Ente. Los moduladores votan intended_direction con peso positivo, así que si D tiene el signo pedido, final_signal también lo tiene. El override de un veto único exige consenso ≥ 0,80, lo que fuerza |final_signal| ≥ 0,6 > 0,28. El llamador sólo lee approved y el signo, y vetoed_by no tiene lector. Ballena, OI, spoofing, LS, taker, staleness macro y Prospect quedan como telemetría sin publicar. Los documentos (entes_consejo.rs:7-8, consejo_seniors.rs:549-553) y los tests (prospect_theory_contract.rs:198-216, consejo_seniors.rs:1842-1845) afirman una modulación que no existe.
- Consecuencia viva: Ninguna entrada se modera por ballenas, OI, spoofing, masa, staleness macro ni Prospect, aunque el diseño lo declare. El operador cree que tiene protecciones que no existen.
- Prueba que falla hoy: cf2_ente_extremo_sin_cascada_modera_alguna_decision: sobre la rejilla de cf2_ente_inerte_en_la_decision_salvo_cascada (20 000 cargas útiles, semilla 0xC0F2_0025, liq ≤ 0,85), exigir al menos un caso en el que el Ente extremo (z = 10, OI = 1, spoof = 1, LS y taker adversos 20/0,05, staleness 1 h, P = ±50 adverso) cambie approved. Hoy cambia en 0 casos, y la prueba de inercia complementaria pasa.
  - Actualización (revisión cruzada, CL-65): la prueba se dividió en cf2_ente_inerte_sin_vetos (activa; ahora sólo exige el signo de final_signal en las cargas aprobadas, porque en una no aprobada el Ente puede invertirlo) y en cf2_ente_decide_por_el_suelo_del_override y cf2_ente_decide_el_override_de_un_veto_unico (activas, formas cerradas a 1e-12).
- Dueño: Línea C (C-F3, consejo). Decidir entre sombra declarada (G-TEO 5) o retirada.
- Fila de §3 del plan: 19
- Escépticos: reformulado (un escéptico lo refuta tal como estaba escrito y otro lo da por dudoso).
  - Lente refutar, refutado: La «demostración algebraica» supone que los asientos direccionales votan ±1, y no es así. SeriesTemporales (consejo_seniors.rs:464, :472), ML (:781-782) y Teleonomía (:883, :887) emiten una señal fraccionaria con confianza |s|. Metacognitivo emite media·{1; 0,7; 0,4} con una confianza que no depende de la magnitud (adjusted_confidence) y un peso de hasta 2 (:834-838). El consenso cuenta capacidad c·w (:1206-1224), no s·c·w. Por tanto, un consenso ≥ 0,80 no fuerza |final_signal| ≥ 0,6. El suelo del override, |final_signal| ≥ 0,28 (:1283-1284), se evalúa sobre un final_signal que incluye a los moduladores, Ente incluido (:1169-1184). Por ahí la convicción del Ente decide si se sobreescribe un veto único. Contraejemplo exacto, alcanzable con la API pública y con valores que el tracker vivo produce: 500 resultados en el tracker (lib.rs:4247 los alimenta en vivo), wr = 0,8 ⇒ safe_wr = 404/508 = 0,79528 (:1119), y peso de tracker 2,0 para Metacognitivo, que es el tope de compute_weights (:1507). Payload: OBI 0, Hurst 0,5, ml_prob = base = 0,30, fused 0,05, persistence 0,5, ATR 0,005, racha 3, dir +1 y VPIN 0,80, lo que da un único veto, el Causal. Direccionales: D = 0,015535 y A'' = 2,81215, con consenso 1,0. Moduladores: Vol 0,36 y Riesgo 0,40. Con el Ente neutro (e = 1), f = 1,77553/4,57215 = 0,3883 ≥ 0,28: hay override y se aprueba. Con el Ente extremo (e = 0,20), f = 0,97553/3,77215 = 0,2586 < 0,28: se mantiene el veto Causal y no se aprueba. La prueba de inercia del auditor pasa sólo porque su rejilla usa un consejo nuevo, donde n = 0 fija safe_wr = 0,5 sea cual sea el wr (:1119), la capacidad de Meta queda en ≤ 0,6875 y la racha ≤ 3 con DD ≤ 0,3 deja a Riesgo en ≥ 0,28. Con eso, los moduladores suman ≥ 0,84 y dominan. Su contraprueba, cf2_ente_extremo_sin_cascada_modera_alguna_decision, falla en su rejilla, pero pasaría con este caso.
  - Corrección (refutar): Nuevo título: «La convicción del Ente sólo decide por un canal: el suelo del override de veto único (consejo_seniors.rs:1281-1287). Fuera de él es inerte, porque la aprobación sale del consenso direccional y del signo de D (:1310-1331)». La severidad sigue en media (teoría mal especificada con consumidor vivo), pero la consecuencia viva es otra: un Ente adverso puede impedir que se sobreescriba un veto Causal, de Ejecución o del Auditor, y uno neutro lo permite. La documentación de modulación general (entes_consejo.rs:7-8; consejo_seniors.rs:549-553) sigue siendo engañosa, porque el efecto es estrecho y no está declarado. La prueba de inercia se sustituye por cf2_ente_decide_por_el_suelo_del_override (código en la prueba cf2_ente_inerte_en_la_decision_salvo_cascada), que pasa hoy.
  - Lente refutar, dudoso: La parte central se confirma. En el camino sin veto, la aprobación exige consenso y directional_signal de asientos que excluyen al Ente (consejo_seniors.rs:1194-1205, :1310-1331), y si D > 0 los moduladores, que votan +dir, no cambian el signo de final_signal. Pero la afirmación «la convicción del Ente no cambia ninguna aprobación» es FALSA en el override de un veto único (consejo_seniors.rs:1281-1287). Ahí se exige |final_signal| ≥ 0,28, y final_signal sí incluye la confianza del Ente (:1166-1175). El argumento «consenso ≥ 0,80 ⇒ |D|/A' ≥ 0,6» supone señales ±1, pero Espectral, ML, Teleonomía y Metacognitivo emiten señal fraccional (:450-455, :779-781, :837, :887). Además, la confianza del Metacognitivo no depende de |s|: vale wr efectivo − penalización (:808, :829, :834). Así, un Metacognitivo con s ≈ 0,007 y capacidad grande da consenso 1,0 con D/A ≈ 0,03, y entonces el Ente decide el override. Contraejemplo determinista con la API pública deliberar_with_weights (consejo_seniors.rs:1069): multiplicador ×3 en Metacognitivo, OBI 0,05, ml_prob 0,54 con base 0,50, atr_pct 0,005, loss_streak 18, intended +1 y do_calculus_risk 0,80, que da un veto Causal único. El resto con Default. Con el Ente neutro (confianza 1,0), final = 1,5343/3,6068 = 0,4254 ≥ 0,28: hay override y se APRUEBA. Con el Ente adverso (confianza 0,2), final = 0,7343/2,8068 = 0,2616 < 0,28: queda VETADO por Causal. En vivo los multiplicadores del tracker llegan hasta 2,0 (:1507) y safe_wr sube con el historial (:1119). Con wr ≈ 0,75 y Metacognitivo ×2, el mismo vuelco aparece (0,2252 frente a 0,377). La frecuencia en vivo no está medida. La prueba de inercia del auditor pasa sólo porque usa un consejo frío: con el tracker vacío, safe_wr = 0,5 sea cual sea el wr que se pase, y los pesos valen 1.
  - Corrección (refutar): Hay que reformular el hallazgo así: «la convicción del Ente sólo decide en dos sitios: el veto de cascada y el override de un veto único (suelo 0,28 sobre final_signal), cuando la capacidad direccional está dominada por asientos con |s| pequeño y confianza grande». La severidad sigue en media, porque los documentos prometen una modulación general que casi nunca opera. La consecuencia viva tampoco es «ninguna»: ballena, OI, spoofing, LS (contado dos veces), taker, staleness y Prospect pueden sostener o levantar un veto Causal, de Ejecución o de Auditor. Test que pasa hoy (la conducta existe): cf2_ente_decide_el_override_de_un_veto_unico (ver faltantes).

### F2.5-09. La EMA del OFI es por evento y cada trade con el libro quieto le mete un 0: la lectura cae con la actividad

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/feature-engine/src/microstructure.rs:190; crates/god-engine-core/src/lib.rs:2806 (process_tick_dual en todo evento) → :3091; consumidor D-492 en :3583-3594
- Causa: update_ofi se llama en trades, depth y klines. En un trade, el libro efectivo es el último medido (D-707, CL-33), así que e = 0 y la EMA se multiplica por 0,9. Con k trades entre dos snapshots de depth, el valor estable es 0,1·OFI/(1−0,9^{k+1}): con k = 9 queda en el 15 % de lo que marcaría sin trades. D-492 corta posiciones con umbrales absolutos (±0,20), justo cuando el VPIN es alto, es decir, cuando hay muchos trades. Además, ofi_value (:3091) es la misma EMA que ema_ofi, así que `ofi_value < −0,25 || ema_ofi < −0,20` (:3586-3587) se reduce a ema_ofi < −0,20. El umbral dinámico tiene además un piso absoluto, el gen dynamic_ofi_threshold (:2883-2891). La suma de CKS en el intervalo no cambia con eventos que no mueven el libro.
- Consecuencia viva: El corte tóxico D-492 y los votos de OFI con piso absoluto se disparan menos cuanto más se opera.
- Prueba que falla hoy: cf2_ofi_no_depende_del_numero_de_trades_entre_snapshots: el libro sube un tick por snapshot (OFI normalizado verdadero +1). Sin trades la lectura es ≈ 1,0 y con 9 trades es 0,1535. Se exige una diferencia ≤ 5 %.
- Dueño: Línea C (alimentación del núcleo y salidas, fuera de las ramas 1-10)
- Fila de §3 del plan: 19 (archivo del dominio; la teoría OFI es de F2.3)
- Relación con hallazgos anteriores: Misma familia que LC-F1-31 (EWMA con memoria en eventos), pero no la duplica: es otro estimador y otro mecanismo (ceros inyectados por trades).
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay un segundo mecanismo que el hallazgo omite. Cuando el print cae fuera del último libro, libro_efectivo desplaza el libro (lib.rs:550-565) y el OFI ve una subida de precio sintética con las cantidades de muro: e = q_b + q_a,prev, un OFI ≈ +1 fabricado por un trade y no por el libro. No medí la magnitud neta. La teoría OFI es de F2.3: hay que coordinar para no publicarlo dos veces.
  - Corrección (refutar): Faltan dos consumidores vivos de ema_ofi. (1) La salida thesis_broken (lib.rs:3630-3631: pnl < −0,85·SL, ema_ofi adverso ±0,20 absoluto y tendencia adversa), otra salida de posiciones vivas que sufre la misma atenuación. (2) La entrada de Hawkes por tick (stateful_engine.rs:913-919), en zona Qoder/GLM y ya señalada como F3-B3 en BARRIDO_EXHAUSTIVO_FASES.md:413. El literal de D-492 ya figura en §4 del plan (fila «lib.rs:3504-3507», C-F3): conviene citarlo para no duplicar.

## Severidad baja

### F2.1-05. Documentación falsa en Hodge y Lundberg

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/risk-engine/src/hodge.rs:28-29 y :34-41; crates/risk-engine/src/cramer_lundberg.rs:30-34, :204, :226 y :249, :278-279 frente a :293; crates/god-engine-core/src/evidence_publication.rs:41 con crates/risk-engine/src/veto_registry.rs:129-130
- Causa: hodge.rs:36-41 dice que el curl «NO veta nada todavía», pero entra en el veto de grupo. :28-29 anuncia eliminación gaussiana O(N³) y «roster ≤ 16», cuando el código es la fórmula cerrada O(n²) y MAX_COINS es 30. cramer_lundberg.rs:31 limita la bisección a (1e-9, 100], pero el techo llega a 100 000. :204 habla de «margen log» y :13 dice «NO log-capital». Los tests se llaman «newton» y usan bisección. El comentario de :278 dice 20 000 caminatas y el código usa 100 000. La clave lundberg_r_nocional guarda la LCB, no R̂, y V-RISK-006 dice «R del estimador».
- Consecuencia viva: Ninguna sobre las órdenes. Induce a error a quien audite el veto.
- Prueba que falla hoy: Con y = ±0,1 % y p = 0,9 (R = ln 9/0,001 ≈ 2197), lundberg() devuelve Some(≈ 2197), contra el doc de :31. agy_aud_p06_hodge_curl_share_systemic_rho_escalation (risk-engine/tests/correlation_admission_contract.rs:538-580) ya demuestra que el curl entra en el veto, contra hodge.rs:36-41.
- Dueño: Línea C
- Fila de §3 del plan: —
- Relación con hallazgos anteriores: F1-B6 del barrido F1 (BARRIDO_EXHAUSTIVO_FASES.md:184) para hodge.rs:28-29; LC-F4-18 para el doc del publicador.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay más documentación falsa que añadir. cramer_lundberg.rs:12-14 dice «caiga por debajo de −m antes de recuperarse», pero la cota de Lundberg acota la probabilidad de caer alguna vez, en horizonte infinito. cramer_lundberg.rs:280 habla de «mínimo log», y el test acumula una suma lineal. El nombre test_r7_r2_a3_cramer_lundberg_analytical_lcb_tightens_ruin_margin (god-engine-core/tests/r7_r4_neutral_bar_spectral_coherence_contract.rs:121-148) afirma que la LCB aprieta, cuando lo que comprueba es m_lcb > m, que en el veto vivo afloja (ver faltantes).
  - Corrección (completitud): Añadir god-engine-core/src/lib.rs:3943-3946 a la lista.

### F2.1-06. Marchenko-Pastur sin consumidor; effective_bets mal nombrado; el veredicto del borde no tiene tamaño del 5 % y un test es vacuo

- Dominio: F2.1, Matemática del riesgo y de la cartera.
- Ubicación: crates/risk-engine/src/random_matrix.rs:22-197, :352-385 y :387-393; crates/risk-engine/src/correlation_guard.rs:358-421
- Causa: Ningún camino vivo llama a la API (grep en la ficha). systematic_mode compara con el borde asintótico, cuya probabilidad de excedencia en ruido es ≈ 0,17 con N = 8 y T = 512 por Tracy-Widom. effective_bets no es un N efectivo. El test B-10 sólo comprueba algo si effective_bets da Some, y «identidad pura» usa la matriz de unos.
- Consecuencia viva: Ninguna hoy. El riesgo es que alguien lo conecte como prueba de independencia o como descuento de diversificación.
- Prueba que falla hoy: cf2_mp_tamano_del_veredicto_en_ruido. Predicción: tamaño ≈ 0,17, por encima de 0,05 + 3·SE = 0,065.
  - Actualización (revisión cruzada, CL-65): la predicción quedó refutada al ejecutarla: tamaño 0,0185 (37 de 2 000), por debajo de 0,0695, y la prueba es activa. Lo que sigue en pie de F2.1-06 es la falta de consumidor y el nombre de effective_bets.
- Dueño: Codex (pedido)
- Fila de §3 del plan: 14
- Relación con hallazgos anteriores: R7-R2-G-2 (miembro B-10) para el test vacuo; plan §3, fila 14, «MP sin consumidor».
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): «El veredicto del borde no tiene tamaño del 5 %» no es un defecto del módulo: su doc lo declara explícitamente (random_matrix.rs:3-7, :174-176 y :182, «not a calibrated p-value»). Es un riesgo para un consumidor futuro. Precisión del tamaño: con (p, n) = (8, 512), el borde (1+√γ)² = 1,2656 queda entre 0,06 y 0,27 σ_TW por encima del centrado de Johnstone (n−1) o del de Ma (n−½, p−½), con σ_TW ≈ 0,0365. Eso da un tamaño ≈ 0,13-0,17, no exactamente P(TW1 > 0) ≈ 0,17.
  - Corrección (completitud): Tamaño en ruido ≈ 0,13-0,19, no «≈ 0,17» como cifra puntual.

### F2.2-09. La cota jerárquica no es monótona, con n = 0 devuelve la media del prior y en vivo es inalcanzable

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/risk-engine/src/evidence.rs:134-166; crates/risk-engine/src/lib.rs:1137-1157 y :1188; crates/god-engine-core/src/lib.rs:7584
- Causa: Con n < 1 devuelve p0 (la media) y con n ≥ 1 devuelve la cota (media − 1,64·sd). Además, el único llamador vivo siempre fija win_probability ∈ (0,1), así que la rama jerárquica no se ejecuta.
- Consecuencia viva: Ninguna hoy en el camino vivo. Si se restaurase la rama (por ejemplo, con win_probability = 0 como centinela, LC-F1-37), una sola ganada bajaría la p del EV de 0,55 a 0,358. El literal 0,55 sigue vivo en la sonda (risk lib.rs:1188).
- Prueba que falla hoy: cf2_cota_jerarquica_es_coherente_y_monotona: con n = 0 devuelve 0,55 (la cota del prior sería 0,304), y con una ganada devuelve 0,358.
- Dueño: Línea C, C-F2.2
- Fila de §3 del plan: 30
- Relación con hallazgos anteriores: R7-R2-F-1 (en la parte del literal 0,55 de la sonda)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Con n = 0 la rama tampoco se ejecutaría aunque se restaurase: el llamador manda trade_count == 0 a None (risk lib.rs:1135 y :1142-1143). La consecuencia «bajaría la p de 0,55 a 0,358» no existe en ningún estado vivo. Los tests evidence.rs:227 (assert_eq 0,55) y veto_logic_contracts.rs:94 fijan la conducta actual y hay que reescribirlos junto con el arreglo.
  - Corrección (completitud): El test evidence.rs:227 no sólo omite la monotonía: fija con assert_eq el valor incoherente (n = 0 → 0,55), así que el arreglo tiene que cambiarlo. La rama tampoco es inalcanzable en sentido estricto: si la sigmoide de Platt satura a 1,0 exacto (a·logit(s) + b ≳ 37, con calibrate_at en Δt = 0), win_probability = 1,0 no cumple < 1 (risk lib.rs:1137-1139). Es improbable y no lo he medido.

### F2.2-11. correlation.rs no tiene consumidor, y su sesgo de R7-R2-E-9 se cancela en el cociente

- Dominio: F2.2, Estadística, calibración y decisión secuencial.
- Ubicación: crates/feature-engine/src/correlation.rs:49-138
- Causa: El factor (1−α) de actualizar la media antes de la desviación aparece en la covarianza y en las dos varianzas, y se cancela. Lo que sigue en pie es el colapso de Err/None en 0,0 (:49-54) y la falta de consumidor.
- Consecuencia viva: Ninguna.
- Prueba que falla hoy: Ninguna falla por el sesgo: cf2_correlacion_media_con_la_cesta_conocida da √0,75 ± 0,02 y pasa. La retirada se justifica por el grep: MarketCorrelationHeatmap sólo aparece en tests.
- Dueño: Línea C, C-F2.2 (retirada)
- Relación con hallazgos anteriores: R7-R2-E-9; R7-R2-G-1
- Escépticos: confirmado por los dos escépticos.

### F2.3-08. BOCPD con hazard por evento, datos W1 solapados y predictiva sin varianza posterior

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/god-engine-core/src/lib.rs:659-666, :704-758 (:722-724), :1860-1878
- Causa: El hazard de 1/100 es por evento, así que la tasa de cambios a priori depende de la tasa de eventos. W1 con retardo de 60 s calculado en cada evento genera datos muy autocorrelados (iid violado). La predictiva N(μ_r, σ_e²) omite la varianza posterior de la media del segmento y sobreestima la sorpresa en segmentos cortos.
- Consecuencia viva: Ninguna: p_transition se descarta (`let _`, lib.rs:7136-7141) y las claves spectral_w1_transport y spectral_p_transition no tienen lector.
- Prueba que falla hoy: cf2_rw_bocpd_no_dispara_en_ruido (predicción incierta). Propuesta adicional: el mismo proceso muestreado a 1 Hz y a 10 Hz debe dar la misma tasa de alarmas por hora; hoy, por construcción, la de 10 Hz es mayor.
  - Actualización (revisión cruzada, CL-65): se registra F2.3-08b. La prueba cf2_bocpd_detecta_un_cambio_de_volatilidad_plantado comprueba ahora antes que W1 vea el cambio (media de W1 en [cambio, cambio + 60 s) por encima del máximo nulo en ≥ 13/16 caminos). Lo ve en 0/16: la media de W1 tras el cambio ×5 (≈ 0,91 entre caminos) es la de la ventana nula (≈ 0,94), porque el transporte espectral está normalizado y es invariante a la escala. El BOCPD sobre W1 no puede detectar un cambio de volatilidad; las 5/16 «detecciones» del resultado de la nube eran falsas alarmas. Severidad baja (el consumidor es el mismo de F2.3-08); dueño Línea C.
- Dueño: Línea C (C-F2.3, fila 10)
- Fila de §3 del plan: 10
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La sub-afirmación «por construcción la de 10 Hz es mayor» no está demostrada: el prior por hora es 10× mayor, pero la tasa posterior de alarmas depende de lo que se mueva W1 entre eventos casi idénticos. Debe quedar como predicción incierta. Severidad baja.
  - Corrección (completitud): Severidad baja (sin consumidor).

### F2.3-09. Documentación falsa o desfasada en el dominio

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: temporal_spectrum.rs:225-230, :242-245, :1193-1199; spectral_regime.rs:21-23, :30-31, :212-217; espectral_multiactivo.rs:48-52
- Causa: temporal_spectrum.rs:225-230 cita 640 bloques y significativo_familia(32), pero el código usa M = 5 (:129, :664). :242-245 dice que el vector es «compartido por todos los lectores», y no lo es. :1193-1199 afirma una validación por Ville que no se hace. spectral_regime.rs:30-31 habla de bandas micro < 60 s, meso < 30 min y macro ≥ 30 min, pero son los puntos 15 s, 5 min y 30 min (:64-66), y hurst_at no es Hurst (:874). spectral_regime.rs:21-23 y el test :212-217 afirman un crash_flux bajo en ruido que no se prueba. espectral_multiactivo.rs:48-52 dice «cero consumidores de política», pero qo_613 es un consumidor vivo.
- Consecuencia viva: Ninguna directa. Induce a error a quien audite o cablee lectores.
- Prueba que falla hoy: La parte numérica la cubren cf2_rw_crash_flux_es_bajo_en_ruido y cf2_un_solo_vector_de_pesos_para_fusion_y_campo; el resto se comprueba por lectura.
- Dueño: Línea C
- Fila de §3 del plan: 1, 2, 9
- Relación con hallazgos anteriores: Relacionado con LC-F1-53 (doc K41 en otra sección)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Refuto dos partes: :242 y :1080-1084 son ciertas en su alcance (ver F2.3-04). La de espectral_multiactivo.rs:48-52 («cero consumidores de política») ya figura como «choque documental» en R7-R2-C-5. Severidad baja.
  - Corrección (completitud): Severidad baja. Añadir temporal_spectrum.rs:658-663, :1699-1701 y :1862-1884.

### F2.3-10. Cálculo muerto en el camino caliente: pronóstico de volumen y campos sólo de escritura

- Dominio: F2.3a, Espectro continuo temporal y régimen.
- Ubicación: crates/quantum-arena/src/spectral_tape.rs:625, :661-664, :676 (volumen actualizado en cada trade); state.rs:214 y :227 (tau_habilidad y spectral_resonant_tau sólo se escriben)
- Causa: SpectralForecastBank::volumen corre un HorizonForecaster por ancla en cada trade sin ningún lector. notional_at, concentracion_at y flujo_at no tienen consumidor. Los campos del arena tau_habilidad (lib.rs:2024-2026) y spectral_resonant_tau (lib.rs:1812, :2937) sólo se escriben.
- Consecuencia viva: Latencia por trade, sin efecto en ninguna decisión.
- Prueba que falla hoy: Se comprueba por grep: `rg -n "\.volumen\b|notional_at|concentracion_at|flujo_at|tau_habilidad|spectral_resonant_tau" --type rust` sólo devuelve las definiciones y las escrituras.
- Dueño: Línea C
- Fila de §3 del plan: 3
- Relación con hallazgos anteriores: Parcial con R7-R2-C-12 (campos sin lector)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Coincide en parte con R7-R2-C-12, que ya lista arena.spectral_resonant_tau y spectral_w1_transport. Son nuevos el banco volumen, tau_habilidad y los tres lectores del tape. El coste por trade es menor de lo que sugiere el hallazgo: before_trade sólo madura cuando vence un objetivo y after_trade sólo muestrea cada h/4. Severidad baja.
  - Corrección (completitud): Severidad baja.

### F2.3b-08. La vela interna del Hurst se re-ancla en el evento de cierre y comprime los huecos en un solo retorno

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:856, :874, :904
- Causa: Se hace kline_start_ms = event_time_ms del evento que cierra (:904), en lugar de inicio + 60 000, y la condición de 60 000 o más (:856) sólo se evalúa cuando llega un evento. Un hueco de k minutos produce un retorno de k minutos que el DFA trata como uno de 1 minuto.
- Consecuencia viva: El Hurst que gobierna TP/SL, bandas y duración mezcla pasos de 1 minuto con pasos de varios minutos en monedas con huecos o tras reconexiones. No medí la magnitud; en mercados líquidos la deriva es menor del 1 %.
- Prueba que falla hoy: cf2_reloj_hurst_rejilla_fija_de_un_minuto (crates/god-engine-core/tests/cf2_hurst_bandas_solucion_conocida.rs): eventos en 0, 61 000, 120 500 y 181 000 ms. La rejilla fija da 2 retornos; predigo que hoy da 1.
- Dueño: Línea C
- Fila de §3 del plan: 6
- Escépticos: confirmado por los dos escépticos. Los dos bajan la severidad de media a baja.
  - Corrección (refutar): Bajaría la severidad de media a baja. El sesgo no está medido; con eventos cada unos 100 ms el sobrante medio es de unos 50 ms por vela (≈0,08 %). Los huecos producen un retorno atípico entre 1024. Una rejilla fija tampoco resuelve los huecos sin un modelo de datos ausentes, que el DFA no tiene. Conviene declarar además que el buffer mezcla velas REST alineadas al minuto del exchange (bootloader.rs:209) con velas internas ancladas a eventos.
  - Corrección (completitud): Bajaría la severidad a baja hasta medir la frecuencia de huecos. En mercado líquido el desfase por minuto es el intervalo entre eventos (unos 0,1 s, el 0,17 %). Además, el calentamiento REST son 1000 velas de 1 minuto (bootloader.rs:189 y :207-241; god_engine.rs:2950-2958), así que el DFA es válido desde el arranque. La duda del auditor sobre las 8,5 h de ceguera se resuelve: no ocurren con acceso REST.

### F2.3b-09. El «Hurst» multifractal está encogido hacia 0,5 y sesgado: con H = 0,7 da unos 0,62-0,64

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/multifractal.rs:113-168
- Causa: Hay tres sesgos que se suman: - el encogimiento n/(n+12), que vale 0,806 con n = 50; - la resta de la media muestral en una fGn de memoria larga, que baja var2/var1 de 2,64 a unos 2,55 y var4/var1 de 6,96 a unos 6,40; - Jensen sobre ln. Con micro (n = 10) sólo se usa h2 y el peso es 0,45.
- Consecuencia viva: Se clasifica baja porque el consumidor es un bosque, invariante a transformaciones monótonas por feature. El daño es de interpretación mientras train y serve coincidan.
- Prueba que falla hoy: cf2_vr_macro_fgn_07_dentro_de_005 (crates/feature-engine/tests/cf2_multifractal_solucion_conocida.rs): media de 400 realizaciones; predigo un macro de unos 0,62-0,64 frente a 0,7 ± 0,05.
- Dueño: Línea C
- Fila de §3 del plan: 6
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Ninguna sobre la severidad baja. Hay que precisar que el margen de fallo frente a 0,65 es de unas 2,5 a 7 SE, no amplio.
  - Corrección (completitud): El argumento de baja severidad («el consumidor es un bosque invariante a transformaciones monótonas») no cubre un consumidor vivo que el auditor no vio. hurst_macro se escribe como «hurst» en el bus mmap al cerrar (lib.rs:4023-4030) y entrena la ranura [4] del bosque sombra (online_daemon.rs:930-940). Ese bosque sirve con el Hurst DFA (online_daemon.rs:1020-1022) y publica forest6_prob, que se lee en vivo (lib.rs:6876). Lo detallo en faltantes.

### F2.3b-10. cache_generation cambia en cada consulta y la deduplicación de D0 nunca actúa

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/feature-engine/src/multifractal.rs:395-410; crates/god-engine-core/src/lib.rs:3215-3220
- Causa: Devuelve falpha_calls, que se incrementa en cada espectro_cacheada, aunque el doc (:403-406) dice que cambia sólo al refrescar, cada 16 consultas.
- Consecuencia viva: Ninguna: multifractal_d0_media y _sd (lib.rs:3221-3252) cuentan cada espectro 16 veces, pero no tienen lector.
- Prueba que falla hoy: cf2_generacion_cache_cambia_una_vez_cada_16: 32 consultas deberían dar como mucho 2 generaciones; predigo 32.
- Dueño: Línea C
- Fila de §3 del plan: 6
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Ninguna.

### F2.3b-11. La cadena Kalman → TensorRing se calcula en cada evento sin lector y con unidades mezcladas

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:716-720, :790-791
- Causa: fair_price y jerk_t no tienen lector de producción. r se pasa como amplitud en USD, no como varianza.
- Consecuencia viva: Ninguna: sólo coste.
- Prueba que falla hoy: No hay prueba de conducta que falle, porque no hay lector. La evidencia es el grep: rg 'fair_price|jerk_t' crates src sólo encuentra stateful_engine.rs y tests/stateful_transition_contract.rs:20 y :58.
- Dueño: Línea C
- Relación con hallazgos anteriores: Las unidades son R7-R2-E-6, y el cómputo sin lector entra en R7-R2-G-1.
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Detalle menor: stateful_transition_contract.rs:20 y :58 leen fair_price, no jerk_t.

### F2.3b-12. hurst_escala_continua → hurst_scale_matched se escribe y nadie lo lee

- Dominio: F2.3b, Física, fractales y señales de una serie.
- Ubicación: crates/god-engine-core/src/lib.rs:3160-3184; crates/quantum-arena/src/state.rs:221, :461
- Causa: La única escritura de producción está en lib.rs:3183, y sus lectores son tests (lib.rs:9673-9681).
- Consecuencia viva: Ninguna.
- Prueba que falla hoy: No hay prueba de conducta. La evidencia es el grep: rg 'hurst_scale_matched' crates src sólo encuentra la declaración, el constructor y la escritura.
- Dueño: Línea C
- Fila de §3 del plan: 6
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Ninguna.

### F2.4-09. Un solo empate de timestamps anula toda la matriz de contagio

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/feature-engine/src/hawkes_cross.rs:152-155; crates/god-engine-core/src/contagion_publisher.rs:54-64
- Causa: contagion_matrix exige un orden estrictamente creciente en TODAS las series; un solo w[1] <= w[0] devuelve None para la matriz entera. El publicador sale sin escribir, y el registro conserva roles viejos sin edad.
- Consecuencia viva: En vivo es improbable, porque los eventos son de depth a ≈ 250 ms. En un replay con depth y trade del mismo ts es posible, y entonces deja roles congelados.
- Prueba que falla hoy: cf2_contagio_un_empate_no_anula_la_matriz: tres series Poisson, una con un empate insertado; espera is_some(); hoy da None.
- Dueño: Qoder/GLM.
- Fila de §3 del plan: 12
- Relación con hallazgos anteriores: LC-F1-35
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): No hay que abrir id nuevo: el test B3 se adjunta como test de función pura del pedido (2) de LC-F1-35. El escenario de replay queda refutado. booktick_replay sólo procesa la moneda 0 (crates/backtest-engine/src/booktick_replay.rs:436 y :477-493), así que el publicador sale en series.len() < 2 (contagion_publisher.rs:47-49) antes de llegar a contagion_matrix. En el modo trade-only, tick_ring queda vacío. En vivo los empates siguen siendo improbables: el guardia de secuencia D-610 descarta los depth duplicados (src/bin/god_engine.rs:3159-3180).

### F2.4-10. HawkesProcessEngine (feature-engine) se escribe en cada evento y nadie lo lee

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/feature-engine/src/hawkes.rs:3-157; crates/feature-engine/src/stateful_engine.rs:913-939
- Causa: El escritor corre en stateful_engine.rs:913-920 y last_hawkes_ratio sólo lo leen tests. Además el impulso es proporcional al silencio (:122-124), lo contrario de Hawkes, y el normalizador de volumen es el literal 10 000 en lugar del gen hawkes_volume_norm.
- Consecuencia viva: Ninguna sobre las decisiones; sólo coste por evento y un gen huérfano.
- Prueba que falla hoy: Grep: rg -n "last_hawkes_ratio|update_hawkes\(|intensity_bull|intensity_bear" --type rust sólo devuelve el escritor y tests (stateful_engine.rs:1670-1690, tests/stateful_open_diagnostics.rs:14-41, tests/stateful_transition_contract.rs:28).
- Dueño: Qoder/GLM.
- Fila de §3 del plan: 11
- Relación con hallazgos anteriores: R7-R2-E-1
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Las rutas están mal: StatefulEngine no está en feature-engine. El escritor vivo es crates/god-engine-core/src/stateful_engine.rs:913-920 (llamado desde process_tick en crates/god-engine-core/src/lib.rs:3088). update_hawkes está en stateful_engine.rs:928-939, el campo en :135 y el reset en :652. Los tests están en stateful_engine.rs:1670-1690, crates/god-engine-core/tests/stateful_open_diagnostics.rs:14-41 y crates/god-engine-core/tests/stateful_transition_contract.rs:28.
  - Corrección (completitud): Re-anclar las rutas de la ficha, del hallazgo y de la retirada a crates/god-engine-core/src/stateful_engine.rs y crates/god-engine-core/tests/.

### F2.4-11. cointegration_zscore publica el momentum del líder bajo un nombre de cointegración

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/god-engine-core/src/lib.rs:5229
- Causa: set_reg("cointegration_zscore", leader_mom.clamp(−3, 3)) etiqueta el momentum de BTC/ETH como z de cointegración. Sus tres lectores (conformal_reversion_filter.rs:183-185, stat_arb.rs:415-418, vecm_arbitrage.rs:176-186) sólo lo usan como respaldo (or_else) de vecm_zscore, que se publica en la línea anterior (:5228) y en el mismo ámbito.
- Consecuencia viva: Ninguna mientras vecm_zscore se publique siempre; sí, si un cambio futuro deja de publicarlo, porque el momentum del líder pasaría a usarse como z de reversión.
- Prueba que falla hoy: Grep: rg -n "\"cointegration_zscore\"" crates src --type rust sólo da el escritor y lecturas .or_else tras "vecm_zscore". Un test de fuente que exija que ninguna clave con «cointegration» se escriba a partir de leader_mom fallaría hoy.
- Dueño: Línea C (núcleo).
- Fila de §3 del plan: 13
- Escépticos: confirmado por los dos escépticos.

### F2.4-12. Cópula t: dominio documentado (0,1) frente a [0,1] aceptado, τ-a con empates y colas simétricas

- Dominio: F2.4, Procesos puntuales, dependencia y contagio.
- Ubicación: crates/feature-engine/src/copulas.rs:27-63, :198-203, :224-229
- Causa: t_cuantil y log_copula_t documentan que devuelven None fuera de (0,1), pero aceptan 0 y 1: t_cuantil(0) devuelve ≈ −1000 por la bisección. tau_kendall es τ-a, y con retornos de 0 repetidos sesga ρ hacia 0. lambda_cola_t es simétrica U/L, cuando el riesgo de un largo vive en la cola inferior (R7-R2-E-8).
- Consecuencia viva: Ninguna directa en el dominio. El manifiesto mensual con colas simétricas alimenta inflar_cola en el guard (zona Codex); se trata en la fila 14 de C-F4.
- Prueba que falla hoy: assert!(log_copula_t(0.0, 0.5, 0.3, 4.0).is_none()) falla hoy, porque devuelve Some(valor finito). En vivo no tiene efecto, porque el manifiesto usa rango/(n+1), que nunca da 0 ni 1.
- Dueño: Línea C (copulas.rs); el lector del guard es de Codex.
- Fila de §3 del plan: 14 (C-F4)
- Relación con hallazgos anteriores: R7-R2-E-8 (la parte de colas simétricas)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (completitud): Añadir el espejo largo/corto (cola cruzada; ver los faltantes).

### F2.5-02. El LS ratio entra dos veces en la convicción del Ente, y la severidad de liquidaciones tres veces

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/god-engine-core/src/lib.rs:7695-7698 y :7744-7752; crates/metacortex-engine/src/consejo_seniors.rs:581-593, :608-613, :548, :562-566
- Causa: "prospect_pressure" no tiene escritor, así que el núcleo siempre calcula la presión de respaldo con el mismo ls_account_ratio que ya entra como crowd_ls_ratio, y con la misma liquidation_severity que ya entra como p_liq y como veto. Con α = β, la presión es δ^0,88·g(ln LS, liq), una función monótona del mismo dato, de modo que el lado que va con la masa se atenúa dos veces.
- Consecuencia viva: Hoy ninguna (inerte por F2.5-01). Pasaría a media si el asiento empieza a decidir.
- Prueba que falla hoy: cf2_ente_un_dato_de_masa_una_entrada: con LS = 2 y atr_pct = 0,002, la confianza con P de respaldo (≈ 0,881, cálculo a mano) es distinta de la que da sólo p_crowd = 1/(1+0,25·ln²2) ≈ 0,8928. Tolerancia 1e-12.
- Dueño: Línea C (lib.rs:7727-7753) y pedido con test a AGY (Prospect)
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): «duplica» no puede quedar vacío: es la semilla LC-F2-08 del plan (PLAN_LINEA_C:292 y :1062), que todavía no está publicada como hallazgo. Consecuencia viva: no es «ninguna». Por el canal del override (F2.5-01 corregido), el doble conteo baja la convicción del Ente en ≈ 1,3 % con LS = 2, y hasta un factor 0,5 en el extremo, así que puede inclinar un override en el borde. La severidad baja se mantiene.
  - Corrección (refutar): La «consecuencia viva: ninguna» es incorrecta. Por la corrección de F2.5-01, el doble conteo cambia el final_signal del override de un veto único y puede inclinarlo. La severidad baja se mantiene mientras no se mida la frecuencia de ese camino.

### F2.5-03. El impulso contrarian de Prospect [1,0; 1,30] queda anulado por el recorte de entity_factor y por el contrato [0,1] de la confianza; el test existente lo enmascara

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/metacortex-engine/src/prospect_theory.rs:180-184; crates/metacortex-engine/src/consejo_seniors.rs:615 y :1136; crates/metacortex-engine/tests/prospect_theory_contract.rs:168-204
- Causa: El producto se recorta a ≤ 1 (:615) y el consejo rechaza una confianza > 1 (:1136). El impulso sólo puede compensar otras atenuaciones. El test fija whale_burst_z = 2,0 y open_interest_norm = 0,5 para que la base quede por debajo de 1 y el impulso se vea.
- Consecuencia viva: Ninguna (inerte por F2.5-01).
- Prueba que falla hoy: cf2_ente_el_impulso_contrarian_de_prospect_llega_a_la_conviccion: con la carga útil neutra, dir = +1 y P = −4, modulation_factor = 1 + 0,3·tanh(1) = 1,2285, pero la confianza es 1,0, igual que la neutra.
- Dueño: AGY (Prospect, pedido con test) y Línea C (recorte del asiento)
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): El hallazgo duplica en parte LC-F2-08, en lo que toca al recorte. Consecuencia: el impulso sólo actúa cuando la base es < 1/1,2285 = 0,814. En ese caso puede rescatar un override de veto único por el canal de F2.5-01, así que «ninguna» no es exacto. La severidad sigue en baja.

### F2.5-04. El poller de LS y taker consulta 30 símbolos arbitrarios de todo el mercado, no el roster

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/data-pipeline/src/omni_multiplexer.rs:727-736 (con el mapa llenado en :686-704)
- Causa: roster_syms = funding_by_symbol.keys().cloned().take(30), y funding_by_symbol se llena con el premiumIndex de todo el mercado. El comentario «ya rotado por el universe manager» es falso. Para la mayoría del roster, ls_account_ratio y taker_ratio nunca existen, el lector devuelve 1,0 (lib.rs:7698, :7702) y el factor queda constante (G-TEO 1).
- Consecuencia viva: Ninguna hoy (inerte por F2.5-01). Bloquea cualquier medición de IC del LS por moneda.
- Prueba que falla hoy: Extraer `fn simbolos_de_sentimiento(claves_funding: &[String], roster: &[String]) -> Vec<String>` y escribir el test simbolos_de_sentimiento_cubren_el_roster: con 500 claves y un roster de 30 contenido en ellas, el resultado debe contener los 30. La lógica actual (take(30) de las claves) no los garantiza. Alternativa como guarda de fuentes: omni_multiplexer.rs no debe contener `keys().cloned().take(30)`.
- Dueño: Línea C (data-pipeline no figura entre las zonas ajenas)
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Hay que añadir al test un requisito de frescura: cada valor publicado debe llevar el sello del periodo de 5 min del que procede, y el lector debe descartarlo si tiene más de 2 periodos.

### F2.5-05. El «sentimiento de la multitud minorista» se alimenta con el LS de top traders

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/data-pipeline/src/omni_multiplexer.rs:739; crates/metacortex-engine/src/consejo_seniors.rs:95-98 y :1006-1008
- Causa: El endpoint topLongShortAccountRatio cubre, según la documentación pública de Binance (no verificable desde la nube), el 20 % superior de cuentas por margen. La teoría contrarian del asiento y CROWD_LS_FEAR hablan de la distribución minorista. Viola G-TEO 2 (semántica).
- Consecuencia viva: Ninguna (inerte).
- Prueba que falla hoy: Guarda de fuentes ls_de_masa_coincide_con_su_endpoint: si omni_multiplexer.rs consulta topLongShortAccountRatio, el doc de crowd_ls_ratio no puede decir «minorista». Hoy lo dice (consejo_seniors.rs:96).
- Dueño: Línea C
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.

### F2.5-06. La presión de Prospect de la masa no tiene punto de referencia: es una transformación de ln LS, en unidades arbitrarias

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/metacortex-engine/src/prospect_theory.rs:191-229 y :3-24; crates/god-engine-core/src/lib.rs:7750
- Causa: Usa p_long, una fracción de cuentas, como probabilidad de un resultado. Usa δ = ATR%·100 a la vez como ganancia y como pérdida, sin precio de referencia. Con α = β, Net = δ^0,88·g(LS, liq), así que no aporta información más allá de LS, liq y ATR. Además, tanh(0,25·P) depende de la unidad de δ, el recorte a ±50 y «P << −1» son inalcanzables con un ATR de 1 min (|P| ≲ 0,5), y el doc dice «Prelec» donde la fórmula es TK92.
- Consecuencia viva: Ninguna (inerte).
- Prueba que falla hoy: La teoría exige que dos masas con el mismo LS y distinto coste medio tengan presiones distintas. Hoy no se puede ni expresar, porque la API no recibe precio de referencia. Prueba de degeneración, que pasa hoy: compute_crowd_net_prospect_pressure(ls, 0, δ)/δ^0,88 no depende de δ para δ ∈ {0,1; 0,3; 1; 3} y LS ∈ {0,5; 2; 5} (tolerancia 1e-12).
- Dueño: AGY (pedido con test)
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): El siguiente paso propuesto tiene un error de teoría (ver faltante «K-T: precio de referencia común»). En un perpetuo, cada contrato abierto a un precio p tiene un largo y un corto a ese mismo p, así que el coste medio NO difiere por lado y no se puede «repartir por lado con el LS». Además, estimar el coste sólo con ΔOI ignora la rotación entre cuentas, que no cambia el OI y es la mayor parte del volumen.

### F2.5-07. SpoofingDetector: ventana igual a la cadencia de @depth5, no separa cancelación de ejecución ni de traslación, y su doc dice «desconectado»

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/feature-engine/src/microstructure.rs:656-664, :701-715; src/bin/god_engine.rs:998, :3205-3231
- Causa: Detecta sólo si dt < 0,25 s, y el host suscribe `@depth5` sin sufijo, cuya cadencia es 250 ms según la documentación de Binance. La detección queda a merced del jitter de E. Sólo recibe el máximo nocional de 5 niveles: un muro consumido por agresores o un muro que sale de la ventana cuando el precio sube un tick dan la misma señal que una cancelación. El score no tiene lado. El doc P-5-DECISIÓN dice «SIGUE DESCONECTADO», pero el host lo cablea desde P-6.
- Consecuencia viva: Ninguna (inerte por F2.5-01). Pasaría a media si spoof_score empieza a decidir.
- Prueba que falla hoy: cf2_spoof_cancelacion_plantada_a_la_cadencia_de_depth5 (una cancelación del 90 % a 250 ms da score 0) y cf2_spoof_nulos_estructurales_sin_cancelacion (ejecución y traslación dan 0,5 cada una).
- Dueño: Línea C
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La consecuencia viva no es «ninguna», sino casi nula. spoof_score sólo entra en el factor del Ente (p_spoof ≥ 0,8), que decide únicamente por el canal del override (F2.5-01 corregido).
  - Corrección (refutar): El siguiente paso, un modelo de cancelación, no se puede realizar con los datos actuales: exige el stream de diferencias de profundidad por nivel, y que el tape guarde más de bq/aq (LC-F1-56). Ver faltantes.

### F2.5-08. El detector de ballenas (z ≥ 2,5 sobre nocional crudo) no está calibrado y las propias ballenas lo enmascaran

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/feature-engine/src/microstructure.rs:228, :244-258; crates/feature-engine/src/welford.rs:35-56; src/bin/god_engine.rs:3329-3346
- Causa: Calcula un z gaussiano sobre una magnitud de cola pesada, con σ̂ EWMA de memoria ≈ 1000 trades que las ballenas inflan. La unidad es el print de @trade, así que un barrido de varios niveles queda repartido.
- Consecuencia viva: Ninguna (inerte).
- Prueba que falla hoy: cf2_ballena_tasa_de_falsas_alarmas_sin_ballenas: con LN(0,1) la tasa es ≈ 0,025 frente al nominal 0,0062. cf2_ballena_plantada_x50_al_uno_por_ciento: el recall esperado es 0,6–0,8, por debajo del 0,9 exigido.
- Dueño: Línea C
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): La precisión que se espera en la prueba de ballenas plantadas está subestimada. Los falsos positivos son ≈ P(Z ≥ 3,6)·0,99·198 000 ≈ 31, frente a ≈ 1 200–1 500 verdaderos positivos, lo que da una precisión de ≈ 0,95–0,98 y no de 0,7–0,95. El veredicto no cambia, porque falla por el recall.
  - Corrección (refutar): Hay dos faltas de completitud. (a) El tracker descarta el lado del agresor: el host sólo pasa el nocional (god_engine.rs:3331-3333), así que una ballena compradora y una vendedora atenúan igual un largo. (b) El estado que lee el consejo es «el ÚLTIMO trade fue ballena» (entes_consejo.rs:25-34), es decir, tiempo de eventos. El criterio recall ≥ 0,9 de la prueba no está derivado (ver pruebas).

### F2.5-10. Tests que dicen certificar Prospect «a través de GodEngineCore» sin observar el camino vivo

- Dominio: F2.5, Conducta humana y de masas.
- Ubicación: crates/god-engine-core/tests/prospect_pressure_integration_contract.rs:150-200; crates/metacortex-engine/tests/prospect_theory_contract.rs:212-216
- Causa: REV-10 ejecuta process_event, pero calcula P por fuera con el motor y nunca lee la carga útil del núcleo. El test de deliberación aprueba igual con P = 0. Son contratos formales que, según G-TEO §30.1, no cuentan como cableado vivo.
- Consecuencia viva: Cobertura falsa.
- Prueba que falla hoy: Mutante: sustituir la presión de respaldo de lib.rs:7751-7752 por 0,0. Los 4 tests de prospect_pressure_integration_contract.rs siguen en verde (predicción por lectura), así que el mutante sobrevive.
- Dueño: AGY (tests de Prospect)
- Fila de §3 del plan: 19
- Escépticos: confirmado por los dos escépticos.

### F2.6-02. feynman_coherence cuenta escalas no observadas y no filtra el ruido: en una caminata aleatoria supera el corte 0,20 con frecuencia

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/god-engine-core/src/lib.rs:2055-2062 y :2430-2441 (malla cruda, sin el gate de observabilidad que el consenso sí aplica en :2534-2537); crates/signal-engine/src/feynman_propagator.rs:181-217 y :290-297
- Causa: Con Δt ≫ τ, las escalas sub-resolución tienen α = 1 y z = dev/|dev| = ±1 exacto (crates/quantum-arena/src/temporal_spectrum.rs:524, :550, :555, :562): con ~1 evento/s son unos 13 fasores idénticos con el signo del último retorno. Las escalas de τ mayor que el tiempo transcurrido comparten z ≈ (p − p₀)/media|p − p₀|: son copias del retorno desde el arranque del proceso. C mide sobre todo si el último tick coincide en signo con el retorno desde el arranque. Contradice D-742/CL-32 («lo no observado no opina»). El corte 0,20 no tiene nulo: sólo con fases uniformes equivaldría a p ≈ e^{−6,4}.
- Consecuencia viva: En c4e8e285, C fija la transmisión del voto escalar de Feynman (feynman_propagator.rs:267-297), que reinyecta el dominante con τ inoperable (el defecto de CL-59). Bajo puro ruido esa transmisión es > 0 una gran parte del tiempo, según la lectura del código. Tras CL-59 (f0902cc7, en origin/main 8632ca3d) C queda como telemetría sin efecto en la decisión.
- Prueba que falla hoy: cf2_6_feynman_caminata_aleatoria_decohiere_como_dice_su_doc (crates/signal-engine/tests/cf2_cuantico_solucion_conocida.rs): exige P(C > 0,20) ≤ 0,05 en una caminata aleatoria; por lectura predigo una fracción del orden de 0,4-0,8 (sin medir).
- Dueño: AGY
- Fila de §3 del plan: ninguna
- Escépticos: confirmado por los dos escépticos. Los dos bajan la severidad de media a baja.
  - Corrección (refutar): Bajo la severidad a baja como hallazgo independiente. En c4e8e285, C sólo llega a una decisión a través del voto escalar de Feynman (feynman_propagator.rs:283-297). Ese voto es distinto de cero sólo con v_dom ≠ 0, y sólo pesa cuando el orquestador no está en modo espectral, es decir, con τ_dom < 30 s (orchestrator.rs:381-384): exactamente el escenario de CL-59. Así que su efecto vivo está contenido en CL-59 (f0902cc7, ya en origin/main 1d1f8bdd). Tras CL-59, C es telemetría. Matiz vivo: en un feed real, los eventos de profundidad que no mueven el mid dan dev = 0 ⇒ ewma_dev_vol = 0 ⇒ z = 0 (temporal_spectrum.rs:555-560). Esas escalas tienen entonces amplitud 0 y salen de active_count. Los 13 fasores idénticos sólo aparecen en eventos que mueven el mid, y la caminata de la prueba, donde todos lo mueven, es el peor caso.
  - Corrección (completitud): La frase «Contradice D-742/CL-32» debe quedar en «contradice CL-32 (sub-resolución)». D-742 pondera por masa en la fusión; no excluye. La parte de las escalas lentas vale también para el consenso de 13 motores. La severidad media vale en c4e8e285; en origin/main, con CL-59, pasa a baja (sólo telemetría).

### F2.6-03. Feynman: la «integral de camino» no tiene caminos ni un ħ efectivo, y la documentación describe constantes y unidades que el código no tiene

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/signal-engine/src/feynman_propagator.rs:6-36, :108-129, :195-208, :290-291
- Causa: Hay una sola fase por escala, atan2(Δz·m, z·√k) + 0,10·tanh(L/(ħ(1+z²+v²))): ħ mueve como mucho 0,1 rad. compute_quantum_phase (S/ħ) no tiene uso fuera de tests. La doc dice τ_k = 4^k μs (:14), pero la malla es 4^k ns (temporal_spectrum.rs:70-71: 1e-6 ms). El umbral «C > 0,40: interferencia constructiva» (:30) no existe en el código, que sólo tiene el corte 0,20 y una rampa lineal (:292-293). La afirmación «< 25 ns» (:35) no está medida. Lo que sí implementa es una longitud resultante media de Rayleigh, normalizada.
- Consecuencia viva: Ninguna directa: es documentación falsa sobre un motor vivo.
- Prueba que falla hoy: Ninguna prueba numérica falla. cf2_6_feynman_coherencia_es_rayleigh_con_fases_uniformes pasa y documenta lo que el código sí es (E[N·C] = 1 y una cola ≈ e^{−6,4} bajo fases uniformes).
- Dueño: AGY
- Fila de §3 del plan: ninguna
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Precisión de la descripción: con amplitudes iguales, C = |ΣA e^{iθ}|²/(N·ΣA²) es el CUADRADO de la longitud resultante media (R̄²), no R̄. El estadístico de Rayleigh es 2N·R̄² ~ χ²₂, y por eso E[N·C] = 1 en la prueba. Cita del plan cuántico: la fila «Física de Decisión … Feynman» está en docs/PLAN_MAESTRO_CUANTICO_UNIVERSO_ESPECTRAL_CONTINUO_2026-10-10.md:251, no en :249.
  - Corrección (completitud): Escribir θ_k = atan2(Δz_k·m, z_k·√k_spring) y citar el plan cuántico en :251.

### F2.6-04. Sombra de NavierStokesManifold: sin lector, ν sin escritor, «flujo de cascada» telescópico y «energía» autonormalizada que no ve la energía inyectada

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/god-engine-core/src/lib.rs:2457-2486; crates/signal-engine/src/navier_stokes_manifold.rs:46-67, :131-161, :192-195
- Causa: sombra_navier_* no tiene lector (sólo sombras_espectrales_telemetria_contract.rs:57). kinematic_viscosity no tiene escritor y ν vale siempre 1. turbulent_cascade_flux = E(31) − E(0), la escala de 146 años contra la de 1 ns. E(k) = ½z² con z autonormalizada (temporal_spectrum.rs:555-562), así que es invariante ante σ: K41 (E ∝ ε^{2/3}) no puede cumplirse, y la pendiente −5/3 no se calcula. El contrato kolmogorov_energy_spectrum_and_cascade_flux_conservation certifica la identidad telescópica como si fuera «conservación».
- Consecuencia viva: Ninguna: es telemetría sin lector. El voto escalar vivo es otro problema (F2.6-01).
- Prueba que falla hoy: cf2_6_navier_energia_espectral_crece_con_la_energia_inyectada: con σ×10 la «energía» media debería crecer ×21,5 (K41) o ×100 (½u²); por lectura la razón es ≈ 1,000. La identidad cf2_6_navier_flujo_de_cascada_es_la_diferencia_de_extremos pasa.
  - Actualización (revisión cruzada, CL-65): el ×21,5 (K41) no aplica a esta prueba: con dev crudo la razón sería 100 (½u²), y es lo que exigen la prueba y el TSV.
- Dueño: AGY
- Fila de §3 del plan: ninguna
- Relación con hallazgos anteriores: CL-58 sólo para ν (docs/audit/CLAVES_SIN_ESCRITOR_2026-10-11.tsv:8, rama de Línea C)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): (1) feature-engine/src/navier_stokes.rs:183-184 sí calcula un ν por evento (self.kinematic_viscosity = L²/(τ_relax·agresión)), pero el núcleo publica Re, laminar y disipación (lib.rs:5328-5332) y no ν. Además ν está en USD²/s y depende del nivel de precio al cuadrado, mientras que navier_stokes_manifold espera un adimensional en [1e-6, 100] (:60-61, :91). Cablearlo tal cual cambiaría el defecto «sin escritor» por uno de unidades. (2) La invariancia de E ante σ vale hasta O(σ·√n). Con los parámetros de la prueba, la desviación de la razón respecto de 1 es del orden de 1e-2, no ≲ 1e-3. Por lectura, sin medir.

### F2.6-05. QuantumTensorStore no tiene consumidor y su «Lyapunov» no es un exponente de Lyapunov

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/feature-engine/src/quantum_tensor_store.rs:60-76; crates/feature-engine/src/lib.rs:17, :39
- Causa: calculate_lyapunov_chaos = RMS(tf0 − tf1) entre dos vectores de features. Es ≥ 0 por construcción, así que no detecta estabilidad (λ < 0), y escala con la unidad de los datos (un λ real es invariante ante x ↦ a·x). Nada lo llama.
- Consecuencia viva: Ninguna: no tiene consumidor.
- Prueba que falla hoy: cf2_6_lyapunov_de_un_punto_fijo_estable_es_negativo (logístico con r = 2,5: λ = ln 0,5 y el código da 0) y cf2_6_lyapunov_no_depende_de_la_unidad_de_medida (logístico con r = 4: el código da ×1000 al multiplicar la serie por 1000).
- Dueño: Línea C (feature-engine, fuera de zonas ajenas); retirada
- Fila de §3 del plan: ninguna
- Relación con hallazgos anteriores: F2-C8 (docs/BARRIDO_EXHAUSTIVO_FASES.md:300) y R7-R2-G-1 D-3 (:2613)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): No es un hallazgo nuevo: duplica F2-C8 y R7-R2-G-1 D-3, como el propio auditor declara. La prueba de r = 2,5 debe reformularse (ver pruebas).

### F2.6-06. SimdNeuralNet se construye y reinicia en cada StatefulEngine pero nunca se evalúa ni se entrena

- Dominio: F2.6, «Cuántico» y analogías sin consumidor.
- Ubicación: crates/god-engine-core/src/stateful_engine.rs:132-133, :258, :651, :1314-1334; crates/feature-engine/src/simd_neural_network.rs:24-34, :188-240
- Causa: infer_simd_alpha y train_simd_step no tienen llamadores. La inicialización «He/Xavier ortogonal» es sin(·)·c. infer_quantized_i8 trunca en vez de redondear y recorta la entrada a ±10, frente a ±100 en infer. El cálculo en coma flotante y el gradiente son correctos por lectura.
- Consecuencia viva: Ninguna en la decisión. Es memoria muerta por moneda (594 f64) y un reset en cada StatefulEngine::reset.
- Prueba que falla hoy: Ninguna numérica: las dos pruebas de solución conocida (bit a bit y gradiente) pasan por lectura. El hallazgo es de consumo y lo demuestra el grep.
- Dueño: Línea C (feature-engine y stateful_engine.rs, que no es la reserva de Línea E en lib.rs); retirada
- Fila de §3 del plan: ninguna
- Relación con hallazgos anteriores: F2-C9 (docs/BARRIDO_EXHAUSTIVO_FASES.md:301) y R7-R0-7 (CENSUS_CODE_MUERTO_RONDA7_2026-10-09.tsv:129-130)
- Escépticos: confirmado por los dos escépticos.
  - Corrección (refutar): Duplica F2-C9 (BARRIDO:301) y R7-R0-7 (BARRIDO:1971; censo, líneas 129-130). No es un hallazgo nuevo.


## Pruebas de solución conocida

Resultado preliminar, al traspaso a la máquina del operador (2026-10-11 ~13:50 UTC). Cada prueba se escribió, compiló y ejecutó una vez en la nube (x86_64-pc-windows-gnu bajo wine, nightly-2026-06-30). La revisión cruzada de las pruebas quedó a medias: empezó en risk-engine y god-engine-core y no terminó. Falta repetirla en todos los crates y correr todas las pruebas juntas antes del PR.

Resultado por prueba (crate, hallazgo, predicción, observado, estado, valores y notas): `docs/audit/PRUEBAS_C-F2_2026-10-11.tsv`.

| Crate | Activas (pasan) | OPEN (fallan como se predijo, `#[ignore]`) | Descartadas |
|---|---|---|---|
| risk-engine | 13 | 12 | 1 |
| god-engine-core | 8 | 18 | 1 |
| quantum-arena | 8 | 10 | 0 |
| feature-engine | 24 | 20 | 3 |
| signal-engine | 7 | 7 | 0 |
| metacortex-engine y strategy-core | 7 | 3 | 0 |
| **Total** | **67** | **70** | **5** |

Una prueba OPEN lleva `#[ignore = "OPEN F2.x-nn (dueño): causa"]`, como los contratos OPEN de Codex. Corre con `cargo test -- --ignored` y falla mientras el defecto siga. Al arreglarlo, su dueño quita el `#[ignore]` en el mismo commit. Una prueba descartada es una predicción que el código refutó (el hallazgo no se sostiene tal como se escribió), o una prueba que no se pudo escribir sin tocar código de producción. En las 32 pruebas donde lo observado difiere de lo predicho, la columna de notas del TSV dice qué se hizo.

### Cierre en el PC del operador (2026-10-11, CL-65)

Revisión cruzada y ejecución completa en la máquina del operador (Windows nativo, x86_64-pc-windows-msvc, nightly-2026-06-30, `CARGO_TARGET_DIR` propio). La revisión la hicieron tres revisores de sólo lectura, que rehicieron a mano las soluciones conocidas de las 137 pruebas contra el código de producción y los hallazgos.

**Lo que encontró la revisión cruzada.**

- Un archivo entero no compilaba en el HEAD que llegó de la nube: `cf2_calibracion_solucion_conocida.rs` leía `cobertura_conformal` como una tupla, y devuelve un struct. Sus 8 pruebas no se habían ejecutado con esa versión; la aserción `err_fijo ≈ 20/201` no se había ejecutado nunca. Corregido; las 8 dan lo predicho.
- Cuatro pruebas activas fijaban el defecto en vez de la solución y se habrían puesto rojas con el arreglo. Quedan rotuladas CARACTERIZACIÓN, con la instrucción de retirarlas o reescribirlas en el mismo commit que active su OPEN:
  - `cf2_dfa_ruido_blanco_sesgo_de_forma_cerrada`;
  - `cf2_re_forma_cerrada_10ms_y_100ms`;
  - `cf2_ym_caracterizacion_retornos_asincronos` (antes `cf2_ym_retornos_asincronos_son_inertes`, que no era un nulo);
  - `cf2_conformal_riesgo_entre_aceptadas_valor_analitico`.
- Se añadió `cf2_ym_triangulo_exacto_sin_dislocacion_es_nulo`, el nulo verdadero de Yang-Mills que sobrevive al arreglo.
- Pruebas OPEN que no se activan solas con el arreglo del defecto. Ahora lo dice su `#[ignore]`:
  - CONTRATO DE API (la firma actual no lleva el dato): espectro irregular, nulos estructurales del spoofing y VPIN vivo;
  - CONTRATO DEL SUCESOR o DEL ASIENTO: Platt fuera del soporte, conformal con selección, cascada de Hodge y LS contado dos veces;
  - RÉPLICA LOCAL, que se traslada a evolution-engine con el arreglo: el vigilante de Ville;
  - requisito de precisión, no se cierra con el sesgo: el DFA por realización.
- Pruebas frágiles o vacuas corregidas:
  - el Lyapunov de un punto fijo era irresoluble con r = 2,5 y pasa a r = 2,9;
  - la ballena plantada usa ahora la EE pareada: 7,4 EE, antes 4,7 frente a un límite de 4;
  - el libro sin cancelaciones usa μ = 2,5 y exige que el precio se mueva;
  - el Lundberg con exposición pasa a 80 000 caminatas con dos comprobaciones deterministas, y se retira una aserción vacua;
  - la cobertura de σ̂ toma la referencia del desarrollo de 2Φ(1,645·e^u) − 1 con los momentos medidos;
  - la τ dominante acepta los dos tamaños de familia;
  - el lead-lag a cadencia viva usa la cota derivada 0,89;
  - el paseo aleatorio del DFA tiene un suelo de 0,01 para el residuo de Jensen conocido;
  - la cópula exige además la λ publicada λ(ρ̂, ν̂);
  - el Ente inerte sólo exige el signo en las cargas aprobadas.
- Hallazgo nuevo F2.3-08b (baja): con un control previo del observable, W1 no ve un cambio de volatilidad ×5 en ninguno de los 16 caminos. La media tras el cambio es ≈ 0,91, frente a ≈ 0,94 en la ventana nula: el transporte espectral es invariante a la escala. Las «detecciones» del BOCPD que se dieron en la nube eran falsas alarmas.
- F2.2-03b: la sub-causa FA-1 (la cota del PF mirada en cada cierre no es anytime-valid) recibe id.
- F2.2-10 recibe una segunda causa, la doble deflación del umbral del DSR, que es la que mide su prueba.
- F2.1-06: su predicción (tamaño ≈ 0,17) queda refutada; da 0,0185.

**Resultado final de las pruebas cf2** (`cargo test -p <crate> --test <archivo> -- --include-ignored`). Hay 138 pruebas: las 137 de la nube más el nulo nuevo de Yang-Mills.

| Crate | Activas (pasan) | OPEN (fallan por su causa) | Coste (pasa) | Descartadas |
|---|---|---|---|---|
| risk-engine | 13 | 12 | 0 | 1 |
| god-engine-core | 8 | 18 | 0 | 1 |
| quantum-arena | 8 | 10 | 0 | 0 |
| feature-engine | 24 | 19 | 1 | 3 |
| signal-engine | 7 | 7 | 0 | 0 |
| metacortex-engine y strategy-core | 8 | 3 | 0 | 0 |
| **Total** | **68** | **69** | **1** | **5** |

Las 69 OPEN fallan y las 68 activas pasan, sin ninguna discrepancia. Además leí el mensaje de cada fallo, y las 69 caen en la aserción del hallazgo que citan, no en una precondición ni en un control. La de coste (`cf2_copula_gaussiana_no_fabrica_cola`) pasa con `--include-ignored` y su estado en el TSV pasa a `ignorado_coste`.

**Suites completas de los siete crates** (`cargo test -p feature-engine -p god-engine-core -p metacortex-engine -p quantum-arena -p risk-engine -p signal-engine -p strategy-core --tests --no-fail-fast`): 168 binarios, 1 632 pasan, 0 fallan y 74 ignoradas (las 69 OPEN, la de coste y 4 anteriores a C-F2).

La columna `pc_cl65` del TSV da el resultado en el PC de cada prueba, y la columna de notas, lo que cambió la revisión cruzada.
