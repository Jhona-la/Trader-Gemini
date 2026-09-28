# MEMORIA DEL PROYECTO — Trader Gemini (estado vivo)

> Memoria compartida entre sesiones de agente (Claude, Gemini, freebuff…).
> Las REGLAS viven en `.agents/AGENTS.md`; aquí va el ESTADO: qué está
> integrado, qué decisiones siguen vigentes y qué queda abierto.
> Actualízala al cerrar cada ola (un bloque por fecha, lo nuevo arriba).
> Detalle forense completo: `INFORME_FORENSE_MAESTRO.md` (ADENDA 22 = lo último).

---

## 2026-09-28 — Claude (cloud): auditoría de gates de evolución y régimen (PR #8)

> Estado: el primer tramo (XLIV-1…7) entró en `main` con 41755422 (merge
> local de GLM). El segundo tramo (XLIV-8…12) sigue en el PR #8.

Canal: PR #8 de GitHub (la sesión cloud no ve el buzón no versionado
`COORDINACION_CODEX_2026-09-28.md`). Sin push directo a `main` (permiso
bloqueado): todo entra por el PR. **No toco** correlation_guard /
random_matrix (Codex) ni Hawkes / flow_excitation (Qoder).

### Corregido (commits atómicos, con tests)
- **XLIV-1 Fisher**: el gate WF-FISHER exigía F > 1,0 y la Fisher de escala
  está acotada por 2/(ln 4)² ≈ 1,04 ⇒ la evolución en vivo NUNCA promovía con
  ≥ 2 monedas. Umbral derivado: masa en ≤ ½ banda operativa (F ≥ ≈ 0,33).
- **XLIV-2 BOCPD**: medias de segmento divididas por el posterior normalizado
  (encogían), densidades sin 1/σ, doble observación por evento en vivo (rompía
  la paridad) y W₁ ausente inventado como 0. p_transition en reposo ruidoso:
  0,40 → < 0,05.
- **XLIV-3 EWMAs de CVD/marea/entropía**: sin corrección de sesgo, «calientes»
  tras UN evento (z ≈ ±22 en cada arranque). Ahora momentos corregidos y 30
  eventos efectivos; rama larga 1 = espejo de la corta. Golden re-certificado.
- **XLIV-4 crash_pressure** (orquestador): sólo cuenta monedas con marea
  BAJISTA (antes una subida fuerte en cualquier moneda recortaba el margen de
  todos los largos).
- **XLIV-5 calibrador con olvido**: `calibrate_at`/`update_at`, semivida 12 h
  (borde lento de la banda). El EV con p calibrada (D-751) era ABSORBENTE: sin
  operar no hay datos que corrijan una p hundida.
- **XLIV-6 funciones de estructura**: referencia browniana (ζ₃ = 1,5), no K41;
  χ = ((3/2)ζ₂ − ζ₃)⁺ (concavidad, independiente de H); se excluyen escalas por
  debajo de la resolución efectiva (intervalo medio entre eventos).
- **XLIV-7 sonda de arranque frío**: una sola sonda abierta por moneda sin
  modelo (antes podía llenar los 3 slots y repetirse tras cada reinicio).
  Golden re-certificado: 1 trade de sonda (antes 2).
- Verificación: arena + núcleo + riesgo 559/559; backtest, evolution, signal y
  metacortex verdes. **T-1 (oráculo, release bajo wine, 30 min): 19/144 genes
  sensibles (13,2 %) ≥ trinquete 11,5 % — PASA.** Comparación gen a gen con
  `main` (988f0478, mismo método): main 20/144; la rama pierde SÓLO el gen 19
  (`min_confidence_btc`) y no gana ni pierde ningún otro. Hipótesis: en el
  fixture las intenciones que sobreviven llevan confianza 0,98 y el umbral en
  su extremo (0,95) ya no cambia nada; en vivo el gen sigue gobernando la
  puerta de confianza. Sin bisecar (≈ 30 min por prueba) no se atribuye a un
  commit concreto.
- Diagnóstico T-1 (fixture sintético): el risk-engine rechaza por
  `suelo_tp_sl` 1 241 740 intenciones (167 161 largos, 1 074 579 cortos) frente
  a 1 605 por comisiones. El suelo de TP/SL domina el embudo; con spread
  sintético de 4 pb frente a fricción de 7 pb es esperable, pero hay que
  medirlo sobre tape real antes de juzgarlo.
- Aislamiento de tests: una vez se vio un fallo de determinismo de
  `replay_*` con tests en paralelo (no reproducido en 13 corridas). Probable
  estado global compartido (mapa global de bosques / registros).

### Defecto propio encontrado por Codex (XLIV-3) — arreglo reservado por Codex
- `momentos_ewma_corregidos` supone momentos sembrados en 0, pero `CoinArena`
  siembra `entropy_mean_ewma`/`entropy_sq_ewma` en 1,0 (state.rs). Con
  entropía constante 0,5, a t = 31: media 9,77 y varianza < 0 (sd = 0). CVD y
  marea no afectados (nacen en 0). Codex tiene en local: semillas de entropía
  en 0, dominio cerrado (peso ∈ (0,1], segundo momento ≥ 0, finito),
  calentamiento con log1p/expm1 y reloj W₁ monótono. **XLIV-3 ya entró en
  `main` (41755422) SIN ese arreglo**: el defecto está vivo hasta que Codex
  publique su parche. Claude no toca esa zona.

### Segundo tramo (XLIV-8 … XLIV-12, con 8b y 9b/9c)
- **XLIV-8 fricción única**: `tp_sl::roundtrip_friction` (2·taker + 2·(piso +
  latencia DIFUSIVA), tope 5 %/lado) para el gate, los brackets del host
  (`genome_protection_prices`, fallback de entrada) y el pre-examen del daemon.
  Los tres últimos seguían con la ley lineal `atr·lat/umbral_pánico` (3,46×
  más por lado con umbral 500 ms y 50 ms). En el gate es idéntico bit a bit.
- **XLIV-9 freno del bosque**: el registro de acierto usaba
  `subio = (is_long == is_win)` con PnL NETO; el bosque se entrena con triple
  barrera sobre el mid BRUTO. Un largo que subía menos que las comisiones
  contaba como bajada y el voto contrario puntuaba como acierto (el freno se
  armaba por la fricción). Ahora `direccion_realizada(is_long, pnl_pct)`.
- **XLIV-10 suelos de confianza tras las puertas**: fusión constructiva
  `(max + 0,1·min).clamp(0,55, 0,96)` fabricaba 0,55 con dos ramas de
  desventaja demostrada (0,40/0,40); modulación espectral `.clamp(0,45, 0,98)`
  deshacía el freno del bosque (0,70 → 0,35 → 0,45). Ahora
  `confianza_superpuesta` (refuerzo sólo si ambas > ½) y `confianza_modulada`
  (sin suelo). Mismo principio que D-642.
- **XLIV-9c (revisión de Codex)**: el signo terminal del mid tampoco es la
  etiqueta del bosque (primer toque TP/SL del LARGO, timeouts descartados).
  `etiqueta_barrera(is_long, reason_code)`: largo TP ⇒ 1, largo SL ⇒ 0,
  corto TP ⇒ 0; corto SL y demás salidas ⇒ sin dato. XLIV-8b: doc de
  `roundtrip_friction` precisada (ATR/latencia no finitos ⇒ latencia 0);
  la paridad de ENTRADAS gate/host/daemon sigue abierta.
- **XLIV-9b ensamble**: el mismo defecto que XLIV-9 en `ModelEnsemble::
  update_with_trade_outcome` (y = 1 si `is_long == is_win` neto): los pesos
  Hedge castigaban al modelo que acertaba «sube» en un largo marginal. Ahora
  recibe `subio` de `direccion_realizada`.
- **XLIV-11 epigenética neta**: sesgo epigenético (tamaño) y modificador del
  umbral de confianza aprendían del movimiento BRUTO del mid; un cierre que
  ganaba menos que las comisiones BAJABA la exigencia y SUBÍA el tamaño tras
  un perdedor neto. Ahora `retorno_neto_pct(net_trade_pnl, entry·qty)`.
  Regla resultante: lo que predice DIRECCIÓN aprende de SU etiqueta
  (bosque: barrera TP/SL; ensamble: signo del mid); lo que gobierna
  RENTABILIDAD (epigenética, ramas D-752, espectro) aprende del neto.
- **XLIV-12 test del replay**: `replay_con_envolvente_sigue_determinista`
  fallaba «0 vs 1» porque el replay no registra specs y el registro global
  lo llenaba otro test en paralelo entre las dos corridas (visto 2 veces).
  `asegurar_spec_nativo` (extraído de B3.20, sin cambio) fija el entorno.
- Verificación: risk-engine 197/197 (con las 4 suites de Codex tras traer
  `main` 6b7c6d37), evolution-engine, god-engine-core 126/126 y golden del
  backtest verdes; el golden no cambia con ninguno de XLIV-8…12.
  T-1 (release, 30 min) sobre d6e5aba1 (XLIV-8/9/10): 19/144 = 13,2 %, la
  MISMA lista de genes inertes que el primer tramo. Head completo: en curso.
- **Lección de build**: tras medir T-1 sobre `main` en el mismo directorio, el
  artefacto release de quantum-arena quedó OBSOLETO con fecha más nueva que
  las fuentes restauradas (carrera checkout ↔ compilación): la siguiente
  build release falló con «no field cvd_ewma_peso». Tras cambiar de commit
  para medir, `touch` a los .rs que difieren (o `cargo clean -p`) antes de
  volver a compilar. La build debug no estaba afectada.

### Hallazgos de este tramo NO corregidos (diseño o zona ajena)
- **Tope micro del stop a 55 pb** (risk-engine, `micro_w_alloc > 0,5`): acota
  el SL por presupuesto de ruina en dólares en vez de RECHAZAR lo que no cabe
  (contradice «el único techo del SL es de capital, lo aplica quien
  dimensiona» y D-750). Con SL dentro del ruido, la deriva de la señal tiene
  menos tiempo para cobrar. Política, no bug numérico: decisión del dueño.
- **Ramas 13 y 15**: confianzas con suelo literal (tanh(…).clamp(0,55, 0,95) y
  0,58 + …); no pasan por `conviccion_de_rama` (D-752). Rediseño pendiente.
- **Atribución en la fusión**: `volume_flow_rate = max(etiquetas)` atribuye el
  resultado a la rama de índice mayor, no a la que aportó la convicción.
- **D-746 freno de apalancamiento**: mide el ATR contra `sl_at_tau` genómico a
  la τ de `operating_tau_ms`, no contra el stop real del gate
  (`tpsl_gate.sl_pct`) ni su τ: exige reordenar el gate (goldens).
- **`suelo_tp_sl`**: el veto es coherente (σ(τ)·k < f/0,65 ⇒ la τ propuesta no
  paga la fricción); el desperdicio está AGUAS ARRIBA: el núcleo propone τ por
  debajo de la banda operable. Candidato: que el generador consulte
  `min_tradeable_tau_ms` con el ATR vivo antes de emitir.
- **Genoma (GLM)**: al insertar `normalize_sl_curve_friction_floor` entre el
  doc de `min_viable_sl` y su `fn`, `min_viable_sl` perdió su doc y su
  `#[inline]` (se los quedó la función nueva). Cosmético + inline cross-crate.
- **Tamaño fuera del espacio de riesgo (meta de crecimiento)**: Kelly entra
  DOS veces — como fracción de MARGEN (`raw_exposure = dir·conf·kelly·capital`)
  y, bajo raíz, en el factor de apalancamiento (`leverage_matrix`,
  `capital_factor = 1 + √(kelly·√L_max)`). Lo que se arriesga al stop,
  `f·conf·L·SL`, nunca se deriva: con f = 0,10, conf 0,7, L = 5 y SL = 0,5 %
  es ≈ 0,18 % del capital por operación, frente a un Kelly de la geometría
  (p = 0,45, TP 1,125 %, SL 0,5 %, fricción 12 pb) de ≈ 11 %. Con 13 USD el
  nocional mínimo (5 USD) fija además el riesgo en ≈ 5·SL. Crecer como pide
  la meta exige apostar cerca de Kelly EN RIESGO AL STOP, y eso sólo es
  seguro con edge medido fuera de muestra, que hoy no existe. Propuesta:
  dimensionar en espacio de riesgo (riesgo = fracción de Kelly acotada por
  la ruina; nocional = riesgo/SL; margen = nocional/L) y dejar al
  apalancamiento como pura consecuencia del margen disponible.

### Señalado a los dueños (no tocado)
- Codex: HY normaliza por la varianza completa con cruce sólo en la ventana
  común; MP guarda pares sin medida como 0 y los declara ruido; `tope/8` con
  riesgo sin medir admite 7 posiciones de la misma apuesta.
- Qoder: check «misma banda» muerto (`diff_ln < 0.80` detrás de
  `find_resonant_slot`, que ya lo descarta): apilar en la misma dirección ya no
  exige ir ganando ≥ 28 pb. ¿Intencionado (D9)? `partition_income` (FMT-285)
  sólo se llama desde tests.
- Término de aceleración de `crash_flux` muerto (se llama con prev = None);
  cablearlo por evento sin suavizar metería ruido (la deriva satura con un
  salto de τ* en ms). Pendiente de diseño.

### Viabilidad de la meta (consejo de seniors — números, no opinión)
- +100 % cada 3 días = crecimiento log 0,231/día = ×4,2·10³⁶ al año. Con Kelly
  pleno y sin fricción exige Sharpe DIARIO 0,68 (≈ 13 anualizado); con ½ Kelly,
  ≈ 15; con ¼ Kelly, ≈ 20. Los mejores fondos del mundo operan en 2–6.
- Capacidad: de 13 USD a 10⁶ USD en 49 días y a 10⁹ en 79 al ritmo pedido; el
  impacto de mercado lo frena mucho antes. Kelly pleno cae ≥ 50 % alguna vez
  con probabilidad 1/2.
- Meta operativa honesta: maximizar crecimiento log con tope de ruina y
  medirlo fuera de muestra (EV > 0 estable en las dos mitades del forense),
  no perseguir el ×2/3 días.

## 2026-09-28 — Qoder (auditoría): Ola 10 / #582 — gen de Hawkes a la superficie viva

- Supervivencia verificada tras las olas: M6-H02 intacto (drain de bracket
  closes incondicional en god_engine.rs:3364, antes del veto en :3426) y
  núcleo #535 intacto (μ̂ empírico + STEADY_STATE_RATIO + bounds [0.50,0.95]).
- Hallazgo: el renombre U-ERR-1 (8afcc677) dejó el gate vivo del motor de
  confluencia en `hawkes >= 1.2` absoluto (BAJO el estado estacionario 1.6)
  y el gen `hawkes_scalp_threshold` huérfano. Mea culpa Ola 8: mi mapeo
  #552 estaba en `should_trigger_micro_scalp` — código muerto (0 llamadores,
  verificado por git grep).
- Fix (sin commit): lib.rs publica `hawkes_excitation_gene` al registro;
  `evaluate_for_coin` consume con umbral = STEADY_STATE_RATIO + (gen−0.50)·2;
  test nuevo. Verificado: signal-engine 59/59, god-engine-core 114/114.
- `obi_zscore_threshold` también quedó huérfano tras U-ERR-1 — decisión de
  diseño pendiente (publicar OBI-z real o retirar el gen).
- Detalle: FORENSIC_INTELLIGENCE_AUDIT.md Ola 10 / #582 (append).

## 2026-09-28 — Codex: contratos espectrales y coordinación (en curso)

- Usuario confirma trabajo paralelo de Claude y GLM, cada uno en su editor.
  Canal compartido: `COORDINACION_CODEX_2026-09-28.md`. Cada agente añade su
  alcance/confirmación; no hay acuse de recibo de los otros dos todavía.
- Trabajo sobre `main`, base observada `dc87cf1d`; XLIV tiene cambios sin
  commit en correlation_guard, risk-engine/lib, god-engine-core/lib y
  flow_excitation_confluence. Codex NO sobrescribe esos cambios.
- Codex reserva `random_matrix.rs`, `spectral_matrix_contract.rs` y
  `correlation_open_contracts.rs`: 9/13 contratos del solver fallaron antes;
  13/13 pasan tras Jacobi + dominio finito/simétrico/diagonal unidad/PSD.
- AVISO al consumidor XLIV: desconocido→cero, matriz estrella incompleta,
  T=capacidad512, AllNoise→independencia y direcciones opuestas descartadas
  siguen abiertos. El solver corregido NO certifica esa integración.
- Cinco contratos OPEN reproducidos en rojo (marcados ignore explícito):
  HY con tiempos duplicados; normalización fuera de ventana común; media
  con pares desconocidos; rho=-0.4 con k=5; overflow de Pearson.
- No operar/desplegar ni prometer duplicación cada 3 días con esta evidencia.
  Regresión amplia e informe detallado en preparación.

### Cierre verificado de este tramo (2026-09-28; adenda al estado inicial)

- Informe: `docs/AUDITORIA_CONTRATOS_ESPECTRALES_2026-09-28.md` (15 fichas,
  2 reparaciones numéricas acotadas, 13 OPEN/limitaciones); artefacto JSON
  homónimo en docs/artifacts. Atlas e informe maestro ampliados sin borrar.
- Risk-engine --all-targets: 165 pasan, 5 ignoradas OPEN; las cinco se
  ejecutaron aparte y fallan. Workspace --all-targets: cargo check OK con
  warnings. Nuevas pruebas únicas: 19. No equivale a certificar el sistema.
- Qoder confirmó recepción y autoría de god-engine-core/lib +
  flow_excitation; se corrige la atribución provisional a XLIV de arriba.
  Su diff fue leído y recibió feedback de unidades/aislamiento por símbolo.
  La segunda sesión sigue sin confirmar; consumidor de riesgo preservado.
- Solver y contratos tienen hashes registrados. Se comprobó preservación
  del prefijo textual normalizado de los informes ampliados y del buzón.
- Artefacto XXXIX ausente reconstruido como resumen histórico explícito,
  NO como reejecución actual ni recuperación de hashes/logs originales.
- Inventario actual: 1.300 versionados, 396 Rust, 24 manifiestos; no se
  traslada automáticamente la cobertura histórica 169/289. Auditoría parcial.
- Sigue main, sin commit/push/merge/fetch ni actividad operativa. La meta
  +100% cada 72h NO queda validada. Prioridad: reparar procedencia y
  contrato de riesgo del consumidor antes de acreditar diversificación.

## 2026-09-25 — Integración ola «espectro predictivo» (PR #4) + F-009/WS de main

### Ramas
- `main` (1fa9a914) llevaba 3 commits que la rama de auditoría no tenía:
  F-009 / D-411 (desasfixia ML), M2-C02 / M1-M02 / WS (Hawkes por símbolo,
  OI en USD, timeout WS adaptativo) y streams WS en minúsculas.
- `claude/decima-ola-auditoria-2` (PR #4) llevaba 13 commits
  (D-742 … D-758). Se unificaron las dos en `claude/elegant-euler-mmtht4`
  (PR #5, auditado y fusionado a `main`; el #4 se cerró como sustituido).
  Una sesión que siga sobre `claude/decima-ola-auditoria-2` debe traer
  `main` antes de continuar.
- Ramas muertas (no integrar): `claude/decima-ola-auditoria-forense`
  (155 detrás), `subagent-*` (121 detrás, de junio).

### Qué quedó integrado (vigente)
- **Espectro predictivo** (`quantum_arena::spectral_tape`,
  `SpectralForecastBank`): puntuación prequencial; volatilidad (R² OOS
  +0,05…+0,18), volumen (+0,25…+0,50) e intensidad (+0,36…+0,64) SÍ son
  predecibles y ganan a la persistencia; flujo de dinero y concentración de
  entes NO (sólo observación).
- **D-742**: el espectro temporal no opina con escalas τ que no ha observado.
- **D-743 `puertas_del_continuo`**: UNA sola función con las cuatro puertas
  (viabilidad, invariante bayesiano D-472, ponderación ML F-009, vetos
  CVD/muro L2) aplicada a `fast_intent` Y `slow_intent`. **Cualquier cambio
  a esas puertas va AHÍ**, nunca inline en una de las dos bandas.
- **F-009 / D-411 (desde main, portado a la puerta)**: dirección ML medida
  contra `ml_model_base` (≈0,30 con etiquetado honesto), veto sólo si
  contradice < −0,80 sin acción de precio extrema, penalización blanda
  `(1 + d·0,5).clamp(0,20, 1)`. Scaler de DarkAlpha: z-scores en [−3, 3].
- **D-744**: cortacircuitos de drawdown con una sola semántica medida.
- **D-745 / U-ERR-5**: una posición, un horizonte. `PositionHorizon` sólo
  tiene `Continuous`; el horizonte REAL es `Position::entry_tau_ms` (τ con la
  que se dimensionó, `ValidatedOrder::tau_ms`). **No reintroducir
  Swing/Scalping** (F-009 de main lo hacía; se revirtió en el merge).
- **D-746**: la volatilidad no es convicción (freno ATR vs distancia al stop).
- **D-747**: flujo de entrenamiento des-espejado (`qty = |bq − aq|`,
  `is_buyer_maker = aq > bq`) en entrenador, exportador, sonda y forense.
- **D-749**: veto de muro compara en el espacio del gen (razón entre muros).
- **D-751/751b/752**: spread de los tapes corregido (`tape_spread_fix`), tick
  inferido del propio tape, el medidor cobra el spread UNA vez.
- **D-752/D-756**: convicción por rama = cota de Wilson de su propio
  historial (`TasaAcierto`, `conviccion_de_rama`), sin literales 0,72/0,68;
  `exigencia_tras_racha` monótona y saturada en 1.
- **D-753 + paridad**: el entrenador (`train_forest`) reproduce el tape por
  `GodEngineCore::process_event` y verifica paridad 48D con tolerancia CERO
  contra `build_54d_tensor`; aborta si difiere. Slippage por latencia
  unificado a la función de difusión pura.
- **D-754**: `compute_tp_sl` usa σ pronosticada (`CoinArena::sigma_forecast_at`)
  sólo si el pronóstico tiene habilidad positiva; si no, idéntico a antes.
- **Riesgo**: Kelly micro sin piso ficticio (se RECHAZA lo que no cabe en el
  tope de ruina), `min_notional` publicado del símbolo, guard de correlación
  = Pearson real, profit factor con cota de Jeffreys.
- **M5-H01**: `fitness_compute(initial, final, dd, trades)` — <30 cierres ⇒
  INVIABLE; dd clamp [0,1]; Darwin arranca `best_all_time` en −∞.
- **CERT-M2-C02**: Hawkes por símbolo (`hawkes_by_coin`), λ/μ real al registry.
- **CERT-M1-M02**: OI normalizado en notional USD (escala log $100M…$10B).
- **WS**: streams en minúsculas; watchdog `BINANCE_WS_TIMEOUT_SECS`
  (defecto 15 s mainnet / 30 s testnet).

### Auditoría del PR #5 (2026-09-25) — corregido antes de fusionar
- **D-744b**: sin `riesgo_por_operacion` medido (arranque del proceso) los
  DOS cortacircuitos de drawdown quedaban desarmados. Ahora
  `drawdown::drawdown_maximo` cae al gen como fracción de caída.
- **D-744c**: la EWMA del riesgo tomado se registra tras el último rechazo.
- **D-754b**: el σ pronosticado vuelve a ceros si la habilidad cae, y sólo
  se publican anclas puntuadas (`MUESTRAS_MADURAS` = 30).
- **D-754c**: el pronóstico σ entra al stop ×√(8/π) (`RANGO_PARKINSON`):
  el respaldo ATR es un rango y el gen del SL está en esa escala.
- **D-750b**: el guard de correlación compara r CON signo (misma dirección
  + anticorrelación = cobertura, no la misma apuesta).

### Hallazgos de la auditoría NO corregidos (diseño — siguiente ola)
1. **EV gate con p calibrada (D-751) puede ser un estado absorbente**: el
   calibrador sólo aprende de lo ejecutado; si baja p, el EV veta todo y p
   no se recupera (lo que D-690 evitó en el gate de confianza). Vigilar en
   el forense si una moneda deja de operar tras ~20 cierres. Remedio
   candidato: calibrar con resultados contrafactuales de intenciones
   vetadas (auditor de trayectorias) o una cota superior de p como piso.
2. **Convicción por rama (D-752) se re-infla aguas abajo** (boost ML ×1,5,
   fusión `max`, persistencia ×1,3): una rama perdedora demostrada puede
   superar `min_confidence_btc`. Y las ramas 11–14 (tensor, impulso,
   tendencia lenta, consenso) no pasan por `conviccion_de_rama`.
3. **VPIN**: el lado lo decide la regla del tick sobre el mid (no el
   `is_buyer_maker` real); y los llamadores legacy de `process_tick`
   (DarwinDaemon opt-in, bin `evolver`) nunca alimentan volumen ⇒ VPIN
   congelado ahí.
4. **Deslizamiento por latencia (D-753)** unificado sólo en gate y física;
   brackets del host (`genome_protection_prices`, fallback de entrada) y el
   pre-screen del daemon siguen con la ley lineal (~3× distinta).
5. **Freno de apalancamiento (D-746)** mide contra el τ del gen y la curva
   de SL genómica, no contra la τ medida ni el stop real (D-745/745b).
6. **Anti-whiplash (D-757)**: sus ventanas de tiempo son código muerto
   (el enfriamiento de `viable_para_entrar` ya las cubre).
7. **Bosque (freno)**: la exactitud direccional se puntúa con PnL NETO
   (`is_win`); un acierto menor que las comisiones cuenta como fallo.
8. Guard de correlación: dirección OPUESTA + anticorrelación también es la
   misma apuesta y el llamador la descarta (preexistente).
9. Proceso: los commits 6c06ee3d y 8cb5504b mezclan varios bloques
   (incumplen «commits atómicos»); no se reescribe historia.

### Abierto / pendiente
- **Nada autoriza a operar**: el motor sigue perdiendo en las tres
  configuraciones medidas (ADENDA 22.7). Falta el forense sobre los tapes
  regenerados con D-751/D-752 y el modelo reentrenado con paridad D-753.
- Modelos se llaman `{SÍMBOLO}_MOTOR` (+ `_VOL` para el predictor P-1).
  BTCUSDT NO tiene `_MOTOR` ⇒ la puerta de roster veta todas sus entradas;
  el forense acepta `FORENSIC_SYMBOL=ATOMUSDT` (u otro con modelo).
- Erradicación U-ERR incompleta: examen walk-forward del daemon, etiquetas y
  puertas del entrenador (ver commit 8afcc677).
- Predictores P-1/P-2: su gate compara R² contra la media, no contra la
  persistencia ⇒ un R² de 0,106 no demuestra habilidad.

### Cómo compilar / probar desde Linux (sesiones cloud)
- `os-guardian` es sólo-Windows (`windows::Win32`) y casi todo depende de él:
  `cargo check` nativo en Linux FALLA en `main` también. No es regresión.
- Chequeo completo: `rustup target add x86_64-pc-windows-gnu`, instalar
  `gcc-mingw-w64-x86-64`, luego
  `cargo check --target x86_64-pc-windows-gnu --workspace --all-targets`.
- Tests: instalar `wine64` y exportar
  `CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUNNER=/usr/lib/wine/wine64 WINEDEBUG=-all`,
  luego `cargo test --target x86_64-pc-windows-gnu -p <crate> --lib`.
- DEMO desde la nube: la política de red del entorno bloquea (403) todos los
  hosts de Binance (`testnet.binancefuture.com`, `stream.binancefuture.com`,
  `fapi.binance.com`, `data.binance.vision`) y no hay claves en el entorno;
  los tapes no están en el repo (viven en el PC del operador). Para operar
  demo en una sesión cloud hay que permitir esos hosts en la configuración
  de red del entorno y definir `USE_TESTNET=true` +
  `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_SECRET_KEY` como variables
  del entorno. Si no, la demo se corre en el PC local.

### Lección de este merge
- Git sólo marcó 4 hunks en `god-engine-core/src/lib.rs`, pero hubo dos
  choques SEMÁNTICOS que auto-mergearon "limpios": `pos_h` Swing/Scalping
  (variantes ya borradas por U-ERR-5) y la ponderación F-009 que main
  editó en el bloque inline que la otra rama había movido a
  `puertas_del_continuo`. Tras cada merge entre sesiones: diff del resultado
  contra CADA padre y compilar con `--all-targets` antes de commitear.

## 2026-09-28 — Codex: reauditoría Git y contratos numéricos (segundo corte)

- Sigue main=origin/main=dc87cf1d tras fetch --prune origin. GitHub: PR #7
  OPEN, head cffff2fd; #6 CLOSED; #5 MERGED. RECTIFICACIÓN del bloque histórico:
  GitHub registra #4 MERGED, no simplemente cerrado/sustituido. No borrar la
  rama elegant-euler: contiene trabajo posterior al merge del PR #5.
- backup-before-cleanup (3 commits exclusivos) y v7-unificacion-wip (1)
  no están integradas ni son patch-equivalent. No se borró ninguna rama.
  Árbol sucio compartido != cambios publicados aunque ahead/behind sea 0/0.
- Nuevos cambios Codex SOLO en funciones preexistentes de correlation_guard:
  validez de ticks, mid, Pearson, HY y fallback. Bloques XLIV y risk/lib se
  preservan. Nueve RED→GREEN más caso de constante n=49 RED→GREEN. Nuevos
  tests: correlation_numeric_contract (10); tres contratos anteriores de
  correlation_open_contracts dejan de ser ignored. Risk-engine all-targets:
  178 pasan, 2 ignoradas OPEN; las dos OPEN ejecutadas siguen RED (008/009).
- Workspace check --offline --workspace --all-targets pasó con advertencias.
  No certifica exposición, independencia, ausencia de bugs ni rentabilidad.
- Genoma editado por otra sesión durante esta ronda: Codex no lo modifica.
  Diagnóstico seed199 inicialmente RED (sin banda); tras edición posterior
  el barrido 20.000 semillas PASA, pero quantum-arena --lib da 79 pasan/1
  falla en D-658: -6.400000000000001 != -6.4 con assert_eq. Revisar contrato
  de reconstrucción/no-op y solape semántico con PR #7, no apilar reparadores.
  Se notificaron ambos cortes por COORDINACION_CODEX_2026-09-28.md.
- Informe: docs/AUDITORIA_INTEGRACION_Y_CONTRATOS_NUMERICOS_2026-09-28.md;
  artefacto: docs/artifacts/auditoria_integracion_numerica_2026-09-28.json.
  Atlas, maestro e informe espectral ampliados; evidencia anterior preservada.
  SPECTRAL-005 parcial; 006/011 reparación local. Consumidor, inferencia MP,
  tamaño efectivo, signos/slots y varianza negativa continúan pendientes.
- Sin commit, push, merge ni staging por Codex; integración/publicación
  consultada al usuario, sin respuesta todavía en este corte. Sin operación,
  promoción, entrenamiento ni despliegue. Inventario no es cobertura total.

## 2026-09-28 — Codex: admisión multiactivo, contratos y publicación concurrente

- Reparación del consumidor D-748: dependency_exposure recorre todos los activos
  y slots, transforma rho por los DOS signos de exposición y mantiene Unknown
  como caso adverso. No usa matriz estrella ni AllNoise para borrar el conteo.
  Se estiman pares una vez por activo; snapshot de posición NO es reserva global.
- SPECTRAL-008: rho_promedio exige matriz completa finita/simétrica/unitaria/PSD;
  009: rho<-1/(k-1) o inválido no recibe crédito por varianza negativa. Los
  dos contratos ignorados ahora están activos. Dos expectativas XLIV erróneas
  se rectifican sin borrar sus testigos; se añaden ejemplos negativos válidos.
- EWMA de pérdida al stop NO es sigma: el consumidor pasa None, sin descuento
  MP/rho. Las APIs se conservan. Frustración de signos NO es Hodge. Falta riesgo
  monetario real ponderado por posición/tau/stop y reserva atómica de cartera.
  tope/8 y umbral binario son deudas legacy, no garantías científicas.
- Risk-engine --all-targets: 195 pasan/0 fallan/0 ignoradas. 15 tests nuevos de
  admisión; 7 RED→GREEN iniciales, 4 casos adicionales sin afirmar RED previo.
  Cuatro suites locales nuevas suman 44 tests (13+10+6+15).
- Quantum-arena --all-targets: 165 pasan/0 fallan/0 ignoradas tras cambio del
  otro editor; 79/80 anterior queda histórico. Codex NO editó esos genomas.
  Workspace check all-targets pasó con advertencias; no es test de todo el
  workspace ni prueba de rentabilidad, latencia operativa o auditoría integral.
- Git cambió DURANTE la ola: GLM reconoció crear/publicar 988f0478 en main con
  arrastre de trabajos, incluyendo solver y correcciones Codex. Fetch/ls-remote
  lo confirman. En ese corte faltan las cuatro suites nuevas y los informes.
  Mensaje del commit dice varianza/Hodge viva, pero SU árbol D-748 ya usa None.
- PR #7 sigue OPEN, ahora CONFLICTING. merge-tree inspeccionado sin iniciar
  merge: conflictos en genome_store (20k semillas frente a rejilla 6561) y
  genome_gate_open_diagnostics (testigo seed199 equivalente). Predicado de
  banda vacía auto-mergea pero no está en violated de main; revisar composición.
  Preservar ambos barridos; no asumir superseded sólo porque seed199 pasa.
- GLM confirmó recepción y prepara validación/publicación; Codex le dejó
  precisión de estados y manifest. No iniciar segundo merge/commit concurrente.
  Backups tienen 3/1 commits exclusivos; no hay otra rama probadamente integrada.
  Codex no hizo staging/commit/push/merge ni borró ramas en esta intervención.
- Informe nuevo: docs/AUDITORIA_ADMISION_MULTIACTIVO_2026-09-28.md; JSON homónimo
  en docs/artifacts. Atlas, maestro e informes previos ampliados por adenda.
  No operación, promoción, entrenamiento, despliegue ni garantía +100%/72 h.

- CORTE REMOTO POSTERIOR 10:57: PR #7 CLOSED, NO merged (mergedAt=null).
  La rama elegant-euler avanzó a ebdf2389 y tiene 8 commits fuera de main,
  nuevo trabajo BOCPD/calibración (6 archivos +360/-68), no auditado funcionalmente
  en esta ola. Main remoto sigue 988f0478; las 44 regresiones/documentos siguen
  pendientes en el último status. No borrar rama por PR cerrado. GLM avisado.
