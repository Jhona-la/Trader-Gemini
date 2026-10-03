# MEMORIA DEL PROYECTO — Trader Gemini (estado vivo)

## 2026-10-03 — Qoder: Ola 49 / #649 — KERNEL HAWKES TRANSVERSAL — ORÁCULO PASA 16/144

- Rama qoder/ola49-kernel-hawkes (worktree .ola49, base ae867860),
  código 65c2978a. Los 3 motores con excitación (hawkes_bessel,
  flow_impulse, flow_excitation_confluence) rediseñados con física
  honesta: excitación = exceso GLOBAL λ/μ̂ sobre SS (moneda de la casa),
  per-escala en x(τ_k); kernel e^{−β·τ_s} muerto en [30s,12h]
  ELIMINADO junto a la mezcla dimensional, el doble conteo, las mallas
  hardcodeadas y el escalón de umbral. Ratio fresco por moneda
  reemplaza 3 knobs muertos. ABSTENCIÓN total en régimen normal (antes
  votaban ±0.9 constante).
- **ORÁCULO T-1: PASA 16/144 = 11.1%** (2756s). Verificación: signal
  106/106 (tests reescritos), core 160/160, ws 0 err.
- Detalle: FORENSIC #649. Buzón: cierre Ola 49.

## 2026-10-02 — Qoder: Ola 48 / #648 — INTEGRIDAD DEL CONSENSO ESPECTRAL — ORÁCULO PASA 16/144

- Rama qoder/ola48-integridad-consenso (worktree .ola48, base 8246c134),
  código 3695380e. H5 (umbral skill anclado a N efectivo 128), H1 (gate
  observabilidad D-742 en composición — escalas sub-resolución a peso 0),
  H3 (TTL del dominante: caduca sin depth por >max(30s,τ)), H6 (score de
  maduración al cierre en todo evento + re-arme en depth con dedup
  propio). Telemetría qo_648_esc_excluidas / qo_648_ttl_expirado_ts.
- Verificación: signal-engine 105/105 (4 nuevos), core 160/160, ws 0
  errores. **ORÁCULO T-1: PASA 16/144 = 11.1%** (4098 s) — push inmediato.
- Detalle: FORENSIC #648. Buzón: entrada Ola 48.

## 2026-10-02 — Qoder: Ola 47 / #626 — PESOS POR HABILIDAD en la composición espectral (13 motores)

- Flujo: rama qoder/ola47-pesos-habilidad (worktree .ola47) → SkillMotores
  + consenso_por_escala + reconciliación con AGY 13/13 (merge 939c6dbc) →
  **ORÁCULO COMBINADO (AGY P29/P30 + #625 + #626): PASA 16/144 = 11.1%**
  (2485s) → docs + push a main. Certifica también la cobertura del merge
  AGY que llegó sin oráculo.
- **#626**: composición ponderada por IC prequential por motor×escala
  (maquinaria #594, voto de ARMADO causal, madurez 30, significancia
  #599, piso 0.15). En frío ≡ pesos iguales — la ponderación sólo entra
  con evidencia. Telemetría qo_626_maduros/_peso_max.
- Verificación: signal-engine 101/101, core 160/160, ws 0 errores.
- **AUDITORÍA SISTEMÁTICA 3 auditores** (~25 hallazgos en memoria de
  sesión): kernel Hawkes MUERTO en banda operable (3 motores, Ola 49),
  solitón invertido, salto sin signo, mea culpa #613 (qo_613_rho_tau sin
  escritor), H5 en #626 (umbral n crudo — ABIERTO, Ola 48 con H1 gate
  observabilidad + H3 TTL + H6 maduración fuera de is_depth).
- Detalle forense: FORENSIC_INTELLIGENCE_AUDIT.md #626. Buzón: cierre
  Ola 47.

## 2026-10-02 — Antigravity: Revisión Base Cuántica — Consenso Espectral Integral (13/13 Motores) + Símplex Continuo de Régimen (Modo Profesor)

- Flujo coordinado: rama `antigravity/revision-base-cuantica` → verificación unitaria (`quantum-arena` 101/101 tests OK, `signal-engine` 96/96 tests OK, `god-engine-core` 158/158 tests OK, `risk-engine` 119/119 tests OK, `evolution-engine` 60/60 tests OK) y workspace (`cargo check` 0 errores) → commit y sincronización.
- AGY-AUD-P29 (CRITICAL): `crates/god-engine-core/src/lib.rs`:
  - **QUÉ**: Consenso espectral unificado e integral incorporando la totalidad de los 13 motores cuánticos del sistema y erradicación de constantes duras.
  - **POR QUÉ**: El consenso sombra previo solo incluía 11 motores (`pesos = [1.0; 11]`), omitiendo `HighPayoffTrendRunner` y `RenyiTsallisEntropyEngine`. Además, utilizaba parámetros literales fijos (`1.0, 0.1, 0.5`, etc.) en lugar de leer los parámetros dinámicos evolucionados del registro omnisciente, duplicando cálculos que ya se realizaban para las sombras individuales.
  - **PARA QUÉ**: Garantizar completitud y consistencia matemática total en la síntesis espectral multiescala (32 escalas temporales), unificando las decisiones de todos los subsistemas cuánticos.
  - **CÓMO**: Se estructuró la evaluación individual de sombras en secuencia una sola vez por tick con parámetros dinámicos del registro, publicando la telemetría `sombra_*` (incluyendo la nueva telemetría `sombra_entropia_*`), y alimentando el array `votos_espectrales` de 13 posiciones con `pesos = [1.0; 13]`.
  - **CUÁNDO**: En cada tick de mercado (`process_tick_dual`).
  - **DÓNDE**: `crates/god-engine-core/src/lib.rs:1880-2025`.
  - **QUIÉN**: `GodEngineCore::process_tick_dual`.
- AGY-AUD-P30 (HIGH): `crates/quantum-arena/src/state.rs` y `crates/god-engine-core/src/lib.rs`:
  - **QUÉ**: Símplex continuo de régimen de mercado $[p_{\text{range}}, p_{\text{bull}}, p_{\text{crash}}, p_{\text{chaos}}] \in \Delta^3$ con funciones de activación suaves $C^\infty$.
  - **POR QUÉ**: El régimen global se colapsaba mediante cortes rígidos de escalón en $z_{\text{btc}} = \pm 1.96$ y Hurst fijo en números enteros discretos (0, 1, 2, 3), causando discontinuidades abruptas en las derivadas de modulación de margen y colchones de riesgo.
  - **PARA QUÉ**: Asegurar continuidad en el universo espectral temporal continuo, permitiendo transiciones probabilísticas suaves sin saltos impulsivos en la gestión de capital y control de riesgo de la cuenta de $13 USD.
  - **CÓMO**: Se agregaron los campos atómicos `regime_p_range`, `regime_p_bull`, `regime_p_crash`, `regime_p_chaos` a `GlobalArena`. Se calcularon las activaciones suaves $C^\infty$ mediante sigmoides con exponentes acotados $[-50, 50]$, publicándose atómicamente y actualizando el `market_regime` MAP para compatibilidad hacia atrás.
  - **CUÁNDO**: Al procesar la microestructura de BTC en cada tick.
  - **DÓNDE**: `crates/quantum-arena/src/state.rs:522, 613` y `crates/god-engine-core/src/lib.rs:2155-2195`.
- AGY-AUD-P31 (HIGH): `crates/risk-engine/src/orchestrator.rs` y `crates/risk-engine/tests/portfolio_admission_contract.rs`:
  - **QUÉ**: Conexión del símplex continuo de régimen de mercado $[p_{\text{range}}, p_{\text{bull}}, p_{\text{crash}}, p_{\text{chaos}}] \in \Delta^3$ al cálculo del colchón direccional de riesgo (`directional_pressure`) y veto sistémico en `PortfolioOrchestrator::allow_trade`.
  - **POR QUÉ**: El cálculo previo de `directional_pressure` solo leía `spectral_crash_flux` por moneda individual y omitía la probabilidad sistémica macro $p_{\text{crash}}$ del mercado, dejando desprotegida a la cuenta de $13 USD si una moneda no había actualizado su flujo por baja cadencia de ticks. Además, para posiciones cortas no existía protección continua frente a *short squeezes* durante rallies sistémicos ($p_{\text{bull}} \to 1.0$).
  - **PARA QUÉ**: Asegurar que el margen admisible para la cuenta micro de $13 USD se contraiga suavemente y de forma $C^\infty$ ante estrés sistémico, evitando saltos de escalón de apalancamiento, llamadas de margen y colapsos de capital.
  - **CÓMO**:
    - Para largos: `directional_pressure = 0.25 * crash_max.max(systemic_crash)`, donde `systemic_crash = self.arena.regime_p_crash.load(Ordering::Relaxed).clamp(0.0, 1.0)`.
    - Para cortos: `directional_pressure = 0.25 * squeeze_max.max(systemic_bull)`, donde `systemic_bull = self.arena.regime_p_bull.load(Ordering::Relaxed).clamp(0.0, 1.0)`.
    - Veto sistémico continuo: `let systemic_crash_veto = { let p = self.arena.regime_p_crash.load(Ordering::Relaxed); p.is_finite() && p >= 0.90 }; if (regime == MarketRegime::Crash || systemic_crash_veto) && intent_is_long { return false; }`.
    - Test de contrato: `continuous_regime_simplex_contracts_margin_smoothly` en `portfolio_admission_contract.rs`.
  - **CUÁNDO**: En cada evaluación de admisión de orden en `PortfolioOrchestrator::allow_trade`.
  - **DÓNDE**: `crates/risk-engine/src/orchestrator.rs:150-188`.
  - **QUIÉN**: `PortfolioOrchestrator::allow_trade`.
- AGY-AUD-P32 (CRITICAL): `crates/execution-engine/src/entry_dispatch.rs`, `crates/execution-engine/src/executor.rs`, `crates/execution-engine/tests/entry_route_contract.rs` y `src/bin/god_engine.rs`:
  - **QUÉ**: Ruteo de órdenes de entrada activas protegido por deslizamiento dinámico mediante `EntryRoute::Ioc` (Immediate-Or-Cancel con precio límite adaptativo) y cierre de arista muerta de ejecución.
  - **POR QUÉ**: Anteriormente, el 100% de las entradas en `god_engine.rs` se despachaban como órdenes `EntryRoute::Market` incondicionales (`type=MARKET`). En libros delgados, desbalances súbitos de liquidez o mechas de alta volatilidad, las órdenes a mercado agresivas sufrían deslizamientos descontrolados (50 a 200 bps), lo cual en una micro-cuenta de $13 USD destruye de 2% a 4% del capital únicamente en el costo de entrada antes de que empiece a operar el trade. Además, `QuantumOrderRouter::route_order` contenía lógica de ruteo IOC que permanecía como arista muerta sin conectar con `god_engine.rs`.
  - **PARA QUÉ**: Garantizar ejecución instantánea como agresor (taker fill) cuando el libro de órdenes es saludable, pero con un techo de precio estricto que aborta/cancela de inmediato (`IOC`) si el deslizamiento excede la tolerancia admisible derivada de la volatilidad instantánea (ATR) y el piso genético evolucionado, blindando el capital micro de $13 USD contra absorciones predatorias y mechas de liquidación.
  - **CÓMO**:
    - Extensión de `EntryRoute` con la variante `EntryRoute::Ioc { price: f64 }` en `crates/execution-engine/src/entry_dispatch.rs`.
    - Validación matemática en `EntryRequest::validate`: verificación de que `price > 0.0`, finito, y `tick_size > 0.0` y finito para rutas IOC.
    - Despacho en `OrderExecutor::submit_entry` invocando `self.execute_ioc_order(...)` con `timeInForce: IOC` y redondeo direccional al tick size exacto del símbolo (`round_price_to_tick(price, tick_size, !is_long)`).
    - En `src/bin/god_engine.rs`, derivación de la cota de deslizamiento dinámico en nanosegundos:
      `dyn_slip = (base_slip + entry_atr_pct * 0.20).clamp(0.0005, 0.0035)`
      `ioc_price = OrderExecutor::round_price_to_tick(raw_ioc_price, dyn_tick_size, !final_is_long)`
    - 12/12 contratos de ejecución aprobados en `crates/execution-engine/tests/entry_route_contract.rs`.
  - **CUÁNDO**: En cada evaluación y despacho de orden de entrada en `src/bin/god_engine.rs`.
  - **DÓNDE**: `crates/execution-engine/src/entry_dispatch.rs`, `crates/execution-engine/src/executor.rs`, `crates/execution-engine/tests/entry_route_contract.rs`, `src/bin/god_engine.rs:4095-4105, 4195-4230`.
  - **QUIÉN**: `OrderExecutor`, `dispatch_entry`, `god_engine.rs`.

## 2026-10-02 — Qoder: Ola 46 / #625 — λ/μ̂ REAL al slot Hawkes del PPO

- Flujo: rama `qoder/ola46-ppo-hawkes-real` (worktree `.ola46`, base
  02903ee1) → código 726a99d2 → **ORÁCULO T-1 PREVIO AL MERGE: PASA
  16/144 = 11.1%** (2688s, --nocapture) → docs + push a main.
- **#625**: el slot Hawkes del `ppo_state` lleva `excitacion_hawkes_
  norm(hawkes_ratio_real)·dir_flow_sign` — exceso REAL λ/μ̂ sobre
  STEADY_STATE_RATIO saturado con tanh (0 = régimen normal = abstención,
  ±1 cascada/calma), misma semántica de exceso que #582/#617. Cierra la
  decisión de consejo abierta desde #554.
- **HALLAZGO — paridad evaluate/update**: el camino de cierre del PPO
  (`ppo_close_features`) usaba VPIN en el MISMO slot 2 — el peso 2
  aprendía de una variable distinta de la que vota. Cableado también
  ahí con λ/μ̂ al ts del cierre. REGLA: tocar un slot del ppo_state
  exige tocar su homólogo en el update en el mismo commit.
- Helper puro `excitacion_hawkes_norm` + 2 tests. Verificación: core
  160/160, check workspace all-targets limpio (7m31s).
- Detalle forense: `FORENSIC_INTELLIGENCE_AUDIT.md` #625. Buzón: cierre
  Ola 46.
- **EN VUELO (apilada sobre ésta)**: Ola 47 / #626 — composición
  espectral ponderada por IC prequential por motor×escala (piso 0.15,
  significancia #599, voto muestreado al ARMARSE el bloque — causal).
  Rama `qoder/ola47-pesos-habilidad`, worktree `.ola47`.
- Lección operativa: comando background con `cd` relativo cuando el CWD
  ya estaba en el worktree = falso fallo silencioso (el oráculo nunca
  corrió). Rutas ABSOLUTAS en todo comando background.

## 2026-10-02 — Qoder: Ola 45 / #624 — INTEGRACIÓN: el orquestador consume el consenso espectral

- Flujo: rama `qoder/ola45-integracion` (worktree `.ola45`) → commit
  76c9db00 → ORÁCULO T-1 PREVIO AL MERGE (obligatorio: cambio de
  pipeline vivo) → **PASA** → merge docs-only de main (GLM LXXVI, buzón
  +38, sin código) → push a main → limpieza de worktree.
- **#624 — el cierre del ciclo espectral**: `evaluate_continuous_
  consensus_for_coin` lee `consenso_espectral_dominante`/`_tau` (#623)
  ANTES de la guardia de peso activo. Dominante ≠0 finito ⇒ dirección y
  convicción = `v_dom·(0.70+0.30·convicción_ensamble)`, y la posición
  vive a `consenso_espectral_tau` clamp [30 s, 12 h] (el router deriva
  la geometría TP/SL de esa τ: `router.rs:74`). Sin dominante:
  fallback escalar BIT A BIT (D-754). Ensamble escalar abstenido ya no
  calla al espectro. Anti-staleness: el core publica dominante 0.0
  explícito cuando `dominante()` es None (espectro plano) — el valor
  del registro es SIEMPRE el del tick en curso.
- Telemetría: `qo_624_decisiones_espectrales` + `qo_624_fraccion_
  espectral` (adopción vs total, cadencia 1024 junto al censo #611).
- Verificación: signal-engine 96/96 (6 contratos qo_624 nuevos: valores
  exactos 0.8·0.97=0.776, short simétrico, override sobre escalar,
  ensamble abstenido, arranque frío bit a bit ante ausente/0.0/NaN, τ
  clamp banda), god-engine-core 158/158, single_consensus_contract 4/4,
  workspace limpio.
- **ORÁCULO T-1**: PASA (exit 0, 3323.66 s, release, --test-threads=1,
  sobre 76c9db00). PASA ⇒ ≥16/144 (≥11.1%): ningún gen certificado de
  main perdió sensibilidad con la integración. Cifra exacta fuera de la
  ventana de captura (tail-60); próxima re-cert con --nocapture la
  recupera.
- Detalle forense: `FORENSIC_INTELLIGENCE_AUDIT.md` #624. Buzón: entrada
  Ola 45.
- La cadena espectral completa está VIVA: #609..#624 — sustrato → 13
  motores → sombra → integración. Siguientes candidatos del consejo:
  λ/μ̂→PPO (oráculo propio), curvatura de Ricci (zona Codex), pisos de
  ramas 13/15 (zona CL).

## 2026-10-02 — Antigravity: Auditoría Base Espectral — Walk-Forward Continuo + TrendRunner Espectral (Modo Profesor)

- Flujo coordinado: rama `antigravity/auditoria-base-espectral` → verificación unitaria completa (`signal-engine` 85/85 tests OK, `evolution-engine` 60/60 tests OK, `god-engine-core` 158/158 tests OK) y workspace (`cargo check --workspace --all-targets` 0 errores) → merge a main → push remoto → limpieza de rama.
- AGY-AUD-P26 (CRITICAL): `crates/evolution-engine/src/online_daemon.rs`:
  - **QUÉ**: Erradicación de la evaluación degenerada de anclas escalares discretas (`scalp_tp_base`, `scalp_sl_base`, `scalp_kelly_fraction`) en la simulación walk-forward del pre-screen evolutivo.
  - **POR QUÉ**: El pre-screen mutaba las curvas continuas de horizonte (`tp_horizon_curve`, `sl_horizon_curve`, `kelly_horizon_curve`), pero evaluaba a los 2 000 mutantes candidatos contra los retornos reales de mercado usando las anclas escalares legacy `candidate.scalp_tp_base` y `candidate.scalp_sl_base` (ancladas en 15 s) y `candidate.scalp_kelly_fraction`. El juez real del host (`GodEngineCore`) opera sobre las curvas continuas $a + b \ln(\tau)$. Esto creaba un desacoplamiento donde el pre-screen seleccionaba mutantes basados en parámetros que diferían de la evaluación continua real.
  - **PARA QUÉ**: Asegurar coherencia matemática estricta y paridad idéntica entre la pre-selección de genomas y la evaluación viva del motor cuántico a la escala exacta de las barras del examen.
  - **CÓMO**: Se evaluaron las curvas continuas a la escala exacta de la barra del walk-forward (`let tau_bar = WF_BAR_MS as f64` = 16 000 ms): `let tp = candidate.tp_at_tau(tau_bar).max(0.0001);`, `let sl = candidate.sl_at_tau(tau_bar).max(0.0001);`, y `let kelly = candidate.kelly_at_tau(tau_bar).clamp(0.05, 0.50);`. El PnL y la evolución del capital simulado ahora escalan proporcionalmente a `kelly` continuo en lugar del escalar truncado.
  - **CUÁNDO**: En cada ciclo de pre-filtrado del demonio de evolución online (`sample_realized_returns` / `walk_forward`).
  - **DÓNDE**: `crates/evolution-engine/src/online_daemon.rs:1525, 1605`.
  - **QUIÉN**: `OnlineEvolutionDaemon::evaluate_candidate_prescreen`.
- AGY-AUD-P27 (HIGH): `crates/signal-engine/src/trend_runner.rs`:
  - **QUÉ**: Implementación de `voto_espectral` multiescala (32 escalas temporales) en `HighPayoffTrendRunner`.
  - **POR QUÉ**: El motor de expansión de tendencia solo emitía un voto escalar $v \in [-1, 1]$, impidiendo que el consenso espectral evalúe qué bandas temporales $\tau_k$ contienen inercia direccional persistente frente al ruido microestructural.
  - **PARA QUÉ**: Permitir que el consenso tensorial espectral determine la escala de máxima convicción ($\tau^*$) y proyecte la expansión de recorrido sin colapsar las 32 dimensiones a un único escalar plano.
  - **CÓMO**: Se implementó `HighPayoffTrendRunner::voto_espectral(desplazamientos, hurst, vpin, atr_pct)` utilizando `VotoEspectral::desde_espectro`. Cada escala $\tau_k$ se evalúa con su desplazamiento $x(\tau_k)$, ponderada suavemente por el exceso de Hurst $(H - 0.50)$ y penalizada por toxicidad de flujo VPIN. Test unitario `qo_espectral_trend_runner_escala_antisimetrica` verifica antisimetría estricta, acotación en $[-1, 1]$ y abstención total cuando $H \le 0.50$.
  - **CUÁNDO**: Al procesar la sombra espectral y el consenso tensorial en cada tick.
  - **DÓNDE**: `crates/signal-engine/src/trend_runner.rs:85`.
  - **QUIÉN**: `HighPayoffTrendRunner::voto_espectral`.
- AGY-AUD-P28 (HIGH): `crates/god-engine-core/src/lib.rs`:
  - **QUÉ**: Publicación observacional de sombras espectrales en el registro omnisciente para `StochasticResonanceEngine`, `CoaxialBreakoutEngine` y `HighPayoffTrendRunner`.
  - **POR QUÉ**: Las implementaciones espectrales de los motores 4/13, 6/13 y 7/13 no tenían sombras observacionales publicadas en el core vivo, impidiendo la trazabilidad del comportamiento espectral en tiempo real.
  - **PARA QUÉ**: Permitir la auditoría de consenso espectral sombra en vivo antes de la migración del orquestador, garantizando cero impacto en el PnL operativo actual (voto vivo bit a bit, T-1 cero).
  - **CÓMO**: Se añadieron las publicaciones de `sombra_res_*`, `sombra_coax_*` y `sombra_trend_*` (tau máxima, amplitud máxima y consenso de banda) en `process_tick_dual`.
  - **CUÁNDO**: En cada actualización de datos de mercado por tick para cada activo en `GodEngineCore::process_tick_dual`.
  - **DÓNDE**: `crates/god-engine-core/src/lib.rs:1880-1965`.
  - **QUIÉN**: `GodEngineCore::process_tick_dual`.

## 2026-09-30 — Antigravity: Ola 7 / Continuidad C^∞ en Señales Cuánticas, Arbitraje Espectral y Desacoplamiento de Skew en Maker (Modo Profesor)

- Flujo coordinado: rama `antigravity/ola7-continuidad-en-senales-y-arbitraje-espectral` → verificación unitaria (`signal-engine` 73/73 tests OK, `strategy-core` 24/24 tests OK + `basket_state_contract` 7/7 OK) y workspace (`cargo check --workspace --all-targets` 0 errores) → merge a main → limpieza de rama.
- AGY-AUD-P21 (CRITICAL): `crates/strategy-core/src/vecm_arbitrage.rs`:
  - **QUÉ**: Erradicación de la discontinuidad escalón en la evaluación del Z-Score del motor de cointegración vectorial Johansen VECM.
  - **POR QUÉ**: La lógica anterior contenía una compuerta discontinua `if z.abs() >= 1.5 { (-z / 3.0).clamp(-1.0, 1.0) } else { 0.0 }`. En $z = 1.499$, la señal devolvía `0.0`; en $z = 1.501$, la señal saltaba abruptamente a $-0.50$. Ese acantilado del 50% causaba temblores (chattering) e inestabilidad extrema en las señales cuando el spread oscilaba alrededor del umbral crítico.
  - **PARA QUÉ**: Mantener la estricta continuidad matemática del universo continuo espectral, donde las fuerzas de reversión a la media nacen suavemente desde 0 en el umbral y crecen de manera proporcional a la significación estadística observada.
  - **CÓMO**: Se implementó una función continua de exceso `let excess = abs_z - 1.5; let smooth_scale = (excess / 1.5).min(1.0); (-z.signum() * smooth_scale).clamp(-1.0, 1.0)`. En $|z| = 1.5$ la salida es exactamente 0.0; a medida que $|z|$ crece hacia 3.0, la escala se amplía continuamente hasta $\pm 1.0$, eliminando el salto de escalón sin romper la semántica de reversión a la media ni los contratos de regresión.
  - **CUÁNDO**: En cada evaluación del consenso o estrategia VECM al consultar `JohansenVecmEngine::evaluate()`.
  - **DÓNDE**: `crates/strategy-core/src/vecm_arbitrage.rs:162`.
  - **QUIÉN**: `JohansenVecmEngine::evaluate`.
- AGY-AUD-P22 (HIGH): `crates/signal-engine/src/trend_runner.rs`:
  - **QUÉ**: Modulación $C^\infty$ continua en la persistencia de tendencia del exponente de Hurst y dirección en `HighPayoffTrendRunner`.
  - **POR QUÉ**: La activación dependía del corte rígido `if safe_hurst <= 0.52 || safe_dir.abs() < 1e-4 { return 0.0; }`. Con $H = 0.5199$, la salida era `0.0`; con $H = 0.5201$, la salida saltaba bruscamente a $\approx 0.197$ (un salto del ~20% en convicción de señal). Además, `safe_dir.signum()` introducía discontinuidad escalón de $\pm 1$ alrededor de cero.
  - **PARA QUÉ**: Asegurar derivabilidad y suavidad espectral continua, de modo que regímenes cercanos a la caminata aleatoria pura ($H \approx 0.50$) tengan convicción cero y aumenten suavemente sin generar choques bruscos en el consenso tensorial.
  - **CÓMO**: Se modeló el exceso de persistencia como `let h_excess = (safe_hurst - 0.50).max(0.0); let h_weight = (h_excess / 0.04).tanh();` y la dirección como `let dir_weight = (safe_dir / 1e-3).tanh(); dir_weight * h_weight * (tp * 10.0).tanh()`. En $H \le 0.50$, la señal es 0.0; entre 0.50 y 0.54 sube suavemente sin escalón. Verificado con el nuevo test unitario `test_trend_runner_continuous_hurst_transition`.
  - **CUÁNDO**: Al evaluar señales de expansión de tendencia en `HighPayoffTrendRunner::evaluate_for_coin`.
  - **DÓNDE**: `crates/signal-engine/src/trend_runner.rs:180`.
  - **QUIÉN**: `HighPayoffTrendRunner::evaluate_for_coin`.
- AGY-AUD-P23 (HIGH): `crates/signal-engine/src/coaxial_breakout.rs`:
  - **QUÉ**: Corrección física del fallback de difusión browniana en escalas multitemporales (1s, 5s, 1m) y continuidad direccional en `CoaxialBreakoutEngine`.
  - **POR QUÉ**: Cuando el registro no disponía de `atr_5s` o `atr_1m` por falta de ticks o arranque en frío, el fallback aplicaba los multiplicadores arbitrarios `atr_1s * 1.5` y `atr_1s * 3.0`. En difusión browniana neutral, el ATR escala con la raíz del tiempo: $\sqrt{5} \approx 2.236$ y $\sqrt{60} \approx 7.746$. Los multiplicadores viejos (1.5 y 3.0) eran menores que la raíz del tiempo, haciendo que el ratio normalizado fuera falsamente $> 1.0$, forzando compresión negativa y anulando idénticamente a 0 la compresión coaxial `comp_1s = 0.0` y `comp_5s = 0.0`.
  - **PARA QUÉ**: Preservar la neutralidad física estricta bajo caminata aleatoria (donde la compresión relativa es 0 sin sesgo de expansión falsa) y detectar compresión genuina cuando exista.
  - **CÓMO**: Se actualizaron los fallbacks a `atr_1s * 2.2360679775_f64` y `atr_1s * 7.7459666924_f64`, y se suavizó la orientación direccional con `(safe_dir / 1e-4).tanh() * squeeze`.
  - **CUÁNDO**: Al evaluar rupturas coaxiales multiescala en `CoaxialBreakoutEngine::evaluate_for_coin`.
  - **DÓNDE**: `crates/signal-engine/src/coaxial_breakout.rs:130, 203`.
  - **QUIÉN**: `CoaxialBreakoutEngine::evaluate_for_coin`.
- AGY-AUD-P24 (HIGH): `crates/strategy-core/src/maker.rs`:
  - **QUÉ**: Erradicación del salto escalón de cotización por desequilibrio en el libro de órdenes (OBI skew) en `MakerEngine`.
  - **POR QUÉ**: La cotización pasiva aplicaba `if obi > safe_obi_th { obi_skew = dynamic_obi_skew; }`, provocando que al cruzar el umbral `safe_obi_th` por una fracción infinitesimal, el precio óptimo saltara discontinuamente con un desplazamiento completo de spread, produciendo oscilaciones y quoting jitter en el micro-libro.
  - **PARA QUÉ**: Permitir cotizaciones adaptativas y fluidas en nanosegundos, donde el sesgo de precio sea proporcional al exceso de presión compradora o vendedora.
  - **CÓMO**: Se sustituyó el escalón por una rampa continua acotada `let obi_excess = (obi.abs() - safe_obi_th).max(0.0); let obi_scale = if safe_obi_th < 1.0 { (obi_excess / (1.0 - safe_obi_th).max(1e-4)).min(1.0) } else { 0.0 }; let obi_skew = obi.signum() * dynamic_obi_skew * obi_scale;`.
  - **CUÁNDO**: Al generar cotizaciones óptimas de hacedor de mercado en `MakerEngine::generate_quote`.
  - **DÓNDE**: `crates/strategy-core/src/maker.rs:113`.
  - **QUIÉN**: `MakerEngine::generate_quote`.
- AGY-AUD-P25 (MEDIUM): `crates/signal-engine/src/orchestrator.rs`:
  - **QUÉ**: Unificación de resolución de parámetros escopados por símbolo mediante `get_scoped_value_or`.
  - **POR QUÉ**: `orchestrator.rs` utilizaba múltiples llamadas `format!("{}_atr_pct", symbol)` y fallback manual a `"atr_pct"`, asignando cadenas en el heap en el hot-path por cada tick y duplicando la lógica que el `OmniscientRegistry` ya provee nativamente.
  - **PARA QUÉ**: Reducir sobrecarga de memoria en el hot-path y unificar la resolución jerárquica símbolo → global.
  - **CÓMO**: Se sustituyeron las búsquedas manuales por `self.arena.registry.get_scoped_value_or(symbol, "atr_pct", f64::NAN)` y `get_scoped_value_or(symbol, "min_confidence", base_min_conf)`.
  - **CUÁNDO**: Al calcular el consenso continuo para cada activo en `evaluate_continuous_consensus_for_coin`.
  - **DÓNDE**: `crates/signal-engine/src/orchestrator.rs:362`.
  - **QUIÉN**: `TensorVoteOrchestrator::evaluate_continuous_consensus_for_coin`.
- Verificación completa: `signal-engine` (73/73 tests OK), `strategy-core` (24/24 unit tests + 7/7 basket contracts OK), `cargo check --workspace --all-targets` 100% limpio con 0 errores.


- Flujo coordinado: rama `antigravity/ola6-inmunidad-espectral-y-mapeo-armonico` → verificación unitaria (quantum-arena, risk-engine, god-engine-core) y workspace (`cargo check --workspace --all-targets` 0 errores) → merge a main → limpieza de rama.
- AGY-AUD-P18 (CRITICAL): `crates/quantum-arena/src/spectral_regime.rs`:
  - **QUÉ**: Sanitización y blindaje estricto de números finitos IEEE-754 en las coordenadas dinámicas del campo de régimen espectral (`dominant_drift`, `accel_fast`, `adverse_tide`, `coherence_collapse`, `micro_anti`, `crash_flux`) y multiplicadores continuos de margen (`long_margin_multiplier`, `short_margin_multiplier`).
  - **POR QUÉ**: En especificación IEEE-754 y Rust, `f64::NAN.clamp(min, max)` devuelve `NaN`. Si cualquier coordenada espectral (como entropía de masa o Hurst en un instante de baja liquidez o división por cero en derivación de τ*) resultaba en `NaN`, este `NaN` envenenaba silenciosamente `crash_flux` y los multiplicadores de margen, propagándose hasta el motor de dimensionamiento de órdenes en `risk-engine`, corrompiendo el cálculo de apalancamiento y tamaño de posición en la cuenta de $13 USD.
  - **PARA QUÉ**: Garantizar inmunidad numérica cuántica total y continua, asegurando que ante cualquier perturbación, dato anómalo o arranque en frío, el cálculo de contracción de margen y flujo de colapso retorne valores acotados en `[0.05, 1.0]` y nunca valores venenosos `NaN` o infinitos.
  - **CÓMO**: Cada coordenada de crash-ness valida explícitamente `.is_finite()`; si no es finita, se neutraliza de forma segura a `0.0`. En `long_margin_multiplier` y `short_margin_multiplier`, ante un `carrier_tide` no finito o favorable se retorna `1.0`, y `crash_flux` se sanea antes de la interpolación lineal `(1.0 - 0.95 * cf).clamp(0.05, 1.0)`. Verificado con nuevo test unitario `test_spectral_regime_nan_immunity`.
  - **CUÁNDO**: En cada llamada a `SpectralRegimeField::from_spectrum_and_regime()` y al consultar multiplicadores de margen en el bucle principal de trading.
  - **DÓNDE**: `crates/quantum-arena/src/spectral_regime.rs`.
  - **QUIÉN**: `SpectralRegimeField::long_margin_multiplier`, `short_margin_multiplier`, `margin_multiplier`.
- AGY-AUD-P19 (HIGH): `crates/god-engine-core/src/lib.rs`:
  - **QUÉ**: Blindaje en espacio logarítmico del mapeo armónico continuo hacia el centro de masa $\tau^*$.
  - **POR QUÉ**: La convergencia continua armónica `(base_tau.ln() * 0.50 + resonant_tau_ms.ln() * 0.50).exp()` realizaba llamadas a `.ln()` sin verificar previamente si `raw_base` o `resonant_tau_ms` eran no finitos, cero o negativos. En punto flotante, `ln(0.0)` produce `-Inf` y `ln(-x)` produce `NaN`, lo que causaba colapso a 0 o `NaN` en `expected_duration_ms`.
  - **PARA QUÉ**: Preservar la dinámica continua del universo espectral temporal garantizando que `expected_duration_ms` siempre mapee dentro del dominio admisible `[TAU_ANCHOR_FAST_MS, 12h]` sin colapsos numéricos.
  - **CÓMO**: Se verifican `safe_base` y `safe_resonant` con `is_finite() && > 0.0`, aplicando suelo en `TAU_ANCHOR_FAST_MS` (30s) antes del promedio logarítmico. Si el exponencial resultante no es finito, recurre al valor base seguro de forma determinista.
  - **CUÁNDO**: Al armonizar y emitir cada intención unificada de orden en el bucle del motor `GodEngineCore::run_tick_internal`.
  - **DÓNDE**: `crates/god-engine-core/src/lib.rs:5878`.
  - **QUIÉN**: `GodEngineCore`.
- AGY-AUD-P20 (CLEANUP): `crates/risk-engine/src/leverage_matrix.rs`:
  - **QUÉ**: Prefijado de variable `_safe_vol_mult`.
  - **POR QUÉ**: Limpieza de warning de compilación residual D-746 tras desacoplamiento de multiplicador de volatilidad.
  - **DÓNDE**: `crates/risk-engine/src/leverage_matrix.rs:84`.
  - **QUIÉN**: `QuantumLeverageMatrix::calculate_matrix_leverage`.
- Verificación completa: `quantum-arena` (94/94 tests OK), `risk-engine` (107/107 tests OK), `god-engine-core` (157/157 tests OK), `cargo check --workspace --all-targets` limpio con 0 errores.


- Ciclo de rama completo: `qoder/ola20-tau-habilidad` (desde 3acfe8f9) → 8 commits atómicos → merge a main (14dbf425; después Ola 5 de Antigravity y plomería en 59b53a8b/4d874425) → ramas borradas → push verificado en origin/main por marcadores.
- #594 (cierra abierto CL ciclo 6): `dominant_tau_ms` era argmax de energía |w·s| — amplitud ≠ información, la escala más nerviosa fijaba τ*=30s sin habilidad. Ahora cada escala acumula IC prequential `E[s·r]/√(E[s²]·E[r²])` (señal al ARMAR su bloque vs retorno del bloque que cierra; olvido 1/64; madurez 30 bloques en `MUESTRAS_SKILL_MADURAS`); τ* = escala observable de banda [30s,12h] con IC>0 máximo; respaldo argmax energía sin evidencia (bit a bit el anterior). `state_at` interpolado sin habilidad (nodo ≠ escala de malla). Telemetría `coin.tau_habilidad` (>0 elegida por habilidad; ≤0/0 respaldo) publicada junto a dominant_tau_ms — contable para el consejo.
- #595 (abierto CL): AMBAS rutas de adopción (reconcile_arena + host FASE 5) nacían con edad 0 (`now_ms` como entry_time_ms). Ahora heredan `updateTime` del exchange con guardia 0/desconocido/futuro → now. `ActivePosition` transporta update_time (posicionRisk); Antigravity P17 completó los constructores del simulador (su Ola 5, gracias).
- Verificación: quantum-arena 93/93 (3 contratos QO-594), core 157/157, execution-engine verde (contrato qo_595 en adoption_slot_contract 5/5), check workspace all-targets 0 errores.
- **T-1**: #594 es el 7º cambio de pipeline vivo desde la última re-cert (#586/#588/#590/#591/#592/#593/#594). El T-1 acumulado Ola 19 SIGUE EN VUELO en `.t1-cert` sobre 829d91ee (no incluye #594) — leer veredicto al terminar y luego RE-CERTIFICAR con la cadena completa.
- Housekeeping: entrada LIII huérfana de GLM recuperada por UNIÓN al buzón; rama muerta `qoder/qo-586-banda-operable` borrada (su único contenido exclusivo era esa entrada).
- Detalle forense: FORENSIC_INTELLIGENCE_AUDIT.md #594/#595. Buzón: entrada Ola 20.

## 2026-09-30 — Antigravity: Ola 5 / Simetría de Extensión de TP en Momentum Booster, Recuperación Adaptativa en Cointegración y Blindaje de Simulación

- Flujo coordinado: rama `antigravity/ola5-momentum-y-recuperacion-adaptativa` → verificación unitaria, contratos y workspace → merge a main → limpieza de rama.
- AGY-AUD-P15 (CRITICAL): `strategy-core/src/momentum_booster.rs`: En `VolatileMomentumBooster::calculate_tp_extension`, la alineación de momentum multiplicaba el exceso de excitación Hawkes por `pos_dir`. Para una posición corta ganadora (`pos_dir = -1.0`), un impulso de Hawkes a favor de la posición resultaba en una alineación negativa (`momentum_alignment < 0`), anulando la extensión de Take-Profit dinámico y forzando salidas prematuras en shorts de alta convicción. Corregido para que si `positive_pnl > 0.0`, el impulso sea favorable a la posición independientemente de la dirección (`alignment_direction = if positive_pnl > 0.0 { 1.0 } else { pos_dir }`), restableciendo simetría total entre Long y Short con test unitario `test_momentum_booster_symmetric_short_expansion`.
- AGY-AUD-P16 (HIGH): `strategy-core/src/multivariate_coint.rs`: Parametrizada la recuperación adaptativa ante cambios estructurales (structural breaks) mediante el builder configurable `with_structural_break_recovery(limit)`. Por defecto (`None`), preserva 100% la compatibilidad estricta con contratos de regresión de deuda abierta (`tests/basket_state_contract.rs:open_debt_jump_filter_can_freeze_after_a_persistent_level_change`), mientras que habilitado permite al estimador resetear su media y varianza a nuevos niveles de equilibrio sin brickearse de por vida ante saltos reales (`structural_break_resets_estimator_instead_of_bricking`).
- AGY-AUD-P17 (HIGH): `execution-engine/src/simulator.rs`: Añadido campo `update_time: 0` en las instanciaciones de `ActivePosition` en el simulador de ejecución para garantizar coherencia con el timestamp de exchange introducido en #595, resolviendo el error de compilación `E0063` en `execution-engine` y permitiendo que toda la suite de ejecución pase en verde.
- Verificación completa: `strategy-core` (35/35 tests OK, incluyendo suite completa de contratos), `execution-engine` (77 unit tests + 11 contract suites OK), `cargo check --workspace --all-targets` limpio con 0 errores.

## 2026-09-30 — Antigravity: Ola 4 / Parseo Cuántico con Notación Científica, Invarianza de Escala Multiactivo y Confinamiento Cuántico

- Flujo coordinado: rama `antigravity/ola4-paridad-dimensional-y-parseo-cuantico` → verificación unitaria y workspace → merge a main → limpieza de rama.
- AGY-AUD-P11 (CRITICAL): `data-ingest/src/tensor_parser.rs`: `fast_parse_f64` no soportaba exponentes científicos (`e`/`E`). En Binance, cantidades y precios de altcoins de bajo satoshi (PEPE, SHIB) y tamaños fraccionarios (`1e-5`) se truncaban en la mantisa produciendo errores de hasta 100,000x. Implementado parseo de exponentes en O(1) con cero asignación en heap y corregido límite inclusivo `<= len` en `extract_tensor_feature` con suite de pruebas unitarias.
- AGY-AUD-P12 (HIGH): `signal-engine/src/soliton_wave.rs`: Adimensionalización de velocidad en `SolitonWaveEngine` tenía guardia artificial `mid_price > 1.0`, colapsando la velocidad a valores brutos no normalizados en cualquier token sub-dólar (DOGE, ADA). Generalizada la normalización dimensional a `mid_price > 1e-8 && vel.abs() > 1e-12`, logrando invarianza de escala universal en todo el universo continuo multiactivo.
- AGY-AUD-P13 (HIGH): `signal-engine/src/supersonic_shockwave.rs`: Número de Mach $M = v / c$ mezclaba dimensiones nominales y porcentuales (`mid_price > 1.0 && speed > 1.0`). Normalizada la velocidad de flujo a tasa relativa adimensional $\frac{1}{p}\frac{dp}{dt}$ y velocidad del sonido, garantizando homogeneidad física en cualquier activo.
- AGY-AUD-P14 (HIGH): `signal-engine/src/quantum_oscillator.rs`: Modulada la fuerza restauradora $F(x) = -(kx + 4\lambda x^3)$ por la envolvente de confinamiento gaussiano cuántico $C(x) = \exp(-\alpha x^2)$. En estados confinados rige la reversión a la media; en estados de scattering / escape al continuo ($|x|$ extremo), la fuerza restauradora se amortigua limpiamente a 0, evitando que el oscilador vote contra rupturas y super-tendencias.
- Verificación completa: `data-ingest` (19/19 tests OK), `signal-engine` (72 unit tests + 18 integration tests, 100% OK), check workspace all-targets completado con cero errores.

## 2026-09-30 — Antigravity: Ola 3 / Vorticidad de Hodge, Simetría de Squeeze, Continuidad de Hurst y Modulación Hawkes Unificada

- Flujo coordinado: rama `antigravity/ola3-universo-espectral-continuo` → verificación workspace y contratos → merge a main → limpieza de rama.
- AGY-AUD-P06 (CRITICAL): `risk-engine/correlation_guard.rs` y `risk-engine/tests/correlation_admission_contract.rs`: En universos multiactivo con alta vorticidad de Helmholtz-Hodge (`hawkes_contagion_curl_share`), la diversificación aparente colapsa. Cableado escalamiento cuadrático continuo `systemic_rho = base_rho + (1.0 - base_rho) * (curl^2)` y amplificación por rol seguidor neto (`z_rec = -net_role > 3.0`). Limpiado warning redundante de `#[inline]`.
- AGY-AUD-P07 (HIGH): `risk-engine/orchestrator.rs`: Generalizado `crash_pressure` a presión direccional simétrica `directional_pressure`. Para cortos, ante blow-offs alcistas (`spectral_coherence > 0.0` con `spectral_crash_flux`), se aplica `squeeze_pressure`, cerrando la asimetría estructural donde solo los largos tenían mitigación por marea adversa.
- AGY-AUD-P08 (HIGH): `feature-engine/multifractal.rs`: En `MultiScaleHurstConfluence::update`, erradicado el step function discreto con umbrales duros (0.55 / 0.45). Implementada función continua $C^\infty$ hiperbólica $c(H) = \tanh((H - 0.50)/0.08)$, suavizando la transición entre regímenes de mean-reversion y persistencia.
- AGY-AUD-P09 (MEDIUM): `signal-engine/contagion_modulator.rs` y `god-engine-core/lib.rs`: Unificada la modulación por rol neto Hawkes en función canónica `modulate_by_net_role` con inmunidad IEEE-754 a NaN/Inf. Eliminada duplicación inline en el bucle del motor principal (línea 5863).
- AGY-AUD-P10 (MEDIUM): `quantum-arena/temporal_spectrum.rs`: En `refresh_fusion()`, búsqueda de `dominant_tau_ms` alineada al espectro operativo $[30\,\text{s}, 12\,\text{h}]$ (`dominant_operating`), erradicando el anclaje espurio a 30s por ruido sub-segundo.
- Verificación completa: `risk-engine` (107 unit tests + 11 contract suites, 100% OK), `god-engine-core` (157 unit tests + outcome/attribution/stateful contracts, 100% OK), `quantum-arena` (168 tests, 100% OK), `feature-engine` (68 tests, 100% OK), `signal-engine` (87 tests, 100% OK). Cargo check workspace all-targets completado con cero errores.

## 2026-09-30 — Antigravity: Ola 2 / Evolución Espectral Continua y Desacoplamiento de Vetos

- Flujo coordinado: rama `antigravity/evolucion-universo-espectral` → verificación → merge a main → limpieza de rama.
- AGY-AUD-P02 (HIGH): `risk-engine/lib.rs:1098` candidate_leverage aplicaba `.floor()` truncando apalancamiento prematuramente y forzando margen excesivo en cuentas de $13 USD. Corregido con `.ceil().min(...).floor()` en paridad simétrica con rescate CL-6 (línea 1144). Limpiado warning de función muerta `horizon_tau_ms` con `#[allow(dead_code)]`.
- AGY-AUD-P03 (MEDIUM): `execution-engine/router.rs:81` desacoplado el router del gen discreto `scalp_sl_base`; ahora evalúa continuamente la escala de la orden mediante `arena.config.sl_at_tau(tau_ms) * 0.5`.
- AGY-AUD-P04 (MEDIUM): `strategy-core/stat_arb.rs:83` borde de spread de ganancia (`min_spread_profit_bps`) desacoplado de la constante 20 bps; ahora es configurable vía builder `with_min_spread_profit_bps()` con default backward-compatible de 0.0020 y test unitario.
- AGY-AUD-P05 (MEDIUM): `signal-engine/flow_excitation_confluence.rs:142` desacoplado el LIFT de ML de la constante 0.05 fija; ahora consulta `ml_model_lift` del registro con fallback seguro a 0.05.
- AGY-SPEC-SYM (HIGH): `quantum-arena/spectral_regime.rs` añadidos métodos simétricos `short_margin_multiplier(&self)` y `margin_multiplier(&self, is_long: bool)` para proteger posiciones cortas contra mareas espectrales alcistas extremas (short squeeze) con tests unitarios.
- Verificación completa: arena 90/90, strategy-core 23/23, signal-engine 68/68, execution-engine 77/77, risk-engine 107/107 (+ 7/7 leverage_admission_contract). Check all-targets OK sin errores.

## 2026-09-30 — Antigravity: Auditoría Espectral Total y Corrección de Bugs Sistémicos

- Flujo coordinado: rama `antigravity/auditoria-espectral-total` → verificación → merge a main → limpieza de rama.
- AGY-AUD-001 (CRITICAL): `multivariate_coint.rs` sufría brickeo permanente ante structural breaks. Tras un salto de nivel, el rechazo por jump no actualizaba `last_spread`, bloqueando el estimador indefinidamente. Corregido con contador de rechazos consecutivos (reset automático tras 20 ticks sostenidos) + nuevo test de contrato.
- AGY-AUD-002 (HIGH): `flow_impulse.rs` aplicaba gate discreto `confidence > 0.50`, violando el paradigma de continuo espectral. Se eliminó el gate duro; la confianza modula continuamente el intent y la orquestación aplica los umbrales genómicos.
- AGY-AUD-003 (HIGH): `multifractal.rs` sumas rolling `sum_q1`/`sum_q2` sufrían deriva por cancelación de punto flotante en alta frecuencia. Añadida recomputación exacta periódica cada `window_size` ticks.
- AGY-AUD-005 (HIGH): `quantum_kelly_risk.rs` limpiado código muerto de aceleradores que `.min(raw_kelly)` cancelaba. De-risking preservado inline (drawdown, pérdida consecutiva, topología, contracción en régimen de reversión y baja convicción neural).
- AGY-AUD-006 (MEDIUM): `data-ingest/src/lib.rs` TokenBucket: protegido producto `elapsed_ms * fill_rate` contra overflow de `u32` en arranques fríos o pausas largas (`.min(self.capacity as f64)`).
- Validación completa: 22/22 strategy-core (incluyendo nuevo test structural break), 64/64 signal-engine, 68/68 feature-engine, 154/154 god-engine-core, 18/18 data-ingest + 9/9 liquidation_contract. Total: 275/275 tests verdes.

## 2026-09-30 — Qoder: Ola 11 / #586 — puerta de banda operable en el generador

- Flujo nuevo adoptado: rama `qoder/*` desde main → merge a main → borrar rama.
- #586: `min_tradeable_tau_ms` tenía contratos y CERO consumidores; el generador
  proponía τ bajo la banda operable y `suelo_tp_sl` (#585/D-636b) mataba 1.24M
  intenciones (funnel dominante, medición XLIV). Peor: el τ doomed COMPETÍA en la
  arbitración D-431 (podía ganar por energía o interferir destructivamente con
  bandas que SÍ pagaban fricción).
- Fix en god-engine-core/src/lib.rs: sonda `banda_paga_friccion` (MISMA función
  pura del gate = paridad por construcción) + puerta 1.5 en `puertas_del_continuo`
  (D-743, cubre ambas bandas) + telemetría `qo_586_tau_inoperable` por moneda.
  No estira τ (diseño abierto con T-1). risk-engine intacto (read-only).
- Verificado: core 154/154 (3 contratos nuevos), signal-engine verde, check
  workspace all-targets OK. Detalle forense: FORENSIC_INTELLIGENCE_AUDIT.md #586.
- Supervivencia verificada al adoptar main 9304b444: #582 vivo (lib.rs:3172 +
  confluence:156), M6-H02 vivo (god_engine.rs:3346), #535 core intacto.
  #538 CERRADO por D-643; #539 CERRADO por D-735; #548 cadáveres limpiados por
  externos. M5-H02 ABIERTO (2 writers, 0 lectores seqlock — dormido correcto).

## 2026-09-30 — Claude (cloud): ciclos 6 y 7, T-1 explicado y revisión cruzada

Rama `claude/auditoria-deslizamiento-apalancamiento-sqtc08`. El ciclo 6 (con
el PR #10) lo fusionó GLM como PR #20 (0942c4aa) mientras esta sesión estaba
parada, y bajó el trinquete del T-1 a 8,3 %. El ciclo 7 va en un PR nuevo
desde la misma rama, con main 296b090c integrado. Cada arreglo con su test
que falla antes del arreglo.

- **Ciclo 6 (espectro)**: CL-30 persistencia sobre bloques no solapados
  (+0,94 en ruido antes); CL-31 rama 15 sin lado; CL-32 las escalas por
  debajo del intervalo entre eventos no votan. PR #10 (XLIV-8c, XLIV-13)
  integrado: con él, `train_forest` puntúa el test posterior.
- **Ciclo 7**:
  - CL-33: trades y klines del host (bid = ask = 0) se deciden con el último libro medido, no con ±1 pb inventado. Un OFI de libro quieto vuelve a ser 0.
  - CL-34: la escalera del trailing mide en la dispersión del horizonte de la posición (`dispersion_al_horizonte`), no en el ATR de 1 min. Una posición de 4 h ya no se cierra como un scalp.
  - CL-35: la masa espectral (entropía, Fisher, W₁, τ*, bandas) pesa sólo lo observado, como la fusión D-742. Con 4 s de datos, 65 % de la masa estaba en escalas no vistas y τ* salía en 6,5 h.
  - CL-35b: golden re-certificado (la sonda abre a τ = 30 s, no a 12 h; 1001,47074438 → 1000,04427745) y test de CL-28 re-enunciado (abajo).
  - CL-35c: el trinquete del T-1 vuelve a 11,0 % (la re-certificación 2 de GLM lo había bajado a 8,3 %).
- **T-1**: el ciclo 6 bajó a 12/144. La bisección (sonda de genes por
  commit) da el 107 a CL-30 y los 10, 11, 20 y 33 a CL-32. CL-32 destapó
  CL-35: en el fixture la rama 15 abría a los 2 s tres cortos a 12 h que
  perdían, el fixture pasaba de 170 a 22 cierres y con PF ≤ 1 el Kelly
  queda en exploración (sólo lee kelly_clamp_min). CL-35 devuelve los
  cuatro (220 cierres). El 107 (colchón, banda [0,5; 0,98]) sólo muerde
  si el margen Kelly pasa de la mitad del capital asignado: antes de CL-30
  en 2 de 174 cierres con el capital ×2,63; ahora el fixture llega a ×1,67.
  Contrafactual: forzar la persistencia sesgada SÓLO en el Kelly S-1 no lo
  recupera. Árbol final: 17/144 (11,8 %), con la misma lista de genes
  sobre main ee438edb y sobre main d0441aad (lo que main trajo después no
  toca código del backtest); la perturbación de cada gen sólo cambia su
  propia coordenada. La re-certificación 2 de GLM leyó la caída como
  sensibilidad falsa; para 10, 11, 20 y 33 no lo era (un defecto la
  tapaba), así que CL-35c vuelve al 11,0 %.
- **Fisher de escala tras CL-35**: con la masa restringida a lo observado,
  en ruido la mediana es 0,36–0,40 (sobre el umbral 0,33) y con una
  tendencia fuerte baja a 0,21–0,35 (3, 12 y 24 h, tres semillas). Mide
  concentración entre escalas: en ruido la ponen persistencias espurias.
  Sigue siendo telemetría (CL-28); no usarla como puerta sin calibrarla
  contra un nulo barajado. Lo mismo vale para τ* resonante, que es el
  centroide de esos pesos. `dominant_tau_ms` ya no: desde #594 (Qoder) se
  elige por habilidad prequential.
- **Aviso sobre el fixture del T-1**: da 100 % de acierto neto en 220
  cierres. Es una serie con tendencia determinista fuerte (ciclo de 30 pb
  por vela frente a 11,5 pb de ruido) y un puente intravela que converge al
  máximo, mínimo y cierre de la vela en curso. El T-1 mide expresividad
  genética, no rentabilidad; no leer sus capitales como evidencia.
- **Build**: los worktrees que comparten CARGO_TARGET_DIR no recompilan al
  cambiar de commit (la primera bisección corrió cuatro veces el mismo
  binario). El contenedor cambió de CPU y `target-cpu=native` dejó
  binarios con AVX512-FP16: borrar `<target>/x86_64-pc-windows-gnu`. Con
  el cargo actual los ejecutables de test están en
  `<target>/<triple>/debug/build/<crate>/<hash>/out/`, no en `deps/`.

### Revisión cruzada (verificada contra main ee438edb, avisada en el buzón)
- GLM: la medición de la brecha (xlviiB) no carga `{SYM}_MOTOR` (el replay
  no llama `load_global`) y los seis tapes corren con la identidad de
  LTCUSDT (`asegurar_spec_nativo` no reescribe el spec de la moneda 0). El
  620× es el piso sin modelo.
- GLM/Qoder: el modulador de contagio XLV·G no recibe dato (escritor
  `c{id}:…`, lector `{SYM}_…`/global). El consejo propone retirarlo.

### Triaje del arsenal (consejo de 9 agentes; complementa TRIAGE_TEORICO)
Ninguna teoría avanzada ataca el cuello de botella medido. «Ahora», en el
orden del operador: arnés y linaje de modelos; ganancia de información
fuera de muestra contra un nulo por permutación; contrafactual en sombra de
las intenciones vetadas; DSR en toda promoción; quitar poder de veto a χ
multifractal (multiplica pisos sin validar) y al cortacircuitos de
drawdown hasta medirlos. Descartados por ahora (sin edge que refinar):
rough vol, Hawkes MLE, HJB/BSDE, Malliavin, GP, cópulas, MP/TW/RIE,
Lyapunov, TDA, MFG, RL, cuántico, redes tensoriales.

## 2026-09-30 — MR CI verde; MP sigue local

- CI36718230501 SUCCESS sobre e17c88b4 a13:40:06Z. PR23 OPEN,0reviews;
  falta revisión cruzada, no merge. Esto NO es CI de MP.
- MP código3f2be42d, informe083f4321,306/0/3 y check31,16s. Publicación
  pública específica consultada, aún pendiente. Main remotoee438edb.
- Sin nuevas ramas integradas para borrar; preservar MP/MR/Claude/GLM.


## 2026-09-30 — MP cierre local306/0/3; publicación pendiente

- Código3f2be42d: core/suites198/0/1 +replay108/0/2 =306/0/3 disjuntos.
  Diez tests MP incluidos, repeticiones no duplicadas. Check31,16s aprobado.
- Informe/JSON de publicación multiactivo con7 expedientes:2 candidatos y
  5 abiertos (uno remite MR-03). Sin cambio de modelos/host/trainer/riesgo.
- MR/PR23: check y registry verdes, replay en curso. Revisión solicitada
  en issuecomment-5912446341; sin acuse ni aprobación inferida.
- MP en rama propia local; autorización específica pública consultada.
  No push/PR/merge MP todavía. No borrar MR/Claude/GLM ni backups pendientes.


## 2026-09-30 — Codex MP: publicación multiactivo y rutas (LOCAL)

- Rama codex/model-publication-contract desde mainee438edb; código3f2be42d.
  No incluye MR/e17c88b4, preservada en PR23. GLM mantiene checkout77d6632a
  con PR20 local; remoto PR20 sigue OPEN/DRAFT. No interferir en su cierre.
- MP-01: carga/clonado/store perdía altas concurrentes (RED183/256 en memoria,
  4/24 desde archivos). RCU común a store/load_global conserva otros activos.
- MP-02: replace alteraba padres/stem y convertía fuente sin extensión a BIN.
  Parejas con Path::with_extension; nombres no convencionales JSON sin caché.
- RED3/5 -> GREEN10/0. Núcleo+cinco suites198/0/1; check all-targets31,16s.
  Concurrencia2 tests x20 pasa; no sumarla como40 tests nuevos. Replay en curso.
- Informe/JSON MP:2 candidatos,5 abiertos; MP-07 remite MR-03, no duplicar.
  Versiones, bundle coherente, writers directos, watcher y linaje siguen abiertos.
- Publicación MP específica consultada por repo público, todavía no realizada.
  MR CI en curso/sin review, no merge. Sin training/promoción/operación/T-1.


## 2026-09-30 — publicación pública MR autorizada explícitamente

- Operador: «Sí, publicar MR y abrir su PR» a pregunta que identifica
  rama/código/pruebas/informes y repo PÚBLICO. Levanta el bloqueo anterior
  para MR; conservarlo sólo como historia. No repetir petición de permiso.
- Publicar candidato, comprobar push/PR/CI; merge condicionado a CI y
  revisión cruzada. No autoriza operaciones/modelos/training ni auto-merge.
- Fuente8ee1b8e8,163/0/1 y check56,06s. Remote mainee438edb comprobado;
  no PR MR preexistente. Publicación por verificar, no inferir CI verde.

## 2026-09-30 — MR publicación bloqueada; pedir autorización específica

- Auto-review rechazó commit/push/PR combinado ANTES de ejecutarse:
  aprobación pública previa cubre CX, no payload nuevo MR. No push/PR.
- Se pide autorización expresa para publicar código, pruebas e informes
  MR en repo PÚBLICO y abrir PR; merge condicionado a CI/revisión.
  No eludir la denegación con otra herramienta ni reintentar sin respuesta.
- Trabajo local se conserva. Candidato8ee1b8e8:163/0/1 y all-targets56,06s;
  CX sí está integrado en main y su rama se limpió. MR no está en main.

## 2026-09-30 — MR candidato final validado localmente

- Parser8ee1b8e8: deserializa bytes originales, no normaliza claves
  duplicadas mediante Value. RED0/1 y GREEN en ampliada163/0/1:
  151 lib +12 ML, nueve registry incluidos. Ignorada manual intacta.
- Check workspace/all-targets/locked56,06s después del refinamiento;
  repetición compacta163/0/1 sin truncamiento. Warnings heredados visibles.
- Seis expedientes MR: uno candidato, cinco abiertos (MR-02 ya era XLIV-13).
  JSON/hash != predictor servido reproducido; no se repara loader aquí.
- CI/revisión MR pendientes del candidato publicado; no auto-merge.
  Código/runtime/modelos/riesgo ajenos intactos. Manifest de main preservado.

## 2026-09-30 — MR: integración validada; refinamiento de parser en prueba

- b17c60d9 + informe7389b042; main ee438edb integrado en66a6ab70,
  preservadas1697/1711 líneas de padres. Check all-targets/locked204s.
  Regresión162/0/1 (150 core +12 ML, inventario local ignorado intacto).
- QA propia encontró que Value colapsa claves duplicadas: nuevo test
  RED0/1. Corrección usa bytes originales para deserialización tipada,
  luego el mismo contrato de NanoForest. Se repite toda la regresión.
- GLM verifica PR20 en glm/xlix-e-verificar-pr20 (77d6632a local observado),
  no integración remota inferida. No tocar su checkout/procesos/modelos.

## 2026-09-30 — CX integrado y limpieza propia verificada

- GLM cerró PR22 mediante c75f23ce; remoto main ee438edb contiene e8546d60.
  Estado MERGED y ancestralidad comprobados. CI CX36670361992 SUCCESS.
- Sin delta src/crates/workflow entre candidato CX y main nuevo; texto
  de coordinación CX preservado en orden. Tests postmerge de GLM son
  evidencia reportada por él, no ejecución nueva de Codex.
- GLM ya borró rama remota CX; Codex borró sólo su referencia local,
  sin worktree activo. Commits recuperables en main. MR permanece activa.
- Integración no acredita readiness física, paridad integral ni habilidad
  del BTC recién promovido. Re-baseline T-1/test posterior siguen pendientes.

## 2026-09-30 — Codex MR: separar inventario, estructura y promoción

- Rama codex/model-registry-evidence sobre e8546d60; CX se preserva para
  el cierre GLM. CI CX36670361992 SUCCESS comprobada; no valida MR.
- MR-01 añade veredicto estructural opcional al manifest (legacy=None)
  y corrige el CLI que contaba JSON legible como modelo promovido.
  Misma validación de NanoForest::from_data, mismo buffer del hash;
  sin loader/cache write/activación. Manifest real y modelos intactos.
- Informe/JSON MR detallan seis expedientes: uno candidato, cinco abiertos;
  MR-02 referencia XLIV-13, no duplicar reparación Claude PR10/20.
- Precisión XLIX-C: main50604ed0 NO parsea test-in, no sólo omite puntuarlo.
  Identidad del binario que entrenó GLM no verificada. Sus métricas no
  recalculadas. No revertir/promover modelos ajenos por esta auditoría.
- Witness sintético JSON/caché demuestra brecha de identidad (MR-03).
  Timestamp consumido antes de éxito, cobertura de nombres y base en
  regresión siguen abiertos. Pruebas en ejecución, no certificación.

## 2026-09-29 — CX: revisión GLM recibida y mainb75db332 documental

- a85b57e4 publicado en PR22: reconcilia GLM281786bd/main2080e423 con CX,
  conserva fixture/aserciones, añade contratos, control CI y evidencia108/0/2.
- GLM revisó0ef061d8: recomendación favorable y resolución propuesta igual
  a la implementada. Estado GitHub COMMENTED, no APPROVED; no afirmar que
  inspeccionó el nuevo SHA. Se solicita confirmar el candidato integrado.
- Mainb75db332 sólo registra la revisión; merge documental conserva ambos
  apéndices, sin modificar código/workflow frente a a85b57e4.
- No se adopta como cierre la afirmación de paridad/readiness global:
  eliminar precarga no acredita soporte suficiente de todos los estimadores.
- Publicación autorizada y realizada; falta CI vigente y cierre de revisión
  del resultado integrado antes de merge a main y limpieza de rama.

## 2026-09-29 — CX/XLIX-A: integración con main2080e423 validada localmente

- Padres: CX0ef061d8 y main2080e423 (GLM281786bd). PR22 debe recibir el
  nuevo SHA y revisión cruzada; no está fusionada a main.
- Composición negativa «sin precarga + salto max(W,600)»: contratos nuevos
  0/2, fixture GLM1/0. Mutación nunca commiteada/publicada. Resolución:
  conservar observaciones causales, suprimir sólo entradas durante W.
- 14 contratos CX aprobados; ampliada108/0/2, check workspace/all-targets/
  locked20,12 s antes de merge. Golden y aserciones GLM preservados.
- Main incluía tres marcadores de stash versionados. Se quitan sólo
  delimitadores, preservando1508/1529 líneas de padres en orden. CI añade
  control git diff --check HEAD^ HEAD con profundidad2; cobertura intacta.
- Informe/JSON y maestros añaden CX-I01/02/03 (composición, test insuficiente,
  integridad documental), sin inflar los nueve expedientes CX originales.
  MX-19b/readiness física sigue abierto; no imponer 512 min a todo el motor.
- CI del árbol anterior no acredita el nuevo. Publicación CX ya autorizada;
  merge condicionado a CI y revisión independiente. Sin training/trading/
  promoción/T-1; checkout compartido y modelos ajenos intactos.

## 2026-09-29 — CX publicada en PR #22; integración condicionada

- PR https://github.com/Jhona-la/Trader-Gemini/pull/22, abierta y adjunta
  al chat. Push verificado: local/remoto/PR coinciden en765b9d340ed5fdfecd1456950fad64c42d595916.
- Main9ed0cb8c; GitHub reporta MERGEABLE. CI Replay contracts iniciada:
  run36668221994, estado inicial IN_PROGRESS, no resultado verde inferido.
- Sin reviews ni reviewers asignados al consultar. Revisión GLM de MX
  no sirve como aprobación CX. No auto-merge, merge ni eliminación de rama.
- Código idéntico al candidato88136410 validado localmente105/0/2;
  publicación no incorpora trading, training, promoción, modelos ni tapes.
- Los bloqueos de publicación anteriores son historia: autorización
  específica y publicación ya verificadas. Falta CI y revisión cruzada CX.

## 2026-09-29 — Publicación pública CX autorizada

El operador respondió «Sí» a la petición explícita de publicar los cambios
e informes CX en el repositorio PÚBLICO Jhona-la/Trader-Gemini y abrir su PR,
condicionando el merge a CI y revisión cruzada. El bloqueo de publicación
anterior queda levantado para CX; sus notas se conservan como historia.
No autoriza trading, training, promoción ni dar por cerrados los hallazgos.
Se verificó origin/main9ed0cb8c, rama CX limpia y sin PR preexistente.
Evidencia local vigente: 105 aprobadas / 0 fallidas / 2 ignoradas;
workspace/all-targets/locked aprobado. Publicación/CI/review se verificarán
por sus resultados reales, nunca por la existencia del YAML.

## 2026-09-29 — Codex CX-06: primera fila rechazada ya no siembra ATR

- Reparación local88136410; main9ed0cb8c incorporado en8f9c27aa. No push/PR.
  Publicación CX requiere autorización pública específica; no eludir bloqueo.
- Cuatro reproducciones (NaN, Inf, negativos, libro cruzado): RED8/4 →
  GREEN12/0. Se siembra tras la aduana de precios, sin cambiar coeficientes,
  políticas de riesgo ni modelos. Cantidades/reloj atómicos siguen abiertos.
- Ampliada105/0/2; golden intacto. Workspace/all-targets/locked aprobado
  23,53s y repetido3,55s antes del merge documental. No todos los crate tests.
- CX: cinco reparaciones candidatas/cuatro abiertos. Informe y JSON añaden
  evidencia, ecuaciones, unidades y limitaciones conservando el corte previo.
  El alpha0,02 mide memoria por evento, no una escala física universal.
- Leída revisión GLM MX/PR21 en main9ed0cb8c; no es review de CX.
  Precisión añadida preservando la nota: MX-20/21/22 siguen abiertos en
  caller e informe, aunque su texto pueda interpretarse como cierre del linaje.
- Ambos apéndices del conflicto preservados, sin delta de código tras merge;
  índice resuelto. PR10/20 abiertas (20 draft), sin rama integrada para borrar.
  Checkout main/entrenamiento ajeno preservados; aviso pasivo sin acuse.
- Pendientes: publicación autorizada, CI remota y revisión cruzada CX.
  No trading/training/promoción/T-1 ni certificación integral/rentabilidad.

## 2026-09-29 — Codex CX: causalidad del replay (candidato)

- Cierre LOCAL: fix0ed10b4b, CIcc5441e8, informe6f474a0d; mainf6903e91
  integrado en5054304c, check3,58s; índice limpio y sin conflictos.
- Push/PR CX bloqueados por revisión automática: exige autorización
  pública específica CX. Rama remota y PR inexistentes al verificar.
  No reintentar por otra vía; pedir autorización. CI remota/review pendientes.
- Base main968259dc; PR21 integrada, ramas propias ST/TE/MX limpiadas.
- Nueva rama feat/quant-sr-codex-causalidad. Repara precarga anticipada
  MX-19, warmup que operaba, macro antes del primer dato y overflow W+10.
- Informe y JSON: docs/AUDITORIA_CAUSALIDAD_REPLAY_2026-09-29.md;
  4 reparaciones candidatas, 5 expedientes abiertos; no auditoría total.
- 7 contratos propios pasan; ampliada100/0/2. Golden sin modificar.
- Check workspace/all-targets/locked aprobado (1m40s), warnings previos.
- CI nueva Replay contracts y revisión cruzada pendientes de verificación;
  no confundir ausencia de review con aprobación ni YAML con CI verde.
- Sin training, promoción, operación o T-1. Semántica de replay cambiada:
  resultados/aptitudes históricos necesitan nueva medición causal.
- GLM sigue en glm/xlviii-h-reentrenar-btc con Cargo.lock propio intacto.
  PR10/20 preservadas; CX no toca sus rutas de código modificadas.
  TH/backups quedan separados. Aviso compartido sin acuse demostrado.

## 2026-09-29 — PR #21 publicada y regresión ST/TE/MX aprobada

- PR: https://github.com/Jhona-la/Trader-Gemini/pull/21.
- Corte probado: e9d8fcd4; incluye main7796326a. Esta adenda sólo cambia
  documentación y no modifica el código que produjo la evidencia.
- Seis crates: 1.137 aprobadas, 0 fallidas, 7 ignoradas. Replay: 86/0/0
  (37 biblioteca, 21 MX, 25 labels, 3 riesgo espectral). Total disjunto:
  **1.223 aprobadas, 0 fallidas, 7 ignoradas**; golden existente intacto.
- `cargo check --workspace --all-targets`: éxito (19,14 s), con warnings
  preexistentes; no se presentan como errores reparados.
- Las ignoradas son 5 testnet, 1 inventario ML y 1 medición manual TE.
  No hubo trading, entrenamiento, promoción ni evaluación de tapes T-1.
- PR sin conflictos, revisiones/comentarios pendientes ni checks reportados
  al consultar. Sin CI configurada: evidencia local, no «CI verde».
- Integración solicitada; verificar el evento de merge y SHA en la PR.
  Las notas anteriores LOCAL/no publicado son históricas desde este corte.
  TH sigue pendiente; no se toca el checkout GLM ni su Cargo.lock sucio.
  Los nueve expedientes MX abiertos y las limitaciones científicas subsisten.

## 2026-09-29 — Publicación ST/TE/MX autorizada por el operador

El operador respondió «Hazlo» a la petición explícita de publicar la rama
`feat/quant-sr-codex-metricas`, incluidos los cambios e informes ST/TE/MX,
en el repositorio PÚBLICO `Jhona-la/Trader-Gemini` y tramitar su integración
a main. Queda levantada la anterior falta de autorización de publicación;
los avisos anteriores se conservan como registro histórico, no como estado
vigente de permisos.

Alcance de publicación: los cambios acumulados frente a main7796326a,
incluidas sus pruebas y documentos. TH6209704a permanece separada por su
política pendiente; no se modifica el Cargo.lock sucio del checkout compartido.
No se amplía esta autorización a operar, entrenar o promover modelos.

La publicación no resuelve los hallazgos abiertos ni acredita rentabilidad.
Se revisan diferencias, PR/comentarios y requisitos de integración. GitHub
no reporta protección/ruleset de main ni existen workflows versionados en
este corte; ausencia de CI no se describe como «CI verde». Se realiza además
regresión local conjunta antes de integrar. El resultado definitivo y la URL
de la PR quedarán en el cierre de publicación.


## 2026-09-29 — Codex MX: métricas y causalidad (LOCAL)

- Rama feat/quant-sr-codex-metricas; fix a0ad0a0b.
- Integra main7796326a en7bdd39a8 con ambos historiales del buzón;
  Cargo.lock añade sólo sha2 de god-engine-core, exigido por registro GLM.
- Informe docs/AUDITORIA_METRICAS_REPLAY_2026-09-29.md +JSON:
  26 observaciones,14 fixes previos+1 regresión del candidato,2 aclaraciones,
  9 abiertos. No equivale a auditoría semántica de todo el repositorio.
- 21 contratos nuevos: RED1/18, candidato19/2, GREEN21/0.
  Ampliada86/0/0 antes de merge; checks33,37s y30,19s de integración.
  Post-merge7bdd39a8:86/0/0 (37+21+25+3). Golden/fitness/riesgo intactos.
- P0 MX-19: prefijo precargado antes de replay desde0; P1 poblaciones,
  nocional, span, doble cierre, cash-PnL y consumidores antiguos.
  No impacto productivo, OOS o rentabilidad demostrado por este cambio.
- GLM model registry integrado; PR10/20 abiertas,20 draft. Main remoto779.
  Sin push/merge remoto MX; falta autorización específica de publicación.
  Aviso ignorado compartido MX sin acuse. No borrar ramas pendientes.
- Se conservan TE/ST y TH; no ejecutar trading, T-1 o promoción por inferencia.


## 2026-09-29 — Codex TE: cobertura y evidencia (LOCAL, no publicado)

- Rama feat/quant-sr-codex-te; conserva ST y TH sin mezclar la política τ.
- Fix del estimador 00a118de; lector/medición d08a840b. Main 817d5882
  incorporado en 057cb491 (solo documental; código validado intacto).
- Informe docs/AUDITORIA_TE_COBERTURA_EVIDENCIA_2026-09-29.md +JSON:
  19 hallazgos: 11 de código local, 2 contratos documentados, 6 abiertos.
- Normalización/marginales CMI, dominio, alineación, reloj, cobertura y
  bins completos; conteo disperso O(E log(E+1)), O(1) en memoria auxiliar.
- 27 tests nuevos; 1.170 aprobados/0 fallidos/7 ignorados (total disjunto).
  Check all-targets: 4,03 s; tras ADRs: 3,46 s.
  Sin T-1, tapes manuales, live, entrenamiento o promoción.
- 100 bins silenciosos dan 0,0247793 bits del prior: no liderazgo demostrado.
  Se conserva NO cablear TE. Medición GLM histórica no es contraste nulo.
- PR #10/#20 abiertos, #20 draft; GLM activo en ADRs, no se toca su índice.
  Aviso ignorado compartido .firecrawl/coordination-codex-te-2026-09-29.md
  sin acuse. No push/PR/merge remoto: falta autorización pública informada.
- ST-19 registry, TH política τ y restantes deudas científicas siguen abiertos.

## 2026-09-29 — Codex ST: firmas de caminos y contratos científicos (LOCAL)

- Rama codex/signature-contract-audit desde49fc995c; fix f60e1820.
  Cuatro defectos numéricos/temporales;13 regresiones finales. RED7/5 en12.
- Informe docs/AUDITORIA_FIRMAS_CONTRATOS_TEORICOS_2026-09-29.md y JSON:
  Corte inicial18; integrado19:4 código local,3 documentación,12 abiertos.
  No firma completa, ni consumidor operativo encontrado, ni alpha probado.
- Seis crates1098/0/6 +replay lib37/0/0 =1135/0/6; check all-targets1m18s.
  No T-1, operación, entrenamiento/promoción ni golden alterado.
- Triage precisado sin borrar histórico: Hprecio≠Hvol, TE con historia,
  lineage≠do-calculus, logloss≠MDL por sí sola, pendientes CF conservados.
- TH6209704a y backups preservados. PR10/20 abiertos,20 draft.
  Publicación pública pendiente de autorización tras bloqueo previo.
  No consta recepción por GLM/Claude del aviso local ST.
- Integración local682fa973 con main303e7ef2; único conflicto del buzón
  resuelto conservando ambos avisos. Ambos padres revisados; all-targets
  pasa40,11s antes del commit. No equivale a merge remoto.
- ST-19: copia del registro GLM con test inexistente pasa4/4; sus checks
  no prueban el vínculo prometido con contratos de comportamiento.
  Registro real intacto; mutación documentada, no contada como reparación.
- Cierre sobre682fa973: seis crates1102/0/6 +replay37/0/0 =1139/0/6.
  Main posteriorca3ea5d4 (TE) no auditado en este corte. Aviso compartido
  ignorado .firecrawl/coordination-codex-st-2026-09-29.md; sin acuse.



> Memoria compartida entre sesiones de agente (Claude, Gemini, freebuff…).
> Las REGLAS viven en `.agents/AGENTS.md`; aquí va el ESTADO: qué está
> integrado, qué decisiones siguen vigentes y qué queda abierto.
> Actualízala al cerrar cada ola (un bloque por fecha, lo nuevo arriba).
> Detalle forense completo: `INFORME_FORENSE_MAESTRO.md` (ADENDA 22 = lo último).

---

## 2026-09-29 — Claude (cloud): ciclos 2 a 6 de vetos, bloqueos y límites

Rama `claude/auditoria-deslizamiento-apalancamiento-sqtc08` (GLM la borra del
remoto tras cada merge; se vuelve a crear con el mismo nombre). Cada hallazgo
viene de la auditoría con 7 auditores y 2 escépticos, y cada arreglo tiene un
test que falla en `main` antes del arreglo.

- **En `main`**: ciclo 2 por el PR #17 (CL-3…CL-7), ciclos 3 y 4 por el PR
  #18 (fa1adae6, CL-8…CL-20). T-1 16/144 en ambos.
- **Ciclo 5 (núcleo, riesgo y evolución)**, T-1 17/144 (11,8 %):
  - CL-21: el núcleo publica `ml_model_base` (FMT-159). `flow_excitation` caía a 0,5 frente a bosques con base 0,18–0,30 y votaba corto casi siempre.
  - CL-22: la rama 13 ya no lleva un segundo enfriamiento (`can_open_at_tau(macro_tau, 120 s)`, artefacto de merge) además del de `puertas_del_continuo`.
  - CL-23: el examen del daemon sólo lleva series que el juez juzga (≥ 60); en la zona 40–59 no juzgaba nada y sumaba 2 001 pruebas al DSR.
  - CL-24: el umbral corto del pre-examen es el espejo del largo (el gen 23 daba siempre 0).
  - CL-25: el rollback vuelve al primer genoma distinto del linaje (`destino_de_rollback`); antes re-promovía el mismo genoma.
  - CL-26: la geometría TP/SL usa el Hurst DFA de 1 min, no el proxy por eventos (S-7 había deshecho D-615b).
  - CL-27: el examen walk-forward juzga barras de mercado de 16 s por moneda, no los deltas de PnL del incumbente (FMT-049).
  - CL-28: la Fisher de escala es telemetría en el daemon (en ruido su mediana queda bajo el umbral; aplazaba casi toda ronda).
  - CL-29: se retira la puerta heurística del incumbente (FMT-055): cerraba la evolución justo cuando el genoma vivo pierde.
- **Ciclo 6 (espectro, en curso)**:
  - CL-30: la persistencia se mide sobre retornos de bloques NO solapados de duración ≥ τ. Antes daba ≈ +0,94 en una caminata aleatoria (y +0,996 en un zigzag que revierte), y todos sus lectores leen 0 como browniano: Kelly S-1, BE/trailing, consejo, Hurst por banda y la masa de la fusión.
  - CL-31: la rama 15 trata la persistencia sin lado (`confluencia_resonante`, con `hurst_at`). Antes comparaba [−1,1] con 0,52/0,48 y sólo emitía largos en la zona moderada.
  - CL-32: las escalas por debajo del intervalo medio entre eventos no votan (`resolucion_efectiva_ms`, como XLIV-6). A 1 evento/s se llevaban el 68 % del peso de la fusión y el 47 % de la masa espectral.
- Abiertos (confirmados, sin arreglar todavía):
  - ~~`dominant_tau_ms` sigue siendo el argmax de |w·s| recortado a [30 s, 12 h]. Con pesos 1/vol, cae en la escala resuelta más rápida y se queda en 30 s. Elegir la escala dominante por habilidad medida exige otro criterio (p. ej. `SpectralForecastBank`).~~ **CERRADO por #594 (Qoder)**: τ dominante = escala con mayor IC prequential > 0. Queda abierto que el umbral es IC > 0 sin significancia ni corrección por el número de escalas maduras (inferido, sin medir: en ruido el máximo de varias IC suele ser positivo).
  - Rama 13/15: suelos literales de confianza (0,55/0,58); B1 OFI tóxico muerto en el núcleo; B3 Coaxial vota 0 y el gen 83 no tiene consumidor.
  - La adopción al arrancar sigue con `entry_time_ms = now`; la distancia del trailing sigue en ATR de 1 min.
  - Libro sintético de ±1 pb en eventos que no son de depth; gate de viabilidad ATR inalcanzable; veto de drawdown absorbente en backtest; `suelo_tp_sl` y la banda del genoma miden stops distintos.
  - R8-A (primer toque de barrera) sigue abierto.
- Descartados tras verificar: freno CL-2 con el ATR que se cancela (física del stop); `rama_abierta` por moneda (OA de Codex); god_engine.rs:4116; `crash_pressure` como máximo de cartera (diseño XLIV-4, política del dueño).

## 2026-09-28 (noche) — Claude (cloud): fricción, freno y auditoría de vetos

Flujo nuevo pedido por el operador: rama con nombre del agente → merge a
`main` resolviendo conflictos → borrar la rama. Mi rama:
`claude/auditoria-deslizamiento-apalancamiento-sqtc08`. Prefijo de mis
bloques: **CL-n** («CL» = Claude; «Ola-20/XLV·A…F» es de otro agente).

- **CL-1** (fallback de gestión del núcleo con la ley lineal de latencia):
  NO entra por esta rama. La otra sesión Claude lo portó como XLIV-8c en el
  PR #10 (rama elegant-euler), abierto. Para no duplicar, queda en ese PR.
- **CL-2**: el freno de volatilidad (D-746) medía contra el stop de la
  curva del gen en la τ del gen; ahora el apalancamiento se decide DESPUÉS
  de la geometría y recibe `expected_loss` (stop real de la orden).
- **CL-2b**: T-1 19/144 → 16/144; trinquete re-certificado a 11,0 %
  (autorizado). Pierde 142/143 (`sl_horizon_curve`, sensibilidad falsa: su
  única lectura efectiva era el stop imaginario) y 20 (`veto_threshold_btc`,
  enmascarado por la cuantización entera del apalancamiento).
- Verificación: 7 crates `--all-targets` en verde bajo wine; T-1 release.
- Estado Git verificado: todo lo de los PR #5–#8, #11 y #12 está en
  `main`; el PR #10 (elegant-euler) sigue abierto. `backup-before-cleanup` y
  `v7-unificacion-wip` (con commits exclusivos según Codex) ya no existen
  en el remoto: si alguien los necesita, están en su clon local.
- Visto de los otros agentes: `effective_bets` (8389432c) sigue sin
  conectar; `hawkes_cross`, `amplificar_por_contagio` (correlation_guard) y
  `contagion_modulator` (signal-engine) sólo tienen definición y tests, sin
  consumidor operativo. No los toco (zonas Codex/Qoder).
- Coordinación: hay DOS sesiones Claude. La de `elegant-euler` (PR #10) y
  ésta. Antes de corregir algo, `git fetch` y revisar el PR #10 y el buzón
  `COORDINACION_CODEX_2026-09-28.md` para no duplicar.
- En curso: auditoría de vetos, bloqueos, límites y rechazos con panel de
  7 auditores + 2 escépticos por hallazgo. Resultados en el siguiente
  bloque o merge.

## 2026-09-28 — Claude (cloud): auditoría de gates de evolución y régimen (PR #8)

> Estado: los dos primeros tramos (XLIV-1…12) están en `main` (PR #8,
> 09245261). El tercero (XLIV-8c, XLIV-13) va en un PR nuevo desde la misma
> rama, reiniciada sobre `main`.

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
  calentamiento con log1p/expm1 y reloj W₁ monótono. XLIV-3 entró en `main`
  (41755422) sin ese arreglo; **CERRADO por el PR #11 de Codex** (fbf299ee /
  b3ba8d80, en `main` desde 92534a9e): semillas de entropía en 0, dominio de
  momentos cerrado y reloj W₁ causal.

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
  T-1 (release, 30 min): sobre d6e5aba1 (XLIV-8/9/10) y sobre el head
  completo ef172aac (≡ 4dac8061), 19/144 = 13,2 % ≥ trinquete 11,5 %, con la
  MISMA lista de genes inertes que el primer tramo, gen a gen.
- **Lección de build**: tras medir T-1 sobre `main` en el mismo directorio, el
  artefacto release de quantum-arena quedó OBSOLETO con fecha más nueva que
  las fuentes restauradas (carrera checkout ↔ compilación): la siguiente
  build release falló con «no field cvd_ewma_peso». Tras cambiar de commit
  para medir, `touch` a los .rs que difieren (o `cargo clean -p`) antes de
  volver a compilar. La build debug no estaba afectada.

### Tercer tramo (tras el merge del PR #8)
- **XLIV-8c**: XLIV-8 olvidó un quinto sitio con la ley lineal de latencia,
  el fallback de gestión del núcleo (`fee_rt_mgmt`). Portado de XLV-1 (PR #9
  de otra sesión Claude, cerrado sin fusionar por solaparse con el #8) junto
  con su guardia sobre las fuentes: función única en los cuatro archivos y
  prohibido `lat_ref`/`latency_ref_ms`.
- **XLIV-13 train_forest (hallazgo de Codex)**: el merge 6fdccd64 del PR #5
  dejó los helpers FMT definidos pero SIN llamadas en `main()`. Verificado:
  `--promote` escribía el modelo VIVO con evidencia de SELECCIÓN (sin
  `--test-in`); el stride efectivo estaba invertido (excedía el presupuesto
  en tapes largos y densificaba ×4 en cortos); `--val-in` suponía el no
  solape sin comprobarlo; la validación no puntuaba el artefacto
  serializado. Recompuesto sin revertir D-753/D-733: contrato de promoción
  antes de E/S, `SamplingBudget`, `purge_training` para `--val-in`,
  `serving_predictions` + control de paridad con `forest_raw`, y test
  posterior (`require_later_holdout`) que también debe pasar el gate.
- **XLIV-13b (revisión de Codex)**: `purge_end` (split interno 80/20) usaba
  frontera abierta (`t + τ > t_val`) y los contratos FMT cerrada; unificado a
  `>=` (López de Prado). El test D-734 pasa de 5 a 6 purgadas y compara con
  `purge_training`.
- **main roto en f0fcf08a (XLV·F)**: `contagion_modulator.rs` importaba
  `crate::hawkes_cross`, que vive en feature-engine ⇒ signal-engine no
  compilaba (`check --all-targets` rojo en 99a19bfb). Arreglado con la misma
  línea (`feature_engine::hawkes_cross`) en el PR #10 (3e7f00bb) y, en
  paralelo, por Codex en el PR #12 (0f31d628, ya en `main` e67a4e2f).
- Abierto (R8-A, Codex): el motivo de salida vivo no reconstruye el primer
  toque de las barreras del entrenador cuando los brackets difieren
  (contraejemplos suyos): XLIV-9c sigue siendo una aproximación.

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
- ~~**D-746 freno de apalancamiento**: mide el ATR contra `sl_at_tau`
  genómico, no contra el stop real del gate.~~ **CERRADO por CL-2** (PR #13
  de la otra sesión Claude, en `main` a3d2eb06): el apalancamiento se decide
  tras la geometría y recibe el stop real de la orden.
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

## 2026-09-28 — Codex: integración EWMA/W1 y publicación parcial verificable

- 6b7c6d37 creado y PUSH confirmado en main con SOLO 31 tests pendientes de
  correlación/admisión. Junto a los 13 de e7bb4c59, las 44 regresiones previas
  están publicadas. Informe de admisión y JSON seguían sin publicar.
- PR #8 head 47a93fe6 revisado (9 diffs), comentarios iniciales sin hilos
  pendientes; checks vacíos NO son CI verde. Otra sesión inició merge local:
  MERGE_HEAD=47a93fe6, 9 archivos staged; no hubo conflictos de índice.
  Codex preserva ese índice, no finaliza/aborta un merge ajeno ni hace add -A.
- Parche adicional WORKING-TREE en state.rs semillas de entropía cero,
  calibration.rs masa/momentos válidos y calentamiento estable log1p/expm1,
  core/lib.rs reloj W1 monótono y transaccional, posterior lognormalizado y
  cola saturada conservando masa. Mis fuentes son diff POSTERIOR al índice.
  Sin editar genomas, Hawkes, fixture T-1, goldens ni presupuestos operativos.
- 8 contraejemplos RED→GREEN reproducidos. 13 tests nuevos pasan (7 EWMA,
  5 temporales y 1 masa privado). Core+arena+risk all-targets: 616 pasan,
  0 fallan, 1 inventario local ignorado. Evolution all-targets: 100 pasan,
  0 fallan, 0 ignoradas. Workspace check all-targets pasa con advertencias.
  Suite de cinco crates sigue en oráculo T-1 largo; NO declararla aprobada
  hasta ver resultado final. Otro editor corre cargo test --workspace;
  no detenerlo ni atribuir sus resultados a Codex.
- Informe nuevo docs/AUDITORIA_INTEGRACION_EWMA_Y_RELOJ_2026-09-28.md y JSON
  homónimo en docs/artifacts; 15 hallazgos con fórmulas, topología y cierre.
  Siguen OPEN Platt sin edad individual, Fisher/anclas fijas, hazard por
  evento, riesgo monetario/reserva atómica y propagación del estado/edad.
  Publicadores de régimen pasan None para deriva: no alimentan esa componente.
  Un extremo por gen NO demuestra inercia global; T-1 tiene alcance condicionado.
- Atlas, maestro e informe de admisión ampliados por adendas. Coordinación:
  buzón local y comentario PR #8 /issuecomment-5877204118. Sin operación,
  promoción, entrenamiento, despliegue ni garantía de +100%/72h.

- DECISIÓN DEL USUARIO posterior: GLM debe terminar el merge actual. Codex
  no lo cierra/aborta ni toca su índice. Manifest explícito dejado en el buzón.
  Revalidación manual FMT-037/190: siguen 8 bosques JSON rechazados por hijos
  fuera del árbol, 12 aceptados sólo estructuralmente y 1 de otro esquema.
  Los 12 aceptados tienen 2 hashes distintos (10 FDUSD iguales + 2 BTCUSDT
  iguales). No se activaron modelos ni se escribieron caches o modelos.

- CIERRE LOCAL 15:18: cinco crates all-targets, 777 pasan/0 fallan/1 inventario
  ignorado, código 0. T-1: 2 pasan, 1405,43 s, fixture/umbral/goldens intactos.
  El inventario se ejecutó aparte (no equivale a modelos válidos). 9 hashes
  de fuente y 21 de modelos verificados. No pruebas propias pendientes.
- Corte remoto: main=6b7c6d37; Claude/PR #8 avanzó a 6ebb2857 con 3c5b0c1a y
  6ebb2857. MERGE_HEAD sigue 47a93fe6. Los 777 NO prueban ese delta remoto.
  Diff nuevo leído, no aplicado: signo terminal != etiqueta primer toque de
  barrera; fricción compartida hereda ATR/latencia no finitos→0. GLM avisado.
  No nuevas operaciones Git de escritura al índice, merge ni borrado de ramas.

## 2026-09-28 — Codex: checkout aislado y publicación EWMA/W1 pendiente

- El usuario cambia el protocolo a rama propia + merge + eliminación segura.
  GLM ya cerró el merge previo en 41755422; ahora trabaja en
  glm/xlv-effective-bets / random_matrix.rs. No se toca ese archivo.
- Codex crea worktree gestionado codex-ewma-w1-audit y rama
  codex/ewma-w1-audit desde 41755422. Snapshot explícito de 16 archivos;
  cinco hashes de fuentes/tests idénticos a los de 777 tests históricos.
  Los originales del checkout compartido se preservan; NO duplicar su commit.
- Compilación desde caché independiente iniciada; resultado se añadirá al cerrar.
  No se confunde la prueba anterior con los nuevos commits del PR #8.
- R8-A sigue OPEN en df01c82c: reason_code tampoco reconstruye primer toque.
  Dos trayectorias con brackets RR=2 producen etiqueta opuesta; el trainer
  aggTrades usa TP/SL literales .0036/.0018, no genes. Detalle §18 del informe.
- Aviso local añadido. Comentario GitHub no publicado (403 y bloqueo de
  autorización externa); se pide permiso explícito para ese repositorio/contenido.
  No push/merge remoto ni eliminación de ramas acreditados en este corte.
- Inventario base: 1309 archivos, 400 Rust, 24 manifests; 78 archivos Rust
  con scalp/swing. Es inventario léxico, NO cobertura ni 78 motores separados.

- Check aislado workspace all-targets pasó (6m54s, advertencias). Pruebas de
  core/risk/arena en curso. Se descubre regresión semántica del merge 6fdccd64:
  main() perdió consumidores de holdout/purga (FMT-193), presupuesto (194)
  y evaluación del artefacto (190); helpers/tests conservados. Reabiertos,
  NO reparados aquí. Evidencia y criterios en informe EWMA/W1 §20.


- CIERRE AISLADO: commits locales fbf299ee (EWMA, 3 archivos) y b3ba8d80
  (W1, 2 archivos). Reejecución core/risk/arena all-targets: 616 pasan,
  0 fallan, 1 inventario ignorado, 49 bloques, código 0. Check workspace
  all-targets pasa; cinco hashes de fuentes/tests sin cambios. Sin pruebas
  propias pendientes. 777/T-1 son históricos, no reejecución de esta pasada.
- Main remoto verificado 7da858ef; Claude 4833d45c (delta sólo memoria).
  Mi rama base 41755422 NO incluye los dos nuevos commits de main. Sus tests
  no prueban ese delta. GLM-RMT-A confirmado en ejecutable aislado sobre blob
  2d1413207425a19f4ef74a642431d256e5e87355: 4*I aceptada por full_spectrum
  y effective_bets=3, largest_eigenvalue la rechaza. Archivo GLM intacto.
- Informes/JSON ampliados, XXI actualizado por adenda de reapertura. Sin
  push/PR/merge/borrado: autorización externa pendiente tras 403/auto-revisión.
  No usar otro canal para eludirla. Buzón local notificado; no asumir acuse.
  Conservar worktree y originales compartidos. Próximo: reconciliar main,
  reparar consumidores del trainer/RMT sin revertir los cambios de otros,
  validar árbol final y publicar cuando exista autorización explícita.

## 2026-09-28 — Codex: autorización recibida y reparación RMT

- Usuario autorizó destino Jhona-la/Trader-Gemini, push/PR/merge y eliminación
  sólo de la propia rama integrada. Aviso publicado PR8 issuecomment-5879667925;
  el bloqueo anterior es histórico. No se autoriza operación/promoción por ello.
- Merge propio 09c86127 incorpora 7da858ef tras comparar ambos padres y check
  workspace all-targets aprobado (37,58 s). PR8 se fusionó externamente en
  09245261 y GLM avanzó main a 5cdc0fe0: falta validar esa combinación.
- 57bae732: largest_eigenvalue/full_spectrum comparten validador/Jacobi.
  Nueva suite: rojo 5 pasan/2 fallan; verde 7 pasan/0 fallan. Se precisa
  effective_bets como participación de modos retenidos, no dimensión real
  de la cartera. No se conecta el placeholder ni se modifica sizing.
- Auditoría detallada §23–25 del informe EWMA/W1 y JSON: Hawkes cuenta
  ventanas con algún evento, no un kernel de intensidad; referencia/nulo,
  censura, selección de lag y amplificación sin contrato estadístico abiertos.
  GLM avisado localmente; no presumir lectura ni modificar su evolución.
- Claude propone reparación del trainer en PR10/e1edc1b3. Consumidores
  recuperados por lectura del diff; NO duplicar la implementación.
  Se comenta frontera interna estricta > frente a contrato cerrado >=:
  issuecomment-5879876894. R8-A (primer toque) sigue abierto.
- Pendiente: integrar último main, check antes del merge, pruebas del árbol
  final, publicar PR propio y verificar SHA remoto antes de retirar rama.
  No confundir 616/777 históricos con pruebas de nuevos commits.

- PR11 publicado en draft, head 32e04af3 sobre e2e0c8ef. Check workspace
  all-targets pasó (30,93 s); cinco crates all-targets: 844 pasan/0 fallan/
  1 inventario ignorado, 65 bloques. Incluye 20 regresiones propias, sin T-1.
- Diagnóstico ejecutado de matriz Hawkes (blob 02e892bd): vacíos→ceros;
  roles NaN→Some(NaN). GLM confirmó recepción y corrige en 5109f357.
  Falta integrar/probar ese último delta. Media por N-1 NO resuelve
  calibración/soporte ni vuelve el z invariante al universo. Siguen OPEN.
- Claude respondió y c7da6e70 corrige purge_end cerrado en PR10, aún abierto
  al consultar. Su reporte 39/39 es evidencia del autor, no ejecución Codex.

- VALIDACIÓN FINAL PRE-MERGE PR11: 8ac27783 incorpora 5109f357. Check
  workspace all-targets pasa (34,89 s). Reejecución mismos cinco crates:
  844/0/1 ignorada, 65 bloques. Backtest lib: 31/0/0 (19,84 s). Total875/0/1;
  20 regresiones propias incluidas; T-1 no reejecutado. Sin tests pendientes.
- Testigos Hawkes sobre blob024bb3d5: vacíos/NaN ya None. Series de dos
  eventos aún Some(ceros); suma de entradas MAX finitas puede dar inf en
  roles. Calibración/soporte siguen abiertos; sin consumidor operativo hallado.
- Informe §28 registra árbol probado y alcance; consultar PR11 para recibo
  de merge remoto posterior, no confundir este corte previo con fusión.

## 2026-09-28 — Codex: recibo de PR11 y limpieza segura

- PR11 merged=true, 22:50:17Z; main92534a9ee7e394fe08264b43319dec7544d5c216.
  c90679dc es ancestro, árbol idéntico. EWMA/W1/RMT e informes integrados;
  875/0/1 pruebas, alcance §28. Rama fuente local/remota retirada después.
- Otra operación cerró PR11 y borró head sin merge a22:47:08/09Z. Se recuperó
  desde el checkout aislado y reabrió. Cuenta compartida: no atribuir agente.
  No limpiar refs activas sin ancestralidad exacta y coordinación del autor.
- Worktree conservado; recibo documental en codex/ewma-w1-receipt. PR10 aún
  OPEN al consultar; backups exclusivos/originales compartidos preservados.
  Sin operaciones, entrenamiento/promoción ni garantía de rentabilidad.

## 2026-09-28 — Codex: PR12 ampliado por rotura posterior de main

- Main99a19bfb/f0fcf08a agregó contagion_modulator con import inexistente.
  E0432 reproducido en cargo check signal-engine. Corregido con
  feature_engine::hawkes_cross::ContagionRole, sin cambiar fórmula/consumidores.
- Merge0f31d628 conserva main99a, diff contra ambos padres revisado y check
  workspace all-targets pasa (19,25s). Seis crates all-targets925/0/1,
  70 bloques; backtest lib31/0/0 (16,65s). Total956/0/1, sin T-1/operación.
- PR12 ya NO es sólo docs: recibo y fix de import. Informe§30–31/JSON.
  GLM luego reconoció borrado de rama; PR11 sí se recuperó en el mismo PR.
- Checkout compartido: cuatro documentos UU tras stash, índice ajeno intacto.
  Se consulta al usuario un único integrador. No confundir este conflicto
  local con el merge remoto exitoso ni duplicar código de PR11.

## 2026-09-28 — Codex: Platt con Newton factible (CAL-01/02/03)

- Rama propia codex/calibration-clock-audit desde a3d2eb06, worktree reutilizado.
  deeca0c6: solver con frontera correcta, descenso Armijo, pérdida estable y
  scores de entrenamiento válidos. Antes2/5fallan; después7/0 y octava prueba
  adicional KKT; núcleo lib127/0. No se altera política temporal ni gates.
- 67c63d32 incorpora main3cdbb6b3. E0308 en nuevo lector Hawkes reproducido;
  fix mínimo &sym. Ambos padres revisados; check all-targets16,62s pasa.
  Seis crates all-targets en curso; no confundir con resultados históricos.
- Auditoría nueva docs/AUDITORIA_CALIBRACION_PLATT_2026-09-28.md y JSON.
  Edad por dato, fallback local/global, sesgo de selección, τ y convergencia
  siguen OPEN. No operación, entrenamiento/promoción ni garantía financiera.
- Coordinación PR10 issuecomment-5881142305 /5881232741. Claude mantiene trainer.
  Checkout compartido pasó de78c81804 con marcadores a3cdb limpio; Codex no
  tocó su índice. Backups locales conservados; no refs remotas observadas.

- CIERRE PRE-MERGE PR14: fuentes67c63d32, main3cdb incorporado. Seis crates
  all-targets934/0/1 y backtest lib31/0/0:965/0/1; ocho regresiones propias
  ya incluidas. Check workspace all-targets pasa16,62s; sin T-1 ni operación.
  Informe Platt§16 y JSON guardan alcance/hashes. Pruebas propias finalizadas.
  PR10 sigue abierto; backups locales3/1 commits exclusivos preservados.
  Recibo remoto posterior y limpieza de rama: consultar PR14, no asumirlos
  por este resultado previo a la fusión.

## 2026-09-28 — Codex CF: contrato conformal por activo

- 836aa9dc desde mainfa23: target local, telemetría de la misma instancia,
  dominio válido y diagnóstico antes del veto de entradas (veto preservado).
  11 regresiones; RED controlado3/7, GREEN10/0 y undécima diagnóstica.
- Seis crates945/0/1 +backtest31/0/0 =976/0/1; check all-targets53,99s pasa.
  Sin T-1/operación/promoción; no modificar golden ni límites para hacer pasar.
- Informe docs/AUDITORIA_CONFORMAL_CONTRATO_2026-09-28.md y JSON,14 hallazgos.
  ACI recortado no hereda la cota original; singleton no garantiza error
  condicionado. Delay/target/tau/feedback/ausencia de evidencia siguen abiertos.
- Rama propia codex/conformal-contract-audit en curso hasta recibo remoto.
  GLM observado en glm/xlv-health-check, Claude PR10c844d4fd. Avisos PR10
  5882626900/5882772751. Conservar ramas activas y backups exclusivos.

## 2026-09-28 — Codex OA: atribución causal y contrato numérico

- Fuentes4737f95b; integración5c4a8c7e con maincf5c. Ocho fallos reparados,
  18 regresiones nuevas: snapshots por activo/slot/generación/símbolo/tiempo,
  crédito rama/bosque y dominio numérico/softmax de modelos participantes.
- Seis crates963/0/1 +backtest31/0/0 =994/0/1; check all-targets26,71s pasa.
  Sin T-1/promoción/live; no se desactivó ningún control de riesgo.
- Informe docs/AUDITORIA_APRENDIZAJE_CAUSAL_2026-09-28.md y JSON:17 hallazgos,
  8 reparados/9 abiertos. Pendientes tau slot2, max(ID), target/horizonte,
  versión durable, garantías estadísticas, namespace/frescura Hawkes y ledger.
- GLM caller cf5c incorporado; namespace aún distinto, avisado PR10
  5883007522/5883110364. Claude mantiene PR10c844d4fd; no acuse observado.
- Rama codex/outcome-clock-audit propia hasta recibo remoto. Backups con3/1
  commits exclusivos y ramas activas preservados. No declarar todo auditado
  ni merged; JSON conserva snapshot previo y recibo posterior confirma Git.

## 2026-09-30 — GO: diagnóstico y contrato de la medición genética

- Rama propia codex/genome-oracle-evidence desde76de9935. Sin producción,
  genoma/fixture/predictor/comparador/trinquete modificados. T-1 completo no ejecutado.
- Corrige trades/PnL/WR del diagnóstico (RED6/1 → GREEN9/0) y registra
  petición/proyección efectiva. Preservación bit a bit de fixture y144 destinos.
- Ruido negativo, extremos sin prueba de inercia, acoplamientos, stats≠fitness,
  no finitos y cobertura monoactivo siguen abiertos: informe GO+JSON detallados.
- Ampliada117/0/2; ignoradas sólo mediciones manuales. CI agrega target rápido,
  no cambia golden ni ignora T-1 para ocultar el fallo informado por GLM.
- GLM revisó MRf3145e27: COMMENTED favorable con condición documental legible.
  Maincdab sólo registra esa revisión. MR/MP/GO son ramas y permisos distintos.
- GO local, publicación específica pendiente; MP40f8685b preservada local.
  No trading/training/promoción. Checkout compartido y PR20 ajenos intactos.
- Check GO workspace/all-targets/locked confirmado en38,18s, sin errores.

- ACTUALIZACIÓN: GLM fusionó MR/PR23 en589a591d y cumplió la condición
  documental enfbf8e9ea; remoto main verificado. Codex no duplicó ese trabajo.
  CI nueva MR/main seguía en curso después del merge: no certificar gate.
- GO integra mainfbf8e9ea localmente, preservando ambos apéndices de tres
  conflictos y comparando ambos padres. Check27,29s pasa. Se mantiene
  separación de permisos/publicaciones GO/MP y PR10/20 pendientes.
- CIERRE integrado: registry9 + replay117 =126/0/2; refuerzo de longitud
  del contrato de fixture validado9/0 y all-targets5,64s. No T-1 extenso.
  JSON conserva hashes previos/finales. MR/main CI todavía en curso;
  no se certifica el gate pre-merge. MP/GO sin referencias remotas.

## 2026-09-30 — Publicación MP y GO autorizada

- Operador: «Hazlo» en respuesta a publicación en repo PÚBLICO mediante
  dos PR separadas. Levanta el bloqueo anterior de MP/GO; no pedir de nuevo.
- Mantener CI del candidato y revisión cruzada antes de cada merge.
  No autoriza trading, training, promoción ni rebajar el trinquete T-1.
- GO9cda4677 conserva hashes validados126/0/2; MP40f8685b debe reconciliar
  mainfbf8e9ea y validar su composición antes de publicar. No mezclar ramas.

## 2026-09-30 — MP/GO publicación autorizada; MP integrada localmente con MR

- Operador respondió Hazlo a publicación pública MP/GO en dos PR con CI
  y revisión cruzada. Levanta bloqueo anterior; no autoriza operación/modelos.
- GO publicada PR24,3ba8c2d6, fuente9cda4677 validada126/0/2. No está en MP.
- MP40f8685b incorpora mainfbf8e9ea (MR integrado por GLM). Cinco conflictos
  de documentación/CI resueltos conservando ambos padres en orden; hashes
  MP originales intactos. Check26,60s pasa, regresión integrada en curso.
- MR/PR23 y main CI aún en curso al verificar; merge no acredita gate.
  PR10/20 abiertas,20 draft. Checkout compartido ajeno intacto.
- CIERRE MP integrado:203/0/1 núcleo/suites +108/0/2 replay =311/0/3.
  Cinco tests MR adicionales explican aumento desde306; no inflar fixes.
  Hashes fuente MP idénticos; check26,60s. Publicar PR propia, sin GO dentro,
  y esperar CI/review del candidato. Sin training, operación o T-1 completo.

## 2026-09-30 — MP-08: caché inválida que bloqueaba fuente válida

- MP PR25/e69797e1 y GO PR24/34324749 reconcilian main6228, sólo coordinación.
  Ambos padres íntegros/en orden; checks all-targets5,46s/26,67s. Sin review.
- QA MP reproduce RED13/1: BIN deserializable con ciclo impide JSON válido.
  Validación movida dentro de aceptación BIN; fallback sólo a JSON validado.
  Ambas fuentes inválidas preservan modelo/archivos; JSON nuevo inválido no
  hace rollback a BIN viejo. Errores conservan causas de caché y fuente.
- GREEN14/0; seis clases inválidas x2 rutas dentro de UN test. Check final
  all-targets25,41s; núcleo/suites207/0/1, replay todavía en curso en este corte.
  Informe/JSON MP§16–17;3 candidatos/5 abiertos, no8 bugs cerrados.
- MR integrado por GLM, CI36725338162 SUCCESS posterior al merge. Nuevo
  aislamiento GLM CX verde/combinado rojo acota efecto marginal; falta dato
  por gen para re-baseline. No tocar rama/procesos GLM ni PR10/20 de Claude.
- Main compartido limpio; ramas activas/backups no integrados preservados.
  No training/promoción/operación/T-1/golden/umbral cambiado ni garantía de lucro.

- CIERRE MP-08: código6ec084e9,207/0/1 núcleo +108/0/2 replay =315/0/3;
  cuatro nuevos tests MP incluidos, no sumar otra vez12 casos internos.
  Check25,41s; hashes re-verificados. Informe§18/JSON qa_mp08_final.
  Publicar en PR25 y pedir review del nuevo SHA; no atribuirle CI del padre.
  GO34324749 permanece en PR24 separada, sin código productivo cambiado.

## 2026-10-01 — Antigravity: Ola 8 — Teorema Analítico de Hodge O(N^2) Zero-Alloc, Reversión Adaptativa en Cointegración y Densidad Espectral O(1)

- **hodge.rs**: Teorema analítico cerrado de Helmholtz-Hodge sobre grafos completos K_n:
  L = n·I - 1·1^T. Con div ortogonal al kernel (Σ div_i = 0), el potencial es φ = div / n.
  La energía de Dirichlet del gradiente satisface idénticamente ‖∇φ‖² = (1/n) · Σ div_i².
  Erradicación total de eliminación gaussiana O(N³), pivoteo y alocaciones de heap. Complejidad
  reducida a O(N²) evaluando cada par una sola vez con buffer de stack para N ≤ 64 (roster ≤ 16).
  Tests: 6/6 verdes en risk-engine + nuevo test `test_hodge_analytical_invariance_multi_asset`.
- **multivariate_coint.rs**: Erradicación del literal fijo `expected_magnitude = 0.015`.
  Se reemplaza por la magnitud medida real del proceso Ornstein-Uhlenbeck:
  `expected_magnitude = (|z_score| * std_dev).clamp(0.002, 0.20)`. Alineación con
  `trajectory_auditor.rs` y con la economía real del EV vs fees. Nuevo test:
  `test_multivariate_cointegration_measured_expected_magnitude`.
- **temporal_spectrum.rs**: Optimización de `continuous_energy_density`: cálculo nodal O(1)
  evaluando únicamente los dos nodos de interpolación activos (i0, i1) en vez de evaluar
  32 exponenciales para toda la malla. Reutilización de `pesos_espectrales()` en
  `micro_resonant_tau_ms` y `macro_resonant_tau_ms` mediante `continuous_resonant_tau_ms_with_pesos`.
- **Verificación**: `cargo check --workspace --all-targets` limpio (1m 28s). Tests de
  risk-engine (111 tests), strategy-core (36 tests) y quantum-arena pasan al 100%.

## 2026-10-01 — Antigravity: Ola 9 — Prequential Volatility Memory Decay, Zero-Alloc Group Risk & Cramér-Lundberg Ruin Bounds, Spectral Hawkes Continuity

- **spectral_tape.rs**: Prequential scoring adaptativo con decaimiento de memoria exponencial (`clim_lambda`) en `HorizonForecaster`. Erradica la petrificación no-ergódica de SSE acumulada de por vida, permitiendo que el R² prequencial refleje la capacidad predictiva reciente del horizonte temporal en vez de acumular indefinidamente shocks del pasado remoto.
- **correlation_guard.rs**: Eliminación de alocación en heap (`Vec<f64>`) en `veto_por_riesgo_real_medido`. Cálculo de suma, suma cuadrática y peor individual en streaming de pasada única O(N) sin heap allocation. Función expuesta `calcular_riesgo_grupo`. Nueva compuerta `veto_por_riesgo_cramer_lundberg` integrando el coeficiente de ajuste de Lundberg R y la cota de supervivencia ψ(m) ≤ e^{-Rm} con el tope de racha Bernoulli.
- **flow_excitation_confluence.rs**: Continuidad espectral y eliminación de escalón rígido en la escala de auto-excitación de Hawkes. Normalización dinámica por el umbral efectivo del proceso (`effective_hawkes_thresh`), permitiendo una modulación suave y monótona sin chattering ni saltos abruptos.
- **Verificación**: Tests en quantum-arena (5/5), risk-engine (113/113), signal-engine (74/74) e integración correlation_admission_contract (26/26) al 100% verdes. `git diff --check` verificado con 0 advertencias de fin de línea.
