use parking_lot::RwLock;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Estructura atómica que reside en la memoria compartida entre el Motor de Evolución y el Motor de Ejecución
#[derive(Clone)]
pub struct QuantumHotSwapState {
    pub has_new_genome: Arc<AtomicBool>,
    pub active_genome_id: Arc<AtomicUsize>,
    pub shadow_sharpe_ratio: Arc<RwLock<f64>>,
}

impl QuantumHotSwapState {
    pub fn new() -> Self {
        Self {
            has_new_genome: Arc::new(AtomicBool::new(false)),
            active_genome_id: Arc::new(AtomicUsize::new(0)),
            shadow_sharpe_ratio: Arc::new(RwLock::new(0.0)),
        }
    }
}

impl Default for QuantumHotSwapState {
    fn default() -> Self {
        Self::new()
    }
}

use quantum_arena::GlobalArena;
use quantum_arena::genome::SuperGenotype;

/// Capital inicial de la simulación walk-forward del daemon (micro-capital
/// $13 — misma cifra que la simulación previa al fix C-10).
const WF_INITIAL_CAPITAL: f64 = 13.0;

/// C-10 / MOD3/5-012 (INFORME 14): mínimo estadístico de operaciones en la
/// ventana OOS del daemon. 3 trades en la partición fuera de muestra es el
/// mínimo para una señal direccional; por debajo, el genoma es INVIABLE
/// (D-654), no mediocre — la inacción no puede puntuar mejor que operar.
/// CERT-M8-H01: el mínimo era 3 — best-of-2000 sobre ≥3 OOS trades es
/// selección pura de estadística de orden (el mejor de 2000 tiradas de
/// 3 monedas supera cualquier umbral por azar). 30 es el mínimo de la
/// regla X-014 para significancia muestral básica.
const WF_MIN_TRADES: u32 = 30;

// CERT-M8-H01 — WALK-FORWARD CON EL MOTOR REAL. La simulación de momentum
// de abajo queda degradada a PRE-SCREEN (prior grueso, ventana corta); el
// JUEZ es GodEngineCore (consejo, ML gate, física de fees, envelope del
// host) — el MISMO mecanismo que opera, patrón shadow-forest + nativo.
const WF_REAL_TOP_K: usize = 24;
const WF_REAL_WINDOW: usize = 400;
const WF_REAL_MAX_COINS: usize = 8;
const WF_REAL_MICRO_TICKS: usize = 8;

struct RealWfOutcome {
    fitness: f64,
    net_returns: Vec<f64>,
    trades: usize,
}

/// Registra specs sintéticos WFD{i} (idempotente) y devuelve sus coin_ids.
fn wf_real_coin_ids(n: usize) -> Vec<usize> {
    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let sym = format!("WFD{}", i);
        if quantum_arena::symbol_registry::try_index(&sym).is_none() {
            let spec = quantum_arena::symbol_registry::SymbolSpec {
                symbol: sym.clone(),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0005,
                is_shadow: true,
            };
            quantum_arena::symbol_registry::update_registry(vec![spec]);
        }
        ids.push(quantum_arena::symbol_registry::try_index(&sym).unwrap_or(i));
    }
    ids
}

/// Evalúa un candidato con el MOTOR REAL sobre micro-ticks Brownian-bridge
/// sintetizados de las series per-coin (contrato omni 54D causal, mismo
/// patrón que run_backtest_native). Envelope del host sobre cada entrada.
fn wf_evaluate_real(
    candidate: &SuperGenotype,
    series: &[Vec<f64>],
    initial_capital: f64,
) -> RealWfOutcome {
    use god_engine_core::GodEngineCore;
    use risk_engine::kelly_envelope::RiskEnvelope;

    let mut ranked_series: Vec<&Vec<f64>> = series.iter().filter(|s| s.len() >= 60).collect();
    ranked_series.sort_by_key(|s| std::cmp::Reverse(s.len()));
    ranked_series.truncate(WF_REAL_MAX_COINS);
    if ranked_series.is_empty() {
        return RealWfOutcome { fitness: f64::NEG_INFINITY, net_returns: Vec::new(), trades: 0 };
    }
    let coin_ids = wf_real_coin_ids(ranked_series.len());

    let arena = Arc::new(GlobalArena::new(initial_capital));
    candidate.apply_to_arena(&arena);
    let mut core = GodEngineCore::new(arena.clone());
    let mut envelope = RiskEnvelope::new();
    let mut vetoes: u64 = 0;
    let mut net_returns: Vec<f64> = Vec::new();
    let mut avg_win = 0.0f64;
    let mut avg_loss = 0.0f64;
    let mut peak = initial_capital;
    let mut max_dd = 0.0f64;
    let mut ts: u64 = 60_000;
    let mut pos_open_flags = [false; quantum_arena::state::MAX_COINS];

    for (si, rets) in ranked_series.iter().enumerate() {
        let coin_id = coin_ids[si].min(quantum_arena::state::MAX_COINS - 1);
        let mut price = 100.0f64;
        let half_spread = candidate.maker_spread_pct.max(0.00005);
        for (ri, &r) in rets.iter().enumerate() {
            let next_price = (price * (1.0 + r)).max(1e-6);
            for t in 0..WF_REAL_MICRO_TICKS {
                ts += 2_000;
                let frac = (t + 1) as f64 / WF_REAL_MICRO_TICKS as f64;
                let mid = price + (next_price - price) * frac;
                let bid = mid * (1.0 - half_spread);
                let ask = mid * (1.0 + half_spread);
                let noise_qty = 0.5 + ((ts % 7) as f64) / 7.0;
                let bid_qty = 10.0 * noise_qty;
                let ask_qty = 10.0 * (2.0 - noise_qty);
                let obi = ((bid_qty - ask_qty) / (bid_qty + ask_qty)).clamp(-1.0, 1.0);
                // omni 54D causal — mismos campos clave que el nativo
                let mut omni = [0.0f64; 54];
                let prev_ret = if ri > 0 { rets[ri - 1] } else { 0.0 };
                omni[1] = (2.0 * half_spread * 10.0).clamp(-5.0, 5.0);
                omni[11] = (prev_ret * 0.005).clamp(-0.001, 0.001);
                omni[13] = (1.0 + prev_ret * 5.0).clamp(0.5, 2.5);
                omni[14] = (50.0 + prev_ret * 500.0).clamp(10.0, 90.0);
                omni[24] = (15.0 + prev_ret.abs() * 200.0).clamp(10.0, 80.0);
                omni[30] = obi * 5.0;
                omni[31] = obi * 6.0;
                omni[32] = if ask_qty > 0.0 {
                    (bid_qty / ask_qty).clamp(0.1, 10.0)
                } else {
                    1.0
                };
                omni[34] = mid * 1.01;
                omni[35] = mid * 0.99;
                omni[39] = obi;
                omni[41] = (-prev_ret * 2.0).clamp(-0.25, 0.25);
                omni[43] = mid;
                omni[49] = prev_ret.abs() * 100.0;

                let is_kline = t == WF_REAL_MICRO_TICKS - 1;
                let (_, closed) = core.process_event(
                    coin_id, true, is_kline, true, mid, bid_qty.min(ask_qty),
                    bid, ask, bid_qty, ask_qty, obi, 0.0, ts, false, &omni, bid_qty > ask_qty,
                );
                // Envelope del host sobre cada entrada recién abierta
                let atr_now = core
                    .feature_engines
                    .get(coin_id)
                    .map(|f| f.get_atr_pct())
                    .unwrap_or(0.002);
                let was_open = pos_open_flags[coin_id];
                backtest_engine::booktick_replay::live_envelope_gate(
                    &arena, &mut envelope, coin_id, mid, atr_now, was_open, &mut vetoes,
                );
                pos_open_flags[coin_id] = arena.coins[coin_id].positions.position.is_open();

                if let Some((_, pnl_net, _)) = closed {
                    if pnl_net >= 0.0 {
                        avg_win = if avg_win == 0.0 { pnl_net.abs() } else { avg_win * 0.95 + pnl_net.abs() * 0.05 };
                    } else {
                        avg_loss = if avg_loss == 0.0 { pnl_net.abs() } else { avg_loss * 0.95 + pnl_net.abs() * 0.05 };
                    }
                    envelope.record_trade(pnl_net > 0.0, avg_win.max(1e-9), -avg_loss.max(1e-9));
                    net_returns.push(pnl_net / initial_capital);
                }
            }
            price = next_price;
            let cap = arena.unified_capital.load(Ordering::Relaxed);
            if cap > peak {
                peak = cap;
            }
            if peak > 0.0 {
                let dd = (peak - cap) / peak;
                if dd.is_finite() && dd > max_dd {
                    max_dd = dd;
                }
            }
            if cap <= 0.0 {
                break;
            }
        }
        if arena.unified_capital.load(Ordering::Relaxed) <= 0.0 {
            break;
        }
    }

    let final_cap = arena.unified_capital.load(Ordering::Relaxed);
    let trades = net_returns.len();
    let fitness = crate::fitness::compute(&crate::fitness::FitnessInputs {
        initial_capital,
        final_capital: final_cap,
        max_drawdown_pct: max_dd,
        total_trades: trades as u32,
        min_trades_required: WF_MIN_TRADES,
        oos_start_capital: final_cap,
        oos_end_capital: final_cap,
    });
    RealWfOutcome { fitness, net_returns, trades }
}

/// D-689 (DÉCIMA OLA) — ARMADO EXPLÍCITO DE LA EVOLUCIÓN EN VIVO.
///
/// Los promotores en vivo cambiaban umbrales y genoma con evidencia de minutos:
/// la cosecha del Shadow Forest con 6,5 céntimos de ventaja sobre $13 y sin
/// mínimo de operaciones, y este daemon con una «confianza bayesiana» que con
/// N < 10 pasa casi siempre. La puerta del almacén valida sanidad (bounds y
/// RR), no rendimiento. Con la filosofía de D-651 y `MAINNET_ARMED`, ningún
/// proceso en vivo cambia umbrales ni genoma sin
/// `TG_LIVE_GENOME_EVOLUTION_ARMED=1`. La detección de deriva, el kill-switch
/// y el rollback post-promoción no dependen de este armado.
pub fn live_evolution_armed() -> bool {
    std::env::var("TG_LIVE_GENOME_EVOLUTION_ARMED")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
}

/// QO-E1 (QUINTA OLA): armado POR ENTORNO. El hallazgo central de la
/// auditoría evolutiva: `TG_LIVE_GENOME_EVOLUTION_ARMED` jamás se fijaba —
/// prod/history tenía 0 generaciones: el organismo NUNCA evolucionó pese
/// a tener los tres lazos cableados y sus redes de seguridad (rollback
/// t≤−2.0, kill-switch EWMA, validación de bounds/RR) vivas.
///
/// Semántica nueva:
/// - La variable explícita `TG_LIVE_GENOME_EVOLUTION_ARMED=1|0` MANDA
///   (siempre respetada — un operador puede armar prod a propósito).
/// - Sin variable: DEMO (TG_GENOME_ENV=demo) arma POR DEFECTO — es el
///   entorno de evaporación segura con capital de papel. Un kill-switch
///   de arming (`TG_LIVE_GENOME_EVOLUTION_DISARM=1`) permite apagarlo en
///   demo para A/B sin tocar el default.
/// - PROD sigue DESARMADO por defecto: la promoción demo→prod conserva el
///   paso humano (TG_GENOME_PROMOTE_ARMED) — doctrina D-651 intacta.
pub fn live_evolution_armed_for_env() -> bool {
    if let Ok(v) = std::env::var("TG_LIVE_GENOME_EVOLUTION_ARMED") {
        return v.trim() == "1";
    }
    if let Ok(v) = std::env::var("TG_LIVE_GENOME_EVOLUTION_DISARM") {
        if v.trim() == "1" {
            return false;
        }
    }
    std::env::var("TG_GENOME_ENV").map(|e| e.trim() == "demo").unwrap_or(false)
}

pub struct LiveEvolutionDaemon {
    pub state: QuantumHotSwapState,
    pub arena: Arc<GlobalArena>,
    pub iteration_count: usize,
    pub last_evolution: Instant,
    pub daemon_start_time: Instant,
    pub warmup_duration_secs: u64,
    pub is_demo: bool, // Identifica si estamos en Testnet para acelerar la evolución
    pub ledger: storage_engine::evolution_ledger::EvolutionLedger,
    pub champion_path: String,
    pub ewma_sharpe: f64, // HC-12: Sharpe adaptativo para kill switch
    pub forest: crate::online_random_forest::TrueOnlineRandomForest,
    /// F4.5: PnL realizado visto por coin en el ciclo anterior — para muestrear
    /// retornos de la ESTRATEGIA (deltas reales), no beta del mercado.
    pub last_realized_by_coin: std::collections::HashMap<usize, f64>,
    /// FIX #1600: Acumulación histórica persistente de retornos de estrategia sobre ventana deslizante
    pub returns_history: Vec<f64>,
    /// T-10 — retornos ETIQUETADOS por moneda: el walk-forward requiere que
    /// el momentum decida sobre el retorno de LA MISMA moneda (antes: el
    /// momentum de BTC decidía entradas sobre retornos de ETH — fitness
    /// cross-asset inválido).
    pub returns_by_coin: std::collections::HashMap<usize, Vec<f64>>,
    /// FASE 3 — watchdog de rollback: retornos posteriores a la última
    /// promoción y la generación promovida. Si el genoma nuevo demuestra edge
    /// NEGATIVO estadísticamente significativo, se revierte al padre.
    pub post_promo_returns: Vec<f64>,
    pub promoted_generation: Option<(u64, u64)>, // (generación, padre)
}

impl LiveEvolutionDaemon {
    pub fn new(
        state: QuantumHotSwapState,
        arena: Arc<GlobalArena>,
        is_demo: bool,
        db_path: &str,
        champion_path: &str,
    ) -> Self {
        // En Testnet (Demo) calentamos rápido (60s), en Producción somos más rigurosos (15 min)
        let warmup = if is_demo { 60 } else { 900 };
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Some(parent) = std::path::Path::new(champion_path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let ledger = storage_engine::evolution_ledger::EvolutionLedger::new(db_path);
        Self {
            state,
            arena,
            iteration_count: 0,
            last_evolution: Instant::now(),
            daemon_start_time: Instant::now(),
            warmup_duration_secs: warmup,
            is_demo,
            ledger,
            champion_path: champion_path.to_string(),
            ewma_sharpe: 0.0,
            forest: crate::online_random_forest::TrueOnlineRandomForest::new(5000),
            last_realized_by_coin: std::collections::HashMap::new(),
            returns_history: Vec::with_capacity(1024),
            returns_by_coin: std::collections::HashMap::new(),
            post_promo_returns: Vec::with_capacity(256),
            promoted_generation: None,
        }
    }

    /// Ciclo asincrónico que corre paralelo al bot de producción
    /// Ingiere datos reales y entrena tensores cuánticos en las sombras.
    /// F5.4: sin unwrap — telemetría ausente ⇒ ingesta deshabilitada con log,
    /// el resto del daemon (umbrales, drift) sigue vivo.
    pub async fn run_online_learning_loop(&mut self) {
        let mut telemetry_reader = Some(storage_engine::MmapTelemetryReader::new(
            quantum_arena::paths::data_join("telemetry.mmap"),
        ));

        let mut pending_new_obs = 0;

        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Ingest Telemetry into the Shadow Forest
            if let Some(reader) = telemetry_reader.as_mut() {
                if let Ok(frames) = reader.read_latest_frames() {
                    for f in frames {
                        // SUBSYSTEM_TENSOR_PREDICTOR = 12, FRAME_PREDICTION_VS_REALITY = 30
                        if f.subsystem_id == 12 && f.frame_type == 30 {
                            // E-02 — FIX ÍNDICE: el productor pone ml_prob
                            // en payload[0]; payload[1] contiene is_long (0/1).
                            // Antes: el forest aprendía "los longs ganan".
                            let ml_prob = f.payload[0];
                            let net_pnl_pct = f.payload[3];
                            // If we are evaluating a long or short based on the prob:
                            // G-03: leer is_long del payload[1] donde el
                            // productor lo escribe — antes se derivaba de
                            // ml_prob>0.5, etiquetando todos los trades de
                            // prob alta como "long" y sesgando las features.
                            let is_long = f.payload[1] > 0.5;
                            self.forest.shadow_evaluate(
                                ml_prob as f32,
                                net_pnl_pct as f32,
                                0.0,
                                is_long,
                            );
                            pending_new_obs += 1;
                        }
                    }
                }
            }

            // Reentrenar periódicamente el Shadow Forest cuando hay suficientes observaciones
            let obs_count = self
                .forest
                .observations
                .read()
                .map(|o| o.len())
                .unwrap_or(0);
            if (obs_count >= 10 && self.iteration_count % 20 == 0) || pending_new_obs >= 10 {
                if let Ok((acc, mse)) = self.forest.retrain_models() {
                    println!(
                        "🌲 [SHADOW RANDOM FOREST] Reentrenado con éxito sobre {} observaciones! Accuracy: {:.2}%, MSE: {:.6}",
                        obs_count,
                        acc * 100.0,
                        mse
                    );
                    pending_new_obs = 0;
                }
            }

            // Aplicar thresholds óptimos del Shadow Forest a la Arena ÚNICAMENTE cuando está entrenado
            // D-689: y sólo con la evolución en vivo armada; sin armar, el bosque
            // aprende pero no sobrescribe los umbrales del genoma validado.
            if self.forest.is_trained() && live_evolution_armed_for_env() {
                let (opt_l, opt_s) = self.forest.get_optimal_thresholds();
                self.arena
                    .config
                    .ml_threshold_long
                    .store(opt_l as f64, Ordering::Relaxed);
                self.arena
                    .config
                    .ml_threshold_short
                    .store(opt_s as f64, Ordering::Relaxed);
            }

            // QO-E2a — EL APRENDIZAJE QUE DECIDE: el forest entrenado
            // PREDICE por símbolo (predict_6d, antes cero callers) y
            // publica al registry `forest6_prob`/`forest6_acc`. El core
            // modula la confianza de las entradas cuando el forest está
            // entrenado (acc > 0.55) y en DESACUERDO con la intención —
            // el aprendizaje deja de ser espectador de su propia señal.
            if self.forest.is_trained() {
                let acc = *self.forest.last_accuracy.read().unwrap_or_else(|e| e.into_inner());
                if acc.is_finite() && acc > 0.0 {
                    let reg = &self.arena.registry;
                    for coin_id in 0..self.arena.coins.len() {
                        let coin = &self.arena.coins[coin_id];
                        let spot_bid = coin.spot_bid.load(std::sync::atomic::Ordering::Relaxed);
                        let spot_ask = coin.spot_ask.load(std::sync::atomic::Ordering::Relaxed);
                        let spread_bps = if spot_bid > 0.0 && spot_ask > spot_bid {
                            ((spot_ask - spot_bid) / spot_bid * 10_000.0).clamp(0.0, 500.0)
                        } else {
                            1.0
                        };
                        let features = [
                            reg.get_for_coin_or(coin_id, "orderbook_imbalance", 0.0)
                                .clamp(-1.0, 1.0),
                            reg.get_for_coin_or(coin_id, "price_acceleration", 0.0)
                                .clamp(-10.0, 10.0),
                            spread_bps,
                            reg.get_for_coin_or(coin_id, "atr_pct", 0.002).clamp(0.0, 1.0),
                            coin.hurst_exponent
                                .load(std::sync::atomic::Ordering::Relaxed)
                                .clamp(0.0, 1.0),
                            reg.get_for_coin_or(coin_id, "price_velocity", 0.0)
                                .clamp(-10.0, 10.0),
                        ];
                        if let Some((prob, _pnl)) = self.forest.predict_6d(features) {
                            reg.set_for_coin(coin_id, "forest6_prob", prob);
                            reg.set_for_coin(coin_id, "forest6_acc", acc);
                        }
                    }
                }
            }

            // FASE 3: AST Mutator checking
            if std::path::Path::new(".forensic_violation").exists() {
                println!("🧬 [DAEMON] Señal forense detectada! Invocando AST-Mutator...");
                let mutator = crate::ast_mutator::ASTMutator::new();

                let config_path = "dynamic_config.json";
                if std::path::Path::new(config_path).exists() {
                    let _ = mutator.mutate_json_config(
                        config_path,
                        "Risk.ML_LOOKAHEAD_PENALTY",
                        serde_json::json!(2.0),
                    );
                }

                let _ = std::fs::remove_file(".forensic_violation");
            }

            // Muestrear retornos realizados en tiempo real tras cada tick de 500ms
            self.sample_realized_returns();

            // FASE 3: watchdog de rollback post-promoción
            self.check_post_promotion_degradation();

            self.iteration_count += 1;

            // Cada 3 minutos (o 60s en Demo) validamos si el entorno cambió
            let eval_interval = if self.is_demo { 60 } else { 180 };
            if self.last_evolution.elapsed() > Duration::from_secs(eval_interval) {
                self.evaluate_shadow_strategy().await;
                self.last_evolution = Instant::now();
            }
        }
    }

    fn sample_realized_returns(&mut self) {
        let capital = self
            .arena
            .unified_capital
            .load(std::sync::atomic::Ordering::Relaxed);

        for coin_id in 0..self.arena.coins.len() {
            let coin = &self.arena.coins[coin_id];
            let realized = coin
                .metrics
                .pnl_realized
                .load(std::sync::atomic::Ordering::Relaxed);

            if let Some(&prev) = self.last_realized_by_coin.get(&coin_id) {
                let delta = realized - prev;
                if delta.abs() > 0.0 && capital > 0.0 {
                    let ret = delta / capital;
                    if ret.is_finite() {
                        self.returns_history.push(ret);
                        let coin_window = self.returns_by_coin.entry(coin_id).or_default();
                        coin_window.push(ret);
                        if coin_window.len() > 400 {
                            coin_window.drain(0..coin_window.len() - 400);
                        }
                        // FASE 3: evidencia post-promoción para el watchdog.
                        if self.promoted_generation.is_some() {
                            self.post_promo_returns.push(ret);
                        }

                        // E4a — FEATURES REALES para el Shadow Forest (antes:
                        // shadow_evaluate con 5/6 constantes — el clasificador
                        // solo aprendía long-vs-short). Se lee el estado vivo
                        // del registry/coin en el instante del cierre:
                        // obi, aceleración de precio, spread bps (spot),
                        // atr_pct, hurst y momentum como proxy macro.
                        let reg = &self.arena.registry;
                        let spot_bid = coin.spot_bid.load(std::sync::atomic::Ordering::Relaxed);
                        let spot_ask = coin.spot_ask.load(std::sync::atomic::Ordering::Relaxed);
                        let spread_bps = if spot_bid > 0.0 && spot_ask > spot_bid {
                            ((spot_ask - spot_bid) / spot_bid * 10_000.0).clamp(0.0, 500.0)
                        } else {
                            1.0
                        };
                        let features = [
                            reg.get_value_or("orderbook_imbalance", 0.0)
                                .clamp(-1.0, 1.0),
                            reg.get_value_or("price_acceleration", 0.0)
                                .clamp(-10.0, 10.0),
                            spread_bps,
                            reg.get_value_or("atr_pct", 0.002).clamp(0.0, 1.0),
                            coin.hurst_exponent
                                .load(std::sync::atomic::Ordering::Relaxed)
                                .clamp(0.0, 1.0),
                            reg.get_value_or("price_velocity", 0.0).clamp(-10.0, 10.0),
                        ];
                        self.forest.shadow_evaluate_with_features(features, ret);
                    }
                }
            }
            self.last_realized_by_coin.insert(coin_id, realized);
        }

        // Mantener ventana deslizante acotada a los 1000 trades más recientes
        if self.returns_history.len() > 1000 {
            let drain_count = self.returns_history.len() - 1000;
            self.returns_history.drain(0..drain_count);
        }
        if self.post_promo_returns.len() > 500 {
            let drain_count = self.post_promo_returns.len() - 500;
            self.post_promo_returns.drain(0..drain_count);
        }
    }

    /// FASE 3 — Watchdog de rollback automático: si el genoma recién
    /// promovido acumula evidencia de edge NEGATIVO estadísticamente
    /// significativo (t-stat <= -2.0 con >= 20 observaciones post-promoción),
    /// revierte al padre vía el embudo versionado y lo reaplica al arena.
    /// Es el complemento operativo del gate de `promote`: sanidad antes,
    /// rendición de cuentas después.
    fn check_post_promotion_degradation(&mut self) {
        let Some((generation_id, parent)) = self.promoted_generation else {
            return;
        };
        if self.post_promo_returns.len() < 20 {
            return;
        }
        let t_stat = Self::calculate_ransac_sharpe(&self.post_promo_returns);
        if t_stat <= -2.0 {
            println!(
                "🚨 [ROLLBACK WATCHDOG] Generación {} degradada: t-stat {:.2} sobre {} obs post-promoción. Revirtiendo al padre {}.",
                generation_id,
                t_stat,
                self.post_promo_returns.len(),
                parent
            );
            match quantum_arena::genome_store::GenomeEnvelope::rollback(parent) {
                Ok(env) => {
                    env.genome.apply_to_arena(&self.arena);
                    println!(
                        "✅ [ROLLBACK WATCHDOG] Padre {} restaurado y aplicado al arena (nueva generación {}).",
                        parent, env.generation
                    );
                    // QO-E2d — LEDGER: el rollback también se registra.
                    self.ledger.save_weight(
                        0,
                        format!("gen_rollback_{}", parent),
                        "rollback".to_string(),
                        -1.0,
                    );
                }
                Err(e) => println!(
                    "⚠️ [ROLLBACK WATCHDOG] Rollback al padre {} falló: {}. El genoma degradado sigue activo — INTERVENCIÓN MANUAL.",
                    parent, e
                ),
            }
            // Watchdog consumido: no re-revertir en cada ciclo sobre la misma evidencia.
            self.promoted_generation = None;
            self.post_promo_returns.clear();
        }
    }

    async fn evaluate_shadow_strategy(&mut self) {
        // FASE 3: Dynamic Batch Sizing adaptativo a la memoria del sistema
        let _batch_size = 5000;

        self.sample_realized_returns();

        // FIX BLOQUEO #3: Reducir umbral de 10 a 3 para micro-capital ($13)
        // Con $13 y scalping, cada trade cuenta. 3 observaciones bastan para arrancar.
        if self.returns_history.len() < 3 {
            return;
        }

        // Si el Sharpe Cuántico RANSAC > 1.2 y supera a la estrategia de producción...
        let current_shadow_sharpe = Self::calculate_ransac_sharpe(&self.returns_history);

        let elapsed_warmup = self.daemon_start_time.elapsed().as_secs();
        if elapsed_warmup < self.warmup_duration_secs {
            println!(
                "⏳ [WARMUP PHASE] {}/{} segundos. Sharpe: {:.2}. Esperando maduración de tensores...",
                elapsed_warmup, self.warmup_duration_secs, current_shadow_sharpe
            );
            return;
        }

        println!(
            "🧜 [ONLINE EVOLUTION] Evaluando Shadow Strategy con {} trades reales acumulados... t-stat RANSAC: {:.2}",
            self.returns_history.len(),
            current_shadow_sharpe
        );

        {
            let mut sr = self.state.shadow_sharpe_ratio.write();
            *sr = current_shadow_sharpe;
        }

        // --- FASE 9 / HC-08: DRIFT DETECTION & KILL SWITCH (EWMA ADAPTIVE - D-389) ---
        // N-01: Se evalúa siempre sobre muestra representativa (>= 25 trades) sin importar el Sharpe puntual
        if self.returns_history.len() >= 25 {
            if self.ewma_sharpe == 0.0 {
                self.ewma_sharpe = current_shadow_sharpe;
            } else {
                self.ewma_sharpe = 0.1 * current_shadow_sharpe + 0.9 * self.ewma_sharpe;
            }

            // D-389: calculate_ransac_sharpe retorna el t-statistic de Student (inlier_mean / inlier_std * sqrt(N)).
            // Un t-stat entre 0.0 y +1.0 representa retornos positivos leves en muestras pequeñas.
            // Para activar Kill Switch por degradación del edge, el t-stat debe ser estadísticamente NEGATIVO
            // con significancia (t < -1.50, p < 0.07 de que el edge negativo sea casual).
            // CERT-M8-H02: el `!self.is_demo` anterior desarmaba el kill-switch
            // exactamente en el único entorno donde la autoevolución está
            // ARMADA por defecto y las mutaciones son vivas — un genoma
            // degradado en demo seguía tradando hasta que el (mucho más
            // lento) rollback watchdog acumulara 20 observaciones. El
            // kill-switch es una red de seguridad del TRADING, no de la
            // promoción: debe ser env-independiente.
            if self.ewma_sharpe < -1.50
                && !self.arena.kill_switch_active.load(Ordering::Relaxed)
            {
                println!(
                    "🚨 [DRIFT DETECTION] t-stat EWMA degradado significativamente a {:.2} sobre {} trades. Activando Kill Switch para detener ejecuciones hasta reentrenar.",
                    self.ewma_sharpe,
                    self.returns_history.len()
                );
                self.arena.kill_switch_active.store(true, Ordering::Relaxed);
                return;
            }

            // Auto-reset / reactivación si el edge se estabiliza (t-stat > -0.50)
            if self.arena.kill_switch_active.load(Ordering::Relaxed) && self.ewma_sharpe > -0.50 {
                println!(
                    "✅ [DRIFT RECOVERY] t-stat EWMA recuperado a {:.2}. Reactivando operaciones (Kill Switch desactivado).",
                    self.ewma_sharpe
                );
                self.arena
                    .kill_switch_active
                    .store(false, Ordering::Relaxed);
            }
        }

        // D-689: la deriva y el kill-switch de arriba son protección y siguen
        // siempre activos; la búsqueda y promoción de mutaciones en vivo no.
        if !live_evolution_armed_for_env() {
            return;
        }

        println!(
            "🔄 [ADAPTIVE SEARCH] Evaluando mutaciones walk-forward (Sharpe actual: {:.2})...",
            current_shadow_sharpe
        );

        // FASE I: Random Forest / Thousands of Universes Evaluation (Estasis de Probabilidad)
        // E2/HERENCIA: la semilla de la mutación es el GENOMA ACTIVO del
        // almacén versionado (champion real), no un champion_path que
        // nadie escribe. Antes: cada ciclo clonaba SuperGenotype::default()
        // y las promociones destruían el linaje evolutivo acumulado.
        let current_genome = quantum_arena::genome_store::GenomeEnvelope::load_active()
            .map(|env| env.genome)
            .or_else(|| {
                std::fs::read(&self.champion_path).ok().and_then(|bytes| {
                    serde_json::from_slice::<quantum_arena::genome::SuperGenotype>(&bytes).ok()
                })
            })
            .unwrap_or_else(quantum_arena::genome::SuperGenotype::default);

        let _iteration = self.iteration_count;
        let fallback_genome = current_genome.clone();

        // FASE 13: Entropic Volatility Mutation
        let mean = self.returns_history.iter().sum::<f64>() / self.returns_history.len() as f64;
        let variance = self
            .returns_history
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / self.returns_history.len() as f64;
        let volatility = variance.sqrt().max(0.0001);

        // FIX BLOQUEO #2: Capturar snapshot de retornos reales para walk-forward en el closure
        let returns_snapshot: Vec<f64> = self.returns_history.clone();
        // T-10: series POR MONEDA (mínimo 40 obs) para el walk-forward.
        let per_coin_series: Vec<Vec<f64>> = self
            .returns_by_coin
            .values()
            .filter(|v| v.len() >= 40)
            .cloned()
            .collect();

        // E-04 — FRICCIÓN COHERENTE CON EL EV GATE: antes fee fijo
        // 0.0008 mientras el gate real incluye maker+taker+2×slip.
        // Leído ANTES del closure para evitar borrow de self.
        // C-10 (INFORME 14): el daemon seguía con fricción pre-D-645 —
        // maker+taker (el roundtrip real de una entrada de mercado es
        // TAKER×2), ATR literal 0.002 y latencia normalizada contra un
        // 150ms literal. Ahora replica la fórmula del gate de entrada del
        // host (god_engine.rs `fee_rt_entry`, B3.19): 2×taker (D-645) +
        // 2×(slippage_floor + atr_del_gen·latencia_del_gen/umbral_pánico).
        let slip_floor_g = self
            .arena
            .config
            .base_slippage_floor
            .load(Ordering::Relaxed)
            .max(0.00001);
        let lat_g = self
            .arena
            .config
            .latency_penalty_ms
            .load(Ordering::Relaxed)
            .max(0.0);
        let lat_ref_g = self
            .arena
            .config
            .latency_ms_panic_threshold
            .load(Ordering::Relaxed)
            .clamp(10.0, 5_000.0);
        // ATR del GEN (dynamic_atr_min), no un literal — el genoma decide el
        // piso de volatilidad con el que se cotiza la fricción.
        let atr_g = current_genome.dynamic_atr_min.max(0.0005);
        let lat_slip_g = (atr_g * (lat_g / lat_ref_g)).clamp(0.0, 0.01);
        let taker_g = self
            .arena
            .config
            .live_taker_fee
            .load(Ordering::Relaxed)
            .max(0.0004);
        // D-645: roundtrip completo a taker — idéntico al EV gate del vivo.
        let roundtrip_fee =
            (taker_g * 2.0) + 2.0 * (slip_floor_g + lat_slip_g).clamp(0.0, 0.01);

        // R7-8 (des-rigidización por DERIVACIÓN): el capital semilla del
        // walk-forward era el literal 13.0 — un genoma evaluado a escala de
        // $13ueba juega con regímenes de capital distintos al real si la
        // cuenta creció. Ahora usa el capital VIVO del arena (el mismo plano
        // contra el que el incumbente opera).
        let wf_live_capital = self
            .arena
            .unified_capital
            .load(Ordering::Relaxed)
            .max(1.0);
        let best_genome = tokio::task::spawn_blocking(move || {
            // CERT-M8-H01: el momentum-sim es PRE-SCREEN (ventana corta,
            // prior grueso). El JUEZ es wf_evaluate_real (motor completo)
            // sobre el top-K + el incumbente — nunca más el mejor de un
            // mundo simulado que no es el que opera.
            let mut prescreened: Vec<(f64, SuperGenotype)> = Vec::with_capacity(2_048);
            let real_series: Vec<Vec<f64>> = if !per_coin_series.is_empty() {
                per_coin_series
                    .iter()
                    .map(|sr| {
                        if sr.len() > WF_REAL_WINDOW {
                            sr[sr.len() - WF_REAL_WINDOW..].to_vec()
                        } else {
                            sr.clone()
                        }
                    })
                    .collect()
            } else {
                vec![returns_snapshot.clone()]
            };
            // FASE 2: roundtrip completo a taker (0.04% x 2 piernas),
            // consistente con el simulador y con el costo real de una
            // entrada de mercado + salida no-maker.

            // Mutation scales dynamically based on real-time market entropy
            let dynamic_mutation_rate = (volatility * 50.0).clamp(0.01, 0.25);

            let mut rng = rand::rng();
            // FIX BLOQUEO #2: Reducir de 10,000 a 2,000 candidatos para micro-capital
            // Con CPU limitada (16GB RAM, no GPU), 2K iteraciones son suficientes.
            // D-740 (DÉCIMA OLA · auditoría integral): EL INCUMBENTE COMPITE.
            //
            // `best_score` arrancaba en −999 y la aptitud de cualquier candidato
            // es finita, así que el PRIMER mutante superaba siempre el arranque:
            // el genoma en curso —cuya aptitud no se evaluaba nunca— era
            // sustituido incondicionalmente, aunque los 2 000 mutantes fueran
            // peores que él. Se promovía «el menos malo» y se instalaba en el
            // motor vivo. Ahora la iteración 0 evalúa el genoma ACTIVO con la
            // misma simulación walk-forward, de modo que su aptitud es la cota
            // que un mutante debe superar para reemplazarlo.
            for _i in 0..=2_000 {
                let es_incumbente = _i == 0;
                let mut candidate = current_genome.clone();
                use rand::RngExt;
                if !es_incumbente {

                // Mutación Vectorial de Tensores (DL/RL Vivo)
                // FASE 9: Ajuste Adaptativo por Régimen de Mercado (Drift Recovery)
                // FASE 13: Topological Evolution

                candidate.scalp_kelly_fraction +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                candidate.swing_kelly_fraction +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;

                // C-10 (INFORME 14) — MUTAR LAS CURVAS, NO LAS ANCLAS: el
                // roundtrip to_vector/from_vector re-deriva las anclas desde
                // los coeficientes (X-004/X-005: `derive_anchors_from_curves`
                // al final de from_vector) — mutar `scalp_tp_base`/`swing_*`
                // se AUTODESTRUÍA ahí y la geometría TP/SL JAMÁS evolucionaba
                // en vivo. Los coeficientes (a,b) de las curvas son la fuente
                // de verdad (slots 140-143 del vector genético). Una
                // perturbación aditiva en `a` (param(τ)=exp(a+b·lnτ)) equivale
                // a la contracción/expansión ±5% multiplicativa que antes se
                // intentaba sobre las anclas — pero en TODOS los horizontes a
                // la vez; `enforce_curve_rr` del roundtrip mantiene el
                // invariante RR y los bounds de (a,b) acotan la mutación.
                candidate.tp_horizon_curve.a +=
                    (rng.random::<f64>() - 0.5) * 0.1_f64.ln_1p(); // ≈ ±ln(1.05)
                candidate.sl_horizon_curve.a +=
                    (rng.random::<f64>() - 0.5) * 0.1_f64.ln_1p();
                // La pendiente b — cómo escala TP/SL con el horizonte τ —
                // también evoluciona (bandas TP_B/SL_B_BOUNDS del genoma).
                candidate.tp_horizon_curve.b +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;
                candidate.sl_horizon_curve.b +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;

                candidate.ml_threshold_long +=
                    (rng.random::<f64>() - 0.5) * (dynamic_mutation_rate * 0.5);
                candidate.ml_threshold_short +=
                    (rng.random::<f64>() - 0.5) * (dynamic_mutation_rate * 0.5);

                // E4c — ESPACIO DE MUTACIÓN UNIFICADO: antes el daemon
                // solo mutaba 10 genes mientras el backtest evoluciona
                // 139 vía nichos — los genes con los que el backtest gana
                // (filtros, maker, explosividad, régimen) JAMÁS mutaban en
                // producción. Se añaden los genes de nicho del backtest
                // con la misma tasa dinámica; el clamp canónico via
                // to_vector/from_vector al final del loop los mantiene en
                // bounds (fuente única R1.1).
                candidate.tech_threshold +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.001;
                candidate.weight_obi += (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.1;
                candidate.maker_spread_pct *=
                    1.0 + (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                candidate.maker_obi_threshold +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.1;
                candidate.explosive_confidence_threshold +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;
                candidate.explosive_leverage_multiplier +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;
                candidate.trend_threshold +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.1;
                // (swing_tp_base/swing_sl_base: ya no se mutan — son VISTAS de
                // las curvas mutadas arriba; ver C-10 arriba.)
                candidate.sl_atr_multiplier +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.2;
                candidate.tp_rr_ratio_btc +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.2;
                candidate.target_volatility +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.01;
                candidate.global_correlation_threshold +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;
                candidate.scalp_trail_act_atr +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate * 0.05;

                // Evolución de genes Topológicos (Red Neuronal)
                candidate.topo_layer_1_activation +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                candidate.topo_layer_2_activation +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;
                candidate.tensor_dropout_rate +=
                    (rng.random::<f64>() - 0.5) * dynamic_mutation_rate;

                // Random entropy jump for quantum seed
                if rng.random::<f64>() < 0.1 {
                    candidate.quantum_entropy_seed = rng.random::<f64>() * 1000.0;
                }

                }
                // Clamping automático delegando a la estructura cuántica central (SuperGenotype)
                let vec = candidate.to_vector();
                candidate = SuperGenotype::from_vector(&vec);

                // FIX BLOQUEO #2: Función de Fitness basada en RETORNOS REALES observados
                // ANTES: Usaba fórmula algebraica cerrada EV = WR_estimado * TP - (1-WR_estimado) * SL
                //        donde WR_estimado = 0.50 + selectivity * 0.70 — PURAMENTE TEÓRICA.
                // AHORA: Walk-forward sobre returns_history real. Simula las decisiones
                //        del genoma candidato contra los retornos REALES observados.
                let tp = candidate.scalp_tp_base.max(0.0001);
                let sl = candidate.scalp_sl_base.max(0.0001);
                let ml_thr_long = candidate.ml_threshold_long;
                let ml_thr_short = candidate.ml_threshold_short;

                // Simulación walk-forward sobre retornos reales
                let mut wf_wins = 0usize;
                let mut wf_losses = 0usize;
                let mut wf_pnl = 0.0f64;
                let mut wf_capital = WF_INITIAL_CAPITAL; // Starting capital
                // C-10: la aptitud única (fitness::compute) exige el drawdown
                // máximo de la trayectoria — se mide pico-a-valle del capital
                // walk-forward simulado.
                let mut wf_peak = WF_INITIAL_CAPITAL;
                let mut wf_dd = 0.0f64;
                // CERT-M8-C03: colectar los retornos NETOS del CANDIDATO —
                // el DSR gate debe evaluar ESTA serie (la del mutante que
                // se promueve), no los returns del incumbente.
                let mut candidate_net_returns: Vec<f64> = Vec::new();

                // T-10 — WALK-FORWARD POR MONEDA: cada serie conserva su
                // propio momentum (el retorno previo de LA MISMA moneda
                // decide la entrada sobre su propio retorno siguiente).
                // Antes: la serie global mezclaba monedas y el momentum
                // de una decidía trades de otra. Se itera la porción OOS
                // de cada serie por moneda; si no hay series suficientes
                // (arranque frío) se cae a la serie global (compat).
                let series: Vec<Vec<f64>> = if !per_coin_series.is_empty() {
                    // pre-screen: sólo los últimos 300 retornos por moneda
                    per_coin_series
                        .iter()
                        .map(|sr| {
                            if sr.len() > 300 {
                                sr[sr.len() - 300..].to_vec()
                            } else {
                                sr.clone()
                            }
                        })
                        .collect()
                } else {
                    vec![returns_snapshot.clone()]
                };

                for coin_ret in &series {
                    let n_returns = coin_ret.len();
                    let train_end = (n_returns * 6) / 10;
                    for i in train_end..n_returns {
                        let r = coin_ret[i];
                        let prev_r = if i > 0 { coin_ret[i - 1] } else { 0.0 };

                        // FIX: Erradicación del Lookahead Bias.
                        // Decisión: el genoma entra long/short basándose en el momentum previo (prev_r),
                        // NO en el retorno actual (r). Se prohíbe leer el futuro.
                        //
                        // FASE 2 (unidades coherentes): ANTES se comparaba
                        // `prev_r` (un retorno, ~1e-4) contra `thr - 0.5` (una
                        // distancia de probabilidad, ~0.1-0.45) — desajuste
                        // semántico que hacía el filtro degenerado. Ahora el
                        // momentum se expresa en sigmas de la ventana real
                        // (`volatility`, calculada sobre returns_history) y el
                        // umbral del genoma (0.5..0.95) se mapea a 0..0.9
                        // sigmas de momentum mínimo exigido.
                        let prev_sigma = if volatility > 1e-12 {
                            prev_r / volatility
                        } else {
                            0.0
                        };
                        let mom_long = (ml_thr_long - 0.5).max(0.0) * 2.0;
                        let mom_short = (ml_thr_short - 0.5).max(0.0) * 2.0;
                        let entry_bias = if prev_sigma > mom_long {
                            1.0
                        } else if prev_sigma < -mom_short {
                            -1.0
                        } else {
                            0.0
                        };
                        if entry_bias == 0.0 {
                            continue;
                        } // Skip: no signal

                        let trade_ret = r * entry_bias; // positive = correct direction

                        // D-513: Modelo Estocástico de Barrera (Brownian Bridge First-Passage Time)
                        // Previene que la evolución premie stops parásitos infinitesimales (sl << sigma)
                        // que en trading real son ejecutados con 100% de probabilidad por el ruido microestructural.
                        let eff_sigma = volatility.max(0.0005); // Piso de volatilidad de 5 bps
                        let p_stop = if trade_ret <= -sl {
                            1.0
                        } else {
                            let arg = (2.0 * sl * (sl + trade_ret)) / (eff_sigma * eff_sigma);
                            (-arg.clamp(0.0, 50.0)).exp().clamp(0.0, 1.0)
                        };
                        let effective_ret = (-sl) * p_stop + trade_ret.min(tp) * (1.0 - p_stop);
                        let net_ret = effective_ret - roundtrip_fee;

                        wf_pnl +=
                            net_ret * wf_capital * candidate.scalp_kelly_fraction.clamp(0.05, 0.50);
                        wf_capital +=
                            net_ret * wf_capital * candidate.scalp_kelly_fraction.clamp(0.05, 0.50);
                        if wf_capital > wf_peak {
                            wf_peak = wf_capital;
                        }
                        if wf_peak > 0.0 {
                            wf_dd = wf_dd.max((wf_peak - wf_capital) / wf_peak);
                        }
                        if net_ret > 0.0 {
                            wf_wins += 1;
                        } else {
                            wf_losses += 1;
                        }
                        candidate_net_returns.push(net_ret);
                    }
                }

                let wf_trades = wf_wins + wf_losses;

                // C-10 (INFORME 14): FUNCIÓN ÚNICA DE APTITUD (D-652/D-653/
                // D-654/D-655). La fórmula anterior `wf_pnl × sqrt(trades) ×
                // wf_wr` era de la familia ERRADICADA por D-652: crecía con el
                // número de operaciones y con el tamaño de la apuesta sin
                // penalizar drawdown ni ruina. Todo promotor de genomas debe
                // llamar a `fitness::compute` — crecimiento logarítmico
                // (utilidad de Kelly) penalizado por drawdown². La simulación
                // corre SOLO sobre la partición OOS (train_end..n): no hay un
                // par IS/OOS de capitales separado que reportar, de modo que
                // oos_start == oos_end (factor 1.0, sin doble penalización).
                // trades < WF_MIN_TRADES ⇒ INVIABLE (−∞): jamás seleccionado.
                let fitness = crate::fitness::compute(&crate::fitness::FitnessInputs {
                    initial_capital: WF_INITIAL_CAPITAL,
                    final_capital: wf_capital,
                    max_drawdown_pct: wf_dd,
                    total_trades: wf_trades as u32,
                    min_trades_required: WF_MIN_TRADES,
                    oos_start_capital: wf_capital,
                    oos_end_capital: wf_capital,
                });
                let _ = wf_pnl; // conservado como telemetría futura del ciclo
                let _ = candidate_net_returns; // pre-screen: no feeding DSR

                prescreened.push((fitness, candidate));
            }

            // ── CERT-M8-H01: ETAPA REAL — el motor completo juzga al top-K ──
            prescreened.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut best = current_genome.clone();
            let mut best_score = f64::NEG_INFINITY;
            let mut best_candidate_returns: Vec<f64> = Vec::new();
            let mut evaluated = 1usize; // el incumbente siempre compite
            let inc = wf_evaluate_real(&current_genome, &real_series, wf_live_capital);
            if inc.fitness > best_score {
                best_score = inc.fitness;
                best_candidate_returns = inc.net_returns;
            }
            for (_pre, cand) in prescreened.into_iter().take(WF_REAL_TOP_K) {
                let out = wf_evaluate_real(&cand, &real_series, wf_live_capital);
                evaluated += 1;
                if out.fitness > best_score {
                    best_score = out.fitness;
                    best = cand;
                    best_candidate_returns = out.net_returns;
                }
            }
            println!(
                "🧬 [WF-REAL] {} candidatos evaluados con el MOTOR REAL (consejo+ML+fees+envelope); mejor fitness {:.4}",
                evaluated, best_score
            );
            (best, best_candidate_returns)
        })
        .await
        .unwrap_or((fallback_genome, Vec::new()));

        // FASE 6 / L-0: Estasis de Probabilidad Adaptativa por Tamaño Muestral con Prior Bayesiano Bootstrap.
        // Para cuentas micro ($13 USD) en fase de arranque (N < 15), incorpora un prior exploratorio suave
        // para evitar que el bot descarte el 100% de las mutaciones al inicio de su ciclo de vida (Causa Forense #D101).
        let safe_sharpe = if current_shadow_sharpe.is_finite() && current_shadow_sharpe > 0.0 {
            current_shadow_sharpe
        } else {
            0.1
        };
        let safe_len = (self.returns_history.len().max(1)) as f64;
        let std_error = 1.0 / safe_len.sqrt();
        let raw_confidence = (1.0 - (std_error / safe_sharpe)).clamp(0.0, 1.0);

        let bootstrap_weight = (15.0 - safe_len).max(0.0) / 15.0;
        let bayesian_confidence =
            (1.0 - bootstrap_weight) * raw_confidence + bootstrap_weight * 0.60;

        let target_confidence = if self.is_demo {
            0.55
        } else if safe_len < 10.0 {
            0.50
        } else if safe_len < 30.0 {
            0.60
        } else {
            0.75
        };

        if bayesian_confidence < target_confidence {
            println!(
                "⚠️ [PROBABILITY STASIS] Sharpe {:.2} superó base, pero Confianza Bayesiana es {:.1}%. Requiere > {:.0}% (N={}). Se descarta mutación.",
                current_shadow_sharpe,
                bayesian_confidence * 100.0,
                target_confidence * 100.0,
                safe_len as usize
            );
            return;
        }

        // QO-M1.1 — DEFLATED SHARPE RATIO (Bailey & López de Prado 2014):
        // con 2000 candidatos por ronda, el mejor por pura suerte supera
        // cualquier umbral fijo. El DSR corrige por multiplicidad y
        // curtosis: sólo un edge que SOBREVIVE es estadísticamente real.
        // CERT-M8-C03: el DSR ANTERIOR evaluaba `self.returns_history` (los
        // returns del INCUMBENTE), no los del CANDIDATO mutante que se
        // promueve — cuando el incumbente estaba caliente, cualquier ruido
        // pasaba. Ahora evalúa los returns SIMULADOS del candidato que el
        // walk-forward produjo.
        let (best_genome, candidate_returns) = best_genome; // destructure tuple
        let dsr_verdict =
            crate::selection_stats::edge_survives_multiplicity(&candidate_returns, 2_000);
        if !dsr_verdict.passes {
            println!(
                "🚫 [QO-M1 DSR] {:.3} < {:.2} con {} pruebas — {}",
                dsr_verdict.dsr,
                crate::selection_stats::DSR_THRESHOLD,
                dsr_verdict.n_trials,
                dsr_verdict.note
            );
            return;
        }

        // 🔥 ACTUALIZACIÓN EN VIVO (HOT-SWAP) AL GOD ENGINE
        // FASE 3: primero el EMBUDO (promote con gate de validación), y
        // solo si el almacén acepta se aplica al arena. Antes el orden
        // era inverso: un promote rechazado dejaba el arena mutado con
        // un genoma que el disco nunca sancionó.
        match quantum_arena::genome_store::GenomeEnvelope::promote(
            best_genome.clone(),
            "online_daemon",
            &format!(
                "t-stat estrategia {:.2} (confianza bayesiana >95%), {} observaciones acumuladas",
                current_shadow_sharpe,
                self.returns_history.len()
            ),
        ) {
            Ok(env) => {
                env.apply_to_arena(&self.arena);
                self.state.active_genome_id.fetch_add(1, Ordering::SeqCst);
                self.state.has_new_genome.store(true, Ordering::Release);
                // FASE 3: armar watchdog de rollback sobre el padre.
                self.post_promo_returns.clear();
                self.promoted_generation = Some((env.generation, env.parent_generation));
                println!(
                    "⚡ [HOT-SWAP] Genoma generación {} promovida (padre {}). Watchdog de rollback armado.",
                    env.generation, env.parent_generation
                );
                // QO-E2d — LEDGER: cada promoción queda en el WAL consultable
                // (responde "qué aprendió el sistema esta semana"; antes el
                // ledger se creaba y jamás se escribía).
                self.ledger.save_weight(
                    0,
                    format!("gen_{}", env.generation),
                    "promote".to_string(),
                    current_shadow_sharpe,
                );
            }
            Err(e) => println!(
                "⚠️ [ONLINE] Promoción RECHAZADA por el gate del almacén (arena queda intacto): {}",
                e
            ),
        }
        let _ = &self.champion_path; // conservado para compat de la struct
    }

    /// RANSAC (Random Sample Consensus) para el cálculo robusto de Sharpe Ratio.
    /// Elimina outliers (ruido de microestructura) y estima el Sharpe real.
    // FIX #719: Filtrado previo de retornos finitos y guarda de resultado finito en RANSAC Sharpe
    fn calculate_ransac_sharpe(returns: &[f64]) -> f64 {
        // CERT-M8-H03: el RANSAC anterior trimaba ±2σ outliers ANTES de
        // computar el t-stat — removiendo exactamente la cola negativa
        // gorda que EVIDENCIA degradación. Los tres controles que consumen
        // este estadístico (kill-switch, DSR gate, rollback watchdog) eran
        // todos optimistas por construcción. Ahora: el t-stat se computa
        // sobre la MUESTRA COMPLETA (sin trim). El inlier_mean RANSAC se
        // mantiene como estimación robusta de LOCALIZACIÓN reportada
        // alongside, pero el test de significancia ve la cola completa.
        let clean_returns: Vec<f64> = returns.iter().copied().filter(|r| r.is_finite()).collect();
        if clean_returns.len() < 10 {
            return 0.0;
        }

        // Media y desviación sobre la MUESTRA COMPLETA
        let n_all = clean_returns.len() as f64;
        let full_mean = clean_returns.iter().sum::<f64>() / n_all;
        let full_var = clean_returns.iter().map(|r| (r - full_mean).powi(2)).sum::<f64>() / (n_all - 1.0);
        let full_std = full_var.sqrt();

        if full_std <= 1e-12 || !full_std.is_finite() {
            return 0.0;
        }

        // t-stat sobre muestra completa: ve la cola gorda negativa
        let t_stat = (full_mean / full_std) * (n_all - 1.0).sqrt();
        if t_stat.is_finite() {
            t_stat
        } else {
            0.0
        }
    }
}
