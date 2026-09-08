#![feature(portable_simd)]

pub mod bootloader;
pub mod conformal;
pub mod darwin;
pub mod latency_accelerator;
pub mod math_kernels;
pub mod ml_inference;
pub mod orchestrator;
pub mod order_flow_aggregator;
pub mod quantum_kelly_risk;
pub mod reality_physics;
pub mod slippage_predictor;
pub mod stateful_engine;
pub mod trailing;

use crate::stateful_engine::StatefulEngine;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{MakerEngine, MakerQuote, ScalpEngine, SignalIntent, SignalType, SwingEngine};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Axioma VII: God Engine Core
/// Este componente contiene la lógica dura del ciclo HFT,
/// unificando la Arena con los motores, y eliminando la duplicación
/// entre Backtest y Producción.
pub struct GodEngineCore {
    pub arena: Arc<GlobalArena>,
    pub risk_engine: RiskEngine,
    pub scalp_engines: Vec<ScalpEngine>,
    pub swing_engines: Vec<SwingEngine>,
    pub maker_engines: Vec<MakerEngine>,
    pub feature_engines: Vec<StatefulEngine>,
    pub tensor_orchestrator: signal_engine::orchestrator::TensorVoteOrchestrator,
    pub scalp_forest: Option<Arc<crate::ml_inference::NanoForest>>,
    pub swing_nn: Option<dark_alpha_engine::DarkAlphaEngine>,
    pub model_rx: Option<std::sync::mpsc::Receiver<dark_alpha_engine::DarkAlphaEngine>>,
    pub last_ml_prob: f32,
    pub flight_recorder: Option<Arc<telemetry_server::FlightRecorder>>,
    pub reality: reality_physics::RealityPhysics,
    pub last_scalp_intent: Vec<SignalIntent>,
    pub last_swing_intent: Vec<SignalIntent>,
    pub last_scalp_senior_signals: Vec<[f64; 10]>,
    pub last_swing_senior_signals: Vec<[f64; 10]>,
    pub lakehouse: Option<Arc<storage_engine::LakehouseWarehouse>>,
    pub consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion,
    pub lead_lag_engine: feature_engine::LeadLagAlphaEngine,
    pub ppo_engine: dark_alpha_engine::online_ppo::OnlinePpoPolicyEngine,
    pub online_learner: metacortex_engine::online_learning::OnlineLearningModule,
    /// Cache de generación del genoma aplicado: refresh_models solo
    /// re-aplica el envelope de disco si su generación es MÁS NUEVA que la
    /// última aplicada. Antes re-aplicaba ciegamente cada 1000 ticks y PISABA
    /// todo hot-swap no promovido (cosecha del shadow forest, mutaciones de
    /// exploración) — el bloqueo #2 del genoma bt/prod.
    pub applied_generation: std::sync::atomic::AtomicU64,
    /// R4.3 — calibrador conformal real (antes: constante 0.95).
    pub conformal: conformal::ConformalCalibrator,
    /// DIAG R4 (transitorio): cuello post-orden.
    pub diag_council_vetoes: u64,
    pub diag_opened: u64,
    pub diag_swing_vetoes: u64,
    pub diag_swing_opened: u64,
    pub diag_close_wins: u64,
    pub diag_close_total: u64,
    pub diag_notional_sum: f64,
    pub diag_notional_max: f64,
    pub diag_pnl_sum: f64,
}

impl GodEngineCore {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        let initial_capital = arena.config.base_capital.load(Ordering::Relaxed);
        let scalp_forest = crate::ml_inference::NanoForest::get_global("BTCUSDT_SCALP");

        let mut scalp_engines = Vec::with_capacity(30);
        let mut swing_engines = Vec::with_capacity(30);
        let mut maker_engines = Vec::with_capacity(30);
        let mut feature_engines = Vec::with_capacity(30);

        for _ in 0..30 {
            scalp_engines.push(ScalpEngine::new());
            swing_engines.push(SwingEngine::default());
            maker_engines.push(MakerEngine::new(0.0005));
            feature_engines.push(StatefulEngine::new());
        }

        let mut tensor_orchestrator =
            signal_engine::orchestrator::TensorVoteOrchestrator::new(Arc::clone(&arena));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::coaxial_breakout::CoaxialBreakoutEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::game_theoretic_nash::GameTheoreticNashEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::hawkes_bessel::HawkesBesselEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::micro_scalp_trigger::MicroScalpTriggerEngine::default()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::perceptron_gate::PerceptronGateEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::quantum_oscillator::QuantumOscillatorEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::renyi_tsallis_entropy::RenyiTsallisEntropyEngine::default()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::soliton_wave::SolitonWaveEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::stochastic_resonance::StochasticResonanceEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::supersonic_shockwave::SupersonicShockwaveEngine::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::swing_conformal_filter::SwingConformalFilterEngine::default()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::trend_runner::HighPayoffTrendRunner::new()));
        tensor_orchestrator.add_strategy(Box::new(signal_engine::turbo_scalper::TurboScalpEngine::default()));
        // FIX #781: Registrar JohansenVecmEngine en el orquestador central
        tensor_orchestrator.add_strategy(Box::new(strategy_core::vecm_arbitrage::JohansenVecmEngine::default()));

        let swing_nn = if let Ok(json_data) =
            std::fs::read_to_string("models/DarkAlpha_BTCUSDT.json")
        {
            if let Ok(mut model) =
                serde_json::from_str::<dark_alpha_engine::DarkAlphaEngine>(&json_data)
            {
                model.init_buffers();
                model.sanitize_denormals();
                // T-04: TODO modelo cargado de disco se CONGELA antes de
                // inferir — los modelos pre-N-11 deserializan descongelados y
                // sus normalizadores Welford mutan en cada predict (la causa
                // raíz del no-determinismo entre corridas).
                model.freeze();
                telemetry_server::telemetry_log!(
                    "🧠 [DARK ALPHA] Initial Boot Model Loaded & Sanitized (in_features: {})",
                    model.layer1.in_features
                );
                Some(model)
            } else {
                telemetry_server::telemetry_log!(
                    "⚠️ [DARK ALPHA] Failed to parse models/DarkAlpha_BTCUSDT.json. Using fallback."
                );
                Some(dark_alpha_engine::DarkAlphaEngine::new(54, 64, 32))
            }
        } else {
            telemetry_server::telemetry_log!(
                "⚠️ [DARK ALPHA] models/DarkAlpha_BTCUSDT.json not found. Using fallback."
            );
            Some(dark_alpha_engine::DarkAlphaEngine::new(54, 64, 32))
        };

        let ppo_engine = dark_alpha_engine::online_ppo::OnlinePpoPolicyEngine::new([0.40, 0.40, 0.20, 0.15, 0.10]);
        let online_learner = metacortex_engine::online_learning::OnlineLearningModule::new(0.001, 0.9);

        Self {
            arena,
            risk_engine: RiskEngine::new(initial_capital),
            scalp_engines,
            swing_engines,
            maker_engines,
            feature_engines,
            tensor_orchestrator,
            scalp_forest,
            swing_nn,
            model_rx: None,
            last_ml_prob: 0.5,
            flight_recorder: None,
            reality: reality_physics::RealityPhysics::default(),
            last_scalp_intent: vec![SignalIntent::flat(); 30],
            last_swing_intent: vec![SignalIntent::flat(); 30],
            last_scalp_senior_signals: vec![[0.0; 10]; 30],
            last_swing_senior_signals: vec![[0.0; 10]; 30],
            lakehouse: None,
            consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion::new(),
            lead_lag_engine: feature_engine::LeadLagAlphaEngine::new(50),
            ppo_engine,
            online_learner,
            applied_generation: std::sync::atomic::AtomicU64::new(0),
            conformal: conformal::ConformalCalibrator::new(),
            diag_council_vetoes: 0,
            diag_opened: 0,
            diag_swing_vetoes: 0,
            diag_swing_opened: 0,
            diag_close_wins: 0,
            diag_close_total: 0,
            diag_notional_sum: 0.0,
            diag_notional_max: 0.0,
            diag_pnl_sum: 0.0,
        }
    }

    pub fn get_features(&self, coin_id: usize) -> [f32; 12] {
        self.feature_engines[coin_id].get_features()
    }

    pub fn reset_engines(&mut self) {
        for i in 0..30 {
            self.scalp_engines[i] = ScalpEngine::new();
            self.swing_engines[i] = SwingEngine::default();
            self.feature_engines[i] = StatefulEngine::new();
        }
        let init_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
        self.risk_engine.reset(init_cap);
    }

    pub fn set_model_rx(
        &mut self,
        rx: std::sync::mpsc::Receiver<dark_alpha_engine::DarkAlphaEngine>,
    ) {
        self.model_rx = Some(rx);
    }

    pub fn set_lakehouse(&mut self, lakehouse: Arc<storage_engine::LakehouseWarehouse>) {
        self.lakehouse = Some(lakehouse);
    }

    /// Rollback local quantum position when order is rejected by exchange or blocked by risk envelope
    pub fn rollback_position(&self, coin_id: usize) {
        if coin_id >= self.arena.coins.len() {
            return;
        }
        let coin = &self.arena.coins[coin_id];
        if coin.positions.position.is_open() {
            let horizon = coin.positions.position.horizon.load(Ordering::Relaxed);
            let (_is_long, _entry_price, _qty, margin_used, entry_fee) =
                coin.positions.position.close_with_fee();
            if horizon == 2 {
                let _ = coin.positions.swing_position.close_with_fee();
                if margin_used > 0.0 {
                    self.arena
                        .used_margin
                        .fetch_add(-margin_used, Ordering::Relaxed);
                    self.arena
                        .swing_used_margin
                        .fetch_add(-margin_used, Ordering::Relaxed);
                }
                if entry_fee > 0.0 {
                    self.arena
                        .unified_capital
                        .fetch_add(entry_fee, Ordering::Relaxed);
                    coin.swing
                        .pnl_realized
                        .fetch_add(entry_fee, Ordering::Relaxed);
                }
            } else {
                let _ = coin.positions.scalp_position.close_with_fee();
                if margin_used > 0.0 {
                    self.arena
                        .used_margin
                        .fetch_add(-margin_used, Ordering::Relaxed);
                    self.arena
                        .scalp_used_margin
                        .fetch_add(-margin_used, Ordering::Relaxed);
                }
                if entry_fee > 0.0 {
                    self.arena
                        .unified_capital
                        .fetch_add(entry_fee, Ordering::Relaxed);
                    coin.scalp
                        .pnl_realized
                        .fetch_add(entry_fee, Ordering::Relaxed);
                }
            }
            if entry_fee > 0.0 {
                coin.metrics
                    .pnl_realized
                    .fetch_add(entry_fee, Ordering::Relaxed);
            }
        }
    }

    /// Rollback local scalp position when order is rejected by exchange or blocked by risk envelope
    pub fn rollback_scalp_position(&self, coin_id: usize) {
        if coin_id >= self.arena.coins.len() {
            return;
        }
        let coin = &self.arena.coins[coin_id];
        if coin.positions.scalp_position.is_open() {
            let (_is_long, _entry_price, _qty, margin_used, entry_fee) =
                coin.positions.scalp_position.close_with_fee();
            if margin_used > 0.0 {
                self.arena
                    .scalp_used_margin
                    .fetch_add(-margin_used, Ordering::Relaxed);
            }
            if entry_fee > 0.0 {
                self.arena
                    .unified_capital
                    .fetch_add(entry_fee, Ordering::Relaxed);
                coin.scalp
                    .pnl_realized
                    .fetch_add(entry_fee, Ordering::Relaxed);
            }
        }
    }

    /// Rollback local swing position when order is rejected by exchange or blocked by risk envelope
    pub fn rollback_swing_position(&self, coin_id: usize) {
        if coin_id >= self.arena.coins.len() {
            return;
        }
        let coin = &self.arena.coins[coin_id];
        if coin.positions.swing_position.is_open() {
            let (_is_long, _entry_price, _qty, margin_used, entry_fee) =
                coin.positions.swing_position.close_with_fee();
            if margin_used > 0.0 {
                self.arena
                    .swing_used_margin
                    .fetch_add(-margin_used, Ordering::Relaxed);
            }
            if entry_fee > 0.0 {
                self.arena
                    .unified_capital
                    .fetch_add(entry_fee, Ordering::Relaxed);
                coin.swing
                    .pnl_realized
                    .fetch_add(entry_fee, Ordering::Relaxed);
            }
        }
    }

    /// Construye el tensor unificado de 54 dimensiones para la Red Neuronal Dark Alpha (Swing)
    #[inline(always)]
    pub fn build_54d_tensor(
        &self,
        coin_id: usize,
        bid_qty: f64,
        ask_qty: f64,
        current_price: f64,
        omni_features: &[f64; 54],
    ) -> [f64; 54] {
        let stateful_feats = self.feature_engines[coin_id].get_swing_features();
        let mut combined = [0.0; 54];
        for j in 0..34 {
            combined[j] = stateful_feats[j] as f64;
        }
        let cur_obi = if bid_qty + ask_qty > 0.0 {
            (bid_qty - ask_qty) / (bid_qty + ask_qty)
        } else {
            0.0
        };
        let cur_atr = self.feature_engines[coin_id].get_atr_pct();
        let cur_vol = bid_qty + ask_qty;
        let delta = self.feature_engines[coin_id].v_t;

        combined[34] = (delta / current_price.max(1.0)).clamp(-0.1, 0.1);
        combined[35] = cur_atr;
        combined[36] = (cur_vol / 1000.0).tanh();
        combined[37] = cur_obi;
        combined[38] = (combined[34] * 10.0).tanh();
        combined[39] = (combined[35] * 100.0).min(5.0);
        combined[40] = 0.50;
        combined[41] = if omni_features.len() > 21 && omni_features[21] > 0.0 {
            (omni_features[21] / 100.0).clamp(0.5, 2.0)
        } else {
            1.04
        };
        combined[42] = if omni_features.len() > 22 && omni_features[22] > 0.0 {
            (omni_features[22] / 5000.0).clamp(0.5, 2.0)
        } else {
            1.02
        };
        combined[43] = if omni_features.len() > 23 && omni_features[23] > 0.0 {
            (omni_features[23] / 18000.0).clamp(0.5, 2.0)
        } else {
            1.00
        };
        combined[44] = if omni_features.len() > 24 && omni_features[24] > 0.0 {
            (omni_features[24] / 20.0).clamp(0.1, 5.0)
        } else {
            0.75
        };
        combined[45] = if omni_features.len() > 25 && omni_features[25] > 0.0 {
            (omni_features[25] / 4.0).clamp(0.1, 5.0)
        } else {
            1.05
        };
        combined[46] = if omni_features.len() > 26 && omni_features[26] > 0.0 {
            (omni_features[26] / 2500.0).clamp(0.5, 2.0)
        } else {
            1.0
        };
        combined[47] = if omni_features.len() > 27 && omni_features[27] > 0.0 {
            (omni_features[27] / 80.0).clamp(0.2, 3.0)
        } else {
            1.00
        };
        combined[48] = if omni_features.len() > 11 {
            (omni_features[11] * 1000.0).clamp(-5.0, 5.0)
        } else {
            0.0
        };
        combined[49] = if omni_features.len() > 29 && omni_features[29] > 0.0 {
            (omni_features[29] / 5.25).clamp(0.0, 3.0)
        } else {
            1.0
        };
        combined[50] = if omni_features.len() > 14 && omni_features[14] > 0.0 {
            (omni_features[14] / 50.0).clamp(0.0, 2.0)
        } else {
            1.0
        };
        combined[51] = if omni_features.len() > 30 {
            (omni_features[30] / 100.0).tanh()
        } else {
            0.0
        };
        combined[52] = if omni_features.len() > 31 {
            (omni_features[31] / 100.0).tanh()
        } else {
            0.0
        };
        combined[53] = if omni_features.len() > 33 {
            (omni_features[33] * 100.0).clamp(-5.0, 5.0)
        } else {
            0.0
        };

        combined
    }

    /// Actualiza los modelos y genomas cargados en caliente si hubieron reentrenamientos asíncronos o evolución genética
    pub fn refresh_models(&mut self) {
        self.scalp_forest = crate::ml_inference::NanoForest::get_global("BTCUSDT_SCALP");
        if let Some(env) = quantum_arena::genome_store::GenomeEnvelope::load_active() {
            let last = self.applied_generation.load(Ordering::Relaxed);
            if env.generation > last {
                env.genome.apply_to_arena(&self.arena);
                self.applied_generation.store(env.generation, Ordering::Relaxed);
            }
            // Generación <= ya aplicada: los hot-swaps en vivo permanecen
            // hasta que el almacén sancione una generación superior.
        }
    }

    /// Procesa un evento unificado continuo (trade, kline, depth) y devuelve las órdenes generadas.
    /// Retorna: (NuevoOrden, None, CerradoOrden, None) para compatibilidad absoluta con ejecutores L3.
    #[inline(always)]
    pub fn process_event(
        &mut self,
        coin_id: usize,
        _is_trade: bool,
        _is_kline_closed: bool,
        is_depth: bool,
        current_price: f64,
        _trade_qty: f64,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        depth_obi: f64,
        depth_micro_div: f64,
        event_time_ms: u64,
        _latency_panic: bool,
        omni_features: &[f64; 54],
    ) -> (
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
    ) {
        telemetry_server::profile_node!("GodEngineCore::process_event", {
            // Fast Invariant Check: Zero-cost anomaly rejection for invalid prices or out-of-bounds coin index
            if current_price <= 0.0 || !current_price.is_finite() || coin_id >= self.arena.coins.len() {
                return (None, None, None, None);
            }

            // Crossed Orderbook Guard: Reject corrupted L2 depth ticks where bid exceeds ask
            if is_depth && bid > 0.0 && ask > 0.0 && bid > ask {
                return (None, None, None, None);
            }

            // Zero-copy Hot-Reloading Check
            if let Some(rx) = &self.model_rx {
                if let Ok(mut new_model) = rx.try_recv() {
                    new_model.init_buffers();
                    new_model.sanitize_denormals();
                    new_model.freeze(); // T-04: inferencia siempre congelada
                    self.swing_nn = Some(new_model);
                    telemetry_server::telemetry_log!(
                        "🧠 [DARK ALPHA] Hot-Reload Successful! New weights absorbed and sanitized in Zero-Copy."
                    );
                }
            }

            if self.arena.kill_switch_active.load(Ordering::Relaxed) {
                return (None, None, None, None);
            }

            if self.arena.tick_counter.load(Ordering::Relaxed).is_multiple_of(1000) {
                self.refresh_models();
            }

            let eff_bid = if bid > 0.0 { bid } else { current_price * 0.9999 };
            let eff_ask = if ask > 0.0 { ask } else { current_price * 1.0001 };
            let eff_bid_qty = if bid_qty > 0.0 { bid_qty } else { _trade_qty.max(0.01) };
            let eff_ask_qty = if ask_qty > 0.0 { ask_qty } else { _trade_qty.max(0.01) };

            if is_depth {
                self.feature_engines[coin_id].update_macro_features(
                    depth_obi,
                    depth_micro_div,
                    0.0,
                    event_time_ms,
                );
                let mid_price = (eff_bid + eff_ask) / 2.0;
                let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                self.arena.coins[coin_id].current_price.store(mid_price, Ordering::Relaxed);
                self.arena.coins[coin_id].current_atr.store(raw_atr_pct * mid_price, Ordering::Relaxed);
                self.arena.update_market_data(coin_id, eff_bid, eff_ask, eff_bid_qty, eff_ask_qty, 0);
            }

            let (new_sc, new_sw, closed_sc, closed_sw, _maker) = self.process_tick_dual(
                coin_id,
                eff_bid,
                eff_ask,
                eff_bid_qty,
                eff_ask_qty,
                event_time_ms,
                omni_features,
            );

            (new_sc, new_sw, closed_sc, closed_sw)
        })
    }

    /// Revierte una posición fantasma (Ghost Position) cuando la API de Binance la rechaza.
    /// Esto evita que el motor local siga trackeando una posición que no existe en el exchange.
    pub fn revert_ghost_position(&mut self, coin_id: usize, is_scalp: bool, is_long: bool) {
        let coin = &self.arena.coins[coin_id];
        if is_scalp {
            if coin.positions.scalp_position.is_open()
                && coin
                    .positions
                    .scalp_position
                    .is_long
                    .load(Ordering::Relaxed)
                    == is_long
            {
                telemetry_server::telemetry_log!(
                    "👻 [GHOST REVERT] Binance rechazó orden Scalp para {}. Revirtiendo estado local.",
                    coin_id
                );
                let (_, _, _, margin) = coin.positions.scalp_position.close();
                self.arena
                    .scalp_used_margin
                    .fetch_add(-margin, Ordering::Relaxed);
            }
        } else {
            if coin.positions.swing_position.is_open()
                && coin
                    .positions
                    .swing_position
                    .is_long
                    .load(Ordering::Relaxed)
                    == is_long
            {
                telemetry_server::telemetry_log!(
                    "👻 [GHOST REVERT] Binance rechazó orden Swing para {}. Revirtiendo estado local.",
                    coin_id
                );
                let (_, _, _, margin) = coin.positions.swing_position.close();
                self.arena
                    .swing_used_margin
                    .fetch_add(-margin, Ordering::Relaxed);
            }
        }

        if coin.positions.position.is_open()
            && coin.positions.position.is_long.load(Ordering::Relaxed) == is_long
        {
            telemetry_server::telemetry_log!(
                "👻 [GHOST REVERT] Binance rechazó orden Unificada para {}. Revirtiendo estado local.",
                coin_id
            );
            let (_, _, _, margin) = coin.positions.position.close();
            self.arena.used_margin.fetch_add(-margin, Ordering::Relaxed);
        }
    }

    /// Revierte de inmediato una posición cuántica continua rechazada por el exchange
    pub fn revert_quantum_ghost_position(&mut self, coin_id: usize) {
        if coin_id >= self.arena.coins.len() {
            return;
        }
        let coin = &self.arena.coins[coin_id];
        if coin.positions.position.is_open() {
            telemetry_server::telemetry_log!(
                "👻 [GHOST REVERT] Revertiendo posición cuántica continua fantasma para coin {}",
                coin_id
            );
            let (_, _, _, margin) = coin.positions.position.close();
            self.arena.used_margin.fetch_add(-margin, Ordering::Relaxed);
        }
    }

    /// Procesa un tick con bifurcación cuántica real e independiente para Scalping y Swing.
    /// Retorna: (new_scalp, new_swing, closed_scalp, closed_swing, maker_quote)
    #[inline(always)]
    pub fn process_tick_dual(
        &mut self,
        coin_id: usize,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        event_time_ms: u64,
        omni_features: &[f64; 54],
    ) -> (
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<MakerQuote>,
    ) {
        telemetry_server::profile_node!("GodEngineCore::process_tick_dual", {
            // 1. Quantum Kill-Switch Check
            if self.arena.kill_switch_active.load(Ordering::Relaxed) {
                return (None, None, None, None, None);
            }

            // 2. Latency Interlock (Fase 6/7: Cisne Negro / HFT Spoofing)
            // F5.2 — FIX UMBRAL ABSURDO: era 3.000.000 ms (¡50 minutos!) con un
            // comentario que decía ">10ms": el guardián de datos obsoletos estaba
            // discapacitado desde su nacimiento. Ahora usa latency_ms_panic_threshold
            // del GENOMA — la MISMA fuente de verdad que el timeout de lectura del
            // WS — con piso absoluto de 50ms (operar con datos más viejos es
            // spoofing garantizado, no HFT).
            let latency_threshold_ms = self
                .arena
                .config
                .latency_ms_panic_threshold
                .load(Ordering::Relaxed)
                .max(50.0);
            let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);
            if latency_ms > latency_threshold_ms as u64 {
                // Modo Pánico: descartar el tick y no emitir señales.
                if self
                    .arena
                    .tick_counter
                    .load(Ordering::Relaxed)
                    .is_multiple_of(1000)
                {
                    telemetry_server::telemetry_log!(
                        "🚨 [LATENCY PANIC] Latencia {}ms > umbral {:.0}ms! Tick descartado — sin señales con datos obsoletos.",
                        latency_ms,
                        latency_threshold_ms
                    );
                }
                return (None, None, None, None, None);
            }

            self.arena.increment_tick();

            let mid_price = (bid + ask) / 2.0;
            let total_vol = bid_qty + ask_qty;

            // Update ML Features
            let pseudo_maker = bid_qty > ask_qty;
            let feature_engine = &mut self.feature_engines[coin_id];
            feature_engine.process_tick(mid_price, total_vol, event_time_ms);
            feature_engine.update_trade_flow(total_vol, pseudo_maker);

            let ofi_value = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
            let sym = quantum_arena::symbol_registry::try_spec(coin_id).map(|s| s.symbol).unwrap_or_default();
            if sym.starts_with("BTC") {
                self.lead_lag_engine.update_leader(true, ofi_value);
            } else if sym.starts_with("ETH") {
                self.lead_lag_engine.update_leader(false, ofi_value);
            }
            // D-36: Predict altcoin cross-asset impulse from real BTC/ETH leaders
            let (leader_mom, lead_lag_div) = self.lead_lag_engine.predict_altcoin_impulse(ofi_value);

            let obi = if total_vol > 0.0 {
                (bid_qty - ask_qty) / total_vol
            } else {
                0.0
            };
            feature_engine.update_macro_features(obi, 0.0, 0.0, event_time_ms);
            let raw_atr_pct = feature_engine.get_atr_pct();
            let hurst_val = feature_engine.hurst.current();
            let coin = &self.arena.coins[coin_id];
            coin.current_price.store(mid_price, Ordering::Relaxed);
            coin.current_atr.store(raw_atr_pct * mid_price, Ordering::Relaxed);
            coin.hurst_exponent.store(hurst_val, Ordering::Relaxed);

            let mut closed_scalp = None;
            let mut closed_swing = None;

            let coin = &self.arena.coins[coin_id];
            let atr_pct = self.feature_engines[coin_id].get_atr_pct();
            let min_atr_pct = self
                .arena
                .config
                .dynamic_atr_min
                .load(Ordering::Relaxed)
                .max(0.001);
            let atr_pct_live = atr_pct.max(min_atr_pct);

            // --- 1A. GESTIÓN DE POSICIÓN SCALPING (TIGHT STOPS, R:R >= 2:1, HIGH-FREQUENCY COMPOUNDING) ---
            let scalp_sl_base = self.arena.config.scalp_sl_base.load(Ordering::Relaxed);
            let scalp_tp_base = self.arena.config.scalp_tp_base.load(Ordering::Relaxed);
            let scalp_sl = scalp_sl_base.max(atr_pct * 1.25).clamp(0.0015, 0.0050);
            let scalp_tp = scalp_tp_base.max(scalp_sl * 2.0).clamp(0.0030, 0.0150);

            let scalp_is_open = coin.positions.scalp_position.is_open()
                || (coin.positions.position.is_open() && coin.positions.position.horizon() == quantum_arena::position::PositionHorizon::Scalping);

            if scalp_is_open {
                let pos = if coin.positions.scalp_position.is_open() {
                    &coin.positions.scalp_position
                } else {
                    &coin.positions.position
                };
                let is_long = pos.is_long.load(Ordering::Relaxed);
                let entry = pos.entry_price.load(Ordering::Relaxed);
                let qty = pos.quantity.load(Ordering::Relaxed);
                let entry_time = pos.entry_time_ms.load(Ordering::Relaxed);

                let pnl_pct = if is_long {
                    (mid_price - entry) / entry
                } else {
                    (entry - mid_price) / entry
                };

                let pseudo_atr = atr_pct_live * entry;
                let side_int = if is_long { 1 } else { -1 };
                let position_age_ms = if event_time_ms > 0 {
                    event_time_ms.saturating_sub(entry_time)
                } else {
                    0
                };

                // Trailing Scalping: Activación temprana para asegurar ganancias rápidas
                let trail_activation_pnl = (scalp_tp * 0.50).max(pseudo_atr / entry.max(1.0) * 1.5).clamp(0.0020, 0.0080);
                let trail_active = position_age_ms > 8_000 && pnl_pct >= trail_activation_pnl;

                let mut trail_hit = false;
                let mut force_close_trail = false;

                if trail_active {
                    let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                    let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                        side_int,
                        entry,
                        mid_price,
                        pseudo_atr,
                        pos.trailing_phase.load(Ordering::Relaxed) as i32,
                        pos.mfe_atr.load(Ordering::Relaxed),
                        pos.max_pnl_pct.load(Ordering::Relaxed),
                        pos.trail_stop.load(Ordering::Relaxed),
                        self.arena.config.scalp_trail_atr_mult_base.load(Ordering::Relaxed).clamp(0.5, 3.0),
                        self.arena.config.scalp_trail_act_atr.load(Ordering::Relaxed).clamp(0.5, 3.0),
                        self.arena.config.scalp_trail_step_atr.load(Ordering::Relaxed).clamp(0.5, 4.0),
                        self.arena.config.scalp_trail_max_atr.load(Ordering::Relaxed).clamp(0.5, 5.0),
                        self.arena.config.scalp_trail_atr_mult_base.load(Ordering::Relaxed).clamp(0.5, 4.0),
                        live_fee,
                    );

                    let sl_floor = if is_long {
                        entry * (1.0 - scalp_sl)
                    } else {
                        entry * (1.0 + scalp_sl)
                    };
                    let safe_stop = if is_long {
                        trail_res.stop_price.max(sl_floor)
                    } else {
                        if trail_res.stop_price > 0.0 {
                            trail_res.stop_price.min(sl_floor)
                        } else {
                            sl_floor
                        }
                    };

                    let current_stored = pos.trail_stop.load(Ordering::Relaxed);
                    let final_stop = if is_long {
                        if current_stored > 0.0 {
                            safe_stop.max(current_stored)
                        } else {
                            safe_stop
                        }
                    } else {
                        if current_stored > 0.0 {
                            safe_stop.min(current_stored)
                        } else {
                            safe_stop
                        }
                    };

                    pos.trail_stop.store(final_stop, Ordering::Relaxed);
                    pos.trailing_phase.store(trail_res.new_phase as u8, Ordering::Relaxed);
                    pos.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
                    pos.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);

                    trail_hit = (is_long && mid_price <= final_stop)
                        || (!is_long && mid_price >= final_stop && final_stop > 0.0);
                    force_close_trail = trail_res.force_close;
                }

                let macro_t = self.feature_engines[coin_id].get_macro_trend();
                let trend_reversed = (is_long && macro_t < -0.0015) || (!is_long && macro_t > 0.0015);
                let hard_timeout = position_age_ms > 7_200_000; // 2h hard limit para scalp
                let is_zombie = event_time_ms > 0 && position_age_ms > 1_800_000 && ((trend_reversed && pnl_pct <= -0.0020) || hard_timeout);
                // D-38: Salida temprana por toxicidad de microestructura (VPIN > 0.70 + OFI adverso) para proteger micro-capital
                let cur_vpin = self.feature_engines[coin_id].cvpin.current_vpin();
                let ofi_adverse = (is_long && ofi_value < -0.45) || (!is_long && ofi_value > 0.45);
                let toxic_flow_exit = ofi_adverse && cur_vpin > 0.70 && pnl_pct < -0.0005;

                // T-06 — el TP limit SOLO se llena con trade-through: un long
                // cierra cuando el BID alcanza el tp_price (no el mid — el mid
                // cruza media-spread antes que el precio ejecutable). El SL se
                // mantiene por mid (conservador, ya maneja gaps con min/max).
                let tp_traded_through = if is_long {
                    bid >= entry * (1.0 + scalp_tp)
                } else {
                    ask <= entry * (1.0 - scalp_tp)
                };
                if tp_traded_through
                    || pnl_pct <= -scalp_sl
                    || trail_hit
                    || force_close_trail
                    || is_zombie
                    || toxic_flow_exit
                {
                    let sl_price = if is_long {
                        entry * (1.0 - scalp_sl)
                    } else {
                        entry * (1.0 + scalp_sl)
                    };
                    let tp_price = if is_long {
                        entry * (1.0 + scalp_tp)
                    } else {
                        entry * (1.0 - scalp_tp)
                    };

                    let mut exit_price = mid_price;
                    if is_long {
                        if pnl_pct <= -scalp_sl {
                            exit_price = exit_price.min(sl_price);
                        }
                        if trail_hit {
                            let ts = pos.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.max(ts); }
                        }
                        if pnl_pct >= scalp_tp {
                            exit_price = tp_price;
                        }
                    } else {
                        if pnl_pct <= -scalp_sl {
                            exit_price = exit_price.max(sl_price);
                        }
                        if trail_hit {
                            let ts = pos.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.min(ts); }
                        }
                        if pnl_pct >= scalp_tp {
                            exit_price = tp_price;
                        }
                    }

                    let gross_pnl = if is_long {
                        (exit_price - entry) * qty
                    } else {
                        (entry - exit_price) * qty
                    };

                    let live_maker = self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002);
                    let live_taker = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004);
                    let close_fee_rate = if pnl_pct >= scalp_tp { live_maker } else { live_taker };
                    let close_fee = (qty * exit_price) * close_fee_rate;
                    // T-01: capturar la predicción ANTES de cerrar —
                    // close_with_fee() resetea ml_prediction a 0.0 y el
                    // calibrador conformal quedaba muerto (p=1.0 perpetuo).
                    let ml_at_entry = coin.positions.scalp_position
                        .ml_prediction
                        .load(Ordering::Relaxed);
                    let (_, _, _, sc_margin, sc_entry_fee) = coin.positions.scalp_position.close_with_fee();
                    // N-04: Cerrar coin.positions.position SOLO si actualmente espeja esta posición de Scalp (horizon == 0)
                    let (gen_margin, gen_entry_fee) = if coin.positions.position.is_open()
                        && (coin.positions.position.horizon.load(Ordering::Relaxed) == 0)
                    {
                        let (_, _, _, gm, gef) = coin.positions.position.close_with_fee();
                        (gm, gef)
                    } else {
                        (0.0, 0.0)
                    };
                    let margin_used = sc_margin.max(gen_margin);
                    let entry_fee_paid = sc_entry_fee.max(gen_entry_fee);

                    let net_realized_pnl = gross_pnl - close_fee;
                    let net_trade_pnl = net_realized_pnl - entry_fee_paid;

                    let current_used = self.arena.used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena.used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.used_margin.store(0.0, Ordering::Relaxed);
                    }
                    let current_sc_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                    if current_sc_used >= margin_used {
                        self.arena.scalp_used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.scalp_used_margin.store(0.0, Ordering::Relaxed);
                    }

                    coin.metrics.pnl_realized.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    coin.scalp.pnl_realized.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    self.arena.unified_capital.fetch_add(net_realized_pnl, Ordering::Relaxed);

                    self.feature_engines[coin_id].last_scalp_exit_tick = self.feature_engines[coin_id].tick_count;
                    self.feature_engines[coin_id].last_scalp_was_loss = net_trade_pnl <= 0.0;

                    let is_win = net_trade_pnl > 0.0;
                    self.diag_close_wins += is_win as u64;
                    self.diag_close_total += 1;
                    self.diag_notional_sum += qty * exit_price;
                    self.diag_notional_max = self.diag_notional_max.max(qty * exit_price);
                    self.diag_pnl_sum += net_trade_pnl;
                    // R4.3 — el cierre alimenta el calibrador conformal con
                    // la predicción capturada antes de cerrar y el resultado neto real.
                    if ml_at_entry > 0.0 {
                        self.conformal.update(ml_at_entry, is_win);
                    }
                    let n = coin.metrics.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.metrics.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.metrics.win_rate.store(new_wr, Ordering::Relaxed);

                    // FIX D41: Profit Factor contable exacto (gross_wins / gross_losses)
                    if is_win {
                        coin.metrics.gross_wins.fetch_add(net_trade_pnl, Ordering::Relaxed);
                        coin.scalp.gross_wins.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    } else {
                        coin.metrics.gross_losses.fetch_add(net_trade_pnl.abs(), Ordering::Relaxed);
                        coin.scalp.gross_losses.fetch_add(net_trade_pnl.abs(), Ordering::Relaxed);
                    }
                    let total_wins = coin.metrics.gross_wins.load(Ordering::Relaxed);
                    let total_losses = coin.metrics.gross_losses.load(Ordering::Relaxed);
                    let new_pf = if total_losses > 0.0 {
                        (total_wins / total_losses).clamp(0.1, 10.0)
                    } else if total_wins > 0.0 {
                        5.0 // Cap de arranque cuando sólo existen ganancias
                    } else {
                        1.50 // Prior Bayesiano inicial
                    };
                    coin.metrics.profit_factor.store(new_pf, Ordering::Relaxed);

                    // FIX D36 & D41: Actualizar métricas aisladas de ScalpState
                    let n_scalp = coin.scalp.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr_scalp = coin.scalp.win_rate.load(Ordering::Relaxed);
                    let new_wr_scalp = old_wr_scalp + (((if is_win { 1.0 } else { 0.0 }) - old_wr_scalp) / n_scalp);
                    coin.scalp.win_rate.store(new_wr_scalp, Ordering::Relaxed);
                    let sc_wins = coin.scalp.gross_wins.load(Ordering::Relaxed);
                    let sc_losses = coin.scalp.gross_losses.load(Ordering::Relaxed);
                    let new_pf_scalp = if sc_losses > 0.0 {
                        (sc_wins / sc_losses).clamp(0.1, 10.0)
                    } else if sc_wins > 0.0 {
                        5.0
                    } else {
                        1.50
                    };
                    coin.scalp.profit_factor.store(new_pf_scalp, Ordering::Relaxed);

                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                    let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                    let clamp_min = self.arena.config.kelly_clamp_min.load(Ordering::Relaxed);
                    let clamp_max = self.arena.config.kelly_clamp_max.load(Ordering::Relaxed);
                    let strategy_base = self.arena.config.scalp_kelly_fraction.load(Ordering::Relaxed);
                    let kelly_f = risk_engine::kelly::calculate_kelly_fraction(
                        new_wr,
                        new_pf,
                        curr_cap,
                        base_cap,
                        survival_ratio,
                        exp_mult,
                        clamp_min,
                        clamp_max,
                        strategy_base,
                    );
                    coin.metrics.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                    coin.scalp.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                    coin.last_scalp_close_ts.store(event_time_ms, Ordering::Relaxed);

                    closed_scalp = Some((is_long, net_realized_pnl, qty));

                    // N-12: Retroalimentación causal al Consejo de Seniors tras cierre de Scalp
                    let notional = (qty * entry).max(1.0);
                    let realized_ret = net_realized_pnl / notional;
                    if coin_id < self.last_scalp_senior_signals.len() {
                        self.consejo_deliberacion.record_outcome(&self.last_scalp_senior_signals[coin_id], realized_ret);
                    }

                    // D-34: Retroalimentación PPO continuo y Online Learning tras cierre de Scalp
                    let hawkes_r = self.feature_engines[coin_id].cvpin.current_vpin();
                    let ppo_close_features = [
                        (ofi_value / 0.35).clamp(-1.5, 1.5),
                        (obi / 0.35).clamp(-1.5, 1.5),
                        hawkes_r.clamp(0.0, 1.0),
                        lead_lag_div.clamp(-1.5, 1.5),
                        ((hurst_val - 0.50) * 2.0).clamp(-1.0, 1.0),
                    ];
                    self.ppo_engine.update_policy(
                        realized_ret,
                        &ppo_close_features,
                        if is_long { 1.0 } else { -1.0 },
                        1.0,
                        0.05,
                        0.01,
                        0.20,
                        0.01,
                    );
                    let cur_features = self.feature_engines[coin_id].get_features();
                    let mut online_feat = [0.0f32; 64];
                    for (idx, &f) in cur_features.iter().enumerate().take(64) {
                        online_feat[idx] = f;
                    }
                    self.online_learner.update_weights_with_kalman_adaptive_vol(
                        &online_feat,
                        (realized_ret - ml_at_entry) as f32,
                        (hurst_val as f32 - 0.5).abs(),
                        raw_atr_pct as f32,
                    );
                } else {
                    let notional = qty * entry;
                    let unrealized = pnl_pct * notional;
                    coin.metrics.pnl_unrealized.store(unrealized, Ordering::Relaxed);
                    coin.scalp.pnl_unrealized.store(unrealized, Ordering::Relaxed);
                }
            }

            // --- 1B. GESTIÓN DE POSICIÓN SWING (WIDE STOPS, MULTI-HOUR / MULTI-DAY TRENDS) ---
            let swing_sl_base = self.arena.config.swing_sl_base.load(Ordering::Relaxed);
            let swing_tp_base = self.arena.config.swing_tp_base.load(Ordering::Relaxed);
            let swing_sl = swing_sl_base.max(atr_pct * 3.0).clamp(0.0080, 0.0350);
            let swing_tp = swing_tp_base.max(swing_sl * 2.5).clamp(0.0200, 0.0900);

            let swing_is_open = coin.positions.swing_position.is_open()
                || (!scalp_is_open && coin.positions.position.is_open() && coin.positions.position.horizon() == quantum_arena::position::PositionHorizon::Swing);

            if swing_is_open {
                let pos = if coin.positions.swing_position.is_open() {
                    &coin.positions.swing_position
                } else {
                    &coin.positions.position
                };
                let is_long = pos.is_long.load(Ordering::Relaxed);
                let entry = pos.entry_price.load(Ordering::Relaxed);
                let qty = pos.quantity.load(Ordering::Relaxed);
                let entry_time = pos.entry_time_ms.load(Ordering::Relaxed);

                let pnl_pct = if is_long {
                    (mid_price - entry) / entry
                } else {
                    (entry - mid_price) / entry
                };

                let pseudo_atr = atr_pct_live * entry;
                let side_int = if is_long { 1 } else { -1 };
                let position_age_ms = if event_time_ms > 0 {
                    event_time_ms.saturating_sub(entry_time)
                } else {
                    0
                };

                // Trailing Swing: Activación tras 60s y PnL >= 50% TP
                let trail_activation_pnl = (swing_tp * 0.50).max(pseudo_atr / entry.max(1.0) * 2.5).clamp(0.0080, 0.0300);
                let trail_active = position_age_ms > 60_000 && pnl_pct >= trail_activation_pnl;

                let mut trail_hit = false;
                let mut force_close_trail = false;

                if trail_active {
                    let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                    let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                        side_int,
                        entry,
                        mid_price,
                        pseudo_atr,
                        pos.trailing_phase.load(Ordering::Relaxed) as i32,
                        pos.mfe_atr.load(Ordering::Relaxed),
                        pos.max_pnl_pct.load(Ordering::Relaxed),
                        pos.trail_stop.load(Ordering::Relaxed),
                        self.arena.config.swing_trail_atr_mult_base.load(Ordering::Relaxed).clamp(0.8, 6.0),
                        self.arena.config.swing_trail_act_atr.load(Ordering::Relaxed).clamp(0.8, 6.0),
                        self.arena.config.swing_trail_step_atr.load(Ordering::Relaxed).clamp(0.8, 6.0),
                        self.arena.config.swing_trail_max_atr.load(Ordering::Relaxed).clamp(0.8, 8.0),
                        self.arena.config.swing_trail_atr_mult_base.load(Ordering::Relaxed).clamp(0.8, 6.0),
                        live_fee,
                    );

                    let sl_floor = if is_long {
                        entry * (1.0 - swing_sl)
                    } else {
                        entry * (1.0 + swing_sl)
                    };
                    let safe_stop = if is_long {
                        trail_res.stop_price.max(sl_floor)
                    } else {
                        if trail_res.stop_price > 0.0 {
                            trail_res.stop_price.min(sl_floor)
                        } else {
                            sl_floor
                        }
                    };

                    let current_stored = pos.trail_stop.load(Ordering::Relaxed);
                    let final_stop = if is_long {
                        if current_stored > 0.0 {
                            safe_stop.max(current_stored)
                        } else {
                            safe_stop
                        }
                    } else {
                        if current_stored > 0.0 {
                            safe_stop.min(current_stored)
                        } else {
                            safe_stop
                        }
                    };

                    pos.trail_stop.store(final_stop, Ordering::Relaxed);
                    pos.trailing_phase.store(trail_res.new_phase as u8, Ordering::Relaxed);
                    pos.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
                    pos.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);

                    trail_hit = (is_long && mid_price <= final_stop)
                        || (!is_long && mid_price >= final_stop && final_stop > 0.0);
                    force_close_trail = trail_res.force_close;
                }

                let macro_t = self.feature_engines[coin_id].get_macro_trend();
                let macro_reversal = (is_long && macro_t < -0.0030) || (!is_long && macro_t > 0.0030);
                let swing_hard_timeout = position_age_ms > 259_200_000; // 72h max swing duration
                let is_swing_zombie = event_time_ms > 0 && position_age_ms > 14_400_000 && ((macro_reversal && pnl_pct <= -swing_sl * 0.70) || swing_hard_timeout);
                // D-38: Salida temprana en Swing por toxicidad de flujo microestructural sostenido
                let cur_vpin_sw = self.feature_engines[coin_id].cvpin.current_vpin();
                let ofi_adverse_sw = (is_long && ofi_value < -0.55) || (!is_long && ofi_value > 0.55);
                let toxic_flow_exit_sw = ofi_adverse_sw && cur_vpin_sw > 0.75 && pnl_pct < -0.0020;

                let tp_traded_through = if is_long {
                    bid >= entry * (1.0 + swing_tp)
                } else {
                    ask <= entry * (1.0 - swing_tp)
                };
                if tp_traded_through
                    || pnl_pct <= -swing_sl
                    || trail_hit
                    || force_close_trail
                    || is_swing_zombie
                    || toxic_flow_exit_sw
                {
                    let sl_price = if is_long {
                        entry * (1.0 - swing_sl)
                    } else {
                        entry * (1.0 + swing_sl)
                    };
                    let tp_price = if is_long {
                        entry * (1.0 + swing_tp)
                    } else {
                        entry * (1.0 - swing_tp)
                    };

                    let mut exit_price = mid_price;
                    if is_long {
                        if pnl_pct <= -swing_sl {
                            exit_price = exit_price.min(sl_price);
                        }
                        if trail_hit {
                            let ts = pos.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.max(ts); }
                        }
                        if pnl_pct >= swing_tp {
                            exit_price = tp_price;
                        }
                    } else {
                        if pnl_pct <= -swing_sl {
                            exit_price = exit_price.max(sl_price);
                        }
                        if trail_hit {
                            let ts = pos.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.min(ts); }
                        }
                        if pnl_pct >= swing_tp {
                            exit_price = tp_price;
                        }
                    }

                    let gross_pnl = if is_long {
                        (exit_price - entry) * qty
                    } else {
                        (entry - exit_price) * qty
                    };

                    let live_maker = self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002);
                    let live_taker = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004);
                    let close_fee_rate = if pnl_pct >= swing_tp { live_maker } else { live_taker };
                    let close_fee = (qty * exit_price) * close_fee_rate;
                    // T-01: predicción capturada antes del reset del close.
                    let ml_at_entry_sw = coin.positions.swing_position
                        .ml_prediction
                        .load(Ordering::Relaxed);
                    let (_, _, _, sw_margin, sw_entry_fee) = coin.positions.swing_position.close_with_fee();
                    // N-04: Cerrar coin.positions.position SOLO si actualmente espeja esta posición de Swing (horizon == 1)
                    let (gen_margin, gen_entry_fee) = if coin.positions.position.is_open()
                        && (coin.positions.position.horizon.load(Ordering::Relaxed) == 1)
                    {
                        let (_, _, _, gm, gef) = coin.positions.position.close_with_fee();
                        (gm, gef)
                    } else {
                        (0.0, 0.0)
                    };
                    let margin_used = sw_margin.max(gen_margin);
                    let entry_fee_paid = sw_entry_fee.max(gen_entry_fee);

                    let net_realized_pnl = gross_pnl - close_fee;
                    let net_trade_pnl = net_realized_pnl - entry_fee_paid;

                    let current_used = self.arena.used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena.used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.used_margin.store(0.0, Ordering::Relaxed);
                    }
                    let current_sw_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                    if current_sw_used >= margin_used {
                        self.arena.swing_used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.swing_used_margin.store(0.0, Ordering::Relaxed);
                    }

                    coin.metrics.pnl_realized.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    coin.swing.pnl_realized.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    self.arena.unified_capital.fetch_add(net_realized_pnl, Ordering::Relaxed);

                    let is_win = net_trade_pnl > 0.0;
                    if ml_at_entry_sw > 0.0 {
                        self.conformal.update(ml_at_entry_sw, is_win);
                    }
                    let n = coin.swing.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.swing.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.swing.win_rate.store(new_wr, Ordering::Relaxed);

                    // FIX D41: Profit Factor contable exacto para Swing (gross_wins / gross_losses)
                    if is_win {
                        coin.metrics.gross_wins.fetch_add(net_trade_pnl, Ordering::Relaxed);
                        coin.swing.gross_wins.fetch_add(net_trade_pnl, Ordering::Relaxed);
                    } else {
                        coin.metrics.gross_losses.fetch_add(net_trade_pnl.abs(), Ordering::Relaxed);
                        coin.swing.gross_losses.fetch_add(net_trade_pnl.abs(), Ordering::Relaxed);
                    }
                    let sw_wins = coin.swing.gross_wins.load(Ordering::Relaxed);
                    let sw_losses = coin.swing.gross_losses.load(Ordering::Relaxed);
                    let new_pf = if sw_losses > 0.0 {
                        (sw_wins / sw_losses).clamp(0.1, 10.0)
                    } else if sw_wins > 0.0 {
                        5.0
                    } else {
                        1.50
                    };
                    coin.swing.profit_factor.store(new_pf, Ordering::Relaxed);

                    // Calibrador conformal para Swing
                    let ml_at_entry = coin.positions.swing_position
                        .ml_prediction
                        .load(Ordering::Relaxed);
                    if ml_at_entry > 0.0 {
                        self.conformal.update(ml_at_entry, is_win);
                    }

                    // FIX D36: Actualizar kelly_fraction en coin.swing
                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                    let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                    let clamp_min = self.arena.config.kelly_clamp_min.load(Ordering::Relaxed);
                    let clamp_max = self.arena.config.kelly_clamp_max.load(Ordering::Relaxed);
                    let strategy_base = self.arena.config.swing_kelly_fraction.load(Ordering::Relaxed);
                    let kelly_f_swing = risk_engine::kelly::calculate_kelly_fraction(
                        new_wr,
                        new_pf,
                        curr_cap,
                        base_cap,
                        survival_ratio,
                        exp_mult,
                        clamp_min,
                        clamp_max,
                        strategy_base,
                    );
                    coin.swing.kelly_fraction.store(kelly_f_swing, Ordering::Relaxed);

                    coin.last_swing_close_ts.store(event_time_ms, Ordering::Relaxed);

                    closed_swing = Some((is_long, net_realized_pnl, qty));

                    // N-12: Retroalimentación causal al Consejo de Seniors tras cierre de Swing
                    let notional = (qty * entry).max(1.0);
                    let realized_ret = net_realized_pnl / notional;
                    if coin_id < self.last_swing_senior_signals.len() {
                        self.consejo_deliberacion.record_outcome(&self.last_swing_senior_signals[coin_id], realized_ret);
                    }

                    // D-34: Retroalimentación PPO continuo tras cierre de Swing
                    let hawkes_r_sw = self.feature_engines[coin_id].cvpin.current_vpin();
                    let ppo_features_sw = [
                        (ofi_value / 0.35).clamp(-1.5, 1.5),
                        (obi / 0.35).clamp(-1.5, 1.5),
                        hawkes_r_sw.clamp(0.0, 1.0),
                        lead_lag_div.clamp(-1.5, 1.5),
                        ((hurst_val - 0.50) * 2.0).clamp(-1.0, 1.0),
                    ];
                    self.ppo_engine.update_policy(
                        realized_ret,
                        &ppo_features_sw,
                        if is_long { 1.0 } else { -1.0 },
                        1.0,
                        0.05,
                        0.01,
                        0.20,
                        0.01,
                    );
                } else {
                    let notional = qty * entry;
                    let unrealized = pnl_pct * notional;
                    coin.swing.pnl_unrealized.store(unrealized, Ordering::Relaxed);
                }
            }

            // --- 2. EVALUAR ENTRADAS ---
            let tick = self.arena.tick_counter.load(Ordering::Relaxed);

            // --- Inteligencia On-Chain (Spot vs Futures Correlation - Fase 7) ---
            let spot_bid = coin.spot_bid.load(Ordering::Relaxed);
            let spot_ask = coin.spot_ask.load(Ordering::Relaxed);
            let mut spot_bias = 0.0;

            if spot_bid > 0.0 && spot_ask > 0.0 {
                let spot_mid = (spot_bid + spot_ask) / 2.0;
                let spread_bps = ((spot_mid - mid_price) / mid_price) * 10000.0;
                if spread_bps > 1.5 {
                    spot_bias = 0.15; // Spot lidera al alza -> Presión compradora real
                } else if spread_bps < -1.5 {
                    spot_bias = -0.15; // Spot lidera a la baja -> Presión vendedora real
                }
            }

            let hurst_val = coin.hurst_exponent.load(Ordering::Relaxed);
            let obi_val = if bid_qty + ask_qty > 0.0 { (bid_qty - ask_qty) / (bid_qty + ask_qty) } else { 0.0 };
            let flow_dir = if bid_qty > ask_qty { 1.0 } else if ask_qty > bid_qty { -1.0 } else { 0.0 };
            let v_t = self.feature_engines[coin_id].v_t;
            let shannon_ent = self.feature_engines[coin_id].entropy.current();
            let vpin_val = self.feature_engines[coin_id].cvpin.current_vpin();
            let total_vol = (bid_qty + ask_qty).max(1e-8);
            let p_bid = (bid_qty / total_vol).clamp(0.0001, 0.9999);
            let p_ask = (ask_qty / total_vol).clamp(0.0001, 0.9999);
            let tsallis_ent = ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5).clamp(0.0, 1.0);

            let micro_v = (v_t.abs() / mid_price.max(1e-8)).clamp(atr_pct * 0.1, atr_pct * 5.0);
            // FIX D38: OmniscientRegistry Namespacing — escribe a nivel global, por coin_id y por símbolo
            let set_reg = |key: &str, val: f64| {
                self.arena.registry.set(key, val);
                self.arena.registry.set_for_coin(coin_id, key, val);
                if !sym.is_empty() {
                    self.arena.registry.set_scoped(&sym, key, val);
                }
            };
            let dir_v = self.feature_engines[coin_id].dir_velocity;
            let a_t = self.feature_engines[coin_id].a_t;
            let spread_pct = if mid_price > 0.0 { (ask - bid) / mid_price } else { 0.0002 };
            let spread_val = (spread_pct * mid_price).max(0.0001);
            let speed_of_sound = spread_val.max(micro_v * mid_price);
            set_reg("spread_speed_of_sound", speed_of_sound);
            set_reg("price_velocity", dir_v);
            set_reg("order_flow_velocity", dir_v);
            set_reg("price_acceleration", a_t);
            set_reg("atr_1s", (micro_v * mid_price).max(0.0001));
            set_reg("atr_5s", (self.feature_engines[coin_id].v_t * 0.5 + atr_pct * mid_price * 0.5).max(0.0005));
            set_reg("atr_1m", (atr_pct * mid_price * 7.746).max(0.0020));
            set_reg("order_flow_direction", flow_dir);
            set_reg("order_book_imbalance", obi_val);
            set_reg("orderbook_imbalance", obi_val);
            set_reg("order_flow_imbalance", ofi_value);
            // D-40: Señal sub-umbral para StochasticResonanceEngine
            set_reg("weak_alpha_signal", (ofi_value * 0.5 + obi_val * 0.5).clamp(-1.0, 1.0));
            // D-36: Alpha de liderazgo Macro Lead-Lag
            set_reg("lead_lag_divergence", lead_lag_div);
            set_reg("leader_momentum", leader_mom);
            set_reg("shannon_entropy", shannon_ent);
            set_reg("tsallis_q_entropy", tsallis_ent);
            set_reg("hurst_exponent", hurst_val);
            set_reg("global_hurst", hurst_val);
            set_reg("atr_pct", atr_pct);
            set_reg("relative_atr_pct", atr_pct);
            set_reg("vpin", vpin_val);
            set_reg("vpin_toxicity", vpin_val);
            set_reg("cvpin", vpin_val);
            set_reg("order_flow_vpin", vpin_val);
            set_reg("hawkes_intensity", (1.0 + obi_val.abs() * 2.0).clamp(0.1, 5.0));
            set_reg("bessel_alpha", 1.5);
            set_reg("hawkes_dt", 0.05);
            set_reg("microstructure_noise_variance", (atr_pct * 0.1).max(0.00001));

            let hebbian_mult = self.arena.registry.get("perceptron_hebbian_weight", "GodEngineCore")
                .map(|p| p.get_value())
                .unwrap_or(1.0)
                .clamp(0.5, 2.0);

            // 54D Features
            let features = self.feature_engines[coin_id].get_features();
            let base_ml_prob = if let Some(f) = &self.scalp_forest {
                f.predict(&features).unwrap_or(0.5) as f64
            } else {
                0.5
            };
            let ml_prob = (base_ml_prob + spot_bias).clamp(0.0, 1.0);
            self.last_ml_prob = ml_prob as f32;
            coin.ml_prob.store(ml_prob, Ordering::Relaxed);
            set_reg("ml_prob", ml_prob);
            set_reg("ml_prob_scalp", ml_prob);

            let nn_score: f64 = self.feature_engines[coin_id].update_ml_prediction(ml_prob);

            let current_obi = obi_val;
            let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
            let dynamic_obi_thr = self.arena.config.dynamic_obi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);
            let dynamic_ema_thr = self.arena.config.dynamic_ema_trend.load(Ordering::Relaxed);
            let dynamic_ofi_thr = self.arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);

            let micro_trend = self.feature_engines[coin_id].get_micro_trend();
            let macro_trend = self.feature_engines[coin_id].get_macro_trend();

            set_reg("ema_trend", micro_trend);
            set_reg("ema_trend_swing", macro_trend);
            set_reg("trend_direction", if macro_trend > 0.0 { 1.0 } else if macro_trend < 0.0 { -1.0 } else { 0.0 });
            // R4.3 — p-valor CONFORMAL CALCULADO (antes constante 0.95 —
            // tautología). El filtro swing/escalp now rechaza de verdad.
            let conf_alpha = self.arena.config.conformal_alpha.load(Ordering::Relaxed).clamp(0.01, 0.30);
            let conformal_p = self.conformal.p_value(
                coin.ml_prob.load(Ordering::Relaxed),
            );
            set_reg("conformal_p_value", conformal_p);
            set_reg("conformal_alpha", conf_alpha);
            let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
            let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
            let total_vol_cvd = buy_vol + sell_vol;
            let rolling_cvd = if total_vol_cvd > 0.0 { (buy_vol - sell_vol) / total_vol_cvd } else { 0.0 };
            let ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;

            let obi_norm = (current_obi / dynamic_obi_thr).clamp(-1.5, 1.5);
            let ofi_norm = (ofi / dynamic_ofi_thr).clamp(-1.5, 1.5);
            // FIX D-49: vecm_zscore mide desviación de base / reversión a la media, NO Order Book Imbalance invertido
            let vecm_basis_z = if spot_bid > 0.0 && spot_ask > 0.0 {
                let spot_mid = (spot_bid + spot_ask) / 2.0;
                ((mid_price - spot_mid) / (mid_price * atr_pct.max(0.0005))).clamp(-3.0, 3.0)
            } else if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                ((mid_price - self.feature_engines[coin_id].kline_ema_slow) / (mid_price * atr_pct.max(0.0005))).clamp(-3.0, 3.0)
            } else {
                0.0
            };
            set_reg("vecm_zscore", vecm_basis_z);
            set_reg("cointegration_zscore", vecm_basis_z);
            let ema_macro = if self.feature_engines[coin_id].kline_ema_slow > 0.0 { self.feature_engines[coin_id].kline_ema_slow } else { mid_price };
            let pos_dev = ((mid_price - ema_macro) / (mid_price * atr_pct.max(0.0005))).clamp(-3.0, 3.0);
            set_reg("quantum_position_deviation", pos_dev);
            // D-34: PPO Policy Engine adaptativo continuo ponderando microestructura
            let hawkes_intensity = (1.0 + current_obi.abs() * 2.0).clamp(0.1, 5.0) / 5.0;
            let regime_code = ((hurst_val - 0.50) * 2.0).clamp(-1.0, 1.0);
            let ppo_state = [
                ofi_norm,
                obi_norm,
                hawkes_intensity,
                lead_lag_div.clamp(-1.5, 1.5),
                regime_code,
            ];
            let ppo_score = self.ppo_engine.evaluate_policy(&ppo_state);
            let micro_score: f64 = (ppo_score * 0.70 + rolling_cvd * 0.30).clamp(-1.0, 1.0);

            let (tensor_scalp, tensor_swing) = self.tensor_orchestrator.evaluate_dual_consensus();
            let tensor_boost = match tensor_scalp.signal {
                SignalType::Long => tensor_scalp.net_confidence.clamp(0.0, 1.0),
                SignalType::Short => -tensor_scalp.net_confidence.clamp(0.0, 1.0),
                SignalType::Flat => 0.0,
            };

            // T-02 — MOTOR DE MICROESTRUCTURA CONECTADO: 30 instancias de
            // ScalpEngine (aceleración OBI con z-score EW-Welford) vivían
            // SIN evaluarse — la mitad del espectro temporal estaba muerta.
            // Se evalúa cada tick y actúa como ruta de intención propia
            // cuando el composite de régimen no dispara.
            let micro_accel_intent = self.scalp_engines[coin_id]
                .evaluate_microstructure_with_vpin(bid_qty, ask_qty, vpin_val, 1.5, &self.arena);

            // Unified Bayesian Fusion: 40% Microstructure L2 (OBI/OFI/CVD) + 35% DarkAlpha ML + 25% Tensor Consensus
            let raw_composite = micro_score * 0.40 + nn_score * 0.35 + tensor_boost * 0.25;
            let composite_score: f64 = (raw_composite * hebbian_mult).clamp(-1.0, 1.0);

            let mut scalp_intent = SignalIntent::flat();
            let spread_pct = if mid_price > 0.0 { (ask - bid) / mid_price } else { 0.0 };
            let dynamic_max_spread = (atr_pct * 0.25).clamp(0.0006, 0.0025);
            let spread_ok = spread_pct <= dynamic_max_spread;

            if atr_pct > dynamic_atr_min && spread_ok && self.feature_engines[coin_id].can_open_scalp(30) {
                let is_mean_reverting = hurst_val < 0.45;
                let is_trending = hurst_val >= 0.50;

                let ema_slow = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let price_stretch = if ema_slow > 0.0 { (mid_price - ema_slow) / cur_atr } else { 0.0 };
                let not_overextended_long = price_stretch <= 1.2;
                let not_overextended_short = price_stretch >= -1.2;

                if is_trending {
                    let dynamic_tech_thr = self.arena.config.tech_threshold.load(Ordering::Relaxed);
                    // Long Scalp: Macro-EMA alcista + Precio por encima de EMA lenta (Macro Bullish) con confluencia positiva robusta
                    if macro_trend > dynamic_ema_thr && mid_price >= ema_slow && composite_score > dynamic_tech_thr && not_overextended_long {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: (0.70f64 + composite_score.max(0.0) * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    // Short Scalp: Macro-EMA bajista + Precio por debajo de EMA lenta (Macro Bearish) con confluencia negativa robusta
                    } else if macro_trend < -dynamic_ema_thr && mid_price <= ema_slow && composite_score < -dynamic_tech_thr && not_overextended_short {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: (0.70f64 + composite_score.min(0.0).abs() * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    }
                } else if is_mean_reverting {
                    // Fade overextensions in oscillating market (Strict Anti-Knife):
                    // Fade Top (Short): Price stretched high + sellers taking control on the book + macro not bull
                    if price_stretch > 1.2 && current_obi < -dynamic_obi_thr * 0.8 && macro_trend <= dynamic_ema_thr * 2.0 {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: (0.70f64 + current_obi.abs() * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    // Fade Bottom (Long): Price stretched low + buyers taking control on the book + macro not bear
                    } else if price_stretch < -1.2 && current_obi > dynamic_obi_thr * 0.8 && macro_trend >= -dynamic_ema_thr * 2.0 {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: (0.70f64 + current_obi * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    }
                } else {
                    // Zona Neutra de Hurst (0.45 <= H < 0.50): Seguimiento de momentum por confluencia de flujo L2
                    let dynamic_tech_thr = self.arena.config.tech_threshold.load(Ordering::Relaxed);
                    if composite_score > dynamic_tech_thr * 1.15 && current_obi > dynamic_obi_thr * 0.7 && not_overextended_long {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: (0.65f64 + composite_score.max(0.0) * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    } else if composite_score < -dynamic_tech_thr * 1.15 && current_obi < -dynamic_obi_thr * 0.7 && not_overextended_short {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: (0.65f64 + composite_score.min(0.0).abs() * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    }
                }

                if scalp_intent.signal == SignalType::Flat && tensor_scalp.signal != SignalType::Flat && tensor_scalp.net_confidence.abs() > 0.65 {
                    scalp_intent = SignalIntent {
                        signal: tensor_scalp.signal,
                        confidence: tensor_scalp.net_confidence.abs().clamp(0.65, 1.0),
                        ..Default::default()
                    };
                }

                // T-02 — fallback de microestructura pura: el z-score de
                // aceleración OBI con confianza suficiente genera la
                // intención aunque el composite de régimen no dispare.
                if scalp_intent.signal == SignalType::Flat
                    && micro_accel_intent.signal != SignalType::Flat
                    && micro_accel_intent.confidence >= 0.60
                    && atr_pct > dynamic_atr_min
                {
                    scalp_intent = micro_accel_intent;
                }

                // FIX D-59: Conectar TurboScalpEngine cuando el ensamble primario está en Flat
                if scalp_intent.signal == SignalType::Flat {
                    let hawkes_r = self.feature_engines[coin_id].cvpin.current_vpin();
                    if let Some(turbo_intent) = signal_engine::turbo_scalper::TurboScalpEngine::evaluate_turbo_scalp(
                        &self.arena, current_obi, ofi, hawkes_r, shannon_ent, mid_price, atr_pct, event_time_ms
                    ) {
                        if (turbo_intent.signal == SignalType::Long && not_overextended_long)
                            || (turbo_intent.signal == SignalType::Short && not_overextended_short)
                        {
                            scalp_intent = turbo_intent;
                        }
                    }
                }

                // Cooldown: 5s post-close para agilidad en micro-scalping
                let last_close = coin.last_scalp_close_ts.load(Ordering::Relaxed);
                if event_time_ms.saturating_sub(last_close) < 5_000 {
                    scalp_intent = SignalIntent::flat();
                }
            }

            // --- CVD & L2 Wall HARD FILTERS (VETOS) ---
            if scalp_intent.signal != SignalType::Flat {
                let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
                let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
                let cvd = buy_vol - sell_vol;
                let total_vol_cvd = buy_vol + sell_vol;
                let cvd_ratio = if total_vol_cvd > 0.0 {
                    cvd / total_vol_cvd
                } else {
                    0.0
                };

                let bid_wall = coin.l2_bid_wall.load(Ordering::Relaxed);
                let ask_wall = coin.l2_ask_wall.load(Ordering::Relaxed);
                let total_wall = bid_wall + ask_wall;
                let wall_imbalance = if total_wall > 0.0 {
                    (bid_wall - ask_wall) / total_wall
                } else {
                    0.0
                };

                let cvd_veto = self.arena.config.cvd_veto_threshold.load(Ordering::Relaxed);
                let wall_veto = self.arena.config.wall_veto_threshold.load(Ordering::Relaxed);

                if scalp_intent.signal == SignalType::Long {
                    if cvd_ratio < -cvd_veto {
                        scalp_intent = SignalIntent::flat();
                    } else if wall_imbalance < -wall_veto {
                        scalp_intent = SignalIntent::flat();
                    }
                } else if scalp_intent.signal == SignalType::Short {
                    if cvd_ratio > cvd_veto {
                        scalp_intent = SignalIntent::flat();
                    } else if wall_imbalance > wall_veto {
                        scalp_intent = SignalIntent::flat();
                    }
                }
            }

            if let Some(fr) = &self.flight_recorder {
                let mut payload = [0u8; 47];
                let prob_bytes = (ml_prob as f32).to_le_bytes();
                payload[0..4].copy_from_slice(&prob_bytes);
                fr.record(telemetry_server::FlightEvent {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64,
                    trace_id: tick,
                    event_type: 1, // 1 = ML Prediction
                    payload,
                });
            }

            let hurst_exponent = features[1] as f64; // [1] = Hurst
            let mut swing_nn_pred = coin.ml_prob.load(Ordering::Relaxed);
            if atr_pct > 0.0001 {
                let combined = self.build_54d_tensor(coin_id, bid_qty, ask_qty, mid_price, omni_features);
                let sf = self.feature_engines[coin_id].get_swing_features();
                if let Some(nn) = &mut self.swing_nn {
                    let in_dim = nn.layer1.in_features;
                    let p_opt = if in_dim == 34 {
                        let mut swing_feats_f64 = [0.0; 34];
                        for idx in 0..34 {
                            swing_feats_f64[idx] = sf[idx] as f64;
                        }
                        nn.predict_for_coin(coin_id, &swing_feats_f64)
                    } else if in_dim == 12 {
                        let mut micro_f64 = [0.0; 12];
                        for idx in 0..12 {
                            micro_f64[idx] = features[idx] as f64;
                        }
                        nn.predict_for_coin(coin_id, &micro_f64)
                    } else {
                        nn.predict_for_coin(coin_id, &combined)
                    };
                    if let Some(p) = p_opt {
                        swing_nn_pred = p.clamp(0.01, 0.99);
                    }
                }
            }

            // --- EVALUACIÓN DE SEÑAL SWING INDEPENDIENTE ---
            let mut swing_intent = SignalIntent::flat();
            let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
            let trend_intent = self.swing_engines[coin_id].evaluate_trend(
                mid_price,
                hurst_exponent,
                trend_threshold,
                swing_nn_pred,
                &self.arena,
            );
            if trend_intent.signal != SignalType::Flat {
                swing_intent = trend_intent;
            } else if tensor_swing.signal != SignalType::Flat && tensor_swing.net_confidence.abs() > 0.60 {
                swing_intent = SignalIntent {
                    signal: tensor_swing.signal,
                    confidence: tensor_swing.net_confidence.abs().clamp(0.60, 1.0),
                    ..Default::default()
                };
            }

            // Swing Cooldown: 30s
            let last_sw_close = coin.last_swing_close_ts.load(Ordering::Relaxed);
            if event_time_ms.saturating_sub(last_sw_close) < 30_000 {
                swing_intent = SignalIntent::flat();
            }

            if coin_id < self.last_scalp_intent.len() {
                self.last_scalp_intent[coin_id] = scalp_intent;
            }
            if coin_id < self.last_swing_intent.len() {
                self.last_swing_intent[coin_id] = swing_intent;
            }

            let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
            let _global_leverage = self.arena.config.global_leverage.load(Ordering::Relaxed).clamp(1.0, 50.0);

            // Zero phantom equity check: si el capital fue liquidado, detener inmediatamente
            if current_cap <= 0.0 {
                self.arena.kill_switch_active.store(true, Ordering::SeqCst);
                return (None, None, closed_scalp, closed_swing, None);
            }

            let mut new_scalp = None;
            let mut new_swing = None;

            // --- APERTURA SCALPING (INDEPENDIENTE) ---
            if scalp_intent.signal != SignalType::Flat && !coin.positions.scalp_position.is_open() {
                let order = self.risk_engine.evaluate_quantum_order_by_horizon(coin_id, &scalp_intent, true, &self.arena);
                if order.signal != SignalType::Flat {
                    // R4.1: Inputs dinámicos reales para deliberación del Consejo de Seniors
                    let scalp_drawdown = if self.risk_engine.scalp_peak_capital > 0.0 {
                        let split = self.risk_engine.smoothed_split.unwrap_or(0.5);
                        let scalp_cap = current_cap * split;
                        ((self.risk_engine.scalp_peak_capital - scalp_cap) / self.risk_engine.scalp_peak_capital).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let vpin_risk = self.feature_engines[coin_id].cvpin.current_vpin().clamp(0.0, 1.0);
                    // D-36: Usar el LeadLagAlphaEngine centralizado actualizado con BTC/ETH
                    let (impulse_mom, _) = self.lead_lag_engine.predict_altcoin_impulse(obi);
                    let graph_corr = impulse_mom.clamp(-1.0, 1.0);

                    let current_spread_bps = if mid_price > 1e-8 && ask >= bid {
                        ((ask - bid) / mid_price) * 10_000.0
                    } else {
                        0.5
                    };
                    let scalp_slip_bps = ((current_spread_bps * 0.5) + 0.5).clamp(0.5, 500.0);

                    let council_snapshot = metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                        horizon: metacortex_engine::consejo_seniors::TradingHorizon::Scalping,
                        book_imbalance: obi,
                        hurst_exponent: hurst_val.clamp(0.0, 1.0),
                        graph_correlation: graph_corr,
                        do_calculus_risk: vpin_risk,
                        causal_veto_threshold: 0.60,
                        current_drawdown_pct: scalp_drawdown,
                        estimated_slippage_bps: scalp_slip_bps,
                    };
                    let wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                    let senior_sigs = self.consejo_deliberacion.extract_senior_signals(&council_snapshot, wr);
                    if coin_id < self.last_scalp_senior_signals.len() {
                        self.last_scalp_senior_signals[coin_id] = senior_sigs;
                    }
                    let deliberation = self.consejo_deliberacion.deliberar_with_weights(&council_snapshot, wr, None);
                    if !deliberation.approved {
                        self.diag_council_vetoes += 1;
                    }

                    if deliberation.approved {
                        let is_long = order.signal == SignalType::Long;
                        let total_used = self.arena.used_margin.load(Ordering::Relaxed);
                        let free_cap = (current_cap - total_used).max(0.0);

                        let eff_leverage = order.leverage.clamp(1.0, 50.0);
                        let min_margin = 5.05 / eff_leverage;
                        let max_margin = (free_cap * 0.85).max(0.0);
                        let mut margin_req = order.volume_usd.clamp(min_margin, max_margin);
                        let max_pos = 50000.0;
                        if margin_req * eff_leverage > max_pos {
                            margin_req = max_pos / eff_leverage;
                        }

                        if margin_req * eff_leverage >= 5.0 && total_used + margin_req <= current_cap * 0.95 {
                            self.diag_opened += 1;
                            self.arena.used_margin.fetch_add(margin_req, Ordering::Relaxed);
                            self.arena.scalp_used_margin.fetch_add(margin_req, Ordering::Relaxed);

                            let base_price = if is_long { ask } else { bid };
                            let nominal_size = margin_req * eff_leverage;
                            let slippage_impact = (nominal_size / 1_000_000.0) * 0.0005;
                            let real_entry_price = if is_long {
                                base_price * (1.0 + slippage_impact)
                            } else {
                                base_price * (1.0 - slippage_impact)
                            };

                            let entry_fee_rate = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0002);
                            let fee_paid = nominal_size * entry_fee_rate;
                            self.arena.unified_capital.fetch_add(-fee_paid, Ordering::Relaxed);

                            let qty = nominal_size / real_entry_price;

                            coin.positions.scalp_position.open_with_fee(
                                is_long,
                                real_entry_price,
                                qty,
                                margin_req,
                                event_time_ms,
                                order.tp_target,
                                order.sl_target,
                                quantum_arena::position::PositionHorizon::Scalping,
                                ml_prob,
                                scalp_intent.confidence,
                                fee_paid,
                            );
                            if !coin.positions.position.is_open() {
                                coin.positions.position.open_with_fee(
                                    is_long,
                                    real_entry_price,
                                    qty,
                                    margin_req,
                                    event_time_ms,
                                    order.tp_target,
                                    order.sl_target,
                                    quantum_arena::position::PositionHorizon::Scalping,
                                    ml_prob,
                                    scalp_intent.confidence,
                                    fee_paid,
                                );
                            }

                            new_scalp = Some((is_long, real_entry_price, qty));
                        }
                    }
                }
            }

            // --- APERTURA SWING (INDEPENDIENTE) ---
            if swing_intent.signal != SignalType::Flat && !coin.positions.swing_position.is_open() {
                let order = self.risk_engine.evaluate_quantum_order_by_horizon(coin_id, &swing_intent, false, &self.arena);
                if order.signal != SignalType::Flat {
                    // R4.1: Inputs dinámicos reales para deliberación del Consejo de Seniors
                    let swing_drawdown = if self.risk_engine.swing_peak_capital > 0.0 {
                        let split = self.risk_engine.smoothed_split.unwrap_or(0.5);
                        let swing_cap = current_cap * (1.0 - split);
                        ((self.risk_engine.swing_peak_capital - swing_cap) / self.risk_engine.swing_peak_capital).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let vpin_risk = self.feature_engines[coin_id].cvpin.current_vpin().clamp(0.0, 1.0);
                    // D-36: Usar el LeadLagAlphaEngine centralizado actualizado con BTC/ETH
                    let (impulse_mom, _) = self.lead_lag_engine.predict_altcoin_impulse(obi);
                    let graph_corr = impulse_mom.clamp(-1.0, 1.0);

                    let current_spread_bps = if mid_price > 1e-8 && ask >= bid {
                        ((ask - bid) / mid_price) * 10_000.0
                    } else {
                        0.5
                    };
                    let swing_slip_bps = ((current_spread_bps * 0.5) + 1.0).clamp(1.0, 500.0);

                    let council_snapshot = metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                        horizon: metacortex_engine::consejo_seniors::TradingHorizon::Swing,
                        book_imbalance: obi,
                        hurst_exponent: hurst_val.clamp(0.0, 1.0),
                        graph_correlation: graph_corr,
                        do_calculus_risk: vpin_risk,
                        causal_veto_threshold: 0.60,
                        current_drawdown_pct: swing_drawdown,
                        estimated_slippage_bps: swing_slip_bps,
                    };
                    let wr = coin.swing.win_rate.load(Ordering::Relaxed);
                    let senior_sigs = self.consejo_deliberacion.extract_senior_signals(&council_snapshot, wr);
                    if coin_id < self.last_swing_senior_signals.len() {
                        self.last_swing_senior_signals[coin_id] = senior_sigs;
                    }
                    let deliberation = self.consejo_deliberacion.deliberar_with_weights(&council_snapshot, wr, None);
                    if !deliberation.approved {
                        self.diag_swing_vetoes += 1;
                        if self.diag_swing_vetoes <= 3 || self.diag_swing_vetoes % 500 == 0 {
                            println!(
                                "🔍 [VETO-SWING] #{} por {:?} | obi={:.3} hurst={:.3} vpin={:.3} graph={:.3} consensus={:.3} final_sig={:.3}",
                                self.diag_swing_vetoes, deliberation.vetoed_by, obi,
                                hurst_val, vpin_risk, graph_corr,
                                deliberation.total_consensus_pct, deliberation.final_signal
                            );
                        }
                    }

                    if deliberation.approved {
                        let is_long = order.signal == SignalType::Long;
                        let total_used = self.arena.used_margin.load(Ordering::Relaxed);
                        let free_cap = (current_cap - total_used).max(0.0);

                        let eff_leverage = order.leverage.clamp(1.0, 50.0);
                        let min_margin = 5.05 / eff_leverage;
                        let max_margin = (free_cap * 0.85).max(0.0);
                        let mut margin_req = order.volume_usd.clamp(min_margin, max_margin);
                        let max_pos = 50000.0;
                        if margin_req * eff_leverage > max_pos {
                            margin_req = max_pos / eff_leverage;
                        }

                        if margin_req * eff_leverage >= 5.0 && total_used + margin_req <= current_cap * 0.95 {
                            self.diag_swing_opened += 1;
                            self.arena.used_margin.fetch_add(margin_req, Ordering::Relaxed);
                            self.arena.swing_used_margin.fetch_add(margin_req, Ordering::Relaxed);

                            let base_price = if is_long { ask } else { bid };
                            let nominal_size = margin_req * eff_leverage;
                            let slippage_impact = (nominal_size / 1_000_000.0) * 0.0005;
                            let real_entry_price = if is_long {
                                base_price * (1.0 + slippage_impact)
                            } else {
                                base_price * (1.0 - slippage_impact)
                            };

                            let entry_fee_rate = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0002);
                            let fee_paid = nominal_size * entry_fee_rate;
                            self.arena.unified_capital.fetch_add(-fee_paid, Ordering::Relaxed);

                            let qty = nominal_size / real_entry_price;

                            coin.positions.swing_position.open_with_fee(
                                is_long,
                                real_entry_price,
                                qty,
                                margin_req,
                                event_time_ms,
                                order.tp_target,
                                order.sl_target,
                                quantum_arena::position::PositionHorizon::Swing,
                                swing_nn_pred,
                                swing_intent.confidence,
                                fee_paid,
                            );
                            if !coin.positions.position.is_open() {
                                coin.positions.position.open_with_fee(
                                    is_long,
                                    real_entry_price,
                                    qty,
                                    margin_req,
                                    event_time_ms,
                                    order.tp_target,
                                    order.sl_target,
                                    quantum_arena::position::PositionHorizon::Swing,
                                    swing_nn_pred,
                                    swing_intent.confidence,
                                    fee_paid,
                                );
                            }

                            new_swing = Some((is_long, real_entry_price, qty));
                        }
                    }
                }
            }

            // --- 3. MARKET MAKING ---
            let mut final_maker_quote = None;
            let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);

            if latency_ms <= 25 {
                // Generar cotización pasiva basada en ATR/Hurst e inventario.
                let volatility = self.feature_engines[coin_id].get_features()[3] as f64; // [3] = v_t (volatility), NOT Hurst

                let mut inventory_delta_usd = 0.0;
                if coin.positions.scalp_position.is_open() {
                    let is_long = coin.positions.scalp_position.is_long.load(Ordering::Relaxed);
                    let notional = coin.positions.scalp_position.quantity.load(Ordering::Relaxed) * mid_price;
                    inventory_delta_usd += if is_long { notional } else { -notional };
                }
                if coin.positions.swing_position.is_open() {
                    let is_long = coin.positions.swing_position.is_long.load(Ordering::Relaxed);
                    let notional = coin.positions.swing_position.quantity.load(Ordering::Relaxed) * mid_price;
                    inventory_delta_usd += if is_long { notional } else { -notional };
                }

                let maker_spread_pct = self.arena.config.maker_spread_pct.load(Ordering::Relaxed);
                let maker_obi_threshold = self
                    .arena
                    .config
                    .maker_obi_threshold
                    .load(Ordering::Relaxed);

                let tensor_poly_a = self.arena.config.tensor_poly_a.load(Ordering::Relaxed);
                let tensor_poly_b = self.arena.config.tensor_poly_b.load(Ordering::Relaxed);

                final_maker_quote = Some(self.maker_engines[coin_id].generate_quote(
                    bid,
                    ask,
                    bid_qty,
                    ask_qty,
                    inventory_delta_usd,
                    volatility,
                    maker_spread_pct,
                    maker_obi_threshold,
                    tensor_poly_a,
                    tensor_poly_b,
                ));
            }

            (
                new_scalp,
                new_swing,
                closed_scalp,
                closed_swing,
                final_maker_quote,
            )
        })
    }

    /// Procesa un tick y devuelve las órdenes generadas (si hay).
    /// Mantiene compatibilidad 100% con tests y llamadores legados.
    /// Retorna: (NuevaPosicion, PosicionCerrada, MakerQuote)
    #[inline(always)]
    pub fn process_tick(
        &mut self,
        coin_id: usize,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        event_time_ms: u64,
        omni_features: &[f64; 54],
    ) -> (
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<MakerQuote>,
    ) {
        let (new_sc, new_sw, closed_sc, closed_sw, maker) = self.process_tick_dual(
            coin_id,
            bid,
            ask,
            bid_qty,
            ask_qty,
            event_time_ms,
            omni_features,
        );
        (new_sc.or(new_sw), closed_sc.or(closed_sw), maker)
    }
}
