#![feature(portable_simd)]

pub mod bootloader;
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

use crate::stateful_engine::{MarketRegime, StatefulEngine};
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
    pub lakehouse: Option<Arc<storage_engine::LakehouseWarehouse>>,
    pub consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion,
    pub lead_lag_engine: feature_engine::LeadLagAlphaEngine,
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
            if let Ok(model) =
                serde_json::from_str::<dark_alpha_engine::DarkAlphaEngine>(&json_data)
            {
                telemetry_server::telemetry_log!(
                    "🧠 [DARK ALPHA] Initial Boot Model Loaded Successfully (in_features: {})",
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
            lakehouse: None,
            consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion::new(),
            lead_lag_engine: feature_engine::LeadLagAlphaEngine::new(50),
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
        combined[41] = if omni_features.len() > 4 && omni_features[4] > 0.0 {
            (omni_features[4] / 100.0).clamp(0.5, 2.0)
        } else {
            1.04
        };
        combined[42] = if omni_features.len() > 0 && omni_features[0] > 0.0 {
            (omni_features[0] / 5000.0).clamp(0.5, 2.0)
        } else {
            1.02
        };
        combined[43] = if omni_features.len() > 1 && omni_features[1] > 0.0 {
            (omni_features[1] / 18000.0).clamp(0.5, 2.0)
        } else {
            1.00
        };
        combined[44] = if omni_features.len() > 2 && omni_features[2] > 0.0 {
            (omni_features[2] / 20.0).clamp(0.1, 5.0)
        } else {
            0.75
        };
        combined[45] = if omni_features.len() > 3 && omni_features[3] > 0.0 {
            (omni_features[3] / 4.0).clamp(0.1, 5.0)
        } else {
            1.05
        };
        combined[46] = 1.0;
        combined[47] = if omni_features.len() > 5 && omni_features[5] > 0.0 {
            (omni_features[5] / 80.0).clamp(0.2, 3.0)
        } else {
            1.00
        };
        combined[48] = 0.0;
        combined[49] = 5.25;
        combined[50] = 1.0;
        combined[51] = 0.0;
        combined[52] = 0.0;
        combined[53] = 0.0;

        combined
    }

    /// Actualiza los modelos cargados en caliente si hubieron reentrenamientos asíncronos
    pub fn refresh_models(&mut self) {
        self.scalp_forest = crate::ml_inference::NanoForest::get_global("BTCUSDT_SCALP");
    }

    /// Procesa un evento (trade, kline, depth) y devuelve las órdenes generadas (si hay).
    /// Retorna: (NuevoScalp, NuevoSwing, CerradoScalp, CerradoSwing)
    /// Donde cada Option es (is_long, entry/exit_price, qty)
    #[inline(always)]
    pub fn process_event(
        &mut self,
        coin_id: usize,
        is_trade: bool,
        is_kline_closed: bool,
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
        latency_panic: bool,
        omni_features: &[f64; 54],
    ) -> (
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
    ) {
        telemetry_server::profile_node!("GodEngineCore::process_event", {
            // Zero-copy Hot-Reloading Check
            if let Some(rx) = &self.model_rx {
                if let Ok(new_model) = rx.try_recv() {
                    self.swing_nn = Some(new_model);
                    telemetry_server::telemetry_log!(
                        "🧠 [DARK ALPHA] Hot-Reload Successful! New weights absorbed in Zero-Copy."
                    );
                }
            }

            if self.arena.kill_switch_active.load(Ordering::Relaxed) {
                return (None, None, None, None);
            }

            // Auto-refresh ML models once every 1000 ticks to support zero-downtime hot-reload
            if self
                .arena
                .tick_counter
                .load(Ordering::Relaxed)
                .is_multiple_of(1000)
            {
                self.refresh_models();
            }

            if is_trade {
                // Signal intent is polled dynamically, no stateful tick processing required for scalp engine here.
            }
            if is_kline_closed {
                // Signal intent is polled dynamically
            }
            if is_depth {
                let mid_price = (bid + ask) / 2.0;
                let total_vol = bid_qty + ask_qty;
                let pseudo_maker = bid_qty > ask_qty;
                let feature_engine = &mut self.feature_engines[coin_id];
                feature_engine.process_tick(mid_price, total_vol, event_time_ms);
                feature_engine.update_trade_flow(total_vol, pseudo_maker);
                let ofi_val = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
                let sym = quantum_arena::symbol_registry::try_spec(coin_id).map(|s| s.symbol).unwrap_or_default();
                if sym.starts_with("BTC") {
                    self.lead_lag_engine.update_leader(true, ofi_val);
                } else if sym.starts_with("ETH") {
                    self.lead_lag_engine.update_leader(false, ofi_val);
                }
                feature_engine.update_macro_features(
                    depth_obi,
                    depth_micro_div,
                    0.0,
                    event_time_ms,
                );
                let raw_atr_pct = feature_engine.get_atr_pct();
                self.arena.coins[coin_id].current_price.store(mid_price, Ordering::Relaxed);
                self.arena.coins[coin_id].current_atr.store(raw_atr_pct * mid_price, Ordering::Relaxed);

                self.arena
                    .update_market_data(coin_id, bid, ask, bid_qty, ask_qty, 0);
                self.arena.increment_tick();
            }

            let mut closed_scalp = None;
            let mut closed_swing = None;
            let mut new_scalp = None;
            let mut new_swing = None;

            let coin = &self.arena.coins[coin_id];
            let swing_regime = self.feature_engines[coin_id].get_market_regime();

            static DBG_TRADE: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            if DBG_TRADE.fetch_add(1, Ordering::Relaxed) < 10 {
                // telemetry_server::telemetry_log!("DEBUG EVOL: process_event start! is_trade={}", is_trade);
            }

            // --- SCALP EVALUATION (Unified Closure Logic) ---
            if is_trade {
                let base_scalp_tp = self.arena.config.scalp_tp_base.load(Ordering::Relaxed);
                let base_scalp_sl = self.arena.config.scalp_sl_base.load(Ordering::Relaxed);
                let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                let min_atr_pct = self
                    .arena
                    .config
                    .dynamic_atr_min
                    .load(Ordering::Relaxed);
                let safe_min_atr = if min_atr_pct.is_finite() && min_atr_pct > 0.0 { min_atr_pct } else { 0.0001 };
                let atr_pct = if raw_atr_pct.is_finite() && raw_atr_pct > 0.0 { raw_atr_pct.max(safe_min_atr) } else { safe_min_atr };

                // Respetar parámetros del genoma con piso de volatilidad institucional (2.0x ATR, 0.40% floor)
                let scalp_sl = if base_scalp_sl.is_finite() && base_scalp_sl > 0.0 {
                    base_scalp_sl.max(0.0040).max(atr_pct * 2.0)
                } else {
                    0.0040
                };
                let scalp_tp = if base_scalp_tp.is_finite() && base_scalp_tp > 0.0 {
                    base_scalp_tp.max(scalp_sl * 1.5)
                } else {
                    0.0060
                };

                if coin.positions.scalp_position.is_open() {
                    let is_long = coin
                        .positions
                        .scalp_position
                        .is_long
                        .load(Ordering::Relaxed);
                    let entry = coin
                        .positions
                        .scalp_position
                        .entry_price
                        .load(Ordering::Relaxed);
                    let qty = coin
                        .positions
                        .scalp_position
                        .quantity
                        .load(Ordering::Relaxed);

                    let pnl_pct = if is_long {
                        (current_price - entry) / entry
                    } else {
                        (entry - current_price) / entry
                    };

                    let pos_dir = if is_long { 1.0 } else { -1.0 };
                    // FIX #565: Calcular ratio Hawkes / flujo de órdenes direccional
                    let vol_delta = self.feature_engines[coin_id].order_flow.get_volume_delta_ratio();
                    let ema_ofi = self.feature_engines[coin_id].ofi_model.ema_ofi.clamp(-1.0, 1.0);
                    let dir_flow = vol_delta * 0.70 + ema_ofi * 0.30;
                    let hawkes_ratio = 1.0 + (dir_flow * pos_dir * 1.5);
                    let tp_boost = strategy_core::momentum_booster::VolatileMomentumBooster::calculate_tp_extension(
                        pos_dir,
                        pnl_pct,
                        hawkes_ratio,
                        atr_pct,
                        &self.arena,
                    );
                    let dynamic_scalp_tp = (scalp_tp * tp_boost).min(scalp_tp * 1.5).max(0.0045);

                    let current_atr = atr_pct * entry;
                    let pseudo_atr = current_atr;

                    let side_int = if is_long { 1 } else { -1 };

                    let entry_time = coin
                        .positions
                        .scalp_position
                        .entry_time_ms
                        .load(Ordering::Relaxed);
                    let position_age_ms = if event_time_ms > 0 {
                        event_time_ms.saturating_sub(entry_time)
                    } else {
                        0
                    };

                    // Only evaluate trailing if position has been alive long enough and ATR is stabilized
                    let trail_active = position_age_ms > 5_000 && pnl_pct > 0.0005; // at least 5s and in profit > 0.05%

                    let mut trail_hit = false;
                    let mut force_close_trail = false;

                    if trail_active {
                        let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                        let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                            side_int,
                            entry,
                            current_price,
                            pseudo_atr,
                            coin.positions
                                .scalp_position
                                .trailing_phase
                                .load(Ordering::Relaxed) as i32,
                            coin.positions
                                .scalp_position
                                .mfe_atr
                                .load(Ordering::Relaxed),
                            coin.positions
                                .scalp_position
                                .max_pnl_pct
                                .load(Ordering::Relaxed),
                            coin.positions
                                .scalp_position
                                .trail_stop
                                .load(Ordering::Relaxed),
                            1.5,
                            1.5,
                            2.0,
                            2.5,
                            1.5, // t_params for scalp
                            live_fee,
                        );

                        // Don't let trailing stop be tighter than the base SL
                        let sl_floor = if is_long {
                            entry * (1.0 - scalp_sl)
                        } else {
                            entry * (1.0 + scalp_sl)
                        };
                        let mut safe_stop = if is_long {
                            trail_res.stop_price.max(sl_floor) // For long, stop should not be below SL floor (but can be above)
                        } else {
                            if trail_res.stop_price > 0.0 {
                                trail_res.stop_price.min(sl_floor)
                            } else {
                                sl_floor
                            }
                        };

                        // Break-Even Lock Adaptativo por Volatilidad:
                        // Activar Break-Even solo cuando el PnL supera 1.2x el ATR local (o 0.5x del TP)
                        let cur_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                        let be_threshold = (dynamic_scalp_tp * 0.5).max(cur_atr_pct * 1.2).clamp(0.0020, 0.0060);
                        if pnl_pct >= be_threshold {
                            let be_profit_buffer = 0.0006; // Cubre comisiones round-trip (0.04% maker/taker) + ganancia neta garantizada
                            if is_long {
                                safe_stop = safe_stop.max(entry * (1.0 + be_profit_buffer));
                            } else {
                                safe_stop = safe_stop.min(entry * (1.0 - be_profit_buffer));
                            }
                        }

                        // Only ratchet up (long) / down (short) — never widen the stop
                        let current_stored = coin
                            .positions
                            .scalp_position
                            .trail_stop
                            .load(Ordering::Relaxed);
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

                        coin.positions
                            .scalp_position
                            .trail_stop
                            .store(final_stop, Ordering::Relaxed);
                        coin.positions
                            .scalp_position
                            .trailing_phase
                            .store(trail_res.new_phase as u8, Ordering::Relaxed);
                        coin.positions
                            .scalp_position
                            .mfe_atr
                            .store(trail_res.mfe_atr, Ordering::Relaxed);
                        coin.positions
                            .scalp_position
                            .max_pnl_pct
                            .store(trail_res.max_pnl_pct, Ordering::Relaxed);

                        trail_hit = (is_long && current_price <= final_stop)
                            || (!is_long && current_price >= final_stop && final_stop > 0.0);
                        force_close_trail = trail_res.force_close;
                    }

                    let notional = qty * entry;
                    let mut unrealized = pnl_pct * notional;

                    let is_zombie = event_time_ms > 0 && position_age_ms > 1_800_000;
                    // FIX #1202: Aislamiento estricto Scalping ↔ Swing. Los scalps estancados se cierran bajo sus propias reglas de salida.
                    let zombie_close = is_zombie;

                    if pnl_pct >= dynamic_scalp_tp
                        || pnl_pct <= -scalp_sl
                        || trail_hit
                        || force_close_trail
                        || zombie_close
                    {
                        let sl_price = if is_long {
                            entry * (1.0 - scalp_sl)
                        } else {
                            entry * (1.0 + scalp_sl)
                        };
                        let tp_price = if is_long {
                            entry * (1.0 + dynamic_scalp_tp)
                        } else {
                            entry * (1.0 - dynamic_scalp_tp)
                        };

                        let mut exit_price = current_price;
                        if is_long {
                            if pnl_pct <= -scalp_sl {
                                exit_price = exit_price.max(sl_price);
                            }
                            if trail_hit {
                                let ts = coin
                                    .positions
                                    .scalp_position
                                    .trail_stop
                                    .load(Ordering::Relaxed);
                                if ts > 0.0 {
                                    exit_price = exit_price.max(ts);
                                }
                            }
                            if pnl_pct >= dynamic_scalp_tp {
                                exit_price = exit_price.min(tp_price);
                            }
                        } else {
                            if pnl_pct <= -scalp_sl {
                                exit_price = exit_price.min(sl_price);
                            }
                            if trail_hit {
                                let ts = coin
                                    .positions
                                    .scalp_position
                                    .trail_stop
                                    .load(Ordering::Relaxed);
                                if ts > 0.0 {
                                    exit_price = exit_price.min(ts);
                                }
                            }
                            if pnl_pct >= dynamic_scalp_tp {
                                exit_price = exit_price.max(tp_price);
                            }
                        }

                        unrealized = if is_long {
                            (exit_price - entry) * qty
                        } else {
                            (entry - exit_price) * qty
                        };
                        let gross_pnl = unrealized;
                        coin.scalp.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);

                        let _reason = if trail_hit {
                            "TRAIL_HIT"
                        } else if force_close_trail {
                            "FORCE_CLOSE"
                        } else if zombie_close {
                            "ZOMBIE"
                        } else if pnl_pct >= dynamic_scalp_tp {
                            "TP"
                        } else {
                            "SL"
                        };
                        let _print_pnl_pct = if is_long {
                            (exit_price - entry) / entry
                        } else {
                            (entry - exit_price) / entry
                        };

                        telemetry_server::telemetry_log!(
                            "🛑 CLOSE SCALP [Coin {}]: Long={}, Entry={:.4}, Exit={:.4}, PnL={:.4}% (${:.4}), pseudo_atr={:.4}, Reason={}",
                            coin_id,
                            is_long,
                            entry,
                            exit_price,
                            _print_pnl_pct * 100.0,
                            unrealized,
                            pseudo_atr,
                            _reason
                        );
                        let close_fee = if pnl_pct >= dynamic_scalp_tp {
                            self.arena
                                .config
                                .live_maker_fee
                                .load(Ordering::Relaxed)
                                .max(0.0002)
                        } else {
                            self.arena
                                .config
                                .live_taker_fee
                                .load(Ordering::Relaxed)
                                .max(0.0004)
                        };
                        unrealized -= notional * close_fee;

                        // FIX #343: Reconcile entry_fee with exit unrealized to compute true net realized PnL
                        let (_, _, _, margin_used, entry_fee_val) = coin.positions.scalp_position.close_with_fee();
                        let total_net_pnl = unrealized - entry_fee_val;

                        let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                        if current_used >= margin_used {
                            self.arena
                                .scalp_used_margin
                                .fetch_add(-margin_used, Ordering::Relaxed);
                        } else {
                            self.arena.scalp_used_margin.store(0.0, Ordering::Relaxed);
                        }
                        coin.scalp
                            .pnl_realized
                            .fetch_add(unrealized, Ordering::Relaxed);
                        self.arena
                            .unified_capital
                            .fetch_add(unrealized, Ordering::Relaxed);

                        // Kelly Feedback: Uses TRUE net realized PnL (accounting for both entry and exit fees)
                        let is_win = total_net_pnl > 0.0;
                        let n = coin.scalp.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                        let old_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                        let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                        coin.scalp.win_rate.store(new_wr, Ordering::Relaxed);
                        let old_pf = coin.scalp.profit_factor.load(Ordering::Relaxed).max(0.1);
                        let decay = (1.0 / n.min(50.0)).clamp(0.02, 0.20);
                        let new_pf = if is_win {
                            (old_pf * (1.0 - decay) + (pnl_pct.abs() / 0.0020) * decay).clamp(0.1, 10.0)
                        } else {
                            (old_pf * (1.0 - decay) + (0.0020 / pnl_pct.abs().max(1e-4)) * decay).clamp(0.1, 10.0)
                        };
                        coin.scalp.profit_factor.store(new_pf, Ordering::Relaxed);
                        let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                        let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                        let survival_ratio = self
                            .arena
                            .config
                            .kelly_survival_cap_ratio
                            .load(Ordering::Relaxed);
                        let exp_mult = self
                            .arena
                            .config
                            .kelly_expansion_mult
                            .load(Ordering::Relaxed);
                        let kelly_f = risk_engine::kelly::calculate_kelly_fraction(
                            new_wr,
                            coin.scalp.profit_factor.load(Ordering::Relaxed),
                            curr_cap,
                            base_cap,
                            survival_ratio,
                            exp_mult,
                        );
                        coin.scalp.kelly_fraction.store(kelly_f, Ordering::Relaxed);

                        // Epigenetic & Hebbian Learning Feedback (BUG-708)
                        let trade_dur = event_time_ms.saturating_sub(coin.positions.scalp_position.entry_time_ms.load(Ordering::Relaxed));
                        coin.apply_epigenetic_feedback(pnl_pct, trade_dur);
                        let mut current_hebbian = self.arena.registry.get("perceptron_hebbian_weight", "GodEngineCore").map(|p| p.get_value()).unwrap_or(1.0);
                        signal_engine::perceptron_gate::PerceptronGateEngine::update_weight(&mut current_hebbian, total_net_pnl, (pseudo_atr / current_price.max(1.0)).clamp(0.0001, 1.0));
                        self.arena.registry.set("perceptron_hebbian_weight", current_hebbian);
                        let cooldown_penalty = if total_net_pnl < 0.0 { 60_000 } else { 0 };
                        coin.last_scalp_close_ts.store(event_time_ms + cooldown_penalty, Ordering::Relaxed);

                        closed_scalp = Some((is_long, total_net_pnl, qty));
                    } else {
                        coin.scalp
                            .pnl_unrealized
                            .store(unrealized, Ordering::Relaxed);
                    }
                }
                static DBG_REGIME: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(0);
                if DBG_REGIME.fetch_add(1, Ordering::Relaxed) < 10 {
                    // telemetry_server::telemetry_log!("DEBUG EVOL: Checking intent... is_open={} latency={} scalp_regime={:?}", coin.positions.scalp_position.is_open(), latency_panic, scalp_regime);
                }

                // Actualización continua de CVD y flujo de órdenes
                let old_buy = coin.agg_buy_vol.load(Ordering::Relaxed);
                let old_sell = coin.agg_sell_vol.load(Ordering::Relaxed);
                let alpha_cvd = 0.01;
                coin.agg_buy_vol.store(old_buy * (1.0 - alpha_cvd) + bid_qty * alpha_cvd, Ordering::Relaxed);
                coin.agg_sell_vol.store(old_sell * (1.0 - alpha_cvd) + ask_qty * alpha_cvd, Ordering::Relaxed);

                let pseudo_maker = bid_qty > ask_qty;
                self.feature_engines[coin_id].update_trade_flow(bid_qty + ask_qty, pseudo_maker);
                let _ = self.feature_engines[coin_id].update_ofi(current_price - 0.05, current_price + 0.05, bid_qty, ask_qty);

                // FASE 17: Removed hardcoded regime gate. Scalp can operate in ANY regime.
                // The RiskEngine and Kelly already modulate exposure based on regime risk.
                if !coin.positions.scalp_position.is_open() && !latency_panic {
                    let features = self.feature_engines[coin_id].get_features();
                    let _coin_symbol = quantum_arena::symbol_registry::try_spec(coin_id)
                        .map(|s| s.symbol)
                        .unwrap_or_else(|| "BTCUSDT".to_string());
                    let combined = self.build_54d_tensor(coin_id, bid_qty, ask_qty, current_price, omni_features);
                    let mut ml_prob = self.swing_nn.as_mut().and_then(|nn| {
                        nn.predict(&combined)
                    }).unwrap_or(0.5);
                    ml_prob = ml_prob.clamp(0.0001, 0.9999); // Clamping asintótico preserva alta convicción
                    self.last_ml_prob = ml_prob as f32;

                    coin.ml_prob.store(ml_prob, Ordering::Relaxed);
                    coin.hurst_exponent
                        .store(features[1] as f64, Ordering::Relaxed); // [1] = Hurst

                    if let Some(lh) = &self.lakehouse {
                        lh.record_tensor(
                            quantum_arena::symbol_registry::try_spec(coin_id).map(|s| s.symbol).unwrap_or_default(),
                            event_time_ms,
                            features.to_vec(),
                            ml_prob as f32,
                            0.0,
                        );
                    }

                    let ml_threshold = self
                        .arena
                        .config
                        .scalp_obi_threshold
                        .load(Ordering::Relaxed);

                    let mut scalp_intent = SignalIntent::flat();
                    let atr_pct = self.feature_engines[coin_id].get_atr_pct();

                    static DBG_EVAL: std::sync::atomic::AtomicUsize =
                        std::sync::atomic::AtomicUsize::new(0);
                    let eval_count = DBG_EVAL.fetch_add(1, Ordering::Relaxed);
                    if eval_count % 100 == 0 {
                        // telemetry_server::telemetry_log!("DEBUG EVOL: Inside scalp eval! atr_pct={:.6} hurst={:.3} scalp_regime={:?} ml_prob={:.3}",
                        //         atr_pct, hurst, scalp_regime, ml_prob);
                    }

                    // 1. PUBLICACIÓN DE MICRO-MÉTRICAS AL OMNISCIENT REGISTRY
                    let hurst_val = coin.hurst_exponent.load(Ordering::Relaxed);
                    let obi_val = if bid_qty + ask_qty > 0.0 { (bid_qty - ask_qty) / (bid_qty + ask_qty) } else { 0.0 };
                    let flow_dir = if bid_qty > ask_qty { 1.0 } else if ask_qty > bid_qty { -1.0 } else { 0.0 };
                    let total_vol = (bid_qty + ask_qty).max(1e-8);
                    let p_bid = (bid_qty / total_vol).clamp(0.0001, 0.9999);
                    let p_ask = (ask_qty / total_vol).clamp(0.0001, 0.9999);
                    let tsallis_ent = ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5).clamp(0.0, 1.0);
                    let shannon_ent = self.feature_engines[coin_id].entropy.current();

                    self.arena.registry.set("atr_1s", (atr_pct * current_price).max(0.0001));
                    self.arena.registry.set("atr_5s", (atr_pct * current_price * 2.236).max(0.0005));
                    self.arena.registry.set("atr_1m", (atr_pct * current_price * 7.746).max(0.0020));
                    self.arena.registry.set("order_flow_direction", flow_dir);
                    self.arena.registry.set("order_book_imbalance", obi_val);
                    self.arena.registry.set("orderbook_imbalance", obi_val);
                    self.arena.registry.set("order_flow_imbalance", obi_val);
                    self.arena.registry.set("shannon_entropy", shannon_ent);
                    self.arena.registry.set("tsallis_q_entropy", tsallis_ent);
                    self.arena.registry.set("hurst_exponent", hurst_val);
                    self.arena.registry.set("global_hurst", hurst_val);
                    self.arena.registry.set("atr_pct", atr_pct);
                    self.arena.registry.set("relative_atr_pct", atr_pct);
                    self.arena.registry.set("hawkes_intensity", (1.0 + obi_val.abs() * 2.0).clamp(0.1, 5.0));

                    // 2. CONECTAR INTELIGENCIA EPIGENÉTICA HEBBIANA
                    let hebbian_mult = self.arena.registry.get("perceptron_hebbian_weight", "GodEngineCore")
                        .map(|p| p.get_value())
                        .unwrap_or(1.0)
                        .clamp(0.5, 2.0);

                    // 3. INTELIGENCIA TENSORIAL (8 Estrategias de Arbitraje y Flujo)
                    let tensor_scalp = self.tensor_orchestrator.evaluate_scalp_consensus();
                    let tensor_score: f64 = match tensor_scalp.signal {
                        SignalType::Long => tensor_scalp.net_confidence,
                        SignalType::Short => -tensor_scalp.net_confidence,
                        SignalType::Flat => 0.0,
                    };

                    // 4. INTELIGENCIA NEURONAL PROFUNDA DARKALPHA 54D
                    let nn_score: f64 = (ml_prob - 0.5) * 2.0;

                    // 5. INTELIGENCIA DE MICROESTRUCTURA L2 (OBI + OFI + CVD + EMA)
                    let current_obi = obi_val;
                    let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
                    let dynamic_obi_thr = self.arena.config.dynamic_obi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);
                    let dynamic_ema_thr = self.arena.config.dynamic_ema_trend.load(Ordering::Relaxed);
                    let dynamic_ofi_thr = self.arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);

                    let ema_trend = if self.feature_engines[coin_id].ema_slow > 0.0 {
                        (self.feature_engines[coin_id].ema_fast - self.feature_engines[coin_id].ema_slow) / self.feature_engines[coin_id].ema_slow
                    } else {
                        0.0
                    };
                    let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
                    let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
                    let total_vol_cvd = buy_vol + sell_vol;
                    let rolling_cvd = if total_vol_cvd > 0.0 { (buy_vol - sell_vol) / total_vol_cvd } else { 0.0 };
                    let ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;

                    let obi_norm = (current_obi / dynamic_obi_thr).clamp(-1.5, 1.5);
                    let ofi_norm = (ofi / dynamic_ofi_thr).clamp(-1.5, 1.5);
                    let micro_score: f64 = (obi_norm * 0.40 + ofi_norm * 0.40 + rolling_cvd * 0.20).clamp(-1.0, 1.0);

                    // 6. SÍNTESIS CUÁNTICA INTEGRAL MULTI-INTELIGENCIA (NEXUS BAYESIANO)
                    // Fusión ponderada: 40% Neural 54D + 35% Microestructura L2 + 25% Tensor Consensus
                    let raw_composite = nn_score * 0.40 + micro_score * 0.35 + tensor_score * 0.25;
                    let composite_score = (raw_composite * hebbian_mult).clamp(-1.0, 1.0);

                    if atr_pct > dynamic_atr_min {
                        let is_mean_reverting = hurst_val < 0.45;
                        let is_trending = hurst_val >= 0.50;

                        let ema_slow = self.feature_engines[coin_id].ema_slow;
                        let cur_atr = self.feature_engines[coin_id].v_t.max(current_price * 0.001);
                        let price_stretch = if ema_slow > 0.0 { (current_price - ema_slow) / cur_atr } else { 0.0 };
                        let not_overextended_long = price_stretch <= 1.2;
                        let not_overextended_short = price_stretch >= -1.2;

                        if is_trending {
                            if composite_score > 0.45 && ema_trend > dynamic_ema_thr && current_price > ema_slow && not_overextended_long {
                                scalp_intent = SignalIntent {
                                    signal: SignalType::Long,
                                    confidence: (0.75 + composite_score * 0.25).min(1.0),
                                    ..Default::default()
                                };
                            } else if composite_score < -0.45 && ema_trend < -dynamic_ema_thr && current_price < ema_slow && not_overextended_short {
                                scalp_intent = SignalIntent {
                                    signal: SignalType::Short,
                                    confidence: (0.75 + composite_score.abs() * 0.25).min(1.0),
                                    ..Default::default()
                                };
                            }
                        } else if is_mean_reverting {
                            // Fade overextensions in oscillating market
                            if composite_score > 0.45 && current_obi > dynamic_obi_thr && price_stretch > 1.0 {
                                scalp_intent = SignalIntent {
                                    signal: SignalType::Short,
                                    confidence: (0.75 + composite_score * 0.25).min(1.0),
                                    ..Default::default()
                                };
                            } else if composite_score < -0.45 && current_obi < -dynamic_obi_thr && price_stretch < -1.0 {
                                scalp_intent = SignalIntent {
                                    signal: SignalType::Long,
                                    confidence: (0.75 + composite_score.abs() * 0.25).min(1.0),
                                    ..Default::default()
                                };
                            }
                        }
                    }

                    // Cooldown: 15s post-close to prevent churn & whipsaw re-entries
                    let last_close = coin.last_scalp_close_ts.load(Ordering::Relaxed);
                    if event_time_ms.saturating_sub(last_close) < 15_000 {
                        scalp_intent = SignalIntent::flat();
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
                        // FASE 17: Vetos CVD/Wall ahora usan thresholds del genoma
                        let cvd_veto = self.arena.config.cvd_veto_threshold.load(Ordering::Relaxed);
                        let wall_veto = self
                            .arena
                            .config
                            .wall_veto_threshold
                            .load(Ordering::Relaxed);

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
                        payload[0..4].copy_from_slice(&(ml_prob as f32).to_le_bytes());
                        fr.record(telemetry_server::FlightEvent {
                            timestamp: std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_nanos() as u64,
                            trace_id: self.arena.tick_counter.load(Ordering::Relaxed),
                            event_type: 1,
                            payload,
                        });
                    }

                    if scalp_intent.signal != SignalType::Flat {
                        let hurst = coin.hurst_exponent.load(Ordering::Relaxed);
                        let slippage_bps = self.feature_engines[coin_id].get_atr_pct() * 10000.0 * 0.1;
                        let base_c = self.arena.config.base_capital.load(Ordering::Relaxed).max(1.0);
                        let cur_c = self.arena.unified_capital.load(Ordering::Relaxed);
                        let dd = (base_c - cur_c).max(0.0) / base_c;
                        let snapshot = metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                            book_imbalance: if bid_qty + ask_qty > 0.0 { (bid_qty - ask_qty) / (bid_qty + ask_qty) } else { 0.0 },
                            hurst_exponent: hurst.clamp(0.0, 1.0),
                            graph_correlation: 0.85,
                            do_calculus_risk: 0.10,
                            current_drawdown_pct: dd.clamp(0.0, 1.0),
                            estimated_slippage_bps: slippage_bps.clamp(0.0, 100.0),
                        };
                        let wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                        let deliberation = self.consejo_deliberacion.deliberar(&snapshot, wr);
                        if !deliberation.approved || deliberation.vetoed_by.is_some() {
                            scalp_intent = SignalIntent::flat();
                        }
                    }

                    self.last_scalp_intent[coin_id] = scalp_intent;
                    let swing_intent = self.last_swing_intent[coin_id];

                    let (scalp_order, _swing_order) = self.risk_engine.evaluate_order(
                        coin_id,
                        scalp_intent,
                        swing_intent,
                        &self.arena,
                    );

                    if self.arena.tick_counter.load(Ordering::Relaxed) % 100_000 == 0 {
                        println!("🔍 [RISK TRACE] tick={} scalp_intent={:?} conf={:.4} scalp_order={:?} vol={:.4} cap={:.4}", 
                            self.arena.tick_counter.load(Ordering::Relaxed), scalp_intent.signal, scalp_intent.confidence, scalp_order.signal, scalp_order.volume_usd, self.arena.unified_capital.load(Ordering::Relaxed));
                    }

                    if scalp_order.signal != SignalType::Flat {
                        let is_long = scalp_order.signal == SignalType::Long;
                        let _cap_split = self
                            .arena
                            .config
                            .capital_split_scalp
                            .load(Ordering::Relaxed);
                        let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                        let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);

                        let mut active_opportunities: f64 = 0.0;
                        for i in 0..30 {
                            let p = self.arena.coins[i].ml_prob.load(Ordering::Relaxed);
                            if p > ml_threshold || (p > 0.01 && p < (1.0 - ml_threshold)) {
                                active_opportunities += 1.0;
                            }
                        }
                        let active_opportunities = active_opportunities.max(1.0);

                        // FASE B: Crecimiento Compuesto Cuántico Micro-Cuenta ($13 USD bootstrap)
                        let min_margin_for_binance = 5.05 / leverage.max(1.0);
                        let kelly_alloc = if current_cap <= 30.0 {
                            (current_cap * 0.40).clamp(min_margin_for_binance, current_cap * 0.70)
                        } else {
                            (scalp_order.volume_usd / active_opportunities).clamp(min_margin_for_binance, current_cap * 0.50)
                        };
                        let mut margin_required = kelly_alloc;

                        let max_position_size = 10000.0;
                        if margin_required * leverage > max_position_size {
                            margin_required = max_position_size / leverage;
                        }
                        let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);

                        if current_cap > 0.0 && (margin_required * leverage >= 5.0) && current_used + margin_required <= current_cap * 0.95
                        {
                            self.arena
                                .scalp_used_margin
                                .fetch_add(margin_required, Ordering::Relaxed);

                            // FRICCION REAL: Entramos cruzando el spread y deslizando el orderbook
                            let nominal_size = margin_required * leverage;
                            let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                            let base_price = if is_long { ask } else { bid };
                            let (real_entry_price, _entry_fee) =
                                self.reality.calculate_market_entry(
                                    base_price,
                                    is_long,
                                    nominal_size,
                                    tick_volatility,
                                );
                            let qty = nominal_size / real_entry_price;

                            static DBG_COUNT_2: std::sync::atomic::AtomicUsize =
                                std::sync::atomic::AtomicUsize::new(0);
                            if DBG_COUNT_2.fetch_add(1, Ordering::Relaxed) < 100 {
                                // telemetry_server::telemetry_log!("DEBUG EVOL: Opening Scalp! is_long={} qty={} margin={}", is_long, qty, margin_required);
                            }

                            // FRICCION REAL: Aplicamos fee Taker real al abrir (market order)
                            let entry_fee_rate = self
                                .arena
                                .config
                                .live_taker_fee
                                .load(Ordering::Relaxed)
                                .max(0.0002); // Floor: 0.02% maker minimum
                            let entry_fee = margin_required * leverage * entry_fee_rate;
                            self.arena
                                .unified_capital
                                .fetch_add(-entry_fee, Ordering::Relaxed);
                            coin.scalp
                                .pnl_realized
                                .fetch_add(-entry_fee, Ordering::Relaxed);

                            coin.positions.scalp_position.open_with_fee(
                                is_long,
                                real_entry_price,
                                qty,
                                margin_required,
                                event_time_ms,
                                0.0,
                                0.0,
                                quantum_arena::position::PositionHorizon::Scalping,
                                ml_prob,
                                scalp_intent.confidence,
                                entry_fee,
                            );

                            let trail_offset = real_entry_price * scalp_sl;
                            let initial_trail = if is_long {
                                real_entry_price - trail_offset
                            } else {
                                real_entry_price + trail_offset
                            };
                            coin.positions
                                .scalp_position
                                .trail_stop
                                .store(initial_trail, Ordering::Relaxed);

                            new_scalp = Some((is_long, real_entry_price, qty));
                        }
                    }
                }
            }

            // --- SWING EVALUATION ---
            if is_kline_closed {
                if coin.positions.swing_position.is_open() {
                    let is_long = coin
                        .positions
                        .swing_position
                        .is_long
                        .load(Ordering::Relaxed);
                    let entry = coin
                        .positions
                        .swing_position
                        .entry_price
                        .load(Ordering::Relaxed);
                    let qty = coin
                        .positions
                        .swing_position
                        .quantity
                        .load(Ordering::Relaxed);

                    // FRICCION REAL:
                    let notional = qty * entry;
                    let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                    let base_price = if is_long { bid } else { ask };
                    let (exit_price, _fee) = self.reality.calculate_exit(
                        base_price,
                        is_long,
                        notional,
                        false,
                        tick_volatility,
                    );
                    let pnl_pct = if is_long {
                        (exit_price - entry) / entry
                    } else {
                        (entry - exit_price) / entry
                    };

                    let current_atr = self.feature_engines[coin_id].get_atr_pct() * entry;
                    let pseudo_atr = if current_atr > 0.0 {
                        current_atr
                    } else {
                        current_price * 0.01
                    };
                    let side_int = if is_long { 1 } else { -1 };

                    let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                    let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                        side_int,
                        entry,
                        current_price,
                        pseudo_atr,
                        coin.positions
                            .swing_position
                            .trailing_phase
                            .load(Ordering::Relaxed) as i32,
                        coin.positions
                            .swing_position
                            .mfe_atr
                            .load(Ordering::Relaxed),
                        coin.positions
                            .swing_position
                            .max_pnl_pct
                            .load(Ordering::Relaxed),
                        coin.positions
                            .swing_position
                            .trail_stop
                            .load(Ordering::Relaxed),
                        2.5, // FIX #591: pullback_tol amplio (2.5 ATR) para permitir respiración en Swing
                        2.0,
                        3.0,
                        4.0,
                        3.5, // trail_runner para macro tendencia
                        live_fee,
                    );

                    let current_stored = coin
                        .positions
                        .swing_position
                        .trail_stop
                        .load(Ordering::Relaxed);
                    let safe_stop = if is_long {
                        if current_stored > 0.0 {
                            trail_res.stop_price.max(current_stored)
                        } else {
                            trail_res.stop_price
                        }
                    } else {
                        if current_stored > 0.0 {
                            trail_res.stop_price.min(current_stored)
                        } else {
                            trail_res.stop_price
                        }
                    };

                    coin.positions
                        .swing_position
                        .trail_stop
                        .store(safe_stop, Ordering::Relaxed);
                    coin.positions
                        .swing_position
                        .max_pnl_pct
                        .store(trail_res.max_pnl_pct, Ordering::Relaxed);

                    let trail_hit = (is_long && current_price <= safe_stop && safe_stop > 0.0)
                        || (!is_long
                            && current_price >= safe_stop
                            && safe_stop > 0.0);

                    let notional = qty * entry;
                    let mut unrealized = pnl_pct * notional;
                    let regime_exit = swing_regime == MarketRegime::Scalping && pnl_pct > 0.0;

                    // Swing Cooldown/Hold Time Mechanism: Prevent micro-whipsaws
                    // Adaptable hold time before trailing or regime exits can trigger
                    let entry_time = coin
                        .positions
                        .swing_position
                        .entry_time_ms
                        .load(Ordering::Relaxed);
                    let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                    let min_atr_pct = self
                        .arena
                        .config
                        .dynamic_atr_min
                        .load(Ordering::Relaxed)
                        .max(0.001);
                    let atr_pct = raw_atr_pct.max(min_atr_pct);

                    let min_hold_ms = (120_000.0 / (atr_pct / 0.005).clamp(0.5, 3.0)).clamp(60_000.0, 600_000.0) as u64;
                    let hold_time_met =
                        (event_time_ms >= entry_time) && ((event_time_ms - entry_time) >= min_hold_ms);

                    let base_swing_tp = self.arena.config.swing_tp_base.load(Ordering::Relaxed);
                    let base_swing_sl = self.arena.config.swing_sl_base.load(Ordering::Relaxed);

                    let swing_tp = base_swing_tp.max(atr_pct * 3.0).max(0.005);
                    let swing_sl = base_swing_sl.max(atr_pct * 2.0).max(0.003);

                    let hurst = coin.hurst_exponent.load(Ordering::Relaxed);
                    let vpin = {
                        let cvpin = &self.feature_engines[coin_id].cvpin;
                        let tot = cvpin.buy_volume + cvpin.sell_volume;
                        if tot > 0.0 { (cvpin.buy_volume - cvpin.sell_volume).abs() / tot } else { 0.0 }
                    };
                    let max_swing_tp = self.arena.config.explosive_leverage_multiplier.load(Ordering::Relaxed).max(1.0) * 0.08;
                    let dynamic_swing_tp = signal_engine::trend_runner::HighPayoffTrendRunner::calculate_expanded_tp(
                        swing_tp,
                        hurst,
                        vpin,
                        atr_pct,
                        max_swing_tp,
                    );

                    if pnl_pct >= dynamic_swing_tp
                        || pnl_pct <= -swing_sl
                        || (hold_time_met && (trail_hit || trail_res.force_close || regime_exit))
                    {
                        let gross_pnl = unrealized;
                        coin.swing.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                        let live_maker = self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002);
                        let live_taker = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004);
                        let close_fee = if pnl_pct >= dynamic_swing_tp { live_maker } else { live_taker };
                        unrealized -= notional * close_fee;

                        // FIX #343: Reconcile entry_fee with exit unrealized to compute true net realized PnL
                        let (_, _, _, margin_used, entry_fee_val) = coin.positions.swing_position.close_with_fee();
                        let total_net_pnl = unrealized - entry_fee_val;

                        let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                        if current_used >= margin_used {
                            self.arena
                                .swing_used_margin
                                .fetch_add(-margin_used, Ordering::Relaxed);
                        } else {
                            self.arena.swing_used_margin.store(0.0, Ordering::Relaxed);
                        }

                        coin.swing
                            .pnl_realized
                            .fetch_add(unrealized, Ordering::Relaxed);
                        self.arena
                            .unified_capital
                            .fetch_add(unrealized, Ordering::Relaxed);

                        // Kelly Feedback: Uses TRUE net realized PnL (accounting for both entry and exit fees)
                        let is_win = total_net_pnl > 0.0;
                        let n = coin.swing.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                        let old_wr = coin.swing.win_rate.load(Ordering::Relaxed);
                        let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                        coin.swing.win_rate.store(new_wr, Ordering::Relaxed);
                        let old_pf = coin.swing.profit_factor.load(Ordering::Relaxed).max(0.1);
                        let decay = (1.0 / n.min(50.0)).clamp(0.02, 0.20);
                        let new_pf = if is_win {
                            (old_pf * (1.0 - decay) + (pnl_pct.abs() / 0.0050) * decay).clamp(0.1, 10.0)
                        } else {
                            (old_pf * (1.0 - decay) + (0.0050 / pnl_pct.abs().max(1e-4)) * decay).clamp(0.1, 10.0)
                        };
                        coin.swing.profit_factor.store(new_pf, Ordering::Relaxed);
                        let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                        let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                        let survival_ratio = self
                            .arena
                            .config
                            .kelly_survival_cap_ratio
                            .load(Ordering::Relaxed);
                        let exp_mult = self
                            .arena
                            .config
                            .kelly_expansion_mult
                            .load(Ordering::Relaxed);
                        let kelly_f = risk_engine::kelly::calculate_kelly_fraction(
                            new_wr,
                            coin.swing.profit_factor.load(Ordering::Relaxed),
                            curr_cap,
                            base_cap,
                            survival_ratio,
                            exp_mult,
                        );
                        coin.swing.kelly_fraction.store(kelly_f, Ordering::Relaxed);

                        // Epigenetic & Hebbian Learning Feedback (BUG-708)
                        let trade_dur = event_time_ms.saturating_sub(coin.positions.swing_position.entry_time_ms.load(Ordering::Relaxed));
                        coin.apply_epigenetic_feedback(pnl_pct, trade_dur);
                        let mut current_hebbian = self.arena.registry.get("perceptron_hebbian_weight", "GodEngineCore").map(|p| p.get_value()).unwrap_or(1.0);
                        signal_engine::perceptron_gate::PerceptronGateEngine::update_weight(&mut current_hebbian, total_net_pnl, (pseudo_atr / current_price.max(1.0)).clamp(0.0001, 1.0));
                        self.arena.registry.set("perceptron_hebbian_weight", current_hebbian);

                        closed_swing = Some((is_long, total_net_pnl, qty));
                    } else {
                        coin.swing
                            .pnl_unrealized
                            .store(unrealized, Ordering::Relaxed);
                    }
                }

                // FASE 17: Removed hardcoded regime/ATR/Hurst/NN gates.
                // All thresholds are now genome-driven and evolvable.
                if !coin.positions.swing_position.is_open() {
                    let features = self.feature_engines[coin_id].get_features();
                    let hurst_exponent = features[1] as f64; // [1] = Hurst
                    let combined = self.build_54d_tensor(coin_id, bid_qty, ask_qty, current_price, omni_features);

                    let nn_prob = if let Some(nn) = &mut self.swing_nn {
                        nn.predict(&combined).unwrap_or(0.5)
                    } else {
                        0.5
                    };

                    if let Some(lh) = &self.lakehouse {
                        let mut feat_vec = Vec::with_capacity(54);
                        for f in omni_features {
                            feat_vec.push(*f as f32);
                        }
                        lh.record_tensor(
                            quantum_arena::symbol_registry::try_spec(coin_id).map(|s| s.symbol).unwrap_or_default(),
                            event_time_ms,
                            feat_vec,
                            nn_prob as f32,
                            0.0,
                        );
                    }

                    let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
                    let mut swing_intent = self.swing_engines[coin_id].evaluate_trend(
                        current_price,
                        hurst_exponent,
                        trend_threshold,
                        nn_prob,
                        &self.arena,
                    );

                    // Publish live swing metrics to OmniscientRegistry for quantum strategies
                    let swing_atr = self.feature_engines[coin_id].get_atr_pct();
                    let trend_dir = if swing_intent.signal == SignalType::Long { 1.0 } else if swing_intent.signal == SignalType::Short { -1.0 } else if nn_prob >= 0.55 { 1.0 } else if nn_prob <= 0.45 { -1.0 } else { 0.0 };
                    let total_vol = (bid_qty + ask_qty).max(1e-8);
                    let p_bid = (bid_qty / total_vol).clamp(0.0001, 0.9999);
                    let p_ask = (ask_qty / total_vol).clamp(0.0001, 0.9999);
                    let tsallis_ent = ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5).clamp(0.0, 1.0);
                    let shannon_ent = self.feature_engines[coin_id].entropy.current();

                    self.arena.registry.set("atr_1s", (swing_atr * current_price).max(0.0001));
                    self.arena.registry.set("atr_5s", (swing_atr * current_price * 2.236).max(0.0005));
                    self.arena.registry.set("atr_1m", (swing_atr * current_price * 7.746).max(0.0020));
                    self.arena.registry.set("order_flow_direction", if nn_prob >= 0.50 { 1.0 } else { -1.0 });
                    self.arena.registry.set("trend_direction", trend_dir);
                    self.arena.registry.set("shannon_entropy", shannon_ent);
                    self.arena.registry.set("tsallis_q_entropy", tsallis_ent);
                    self.arena.registry.set("hurst_exponent", hurst_exponent);
                    self.arena.registry.set("global_hurst", hurst_exponent);
                    self.arena.registry.set("atr_pct", swing_atr);
                    self.arena.registry.set("relative_atr_pct", swing_atr);

                    // Tensor Consensus Evaluation for Swing Horizon
                    let tensor_swing = self.tensor_orchestrator.evaluate_swing_consensus();
                    if swing_intent.signal == SignalType::Flat {
                        if tensor_swing.signal != SignalType::Flat && tensor_swing.net_confidence > 0.60 {
                            swing_intent = SignalIntent {
                                signal: tensor_swing.signal,
                                confidence: tensor_swing.net_confidence,
                                ..Default::default()
                            };
                        }
                    } else if tensor_swing.signal == swing_intent.signal {
                        swing_intent.confidence = (swing_intent.confidence * 0.7 + tensor_swing.net_confidence * 0.3).min(1.0);
                    } else if tensor_swing.signal != SignalType::Flat {
                        // FIX #785: Penalización por desacuerdo bayesiano/adversarial para Swing
                        swing_intent.confidence = (swing_intent.confidence - tensor_swing.net_confidence * 0.5).max(0.0);
                        if swing_intent.confidence < 0.45 {
                            swing_intent = SignalIntent::flat();
                        }
                    }

                    if swing_intent.signal != SignalType::Flat {
                        let slippage_bps = self.feature_engines[coin_id].get_atr_pct() * 10000.0 * 0.1;
                        let base_c = self.arena.config.base_capital.load(Ordering::Relaxed).max(1.0);
                        let cur_c = self.arena.unified_capital.load(Ordering::Relaxed);
                        let dd = (base_c - cur_c).max(0.0) / base_c;
                        let snapshot = metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                            book_imbalance: if bid_qty + ask_qty > 0.0 { (bid_qty - ask_qty) / (bid_qty + ask_qty) } else { 0.0 },
                            hurst_exponent: hurst_exponent.clamp(0.0, 1.0),
                            graph_correlation: 0.85,
                            do_calculus_risk: 0.10,
                            current_drawdown_pct: dd.clamp(0.0, 1.0),
                            estimated_slippage_bps: slippage_bps.clamp(0.0, 100.0),
                        };
                        let wr = coin.swing.win_rate.load(Ordering::Relaxed);
                        let deliberation = self.consejo_deliberacion.deliberar(&snapshot, wr);
                        if !deliberation.approved || deliberation.vetoed_by.is_some() {
                            swing_intent = SignalIntent::flat();
                        }
                    }

                    self.last_swing_intent[coin_id] = swing_intent;
                    let scalp_intent = self.last_scalp_intent[coin_id];

                    let (_scalp_order, swing_order) = self.risk_engine.evaluate_order(
                        coin_id,
                        scalp_intent,
                        swing_intent,
                        &self.arena,
                    );

                    if swing_order.signal != SignalType::Flat {
                        let is_long = swing_order.signal == SignalType::Long;
                        let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                        let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);

                        let trend_threshold =
                            self.arena.config.trend_threshold.load(Ordering::Relaxed);

                        // --- GLOBAL TENSOR SWING ---
                        let mut active_swing_opportunities: f64 = 0.0;
                        for i in 0..30 {
                            let h = self.arena.coins[i].hurst_exponent.load(Ordering::Relaxed);
                            if h > trend_threshold {
                                active_swing_opportunities += 1.0;
                            }
                        }
                        let active_swing_opportunities = active_swing_opportunities.max(1.0);

                        // FASE B: Sinergia Real. Usamos el margen dictado por el RiskEngine.
                        let mut margin_required =
                            swing_order.volume_usd / active_swing_opportunities;

                        // Binance Futures $5.00 MIN_NOTIONAL: Elevate margin to satisfy minimum exchange notional
                        let min_margin_for_binance = 5.05 / leverage.max(1.0);
                        if margin_required < min_margin_for_binance {
                            margin_required = min_margin_for_binance;
                        }

                        let max_position_size = 10000.0;
                        if margin_required * leverage > max_position_size {
                            margin_required = max_position_size / leverage;
                        }
                        let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);

                        if current_cap > 0.0 && (margin_required * leverage >= 5.0) && current_used + margin_required <= current_cap * 0.95
                        {
                            self.arena
                                .swing_used_margin
                                .fetch_add(margin_required, Ordering::Relaxed);

                            // FRICCION REAL: Entramos cruzando el spread
                            let nominal_size = margin_required * leverage;
                            let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                            let base_price = if is_long { ask } else { bid };
                            let (real_entry_price, _entry_fee) =
                                self.reality.calculate_market_entry(
                                    base_price,
                                    is_long,
                                    nominal_size,
                                    tick_volatility,
                                );
                            let qty = nominal_size / real_entry_price;

                            // FRICCION REAL: Aplicamos fee Taker real al abrir (market order)
                            let entry_fee_rate = self
                                .arena
                                .config
                                .live_taker_fee
                                .load(Ordering::Relaxed)
                                .max(0.0002);
                            let entry_fee = margin_required * leverage * entry_fee_rate;
                            self.arena
                                .unified_capital
                                .fetch_add(-entry_fee, Ordering::Relaxed);
                            coin.swing
                                .pnl_realized
                                .fetch_add(-entry_fee, Ordering::Relaxed);

                            coin.positions.swing_position.open_with_fee(
                                is_long,
                                real_entry_price,
                                qty,
                                margin_required,
                                event_time_ms,
                                0.0,
                                0.0,
                                quantum_arena::position::PositionHorizon::Swing,
                                nn_prob,
                                swing_intent.confidence,
                                entry_fee,
                            );

                            let base_swing_sl = self.arena.config.swing_sl_base.load(Ordering::Relaxed);
                            let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                            let min_atr_pct = self
                                .arena
                                .config
                                .dynamic_atr_min
                                .load(Ordering::Relaxed);
                            let safe_min_atr = if min_atr_pct.is_finite() && min_atr_pct > 0.0 { min_atr_pct } else { 0.0001 };
                            let atr_pct = if raw_atr_pct.is_finite() && raw_atr_pct > 0.0 { raw_atr_pct.max(safe_min_atr) } else { safe_min_atr };
                            let swing_sl = if base_swing_sl.is_finite() && base_swing_sl > 0.0 {
                                base_swing_sl.max(atr_pct * 1.5)
                            } else {
                                0.010
                            };

                            let trail_offset = real_entry_price * swing_sl;
                            let initial_trail = if is_long {
                                real_entry_price - trail_offset
                            } else {
                                real_entry_price + trail_offset
                            };
                            coin.positions
                                .swing_position
                                .trail_stop
                                .store(initial_trail, Ordering::Relaxed);

                            new_swing = Some((is_long, real_entry_price, qty));
                        }
                    }
                }
            }

            (new_scalp, new_swing, closed_scalp, closed_swing)
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
    }

    /// Procesa un tick y devuelve las órdenes generadas (si hay).
    /// Retorna: (NuevoScalp, NuevoSwing, CerradoScalp, CerradoSwing, MakerQuote)
    /// Donde cada Option es (is_long, entry/exit_price, qty)
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
        Option<(bool, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<MakerQuote>,
    ) {
        telemetry_server::profile_node!("GodEngineCore::process_tick", {
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

            // --- 1. GESTION DE POSICIONES (TP/SL/TRAILING) ---
            let base_scalp_tp = self.arena.config.scalp_tp_base.load(Ordering::Relaxed);
            let base_scalp_sl = self.arena.config.scalp_sl_base.load(Ordering::Relaxed);
            let swing_tp = self.arena.config.swing_tp_base.load(Ordering::Relaxed);
            let swing_sl = self.arena.config.swing_sl_base.load(Ordering::Relaxed);
            let _sim_fee_rate = self.arena.config.max_fee_pct.load(Ordering::Relaxed);

            let coin = &self.arena.coins[coin_id];
            let atr_pct = self.feature_engines[coin_id].get_atr_pct();

            // CONSEJO DE SAGES: Dynamic SL/TP with asymmetric Risk:Reward ratio (TP >= 1.8x SL) to guarantee positive edge over exchange fees
            let scalp_sl = base_scalp_sl.max(atr_pct * 1.2).max(0.0015);
            let scalp_tp = base_scalp_tp.max(scalp_sl * 1.8).max(0.0035);

            if coin.positions.scalp_position.is_open() {
                let is_long = coin
                    .positions
                    .scalp_position
                    .is_long
                    .load(Ordering::Relaxed);
                let entry = coin
                    .positions
                    .scalp_position
                    .entry_price
                    .load(Ordering::Relaxed);
                let qty = coin
                    .positions
                    .scalp_position
                    .quantity
                    .load(Ordering::Relaxed);

                let pnl_pct = if is_long {
                    (mid_price - entry) / entry
                } else {
                    (entry - mid_price) / entry
                };

                let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                let min_atr_pct = self
                    .arena
                    .config
                    .dynamic_atr_min
                    .load(Ordering::Relaxed)
                    .max(0.001);
                let atr_pct_live = raw_atr_pct.max(min_atr_pct);
                let current_atr = atr_pct_live * entry;
                let pseudo_atr = current_atr;

                let side_int = if is_long { 1 } else { -1 };

                let entry_time = coin
                    .positions
                    .scalp_position
                    .entry_time_ms
                    .load(Ordering::Relaxed);
                let position_age_ms = if event_time_ms > 0 {
                    event_time_ms.saturating_sub(entry_time)
                } else {
                    0
                };

                // Gate trailing: at least 5s and in profit > 0.05%
                let trail_active = position_age_ms > 5_000 && pnl_pct > 0.0005;

                let mut trail_hit = false;
                let mut force_close_trail = false;

                if trail_active {
                    let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                    let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                        side_int,
                        entry,
                        mid_price,
                        pseudo_atr,
                        coin.positions
                            .scalp_position
                            .trailing_phase
                            .load(Ordering::Relaxed) as i32,
                        coin.positions
                            .scalp_position
                            .mfe_atr
                            .load(Ordering::Relaxed),
                        coin.positions
                            .scalp_position
                            .max_pnl_pct
                            .load(Ordering::Relaxed),
                        coin.positions
                            .scalp_position
                            .trail_stop
                            .load(Ordering::Relaxed),
                        1.5,
                        1.5,
                        2.0,
                        2.5,
                        1.5,
                        live_fee,
                    );

                    // Don't let trailing stop be tighter than the base SL
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

                    // Only ratchet up (long) / down (short) — never widen
                    let current_stored = coin
                        .positions
                        .scalp_position
                        .trail_stop
                        .load(Ordering::Relaxed);
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

                    coin.positions
                        .scalp_position
                        .trail_stop
                        .store(final_stop, Ordering::Relaxed);
                    coin.positions
                        .scalp_position
                        .trailing_phase
                        .store(trail_res.new_phase as u8, Ordering::Relaxed);
                    coin.positions
                        .scalp_position
                        .mfe_atr
                        .store(trail_res.mfe_atr, Ordering::Relaxed);
                    coin.positions
                        .scalp_position
                        .max_pnl_pct
                        .store(trail_res.max_pnl_pct, Ordering::Relaxed);

                    trail_hit = (is_long && mid_price <= final_stop)
                        || (!is_long && mid_price >= final_stop && final_stop > 0.0);
                    force_close_trail = trail_res.force_close;
                }

                let notional = qty * entry;
                let mut unrealized = pnl_pct * notional;

                let is_zombie = event_time_ms > 0 && position_age_ms > 1_800_000;
                let hurst_exponent = self.feature_engines[coin_id].get_features()[1] as f64;
                let zombie_close = is_zombie && hurst_exponent < 0.65;
                let zombie_promote = is_zombie && hurst_exponent >= 0.65;

                if zombie_promote && !coin.positions.swing_position.is_open() {
                    let old_trail = coin
                        .positions
                        .scalp_position
                        .trail_stop
                        .load(Ordering::Relaxed);
                    let (_, price, qty, margin_used) = coin.positions.scalp_position.close();
                    coin.positions.swing_position.open(
                        is_long,
                        price,
                        qty,
                        margin_used,
                        entry_time,
                        0.0,
                        0.0,
                    );
                    coin.positions
                        .swing_position
                        .trail_stop
                        .store(old_trail, Ordering::Relaxed);
                } else if pnl_pct >= scalp_tp
                    || pnl_pct <= -scalp_sl
                    || trail_hit
                    || force_close_trail
                    || zombie_close
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
                            exit_price = exit_price.max(sl_price);
                        }
                        if trail_hit {
                            let ts = coin
                                .positions
                                .scalp_position
                                .trail_stop
                                .load(Ordering::Relaxed);
                            if ts > 0.0 {
                                exit_price = exit_price.max(ts);
                            }
                        }
                        if pnl_pct >= scalp_tp {
                            exit_price = tp_price;
                        }
                    } else {
                        if pnl_pct <= -scalp_sl {
                            exit_price = exit_price.min(sl_price);
                        }
                        if trail_hit {
                            let ts = coin
                                .positions
                                .scalp_position
                                .trail_stop
                                .load(Ordering::Relaxed);
                            if ts > 0.0 {
                                exit_price = exit_price.min(ts);
                            }
                        }
                        if pnl_pct >= scalp_tp {
                            exit_price = tp_price;
                        }
                    }

                    unrealized = if is_long {
                        (exit_price - entry) * qty
                    } else {
                        (entry - exit_price) * qty
                    };
                    let gross_pnl = unrealized;
                    coin.scalp.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);

                    let _print_pnl_pct = if is_long {
                        (exit_price - entry) / entry
                    } else {
                        (entry - exit_price) / entry
                    };

                    let _reason = if trail_hit {
                        "TRAIL_HIT"
                    } else if force_close_trail {
                        "FORCE_CLOSE"
                    } else if zombie_close {
                        "ZOMBIE"
                    } else if pnl_pct >= scalp_tp {
                        "TP"
                    } else {
                        "SL"
                    };
                    telemetry_server::telemetry_log!(
                        "🛑 CLOSE SCALP [Coin {}]: Long={}, Entry={:.4}, Exit={:.4}, PnL={:.4}% (${:.4}), MFE_ATR={:.2}, Reason={}",
                        coin_id,
                        is_long,
                        entry,
                        exit_price,
                        _print_pnl_pct * 100.0,
                        unrealized,
                        coin.positions
                            .scalp_position
                            .mfe_atr
                            .load(Ordering::Relaxed),
                        _reason
                    );

                    // Fee: use real Binance rates
                    let close_fee = if pnl_pct >= scalp_tp {
                        self.arena
                            .config
                            .live_maker_fee
                            .load(Ordering::Relaxed)
                            .max(0.0002)
                    } else {
                        self.arena
                            .config
                            .live_taker_fee
                            .load(Ordering::Relaxed)
                            .max(0.0004)
                    };
                    unrealized -= notional * close_fee;

                    let (_, _, _, margin_used) = coin.positions.scalp_position.close();
                    let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena
                            .scalp_used_margin
                            .fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.scalp_used_margin.store(0.0, Ordering::Relaxed);
                    }
                    coin.scalp
                        .pnl_realized
                        .fetch_add(unrealized, Ordering::Relaxed);
                    self.arena
                        .unified_capital
                        .fetch_add(unrealized, Ordering::Relaxed);

                    // Kelly Feedback
                    let is_win = unrealized > 0.0;
                    let n = coin.scalp.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.scalp.win_rate.store(new_wr, Ordering::Relaxed);
                    let old_pf = coin.scalp.profit_factor.load(Ordering::Relaxed).max(0.1);
                    let decay = (1.0 / n.min(50.0)).clamp(0.02, 0.20);
                    let new_pf = if is_win {
                        (old_pf * (1.0 - decay) + (pnl_pct.abs() / 0.0020) * decay).clamp(0.1, 10.0)
                    } else {
                        (old_pf * (1.0 - decay) + (0.0020 / pnl_pct.abs().max(1e-4)) * decay).clamp(0.1, 10.0)
                    };
                    coin.scalp.profit_factor.store(new_pf, Ordering::Relaxed);
                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self
                        .arena
                        .config
                        .kelly_survival_cap_ratio
                        .load(Ordering::Relaxed);
                    let exp_mult = self
                        .arena
                        .config
                        .kelly_expansion_mult
                        .load(Ordering::Relaxed);
                    let kelly_f = risk_engine::kelly::calculate_kelly_fraction(
                        new_wr,
                        coin.scalp.profit_factor.load(Ordering::Relaxed),
                        curr_cap,
                        base_cap,
                        survival_ratio,
                        exp_mult,
                    );
                    coin.scalp.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                    coin.last_scalp_close_ts.store(event_time_ms, Ordering::Relaxed);

                    closed_scalp = Some((is_long, unrealized, qty));
                } else {
                    coin.scalp
                        .pnl_unrealized
                        .store(unrealized, Ordering::Relaxed);
                }
            }

            if coin.positions.swing_position.is_open() {
                let is_long = coin
                    .positions
                    .swing_position
                    .is_long
                    .load(Ordering::Relaxed);
                let entry = coin
                    .positions
                    .swing_position
                    .entry_price
                    .load(Ordering::Relaxed);
                let qty = coin
                    .positions
                    .swing_position
                    .quantity
                    .load(Ordering::Relaxed);

                let pnl_pct = if is_long {
                    (mid_price - entry) / entry
                } else {
                    (entry - mid_price) / entry
                };

                let current_atr = self.feature_engines[coin_id].get_atr_pct() * entry;
                let pseudo_atr = if current_atr > 0.0 {
                    current_atr
                } else {
                    mid_price * 0.01
                };

                let side_int = if is_long { 1 } else { -1 };

                let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed) + self.arena.config.live_taker_fee.load(Ordering::Relaxed);
                let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                    side_int,
                    entry,
                    mid_price,
                    pseudo_atr,
                    coin.positions
                        .swing_position
                        .trailing_phase
                        .load(Ordering::Relaxed) as i32,
                    coin.positions
                        .swing_position
                        .mfe_atr
                        .load(Ordering::Relaxed),
                    coin.positions
                        .swing_position
                        .max_pnl_pct
                        .load(Ordering::Relaxed),
                    coin.positions
                        .swing_position
                        .trail_stop
                        .load(Ordering::Relaxed),
                    2.5, // FIX #591: pullback_tol amplio (2.5 ATR) para permitir respiración en Swing
                    2.0,
                    3.0,
                    4.0,
                    3.5, // t_params for swing (trail_runner scale)
                    live_fee,
                );

                let current_stored = coin
                    .positions
                    .swing_position
                    .trail_stop
                    .load(Ordering::Relaxed);
                let safe_stop = if is_long {
                    if current_stored > 0.0 {
                        trail_res.stop_price.max(current_stored)
                    } else {
                        trail_res.stop_price
                    }
                } else {
                    if current_stored > 0.0 {
                        trail_res.stop_price.min(current_stored)
                    } else {
                        trail_res.stop_price
                    }
                };

                coin.positions
                    .swing_position
                    .trail_stop
                    .store(safe_stop, Ordering::Relaxed);
                coin.positions
                    .swing_position
                    .trailing_phase
                    .store(trail_res.new_phase as u8, Ordering::Relaxed);
                coin.positions
                    .swing_position
                    .mfe_atr
                    .store(trail_res.mfe_atr, Ordering::Relaxed);
                coin.positions
                    .swing_position
                    .max_pnl_pct
                    .store(trail_res.max_pnl_pct, Ordering::Relaxed);

                let trail_hit = (is_long && mid_price <= safe_stop && safe_stop > 0.0)
                    || (!is_long
                        && mid_price >= safe_stop
                        && safe_stop > 0.0);

                let notional = qty * entry;
                let mut unrealized = pnl_pct * notional;

                // Swing Cooldown/Hold Time Mechanism
                let entry_time = coin
                    .positions
                    .swing_position
                    .entry_time_ms
                    .load(Ordering::Relaxed);
                let hold_time_met =
                    (event_time_ms >= entry_time) && ((event_time_ms - entry_time) >= 300_000);

                if pnl_pct >= swing_tp
                    || pnl_pct <= -swing_sl
                    || (hold_time_met && (trail_hit || trail_res.force_close))
                {
                    let gross_pnl = unrealized;
                    coin.swing.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                    let live_maker = self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002);
                    let live_taker = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004);
                    let close_fee = if pnl_pct >= swing_tp { live_maker } else { live_taker };
                    unrealized -= notional * close_fee;

                    let (_, _, _, margin_used) = coin.positions.swing_position.close();
                    let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena
                            .swing_used_margin
                            .fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.swing_used_margin.store(0.0, Ordering::Relaxed);
                    }
                    coin.swing
                        .pnl_realized
                        .fetch_add(unrealized, Ordering::Relaxed);
                    self.arena
                        .unified_capital
                        .fetch_add(unrealized, Ordering::Relaxed);

                    // Kelly Feedback
                    let is_win = unrealized > 0.0;
                    let n = coin.swing.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.swing.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.swing.win_rate.store(new_wr, Ordering::Relaxed);
                    let old_pf = coin.swing.profit_factor.load(Ordering::Relaxed).max(0.1);
                    let decay = (1.0 / n.min(50.0)).clamp(0.02, 0.20);
                    let new_pf = if is_win {
                        (old_pf * (1.0 - decay) + (pnl_pct.abs() / 0.0050) * decay).clamp(0.1, 10.0)
                    } else {
                        (old_pf * (1.0 - decay) + (0.0050 / pnl_pct.abs().max(1e-4)) * decay).clamp(0.1, 10.0)
                    };
                    coin.swing.profit_factor.store(new_pf, Ordering::Relaxed);
                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self
                        .arena
                        .config
                        .kelly_survival_cap_ratio
                        .load(Ordering::Relaxed);
                    let exp_mult = self
                        .arena
                        .config
                        .kelly_expansion_mult
                        .load(Ordering::Relaxed);
                    let kelly_f = risk_engine::kelly::calculate_kelly_fraction(
                        new_wr,
                        coin.swing.profit_factor.load(Ordering::Relaxed),
                        curr_cap,
                        base_cap,
                        survival_ratio,
                        exp_mult,
                    );
                    coin.swing.kelly_fraction.store(kelly_f, Ordering::Relaxed);

                    closed_swing = Some((is_long, unrealized, qty));
                } else {
                    coin.swing
                        .pnl_unrealized
                        .store(unrealized, Ordering::Relaxed);
                }
            }

            let mut new_scalp = None;
            let mut new_swing = None;

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
            let a_t = self.feature_engines[coin_id].a_t;
            let shannon_ent = self.feature_engines[coin_id].entropy.current();
            let vpin_val = self.feature_engines[coin_id].cvpin.current_vpin();
            let total_vol = (bid_qty + ask_qty).max(1e-8);
            let p_bid = (bid_qty / total_vol).clamp(0.0001, 0.9999);
            let p_ask = (ask_qty / total_vol).clamp(0.0001, 0.9999);
            let tsallis_ent = ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5).clamp(0.0, 1.0);

            self.arena.registry.set("atr_1s", (atr_pct * mid_price).max(0.0001));
            self.arena.registry.set("atr_5s", (atr_pct * mid_price * 2.236).max(0.0005));
            self.arena.registry.set("atr_1m", (atr_pct * mid_price * 7.746).max(0.0020));
            self.arena.registry.set("order_flow_direction", flow_dir);
            self.arena.registry.set("order_book_imbalance", obi_val);
            self.arena.registry.set("orderbook_imbalance", obi_val);
            self.arena.registry.set("order_flow_imbalance", obi_val);
            self.arena.registry.set("price_velocity", v_t);
            self.arena.registry.set("order_flow_velocity", v_t);
            self.arena.registry.set("price_acceleration", a_t);
            self.arena.registry.set("shannon_entropy", shannon_ent);
            self.arena.registry.set("tsallis_q_entropy", tsallis_ent);
            self.arena.registry.set("hurst_exponent", hurst_val);
            self.arena.registry.set("global_hurst", hurst_val);
            self.arena.registry.set("atr_pct", atr_pct);
            self.arena.registry.set("relative_atr_pct", atr_pct);
            self.arena.registry.set("vpin", vpin_val);
            self.arena.registry.set("vpin_toxicity", vpin_val);
            self.arena.registry.set("cvpin", vpin_val);
            self.arena.registry.set("order_flow_vpin", vpin_val);
            self.arena.registry.set("hawkes_intensity", (1.0 + obi_val.abs() * 2.0).clamp(0.1, 5.0));
            self.arena.registry.set("bessel_alpha", 1.5);
            self.arena.registry.set("hawkes_dt", 0.05);
            self.arena.registry.set("microstructure_noise_variance", (atr_pct * 0.1).max(0.00001));

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
            self.arena.registry.set("ml_prob", ml_prob);
            self.arena.registry.set("ml_prob_scalp", ml_prob);

            let nn_score: f64 = (ml_prob - 0.5) * 2.0;

            let current_obi = obi_val;
            let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
            let dynamic_obi_thr = self.arena.config.dynamic_obi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);
            let dynamic_ema_thr = self.arena.config.dynamic_ema_trend.load(Ordering::Relaxed);
            let dynamic_ofi_thr = self.arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed).clamp(0.15, 0.95);

            let micro_trend = self.feature_engines[coin_id].get_micro_trend();
            let macro_trend = self.feature_engines[coin_id].get_macro_trend();

            self.arena.registry.set("ema_trend", micro_trend);
            self.arena.registry.set("ema_trend_swing", macro_trend);
            self.arena.registry.set("trend_direction", if macro_trend > 0.0 { 1.0 } else if macro_trend < 0.0 { -1.0 } else { 0.0 });
            self.arena.registry.set("conformal_p_value", 0.95);
            self.arena.registry.set("conformal_alpha", 0.10);
            let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
            let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
            let total_vol_cvd = buy_vol + sell_vol;
            let rolling_cvd = if total_vol_cvd > 0.0 { (buy_vol - sell_vol) / total_vol_cvd } else { 0.0 };
            let ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;

            let obi_norm = (current_obi / dynamic_obi_thr).clamp(-1.5, 1.5);
            let ofi_norm = (ofi / dynamic_ofi_thr).clamp(-1.5, 1.5);
            self.arena.registry.set("vecm_zscore", obi_norm * 1.5);
            self.arena.registry.set("cointegration_zscore", obi_norm * 1.5);
            let micro_score: f64 = (obi_norm * 0.40 + ofi_norm * 0.40 + rolling_cvd * 0.20).clamp(-1.0, 1.0);

            let (tensor_scalp, tensor_swing) = self.tensor_orchestrator.evaluate_dual_consensus();
            let tensor_boost = match tensor_scalp.signal {
                SignalType::Long => tensor_scalp.net_confidence.clamp(0.0, 1.0),
                SignalType::Short => -tensor_scalp.net_confidence.clamp(0.0, 1.0),
                SignalType::Flat => 0.0,
            };

            // Unified Bayesian Fusion: DarkAlpha ML (35%) + Microstructure L2 (40%) + Multi-Strategy Ensemble (25%)
            let raw_composite = nn_score * 0.35 + micro_score * 0.40 + tensor_boost * 0.25;
            let composite_score: f64 = (raw_composite * hebbian_mult).clamp(-1.0, 1.0);

            let mut scalp_intent = SignalIntent::flat();
            if atr_pct > dynamic_atr_min {
                let is_mean_reverting = hurst_val < 0.45;
                let is_trending = hurst_val >= 0.50;

                let ema_slow = self.feature_engines[coin_id].ema_slow;
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let price_stretch = if ema_slow > 0.0 { (mid_price - ema_slow) / cur_atr } else { 0.0 };
                let not_overextended_long = price_stretch <= 1.2;
                let not_overextended_short = price_stretch >= -1.2;

                if is_trending {
                    if composite_score > 0.25 && micro_trend > dynamic_ema_thr && not_overextended_long {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: (0.70f64 + composite_score * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    } else if composite_score < -0.25 && micro_trend < -dynamic_ema_thr && not_overextended_short {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: (0.70f64 + composite_score.abs() * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    }
                } else if is_mean_reverting {
                    // Fade overextensions in oscillating market
                    if composite_score > 0.25 && current_obi > dynamic_obi_thr && micro_trend > dynamic_ema_thr {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: (0.70f64 + composite_score * 0.30f64).min(1.0f64),
                            ..Default::default()
                        };
                    } else if composite_score < -0.25 && current_obi < -dynamic_obi_thr && micro_trend < -dynamic_ema_thr {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: (0.70f64 + composite_score.abs() * 0.30f64).min(1.0f64),
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

                // Cooldown: 10s post-close to prevent churn on false micro-breakouts
                let last_close = coin.last_scalp_close_ts.load(Ordering::Relaxed);
                if event_time_ms.saturating_sub(last_close) < 10_000 {
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

                if scalp_intent.signal == SignalType::Long {
                    if cvd_ratio < -0.10 {
                        // telemetry_server::telemetry_log!("🛡️ [VETO CVD] Scalp Long abortado por CVD negativo ({:.2}%)", cvd_ratio * 100.0);
                        scalp_intent = SignalIntent::flat();
                    } else if wall_imbalance < -0.40 {
                        // telemetry_server::telemetry_log!("🧱 [VETO WALL] Scalp Long abortado frente a Muro Ask masivo (Imbalance: {:.2})", wall_imbalance);
                        scalp_intent = SignalIntent::flat();
                    }
                } else if scalp_intent.signal == SignalType::Short {
                    if cvd_ratio > 0.10 {
                        // telemetry_server::telemetry_log!("🛡️ [VETO CVD] Scalp Short abortado por CVD positivo ({:.2}%)", cvd_ratio * 100.0);
                        scalp_intent = SignalIntent::flat();
                    } else if wall_imbalance > 0.40 {
                        // telemetry_server::telemetry_log!("🧱 [VETO WALL] Scalp Short abortado frente a Muro Bid masivo (Imbalance: {:.2})", wall_imbalance);
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
            if (swing_nn_pred <= 0.0001 || swing_nn_pred >= 0.9999 || swing_nn_pred == 0.5) && atr_pct > 0.0001 {
                let combined = self.build_54d_tensor(coin_id, bid_qty, ask_qty, mid_price, omni_features);
                let sf = self.feature_engines[coin_id].get_swing_features();
                if let Some(nn) = &mut self.swing_nn {
                    let in_dim = nn.layer1.in_features;
                    let p_opt = if in_dim == 34 {
                        let mut swing_feats_f64 = [0.0; 34];
                        for idx in 0..34 {
                            swing_feats_f64[idx] = sf[idx] as f64;
                        }
                        nn.predict(&swing_feats_f64)
                    } else if in_dim == 12 {
                        let mut micro_f64 = [0.0; 12];
                        for idx in 0..12 {
                            micro_f64[idx] = features[idx] as f64;
                        }
                        nn.predict(&micro_f64)
                    } else {
                        nn.predict(&combined)
                    };
                    if let Some(p) = p_opt {
                        swing_nn_pred = p.clamp(0.01, 0.99);
                    }
                }
            }

            let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
            let mut swing_intent = self.swing_engines[coin_id].evaluate_trend(
                mid_price,
                hurst_exponent,
                trend_threshold,
                swing_nn_pred,
                &self.arena,
            );
            if swing_intent.signal == SignalType::Flat && tensor_swing.signal != SignalType::Flat && tensor_swing.net_confidence.abs() > 0.60 {
                swing_intent = SignalIntent {
                    signal: tensor_swing.signal,
                    confidence: tensor_swing.net_confidence.abs().clamp(0.60, 1.0),
                    ..Default::default()
                };
            }

            // Unificación Cuántica: Evaluacion Independiente
            let (scalp_order, swing_order) =
                self.risk_engine
                    .evaluate_order(coin_id, scalp_intent, swing_intent, &self.arena);

            if scalp_order.signal != SignalType::Flat || swing_order.signal != SignalType::Flat {
                // Se autoriza la ejecución (escalado dual).
                // Registramos las posiciones virtuales para backtesting/P&L Tracking.
                if !coin.positions.scalp_position.is_open()
                    && scalp_order.signal != SignalType::Flat
                {
                    let is_long = scalp_order.signal == SignalType::Long;
                    // Bolsillo de Capital Scalp (50%)
                    let cap_split = self
                        .arena
                        .config
                        .capital_split_scalp
                        .load(Ordering::Relaxed);
                    let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);

                    // --- GLOBAL TENSOR SCALP (Tick) ---
                    let ml_thr_long = self
                        .arena
                        .config
                        .ml_threshold_long
                        .load(Ordering::Relaxed);
                    let ml_thr_short = self
                        .arena
                        .config
                        .ml_threshold_short
                        .load(Ordering::Relaxed);
                    let mut active_opportunities: f64 = 0.0;
                    for i in 0..30 {
                        let p = self.arena.coins[i].ml_prob.load(Ordering::Relaxed);
                        if p > ml_thr_long || p < ml_thr_short {
                            active_opportunities += 1.0;
                        }
                    }
                    let active_opportunities = active_opportunities.max(1.0);

                    let kelly = coin
                        .scalp
                        .kelly_fraction
                        .load(Ordering::Relaxed)
                        .clamp(0.1, 1.0);
                    let adjusted_kelly = (kelly / active_opportunities).max(0.1).min(1.0);

                    let min_margin_for_binance = 5.05 / leverage.max(1.0);
                    let kelly_alloc = if current_cap <= 30.0 {
                        (current_cap * 0.40).clamp(min_margin_for_binance, current_cap * 0.70)
                    } else {
                        (current_cap.max(0.0) * cap_split * adjusted_kelly * scalp_intent.confidence).clamp(min_margin_for_binance, current_cap * 0.50)
                    };
                    let mut margin_required = kelly_alloc;

                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                    let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);

                    // Solo abrimos si cumplimos MIN_NOTIONAL ($5.00), margen disponible (< 95% de utilizacion) y capital positivo
                    if current_cap > 0.0 && margin_required * leverage >= 5.0 && current_used + margin_required <= current_cap * 0.95 {
                        self.arena
                            .scalp_used_margin
                            .fetch_add(margin_required, Ordering::Relaxed);
                        let entry_price = if is_long { ask } else { bid };
                        let qty = (margin_required * leverage) / entry_price;
                        coin.positions.scalp_position.open(
                            is_long,
                            entry_price,
                            qty,
                            margin_required,
                            event_time_ms,
                            0.0,
                            0.0,
                        );
                        new_scalp = Some((is_long, entry_price, qty));
                    }
                }

                if !coin.positions.swing_position.is_open()
                    && swing_order.signal != SignalType::Flat
                {
                    let is_long = swing_order.signal == SignalType::Long;
                    // Bolsillo de Capital Swing (resto del scalp)
                    let cap_split = 1.0
                        - self
                            .arena
                            .config
                            .capital_split_scalp
                            .load(Ordering::Relaxed);
                    let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    // Swing usa el MISMO leverage del genoma (sin cap artificial)
                    let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);

                    let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);

                    // --- GLOBAL TENSOR SWING (Tick) ---
                    let mut active_swing_opportunities: f64 = 0.0;
                    for i in 0..30 {
                        let h = self.arena.coins[i].hurst_exponent.load(Ordering::Relaxed);
                        if h > trend_threshold {
                            active_swing_opportunities += 1.0;
                        }
                    }
                    let active_swing_opportunities = active_swing_opportunities.max(1.0);

                    let kelly = coin
                        .swing
                        .kelly_fraction
                        .load(Ordering::Relaxed)
                        .clamp(0.1, 1.0);
                    let adjusted_kelly = (kelly / active_swing_opportunities).max(0.01).min(1.0);

                    let mut margin_required =
                        current_cap.max(0.0) * cap_split * adjusted_kelly * swing_intent.confidence;
                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                    // FIX #1409: Enforce Binance Futures $5.00 MIN_NOTIONAL for Swing
                    if margin_required * leverage < 5.0 && leverage > 0.0 {
                        margin_required = 5.05 / leverage;
                    }
                    let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);

                    if current_cap > 0.0 && margin_required * leverage >= 5.0 && current_used + margin_required <= current_cap * 0.95 {
                        self.arena
                            .swing_used_margin
                            .fetch_add(margin_required, Ordering::Relaxed);

                        let base_price = if is_long { ask } else { bid };
                        let nominal_size = margin_required * leverage;

                        // Slippage model (Market Impact): 0.05% por cada 1M nominal
                        let slippage_impact = (nominal_size / 1_000_000.0) * 0.0005;
                        let real_entry_price = if is_long {
                            base_price * (1.0 + slippage_impact)
                        } else {
                            base_price * (1.0 - slippage_impact)
                        };

                        let entry_fee_rate = self
                            .arena
                            .config
                            .live_taker_fee
                            .load(Ordering::Relaxed)
                            .max(0.0002);
                        let fee_paid = nominal_size * entry_fee_rate;
                        self.arena
                            .unified_capital
                            .fetch_add(-fee_paid, Ordering::Relaxed);

                        let qty = nominal_size / real_entry_price;
                        coin.positions.swing_position.open(
                            is_long,
                            real_entry_price,
                            qty,
                            margin_required,
                            event_time_ms,
                            0.0,
                            0.0,
                        );
                        new_swing = Some((is_long, real_entry_price, qty));
                    }
                }

                // En Producción, retornaríamos `net_order` a través de un canal para la API REST/WS.
                // Por ahora mantenemos la firma, pero `net_order` es el estado real.
            }
            // --- 3. MARKET MAKING ---
            let mut final_maker_quote = None;
            let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);

            if latency_ms <= 25 {
                // Generar cotización pasiva basada en ATR/Hurst e inventario.
                let volatility = self.feature_engines[coin_id].get_features()[3] as f64; // [3] = v_t (volatility), NOT Hurst

                let mut inventory_delta_usd = 0.0;
                if coin.positions.scalp_position.is_open() {
                    let is_long = coin
                        .positions
                        .scalp_position
                        .is_long
                        .load(Ordering::Relaxed);
                    let notional = coin
                        .positions
                        .scalp_position
                        .quantity
                        .load(Ordering::Relaxed)
                        * mid_price;
                    inventory_delta_usd += if is_long { notional } else { -notional };
                }
                // FIX #566: Incluir exposición de Swing para control de inventario Avellaneda-Stoikov exacto
                if coin.positions.swing_position.is_open() {
                    let is_long = coin
                        .positions
                        .swing_position
                        .is_long
                        .load(Ordering::Relaxed);
                    let notional = coin
                        .positions
                        .swing_position
                        .quantity
                        .load(Ordering::Relaxed)
                        * mid_price;
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
}
