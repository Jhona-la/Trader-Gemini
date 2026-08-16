#![feature(portable_simd)]

pub mod darwin;
pub mod trailing;
pub mod reality_physics;
pub mod bootloader;
pub mod orchestrator;
pub mod slippage_predictor;

use quantum_arena::GlobalArena;
use signal_engine::{ScalpEngine, SwingEngine, MakerEngine, MakerQuote, SignalType, SignalIntent};
use risk_engine::RiskEngine;
use crate::stateful_engine::{StatefulEngine, MarketRegime};
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
    pub scalp_forest: Option<Arc<crate::ml_inference::NanoForest>>,
    pub swing_nn: Option<dark_alpha_engine::DarkAlphaEngine>,
    pub model_rx: Option<std::sync::mpsc::Receiver<dark_alpha_engine::DarkAlphaEngine>>,
    pub last_ml_prob: f32,
    pub flight_recorder: Option<Arc<telemetry_server::FlightRecorder>>,
    pub reality: reality_physics::RealityPhysics,
    pub last_scalp_intent: Vec<SignalIntent>,
    pub last_swing_intent: Vec<SignalIntent>,
    pub lakehouse: Option<Arc<storage_engine::LakehouseWarehouse>>,
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
        let swing_nn = if let Ok(json_data) = std::fs::read_to_string("models/DarkAlpha_BTCUSDT.json") {
            if let Ok(model) = serde_json::from_str::<dark_alpha_engine::DarkAlphaEngine>(&json_data) {
                telemetry_server::telemetry_log!("🧠 [DARK ALPHA] Initial Boot Model Loaded Successfully (in_features: {})", model.layer1.in_features);
                Some(model)
            } else {
                telemetry_server::telemetry_log!("⚠️ [DARK ALPHA] Failed to parse models/DarkAlpha_BTCUSDT.json. Using fallback.");
                Some(dark_alpha_engine::DarkAlphaEngine::new(54, 64, 32))
            }
        } else {
            telemetry_server::telemetry_log!("⚠️ [DARK ALPHA] models/DarkAlpha_BTCUSDT.json not found. Using fallback.");
            Some(dark_alpha_engine::DarkAlphaEngine::new(54, 64, 32))
        };
        
        Self {
            arena,
            risk_engine: RiskEngine::new(initial_capital),
            scalp_engines,
            swing_engines,
            maker_engines,
            feature_engines,
            scalp_forest,
            swing_nn,
            model_rx: None,
            last_ml_prob: 0.5,
            flight_recorder: None,
            reality: reality_physics::RealityPhysics::default(),
            last_scalp_intent: vec![SignalIntent::flat(); 30],
            last_swing_intent: vec![SignalIntent::flat(); 30],
            lakehouse: None,
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
    }
    
    pub fn set_model_rx(&mut self, rx: std::sync::mpsc::Receiver<dark_alpha_engine::DarkAlphaEngine>) {
        self.model_rx = Some(rx);
    }
    
    pub fn set_lakehouse(&mut self, lakehouse: Arc<storage_engine::LakehouseWarehouse>) {
        self.lakehouse = Some(lakehouse);
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
        event_time_ms: u64, latency_panic: bool, omni_features: &[f64; 54]) -> (Option<(bool, f64, f64)>, Option<(bool, f64, f64)>, Option<(bool, f64, f64)>, Option<(bool, f64, f64)>) {
        telemetry_server::profile_node!("GodEngineCore::process_event", {
        
        // Zero-copy Hot-Reloading Check
        if let Some(rx) = &self.model_rx {
            if let Ok(new_model) = rx.try_recv() {
                self.swing_nn = Some(new_model);
                telemetry_server::telemetry_log!("🧠 [DARK ALPHA] Hot-Reload Successful! New weights absorbed in Zero-Copy.");
            }
        }
        
        if self.arena.kill_switch_active.load(Ordering::Relaxed) {
            return (None, None, None, None);
        }

        // Auto-refresh ML models once every 1000 ticks to support zero-downtime hot-reload
        if self.arena.tick_counter.load(Ordering::Relaxed).is_multiple_of(1000) {
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
            let _ = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
            feature_engine.update_macro_features(depth_obi, depth_micro_div, 0.0, event_time_ms);
            self.arena.update_market_data(coin_id, bid, ask, bid_qty, ask_qty, 0);
            self.arena.increment_tick();
        }
        
        let mut closed_scalp = None;
        let mut closed_swing = None;
        let mut new_scalp = None;
        let mut new_swing = None;
        
        let scalp_tp = self.arena.config.scalp_tp_base.load(Ordering::Relaxed);
        let scalp_sl = self.arena.config.scalp_sl_base.load(Ordering::Relaxed);
        let swing_tp = self.arena.config.swing_tp_base.load(Ordering::Relaxed);
        let swing_sl = self.arena.config.swing_sl_base.load(Ordering::Relaxed);
        let sim_fee_rate = self.arena.config.max_fee_pct.load(Ordering::Relaxed);
        
        let coin = &self.arena.coins[coin_id];
        let scalp_regime = self.feature_engines[coin_id].get_market_regime();
        let swing_regime = self.feature_engines[coin_id].get_market_regime();
        
        static DBG_TRADE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if DBG_TRADE.fetch_add(1, Ordering::Relaxed) < 10 {
            // telemetry_server::telemetry_log!("DEBUG EVOL: process_event start! is_trade={}", is_trade);
        }
        
        // --- SCALP EVALUATION (Unified Closure Logic) ---
        if is_trade {
            let base_scalp_tp = self.arena.config.scalp_tp_base.load(Ordering::Relaxed);
            let base_scalp_sl = self.arena.config.scalp_sl_base.load(Ordering::Relaxed);
            let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
            let min_atr_pct = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed).max(0.001); // Force at least 0.1% floor
            let atr_pct = raw_atr_pct.max(min_atr_pct);
            
            // Dynamic SL/TP floor to prevent fee-chop
            let scalp_tp = base_scalp_tp.max(atr_pct * 2.0).max(0.002);
            let scalp_sl = base_scalp_sl.max(atr_pct * 1.5).max(0.0015);
            
            if coin.positions.scalp_position.is_open() {
                let is_long = coin.positions.scalp_position.is_long.load(Ordering::Relaxed);
                let entry = coin.positions.scalp_position.entry_price.load(Ordering::Relaxed);
                let qty = coin.positions.scalp_position.quantity.load(Ordering::Relaxed);
                
                let pnl_pct = if is_long {
                    (current_price - entry) / entry
                } else {
                    (entry - current_price) / entry
                };
                
                let current_atr = atr_pct * entry;
                let pseudo_atr = current_atr;
                
                let side_int = if is_long { 1 } else { -1 };
                
                let entry_time = coin.positions.scalp_position.entry_time_ms.load(Ordering::Relaxed);
                let position_age_ms = if event_time_ms > 0 { event_time_ms.saturating_sub(entry_time) } else { 0 };
                let warmup_ticks = 50; // Don't trail until we have stable ATR
                let ticks_since_open = coin.positions.scalp_position.trailing_phase.load(Ordering::Relaxed) as u64; // reuse for tick count in phase 0
                
                // Only evaluate trailing if position has been alive long enough and ATR is stabilized
                let trail_active = position_age_ms > 5_000 && pnl_pct > 0.0005; // at least 5s and in profit > 0.05%
                
                let mut trail_hit = false;
                let mut force_close_trail = false;
                
                if trail_active {
                    let trail_res = crate::trailing::evaluate_quantum_trailing(
                        side_int, entry, current_price, pseudo_atr,
                        coin.positions.scalp_position.trailing_phase.load(Ordering::Relaxed) as i32,
                        coin.positions.scalp_position.mfe_atr.load(Ordering::Relaxed),
                        coin.positions.scalp_position.max_pnl_pct.load(Ordering::Relaxed),
                        coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed),
                        1.5, 1.5, 2.0, 2.5, 1.5 // t_params for scalp
                    );
                    
                    // Don't let trailing stop be tighter than the base SL
                    let sl_floor = if is_long { entry * (1.0 - scalp_sl) } else { entry * (1.0 + scalp_sl) };
                    let safe_stop = if is_long {
                        trail_res.stop_price.max(sl_floor) // For long, stop should not be below SL floor (but can be above)
                    } else {
                        if trail_res.stop_price > 0.0 { trail_res.stop_price.min(sl_floor) } else { sl_floor }
                    };
                    
                    // Only ratchet up (long) / down (short) — never widen the stop
                    let current_stored = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                    let final_stop = if is_long {
                        if current_stored > 0.0 { safe_stop.max(current_stored) } else { safe_stop }
                    } else {
                        if current_stored > 0.0 { safe_stop.min(current_stored) } else { safe_stop }
                    };
                    
                    coin.positions.scalp_position.trail_stop.store(final_stop, Ordering::Relaxed);
                    coin.positions.scalp_position.trailing_phase.store(trail_res.new_phase as u8, Ordering::Relaxed);
                    coin.positions.scalp_position.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
                    coin.positions.scalp_position.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);
                    
                    trail_hit = (is_long && current_price <= final_stop) || (!is_long && current_price >= final_stop && final_stop > 0.0);
                    force_close_trail = trail_res.force_close;
                }
                
                let notional = qty * entry;
                let mut unrealized = pnl_pct * notional;
                
                let is_zombie = event_time_ms > 0 && position_age_ms > 1_800_000;
                let hurst_exponent = self.feature_engines[coin_id].get_features()[1] as f64;
                let zombie_close = is_zombie && hurst_exponent < 0.65;
                let zombie_promote = is_zombie && hurst_exponent >= 0.65;
                
                if zombie_promote && !coin.positions.swing_position.is_open() {
                    let (_, price, qty, margin_used) = coin.positions.scalp_position.close();
                    coin.positions.swing_position.open(is_long, price, qty, margin_used, entry_time, 0.0, 0.0);
                    coin.positions.swing_position.trail_stop.store(coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed), Ordering::Relaxed);
                } else if pnl_pct >= scalp_tp || pnl_pct <= -scalp_sl || trail_hit || force_close_trail || zombie_close {
                    let sl_price = if is_long { entry * (1.0 - scalp_sl) } else { entry * (1.0 + scalp_sl) };
                    let tp_price = if is_long { entry * (1.0 + scalp_tp) } else { entry * (1.0 - scalp_tp) };
                    
                    let mut exit_price = current_price;
                    if is_long {
                        if pnl_pct <= -scalp_sl { exit_price = exit_price.max(sl_price); }
                        if trail_hit {
                            let ts = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.max(ts); }
                        }
                        if pnl_pct >= scalp_tp { exit_price = tp_price; }
                    } else {
                        if pnl_pct <= -scalp_sl { exit_price = exit_price.min(sl_price); }
                        if trail_hit {
                            let ts = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                            if ts > 0.0 { exit_price = exit_price.min(ts); }
                        }
                        if pnl_pct >= scalp_tp { exit_price = tp_price; }
                    }
                    
                    unrealized = if is_long { (exit_price - entry) * qty } else { (entry - exit_price) * qty };
                    let gross_pnl = unrealized;
                    coin.scalp.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                    
                    let _reason = if trail_hit { "TRAIL_HIT" } else if force_close_trail { "FORCE_CLOSE" } else if zombie_close { "ZOMBIE" } else if pnl_pct >= scalp_tp { "TP" } else { "SL" };
                    let _print_pnl_pct = if is_long { (exit_price - entry) / entry } else { (entry - exit_price) / entry };
                    
                    telemetry_server::telemetry_log!("🛑 CLOSE SCALP [Coin {}]: Long={}, Entry={:.4}, Exit={:.4}, PnL={:.4}% (${:.4}), pseudo_atr={:.4}, Reason={}", coin_id, is_long, entry, exit_price, _print_pnl_pct * 100.0, unrealized, pseudo_atr, _reason);
                    let close_fee = if pnl_pct >= scalp_tp { self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002) } else { self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004) }; 
                    unrealized -= notional * close_fee; 
                    
                    let (_, _, _, margin_used) = coin.positions.scalp_position.close();
                    let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena.scalp_used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.scalp_used_margin.store(0.0, Ordering::Relaxed);
                    }
                    coin.scalp.pnl_realized.fetch_add(unrealized, Ordering::Relaxed);
                    self.arena.unified_capital.fetch_add(unrealized, Ordering::Relaxed);
                    
                    // Kelly Feedback
                    let is_win = unrealized > 0.0;
                    let n = coin.scalp.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.scalp.win_rate.store(new_wr, Ordering::Relaxed);
                    if is_win && pnl_pct.abs() > 0.0 {
                        let old_pf = coin.scalp.profit_factor.load(Ordering::Relaxed);
                        let new_pf = old_pf + ((pnl_pct.abs() / old_pf.max(0.001) - old_pf) / n);
                        coin.scalp.profit_factor.store(new_pf.max(0.1), Ordering::Relaxed);
                    }
                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                    let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                    let kelly_f = risk_engine::kelly::calculate_kelly_fraction(new_wr, coin.scalp.profit_factor.load(Ordering::Relaxed), curr_cap, base_cap, survival_ratio, exp_mult);
                    coin.scalp.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                    
                    closed_scalp = Some((is_long, unrealized, qty));
                } else {
                    coin.scalp.pnl_unrealized.store(unrealized, Ordering::Relaxed);
                }
            }
            static DBG_REGIME: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            if DBG_REGIME.fetch_add(1, Ordering::Relaxed) < 10 {
                // telemetry_server::telemetry_log!("DEBUG EVOL: Checking intent... is_open={} latency={} scalp_regime={:?}", coin.positions.scalp_position.is_open(), latency_panic, scalp_regime);
            }
            
            // FASE 17: Removed hardcoded regime gate. Scalp can operate in ANY regime.
            // The RiskEngine and Kelly already modulate exposure based on regime risk.
            if !coin.positions.scalp_position.is_open() && !latency_panic {
                let features = self.feature_engines[coin_id].get_features();
                let mut ml_prob = self.scalp_forest.as_ref().map(|f| f.predict(&features).unwrap_or(0.5) as f64).unwrap_or(0.5);
                if ml_prob >= 0.9999 || ml_prob <= 0.0001 {
                    ml_prob = 0.5; // Fallback para modelo degenerado / saturado
                }
                self.last_ml_prob = ml_prob as f32;
                
                coin.ml_prob.store(ml_prob, Ordering::Relaxed);
                coin.hurst_exponent.store(features[1] as f64, Ordering::Relaxed); // [1] = Hurst
                
                if let Some(lh) = &self.lakehouse {
                    lh.record_tensor(quantum_arena::symbol_registry::spec(coin_id).symbol, event_time_ms, features.to_vec(), ml_prob as f32, 0.0);
                }
                
                let ml_threshold = self.arena.config.scalp_obi_threshold.load(Ordering::Relaxed);
                
                let mut scalp_intent = SignalIntent::flat();
                let atr_pct = self.feature_engines[coin_id].get_atr_pct();
                let hurst = features[1] as f64;
                
                static DBG_EVAL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                let eval_count = DBG_EVAL.fetch_add(1, Ordering::Relaxed);
                if eval_count % 100 == 0 {
                    // telemetry_server::telemetry_log!("DEBUG EVOL: Inside scalp eval! atr_pct={:.6} hurst={:.3} scalp_regime={:?} ml_prob={:.3}", 
                    //         atr_pct, hurst, scalp_regime, ml_prob);
                }
                
                // G-04: Scalping Entry — Hybrid ML + Microstructure Signal System
                // FASE 17: Si el ML es indeciso (~0.5), usamos microestructura pura (OBI+OFI)
                // como señal direccional. Esto es estándar institucional para HFT.
                let effective_threshold = ml_threshold.max(0.51);
                
                if atr_pct > 0.000001 { 
                    if ml_prob > effective_threshold {
                        scalp_intent = SignalIntent { signal: SignalType::Long, confidence: ml_prob, ..Default::default() };
                    } else if ml_prob < (1.0 - effective_threshold) {
                        scalp_intent = SignalIntent { signal: SignalType::Short, confidence: 1.0 - ml_prob, ..Default::default() };
                    } else {
                        // ML INDECISO (~0.5): Bypass con Microestructura Pura usando el Motor de Scalp
                        let z_target = self.arena.config.turbo_z_score_stdev.load(Ordering::Relaxed);
                        scalp_intent = self.scalp_engines[coin_id].evaluate_microstructure(bid_qty, ask_qty, z_target, &self.arena);
                    }
                }
                
                // --- CVD & L2 Wall HARD FILTERS (VETOS) ---
                if scalp_intent.signal != SignalType::Flat {
                    let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
                    let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
                    let cvd = buy_vol - sell_vol;
                    let total_vol_cvd = buy_vol + sell_vol;
                    let cvd_ratio = if total_vol_cvd > 0.0 { cvd / total_vol_cvd } else { 0.0 };

                    let bid_wall = coin.l2_bid_wall.load(Ordering::Relaxed);
                    let ask_wall = coin.l2_ask_wall.load(Ordering::Relaxed);
                    let total_wall = bid_wall + ask_wall;
                    let wall_imbalance = if total_wall > 0.0 { (bid_wall - ask_wall) / total_wall } else { 0.0 };
                    // FASE 17: Vetos CVD/Wall ahora usan thresholds del genoma
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
                    payload[0..4].copy_from_slice(&(ml_prob as f32).to_le_bytes());
                    fr.record(telemetry_server::FlightEvent {
                        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,
                        trace_id: self.arena.tick_counter.load(Ordering::Relaxed),
                        event_type: 1,
                        payload,
                    });
                }
                
                self.last_scalp_intent[coin_id] = scalp_intent;
                let swing_intent = self.last_swing_intent[coin_id];
                
                let (scalp_order, _swing_order) = self.risk_engine.evaluate_order(coin_id, scalp_intent, swing_intent, &self.arena);
                
                if scalp_order.signal != SignalType::Flat {
                    let is_long = scalp_order.signal == SignalType::Long;
                    let _cap_split = self.arena.config.capital_split_scalp.load(Ordering::Relaxed);
                    let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);
                    
                    let mut active_opportunities: f64 = 0.0;
                    for i in 0..30 {
                        let p = self.arena.coins[i].ml_prob.load(Ordering::Relaxed);
                        if p > ml_threshold || p < (1.0 - ml_threshold) {
                            active_opportunities += 1.0;
                        }
                    }
                    let active_opportunities = active_opportunities.max(1.0); 
                    
                    // FASE B: Sinergia Real. Usamos el margen dictado por el RiskEngine.
                    let mut margin_required = scalp_order.volume_usd / active_opportunities;

                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                    let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                    
                    if current_cap > 0.0 && current_used + margin_required <= current_cap * 0.95 {
                        self.arena.scalp_used_margin.fetch_add(margin_required, Ordering::Relaxed);
                        
                        // FRICCION REAL: Entramos cruzando el spread y deslizando el orderbook
                        let nominal_size = margin_required * leverage;
                        let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                        let base_price = if is_long { ask } else { bid };
                        let (real_entry_price, _entry_fee) = self.reality.calculate_market_entry(base_price, is_long, nominal_size, tick_volatility);
                        let qty = nominal_size / real_entry_price;
                        
                        static DBG_COUNT_2: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                        if DBG_COUNT_2.fetch_add(1, Ordering::Relaxed) < 100 {
                            // telemetry_server::telemetry_log!("DEBUG EVOL: Opening Scalp! is_long={} qty={} margin={}", is_long, qty, margin_required);
                        }
                        
                        // FRICCION REAL: Aplicamos fee Taker real al abrir (market order)
                        let entry_fee_rate = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0002); // Floor: 0.02% maker minimum
                        let entry_fee = margin_required * leverage * entry_fee_rate; 
                        self.arena.unified_capital.fetch_add(-entry_fee, Ordering::Relaxed);
                        coin.scalp.pnl_realized.fetch_add(-entry_fee, Ordering::Relaxed);
                        
                        coin.positions.scalp_position.open(is_long, real_entry_price, qty, margin_required, event_time_ms, 0.0, 0.0);
                        
                        let trail_offset = real_entry_price * scalp_sl;
                        let initial_trail = if is_long { real_entry_price - trail_offset } else { real_entry_price + trail_offset };
                        coin.positions.scalp_position.trail_stop.store(initial_trail, Ordering::Relaxed);
                        
                        new_scalp = Some((is_long, real_entry_price, qty));
                    }
                }
            }
        }
        
        // --- SWING EVALUATION ---
        if is_kline_closed {
            if coin.positions.swing_position.is_open() {
                let is_long = coin.positions.swing_position.is_long.load(Ordering::Relaxed);
                let entry = coin.positions.swing_position.entry_price.load(Ordering::Relaxed);
                let qty = coin.positions.swing_position.quantity.load(Ordering::Relaxed);
                
                // FRICCION REAL:
                let notional = qty * entry;
                let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                let base_price = if is_long { bid } else { ask };
                let (exit_price, _fee) = self.reality.calculate_exit(base_price, is_long, notional, false, tick_volatility);
                let pnl_pct = if is_long { (exit_price - entry) / entry } else { (entry - exit_price) / entry };
                
                let current_atr = self.feature_engines[coin_id].get_atr_pct() * entry;
                let pseudo_atr = if current_atr > 0.0 { current_atr } else { current_price * 0.01 };
                let side_int = if is_long { 1 } else { -1 };
                
                let trail_res = crate::trailing::evaluate_quantum_trailing(
                    side_int, entry, current_price, pseudo_atr,
                    coin.positions.swing_position.trailing_phase.load(Ordering::Relaxed) as i32,
                    coin.positions.swing_position.mfe_atr.load(Ordering::Relaxed),
                    coin.positions.swing_position.max_pnl_pct.load(Ordering::Relaxed),
                    coin.positions.swing_position.trail_stop.load(Ordering::Relaxed),
                    0.8, 1.5, 3.0, 4.0, 1.0
                );
                
                coin.positions.swing_position.trail_stop.store(trail_res.stop_price, Ordering::Relaxed);
                coin.positions.swing_position.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);
                
                let trail_hit = (is_long && current_price <= trail_res.stop_price) || (!is_long && current_price >= trail_res.stop_price && trail_res.stop_price > 0.0);
                
                let notional = qty * entry;
                let mut unrealized = pnl_pct * notional;
                let regime_exit = swing_regime == MarketRegime::Scalping && pnl_pct > 0.0;
                
                // Swing Cooldown/Hold Time Mechanism: Prevent micro-whipsaws
                // Minimum hold time before trailing or regime exits can trigger (e.g., 5 mins / 300_000 ms)
                let entry_time = coin.positions.swing_position.entry_time_ms.load(Ordering::Relaxed);
                let hold_time_met = (event_time_ms >= entry_time) && ((event_time_ms - entry_time) >= 300_000);
                
                if pnl_pct >= swing_tp || pnl_pct <= -swing_sl || (hold_time_met && (trail_hit || trail_res.force_close || regime_exit)) {
                    let gross_pnl = unrealized;
                    coin.swing.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                    let close_fee = if pnl_pct >= scalp_tp { 0.0002 } else { 0.0005 }; 
                    unrealized -= notional * close_fee;
                    let (_, _, _, margin_used) = coin.positions.swing_position.close();
                    let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used { self.arena.swing_used_margin.fetch_add(-margin_used, Ordering::Relaxed); }
                    else { self.arena.swing_used_margin.store(0.0, Ordering::Relaxed); }
                    
                    coin.swing.pnl_realized.fetch_add(unrealized, Ordering::Relaxed);
                    self.arena.unified_capital.fetch_add(unrealized, Ordering::Relaxed);
                    
                    let is_win = unrealized > 0.0;
                    let n = coin.swing.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                    let old_wr = coin.swing.win_rate.load(Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.swing.win_rate.store(new_wr, Ordering::Relaxed);
                    if is_win && pnl_pct.abs() > 0.0 {
                        let old_pf = coin.swing.profit_factor.load(Ordering::Relaxed);
                        let new_pf = old_pf + ((pnl_pct.abs() / old_pf.max(0.001) - old_pf) / n);
                        coin.swing.profit_factor.store(new_pf.max(0.1), Ordering::Relaxed);
                    }
                    let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                    let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                    let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                    let kelly_f = risk_engine::kelly::calculate_kelly_fraction(new_wr, coin.swing.profit_factor.load(Ordering::Relaxed), curr_cap, base_cap, survival_ratio, exp_mult);
                    coin.swing.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                    
                    closed_swing = Some((is_long, unrealized, qty));
                } else {
                    coin.swing.pnl_unrealized.store(unrealized, Ordering::Relaxed);
                }
            }
            
            // FASE 17: Removed hardcoded regime/ATR/Hurst/NN gates.
            // All thresholds are now genome-driven and evolvable.
            if !coin.positions.swing_position.is_open() && (swing_regime == MarketRegime::Swing || swing_regime == MarketRegime::Neutral) {
                let features = self.feature_engines[coin_id].get_features();
                let hurst_exponent = features[1] as f64; // [1] = Hurst
                let atr_pct = self.feature_engines[coin_id].get_atr_pct();
                let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
                let hurst_swing_th = self.arena.config.hurst_swing_threshold.load(Ordering::Relaxed);
                let ml_thresh_long = self.arena.config.ml_threshold_long.load(Ordering::Relaxed);
                let ml_thresh_short = self.arena.config.ml_threshold_short.load(Ordering::Relaxed);
                
                let nn_prob = if let Some(nn) = &mut self.swing_nn { nn.predict(omni_features).unwrap_or(0.5) } else {
                    0.5
                };
                
                if let Some(lh) = &self.lakehouse {
                    let mut feat_vec = Vec::with_capacity(54);
                    for f in omni_features { feat_vec.push(*f as f32); }
                    lh.record_tensor(quantum_arena::symbol_registry::spec(coin_id).symbol, event_time_ms, feat_vec, nn_prob as f32, 0.0);
                }
                
                let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
                let swing_intent = self.swing_engines[coin_id].evaluate_trend(current_price, hurst_exponent, trend_threshold, nn_prob, &self.arena);
                
                self.last_swing_intent[coin_id] = swing_intent;
                let scalp_intent = self.last_scalp_intent[coin_id];
                
                let (_scalp_order, swing_order) = self.risk_engine.evaluate_order(coin_id, scalp_intent, swing_intent, &self.arena);
                
                if swing_order.signal != SignalType::Flat {
                    let is_long = swing_order.signal == SignalType::Long;
                    let cap_split = 1.0 - self.arena.config.capital_split_scalp.load(Ordering::Relaxed);
                    let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                    let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);
                    
                    let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
                    
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
                    let mut margin_required = swing_order.volume_usd / active_swing_opportunities;

                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                    let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                    
                    if current_cap > 0.0 && current_used + margin_required <= current_cap * 0.95 {
                        self.arena.swing_used_margin.fetch_add(margin_required, Ordering::Relaxed);
                        
                        // FRICCION REAL: Entramos cruzando el spread
                        let nominal_size = margin_required * leverage;
                        let tick_volatility = self.feature_engines[coin_id].get_atr_pct();
                        let base_price = if is_long { ask } else { bid };
                        let (real_entry_price, _entry_fee) = self.reality.calculate_market_entry(base_price, is_long, nominal_size, tick_volatility);
                        let qty = nominal_size / real_entry_price;
                        
                        // FRICCION REAL: Aplicamos fee Taker real al abrir (market order)
                        let entry_fee_rate = self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0002);
                        let entry_fee = margin_required * leverage * entry_fee_rate;
                        self.arena.unified_capital.fetch_add(-entry_fee, Ordering::Relaxed);
                        coin.swing.pnl_realized.fetch_add(-entry_fee, Ordering::Relaxed);
                        
                        coin.positions.swing_position.open(is_long, real_entry_price, qty, margin_required, event_time_ms, 0.0, 0.0);
                        
                        let trail_offset = real_entry_price * swing_sl;
                        let initial_trail = if is_long { real_entry_price - trail_offset } else { real_entry_price + trail_offset };
                        coin.positions.swing_position.trail_stop.store(initial_trail, Ordering::Relaxed);
                        
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
            if coin.positions.scalp_position.is_open() && coin.positions.scalp_position.is_long.load(Ordering::Relaxed) == is_long {
                telemetry_server::telemetry_log!("👻 [GHOST REVERT] Binance rechazó orden Scalp para {}. Revirtiendo estado local.", coin_id);
                let (_, _, _, margin) = coin.positions.scalp_position.close();
                self.arena.scalp_used_margin.fetch_add(-margin, Ordering::Relaxed);
            }
        } else {
            if coin.positions.swing_position.is_open() && coin.positions.swing_position.is_long.load(Ordering::Relaxed) == is_long {
                telemetry_server::telemetry_log!("👻 [GHOST REVERT] Binance rechazó orden Swing para {}. Revirtiendo estado local.", coin_id);
                let (_, _, _, margin) = coin.positions.swing_position.close();
                self.arena.swing_used_margin.fetch_add(-margin, Ordering::Relaxed);
            }
        }
    }

    /// Procesa un tick y devuelve las órdenes generadas (si hay).
    /// Retorna: (NuevoScalp, NuevoSwing, CerradoScalp, CerradoSwing, MakerQuote)
    /// Donde cada Option es (is_long, entry/exit_price, qty)
    #[inline(always)]
    pub fn process_tick(&mut self, coin_id: usize, bid: f64, ask: f64, bid_qty: f64, ask_qty: f64, event_time_ms: u64, omni_features: &[f64; 54]) -> (Option<(bool, f64, f64)>, Option<(bool, f64, f64)>, Option<(bool, f64, f64)>, Option<(bool, f64, f64)>, Option<MakerQuote>) {
        telemetry_server::profile_node!("GodEngineCore::process_tick", {
        
        // 1. Quantum Kill-Switch Check
        if self.arena.kill_switch_active.load(Ordering::Relaxed) {
            return (None, None, None, None, None);
        }
        
        // 2. Latency Interlock (Fase 6/7: Cisne Negro / HFT Spoofing)
        // Si el tiempo desde que Binance generó el tick hasta ahora > 10ms
        let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);
        if latency_ms > 3000000 {
            // Entramos en "Modo Pánico": descartar el procesamiento de este tick
            // y no emitir señales para evitar trades atrasados
            if self.arena.tick_counter.load(Ordering::Relaxed).is_multiple_of(1000) {
                telemetry_server::telemetry_log!("🚨 [LATENCY PANIC] Latencia de {}ms detectada! Cortando motor cuántico para evitar spoofing.", latency_ms);
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
        
        let _ofi_value = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
        
        let obi = if total_vol > 0.0 { (bid_qty - ask_qty) / total_vol } else { 0.0 };
        feature_engine.update_macro_features(obi, 0.0, 0.0, 0);
        
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
        
        // CONSEJO DE SAGES: Dynamic SL/TP floor to prevent fee-chop (Floor = 0.15% SL, 0.2% TP)
        let scalp_tp = base_scalp_tp.max(atr_pct * 2.0).max(0.002);
        let scalp_sl = base_scalp_sl.max(atr_pct * 1.5).max(0.0015);
        
        if coin.positions.scalp_position.is_open() {
            let is_long = coin.positions.scalp_position.is_long.load(Ordering::Relaxed);
            let entry = coin.positions.scalp_position.entry_price.load(Ordering::Relaxed);
            let qty = coin.positions.scalp_position.quantity.load(Ordering::Relaxed);
            
            let pnl_pct = if is_long {
                (mid_price - entry) / entry
            } else {
                (entry - mid_price) / entry
            };
            
            let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
            let min_atr_pct = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed).max(0.001);
            let atr_pct_live = raw_atr_pct.max(min_atr_pct);
            let current_atr = atr_pct_live * entry;
            let pseudo_atr = current_atr;
            
            let side_int = if is_long { 1 } else { -1 };
            
            let entry_time = coin.positions.scalp_position.entry_time_ms.load(Ordering::Relaxed);
            let position_age_ms = if event_time_ms > 0 { event_time_ms.saturating_sub(entry_time) } else { 0 };
            
            // Gate trailing: at least 5s and in profit > 0.05%
            let trail_active = position_age_ms > 5_000 && pnl_pct > 0.0005;
            
            let mut trail_hit = false;
            let mut force_close_trail = false;
            
            if trail_active {
                let trail_res = crate::trailing::evaluate_quantum_trailing(
                    side_int, entry, mid_price, pseudo_atr,
                    coin.positions.scalp_position.trailing_phase.load(Ordering::Relaxed) as i32,
                    coin.positions.scalp_position.mfe_atr.load(Ordering::Relaxed),
                    coin.positions.scalp_position.max_pnl_pct.load(Ordering::Relaxed),
                    coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed),
                    1.5, 1.5, 2.0, 2.5, 1.5
                );
                
                // Don't let trailing stop be tighter than the base SL
                let sl_floor = if is_long { entry * (1.0 - scalp_sl) } else { entry * (1.0 + scalp_sl) };
                let safe_stop = if is_long {
                    trail_res.stop_price.max(sl_floor)
                } else {
                    if trail_res.stop_price > 0.0 { trail_res.stop_price.min(sl_floor) } else { sl_floor }
                };
                
                // Only ratchet up (long) / down (short) — never widen
                let current_stored = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                let final_stop = if is_long {
                    if current_stored > 0.0 { safe_stop.max(current_stored) } else { safe_stop }
                } else {
                    if current_stored > 0.0 { safe_stop.min(current_stored) } else { safe_stop }
                };
                
                coin.positions.scalp_position.trail_stop.store(final_stop, Ordering::Relaxed);
                coin.positions.scalp_position.trailing_phase.store(trail_res.new_phase as u8, Ordering::Relaxed);
                coin.positions.scalp_position.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
                coin.positions.scalp_position.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);
                
                trail_hit = (is_long && mid_price <= final_stop) || (!is_long && mid_price >= final_stop && final_stop > 0.0);
                force_close_trail = trail_res.force_close;
            }
            
            let notional = qty * entry;
            let mut unrealized = pnl_pct * notional;
            
            let is_zombie = event_time_ms > 0 && position_age_ms > 1_800_000;
            let hurst_exponent = self.feature_engines[coin_id].get_features()[1] as f64;
            let zombie_close = is_zombie && hurst_exponent < 0.65;
            let zombie_promote = is_zombie && hurst_exponent >= 0.65;
            
            if zombie_promote && !coin.positions.swing_position.is_open() {
                let (_, price, qty, margin_used) = coin.positions.scalp_position.close();
                coin.positions.swing_position.open(is_long, price, qty, margin_used, entry_time, 0.0, 0.0);
                coin.positions.swing_position.trail_stop.store(coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed), Ordering::Relaxed);
            } else if pnl_pct >= scalp_tp || pnl_pct <= -scalp_sl || trail_hit || force_close_trail || zombie_close {
                let sl_price = if is_long { entry * (1.0 - scalp_sl) } else { entry * (1.0 + scalp_sl) };
                let tp_price = if is_long { entry * (1.0 + scalp_tp) } else { entry * (1.0 - scalp_tp) };
                
                let mut exit_price = mid_price;
                if is_long {
                    if pnl_pct <= -scalp_sl { exit_price = exit_price.max(sl_price); }
                    if trail_hit {
                        let ts = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                        if ts > 0.0 { exit_price = exit_price.max(ts); }
                    }
                    if pnl_pct >= scalp_tp { exit_price = tp_price; }
                } else {
                    if pnl_pct <= -scalp_sl { exit_price = exit_price.min(sl_price); }
                    if trail_hit {
                        let ts = coin.positions.scalp_position.trail_stop.load(Ordering::Relaxed);
                        if ts > 0.0 { exit_price = exit_price.min(ts); }
                    }
                    if pnl_pct >= scalp_tp { exit_price = tp_price; }
                }
                
                unrealized = if is_long { (exit_price - entry) * qty } else { (entry - exit_price) * qty };
                let gross_pnl = unrealized;
                coin.scalp.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                
                let _print_pnl_pct = if is_long { (exit_price - entry) / entry } else { (entry - exit_price) / entry };

                let _reason = if trail_hit { "TRAIL_HIT" } else if force_close_trail { "FORCE_CLOSE" } else if zombie_close { "ZOMBIE" } else if pnl_pct >= scalp_tp { "TP" } else { "SL" };
                telemetry_server::telemetry_log!("🛑 CLOSE SCALP [Coin {}]: Long={}, Entry={:.4}, Exit={:.4}, PnL={:.4}% (${:.4}), MFE_ATR={:.2}, Reason={}", coin_id, is_long, entry, exit_price, _print_pnl_pct * 100.0, unrealized, coin.positions.scalp_position.mfe_atr.load(Ordering::Relaxed), _reason);
                
                // Fee: use real Binance rates
                let close_fee = if pnl_pct >= scalp_tp { self.arena.config.live_maker_fee.load(Ordering::Relaxed).max(0.0002) } else { self.arena.config.live_taker_fee.load(Ordering::Relaxed).max(0.0004) }; 
                unrealized -= notional * close_fee; 
                
                let (_, _, _, margin_used) = coin.positions.scalp_position.close();
                let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                if current_used >= margin_used {
                    self.arena.scalp_used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                } else {
                    self.arena.scalp_used_margin.store(0.0, Ordering::Relaxed);
                }
                coin.scalp.pnl_realized.fetch_add(unrealized, Ordering::Relaxed);
                self.arena.unified_capital.fetch_add(unrealized, Ordering::Relaxed);
                
                // Kelly Feedback
                let is_win = unrealized > 0.0;
                let n = coin.scalp.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                let old_wr = coin.scalp.win_rate.load(Ordering::Relaxed);
                let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                coin.scalp.win_rate.store(new_wr, Ordering::Relaxed);
                if is_win && pnl_pct.abs() > 0.0 {
                    let old_pf = coin.scalp.profit_factor.load(Ordering::Relaxed);
                    let new_pf = old_pf + ((pnl_pct.abs() / old_pf.max(0.001) - old_pf) / n);
                    coin.scalp.profit_factor.store(new_pf.max(0.1), Ordering::Relaxed);
                }
                let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                let kelly_f = risk_engine::kelly::calculate_kelly_fraction(new_wr, coin.scalp.profit_factor.load(Ordering::Relaxed), curr_cap, base_cap, survival_ratio, exp_mult);
                coin.scalp.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                
                closed_scalp = Some((is_long, unrealized, qty));
            } else {
                coin.scalp.pnl_unrealized.store(unrealized, Ordering::Relaxed);
            }
        }
        
        if coin.positions.swing_position.is_open() {
            let is_long = coin.positions.swing_position.is_long.load(Ordering::Relaxed);
            let entry = coin.positions.swing_position.entry_price.load(Ordering::Relaxed);
            let qty = coin.positions.swing_position.quantity.load(Ordering::Relaxed);
            
            let pnl_pct = if is_long {
                (mid_price - entry) / entry
            } else {
                (entry - mid_price) / entry
            };
            
            let current_atr = self.feature_engines[coin_id].get_atr_pct() * entry;
            let pseudo_atr = if current_atr > 0.0 { current_atr } else { mid_price * 0.01 };
            
            let side_int = if is_long { 1 } else { -1 };
            
            let trail_res = crate::trailing::evaluate_quantum_trailing(
                side_int, entry, mid_price, pseudo_atr,
                coin.positions.swing_position.trailing_phase.load(Ordering::Relaxed) as i32,
                coin.positions.swing_position.mfe_atr.load(Ordering::Relaxed),
                coin.positions.swing_position.max_pnl_pct.load(Ordering::Relaxed),
                coin.positions.swing_position.trail_stop.load(Ordering::Relaxed),
                0.7, 1.5, 2.0, 3.0, 0.7 // t_params for swing (pullback_tol as ATR scale)
            );
            
            coin.positions.swing_position.trail_stop.store(trail_res.stop_price, Ordering::Relaxed);
            coin.positions.swing_position.trailing_phase.store(trail_res.new_phase as u8, Ordering::Relaxed);
            coin.positions.swing_position.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
            coin.positions.swing_position.max_pnl_pct.store(trail_res.max_pnl_pct, Ordering::Relaxed);
            
            let trail_hit = (is_long && mid_price <= trail_res.stop_price) || (!is_long && mid_price >= trail_res.stop_price && trail_res.stop_price > 0.0);
            
            let notional = qty * entry;
            let mut unrealized = pnl_pct * notional;
            
            // Swing Cooldown/Hold Time Mechanism
            let entry_time = coin.positions.swing_position.entry_time_ms.load(Ordering::Relaxed);
            let hold_time_met = (event_time_ms >= entry_time) && ((event_time_ms - entry_time) >= 300_000);
            
            if pnl_pct >= swing_tp || pnl_pct <= -swing_sl || (hold_time_met && (trail_hit || trail_res.force_close)) {
                let gross_pnl = unrealized;
                coin.swing.pnl_gross.fetch_add(gross_pnl, Ordering::Relaxed);
                // Fee = notional * 0.04% (Binance Maker round-trip)
                let close_fee = if pnl_pct >= scalp_tp { 0.0002 } else { 0.0005 }; 
                unrealized -= notional * close_fee;
                
                let (_, _, _, margin_used) = coin.positions.swing_position.close();
                let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                if current_used >= margin_used {
                    self.arena.swing_used_margin.fetch_add(-margin_used, Ordering::Relaxed);
                } else {
                    self.arena.swing_used_margin.store(0.0, Ordering::Relaxed);
                }
                coin.swing.pnl_realized.fetch_add(unrealized, Ordering::Relaxed);
                self.arena.unified_capital.fetch_add(unrealized, Ordering::Relaxed);
                
                // Kelly Feedback
                let is_win = unrealized > 0.0;
                let n = coin.swing.trade_count.fetch_add(1, Ordering::Relaxed) as f64 + 1.0;
                let old_wr = coin.swing.win_rate.load(Ordering::Relaxed);
                let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                coin.swing.win_rate.store(new_wr, Ordering::Relaxed);
                if is_win && pnl_pct.abs() > 0.0 {
                    let old_pf = coin.swing.profit_factor.load(Ordering::Relaxed);
                    let new_pf = old_pf + ((pnl_pct.abs() / old_pf.max(0.001) - old_pf) / n);
                    coin.swing.profit_factor.store(new_pf.max(0.1), Ordering::Relaxed);
                }
                let curr_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                let base_cap = self.arena.config.base_capital.load(Ordering::Relaxed);
                let survival_ratio = self.arena.config.kelly_survival_cap_ratio.load(Ordering::Relaxed);
                let exp_mult = self.arena.config.kelly_expansion_mult.load(Ordering::Relaxed);
                let kelly_f = risk_engine::kelly::calculate_kelly_fraction(new_wr, coin.swing.profit_factor.load(Ordering::Relaxed), curr_cap, base_cap, survival_ratio, exp_mult);
                coin.swing.kelly_fraction.store(kelly_f, Ordering::Relaxed);
                
                closed_swing = Some((is_long, unrealized, qty));
            } else {
                coin.swing.pnl_unrealized.store(unrealized, Ordering::Relaxed);
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
        
        // 12-Feature array
        let features = self.feature_engines[coin_id].get_features();
        let base_ml_prob = self.scalp_forest.as_ref().map(|f| f.predict(&features).unwrap_or(0.5) as f64).unwrap_or(0.5);
        let ml_prob = (base_ml_prob + spot_bias).clamp(0.0, 1.0);
        self.last_ml_prob = ml_prob as f32;
        
        // CONSEJO DE SAGES: Mutación Cuántica - Scalp = Mean Reversion (Hurst < 0.40), Swing = Momentum
        let mut scalp_intent = SignalIntent::flat();
        let atr_pct = self.feature_engines[coin_id].get_atr_pct();
        let _hurst = features[1] as f64;
        let total_vol = bid_qty + ask_qty;
        let current_obi = if total_vol > 0.0 { (bid_qty - ask_qty) / total_vol } else { 0.0 };
        let ml_threshold = self.arena.config.scalp_obi_threshold.load(Ordering::Relaxed);
        
        let ema_trend = features[0] as f64;
        let ofi = features[2] as f64;
        
        let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
        let dynamic_obi_thr = self.arena.config.dynamic_obi_threshold.load(Ordering::Relaxed);
        let dynamic_ema_thr = self.arena.config.dynamic_ema_trend.load(Ordering::Relaxed);
        let dynamic_ofi_thr = self.arena.config.dynamic_ofi_threshold.load(Ordering::Relaxed);

        // SCALP: Universal Mean Reversion (Fade all Momentum Spikes)
        if atr_pct > dynamic_atr_min { 
            if (ml_prob - 0.5).abs() < 0.001 {
                if current_obi > dynamic_obi_thr && ema_trend > dynamic_ema_thr && ofi > dynamic_ofi_thr { 
                    scalp_intent = SignalIntent { signal: SignalType::Short, confidence: 0.90, ..Default::default() };
                } else if current_obi < -dynamic_obi_thr && ema_trend < -dynamic_ema_thr && ofi < -dynamic_ofi_thr { 
                    scalp_intent = SignalIntent { signal: SignalType::Long, confidence: 0.90, ..Default::default() };
                }
            } else {
                if spot_bias > 0.1 || ml_prob > ml_threshold { 
                    scalp_intent = SignalIntent { signal: SignalType::Long, confidence: ml_prob.max(0.85), ..Default::default() };
                } else if spot_bias < -0.1 || ml_prob < (1.0 - ml_threshold) { 
                    scalp_intent = SignalIntent { signal: SignalType::Short, confidence: (1.0 - ml_prob).max(0.85), ..Default::default() };
                }
            }
        }

        // --- CVD & L2 Wall HARD FILTERS (VETOS) ---
        if scalp_intent.signal != SignalType::Flat {
            let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
            let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
            let cvd = buy_vol - sell_vol;
            let total_vol_cvd = buy_vol + sell_vol;
            let cvd_ratio = if total_vol_cvd > 0.0 { cvd / total_vol_cvd } else { 0.0 };

            let bid_wall = coin.l2_bid_wall.load(Ordering::Relaxed);
            let ask_wall = coin.l2_ask_wall.load(Ordering::Relaxed);
            let total_wall = bid_wall + ask_wall;
            let wall_imbalance = if total_wall > 0.0 { (bid_wall - ask_wall) / total_wall } else { 0.0 };

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
                timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,
                trace_id: tick,
                event_type: 1, // 1 = ML Prediction
                payload,
            });
        }
        
        let hurst_exponent = features[1] as f64; // [1] = Hurst
        let mut swing_nn_pred = 0.5_f64;
        if atr_pct > 0.0005 && hurst_exponent > 0.55 {
            if let Some(nn) = &mut self.swing_nn { 
                swing_nn_pred = nn.predict(omni_features).unwrap_or(0.5); 
            }
        }
        
        let trend_threshold = self.arena.config.trend_threshold.load(Ordering::Relaxed);
        let mut swing_intent = self.swing_engines[coin_id].evaluate_trend(mid_price, hurst_exponent, trend_threshold, swing_nn_pred, &self.arena);
        
        // Unificación Cuántica: Evaluacion Independiente
        let (scalp_order, swing_order) = self.risk_engine.evaluate_order(coin_id, scalp_intent, swing_intent, &self.arena);
        
        if scalp_order.signal != SignalType::Flat || swing_order.signal != SignalType::Flat {
            // Se autoriza la ejecución (escalado dual). 
            // Registramos las posiciones virtuales para backtesting/P&L Tracking.
            if !coin.positions.scalp_position.is_open() && scalp_order.signal != SignalType::Flat {
                let is_long = scalp_order.signal == SignalType::Long;
                // Bolsillo de Capital Scalp (50%)
                let cap_split = self.arena.config.capital_split_scalp.load(Ordering::Relaxed);
                let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
                let leverage = self.arena.config.global_leverage.load(Ordering::Relaxed);
                
                // --- GLOBAL TENSOR SCALP (Tick) ---
                let ml_threshold = self.arena.config.scalp_obi_threshold.load(Ordering::Relaxed);
                let mut active_opportunities: f64 = 0.0;
                for i in 0..30 {
                    let p = self.arena.coins[i].ml_prob.load(Ordering::Relaxed);
                    if p > (0.5 + ml_threshold) || p < (0.5 - ml_threshold) {
                        active_opportunities += 1.0;
                    }
                }
                let active_opportunities = active_opportunities.max(1.0);
                
                let kelly = coin.scalp.kelly_fraction.load(Ordering::Relaxed).clamp(0.1, 1.0);
                let adjusted_kelly = (kelly / active_opportunities).max(0.1).min(1.0);
                
                let mut margin_required = current_cap.max(0.0) * cap_split * adjusted_kelly * scalp_intent.confidence;
                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                let current_used = self.arena.scalp_used_margin.load(Ordering::Relaxed);
                
                // Solo abrimos si tenemos margen disponible (< 95% de utilizacion) y capital positivo
                if current_cap > 0.0 && current_used + margin_required <= current_cap * 0.95 {
                    self.arena.scalp_used_margin.fetch_add(margin_required, Ordering::Relaxed);
                    let qty = (margin_required * leverage) / mid_price;
                    coin.positions.scalp_position.open(is_long, mid_price, qty, margin_required, event_time_ms, 0.0, 0.0);
                    new_scalp = Some((is_long, mid_price, qty));
                }
            }
            
            if !coin.positions.swing_position.is_open() && swing_order.signal != SignalType::Flat {
                let is_long = swing_order.signal == SignalType::Long;
                // Bolsillo de Capital Swing (resto del scalp)
                let cap_split = 1.0 - self.arena.config.capital_split_scalp.load(Ordering::Relaxed);
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
                
                let kelly = coin.swing.kelly_fraction.load(Ordering::Relaxed).clamp(0.1, 1.0);
                let adjusted_kelly = (kelly / active_swing_opportunities).max(0.01).min(1.0);
                
                let mut margin_required = current_cap.max(0.0) * cap_split * adjusted_kelly * swing_intent.confidence;
                    let max_position_size = 10000.0;
                    if margin_required * leverage > max_position_size {
                        margin_required = max_position_size / leverage;
                    }
                let current_used = self.arena.swing_used_margin.load(Ordering::Relaxed);
                
                if current_cap > 0.0 && current_used + margin_required <= current_cap * 0.95 {
                    self.arena.swing_used_margin.fetch_add(margin_required, Ordering::Relaxed);
                    
                    let base_price = if is_long { ask } else { bid };
                    let nominal_size = margin_required * leverage;
                    
                    // Slippage model (Market Impact): 0.05% por cada 1M nominal
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
                    coin.positions.swing_position.open(is_long, real_entry_price, qty, margin_required, event_time_ms, 0.0, 0.0);
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
                let is_long = coin.positions.scalp_position.is_long.load(Ordering::Relaxed);
                let notional = coin.positions.scalp_position.quantity.load(Ordering::Relaxed) * mid_price;
                inventory_delta_usd += if is_long { notional } else { -notional };
            }
            
            let maker_spread_pct = self.arena.config.maker_spread_pct.load(Ordering::Relaxed);
            let maker_obi_threshold = self.arena.config.maker_obi_threshold.load(Ordering::Relaxed);
            
            final_maker_quote = Some(self.maker_engines[coin_id].generate_quote(
                bid, ask, bid_qty, ask_qty, inventory_delta_usd, volatility,
                maker_spread_pct, maker_obi_threshold, 0.0, 0.0
            ));
        }
        
        (new_scalp, new_swing, closed_scalp, closed_swing, final_maker_quote)
        })
    }
}

pub mod stateful_engine;
pub mod ml_inference;
pub mod math_kernels;











