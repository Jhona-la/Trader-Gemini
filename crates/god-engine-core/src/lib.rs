#![feature(portable_simd)]

pub mod bootloader;
pub mod calibration;
pub mod conformal;
pub mod darwin;
pub mod diffusion;
pub mod direction_diag;
pub mod ensemble;
pub mod latency_accelerator;
pub mod math_kernels;
pub mod ml_inference;
pub mod orchestrator;
pub mod order_flow_aggregator;
pub mod quantum_kelly_risk;
pub mod reality_physics;
pub mod reexport_storage {
    pub use storage_engine::mmap_bus;
}
pub mod slippage_predictor;
pub mod stateful_engine;
pub mod trailing;

use crate::stateful_engine::StatefulEngine;
use quantum_arena::GlobalArena;
use risk_engine::RiskEngine;
use signal_engine::{MakerEngine, MakerQuote, SignalIntent, SignalType};
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// Axioma VII: God Engine Core
/// Este componente contiene la lógica dura del ciclo HFT,
/// unificando la Arena con los motores, y eliminando la duplicación
/// entre Backtest y Producción.
pub struct GodEngineCore {
    pub arena: Arc<GlobalArena>,
    pub risk_engine: RiskEngine,
    pub maker_engines: Vec<MakerEngine>,
    pub feature_engines: Vec<StatefulEngine>,
    pub tensor_orchestrator: signal_engine::orchestrator::TensorVoteOrchestrator,
    pub scalp_forest: Option<Arc<crate::ml_inference::NanoForest>>,
    pub swing_nn: Option<dark_alpha_engine::DarkAlphaEngine>,
    /// F4.7: ensamble online — ambos modelos opinan y el peso emerge del
    /// Brier acumulado (Hedge). Antes: cadena if-else donde el segundo
    /// modelo solo opinaba si el primero no existía.
    pub ensemble: crate::ensemble::ModelEnsemble,
    /// D-432: Ensamble bayesiano escopado por símbolo (1 por moneda) para evitar
    /// contaminación cruzada causal entre activos del universo.
    pub ensembles: Vec<crate::ensemble::ModelEnsemble>,
    /// F8 — ESPECTRO TEMPORAL CONTINUO por símbolo: 19 escalas log-espaciadas
    /// (1ms → ~2.18 años) actualizadas en CADA evento. Reemplaza la visión
    /// binaria scalp/swing: el motor observa todas las escalas a la vez, con
    /// fusión por paridad de riesgo (w ∝ 1/vol_de_desviación).
    pub temporal_spectrum: Vec<quantum_arena::temporal_spectrum::TemporalSpectrum>,
    /// F4.7: último precio de kline CERRADO por coin — para calibrar el
    /// ensamble con la dirección realizada de cada vela.
    kline_close_memory: Vec<f64>,
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
    /// X-019: mtime del último sobre de genomas visto — evita disco+parse
    /// en el hot loop cuando no hay evolución nueva.
    genomes_mtime: Option<std::time::SystemTime>,
    /// R4.3 — calibrador conformal real (antes: constante 0.95).
    pub conformal: conformal::ConformalCalibrator,
    /// D-619 (DÉCIMA OLA): mapa aprendido de la puntuación de confianza a la
    /// probabilidad real de acierto (escalado de Platt con prior identidad).
    pub confidence_calibrator: calibration::PlattCalibrator,
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
    /// Diagnóstico por dirección del embudo de entrada (sólo telemetría).
    pub diag_dir: direction_diag::DirectionDiag,
}

impl GodEngineCore {
    pub fn new(arena: Arc<GlobalArena>) -> Self {
        let initial_capital = arena.config.base_capital.load(Ordering::Relaxed);
        // F-012 FIX: Universal model key — the NanoForest serves ALL assets, not just BTC.
        // Search order: UNIVERSAL → legacy BTCUSDT_SCALP fallback for backward compatibility.
        let scalp_forest =
            crate::ml_inference::NanoForest::get_global("UNIVERSAL").or_else(|| {
                // Try universal model file first
                let universal_path = "models/UNIVERSAL_FOREST.json";
                if std::path::Path::new(universal_path).exists() {
                    let _ = crate::ml_inference::NanoForest::load_global("UNIVERSAL", universal_path);
                    return crate::ml_inference::NanoForest::get_global("UNIVERSAL");
                }
                // Fallback: legacy BTCUSDT_SCALP model (backward compatible)
                let legacy_path = "models/BTCUSDT_SCALP.json";
                if std::path::Path::new(legacy_path).exists() {
                    let _ = crate::ml_inference::NanoForest::load_global("UNIVERSAL", legacy_path);
                    crate::ml_inference::NanoForest::get_global("UNIVERSAL")
                } else {
                    None
                }
            });

        // D-605: una entrada por moneda del arena, con su misma capacidad.
        let n_coins = quantum_arena::state::MAX_COINS;
        let mut maker_engines = Vec::with_capacity(n_coins);
        let mut feature_engines = Vec::with_capacity(n_coins);

        for _ in 0..n_coins {
            maker_engines.push(MakerEngine::new(0.0005));
            feature_engines.push(StatefulEngine::new());
        }

        let mut tensor_orchestrator =
            signal_engine::orchestrator::TensorVoteOrchestrator::new(Arc::clone(&arena));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::coaxial_breakout::CoaxialBreakoutEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::game_theoretic_nash::GameTheoreticNashEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::hawkes_bessel::HawkesBesselEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::micro_scalp_trigger::MicroScalpTriggerEngine::default(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::perceptron_gate::PerceptronGateEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::quantum_oscillator::QuantumOscillatorEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::renyi_tsallis_entropy::RenyiTsallisEntropyEngine::default(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::soliton_wave::SolitonWaveEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::stochastic_resonance::StochasticResonanceEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::supersonic_shockwave::SupersonicShockwaveEngine::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::swing_conformal_filter::SwingConformalFilterEngine::default(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::trend_runner::HighPayoffTrendRunner::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::turbo_scalper::TurboScalpEngine::default(),
        ));
        // FIX #781: Registrar JohansenVecmEngine en el orquestador central
        tensor_orchestrator.add_strategy(Box::new(
            strategy_core::vecm_arbitrage::JohansenVecmEngine::default(),
        ));

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

        let ppo_engine = dark_alpha_engine::online_ppo::OnlinePpoPolicyEngine::new([
            0.40, 0.40, 0.20, 0.15, 0.10,
        ]);
        let online_learner =
            metacortex_engine::online_learning::OnlineLearningModule::new(0.001, 0.9);

        Self {
            arena,
            risk_engine: RiskEngine::new(initial_capital),
            maker_engines,
            feature_engines,
            tensor_orchestrator,
            scalp_forest,
            swing_nn,
            ensemble: crate::ensemble::ModelEnsemble::new(),
            ensembles: (0..n_coins)
                .map(|_| crate::ensemble::ModelEnsemble::new())
                .collect(),
            temporal_spectrum: (0..n_coins)
                .map(|_| quantum_arena::temporal_spectrum::TemporalSpectrum::new())
                .collect(),
            kline_close_memory: vec![0.0; n_coins],
            model_rx: None,
            last_ml_prob: 0.5,
            flight_recorder: None,
            reality: reality_physics::RealityPhysics::default(),
            last_scalp_intent: vec![SignalIntent::flat(); n_coins],
            last_swing_intent: vec![SignalIntent::flat(); n_coins],
            last_scalp_senior_signals: vec![[0.0; 10]; n_coins],
            last_swing_senior_signals: vec![[0.0; 10]; n_coins],
            lakehouse: None,
            consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion::new(),
            lead_lag_engine: feature_engine::LeadLagAlphaEngine::new(50),
            ppo_engine,
            online_learner,
            applied_generation: std::sync::atomic::AtomicU64::new(0),
            genomes_mtime: None,
            conformal: conformal::ConformalCalibrator::new(),
            confidence_calibrator: calibration::PlattCalibrator::new(),
            diag_council_vetoes: 0,
            diag_opened: 0,
            diag_swing_vetoes: 0,
            diag_swing_opened: 0,
            diag_close_wins: 0,
            diag_close_total: 0,
            diag_notional_sum: 0.0,
            diag_notional_max: 0.0,
            diag_pnl_sum: 0.0,
            diag_dir: direction_diag::DirectionDiag::default(),
        }
    }

    pub fn get_features(&self, coin_id: usize) -> [f32; 12] {
        self.feature_engines[coin_id].get_features()
    }

    pub fn reset_engines(&mut self) {
        for i in 0..quantum_arena::state::MAX_COINS {
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
            let (_is_long, _entry_price, _qty, margin_used, entry_fee) =
                coin.positions.position.close_with_fee();
            if margin_used > 0.0 {
                let cur = self.arena.used_margin.load(Ordering::Relaxed);
                self.arena
                    .used_margin
                    .store((cur - margin_used).max(0.0), Ordering::Relaxed);
            }
            if entry_fee > 0.0 {
                self.arena
                    .unified_capital
                    .fetch_add(entry_fee, Ordering::Relaxed);
                // D-180: No sumar entry_fee a pnl_realized (nunca fue ganancia)
            }
        }
    }

    /// Rollback local scalp position (alias hacia rollback_position unificado)
    pub fn rollback_scalp_position(&self, coin_id: usize) {
        self.rollback_position(coin_id);
    }

    /// Rollback local swing position (alias hacia rollback_position unificado)
    pub fn rollback_swing_position(&self, coin_id: usize) {
        self.rollback_position(coin_id);
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

    /// Actualiza los modelos y genomas cargados en caliente si hubieron reentrenamientos asíncronos o evolución genética.
    /// X-019 (REHAB-6): cache de mtime — antes CADA llamada (cada 1000 ticks)
    /// hacía fs::read + parse JSON del sobre en el hilo TIME_CRITICAL; ahora
    /// un stat() barato y parse solo si el archivo cambió.
    pub fn refresh_models(&mut self) {
        self.scalp_forest = crate::ml_inference::NanoForest::get_global("UNIVERSAL");
        let mtime_now = std::fs::metadata(quantum_arena::genome_store::active_json_path())
            .and_then(|m| m.modified())
            .ok();
        let changed = match (mtime_now, self.genomes_mtime) {
            (Some(a), Some(b)) => a != b,
            (Some(_), None) => true,
            _ => false,
        };
        if changed {
            self.genomes_mtime = mtime_now;
            if let Some(env) = quantum_arena::genome_store::GenomeEnvelope::load_active() {
                let last = self.applied_generation.load(Ordering::Relaxed);
                if env.generation > last {
                    env.genome.apply_to_arena(&self.arena);
                    self.applied_generation
                        .store(env.generation, Ordering::Relaxed);
                }
                // Generación <= ya aplicada: los hot-swaps en vivo permanecen
                // hasta que el almacén sancione una generación superior.
            }
        }
    }

    /// Procesa un evento unificado continuo (trade, kline, depth) y devuelve las órdenes generadas.
    /// Retorna: (NuevoOrden, CerradoOrden) en arquitectura universal continua.
    #[inline(always)]
    pub fn process_event(
        &mut self,
        coin_id: usize,
        is_trade: bool,
        is_kline_closed: bool,
        is_depth: bool,
        current_price: f64,
        trade_qty: f64,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        depth_obi: f64,
        depth_micro_div: f64,
        event_time_ms: u64,
        latency_panic: bool,
        omni_features: &[f64; 54],
        is_buyer_maker: bool,
    ) -> (Option<(bool, f64, f64, f64, f64)>, Option<(bool, f64, f64)>) {
        telemetry_server::profile_node!("GodEngineCore::process_event", {
            // Fast Invariant Check: Zero-cost anomaly rejection for invalid prices or out-of-bounds coin index
            if current_price <= 0.0
                || !current_price.is_finite()
                || coin_id >= self.arena.coins.len()
            {
                return (None, None);
            }

            // Crossed Orderbook Guard: Reject corrupted L2 depth ticks where bid exceeds ask
            if is_depth && bid > 0.0 && ask > 0.0 && bid > ask {
                return (None, None);
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

            // F8 — ESPECTRO CONTINUO: TODAS las escalas (1ms → ~2.18 años)
            // se actualizan en CADA evento del motor — la observación es
            // continua, no por buckets. O(19) ~ 120 FLOPs. La fusión
            // (paridad de riesgo) y la escala dominante quedan disponibles
            // para señales/telemetría; los caminos scalp/swing siguen
            // operando pero sus parámetros ya provienen de las CURVAS de
            // horizonte del genoma (ver apply_to_arena) — el binario se
            // vuelve una VISTA del continuo, no la fuente.
            if let Some(spec) = self.temporal_spectrum.get_mut(coin_id) {
                spec.update(current_price, event_time_ms);
            }

            // F4.7 — CALIBRACIÓN CONTINUA DEL ENSAMBLE: cada kline CERRADO
            // evalúa las predicciones vivas contra la dirección realizada del
            // bar (y=1 subió / 0 bajó) y actualiza los pesos Hedge por Brier.
            // El flag is_kline_closed estaba IGNORADO desde el origen; ahora
            // es el reloj de calibración del sistema (stream continuo, no la
            // escasez de trades cerrados).
            if is_kline_closed && coin_id < self.kline_close_memory.len() {
                let prev_close = self.kline_close_memory[coin_id];
                if prev_close > 0.0 {
                    let y = if current_price > prev_close { 1.0 } else { 0.0 };
                    if coin_id < self.ensembles.len() {
                        self.ensembles[coin_id].update_with_outcome(y);
                    } else {
                        self.ensemble.update_with_outcome(y);
                    }
                }
                self.kline_close_memory[coin_id] = current_price;
            }

            if self.arena.kill_switch_active.load(Ordering::Relaxed) {
                return (None, None);
            }

            if self
                .arena
                .tick_counter
                .load(Ordering::Relaxed)
                .is_multiple_of(1000)
            {
                self.refresh_models();
            }

            let eff_bid = if bid > 0.0 {
                bid
            } else {
                current_price * 0.9999
            };
            let eff_ask = if ask > 0.0 {
                ask
            } else {
                current_price * 1.0001
            };
            let eff_bid_qty = if bid_qty > 0.0 {
                bid_qty
            } else {
                trade_qty.max(0.01)
            };
            let eff_ask_qty = if ask_qty > 0.0 {
                ask_qty
            } else {
                trade_qty.max(0.01)
            };

            if is_depth {
                self.feature_engines[coin_id].update_macro_features(
                    depth_obi,
                    depth_micro_div,
                    0.0,
                    event_time_ms,
                );
                let mid_price = (eff_bid + eff_ask) / 2.0;
                let raw_atr_pct = self.feature_engines[coin_id].get_atr_pct();
                self.arena.coins[coin_id]
                    .current_price
                    .store(mid_price, Ordering::Relaxed);
                self.arena.coins[coin_id]
                    .current_atr
                    .store(raw_atr_pct * mid_price, Ordering::Relaxed);
                self.arena.update_market_data(
                    coin_id,
                    eff_bid,
                    eff_ask,
                    eff_bid_qty,
                    eff_ask_qty,
                    event_time_ms,
                );

                if coin_id == 0 {
                    // D-404: Conectar actualización en caliente de arena.market_regime basada en microestructura y tendencia de BTC
                    let btc_fe = &self.feature_engines[0];
                    let btc_trend = if btc_fe.ema_slow > 0.0 {
                        (btc_fe.ema_fast - btc_fe.ema_slow) / btc_fe.ema_slow
                    } else {
                        0.0
                    };
                    let btc_hurst = btc_fe.hurst.current();
                    // X-029 (REHAB-1b): cortes de Hurst desde el GENOMA — los
                    // genes existían y estaban huérfanos mientras el código
                    // usaba 0.52/0.42 congelados. Umbrales evolucionables:
                    // tendencia exige H > hurst_trend_threshold; caos/reversión
                    // bajo hurst_scalp_threshold (la banda rápida). El 0.5 de
                    // otras fórmulas es la frontera browniana TEÓRICA — física,
                    // no arbitrariedad — y se queda.
                    let h_trend = self
                        .arena
                        .config
                        .hurst_trend_threshold
                        .load(Ordering::Relaxed)
                        .clamp(0.50, 0.90);
                    let h_chaos = self
                        .arena
                        .config
                        .hurst_scalp_threshold
                        .load(Ordering::Relaxed)
                        .clamp(0.30, h_trend - 0.02);
                    let new_regime = if btc_trend > 0.015 && btc_hurst > h_trend {
                        1u8 // BullRun
                    } else if btc_trend < -0.015 && btc_hurst > h_trend {
                        2u8 // Crash
                    } else if btc_hurst < h_chaos {
                        3u8 // Chaotic / Mean Reverting
                    } else {
                        0u8 // Range
                    };
                    self.arena
                        .market_regime
                        .store(new_regime, Ordering::Relaxed);
                }
            }

            if is_trade {
                // D-220 & D-247: Ingesta física real de microestructura agresora (Taker Buy vs Taker Sell)
                self.feature_engines[coin_id].update_trade_flow(trade_qty, is_buyer_maker);
                self.arena.coins[coin_id]
                    .current_price
                    .store(current_price, Ordering::Relaxed);
            }

            let (new_order, closed_order, _maker) = self.process_tick_dual(
                coin_id,
                eff_bid,
                eff_ask,
                eff_bid_qty,
                eff_ask_qty,
                event_time_ms,
                omni_features,
            );

            // Si hay pánico de latencia, no abrimos nuevas órdenes pero permitimos cierres defensivos
            if latency_panic {
                (None, closed_order)
            } else {
                (new_order, closed_order)
            }
        })
    }

    /// Revierte una posición fantasma (Ghost Position) cuando la API de Binance la rechaza.
    pub fn revert_ghost_position(&mut self, coin_id: usize, _is_scalp: bool, _is_long: bool) {
        self.revert_quantum_ghost_position(coin_id);
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
            if margin > 0.0 {
                let cur = self.arena.used_margin.load(Ordering::Relaxed);
                self.arena
                    .used_margin
                    .store((cur - margin).max(0.0), Ordering::Relaxed);
            }
        }
    }

    /// Procesa un tick en el motor universal continuo unificado.
    /// Retorna: (new_order, closed_order, maker_quote)
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
        Option<(bool, f64, f64, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<MakerQuote>,
    ) {
        telemetry_server::profile_node!("GodEngineCore::process_tick_dual", {
            // X-016 (REHAB-1): el espectro se actualiza TAMBIÉN aquí — los
            // callers directos de dual (backtests) congelaban el espectro al
            // no pasar por process_event. Idempotente para producción (process_
            // event ya actualizó con el mismo tick: dt=0 ⇒ α=0, sin doble peso).
            if let Some(spec) = self.temporal_spectrum.get_mut(coin_id) {
                let mid = (bid + ask) * 0.5;
                spec.update(mid, event_time_ms);
            }

            // 1. Quantum Kill-Switch Check
            if self.arena.kill_switch_active.load(Ordering::Relaxed) {
                return (None, None, None);
            }

            // 2. Latency Interlock
            let latency_threshold_ms = self
                .arena
                .config
                .latency_ms_panic_threshold
                .load(Ordering::Relaxed)
                .max(50.0);
            let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);
            // X-012 (REHAB-4): el interlock NUNCA bloquea salidas. Antes este
            // `return` saltaba la gestión de posición completa (SL/trailing/
            // zombie/toxic) justo durante picos de latencia — exactamente
            // cuando las salidas defensivas son vitales. Ahora: bandera; la
            // gestión corre; solo la GENERACIÓN de entradas se bloquea (abajo,
            // frontera de EVALUAR ENTRADAS).
            // X-010: stalled (watchdog del WS sin datos 5s) también bloquea
            // entradas — la muerte silenciosa del feed ya no es invisible.
            let entries_blocked = quantum_arena::feed_health::is_stalled()
                || latency_ms > latency_threshold_ms as u64;
            if entries_blocked
                && self
                    .arena
                    .tick_counter
                    .load(Ordering::Relaxed)
                    .is_multiple_of(1000)
            {
                telemetry_server::telemetry_log!(
                    "🚨 [LATENCY PANIC] Latencia {}ms > umbral {:.0}ms! ENTRADAS bloqueadas — salidas defensivas siguen activas.",
                    latency_ms,
                    latency_threshold_ms
                );
            }

            self.arena.increment_tick();

            let mid_price = (bid + ask) / 2.0;
            let total_vol = bid_qty + ask_qty;

            // D-119: cvpin solo debe recibir volumen normalizado o trades reales, no millones de USD de profundidad L2
            let tick_vol = if total_vol > 0.0 {
                (total_vol * 0.005).clamp(0.01, 10.0)
            } else {
                0.01
            };
            let feature_engine = &mut self.feature_engines[coin_id];
            feature_engine.process_tick(mid_price, tick_vol, event_time_ms);
            // D-220 & D-247: Depth snapshots must NOT corrupt OrderFlow with synthetic trades. Real trades update order flow via process_event when is_trade=true.

            let ofi_value = feature_engine.update_ofi(bid, ask, bid_qty, ask_qty);
            let sym = quantum_arena::symbol_registry::try_spec(coin_id)
                .map(|s| s.symbol)
                .unwrap_or_default();
            if sym.starts_with("BTC") {
                self.lead_lag_engine.update_leader(true, ofi_value);
            } else if sym.starts_with("ETH") {
                self.lead_lag_engine.update_leader(false, ofi_value);
            }
            let (leader_mom, lead_lag_div) =
                self.lead_lag_engine.predict_altcoin_impulse(ofi_value);

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
            coin.current_atr
                .store(raw_atr_pct * mid_price, Ordering::Relaxed);
            coin.hurst_exponent.store(hurst_val, Ordering::Relaxed);

            let mut closed_order = None;

            let atr_pct = self.feature_engines[coin_id].get_atr_pct();
            let min_atr_pct = self
                .arena
                .config
                .dynamic_atr_min
                .load(Ordering::Relaxed)
                .max(0.001);
            let atr_pct_live = atr_pct.max(min_atr_pct);

            // --- 1. GESTIÓN DE POSICIÓN CONTINUA UNIFICADA (TRAILING STOPS, HIGH-FREQUENCY COMPOUNDING) ---
            if coin.positions.position.is_open() {
                let pos = &coin.positions.position;
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

                let (sl, tp) = {
                    let pos_tp = pos.tp_price.load(Ordering::Relaxed);
                    let pos_sl = pos.sl_price.load(Ordering::Relaxed);
                    // FASE 23: Espectro Continuo Universal — evaluación continua a tau dominante
                    let tau_entry = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
                    let tau = if tau_entry > 0.0 {
                        tau_entry
                    } else {
                        let l_fast = quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS.ln();
                        let l_slow = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS.ln();
                        let s = self.arena.config.temporal_scale.load(Ordering::Relaxed).clamp(0.05, 0.95);
                        (l_fast + s * (l_slow - l_fast)).exp()
                    };
                    let sl_base = self.arena.config.sl_at_tau(tau).clamp(0.0005, 0.0500);
                    let tp_base = self.arena.config.tp_at_tau(tau).clamp(0.0010, 0.1000);
                    let fallback_sl = sl_base.max(atr_pct * 1.5).clamp(0.0010, 0.0300);
                    let rr_ratio = self
                        .arena
                        .config
                        .tp_rr_ratio_btc
                        .load(Ordering::Relaxed)
                        .clamp(1.0, 10.0);
                    let fallback_tp = tp_base.max(fallback_sl * rr_ratio).clamp(0.0020, 0.0800);
                    if pos_tp > 0.0 && pos_sl > 0.0 {
                        if entry > 0.0 {
                            (
                                ((entry - pos_sl) / entry).abs(),
                                ((pos_tp - entry) / entry).abs(),
                            )
                        } else {
                            (fallback_sl, fallback_tp)
                        }
                    } else {
                        (fallback_sl, fallback_tp)
                    }
                };

                // D-464: Arquitectura Integrada de Breakeven & Trailing Ratchet
                let current_max_pnl = pos.max_pnl_pct.load(Ordering::Relaxed);
                if pnl_pct > current_max_pnl {
                    pos.max_pnl_pct.store(pnl_pct, Ordering::Relaxed);
                }
                let peak_pnl = pos.max_pnl_pct.load(Ordering::Relaxed);

                let live_fee = self.arena.config.live_maker_fee.load(Ordering::Relaxed)
                    + self.arena.config.live_taker_fee.load(Ordering::Relaxed);

                // D-465, D-472, D-474, D-475 & D-495: Escudo Breakeven Progresivo Calibrado Antiasfixia.
                // Activa cuando el trade ha alcanzado al menos 2.0 ATR o el 55% de su TP objetivo (mínimo 62 bps).
                // En cuanto el trade demuestra inercia direccional probada, el stop se ajusta a Entry + buffer (+10 a +18 bps post-fees).
                let be_activation = (tp * 0.55).max(atr_pct_live * 2.0).clamp(0.0062, 0.0160);
                if peak_pnl >= be_activation {
                    let be_buffer = (live_fee * 2.0).clamp(0.0010, 0.0018);
                    let be_stop = if is_long {
                        entry * (1.0 + be_buffer)
                    } else {
                        entry * (1.0 - be_buffer)
                    };
                    let cur_stop = pos.trail_stop.load(Ordering::Relaxed);
                    if is_long {
                        if cur_stop == 0.0 || be_stop > cur_stop {
                            pos.trail_stop.store(be_stop, Ordering::Relaxed);
                        }
                    } else {
                        if cur_stop == 0.0 || be_stop < cur_stop {
                            pos.trail_stop.store(be_stop, Ordering::Relaxed);
                        }
                    }
                }

                // 2. Trailing Stop Ratchet Dinámico: activa cuando el pico alcanza >= 70% de TP (o mínimo 72 bps)
                let trail_activation_pnl =
                    (tp * 0.70).max(be_activation * 1.25).clamp(0.0072, 0.0200);
                let trail_active = peak_pnl >= trail_activation_pnl;

                let mut force_close_trail = false;

                if trail_active {
                    let tau_entry = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
                    let tau = if tau_entry > 0.0 {
                        tau_entry
                    } else {
                        let l_fast = quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS.ln();
                        let l_slow = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS.ln();
                        let s = self.arena.config.temporal_scale.load(Ordering::Relaxed).clamp(0.05, 0.95);
                        (l_fast + s * (l_slow - l_fast)).exp()
                    };
                    let (trail_atr_mult, trail_act, trail_step, trail_max) =
                        self.arena.config.trail_params_at_tau(tau);

                    let trail_res = crate::trailing::evaluate_quantum_trailing_with_fee(
                        side_int,
                        entry,
                        mid_price,
                        pseudo_atr,
                        pos.trailing_phase.load(Ordering::Relaxed) as i32,
                        pos.mfe_atr.load(Ordering::Relaxed),
                        peak_pnl,
                        pos.trail_stop.load(Ordering::Relaxed),
                        trail_atr_mult,
                        trail_act,
                        trail_step,
                        trail_max,
                        trail_atr_mult,
                        live_fee,
                    );

                    let sl_floor = if is_long {
                        entry * (1.0 - sl)
                    } else {
                        entry * (1.0 + sl)
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
                    pos.trailing_phase
                        .store(trail_res.new_phase as u8, Ordering::Relaxed);
                    pos.mfe_atr.store(trail_res.mfe_atr, Ordering::Relaxed);
                    force_close_trail = trail_res.force_close;
                }

                // 3. Verificación Universal de Stop Ejecutado (Breakeven / Trailing)
                let active_trail_stop = pos.trail_stop.load(Ordering::Relaxed);
                let trail_hit = if active_trail_stop > 0.0 {
                    (is_long && mid_price <= active_trail_stop)
                        || (!is_long && mid_price >= active_trail_stop)
                } else {
                    false
                };

                let tau_exit = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
                let temporal_s = if tau_exit > 0.0 {
                    let l_fast = quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS.ln();
                    let l_slow = quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS.ln();
                    ((tau_exit.ln() - l_fast) / (l_slow - l_fast)).clamp(0.0, 1.0)
                } else {
                    self.arena
                        .config
                        .temporal_scale
                        .load(Ordering::Relaxed)
                        .clamp(0.0, 1.0)
                };

                let macro_t = self.feature_engines[coin_id].get_macro_trend();
                // D-455: Inversión de tendencia genuina (45 bps de pendiente EMA) en lugar de ruido browniano
                let trend_reversed =
                    (is_long && macro_t < -0.0045) || (!is_long && macro_t > 0.0045);

                // D-649b: el gen entra acotado a [4 h, 8 h] (ver
                // SuperGenotype::SLOT_ZOMBIE_TIMEOUT). En su suelo, debounce y
                // hard-timeout reproducen exactamente las fórmulas anteriores.
                // D-649 (DÉCIMA OLA) — EL GEN `zombie_timeout_ms` ENTRA EN SERVICIO.
                //
                // El gen existía, se escribía en el arena y NINGÚN consumidor lo
                // leía: la caducidad de posiciones se regía por dos literales
                // (43_200_000 y 14_400_000). Peor: toda la lógica de zombi estaba
                // condicionada a `pnl_pct <= -0.0050` o a inversión de tendencia,
                // de modo que **una posición plana y antigua no expiraba nunca**.
                // Eso dejaba al sistema sin red de seguridad frente a posiciones
                // huérfanas tras una desconexión.
                //
                // Ahora el gen define la escala temporal de caducidad y existe un
                // TECHO ABSOLUTO independiente del PnL.
                let zombie_gene_ms = self
                    .arena
                    .config
                    .zombie_timeout_ms
                    .load(Ordering::Relaxed);
                let zombie_base_ms = if zombie_gene_ms.is_finite() && zombie_gene_ms > 0.0 {
                    zombie_gene_ms
                } else {
                    3_600_000.0
                };
                // El horizonte continuo dilata la caducidad: una tesis larga
                // necesita más tiempo que una corta. Factor continuo en s.
                let horizon_dilation = 1.0 + temporal_s;
                let dynamic_zombie_debounce_ms =
                    (zombie_base_ms * horizon_dilation) as u64;
                let dynamic_hard_timeout_ms =
                    (zombie_base_ms * 3.0 * horizon_dilation) as u64;
                // TECHO ABSOLUTO: ninguna posición sobrevive más de 12x la escala
                // genómica, gane, pierda o esté plana. Es la red de seguridad que
                // faltaba — la única defensa contra una posición que quedó viva
                // por una desconexión, un fill perdido o un estado corrupto.
                let absolute_expiry_ms = (zombie_base_ms * 12.0 * horizon_dilation) as u64;
                let expired_by_age =
                    event_time_ms > 0 && position_age_ms > absolute_expiry_ms;
                let hard_timeout = position_age_ms > dynamic_hard_timeout_ms && pnl_pct <= -0.0050;
                let is_zombie = expired_by_age
                    || (event_time_ms > 0
                        && position_age_ms > dynamic_zombie_debounce_ms
                        && ((trend_reversed && pnl_pct <= -0.0060) || hard_timeout)); // D-649b: umbral original

                // D-492: Dynamic Adverse Order Flow Stop Cutting (Toxic Flow Cutoff)
                // Se activa únicamente cuando el trade ha consumido la gran mayoría de su stop loss continuo
                // (pnl_pct <= -0.85 * sl) y el flujo L2 y micro-tendencia confirman toxicidad adversa terminal,
                // salvando el 15% restante del SL sin asfixiar trades en retrocesos normales de mercado.
                let micro_t = self.feature_engines[coin_id].get_micro_trend();
                let ema_ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;
                let cur_vpin = self.feature_engines[coin_id].cvpin.current_vpin();

                let ofi_adverse = (is_long && (ofi_value < -0.30 || ema_ofi < -0.25))
                    || (!is_long && (ofi_value > 0.30 || ema_ofi > 0.25));
                let trend_adverse = (is_long && micro_t < -0.00060)
                    || (!is_long && micro_t > 0.00060);

                let toxic_cut_sl = (sl * 0.85).max(0.0065);
                let toxic_flow_exit = (pnl_pct <= -toxic_cut_sl && ofi_adverse && trend_adverse)
                    || (pnl_pct <= -toxic_cut_sl && cur_vpin > 0.65 && ofi_adverse);

                let tp_traded_through = if is_long {
                    bid >= entry * (1.0 + tp)
                } else {
                    ask <= entry * (1.0 - tp)
                };

                if tp_traded_through
                    || pnl_pct <= -sl
                    || trail_hit
                    || force_close_trail
                    || is_zombie
                    || toxic_flow_exit
                {
                    let sl_price = if is_long {
                        entry * (1.0 - sl)
                    } else {
                        entry * (1.0 + sl)
                    };
                    let tp_price = if is_long {
                        entry * (1.0 + tp)
                    } else {
                        entry * (1.0 - tp)
                    };

                    let mut exit_price = mid_price;
                    if is_long {
                        if pnl_pct <= -sl {
                            exit_price = exit_price.min(sl_price);
                        }
                        if trail_hit {
                            // D-120: Erradicar lookahead bias: el precio de salida es el precio de mercado real
                            exit_price = mid_price;
                        }
                        if pnl_pct >= tp {
                            exit_price = tp_price;
                        }
                    } else {
                        if pnl_pct <= -sl {
                            exit_price = exit_price.max(sl_price);
                        }
                        if trail_hit {
                            // D-120: Erradicar lookahead bias: el precio de salida es el precio de mercado real
                            exit_price = mid_price;
                        }
                        if pnl_pct >= tp {
                            exit_price = tp_price;
                        }
                    }

                    // N-02 — FÍSICA DE SALIDA CONECTADA: calculate_exit
                    // modela maker (precio límite exacto) vs taker (cruce
                    // adverso del book + latency-slippage + floor del genoma).
                    // El TP con trade-through califica como maker; el resto
                    // (SL, trailing, zombie, timeout) son taker. Antes: TODO
                    // exit llenaba a precio perfecto — asimetría que inflaba
                    // el edge de scalps con TP estrecho.
                    let exit_is_maker = pnl_pct >= tp;
                    let exit_nominal = qty * exit_price;
                    let (phys_exit_price, phys_exit_fee) = self.reality.calculate_exit(
                        exit_price,
                        is_long,
                        exit_nominal,
                        exit_is_maker,
                        atr_pct, // fracción (N-01)
                        self.arena
                            .config
                            .base_slippage_floor
                            .load(Ordering::Relaxed)
                            .max(0.00001),
                        self.arena
                            .config
                            .latency_penalty_ms
                            .load(Ordering::Relaxed)
                            .max(0.0),
                    );
                    if phys_exit_price > 0.0 {
                        exit_price = phys_exit_price;
                    }

                    let gross_pnl = if is_long {
                        (exit_price - entry) * qty
                    } else {
                        (entry - exit_price) * qty
                    };

                    let (reason_code, reason) = if tp_traded_through {
                        (1u8, "TP")
                    } else if pnl_pct <= -sl {
                        (2u8, "SL")
                    } else if trail_hit {
                        (3u8, "TRAIL_HIT")
                    } else if force_close_trail {
                        (4u8, "FORCE_TRAIL")
                    } else if is_zombie {
                        (5u8, "ZOMBIE")
                    } else {
                        (6u8, "TOXIC_FLOW")
                    };

                    if self.diag_close_total < 100 {
                        println!(
                            "🚪 [CLOSE TRACE] #{} reason={} pnl_pct={:.4}% gross_pnl=${:.4} exit={:.2} entry={:.2} h={:?} ts={}",
                            self.diag_close_total,
                            reason,
                            pnl_pct * 100.0,
                            gross_pnl,
                            exit_price,
                            entry,
                            pos.horizon(),
                            event_time_ms
                        );
                    }

                    // H-4: fee de la FÍSICA (calculate_exit ya diferenció
                    // maker/taker según exit_is_maker) — las variables
                    // live_taker/live_maker quedaron muertas tras la
                    // consolidación y se eliminan.
                    let close_fee = phys_exit_fee;

                    let ml_at_entry = pos.ml_prediction.load(Ordering::Relaxed);
                    // D-619: puntuación cruda con la que se abrió (se lee antes de
                    // que `close_with_fee` limpie la posición).
                    let score_at_entry = pos.confidence.load(Ordering::Relaxed);
                    let (_, _, _, margin_used, entry_fee_paid) = pos.close_with_fee();

                    let net_realized_pnl = gross_pnl - close_fee;
                    let net_trade_pnl = net_realized_pnl - entry_fee_paid;

                    let current_used = self.arena.used_margin.load(Ordering::Relaxed);
                    if current_used >= margin_used {
                        self.arena
                            .used_margin
                            .fetch_add(-margin_used, Ordering::Relaxed);
                    } else {
                        self.arena.used_margin.store(0.0, Ordering::Relaxed);
                    }

                    // FASE 23: Métricas continuas unificadas — sin bifurcaciones scalp/swing
                    coin.metrics
                        .pnl_realized
                        .fetch_add(net_trade_pnl, Ordering::Relaxed);
                    coin.scalp
                        .pnl_realized
                        .fetch_add(net_trade_pnl, Ordering::Relaxed);
                    coin.swing
                        .pnl_realized
                        .fetch_add(net_trade_pnl, Ordering::Relaxed);

                    self.feature_engines[coin_id].last_scalp_exit_tick =
                        self.feature_engines[coin_id].tick_count;
                    let was_loss = net_trade_pnl <= 0.0;
                    self.feature_engines[coin_id].last_scalp_was_loss = was_loss;
                    if was_loss {
                        self.feature_engines[coin_id].scalp_loss_streak += 1;
                        if is_long {
                            self.feature_engines[coin_id].scalp_long_loss_streak += 1;
                        } else {
                            self.feature_engines[coin_id].scalp_short_loss_streak += 1;
                        }
                    } else {
                        self.feature_engines[coin_id].scalp_loss_streak = 0;
                        if is_long {
                            self.feature_engines[coin_id].scalp_long_loss_streak = 0;
                        } else {
                            self.feature_engines[coin_id].scalp_short_loss_streak = 0;
                        }
                    }
                    self.arena
                        .unified_capital
                        .fetch_add(net_realized_pnl, Ordering::Relaxed);

                    let is_win = net_trade_pnl > 0.0;

                    // MMAP TELEMETRY BUS — PRODUCTOR CONECTADO (el feedback
                    // de autoevolución estaba muerto: el daemon leía un bus
                    // que nadie escribía). Cada cierre emite predicción-vs-
                    // realidad para el Shadow Forest del online_daemon.
                    {
                        let cap_pct = if entry > 0.0 {
                            net_trade_pnl / (entry * qty).max(1e-8)
                        } else {
                            0.0
                        };
                        storage_engine::mmap_bus::write_prediction_vs_reality(
                            ml_at_entry,
                            is_long,
                            cap_pct,
                            atr_pct,
                        );
                    }
                    self.diag_close_wins += is_win as u64;
                    self.diag_close_total += 1;
                    self.diag_notional_sum += qty * exit_price;
                    self.diag_notional_max = self.diag_notional_max.max(qty * exit_price);
                    self.diag_pnl_sum += net_trade_pnl;

                    if ml_at_entry > 0.0 {
                        // D-676 (DÉCIMA OLA): la posición guarda `ml_prob` crudo, la
                        // probabilidad de que el precio SUBA. La convención del
                        // calibrador es la probabilidad de que la operación gane EN
                        // SU DIRECCIÓN: para un corto es la complementaria. Antes los
                        // cortos entraban invertidos y contaminaban la calibración
                        // de todas las operaciones.
                        let p_win_at_entry = if is_long { ml_at_entry } else { 1.0 - ml_at_entry };
                        self.conformal.update(p_win_at_entry, is_win);
                    }
                    // D-619: el calibrador aprende de la puntuación CRUDA y del
                    // resultado neto de comisiones. Nunca de su propia salida.
                    if score_at_entry > 0.0 {
                        self.confidence_calibrator.update(score_at_entry, is_win);
                    }

                    // D-680 (DÉCIMA OLA): media posterior con prior de diseño. Antes
                    // `n` empezaba en 1 y la primera operación sobrescribía el
                    // prior entero: tras una pérdida inicial el win rate valía 0.
                    let n_prev = coin.metrics.trade_count.fetch_add(1, Ordering::Relaxed) as f64;
                    let old_wr = coin.metrics.win_rate.load(Ordering::Relaxed);
                    let new_wr = quantum_arena::genome::SuperGenotype::posterior_win_rate(
                        old_wr, n_prev, is_win,
                    );
                    coin.metrics.win_rate.store(new_wr, Ordering::Relaxed);

                    // F-014 FIX: Removed bifurcated writes to coin.scalp/coin.swing.
                    // All metric updates flow exclusively through coin.metrics (unified source of truth).

                    if is_win {
                        coin.metrics
                            .gross_wins
                            .fetch_add(net_trade_pnl, Ordering::Relaxed);
                    } else {
                        coin.metrics
                            .gross_losses
                            .fetch_add(net_trade_pnl.abs(), Ordering::Relaxed);
                    }
                    let total_wins = coin.metrics.gross_wins.load(Ordering::Relaxed);
                    let total_losses = coin.metrics.gross_losses.load(Ordering::Relaxed);
                    let new_pf = if total_losses > 0.0 {
                        (total_wins / total_losses).clamp(0.1, 10.0)
                    } else if total_wins > 0.0 {
                        5.0
                    } else {
                        1.50
                    };
                    coin.metrics.profit_factor.store(new_pf, Ordering::Relaxed);
                    // F-014 FIX: Removed bifurcated profit_factor writes to coin.scalp/coin.swing.

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
                    let clamp_min = self.arena.config.kelly_clamp_min.load(Ordering::Relaxed);
                    let clamp_max = self.arena.config.kelly_clamp_max.load(Ordering::Relaxed);
                    let tau_pos = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
                    let strategy_base = self
                        .arena
                        .config
                        .kelly_at_tau(if tau_pos > 0.0 { tau_pos } else { 30_000.0 });
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
                    coin.metrics
                        .kelly_fraction
                        .store(kelly_f, Ordering::Relaxed);
                    coin.last_close_ts.store(event_time_ms, Ordering::Relaxed);
                    coin.last_close_is_long.store(is_long, Ordering::Relaxed);
                    coin.last_close_was_win.store(is_win, Ordering::Relaxed);
                    coin.last_close_reason.store(reason_code, Ordering::Relaxed);
                    coin.last_scalp_close_ts.store(event_time_ms, Ordering::Relaxed);
                    coin.last_swing_close_ts.store(event_time_ms, Ordering::Relaxed);

                    // D-181: closed_order debe reflejar el PnL neto definitivo deduciendo ambas comisiones (entry + close)
                    closed_order = Some((is_long, net_trade_pnl, qty));

                    let notional = (qty * entry).max(1.0);
                    let realized_ret = net_trade_pnl / notional;
                    if coin_id < self.last_scalp_senior_signals.len() {
                        self.consejo_deliberacion.record_outcome(
                            &self.last_scalp_senior_signals[coin_id],
                            realized_ret,
                        );
                    }
                    if coin_id < self.last_swing_senior_signals.len() {
                        self.consejo_deliberacion.record_outcome(
                            &self.last_swing_senior_signals[coin_id],
                            realized_ret,
                        );
                    }

                    // D-190: Evitar contaminación cruzada en Hebbian. Escopar por símbolo con fallback global.
                    let hebb_key = format!("{}_hebbian_weight", sym);
                    let mut cur_hebbian = self
                        .arena
                        .registry
                        .get(&hebb_key, "GodEngineCore")
                        .or_else(|| {
                            self.arena
                                .registry
                                .get("perceptron_hebbian_weight", "GodEngineCore")
                        })
                        .map(|p| p.get_value())
                        .unwrap_or(1.0);
                    signal_engine::perceptron_gate::PerceptronGateEngine::update_weight(
                        &mut cur_hebbian,
                        net_trade_pnl,
                        atr_pct,
                    );
                    self.arena.registry.set(&hebb_key, cur_hebbian);
                    self.arena
                        .registry
                        .set("perceptron_hebbian_weight", cur_hebbian);

                    let hawkes_r = self.feature_engines[coin_id].cvpin.current_vpin();
                    let close_dir_sign = if obi.abs() > 0.05 {
                        obi.signum()
                    } else if ofi_value.abs() > 0.05 {
                        ofi_value.signum()
                    } else if is_long {
                        1.0
                    } else {
                        -1.0
                    };
                    let dir_macro = self.feature_engines[coin_id].get_macro_trend();
                    let ppo_close_features = [
                        (ofi_value / 0.35).clamp(-1.5, 1.5),
                        (obi / 0.35).clamp(-1.5, 1.5),
                        hawkes_r.clamp(0.0, 1.0) * close_dir_sign,
                        lead_lag_div.clamp(-1.5, 1.5),
                        ((hurst_val - 0.50) * 2.0).clamp(-1.0, 1.0)
                            * (if dir_macro.abs() > 0.0005 {
                                dir_macro.signum()
                            } else {
                                close_dir_sign
                            }),
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
                    coin.metrics
                        .pnl_unrealized
                        .store(unrealized, Ordering::Relaxed);
                }
            }

            // --- 2. EVALUAR ENTRADAS ---
            // X-012: frontera exacta del bloqueo por latencia — la gestión de
            // posiciones (sección 1, arriba) corrió COMPLETA antes de llegar
            // aquí. Con datos obsoletos no se abren posiciones nuevas; las
            // abiertas ya fueron gestionadas con este mismo tick.
            if entries_blocked {
                return (None, None, None);
            }
            let tick = self.arena.tick_counter.load(Ordering::Relaxed);

            // --- Inteligencia On-Chain (Spot vs Futures Correlation) ---
            let spot_bid = coin.spot_bid.load(Ordering::Relaxed);
            let spot_ask = coin.spot_ask.load(Ordering::Relaxed);
            let mut spot_bias = 0.0;

            if spot_bid > 0.0 && spot_ask > 0.0 {
                let spot_mid = (spot_bid + spot_ask) / 2.0;
                let spread_bps = ((spot_mid - mid_price) / mid_price) * 10000.0;
                if spread_bps > 1.5 {
                    spot_bias = 0.15;
                } else if spread_bps < -1.5 {
                    spot_bias = -0.15;
                }
            }

            let hurst_val = coin.hurst_exponent.load(Ordering::Relaxed);
            let obi_val = if bid_qty + ask_qty > 0.0 {
                (bid_qty - ask_qty) / (bid_qty + ask_qty)
            } else {
                0.0
            };
            let flow_dir = if bid_qty > ask_qty {
                1.0
            } else if ask_qty > bid_qty {
                -1.0
            } else {
                0.0
            };
            let v_t = self.feature_engines[coin_id].v_t;
            let shannon_ent = self.feature_engines[coin_id].entropy.current();
            let vpin_val = self.feature_engines[coin_id].cvpin.current_vpin();
            let total_vol = (bid_qty + ask_qty).max(1e-8);
            let p_bid = (bid_qty / total_vol).clamp(0.0001, 0.9999);
            let p_ask = (ask_qty / total_vol).clamp(0.0001, 0.9999);
            let tsallis_ent = ((1.0 - (p_bid.powf(1.5) + p_ask.powf(1.5))) / 0.5).clamp(0.0, 1.0);

            let micro_v = (v_t.abs() / mid_price.max(1e-8)).clamp(atr_pct * 0.1, atr_pct * 5.0);
            let set_reg = |key: &str, val: f64| {
                self.arena.registry.set(key, val);
                self.arena.registry.set_for_coin(coin_id, key, val);
                if !sym.is_empty() {
                    self.arena.registry.set_scoped(&sym, key, val);
                }
            };
            let dir_v = self.feature_engines[coin_id].dir_velocity;
            let a_t = self.feature_engines[coin_id].a_t;
            let spread_pct = if mid_price > 0.0 {
                (ask - bid) / mid_price
            } else {
                0.0002
            };
            let spread_val = (spread_pct * mid_price).max(0.0001);
            let speed_of_sound = spread_val.max(micro_v * mid_price);
            set_reg("spread_speed_of_sound", speed_of_sound);
            set_reg("price_velocity", dir_v);
            set_reg("order_flow_velocity", dir_v);
            set_reg("price_acceleration", a_t);
            set_reg("atr_1s", (micro_v * mid_price).max(0.0001));
            set_reg(
                "atr_5s",
                (self.feature_engines[coin_id].v_t * 0.5 + atr_pct * mid_price * 0.5).max(0.0005),
            );
            set_reg("atr_1m", (atr_pct * mid_price * 7.746).max(0.0020));
            set_reg("order_flow_direction", flow_dir);
            set_reg("order_book_imbalance", obi_val);
            set_reg("orderbook_imbalance", obi_val);
            set_reg("order_flow_imbalance", ofi_value);
            set_reg(
                "weak_alpha_signal",
                (ofi_value * 0.5 + obi_val * 0.5).clamp(-1.0, 1.0),
            );
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
            let atr_abs = (atr_pct * mid_price).max(1e-8);
            set_reg(
                "hawkes_intensity",
                (1.0 + (a_t.abs() / atr_abs).clamp(0.0, 4.0)).clamp(0.1, 5.0),
            );
            set_reg("bessel_alpha", 1.5);
            set_reg("hawkes_dt", 0.05);
            set_reg(
                "microstructure_noise_variance",
                (atr_pct * 0.1).max(0.00001),
            );

            let hebbian_mult = self
                .arena
                .registry
                .get_scoped_value_or(&sym, "perceptron_hebbian_weight", 1.0)
                .clamp(0.5, 2.0);

            // 34D Macro+Micro Features for NanoForest (Indices 0..24 used by trained trees)
            let features = self.feature_engines[coin_id].get_features();
            let swing_feats = self.feature_engines[coin_id].get_swing_features();

            // F4.7 — ENSAMBLE REAL: ambos modelos opinan; la probabilidad es
            // el promedio ponderado con pesos que evolucionan por Brier real.
            // (Antes: if-else — el NN solo opinaba si el forest NO existía.)
            let combined_tensor =
                self.build_54d_tensor(coin_id, bid_qty, ask_qty, mid_price, omni_features);
            // D-406 & D-407: Soporte para modelo por activo ({sym}_SCALP) con fallback a BTCUSDT_SCALP
            let sym = quantum_arena::symbol_registry::try_symbol(coin_id)
                .unwrap_or_else(|| "BTCUSDT".to_string());
            let coin_model_key = format!("{}_SCALP", sym);
            let active_forest = crate::ml_inference::NanoForest::get_global(&coin_model_key)
                .or_else(|| self.scalp_forest.clone());
            let coin_ensemble = if coin_id < self.ensembles.len() {
                &mut self.ensembles[coin_id]
            } else {
                &mut self.ensemble
            };
            if let Some(f) = &active_forest {
                if let Some(p) = f.predict(&swing_feats) {
                    coin_ensemble.submit(crate::ensemble::ModelId::ScalpForest, p as f64);
                }
            }
            if let Some(nn) = self.swing_nn.as_mut() {
                let in_dim = nn.layer1.in_features;
                let p_opt = if in_dim == 34 {
                    let mut swing_feats_f64 = [0.0; 34];
                    for idx in 0..34 {
                        swing_feats_f64[idx] = swing_feats[idx] as f64;
                    }
                    nn.predict_for_coin(coin_id, &swing_feats_f64)
                } else if in_dim == 12 {
                    let mut micro_f64 = [0.0; 12];
                    for idx in 0..12 {
                        micro_f64[idx] = features[idx] as f64;
                    }
                    nn.predict_for_coin(coin_id, &micro_f64)
                } else {
                    nn.predict_for_coin(coin_id, &combined_tensor)
                };
                if let Some(p) = p_opt {
                    coin_ensemble.submit(crate::ensemble::ModelId::SwingNN, p);
                }
            }
            let mut online_feat = [0.0f32; 64];
            for (idx, &f) in features.iter().enumerate().take(64) {
                online_feat[idx] = f;
            }
            // F-026: Conectar Online Learning (Kalman Adaptive Module) a la inferencia activa
            let online_residual = self.online_learner.predict(&online_feat) as f64;
            let base_ml_prob = if let Some(p) = coin_ensemble.combined() {
                p
            } else {
                0.5
            };
            let ml_prob = (base_ml_prob + online_residual.clamp(-0.15, 0.15) + spot_bias).clamp(0.0, 1.0);
            self.last_ml_prob = ml_prob as f32;
            coin.ml_prob.store(ml_prob, Ordering::Relaxed);
            set_reg("ml_prob", ml_prob);
            set_reg("ml_prob_scalp", ml_prob);

            let nn_score: f64 = self.feature_engines[coin_id].update_ml_prediction(ml_prob);

            let current_obi = obi_val;
            let dynamic_atr_min = self.arena.config.dynamic_atr_min.load(Ordering::Relaxed);
            let tau_dom = self
                .temporal_spectrum
                .get(coin_id)
                .map(|s| s.dominant_tau_ms)
                .unwrap_or(30_000.0);
            let dynamic_obi_thr = self
                .arena
                .config
                .obi_threshold_at_tau(tau_dom)
                .clamp(0.12, 0.60);
            let dynamic_ema_thr = self
                .arena
                .config
                .dynamic_ema_trend
                .load(Ordering::Relaxed)
                .clamp(0.0008, 0.0040);
            let dynamic_ofi_thr = self
                .arena
                .config
                .dynamic_ofi_threshold
                .load(Ordering::Relaxed)
                .clamp(0.15, 0.95);

            let micro_trend = self.feature_engines[coin_id].get_micro_trend();
            let macro_trend = self.feature_engines[coin_id].get_macro_trend();
            let higher_trend = self.feature_engines[coin_id].get_higher_trend();
            let secular_trend = self.feature_engines[coin_id].get_secular_trend();
            let is_bear = self.feature_engines[coin_id].is_macro_bear();
            let is_bull = self.feature_engines[coin_id].is_macro_bull();

            let bear_signal = is_bear && macro_trend < -dynamic_ema_thr;
            let bull_signal = is_bull && macro_trend > dynamic_ema_thr;

            let is_confirmed_downtrend = bear_signal && !bull_signal;
            let is_confirmed_uptrend = bull_signal && !bear_signal;

            set_reg("ema_trend", micro_trend);
            set_reg("ema_trend_swing", macro_trend);
            set_reg("higher_trend", higher_trend);
            set_reg(
                "trend_direction",
                if higher_trend.abs() > 0.0010 {
                    higher_trend.signum()
                } else if macro_trend > 0.0 {
                    1.0
                } else if macro_trend < 0.0 {
                    -1.0
                } else {
                    0.0
                },
            );
            let conf_alpha = self
                .arena
                .config
                .conformal_alpha
                .load(Ordering::Relaxed)
                .clamp(0.01, 0.30);
            // D-617/D-618: el calibrador recibe el nivel objetivo del genoma y
            // decide con la regla selectiva conformal (conjunto = {gana}) sobre
            // su nivel efectivo corregido por ACI. `conformal_accept` es lo que
            // consume el filtro; el p-valor queda para telemetría.
            self.conformal.set_target_alpha(conf_alpha);
            let ml_prob_now = coin.ml_prob.load(Ordering::Relaxed);
            // D-676: la aceptación depende de la dirección — un largo gana si el
            // precio sube (p = ml_prob) y un corto si baja (p = 1 − ml_prob).
            let conformal_p = self.conformal.p_value(ml_prob_now);
            let conformal_p_short = self.conformal.p_value(1.0 - ml_prob_now);
            let accept_long = self.conformal.accepts(ml_prob_now);
            let accept_short = self.conformal.accepts(1.0 - ml_prob_now);
            set_reg("conformal_p_value", conformal_p);
            set_reg("conformal_p_value_short", conformal_p_short);
            set_reg("conformal_alpha", conf_alpha);
            set_reg("conformal_alpha_eff", self.conformal.effective_alpha());
            set_reg("conformal_accept_long", if accept_long { 1.0 } else { 0.0 });
            set_reg("conformal_accept_short", if accept_short { 1.0 } else { 0.0 });
            let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
            let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
            let total_vol_cvd = buy_vol + sell_vol;
            let rolling_cvd = if total_vol_cvd > 0.0 {
                (buy_vol - sell_vol) / total_vol_cvd
            } else {
                0.0
            };
            let ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;

            let obi_norm = (current_obi / dynamic_obi_thr).clamp(-1.5, 1.5);
            let ofi_norm = (ofi / dynamic_ofi_thr).clamp(-1.5, 1.5);
            let vecm_basis_z = if spot_bid > 0.0 && spot_ask > 0.0 {
                let spot_mid = (spot_bid + spot_ask) / 2.0;
                ((mid_price - spot_mid) / (mid_price * atr_pct.max(0.0005))).clamp(-3.0, 3.0)
            } else if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                ((mid_price - self.feature_engines[coin_id].kline_ema_slow)
                    / (mid_price * atr_pct.max(0.0005)))
                .clamp(-3.0, 3.0)
            } else {
                0.0
            };
            set_reg("vecm_zscore", vecm_basis_z);
            set_reg("cointegration_zscore", leader_mom.clamp(-3.0, 3.0));
            let ema_macro = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                self.feature_engines[coin_id].kline_ema_slow
            } else {
                mid_price
            };
            let pos_dev =
                ((mid_price - ema_macro) / (mid_price * atr_pct.max(0.0005))).clamp(-3.0, 3.0);
            set_reg("quantum_position_deviation", pos_dev);
            let hawkes_intensity = (1.0 + current_obi.abs() * 2.0).clamp(0.1, 5.0) / 5.0;
            let dir_flow_sign = if current_obi.abs() > 0.05 {
                current_obi.signum()
            } else if ofi.abs() > 0.05 {
                ofi.signum()
            } else {
                macro_trend.signum()
            };
            let dir_hawkes = hawkes_intensity * dir_flow_sign;
            let regime_code = ((hurst_val - 0.50) * 2.0).clamp(-1.0, 1.0);
            let dir_regime = regime_code
                * (if macro_trend.abs() > 0.0005 {
                    macro_trend.signum()
                } else {
                    dir_flow_sign
                });
            // D-643 (DÉCIMA OLA): el gen `min_confidence_btc` entra acotado a sus
            // bounds [0,50; 0,95] (`clamp_slot`). El `.max(0.70).clamp(0.70, 0.90)`
            // de lectura anulaba la parte de la banda que la evolución explora.
            let tensor_min_conf = self
                .arena
                .config
                .min_confidence_btc
                .load(Ordering::Relaxed);
            let ppo_state = [
                ofi_norm,
                obi_norm,
                dir_hawkes,
                lead_lag_div.clamp(-1.5, 1.5),
                dir_regime,
            ];
            let ppo_score = self.ppo_engine.evaluate_policy(&ppo_state);
            // 70% política PPO aprendida + 30% flujo acumulado CVD
            let micro_score: f64 = (ppo_score * 0.70 + rolling_cvd * 0.30).clamp(-1.0, 1.0);

            let sym = quantum_arena::symbol_registry::try_spec(coin_id)
                .map(|s| s.symbol)
                .unwrap_or_else(|| "BTCUSDT".to_string());
            // S-08 — DES-DUPLICACIÓN: scalp y swing son ALIASES del consenso
            // continuo — evaluarlos por separado gastaba 42 evaluaciones de
            // estrategia por tick para obtener 1 señal idéntica. Ahora UNA
            // sola evaluación; las tres referencias son Copys del mismo valor.
            let tensor_cont = self
                .tensor_orchestrator
                .evaluate_continuous_consensus_for_coin(coin_id, &sym);
            let tensor_scalp = tensor_cont;
            let tensor_swing = tensor_cont;

            let tensor_boost = match tensor_cont.signal {
                SignalType::Long => tensor_cont.net_confidence.clamp(0.0, 1.0),
                SignalType::Short => -tensor_cont.net_confidence.clamp(0.0, 1.0),
                SignalType::Flat => 0.0,
            };

            // Fusión Bayesiana Calibrada: w_micro (L2) + 60% del residuo a DarkAlpha ML + 40% al consenso tensorial
            let w_micro = self
                .arena
                .config
                .weight_obi
                .load(Ordering::Relaxed)
                .clamp(0.1, 0.8);
            let remaining = 1.0 - w_micro;
            let w_nn = remaining * 0.60;
            let w_tensor = remaining * 0.40;
            let raw_composite = micro_score * w_micro + nn_score * w_nn + tensor_boost * w_tensor;
            let composite_score: f64 = (raw_composite * hebbian_mult).clamp(-1.0, 1.0);
            self.diag_dir.record_evaluation(
                composite_score,
                ml_prob,
                [
                    micro_score * w_micro * hebbian_mult,
                    nn_score * w_nn * hebbian_mult,
                    tensor_boost * w_tensor * hebbian_mult,
                ],
                tensor_cont.signal,
            );

            let mut scalp_intent = SignalIntent::flat();
            let spread_pct = if mid_price > 0.0 {
                (ask - bid) / mid_price
            } else {
                0.0
            };
            let dynamic_max_spread = (atr_pct * 0.25).clamp(0.0006, 0.0025);
            let spread_ok = spread_pct <= dynamic_max_spread;

            if atr_pct > dynamic_atr_min
                && spread_ok
                && self.feature_engines[coin_id].can_open_scalp(600)
            {
                let is_mean_reverting = hurst_val < 0.45;

                let ema_slow = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let price_stretch = if ema_slow > 0.0 {
                    (mid_price - ema_slow) / cur_atr
                } else {
                    0.0
                };
                // D-472, D-474 & D-478: Disciplina Antiextensión y Cero Persecución (No Chasing Law)
                // Prohibido vender por debajo de la media (price_stretch < 0.0) o comprar por encima (price_stretch > 0.0).
                // En tendencia, entrar exclusivamente en o por encima de la EMA slow para cortos (y en o por debajo para largos).
                let (min_stretch_short, max_stretch_long) = if is_mean_reverting {
                    (0.05, -0.05)
                } else {
                    (0.00, 0.00)
                };
                // D-621 (DÉCIMA OLA): la cota de magnitud era ±1,50 ATR. Se expresa
                // en desviaciones típicas de la distancia a la EMA de 21 velas: más
                // allá del 95 % el desplazamiento es significativo y deja de ser un
                // retroceso que se pueda operar.
                let stretch_z = crate::diffusion::atr_stretch_z(
                    price_stretch,
                    crate::diffusion::EMA_SLOW_BARS,
                );
                let not_overextended_long =
                    price_stretch <= max_stretch_long && stretch_z >= -crate::diffusion::Z95;
                let not_overextended_short =
                    price_stretch >= min_stretch_short && stretch_z <= crate::diffusion::Z95;

                // D-105: Mapeo de convicción Bayesiana calibrada para Kelly sizing realista
                let sig_conf = |score: f64| -> f64 {
                    (0.50 + 0.40 * score.abs().clamp(0.0, 1.0)).clamp(0.51, 0.90)
                };

                let is_anti_persistent = hurst_val < 0.42;
                let mut dynamic_tech_thr = self
                    .arena
                    .config
                    .tech_threshold
                    .load(Ordering::Relaxed);

                if is_anti_persistent {
                    dynamic_tech_thr *= 1.20; // Elevar exigencia analítica 20% en régimen de ruido/chop
                }

                let min_obi_trend: f64 = if is_anti_persistent {
                    (dynamic_obi_thr * 1.10).min(0.35)
                } else {
                    (dynamic_obi_thr * 0.85).max(0.12)
                };
                let min_obi_pullback: f64 = if is_anti_persistent {
                    (dynamic_obi_thr * 1.10).min(0.35)
                } else {
                    dynamic_obi_thr.max(0.14)
                };

                // D-500: Anti-Chop & Post-Loss Conviction Firewall con Direccionalidad y Decaimiento Temporal
                let short_streak = self.feature_engines[coin_id].get_active_directional_streak(false);
                let long_streak = self.feature_engines[coin_id].get_active_directional_streak(true);

                // D-460 & D-466: Unificación Continua del Generador de Señales (Multiscale Vector Field).
                // Confluencia de triple escala temporal: Micro (1m ticks), Intermedio (EMA 9 vs 21), y Macro Superior (2-Hour EMA 120).
                if is_confirmed_downtrend {
                    // RÉGIMEN BAJISTA CONFIRMADO (MULTISCALE DOWNTREND)
                    let d_tech_thr = if short_streak >= 2 { dynamic_tech_thr.max(0.30) } else { dynamic_tech_thr };
                    let d_obi_trend = if short_streak >= 2 { min_obi_trend.max(0.22) } else { min_obi_trend };
                    // 1. Tendencial Short: Flujo institucional, confluencia L2 y ML apuntan a la baja
                    if composite_score < -d_tech_thr
                        && not_overextended_short
                        && current_obi < -d_obi_trend
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 1.0,
                            ..Default::default()
                        };
                    // 2. Pullback Short: Rebote hacia ema_slow vendido con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score <= -d_tech_thr),
                    // filtro de tendencia superior (higher_trend <= -0.0010 para evitar vender en rallies)
                    // y desbalance de libro sólido (current_obi < -0.20)
                    } else if short_streak < 2
                        && higher_trend <= -0.0010
                        && price_stretch >= 0.15
                        && price_stretch <= 1.20
                        && current_obi < -min_obi_pullback.max(0.20)
                        && composite_score <= -d_tech_thr
                        && micro_trend <= 0.0
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score.abs()),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 2.0,
                            ..Default::default()
                        };
                    // 3. Reversión Long: Exclusivamente ante capitulación estadística extrema (pánico masivo con absorción)
                    } else if price_stretch < -2.5
                        && current_obi > dynamic_obi_thr * 0.8
                        && composite_score > dynamic_tech_thr
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 3.0,
                            ..Default::default()
                        };
                    }
                } else if is_confirmed_uptrend {
                    // RÉGIMEN ALCISTA CONFIRMADO (MULTISCALE UPTREND)
                    let u_tech_thr = if long_streak >= 2 { dynamic_tech_thr.max(0.30) } else { dynamic_tech_thr };
                    let u_obi_trend = if long_streak >= 2 { min_obi_trend.max(0.22) } else { min_obi_trend };
                    // 1. Tendencial Long: Flujo institucional, confluencia L2 y ML apuntan al alza
                    if composite_score > u_tech_thr
                        && not_overextended_long
                        && current_obi > u_obi_trend
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 4.0,
                            ..Default::default()
                        };
                    // 2. Dip Long: Corrección hacia ema_slow comprada con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score >= u_tech_thr),
                    // filtro de tendencia superior (higher_trend >= 0.0010)
                    // y desbalance de libro sólido (current_obi > min_obi_pullback.max(0.20))
                    } else if long_streak < 2
                        && higher_trend >= 0.0010
                        && price_stretch <= -0.15
                        && price_stretch >= -1.20
                        && current_obi > min_obi_pullback.max(0.20)
                        && composite_score >= u_tech_thr
                        && micro_trend >= 0.0
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 5.0,
                            ..Default::default()
                        };
                    // 3. Reversión Short: Exclusivamente ante euforia parabólica extrema con ventas masivas L2
                    } else if price_stretch > 2.5
                        && current_obi < -dynamic_obi_thr * 0.8
                        && composite_score < -dynamic_tech_thr
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score.abs()),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 6.0,
                            ..Default::default()
                        };
                    }
                } else {
                    // RÉGIMEN NEUTRO / RANGO LATERAL (Disciplina de reversión a la media: comprar en soporte, vender en resistencia)
                    let range_thr = dynamic_tech_thr * 1.15;
                    let range_obi = (dynamic_obi_thr * 0.85).clamp(0.12, 0.35);
                    if composite_score > range_thr && current_obi > range_obi && price_stretch <= -0.15 && micro_trend >= 0.0 {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 7.0,
                            ..Default::default()
                        };
                    } else if composite_score < -range_thr
                        && current_obi < -range_obi
                        && price_stretch >= 0.15
                        && micro_trend <= 0.0
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 8.0,
                            ..Default::default()
                        };
                    } else if short_streak < 2 && price_stretch > 1.0 && current_obi < -range_obi * 1.15 && composite_score <= -0.24 && micro_trend <= 0.0
                    {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(current_obi.abs().min(composite_score.abs())),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 9.0,
                            ..Default::default()
                        };
                    } else if long_streak < 2 && price_stretch < -1.0 && current_obi > range_obi * 1.15 && composite_score >= 0.24 && micro_trend >= 0.0 {
                        scalp_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(current_obi.abs().min(composite_score.abs())),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 10.0,
                            ..Default::default()
                        };
                    }
                }

                let tensor_cutoff = ((tensor_min_conf - 0.50) * 2.0).clamp(0.35, 0.80);
                if scalp_intent.signal == SignalType::Flat
                    && tensor_scalp.signal != SignalType::Flat
                    && !is_anti_persistent
                    && tensor_scalp.net_confidence.abs() >= tensor_cutoff
                {
                    // D-502, D-503 & D-505: Invariante de Momentum Jerárquico, Micro-Surge y Concurrencia Multiescala
                    // 1. Prohibido abrir Short ante micro-spikes de ticks adversos > +4 bps (mic > 0.00040).
                    // 2. Prohibido abrir Short si 2h, 1m y ticks están concurrentemente subiendo (ht > 0.00020 && mac > 0.00005 && mic > 0.00015).
                    // 3. Prohibido abrir Short si momentum micro y macro son adversos sin soporte secular fuerte (mac > 0.00005 && mic > 0.00010 && st > -0.0020).
                    // Simétrico para órdenes Long.
                    let is_adverse_momentum_short = (macro_trend > 0.00005 && micro_trend > 0.00010 && secular_trend > -0.0020)
                        || (macro_trend > 0.00010 && micro_trend > 0.00015)
                        || (micro_trend > 0.00040)
                        || (higher_trend > 0.00010 && macro_trend > 0.0 && micro_trend > 0.00010 && secular_trend > -0.0010);
                    let is_adverse_momentum_long = (macro_trend < -0.00005 && micro_trend < -0.00010 && secular_trend < 0.0020)
                        || (macro_trend < -0.00010 && micro_trend < -0.00015)
                        || (micro_trend < -0.00040)
                        || (higher_trend < -0.00010 && macro_trend < 0.0 && micro_trend < -0.00010 && secular_trend < 0.0010);
                    let tensor_tech_thr = (dynamic_tech_thr * 0.90).max(0.22);
                    let range_obi = (dynamic_obi_thr * 0.85).clamp(0.12, 0.35);

                    // Diagnóstico por dirección: las condiciones del gate se nombran una
                    // sola vez y el diagnóstico cuenta cuál falla. La semántica es la de la
                    // conjunción anterior: comparaciones puras, sin efectos laterales.
                    let long_conditions = [
                        long_streak < 2,
                        !is_confirmed_downtrend,
                        !is_adverse_momentum_long,
                        !(higher_trend < -0.0002 && secular_trend < 0.0),
                        !(price_stretch < -0.80 && secular_trend < 0.0010),
                        higher_trend >= -0.0008,
                        composite_score >= tensor_tech_thr,
                        current_obi > range_obi,
                        not_overextended_long,
                    ];
                    let short_conditions = [
                        short_streak < 2,
                        !is_confirmed_uptrend,
                        !is_adverse_momentum_short,
                        !(higher_trend > 0.0002 && secular_trend > 0.0),
                        !(price_stretch > 0.80 && secular_trend > -0.0010),
                        higher_trend <= 0.0008,
                        composite_score <= -tensor_tech_thr,
                        current_obi < -range_obi,
                        not_overextended_short,
                    ];
                    let tensor_allowed = match tensor_scalp.signal {
                        SignalType::Long => {
                            self.diag_dir.record_gate(true, &long_conditions);
                            long_conditions.iter().all(|&ok| ok)
                        }
                        SignalType::Short => {
                            self.diag_dir.record_gate(false, &short_conditions);
                            short_conditions.iter().all(|&ok| ok)
                        }
                        SignalType::Flat => false,
                    };
                    if tensor_allowed {
                        scalp_intent = SignalIntent {
                            signal: tensor_scalp.signal,
                            confidence: tensor_scalp
                                .net_confidence
                                .abs()
                                .clamp(tensor_min_conf, 1.0),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 11.0,
                            ..Default::default()
                        };
                    }
                }

                if scalp_intent.signal == SignalType::Flat {
                    let hawkes_r = self.feature_engines[coin_id].cvpin.current_vpin();
                    if let Some(mut turbo_intent) =
                        signal_engine::turbo_scalper::TurboScalpEngine::evaluate_turbo_scalp(
                            &self.arena,
                            current_obi,
                            ofi,
                            hawkes_r,
                            shannon_ent,
                            mid_price,
                            atr_pct,
                            event_time_ms,
                        )
                    {
                        let turbo_streak = if turbo_intent.signal == SignalType::Long {
                            long_streak
                        } else {
                            short_streak
                        };
                        let turbo_aligned_with_regime = if is_confirmed_downtrend {
                            turbo_intent.signal == SignalType::Short
                        } else if is_confirmed_uptrend {
                            turbo_intent.signal == SignalType::Long
                        } else if is_mean_reverting {
                            (turbo_intent.signal == SignalType::Long
                                && price_stretch < -0.5
                                && macro_trend >= -dynamic_ema_thr)
                                || (turbo_intent.signal == SignalType::Short
                                    && price_stretch > 0.5
                                    && macro_trend <= dynamic_ema_thr)
                        } else {
                            (turbo_intent.signal == SignalType::Long && macro_trend >= 0.0)
                                || (turbo_intent.signal == SignalType::Short && macro_trend <= 0.0)
                        };

                        if turbo_streak < 2
                            && turbo_aligned_with_regime
                            && ((turbo_intent.signal == SignalType::Long && not_overextended_long)
                                || (turbo_intent.signal == SignalType::Short
                                    && not_overextended_short))
                        {
                            turbo_intent.volume_flow_rate = 12.0;
                            scalp_intent = turbo_intent;
                        }
                    }
                }

                // D-472: Invariante Bayesiano Absoluto — Prohibir cualquier scalp que contradiga el composite score
                if scalp_intent.signal == SignalType::Long && composite_score < 0.0 {
                    scalp_intent = SignalIntent::flat();
                } else if scalp_intent.signal == SignalType::Short && composite_score > 0.0 {
                    scalp_intent = SignalIntent::flat();
                }

                // F-009 FIX: Continuous ML Probability Weighting (replaces binary switch)
                // Instead of killing signals when ml_prob crosses 0.51/0.49, modulate
                // confidence continuously. The farther ml_prob is from 0.5 in the signal's
                // direction, the more the confidence is amplified. Against the signal,
                // confidence is reduced proportionally.
                {
                    let ml_directional = match scalp_intent.signal {
                        SignalType::Long => (ml_prob - 0.5) * 2.0,   // [-1, +1] where +1 = strong bullish
                        SignalType::Short => (0.5 - ml_prob) * 2.0,  // [-1, +1] where +1 = strong bearish
                        _ => 0.0,
                    };
                    // If ML contradicts signal (ml_directional < 0) AND no extreme price action,
                    // reduce confidence. If ml_directional < -0.5, kill signal entirely.
                    if ml_directional < -0.50 && price_stretch.abs() < 2.5 {
                        scalp_intent = SignalIntent::flat();
                    } else if ml_directional < 0.0 && price_stretch.abs() < 2.5 {
                        // Soft penalty: scale confidence by (1 + ml_directional) where ml_directional is [-0.5, 0)
                        scalp_intent.confidence *= (1.0 + ml_directional).max(0.1);
                    } else if ml_directional > 0.0 {
                        // ML confirms signal direction: boost confidence proportionally
                        scalp_intent.confidence *= 1.0 + ml_directional * 0.5;
                        scalp_intent.confidence = scalp_intent.confidence.min(0.99);
                    }
                }

                // D-622 (DÉCIMA OLA): el cooldown binario del lado scalp (20 s tras
                // ganar; 90 s o 180 s tras perder según volatilidad) se retira. El
                // guard unificado D-463 cubre la reentrada tras cualquier cierre
                // —dirección, ganancia o pérdida, racha— con ventanas iguales o
                // mayores, y sin distinguir de qué «motor» vino la señal.
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
                let prob_bytes = (ml_prob as f32).to_le_bytes();
                payload[0..4].copy_from_slice(&prob_bytes);
                fr.record(telemetry_server::FlightEvent {
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos() as u64,
                    trace_id: tick,
                    event_type: 1,
                    payload,
                });
            }

            let hurst_exponent = features[1] as f64;
            // X-024 (REHAB-5): el gate swing YA NO consume la NN cruda — pasa
            // por el ENSAMBLE (pesos aprendidos por Brier real). Antes: doble
            // inferencia por tick (2× actualización Welford) y los pesos del
            // ensamble solo afectaban el scalp — el swing ignoraba el aprendizaje.
            let mut swing_nn_pred = coin.ml_prob.load(Ordering::Relaxed);
            if let Some(ens) = self.ensembles.get(coin_id) {
                if let Some(p) = ens.combined() {
                    swing_nn_pred = p.clamp(0.01, 0.99);
                }
            }

            // --- EVALUACIÓN DE SEÑAL SWING / TENDENCIAL NATIVA (MOTOR UNIFICADO CONTINUO) ---
            let mut swing_intent = SignalIntent::flat();
            let trend_threshold = self
                .arena
                .config
                .trend_threshold
                .load(Ordering::Relaxed);

            let ml_long = self.arena.config.ml_threshold_long.load(Ordering::Relaxed);
            let ml_short = self.arena.config.ml_threshold_short.load(Ordering::Relaxed);
            let effective_ml_long = if ml_long <= 0.50 {
                1.0 - ml_long
            } else {
                ml_long
            }
            .clamp(0.51, 0.95);
            let effective_ml_short = if ml_short >= 0.50 {
                1.0 - ml_short
            } else {
                ml_short
            }
            .clamp(0.05, 0.49);

            let raw_base = self.arena.config.base_duration_ms.load(Ordering::Relaxed);
            let swing_duration_ms = if raw_base.is_finite() && raw_base > 0.0 {
                (raw_base * 60.0) as u64
            } else {
                3_600_000
            }
            .max(1_800_000);

            let ema_fast = if self.feature_engines[coin_id].kline_ema_fast > 0.0 {
                self.feature_engines[coin_id].kline_ema_fast
            } else {
                self.feature_engines[coin_id].ema_fast
            };
            let ema_slow = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                self.feature_engines[coin_id].kline_ema_slow
            } else {
                self.feature_engines[coin_id].ema_slow
            };
            let ma_trend_strength = if ema_slow > 0.0 {
                ((ema_fast - ema_slow) / ema_slow).abs()
            } else {
                0.0
            };
            let is_trend_candidate = hurst_exponent >= 0.48
                && (hurst_exponent >= trend_threshold || ma_trend_strength > 0.0020);

            if is_trend_candidate {
                if ema_slow > 0.0 {
                    let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                    let price_stretch = (mid_price - ema_slow) / cur_atr;
                    // D-621 (DÉCIMA OLA): antes 0,20 ATR a favor y 1,50 ATR en contra, una
                    // asimetría de 7,5× sin derivación. La regla declarada es no comprar
                    // por encima de la media ni vender por debajo (cota 0), y no
                    // entrar si el desplazamiento en contra ya es significativo (z95).
                    let swing_stretch_z = crate::diffusion::atr_stretch_z(
                        price_stretch,
                        crate::diffusion::EMA_SLOW_BARS,
                    );
                    let not_chasing_swing_long =
                        swing_stretch_z <= 0.0 && swing_stretch_z >= -crate::diffusion::Z95;
                    let not_chasing_swing_short =
                        swing_stretch_z >= 0.0 && swing_stretch_z <= crate::diffusion::Z95;

                    let macd_diff = (ema_fast - ema_slow) / ema_slow;
                    let swing_tp = self.arena.config.swing_tp_base.load(Ordering::Relaxed);
                    let threshold =
                        (swing_tp * 0.003).max(0.0001) * (1.0 / hurst_exponent.max(0.1));

                    let is_bull_trend = is_confirmed_uptrend && ema_fast > ema_slow;
                    let is_bear_trend = is_confirmed_downtrend && ema_fast < ema_slow;

                    if macd_diff > threshold
                        && swing_nn_pred >= effective_ml_long
                        && is_bull_trend
                        && not_chasing_swing_long
                    {
                        let raw_conf = (macd_diff.abs() * hurst_exponent * 50.0)
                            .max((swing_nn_pred - 0.5).max(0.0) * 2.0);
                        let confidence = if raw_conf.is_finite() {
                            raw_conf.tanh().clamp(0.55, 0.95)
                        } else {
                            0.55
                        };
                        swing_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence,
                            expected_duration_ms: swing_duration_ms,
                            horizon: strategy_core::TradeHorizon::Continuous,
                            // D-678: rama 13 · camino de tendencia.
                            volume_flow_rate: 13.0,
                            ..Default::default()
                        };
                    } else if macd_diff < -threshold
                        && swing_nn_pred <= effective_ml_short
                        && is_bear_trend
                        && not_chasing_swing_short
                    {
                        let raw_conf = (macd_diff.abs() * hurst_exponent * 50.0)
                            .max((0.5 - swing_nn_pred).max(0.0) * 2.0);
                        let confidence = if raw_conf.is_finite() {
                            raw_conf.tanh().clamp(0.55, 0.95)
                        } else {
                            0.55
                        };
                        swing_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence,
                            expected_duration_ms: swing_duration_ms,
                            horizon: strategy_core::TradeHorizon::Continuous,
                            // D-678: rama 13 · camino de tendencia.
                            volume_flow_rate: 13.0,
                            ..Default::default()
                        };
                    }
                }
            }

            // D-685: rama de consenso tensorial contenida (ver
            // `CONSENSUS_BRANCH_ENABLED`).
            if CONSENSUS_BRANCH_ENABLED && swing_intent.signal == SignalType::Flat {
                let ema_fast = if self.feature_engines[coin_id].kline_ema_fast > 0.0 {
                    self.feature_engines[coin_id].kline_ema_fast
                } else {
                    self.feature_engines[coin_id].ema_fast
                };
                let ema_slow = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let is_bull = is_confirmed_uptrend && ema_fast > ema_slow;
                let is_bear = is_confirmed_downtrend && ema_fast < ema_slow;

                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let price_stretch = if ema_slow > 0.0 {
                    (mid_price - ema_slow) / cur_atr
                } else {
                    0.0
                };
                // D-621 (DÉCIMA OLA): mismas cotas en z que el camino de tendencia.
                let consensus_stretch_z = crate::diffusion::atr_stretch_z(
                    price_stretch,
                    crate::diffusion::EMA_SLOW_BARS,
                );
                let not_chasing_long =
                    consensus_stretch_z <= 0.0 && consensus_stretch_z >= -crate::diffusion::Z95;
                let not_chasing_short =
                    consensus_stretch_z >= 0.0 && consensus_stretch_z <= crate::diffusion::Z95;

                // D-458 & D-462: Desbloquear consenso continuo evitando la persecución tardía (anti-chase guard)
                if tensor_swing.signal == SignalType::Long
                    && is_bull
                    && not_chasing_long
                    && tensor_swing.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    swing_intent = SignalIntent {
                        signal: tensor_swing.signal,
                        confidence: tensor_swing
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_swing.signal == SignalType::Short
                    && is_bear
                    && not_chasing_short
                    && tensor_swing.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    swing_intent = SignalIntent {
                        signal: tensor_swing.signal,
                        confidence: tensor_swing
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_cont.signal == SignalType::Long
                    && is_bull
                    && not_chasing_long
                    && tensor_cont.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    swing_intent = SignalIntent {
                        signal: tensor_cont.signal,
                        confidence: tensor_cont
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_cont.signal == SignalType::Short
                    && is_bear
                    && not_chasing_short
                    && tensor_cont.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    swing_intent = SignalIntent {
                        signal: tensor_cont.signal,
                        confidence: tensor_cont
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                }
            }

            // D-622 (DÉCIMA OLA): el cooldown del lado swing dependía de que el
            // ÚLTIMO CIERRE SCALP hubiera perdido — una pérdida de 30 segundos
            // bloqueaba una tesis de horas. Contaminación cruzada entre horizontes
            // que el sistema declara unificados. Lo cubre el guard D-463.

            if coin_id < self.last_scalp_intent.len() {
                self.last_scalp_intent[coin_id] = scalp_intent;
            }
            if coin_id < self.last_swing_intent.len() {
                self.last_swing_intent[coin_id] = swing_intent;
            }

            // D-431: Composición de onda multiescala no destructiva (Continuous Wave Mechanics).
            // Evita el canibalismo ciego donde una discrepancia de 0.01 abre operaciones contratendencia.
            let mut unified_intent = SignalIntent::flat();
            if scalp_intent.signal != SignalType::Flat && swing_intent.signal != SignalType::Flat {
                if scalp_intent.signal == swing_intent.signal {
                    // D-623 (DÉCIMA OLA): antes `máx(p₁, p₂)·1,10` con suelo 0,60. Dos
                    // evidencias sólo se combinan sumando log-odds si son
                    // condicionalmente independientes, y éstas no lo son: ambas leen el
                    // mismo consenso tensorial (`tensor_scalp` y `tensor_swing` son copias
                    // de `tensor_cont`). Con evidencia dependiente, la combinación que no
                    // inventa certeza es el máximo.
                    let boosted_conf = scalp_intent.confidence.max(swing_intent.confidence);
                    unified_intent = SignalIntent {
                        signal: scalp_intent.signal,
                        confidence: boosted_conf,
                        horizon: strategy_core::TradeHorizon::Continuous,
                        ..scalp_intent
                    };
                } else {
                    // Señales opuestas (conflicto de frecuencia):
                    // La tendencia mayor confirmada decide la dirección en el continuo temporal universal
                    if is_confirmed_downtrend {
                        if scalp_intent.signal == SignalType::Short {
                            unified_intent = SignalIntent {
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..scalp_intent
                            };
                        } else if swing_intent.signal == SignalType::Short {
                            unified_intent = SignalIntent {
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..swing_intent
                            };
                        } else {
                            unified_intent = SignalIntent::flat();
                        }
                    } else if is_confirmed_uptrend {
                        if scalp_intent.signal == SignalType::Long {
                            unified_intent = SignalIntent {
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..scalp_intent
                            };
                        } else if swing_intent.signal == SignalType::Long {
                            unified_intent = SignalIntent {
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..swing_intent
                            };
                        } else {
                            unified_intent = SignalIntent::flat();
                        }
                    } else {
                        // Conflicto sin tendencia dominante confirmada: preservar capital -> Flat
                        unified_intent = SignalIntent::flat();
                    }
                }
            } else if scalp_intent.signal != SignalType::Flat {
                unified_intent = SignalIntent {
                    horizon: strategy_core::TradeHorizon::Continuous,
                    ..scalp_intent
                };
            } else if swing_intent.signal != SignalType::Flat {
                unified_intent = SignalIntent {
                    horizon: strategy_core::TradeHorizon::Continuous,
                    ..swing_intent
                };
            }

            // D-474: Invariante Fractal Universal de Horizonte Continuo (Persistencia Browniana Hurst)
            // En el espectro continuo universal, la duración esperada se modula continuamente por Hurst:
            // Hurst < 0.48 (anti-persistente) comprime la duración hacia la microestructura (tau bajo);
            // Hurst >= 0.52 (persistente/trending) expande la duración temporal para capturar la tendencia.
            // D-609 (DÉCIMA OLA): antes escalones en H = 0,48 y 0,52 (duración ×½ o
            // ×2). Ahora la misma modulación es continua en H, y sólo actúa sobre
            // duraciones DECLARADAS: la versión anterior asignaba 15 s a cualquier
            // intención sin duración en régimen anti-persistente por un artefacto
            // de `(0 / 2).max(15_000)` — las entradas del consenso tensorial, que no
            // declaran duración, recibían un horizonte de 15 segundos.
            unified_intent.expected_duration_ms = hurst_duration_modulation(
                unified_intent.expected_duration_ms,
                hurst_exponent,
            );
            unified_intent.horizon = strategy_core::TradeHorizon::Continuous;

            // D-467, D-472, D-488 & D-496: Escudo Invariante Macro Multiescala (Secular 12h, Superior 2h y Macro 1m/15m).
            // Erradica operaciones a contratendencia del régimen mayor (e.g. comprar Longs en caída o vender Shorts en rally)
            // D-624 (DÉCIMA OLA): antes −3, −8 y −6 pb para tres horizontes (0,04 σ,
            // 0,24 σ y 0,87 σ con ATR del 0,10 %): el escudo bloqueaba largos en
            // cuanto el precio bajaba unos puntos básicos de la EMA de 12 h. Cada
            // tendencia se estandariza con su propia desviación de difusión y se
            // exige significación al 95 %, el mismo criterio para los tres horizontes.
            let shield_atr_ratio =
                self.feature_engines[coin_id].v_t.max(mid_price * 0.001) / mid_price.max(1e-12);
            let z_secular = crate::diffusion::ema_distance_z(
                secular_trend,
                shield_atr_ratio,
                crate::diffusion::EMA_MACRO_BARS,
            );
            let z_higher = crate::diffusion::ema_distance_z(
                higher_trend,
                shield_atr_ratio,
                crate::diffusion::EMA_TREND_BARS,
            );
            let z_macro = crate::diffusion::ema_spread_z(
                macro_trend,
                shield_atr_ratio,
                crate::diffusion::EMA_FAST_BARS,
                crate::diffusion::EMA_SLOW_BARS,
            );
            let z95 = crate::diffusion::Z95;
            if (is_confirmed_downtrend || z_secular < -z95 || z_higher < -z95 || z_macro < -z95)
                && unified_intent.signal == SignalType::Long
            {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                let dynamic_tech_thr = self
                    .arena
                    .config
                    .tech_threshold
                    .load(Ordering::Relaxed);
                // D-624: capitulación = desplazamiento significativo bajo la EMA de 21 velas.
                let extreme_capitulation = crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) < -z95
                    && current_obi > dynamic_obi_thr * 0.8
                    && composite_score > dynamic_tech_thr;
                if !extreme_capitulation {
                    unified_intent = SignalIntent::flat();
                }
            } else if (is_confirmed_uptrend || z_secular > z95 || z_higher > z95 || z_macro > z95)
                && unified_intent.signal == SignalType::Short
            {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                let dynamic_tech_thr = self
                    .arena
                    .config
                    .tech_threshold
                    .load(Ordering::Relaxed);
                // D-624: euforia = desplazamiento significativo sobre la EMA de 21 velas.
                let extreme_blowoff = crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) > z95
                    && current_obi < -dynamic_obi_thr * 0.8
                    && composite_score < -dynamic_tech_thr;
                if !extreme_blowoff {
                    unified_intent = SignalIntent::flat();
                }
            }

            // X-016 plenitud (REHAB-1b): ACONDICIONAMIENTO ESPECTRAL de la
            // entrada unificada. La persistencia de la escala dominante
            // (medida: autocorrelación de sorpresas — tendencia +1, reversión
            // −1, ruido 0) modula la confianza: tendencia confirmada la
            // preserva (×1), ruido la modula suavemente, reversión la castiga si es tendencial.
            if unified_intent.signal != SignalType::Flat {
                if let Some(spec) = self.temporal_spectrum.get(coin_id) {
                    let tau_dom = spec.dominant_tau_ms;
                    let persist = spec.persistence_at(tau_dom);
                    let is_trending_mode = is_confirmed_uptrend || is_confirmed_downtrend;
                    let directional_persist = if is_trending_mode { persist } else { -persist };
                    let factor = (1.0 + 0.25 * directional_persist).clamp(0.70, 1.30);
                    unified_intent.confidence =
                        (unified_intent.confidence * factor).clamp(0.10, 0.99);
                }
            }

            // D-620 (DÉCIMA OLA): aquí había un segundo gate de confianza sobre el
            // mismo gen, `(min_confidence_btc·0,85).clamp(0,48; 0,72)`. El
            // risk-engine exige después `min_confidence_btc·[1; 1,0645]`, que
            // siempre es mayor: este gate nunca decidía nada y sólo repetía con
            // otra fórmula una regla que vive en un único sitio.

            let current_cap = self.arena.unified_capital.load(Ordering::Relaxed);
            let _global_leverage = self
                .arena
                .config
                .global_leverage
                .load(Ordering::Relaxed)
                .clamp(1.0, 50.0);

            // Zero phantom equity check: si el capital fue liquidado, detener inmediatamente
            if current_cap <= 0.0 {
                self.arena.kill_switch_active.store(true, Ordering::SeqCst);
                return (None, closed_order, None);
            }

            // D-463: Unified Anti-Whiplash & Cycle Reset Guard (Erradicación del Chopping Post-Salida)
            if unified_intent.signal != SignalType::Flat {
                let last_close = coin.last_close_ts.load(Ordering::Relaxed);
                if last_close > 0 {
                    let elapsed_ms = event_time_ms.saturating_sub(last_close);
                    let last_is_long = coin.last_close_is_long.load(Ordering::Relaxed);
                    let last_was_win = coin.last_close_was_win.load(Ordering::Relaxed);
                    let is_same_dir = (unified_intent.signal == SignalType::Long && last_is_long)
                        || (unified_intent.signal == SignalType::Short && !last_is_long);

                    let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                    let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                        self.feature_engines[coin_id].kline_ema_slow
                    } else {
                        self.feature_engines[coin_id].ema_slow
                    };
                    let p_stretch = if ema_ref > 0.0 {
                        (mid_price - ema_ref) / cur_atr
                    } else {
                        0.0
                    };

                    let whiplash_veto = if last_was_win {
                        // POST-WIN: La pata impulsiva se monetizó con éxito.
                        // Prohibido comprar el techo del rally o vender el piso del dump recién tomado.
                        if is_same_dir {
                            if elapsed_ms < 180_000 {
                                (unified_intent.signal == SignalType::Long && p_stretch > 0.25)
                                    || (unified_intent.signal == SignalType::Short
                                        && p_stretch < -0.25)
                                    || elapsed_ms < 45_000
                            } else {
                                false
                            }
                        } else {
                            // Giro a contratendencia post-win: requiere al menos 60s de confirmación
                            elapsed_ms < 60_000
                        }
                    } else {
                        // POST-LOSS (SL): El trade fue liquidado por movimiento adverso brusco.
                        // Prohibido vender en capitulación tras saltar stop de Long o comprar en euforia tras stop de Short.
                        if !is_same_dir {
                            if elapsed_ms < 180_000 {
                                true
                            } else {
                                (unified_intent.signal == SignalType::Short && p_stretch < -0.30)
                                    || (unified_intent.signal == SignalType::Long
                                        && p_stretch > 0.30)
                            }
                        } else {
                            // En la MISMA DIRECCIÓN: si hay racha de pérdidas consecutivas,
                            // aplicar retroceso exponencial para cortar la hemorragia de trades repetidos
                            let is_scalp_long = unified_intent.signal == SignalType::Long;
                            let streak = self.feature_engines[coin_id].get_active_directional_streak(is_scalp_long);
                            let required_ms = match streak {
                                0 | 1 => 180_000,    // 3 minutos
                                2 => 1_200_000,      // 20 minutos (antes 30 min)
                                3 => 3_600_000,      // 1 hora (antes 2 horas)
                                _ => 7_200_000,      // 2 horas (antes 4 horas)
                            };
                            let time_veto = elapsed_ms < required_ms;
                            let stretch_veto = if streak >= 2 {
                                (unified_intent.signal == SignalType::Short && p_stretch < 0.10)
                                    || (unified_intent.signal == SignalType::Long && p_stretch > -0.10)
                            } else {
                                false
                            };
                            time_veto || stretch_veto
                        }
                    };

                    if whiplash_veto {
                        unified_intent = SignalIntent::flat();
                    }
                }
            }

            // D-472: Invariante Bayesiano Absoluto Universal (Cross-Horizon)
            // Ninguna orden unificada puede entrar si contradice la convicción Bayesiana (composite_score)
            if unified_intent.signal == SignalType::Long && composite_score < 0.0 {
                unified_intent = SignalIntent::flat();
            } else if unified_intent.signal == SignalType::Short && composite_score > 0.0 {
                unified_intent = SignalIntent::flat();
            }

            // D-499: Invariante de Convicción Post-Racha Direccional Universal (Cross-Horizon Directional Loss Streak Firewall)
            // Si el activo acumula una racha de 2 o más pérdidas consecutivas activas en su dirección (short/long),
            // se exige convicción Bayesiana institucional (|score| >= 0.28, |current_obi| >= 0.18).
            // Si la racha es >= 3, se exige convicción superlativa (|score| >= 0.32, |current_obi| >= 0.22).
            if unified_intent.signal == SignalType::Short {
                let streak = self.feature_engines[coin_id].get_active_directional_streak(false);
                if streak >= 2 {
                    let (min_score, min_obi) = if streak >= 3 {
                        (0.32, 0.22)
                    } else {
                        (0.28, 0.18)
                    };
                    if composite_score > -min_score || current_obi > -min_obi {
                        unified_intent = SignalIntent::flat();
                    }
                }
            } else if unified_intent.signal == SignalType::Long {
                let streak = self.feature_engines[coin_id].get_active_directional_streak(true);
                if streak >= 2 {
                    let (min_score, min_obi) = if streak >= 3 {
                        (0.32, 0.22)
                    } else {
                        (0.28, 0.18)
                    };
                    if composite_score < min_score || current_obi < min_obi {
                        unified_intent = SignalIntent::flat();
                    }
                }
            }

            // D-475: Escudo Invariante de Microestructura L2 Universal (Cross-Horizon OBI Veto)
            // Prohibido abrir Long si el libro L2 muestra presión vendedora pasiva y prohibido abrir
            // Short si muestra soporte comprador pasivo, salvo capitulación/euforia estadística.
            // D-688 (DÉCIMA OLA): la presión se exigía con |OBI| > 0,10 y la excepción con
            // «|Z| > 2,5», que en realidad eran 2,5 ATR. Ahora la presión debe superar z95 veces
            // la desviación del ruido del propio libro, y la excepción es el mismo z95 sobre la
            // distancia a la EMA de 21 velas. Durante el calentamiento del ruido no hay veto.
            let obi_pressure_threshold = self.feature_engines[coin_id]
                .obi_noise
                .sd()
                .map(|sd| crate::diffusion::Z95 * sd)
                .unwrap_or(f64::INFINITY);
            if unified_intent.signal == SignalType::Long && current_obi < -obi_pressure_threshold {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                if crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) >= -crate::diffusion::Z95 {
                    unified_intent = SignalIntent::flat();
                }
            } else if unified_intent.signal == SignalType::Short && current_obi > obi_pressure_threshold {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                if crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) <= crate::diffusion::Z95 {
                    unified_intent = SignalIntent::flat();
                }
            }

            // D-473 & D-477: Escudo Invariante Neuronal DarkAlpha Universal (Cross-Horizon ML Filter)
            // Veto estricto si el modelo ML predice activamente en contra de la dirección deseada
            // D-688 (DÉCIMA OLA): «en contra» era una banda 0,460/0,540 sin derivación. La regla
            // declarada es que el modelo prediga contra la dirección: P(sube) < ½ para un largo y
            // > ½ para un corto. La excepción de extensión pasa a z95, como el resto del motor.
            if unified_intent.signal == SignalType::Long && ml_prob < 0.5 {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                if crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) >= -crate::diffusion::Z95 {
                    unified_intent = SignalIntent::flat();
                }
            } else if unified_intent.signal == SignalType::Short && ml_prob > 0.5 {
                let cur_atr = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
                let ema_ref = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                    self.feature_engines[coin_id].kline_ema_slow
                } else {
                    self.feature_engines[coin_id].ema_slow
                };
                let p_stretch = if ema_ref > 0.0 {
                    (mid_price - ema_ref) / cur_atr
                } else {
                    0.0
                };
                if crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) <= crate::diffusion::Z95 {
                    unified_intent = SignalIntent::flat();
                }
            }

            let mut new_order = None;

            // --- APERTURA CONTINUA UNIFICADA (100% CAPITAL ALLOCATION) ---
            if unified_intent.signal != SignalType::Flat && !coin.positions.position.is_open() {
                // D-619 (DÉCIMA OLA): la confianza que entra en el Kelly era una
                // puntuación heurística tratada como probabilidad. La posición guarda
                // la puntuación cruda para que el calibrador aprenda de ella.
                // D-690: la probabilidad calibrada NO sustituye a la puntuación. Con
                // ella en `confidence`, el gate del risk-engine (en la escala de la
                // puntuación) lo rechazaba todo tras las primeras pérdidas y el
                // calibrador —que sólo aprende de lo ejecutado— dejaba de recibir
                // datos: 2 operaciones por ventana frente a 56 y 40. Viaja aparte y
                // sólo alimenta el Kelly, donde reducir tamaño no crea un estado
                // absorbente. Calibrar la selección exige resultados contrafactuales.
                let raw_confidence_score = unified_intent.confidence;
                let mut calibrated_intent = unified_intent;
                calibrated_intent.win_probability =
                    self.confidence_calibrator.calibrate(raw_confidence_score);
                let order =
                    self.risk_engine
                        .evaluate_quantum_order(coin_id, &calibrated_intent, &self.arena);
                if order.signal != SignalType::Flat {
                    let drawdown = if self.risk_engine.peak_capital > 0.0 {
                        ((self.risk_engine.peak_capital - current_cap)
                            / self.risk_engine.peak_capital)
                            .clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let vpin_risk = self.feature_engines[coin_id]
                        .cvpin
                        .current_vpin()
                        .clamp(0.0, 1.0);
                    let (impulse_mom, _) = self.lead_lag_engine.predict_altcoin_impulse(obi);
                    let graph_corr = impulse_mom.clamp(-1.0, 1.0);

                    let current_spread_bps = if mid_price > 1e-8 && ask >= bid {
                        ((ask - bid) / mid_price) * 10_000.0
                    } else {
                        0.5
                    };
                    let slip_bps = ((current_spread_bps * 0.5) + 0.5).clamp(0.5, 500.0);

                    let council_horizon =
                        metacortex_engine::consejo_seniors::TradingHorizon::Continuous;

                    let council_snapshot =
                        metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                            horizon: council_horizon,
                            book_imbalance: obi,
                            hurst_exponent: hurst_val.clamp(0.0, 1.0),
                            graph_correlation: graph_corr,
                            do_calculus_risk: vpin_risk,
                            causal_veto_threshold: 0.75,
                            current_drawdown_pct: drawdown,
                            estimated_slippage_bps: slip_bps,
                        };
                    let wr = coin.metrics.win_rate.load(Ordering::Relaxed);
                    let senior_sigs = self
                        .consejo_deliberacion
                        .extract_senior_signals(&council_snapshot, wr);
                    if coin_id < self.last_scalp_senior_signals.len() {
                        self.last_scalp_senior_signals[coin_id] = senior_sigs;
                    }
                    if coin_id < self.last_swing_senior_signals.len() {
                        self.last_swing_senior_signals[coin_id] = senior_sigs;
                    }
                    let deliberation = self.consejo_deliberacion.deliberar_with_weights(
                        &council_snapshot,
                        wr,
                        None,
                    );
                    if !deliberation.approved {
                        self.diag_council_vetoes += 1;
                    }

                    if deliberation.approved {
                        let is_long = order.signal == SignalType::Long;
                        let total_used = self.arena.used_margin.load(Ordering::Relaxed);
                        let free_cap = (current_cap - total_used).max(0.0);

                        let eff_leverage = order.leverage.clamp(1.0, 50.0);
                        // D-634/D-635 (DÉCIMA OLA): aquí se revalidaba la orden con
                        // literales propios —notional 5,05, techo de 50 000 y colchón
                        // 0,98— que ignoraban el gen `margin_cushion_pct` que el
                        // risk-engine acababa de respetar, y ganaba el más permisivo.
                        // El risk-engine valida notional mínimo, colchón y límite por
                        // operación con UNA función compartida; aquí sólo queda lo que
                        // él no puede ver: el margen que otras monedas hayan comprometido
                        // desde que validó la orden.
                        let min_notional = risk_engine::capital_regime::effective_min_notional(
                            quantum_arena::symbol_registry::try_spec(coin_id)
                                .map(|s| s.min_notional)
                                .unwrap_or(0.0),
                        );
                        let cushion = risk_engine::capital_regime::margin_cushion(
                            self.arena.config.margin_cushion_pct.load(Ordering::Relaxed),
                            risk_engine::capital_regime::micro_weight(current_cap, min_notional),
                        );
                        let margin_req = order.volume_usd;
                        if margin_req > 0.0 && margin_req <= free_cap {
                            if margin_req * eff_leverage >= min_notional
                                && total_used + margin_req <= current_cap * cushion
                            {
                                self.diag_opened += 1;
                                self.diag_dir.record_open(is_long);
                                self.arena
                                    .used_margin
                                    .fetch_add(margin_req, Ordering::Relaxed);

                                let base_price = if is_long { ask } else { bid };
                                let nominal_size = margin_req * eff_leverage;
                                // O-01/O-02 — REALITY PHYSICS CONECTADO: el
                                // fill lineal 0.5bps/$1M daba impacto ≈0 a
                                // escala operativa (un edge de pocos bps era
                                // "certificado" sin fricción). Ahora:
                                // impacto CUADRÁTICO (powf 1.2) que castiga
                                // tamaños grandes + latency-slippage (el
                                // precio se mueve durante la latencia) +
                                // base_slippage_floor del GENOMA. Puede volcar
                                // el signo del PnL certificado — es el punto.
                                // N-01 — FIX UNIDADES: atr_pct YA es la
                                // fracción que calculate_market_entry espera
                                // (sus tests usan 0.001-0.002). Multiplicar
                                // por mid_price producía unidades ABSOLUTAS
                                // (BTC ~60) que saturaban el latency-slippage
                                // al clamp del 5% en TODO trade.
                                let tick_vol = atr_pct;
                                let slip_floor = self
                                    .arena
                                    .config
                                    .base_slippage_floor
                                    .load(Ordering::Relaxed)
                                    .max(0.00001);
                                let lat_ms = self
                                    .arena
                                    .config
                                    .latency_penalty_ms
                                    .load(Ordering::Relaxed)
                                    .max(0.0);
                                let (real_entry_price, phys_entry_fee) =
                                    self.reality.calculate_market_entry(
                                        base_price,
                                        is_long,
                                        nominal_size,
                                        tick_vol,
                                        slip_floor,
                                        lat_ms,
                                    );
                                let real_entry_price = if real_entry_price <= 0.0 {
                                    base_price
                                } else {
                                    real_entry_price
                                };

                                // H-4: usar el fee de la FÍSICA — antes se recalculaba aparte
                                let fee_paid = phys_entry_fee;
                                self.arena
                                    .unified_capital
                                    .fetch_add(-fee_paid, Ordering::Relaxed);

                                let qty = nominal_size / real_entry_price;

                                let pos_h = quantum_arena::position::PositionHorizon::Continuous;

                                coin.positions.position.open_with_fee(
                                    is_long,
                                    real_entry_price,
                                    qty,
                                    margin_req,
                                    event_time_ms,
                                    order.tp_target,
                                    order.sl_target,
                                    pos_h,
                                    ml_prob,
                                    raw_confidence_score,
                                    fee_paid,
                                );
                                // REHAB-1b: la posición NACE con su τ dominante
                                // VIVA del espectro — horizonte continuo real,
                                // no etiqueta. Los cierres (temporal_s lerp)
                                // podrán leerla directamente.
                                let tau_entry = self
                                    .temporal_spectrum
                                    .get(coin_id)
                                    .map(|s| s.dominant_tau_ms as u64)
                                    .unwrap_or(0);
                                coin.positions
                                    .position
                                    .entry_tau_ms
                                    .store(tau_entry, Ordering::Relaxed);

                                new_order = Some((
                                    is_long,
                                    real_entry_price,
                                    qty,
                                    order.tp_target,
                                    order.sl_target,
                                ));

                                if self.diag_opened <= 100 {
                                    println!(
                                        "🚀 [OPEN TRACE] #{} dir={} rama={:.0} h={:?} conf={:.4} lev={:.1}x margin=${:.2} notional=${:.2} sc={:.3} obi={:.3} ht={:.5} st={:.5} mac={:.5} mic={:.5} ts={}",
                                        self.diag_opened,
                                        if is_long { "LONG" } else { "SHORT" },
                                        unified_intent.volume_flow_rate,
                                        unified_intent.horizon,
                                        unified_intent.confidence,
                                        eff_leverage,
                                        margin_req,
                                        nominal_size,
                                        composite_score,
                                        current_obi,
                                        higher_trend,
                                        secular_trend,
                                        macro_trend,
                                        micro_trend,
                                        event_time_ms
                                    );
                                }
                            }
                        }
                    }
                }
            }

            // --- 3. MARKET MAKING ---
            let mut final_maker_quote = None;
            let latency_ms = self.arena.last_ws_latency_ms.load(Ordering::Relaxed);

            if latency_ms <= 25 {
                let volatility = self.feature_engines[coin_id].get_features()[3] as f64;

                let mut inventory_delta_usd = 0.0;
                if coin.positions.position.is_open() {
                    let is_long = coin.positions.position.is_long.load(Ordering::Relaxed);
                    let notional =
                        coin.positions.position.quantity.load(Ordering::Relaxed) * mid_price;
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

            (new_order, closed_order, final_maker_quote)
        })
    }

    /// Procesa un tick y devuelve las órdenes generadas (si hay).
    /// Retorna: (new_order, closed_order, maker_quote)
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
        Option<(bool, f64, f64, f64, f64)>,
        Option<(bool, f64, f64)>,
        Option<MakerQuote>,
    ) {
        self.process_tick_dual(
            coin_id,
            bid,
            ask,
            bid_qty,
            ask_qty,
            event_time_ms,
            omni_features,
        )
    }
}

/// D-609 (DÉCIMA OLA) — modulación continua de la duración esperada por el
/// exponente de Hurst.
///
/// Reproduce los extremos calibrados del diseño anterior —×2 a H = 0,52, ×½ a
/// H = 0,48— con `factor = 2^((H − 0,5)/0,02)` acotado a [½, 2], sin escalones.
/// El recorte a [15 s, 1 h] se aplica sólo en la dirección del cambio, de modo
/// que a H = 0,5 la duración no se altera. Una duración no declarada (0) se deja
/// en 0: el horizonte lo decide entonces el mapeo τ único del sistema.
pub fn hurst_duration_modulation(expected_duration_ms: u64, hurst: f64) -> u64 {
    if expected_duration_ms == 0 || !hurst.is_finite() {
        return expected_duration_ms;
    }
    let dur = expected_duration_ms as f64;
    let factor = 2f64.powf((hurst - 0.5) / 0.02).clamp(0.5, 2.0);
    let modulated = dur * factor;
    let bounded = if factor > 1.0 {
        modulated.min(3_600_000.0).max(dur)
    } else {
        modulated.max(15_000.0).min(dur)
    };
    bounded.round() as u64
}

#[cfg(test)]
mod tests_d609 {
    use super::hurst_duration_modulation;

    #[test]
    fn conserva_los_extremos_calibrados() {
        assert_eq!(hurst_duration_modulation(600_000, 0.52), 1_200_000);
        assert_eq!(hurst_duration_modulation(600_000, 0.48), 300_000);
        assert_eq!(hurst_duration_modulation(600_000, 0.50), 600_000);
    }

    #[test]
    fn es_continua_alrededor_de_los_antiguos_escalones() {
        for &h in &[0.48, 0.52] {
            let a = hurst_duration_modulation(600_000, h - 1e-6) as f64;
            let b = hurst_duration_modulation(600_000, h + 1e-6) as f64;
            assert!((a - b).abs() / 600_000.0 < 1e-3, "salto en H = {h}: {a} vs {b}");
        }
    }

    #[test]
    fn no_inventa_duracion_para_intenciones_sin_duracion() {
        assert_eq!(hurst_duration_modulation(0, 0.30), 0);
    }

    #[test]
    fn respeta_los_limites_solo_en_la_direccion_del_cambio() {
        assert_eq!(hurst_duration_modulation(7_200_000, 0.60), 7_200_000);
        assert_eq!(hurst_duration_modulation(10_000, 0.40), 10_000);
    }
}

/// D-685 (DÉCIMA OLA) — CONTENCIÓN DE LA RAMA DE CONSENSO TENSORIAL (rama 14).
///
/// Cuando ninguna otra rama emite señal, el consenso tensorial abre una
/// posición si hay tendencia confirmada, EMAs ordenadas y confianza suficiente,
/// sin ninguna medida de cuánto se ha extendido ya la tendencia. En la
/// verificación forense perdió en todas las corridas medidas, con aperturas en
/// tendencias ya desplazadas (|st| medio 1,6–5,5 %).
///
/// Una guarda de extensión en unidades de difusión (z > 1,96 respecto a la EMA
/// de 12 h) resultó inerte: sus aperturas quedan justo por debajo del umbral, y
/// bajarlo sería ajustar un literal al backtest. La decisión se tomó con una
/// regla fijada antes de ver los datos —elegir por aptitud en entrenamiento
/// (ticks [0, 628 992)) y adoptar sólo si la validación ([628 992, 1 048 320))
/// no empeora—, y se repitió sobre el motor con el lote de literales:
///
/// | Motor | Variante | Entrenamiento (aptitud) | Validación (aptitud) |
/// |---|---|---|---|
/// | 16f37ccd | activa | −17,92 % · 65 ops · DD 17,9 % (−0,287) | −5,15 % · 37 ops (−0,070) |
/// | 16f37ccd | contenida | −5,15 % · 45 ops · DD 6,0 % (−0,063) | −4,84 % · 36 ops (−0,061) |
/// | 3f484361 | activa | −14,86 % · 64 ops · DD 15,3 % (−0,226) | −8,04 % · 41 ops (−0,108) |
/// | 3f484361 | contenida | −5,62 % · 56 ops · DD 6,7 % (−0,070) | −7,14 % · 40 ops (−0,090) |
///
/// El código se conserva para su rediseño: una entrada de consenso necesita
/// una medida de ventaja propia, validada del mismo modo, antes de reactivarse.
const CONSENSUS_BRANCH_ENABLED: bool = false;
