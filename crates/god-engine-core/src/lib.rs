#![feature(portable_simd)]

/// CERT-M5-H01 — FUNCIÓN DE FITNESS UNIFICADA para Darwin (el core no puede
/// importar evolution-engine::fitness sin ciclo de dependencias). Misma
/// matemática que fitness::compute: crecimiento logarítmico (utilidad Kelly)
/// penalizado por drawdown². Inacción = INVIABLE (f64::NEG_INFINITY).
///
/// M5-H01 (2026-09-19): alineado con el canónico en los DOS mecanismos que
/// faltaban —
/// (1) gate min_trades: <30 cierres ⇒ evidencia insuficiente ⇒ INVIABLE
///     (D-654). 30 = WF_MIN_TRADES (online_daemon), el mismo estándar que la
///     promoción walk-forward exige al candidato vivo. Antes, un genoma con
///     1-2 trades de suerte podía liderar el leaderboard de Darwin.
/// (2) dd.clamp(0,1): λ·dd² no crece más allá del ancla "dd 50% = duplicar
///     capital" (paridad fitness::compute).
/// El factor OOS del canónico NO se replica: los evaluadores de período
/// único lo desactivan por convención (oos_start == oos_end ⇒ factor 1.0;
/// ver evolver.rs:443 "sin split IS/OOS aquí").
#[inline]
pub fn fitness_compute(initial: f64, final_cap: f64, max_dd: f64, total_trades: u32) -> f64 {
    if initial <= 0.0 || !initial.is_finite() || !final_cap.is_finite() || final_cap <= 0.0 {
        return f64::NEG_INFINITY; // ruina o datos inválidos
    }
    const MIN_TRADES: u32 = 30;
    if total_trades < MIN_TRADES {
        return f64::NEG_INFINITY; // D-654: poca actividad = ruido, no edge
    }
    let growth = (final_cap / initial).ln();
    // λ = 4·ln(2): el dd del 50% cuesta exactamente lo que duplicar capital gana
    const DRAWDOWN_LAMBDA: f64 = 2.772_588_722_239_781;
    let dd = max_dd.clamp(0.0, 1.0);
    growth - DRAWDOWN_LAMBDA * dd * dd
}

pub mod bootloader;
pub mod calibration;
pub mod conformal;
pub mod darwin;
pub mod diffusion;
pub mod direction_diag;
pub mod ensemble;
pub mod latency_accelerator;
pub mod liquidation_feed;
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
    /// F8 — ESPECTRO TEMPORAL CONTINUO por símbolo: 32 escalas log-espaciadas
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
    pub last_fast_intent: Vec<SignalIntent>,
    pub last_slow_intent: Vec<SignalIntent>,
    /// U-2: señales del consejo que acompañan la ÚLTIMA intención evaluada
    /// (la del motor continuo — ya no hay dualidad de slots por horizonte).
    pub last_senior_signals: Vec<[f64; 11]>,
    pub lakehouse: Option<Arc<storage_engine::LakehouseWarehouse>>,
    pub consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion,
    pub lead_lag_engine: feature_engine::LeadLagAlphaEngine,
    pub ppo_engine: dark_alpha_engine::online_ppo::OnlinePpoPolicyEngine,
    pub online_learner: metacortex_engine::online_learning::OnlineLearningModule,
    /// #26: Sistema inmune vivo para registro y amortiguación de traumas de predicción
    pub immune_system: metacortex_engine::immune_system::LivingImmuneSystem,
    /// #15: Auditor forense ShadowGraph para detección lock-free de concept drift
    pub shadow_auditor: std::sync::Arc<metacortex_engine::shadow_graph_auditor::ShadowGraphAuditor>,
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
    /// CERT-M2-H03 — calibradores PER-SÍMBOLO: los globales arriba eran
    /// alimentados por TODAS las monedas (exchangeability rota cross-asset:
    /// el conformal aprendía un blend BTC+NEAR+ATOM). El índice es coin_id;
    /// si el slot no existe, se cae al global (compat).
    pub conformal_by_coin: Vec<conformal::ConformalCalibrator>,
    pub calibrator_by_coin: Vec<calibration::PlattCalibrator>,
    /// CERT-M2-C02 — proceso de Hawkes POR SÍMBOLO. Cada TRADE real excita
    /// el proceso (record_event); λ/μ verdadero se publica al registry.
    /// Antes el slot 'hawkes_intensity' llevaba un proxy de aceleración/ATR
    /// (mislabel documentado por la auditoría décima) y la matemática
    /// Σα·e^(−βΔt) de hawkes_bessel.rs vivía muerta sin callers.
    pub hawkes_by_coin: Vec<signal_engine::hawkes_bessel::HawkesBesselEngine>,
    /// DIAG R4 (transitorio): cuello post-orden.
    pub diag_council_vetoes: u64,
    pub diag_opened: u64,
    /// B3.18 — entradas vetadas por el gate de ensamble (la predicción
    /// decidió NO): diagnóstico de cuánto consume el sistema la predicción.
    pub diag_ml_vetoes: u64,
    /// MOD2/7-010 (DEC-14): declarados, exportados al backtest de evolución
    /// y JAMÁS incrementados — telemetría estructuralmente falsa (siempre 0).
    /// Conservados por compatibilidad del struct; su eliminación pertenece
    /// a la FASE 2 de erradicación swing/scalp.
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
                let legacy_path = "models/BTCUSDT_MOTOR.json";
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
            last_fast_intent: vec![SignalIntent::flat(); n_coins],
            last_slow_intent: vec![SignalIntent::flat(); n_coins],
            last_senior_signals: vec![[0.0; 11]; n_coins],
            lakehouse: None,
            consejo_deliberacion: metacortex_engine::consejo_seniors::ConsejoDeliberacion::new(),
            lead_lag_engine: feature_engine::LeadLagAlphaEngine::new(50),
            ppo_engine,
            online_learner,
            immune_system: metacortex_engine::immune_system::LivingImmuneSystem::new("."),
            shadow_auditor: std::sync::Arc::new(
                metacortex_engine::shadow_graph_auditor::ShadowGraphAuditor::new(),
            ),
            applied_generation: std::sync::atomic::AtomicU64::new(0),
            genomes_mtime: None,
            conformal: conformal::ConformalCalibrator::new(),
            confidence_calibrator: calibration::PlattCalibrator::new(),
            conformal_by_coin: (0..n_coins)
                .map(|_| conformal::ConformalCalibrator::new())
                .collect(),
            calibrator_by_coin: (0..n_coins)
                .map(|_| calibration::PlattCalibrator::new())
                .collect(),
            hawkes_by_coin: (0..n_coins)
                .map(|_| signal_engine::hawkes_bessel::HawkesBesselEngine::new())
                .collect(),
            diag_council_vetoes: 0,
            diag_opened: 0,
            diag_ml_vetoes: 0,
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

    /// #15: Evalúa si el drift de predicción o las anomalías acumuladas requieren reentrenamiento urgente
    pub fn evaluate_system_drift(&self) -> bool {
        self.shadow_auditor.evaluate_system_drift()
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
        for p in [&coin.positions.scalp, &coin.positions.swing, &coin.positions.position] {
            if p.is_open() {
                let (_is_long, _entry_price, _qty, margin_used, entry_fee) =
                    p.close_with_fee();
                if margin_used > 0.0 {
                    let _ = self.arena.used_margin.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |v| Some((v - margin_used).max(0.0)),
                    );
                }
                if entry_fee > 0.0 {
                    self.arena
                        .unified_capital
                        .fetch_add(entry_fee, Ordering::Relaxed);
                }
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
        // C-02 (DEC-14) — MAPA DE FEATURES VIVAS vs MUERTAS del bloque
        // omni[34..54] y CONTRATO DE SERVIDO: toda feature SIN PRODUCTOR
        // en vivo se sirve como 0.0 (centinela "sin dato"), NUNCA como un
        // literal plausible (1.05, 1.00…). El NN fue entrenado con cierto
        // rango de valores: un literal que parece dato real contamina la
        // inferencia de forma silenciosa; el cero al menos señala ausencia.
        // Productores vivos confirmados: omni[11] funding (premiumIndex
        // poller) · omni[14] fear&greed (alternative.me) · omni[21-24]
        // VIX/SP500/DXY/NASDAQ (Yahoo B3.23) · omni[26] gold (PAXG
        // Binance). OJO: los defaults del constructor del multiplexer
        // (us10y=4.2, oil=80, fed=5.5…) son NO-cero, así que el guard
        // `> 0.0` los dejaba pasar por la rama "con datos" — por eso las
        // muertas se zeran aquí incondicionalmente. Para re-activar un
        // canal: cablear su productor y restaurar la lectura del slot.
        let stateful_feats = self.feature_engines[coin_id].get_universal_features();
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
        // [40] MUERTA (segunda slot fear_greed sin productor) — era el
        // literal 0.50: servía "neutralidad" inventada al NN.
        combined[40] = 0.0;
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
        // [45] MUERTA (us10Y): omni[25] no tiene productor — el slot carga
        // el literal 4.2 del constructor y el guard `> 0.0` lo servía como
        // dato real (1.05 plausible). Cero incondicional.
        combined[45] = 0.0;
        // [46] gold: VIVA vía PAXG Binance (omni[26]). El fallback pasa a
        // 0.0: si PAXG aún no entregó, "sin dato" — no el 1.0 de antes.
        combined[46] = if omni_features.len() > 26 && omni_features[26] > 0.0 {
            (omni_features[26] / 2500.0).clamp(0.5, 2.0)
        } else {
            0.0
        };
        // [47] MUERTA (oil WTI): omni[27] sin productor — literal 80.0 del
        // constructor servido como 1.00 plausible. Cero incondicional.
        combined[47] = 0.0;
        combined[48] = if omni_features.len() > 11 {
            (omni_features[11] * 1000.0).clamp(-5.0, 5.0)
        } else {
            0.0
        };
        // [49] MUERTA (fed_rate): omni[29] sin productor — literal 5.5 del
        // constructor servido como ~1.05 plausible. Cero incondicional.
        combined[49] = 0.0;
        combined[50] = if omni_features.len() > 14 && omni_features[14] > 0.0 {
            (omni_features[14] / 50.0).clamp(0.0, 2.0)
        } else {
            1.0
        };
        // [51]/[52]/[53] MUERTAS (CVD spot/futuros y basis premium: sin
        // productor). Los slots por defecto valen 0.0 en el multiplexer y
        // tanh(0)=0 / clamp(0)=0, así que HOY ya sirven exactamente 0.0 —
        // se conserva la lectura para que un productor futuro cablee solo,
        // pero cualquier default no-cero del multiplexer contaminaría el
        // tensor de nuevo (ver nota C-02 arriba).
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
        if std::env::var_os("GOD_NO_HOT_RELOAD").is_some() {
            return;
        }
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
                let last = self.applied_generation.load(Ordering::SeqCst);
                if env.generation > last {
                    // CERT-M5-H02: WRITER PROTOCOL — incrementar la generación
                    // ANTES de aplicar (seqlock write-begin). Un reader que
                    // capture la generación ANTES y la re-verifique DESPUÉS
                    // de computar su decisión detectará el swap si la gen
                    // cambió, evitando mezclar leverage viejo con SL nueva.
                    self.applied_generation
                        .store(env.generation + 1, Ordering::SeqCst);
                    env.genome.apply_to_arena(&self.arena);
                    self.applied_generation
                        .store(env.generation, Ordering::SeqCst);
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
        // MOD2/7-004: era el "funding" falso del camino depth (semántica
        // cruzada); el funding real viaja en omni_features[11]. Se conserva
        // el slot para no romper la firma pública del evento unificado.
        _depth_micro_div: f64,
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
            // CERT-M2-H04: ANTES calificaba contra dirección de kline
            // (subió/bajó el 1m bar) — pero el forest MOTOR predice una
            // BARRERA TRIPLE (TP +0.36%/SL −0.18% a horizonte de minutos).
            // Los pesos Hedge castigaban/premiaban al forest por una pregunta
            // que no fue entrenado para responder. Ahora el label usa una
            // aproximación de barrera: retorno del bar comparado contra
            // el umbral de fee+SL (proxy de "superó la barrera" vs "no").
            if is_kline_closed && coin_id < self.kline_close_memory.len() {
                let prev_close = self.kline_close_memory[coin_id];
                if prev_close > 0.0 {
                    // Barrera aproximada: retorno del bar vs umbral de
                    // fricción (SL base del genoma). Si el retorno supera
                    // el umbral en la dirección predicha → win (1.0); si
                    // lo supera en contra → loss (0.0); entre ambos →
                    // neutral descartado (como el trainer descarta neutros).
                    let bar_ret = (current_price - prev_close) / prev_close;
                    let fee_hurdle = self
                        .arena
                        .config
                        .sl_at_tau(30_000.0) // τ rápida: horizonte del 1m bar
                        .max(0.001); // piso 10 bps
                    let y = if bar_ret > fee_hurdle {
                        1.0 // superó la barrera alcista
                    } else if bar_ret < -fee_hurdle {
                        0.0 // superó la barrera bajista
                    } else {
                        // dentro del rango de fricción: neutro, DESCARTAR
                        // (el ensemble no aprende de samples sin resolución)
                        self.kline_close_memory[coin_id] = current_price;
                        // skip update pero actualizar memoria
                        if let Some(spec) = self.temporal_spectrum.get_mut(coin_id) {
                            let _ = spec; // ya actualizado arriba
                        }
                        // continue to next processing without calibrating
                        0.5_f64.signum() * 0.0 // señal neutra — no usada
                    };
                    // Sólo calibrar con samples DECISIVOS (y ∈ {0.0, 1.0})
                    if y == 0.0 || y == 1.0 {
                        if coin_id < self.ensembles.len() {
                            self.ensembles[coin_id].update_with_outcome(y);
                        } else {
                            self.ensemble.update_with_outcome(y);
                        }
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
                // MOD2/7-004 (INFORME DECIMOCUARTO): funding_rate REAL — el
                // omni slot 11 (`agg_funding_rate`) lo produce el poller
                // REST /fapi/v1/premiumIndex (clamp [-1,1], saneado a
                // finito por OmniState::get_features). Antes este camino
                // pasaba `depth_micro_div` como "funding" (semántica
                // cruzada) y el camino por-tick pasaba 0.0 constante:
                // columna muerta del vector. TODO: el productor es
                // BTC-only (premiumIndex?symbol=BTCUSDT) — cablear funding
                // per-símbolo cuando el poller lo publique.
                // dex_severity: SIN productor en vivo (feed MEV/DEX no
                // existe) — se documenta el 0.0 en vez de falsificar señal.
                // QO-U1c: funding PER-SÍMBOLO del registry (escrito por el
                // poller cada 120s); fallback al global omni[11] (BTC).
                let sym_funding = quantum_arena::symbol_registry::try_symbol(coin_id)
                    .map(|s| {
                        self.arena
                            .registry
                            .get_scoped_value_or(&s, "funding_rate", f64::NAN)
                    })
                    .unwrap_or(f64::NAN);
                let funding_rate_live = if sym_funding.is_finite() {
                    sym_funding
                } else {
                    omni_features[11]
                };
                // P-4: severidad de liquidación VIVA (event-driven, swap).
                let liq_severity = crate::liquidation_feed::take_pending();
                self.feature_engines[coin_id].update_macro_features(
                    depth_obi,
                    funding_rate_live,
                    liq_severity,
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
                // CERT-M2-C02: cada trade real EXCITA el proceso de Hawkes
                // del símbolo (λ(t) = μ + Σα·e^(−β(t−tᵢ)) vivo, por fin).
                if let Some(hk) = self.hawkes_by_coin.get_mut(coin_id) {
                    hk.record_event(event_time_ms as f64 / 1000.0);
                }
                // D-220 & D-247: Ingesta física real de microestructura agresora (Taker Buy vs Taker Sell)
                self.feature_engines[coin_id].update_trade_flow(trade_qty, is_buyer_maker);
                // D-708 (DÉCIMA OLA · auditoría integral): EL FLUJO AGREGADO ES
                // ESTADO DEL NÚCLEO, NO DEL LLAMADOR.
                //
                // `agg_buy_vol`/`agg_sell_vol` —de donde sale `rolling_cvd`, que
                // en ausencia de libro SUSTITUYE al micro-score y pesa hasta el
                // 80 % del `composite_score`— los alimentaban los llamadores:
                // `god_engine` y el forense sí, `booktick_replay` (el motor de
                // `backtest_windows` y de `evolution`) NO. Allí el CVD era
                // idénticamente 0, las ramas que exigen |OBI efectivo| por encima
                // de su umbral eran inalcanzables y la aptitud se medía sobre un
                // motor mutilado. Con la actualización aquí, todo llamador
                // alimenta la misma fuente con el mismo dato; las llamadas
                // externas se retiran para no contar dos veces.
                self.arena
                    .update_agg_trade(coin_id, is_buyer_maker, trade_qty);
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
                is_depth, // CERT-M2-H01: el depth path ya actualizó macro arriba
            );

            // Si hay pánico de latencia o es un evento puro de profundidad sin trade,
            // no abrimos nuevas órdenes pero permitimos cierres defensivos (SL/TP/trailing)
            if latency_panic || (is_depth && !is_trade) {
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
        for p in [&coin.positions.scalp, &coin.positions.swing, &coin.positions.position] {
            if p.is_open() {
                telemetry_server::telemetry_log!(
                    "👻 [GHOST REVERT] Revertiendo posición cuántica fantasma para coin {}",
                    coin_id
                );
                let (_, _, _, margin) = p.close();
                if margin > 0.0 {
                    let _ = self.arena.used_margin.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |v| Some((v - margin).max(0.0)),
                    );
                }
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
        _skip_macro_update: bool,
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
            // MOD2/7-004 (INFORME DECIMOCUARTO): funding_rate REAL del omni
            // slot 11 (poller premiumIndex, ver camino is_depth arriba) —
            // antes 0.0 constante: columna muerta del vector 34D/48D.
            // TODO: funding per-símbolo (el productor hoy es BTC-only).
            // dex_severity: sin productor en vivo — 0.0 documentado.
            // CERT-M2-H01: ANTES el camino per-tick llamaba
            // update_macro_features SIEMPRE, duplicando la llamada que el
            // branch is_depth ya hizo arriba (~600) — obi_noise,
            // obi_accel, fr_elasticity se actualizaban al DOBLE en
            // eventos depth pero una vez en trades. Ahora: el per-tick
            // SÓLO actualiza si el camino is_depth NO corrió (para trades
            // y klines); los depth events ya fueron actualizados arriba
            // con el funding per-símbolo (mejor dato).
            if !_skip_macro_update {
                let liq_sev_tick = crate::liquidation_feed::take_pending();
                feature_engine.update_macro_features(obi, omni_features[11], liq_sev_tick, event_time_ms);
            }
            let raw_atr_pct = feature_engine.get_atr_pct();
            let hurst_val = feature_engine.hurst.current();
            let coin = &self.arena.coins[coin_id];
            coin.current_price.store(mid_price, Ordering::Relaxed);
            coin.current_atr
                .store(raw_atr_pct * mid_price, Ordering::Relaxed);
            coin.hurst_exponent.store(hurst_val, Ordering::Relaxed);
            // S-7 — Hurst multifractal SELECCIONADO POR τ: micro (<2min),
            // meso (<1h), macro (≥1h). La geometría TP/SL consume el H del
            // horizonte que el motor opera, no el escalar global.
            {
                let tau_dom_h = self
                    .temporal_spectrum
                    .get(coin_id)
                    .map(|s| s.dominant_tau_ms)
                    .unwrap_or(600_000.0);
                let fe_h = &self.feature_engines[coin_id];
                let h_scale = if tau_dom_h < 120_000.0 {
                    fe_h.hurst_micro
                } else if tau_dom_h < 3_600_000.0 {
                    fe_h.hurst_meso
                } else {
                    fe_h.hurst_macro
                };
                let h_val = if h_scale.is_finite() && h_scale > 0.0 {
                    h_scale
                } else {
                    hurst_val as f32
                };
                coin.hurst_scale_matched
                    .store(h_val as f64, Ordering::Relaxed);
            }

            let mut closed_order = None;

            let atr_pct = self.feature_engines[coin_id].get_atr_pct();
            let min_atr_pct = self
                .arena
                .config
                .dynamic_atr_min
                .load(Ordering::Relaxed)
                .max(0.001);
            let atr_pct_live = atr_pct.max(min_atr_pct);

            // --- 1. GESTIÓN MULTI-HORIZONTE CONTINUA INTEGRAL (SCALP, SWING & CONTINUOUS) ---
            for pos in [&coin.positions.scalp, &coin.positions.swing, &coin.positions.position] {
                if !pos.is_open() {
                    continue;
                }
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

                // #560: Espectro Continuo Universal Multivariante — longitud de onda tau_trade_ms y coordenada s in [0, 1]
                let tau_entry = pos.entry_tau_ms.load(Ordering::Relaxed) as f64;
                let tau_trade_ms = if tau_entry > 0.0 {
                    tau_entry
                } else {
                    self.temporal_spectrum
                        .get(coin_id)
                        .map(|s| s.dominant_tau_ms)
                        .filter(|&t| t > 0.0)
                        .unwrap_or_else(|| {
                            quantum_arena::temporal_spectrum::tau_from_temporal_scale(
                                self.arena.config.temporal_scale.load(Ordering::Relaxed),
                            )
                        })
                };
                let temporal_s = quantum_arena::temporal_spectrum::temporal_scale_from_tau(tau_trade_ms);

                let (sl, tp) = {
                    let pos_tp = pos.tp_price.load(Ordering::Relaxed);
                    let pos_sl = pos.sl_price.load(Ordering::Relaxed);
                    let sl_base = self.arena.config.sl_at_tau(tau_trade_ms).clamp(0.0005, 0.0500);
                    let tp_base = self.arena.config.tp_at_tau(tau_trade_ms).clamp(0.0010, 0.1000);
                    let fallback_sl = sl_base.max(atr_pct * 1.5).clamp(0.0010, 0.0300);
                    let rr_ratio = self
                        .arena
                        .config
                        .tp_rr_ratio_btc
                        .load(Ordering::Relaxed)
                        .clamp(1.0, 10.0);
                    let fallback_tp = tp_base.max(fallback_sl * rr_ratio).clamp(0.0020, 0.0800);
                    // B3.2: el fallback de gestión tampoco propone salidas que
                    // no pagan sus comisiones — pisos de viabilidad por
                    // fricción (mismo invariante que el gate y los brackets).
                    // Los pisos previos (0,05 % / 0,10 %) eran menores que la
                    // fricción roundtrip VIP0: TP garantizado en pérdida neta.
                    // B3.19: + latency_slip (atr·lat/ref), como el gate.
                    let lat_ref_mgmt = self
                        .arena
                        .config
                        .latency_ms_panic_threshold
                        .load(Ordering::Relaxed)
                        .clamp(10.0, 5_000.0);
                    let lat_slip_mgmt = (atr_pct_live
                        * (self
                            .arena
                            .config
                            .latency_penalty_ms
                            .load(Ordering::Relaxed)
                            .max(0.0)
                            / lat_ref_mgmt))
                    .clamp(0.0, 0.05);
                    let fee_rt_mgmt = 2.0 * self.arena.config.live_taker_fee.load(Ordering::Relaxed)
                        + 2.0 * (self
                            .arena
                            .config
                            .base_slippage_floor
                            .load(Ordering::Relaxed)
                            .max(0.00001)
                            + lat_slip_mgmt);
                    let (fallback_sl, fallback_tp) =
                        quantum_arena::genome::SuperGenotype::friction_floors(
                            fee_rt_mgmt,
                            fallback_sl,
                            fallback_tp,
                        );
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
                // Activa cuando el recorrido ya cubre la fricción de ida y
                // vuelta y una fracción espectral del objetivo. S-3
                // (ESPECTRALIZACIÓN): las activaciones de BE y trailing
                // respiran con la persistencia de la escala dominante —
                // tendencial (pers→+1) activa TARDE (deja correr), mean-
                // revert (pers→−1) activa PRONTO (asegura el retroceso).
                // Fracción del TP: BE lerp(0.45,0.65), trail lerp(0.60,0.80);
                // pers=0 ⇒ 0.55/0.70 exactos (comportamiento B3.27).
                //
                // D-727 (DÉCIMA OLA · auditoría integral): LA PROTECCIÓN NO PUEDE
                // ARMARSE POR ENCIMA DEL OBJETIVO. Aquí había dos pisos ABSOLUTOS
                // —62 pb para el breakeven y 72 pb para el trailing— que no
                // dependían del TP ni de la volatilidad. La geometría de entrada
                // fija el objetivo en TP = 5,5·f (fricción): con la fricción de
                // referencia de 10 pb, TP = 0,55 % y el piso de 62 pb quedaba en
                // el 113 % del objetivo — el TP cierra la posición antes de que
                // exista protección alguna, y el trailing era inalcanzable por
                // construcción. Las activaciones son fracciones espectrales del
                // recorrido REAL al objetivo; el suelo es la única magnitud física
                // que justifica mover el stop (que la ganancia cubra la fricción
                // de ida y vuelta, o dos ATR de ruido) y el TECHO es el propio
                // objetivo: nada puede armarse por encima del TP.
                let pers_dom = self
                    .temporal_spectrum
                    .get(coin_id)
                    .map(|spec| spec.persistence_at(tau_trade_ms).clamp(-1.0, 1.0))
                    .unwrap_or(0.0);
                let s_t = (pers_dom + 1.0) * 0.5;
                let be_frac = 0.45 + 0.20 * s_t;
                let trail_frac = 0.60 + 0.20 * s_t;

                let slip_floor = self
                    .arena
                    .config
                    .base_slippage_floor
                    .load(Ordering::Relaxed)
                    .max(0.0001);

                // #544, #548, #555, #559 & #560: Breakeven Físico con Garantía EV >= 0 en Espectro Continuo Multivariante
                // Erradicada la dicotomía discreta (scalping vs swing).
                // A escala rápida (s=0, tau=30s): buffer ágil (7-11 bps) que garantiza ganancia neta post-fees sin redundancia,
                // activándose limpiamente a 12-16 bps (buf + 3.5 bps) para capturar micro-impulsos y blindar los $13 USD.
                // A escala lenta (s=1, tau=12h): buffer amplio (22-35 bps) y activación escalada con el objetivo TP.
                // En todo el continuo s in [0, 1]: interpolación suave lerp(fast, slow, s) sin escalones ni acantilados.
                // VIP0 Binance taker fee = 0.05% (5 bps). Roundtrip = 10 bps. Slippage floor + taker impact = ~3.0 bps.
                // Total friction real = ~13.0 bps.
                let buf_fast = (live_fee * 1.25 + slip_floor * 1.2).clamp(0.00130, 0.00160);
                let min_breathing_fast = (atr_pct_live * 0.60).clamp(0.00040, 0.00080);
                let act_fast = (buf_fast + min_breathing_fast).clamp(0.00170, 0.00220);

                let buf_slow = (live_fee * 1.50 + slip_floor * 1.5).clamp(0.0018, 0.0028);
                let act_slow = (tp * be_frac * 0.70)
                    .max(buf_slow + 0.0010)
                    .min(0.0050);

                let be_buffer = (1.0 - temporal_s) * buf_fast + temporal_s * buf_slow;
                let be_activation = (1.0 - temporal_s) * act_fast + temporal_s * act_slow;

                if peak_pnl >= be_activation {
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

                // 2. Trailing Stop Ratchet Espectral Continuo (#559, #560)
                let trail_act_fast = (be_activation + min_breathing_fast * 0.5).clamp(0.00250, 0.00320);
                let trail_act_slow = (tp * trail_frac).max(be_activation * 1.25).min(tp * 0.95);
                let trail_activation_pnl = (1.0 - temporal_s) * trail_act_fast + temporal_s * trail_act_slow;
                let trail_active = peak_pnl >= trail_activation_pnl;

                let mut force_close_trail = false;

                if trail_active {
                    let (trail_atr_mult, trail_act, trail_step, trail_max) =
                        self.arena.config.trail_params_at_tau(tau_trade_ms);

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
                        tp, // B3.27 — escalera relativa al TP
                        // S-2 / #560: persistencia continua de la longitud de onda tau_trade_ms
                        self.temporal_spectrum
                            .get(coin_id)
                            .map(|spec| {
                                spec.persistence_at(tau_trade_ms).clamp(-1.0, 1.0)
                            })
                            .unwrap_or(0.0),
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

                let macro_t = self.feature_engines[coin_id].get_macro_trend();
                // D-455: Inversión de tendencia genuina (45 bps de pendiente EMA) en escala macro
                let trend_reversed =
                    (is_long && macro_t < -0.0045) || (!is_long && macro_t > 0.0045);

                // D-492: Dynamic Adverse Order Flow Stop Cutting (Toxic Flow Cutoff)
                let micro_t = self.feature_engines[coin_id].get_micro_trend();
                let ema_ofi = self.feature_engines[coin_id].ofi_model.ema_ofi;
                let cur_vpin = self.feature_engines[coin_id].cvpin.current_vpin();

                let ofi_adverse = (is_long && (ofi_value < -0.25 || ema_ofi < -0.20))
                    || (!is_long && (ofi_value > 0.25 || ema_ofi > 0.20));
                let trend_adverse = (is_long && micro_t < -0.00045)
                    || (!is_long && micro_t > 0.00045);

                // D-492 / #548: toxic_cut_sl relativo al SL de la posición (95% del SL, o min 3.0x live fee)
                // Evita cortes apresurados a -24 bps dentro del ruido difusivo natural.
                // Exige VPIN severo (> 0.85) y desbalance real de OFI adverso.
                let toxic_cut_sl = (sl * 0.95).max(live_fee * 3.0);
                let toxic_flow_exit = pnl_pct <= -toxic_cut_sl && cur_vpin > 0.85 && ofi_adverse;

                // #548, #556 & #560: Alpha Decay y Peak Harvest Continuo Multivariante
                // La predictibilidad de microestructura y flujo se extingue continuamente
                // según la longitud de onda tau_trade_ms y su coordenada espectral temporal_s.
                let harvest_age_ms = (tau_trade_ms * (0.8 + 1.0 * temporal_s)).clamp(120_000.0, 7_200_000.0) as u64;
                let peak_harvest_thresh = (act_fast * 0.95).max(0.00120) + 0.0035 * temporal_s;
                let min_stagnant_ms = (tau_trade_ms * (4.0 + 3.0 * temporal_s)).clamp(240_000.0, 14_400_000.0) as u64;
                let hard_stagnant_ms = (tau_trade_ms * (6.0 + 5.0 * temporal_s)).clamp(450_000.0, 28_800_000.0) as u64;
                let absolute_trade_life_ms = (tau_trade_ms * (10.0 + 10.0 * temporal_s)).clamp(900_000.0, 86_400_000.0) as u64;

                let mut is_peak_harvest = false;
                let alpha_decay_exit = if event_time_ms > 0 {
                    let harvest_ratio = if peak_pnl >= 0.0030 {
                        0.78
                    } else if peak_pnl >= 0.0020 {
                        0.70
                    } else {
                        0.60
                    };
                    let peak_harvest_decay = position_age_ms > harvest_age_ms
                        && peak_pnl >= peak_harvest_thresh
                        && pnl_pct >= be_buffer
                        && pnl_pct <= (peak_pnl * harvest_ratio).max(be_buffer);

                    if peak_harvest_decay {
                        is_peak_harvest = true;
                    }

                    // NUNCA liquidar un trade por fluctuación normal de spread (-2 bps).
                    // Solo cerrar si la tesis direccional se rompió con significancia estadística (> 70% del SL con flujo y microtendencia en contra)
                    // o si superó su tiempo máximo de vida física.
                    let time_stagnant_decay = if position_age_ms > min_stagnant_ms {
                        let ema_ofi_adverse = (is_long && ema_ofi < -0.30) || (!is_long && ema_ofi > 0.30);
                        let thesis_broken = pnl_pct < -sl * 0.70 && ema_ofi_adverse && trend_adverse;
                        let stillborn_cut = peak_pnl <= 0.0001 && pnl_pct < -sl * 0.75 && ema_ofi_adverse;
                        let time_expired = position_age_ms > hard_stagnant_ms && pnl_pct < -sl * 0.50;
                        let absolute_expired = position_age_ms > absolute_trade_life_ms;

                        thesis_broken || stillborn_cut || time_expired || absolute_expired
                    } else {
                        false
                    };

                    peak_harvest_decay || time_stagnant_decay
                } else {
                    false
                };

                // D-649 (DÉCIMA OLA), #548 & #560: Timeout y Zombi adaptativo continuo por escala espectral
                let dynamic_zombie_debounce_ms = (tau_trade_ms * (3.0 + 2.0 * temporal_s)).clamp(300_000.0, 7_200_000.0) as u64;
                let dynamic_hard_timeout_ms = (tau_trade_ms * (6.0 + 6.0 * temporal_s)).clamp(600_000.0, 21_600_000.0) as u64;
                let absolute_expiry_ms = (tau_trade_ms * (10.0 + 10.0 * temporal_s)).clamp(900_000.0, 43_200_000.0) as u64;
                let z_loss_hard = (sl * (0.75 + 0.15 * temporal_s)).max(live_fee * 2.0);
                let z_loss_trend = (sl * (0.65 + 0.15 * temporal_s)).max(live_fee * 2.0);

                let expired_by_age =
                    event_time_ms > 0 && position_age_ms > absolute_expiry_ms && pnl_pct <= 0.0;
                let hard_timeout = position_age_ms > dynamic_hard_timeout_ms && pnl_pct <= -z_loss_hard;
                let is_zombie = expired_by_age
                    || (event_time_ms > 0
                        && position_age_ms > dynamic_zombie_debounce_ms
                        && ((trend_reversed && pnl_pct <= -z_loss_trend) || hard_timeout));

                let tp_traded_through = if is_long {
                    bid >= entry * (1.0 + tp)
                } else {
                    ask <= entry * (1.0 - tp)
                };

                if tp_traded_through
                    || pnl_pct <= -sl
                    || trail_hit
                    || force_close_trail
                    || alpha_decay_exit
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
                        // B3.19 — writer de zombie_promotions (auditoría: el
                        // contador existía y /api/state lo leía, pero NADIE
                        // lo escribía — zombie_count siempre 0). U-1: métrica
                        // unificada (el slot gemelo scalp ya no existe).
                        coin.metrics
                            .zombie_promotions
                            .fetch_add(1, Ordering::Relaxed);
                        (5u8, "ZOMBIE")
                    } else if toxic_flow_exit {
                        (6u8, "TOXIC_FLOW")
                    } else if is_peak_harvest {
                        (8u8, "PEAK_HARVEST")
                    } else {
                        (7u8, "ALPHA_DECAY")
                    };

                    if self.diag_close_total < 100 {
                        println!(
                            "🚪 [CLOSE TRACE] #{} reason={} pnl_pct={:.4}% peak={:.4}% age={}s gross_pnl=${:.4} exit={:.2} entry={:.2} h={:?} ts={}",
                            self.diag_close_total,
                            reason,
                            pnl_pct * 100.0,
                            pos.max_pnl_pct.load(Ordering::Relaxed) * 100.0,
                            position_age_ms / 1000,
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

                    // B3.14 — ¿la entrada EXISTIÓ en el exchange? La
                    // posición local nace ANTES de la ejecución asíncrona;
                    // si la entrada fue vetada/rechazada y aun así el ciclo
                    // del core la cerró, el PnL es PAPEL y NO contabiliza
                    // (caso KOMA: +$36 con WR 1.0 jamás operados). El swap
                    // consume la confirmación; el host lee
                    // last_close_confirmed para su propia contabilidad.
                    let was_exchange_confirmed =
                        pos.exchange_confirmed.swap(false, Ordering::Relaxed);
                    pos.last_close_confirmed
                        .store(was_exchange_confirmed, Ordering::Relaxed);

                    let net_realized_pnl = gross_pnl - close_fee;
                    let net_trade_pnl = net_realized_pnl - entry_fee_paid;

                    // D-731: la rama `else` ponía el acumulador GLOBAL a cero
                    // —borrando el margen de las demás monedas— cuando el
                    // acumulado era menor que el margen de esta posición, que es
                    // justo el síntoma de una carrera previa. Resta atómica
                    // acotada en cero, sin tocar lo ajeno.
                    let _ = self.arena.used_margin.fetch_update(
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                        |v| Some((v - margin_used).max(0.0)),
                    );

                    // FASE 23 / F-014 / C-07 (INFORME DECIMOCUARTO): métricas
                    // continuas unificadas — el PnL se escribe SOLO en
                    // coin.metrics (fuente única de verdad). Las escrituras
                    // gemelas a coin.scalp/coin.swing contaban cada trade TRES
                    // veces y hacían mentir el comentario F-014 de arriba;
                    // quedan en 0 (los readers D-441 de telemetría ya caen al
                    // fallback contra coin.metrics cuando scalp == 0).
                    // B3.14: SOLO posiciones cuya entrada existió en el exchange.
                    if was_exchange_confirmed {
                        // D-739 (DÉCIMA OLA · auditoría integral): el MISMO PnL se
                        // escribía en las tres celdas —`metrics`, `scalp` y
                        // `swing`—, de modo que una operación de +1,00 USD
                        // producía 1,00 en cada una: los consumidores que suman
                        // las dos piernas (el panel, el bus mmap, el simulador
                        // multiactivo) veían el DOBLE del PnL real, y el desglose
                        // por horizonte era ficción — dos motores con idéntico
                        // resultado donde sólo hubo una operación. El motor es
                        // continuo: la única celda es `metrics`, como ya declaraba
                        // el comentario de F-014 unas líneas más abajo.
                        coin.metrics
                            .pnl_realized
                            .fetch_add(net_trade_pnl, Ordering::Relaxed);
                    }

                    self.feature_engines[coin_id].last_scalp_exit_tick =
                        self.feature_engines[coin_id].tick_count;
                    self.feature_engines[coin_id].last_scalp_exit_ts = event_time_ms;
                    // #548: Calibración de racha direccional
                    // Un cierre por decaimiento de alfa plano o scratch de comisiones (-0.0005 < pnl_pct <= 0)
                    // es un evento neutro de rango, no una falla direccional tóxica de tendencia contraria.
                    // Solo pérdidas direccionales genuinas (pnl_pct <= -0.0005) incrementan la racha de pérdidas.
                    let is_directional_loss = net_trade_pnl < 0.0 && pnl_pct <= -0.0005;
                    self.feature_engines[coin_id].last_scalp_was_loss = is_directional_loss;
                    if is_directional_loss {
                        self.feature_engines[coin_id].scalp_loss_streak += 1;
                        if is_long {
                            self.feature_engines[coin_id].scalp_long_loss_streak += 1;
                        } else {
                            self.feature_engines[coin_id].scalp_short_loss_streak += 1;
                        }
                    } else if net_trade_pnl > 0.0 {
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
                        let obi_val = self.feature_engines[coin_id].obi_accel.prev_obi;
                        let hurst_val = self.feature_engines[coin_id].hurst_macro as f64;
                        storage_engine::mmap_bus::write_prediction_vs_reality_ext(
                            ml_at_entry,
                            is_long,
                            cap_pct,
                            atr_pct,
                            obi_val,
                            hurst_val,
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
                        if coin_id < self.conformal_by_coin.len() {
                            self.conformal_by_coin[coin_id].update(p_win_at_entry, is_win);
                        } else {
                            self.conformal.update(p_win_at_entry, is_win);
                        }
                    }
                    // D-619: el calibrador aprende de la puntuación CRUDA y del
                    // resultado neto de comisiones. Nunca de su propia salida.
                    if score_at_entry > 0.0 {
                        if coin_id < self.calibrator_by_coin.len() {
                            self.calibrator_by_coin[coin_id].update(score_at_entry, is_win);
                        } else {
                            self.confidence_calibrator.update(score_at_entry, is_win);
                        }
                    }

                    // #25: Actualización del motor de refuerzo continuo PPO (OnlinePpoPolicyEngine)
                    let ppo_reward = net_trade_pnl / (entry * qty).max(1e-8);
                    let action_sign = if is_long { 1.0 } else { -1.0 };
                    let fe_c = &self.feature_engines[coin_id];
                    let state_feats = [
                        fe_c.ofi_model.ema_ofi,
                        fe_c.obi_accel.prev_obi,
                        fe_c.cvpin.current_vpin(),
                        0.0,
                        fe_c.a_t,
                    ];
                    self.ppo_engine.update_policy(
                        ppo_reward,
                        &state_feats,
                        action_sign,
                        1.0,
                        0.05,
                        0.01,
                        0.20,
                        0.01,
                    );

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

                    // QO-E2b — EMISOR: aparear el tensor congelado en la
                    // APERTURA con el retorno neto REALIZADO y appendar la
                    // fila al dataset del auto-trainer NN. Append directo:
                    // los cierres son eventos raros (segundos-minutos), el
                    // costo de abrir el archivo es irrelevante fuera del
                    // hot path de ticks.
                    {
                        let tensor_ready = pos
                            .nn_entry_tensor
                            .lock()
                            .map(|t| !t.is_empty() && t.len() == 54)
                            .unwrap_or(false);
                        if tensor_ready {
                            let notional = (qty * entry).max(1.0);
                            let target_ret = net_trade_pnl / notional;
                            if target_ret.is_finite() {
                                let sym_ds = quantum_arena::symbol_registry::try_symbol(coin_id)
                                    .unwrap_or_default();
                                if !sym_ds.is_empty() {
                                    let path =
                                        format!("data/dark_alpha_dataset_{}.csv", sym_ds);
                                    let need_header = !std::path::Path::new(&path).exists();
                                    use std::io::Write as _;
                                    if let (Ok(mut f), Ok(t)) = (
                                        std::fs::OpenOptions::new().create(true).append(true).open(&path),
                                        pos.nn_entry_tensor.lock(),
                                    ) {
                                        if need_header {
                                            let _ = writeln!(
                                                f,
                                                "target_return,{}",
                                                (0..54).map(|i| format!("f{i}")).collect::<Vec<_>>().join(",")
                                            );
                                        }
                                        let mut row = format!("{:.8}", target_ret);
                                        for v in t.iter() {
                                            row.push_str(&format!(",{:.8}", v));
                                        }
                                        let _ = writeln!(f, "{}", row);
                                    }
                                }
                            }
                        }
                    }

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
                    // S-1: confianza espectral — persistencia de la escala
                    // dominante mapeada de [-1,1] a [0,1] (0.5 = browniano
                    // neutral). La banda de Kelly respira con el régimen.
                    let spectral_conf = self
                        .temporal_spectrum
                        .get(coin_id)
                        .map(|spec| {
                            let pers = spec
                                .persistence_at(spec.dominant_tau_ms)
                                .clamp(-1.0, 1.0);
                            (0.5 + pers * 0.5).clamp(0.0, 1.0)
                        })
                        .unwrap_or(0.5);
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
                        spectral_conf,
                    );
                    coin.metrics
                        .kelly_fraction
                        .store(kelly_f, Ordering::Relaxed);
                    coin.last_close_ts.store(event_time_ms, Ordering::Relaxed);
                    coin.last_close_is_long.store(is_long, Ordering::Relaxed);
                    coin.last_close_was_win.store(is_win, Ordering::Relaxed);
                    coin.last_close_reason.store(reason_code, Ordering::Relaxed);
                    coin.last_scalp_close_ts.store(event_time_ms, Ordering::Relaxed);
                    // MOD2/7-021 (INFORME DECIMOCUARTO): `last_swing_close_ts`
                    // se escribía con el MISMO valor que el scalp (timestamps
                    // gemelos — residuo de la bifurcación swing/scalp). Sin
                    // lectores en todo el workspace: se deja en 0 para siempre.

                    // D-181: closed_order debe reflejar el PnL neto definitivo deduciendo ambas comisiones (entry + close)
                    closed_order = Some((is_long, net_trade_pnl, qty));

                    let notional = (qty * entry).max(1.0);
                    let realized_ret = net_trade_pnl / notional;
                    // C-07 (INFORME DECIMOCUARTO): record_outcome UNA VEZ por
                    // cierre. Antes se registraba con las señales scalp Y swing
                    // — idénticas tras la unificación— duplicando cada trade en
                    // el tracker (los pesos adaptativos aprendían de un dataset
                    // con cada observación repetida).
                    if coin_id < self.last_senior_signals.len() {
                        self.consejo_deliberacion.record_outcome(
                            &self.last_senior_signals[coin_id],
                            realized_ret,
                        );
                    }

                    // #26: Registrar trauma en el sistema inmune vivo si la pérdida excede 1.5%
                    if realized_ret < -0.015 {
                        let record = metacortex_engine::immune_system::TraumaRecord {
                            id: format!("trauma_{}_{}", sym, event_time_ms),
                            timestamp: chrono::Utc::now(),
                            symbol: sym.clone(),
                            regime: format!("{:?}", self.feature_engines[coin_id].regime),
                            expected_pnl_pct: 0.01,
                            actual_pnl_pct: realized_ret,
                            predictor_name: "GodEngineCore".to_string(),
                            inputs_snapshot: self.feature_engines[coin_id]
                                .get_universal_features()
                                .iter()
                                .map(|&x| x as f64)
                                .collect(),
                        };
                        let _ = self.immune_system.record_trauma(&record);
                    }

                    // #15: Auditor forense ShadowGraph (Metacórtex Concept Drift)
                    let pnl_drift = realized_ret - (ml_at_entry - 0.5) * 2.0 * 0.01;
                    let actual_slippage = if mid_price > 0.0 {
                        (exit_price - mid_price).abs() / mid_price
                    } else {
                        0.0
                    };
                    self.shadow_auditor.record_event(
                        metacortex_engine::shadow_graph_auditor::ShadowEvent {
                            tick_id: event_time_ms,
                            expected_prob: ml_at_entry,
                            actual_slippage,
                            latency_ms: 0,
                            pnl_drift,
                        },
                    );

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
                    self.diag_dir.record_online_update(realized_ret - ml_at_entry);
                    self.online_learner.update_weights_with_kalman_adaptive_vol(
                        &online_feat,
                        (realized_ret - ml_at_entry) as f32,
                        (hurst_val as f32 - 0.5).abs(),
                        raw_atr_pct as f32,
                    );
                } else {
                    let notional = qty * entry;
                    let mut unrealized = pnl_pct * notional;
                    // B3.13 — GUARD DE MARCADO anclado a CAPITAL (no a
                    // notional: un entry local basura infla el notional con
                    // él). Medido: +$2.27M en cuenta de $2.2K (2026-09-15).
                    // El unrealized de UNA posición no puede exceder 2× el
                    // capital de la cuenta; si lo excede, el marcado local
                    // está roto (glitch de precio o posición fantasma de
                    // rotación) y se acota.
                    let cap_anchor = self.arena.unified_capital.load(Ordering::Relaxed).max(1.0);
                    let bound = cap_anchor * 2.0;
                    if !unrealized.is_finite() {
                        unrealized = 0.0;
                    } else if unrealized.abs() > bound {
                        unrealized = unrealized.clamp(-bound, bound);
                    }
                    coin.metrics
                        .pnl_unrealized
                        .store(unrealized, Ordering::Relaxed);
                }
                if closed_order.is_some() {
                    break;
                }
            }

            // --- 2. ANALÍTICA COMPLETA (ML + espectro + registro) ---
            // X-012 + B2.5-fix: antes, el interlock de entradas retornaba
            // AQUÍ — antes del bloque ML — y un feed marcado stalled/lento
            // congelaba TODA la analítica (ml_prob en 0.5 exacto por defecto,
            // espectro muerto): el síntoma "ml congelado" que perseguíamos
            // desde la primera auditoría. El interlock debe bloquear
            // ENTRADAS, jamás análisis. El return se movió a tras el bloque
            // ML, inmediatamente antes de la maquinaria de señales.
            let tick = self.arena.tick_counter.load(Ordering::Relaxed);

            // --- Inteligencia On-Chain (Spot vs Futures Correlation) ---
            let spot_bid = coin.spot_bid.load(Ordering::Relaxed);
            let spot_ask = coin.spot_ask.load(Ordering::Relaxed);
            // MOD2/7-028 (INFORME DECIMOCUARTO): el salto ±0.15 ABSOLUTO al
            // cruzar 1.5 bps no tenía base — 15 puntos de probabilidad por
            // 1.5 bps de spread podían mover el gate sin opinión de modelo.
            // Ahora el sesgo escala con la magnitud REAL del spread:
            // 10 bps ⇒ ±0.05 (la mitad del efecto anterior, proporcional al
            // fenómeno), con clamp simétrico ±0.05. Además (MOD2/7-029) el
            // gate B3.18 lee el ensamble PURO — este sesgo sólo ajusta las
            // ramas de señal.
            let mut spot_bias = 0.0;

            if spot_bid > 0.0 && spot_ask > 0.0 {
                let spot_mid = (spot_bid + spot_ask) / 2.0;
                let spread_bps = ((spot_mid - mid_price) / mid_price) * 10000.0;
                spot_bias = (spread_bps / 10.0).clamp(-0.05, 0.05);
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
            // CERT-M2-C02 — λ/μ verdadero del proceso excitado por TRADES
            // (la excitación ocurre en process_event::is_trade, que llama a
            // este dual en el MISMO evento — la lectura es fresca; los
            // callers directos de dual sin trades previos leen ratio 1.0).
            let hawkes_ev_s = event_time_ms as f64 / 1000.0;
            let (hawkes_ratio_real, hawkes_eta_real) = match self.hawkes_by_coin.get(coin_id) {
                Some(hk) => (hk.intensity_ratio(hawkes_ev_s), hk.branching_ratio()),
                None => (1.0, 0.0),
            };
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
            // CERT-M2-C02: λ/μ VERDADERO del proceso excitado por trades
            // (antes: proxy 1+|a_t|/ATR — aceleración, no intensidad).
            // Clamp [0.1, 10] sólo acota telemetría: ≈1 calma, >3 cascada.
            set_reg("hawkes_intensity", hawkes_ratio_real.clamp(0.1, 10.0));
            set_reg("hawkes_branching", hawkes_eta_real);
            set_reg("bessel_alpha", 1.5);
            set_reg("hawkes_dt", 0.05);
            set_reg(
                "microstructure_noise_variance",
                (atr_pct * 0.1).max(0.00001),
            );

            // MOD2/7-005 (INFORME DECIMOCUARTO): key-mismatch — el cierre
            // escribe `{sym}_hebbian_weight` (ver bloque de cierre arriba)
            // pero aquí se leía `{sym}_perceptron_hebbian_weight`, clave que
            // NADIE escribió: la resolución caía SIEMPRE al global y el
            // aislamiento per-símbolo era inoperante (la racha perdedora de
            // DOGE reducía la convicción de BTC). Ahora se lee la clave que
            // el cierre realmente escribe; sin historia propia el símbolo
            // obtiene el neutro 1.0 — jamás el peso aprendido por otro.
            let hebbian_mult = self
                .arena
                .registry
                .get_scoped_value_or(&sym, "hebbian_weight", 1.0)
                .clamp(0.5, 2.0);

            // 34D Macro+Micro Features for NanoForest (Indices 0..24 used by trained trees)
            let features = self.feature_engines[coin_id].get_features();
            let swing_feats = self.feature_engines[coin_id].get_universal_features();

            // F4.7 — ENSAMBLE REAL: ambos modelos opinan; la probabilidad es
            // el promedio ponderado con pesos que evolucionan por Brier real.
            // (Antes: if-else — el NN solo opinaba si el forest NO existía.)
            let combined_tensor =
                self.build_54d_tensor(coin_id, bid_qty, ask_qty, mid_price, omni_features);
            // D-406 & D-407: Soporte para modelo por activo ({sym}_SCALP).
            // B3.18b — SIN FALLBACK a BTC: los símbolos sin modelo validado
            // NO heredan la opinión de otro símbolo (transferencia
            // cross-símbolo jamás validada — ETH la falló; y el BTC rolling
            // delgado con init sesgado inyectaba sesgo long global como
            // fallback). El ensamble sigue opinando con NN + espectro; la
            // entrada además pasa el gate B3.18.
            let sym = quantum_arena::symbol_registry::try_symbol(coin_id)
                .unwrap_or_else(|| "BTCUSDT".to_string());
            let coin_model_key = format!("{}_MOTOR", sym); // U-7: contrato migrado
            let active_forest = crate::ml_inference::NanoForest::get_global(&coin_model_key);
            // B3.25 — DISCIPLINA DE ROSTER: hueco medido en vivo (SOL, modelo
            // retirado, abrió posición nueva): sin forest, el NN SOLO puede
            // empujar el ml sobre el umbral del gate — el ensamble nunca es
            // neutral de verdad. La regla estructural: SIN MODELO VALIDADO
            // del roster, NO se opera (el gate B3.18 exigirá además este
            // flag). La opinión NN sigue viva para análisis/telemetría.
            let has_roster_model = active_forest.is_some();
            // B3.36 — base del PROPIO modelo (sigmoid(init_score)). Con el
            // etiquetado honesto HOST-010 la base es ~30%, no 50%: los gates
            // de entrada (B3.18, fallback F7, rama swing) se expresan como
            // LIFT sobre esta base, no como umbral absoluto — así un cambio
            // de geometría de labels NO cambia la selectividad del sistema.
            let ml_model_base = active_forest
                .as_ref()
                .map(|f| f.base_prob())
                .unwrap_or(0.5)
                .clamp(0.05, 0.95);
            // P-1 (PREDICTORES) — VOLATILIDAD FUTURA {SYM}_VOL: regresión
            // entrenada con --label vol (RMS de retornos a horizonte, ×100).
            // Se sirve con predict_raw (SIN sigmoid — es una σ, no una
            // probabilidad) sobre el MISMO vector 48D del motor. Publicada
            // al registry como `vol_forecast_pct`: los consumidores de
            // sizing/cooldowns/telemetría leen esa clave; 0.0 = sin modelo
            // (el gate del trainer ya bloqueó los símbolos sin edge).
            {
                let vol_key = format!("{}_VOL", sym);
                let (vol_forecast, vol_base) = crate::ml_inference::NanoForest::get_global(&vol_key)
                    .map(|m| {
                        const VD: usize = crate::ml_inference::NanoForest::ML_VECTOR_DIM;
                        let mut vin = [0f32; VD];
                        vin[..34].copy_from_slice(&swing_feats);
                        vin[34..44].copy_from_slice(
                            &self.feature_engines[coin_id].get_spectral_ml_features(),
                        );
                        vin[44..]
                            .copy_from_slice(&crate::ml_inference::macro_ml_features(omni_features));
                        for v in vin.iter_mut() {
                            if !v.is_finite() {
                                *v = 0.0;
                            }
                        }
                        let (raw, _) = m.predict_raw(&vin);
                        // P-1b: la BASE del modelo de regresión es su init
                        // (media del mes de entrenamiento) — el freno de
                        // sizing compara pronóstico contra base en unidades
                        // exactas (σ del régimen en que se entrenó).
                        let base = m.init_value();
                        (
                            if raw.is_finite() && raw > 0.0 {
                                raw as f64
                            } else {
                                0.0
                            },
                            if base.is_finite() && base > 0.0 { base } else { 0.0 },
                        )
                    })
                    .unwrap_or((0.0, 0.0));
                set_reg("vol_forecast_pct", vol_forecast);
                set_reg("vol_forecast_base", vol_base);
                // P-3c — delta de OI pronosticado ({SYM}_OI, regresión del
                // histórico acumulativo). Hoy NINGÚN símbolo pasó el gate
                // (21d de ventana horaria: sin edge medible) — el serving
                // existe para que el predictor se ENCIENDA solo el día que
                // el gate lo permita; sin modelo ⇒ 0.0.
                let oi_key = format!("{}_OI", sym);
                let oi_delta = crate::ml_inference::NanoForest::get_global(&oi_key)
                    .and_then(|m| {
                        const VD: usize = crate::ml_inference::NanoForest::ML_VECTOR_DIM;
                        let mut vin = [0f32; VD];
                        vin[..34].copy_from_slice(&swing_feats);
                        vin[34..44].copy_from_slice(
                            &self.feature_engines[coin_id].get_spectral_ml_features(),
                        );
                        vin[44..]
                            .copy_from_slice(&crate::ml_inference::macro_ml_features(omni_features));
                        for v in vin.iter_mut() {
                            if !v.is_finite() {
                                *v = 0.0;
                            }
                        }
                        let (raw, _) = m.predict_raw(&vin);
                        raw.is_finite().then_some(raw as f64)
                    })
                    .unwrap_or(0.0);
                set_reg("oi_delta_forecast_pct", oi_delta);
            }
            let coin_ensemble = if coin_id < self.ensembles.len() {
                &mut self.ensembles[coin_id]
            } else {
                &mut self.ensemble
            };
            // Diagnóstico: predicción de cada modelo del ensamble (sólo telemetría).
            let mut diag_forest_p: Option<f64> = None;
            let mut diag_nn_p: Option<f64> = None;
            if let Some(f) = &active_forest {
                // B2.3: el input del forest es swing(34) ⊕ espectral(10) —
                // idéntico al de train_forest. Los árboles entrenados con el
                // esquema viejo (splits <34) no se ven afectados; los nuevos
                // pueden explotar el bloque espectral (F8: el espectro decide).
                // B3.4: + macro(4) — niveles FRED vivos del omni_state con el
                // MISMO contrato del trainer (macro_ml_features). El vector
                // vivo es superconjunto: modelos viejos (splits <44)
                // siguen válidos; los nuevos pueden partir por régimen macro.
                const FOREST_INPUT_DIM: usize = crate::ml_inference::NanoForest::ML_VECTOR_DIM;
                let mut forest_input = [0f32; FOREST_INPUT_DIM];
                forest_input[..34].copy_from_slice(&swing_feats);
                forest_input[34..44]
                    .copy_from_slice(&self.feature_engines[coin_id].get_spectral_ml_features());
                forest_input[44..].copy_from_slice(&crate::ml_inference::macro_ml_features(
                    omni_features,
                ));
                // Saneo: un feature NaN (p.ej. omni sin feed macro para ese
                // símbolo) mataba predict() completo → ml congelado en 0.5.
                // NaN = "sin dato" ⇒ neutro 0.0, el resto del vector sigue
                // opinando. El trainer descarta esas muestras; aquí el
                // neutro preserva el flujo de análisis en vivo.
                for v in forest_input.iter_mut() {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                }
                if let Some(p) = f.predict(&forest_input) {
                    diag_forest_p = Some(p as f64);
                    coin_ensemble.submit(crate::ensemble::ModelId::MotorForest, p as f64);
                } else if tick % 100 == 0 {
                    // B2.5-aud: el input ya está saneado arriba (NaN ⇒ 0.0), así
                    // que predict()=None aquí SÓLO puede ser un modelo
                    // degenerado (tree_offsets ≤ 1: sin árboles usables). El
                    // viejo mensaje de "features no finitos" listaba siempre
                    // un vector vacío y ocultaba la causa real.
                    println!(
                        "🔬 [ML-DIAG] {} predict=None (modelo {:?}) — modelo degenerado (tree_offsets ≤ 1), sin splits evaluables; reciclar modelo",
                        sym, coin_model_key
                    );
                }
            } else if tick % 100 == 0 {
                println!(
                    "🔬 [ML-DIAG] {} SIN forest activo (key {:?} sin modelo y sin fallback)",
                    sym, coin_model_key
                );
            }
            if let Some(nn) = self.swing_nn.as_mut() {
                // C-06 (INFORME DECIMOCUARTO): el swing_nn (DarkAlpha) es UN
                // modelo entrenado con datos de BTC; antes votaba en el
                // ensamble de TODAS las monedas vía predict_for_coin — la
                // entrada en un alt la podía decidir el modelo de BTC.
                // NN restringido a BTC hasta que exista un modelo por símbolo
                // (paridad con B3.18b del forest): fuera de su símbolo de
                // entrenamiento el voto es NEUTRAL (0.5) — no se evalúa la
                // inferencia, el modelo simplemente no opina.
                let nn_trained_for_symbol = sym == "BTCUSDT";
                let in_dim = nn.layer1.in_features;
                let p_opt = if nn_trained_for_symbol {
                    if in_dim == 34 {
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
                    }
                } else {
                    Some(0.5)
                };
                if let Some(p) = p_opt {
                    diag_nn_p = Some(p);
                    coin_ensemble.submit(crate::ensemble::ModelId::DarkAlphaNN, p);
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
            // D-693 (DÉCIMA OLA): el residuo online ya no entra en la probabilidad
            // (ver `calibration::compose_ml_prob`). Se sigue calculando para que el
            // diagnóstico muestre lo que habría sumado.
            // MOD2/7-029 (INFORME DECIMOCUARTO): `ml_prob_pure` es la opinión del
            // ensamble SIN el sesgo spot — el valor que evalúa el gate B3.18. El
            // spot_bias es una corrección de microestructura, no un modelo: con
            // ±0.15 absolutos podía cruzar el umbral B3.18 él solo, sin opinión de
            // modelo. Las ramas de señal (p.ej. la 13) siguen consumiendo la
            // versión con sesgo (`ml_prob`), que es la publicada en coin.ml_prob.
            let ml_prob_pure = base_ml_prob;
            let ml_prob = crate::calibration::compose_ml_prob(base_ml_prob, spot_bias);
            self.diag_dir.record_ml_components(
                base_ml_prob,
                online_residual.clamp(-0.15, 0.15),
                diag_forest_p,
                diag_nn_p,
            );
            self.last_ml_prob = ml_prob as f32;
            coin.ml_prob.store(ml_prob, Ordering::Relaxed);
            set_reg("ml_prob", ml_prob);
            set_reg("ml_prob_motor", ml_prob);
            // B3.37-diag — latido del camino ML completo para los símbolos
            // con modelo: forest→ensamble→store. Si este línea imprime
            // valores vivos pero ESPECTRO sigue en 0.5000, el defecto está
            // en el LECTOR; si imprime 0.5 vacío, está en el CAMINO.
            if matches!(sym.as_str(), "NEARUSDT" | "ATOMUSDT" | "BNBUSDT")
                && self.arena.tick_counter.load(Ordering::Relaxed) % 5_000 < 26
            {
                telemetry_server::telemetry_log!(
                    "🫀 [ML-HEARTBEAT] {} forest={:?} base={:.4} combined={:.4} stored={:.4} roster={}",
                    sym,
                    diag_forest_p,
                    ml_model_base,
                    base_ml_prob,
                    ml_prob,
                    has_roster_model
                );
            }

            let nn_score: f64 = self.feature_engines[coin_id].update_ml_prediction(ml_prob);

            // X-012 (reubicado por B2.5-fix): frontera REAL del bloqueo —
            // gestión de posiciones (sección 1) y analítica ML/espectral ya
            // corrieron completas. Con datos obsoletos no se EVALÚAN ni
            // abren posiciones nuevas desde aquí hacia abajo.
            // CERT-M2-C01: el return anterior `(None, None, None)`
            // DESCARTABA el `closed_order` ya computado en la sección 1
            // (~1200 líneas arriba) — durante tormentas de latencia/stalls
            // (exactamente cuando las salidas defensivas son VITALES), el
            // host nunca recibía el evento de cierre: estado divergente,
            // OCO rancio, contabilidad perdida. Ahora el cierre viaja.
            if entries_blocked {
                return (None, closed_order, None);
            }

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
                .clamp(0.0003, 0.0040);
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
            let conformal_p = if coin_id < self.conformal_by_coin.len() {
                self.conformal_by_coin[coin_id].p_value(ml_prob_now)
            } else {
                self.conformal.p_value(ml_prob_now)
            };
            let conformal_p_short = if coin_id < self.conformal_by_coin.len() {
                self.conformal_by_coin[coin_id].p_value(1.0 - ml_prob_now)
            } else {
                self.conformal.p_value(1.0 - ml_prob_now)
            };
            let accept_long = if coin_id < self.conformal_by_coin.len() {
                            self.conformal_by_coin[coin_id].accepts(ml_prob_now)
                        } else {
                            self.conformal.accepts(ml_prob_now)
                        };
            let accept_short = if coin_id < self.conformal_by_coin.len() {
                            self.conformal_by_coin[coin_id].accepts(1.0 - ml_prob_now)
                        } else {
                            self.conformal.accepts(1.0 - ml_prob_now)
                        };
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

            // ── BACKTEST ADAPTATIVO & CIERRE M2-C05 ─────────────────────────────
            // Detección honesta de libro ausente: un libro solo está ausente si NO hay
            // liquidez en los niveles bid/ask ((bid_qty + ask_qty) <= 1e-9 o cantidades no positivas).
            // OBI ≈ 0 (|obi| < 0.005) es un libro BALANCEADO normal en pares líquidos (BTC, ETH),
            // NUNCA un libro ausente. Confundirlo provocaba bypass espurio de PPO y gates en vivo.
            let book_absent = (bid_qty + ask_qty) <= 1e-9 || bid_qty <= 0.0 || ask_qty <= 0.0;
            let adaptive_micro_score = if book_absent {
                // Sin libro real (trade-only): CVD del flujo de trades (dirección agresora ×
                // volumen) ES la señal de microestructura disponible.
                rolling_cvd.clamp(-1.0, 1.0)
            } else {
                micro_score
            };

            // FIX SESGO ML (auditoría): el forest entrenado con features de
            // libro predice 0.337-0.476 cuando 5/12 micro-features son cero
            // — sesgo bearish sistemático que bloquea toda señal Long. Con
            // libro ausente, RE-CENTRAR la predicción: si el forest dice
            // "menos de 0.5", eso es su offset, no su señal. Normalizamos
            // mapeando el rango observado [0.35, 0.50] a [0.30, 0.70] para
            // restaurar simetría direccional.
            // D-735 (DÉCIMA OLA · auditoría integral): SIN AMPLIFICACIÓN
            // ASIMÉTRICA DE LA PROBABILIDAD.
            //
            // Aquí se doblaba la distancia a la neutralidad SÓLO en el lado
            // bajista (`ml_prob < 0,50`) cuando falta el libro, y el comentario
            // afirmaba lo contrario —«re-centrar para restaurar simetría
            // direccional»—: con ml = 0,35 la transformación daba 0,20, no 0,30.
            // El consumidor largo comparaba el valor CRUDO contra su umbral y el
            // corto el valor DUPLICADO contra el suyo, de modo que un umbral corto
            // de 0,30 disparaba en realidad con ml < 0,40. Como sin libro (todo
            // evento de trade en producción, antes de D-707) esa rama era la
            // habitual, el efecto era un sesgo estructural a corto en la misma
            // magnitud que el motor usa para decidir.
            //
            // El sesgo del bosque cuando faltan features no se corrige con una
            // recta ad hoc en el consumidor: se corrige recalibrando el modelo
            // —`calibration::PlattCalibrator` existe para eso— o no inyectando
            // features sintéticas de libro (D-707).
            let ml_prob_adaptive = ml_prob;

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
            let raw_composite =
                adaptive_micro_score * w_micro + nn_score * w_nn + tensor_boost * w_tensor;
            let composite_score: f64 = (raw_composite * hebbian_mult).clamp(-1.0, 1.0);
            self.diag_dir.record_evaluation(
                composite_score,
                ml_prob,
                [
                    adaptive_micro_score * w_micro * hebbian_mult,
                    nn_score * w_nn * hebbian_mult,
                    tensor_boost * w_tensor * hebbian_mult,
                ],
                tensor_cont.signal,
            );

            // U-2 (MOTOR UNIVERSAL CONTINUO): lo que fue la rama "scalp" es la
            // LECTURA DE BANDA RÁPIDA del continuo (τ corto: microestructura,
            // flujo del libro, triggers de sub-segundo). Lo que fue "swing" es
            // la LECTURA DE BANDA LENTA (τ largo: tendencia, EMAs, stretch).
            // No son estrategias: son dos vistas del MISMO espectro temporal
            // (doctrina F8). La fusión final arbitra el ESPECTRO (banda más
            // cercana a τ dominante), no una etiqueta de horizonte.
            let mut fast_intent = SignalIntent::flat();
            let micro_tau = self
                .temporal_spectrum
                .get(coin_id)
                .map(|s| s.micro_resonant_tau_ms())
                .unwrap_or(30_000.0);
            let fast_duration_ms = (micro_tau.clamp(1_000.0, 180_000.0)).round() as u64;
            let spread_pct = if mid_price > 0.0 {
                (ask - bid) / mid_price
            } else {
                0.0
            };
            let dynamic_max_spread = (atr_pct * 0.25).clamp(0.0006, 0.0025);
            let spread_ok = spread_pct <= dynamic_max_spread;

            // MOD2/7-014: clasificación canónica de Hurst — UNA definición
            // (consts HURST_*) usada por TODAS las ramas de señal de abajo.
            let is_anti_persistent = hurst_val < HURST_ANTI_PERSISTENT;
            let is_persistent = hurst_val > HURST_PERSISTENT;

            if atr_pct > dynamic_atr_min
                && spread_ok
                && self.feature_engines[coin_id].can_open_position(60_000)
            {
                // (misma banda canónica: reversión a la media ≡ anti-persistencia)
                let is_mean_reverting = is_anti_persistent;

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

                // MOD2/7-014: antes `hurst_val < 0.42` — un tercer literal
                // de banda que contradecía las demás ramas. Banda canónica.
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
                let total_loss_streak = self.feature_engines[coin_id].get_active_total_loss_streak();
                // Si ambas direcciones han fallado recientemente (chop alternante), total_loss_streak modula.
                // Si solo una dirección falla mientras el mercado se mueve en la otra, la dirección a favor del flujo permanece libre.
                let is_alternating_chop = short_streak >= 1 && long_streak >= 1;
                let effective_short_streak = if is_alternating_chop { short_streak.max(total_loss_streak) } else { short_streak };
                let effective_long_streak = if is_alternating_chop { long_streak.max(total_loss_streak) } else { long_streak };

                // F7-backtest: cuando el libro está AUSENTE, las condiciones
                // OBI (< -min_obi_trend) son imposibles de satisfacer (OBI=0).
                // Sustitución honesta: CVD del flujo de trades reemplaza OBI
                // como confirmación direccional. Con libro real, OBI original.
                let effective_obi_short = if book_absent { rolling_cvd } else { current_obi };
                let effective_obi_long = if book_absent { rolling_cvd } else { current_obi };

                // DIAGNÓSTICO (temporal — eliminar tras calibrar): ver por qué
                // no se generan señales en backtest trade-only.
                if self.arena.tick_counter.load(Ordering::Relaxed) % 50_000 == 0 {
                    telemetry_server::telemetry_log!(
                        "🔍 [DIAG] tick={} atr_pct={:.6} atr_min={:.6} spread_ok={} can_scalp={} hurst={:.3} ema_s={:.2} mid={:.2} ml={:.3} book_absent={} obi={:.3} ofi={:.3} cvd={:.3}",
                        self.arena.tick_counter.load(Ordering::Relaxed),
                        atr_pct,
                        dynamic_atr_min,
                        spread_ok,
                        self.feature_engines[coin_id].can_open_position(60_000),
                        hurst_val,
                        self.feature_engines[coin_id].ema_slow,
                        mid_price,
                        ml_prob,
                        book_absent,
                        obi_val,
                        ofi_value,
                        rolling_cvd,
                    );
                }

                // ── F7-backtest: CANAL ML-ONLY + PRICE-ACTION FALLBACK ──────
                // Ruta 1: ML directo (umbral reducido cuando libro ausente —
                // el forest tiende a predecir ~0.5 sin features de libro).
                // Ruta 2: PRICE-ACTION puro (momentum/ATR/Hurst — funciona
                // SIN ML y SIN libro; computable de precio/volumen solos).
                if book_absent && fast_intent.signal == SignalType::Flat && atr_pct > 0.00005 {
                    // B3.36 — mismos gates POR LIFT que el camino vivo: el
                    // fallback usa los MISMOS genes reinterpretados como lift
                    // sobre la base del modelo del símbolo.
                    // D-715 (unión): los genes pasan ANTES por
                    // ml_gate_thresholds — la reparación canónica (largo ≥ ½
                    // ≥ corto, no finitos neutralizados) — y el umbral YA
                    // reparado es el que se reinterpreta como lift. Un solo
                    // invariante para los mismos dos genes en todo el motor.
                    let (ml_thr_long, _ml_thr_short) = crate::calibration::ml_gate_thresholds(
                        self.arena.config.ml_threshold_long.load(Ordering::Relaxed),
                        self.arena.config.ml_threshold_short.load(Ordering::Relaxed),
                    );
                    let ml_lift = (ml_thr_long - 0.50).clamp(0.02, 0.25);
                    // Ruta 1: ML RE-CENTRADO (sesgo eliminado) con confianza
                    // proporcional al LIFT sobre la base del modelo.
                    if ml_prob_adaptive > ml_model_base + ml_lift {
                        let conviction =
                            0.5 + (ml_prob_adaptive - ml_model_base).abs().min(0.45);
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviction.clamp(0.60, 0.95),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 1.0,
                            ..Default::default()
                        };
                    } else if ml_prob_adaptive < ml_model_base - ml_lift {
                        let conviction =
                            0.5 + (ml_prob_adaptive - ml_model_base).abs().min(0.45);
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviction.clamp(0.60, 0.95),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 1.0,
                            ..Default::default()
                        };
                    }
                    // Ruta 1b: SEÑAL ESPECTRAL DIRECTA — la fusión por paridad
                    // de riesgo del espectro temporal (19 escalas) produce un
                    // score [-1,+1] computado SOLO de precios reales. Si el
                    // score es fuerte en una dirección Y la persistencia lo
                    // confirma, es una señal legítima independiente del ML.
                    if fast_intent.signal == SignalType::Flat {
                        if let Some(spec) = self.temporal_spectrum.get(coin_id) {
                            let field_long = spec.spectral_field(true);
                            let field_short = spec.spectral_field(false);
                            let fused = spec.fused_score;
                            let tau_star = field_long.resonant_tau_ms;
                            let expected_tau = (tau_star.clamp(500.0, 3_600_000.0)).round() as u64;
                            let dyn_flow = (1.0 + fused.abs() * 2.0).clamp(1.0, 5.0);

                            // Señal Directa del Campo Espectral Continuo (#563):
                            // Se activa cuando el momento espectral unificado y la coherencia armónica
                            // sobre las 32 partes espectrales están firmemente alineados.
                            if fused > 0.60 && field_long.global_coherence > 0.15 {
                                fast_intent = SignalIntent {
                                    signal: SignalType::Long,
                                    confidence: (0.55 + field_long.global_coherence * 0.35).min(0.92),
                                    horizon: strategy_core::TradeHorizon::Continuous,
                                    expected_duration_ms: expected_tau,
                                    volume_flow_rate: dyn_flow,
                                    ..Default::default()
                                };
                            } else if fused < -0.60 && field_short.global_coherence > 0.15 {
                                fast_intent = SignalIntent {
                                    signal: SignalType::Short,
                                    confidence: (0.55 + field_short.global_coherence * 0.35).min(0.92),
                                    horizon: strategy_core::TradeHorizon::Continuous,
                                    expected_duration_ms: expected_tau,
                                    volume_flow_rate: dyn_flow,
                                    ..Default::default()
                                };
                            }
                        }
                    }
                    // Ruta 2: PRICE-ACTION (cuando ML sigue neutral)
                    // FIX AUDIT: Hurst puede estar clavado en 0.5 (DFA sin
                    // datos suficientes). Rama A: Hurst activo (anti-persistente
                    // O persistente, bandas canónicas MOD2/7-014 — antes el
                    // par 0.49/0.51 creaba una tercera clasificación ad hoc).
                    // Rama B: Hurst neutral — momentum directo sin régimen.
                    //
                    // D-737 (DÉCIMA OLA · auditoría integral): este `else` se ligaba
                    // al `if` de la ruta 1b, no al bloque del ML. Consecuencia
                    // exactamente contraria a la declarada: si el ML NO opinaba se
                    // entraba en 1b y la ruta 2 no se evaluaba nunca —el respaldo
                    // estaba muerto justo en el caso para el que se escribió—, y si
                    // el ML SÍ opinaba se saltaba 1b y la ruta 2 podía REESCRIBIR la
                    // intención: un Long del ML con confianza 0,62 se convertía en un
                    // Short de confianza literal 0,68 cuando el Hurst caía por debajo
                    // de 0,45. El motor invertía la dirección de su propia señal.
                    //
                    // Ahora las tres rutas son una cascada explícita: cada respaldo
                    // se evalúa sólo si la intención sigue plana.
                    if fast_intent.signal == SignalType::Flat
                        && self.feature_engines[coin_id].ema_slow > 0.0
                    {
                        let ema_s = self.feature_engines[coin_id].ema_slow;
                        let atr_abs = (atr_pct * mid_price).max(0.01);
                        let dev_atr = (mid_price - ema_s) / atr_abs;
                        let hurst_active = is_anti_persistent || is_persistent;

                        if hurst_active && is_persistent && dev_atr > 1.5 && dev_atr < 4.0 && rolling_cvd > 0.0 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Long,
                                confidence: 0.72,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.0,
                                ..Default::default()
                            };
                        } else if hurst_active && is_persistent && dev_atr < -1.5 && dev_atr > -4.0 && rolling_cvd < 0.0 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: 0.72,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.0,
                                ..Default::default()
                            };
                        } else if hurst_active && is_anti_persistent && dev_atr > 2.5 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: 0.68,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.2,
                                ..Default::default()
                            };
                        } else if hurst_active && is_anti_persistent && dev_atr < -2.5 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Long,
                                confidence: 0.68,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.2,
                                ..Default::default()
                            };
                        }
                        // Rama B: Hurst neutral (banda canónica [0.45, 0.52]) —
                        // momentum directo sin confirmación de régimen. Requiere
                        // desviación mayor (+0.5 ATR extra) y CVD alineado como
                        // substituto.
                        else if !hurst_active && dev_atr > 2.0 && dev_atr < 5.0 && rolling_cvd > 0.05 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Long,
                                confidence: 0.65,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.5,
                                ..Default::default()
                            };
                        } else if !hurst_active && dev_atr < -2.0 && dev_atr > -5.0 && rolling_cvd < -0.05 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: 0.65,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                expected_duration_ms: fast_duration_ms,
                                volume_flow_rate: 2.5,
                                ..Default::default()
                            };
                        }
                    }
                }

                // D-460 & D-466: Unificación Continua del Generador de Señales (Multiscale Vector Field).
                // Confluencia de triple escala temporal: Micro (1m ticks), Intermedio (EMA 9 vs 21), y Macro Superior (2-Hour EMA 120).
                if is_confirmed_downtrend {
                    // RÉGIMEN BAJISTA CONFIRMADO (MULTISCALE DOWNTREND)
                    let d_tech_thr = if effective_short_streak >= 2 { dynamic_tech_thr.max(0.30) } else { dynamic_tech_thr };
                    let d_obi_trend = if effective_short_streak >= 2 { min_obi_trend.max(0.22) } else { min_obi_trend };
                    // 1. Tendencial Short: Flujo institucional, confluencia L2 y ML apuntan a la baja
                    if composite_score < -d_tech_thr
                        && not_overextended_short
                        && effective_obi_short < -d_obi_trend
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 1.0,
                            ..Default::default()
                        };
                    // 2. Pullback Short: Rebote hacia ema_slow vendido con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score <= -d_tech_thr),
                    // filtro de tendencia superior (higher_trend <= -0.0010 para evitar vender en rallies)
                    // y desbalance de libro sólido (current_obi < -0.20)
                    } else if effective_short_streak < 2
                        && higher_trend <= -0.0010
                        && price_stretch >= 0.15
                        && price_stretch <= 1.20
                        && effective_obi_short < -min_obi_pullback.max(0.20)
                        && composite_score <= -d_tech_thr
                        && micro_trend <= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score.abs()),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 2.0,
                            ..Default::default()
                        };
                    // 3. Reversión Long — ELIMINADA (MOD2/7-011, INFORME
                    // DECIMOCUARTO): esta rama duplicaba la excepción de
                    // capitulación extrema que el escudo macro (D-467/D-624,
                    // aguas abajo del flujo) ya evalúa de forma más tardía y
                    // mejor informada (z-secular/z-higher/z-macro de difusión
                    // + z95 sobre la distancia a la EMA de 21 velas, en vez
                    // del stretch ATR crudo −2,5). Dos implementaciones
                    // independientes de la misma tautología derivaban en
                    // bandas mutuamente excluyentes; la excepción vive AHORA
                    // SOLO en el escudo macro. Una capitulación Long debe
                    // nacer de otra rama (ML-only/price-action, neutra) y
                    // superar el escudo con `extreme_capitulation`.
                    }
                } else if is_confirmed_uptrend {
                    // RÉGIMEN ALCISTA CONFIRMADO (MULTISCALE UPTREND)
                    let u_tech_thr = if effective_long_streak >= 2 { dynamic_tech_thr.max(0.30) } else { dynamic_tech_thr };
                    let u_obi_trend = if effective_long_streak >= 2 { min_obi_trend.max(0.22) } else { min_obi_trend };
                    // 1. Tendencial Long: Flujo institucional, confluencia L2 y ML apuntan al alza
                    if composite_score > u_tech_thr
                        && not_overextended_long
                        && effective_obi_long > u_obi_trend
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 4.0,
                            ..Default::default()
                        };
                    // 2. Dip Long: Corrección hacia ema_slow comprada con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score >= u_tech_thr),
                    // filtro de tendencia superior (higher_trend >= 0.0010)
                    // y desbalance de libro sólido (current_obi > min_obi_pullback.max(0.20))
                    } else if effective_long_streak < 2
                        && higher_trend >= 0.0010
                        && price_stretch <= -0.15
                        && price_stretch >= -1.20
                        && effective_obi_long > min_obi_pullback.max(0.20)
                        && composite_score >= u_tech_thr
                        && micro_trend >= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 5.0,
                            ..Default::default()
                        };
                    // 3. Reversión Short — ELIMINADA (MOD2/7-011): espejo
                    // exacto de la rama 3 de downtrend; la excepción de
                    // euforia parabólica extrema (`extreme_blowoff`) vive
                    // SOLO en el escudo macro, más tardío y mejor informado.
                    }
                } else {
                    // RÉGIMEN NEUTRO / RANGO LATERAL (Disciplina de reversión a la media: comprar en soporte, vender en resistencia)
                    let range_thr = dynamic_tech_thr * 1.15;
                    let range_obi = (dynamic_obi_thr * 0.85).clamp(0.12, 0.35);

                    // D-745 (UNDÉCIMA OLA · ALINEACIÓN JERÁRQUICA MULTIESCALA EN RANGO):
                    // Las ramas de reversión a la media (7, 8, 9, 10) solo deben operar
                    // cuando el mercado está GENUINAMENTE en rango o retroceso no impulsivo.
                    // Si higher_trend (2h) o secular_trend imponen una dirección macro clara,
                    // operar contra la marea produce pérdidas directas por parada (SL).
                    let range_long_trend_ok = higher_trend >= -0.0008
                        && !(higher_trend < -0.0002 && secular_trend < 0.0)
                        && macro_trend >= -0.00025;
                    let range_short_trend_ok = higher_trend <= 0.0008
                        && !(higher_trend > 0.0002 && secular_trend > 0.0)
                        && macro_trend <= 0.00025;

                    if composite_score > range_thr
                        && effective_obi_long > range_obi
                        && price_stretch <= -0.15
                        && micro_trend >= 0.0
                        && range_long_trend_ok
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 7.0,
                            ..Default::default()
                        };
                    } else if composite_score < -range_thr
                        && effective_obi_short < -range_obi
                        && price_stretch >= 0.15
                        && micro_trend <= 0.0
                        && range_short_trend_ok
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(composite_score),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 8.0,
                            ..Default::default()
                        };
                    } else if effective_short_streak < 2
                        && price_stretch > 1.0
                        && effective_obi_short < -range_obi * 1.15
                        && composite_score <= -0.24
                        && micro_trend <= 0.0
                        && range_short_trend_ok
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: sig_conf(effective_obi_short.abs().min(composite_score.abs())),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 9.0,
                            ..Default::default()
                        };
                    // D-736 (DÉCIMA OLA · auditoría integral): esta rama —reversión
                    // a la media alcista— leía `current_obi` mientras su espejo
                    // bajista y las otras ocho ramas del bloque leen
                    // `effective_obi_*`. Sin libro real, `current_obi` vale
                    // exactamente 0 y la condición era imposible: la rama alcista
                    // NUNCA disparaba mientras su simétrica bajista sí. Es una de
                    // las causas mecánicas del «un solo largo en ~140 operaciones».
                    } else if effective_long_streak < 2
                        && price_stretch < -1.0
                        && effective_obi_long > range_obi * 1.15
                        && composite_score >= 0.24
                        && micro_trend >= 0.0
                        && range_long_trend_ok
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: sig_conf(effective_obi_long.abs().min(composite_score.abs())),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 10.0,
                            ..Default::default()
                        };
                    }
                }

                let tensor_cutoff = tensor_min_conf.clamp(0.55, 0.85);
                if fast_intent.signal == SignalType::Flat
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

                    let long_macro_slope_ok = macro_trend >= 0.0 || (higher_trend > 0.00020 && micro_trend > 0.00015);
                    let short_macro_slope_ok = macro_trend <= 0.0 || (higher_trend < -0.00020 && micro_trend < -0.00015);

                    // Diagnóstico por dirección: las condiciones del gate se nombran una
                    // sola vez y el diagnóstico cuenta cuál falla. La semántica es la de la
                    // conjunción anterior: comparaciones puras, sin efectos laterales.
                    let long_conditions = [
                        effective_long_streak < 2,
                        !is_confirmed_downtrend,
                        !is_adverse_momentum_long,
                        !(higher_trend < -0.0001 && (secular_trend < 0.0 || macro_trend < -0.0001))
                            && !(secular_trend < -0.0002 && macro_trend < 0.0),
                        !(price_stretch < -0.80 && secular_trend < 0.0010),
                        higher_trend >= -0.0001 && long_macro_slope_ok,
                        composite_score >= tensor_tech_thr,
                        effective_obi_long > range_obi,
                        not_overextended_long,
                    ];
                    let short_conditions = [
                        effective_short_streak < 2,
                        !is_confirmed_uptrend,
                        !is_adverse_momentum_short,
                        !(higher_trend > 0.0001 && (secular_trend > 0.0 || macro_trend > 0.0001))
                            && !(secular_trend > 0.0002 && macro_trend > 0.0),
                        !(price_stretch > 0.80 && secular_trend > -0.0010),
                        higher_trend <= 0.0001 && short_macro_slope_ok,
                        composite_score <= -tensor_tech_thr,
                        effective_obi_short < -range_obi,
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
                        fast_intent = SignalIntent {
                            signal: tensor_scalp.signal,
                            confidence: tensor_scalp
                                .net_confidence
                                .abs()
                                .clamp(0.50, 0.95),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            expected_duration_ms: fast_duration_ms,
                            volume_flow_rate: 11.0,
                            ..Default::default()
                        };
                    }
                }

                if fast_intent.signal == SignalType::Flat {
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
                            effective_long_streak
                        } else {
                            effective_short_streak
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
                            fast_intent = turbo_intent;
                        }
                    }
                }

                // D-472: Invariante Bayesiano Absoluto — Prohibir cualquier scalp que contradiga el composite score
                if fast_intent.signal == SignalType::Long && composite_score < 0.0 {
                    fast_intent = SignalIntent::flat();
                } else if fast_intent.signal == SignalType::Short && composite_score > 0.0 {
                    fast_intent = SignalIntent::flat();
                }

                // F-009 FIX: Continuous ML Probability Weighting (replaces binary switch)
                // Instead of killing signals when ml_prob crosses 0.51/0.49, modulate
                // confidence continuously relative to the model's baseline (ml_model_base).
                // The farther ml_prob is from ml_base in the signal's direction, the more
                // confidence is amplified. Against the signal, confidence is reduced smoothly.
                {
                    let ml_base = if ml_model_base > 0.05 && ml_model_base < 0.95 {
                        ml_model_base
                    } else {
                        0.5
                    };
                    let ml_directional = match fast_intent.signal {
                        SignalType::Long => (ml_prob - ml_base) * 2.0,   // [-1, +1] where +1 = strong bullish
                        SignalType::Short => (ml_base - ml_prob) * 2.0,  // [-1, +1] where +1 = strong bearish
                        _ => 0.0,
                    };
                    // #539: Simetría y Veto Estricto del ML-Gate (Restitución de Veto Causal).
                    // Si el modelo contradice con convicción (ml_directional < -0.50), veto total: prohibido
                    // abrir órdenes en contra de la predicción de IA salvo estiramiento de reversión (|stretch| >= 2.5).
                    // Para contradicción leve (-0.50 <= ml_directional < 0.0), atenuación continua y simétrica
                    // sin piso arbitrario de 0.20 que mantenía vivas operaciones con esperanza negativa.
                    if ml_directional < -0.50 && price_stretch.abs() < 2.5 {
                        fast_intent = SignalIntent::flat();
                    } else if ml_directional < 0.0 && price_stretch.abs() < 2.5 {
                        fast_intent.confidence *= (1.0 + ml_directional).clamp(0.05, 1.0);
                    } else if ml_directional > 0.0 {
                        // ML confirma la dirección: boost simétrico y suave acotado a 0.95
                        fast_intent.confidence *= 1.0 + ml_directional * 0.5;
                        fast_intent.confidence = fast_intent.confidence.min(0.95);
                    }
                }

                // D-622 (DÉCIMA OLA): el cooldown binario del lado scalp (20 s tras
                // ganar; 90 s o 180 s tras perder según volatilidad) se retira. El
                // guard unificado D-463 cubre la reentrada tras cualquier cierre
                // —dirección, ganancia o pérdida, racha— con ventanas iguales o
                // mayores, y sin distinguir de qué «motor» vino la señal.
            }

            // --- CVD & L2 Wall HARD FILTERS (VETOS) ---
            if fast_intent.signal != SignalType::Flat {
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

                if fast_intent.signal == SignalType::Long {
                    if cvd_ratio < -cvd_veto {
                        fast_intent = SignalIntent::flat();
                    } else if wall_imbalance < -wall_veto {
                        fast_intent = SignalIntent::flat();
                    }
                } else if fast_intent.signal == SignalType::Short {
                    if cvd_ratio > cvd_veto {
                        fast_intent = SignalIntent::flat();
                    } else if wall_imbalance > wall_veto {
                        fast_intent = SignalIntent::flat();
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
            let mut slow_intent = SignalIntent::flat();
            let trend_threshold = self
                .arena
                .config
                .trend_threshold
                .load(Ordering::Relaxed);

            // B3.36 + D-715 (unión): la rama swing gatea por LIFT sobre la
            // base del modelo (ml_model_base) — los umbrales absolutos
            // mataban los largos con la base ~30% del etiquetado honesto.
            // Los genes se reparan por la MISMA función que la puerta de
            // entrada (ml_gate_thresholds): un solo invariante, sin la doble
            // reparación divergente (reflejar 1−ml_long) que D-715 extirpó.
            let (ml_thr_long, ml_thr_short) = crate::calibration::ml_gate_thresholds(
                self.arena.config.ml_threshold_long.load(Ordering::Relaxed),
                self.arena.config.ml_threshold_short.load(Ordering::Relaxed),
            );
            let ml_lift_long = (ml_thr_long - 0.50).clamp(0.02, 0.25);
            let ml_lift_short = (0.50 - ml_thr_short).clamp(0.02, 0.25);
            let effective_ml_long = ml_model_base + ml_lift_long;
            let effective_ml_short = ml_model_base - ml_lift_short;

            let macro_tau = self
                .temporal_spectrum
                .get(coin_id)
                .map(|s| s.macro_resonant_tau_ms())
                .unwrap_or(3_600_000.0);
            let swing_duration_ms = (macro_tau.clamp(180_000.0, 43_200_000.0)).round() as u64;

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
            // MOD2/7-014: antes `hurst_exponent >= 0.48` — quinto literal de
            // banda. Candidato a tendencia = NO anti-persistente (banda
            // canónica 0.45; el umbral fino lo pone trend_threshold/EMA).
            let is_trend_candidate = hurst_exponent >= HURST_ANTI_PERSISTENT
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
                            .max((swing_nn_pred - ml_model_base).max(0.0) * 2.0);
                        let confidence = if raw_conf.is_finite() {
                            raw_conf.tanh().clamp(0.55, 0.95)
                        } else {
                            0.55
                        };
                        slow_intent = SignalIntent {
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
                            .max((ml_model_base - swing_nn_pred).max(0.0) * 2.0);
                        let confidence = if raw_conf.is_finite() {
                            raw_conf.tanh().clamp(0.55, 0.95)
                        } else {
                            0.55
                        };
                        slow_intent = SignalIntent {
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
            if CONSENSUS_BRANCH_ENABLED && slow_intent.signal == SignalType::Flat {
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
                    slow_intent = SignalIntent {
                        signal: tensor_swing.signal,
                        confidence: tensor_swing
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        expected_duration_ms: swing_duration_ms,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_swing.signal == SignalType::Short
                    && is_bear
                    && not_chasing_short
                    && tensor_swing.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    slow_intent = SignalIntent {
                        signal: tensor_swing.signal,
                        confidence: tensor_swing
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        expected_duration_ms: swing_duration_ms,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_cont.signal == SignalType::Long
                    && is_bull
                    && not_chasing_long
                    && tensor_cont.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    slow_intent = SignalIntent {
                        signal: tensor_cont.signal,
                        confidence: tensor_cont
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        expected_duration_ms: swing_duration_ms,
                        // D-678: rama 14 · consenso tensorial.
                        volume_flow_rate: 14.0,
                        ..Default::default()
                    };
                } else if tensor_cont.signal == SignalType::Short
                    && is_bear
                    && not_chasing_short
                    && tensor_cont.net_confidence.abs() > tensor_min_conf * 0.95
                {
                    slow_intent = SignalIntent {
                        signal: tensor_cont.signal,
                        confidence: tensor_cont
                            .net_confidence
                            .abs()
                            .clamp(tensor_min_conf * 0.95, 1.0),
                        horizon: strategy_core::TradeHorizon::Continuous,
                        expected_duration_ms: swing_duration_ms,
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

            if coin_id < self.last_fast_intent.len() {
                self.last_fast_intent[coin_id] = fast_intent;
            }
            if coin_id < self.last_slow_intent.len() {
                self.last_slow_intent[coin_id] = slow_intent;
            }

            // D-431 & #565: Despacho Concurrente Integral Multi-Banda (Sin Canibalismo ni Anulación Ciega).
            // Si ambas bandas ven oportunidades, el sistema preserva la especialización por slot:
            // - fast_intent opera sobre la banda microestructura/rápida (scalp).
            // - slow_intent opera sobre la banda de régimen/portadora (swing).
            // Ambas pueden coexistir en la misma moneda sin pisarse.
            let scalp_free = !coin.positions.scalp.is_open();
            let swing_free = !coin.positions.swing.is_open();

            let mut unified_intent = SignalIntent::flat();
            if fast_intent.signal != SignalType::Flat && slow_intent.signal != SignalType::Flat {
                if fast_intent.signal == slow_intent.signal {
                    let boosted_conf = fast_intent.confidence.max(slow_intent.confidence);
                    // Ambas bandas alineadas direccionalmente: si scalp está libre, capturar micro-impulso;
                    // si scalp ya está ocupado pero swing está libre, reforzar marea macro en swing.
                    let winner_intent = if scalp_free {
                        fast_intent
                    } else if swing_free {
                        slow_intent
                    } else {
                        fast_intent
                    };
                    unified_intent = SignalIntent {
                        signal: fast_intent.signal,
                        confidence: boosted_conf,
                        horizon: strategy_core::TradeHorizon::Continuous,
                        ..winner_intent
                    };
                } else {
                    // Conflicto de banda:
                    // 1. Si sólo un slot está libre, despachar la banda que tiene su slot disponible.
                    // 2. Si ambos están libres, arbitrar por resonancia energética espectral viva.
                    let tau_dom_now = self
                        .temporal_spectrum
                        .get(coin_id)
                        .map(|s| s.dominant_tau_ms)
                        .unwrap_or(30_000.0);
                    const TAU_MID_MS: f64 = 1_138_000.0;
                    let fast_band_governs = tau_dom_now < TAU_MID_MS;

                    let winner = if scalp_free && !swing_free {
                        fast_intent
                    } else if swing_free && !scalp_free {
                        slow_intent
                    } else if fast_band_governs {
                        fast_intent
                    } else {
                        slow_intent
                    };
                    unified_intent = SignalIntent {
                        horizon: strategy_core::TradeHorizon::Continuous,
                        ..winner
                    };
                }
            } else if fast_intent.signal != SignalType::Flat {
                unified_intent = SignalIntent {
                    horizon: strategy_core::TradeHorizon::Continuous,
                    ..fast_intent
                };
            } else if slow_intent.signal != SignalType::Flat {
                unified_intent = SignalIntent {
                    horizon: strategy_core::TradeHorizon::Continuous,
                    ..slow_intent
                };
            }

            // D-474: Invariante Fractal Universal de Horizonte Continuo (Persistencia Browniana Hurst)
            // En el espectro continuo universal, la duración esperada se modula continuamente por Hurst:
            // Hurst < 0.48 (anti-persistente) comprime la duración hacia la microestructura (tau bajo);
            // Hurst >= 0.52 (persistente/trending) expande la duración temporal para capturar la tendencia.
            unified_intent.expected_duration_ms = hurst_duration_modulation(
                unified_intent.expected_duration_ms,
                hurst_exponent,
            );

            // QO-E2a — EL APRENDIZAJE MODULA: el forest online (entrenado
            // con los resultados REALES de cierres previos vía telemetry
            // mmap) opinaba sólo sobre 2 umbrales — su predictor predict_6d
            // no lo llamaba nadie. Ahora: cuando está entrenado (acc>0.55)
            // y predice DESACUERDO fuerte con la intención (prob<0.40 para
            // largo / >0.60 para corto), la confianza se reduce ×0.8.
            // UNILATERAL: el forest nunca AMPLÍA confianza (su acc de
            // clasificación binaria no justifica más agresividad) — sólo
            // puede frenar. Sin desacuerdo o sin entrenamiento: intacto.
            if unified_intent.signal != SignalType::Flat {
                let f6_acc = self
                    .arena
                    .registry
                    .get_for_coin_or(coin_id, "forest6_acc", 0.0);
                if f6_acc > 0.55 {
                    let f6_prob = self
                        .arena
                        .registry
                        .get_for_coin_or(coin_id, "forest6_prob", 0.5);
                    let disagrees = match unified_intent.signal {
                        SignalType::Long => f6_prob < 0.40,
                        SignalType::Short => f6_prob > 0.60,
                        SignalType::Flat => false,
                    };
                    if disagrees && unified_intent.confidence.is_finite() {
                        unified_intent.confidence =
                            (unified_intent.confidence * 0.8).clamp(0.0, 1.0);
                    }
                }
            }

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
            self.diag_dir
                .funnel_begin(unified_intent.signal, !coin.positions.position.is_open());
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
                // MOD2/7-011: tras eliminar la rama 3 de downtrend, esta es la
                // ÚNICA excepción de capitulación extrema del motor (la más
                // tardía y mejor informada del flujo).
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
                // MOD2/7-011: única excepción de blow-off tras eliminar la
                // rama 3 de uptrend.
                let extreme_blowoff = crate::diffusion::atr_stretch_z(p_stretch, crate::diffusion::EMA_SLOW_BARS) > z95
                    && current_obi < -dynamic_obi_thr * 0.8
                    && composite_score < -dynamic_tech_thr;
                if !extreme_blowoff {
                    unified_intent = SignalIntent::flat();
                }
            }

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_MACRO);
            // X-016 plenitud (REHAB-1b): ACONDICIONAMIENTO ESPECTRAL de la
            // entrada unificada. La persistencia de la escala dominante
            // (medida: autocorrelación de sorpresas — tendencia +1, reversión
            // −1, ruido 0) modula la confianza: tendencia confirmada la
            // preserva (×1), ruido la modula suavemente, reversión la castiga si es tendencial.
            // X-016 & #563: ACONDICIONAMIENTO BAJO EL CAMPO MULTIVARIANTE CONTINUO TEMPORAL ESPECTRAL
            // El mercado se comprende en todas sus 32 partes espectrales (1 ns a 146 años) como un campo continuo.
            // X-016 & #564: ACONDICIONAMIENTO INTEGRAL DEL CAMPO CONTINUO TEMPORAL ESPECTRAL
            // Evalúa el tensor espectral en todas sus dimensiones físicas:
            // 1. Marea macro (swing 60% + secular 40%): portadora de energía de fondo.
            // 2. Coherencia espectral multiescala (táctico 40% + swing 35% + secular 25%).
            // 3. Alineación táctica inmediata (no entrar en cuchillo cayendo o clímax).
            // 4. Entropía de Shannon del campo (rechazar desorden térmico).
            if unified_intent.signal != SignalType::Flat {
                if let Some(spec) = self.temporal_spectrum.get(coin_id) {
                    let is_long = unified_intent.signal == SignalType::Long;
                    let sign = if is_long { 1.0 } else { -1.0 };
                    let coherence = spec.spectral_coherence(is_long);
                    let macro_tide = (spec.swing_score() * 0.60 + spec.secular_score() * 0.40) * sign;
                    let tactical_align = spec.tactical_score() * sign;
                    let field = spec.spectral_field(is_long);
                    let tau_dom = spec.dominant_tau_ms;
                    let persist = spec.persistence_at(tau_dom);

                    // LEY DE RESONANCIA CUÁNTICA ESPECTRAL MULTIVARIANTE (#565 & #566):
                    // - Prohibido operar contra la marea macro portadora (macro_tide < 0.00).
                    // - Prohibido operar con interferencia destructiva multiescala (coherence < 0.05).
                    // - Prohibido operar con la banda táctica en contra (tactical_align < 0.00).
                    // - Prohibido comprar Long si macro_trend < 0.0 sin rebote micro agresivo (micro_trend > 0.00012 && ofi > 0.20).
                    // - Prohibido vender Short si macro_trend > 0.0 sin rechazo micro agresivo (micro_trend < -0.00012 && ofi < -0.20).
                    // - LEY DE FASE ARMÓNICA (ANTI-CRESTA / ANTI-VALLE):
                    //   * Comprar Long sólo en fase de absorción de soporte (valle armónico: z_carrier <= 0.35 && z_resonant <= 0.55).
                    //   * Vender Short sólo en fase de rechazo de resistencia (cresta armónica: z_carrier >= -0.35 && z_resonant >= -0.55).
                    //   * Excepción única: persistencia super-crítica (persist > 0.50).
                    // - Prohibido operar en caos térmico desordenado (spectral_entropy > 0.93).
                    let carrier_tau_ms = 14_400_000.0; // 4 horas en la variedad espectral
                    let z_carrier = spec.momentum_z_at(carrier_tau_ms);
                    let z_resonant = spec.momentum_z_at(field.resonant_tau_ms);

                    let wave_phase_ok = if is_long {
                        (z_carrier <= 0.35 && z_resonant <= 0.55) || persist > 0.50
                    } else {
                        (z_carrier >= -0.35 && z_resonant >= -0.55) || persist > 0.50
                    };

                    let micro_rebound_ok = if is_long {
                        macro_trend >= 0.0 || (micro_trend > 0.00012 && ofi_value > 0.20)
                    } else {
                        macro_trend <= 0.0 || (micro_trend < -0.00012 && ofi_value < -0.20)
                    };

                    let phase_res = spec.phase_resonance(60_000.0, 14_400_000.0);
                    let spec_grad = spec.spectral_gradient_at(field.resonant_tau_ms);
                    let spec_grad_ok = if is_long {
                        spec_grad >= -0.05
                    } else {
                        spec_grad <= 0.05
                    };

                    if macro_tide < 0.00
                        || coherence < 0.12
                        || tactical_align < 0.00
                        || !micro_rebound_ok
                        || !wave_phase_ok
                        || field.spectral_entropy > 0.92
                        || field.confluence_ratio < 0.42
                        || phase_res < 0.0
                        || !spec_grad_ok
                    {
                        unified_intent.signal = SignalType::Flat;
                    } else {
                        // Mapeo armónico continuo en el Universo Multivariante Continuo Temporal Espectral:
                        // Elimina la discretización binaria rígida y converge continuamente hacia el centro de masa tau*.
                        let base_tau = if unified_intent.expected_duration_ms > 0 {
                            unified_intent.expected_duration_ms as f64
                        } else {
                            field.resonant_tau_ms
                        };
                        let continuous_tau = (base_tau.ln() * 0.60 + field.resonant_tau_ms.ln() * 0.40).exp();
                        unified_intent.expected_duration_ms = continuous_tau.clamp(30_000.0, 43_200_000.0) as u64;

                        let is_trending_mode = is_confirmed_uptrend || is_confirmed_downtrend;
                        let directional_persist = if is_trending_mode { persist } else { -persist };
                        let coherence_boost = 0.25 * coherence;
                        let entropy_boost = 0.10 * (1.0 - field.spectral_entropy).max(0.0);
                        let persist_boost = 0.15 * directional_persist;
                        let phase_boost = 0.10 * phase_res;
                        let spectral_factor = (1.0 + coherence_boost + entropy_boost + persist_boost + phase_boost).clamp(0.65, 1.45);
                        unified_intent.confidence =
                            (unified_intent.confidence * spectral_factor).clamp(0.10, 0.99);
                    }
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

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_SPECTRAL);
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
                                (unified_intent.signal == SignalType::Long && p_stretch > 1.20)
                                    || (unified_intent.signal == SignalType::Short
                                        && p_stretch < -1.20)
                                    || elapsed_ms < 30_000
                            } else {
                                false
                            }
                        } else {
                            // Giro a contratendencia post-win: requiere al menos 45s de confirmación
                            elapsed_ms < 45_000
                        }
                    } else {
                        // POST-LOSS: Prohibido capitular al final de extensiones extremas (> 1.5 ATR)
                        let is_scalp_long = unified_intent.signal == SignalType::Long;
                        let dir_streak = self.feature_engines[coin_id].get_active_directional_streak(is_scalp_long);
                        let tot_streak = self.feature_engines[coin_id].get_active_total_loss_streak();
                        let effective_streak = dir_streak.max(tot_streak);

                        if !is_same_dir {
                            if effective_streak >= 2 {
                                // Racha de pérdidas consecutivas alternantes: cooldown exponencial antes de revertir
                                let required_ms = match effective_streak {
                                    2 => 600_000,    // 10 minutos
                                    3 => 1_200_000,  // 20 minutos
                                    _ => 1_800_000,  // 30 minutos
                                };
                                let time_veto = elapsed_ms < required_ms;
                                let extreme_stretch_veto = (unified_intent.signal == SignalType::Short && p_stretch < -1.50)
                                    || (unified_intent.signal == SignalType::Long && p_stretch > 1.50);
                                time_veto || extreme_stretch_veto
                            } else if elapsed_ms < 60_000 {
                                true
                            } else {
                                (unified_intent.signal == SignalType::Short && p_stretch < -1.20)
                                    || (unified_intent.signal == SignalType::Long
                                        && p_stretch > 1.20)
                            }
                        } else {
                            // En la MISMA DIRECCIÓN: cooldown si hay racha de pérdidas consecutivas
                            let required_ms = match effective_streak {
                                0 | 1 => 60_000,     // 1 minuto
                                2 => 600_000,       // 10 minutos
                                3 => 1_200_000,     // 20 minutos
                                _ => 1_800_000,     // 30 minutos
                            };
                            let time_veto = elapsed_ms < required_ms;
                            let extreme_stretch_veto = if effective_streak >= 2 {
                                (unified_intent.signal == SignalType::Short && p_stretch < -1.50)
                                    || (unified_intent.signal == SignalType::Long && p_stretch > 1.50)
                            } else {
                                false
                            };
                            time_veto || extreme_stretch_veto
                        }
                    };

                    if whiplash_veto {
                        unified_intent = SignalIntent::flat();
                    }
                }
            }

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_WHIPLASH);
            // MOD2/7-011 (INFORME DECIMOCUARTO, FOCO 2 «Rigidez de filtros»):
            // aquí existía una SEGUNDA evaluación del veto Bayesiano D-472 —
            // la repetición literal de la que ya aplicó al `fast_intent`
            // aguas arriba (misma condición: Long ⇒ composite ≥ 0, Short ⇒
            // composite ≤ 0). Tautología serial: cualquier intención que la
            // primera ya mató no llega aquí, y la que pasa la primera no
            // aporta información nueva al flujo. Se elimina la segunda
            // evaluación; la única vive junto a la generación de señales.
            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_BAYES);
            // D-499: Invariante de Convicción Post-Racha Universal (Cross-Horizon Loss Streak Firewall)
            // Si el activo acumula una racha de 2 o más pérdidas consecutivas activas (direccionales o totales),
            // se exige convicción Bayesiana institucional (|score| >= 0.28, |current_obi| >= 0.18).
            // Si la racha es >= 3, se exige convicción superlativa (|score| >= 0.32, |current_obi| >= 0.22).
            let is_short_intent = unified_intent.signal == SignalType::Short;
            let dir_streak = self.feature_engines[coin_id].get_active_directional_streak(!is_short_intent);
            let other_streak = self.feature_engines[coin_id].get_active_directional_streak(is_short_intent);
            let tot_streak = self.feature_engines[coin_id].get_active_total_loss_streak();
            let is_alternating_chop = dir_streak >= 1 && other_streak >= 1;
            let effective_streak = if is_alternating_chop { dir_streak.max(tot_streak) } else { dir_streak };

            if effective_streak >= 2 {
                let (min_score, min_obi) = if effective_streak >= 3 {
                    (0.32, 0.22)
                } else {
                    (0.28, 0.18)
                };
                let eff_obi = if book_absent { rolling_cvd } else { current_obi };
                if unified_intent.signal == SignalType::Short {
                    if composite_score > -min_score || eff_obi > -min_obi {
                        unified_intent = SignalIntent::flat();
                    }
                } else if unified_intent.signal == SignalType::Long {
                    if composite_score < min_score || eff_obi < min_obi {
                        unified_intent = SignalIntent::flat();
                    }
                }
            }

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_STREAK);
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

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_L2);
            // D-696 (DÉCIMA OLA · auditoría integral): UN SOLO SITIO DECIDE SI LA
            // PREDICCIÓN ESTÁ DE ACUERDO CON LA DIRECCIÓN.
            //
            // Aquí vivía el escudo neuronal (D-473/D-477, con su banda D-688 y la
            // puerta de habilidad D-695): vetaba un largo con `ml_prob < 0,5` y un
            // corto con `ml_prob > 0,5`. B3.18 puso el mismo juicio —con los
            // umbrales del GENOMA, no con el literal ½— en el punto único de
            // entrada. Los dos umbrales salen del mismo `ml_prob` (línea 1717) y
            // `ml_gate_thresholds` garantiza `largo ≥ ½ ≥ corto`, de modo que todo
            // lo que este escudo vetaba lo veta después la puerta genómica: era
            // una segunda fuente de verdad, más laxa, con una excepción de
            // extensión que no cambiaba ninguna apertura.
            //
            // Medición sobre datos reales (junio-julio 2026, 34 M de eventos por
            // ventana) de la puerta de habilidad D-695 frente al escudo sin
            // puerta: aptitud media −0,1428 con la puerta y −0,1290 sin ella. La
            // regla pre-registrada era conservar D-695 sólo si no empeoraba; no la
            // supera, y al unificar el juicio en la puerta genómica desaparece.
            // El `SkillTracker` se conserva como TELEMETRÍA: mide si el ensamble
            // tiene habilidad, sin decidir nada.
            let neural_skill = if coin_id < self.ensembles.len() {
                self.ensembles[coin_id].has_significant_skill()
            } else {
                self.ensemble.has_significant_skill()
            };
            let (ml_gate_long, ml_gate_short) = crate::calibration::ml_gate_thresholds(
                self.arena.config.ml_threshold_long.load(Ordering::Relaxed),
                self.arena.config.ml_threshold_short.load(Ordering::Relaxed),
            );
            let neural_against_long =
                unified_intent.signal == SignalType::Long && ml_prob < ml_gate_long;
            let neural_against_short =
                unified_intent.signal == SignalType::Short && ml_prob > ml_gate_short;
            if neural_against_long || neural_against_short {
                self.diag_dir.record_neural_gate(neural_against_long, neural_skill);
            }

            self.diag_dir.funnel_checkpoint(unified_intent.signal, direction_diag::STAGE_NEURAL);
            let mut new_order = None;

            let is_scalp_candidate = unified_intent.expected_duration_ms > 0
                && unified_intent.expected_duration_ms <= 300_000;
            let is_swing_candidate = unified_intent.expected_duration_ms > 300_000;

            // Especialización física continua por longitud de onda espectral:
            // Cada intención se despacha a su propio slot especializado sin canibalismo ni mezclas.
            // Si el slot primario está ocupado, el slot continuo universal absorbe el flujo armónico si está disponible.
            let (target_pos_slot, pos_h, slot_available) = if is_scalp_candidate {
                (0usize, quantum_arena::position::PositionHorizon::Scalping, !coin.positions.scalp.is_open())
            } else if is_swing_candidate {
                (1usize, quantum_arena::position::PositionHorizon::Swing, !coin.positions.swing.is_open())
            } else {
                (2usize, quantum_arena::position::PositionHorizon::Continuous, !coin.positions.position.is_open())
            };

            // --- APERTURA MULTI-HORIZONTE CONTINUA INTEGRAL ---
            if unified_intent.signal != SignalType::Flat && slot_available {
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
                self.diag_dir.record_risk(
                    calibrated_intent.signal == SignalType::Long,
                    calibrated_intent.confidence,
                    order.signal != SignalType::Flat,
                );
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

                    let current_spread_bps = if mid_price > 1e-8 && ask >= bid {
                        ((ask - bid) / mid_price) * 10_000.0
                    } else {
                        0.5
                    };
                    let slip_bps = ((current_spread_bps * 0.5) + 0.5).clamp(0.5, 500.0);

                    let council_horizon =
                        metacortex_engine::consejo_seniors::TradingHorizon::Continuous;

                    // MOD2/7-006 (INFORME DECIMOCUARTO): el consejo delibera
                    // ahora sobre perspectivas GENUINAMENTE diversas, cada una
                    // con su propio dato crudo. El asiento "grafos" leía
                    // impulse_mom del lead_lag_engine — computado A PARTIR del
                    // OBI, colineal por construcción — y fue retirado del
                    // payload. Fuentes nuevas, todas ya en scope del tick:
                    //   - fused_score/persistence: espectro temporal (X-016);
                    //   - atr_pct: v_t relativo al precio;
                    //   - loss_streak: racha de pérdidas de la dirección;
                    //   - ml_prob: ensamble PURO (sin spot_bias, MOD2/7-029) —
                    //     un despegue del spot no puede mover al asiento ML;
                    //   - intended_direction: la entrada bajo deliberación
                    //     (contexto de los moduladores Riesgo/Volatilidad).
                    let (council_fused, council_persistence) = self
                        .temporal_spectrum
                        .get(coin_id)
                        .map(|spec| {
                            (
                                spec.fused_score.clamp(-1.0, 1.0),
                                spec.persistence_at(spec.dominant_tau_ms).clamp(-1.0, 1.0),
                            )
                        })
                        .unwrap_or((0.0, 0.0));
                    let council_atr_pct = if mid_price > 1e-8 {
                        (self.feature_engines[coin_id].v_t / mid_price).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let council_loss_streak = self
                        .feature_engines[coin_id]
                        .get_active_directional_streak(order.signal == SignalType::Long)
                        .max(self.feature_engines[coin_id].get_active_total_loss_streak());
                    let council_intended_dir = match order.signal {
                        SignalType::Long => 1.0,
                        SignalType::Short => -1.0,
                        SignalType::Flat => 0.0,
                    };

                    let council_snapshot =
                        metacortex_engine::consejo_seniors::MarketSnapshotPayload {
                            horizon: council_horizon,
                            book_imbalance: obi,
                            hurst_exponent: hurst_val.clamp(0.0, 1.0),
                            ml_prob: ml_prob_pure.clamp(0.0, 1.0),
                            fused_score: council_fused,
                            persistence: council_persistence,
                            atr_pct: council_atr_pct,
                            loss_streak: council_loss_streak,
                            intended_direction: council_intended_dir,
                            do_calculus_risk: vpin_risk,
                            causal_veto_threshold: 0.75,
                            current_drawdown_pct: drawdown,
                            estimated_slippage_bps: slip_bps,
                            // U-4: τ dominante — los asientos interpolan sus
                            // umbrales en el continuo, sin etiquetas de horizonte.
                            dominant_tau_ms: self
                                .temporal_spectrum
                                .get(coin_id)
                                .map(|s| s.dominant_tau_ms)
                                .unwrap_or(1_138_000.0),
                            // P-5b: datos EXCLUSIVOS del asiento Ente del
                            // Mercado — ballena (z de burst del @trade real),
                            // cascada (PEEK: observa sin robarle el evento al
                            // camino per-tick), apalancamiento (OI per-símbolo).
                            whale_burst_z: self
                                .arena
                                .registry
                                .get_for_coin_or(coin_id, "whale_burst_z", 0.0),
                            liquidation_severity:
                                crate::liquidation_feed::peek_pending(),
                            open_interest_norm: coin
                                .open_interest_norm
                                .load(Ordering::Relaxed),
                            spoof_score: self
                                .arena
                                .registry
                                .get_for_coin_or(coin_id, "spoof_score", 0.0),
                            // QO-U2: sentimiento de masas contrarian
                            crowd_ls_ratio: self
                                .arena
                                .registry
                                .get_scoped_value_or(&sym, "ls_account_ratio", 1.0),
                            crowd_taker_ratio: self
                                .arena
                                .registry
                                .get_scoped_value_or(&sym, "taker_ratio", 1.0),
                            ml_model_base,
                        };
                    let wr = coin.metrics.win_rate.load(Ordering::Relaxed);
                    let senior_sigs = self
                        .consejo_deliberacion
                        .extract_senior_signals(&council_snapshot, wr);
                    // C-07 (INFORME DECIMOCUARTO): la deliberación es UNA por
                    // trade. Antes se guardaban las MISMAS señales en los slots
                    // scalp y swing y el cierre llamaba record_outcome con AMBOS
                    // → cada trade duplicado en el tracker de pesos adaptativos
                    // (dataset 2×, window_size al 50% de historia real). El
                    // swing queda zeroed (la erradicación swing/scalp es
                    // cosmética; sólo queda el slot scalp, ahora "la" señal).
                    if coin_id < self.last_senior_signals.len() {
                        self.last_senior_signals[coin_id] = senior_sigs;
                    }
                    let deliberation = self.consejo_deliberacion.deliberar_with_weights(
                        &council_snapshot,
                        wr,
                        None,
                    );
                    // D-738 (DÉCIMA OLA · auditoría integral): EL CONSEJO APRUEBA
                    // UNA OPERACIÓN, NO «TENGO UNA OPINIÓN».
                    //
                    // `approved` sólo decía que existía supermayoría de ALGO: su
                    // dirección viaja en `final_signal`, que no leía nadie en todo
                    // el repositorio. Con el libro dado la vuelta (OBI −0,40,
                    // momento −0,35) el Consejo alcanzaba un 92 % de consenso
                    // BAJISTA, marcaba `approved = true`, y el llamador abría el
                    // LARGO que traía el risk-engine: el órgano que existe para
                    // vetar entradas contra la microestructura las bendecía.
                    // Ahora se exige que el consenso sea del lado que se va a
                    // operar.
                    let quiere_largo = order.signal == SignalType::Long;
                    let consejo_en_la_misma_direccion = if quiere_largo {
                        deliberation.final_signal > 0.0
                    } else {
                        deliberation.final_signal < 0.0
                    };
                    let aprobado_por_consejo =
                        deliberation.approved && consejo_en_la_misma_direccion;
                    if !aprobado_por_consejo {
                        self.diag_council_vetoes += 1;
                        self.diag_dir.record_council(quiere_largo, false);
                    }

                    // B3.18 — LA PREDICCIÓN DECIDE. Descubrimiento 2026-09-15:
                    // el PnL era INSENSIBLE al forest (BNB sept idéntico al
                    // centavo con modelos distintos) porque las ramas de
                    // entrada gatean con NN/flujo/tendencia y el ensamble
                    // (forest validado ⊕ NN ⊕ espectro) no consumía nadie.
                    // Ahora TODA entrada exige el acuerdo del ensamble:
                    // long ⇒ ml ≥ umbral genómico, short ⇒ ml ≤ umbral.
                    //
                    // B3.18-aud (FRESCURA): se lee la variable LOCAL ml_prob_pure
                    // — computada en el bloque de ANALÍTICA COMPLETA ~1.5k
                    // líneas arriba, DENTRO de esta misma invocación, ANTES de
                    // las ramas de señal/deliberación (orden verificado).
                    // MOD2/7-029: el gate lee el ensamble PURO (sin
                    // spot_bias); la versión con sesgo alimenta las ramas de
                    // señal. B3.25: sin modelo validado del roster, no se opera.
                    // B3.36 — GATE POR LIFT sobre la base del SÍMBOLO: con el
                    // etiquetado honesto (base ~30%) el gate absoluto volvía
                    // el sistema short-only por artefacto de escala.
                    // D-715 (unión): los genes se reparan ANTES por
                    // ml_gate_thresholds (largo ≥ ½ ≥ corto, finitud) — la
                    // ÚNICA reparación de esos dos genes en el motor — y el
                    // umbral YA reparado es el que se reinterpreta como lift.
                    let ml_now = ml_prob_pure;
                    let (ml_thr_long_gate, ml_thr_short_gate) = crate::calibration::ml_gate_thresholds(
                        self.arena.config.ml_threshold_long.load(Ordering::Relaxed),
                        self.arena.config.ml_threshold_short.load(Ordering::Relaxed),
                    );
                    let ml_lift_long = (ml_thr_long_gate - 0.50).clamp(0.02, 0.25);
                    let ml_lift_short = (0.50 - ml_thr_short_gate).clamp(0.02, 0.25);
                    // S-6 (ESPECTRALIZACIÓN): el lift exigido respira con el
                    // ACUERDO espectral — cuando la fusión del espectro apunta
                    // en la MISMA dirección que el modelo, la exigencia baja
                    // (×0.7); cuando divergen, sube (×1.3). agree ∈ [-1,1].
                    let agree = (council_fused
                        * (ml_now - ml_model_base).signum())
                        .clamp(-1.0, 1.0);
                    let lift_eff_long = (ml_lift_long * (1.0 - 0.3 * agree)).clamp(0.01, 0.30);
                    let lift_eff_short = (ml_lift_short * (1.0 - 0.3 * agree)).clamp(0.01, 0.30);
                    let ml_gate_ok = has_roster_model
                        && if order.signal == SignalType::Long {
                            ml_now >= ml_model_base + lift_eff_long
                        } else {
                            ml_now <= ml_model_base - lift_eff_short
                        };
                    if aprobado_por_consejo && !ml_gate_ok {
                        self.diag_ml_vetoes += 1;
                        if self.diag_ml_vetoes % 50 == 1 {
                            telemetry_server::telemetry_log!(
                                "🧠 [ML-GATE] {} entradas vetadas por el ensamble (última: {} ml={:.3}) — la predicción decide",
                                self.diag_ml_vetoes,
                                quantum_arena::symbol_registry::try_symbol(coin_id)
                                    .unwrap_or_default(),
                                ml_now
                            );
                        }
                    }

                    if aprobado_por_consejo && ml_gate_ok {
                        let is_long = order.signal == SignalType::Long;
                        // MOD6/8-010: lector saturado — un used_margin
                        // levemente negativo (drift de doble liberación)
                        // NO puede inflamar free_cap ni el chequeo de
                        // colchón.
                        let total_used = self.arena.used_margin_saturated();
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

                                let tau_coin = self
                                    .temporal_spectrum
                                    .get(coin_id)
                                    .map(|s| s.dominant_tau_ms)
                                    .unwrap_or(60_000.0);
                                let target_pos = match target_pos_slot {
                                    0 => &coin.positions.scalp,
                                    1 => &coin.positions.swing,
                                    _ => &coin.positions.position,
                                };

                                target_pos.open_with_fee(
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
                                // QO-E2b — PRODUCTOR DEL DATASET NN: congelar
                                // el tensor 54D de la APERTURA. El cierre lo
                                // empareja con el retorno neto realizado y
                                // appenda la fila a data/dark_alpha_dataset_
                                // {SYM}.csv — el auto-trainer NN esperaba ese
                                // archivo desde su creación y NADIE lo
                                // escribía (lazo doble-muerto).
                                {
                                    let tensor_snapshot = self.build_54d_tensor(
                                        coin_id,
                                        bid_qty,
                                        ask_qty,
                                        mid_price,
                                        omni_features,
                                    );
                                    if let Ok(mut t) =
                                        target_pos.nn_entry_tensor.lock()
                                    {
                                        *t = tensor_snapshot.to_vec();
                                    }
                                }
                                // REHAB-1b & #560: la posición NACE con su τ dominante
                                // VIVA del espectro continuo — horizonte físico real, no etiqueta discreta.
                                let tau_entry = if calibrated_intent.expected_duration_ms > 0 {
                                    calibrated_intent.expected_duration_ms
                                } else {
                                    tau_coin
                                        .clamp(
                                            quantum_arena::temporal_spectrum::TAU_ANCHOR_FAST_MS,
                                            quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS,
                                        )
                                        .round() as u64
                                };
                                target_pos
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
            false, // CERT-M2-H01: wrapper legacy — siempre actualiza macro
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

/// MOD2/7-014 (INFORME DECIMOCUARTO, FOCO 2 «Rigidez de filtros»):
/// definición CANÓNICA única de las bandas del exponente de Hurst.
/// Antes convivían seis literales (0.42/0.45/0.48/0.49/0.51/0.52)
/// clasificando el MISMO estimador con solapes mutuamente excluyentes
/// según la rama (p.ej. H=0.43 era «anti-persistente» para el fallback
/// price-action pero «neutral» para el gate de OBI). Una sola semántica:
/// anti-persistente = H < 0.45 · persistente = H > 0.52 · neutral = [0.45, 0.52].
pub const HURST_ANTI_PERSISTENT: f64 = 0.45;
pub const HURST_PERSISTENT: f64 = 0.52;

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

#[cfg(test)]
mod tests_b3_ml_wiring {
    //! B3.18-aud — regresiones del cableado ML del core:
    //! 1. el ml_prob que lee el gate es del MISMO tick (frescura),
    //! 2. el forest ({SYM}_SCALP) realmente pesa en ese ml_prob vía el
    //!    ensamble (con el bloque MACRO del contrato 48D en dims 44..48),
    //! 3. la analítica ML sigue viva con el feed stalled (X-012/B2.5-fix):
    //!    el bloqueo es de ENTRADAS, jamás de análisis.

    use super::*;
    use crate::ml_inference::{NanoForest, NanoForestData, GLOBAL_FORESTS};
    use std::sync::atomic::Ordering;

    /// Forest cuyo ÚNICO split vive en la dim 44 — el primer slot del bloque
    /// MACRO (B3.4) del vector 48D. VIX<20 ⇒ macro[0]<0 ⇒ hoja -3 (p≈0.047);
    /// VIX>20 ⇒ macro[0]>0 ⇒ hoja +3 (p≈0.953). Si el vector vivo no llevara
    /// el bloque macro de ESTE tick, el split leería 0.0 y la predicción no
    /// podría moverse entre ticks.
    fn macro_split_forest() -> NanoForestData {
        NanoForestData {
            children_left: vec![1, -1, -1],
            children_right: vec![2, -1, -1],
            feature: vec![44, -1, -1],
            threshold: vec![0.0, 0.0, 0.0],
            value: vec![0.0, -3.0, 3.0],
            tree_offsets: vec![0, 3],
            init_score: 0.0,
        }
    }

    fn install_testusdt_forest() {
        quantum_arena::symbol_registry::update_registry(vec![
            quantum_arena::symbol_registry::get_official_binance_spec("TESTUSDT"),
        ]);
        let forest = NanoForest::from_data(macro_split_forest()).unwrap();
        let current = GLOBAL_FORESTS.load();
        let mut map = (**current).clone();
        map.insert("TESTUSDT_MOTOR".to_string(), Arc::new(forest));
        GLOBAL_FORESTS.store(Arc::new(map));
    }

    fn tick(core: &mut GodEngineCore, omni: &[f64; 54], t_ms: u64) {
        core.process_event(
            0, true, false, false, 100.0, 1.0, 99.99, 100.01, 5.0, 5.0, 0.0, 0.0, t_ms, false,
            omni, false,
        );
    }

    #[test]
    fn b3_18_el_ml_prob_del_gate_es_del_mismo_tick_y_lleva_al_forest() {
        install_testusdt_forest();
        // D-714: pila suficiente para construir el arena.
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let mut core = GodEngineCore::new(Arc::clone(&arena));
        // Ensamble forest-only: sin NN la opinión combinada ES la del forest
        // (peso Hedge inicial 1.0) — determinista para la aserción.
        core.swing_nn = None;

        // Tick 1 — VIX=10 ⇒ macro dim44 = (10-20)/10 = -1 ⇒ hoja -3.
        let mut omni = [0.0f64; 54];
        omni[24] = 10.0;
        tick(&mut core, &omni, 1_000);
        let ml_bear = arena.coins[0].ml_prob.load(Ordering::Relaxed);
        let exp_bear = 1.0 / (1.0 + (3.0f64).exp()); // sigmoid(-3) ≈ 0.04743
        assert!(
            (ml_bear - exp_bear).abs() < 1e-6,
            "ml_prob={} debía ser la opinión del forest {}",
            ml_bear,
            exp_bear
        );
        assert_eq!(core.last_ml_prob as f64, ml_bear);

        // Tick 2 — MISMA cadena, VIX=30 ⇒ dim44 = +1 ⇒ hoja +3. El valor que
        // lee el gate (ml_prob local / coin.ml_prob) debe ser el de ESTE tick:
        // si el gate leyera el del tick anterior, veríamos 0.047.
        omni[24] = 30.0;
        tick(&mut core, &omni, 2_000);
        let ml_bull = arena.coins[0].ml_prob.load(Ordering::Relaxed);
        let exp_bull = 1.0 / (1.0 + (-3.0f64).exp()); // sigmoid(+3) ≈ 0.95257
        assert!(
            (ml_bull - exp_bull).abs() < 1e-6,
            "ml_prob={} es stale: el gate habría leído el tick anterior",
            ml_bull
        );
        // El ensamble que alimenta el gate refleja la mezcla de ESTE tick.
        assert!((core.ensembles[0].combined().unwrap() - exp_bull).abs() < 1e-6);
    }

    #[test]
    fn b2_5_ml_prob_sigue_vivo_con_feed_stalled() {
        install_testusdt_forest();
        // D-714: pila suficiente para construir el arena.
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let mut core = GodEngineCore::new(Arc::clone(&arena));
        core.swing_nn = None;

        quantum_arena::feed_health::stall();
        let mut omni = [0.0f64; 54];
        omni[24] = 30.0;
        // Con el feed stalled las ENTRADAS se bloquean (ninguna orden)…
        let (new_order, closed_order) = core.process_event(
            0, true, false, false, 100.0, 1.0, 99.99, 100.01, 5.0, 5.0, 0.0, 0.0, 1_000, false,
            &omni, false,
        );
        assert!(new_order.is_none(), "entradas bloqueadas con feed stalled");
        assert!(closed_order.is_none());
        // …pero la analítica ML corrió completa: ml_prob fresco del forest,
        // no el 0.5 por defecto ni un valor stale. Al recuperar el feed, el
        // PRIMER tick ya decide el gate con predicción viva.
        let ml = arena.coins[0].ml_prob.load(Ordering::Relaxed);
        let exp_bull = 1.0 / (1.0 + (-3.0f64).exp());
        assert!(
            (ml - exp_bull).abs() < 1e-6,
            "analítica ML no debe congelarse con feed stalled: ml={ml}"
        );
        quantum_arena::feed_health::clear();
    }
}

#[cfg(test)]
mod tests_m5_h01 {
    //! M5-H01 — paridad del fitness de Darwin con el canónico
    //! (evolution-engine::fitness::compute). El core no puede importarlo
    //! (ciclo de deps), así que este módulo CLAVA el contrato: gate
    //! min_trades=30, clamp de dd∈[0,1], curva base growth − λ·dd².

    use super::fitness_compute;

    #[test]
    fn m5_h01_gate_min_trades() {
        // <30 cierres ⇒ INVIABLE aunque el crecimiento sea espectacular:
        // 1-2 trades de suerte no pueden liderar el leaderboard de Darwin.
        assert_eq!(fitness_compute(100.0, 200.0, 0.0, 0), f64::NEG_INFINITY);
        assert_eq!(fitness_compute(100.0, 200.0, 0.0, 29), f64::NEG_INFINITY);
        // Exactamente 30 ⇒ viable, y sin dd el fitness es el crecimiento log.
        let f = fitness_compute(100.0, 200.0, 0.0, 30);
        assert!((f - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn m5_h01_dd_clamp_y_lambda() {
        // Ancla λ=4·ln2: dd 50% con crecimiento 0 cuesta exactamente ln(2).
        let f = fitness_compute(100.0, 100.0, 0.5, 100);
        assert!((f + std::f64::consts::LN_2).abs() < 1e-12);
        // dd>100% se clampa a 1: el castigo no crece más allá del ancla.
        let f_clamped = fitness_compute(100.0, 100.0, 1.7, 100);
        let f_full = fitness_compute(100.0, 100.0, 1.0, 100);
        assert!((f_clamped - f_full).abs() < 1e-12);
        // Ruina / datos inválidos siguen siendo INVIABLE con trades de sobra.
        assert_eq!(fitness_compute(0.0, 100.0, 0.0, 100), f64::NEG_INFINITY);
        assert_eq!(fitness_compute(100.0, -1.0, 0.0, 100), f64::NEG_INFINITY);
    }
}
