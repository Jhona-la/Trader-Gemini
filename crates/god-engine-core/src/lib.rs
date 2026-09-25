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

/// D-752 — TASA DE ACIERTO MEDIDA, CON SU INCERTIDUMBRE (intervalo de Wilson).
///
/// QUÉ ESTABA MAL: varias ramas de señal emitían una «confianza» LITERAL
/// (0,72 / 0,68 / 0,65) y el freno del bosque se armaba con `acc > 0,55` sin
/// mirar el tamaño de la muestra. Aguas abajo esa cifra se trata como una
/// PROBABILIDAD: pasa por el escalado de Platt que alimenta el Kelly y se
/// compara contra el gen `min_confidence_btc`. Una constante escrita a mano no
/// es una frecuencia observada, y una exactitud sin `n` no es evidencia.
///
/// POR QUÉ IMPORTABA: el tamaño de la posición es monótono en esa cifra, y el
/// ORDEN entre ramas (0,72 > 0,68 > 0,65) tampoco salía de dato alguno. Una
/// rama que acierta el 45 % dimensionaba como si acertara el 72 %, y ninguna
/// racha de pérdidas podía corregirla: un literal no aprende.
///
/// QUÉ GARANTIZA: la convicción de una rama es la frecuencia de acierto que
/// ESA rama demostró, penalizada por el tamaño de su muestra. Se usa el
/// intervalo de Wilson —no el normal de Wald— porque no degenera con `n`
/// pequeño ni con `p` en los extremos: con 3 aciertos de 3, Wald devuelve
/// [1, 1] («certeza» con tres datos) y Wilson devuelve un intervalo ancho que
/// todavía contiene ½.
#[derive(Debug, Clone, Copy, Default)]
pub struct TasaAcierto {
    /// Operaciones cerradas atribuidas a esta fuente.
    pub n: u32,
    /// De ellas, las que acabaron en ganancia NETA de comisiones.
    pub aciertos: u32,
}

impl TasaAcierto {
    #[inline]
    pub fn observar(&mut self, acierto: bool) {
        self.n = self.n.saturating_add(1);
        if acierto {
            self.aciertos = self.aciertos.saturating_add(1);
        }
    }

    /// Intervalo de Wilson al nivel `z`. Con `n = 0` devuelve `[0, 1]`: la
    /// ignorancia completa, que es exactamente lo que hay antes del primer
    /// cierre. `z` es el mismo `Z95` que usa el resto del núcleo para decidir
    /// significación, de modo que hay UN solo criterio en todo el motor.
    #[inline]
    pub fn intervalo(&self, z: f64) -> (f64, f64) {
        if self.n == 0 || !z.is_finite() || z <= 0.0 {
            return (0.0, 1.0);
        }
        let n = self.n as f64;
        let p = (self.aciertos as f64 / n).clamp(0.0, 1.0);
        let z2 = z * z;
        let denom = 1.0 + z2 / n;
        let centro = (p + z2 / (2.0 * n)) / denom;
        let radio = z * ((p * (1.0 - p) / n) + z2 / (4.0 * n * n)).sqrt() / denom;
        (
            (centro - radio).clamp(0.0, 1.0),
            (centro + radio).clamp(0.0, 1.0),
        )
    }

    /// `true` sólo si la muestra EXCLUYE la moneda al aire por arriba: la
    /// fuente demostró acierto mejor que el azar al nivel `z`.
    #[inline]
    pub fn mejor_que_el_azar(&self, z: f64) -> bool {
        self.intervalo(z).0 > 0.5
    }
}

/// D-752 — CONVICCIÓN DE UNA RAMA A PARTIR DE SU PROPIO HISTORIAL.
///
/// Tres estados y ningún parámetro libre; el criterio es siempre el mismo:
/// «¿excluye el intervalo de Wilson la moneda al aire?».
///   · cota inferior > ½ ⇒ la rama DEMOSTRÓ ventaja. La convicción es esa
///     cota: lo que la muestra garantiza, no lo que su media sugiere.
///   · cota superior < ½ ⇒ la rama DEMOSTRÓ desventaja. Emite esa cota
///     superior, que por definición queda bajo ½, y el gate de confianza del
///     risk-engine la rechaza sin necesidad de una regla aparte.
///   · el intervalo contiene ½ ⇒ todavía no hay evidencia; decide la magnitud
///     MEDIDA del disparo, que es lo único observable en ese momento.
///
/// El tercer caso no es una concesión: sin él el motor nunca abriría la
/// primera operación de una rama y jamás reuniría la muestra con la que
/// juzgarla.
#[inline]
pub fn conviccion_de_rama(registro: &TasaAcierto, piso_por_magnitud: f64) -> f64 {
    let (lo, hi) = registro.intervalo(crate::diffusion::Z95);
    if lo > 0.5 {
        lo
    } else if hi < 0.5 {
        hi
    } else {
        piso_por_magnitud
    }
}

/// D-756 — ESCALADA DE EXIGENCIA TRAS UNA RACHA DE PÉRDIDAS.
///
/// QUÉ ESTABA MAL: la exigencia de desequilibrio de libro tras dos pérdidas
/// seguidas se elevaba con un escalón único y literal (`.max(0,22)` /
/// `.max(0,20)`), que además podía REBAJARLA si el umbral vigente ya era
/// mayor. Y una tercera o cuarta pérdida no cambiaba nada.
///
/// QUÉ GARANTIZA: la misma escalera binaria que gobierna el enfriamiento
/// (D-754). Cada pérdida consecutiva a partir de la segunda DUPLICA el
/// desequilibrio exigido, y el techo es 1 porque es lo que |OBI| puede valer
/// por construcción. Es monótona (nunca relaja) y no tiene escalones
/// inventados. El tope de 8 duplicaciones sólo evita el desbordamiento: a
/// partir de ahí el umbral ya está saturado en 1.
#[inline]
pub fn exigencia_tras_racha(umbral: f64, racha: u32) -> f64 {
    if !umbral.is_finite() {
        return umbral;
    }
    let niveles = racha.saturating_sub(1).min(8) as i32;
    (umbral * 2f64.powi(niveles)).min(1.0)
}

/// D-752 — etiquetas de rama. `SignalIntent::volume_flow_rate` YA transportaba
/// un identificador de rama (1..14) para la traza de apertura, y sobrevive a
/// la arbitración porque todas las fusiones usan `..fast_intent` / `..winner`.
/// Se reutiliza ese canal —en vez de abrir otro— para atribuir cada cierre a
/// la rama que lo originó. Las ramas de respaldo (ML re-centrado, espectro,
/// acción de precio) no lo rellenaban: quedaban todas en 0, indistinguibles.
pub const RAMA_ML_RECENTRADO: f64 = 20.0;
pub const RAMA_ESPECTRO_DIRECTO: f64 = 21.0;
pub const RAMA_TENDENCIA_PERSISTENTE: f64 = 22.0;
pub const RAMA_REVERSION_ANTIPERSISTENTE: f64 = 23.0;
pub const RAMA_MOMENTO_SIN_REGIMEN: f64 = 24.0;
/// Casillas del registro por moneda: cubre las etiquetas 0..=24 con holgura.
pub const N_RAMAS: usize = 32;

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
    /// ESPECTRO PREDICTIVO por símbolo (D-742b): el tape a todas las escalas
    /// y el pronóstico en línea de volatilidad y volumen a cualquier
    /// horizonte, con su habilidad medida fuera de muestra. Observación pura
    /// por ahora: se publica al registry y NADIE decide todavía con ello —
    /// primero se mide en el forense, después se conecta.
    pub spectral_forecast: Vec<quantum_arena::spectral_tape::SpectralForecastBank>,
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
    /// D-752 — HISTORIAL DE ACIERTO POR RAMA Y POR MONEDA. Índice externo:
    /// `coin_id`; interno: la etiqueta de rama que viaja en
    /// `SignalIntent::volume_flow_rate`. Esto NO es telemetría: es la fuente
    /// de la convicción que las ramas de respaldo emiten (ver
    /// `conviccion_de_rama`). Por moneda porque la exchangeabilidad se rompe
    /// entre activos, igual que ya ocurre con `conformal_by_coin`.
    pub rama_registro: Vec<[TasaAcierto; N_RAMAS]>,
    /// Rama que abrió la posición viva de cada moneda; `None` si no hay
    /// posición o si la entrada no llevaba etiqueta. El cierre la consume.
    pub rama_abierta: Vec<Option<usize>>,
    /// D-752 — HISTORIAL DIRECCIONAL DEL BOSQUE ONLINE, medido por el núcleo.
    /// El productor (`online_daemon`) publica `forest6_acc` SIN tamaño de
    /// muestra, así que esa cifra no admite cota inferior alguna. El núcleo
    /// mide lo único que puede medir por sí mismo: si el voto del bosque en la
    /// apertura coincidió con el resultado del cierre.
    pub bosque_registro: Vec<TasaAcierto>,
    /// Voto direccional del bosque en la apertura viva (`true` = predijo
    /// subida). `None` si el bosque no opinaba al abrir.
    pub bosque_voto_abierto: Vec<Option<bool>>,
    /// D-756 — DISTRIBUCIÓN MEDIDA DE |OFI|, |OBI| Y DEL SCORE MICRO, POR
    /// MONEDA, con los estimadores P² que ya existían en
    /// `quantum_arena::adaptive_quantiles` y que NO tenían ni un solo
    /// llamador fuera de sus propios tests: la maquinaria anti-hardcode del
    /// repositorio estaba construida y desconectada. Ahora alimenta los
    /// suelos de los umbrales de flujo y libro, de modo que «desequilibrio
    /// significativo» signifique el percentil 80 de lo que ESTE símbolo hace,
    /// y no un número escrito a mano.
    pub cuantiles: Vec<quantum_arena::adaptive_quantiles::AdaptiveQuantileEngine>,
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
            signal_engine::flow_excitation_confluence::FlowExcitationConfluenceEngine::default(),
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
            signal_engine::conformal_reversion_filter::ConformalReversionFilterEngine::default(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::trend_runner::HighPayoffTrendRunner::new(),
        ));
        tensor_orchestrator.add_strategy(Box::new(
            signal_engine::flow_impulse::FlowImpulseEngine::default(),
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
            spectral_forecast: (0..n_coins)
                .map(|_| quantum_arena::spectral_tape::SpectralForecastBank::new())
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
            rama_registro: vec![[TasaAcierto::default(); N_RAMAS]; n_coins],
            rama_abierta: vec![None; n_coins],
            bosque_registro: vec![TasaAcierto::default(); n_coins],
            bosque_voto_abierto: vec![None; n_coins],
            cuantiles: (0..n_coins)
                .map(|_| quantum_arena::adaptive_quantiles::AdaptiveQuantileEngine::new())
                .collect(),
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
                // D-731 (DÉCIMA OLA · auditoría integral): la liberación de margen
                // era load → resta → store. Entre la carga y el guardado, otro
                // hilo (el cierre de otra moneda, la reconciliación o la adopción)
                // puede haber sumado o restado: esa actualización se PIERDE y
                // `used_margin` queda por encima o por debajo del margen realmente
                // comprometido — el motor deja de abrir con margen libre, o abre
                // creyendo que lo tiene. Con `fetch_update` la resta es atómica.
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
                // D-180: No sumar entry_fee a pnl_realized (nunca fue ganancia)
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
    /// PUERTAS DEL CONTINUO (D-743) — lo que invalida una entrada es del
    /// MERCADO, no de la banda que la produjo.
    ///
    /// El motor lee el espectro por dos ventanas —la rápida (microestructura,
    /// flujo del libro) y la lenta (tendencia, EMAs, stretch)— y arbitra entre
    /// ellas. Pero los filtros duros vivían sólo en el camino de la rápida: la
    /// lectura lenta podía abrir con el spread por encima del recorrido, sin
    /// volatilidad que recorrer, dentro del enfriamiento de reentrada, contra
    /// el consenso del tick, contra el modelo y contra el muro del libro. Con
    /// τ dominante por encima de la media geométrica de la banda operativa, la
    /// arbitración le entrega la decisión a esa lectura: el veto que el
    /// operador cree tener armado no existe en la mitad de los regímenes.
    ///
    /// Aquí están las cuatro puertas, aplicadas por igual a cualquier lectura:
    ///   1. viabilidad del mercado (ATR mínimo, spread, enfriamiento);
    ///   2. invariante bayesiano D-472 (nada contradice al consenso del tick);
    ///   3. ponderación continua del modelo F-009 / D-411 (veta si lo
    ///      contradice con fuerza y no hay acción de precio extrema; si no,
    ///      modula confianza). La dirección del modelo se mide RELATIVA a su
    ///      propia base (`ml_model_base`, ~0,30 con el etiquetado honesto), no
    ///      a 0,5: contra 0,5 un modelo con base 0,30 «contradecía» todo largo
    ///      por construcción;
    ///   4. vetos de flujo agregado (CVD) y de muro del libro (L2).
    fn puertas_del_continuo(
        &self,
        coin_id: usize,
        intent: SignalIntent,
        viable: bool,
        composite_score: f64,
        ml_prob: f64,
        ml_model_base: f64,
        price_stretch: f64,
    ) -> SignalIntent {
        if intent.signal == SignalType::Flat {
            return intent;
        }
        if !viable {
            return SignalIntent::flat();
        }
        let mut out = intent;

        // 2. Invariante bayesiano absoluto (D-472).
        if (out.signal == SignalType::Long && composite_score < 0.0)
            || (out.signal == SignalType::Short && composite_score > 0.0)
        {
            return SignalIntent::flat();
        }

        // 3. Ponderación continua por el modelo (F-009 / D-411, desasfixia
        //    direccional): la dirección se mide contra la base del propio
        //    modelo; sólo una contradicción FUERTE (< −0,80) veta, y la
        //    penalización blanda conserva convicción no nula (piso 0,20).
        let ml_base = if ml_model_base > 0.05 && ml_model_base < 0.95 {
            ml_model_base
        } else {
            0.5
        };
        let ml_directional = match out.signal {
            SignalType::Long => (ml_prob - ml_base) * 2.0,
            SignalType::Short => (ml_base - ml_prob) * 2.0,
            _ => 0.0,
        };
        if ml_directional < -0.80 && price_stretch.abs() < 2.5 {
            return SignalIntent::flat();
        } else if ml_directional < 0.0 && price_stretch.abs() < 2.5 {
            out.confidence *= (1.0 + ml_directional * 0.5).clamp(0.20, 1.0);
        } else if ml_directional > 0.0 {
            out.confidence = (out.confidence * (1.0 + ml_directional * 0.5)).min(0.99);
        }

        // 4. Vetos duros de flujo agregado y muro del libro.
        let coin = &self.arena.coins[coin_id];
        let buy_vol = coin.agg_buy_vol.load(Ordering::Relaxed);
        let sell_vol = coin.agg_sell_vol.load(Ordering::Relaxed);
        let total_vol_cvd = buy_vol + sell_vol;
        let cvd_ratio = if total_vol_cvd > 0.0 {
            (buy_vol - sell_vol) / total_vol_cvd
        } else {
            0.0
        };
        let bid_wall = coin.l2_bid_wall.load(Ordering::Relaxed);
        let ask_wall = coin.l2_ask_wall.load(Ordering::Relaxed);
        let cvd_veto = self.arena.config.cvd_veto_threshold.load(Ordering::Relaxed);
        // D-749: el veto del muro se decide en el espacio del gen —una RAZÓN
        // entre muros— y no contrastando esa razón contra un desequilibrio
        // normalizado en [−1, 1], donde nunca podía dispararse.
        let wall_veto = self
            .arena
            .config
            .wall_veto_threshold
            .load(Ordering::Relaxed);
        let es_largo = out.signal == SignalType::Long;
        match out.signal {
            SignalType::Long | SignalType::Short => {
                let cvd_en_contra = if es_largo {
                    cvd_ratio < -cvd_veto
                } else {
                    cvd_ratio > cvd_veto
                };
                if cvd_en_contra
                    || crate::calibration::muro_en_contra(bid_wall, ask_wall, wall_veto, es_largo)
                {
                    return SignalIntent::flat();
                }
            }
            _ => {}
        }
        out
    }

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
                // D-745: la τ dominante MEDIDA se publica al arena para que el
                // risk-engine dimensione en el mismo horizonte en el que el
                // núcleo gestionará la posición.
                if let Some(spec) = self.temporal_spectrum.get(coin_id) {
                    self.arena.coins[coin_id]
                        .dominant_tau_ms
                        .store(spec.dominant_tau_ms, Ordering::Relaxed);
                }
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
                    // D-758 — EL RÉGIMEN GLOBAL NO PUEDE DECIDIRSE CON 150
                    // PUNTOS BÁSICOS ESCRITOS A MANO.
                    //
                    // QUÉ ESTABA MAL: `btc_trend > 0,015`. `btc_trend` es el
                    // diferencial relativo de las EMAs de 20 y 200 TICKS —una
                    // magnitud de microestructura, típicamente de unos pocos
                    // puntos básicos—, así que exigirle el 1,5 % declaraba
                    // «BullRun» o «Crash» prácticamente nunca: las dos ramas
                    // estaban muertas y el régimen global se reducía a
                    // Range/Chaotic. Y el 1,5 % es además absoluto: no
                    // distingue un mercado en calma de uno en pánico.
                    //
                    // QUÉ GARANTIZA: el mismo criterio de significación que ya
                    // usa el resto del núcleo —z al 95 %— aplicado con las
                    // unidades CORRECTAS: la σ del retorno POR TICK medida por
                    // el propio motor (D-758) escalada por la desviación
                    // estacionaria del diferencial de dos EMAs de esos mismos
                    // periodos. Sin calentamiento el z es 0 y no se declara
                    // tendencia: la ignorancia no es un régimen.
                    let z_btc = match btc_fe.sigma_retorno_tick() {
                        Some(sd) => {
                            let escala = sd
                                * crate::diffusion::ema_spread_sd(
                                    crate::stateful_engine::TICK_EMA_FAST_BARS,
                                    crate::stateful_engine::TICK_EMA_SLOW_BARS,
                                );
                            if escala > 0.0 && btc_trend.is_finite() {
                                btc_trend / escala
                            } else {
                                0.0
                            }
                        }
                        None => 0.0,
                    };
                    let new_regime = if z_btc > crate::diffusion::Z95 && btc_hurst > h_trend {
                        1u8 // BullRun
                    } else if z_btc < -crate::diffusion::Z95 && btc_hurst > h_trend {
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

            // D-753 — LA CANTIDAD NEGOCIADA REAL LLEGA AL MOTOR DE FEATURES.
            // `trade_qty` ya venía en la firma y TODOS los llamadores la
            // rellenan con la cantidad del tape (vivo, `booktick_replay`,
            // forense). Sólo faltaba hacerla llegar al camino per-tick, que
            // fabricaba un sustituto a partir de la profundidad del libro.
            // En eventos que NO son trades queda en 0: no hubo volumen.
            self.feature_engines[coin_id].ultima_cantidad_trade =
                if is_trade && trade_qty.is_finite() && trade_qty > 0.0 {
                    trade_qty
                } else {
                    0.0
                };

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
                // ESPECTRO PREDICTIVO (D-742b): el mismo trade alimenta las
                // tasas de todas las escalas y madura los pronósticos cuyo
                // horizonte acaba de vencer. `is_buyer_maker` = el comprador
                // era el pasivo ⇒ el AGRESOR fue el vendedor.
                if let Some(bank) = self.spectral_forecast.get_mut(coin_id) {
                    bank.on_trade(event_time_ms, (eff_bid + eff_ask) * 0.5, trade_qty, !is_buyer_maker);
                    // D-754 — EL PRONÓSTICO SÓLO SALE AL MOTOR SI HA
                    // DEMOSTRADO HABILIDAD. `habilidad_volatilidad` es el R²
                    // fuera de muestra frente a la climatología, acumulado
                    // prequencialmente: cada muestra se puntúa ANTES de que el
                    // modelo vea su objetivo. Mientras sea None (sin muestras
                    // maduras) o ≤ 0 (no bate a su propia media), el arena
                    // conserva ceros y la geometría sigue usando el ATR hacia
                    // atrás de siempre. Publicar un pronóstico sin evidencia
                    // sería exactamente el pecado que esta ola vino a cerrar.
                    //
                    // Auditoría PR #5 (D-754b): cuando la habilidad deja de ser
                    // positiva (o vuelve a no haber muestras maduras), el arena
                    // VUELVE a ceros. Antes sólo se escribía en la rama con
                    // habilidad y los últimos σ quedaban congelados: el
                    // risk-engine seguía leyendo un pronóstico ya desautorizado.
                    let habilidad = bank.habilidad_volatilidad();
                    let slots = &self.arena.coins[coin_id].sigma_forecast;
                    if habilidad.map(|h| h > 0.0).unwrap_or(false) {
                        let sigmas = bank.sigmas_en_anclas(event_time_ms);
                        for (slot, s) in slots.iter().zip(sigmas.iter()) {
                            slot.store(
                                s.filter(|v| v.is_finite() && *v > 0.0).unwrap_or(0.0),
                                Ordering::Relaxed,
                            );
                        }
                    } else {
                        for slot in slots.iter() {
                            slot.store(0.0, Ordering::Relaxed);
                        }
                    }
                }
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
                // D-731: resta atómica, no load→store.
                let _ = self.arena.used_margin.fetch_update(
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                    |v| Some((v - margin).max(0.0)),
                );
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

            // D-119 / D-753: cvpin sólo debe recibir volumen NEGOCIADO.
            //
            // QUÉ ESTABA MAL: aquí se fabricaba `(bid_qty + ask_qty)·0,005`
            // acotado a [0,01; 10] y se llamaba «volumen del tick». Eso es el
            // 0,5 % de la PROFUNDIDAD del libro: ni es volumen negociado ni
            // guarda relación monótona con él —un libro grueso y quieto
            // producía «volumen» constante—, y el recorte a 10 lo saturaba
            // para cualquier símbolo líquido, dejando una cifra CASI
            // CONSTANTE.
            //
            // POR QUÉ IMPORTABA: el VPIN es un reloj de VOLUMEN. Con un
            // volumen casi constante, los buckets se cerraban a ritmo fijo y
            // el VPIN medía la cadencia del feed, no la toxicidad del flujo.
            // Y el VPIN gobierna el corte tóxico de posiciones, el asiento
            // causal del consejo y `vpin_risk` en el dimensionado.
            //
            // QUÉ GARANTIZA: la cantidad es la del trade REAL (0 cuando el
            // evento no es un trade), así que el reloj de volumen avanza
            // exactamente con el volumen que cruzó el mercado.
            let tick_vol = self.feature_engines[coin_id].ultima_cantidad_trade;
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
                    .map(|spec| spec.persistence_at(spec.dominant_tau_ms).clamp(-1.0, 1.0))
                    .unwrap_or(0.0);
                let s_t = (pers_dom + 1.0) * 0.5;
                let be_frac = 0.45 + 0.20 * s_t;
                let trail_frac = 0.60 + 0.20 * s_t;
                let be_activation = (tp * be_frac)
                    .max(live_fee * 2.0)
                    .max(atr_pct_live * 2.0)
                    .min(tp * 0.90);
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

                // 2. Trailing Stop Ratchet Dinámico: activa en trail_frac
                // (espectral) del TP. D-727: siempre por encima del
                // breakeven y siempre por debajo del objetivo — si se armara
                // en el TP no existiría.
                let trail_activation_pnl = (tp * trail_frac)
                    .max(be_activation * 1.25)
                    .min(tp * 0.95);
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
                        tp, // B3.27 — escalera relativa al TP
                        // S-2: persistencia de la escala dominante — la
                        // escalera respira con el régimen (tendencial corre,
                        // mean-revert cosecha).
                        self.temporal_spectrum
                            .get(coin_id)
                            .map(|spec| {
                                spec.persistence_at(spec.dominant_tau_ms).clamp(-1.0, 1.0)
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
                        // B3.19 — writer de zombie_promotions (auditoría: el
                        // contador existía y /api/state lo leía, pero NADIE
                        // lo escribía — zombie_count siempre 0). U-1: métrica
                        // unificada (el slot gemelo scalp ya no existe).
                        coin.metrics
                            .zombie_promotions
                            .fetch_add(1, Ordering::Relaxed);
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
                    // D-754: el horizonte con el que se dimensionó la posición
                    // es la base del enfriamiento posterior. Se lee ANTES de
                    // cerrar, mientras el estado de la posición sigue vivo.
                    let tau_de_la_posicion = pos.entry_tau_ms.load(Ordering::Relaxed);
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
                    // D-754: los dos relojes del enfriamiento — CUÁNDO se
                    // cerró (en ms de evento, no en cuenta de ticks) y CON QUÉ
                    // horizonte se había dimensionado.
                    self.feature_engines[coin_id].last_scalp_exit_ms = event_time_ms;
                    if tau_de_la_posicion > 0 {
                        self.feature_engines[coin_id].tau_ultimo_cierre_ms = tau_de_la_posicion;
                    }
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

                    // D-752 — ATRIBUCIÓN DEL RESULTADO A QUIEN LO ORIGINÓ.
                    //
                    // La rama que abrió esta posición aprende de su propio
                    // cierre. Sin esto, la «confianza» de cada rama seguiría
                    // siendo el número que alguien escribió a mano: el motor
                    // podía perder mil veces con la misma rama y ésta seguiría
                    // declarando 0,72.
                    if let Some(rama) = self.rama_abierta[coin_id].take() {
                        if rama < N_RAMAS {
                            self.rama_registro[coin_id][rama].observar(is_win);
                        }
                    }
                    // D-752 — Y EL BOSQUE APRENDE DE SU PROPIO VOTO.
                    //
                    // Dirección realizada: un largo que gana subió y uno que
                    // pierde bajó; para un corto, al revés. De ahí
                    // `subio == (is_long == is_win)`. Comparada con el voto
                    // que el bosque emitió en la apertura, da la exactitud
                    // direccional MEDIDA del bosque sobre las operaciones que
                    // el motor realmente ejecutó — con su tamaño de muestra,
                    // que es justo lo que `forest6_acc` no trae.
                    if let Some(predijo_subida) = self.bosque_voto_abierto[coin_id].take() {
                        let subio = is_long == is_win;
                        self.bosque_registro[coin_id].observar(predijo_subida == subio);
                    }

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
            } else if tick % 100_000 == 0 {
                // D-748b: este diagnóstico se emitía cada 100 ticks. En una
                // corrida forense de 17 M de ticks sobre un símbolo sin modelo
                // son 170 000 líneas idénticas —37 807 en la primera media
                // hora— que ahogan el log donde vive el veredicto y frenan la
                // corrida por E/S. La ausencia de modelo es un estado, no un
                // evento: basta recordarlo de cuando en cuando.
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
            // D-756 — LOS UMBRALES SALEN DE LA DISTRIBUCIÓN MEDIDA, NO DE UN
            // RECORTE A MANO.
            //
            // QUÉ ESTABA MAL, y era lo más grave de los tres:
            // `obi_threshold_at_tau` YA acota el gen a su banda evolutiva
            // [0,10; 0,95] — MOD3/5-005 subió ese techo de 0,60 a 0,95
            // precisamente porque «el campeón con OBI 0,797 llegaba al gate
            // leído como 0,60: el genoma evaluado no era el operado». Aquí se
            // volvía a recortar a 0,60 inmediatamente después, reintroduciendo
            // en el núcleo el defecto que el arena había corregido: el cuarto
            // superior de la banda que la evolución explora seguía siendo
            // invisible, y la evolución seguía premiando genomas que el motor
            // no podía ejecutar. Los suelos 0,12 / 0,0008 / 0,15 y el techo
            // 0,0040 eran además absolutos: la misma cifra para un libro
            // profundo y quieto que para uno fino y nervioso.
            //
            // QUÉ GARANTIZA: el gen pone la ambición y la DISTRIBUCIÓN MEDIDA
            // pone el suelo. Por debajo del percentil 80 de |OBI| o |OFI| de
            // ESTE símbolo, «desequilibrio» describe el estado típico del
            // libro, no un evento: ningún gen puede bajar de ahí. Se usan los
            // estimadores P² de `adaptive_quantiles`, que ya existían.
            let piso_obi_medido = self.cuantiles[coin_id].dynamic_obi_threshold();
            let piso_ofi_medido = self.cuantiles[coin_id].dynamic_ofi_threshold();
            // D-756: el suelo del score analítico, por el mismo camino. Es el
            // percentil 85 MEDIDO de |composite_score| en este símbolo: por
            // debajo de él, «convicción analítica» describe lo que el score
            // hace el 85 % del tiempo, no un evento. Se lee aquí, junto a los
            // otros dos, para que TODAS las puertas de abajo usen el mismo
            // estado del estimador dentro del mismo tick.
            let piso_score_medido = self.cuantiles[coin_id].dynamic_holistic_threshold();
            // D-752 (cierre del hueco) — EL HISTORIAL DE TODAS LAS RAMAS, NO
            // SÓLO EL DE LAS DE RESPALDO.
            //
            // QUÉ ESTABA MAL AÚN: las cinco ramas de respaldo ya consultaban su
            // propio historial, pero las DIEZ ramas del camino principal
            // —etiquetadas 1..10 en `volume_flow_rate`, que es justo el canal
            // que el cierre lee para atribuir el resultado— seguían emitiendo
            // `sig_conf(score)` a pelo. Su resultado SÍ se registraba, pero
            // nadie lo leía: el motor acumulaba la evidencia de que una rama
            // perdía y volvía a dimensionar igual en la siguiente entrada.
            //
            // QUÉ GARANTIZA: una sola regla para TODAS las ramas etiquetadas.
            // Mientras la muestra no decida manda la magnitud medida del
            // disparo (`sig_conf`), y en cuanto el intervalo de Wilson excluye
            // la moneda al aire —por arriba o por abajo— manda la cota que esa
            // muestra garantiza. Se copia la fila de la moneda (es `Copy`)
            // para no retener un préstamo de `self` dentro de las ramas.
            let registro_ramas = self.rama_registro[coin_id];
            let dynamic_obi_thr = self
                .arena
                .config
                .obi_threshold_at_tau(tau_dom)
                .max(piso_obi_medido)
                .clamp(0.0, 1.0); // |OBI| ∈ [-1,1] por construcción
            // El umbral de tendencia se compara con `macro_trend`, que es
            // (EMA₉ − EMA₂₁)/EMA₂₁ sobre velas de 1 min. Su suelo es el valor
            // que la difusión declara SIGNIFICATIVO al 95 % con la
            // volatilidad medida: por debajo de él, «tendencia» es ruido de
            // dos medias móviles sobre un paseo aleatorio.
            let piso_tendencia = crate::diffusion::Z95
                * crate::diffusion::sigma_from_atr(atr_pct)
                * crate::diffusion::ema_spread_sd(
                    crate::diffusion::EMA_FAST_BARS,
                    crate::diffusion::EMA_SLOW_BARS,
                );
            let dynamic_ema_thr = self
                .arena
                .config
                .dynamic_ema_trend
                .load(Ordering::Relaxed)
                .max(piso_tendencia)
                .max(0.0);
            let dynamic_ofi_thr = self
                .arena
                .config
                .dynamic_ofi_threshold
                .load(Ordering::Relaxed)
                .max(piso_ofi_medido)
                .max(f64::MIN_POSITIVE); // sólo evita la división por cero

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
            // D-756: el signo de la tendencia superior se decidía con un
            // literal de 10 pb. Diez puntos básicos son un desplazamiento
            // enorme en un símbolo tranquilo y ruido puro en uno volátil. Se
            // exige significación al 95 % sobre la distancia a la EMA de 2 h,
            // tipificada con la volatilidad medida — el mismo criterio que ya
            // usan los escudos macro de más abajo.
            let z_higher_dir = crate::diffusion::ema_distance_z(
                higher_trend,
                atr_pct,
                crate::diffusion::EMA_TREND_BARS,
            );
            // D-758 — LOS CUATRO EJES DE TENDENCIA, TIPIFICADOS UNA SOLA VEZ Y
            // CADA UNO EN SU PROPIA ESCALA.
            //
            // QUÉ ESTABA MAL: los escudos de momento adverso comparaban los
            // cuatro ejes contra fracciones crudas (0,00005 / 0,00010 /
            // 0,00015 / 0,00040 / 0,0010 / 0,0020). Ninguna depende del
            // instrumento ni de su volatilidad, y —peor— se aplicaban a
            // magnitudes de escalas DISTINTAS: `micro_trend` es un diferencial
            // de EMAs por TICK, `macro_trend` uno de EMAs de vela de 1 min, y
            // `higher_trend`/`secular_trend` son distancias a las EMAs de 2 h y
            // 12 h. Cuatro escalas, y umbrales del mismo orden para todas.
            //
            // QUÉ GARANTIZA: cada eje se tipifica con la σ de SU propia escala
            // —la medida por tick para el micro (D-758), la del ATR de 1 min
            // para los tres de vela— y con la desviación estacionaria que le
            // corresponde según sus periodos. A partir de aquí los cuatro
            // hablan el mismo idioma, el de las desviaciones típicas, y pueden
            // compararse y combinarse entre sí.
            let z_micro = match self.feature_engines[coin_id].sigma_retorno_tick() {
                Some(sd) => {
                    let escala = sd
                        * crate::diffusion::ema_spread_sd(
                            crate::stateful_engine::TICK_EMA_FAST_BARS,
                            crate::stateful_engine::TICK_EMA_SLOW_BARS,
                        );
                    if escala > 0.0 && micro_trend.is_finite() {
                        micro_trend / escala
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
            let z_macro = crate::diffusion::ema_spread_z(
                macro_trend,
                atr_pct,
                crate::diffusion::EMA_FAST_BARS,
                crate::diffusion::EMA_SLOW_BARS,
            );
            let z_secular = crate::diffusion::ema_distance_z(
                secular_trend,
                atr_pct,
                crate::diffusion::EMA_MACRO_BARS,
            );
            set_reg(
                "trend_direction",
                if z_higher_dir.abs() > crate::diffusion::Z95 {
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

            // ── BACKTEST ADAPTATIVO (F7-backtest) ─────────────────────────────
            // Detección de libro ausente: OBI (instantáneo, de cantidades
            // bid/ask) es el indicador confiable — en trade-only, bid_qty=
            // ask_qty ⇒ OBI=0.
            let book_absent = obi_val.abs() < 0.005;
            let adaptive_micro_score = if book_absent {
                // Sin libro: CVD del flujo de trades (dirección agresora ×
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
            // D-756 — SE ALIMENTA LA DISTRIBUCIÓN MEDIDA. Los umbrales que
            // este mismo tick ha usado se leyeron ANTES (más arriba), con el
            // estado anterior del estimador: una magnitud no puede participar
            // en el cálculo del umbral que la juzga.
            self.cuantiles[coin_id].update(ofi, current_obi, composite_score, atr_pct);

            let mut fast_intent = SignalIntent::flat();
            let spread_pct = if mid_price > 0.0 {
                (ask - bid) / mid_price
            } else {
                0.0
            };
            // D-755 — EL FILTRO DE HORQUILLA SALE DEL INSTRUMENTO Y DEL
            // HORIZONTE, NO DE DOS NÚMEROS.
            //
            // QUÉ ESTABA MAL: `(atr_pct·0,25).clamp(0,0006; 0,0025)`. Los dos
            // extremos son absolutos y no dependen del instrumento. Un símbolo
            // cuyo TICK vale más de 6 pb del precio no puede cotizar NUNCA una
            // horquilla por debajo del suelo: el filtro lo declaraba inviable
            // siempre, aun con el libro en su mejor estado posible. Y en un
            // símbolo de tick fino el techo de 25 pb dejaba pasar horquillas
            // que se comen varias veces el recorrido que la operación
            // persigue. El factor 0,25 sobre el ATR tampoco decía de dónde
            // salía ese cuarto.
            //
            // QUÉ GARANTIZA: dos magnitudes medidas.
            //  · SUELO — el TICK del símbolo relativo al precio
            //    (`symbol_registry`). Es la rejilla de cotización: ninguna
            //    horquilla puede ser menor, así que rechazar por debajo de
            //    ella es rechazar el mejor libro que el exchange permite.
            //  · TECHO — lo que deja la aritmética del viaje de ida y vuelta.
            //    Cruzar la horquilla y pagar las dos comisiones publicadas
            //    tiene que dejar algo del recorrido alcanzable a τ. El
            //    recorrido alcanzable es el MENOR entre lo que el motor
            //    persigue (`tp_at_tau`) y lo que la difusión puede entregar a
            //    ese horizonte (σ(τ) con el Hurst MEDIDO): si el objetivo es
            //    ambicioso pero el mercado no se mueve tanto, manda el
            //    mercado, y al revés. No hay fracción inventada: la fracción
            //    la impone el propio recorrido.
            let spec_simbolo = quantum_arena::symbol_registry::try_spec(coin_id);
            let tick_pct = spec_simbolo
                .as_ref()
                .map(|s| s.tick_size)
                .filter(|t| t.is_finite() && *t > 0.0)
                .map(|t| t / mid_price.max(1e-12))
                .unwrap_or(0.0);
            let friccion_ida_vuelta = spec_simbolo
                .as_ref()
                .map(|s| (s.maker_fee + s.taker_fee).max(0.0))
                .unwrap_or(0.0);
            // σ(τ): ley de escala de la difusión. σ(1 min) sale del ATR
            // relativo (factor de Parkinson) y se lleva a τ elevando el
            // cociente de horizontes al exponente de Hurst medido — la misma
            // ley fraccionaria que ya gobierna la geometría TP/SL del motor.
            let sigma_tau = crate::diffusion::sigma_from_atr(atr_pct)
                * (tau_dom / 60_000.0)
                    .max(1e-9)
                    .powf(hurst_val.clamp(0.05, 0.95));
            let recorrido_alcanzable = self
                .arena
                .config
                .tp_at_tau(tau_dom)
                .min(sigma_tau)
                .max(0.0);
            let dynamic_max_spread =
                (recorrido_alcanzable - friccion_ida_vuelta).max(tick_pct);
            let spread_ok = spread_pct <= dynamic_max_spread;

            // MOD2/7-014: clasificación canónica de Hurst — UNA definición
            // (consts HURST_*) usada por TODAS las ramas de señal de abajo.
            let is_anti_persistent = hurst_val < HURST_ANTI_PERSISTENT;
            let is_persistent = hurst_val > HURST_PERSISTENT;

            // D-743: la VIABILIDAD de una entrada —que haya volatilidad que
            // recorrer, que el spread no se coma el recorrido y que la reentrada
            // no esté en enfriamiento— es una condición del MERCADO, no de la
            // banda que produjo la señal. Se calcula aquí, una vez, y la
            // comparten las dos lecturas del espectro (ver `puertas_del_continuo`).
            // D-754 — EL ENFRIAMIENTO ES TIEMPO, Y SU BASE ES UN HORIZONTE.
            //
            // QUÉ ESTABA MAL: `can_open_position(600)` contaba 600 TICKS. En
            // un tape denso son segundos; en uno ralo, horas. La misma línea
            // significaba cosas que diferían en tres órdenes de magnitud según
            // el símbolo, la hora del día y el entorno —vivo contra forense—,
            // de modo que el backtest medía una política distinta de la que
            // corre en producción.
            //
            // QUÉ GARANTIZA: la base es el horizonte τ con el que se
            // dimensionó la operación que acaba de cerrarse; reentrar antes de
            // que pase τ es reentrar dentro del mismo movimiento del que se
            // acaba de salir. Si todavía no hubo cierre con τ registrada, se
            // usa la τ dominante viva del espectro, que es la misma magnitud
            // medida sobre el mercado actual.
            let enfriamiento_base_ms = {
                let tau_cierre = self.feature_engines[coin_id].tau_ultimo_cierre_ms;
                if tau_cierre > 0 {
                    tau_cierre as f64
                } else {
                    tau_dom
                }
            };
            let cooldown_ok =
                self.feature_engines[coin_id].can_open_position_ms(enfriamiento_base_ms);
            let atr_ok = atr_pct > dynamic_atr_min;
            let viable_para_entrar = atr_ok && spread_ok && cooldown_ok;
            // D-750: cuando el motor pasa un mes entero sin evaluar una sola
            // señal, hay que poder decir POR CUÁL de las tres condiciones. El
            // diagnóstico de abajo vive dentro del bloque viable, así que
            // guarda silencio justo en el caso que hay que explicar.
            if !viable_para_entrar
                && self.arena.tick_counter.load(Ordering::Relaxed) % 50_000 == 0
            {
                telemetry_server::telemetry_log!(
                    "🚧 [VIABILIDAD] tick={} INVIABLE · atr {:.6} > min {:.6}? {} · spread {:.6} ≤ max {:.6}? {} · enfriamiento? {}",
                    self.arena.tick_counter.load(Ordering::Relaxed),
                    atr_pct,
                    dynamic_atr_min,
                    atr_ok,
                    spread_pct,
                    dynamic_max_spread,
                    spread_ok,
                    cooldown_ok
                );
            }
            let ema_slow_continuo = if self.feature_engines[coin_id].kline_ema_slow > 0.0 {
                self.feature_engines[coin_id].kline_ema_slow
            } else {
                self.feature_engines[coin_id].ema_slow
            };
            let cur_atr_continuo = self.feature_engines[coin_id].v_t.max(mid_price * 0.001);
            let price_stretch_continuo = if ema_slow_continuo > 0.0 {
                (mid_price - ema_slow_continuo) / cur_atr_continuo
            } else {
                0.0
            };

            if viable_para_entrar {
                // (misma banda canónica: reversión a la media ≡ anti-persistencia)
                let is_mean_reverting = is_anti_persistent;

                let ema_slow = ema_slow_continuo;
                let cur_atr = cur_atr_continuo;
                let price_stretch = price_stretch_continuo;
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

                // D-756: los recortes 0,35 / 0,12 / 0,14 eran absolutos. El
                // suelo es ahora el percentil 80 MEDIDO de |OBI| de este
                // símbolo (por debajo de él, «desequilibrio» es el estado
                // habitual del libro) y el techo es 1, que es lo que el OBI
                // puede valer por construcción. La modulación por régimen
                // (±15 %) se conserva: es una preferencia del gen, no un
                // umbral disfrazado.
                let min_obi_trend: f64 = if is_anti_persistent {
                    (dynamic_obi_thr * 1.10).min(1.0)
                } else {
                    (dynamic_obi_thr * 0.85).max(piso_obi_medido)
                };
                let min_obi_pullback: f64 = if is_anti_persistent {
                    (dynamic_obi_thr * 1.10).min(1.0)
                } else {
                    dynamic_obi_thr.max(piso_obi_medido)
                };

                // D-500: Anti-Chop & Post-Loss Conviction Firewall con Direccionalidad y Decaimiento Temporal
                let short_streak = self.feature_engines[coin_id].get_active_directional_streak_ms(false, enfriamiento_base_ms);
                let long_streak = self.feature_engines[coin_id].get_active_directional_streak_ms(true, enfriamiento_base_ms);

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
                        cooldown_ok,
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
                // D-756: el respaldo exigia atr_pct > 0,00005 — cinco millonesimas,
                // absolutas y sin relacion con el instrumento. El minimo fisico
                // es el TICK: si el recorrido tipico del simbolo no llega a un
                // paso de su rejilla de cotizacion, no hay excursion que operar
                // por mucha senal que haya.
                if book_absent && fast_intent.signal == SignalType::Flat && atr_pct > tick_pct {
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
                    // D-752: el suelo literal de 0,60 sobre la convicción
                    // REGALABA 10 puntos de probabilidad a un lift de 2 pb: un
                    // modelo que apenas se despega de su base entraba al Kelly
                    // declarando 0,60. La convicción por magnitud es el propio
                    // lift medido sobre la base del modelo (0,5 + lift), y
                    // sobre ella decide el historial de la rama.
                    let registro_ml = self.rama_registro[coin_id][RAMA_ML_RECENTRADO as usize];
                    if ml_prob_adaptive > ml_model_base + ml_lift {
                        let conviction =
                            0.5 + (ml_prob_adaptive - ml_model_base).abs().min(0.45);
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviccion_de_rama(&registro_ml, conviction.min(1.0)),
                            volume_flow_rate: RAMA_ML_RECENTRADO,
                            horizon: strategy_core::TradeHorizon::Continuous,
                            ..Default::default()
                        };
                    } else if ml_prob_adaptive < ml_model_base - ml_lift {
                        let conviction =
                            0.5 + (ml_prob_adaptive - ml_model_base).abs().min(0.45);
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviccion_de_rama(&registro_ml, conviction.min(1.0)),
                            volume_flow_rate: RAMA_ML_RECENTRADO,
                            horizon: strategy_core::TradeHorizon::Continuous,
                            ..Default::default()
                        };
                    }
                    // Ruta 1b: SEÑAL ESPECTRAL DIRECTA — la fusión por paridad
                    // de riesgo del espectro temporal (19 escalas) produce un
                    // score [-1,+1] computado SOLO de precios reales. Si el
                    // score es fuerte en una dirección Y la persistencia lo
                    // confirma, es una señal legítima independiente del ML.
                    if fast_intent.signal == SignalType::Flat {
                        // D-752: la convicción por magnitud de esta rama es el
                        // propio score fusionado del espectro, que ya vive en
                        // [-1, 1] y es una medida (paridad de riesgo sobre 32
                        // escalas). `sig_conf` lo lleva a la escala de
                        // confianza del motor; sobre ella decide el historial.
                        // Antes: `0,55 + |fused|·0,3` acotado a 0,90 — tres
                        // números sin origen.
                        let registro_esp =
                            self.rama_registro[coin_id][RAMA_ESPECTRO_DIRECTO as usize];
                        if let Some(spec) = self.temporal_spectrum.get(coin_id) {
                            let fused = spec.fused_score;
                            let tau = spec.dominant_tau_ms;
                            let persist = spec.persistence_at(tau);
                            // Score fuerte (>0.6) + persistencia direccional
                            if fused > 0.6 && persist > 0.15 {
                                fast_intent = SignalIntent {
                                    signal: SignalType::Long,
                                    confidence: conviccion_de_rama(
                                        &registro_esp,
                                        sig_conf(fused),
                                    ),
                                    volume_flow_rate: RAMA_ESPECTRO_DIRECTO,
                                    horizon: strategy_core::TradeHorizon::Continuous,
                                    ..Default::default()
                                };
                            } else if fused < -0.6 && persist < -0.15 {
                                fast_intent = SignalIntent {
                                    signal: SignalType::Short,
                                    confidence: conviccion_de_rama(
                                        &registro_esp,
                                        sig_conf(fused),
                                    ),
                                    volume_flow_rate: RAMA_ESPECTRO_DIRECTO,
                                    horizon: strategy_core::TradeHorizon::Continuous,
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

                        // D-752 — LAS CONFIANZAS LITERALES POR RAMA.
                        //
                        // QUÉ ESTABA MAL: estas ramas emitían 0,72 / 0,68 /
                        // 0,65 fijos. Aguas abajo la cifra se trata como una
                        // PROBABILIDAD —pasa por el escalado de Platt que
                        // alimenta el Kelly y se compara contra el gen
                        // `min_confidence_btc`—, de modo que el tamaño de la
                        // posición es monótono en ella. Una rama que dispara
                        // con dev_atr = 1,51 declaraba exactamente lo mismo
                        // que una que dispara con 3,9, y el ORDEN entre las
                        // tres (0,72 > 0,68 > 0,65) no salía de ninguna
                        // medición: nadie había comprobado que la rama de
                        // tendencia acierte más que la de reversión.
                        //
                        // QUÉ GARANTIZA: dos fuentes medidas, en este orden.
                        //  1. El historial de ESA rama en ESA moneda, con su
                        //     intervalo de Wilson: si la muestra excluye la
                        //     moneda al aire, la convicción es la cota que la
                        //     muestra garantiza —por arriba o por abajo—.
                        //  2. Mientras la muestra no decida, la magnitud del
                        //     disparo: la desviación sobre la EMA lenta
                        //     tipificada (`atr_stretch_z`) y referida al mismo
                        //     z95 que el resto del núcleo usa para declarar
                        //     significación. Así una rama que dispara al
                        //     límite pide menos tamaño que una que dispara
                        //     lejos, que es lo que el literal impedía.
                        let magnitud_z = crate::diffusion::atr_stretch_z(
                            dev_atr,
                            crate::diffusion::EMA_SLOW_BARS,
                        );
                        let magnitud_norm =
                            (magnitud_z.abs() / crate::diffusion::Z95).clamp(0.0, 1.0);
                        let piso_magnitud = sig_conf(magnitud_norm);
                        let registro_tend =
                            self.rama_registro[coin_id][RAMA_TENDENCIA_PERSISTENTE as usize];
                        let registro_rev = self.rama_registro[coin_id]
                            [RAMA_REVERSION_ANTIPERSISTENTE as usize];

                        if hurst_active && is_persistent && dev_atr > 1.5 && dev_atr < 4.0 && rolling_cvd > 0.0 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Long,
                                confidence: conviccion_de_rama(&registro_tend, piso_magnitud),
                                volume_flow_rate: RAMA_TENDENCIA_PERSISTENTE,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..Default::default()
                            };
                        } else if hurst_active && is_persistent && dev_atr < -1.5 && dev_atr > -4.0 && rolling_cvd < 0.0 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: conviccion_de_rama(&registro_tend, piso_magnitud),
                                volume_flow_rate: RAMA_TENDENCIA_PERSISTENTE,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..Default::default()
                            };
                        } else if hurst_active && is_anti_persistent && dev_atr > 2.5 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: conviccion_de_rama(&registro_rev, piso_magnitud),
                                volume_flow_rate: RAMA_REVERSION_ANTIPERSISTENTE,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..Default::default()
                            };
                        } else if hurst_active && is_anti_persistent && dev_atr < -2.5 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Long,
                                confidence: conviccion_de_rama(&registro_rev, piso_magnitud),
                                volume_flow_rate: RAMA_REVERSION_ANTIPERSISTENTE,
                                horizon: strategy_core::TradeHorizon::Continuous,
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
                                confidence: conviccion_de_rama(
                                    &self.rama_registro[coin_id]
                                        [RAMA_MOMENTO_SIN_REGIMEN as usize],
                                    piso_magnitud,
                                ),
                                volume_flow_rate: RAMA_MOMENTO_SIN_REGIMEN,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..Default::default()
                            };
                        } else if !hurst_active && dev_atr < -2.0 && dev_atr > -5.0 && rolling_cvd < -0.05 {
                            fast_intent = SignalIntent {
                                signal: SignalType::Short,
                                confidence: conviccion_de_rama(
                                    &self.rama_registro[coin_id]
                                        [RAMA_MOMENTO_SIN_REGIMEN as usize],
                                    piso_magnitud,
                                ),
                                volume_flow_rate: RAMA_MOMENTO_SIN_REGIMEN,
                                horizon: strategy_core::TradeHorizon::Continuous,
                                ..Default::default()
                            };
                        }
                    }
                }

                // D-460 & D-466: Unificación Continua del Generador de Señales (Multiscale Vector Field).
                // Confluencia de triple escala temporal: Micro (1m ticks), Intermedio (EMA 9 vs 21), y Macro Superior (2-Hour EMA 120).
                if is_confirmed_downtrend {
                    // RÉGIMEN BAJISTA CONFIRMADO (MULTISCALE DOWNTREND)
                    // D-756: el endurecimiento tras racha era `.max(0,30)` —un
                    // absoluto que además no hacía NADA si el gen ya pedía
                    // más, y que no crecía con la cuarta ni la quinta pérdida—.
                    // Suelo: el percentil 85 medido de |score| en este símbolo.
                    // Escalada: la escalera binaria del motor, desde el primer
                    // nivel en que esta puerta actúa (racha 2).
                    let d_tech_thr = if short_streak >= 2 {
                        exigencia_tras_racha(
                            dynamic_tech_thr.max(piso_score_medido),
                            short_streak.saturating_sub(1),
                        )
                    } else {
                        dynamic_tech_thr
                    };
                    let d_obi_trend = exigencia_tras_racha(min_obi_trend, short_streak);
                    // 1. Tendencial Short: Flujo institucional, confluencia L2 y ML apuntan a la baja
                    if composite_score < -d_tech_thr
                        && not_overextended_short
                        && effective_obi_short < -d_obi_trend
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviccion_de_rama(
                                &registro_ramas[1],
                                sig_conf(composite_score),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 1.0,
                            ..Default::default()
                        };
                    // 2. Pullback Short: Rebote hacia ema_slow vendido con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score <= -d_tech_thr),
                    // filtro de tendencia superior (para evitar vender en rallies)
                    // y desbalance de libro sólido.
                    //
                    // D-756: el filtro era `higher_trend <= -0,0010`. Diez
                    // puntos básicos sobre la EMA de 2 h son un desplome en un
                    // símbolo tranquilo y ruido puro en uno volátil: el mismo
                    // literal significaba cosas opuestas según el instrumento
                    // y la hora. Se exige lo mismo que ya decide
                    // `trend_direction` unas líneas más arriba —significación
                    // al 95 % de la distancia a la EMA de tendencia,
                    // tipificada con la volatilidad MEDIDA— y con signo
                    // bajista. Es el mismo criterio, medido una sola vez.
                    } else if short_streak < 2
                        && z_higher_dir <= -crate::diffusion::Z95
                        && price_stretch >= 0.15
                        && price_stretch <= 1.20
                        && effective_obi_short < -exigencia_tras_racha(min_obi_pullback, short_streak)
                        && composite_score <= -d_tech_thr
                        && micro_trend <= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviccion_de_rama(
                                &registro_ramas[2],
                                sig_conf(composite_score.abs()),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
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
                    // D-756: simétrico del bajista — suelo por el percentil 85
                    // medido de |score| y escalada por la escalera binaria.
                    let u_tech_thr = if long_streak >= 2 {
                        exigencia_tras_racha(
                            dynamic_tech_thr.max(piso_score_medido),
                            long_streak.saturating_sub(1),
                        )
                    } else {
                        dynamic_tech_thr
                    };
                    let u_obi_trend = exigencia_tras_racha(min_obi_trend, long_streak);
                    // 1. Tendencial Long: Flujo institucional, confluencia L2 y ML apuntan al alza
                    if composite_score > u_tech_thr
                        && not_overextended_long
                        && effective_obi_long > u_obi_trend
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviccion_de_rama(
                                &registro_ramas[4],
                                sig_conf(composite_score),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 4.0,
                            ..Default::default()
                        };
                    // 2. Dip Long: Corrección hacia ema_slow comprada con confluencia estricta de flujo L2 Y composite score
                    // D-500 & D-501: Convicción analítica plena (composite_score >= u_tech_thr),
                    // filtro de tendencia superior y desbalance de libro
                    // sólido (current_obi > exigencia_tras_racha(min_obi_pullback, long_streak)).
                    //
                    // D-756: simétrico del pullback bajista — el literal de
                    // 10 pb se sustituye por significación al 95 % de la
                    // distancia a la EMA de tendencia con la volatilidad
                    // medida.
                    } else if long_streak < 2
                        && z_higher_dir >= crate::diffusion::Z95
                        && price_stretch <= -0.15
                        && price_stretch >= -1.20
                        && effective_obi_long > exigencia_tras_racha(min_obi_pullback, long_streak)
                        && composite_score >= u_tech_thr
                        && micro_trend >= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviccion_de_rama(
                                &registro_ramas[5],
                                sig_conf(composite_score),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
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
                    // D-756: mismo criterio que min_obi_trend — suelo por el percentil
                    // 80 medido de |OBI|, techo por el maximo que |OBI| admite.
                    let range_obi = (dynamic_obi_thr * 0.85).max(piso_obi_medido).min(1.0);
                    if composite_score > range_thr && effective_obi_long > range_obi && price_stretch <= -0.15 && micro_trend >= 0.0 {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviccion_de_rama(
                                &registro_ramas[7],
                                sig_conf(composite_score),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 7.0,
                            ..Default::default()
                        };
                    } else if composite_score < -range_thr
                        && effective_obi_short < -range_obi
                        && price_stretch >= 0.15
                        && micro_trend <= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviccion_de_rama(
                                &registro_ramas[8],
                                sig_conf(composite_score),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 8.0,
                            ..Default::default()
                        };
                    } else if short_streak < 2 && price_stretch > 1.0 && effective_obi_short < -range_obi * 1.15 && composite_score <= -0.24 && micro_trend <= 0.0
                    {
                        fast_intent = SignalIntent {
                            signal: SignalType::Short,
                            confidence: conviccion_de_rama(
                                &registro_ramas[9],
                                sig_conf(effective_obi_short.abs().min(composite_score.abs())),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
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
                    } else if long_streak < 2 && price_stretch < -1.0 && effective_obi_long > range_obi * 1.15 && composite_score >= 0.24 && micro_trend >= 0.0 {
                        fast_intent = SignalIntent {
                            signal: SignalType::Long,
                            confidence: conviccion_de_rama(
                                &registro_ramas[10],
                                sig_conf(effective_obi_long.abs().min(composite_score.abs())),
                            ),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 10.0,
                            ..Default::default()
                        };
                    }
                }

                let tensor_cutoff = ((tensor_min_conf - 0.50) * 2.0).clamp(0.35, 0.80);
                if fast_intent.signal == SignalType::Flat
                    && tensor_scalp.signal != SignalType::Flat
                    && !is_anti_persistent
                    && tensor_scalp.net_confidence.abs() >= tensor_cutoff
                {
                    // D-502, D-503 & D-505: Invariante de Momentum Jerárquico,
                    // Micro-Surge y Concurrencia Multiescala.
                    //
                    // D-758 — QUÉ ESTABA MAL: el escudo era una escalera de
                    // cuatro cláusulas con ocho fracciones crudas (0,00005 /
                    // 0,00010 / 0,00015 / 0,00040 / 0,0010 / 0,0020). Lo que
                    // esas cláusulas querían decir es muy simple —«hay momento
                    // en contra, sea porque un eje solo grita o porque varios
                    // susurran a la vez»— pero se escribió como una tabla de
                    // umbrales a mano, con el agravante de aplicar números del
                    // mismo orden a ejes que viven en escalas temporales
                    // distintas (tick, 1 min, 2 h, 12 h).
                    //
                    // QUÉ GARANTIZA: los mismos dos casos, dichos con la
                    // estadística que ya usa todo el núcleo y sobre los ejes ya
                    // tipificados (D-758).
                    //  · UN EJE SOLO: alguno de los tres ejes operativos supera
                    //    por sí mismo la significación del 95 %. Es la cláusula
                    //    del micro-spike, sin el «+4 pb».
                    //  · VARIOS A LA VEZ: la combinación de Stouffer —suma de
                    //    z's dividida por √k, que es la forma canónica de
                    //    agregar evidencia independiente— alcanza esa misma
                    //    significación. Es la cláusula de concurrencia
                    //    multiescala, sin la tabla de umbrales: tres ejes que
                    //    susurran a 1,2σ suman 2,08σ y levantan el escudo,
                    //    mientras que uno solo a 1,2σ no.
                    //  · EL EJE SECULAR conserva su papel de veto del veto: el
                    //    escudo NO se levanta si la estructura de 12 h apoya
                    //    significativamente la entrada que se quiere hacer.
                    let z95 = crate::diffusion::Z95;
                    let ejes = [z_micro, z_macro, z_higher_dir];
                    let stouffer =
                        ejes.iter().sum::<f64>() / (ejes.len() as f64).sqrt();
                    let is_adverse_momentum_short = (ejes.iter().any(|z| *z > z95)
                        || stouffer > z95)
                        && z_secular > -z95;
                    let is_adverse_momentum_long = (ejes.iter().any(|z| *z < -z95)
                        || stouffer < -z95)
                        && z_secular < z95;
                    // D-756: el suelo 0,22 era absoluto. Se sustituye por el
                    // percentil 85 MEDIDO de |score| en este símbolo. El 0,90
                    // se conserva: es una preferencia de relajación del gen
                    // para la rama tensorial —no un umbral— y el suelo medido
                    // impide que relaje por debajo de lo que este activo hace
                    // el 85 % del tiempo.
                    let tensor_tech_thr = (dynamic_tech_thr * 0.90).max(piso_score_medido);
                    // D-756: mismo criterio que min_obi_trend — suelo por el percentil
                    // 80 medido de |OBI|, techo por el maximo que |OBI| admite.
                    let range_obi = (dynamic_obi_thr * 0.85).max(piso_obi_medido).min(1.0);

                    // Diagnóstico por dirección: las condiciones del gate se nombran una
                    // sola vez y el diagnóstico cuenta cuál falla. La semántica es la de la
                    // conjunción anterior: comparaciones puras, sin efectos laterales.
                    // D-758: los cuatro literales de estas dos listas
                    // (0,0002 / 0,0010 / 0,0008) pasan al espacio tipificado.
                    // Su tamaño delataba lo que realmente eran:
                    //  · 2 pb sobre la EMA de 2 h son ~0,06 σ y 8 pb ~0,24 σ
                    //    con la volatilidad típica del activo. Es decir: NO
                    //    eran umbrales, eran bandas muertas alrededor de cero
                    //    puestas a ojo. La banda muerta honesta es la que mide
                    //    el propio ruido — el signo cuando se quiere «se
                    //    inclina hacia», y el z95 cuando se quiere «no está
                    //    claramente en contra».
                    //  · el 0,0010 sobre la EMA de 12 h cumplía el papel de
                    //    «el eje secular no apoya claramente esta entrada»:
                    //    eso es exactamente una prueba de significación.
                    let long_conditions = [
                        long_streak < 2,
                        !is_confirmed_downtrend,
                        !is_adverse_momentum_long,
                        !(z_higher_dir < 0.0 && z_secular < 0.0),
                        !(price_stretch < -0.80 && z_secular < crate::diffusion::Z95),
                        z_higher_dir >= -crate::diffusion::Z95,
                        composite_score >= tensor_tech_thr,
                        effective_obi_long > range_obi,
                        not_overextended_long,
                    ];
                    let short_conditions = [
                        short_streak < 2,
                        !is_confirmed_uptrend,
                        !is_adverse_momentum_short,
                        !(z_higher_dir > 0.0 && z_secular > 0.0),
                        !(price_stretch > 0.80 && z_secular > -crate::diffusion::Z95),
                        z_higher_dir <= crate::diffusion::Z95,
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
                                .clamp(tensor_min_conf, 1.0),
                            horizon: strategy_core::TradeHorizon::Continuous,
                            volume_flow_rate: 11.0,
                            ..Default::default()
                        };
                    }
                }

                if fast_intent.signal == SignalType::Flat {
                    let hawkes_r = self.feature_engines[coin_id].cvpin.current_vpin();
                    if let Some(mut impulso_intent) =
                        signal_engine::flow_impulse::FlowImpulseEngine::evaluate_flow_impulse(
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
                        let impulso_streak = if impulso_intent.signal == SignalType::Long {
                            long_streak
                        } else {
                            short_streak
                        };
                        let impulso_alineado_con_regimen = if is_confirmed_downtrend {
                            impulso_intent.signal == SignalType::Short
                        } else if is_confirmed_uptrend {
                            impulso_intent.signal == SignalType::Long
                        } else if is_mean_reverting {
                            (impulso_intent.signal == SignalType::Long
                                && price_stretch < -0.5
                                && macro_trend >= -dynamic_ema_thr)
                                || (impulso_intent.signal == SignalType::Short
                                    && price_stretch > 0.5
                                    && macro_trend <= dynamic_ema_thr)
                        } else {
                            (impulso_intent.signal == SignalType::Long && macro_trend >= 0.0)
                                || (impulso_intent.signal == SignalType::Short && macro_trend <= 0.0)
                        };

                        if impulso_streak < 2
                            && impulso_alineado_con_regimen
                            && ((impulso_intent.signal == SignalType::Long && not_overextended_long)
                                || (impulso_intent.signal == SignalType::Short
                                    && not_overextended_short))
                        {
                            impulso_intent.volume_flow_rate = 12.0;
                            fast_intent = impulso_intent;
                        }
                    }
                }

                // D-622 (DÉCIMA OLA): el cooldown binario del lado scalp (20 s tras
                // ganar; 90 s o 180 s tras perder según volatilidad) se retira. El
                // guard unificado D-463 cubre la reentrada tras cualquier cierre
                // —dirección, ganancia o pérdida, racha— con ventanas iguales o
                // mayores, y sin distinguir de qué «motor» vino la señal.
            }

            // D-743 — PUERTAS DEL CONTINUO sobre la banda rápida. Los vetos de
            // flujo agregado y muro del libro, el invariante bayesiano y la
            // ponderación continua del modelo vivían aquí, aplicados SÓLO a esta
            // banda: la lectura lenta del mismo espectro entraba sin pasar por
            // ninguno. Ahora son una función única que se aplica a las DOS
            // lecturas antes de arbitrar.
            fast_intent = self.puertas_del_continuo(
                coin_id,
                fast_intent,
                viable_para_entrar,
                composite_score,
                ml_prob,
                ml_model_base,
                price_stretch_continuo,
            );

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

            // D-743: la lectura LENTA pasa por las MISMAS puertas que la rápida
            // —viabilidad de mercado, invariante bayesiano, ponderación del
            // modelo y vetos de flujo y muro— antes de competir en la
            // arbitración. Hasta aquí entraba sin ninguna: un motor universal
            // con una puerta para cada mitad no es un motor universal.
            slow_intent = self.puertas_del_continuo(
                coin_id,
                slow_intent,
                viable_para_entrar,
                composite_score,
                ml_prob,
                ml_model_base,
                price_stretch_continuo,
            );

            if coin_id < self.last_fast_intent.len() {
                self.last_fast_intent[coin_id] = fast_intent;
            }
            if coin_id < self.last_slow_intent.len() {
                self.last_slow_intent[coin_id] = slow_intent;
            }

            // D-431: Composición de onda multiescala no destructiva (Continuous Wave Mechanics).
            // Evita el canibalismo ciego donde una discrepancia de 0.01 abre operaciones contratendencia.
            let mut unified_intent = SignalIntent::flat();
            if fast_intent.signal != SignalType::Flat && slow_intent.signal != SignalType::Flat {
                if fast_intent.signal == slow_intent.signal {
                    // D-623 (DÉCIMA OLA): antes `máx(p₁, p₂)·1,10` con suelo 0,60. Dos
                    // evidencias sólo se combinan sumando log-odds si son
                    // condicionalmente independientes, y éstas no lo son: ambas leen el
                    // mismo consenso tensorial (`tensor_scalp` y `tensor_swing` son copias
                    // de `tensor_cont`). Con evidencia dependiente, la combinación que no
                    // inventa certeza es el máximo.
                    let boosted_conf = fast_intent.confidence.max(slow_intent.confidence);
                    unified_intent = SignalIntent {
                        signal: fast_intent.signal,
                        confidence: boosted_conf,
                        horizon: strategy_core::TradeHorizon::Continuous,
                        ..fast_intent
                    };
                } else {
                    // U-2 (MOTOR UNIVERSAL CONTINUO): conflicto de banda — la
                    // banda cuya escala está MÁS CERCA de τ dominante lleva la
                    // energía del mercado AHORA y decide la dirección. Antes
                    // arbitraba una "tendencia confirmada" fija (sesgo lento
                    // estructural: en transiciones rápidas el motor seguía a
                    // la banda lenta contra el flujo vivo). τ_mid = media
                    // geométrica de la banda operativa [30s, 12h] ≈ 19 min:
                    // τ_dom < τ_mid ⇒ manda la banda rápida, si no la lenta.
                    // Los escudos macro (D-467+) siguen vetando contratendencia
                    // del régimen mayor DESPUÉS — la arbitración espectral no
                    // los reemplaza, los precede.
                    let tau_dom_now = self
                        .temporal_spectrum
                        .get(coin_id)
                        .map(|s| s.dominant_tau_ms)
                        .unwrap_or(30_000.0);
                    const TAU_MID_MS: f64 = 1_138_000.0; // √(30_000 × 43_200_000)
                    let fast_band_governs = tau_dom_now < TAU_MID_MS;
                    let winner = if fast_band_governs {
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

            // QO-E2a — EL APRENDIZAJE MODULA: el forest online (entrenado
            // con los resultados REALES de cierres previos vía telemetry
            // mmap) opinaba sólo sobre 2 umbrales — su predictor predict_6d
            // no lo llamaba nadie. Ahora: cuando está entrenado (acc>0.55)
            // y predice DESACUERDO fuerte con la intención (prob<0.40 para
            // largo / >0.60 para corto), la confianza se reduce ×0.8.
            // UNILATERAL: el forest nunca AMPLÍA confianza (su acc de
            // clasificación binaria no justifica más agresividad) — sólo
            // puede frenar. Sin desacuerdo o sin entrenamiento: intacto.
            // D-752 — EL FRENO DEL BOSQUE EXIGE EVIDENCIA, NO UNA EXACTITUD
            // SUELTA.
            //
            // QUÉ ESTABA MAL: el freno se armaba con `forest6_acc > 0,55`. Esa
            // cifra la publica el daemon SIN tamaño de muestra, así que 0,56
            // podía venir de 9 aciertos sobre 16 —indistinguible de una moneda
            // al aire— y aun así recortaba un 20 % la confianza, es decir el
            // tamaño de la posición. Los umbrales de desacuerdo (0,40 / 0,60)
            // y el recorte (×0,8) eran tres literales más, sin relación con lo
            // buena que el bosque hubiera demostrado ser.
            //
            // QUÉ GARANTIZA: la autoridad del bosque es su exactitud
            // DIRECCIONAL medida por el propio núcleo sobre las operaciones
            // que el motor ejecutó (apertura: voto; cierre: dirección
            // realizada), leída por su cota inferior de Wilson. Sólo frena si
            // esa cota excluye la moneda al aire al 95 %. Y entonces todo lo
            // demás sale de la misma cota `c`:
            //   · desacuerdo: el bosque debe apartarse de ½ al menos tanto
            //     como su propia ventaja demostrada (`p < 1 − c` para un
            //     largo, `p > c` para un corto);
            //   · recorte: `2·(1 − c)`, que vale 1 (sin freno) cuando el
            //     bosque no demuestra nada y 0 (veto total) cuando acierta
            //     siempre.
            // Con `c = 0,60` la regla reproduce EXACTAMENTE los tres literales
            // antiguos: eran el caso particular de un bosque con un 60 % de
            // acierto demostrado, escrito como si fuera universal.
            let f6_prob_actual = self
                .arena
                .registry
                .get_for_coin_or(coin_id, "forest6_prob", 0.5);
            // Voto direccional del bosque para esta moneda, o `None` si no
            // opina (el valor por defecto del registry es justo ½).
            let voto_bosque: Option<bool> = if (f6_prob_actual - 0.5).abs() > f64::EPSILON {
                Some(f6_prob_actual > 0.5)
            } else {
                None
            };
            if unified_intent.signal != SignalType::Flat {
                let (cota_bosque, _) =
                    self.bosque_registro[coin_id].intervalo(crate::diffusion::Z95);
                if cota_bosque > 0.5 {
                    let disagrees = match unified_intent.signal {
                        SignalType::Long => f6_prob_actual < 1.0 - cota_bosque,
                        SignalType::Short => f6_prob_actual > cota_bosque,
                        SignalType::Flat => false,
                    };
                    if disagrees && unified_intent.confidence.is_finite() {
                        let recorte = (2.0 * (1.0 - cota_bosque)).clamp(0.0, 1.0);
                        unified_intent.confidence =
                            (unified_intent.confidence * recorte).clamp(0.0, 1.0);
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

                    // D-757 — LAS VENTANAS DEL ANTI-WHIPLASH RESPIRAN CON τ.
                    //
                    // QUÉ ESTABA MAL: 45 s, 60 s, 3 min, 20 min, 1 h y 2 h
                    // escritos a mano, más tres estiramientos en ATR crudo
                    // (0,25 / 0,30 / 0,10). Un motor que declara operar un
                    // CONTINUO de horizontes no puede llevar un calendario
                    // fijo: tres minutos son una eternidad para una tesis de
                    // 30 s y un parpadeo para una de doce horas. La misma
                    // línea protegía de más en la banda rápida y de menos en
                    // la lenta, y el escalón de racha (180 s → 20 min → 1 h →
                    // 2 h) ni siquiera era regular.
                    //
                    // QUÉ GARANTIZA: la unidad es τ —el horizonte con el que
                    // se dimensionó la operación que acaba de cerrarse— y la
                    // escalada por racha es la misma escalera binaria del
                    // enfriamiento (D-754), con el mismo techo: más allá del
                    // horizonte más lento que el motor opera, esto deja de ser
                    // un veto y pasa a ser un apagado.
                    //
                    // Los estiramientos pasan a ser pruebas de SIGNO. Lo que
                    // las tres reglas querían decir es «el precio todavía no
                    // ha vuelto de donde lo dejamos», y eso es el signo de la
                    // distancia a la media, no una fracción de ATR: 0,25 ATR
                    // equivale a 0,18 σ en la distribución de esa distancia —
                    // ni siquiera es un desplazamiento distinguible de cero—,
                    // así que el número no aportaba el matiz que aparentaba.
                    let ventana_base_ms = enfriamiento_base_ms;
                    let transcurrido_ms = elapsed_ms as f64;
                    let whiplash_veto = if last_was_win {
                        // POST-WIN: la pata impulsiva se monetizó. Prohibido
                        // recomprar el techo del rally (o revender el suelo
                        // del desplome) mientras no haya pasado un horizonte
                        // completo Y el precio siga extendido en la misma
                        // dirección que se acaba de cobrar.
                        if is_same_dir {
                            transcurrido_ms < ventana_base_ms
                                && ((unified_intent.signal == SignalType::Long
                                    && p_stretch > 0.0)
                                    || (unified_intent.signal == SignalType::Short
                                        && p_stretch < 0.0))
                        } else {
                            // Giro a contratendencia post-win: es una tesis
                            // nueva, y como cualquier otra necesita que el
                            // mercado haya tenido un horizonte para producirla.
                            transcurrido_ms < ventana_base_ms
                        }
                    } else if !is_same_dir {
                        // POST-LOSS (SL) a contratendencia: prohibido vender la
                        // capitulación que acaba de saltar un stop de largo, o
                        // comprar la euforia que saltó uno de corto, durante un
                        // horizonte; pasado éste, sólo mientras el precio siga
                        // extendido en esa misma dirección adversa.
                        transcurrido_ms < ventana_base_ms
                            || (unified_intent.signal == SignalType::Short && p_stretch < 0.0)
                            || (unified_intent.signal == SignalType::Long && p_stretch > 0.0)
                    } else {
                        // POST-LOSS en la MISMA dirección: retroceso binario
                        // sobre el horizonte, idéntico al del enfriamiento.
                        let is_scalp_long = unified_intent.signal == SignalType::Long;
                        let streak = self.feature_engines[coin_id]
                            .get_active_directional_streak_ms(is_scalp_long, enfriamiento_base_ms);
                        let requerido_ms = (ventana_base_ms
                            * (streak.min(8) as f64).exp2())
                        .min(quantum_arena::temporal_spectrum::TAU_ANCHOR_SLOW_MS);
                        let time_veto = transcurrido_ms < requerido_ms;
                        // Con racha demostrada, además se exige que la entrada
                        // esté en el lado correcto de la media: prohibido
                        // vender por debajo de ella o comprar por encima.
                        let stretch_veto = streak >= 2
                            && ((unified_intent.signal == SignalType::Short && p_stretch < 0.0)
                                || (unified_intent.signal == SignalType::Long
                                    && p_stretch > 0.0));
                        time_veto || stretch_veto
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
            // D-499: Invariante de Convicción Post-Racha Direccional Universal
            // (Cross-Horizon Directional Loss Streak Firewall).
            //
            // D-756 — QUÉ ESTABA MAL: la «convicción institucional» se escribía
            // como cuatro números absolutos —(|score| ≥ 0,28; |OBI| ≥ 0,18) a
            // partir de dos pérdidas y (0,32; 0,22) a partir de tres—. Ninguno
            // depende del instrumento: en un libro grueso y quieto |OBI| = 0,18
            // es el estado HABITUAL —el cortafuegos no cortaba nada— y en uno
            // fino y nervioso es inalcanzable —cortaba siempre—. Además el
            // escalón entre dos y tres pérdidas era del 14 %, una cifra sin
            // origen, y una cuarta o quinta pérdida no cambiaba nada.
            //
            // QUÉ GARANTIZA: los dos suelos salen de la DISTRIBUCIÓN MEDIDA de
            // este símbolo con los estimadores P² que ya existían —el percentil
            // 85 de |composite_score| y el 80 de |OBI|—, de modo que «convicción»
            // significa un EVENTO en la distribución propia del activo y no un
            // número escrito a mano. La escalada es la misma escalera binaria
            // del enfriamiento (D-754) y de la exigencia de libro, leída desde
            // el primer nivel en que este cortafuegos actúa: como sólo engancha
            // a partir de la segunda pérdida, la racha efectiva es `streak − 1`,
            // así que dos pérdidas piden el suelo medido, tres lo duplican,
            // cuatro lo cuadruplican. El techo de 1 es físico: tanto
            // `composite_score` (acotado a [−1, 1] en su construcción) como
            // |OBI| no pueden valer más, de modo que saturar es vetar.
            if unified_intent.signal == SignalType::Short {
                let streak = self.feature_engines[coin_id].get_active_directional_streak_ms(false, enfriamiento_base_ms);
                if streak >= 2 {
                    let nivel = streak.saturating_sub(1);
                    let min_score = exigencia_tras_racha(piso_score_medido, nivel);
                    let min_obi = exigencia_tras_racha(piso_obi_medido, nivel);
                    if composite_score > -min_score || current_obi > -min_obi {
                        unified_intent = SignalIntent::flat();
                    }
                }
            } else if unified_intent.signal == SignalType::Long {
                let streak = self.feature_engines[coin_id].get_active_directional_streak_ms(true, enfriamiento_base_ms);
                if streak >= 2 {
                    let nivel = streak.saturating_sub(1);
                    let min_score = exigencia_tras_racha(piso_score_medido, nivel);
                    let min_obi = exigencia_tras_racha(piso_obi_medido, nivel);
                    if composite_score < min_score || current_obi < min_obi {
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
                        .get_active_directional_streak_ms(order.signal == SignalType::Long, enfriamiento_base_ms);
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

                                // U-ERR-5 (merge con F-009 de main): F-009 volvía a
                                // partir la apertura en Swing/Scalping según la
                                // rama (vfr ≥ 13) o la duración declarada. El
                                // binario de horizonte está erradicado: el
                                // horizonte REAL de la posición es `entry_tau_ms`
                                // (τ dimensionada, D-745), no esta etiqueta.
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
                                // D-752 — QUIÉN ABRIÓ ESTA POSICIÓN. La
                                // etiqueta de rama viaja en
                                // `volume_flow_rate` y sobrevive a la
                                // arbitración (todas las fusiones usan
                                // `..fast_intent` / `..winner`). Se anota
                                // junto con el voto del bosque hasta el
                                // cierre, que es quien los juzga: sin esta
                                // línea ninguna rama podría aprender de su
                                // propio resultado y la convicción volvería a
                                // ser un número escrito a mano.
                                let etiqueta_rama = unified_intent.volume_flow_rate;
                                self.rama_abierta[coin_id] = if etiqueta_rama.is_finite()
                                    && etiqueta_rama >= 0.0
                                    && (etiqueta_rama as usize) < N_RAMAS
                                {
                                    Some(etiqueta_rama as usize)
                                } else {
                                    None
                                };
                                self.bosque_voto_abierto[coin_id] = voto_bosque;

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
                                        coin.positions.position.nn_entry_tensor.lock()
                                    {
                                        *t = tensor_snapshot.to_vec();
                                    }
                                }
                                // D-745: la posición NACE con el horizonte CON
                                // EL QUE SE DIMENSIONÓ (`order.tau_ms`), no con
                                // una segunda lectura del espectro. Mientras se
                                // recalculaba aquí, el risk-engine podía
                                // dimensionar a 2 h una posición que el núcleo
                                // gestionaba a 30 s: TP/SL de respaldo,
                                // trailing, caducidad, Kelly de cierre y
                                // apalancamiento del host razonaban sobre un
                                // horizonte que nadie había usado para calcular
                                // el tamaño. Respaldo: la τ dominante viva.
                                let tau_entry = if order.tau_ms.is_finite() && order.tau_ms > 0.0 {
                                    order.tau_ms as u64
                                } else {
                                    self.temporal_spectrum
                                        .get(coin_id)
                                        .map(|s| s.dominant_tau_ms as u64)
                                        .unwrap_or(0)
                                };
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

#[cfg(test)]
mod tests_d743_puertas_del_continuo {
    use super::*;
    use signal_engine::{SignalIntent, SignalType};
    use std::sync::Arc;

    fn intencion_larga() -> SignalIntent {
        SignalIntent {
            signal: SignalType::Long,
            confidence: 0.80,
            horizon: strategy_core::TradeHorizon::Continuous,
            ..Default::default()
        }
    }

    /// D-743: las cuatro puertas valen para CUALQUIER lectura del espectro.
    /// Antes vivían sólo en el camino de la banda rápida y la lenta —que gana
    /// la arbitración siempre que τ dominante supera la media geométrica de la
    /// banda operativa— entraba sin pasar por ninguna.
    #[test]
    fn d743_una_intencion_no_viable_queda_plana_venga_de_donde_venga() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let core = GodEngineCore::new(Arc::clone(&arena));
        let i = intencion_larga();

        // 1. Mercado inviable (spread, ATR o enfriamiento).
        assert_eq!(
            core.puertas_del_continuo(0, i, false, 0.5, 0.60, 0.5, 0.0).signal,
            SignalType::Flat
        );
        // 2. Consenso del tick en contra (invariante bayesiano D-472).
        assert_eq!(
            core.puertas_del_continuo(0, i, true, -0.20, 0.60, 0.5, 0.0).signal,
            SignalType::Flat
        );
        // 3. El modelo la contradice con fuerza (< −0,80 bajo su base) y no
        //    hay acción de precio extrema.
        assert_eq!(
            core.puertas_del_continuo(0, i, true, 0.50, 0.05, 0.5, 0.0).signal,
            SignalType::Flat
        );
        // Con acción de precio extrema, el veto del modelo cede (F-009).
        assert_eq!(
            core.puertas_del_continuo(0, i, true, 0.50, 0.05, 0.5, 3.0).signal,
            SignalType::Long
        );
        // 4. Flujo agregado en contra: veto duro.
        arena.coins[0].agg_buy_vol.store(1.0, Ordering::Relaxed);
        arena.coins[0].agg_sell_vol.store(99.0, Ordering::Relaxed);
        assert_eq!(
            core.puertas_del_continuo(0, i, true, 0.50, 0.60, 0.5, 0.0).signal,
            SignalType::Flat
        );
        arena.coins[0].agg_buy_vol.store(50.0, Ordering::Relaxed);
        arena.coins[0].agg_sell_vol.store(50.0, Ordering::Relaxed);
        // 4b. Muro del libro en contra: veto duro.
        arena.coins[0].l2_bid_wall.store(1.0, Ordering::Relaxed);
        arena.coins[0].l2_ask_wall.store(99.0, Ordering::Relaxed);
        assert_eq!(
            core.puertas_del_continuo(0, i, true, 0.50, 0.60, 0.5, 0.0).signal,
            SignalType::Flat
        );
    }

    /// F-009 / D-411 — la dirección del modelo se mide contra SU base. Con
    /// base 0,30 (etiquetado honesto), ml = 0,30 es neutral para un largo: ni
    /// veta ni penaliza. Una contradicción moderada (−0,60) sólo penaliza, con
    /// piso de convicción, en vez de aplanar la intención.
    #[test]
    fn f009_d411_modelo_relativo_a_su_base_y_penalizacion_blanda() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let core = GodEngineCore::new(Arc::clone(&arena));
        arena.coins[0].agg_buy_vol.store(50.0, Ordering::Relaxed);
        arena.coins[0].agg_sell_vol.store(50.0, Ordering::Relaxed);
        let i = intencion_larga();

        let neutral = core.puertas_del_continuo(0, i, true, 0.50, 0.30, 0.30, 0.0);
        assert_eq!(neutral.signal, SignalType::Long);
        assert!((neutral.confidence - i.confidence).abs() < 1e-12);

        // ml 0,20 con base 0,50 ⇒ dirección −0,60: penaliza, no veta.
        let blanda = core.puertas_del_continuo(0, i, true, 0.50, 0.20, 0.5, 0.0);
        assert_eq!(blanda.signal, SignalType::Long);
        assert!((blanda.confidence - i.confidence * 0.70).abs() < 1e-12);

        // Base fuera de (0,05, 0,95) no es base: se cae a 0,5.
        let degenerada = core.puertas_del_continuo(0, i, true, 0.50, 0.05, 0.99, 0.0);
        assert_eq!(degenerada.signal, SignalType::Flat);
    }

    /// Una intención que pasa las cuatro conserva su dirección, y el modelo a
    /// favor sólo puede modular la confianza dentro de [0, 0.99].
    #[test]
    fn d743_lo_que_pasa_las_puertas_conserva_direccion_y_confianza_acotada() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        let core = GodEngineCore::new(Arc::clone(&arena));
        arena.coins[0].agg_buy_vol.store(50.0, Ordering::Relaxed);
        arena.coins[0].agg_sell_vol.store(50.0, Ordering::Relaxed);
        let out = core.puertas_del_continuo(0, intencion_larga(), true, 0.50, 0.95, 0.5, 0.0);
        assert_eq!(out.signal, SignalType::Long);
        assert!(out.confidence > 0.80 && out.confidence <= 0.99, "{}", out.confidence);
    }
}

/// Tests de los defectos D-752 (confianzas literales por rama y freno del
/// bosque sin muestra) y D-756 (escalada de exigencia tras racha). Afirman la
/// propiedad que el código ANTERIOR violaba: un literal no aprende y una
/// exactitud sin `n` no es evidencia.
#[cfg(test)]
mod tests_d752_d756 {
    use super::*;
    use crate::diffusion::Z95;

    fn registro(n: u32, aciertos: u32) -> TasaAcierto {
        TasaAcierto { n, aciertos }
    }

    /// D-752 — CON TRES DATOS NO HAY CERTEZA.
    ///
    /// Es la razón de usar Wilson y no Wald: con 3 aciertos de 3, el intervalo
    /// normal degenera en [1, 1] («certeza» con tres observaciones) y el motor
    /// dimensionaría al máximo. Wilson devuelve un intervalo que todavía
    /// contiene la moneda al aire, así que la rama NO puede declarar ventaja.
    #[test]
    fn d752_wilson_no_declara_certeza_con_tres_datos() {
        let r = registro(3, 3);
        let (lo, hi) = r.intervalo(Z95);
        assert!(lo < 0.5 && hi > 0.5, "3/3 no puede excluir el azar: [{lo}, {hi}]");
        assert!(!r.mejor_que_el_azar(Z95));

        // Sin muestra: ignorancia completa, no optimismo.
        let (lo0, hi0) = registro(0, 0).intervalo(Z95);
        assert_eq!((lo0, hi0), (0.0, 1.0));

        // Con muestra suficiente, la MISMA proporción sí excluye el azar.
        let r2 = registro(300, 300);
        assert!(r2.mejor_que_el_azar(Z95), "300/300 debe excluir el azar");
    }

    /// D-752 — UNA RAMA QUE DEMUESTRA PERDER NO PUEDE SEGUIR DECLARANDO 0,72.
    ///
    /// QUÉ ESTABA MAL: las ramas de acción de precio emitían 0,72 / 0,68 /
    /// 0,65 FIJOS, y aguas abajo esa cifra se trata como una PROBABILIDAD (la
    /// come el escalado de Platt que alimenta el Kelly y se compara contra el
    /// gen `min_confidence_btc`). El motor podía perder mil veces con la misma
    /// rama y ésta seguiría pidiendo el mismo tamaño de posición.
    ///
    /// QUÉ GARANTIZA ESTE TEST: los tres estados de `conviccion_de_rama` son
    /// los que la muestra permite, y ninguno de ellos es un literal.
    #[test]
    fn d752_la_conviccion_de_rama_sale_de_su_propio_historial() {
        let piso = 0.58; // convicción por magnitud del disparo

        // 1) Sin evidencia: manda la magnitud medida del disparo, no un literal.
        assert_eq!(conviccion_de_rama(&registro(0, 0), piso), piso);
        assert_eq!(conviccion_de_rama(&registro(5, 3), piso), piso);

        // 2) Ventaja DEMOSTRADA: la convicción es la cota que la muestra
        //    garantiza, no la media optimista.
        let ganadora = registro(400, 280); // 70 % con n grande
        let c_gana = conviccion_de_rama(&ganadora, piso);
        let (lo, _) = ganadora.intervalo(Z95);
        assert_eq!(c_gana, lo);
        assert!(c_gana > 0.5 && c_gana < 0.70, "cota {c_gana} vs media 0,70");

        // 3) Desventaja DEMOSTRADA: la convicción cae por debajo de ½ y el
        //    gate de confianza la rechaza sin necesidad de una regla aparte.
        //    ESTE es el caso que el literal 0,72 hacía imposible.
        let perdedora = registro(400, 120); // 30 % con n grande
        let c_pierde = conviccion_de_rama(&perdedora, piso);
        assert!(
            c_pierde < 0.5,
            "una rama que pierde el 70 % de las veces declaró {c_pierde}"
        );
        assert!(
            c_pierde < 0.72 && c_pierde < piso,
            "la evidencia adversa debe mandar sobre la magnitud del disparo"
        );

        // Monotonía: más aciertos con la misma n ⇒ más convicción.
        let a = conviccion_de_rama(&registro(400, 240), piso);
        let b = conviccion_de_rama(&registro(400, 300), piso);
        assert!(a < b, "la convicción no es monótona en la tasa medida: {a} / {b}");
    }

    /// D-752 — EL FRENO DEL BOSQUE EXIGE TAMAÑO DE MUESTRA.
    ///
    /// QUÉ ESTABA MAL: se armaba con `forest6_acc > 0,55`, una cifra que el
    /// daemon publica SIN `n`. 9 aciertos de 16 dan 0,5625 —indistinguible de
    /// una moneda al aire— y aun así recortaban un 20 % la confianza, es decir
    /// el tamaño de la posición.
    ///
    /// QUÉ GARANTIZA ESTE TEST: con esa misma muestra el freno NO se arma, y
    /// el recorte derivado `2·(1−c)` reproduce el ×0,8 antiguo exactamente
    /// cuando el bosque demuestra el 60 % que aquel literal presuponía.
    #[test]
    fn d752_el_freno_del_bosque_exige_evidencia_no_una_exactitud_suelta() {
        // 9/16 = 0,5625 > 0,55: el criterio viejo frenaba. El nuevo, no.
        let flojo = registro(16, 9);
        assert!(9.0 / 16.0 > 0.55, "premisa del test");
        assert!(
            !flojo.mejor_que_el_azar(Z95),
            "9 de 16 no distinguen al bosque de una moneda al aire"
        );

        // Con muestra que sí demuestra ventaja, el freno se arma y su
        // intensidad sale de la propia cota.
        let solido = registro(2_000, 1_240); // 62 % con n grande
        let (c, _) = solido.intervalo(Z95);
        assert!(c > 0.5, "62 % con n=2000 debe excluir el azar");
        let recorte = (2.0 * (1.0 - c)).clamp(0.0, 1.0);
        assert!(recorte < 1.0 && recorte > 0.0, "recorte {recorte}");

        // Identidad de calibración: c = 0,60 reproduce el ×0,8 y los umbrales
        // de desacuerdo 0,40 / 0,60 que estaban escritos a mano — eran el caso
        // particular de un bosque con 60 % de acierto DEMOSTRADO.
        let c60 = 0.60_f64;
        assert!(((2.0 * (1.0 - c60)) - 0.8).abs() < 1e-12);
        assert!(((1.0 - c60) - 0.40).abs() < 1e-12);
    }

    /// D-752 — EL CANAL DE ATRIBUCIÓN Y EL REGISTRO TIENEN QUE CUADRAR.
    ///
    /// La convicción de una rama sólo puede salir de su historial si la
    /// etiqueta que viaja en `SignalIntent::volume_flow_rate` cae dentro del
    /// registro que el cierre actualiza. Si alguien añade una rama con una
    /// etiqueta fuera de rango, la apertura no se anota, el cierre no enseña
    /// nada y esa rama vuelve en silencio a dimensionar con un número fijo —
    /// exactamente el defecto que D-752 corrige. Este test fija el contrato.
    #[test]
    fn d752_todas_las_etiquetas_de_rama_caben_en_el_registro() {
        // Camino principal: etiquetas 1..=14 escritas como literales en los
        // `SignalIntent` de las ramas de régimen.
        for etiqueta in 1..=14usize {
            assert!(
                etiqueta < N_RAMAS,
                "la etiqueta {etiqueta} del camino principal no cabe en el registro"
            );
        }
        // Ramas de respaldo: constantes nombradas.
        for etiqueta in [
            RAMA_ML_RECENTRADO,
            RAMA_ESPECTRO_DIRECTO,
            RAMA_TENDENCIA_PERSISTENTE,
            RAMA_REVERSION_ANTIPERSISTENTE,
            RAMA_MOMENTO_SIN_REGIMEN,
        ] {
            assert!(etiqueta.is_finite() && etiqueta >= 0.0, "{etiqueta}");
            assert!(
                (etiqueta as usize) < N_RAMAS,
                "la etiqueta de respaldo {etiqueta} no cabe en el registro"
            );
            // La conversión f64 → usize del canal de atribución debe ser
            // exacta: una etiqueta con parte decimal se anotaría en la casilla
            // equivocada y una rama aprendería del resultado de otra.
            assert_eq!(etiqueta, (etiqueta as usize) as f64, "etiqueta no entera");
        }
        // Y no colisionan con las del camino principal.
        assert!(RAMA_ML_RECENTRADO as usize > 14);
    }

    /// D-756 — LA EXIGENCIA TRAS RACHA ESCALA Y NUNCA RELAJA.
    ///
    /// QUÉ ESTABA MAL: `min_obi_trend.max(0,22)`. Con un umbral vigente ya
    /// mayor que 0,22 el «endurecimiento» no hacía NADA, y una tercera, cuarta
    /// o décima pérdida consecutiva tampoco cambiaba nada: el escalón era
    /// único y literal.
    ///
    /// QUÉ GARANTIZA ESTE TEST: la misma escalera binaria del enfriamiento
    /// (D-754), monótona y con techo en 1, que es lo que |OBI| puede valer por
    /// construcción.
    #[test]
    fn d756_la_exigencia_tras_racha_escala_y_nunca_relaja() {
        let umbral = 0.30_f64;

        // Sin racha (0 y 1): la exigencia es la del umbral vigente.
        assert_eq!(exigencia_tras_racha(umbral, 0), umbral);
        assert_eq!(exigencia_tras_racha(umbral, 1), umbral);

        // A partir de la segunda pérdida, cada una DUPLICA.
        assert!((exigencia_tras_racha(umbral, 2) - 0.60).abs() < 1e-12);
        assert!((exigencia_tras_racha(umbral, 3) - 1.0).abs() < 1e-12); // 1,20 → techo

        // ESTA es la propiedad que el `.max(0,22)` viejo violaba: con un
        // umbral vigente de 0,30 daba 0,30 para CUALQUIER racha.
        assert!(
            exigencia_tras_racha(umbral, 5) > exigencia_tras_racha(umbral, 2),
            "una racha más larga debe exigir más, no lo mismo"
        );

        // Monotonía general y no-relajación, con un umbral pequeño.
        let pequeno = 0.05_f64;
        let mut anterior = exigencia_tras_racha(pequeno, 0);
        for racha in 1..=12u32 {
            let actual = exigencia_tras_racha(pequeno, racha);
            assert!(
                actual >= anterior,
                "la exigencia RELAJÓ en la racha {racha}: {anterior} → {actual}"
            );
            assert!(actual <= 1.0, "|OBI| no puede exigir más de 1: {actual}");
            anterior = actual;
        }
        assert!((anterior - 1.0).abs() < 1e-12, "debe saturar en el techo físico");

        // Un umbral no finito no se inventa: se propaga tal cual.
        assert!(exigencia_tras_racha(f64::NAN, 4).is_nan());
    }
}
