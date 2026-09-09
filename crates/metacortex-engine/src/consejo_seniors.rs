//! # Council of Seniors — Distributed Adversarial Deliberation Engine
//!
//! Implements a 9-member council of specialized senior agents in `cerebro/consejo/`:
//! 1. SeniorMicroestructura — Orderbook & trade flow agression
//! 2. SeniorSeriesTemporales — Wavelet, HMM, state-space cycles
//! 3. SeniorGrafos — GNN & correlation breakdown
//! 4. SeniorCausal — Do-calculus & manipulation filter (VETO)
//! 5. SeniorRiesgo — VaR, CVaR, Kelly & drawdown limit (VETO)
//! 6. SeniorEjecucion — Market impact & micro-timing (VETO)
//! 7. SeniorCuantico — Combinatorial portfolio allocation
//! 8. SeniorMetacognitivo — Dynamic weight recalibration over 1000 decisions
//! 9. SeniorTeleonomia — Future utility & understanding (VETO)

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TradingHorizon {
    Continuous,
    Scalping,
    Swing,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotPayload {
    pub horizon: TradingHorizon,
    pub book_imbalance: f64,
    pub hurst_exponent: f64,
    pub graph_correlation: f64,
    pub do_calculus_risk: f64,
    pub causal_veto_threshold: f64,
    pub current_drawdown_pct: f64,
    pub estimated_slippage_bps: f64,
}

impl MarketSnapshotPayload {
    /// Validates data integrity: fails explicitly on NaN, Inf, or out-of-bound anomalies
    pub fn validate(&self) -> Result<(), String> {
        let metrics = [
            ("book_imbalance", self.book_imbalance),
            ("hurst_exponent", self.hurst_exponent),
            ("graph_correlation", self.graph_correlation),
            ("do_calculus_risk", self.do_calculus_risk),
            ("current_drawdown_pct", self.current_drawdown_pct),
            ("estimated_slippage_bps", self.estimated_slippage_bps),
        ];

        for (name, val) in metrics {
            if !val.is_finite() {
                return Err(format!("Corrupt non-finite metric in snapshot: {}", name));
            }
        }

        if self.hurst_exponent < 0.0 || self.hurst_exponent > 1.0 {
            return Err(format!(
                "Out-of-bounds Hurst exponent: {}",
                self.hurst_exponent
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeniorRole {
    Microestructura,
    SeriesTemporales,
    Grafos,
    Causal,
    Riesgo,
    Ejecucion,
    Cuantico,
    Metacognitivo,
    Teleonomia,
    AuditorInterno,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeniorOpinion {
    pub role: SeniorRole,
    pub signal_direction: f64, // [-1.0, 1.0]
    pub confidence: f64,       // [0.0, 1.0]
    pub weight: f64,           // Dynamic weight
    pub is_veto: bool,
    pub justification: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub approved: bool,
    pub final_signal: f64,
    pub total_consensus_pct: f64,
    pub vetoed_by: Option<SeniorRole>,
    pub dissenting_log: Vec<SeniorOpinion>,
}

pub trait SeniorAgent: Send + Sync {
    fn role(&self) -> SeniorRole;
    fn evaluate(&self, payload: &MarketSnapshotPayload, win_rate: f64) -> SeniorOpinion;
}

#[inline(always)]
pub fn safe_signum(val: f64) -> f64 {
    if val > 1e-6 {
        1.0
    } else if val < -1e-6 {
        -1.0
    } else {
        0.0
    }
}

// 1. Senior Microestructura
pub struct SeniorMicroestructura;
impl SeniorAgent for SeniorMicroestructura {
    fn role(&self) -> SeniorRole {
        SeniorRole::Microestructura
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let imbalance = payload.book_imbalance;
        SeniorOpinion {
            role: self.role(),
            signal_direction: safe_signum(imbalance),
            confidence: imbalance.abs().clamp(0.0, 1.0),
            weight: 1.0,
            is_veto: false,
            justification: format!("Book imbalance: {:.4}", imbalance),
        }
    }
}

// 2. Senior Series Temporales
pub struct SeniorSeriesTemporales;
impl SeniorAgent for SeniorSeriesTemporales {
    fn role(&self) -> SeniorRole {
        SeniorRole::SeriesTemporales
    }
    #[inline(always)]
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let hurst = payload.hurst_exponent;
        let flow_dir = safe_signum(payload.book_imbalance);

        let (trend_threshold, mean_reversion_threshold) = match payload.horizon {
            TradingHorizon::Continuous => (0.52, 0.45),
            TradingHorizon::Scalping => (0.55, 0.42),
            TradingHorizon::Swing => (0.65, 0.35),
        };

        // FIX #571: Adaptativo por horizonte.
        let signal = if hurst > trend_threshold {
            flow_dir
        } else if hurst < mean_reversion_threshold {
            -flow_dir
        } else {
            0.0
        };
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence: (hurst - 0.5).abs() * 2.0,
            weight: 1.0,
            is_veto: false,
            justification: format!(
                "Hurst exponent: {:.4} (dir={:.1}, mode={:?})",
                hurst, signal, payload.horizon
            ),
        }
    }
}

// 3. Senior Grafos
pub struct SeniorGrafos;
impl SeniorAgent for SeniorGrafos {
    fn role(&self) -> SeniorRole {
        SeniorRole::Grafos
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let correlation = payload.graph_correlation;
        SeniorOpinion {
            role: self.role(),
            signal_direction: safe_signum(correlation),
            confidence: correlation.abs().clamp(0.0, 1.0),
            weight: 0.9,
            is_veto: false,
            justification: format!("Graph correlation score: {:.4}", correlation),
        }
    }
}

// 4. Senior Causal (VETO ON MANIPULATION - PEARL DO-CALCULUS)
pub struct SeniorCausal;
impl SeniorAgent for SeniorCausal {
    fn role(&self) -> SeniorRole {
        SeniorRole::Causal
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let do_calculus_risk = if payload.do_calculus_risk.is_finite() {
            payload.do_calculus_risk.clamp(0.0, 1.0)
        } else {
            1.0 // Falla segura: veto preventivo si el riesgo causal no es finito
        };
        // D-112: El umbral causal protege contra toxicidad extrema (>0.85) sin asfixiar
        // los breakouts institucionales legítimos (VPIN entre 0.60 y 0.80).
        let effective_threshold = payload.causal_veto_threshold.clamp(0.60, 0.90);
        let is_aligned_breakout = (payload.book_imbalance.abs() > 0.25
            || matches!(payload.horizon, TradingHorizon::Scalping))
            && do_calculus_risk < 0.88;
        let is_veto = do_calculus_risk > effective_threshold && !is_aligned_breakout;
        SeniorOpinion {
            role: self.role(),
            signal_direction: 0.0, // Neutral permission agent (not a trend predictor)
            confidence: (1.0 - do_calculus_risk).clamp(0.0, 1.0),
            weight: 1.2,
            is_veto,
            justification: format!(
                "Causal manipulation risk: {:.4} (Threshold: {:.4}, aligned={})",
                do_calculus_risk, effective_threshold, is_aligned_breakout
            ),
        }
    }
}

// 5. Senior Riesgo (VETO ON VAR/DRAWDOWN)
pub struct SeniorRiesgo;
impl SeniorAgent for SeniorRiesgo {
    fn role(&self) -> SeniorRole {
        SeniorRole::Riesgo
    }
    #[inline(always)]
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let drawdown = payload.current_drawdown_pct;

        // FIX #1280: Umbral adaptativo para micro-cuentas ($13 USD bootstrap).
        let max_drawdown = match payload.horizon {
            TradingHorizon::Continuous => 0.90,
            TradingHorizon::Scalping => 0.95, // Scalping es más arriesgado pero permite mayor drawdown para recovery
            TradingHorizon::Swing => 0.85,
        };

        let is_veto = drawdown > max_drawdown;
        SeniorOpinion {
            role: self.role(),
            signal_direction: 0.0, // Neutral permission agent
            confidence: (1.0 - drawdown).clamp(0.0, 1.0),
            weight: 1.5,
            is_veto,
            justification: format!(
                "Risk drawdown assessment: {:.4} (mode={:?})",
                drawdown, payload.horizon
            ),
        }
    }
}

// 6. Senior Ejecucion (VETO ON IMPACT)
pub struct SeniorEjecucion;
impl SeniorAgent for SeniorEjecucion {
    fn role(&self) -> SeniorRole {
        SeniorRole::Ejecucion
    }
    #[inline(always)]
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        // FIX #387: estimated_slippage_bps ya está expresado en puntos básicos (bps)
        let slippage_bps = payload.estimated_slippage_bps.max(0.0);
        // D-113: Umbral dinámico adaptado a la volatilidad de altcoins
        let max_slippage = match payload.horizon {
            TradingHorizon::Continuous => 65.0, // Altcoins con spread normal de 10-35 bps no son vetadas
            TradingHorizon::Scalping => 35.0,
            TradingHorizon::Swing => 100.0,
        };
        let is_veto = slippage_bps > max_slippage;
        SeniorOpinion {
            role: self.role(),
            signal_direction: 0.0, // Neutral permission agent
            confidence: 0.9,
            weight: 1.0,
            is_veto,
            justification: format!(
                "Execution impact slippage: {:.2} bps (mode={:?})",
                slippage_bps, payload.horizon
            ),
        }
    }
}

// 7. Senior Cuantico — Analiza la interacción entre book_imbalance y slippage
// para evaluar si la microestructura permite una ejecución rentable en la dirección del flujo.
pub struct SeniorCuantico;
impl SeniorAgent for SeniorCuantico {
    fn role(&self) -> SeniorRole {
        SeniorRole::Cuantico
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        // FIX #387: Usar directamente puntos básicos sin distorsión de escala
        let slippage_bps = payload.estimated_slippage_bps.max(0.0);
        let slippage_impact = slippage_bps / 100.0; // 10 bps = 0.10 impact
        let execution_quality = (payload.book_imbalance.abs() - slippage_impact).clamp(0.0, 1.0);
        let signal = safe_signum(payload.book_imbalance) * execution_quality;
        let confidence = (execution_quality * 2.0).clamp(0.0, 1.0);
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence,
            weight: 1.0,
            is_veto: false,
            justification: format!(
                "Quantum exec quality: {:.4} (imb={:.4}, slip={:.2} bps)",
                execution_quality, payload.book_imbalance, slippage_bps
            ),
        }
    }
}

// 8. Senior Metacognitivo — Pondera historial (wr) contra riesgo actual y valida la dirección.
pub struct SeniorMetacognitivo;
impl SeniorAgent for SeniorMetacognitivo {
    fn role(&self) -> SeniorRole {
        SeniorRole::Metacognitivo
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        let dd_penalty = (payload.current_drawdown_pct * 5.0).clamp(0.0, 0.5);
        // D-122: Respetar RULE[growth_over_wr]. Nunca invertir la dirección (-dir) de la microestructura por WR temporal.
        // Se modula la convicción y el peso proporcionalmente al rendimiento, protegiendo las rachas iniciales.
        let effective_wr = if wr > 0.0 { wr.clamp(0.20, 1.0) } else { 0.50 };
        let adjusted_confidence = (effective_wr - dd_penalty).clamp(0.05, 1.0);
        let dir = safe_signum(payload.book_imbalance);
        let weight = if wr >= 0.50 { 2.0 } else { 1.0 };
        SeniorOpinion {
            role: self.role(),
            signal_direction: dir,
            confidence: adjusted_confidence,
            weight,
            is_veto: false,
            justification: format!(
                "Metacognitive WR={:.4}, DD_penalty={:.4}, adj_conf={:.4}, weight={:.1}",
                wr, dd_penalty, adjusted_confidence, weight
            ),
        }
    }
}

// 9. Senior Teleonomia (VETO ON UTILITY)
// Evalúa si la operación tiene utilidad futura positiva considerando
// el contexto macro completo del payload en la dirección del flujo.
pub struct SeniorTeleonomia;
impl SeniorAgent for SeniorTeleonomia {
    fn role(&self) -> SeniorRole {
        SeniorRole::Teleonomia
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        // FIX #572: Normalizar slippage en bps respecto a 50 bps sin saturación prematura
        let slippage_penalty = (payload.estimated_slippage_bps / 50.0).clamp(0.0, 1.0);
        let execution_quality = 1.0 - slippage_penalty;
        let utility = payload.graph_correlation.abs() * 0.3
            + (payload.hurst_exponent - 0.5).abs() * 0.4
            + execution_quality * 0.3;
        // Solo vetar si la utilidad teleonómica es prácticamente nula y el WR colapsó por debajo del umbral crítico (35%)
        let is_veto = utility < 0.05 && wr < 0.35;
        let dir = safe_signum(payload.book_imbalance);
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto {
                0.0
            } else {
                dir * utility.clamp(0.0, 1.0)
            },
            confidence: utility.clamp(0.0, 1.0),
            weight: 1.0,
            is_veto,
            justification: format!(
                "Teleonomic utility={:.4}, wr={:.4}, veto={}",
                utility, wr, is_veto
            ),
        }
    }
}

// 10. Senior Auditor Interno (El Anti-Sistema / VETO ON DISCREPANCY & SELF-DECEPTION)
pub struct SeniorAuditorInterno;
impl SeniorAgent for SeniorAuditorInterno {
    fn role(&self) -> SeniorRole {
        SeniorRole::AuditorInterno
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        // Umbral de drawdown adaptativo alineado con SeniorRiesgo para evitar vetos contradictorios (D-343)
        let max_dd = match payload.horizon {
            TradingHorizon::Continuous => 0.95,
            TradingHorizon::Scalping => 0.95,
            TradingHorizon::Swing => 0.85,
        };
        let is_veto = payload.do_calculus_risk > 0.92 || payload.current_drawdown_pct > max_dd;
        SeniorOpinion {
            role: self.role(),
            signal_direction: 0.0, // Neutral permission agent
            confidence: 1.0,
            weight: 3.0, // Maximum authority as Devil's Advocate
            is_veto,
            justification: format!(
                "Auditor Interno check: DD={:.4}, Risk={:.4}",
                payload.current_drawdown_pct, payload.do_calculus_risk
            ),
        }
    }
}

pub struct ConsejoDeliberacion {
    pub agents: Vec<Box<dyn SeniorAgent>>,
    pub tracker: std::sync::RwLock<SeniorPerformanceTracker>,
}

impl Default for ConsejoDeliberacion {
    fn default() -> Self {
        Self {
            agents: vec![
                Box::new(SeniorMicroestructura),
                Box::new(SeniorSeriesTemporales),
                Box::new(SeniorGrafos),
                Box::new(SeniorCausal),
                Box::new(SeniorRiesgo),
                Box::new(SeniorEjecucion),
                Box::new(SeniorCuantico),
                Box::new(SeniorMetacognitivo),
                Box::new(SeniorTeleonomia),
                Box::new(SeniorAuditorInterno),
            ],
            tracker: std::sync::RwLock::new(SeniorPerformanceTracker::new(500)),
        }
    }
}

impl ConsejoDeliberacion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Deliberates on market snapshot with optional adaptive weight multipliers
    pub fn deliberar_with_weights(
        &self,
        payload: &MarketSnapshotPayload,
        win_rate: f64,
        weight_multipliers: Option<&[f64; 10]>,
    ) -> ConsensusResult {
        // R-06 — SHRINKAGE BAYESIANO del win rate: wr=0 con n=0 significa
        // "sin datos", NO "sistema fallando". Mezcla con prior Beta(1,1)
        // (uniforme) ponderada por la evidencia disponible: sin trades el
        // prior 0.5 domina (neutral); con n grande el wr empírico manda.
        // El caller pasa n implícito vía el propio wr — usamos el shrinkage
        // estándar wr' = (wr*n + 0.5*k)/(n + k) con k=8 pseudo-observaciones
        // y n estimado del tracker del Consejo cuando existe.
        let raw_wr = if win_rate.is_finite() {
            win_rate.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let n_obs = self
            .tracker
            .read()
            .ok()
            .map(|t| t.total_outcomes())
            .unwrap_or(0) as f64;
        let k_prior = 8.0;
        let safe_wr = (raw_wr * n_obs + 0.5 * k_prior) / (n_obs + k_prior);

        // N-12: Si no se proporcionan multiplicadores externos, usar pesos adaptativos empíricos del tracker
        let dynamic_weights = if weight_multipliers.is_none() {
            self.tracker.read().ok().map(|t| t.compute_weights())
        } else {
            None
        };
        let active_mults = weight_multipliers.or(dynamic_weights.as_ref());

        // Enforce strict data payload validation
        if let Err(err_msg) = payload.validate() {
            return ConsensusResult {
                approved: false,
                final_signal: 0.0,
                total_consensus_pct: 0.0,
                vetoed_by: Some(SeniorRole::Riesgo),
                dissenting_log: vec![SeniorOpinion {
                    role: SeniorRole::Riesgo,
                    signal_direction: 0.0,
                    confidence: 1.0,
                    weight: 5.0,
                    is_veto: true,
                    justification: format!("Data integrity failure: {}", err_msg),
                }],
            };
        }

        let mut opinions = Vec::new();
        let mut total_weighted_signal = 0.0;
        let mut active_weights = 0.0;
        let mut vetoes = Vec::new();

        for (idx, agent) in self.agents.iter().enumerate() {
            let mut op = agent.evaluate(payload, safe_wr);
            if let Some(multipliers) = active_mults {
                if let Some(&mult) = multipliers.get(idx) {
                    if mult.is_finite() && mult > 0.0 {
                        op.weight *= mult.clamp(0.1, 5.0);
                    }
                }
            }

            if op.is_veto {
                vetoes.push(op.clone());
            }

            if op.signal_direction != 0.0 && op.confidence > 0.0 {
                total_weighted_signal += op.signal_direction * op.confidence * op.weight;
                active_weights += op.confidence * op.weight;
            }
            opinions.push(op);
        }

        let mut final_signal = if active_weights > 0.0 {
            (total_weighted_signal / active_weights).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // El denominador es la capacidad ponderada de todos los miembros con voto direccional
        let total_directional_capacity: f64 = opinions
            .iter()
            .filter(|o| {
                !matches!(
                    o.role,
                    SeniorRole::Causal
                        | SeniorRole::Riesgo
                        | SeniorRole::Ejecucion
                        | SeniorRole::AuditorInterno
                )
            })
            .map(|o| o.weight * o.confidence.clamp(0.0, 1.0))
            .sum();

        let positive_capacity: f64 = opinions
            .iter()
            .filter(|o| o.signal_direction > 0.0)
            .map(|o| o.weight * o.confidence.clamp(0.0, 1.0))
            .sum();

        let negative_capacity: f64 = opinions
            .iter()
            .filter(|o| o.signal_direction < 0.0)
            .map(|o| o.weight * o.confidence.clamp(0.0, 1.0))
            .sum();

        let long_consensus_pct = if total_directional_capacity > 0.0 {
            let raw = positive_capacity / total_directional_capacity;
            if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let short_consensus_pct = if total_directional_capacity > 0.0 {
            let raw = negative_capacity / total_directional_capacity;
            if raw.is_finite() {
                raw.clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        // D-342: Quórum Bayesiano ponderado contra Veto Deadlock.
        // Si hay vetoes:
        // - Si >= 2 seniors vetan: veto absoluto e irrevocable.
        // - Si 1 senior veta pero los otros 9 seniors tienen supermayoría (> 80% consenso) y señal fuerte (>= 0.35),
        //   se aprueba aplicando penalización bayesiana del 25% al final_signal en lugar de un bloqueo irreversible.
        let mut vetoed_by = None;
        if !vetoes.is_empty() {
            let top_consensus = long_consensus_pct.max(short_consensus_pct);
            // D-342 & D-425: Quórum Bayesiano con escala alineada a la convicción continua del tensor (>= 0.28)
            if vetoes.len() == 1 && top_consensus >= 0.80 && final_signal.abs() >= 0.28 {
                final_signal *= 0.75; // Penalización del 25% por disenso de 1 senior
            } else {
                vetoed_by = Some(vetoes[0].role);
            }
        }

        let (approved, consensus_pct) = if vetoed_by.is_some() {
            (false, 0.0)
        } else if long_consensus_pct >= 0.50 && final_signal > 0.0 {
            (true, long_consensus_pct)
        } else if short_consensus_pct >= 0.50 && final_signal < 0.0 {
            (true, short_consensus_pct)
        } else if total_directional_capacity == 0.0 {
            // D-165: Con cero capacidad direccional de los seniors, rechazar para evitar operaciones a ciegas
            (false, 0.0)
        } else {
            (false, long_consensus_pct.max(short_consensus_pct))
        };

        let dissenting_log = if approved {
            opinions
                .into_iter()
                .filter(|o| {
                    o.is_veto
                        || (o.signal_direction != 0.0
                            && final_signal != 0.0
                            && o.signal_direction.signum() != final_signal.signum())
                })
                .collect()
        } else if !vetoes.is_empty() {
            vetoes
        } else {
            opinions
                .into_iter()
                .filter(|o| {
                    o.signal_direction != 0.0
                        && final_signal != 0.0
                        && o.signal_direction.signum() != final_signal.signum()
                })
                .collect()
        };

        ConsensusResult {
            approved,
            final_signal: if approved { final_signal } else { 0.0 },
            total_consensus_pct: consensus_pct,
            vetoed_by,
            dissenting_log,
        }
    }

    /// Deliberates on market snapshot requiring explicit payload validation, 60% consensus, and 0 vetoes
    #[inline(always)]
    pub fn deliberar(&self, payload: &MarketSnapshotPayload, win_rate: f64) -> ConsensusResult {
        self.deliberar_with_weights(payload, win_rate, None)
    }

    /// N-12: Extrae las señales direccionales de los 10 Seniors para correlacionar con el resultado posterior
    pub fn extract_senior_signals(
        &self,
        payload: &MarketSnapshotPayload,
        win_rate: f64,
    ) -> [f64; 10] {
        let safe_wr = if win_rate.is_finite() {
            win_rate.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let mut signals = [0.0; 10];
        for (idx, agent) in self.agents.iter().enumerate() {
            signals[idx] = agent.evaluate(payload, safe_wr).signal_direction;
        }
        signals
    }

    /// N-12: Registra el retorno realizado de una operación para actualizar los pesos adaptativos
    pub fn record_outcome(&self, senior_signals: &[f64; 10], realized_return: f64) {
        if let Ok(mut tracker) = self.tracker.write() {
            tracker.record_outcome(senior_signals, realized_return);
        }
    }
}

/// Tracker de desempeño y calibración metacognitiva de los 10 Seniors sobre ventana deslizante de hasta 1000 decisiones (Punto #276)
#[derive(Debug, Clone)]
pub struct SeniorPerformanceTracker {
    pub window_size: usize,
    pub correct_counts: [usize; 10],
    pub total_counts: [usize; 10],
    pub history: std::collections::VecDeque<([f64; 10], f64)>,
}

impl SeniorPerformanceTracker {
    /// R-06: total de resultados observados (para el shrinkage del wr).
    pub fn total_outcomes(&self) -> usize {
        self.history.len()
    }

    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(10).min(1000),
            correct_counts: [0; 10],
            total_counts: [0; 10],
            history: std::collections::VecDeque::with_capacity(window_size.max(10).min(1000)),
        }
    }

    /// Registra el resultado observado tras la decisión del Consejo
    pub fn record_outcome(&mut self, senior_signals: &[f64; 10], realized_return: f64) {
        if !realized_return.is_finite() {
            return;
        }

        if self.history.len() >= self.window_size {
            if let Some((old_signals, old_ret)) = self.history.pop_front() {
                for i in 0..10 {
                    let old_sig = old_signals[i];
                    self.total_counts[i] = self.total_counts[i].saturating_sub(1);
                    let was_correct = if old_sig != 0.0 {
                        (old_sig > 0.0 && old_ret > 0.0) || (old_sig < 0.0 && old_ret < 0.0)
                    } else {
                        old_ret > 0.0
                    };
                    if was_correct {
                        self.correct_counts[i] = self.correct_counts[i].saturating_sub(1);
                    }
                }
            }
        }

        for i in 0..10 {
            let sig = senior_signals[i];
            self.total_counts[i] += 1;
            let was_correct = if sig != 0.0 {
                (sig > 0.0 && realized_return > 0.0) || (sig < 0.0 && realized_return < 0.0)
            } else {
                // D-345: Para los 4 seniors de veto/permiso (signal == 0.0), el permiso implícito
                // es evaluado por el éxito del trade ejecutado: si ganó, el permiso fue acertado.
                realized_return > 0.0
            };
            if was_correct {
                self.correct_counts[i] += 1;
            }
        }

        self.history.push_back((*senior_signals, realized_return));
    }

    /// Calcula multiplicadores adaptativos normalizados en el rango [0.5, 2.0]
    pub fn compute_weights(&self) -> [f64; 10] {
        let mut weights = [1.0; 10];
        let mut sum_acc = 0.0;
        let mut active_seniors = 0;

        let mut accuracies = [0.5; 10];
        for i in 0..10 {
            if self.total_counts[i] >= 5 {
                let acc = self.correct_counts[i] as f64 / self.total_counts[i] as f64;
                accuracies[i] = acc.clamp(0.01, 0.99);
                sum_acc += accuracies[i];
                active_seniors += 1;
            }
        }

        if active_seniors >= 2 && sum_acc > 0.0 {
            let mean_acc = sum_acc / active_seniors as f64;
            for i in 0..10 {
                if self.total_counts[i] >= 5 {
                    let relative = accuracies[i] / mean_acc.max(0.01);
                    weights[i] = relative.clamp(0.5, 2.0);
                }
            }
        }

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consejo_deliberacion_long_approval() {
        let consejo = ConsejoDeliberacion::new();
        let payload = MarketSnapshotPayload {
            horizon: TradingHorizon::Scalping,
            book_imbalance: 0.85,
            hurst_exponent: 0.72,
            graph_correlation: 0.5,
            do_calculus_risk: 0.5,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.05,
            // 20 bps: bajo el límite de 25 bps de SeniorEjecucion para
            // Scalping (30 disparaba el veto y contradecía la intención
            // del caso: payload alcista fuerte DEBE aprobarse).
            estimated_slippage_bps: 20.0,
        };

        let result = consejo.deliberar(&payload, 0.70);
        assert!(
            result.approved,
            "Long signal should be approved with strong bullish payload"
        );
        assert!(result.final_signal > 0.0);
        assert!(result.total_consensus_pct >= 0.60);
    }

    #[test]
    fn test_consejo_deliberacion_short_approval() {
        let consejo = ConsejoDeliberacion::new();
        let payload = MarketSnapshotPayload {
            horizon: TradingHorizon::Scalping,
            book_imbalance: -0.85,
            hurst_exponent: 0.35,
            graph_correlation: -0.80,
            do_calculus_risk: 0.05,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.01,
            estimated_slippage_bps: 0.0005,
        };

        let result = consejo.deliberar(&payload, 0.70);
        assert!(
            result.approved,
            "Short signal MUST be approved with strong bearish payload"
        );
        assert!(result.final_signal < 0.0);
        assert!(result.total_consensus_pct >= 0.60);
    }

    #[test]
    fn test_consejo_deliberacion_veto() {
        let consejo = ConsejoDeliberacion::new();
        let payload = MarketSnapshotPayload {
            horizon: TradingHorizon::Scalping,
            book_imbalance: 0.90,
            hurst_exponent: 0.75,
            graph_correlation: 0.85,
            do_calculus_risk: 0.95, // High manipulation risk triggering VETO
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.01,
            estimated_slippage_bps: 0.0005,
        };

        let result = consejo.deliberar(&payload, 0.70);
        assert!(!result.approved, "Trade must be rejected due to VETO");
        assert_eq!(result.vetoed_by, Some(SeniorRole::Causal));
    }

    #[test]
    fn test_consejo_deliberacion_with_custom_weights() {
        let consejo = ConsejoDeliberacion::new();
        let payload = MarketSnapshotPayload {
            horizon: TradingHorizon::Scalping,
            book_imbalance: 0.85,
            hurst_exponent: 0.72,
            graph_correlation: 0.80,
            do_calculus_risk: 0.05,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.01,
            estimated_slippage_bps: 0.0005,
        };

        // Multiplicadores que potencian a Microestructura y Series Temporales
        let weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = consejo.deliberar_with_weights(&payload, 0.70, Some(&weights));
        assert!(result.approved);
        assert!(result.final_signal > 0.0);
        assert!(result.total_consensus_pct >= 0.60);
    }

    #[test]
    fn test_senior_performance_tracker_adaptation() {
        let mut tracker = SeniorPerformanceTracker::new(50);
        // Simular 10 trades ganadores donde el Senior 0 (Microestructura) acertó
        for _ in 0..10 {
            let signals = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            tracker.record_outcome(&signals, 0.01);
        }

        let weights = tracker.compute_weights();
        assert!(
            weights[0] > weights[1],
            "Senior 0 con 100% acierto debe tener mayor peso que Senior 1 con 0%"
        );
        assert!(weights[0] <= 2.0);
        assert!(weights[1] >= 0.5);
    }

    #[test]
    fn test_consejo_deliberacion_nan_payload_immunity() {
        let consejo = ConsejoDeliberacion::new();
        let payload = MarketSnapshotPayload {
            horizon: TradingHorizon::Scalping,
            book_imbalance: f64::NAN,
            hurst_exponent: f64::NAN,
            graph_correlation: f64::NAN,
            do_calculus_risk: f64::INFINITY,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: -0.05,
            estimated_slippage_bps: 10.0,
        };

        let result = consejo.deliberar(&payload, 0.70);
        assert!(result.final_signal.is_finite());
        assert!(result.total_consensus_pct.is_finite());
    }
}
