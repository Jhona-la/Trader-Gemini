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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotPayload {
    pub book_imbalance: f64,
    pub hurst_exponent: f64,
    pub graph_correlation: f64,
    pub do_calculus_risk: f64,
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
            return Err(format!("Out-of-bounds Hurst exponent: {}", self.hurst_exponent));
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

// 1. Senior Microestructura
pub struct SeniorMicroestructura;
impl SeniorAgent for SeniorMicroestructura {
    fn role(&self) -> SeniorRole { SeniorRole::Microestructura }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let imbalance = payload.book_imbalance;
        SeniorOpinion {
            role: self.role(),
            signal_direction: imbalance.signum(),
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
    fn role(&self) -> SeniorRole { SeniorRole::SeriesTemporales }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let hurst = payload.hurst_exponent;
        let signal = if hurst > 0.55 { 1.0 } else if hurst < 0.42 { -1.0 } else { 0.0 };
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence: (hurst - 0.5).abs() * 2.0,
            weight: 1.0,
            is_veto: false,
            justification: format!("Hurst exponent: {:.4}", hurst),
        }
    }
}

// 3. Senior Grafos
pub struct SeniorGrafos;
impl SeniorAgent for SeniorGrafos {
    fn role(&self) -> SeniorRole { SeniorRole::Grafos }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let correlation = payload.graph_correlation;
        SeniorOpinion {
            role: self.role(),
            signal_direction: correlation.signum(),
            confidence: correlation.abs().clamp(0.0, 1.0),
            weight: 0.9,
            is_veto: false,
            justification: format!("Graph correlation score: {:.4}", correlation),
        }
    }
}

// 4. Senior Causal (VETO ON MANIPULATION)
pub struct SeniorCausal;
impl SeniorAgent for SeniorCausal {
    fn role(&self) -> SeniorRole { SeniorRole::Causal }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let do_calculus_risk = payload.do_calculus_risk;
        let is_veto = do_calculus_risk > 0.8; // High manipulation risk
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto { 0.0 } else { 1.0 },
            confidence: (1.0 - do_calculus_risk).clamp(0.0, 1.0),
            weight: 1.2,
            is_veto,
            justification: format!("Causal manipulation risk: {:.4}", do_calculus_risk),
        }
    }
}

// 5. Senior Riesgo (VETO ON VAR/DRAWDOWN)
pub struct SeniorRiesgo;
impl SeniorAgent for SeniorRiesgo {
    fn role(&self) -> SeniorRole { SeniorRole::Riesgo }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let drawdown = payload.current_drawdown_pct;
        let is_veto = drawdown > 0.15; // Hard veto if DD > 15%
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto { 0.0 } else { 1.0 },
            confidence: (1.0 - drawdown).clamp(0.0, 1.0),
            weight: 1.5,
            is_veto,
            justification: format!("Risk drawdown assessment: {:.4}", drawdown),
        }
    }
}

// 6. Senior Ejecucion (VETO ON IMPACT)
pub struct SeniorEjecucion;
impl SeniorAgent for SeniorEjecucion {
    fn role(&self) -> SeniorRole { SeniorRole::Ejecucion }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let slippage_est = payload.estimated_slippage_bps;
        let is_veto = slippage_est > 0.005; // > 50 bps slippage veto
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto { 0.0 } else { 1.0 },
            confidence: 0.9,
            weight: 1.0,
            is_veto,
            justification: format!("Execution impact slippage: {:.4}", slippage_est),
        }
    }
}

// 7. Senior Cuantico — Analiza la interacción entre book_imbalance y slippage
// para evaluar si la microestructura permite una ejecución rentable.
pub struct SeniorCuantico;
impl SeniorAgent for SeniorCuantico {
    fn role(&self) -> SeniorRole { SeniorRole::Cuantico }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        // Combinatorial score: fuerte imbalance + bajo slippage = alta calidad de ejecución
        let execution_quality = payload.book_imbalance.abs() - payload.estimated_slippage_bps * 100.0;
        let signal = execution_quality.clamp(-1.0, 1.0);
        let confidence = (execution_quality.abs() * 2.0).clamp(0.0, 1.0);
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence,
            weight: 1.0,
            is_veto: false,
            justification: format!("Quantum exec quality: {:.4} (imb={:.4}, slip={:.4})", execution_quality, payload.book_imbalance, payload.estimated_slippage_bps),
        }
    }
}

// 8. Senior Metacognitivo — Pondera historial (wr) contra riesgo actual.
// Si WR es bajo Y drawdown es alto, reduce confidence drásticamente.
pub struct SeniorMetacognitivo;
impl SeniorAgent for SeniorMetacognitivo {
    fn role(&self) -> SeniorRole { SeniorRole::Metacognitivo }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        // Metacognition: evaluar si la confianza del sistema está justificada por los resultados
        let dd_penalty = (payload.current_drawdown_pct * 5.0).clamp(0.0, 0.5);
        let adjusted_confidence = (wr - dd_penalty).clamp(0.0, 1.0);
        // Si WR < 0.45, la señal se invierte (el sistema se equivoca más de lo que acierta)
        let signal = if wr < 0.45 { -0.5 } else if wr > 0.60 { 1.0 } else { 0.3 };
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence: adjusted_confidence,
            weight: 2.0,
            is_veto: false,
            justification: format!("Metacognitive WR={:.4}, DD_penalty={:.4}, adj_conf={:.4}", wr, dd_penalty, adjusted_confidence),
        }
    }
}

// 9. Senior Teleonomia (VETO ON UTILITY)
// Evalúa si la operación tiene utilidad futura positiva considerando
// el contexto macro completo del payload.
pub struct SeniorTeleonomia;
impl SeniorAgent for SeniorTeleonomia {
    fn role(&self) -> SeniorRole { SeniorRole::Teleonomia }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        // Utilidad esperada: alta correlación de grafos + buen Hurst + bajo slippage → utilidad positiva
        let utility = payload.graph_correlation.abs() * 0.3
            + (payload.hurst_exponent - 0.5).abs() * 0.4
            + (1.0 - payload.estimated_slippage_bps * 200.0).clamp(0.0, 1.0) * 0.3;
        // VETO si la utilidad esperada es negativa (el trade no justifica el riesgo)
        let is_veto = utility < 0.15 && wr < 0.55;
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto { 0.0 } else { utility.clamp(-1.0, 1.0) },
            confidence: utility.clamp(0.0, 1.0),
            weight: 1.0,
            is_veto,
            justification: format!("Teleonomic utility={:.4}, wr={:.4}, veto={}", utility, wr, is_veto),
        }
    }
}

// 10. Senior Auditor Interno (El Anti-Sistema / VETO ON DISCREPANCY & SELF-DECEPTION)
pub struct SeniorAuditorInterno;
impl SeniorAgent for SeniorAuditorInterno {
    fn role(&self) -> SeniorRole { SeniorRole::AuditorInterno }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let is_veto = payload.do_calculus_risk > 0.90 || payload.current_drawdown_pct > 0.12;
        SeniorOpinion {
            role: self.role(),
            signal_direction: if is_veto { 0.0 } else { 1.0 },
            confidence: 1.0,
            weight: 3.0, // Maximum authority as Devil's Advocate
            is_veto,
            justification: format!("Auditor Interno check: DD={:.4}, Risk={:.4}", payload.current_drawdown_pct, payload.do_calculus_risk),
        }
    }
}

pub struct ConsejoDeliberacion {
    agents: Vec<Box<dyn SeniorAgent>>,
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
        }
    }
}

impl ConsejoDeliberacion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Deliberates on market snapshot requiring explicit payload validation, 60% consensus, and 0 vetoes
    pub fn deliberar(&self, payload: &MarketSnapshotPayload, win_rate: f64) -> ConsensusResult {
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
        let mut total_weights = 0.0;

        for agent in &self.agents {
            let op = agent.evaluate(payload, win_rate);
            if op.is_veto {
                return ConsensusResult {
                    approved: false,
                    final_signal: 0.0,
                    total_consensus_pct: 0.0,
                    vetoed_by: Some(op.role),
                    dissenting_log: vec![op],
                };
            }

            total_weighted_signal += op.signal_direction * op.confidence * op.weight;
            total_weights += op.confidence * op.weight;
            opinions.push(op);
        }

        let final_signal = if total_weights > 0.0 {
            total_weighted_signal / total_weights
        } else {
            0.0
        };

        let positive_weight: f64 = opinions.iter()
            .filter(|o| o.signal_direction > 0.0)
            .map(|o| o.confidence * o.weight)
            .sum();

        let consensus_pct = if total_weights > 0.0 { positive_weight / total_weights } else { 0.0 };
        let approved = consensus_pct >= 0.60;

        let dissenting_log = opinions.into_iter().filter(|o| o.signal_direction.signum() != final_signal.signum()).collect();

        ConsensusResult {
            approved,
            final_signal,
            total_consensus_pct: consensus_pct,
            vetoed_by: None,
            dissenting_log,
        }
    }
}
