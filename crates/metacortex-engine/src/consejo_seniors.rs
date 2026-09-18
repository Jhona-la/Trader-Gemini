//! # Council of Seniors — Distributed Adversarial Deliberation Engine
//!
//! Implements a 10-member council of specialized senior agents in `cerebro/consejo/`.
//!
//! MOD2/7-006 (INFORME DECIMOCUARTO) — DIVERSIFICACIÓN DEL CONSEJO:
//! antes, 5 de 6 seniors direccionales eran transformadas colineales de
//! {OBI, f(OBI), Hurst} (grafos = sign(impulse_mom) que ES OBI del
//! lead_lag_engine; cuántico = promedio de micro+grafos; metacognitivo =
//! f(micro, grafos); series = tanh(Hurst)·dirección de grafos/OBI). El
//! consejo era un eco de cámara de 3 señales ya gateadas ≥6 veces aguas
//! arriba. Ahora cada asiento opina desde una PERSPECTIVA DISTINTA con su
//! PROPIO dato crudo e independiente del `MarketSnapshotPayload`:
//!
//! | # | Asiento | Dato único | Papel |
//! |---|---------|-----------|-------|
//! | 0 | SeniorMicroestructura (Flujo) | `book_imbalance` (OBI L2) | direccional — la única señal real de microestructura |
//! | 1 | SeniorSeriesTemporales (Espectral) | `fused_score` + `persistence` | direccional — momentum sostenible vs mean-reversion |
//! | 2 | SeniorVolatilidad | `atr_pct` + `intended_direction` | modulador de convicción — mercados tranquilos vs peligrosos |
//! | 3 | SeniorCausal | `do_calculus_risk` (VPIN) | VETO de manipulación |
//! | 4 | SeniorRiesgo | `current_drawdown_pct` + `loss_streak` + `intended_direction` | VETO de drawdown + modulador de convicción por racha |
//! | 5 | SeniorEjecucion | `estimated_slippage_bps` | VETO de impacto |
//! | 6 | SeniorML | `ml_prob` | direccional — ¿el ensamble apoya esta dirección? |
//! | 7 | SeniorMetacognitivo | divergencia entre {ml, espectral, flujo} + win_rate | direccional — calidad/divergencia de las otras opiniones |
//! | 8 | SeniorTeleonomia | `hurst` + `fused_score` + `ml_prob` + fricción | VETO de utilidad esperada |
//! | 9 | SeniorAuditorInterno | `do_calculus_risk` + drawdown | VETO del abogado del diablo |
//!
//! `graph_correlation` fue RETIRADO del payload: era `impulse_mom` del
//! lead_lag_engine, computado A PARTIR del OBI — un duplicado colineal, no
//! una perspectiva. El consenso 0.35 (B3.31) con este consejo significa
//! "al menos 2-3 perspectivas independientes alineadas".

use serde::{Deserialize, Serialize};

/// U-6 (MOTOR UNIVERSAL CONTINUO): variantes Scalping/Swing extirpadas —
/// el eje temporal del consejo es `dominant_tau_ms` del payload (continuo).
/// `Continuous` se conserva como único valor del contrato serde.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TradingHorizon {
    #[default]
    Continuous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketSnapshotPayload {
    pub horizon: TradingHorizon,
    /// OBI del libro L2 — dato EXCLUSIVO del asiento Flujo (Microestructura).
    pub book_imbalance: f64,
    /// Hurst — usado SOLO por Teleonomia como escala de predictibilidad.
    pub hurst_exponent: f64,
    /// MOD2/7-006: probabilidades del ensamble ML [0,1] — dato EXCLUSIVO del
    /// asiento ML. >0.55 alcista, <0.45 bajista, neutral en la banda muerta.
    pub ml_prob: f64,
    /// Score espectral fusionado [-1,1] — dato EXCLUSIVO del asiento Espectral.
    pub fused_score: f64,
    /// Persistencia de la escala dominante [-1,1] (>0 momentum sostenible,
    /// <0 mean-reversion probable) — dato EXCLUSIVO del asiento Espectral.
    pub persistence: f64,
    /// ATR relativo al precio (v_t/mid) — dato EXCLUSIVO del asiento Volatilidad.
    pub atr_pct: f64,
    /// Racha de pérdidas reciente de la dirección bajo deliberación — dato
    /// EXCLUSIVO del asiento Riesgo.
    pub loss_streak: u32,
    /// Dirección de la entrada bajo deliberación (-1/0/+1). NO es una opinión:
    /// es el contexto que los asientos de convicción (Riesgo, Volatilidad)
    /// modulan sin generar dirección propia.
    pub intended_direction: f64,
    pub do_calculus_risk: f64,
    pub causal_veto_threshold: f64,
    pub current_drawdown_pct: f64,
    pub estimated_slippage_bps: f64,
    /// U-4 (MOTOR UNIVERSAL CONTINUO): τ dominante del espectro (ms) del
    /// símbolo bajo deliberación. Los asientos que arbitraban por horizonte
    /// (Riesgo: caps de DD 0.95/0.90/0.85; Ejecución: slippage 35/65/100
    /// bps) ahora interpolan LOG-LINEALMENTE en τ sobre la banda operativa
    /// [30 s, 12 h] — misma semántica, sin saltos, sin etiquetas.
    pub dominant_tau_ms: f64,
}

impl MarketSnapshotPayload {
    /// U-4 — posición espectral s∈[0,1] de τ en la banda operativa
    /// (log-lineal): 0 = banda rápida (30 s), 1 = banda lenta (12 h).
    /// τ fuera de banda se clampa a los extremos.
    pub fn spectral_s(&self) -> f64 {
        const TAU_FAST: f64 = 30_000.0;
        const TAU_SLOW: f64 = 43_200_000.0;
        let tau = if self.dominant_tau_ms.is_finite() && self.dominant_tau_ms > 0.0 {
            self.dominant_tau_ms
        } else {
            (TAU_FAST * TAU_SLOW).sqrt()
        };
        ((tau / TAU_FAST).ln() / (TAU_SLOW / TAU_FAST).ln()).clamp(0.0, 1.0)
    }

    /// Validates data integrity: fails explicitly on NaN, Inf, or out-of-bound anomalies
    pub fn validate(&self) -> Result<(), String> {
        let metrics = [
            ("book_imbalance", self.book_imbalance),
            ("hurst_exponent", self.hurst_exponent),
            ("ml_prob", self.ml_prob),
            ("fused_score", self.fused_score),
            ("persistence", self.persistence),
            ("atr_pct", self.atr_pct),
            ("intended_direction", self.intended_direction),
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
        if !(0.0..=1.0).contains(&self.ml_prob) {
            return Err(format!("Out-of-bounds ml_prob: {}", self.ml_prob));
        }
        if !(-1.0..=1.0).contains(&self.persistence) {
            return Err(format!("Out-of-bounds persistence: {}", self.persistence));
        }
        if self.atr_pct < 0.0 {
            return Err(format!("Out-of-bounds atr_pct: {}", self.atr_pct));
        }
        if !(-1.0..=1.0).contains(&self.intended_direction) {
            return Err(format!(
                "Out-of-bounds intended_direction: {}",
                self.intended_direction
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeniorRole {
    Microestructura,
    SeriesTemporales,
    /// MOD2/7-006: asiento de VOLATILIDAD (antes Grafos: leía impulse_mom
    /// del lead_lag_engine, que ES OBI — colineal por construcción).
    Volatilidad,
    Causal,
    Riesgo,
    Ejecucion,
    /// MOD2/7-006: asiento ML (antes Cuántico: era el promedio de OBI+grafos).
    Ml,
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

/// MOD2/7-006: opinión espectral pura a partir de los observables INDEPENDIENTES
/// del espectro temporal (fused_score y persistence). No toca OBI ni Hurst.
/// - persistence > 0 → momentum sostenible: CONFIRMA la dirección de fused_score.
/// - persistence < 0 → mean-reversion probable: FADEA la dirección de fused_score.
/// - persistence ≈ 0 → ruido: la transición tanh es C^inf y anula la opinión.
#[inline(always)]
fn spectral_opinion(fused_score: f64, persistence: f64) -> f64 {
    (fused_score * (persistence * 2.0).tanh()).clamp(-1.0, 1.0)
}

/// MOD2/7-006: opinión del ensamble ML con banda muerta. ml_prob > 0.55 →
/// alcista, < 0.45 → bajista, neutral en la banda muerta (±0.05 alrededor de
/// 0.5). Convicción plena con |edge| ≥ 0.20 (ml 0.70/0.30 — el techo B3.19).
#[inline(always)]
fn ml_opinion(ml_prob: f64) -> f64 {
    let edge = ml_prob - 0.5;
    const DEAD_ZONE: f64 = 0.05;
    const FULL_EDGE: f64 = 0.20;
    if edge.abs() < DEAD_ZONE {
        0.0
    } else {
        ((edge - DEAD_ZONE * edge.signum()) / (FULL_EDGE - DEAD_ZONE)).clamp(-1.0, 1.0)
    }
}

// 1. Senior Microestructura (Flujo) — MOD2/7-006: MANTIENE el OBI actual:
// es la ÚNICA señal real de microestructura del consejo (presión del libro L2).
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
            justification: format!("Flujo L2 (OBI): {:.4}", imbalance),
        }
    }
}

// 2. Senior Series Temporales (Espectral) — MOD2/7-006: ya NO lee Hurst·OBI
// (colineal). Opina desde el espectro temporal: fused_score direcciona y
// persistence decide si el momentum es sostenible (>0) o si impera la
// mean-reversion (<0, se fadea la señal espectral).
pub struct SeniorSeriesTemporales;
impl SeniorAgent for SeniorSeriesTemporales {
    fn role(&self) -> SeniorRole {
        SeniorRole::SeriesTemporales
    }
    #[inline(always)]
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let signal = spectral_opinion(payload.fused_score, payload.persistence);
        let confidence = signal.abs().clamp(0.0, 1.0);
        let regime = if payload.persistence >= 0.0 {
            "momentum"
        } else {
            "mean-reversion"
        };
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence,
            weight: 1.0,
            is_veto: false,
            justification: format!(
                "Espectral: fused={:.4}, persistence={:.4} ({}) → sig={:.3}",
                payload.fused_score, payload.persistence, regime, signal
            ),
        }
    }
}

// 3. Senior Volatilidad (antes Grafos) — MOD2/7-006: el asiento "grafos" leía
// impulse_mom del lead_lag_engine, computado A PARTIR del OBI (colinealidad
// literal). Ahora opina desde el ATR relativo: mercados tranquilos autorizan
// convicción plena; mercados peligrosos la reducen. NO genera dirección:
// modula la convicción de la entrada bajo deliberación.
pub struct SeniorVolatilidad;
impl SeniorAgent for SeniorVolatilidad {
    fn role(&self) -> SeniorRole {
        SeniorRole::Volatilidad
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let atr_pct = payload.atr_pct.max(0.0);
        // Bandas de referencia crypto (scalp 1m): calma ≤ 10 bps del precio,
        // peligro ≥ 40 bps. La convicción decae linealmente hasta un piso de
        // 0.4 — nunca cero: la volatilidad es riesgo, no veto direccional.
        const CALM_ATR_PCT: f64 = 0.0010;
        const DANGER_ATR_PCT: f64 = 0.0040;
        let regime_factor =
            1.0 - 0.6 * ((atr_pct - CALM_ATR_PCT) / (DANGER_ATR_PCT - CALM_ATR_PCT)).clamp(0.0, 1.0);
        let dir = safe_signum(payload.intended_direction);
        SeniorOpinion {
            role: self.role(),
            signal_direction: dir,
            confidence: if dir != 0.0 { regime_factor } else { 0.0 },
            weight: 0.9,
            is_veto: false,
            justification: format!(
                "Volatilidad: ATR%={:.4} → convicción {:.2} (dir={:+.0})",
                atr_pct, regime_factor, dir
            ),
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
        // U-4: el breakout alineado antes dependía de etiquetas de horizonte
        // — en el continuo, un libro alineado O una τ de banda rápida (la
        // microestructura lidera los breakouts) califican igual.
        let is_aligned_breakout = (payload.book_imbalance.abs() > 0.25 || payload.spectral_s() < 0.5)
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

// 5. Senior Riesgo — MOD2/7-006: conserva el VETO de drawdown extremo, y su
// voto pasa a ser un MODULADOR DE CONVICCIÓN direccional: opina sobre la
// entrada bajo deliberación (`intended_direction`) con la convicción que
// merecen el drawdown reciente y la racha de pérdidas de esa dirección.
// streak = 0 → convicción plena; streak > 2 → convicción claramente reducida.
pub struct SeniorRiesgo;
impl SeniorAgent for SeniorRiesgo {
    fn role(&self) -> SeniorRole {
        SeniorRole::Riesgo
    }
    #[inline(always)]
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let drawdown = payload.current_drawdown_pct;

        // U-4 (MOTOR UNIVERSAL CONTINUO): cap de DD continuo en τ — antes
        // escalones por horizonte (Scalping 0.95 / Continuous 0.90 / Swing
        // 0.85). Misma semántica (banda rápida tolera más DD para recovery,
        // banda lenta menos), sin saltos ni etiquetas: s=0 (τ 30s) ⇒ 0.95,
        // s=1 (τ 12h) ⇒ 0.85.
        let max_drawdown = 0.95 - 0.10 * payload.spectral_s();

        let is_veto = drawdown > max_drawdown;

        // MOD2/7-006: convicción por racha — 1/(1 + 0.5·streak): plena con
        // streak 0, ~0.67 con 1, 0.5 con 2, ≤0.4 cuando streak > 2.
        let streak = payload.loss_streak;
        let streak_factor = 1.0 / (1.0 + 0.5 * streak as f64);
        let conviction = (streak_factor * (1.0 - drawdown.clamp(0.0, 1.0))).clamp(0.05, 1.0);
        let dir = safe_signum(payload.intended_direction);

        SeniorOpinion {
            role: self.role(),
            signal_direction: dir, // Modula la entrada bajo deliberación
            confidence: if dir != 0.0 { conviction } else { 0.0 },
            weight: 1.5,
            is_veto,
            justification: format!(
                "Riesgo: DD={:.4}, racha={} → convicción {:.2} (dir={:+.0})",
                drawdown, streak, conviction, dir
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
        // U-4: umbral de slippage continuo en τ — antes escalones por
        // horizonte (Scalping 35 / Continuous 65 / Swing 100 bps). Un trade
        // de τ corto no puede pagar 100 bps; uno de 12 h puede absorberlos.
        // D-113: extremo rápido conserva la tolerancia estrecha de altcoins.
        let max_slippage = 35.0 + 65.0 * payload.spectral_s();
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

// 7. Senior ML (antes Cuántico) — MOD2/7-006: la "superposición cuántica"
// era (OBI + f(OBI))/√2 — el promedio de dos seniors previos, cero
// información nueva. Ahora es la opinión del ENSAMBLE ML: ml_prob > 0.55 →
// alcista, < 0.45 → bajista, banda muerta neutral en medio. Dato 100%
// independiente del OBI (el ensamble forest⊕NN no ve el libro L2).
pub struct SeniorML;
impl SeniorAgent for SeniorML {
    fn role(&self) -> SeniorRole {
        SeniorRole::Ml
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, _wr: f64) -> SeniorOpinion {
        let p = if payload.ml_prob.is_finite() {
            payload.ml_prob.clamp(0.0, 1.0)
        } else {
            0.5 // Falla segura: sin modelo no hay opinión
        };
        let signal = ml_opinion(p);
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence: signal.abs().clamp(0.0, 1.0),
            weight: 1.0,
            is_veto: false,
            justification: format!("ML ensemble: ml_prob={:.4} → sig={:.3}", p, signal),
        }
    }
}

// 8. Senior Metacognitivo — MOD2/7-006: MANTIENE su papel de evaluador de la
// CALIDAD de las otras opiniones, pero la "resonancia" ya no es OBI·f(OBI)
// (eco de cámara). Ahora mide la DIVERGENCIA entre las tres perspectivas
// direccionales genuinamente independientes del consejo — ML (modelo),
// Espectral (régimen) y Flujo (libro L2) — y modera su convicción por el
// rendimiento histórico (win rate) y la humildad epistémica:
//   - unánimidad (3/3)  → convicción plena en la dirección media;
//   - mayoría (2/3, sin disenso) → convicción moderada;
//   - evidencia fina (1/3)  → convicción reducida;
//   - divergencia real (≥1 vs ≥1) → ABSTENCIÓN: incertidumbre.
pub struct SeniorMetacognitivo;
impl SeniorAgent for SeniorMetacognitivo {
    fn role(&self) -> SeniorRole {
        SeniorRole::Metacognitivo
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        let dd_penalty = (payload.current_drawdown_pct * 5.0).clamp(0.0, 0.5);
        let effective_wr = if wr > 0.0 { wr.clamp(0.20, 1.0) } else { 0.50 };
        let adjusted_confidence = (effective_wr - dd_penalty).clamp(0.05, 1.0);

        // Las tres perspectivas independientes, cada una con su propio dato:
        let ml_sig = ml_opinion(payload.ml_prob.clamp(0.0, 1.0));
        let spec_sig = spectral_opinion(payload.fused_score, payload.persistence);
        let flow_sig = payload.book_imbalance.clamp(-1.0, 1.0);

        let dirs = [
            safe_signum(ml_sig),
            safe_signum(spec_sig),
            safe_signum(flow_sig),
        ];
        let pos = dirs.iter().filter(|&&d| d > 0.0).count();
        let neg = dirs.iter().filter(|&&d| d < 0.0).count();
        let mean = (ml_sig + spec_sig + flow_sig) / 3.0;

        let (signal_dir, divergence) = if pos == 3 || neg == 3 {
            (mean, false) // Unanimidad de las 3 perspectivas
        } else if (pos >= 2 && neg == 0) || (neg >= 2 && pos == 0) {
            (mean * 0.7, false) // Mayoría sin disenso
        } else if pos + neg == 1 {
            (mean * 0.4, false) // Una sola voz: evidencia fina
        } else {
            (0.0, true) // Disonancia genuina: abstención / humildad epistémica
        };

        let weight = (1.0 + (effective_wr - 0.20) / 0.80).clamp(1.0, 2.0);
        SeniorOpinion {
            role: self.role(),
            signal_direction: signal_dir,
            confidence: if divergence { 0.0 } else { adjusted_confidence },
            weight,
            is_veto: false,
            justification: format!(
                "Metacognitivo: ml={:.2}, spec={:.2}, flujo={:.2} → {} (wr={:.3})",
                ml_sig,
                spec_sig,
                flow_sig,
                if divergence { "divergencia → abstención" } else { "consenso de perspectivas" },
                wr
            ),
        }
    }
}

// 9. Senior Teleonomia (VETO ON UTILITY)
// MOD2/7-006: descolinearizado — la utilidad ya NO se compone de
// graph_correlation (OBI) y book_imbalance (OBI). Ahora: contribución
// espectral escalada por la predictibilidad fractal (Hurst) + contribución
// del ensamble ML, penalizadas por fricción de ejecución y toxicidad.
pub struct SeniorTeleonomia;
impl SeniorAgent for SeniorTeleonomia {
    fn role(&self) -> SeniorRole {
        SeniorRole::Teleonomia
    }
    fn evaluate(&self, payload: &MarketSnapshotPayload, wr: f64) -> SeniorOpinion {
        let slippage_penalty = (payload.estimated_slippage_bps / 50.0).clamp(0.0, 1.0);
        let execution_quality = 1.0 - slippage_penalty;

        let predictability = ((payload.hurst_exponent - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let spectral = spectral_opinion(payload.fused_score, payload.persistence);
        let ml_edge = ((payload.ml_prob.clamp(0.0, 1.0) - 0.5) * 2.0).clamp(-1.0, 1.0);
        let toxicity_penalty = 1.0 - 0.30 * payload.do_calculus_risk.clamp(0.0, 1.0);
        let expected_utility = (spectral * predictability * 0.6 + ml_edge * 0.4)
            * execution_quality
            * toxicity_penalty;

        let is_veto = expected_utility.abs() < 0.02 && wr < 0.35;
        let signal = if is_veto {
            0.0
        } else {
            expected_utility.clamp(-1.0, 1.0)
        };
        let confidence = expected_utility.abs().clamp(0.0, 1.0);

        SeniorOpinion {
            role: self.role(),
            signal_direction: signal,
            confidence,
            weight: 1.0,
            is_veto,
            justification: format!(
                "Teleonomia: utilidad={:.4}, sig={:.4}, wr={:.4}, veto={}",
                expected_utility, signal, wr, is_veto
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
        // Umbral de drawdown continuo en τ alineado con SeniorRiesgo para
        // evitar vetos contradictorios (D-343, U-4: antes escalones
        // Scalping/Continuous/Swing — misma curva que SeniorRiesgo).
        let max_dd = 0.95 - 0.10 * payload.spectral_s();
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
                Box::new(SeniorVolatilidad),
                Box::new(SeniorCausal),
                Box::new(SeniorRiesgo),
                Box::new(SeniorEjecucion),
                Box::new(SeniorML),
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

        // MOD2/7-006: el denominador del consenso es la capacidad ponderada de
        // los asientos con PERSPECTIVA DIRECCIONAL PROPIA (Flujo, Espectral,
        // ML, Metacognitivo, Teleonomia). Riesgo y Volatilidad son MODULADORES
        // de convicción: heredan `intended_direction` (no generan dirección) —
        // cuentan en `final_signal` pero NO en el consenso, o el consejo
        // rubber-stamp-earía su propia entrada. Causal/Ejecución/Auditor son
        // asientos de permiso/veto. El MISMO filtro se aplica al numerador
        // (positive/negative capacity) o el consenso excedería 1.0.
        let is_directional_seat = |o: &SeniorOpinion| {
            !matches!(
                o.role,
                SeniorRole::Causal
                    | SeniorRole::Riesgo
                    | SeniorRole::Ejecucion
                    | SeniorRole::Volatilidad
                    | SeniorRole::AuditorInterno
            )
        };

        let total_directional_capacity: f64 = opinions
            .iter()
            .filter(|o| is_directional_seat(o))
            .map(|o| o.weight * o.confidence.clamp(0.0, 1.0))
            .sum();

        let positive_capacity: f64 = opinions
            .iter()
            .filter(|o| is_directional_seat(o) && o.signal_direction > 0.0)
            .map(|o| o.weight * o.confidence.clamp(0.0, 1.0))
            .sum();

        let negative_capacity: f64 = opinions
            .iter()
            .filter(|o| is_directional_seat(o) && o.signal_direction < 0.0)
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

        // MOD2/7-013 (INFORME DECIMOCUARTO, FOCO 2 «Rigidez de filtros»): el
        // umbral de aprobación baja de 0.50 (mayoría absoluta) a 0.35
        // (minoría sustancial). MOD2/7-006: con el consejo ahora DIVERSO
        // (Flujo/Espectral/ML/Metacognitivo/Teleonomia como perspectivas
        // independientes), consenso 0.35 significa «al menos 2-3 perspectivas
        // independientes alineadas», no «el mismo OBI visto desde 5 ángulos».
        // El ML (B3.18) y el risk-engine ya gatearon la entrada; el consejo es
        // la última deliberación cualitativa, no un segundo embudo cuantitativo.
        let (approved, consensus_pct) = if vetoed_by.is_some() {
            (false, 0.0)
        } else if long_consensus_pct >= 0.35 && final_signal > 0.0 {
            (true, long_consensus_pct)
        } else if short_consensus_pct >= 0.35 && final_signal < 0.0 {
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

    /// Payload base diverso: todas las perspectivas independientes alineadas
    /// al alza (flujo, espectro, ML) en un mercado tranquilo sin racha.
    fn diverse_bullish_payload() -> MarketSnapshotPayload {
        MarketSnapshotPayload {
            horizon: TradingHorizon::Continuous,
            book_imbalance: 0.85,
            hurst_exponent: 0.72,
            ml_prob: 0.70,
            fused_score: 0.50,
            persistence: 0.60,
            atr_pct: 0.0010,
            loss_streak: 0,
            intended_direction: 1.0,
            do_calculus_risk: 0.5,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.05,
            // 20 bps: bajo el límite de 35 bps de SeniorEjecucion para Scalping.
            estimated_slippage_bps: 20.0,
            dominant_tau_ms: 1_138_000.0,
        }
    }

    #[test]
    fn test_consejo_deliberacion_long_approval() {
        let consejo = ConsejoDeliberacion::new();
        let payload = diverse_bullish_payload();

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
            horizon: TradingHorizon::Continuous,
            book_imbalance: -0.85,
            hurst_exponent: 0.35,
            ml_prob: 0.28,
            fused_score: -0.55,
            persistence: 0.50,
            atr_pct: 0.0010,
            loss_streak: 0,
            intended_direction: -1.0,
            do_calculus_risk: 0.05,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: 0.01,
            estimated_slippage_bps: 0.0005,
            dominant_tau_ms: 1_138_000.0,
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
        let mut payload = diverse_bullish_payload();
        payload.do_calculus_risk = 0.95; // Doble veto: Causal (>0.80) + Auditor (>0.92)

        let result = consejo.deliberar(&payload, 0.70);
        assert!(!result.approved, "Trade must be rejected due to VETO");
        assert_eq!(result.vetoed_by, Some(SeniorRole::Causal));
    }

    #[test]
    fn test_consejo_deliberacion_with_custom_weights() {
        let consejo = ConsejoDeliberacion::new();
        let payload = diverse_bullish_payload();

        // Multiplicadores que potencian a Flujo y Espectral
        let weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = consejo.deliberar_with_weights(&payload, 0.70, Some(&weights));
        assert!(result.approved);
        assert!(result.final_signal > 0.0);
        assert!(result.total_consensus_pct >= 0.60);
    }

    #[test]
    fn test_senior_performance_tracker_adaptation() {
        let mut tracker = SeniorPerformanceTracker::new(50);
        // Simular 10 trades ganadores donde el Senior 0 (Flujo) acertó
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
            horizon: TradingHorizon::Continuous,
            book_imbalance: f64::NAN,
            hurst_exponent: f64::NAN,
            ml_prob: f64::NAN,
            fused_score: f64::INFINITY,
            persistence: f64::NAN,
            atr_pct: f64::NAN,
            loss_streak: 0,
            intended_direction: f64::NAN,
            do_calculus_risk: f64::INFINITY,
            causal_veto_threshold: 0.80,
            current_drawdown_pct: -0.05,
            estimated_slippage_bps: 10.0,
            dominant_tau_ms: 1_138_000.0,
        };

        let result = consejo.deliberar(&payload, 0.70);
        assert!(result.final_signal.is_finite());
        assert!(result.total_consensus_pct.is_finite());
    }

    /// MOD2/7-006 (INFORME DECIMOCUARTO): test de DIVERSIDAD REAL del consejo.
    /// Antes, 5 de 6 seniors direccionales eran transformadas colineales de
    /// {OBI, f(OBI), Hurst}: cambiar OBI movía TODO el consejo y cambiar
    /// ml/espectro/ATR no movía NADA. Ahora:
    ///   1. Cambiar OBI (sin tocar ml/espectro/ATR) cambia el output del
    ///      consejo — pero SOLO mediante los asientos que legítimamente leen
    ///      flujo (Flujo y Metacognitivo).
    ///   2. Cambiar ml_prob cambia el output — mediante ML y Metacognitivo,
    ///      SIN mover a Flujo.
    ///   3. Los asientos independientes (ML, Espectral, Volatilidad, Riesgo)
    ///      son INVARIANTES a cambios de OBI: no son funciones del OBI.
    #[test]
    fn test_consejo_diversidad_no_colinealidad_mod2_7_006() {
        let consejo = ConsejoDeliberacion::new();

        // Consejo diverso: ML y espectro alcistas, flujo ALCISTA fuerte.
        let base = diverse_bullish_payload();

        // (1) Flip de OBI: +0.85 → −0.85 con ml/espectro/ATR/racha intactos.
        let obi_flip = MarketSnapshotPayload {
            book_imbalance: -0.85,
            ..base.clone()
        };

        let res_base = consejo.deliberar(&base, 0.70);
        let res_obi = consejo.deliberar(&obi_flip, 0.70);

        // Output del consejo DISTINTO al cambiar OBI (diversidad real: el
        // consejo sigue sensible al flujo L2)...
        assert_ne!(
            res_base.final_signal, res_obi.final_signal,
            "cambiar OBI debe cambiar el final_signal del consejo"
        );
        assert!(
            (res_base.final_signal - res_obi.final_signal).abs() > 0.10,
            "el cambio debe ser material (>0.10), no cosmético: base={:.3} vs obi_flip={:.3}",
            res_base.final_signal,
            res_obi.final_signal
        );
        assert_ne!(
            res_base.total_consensus_pct, res_obi.total_consensus_pct,
            "cambiar OBI debe cambiar el consenso del consejo"
        );

        // ...pero SOLO los asientos que legítimamente leen OBI se mueven.
        // Índices: 0=Flujo, 1=Espectral, 2=Volatilidad, 3=Causal, 4=Riesgo,
        //          5=Ejecución, 6=ML, 7=Metacognitivo, 8=Teleonomia, 9=Auditor.
        let sigs_base = consejo.extract_senior_signals(&base, 0.70);
        let sigs_obi = consejo.extract_senior_signals(&obi_flip, 0.70);
        for (idx, (a, b)) in sigs_base.iter().zip(sigs_obi.iter()).enumerate() {
            let moved = (a - b).abs() > 1e-9;
            let reads_obi = matches!(idx, 0 | 7); // Flujo y Metacognitivo
            assert_eq!(
                moved,
                reads_obi,
                "asiento {} {} ante flip de OBI (antes: TODO el consejo era f(OBI))",
                idx,
                if moved { "se movió" } else { "no se movió" }
            );
        }

        // (2) Flip de ml_prob: 0.70 → 0.30 con OBI/espectro/ATR intactos.
        let ml_flip = MarketSnapshotPayload {
            ml_prob: 0.30,
            ..base.clone()
        };
        let res_ml = consejo.deliberar(&ml_flip, 0.70);
        assert_ne!(
            res_base.final_signal, res_ml.final_signal,
            "cambiar ml_prob debe cambiar el final_signal del consejo"
        );

        let sigs_ml = consejo.extract_senior_signals(&ml_flip, 0.70);
        for (idx, (a, b)) in sigs_base.iter().zip(sigs_ml.iter()).enumerate() {
            let moved = (a - b).abs() > 1e-9;
            let reads_ml = matches!(idx, 6 | 7 | 8); // ML, Metacognitivo, Teleonomia
            assert_eq!(
                moved, reads_ml,
                "asiento {} {} ante flip de ml_prob",
                idx,
                if moved { "se movió" } else { "no se movió" }
            );
        }

        // (3) Colinealidad estructural imposible: el asiento ML NO es función
        // del OBI ni el asiento Flujo es función del ml_prob.
        let senior_ml = SeniorML;
        assert_eq!(
            senior_ml.evaluate(&base, 0.5).signal_direction,
            senior_ml.evaluate(&obi_flip, 0.5).signal_direction,
            "SeniorML no debe leer OBI"
        );
        let senior_flujo = SeniorMicroestructura;
        assert_eq!(
            senior_flujo.evaluate(&base, 0.5).signal_direction,
            senior_flujo.evaluate(&ml_flip, 0.5).signal_direction,
            "SeniorFlujo no debe leer ml_prob"
        );

        // (4) Independencia de los moduladores: Riesgo responde a la RACHA
        // (streak 0 vs 3 con mismo OBI/ml/espectro) y Volatilidad al ATR
        // (calma vs peligro), sin tocar ninguna señal direccional.
        let streak3 = MarketSnapshotPayload {
            loss_streak: 3,
            ..base.clone()
        };
        let senior_riesgo = SeniorRiesgo;
        let conf0 = senior_riesgo.evaluate(&base, 0.5).confidence;
        let conf3 = senior_riesgo.evaluate(&streak3, 0.5).confidence;
        assert!(
            conf0 > conf3,
            "racha 3 debe reducir la convicción del SeniorRiesgo ({:.2} > {:.2})",
            conf0,
            conf3
        );

        let high_atr = MarketSnapshotPayload {
            atr_pct: 0.0040, // ≥ 40 bps: mercado peligroso
            ..base.clone()
        };
        let senior_vol = SeniorVolatilidad;
        assert!(
            senior_vol.evaluate(&base, 0.5).confidence
                > senior_vol.evaluate(&high_atr, 0.5).confidence,
            "ATR alto debe reducir la convicción del SeniorVolatilidad"
        );
    }
}
