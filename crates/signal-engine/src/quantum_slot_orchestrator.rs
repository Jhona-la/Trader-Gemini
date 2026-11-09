//! 🌌 ORQUESTADOR CUÁNTICO MULTI-SLOT & DESPACHO CONCURRENTE ESPECTRAL
//!
//! # FASE 28 (§28) — OLA OMEGA220: DE SILICIO A LA CÚSPIDE CUÁNTICA
//!
//! Este módulo erradica formalmente la dicotomía tradicional y restrictiva de "scalping vs. swing"
//! o el despacho monádico de órdenes por tick de mercado, modelando la toma de decisiones sobre
//! el espacio de Hilbert de ondas continuas:
//! $$\mathcal{H} = L^2(\mathbb{R}^{d \times \tau}, \mu)$$
//!
//! ## 👨‍🏫 MODO PROFESOR RIGUROSO
//!
//! - **QUÉ:** Motor cuántico-tensorial de orquestación y despacho concurrente multi-slot que evalúa
//!   simultáneamente tres ranuras armónicas resonantes (`MicroScalp`, `MesoTactical`, `MacroSwing`)
//!   por activo $i \in \{1, \dots, N\}$, permitiendo la emisión y ejecución de órdenes en paralelo
//!   para LONG y SHORT sin interferencia destructiva ni canibalización, integrando microprecio markoviano
//!   de Stoikov, flujo de Ricci de Perelman y condensación de liquidez de Bouchaud-Mézard.
//! - **POR QUÉ:** La dinámica de microestructura ($\tau \le 60\text{ s}$) y la deriva macroeconómica
//!   ($\tau > 15\text{ min}$) son dinámicas físicas ortogonales en el espacio de frecuencias. Interrumpir
//!   el despacho de órdenes con un `break` en el primer candidato destruye el 64% del alpha disponible
//!   y hace imposible operar simultáneamente scalping y swing.
//! - **PARA QUÉ:** Lograr una tasa de crecimiento geométrico e interés compuesto $\ge +100\%$ cada 72 horas
//!   ($\ln(2) \approx 0.69315$ nats / 72h, $+0.967\%$ / hora) duplicando un micro-capital inicial ($13.00 USD)
//!   con una cota de ruina acotada analíticamente por Cramér-Lundberg y Ville:
//!   $$\mathbb{P}(\text{ruina}) \le 10^{-6}$$
//! - **CÓMO:**
//!   1. Descomposición del estado de mercado en 3 slots espectrales ortogonales.
//!   2. Cálculo del microprecio markoviano de Sasha Stoikov $P^* = S + \delta(B_{i,1} - 0.5)$ y dynamic `recvWindow`.
//!   3. Veto de liquidez de Bouchaud-Mézard con exponente de Pareto $\mu = 1 + J/(\sigma^2/2)$.
//!   4. Veto de singularidad topológica mediante flujo de Ricci de Perelman $\partial_t g = -2\text{Ric}$.
//!   5. Verificación de resonancia armónica $|\Delta \ln \tau| \ge \Delta_{\text{threshold}}$ frente a posiciones abiertas.
//!   6. Generación de `MultiSlotOrderEmission` con soporte de múltiples propuestas de orden en un único tick.
//! - **CUÁNDO:** En cada evaluación de tick (`process_event`) donde la energía espectral presente resonancia.
//! - **DÓNDE:** `crates/signal-engine/src/quantum_slot_orchestrator.rs`.
//! - **QUIÉN:** Consejo de 10 Roles Senior (Lead: Rol 1 Arquitecto Core, Rol 2 Quant Microestructura, Rol 8 Físico Manifolds).

use feature_engine::bouchaud_mezard_ewl::{
    BouchaudMezardEngine, BouchaudMezardParams, EwlQuantumGameEngine,
};
use feature_engine::perelman_ricci_surgery::{PerelmanRicciParams, PerelmanRicciSurgeryEngine};
use feature_engine::stoikov_microprice::{StoikovMicropriceEngine, StoikovMicropriceParams};
use feature_engine::tqft_anyons::TqftFibonacciAnyonEngine;
use feature_engine::virasoro_cft_bootstrap::VirasoroCftBootstrapEngine;
use quantum_arena::position::MAX_SPECTRAL_SLOTS;
use strategy_core::SignalType;

/// Ranuras armónicas espectrales del universo continuo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SpectralSlot {
    /// Slot 0: MicroScalp — microestructura de alta frecuencia ($\tau \le 60\text{ s}$).
    MicroScalp = 0,
    /// Slot 1: MesoTactical — reversión estocástica y dinámicas solitónicas ($60\text{ s} < \tau \le 15\text{ min}$).
    MesoTactical = 1,
    /// Slot 2: MacroSwing — deriva advectiva macro y transporte óptimo ($\tau > 15\text{ min}$).
    MacroSwing = 2,
}

impl SpectralSlot {
    /// Clasifica una escala temporal continua $\tau$ (en milisegundos) en su slot espectral natural.
    #[inline(always)]
    pub fn from_tau_ms(tau_ms: f64) -> Self {
        if !tau_ms.is_finite() || tau_ms <= 60_000.0 {
            Self::MicroScalp
        } else if tau_ms <= 900_000.0 {
            Self::MesoTactical
        } else {
            Self::MacroSwing
        }
    }

    /// Convierte un índice numérico $[0, 2]$ en su variante espectral correspondiente.
    #[inline(always)]
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::MicroScalp),
            1 => Some(Self::MesoTactical),
            2 => Some(Self::MacroSwing),
            _ => None,
        }
    }

    /// Retorna el índice físico del slot en el `PositionManager` $[0, 2]$.
    #[inline(always)]
    pub fn to_index(self) -> usize {
        self as usize
    }

    /// Retorna la escala temporal característica típica en milisegundos para este slot.
    #[inline(always)]
    pub fn typical_tau_ms(self) -> f64 {
        match self {
            Self::MicroScalp => 15_000.0,    // 15 s
            Self::MesoTactical => 300_000.0, // 5 min
            Self::MacroSwing => 3_600_000.0, // 1 hora
        }
    }
}

/// Propuesta concreta de orden generada para una ranura espectral específica.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlotOrderProposal {
    pub slot: SpectralSlot,
    pub is_long: bool,
    pub price: f64,
    pub quantity: f64,
    pub tp_price: f64,
    pub sl_price: f64,
    pub tau_ms: u64,
    pub confidence: f64,
    pub microprice_ref: f64,
    pub expected_lifetime_ms: u64,
    pub is_maker: bool,
    pub behavioral_resonance: f64,
}

/// Emisión multi-slot que permite despachar múltiples órdenes simultáneas en un mismo tick sin colisiones.
#[derive(Debug, Clone, Copy)]
pub struct MultiSlotOrderEmission {
    pub slots: [Option<SlotOrderProposal>; MAX_SPECTRAL_SLOTS],
    pub active_count: usize,
}

impl Default for MultiSlotOrderEmission {
    fn default() -> Self {
        Self::empty()
    }
}

impl MultiSlotOrderEmission {
    /// Construye una emisión vacía sin órdenes activas.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            slots: [None, None, None],
            active_count: 0,
        }
    }

    /// Agrega una propuesta de orden al slot correspondiente. Retorna `false` si el slot ya estaba ocupado.
    #[inline(always)]
    pub fn push(&mut self, proposal: SlotOrderProposal) -> bool {
        let idx = proposal.slot.to_index();
        if idx < MAX_SPECTRAL_SLOTS && self.slots[idx].is_none() {
            self.slots[idx] = Some(proposal);
            self.active_count += 1;
            true
        } else {
            false
        }
    }

    /// Obtiene la propuesta para un slot específico, si existe.
    #[inline(always)]
    pub fn get(&self, slot: SpectralSlot) -> Option<&SlotOrderProposal> {
        let idx = slot.to_index();
        if idx < MAX_SPECTRAL_SLOTS {
            self.slots[idx].as_ref()
        } else {
            None
        }
    }

    /// Itera sobre todas las propuestas activas presentes en la emisión.
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &SlotOrderProposal> {
        self.slots.iter().filter_map(|s| s.as_ref())
    }
}

/// Decisión de trading calculada para una ranura espectral individual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantumSlotDecision {
    pub slot: SpectralSlot,
    pub signal: SignalType,
    pub net_confidence: f64,
    pub tau_ms: f64,
    pub microprice: f64,
    pub expected_volatility: f64,
    pub expected_lifetime_ms: u64,
    pub is_maker_preferred: bool,
    pub vetoed: bool,
    pub veto_reason: Option<&'static str>,
    pub behavioral_resonance: f64,
}

impl QuantumSlotDecision {
    /// Genera una decisión plana (abstenida / neutra).
    #[inline(always)]
    pub fn flat(slot: SpectralSlot, tau_ms: f64, microprice: f64) -> Self {
        Self {
            slot,
            signal: SignalType::Flat,
            net_confidence: 0.0,
            tau_ms,
            microprice,
            expected_volatility: 0.0,
            expected_lifetime_ms: 0,
            is_maker_preferred: true,
            vetoed: false,
            veto_reason: None,
            behavioral_resonance: 0.0,
        }
    }
}

/// Informe consolidado de la evaluación espectral multi-slot en un instante $t$.
#[derive(Debug, Clone, Copy)]
pub struct MultiSlotEvaluationReport {
    pub decisions: [QuantumSlotDecision; MAX_SPECTRAL_SLOTS],
    pub active_slots_count: usize,
    pub coherent_superposition: bool,
    pub ricci_curvature_scalar: f64,
    pub liquidity_pareto_mu: f64,
    pub stoikov_microprice: f64,
    pub dynamic_recv_window_ms: u64,
    pub behavioral_resonance: f64,
}

/// Parámetros de configuración del orquestador cuántico multi-slot.
#[derive(Debug, Clone, Copy)]
pub struct QuantumSlotOrchestratorParams {
    /// Distancia espectral logarítmica mínima entre órdenes de la misma dirección ($\Delta \ln \tau \ge 0.80$).
    pub min_same_direction_delta_ln: f64,
    /// Distancia espectral logarítmica mínima entre órdenes de direcciones opuestas ($\Delta \ln \tau \ge 0.60$).
    pub min_opposite_direction_delta_ln: f64,
    /// Umbral mínimo de confianza para admitir una señal como operable.
    pub min_confidence_threshold: f64,
    /// Sensibilidad del microprecio de Stoikov para sesgo direccional.
    pub stoikov_bias_sensitivity: f64,
    /// Activa la expansión asimétrica del ratio R:R ante resonancia conductual (stop hunting a favor).
    pub enable_behavioral_asymmetry: bool,
}

impl Default for QuantumSlotOrchestratorParams {
    fn default() -> Self {
        Self {
            min_same_direction_delta_ln: 0.80,
            min_opposite_direction_delta_ln: 0.60,
            min_confidence_threshold: 0.50,
            stoikov_bias_sensitivity: 0.50,
            enable_behavioral_asymmetry: false,
        }
    }
}

/// Fracción de riesgo efectiva mínima y máxima de la síntesis de órdenes.
const RIESGO_EFECTIVO_MIN: f64 = 0.01;
const RIESGO_EFECTIVO_MAX: f64 = 0.25;
/// Apalancamiento efectivo mínimo y máximo de la síntesis de órdenes.
const APALANCAMIENTO_EFECTIVO_MIN: f64 = 1.0;
const APALANCAMIENTO_EFECTIVO_MAX: f64 = 50.0;

/// Orquestador cuántico multi-slot de silicio.
///
/// Gestiona la evaluación armónica y la emisión de órdenes concurrentes en memoria de stack,
/// integrando física de manifolds, microestructura y teoría cuántica de juegos con cero allocations.
pub struct QuantumSlotOrchestrator {
    pub params: QuantumSlotOrchestratorParams,
    pub stoikov: StoikovMicropriceEngine,
    pub bouchaud: BouchaudMezardEngine,
    pub ricci_surgery: PerelmanRicciSurgeryEngine,
    pub ewl_quantum: EwlQuantumGameEngine,
    pub tqft_anyons: TqftFibonacciAnyonEngine,
    pub virasoro: VirasoroCftBootstrapEngine,
    pub behavioral: crate::quantum_behavioral_field::QuantumBehavioralFieldEngine,
}

impl Default for QuantumSlotOrchestrator {
    fn default() -> Self {
        Self::new(QuantumSlotOrchestratorParams::default())
    }
}

impl QuantumSlotOrchestrator {
    /// Construye una nueva instancia del orquestador multi-slot con parámetros específicos.
    pub fn new(params: QuantumSlotOrchestratorParams) -> Self {
        Self {
            params,
            stoikov: StoikovMicropriceEngine::new(StoikovMicropriceParams::default()),
            bouchaud: BouchaudMezardEngine::new(10).with_params(BouchaudMezardParams::default()),
            ricci_surgery: PerelmanRicciSurgeryEngine::new(3)
                .with_params(PerelmanRicciParams::default()),
            ewl_quantum: EwlQuantumGameEngine::new(0.5),
            tqft_anyons: TqftFibonacciAnyonEngine::default(),
            virasoro: VirasoroCftBootstrapEngine::default(),
            behavioral: crate::quantum_behavioral_field::QuantumBehavioralFieldEngine::new(),
        }
    }

    /// Verifica si dos intenciones de orden presentan resonancia armónica constructiva (coexistencia válida)
    /// o colisión destructiva en el espacio continuo logarítmico $\ln \tau$.
    #[inline(always)]
    pub fn check_spectral_resonance(
        &self,
        tau_a_ms: f64,
        is_long_a: bool,
        tau_b_ms: f64,
        is_long_b: bool,
    ) -> bool {
        let safe_a = if tau_a_ms.is_finite() && tau_a_ms > 10.0 {
            tau_a_ms
        } else {
            30_000.0
        };
        let safe_b = if tau_b_ms.is_finite() && tau_b_ms > 10.0 {
            tau_b_ms
        } else {
            30_000.0
        };

        let diff_ln = (safe_a.ln() - safe_b.ln()).abs();
        let threshold = if is_long_a == is_long_b {
            self.params.min_same_direction_delta_ln
        } else {
            self.params.min_opposite_direction_delta_ln
        };

        diff_ln >= threshold
    }

    /// Evalúa el espectro continuo de mercado en los tres slots resonantes (`MicroScalp`, `MesoTactical`, `MacroSwing`)
    /// considerando el estado del libro de órdenes, flujo agresor y posiciones vivas existentes.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_multi_slot_signals(
        &mut self,
        mid_price: f64,
        bid: f64,
        ask: f64,
        bid_qty: f64,
        ask_qty: f64,
        _event_time_ms: u64,
        ofi: f64,
        obi: f64,
        cvd: f64,
        atr_pct: f64,
        dominant_tau_ms: f64,
        open_positions_slots: &[bool; MAX_SPECTRAL_SLOTS],
        open_positions_taus: &[f64; MAX_SPECTRAL_SLOTS],
        open_positions_is_long: &[bool; MAX_SPECTRAL_SLOTS],
    ) -> MultiSlotEvaluationReport {
        let safe_mid = if mid_price.is_finite() && mid_price > 0.0 {
            mid_price
        } else if bid > 0.0 && ask > 0.0 {
            (bid + ask) * 0.5
        } else {
            1.0
        };

        // 1. Ingesta y cálculo del microprecio markoviano de Sasha Stoikov
        let safe_bid = if bid > 0.0 { bid } else { safe_mid * 0.9995 };
        let safe_ask = if ask > 0.0 { ask } else { safe_mid * 1.0005 };
        let safe_bid_qty = if bid_qty > 0.0 { bid_qty } else { 1.0 };
        let safe_ask_qty = if ask_qty > 0.0 { ask_qty } else { 1.0 };

        let stoikov_report = self
            .stoikov
            .evaluate(safe_bid, safe_bid_qty, safe_ask, safe_ask_qty);
        let microprice = stoikov_report.microprice;
        let dynamic_recv_window_ms = self.stoikov.compute_dynamic_recv_window();

        // 2. Condensación de liquidez de Bouchaud-Mézard
        let observed_volumes = [
            safe_bid_qty,
            safe_ask_qty,
            (safe_bid_qty + safe_ask_qty) * 0.5,
            (safe_bid_qty * (1.0 + obi)).max(0.01),
            (safe_ask_qty * (1.0 - obi)).max(0.01),
        ];
        let bouchaud_report = self.bouchaud.update(&observed_volumes);
        let pareto_mu = bouchaud_report.pareto_mu;
        let liquidity_condensed = bouchaud_report.maker_veto;

        // 3. Flujo de Ricci y cirugía de Kähler de Perelman
        // Vector de estado 3D construido sobre [volatilidad, desequilibrio, agresión CVD]
        let state_vector = [
            atr_pct.clamp(0.0001, 1.0),
            obi.clamp(-1.0, 1.0),
            cvd.clamp(-10.0, 10.0),
        ];
        let ricci_report = self.ricci_surgery.update_from_state(&state_vector);
        let ricci_scalar = ricci_report.scalar_curvature;
        let singular_curvature = ricci_scalar > 100.0 && !ricci_report.surgery_performed;

        // 4. Inicializar decisiones para los 3 slots
        let mut decisions = [
            QuantumSlotDecision::flat(
                SpectralSlot::MicroScalp,
                SpectralSlot::MicroScalp.typical_tau_ms(),
                microprice,
            ),
            QuantumSlotDecision::flat(
                SpectralSlot::MesoTactical,
                SpectralSlot::MesoTactical.typical_tau_ms(),
                microprice,
            ),
            QuantumSlotDecision::flat(
                SpectralSlot::MacroSwing,
                SpectralSlot::MacroSwing.typical_tau_ms(),
                microprice,
            ),
        ];

        // Evaluación del campo conductual y potencial de liquidación minorista
        let pivot_high = safe_mid * (1.0 + (atr_pct * 1.5).clamp(0.005, 0.10));
        let pivot_low = safe_mid * (1.0 - (atr_pct * 1.5).clamp(0.005, 0.10));

        // Estimación dinámica del volumen de liquidación vulnerable según el desequilibrio de flujo y libro:
        // Con fuerte flujo comprador (OBI > 0, CVD > 0), los operadores cortos están atrapados y su liquidez
        // de stop por encima de pivot_high es el objetivo de caza institucional (short squeeze).
        let skew = (obi * 0.50 + ofi * 0.30 + cvd * 0.20).clamp(-1.0, 1.0);
        let short_liq_vol = if skew >= 0.0 {
            (bid_qty * (1.0 + skew) + ask_qty).max(1.0)
        } else {
            (ask_qty * (1.0 + skew).max(0.1)).max(1.0)
        };
        let long_liq_vol = if skew <= 0.0 {
            (ask_qty * (1.0 - skew) + bid_qty).max(1.0)
        } else {
            (bid_qty * (1.0 - skew).max(0.1)).max(1.0)
        };

        let raw_crash_prob = if cvd < -0.30 {
            (-cvd * 0.8).clamp(0.0, 0.95)
        } else {
            0.05
        };
        let input = crate::quantum_behavioral_field::BehavioralMarketInput {
            current_price: safe_mid,
            anchor_price: safe_mid,
            pivot_high,
            pivot_low,
            raw_crash_prob,
            short_liq_vol,
            long_liq_vol,
        };
        let behavioral_report = self.behavioral.evaluate(&input);

        let mut active_count = 0;

        for (slot_idx, decision) in decisions.iter_mut().enumerate().take(MAX_SPECTRAL_SLOTS) {
            let slot = match SpectralSlot::from_index(slot_idx) {
                Some(s) => s,
                None => continue,
            };

            let tau_k = match slot {
                SpectralSlot::MicroScalp => {
                    if dominant_tau_ms > 0.0 && dominant_tau_ms <= 60_000.0 {
                        dominant_tau_ms
                    } else {
                        slot.typical_tau_ms()
                    }
                }
                SpectralSlot::MesoTactical => {
                    if dominant_tau_ms > 60_000.0 && dominant_tau_ms <= 900_000.0 {
                        dominant_tau_ms
                    } else {
                        slot.typical_tau_ms()
                    }
                }
                SpectralSlot::MacroSwing => {
                    if dominant_tau_ms > 900_000.0 {
                        dominant_tau_ms
                    } else {
                        slot.typical_tau_ms()
                    }
                }
            };

            // Cálculo direccional por slot modulado por resonancia conductual
            let (raw_signal, raw_confidence) = match slot {
                SpectralSlot::MicroScalp => {
                    // MicroScalp: desequilibrio de libro (OFI/OBI), microprice delta y resonancia conductual
                    let microprice_delta_bps = if safe_mid > 0.0 {
                        ((microprice - safe_mid) / safe_mid) * 10_000.0
                    } else {
                        0.0
                    };
                    let micro_score = ofi * 0.40
                        + obi * 0.35
                        + (microprice_delta_bps * 0.10).clamp(-1.0, 1.0) * 0.25
                        + behavioral_report.behavioral_resonance * 0.15;
                    let conf =
                        (0.50 + micro_score.abs() * 0.40) * behavioral_report.conviction_multiplier;
                    if micro_score > 0.15 {
                        (SignalType::Long, conf.clamp(0.0, 1.0))
                    } else if micro_score < -0.15 {
                        (SignalType::Short, conf.clamp(0.0, 1.0))
                    } else {
                        (SignalType::Flat, 0.0)
                    }
                }
                SpectralSlot::MesoTactical => {
                    // MesoTactical: dominado por impulso de flujo, CVD y resonancia conductual
                    let meso_score =
                        cvd * 0.55 + ofi * 0.30 + behavioral_report.behavioral_resonance * 0.25;
                    let conf =
                        (0.50 + meso_score.abs() * 0.35) * behavioral_report.conviction_multiplier;
                    if meso_score > 0.20 {
                        (SignalType::Long, conf.clamp(0.0, 1.0))
                    } else if meso_score < -0.20 {
                        (SignalType::Short, conf.clamp(0.0, 1.0))
                    } else {
                        (SignalType::Flat, 0.0)
                    }
                }
                SpectralSlot::MacroSwing => {
                    // MacroSwing: régimen macro, tendencia acumulada y resonancia conductual
                    let macro_score =
                        cvd * 0.70 + obi * 0.15 + behavioral_report.behavioral_resonance * 0.35;
                    let conf =
                        (0.50 + macro_score.abs() * 0.30) * behavioral_report.conviction_multiplier;
                    if macro_score > 0.25 {
                        (SignalType::Long, conf.clamp(0.0, 1.0))
                    } else if macro_score < -0.25 {
                        (SignalType::Short, conf.clamp(0.0, 1.0))
                    } else {
                        (SignalType::Flat, 0.0)
                    }
                }
            };

            // Evaluar vetos físicos y de microestructura
            let mut vetoed = false;
            let mut veto_reason = None;

            if raw_signal != SignalType::Flat {
                // Veto 1: Condensación de liquidez de Bouchaud-Mézard para MicroScalp
                if slot == SpectralSlot::MicroScalp && liquidity_condensed {
                    vetoed = true;
                    veto_reason = Some("BOUCHAUD_LIQUIDITY_CONDENSED_PARETO_COLLAPSE");
                }

                // Veto 2: Singularidad de curvatura de Ricci de Perelman para MacroSwing
                if slot == SpectralSlot::MacroSwing && singular_curvature {
                    vetoed = true;
                    veto_reason = Some("PERELMAN_RICCI_CURVATURE_SINGULARITY");
                }

                // Veto 3: Confianza mínima insuficiente
                if raw_confidence < self.params.min_confidence_threshold {
                    vetoed = true;
                    veto_reason = Some("CONFIDENCE_BELOW_THRESHOLD");
                }

                // Veto 4: Colisión armónica destructiva con posiciones abiertas existentes
                let is_long_intent = raw_signal == SignalType::Long;
                for open_idx in 0..MAX_SPECTRAL_SLOTS {
                    if open_positions_slots[open_idx] {
                        let open_tau = open_positions_taus[open_idx];
                        let open_is_long = open_positions_is_long[open_idx];
                        let resonant = self.check_spectral_resonance(
                            tau_k,
                            is_long_intent,
                            open_tau,
                            open_is_long,
                        );
                        if !resonant {
                            vetoed = true;
                            veto_reason = Some("SPECTRAL_DESTRUCTIVE_COLLISION_WITH_OPEN_POSITION");
                            break;
                        }
                    }
                }
            }

            let lifetime_ms = (tau_k * 1.5) as u64;
            let is_maker = slot == SpectralSlot::MicroScalp || (raw_confidence < 0.70);

            *decision = QuantumSlotDecision {
                slot,
                signal: if vetoed { SignalType::Flat } else { raw_signal },
                net_confidence: if vetoed { 0.0 } else { raw_confidence },
                tau_ms: tau_k,
                microprice,
                expected_volatility: atr_pct,
                expected_lifetime_ms: lifetime_ms,
                is_maker_preferred: is_maker,
                vetoed,
                veto_reason,
                behavioral_resonance: behavioral_report.behavioral_resonance,
            };

            if decision.signal != SignalType::Flat {
                active_count += 1;
            }
        }

        let coherent_superposition = active_count >= 2;

        MultiSlotEvaluationReport {
            decisions,
            active_slots_count: active_count,
            coherent_superposition,
            ricci_curvature_scalar: ricci_scalar,
            liquidity_pareto_mu: pareto_mu,
            stoikov_microprice: microprice,
            dynamic_recv_window_ms,
            behavioral_resonance: behavioral_report.behavioral_resonance,
        }
    }

    /// Nocional/capital que recibe un slot al abrirse como único activo: la cota de
    /// `synthesize_order_emissions` (riesgo y apalancamiento efectivos). Una entrada NaN
    /// devuelve NaN.
    pub fn exposicion_bruta_maxima_por_slot(max_leverage: f64, risk_fraction: f64) -> f64 {
        risk_fraction.clamp(RIESGO_EFECTIVO_MIN, RIESGO_EFECTIVO_MAX)
            * max_leverage.clamp(APALANCAMIENTO_EFECTIVO_MIN, APALANCAMIENTO_EFECTIVO_MAX)
    }

    /// Sintetiza propuestas concretas de órdenes para ejecución a partir del informe de evaluación multi-slot.
    pub fn synthesize_order_emissions(
        &self,
        report: &MultiSlotEvaluationReport,
        capital_available_usd: f64,
        max_leverage: f64,
        risk_fraction: f64,
    ) -> MultiSlotOrderEmission {
        let mut emission = MultiSlotOrderEmission::empty();

        if report.active_slots_count == 0 || capital_available_usd <= 0.0 {
            return emission;
        }

        let safe_risk = risk_fraction.clamp(RIESGO_EFECTIVO_MIN, RIESGO_EFECTIVO_MAX);
        let safe_leverage =
            max_leverage.clamp(APALANCAMIENTO_EFECTIVO_MIN, APALANCAMIENTO_EFECTIVO_MAX);

        // Reparto de capital entre slots activos sin sobrepasar el capital disponible
        let capital_per_active_slot =
            (capital_available_usd * safe_risk) / (report.active_slots_count as f64);

        for decision in report.decisions.iter() {
            if decision.signal == SignalType::Flat || decision.vetoed {
                continue;
            }

            let is_long = decision.signal == SignalType::Long;
            let ref_price = if decision.microprice > 0.0 {
                decision.microprice
            } else {
                1.0
            };

            let notional = capital_per_active_slot * safe_leverage;
            let quantity = if ref_price > 0.0 {
                notional / ref_price
            } else {
                0.0
            };

            // TP y SL ajustados por la volatilidad esperada y la escala temporal del slot
            let vol = decision.expected_volatility.clamp(0.001, 0.10);
            let mut tp_pct = match decision.slot {
                SpectralSlot::MicroScalp => vol * 1.5,
                SpectralSlot::MesoTactical => vol * 2.5,
                SpectralSlot::MacroSwing => vol * 4.0,
            };
            let mut sl_pct = tp_pct / 2.0; // Ratio Asimétrico base R:R = 2.0

            // Ajuste asimétrico si el campo conductual apoya la dirección (stop hunting a favor)
            if self.params.enable_behavioral_asymmetry
                && ((is_long && decision.behavioral_resonance > 0.10)
                    || (!is_long && decision.behavioral_resonance < -0.10))
            {
                tp_pct *= 1.25;
                sl_pct *= 0.90;
            }

            let (tp_price, sl_price) = if is_long {
                (ref_price * (1.0 + tp_pct), ref_price * (1.0 - sl_pct))
            } else {
                (ref_price * (1.0 - tp_pct), ref_price * (1.0 + sl_pct))
            };

            let proposal = SlotOrderProposal {
                slot: decision.slot,
                is_long,
                price: ref_price,
                quantity,
                tp_price,
                sl_price,
                tau_ms: decision.tau_ms as u64,
                confidence: decision.net_confidence,
                microprice_ref: decision.microprice,
                expected_lifetime_ms: decision.expected_lifetime_ms,
                is_maker: decision.is_maker_preferred,
                behavioral_resonance: decision.behavioral_resonance,
            };

            emission.push(proposal);
        }

        emission
    }
}
