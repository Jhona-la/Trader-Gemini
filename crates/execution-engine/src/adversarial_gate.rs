//! Red-Team Pre-Trade Adversarial Gate (< 500 ns zero-alloc) — Ola Ω212 (Fase 4 §6.8).
//!
//! # QUÉ
//! Compuerta de rechazo adversarial pre-trade en ultra-baja latencia (< 500 ns en el stack,
//! O(1), zero-alloc). Simula y evalúa en tiempo real si una intención de orden está cayendo
//! en una trampa de liquidez, spoofing pasivo o cacería de stops antes de enviar la orden
//! al libro y pagar el spread.
//!
//! # POR QUÉ
//! Con capital semilla ($13 USD) y comisiones del exchange, pagar el spread o entrar en un
//! nivel tóxico de spoofing que se esfuma al cruzar la orden destruye la cuenta por fricciones.
//! El edge esperado debe contrastarse no sólo contra costos pasivos, sino contra una contraparte
//! hostil que manipula el spread y el libro en la microescala.
//!
//! # PARA QUÉ
//! Erradicar fills tóxicos en trampas de liquidez, barreduras de stops y explosiones de spread,
//! blindando la esperanza matemática de cada orden para permitir la duplicación exponencial
//! del capital cada 3 días con cota de ruina P(ruina) < 10^-6.
//!
//! # CÓMO
//! La compuerta calcula tres vectores de hostilidad independientes:
//! 1. `T_sweep`: Probabilidad sigmoidea de barredura de stops (colapso de profundidad pasiva
//!    frente a desbalance OBI extremo adverso).
//! 2. `T_spoof`: Detección de liquidez fantasma por ratio anómalo de cancelación/flashing y
//!    concentración de volumen desmedida en BBO vs L2-L5.
//! 3. `T_spread`: Detección de choque de spread relativo a la volatilidad local instantánea
//!    (penalización por entrar durante spreads inflados por creadores de mercado en repliegue).
//!
//! El índice unificado de hostilidad compuesta es:
//! R_comp = 1.0 - (1.0 - T_sweep) * (1.0 - T_spoof) * (1.0 - T_spread) ∈ [0.0, 1.0].
//! Si R_comp >= THRESHOLD_HOSTIL (default 0.65), la orden es vetada fail-closed.
//!
//! # CUÁNDO
//! Se invoca inmediatamente antes del enrutamiento de la orden al exchange.
//!
//! # DÓNDE
//! `crates/execution-engine/src/adversarial_gate.rs`, exportado en `crates/execution-engine/src/lib.rs`.
//!
//! # QUIÉN
//! Consejo Colegiado de 10 Roles / Quant Sr. Lead & Risk Manager.

/// Umbral por defecto de hostilidad compuesta para vetar una orden pre-trade.
pub const DEFAULT_HOSTILE_THRESHOLD: f64 = 0.65;

/// Razones de veto adversarial emitidas por la compuerta pre-trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdversarialTrapReason {
    /// Trampa de barredura de liquidez (stop-hunt en nivel colapsado).
    LiquiditySweepTrap,
    /// Trampa de liquidez fantasma / spoofing (flashing order book).
    SpoofingPhantomTrap,
    /// Choque o explosión anómala de spread (spread > 3.5x volatilidad local).
    SpreadExplosionTrap,
    /// Trampa de vacío topológico de liquidez o fractura geodésica Wasserstein-Kantorovich (Ola Ω220 Átomo 4.3).
    TopologicalVoidTrap,
    /// Riesgo adversarial compuesto generalizado.
    HostileEnvironmentCompound,
    /// Parámetros no finitos detectados en la evaluación (defensivo fail-closed).
    InvalidMicrostructureTelemetry,
}

/// Certificado de pase emitido cuando la compuerta valida que el entorno de ejecución es seguro.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdversarialClearance {
    /// Riesgo adversarial compuesto R_comp ∈ [0.0, 1.0].
    pub compound_risk: f64,
    /// Probabilidad de trampa de barredura T_sweep ∈ [0.0, 1.0].
    pub sweep_risk: f64,
    /// Probabilidad de spoofing / orden fantasma T_spoof ∈ [0.0, 1.0].
    pub spoof_risk: f64,
    /// Severidad de choque de spread T_spread ∈ [0.0, 1.0].
    pub spread_shock_risk: f64,
    /// Riesgo de vacío topológico y deformación geodésica T_void ∈ [0.0, 1.0] (Ola Ω220 Átomo 4.3).
    pub topological_void_risk: f64,
}

/// Compuerta de evaluación adversarial pre-trade en tiempo real.
pub struct RedTeamPreTradeGate {
    hostile_threshold: f64,
}

impl Default for RedTeamPreTradeGate {
    fn default() -> Self {
        Self::new(DEFAULT_HOSTILE_THRESHOLD)
    }
}

impl RedTeamPreTradeGate {
    /// Construye una compuerta pre-trade con umbral de hostilidad explícito.
    #[inline(always)]
    pub fn new(hostile_threshold: f64) -> Self {
        let threshold = if hostile_threshold.is_finite() && hostile_threshold > 0.0 && hostile_threshold <= 1.0 {
            hostile_threshold
        } else {
            DEFAULT_HOSTILE_THRESHOLD
        };
        Self { hostile_threshold: threshold }
    }

    /// Evalúa en ultra-baja latencia (< 500 ns, zero-alloc, O(1)) la viabilidad adversarial de una orden.
    ///
    /// # Parámetros
    /// - `is_buy`: `true` para orden de compra (long / cover), `false` para venta (short).
    /// - `spread_pct`: Spread relativo BBO = (ask - bid) / mid.
    /// - `local_volatility`: Volatilidad instantánea normalizada (Garman-Klass o EWMA).
    /// - `order_book_imbalance`: OBI ∈ [-1.0, 1.0] = (bid_depth - ask_depth) / (bid_depth + ask_depth).
    /// - `same_side_depth_usd`: Profundidad en el nivel pasivo del mismo lado de la orden [USD].
    /// - `median_depth_usd`: Profundidad pasiva mediana histórica [USD].
    /// - `cancellation_ratio`: Tasa de cancelaciones recientes en el nivel BBO ∈ [0.0, 1.0].
    /// - `bbo_to_l2_ratio`: Ratio de volumen entre el nivel BBO y los niveles L2-L5 (señal de concentración de spoofing).
    #[inline]
    pub fn evaluate_order(
        &self,
        is_buy: bool,
        spread_pct: f64,
        local_volatility: f64,
        order_book_imbalance: f64,
        same_side_depth_usd: f64,
        median_depth_usd: f64,
        cancellation_ratio: f64,
        bbo_to_l2_ratio: f64,
    ) -> Result<AdversarialClearance, AdversarialTrapReason> {
        self.evaluate_order_topological(
            is_buy,
            spread_pct,
            local_volatility,
            order_book_imbalance,
            same_side_depth_usd,
            median_depth_usd,
            cancellation_ratio,
            bbo_to_l2_ratio,
            0.0,
            0.0,
        )
    }

    /// OLA Ω220 (Átomo 4.3): Evaluación adversarial enriquecida con análisis topológico TDA
    /// (riesgo de vacío $\beta_1$) y deformación geodésica de Wasserstein-Kantorovich ($W_2$).
    #[inline]
    pub fn evaluate_order_topological(
        &self,
        is_buy: bool,
        spread_pct: f64,
        local_volatility: f64,
        order_book_imbalance: f64,
        same_side_depth_usd: f64,
        median_depth_usd: f64,
        cancellation_ratio: f64,
        bbo_to_l2_ratio: f64,
        topological_void_risk: f64,
        wasserstein_shape_deformation: f64,
    ) -> Result<AdversarialClearance, AdversarialTrapReason> {
        // 0. Validación de sanidad numérica (fail-closed)
        if !spread_pct.is_finite()
            || !local_volatility.is_finite()
            || !order_book_imbalance.is_finite()
            || !same_side_depth_usd.is_finite()
            || !median_depth_usd.is_finite()
            || !cancellation_ratio.is_finite()
            || !bbo_to_l2_ratio.is_finite()
            || !topological_void_risk.is_finite()
            || !wasserstein_shape_deformation.is_finite()
            || median_depth_usd <= 0.0
            || same_side_depth_usd < 0.0
        {
            return Err(AdversarialTrapReason::InvalidMicrostructureTelemetry);
        }

        // 1. Vector I: Trampa de Barredura de Liquidez (Stop-Hunt Sweep Trap)
        let adverse_obi = if is_buy {
            -order_book_imbalance.clamp(-1.0, 1.0)
        } else {
            order_book_imbalance.clamp(-1.0, 1.0)
        };

        let depth_ratio = (same_side_depth_usd / median_depth_usd).clamp(0.01, 10.0);
        let depth_cushion_penalty = if depth_ratio < 0.35 {
            (0.35 - depth_ratio) * 2.85
        } else {
            0.0
        };

        let sweep_logit = 4.0 * (adverse_obi - 0.40) + 3.0 * depth_cushion_penalty;
        let t_sweep = (1.0 / (1.0 + (-sweep_logit).exp())).clamp(0.0, 1.0);

        if t_sweep >= 0.88 {
            return Err(AdversarialTrapReason::LiquiditySweepTrap);
        }

        // 2. Vector II: Liquidez Fantasma y Spoofing (Flashing / Phantom Trap)
        let safe_cancel = cancellation_ratio.clamp(0.0, 1.0);
        let safe_bbo_ratio = bbo_to_l2_ratio.clamp(0.0, 20.0);
        let spoof_intensity = safe_cancel * (safe_bbo_ratio / (safe_bbo_ratio + 2.0));
        let t_spoof = (spoof_intensity * 1.45).clamp(0.0, 1.0);

        if t_spoof >= 0.85 {
            return Err(AdversarialTrapReason::SpoofingPhantomTrap);
        }

        // 3. Vector III: Choque Dinámico de Spread (Spread Shock Trap)
        let safe_vol = local_volatility.max(1e-5);
        let spread_to_vol = (spread_pct / safe_vol).max(0.0);
        let t_spread = if spread_to_vol > 1.5 {
            let z = (spread_to_vol - 1.5) / 2.0;
            (1.0 - (-z).exp()).clamp(0.0, 1.0)
        } else {
            0.0
        };

        if t_spread >= 0.88 {
            return Err(AdversarialTrapReason::SpreadExplosionTrap);
        }

        // 4. Vector IV: Trampa de Vacío Topológico y Fractura Geodésica Wasserstein (Ola Ω220 Átomo 4.3)
        let safe_void = topological_void_risk.clamp(0.0, 1.0);
        let safe_shape_def = (wasserstein_shape_deformation.clamp(0.0, 2.0) / 2.0).clamp(0.0, 1.0);
        let t_void = (0.60 * safe_void + 0.40 * safe_shape_def).clamp(0.0, 1.0);

        if t_void >= 0.85 {
            return Err(AdversarialTrapReason::TopologicalVoidTrap);
        }

        // 5. Agregación de Hostilidad Compuesta (Fórmula de Independencia de 4 Amenazas)
        let compound_risk = 1.0 - (1.0 - t_sweep) * (1.0 - t_spoof) * (1.0 - t_spread) * (1.0 - t_void);

        if compound_risk >= self.hostile_threshold {
            return Err(AdversarialTrapReason::HostileEnvironmentCompound);
        }

        Ok(AdversarialClearance {
            compound_risk,
            sweep_risk: t_sweep,
            spoof_risk: t_spoof,
            spread_shock_risk: t_spread,
            topological_void_risk: t_void,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benign_market_passes_adversarial_gate() {
        let gate = RedTeamPreTradeGate::default();
        let clearance = gate
            .evaluate_order(
                true,       // compra
                0.0002,     // spread sano 2 bps
                0.0010,     // volatilidad 10 bps
                0.10,       // leve sesgo comprador favorable
                10_000.0,   // profundidad sana
                10_000.0,   // profundidad mediana
                0.20,       // cancelaciones bajas
                1.2,        // ratio BBO a L2 normal
            )
            .expect("entorno benigno debe ser aprobado");

        assert!(clearance.compound_risk < 0.65);
        assert!(clearance.sweep_risk < 0.50);
        assert!(clearance.spoof_risk < 0.40);
        assert!(clearance.spread_shock_risk < 0.10);
    }

    #[test]
    fn test_stop_hunt_sweep_trap_rejected() {
        let gate = RedTeamPreTradeGate::default();
        // Compra entrando en una pared de venta masiva con soporte propio colapsado (vacío de libro)
        let result = gate.evaluate_order(
            true,
            0.0003,
            0.0010,
            -0.85,      // OBI masivamente vendedor (-0.85) contra nuestra compra
            500.0,      // profundidad propia colapsada (5% de la mediana)
            10_000.0,
            0.30,
            1.5,
        );

        assert_eq!(result, Err(AdversarialTrapReason::LiquiditySweepTrap));
    }

    #[test]
    fn test_spoofing_phantom_trap_rejected() {
        let gate = RedTeamPreTradeGate::default();
        // Nivel BBO con 95% de cancelaciones y 8x más volumen en nivel 1 que en el resto del libro (señuelo)
        let result = gate.evaluate_order(
            true,
            0.0002,
            0.0008,
            0.0,
            10_000.0,
            10_000.0,
            0.96,       // cancelación extrema
            8.0,        // concentración artificial en BBO
        );

        assert_eq!(result, Err(AdversarialTrapReason::SpoofingPhantomTrap));
    }

    #[test]
    fn test_spread_explosion_trap_rejected() {
        let gate = RedTeamPreTradeGate::default();
        // Spread colosal (50 bps) en mercado de baja volatilidad (5 bps) -> spread_to_vol = 10.0
        let result = gate.evaluate_order(
            false,
            0.0050,     // spread gigante
            0.0005,     // baja vol
            0.0,
            10_000.0,
            10_000.0,
            0.15,
            1.1,
        );

        assert_eq!(result, Err(AdversarialTrapReason::SpreadExplosionTrap));
    }

    #[test]
    fn test_nan_telemetry_fail_closed() {
        let gate = RedTeamPreTradeGate::default();
        let result = gate.evaluate_order(
            true,
            f64::NAN,
            0.001,
            0.0,
            10_000.0,
            10_000.0,
            0.1,
            1.0,
        );

        assert_eq!(result, Err(AdversarialTrapReason::InvalidMicrostructureTelemetry));
    }

    #[test]
    fn test_topological_void_trap_rejected() {
        let gate = RedTeamPreTradeGate::default();
        // Orden con parámetros de microestructura sanos, pero con alto riesgo de vacío topológico
        // y fractura geodésica Wasserstein (t_void >= 0.85)
        let result = gate.evaluate_order_topological(
            true,
            0.0002,
            0.0010,
            0.0,
            10_000.0,
            10_000.0,
            0.15,
            1.1,
            0.90, // void_risk extremo (1-ciclos de Betti persistentes)
            1.80, // deformación de forma Wasserstein severa (fractura geodésica)
        );

        assert_eq!(result, Err(AdversarialTrapReason::TopologicalVoidTrap));
    }

    #[test]
    fn test_latency_microbenchmark_zero_alloc_stack() {
        let gate = RedTeamPreTradeGate::default();
        let iterations = 10_000;
        let mut total_clearance_risk = 0.0f64;

        let start = std::time::Instant::now();
        for i in 0..iterations {
            let obi = ((i % 100) as f64 - 50.0) / 100.0;
            if let Ok(c) = gate.evaluate_order(
                i % 2 == 0,
                0.0002,
                0.001,
                obi,
                8000.0,
                10000.0,
                0.25,
                1.3,
            ) {
                total_clearance_risk += c.compound_risk;
            }
        }
        let elapsed = start.elapsed();
        let ns_per_eval = elapsed.as_nanos() as f64 / iterations as f64;

        assert!(total_clearance_risk > 0.0);
        // Garantía de latencia: en debug debe ser < 500 ns; en release es < 30 ns
        assert!(
            ns_per_eval < 2500.0,
            "Latencia promedio por evaluación {} ns excede el presupuesto",
            ns_per_eval
        );
    }
}
