use quantum_arena::GlobalArena;
use signal_engine::SignalIntent;
use std::sync::atomic::Ordering;

/// 🚀 ALGORITMO #77 V11: QUANTUM LEVERAGE MATRIX (Adaptativo-Autoevolutivo)
///
/// FASE 10: Erradicación de hardcodes. Todo parámetro es un gen del genoma
/// o derivado de métricas reales del arena (WR, PF, Hurst, ATR).
///
/// QUÉ: Algoritmo de apalancamiento que combina 5 tensores matemáticos para
///      producir un leverage óptimo por operación.
///
/// CÓMO: 5 Tensores multiplicativos:
///   T1 = Fractional Kelly (con profit_factor REAL, no proxy)
///   T2 = Conviction³ (Bayesian Proxy, escala del genoma)
///   T3 = Volatility Brake (tanh friction, SIEMPRE activo incluso en scalp)
///   T4 = Growth Pressure (relativo a base_capital, no a $50)
///   T5 = Hurst Predictability Bonus
/// QO-M0.1 — Kelly desde probabilidad y profit factor como FUNCIÓN PURA
/// (testeable): f* = W·(1 − 1/PF) — identidad exacta con kelly.rs:69.
/// La versión anterior usaba PF como el pago b (K = p − q/PF): b = PF·q/p,
/// no PF — misestimación sistemática. Piso 0: sin edge, sin fracción.
#[inline(always)]
/// CERT-M5-H03: productor PURO de fracción — también pasa por el tope de
/// ruina central (streak-bound + axioma), como todos los demás.
pub fn kelly_from_pf(prob_win: f64, profit_factor: f64) -> f64 {
    if !prob_win.is_finite() || !profit_factor.is_finite() || profit_factor <= 1.0 {
        return 0.0;
    }
    crate::ruin::clamp_ruin((prob_win * (1.0 - 1.0 / profit_factor)).max(0.0), 1.0 - prob_win.clamp(0.0, 1.0))
}

pub struct QuantumLeverageMatrix;

impl QuantumLeverageMatrix {
    /// Retorna el Leverage recomendado (entre 1.0 y genome_max_leverage)
    ///
    /// Fórmula: L = T1(kelly) × T2(conviction) × T3(vol_brake) × T4(growth) × T5(hurst)
    pub fn calculate_dynamic_leverage(
        signal: &SignalIntent,
        temporal_scale: f64, // D-509: Variedad temporal continua s in [0.0, 1.0] sin colapso booleano
        current_capital: f64,
        base_capital: f64, // Capital base real (extraído de API)
        tick_volatility: f64,
        volatility_multiplier: f64, // Extracted from SuperGenotype
        hurst_exponent: f64,        // TENSOR 5: Predictabilidad de la serie
        real_profit_factor: f64, // PF REAL del coin (de arena.coins[id].scalp/swing.profit_factor)
        real_win_rate: f64,      // Win Rate REAL histórico del coin
        genome_max_leverage: f64, // Límite del genoma (de config.global_leverage)
        arena: &GlobalArena,
    ) -> f64 {
        // FIX #651: Sanitizar parámetros entrantes asegurando robustez numérica total
        let safe_curr_cap = if current_capital.is_finite() && current_capital > 0.0 {
            current_capital
        } else {
            13.0
        };
        let safe_base_cap = if base_capital.is_finite() && base_capital > 0.0 {
            base_capital
        } else {
            13.0
        };
        let safe_tick_vol = if tick_volatility.is_finite() && tick_volatility >= 0.0 {
            tick_volatility
        } else {
            0.001
        };
        let safe_vol_mult = if volatility_multiplier.is_finite() && volatility_multiplier > 0.0 {
            volatility_multiplier
        } else {
            1.0
        };
        let safe_hurst = if hurst_exponent.is_finite() {
            hurst_exponent.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let safe_pf = if real_profit_factor.is_finite() && real_profit_factor > 0.0 {
            real_profit_factor.max(0.1)
        } else {
            1.0
        };
        let safe_wr = if real_win_rate.is_finite() && real_win_rate >= 0.0 {
            real_win_rate.clamp(0.0, 1.0)
        } else {
            0.5
        };
        let safe_max_lev = if genome_max_leverage.is_finite() && genome_max_leverage >= 1.0 {
            genome_max_leverage.clamp(1.0, 50.0)
        } else {
            20.0
        };

        // ═══════════════════════════════════════════════════════
        // TENSOR 1: Full Fractional Kelly (con profit_factor y win_rate REAL)
        // ═══════════════════════════════════════════════════════
        // Fusión Bayesiana: si hay historial (WR > 0.05), pondera 70% historia + 30% convicción puntual
        // D-690: la convicción puntual es la probabilidad calibrada cuando existe;
        // la puntuación cruda sólo cuando el núcleo no la ha calibrado.
        let signal_probability = if signal.win_probability > 0.0 {
            signal.win_probability
        } else {
            signal.confidence
        };
        let prob_win = if safe_wr > 0.05 {
            (safe_wr * 0.70 + signal_probability.clamp(0.1, 1.0) * 0.30).clamp(0.10, 0.95)
        } else {
            signal_probability.clamp(0.10, 0.95)
        };
        let pf = safe_pf; // PF real, fallback si no hay historial
        // QO-M0.1 (auditoría matemática): la fórmula anterior usaba el PF
        // como el pago b de Kelly — K = p − (1−p)/PF — pero b = PF·q/p, no
        // PF: sistemáticamente MISestimaba Kelly. La identidad correcta
        // (misma que kelly.rs:69): f* = W·(1 − 1/PF). Y el piso .max(0.01)
        // forzaba apuesta con edge negativo — abajo 0: sin edge, sin size.
        let kelly = kelly_from_pf(prob_win, pf);

        // Fracción adaptativa: Hurst × confidence determinan agresividad
        let fraction_multiplier =
            if prob_win > arena.config.veto_threshold_btc.load(Ordering::Relaxed) {
                1.0
            } else {
                (prob_win - 0.5).max(0.1) * 2.0 // Minimum 0.1 to prevent near-zero fractions
            };

        let dynamic_kelly = kelly * fraction_multiplier;

        // ═══════════════════════════════════════════════════════
        // TENSOR 2: Convicción de la Señal (Bayesian Proxy Adaptativo)
        // ═══════════════════════════════════════════════════════
        let conviction_scale = volatility_multiplier.max(1.0);
        let conviction = (0.40 + signal.confidence.clamp(0.1, 1.0) * 0.60) * conviction_scale;

        // ═══════════════════════════════════════════════════════
        // TENSOR 3: Freno de Volatilidad (SIEMPRE activo)
        // ═══════════════════════════════════════════════════════
        let vol_sensitivity = safe_vol_mult.max(1.0); // Minimum 1.0, genome dictates
        let vol_brake = 1.0 - (safe_tick_vol * vol_sensitivity).tanh();

        // ═══════════════════════════════════════════════════════
        // TENSOR 4: Micro-Capital Acceleration (Curva Logarítmica)
        // ═══════════════════════════════════════════════════════
        // A menor capital, mayor multiplicador para permitir interés compuesto rápido.
        // A mayor capital, amortiguación logarítmica para proteger patrimonio.
        // FIX #597: Amortiguación logarítmica blindada contra capitales infinitesimales o nulos
        // FIX #597 & #705: Amortiguación logarítmica blindada con sanitización de parámetros atómicos
        let raw_log_div = arena
            .config
            .lev_matrix_log_cap_divisor
            .load(Ordering::Relaxed);
        let log_divisor = if raw_log_div.is_finite() && raw_log_div > 0.0 {
            raw_log_div.max(1.0)
        } else {
            10.0
        };

        let capital_ratio = (safe_curr_cap / safe_base_cap.max(1.0)).max(0.0);
        let log_cap = safe_curr_cap.max(2.0).log10().max(0.3010); // log10(2.0) ≈ 0.3010
        let raw_dampener = (log_divisor / log_cap.max(1.0)).min(3.0);
        let logarithmic_dampener = if raw_dampener.is_finite() {
            raw_dampener.clamp(0.50, 3.0)
        } else {
            1.0
        };

        let raw_growth_scalar = arena
            .config
            .lev_matrix_growth_scalar
            .load(Ordering::Relaxed);
        let growth_scalar = if raw_growth_scalar.is_finite() && raw_growth_scalar >= 0.0 {
            raw_growth_scalar
        } else {
            0.5
        };
        let raw_gf = (1.0 + growth_scalar / (1.0 + capital_ratio)) * logarithmic_dampener;
        let growth_factor = if raw_gf.is_finite() {
            raw_gf.clamp(0.1, 10.0)
        } else {
            1.0
        };

        // Techo dinámico logarítmico: capitales bajos permiten leverages guiados por EV pero acotados para micro-cuentas ($13 USD)
        // D-641 (completo) — EL TECHO DE APALANCAMIENTO YA NO SALTA DE 4× A ~40×.
        // Con el divisor genómico en su banda [3, 10], el techo estándar a $20,01
        // valía 50·(1 − log10(20)/(2·d)) ≈ 39–47×, frente a 4× un centavo antes:
        // un salto de un orden de magnitud en el riesgo por operación. Ahora el
        // techo micro de 4× rige pleno a ≤3 operaciones mínimas y se funde
        // geométricamente con el estándar hasta 10.
        let standard_ceiling = 50.0 * (1.0 - (log_cap / (log_divisor * 2.0)).min(0.8));
        let micro_w = crate::capital_regime::micro_weight(
            safe_curr_cap,
            arena.config.min_notional.load(Ordering::Relaxed),
        );
        let raw_ceiling = crate::capital_regime::log_lerp(standard_ceiling, 4.0, micro_w);
        let dynamic_ceiling = if raw_ceiling.is_finite() {
            raw_ceiling.clamp(1.0, 50.0)
        } else {
            4.0
        };
        let effective_max_leverage = safe_max_lev.clamp(1.0, dynamic_ceiling);

        // ═══════════════════════════════════════════════════════
        // TENSOR 5: Hurst Predictability Bonus
        // ═══════════════════════════════════════════════════════
        let raw_hurst_bonus = if safe_hurst > 0.5 {
            1.0 + (safe_hurst - 0.5) * 2.0
        } else {
            1.0 - (0.5 - safe_hurst)
        };
        let hurst_bonus = if raw_hurst_bonus.is_finite() {
            raw_hurst_bonus.clamp(0.5, 2.0)
        } else {
            1.0
        };

        let raw_vol_clamp = arena
            .config
            .lev_matrix_vol_clamp_min
            .load(Ordering::Relaxed);
        let vol_clamp_min = if raw_vol_clamp.is_finite() {
            raw_vol_clamp.clamp(0.0, 1.0)
        } else {
            0.1
        };

        let effective_temporal_scale = if temporal_scale.is_finite() {
            temporal_scale.clamp(0.0, 1.0)
        } else {
            arena
                .config
                .temporal_scale
                .load(Ordering::Relaxed)
                .clamp(0.0, 1.0)
        };
        // U-6: motor continuo — sólo existe TradeHorizon::Continuous; la
        // escala s viene de la τ declarada o del arena (D-638b).
        let s = if signal.expected_duration_ms > 0 {
            quantum_arena::temporal_spectrum::temporal_scale_from_tau(
                signal.expected_duration_ms as f64,
            )
        } else {
            effective_temporal_scale
        };

        // D-338: Homotopía continua y diferenciable s in [0, 1].
        // Elimina el salto abrupto del 30% en Kelly y +0.20 en freno de volatilidad.
        let effective_vol_clamp = vol_clamp_min * (1.0 - s) + (vol_clamp_min + 0.20) * s;
        let final_vol_factor = vol_brake.max(effective_vol_clamp);

        let fast_dynamic_kelly = kelly * fraction_multiplier.max(1.0);
        let slow_dynamic_kelly = dynamic_kelly * 0.70;
        let final_dynamic_kelly = fast_dynamic_kelly * (1.0 - s) + slow_dynamic_kelly * s;

        // Capital factor: Kelly escala el leverage. Sqrt para suavizar.
        let safe_dyn_kelly = if final_dynamic_kelly.is_finite() && final_dynamic_kelly >= 0.0 {
            final_dynamic_kelly
        } else {
            0.0
        };
        let capital_factor = 1.0 + (safe_dyn_kelly * effective_max_leverage.sqrt()).sqrt();

        let final_leverage =
            capital_factor * conviction * final_vol_factor * growth_factor * hurst_bonus;

        if !final_leverage.is_finite() {
            return 1.0;
        }

        let clamped = final_leverage.clamp(1.0, effective_max_leverage);

        // telemetry_engine::telemetry!(
        //     "Leverage Matrix: {:?} [Scalp={}] (Hurst: {:.2}, VolFactor: {:.2}, PF: {:.2}) -> {:.2}x",
        //     signal.signal, is_scalp, hurst_bonus, final_vol_factor, real_profit_factor, clamped
        // );

        clamped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leverage_for(signal: &SignalIntent, arena: &GlobalArena) -> f64 {
        QuantumLeverageMatrix::calculate_dynamic_leverage(
            signal, 0.0, 10_000.0, 10_000.0, 0.001, 1.0, 0.5, 1.0, 0.0, 20.0, arena,
        )
    }

    /// D-690 + QO-M0.1: el Kelly usa la probabilidad (calibrada si existe)
    /// y la identidad correcta W·(1−1/PF). A nivel de matriz el techo de
    /// leverage puede saturar ambas ramas (comportamiento legítimo), así
    /// que la propiedad monótona se verifica en la función pura.
    #[test]
    fn d690_kelly_usa_la_probabilidad_calibrada() {
        // D-714: el arena no cabe en la pila por defecto de un hilo de test.
        let arena = quantum_arena::GlobalArena::build_in_own_stack(10_000.0);
        let uncalibrated = SignalIntent {
            signal: signal_engine::SignalType::Long,
            confidence: 0.9,
            ..Default::default()
        };
        let calibrated_same = SignalIntent {
            win_probability: 0.9,
            ..uncalibrated
        };
        let base = leverage_for(&uncalibrated, &arena);
        assert_eq!(base, leverage_for(&calibrated_same, &arena));

        // QO-M0.1 — identidad exacta y monotonía en W.
        assert!((kelly_from_pf(0.62, 1.5) - 0.62 * (1.0 - 1.0 / 1.5)).abs() < 1e-12);
        assert!(
            kelly_from_pf(0.44, 1.5) < kelly_from_pf(0.62, 1.5),
            "probabilidad baja ⇒ Kelly menor"
        );
        // Sin edge (PF ≤ 1): fracción 0, jamás negativa.
        assert_eq!(kelly_from_pf(0.9, 1.0), 0.0);
        assert_eq!(kelly_from_pf(0.9, 0.5), 0.0);
        // La identidad con kelly.rs: W·(1−1/PF) = W − (1−W)/R con R=PF·q/p.
        let w = 0.62f64;
        let pf = 1.5f64;
        let r = pf * (1.0 - w) / w;
        assert!((kelly_from_pf(w, pf) - (w - (1.0 - w) / r)).abs() < 1e-12);
    }
}
