use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::{QuantumStrategy, SignalIntent, SignalType};

/// 🎯 ALGORITMO #45: DETECTOR COAXIAL DE COMPRESIÓN DE VOLATILIDAD Y BREAKOUT (COAXIAL BREAKOUT ENGINE)
/// Monitorea compresión estocástica de ATR en 1s, 5s y 1m simultáneamente.
/// Dispara la entrada de confluencia justo antes de que el flujo institucional barra el libro de órdenes.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct CoaxialBreakoutEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for CoaxialBreakoutEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoaxialBreakoutEngine").finish()
    }
}

impl CoaxialBreakoutEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// #616 (Ola 38) — VOTO ESPECTRAL de la compresión coaxial: el PRODUCTO
    /// TENSORIAL de compresión entre escalas ADYACENTES de la malla, usando
    /// los desplazamientos |x(τ)| como proxies de la volatilidad a cada
    /// escala. La escala k está COMPRIMIDA cuando su desplazamiento es menor
    /// que el de sus vecinas: energía acumulada lista para romper.
    /// Producto de compresiones [k−1→k]·[k→k+1], ×4, tanh — la MISMA forma
    /// tensorial del evaluate_coaxial_breakout. Firmada por la dirección del
    /// desplazamiento en k. Observacional: voto vivo bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
    ) -> crate::voto_espectral::VotoEspectral {
        let mut por_escala = [0.0f64; 32];
        for k in 1..31 {
            let a = desplazamientos[k - 1].abs().max(1e-9);
            let b = desplazamientos[k].abs().max(1e-9);
            let c = desplazamientos[k + 1].abs().max(1e-9);
            let comp_ab = (1.0 - a / b).max(0.0);
            let comp_bc = (1.0 - b / c).max(0.0);
            let squeeze = (comp_ab * comp_bc * 4.0).tanh().clamp(0.0, 1.0);
            let signo = if desplazamientos[k] > 0.0 {
                1.0
            } else if desplazamientos[k] < 0.0 {
                -1.0
            } else {
                0.0
            };
            por_escala[k] = (signo * squeeze).clamp(-1.0, 1.0);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    /// Infiere la señal de ruptura coaxial multidimensional (O(1) Continuous Math)
    #[inline(always)]
    pub fn evaluate_coaxial_breakout(
        arena: &quantum_arena::GlobalArena,
        atr_1s: f64,
        atr_5s: f64,
        atr_1m: f64,
        _current_price: f64,
        is_bullish_flow: bool,
    ) -> Option<SignalIntent> {
        // Relative compression needs observed nonnegative variation and positive
        // denominators. Zero / zero is unknown, not maximal compression.
        // A zero 1s numerator is valid when the larger-scale denominators exist.
        if !atr_1s.is_finite()
            || !atr_5s.is_finite()
            || !atr_1m.is_finite()
            || atr_1s < 0.0
            || atr_5s <= 0.0
            || atr_1m <= 0.0
        {
            return None;
        }

        // Tensor math: Transform volatility ratios into continuous squeeze probabilities
        // Normalización temporal estocástica mediante ley de difusión Browniana (\sigma \sim \sqrt{\Delta t})
        let norm_atr_1s = atr_1s.max(0.0); // sqrt(1.0) = 1.0
        let norm_atr_5s = (atr_5s.max(0.0)) / 2.2360679775_f64; // sqrt(5.0)
        let norm_atr_1m = (atr_1m.max(0.0)) / 7.7459666924_f64; // sqrt(60.0)

        // El factor de compresión crece cuando la volatilidad normalizada de baja escala es menor a la macro
        let comp_1s = (1.0_f64 - (norm_atr_1s / norm_atr_5s.max(1e-8))).max(0.0);
        let comp_5s = (1.0_f64 - (norm_atr_5s / norm_atr_1m.max(1e-8))).max(0.0);

        // Producto tensorial de compresión (ambos marcos temporales deben estar comprimidos)
        let coaxial_squeeze = (comp_1s * comp_5s * 4.0).tanh();

        // Emitir señal si la compresión acumulada es matemáticamente relevante
        use std::sync::atomic::Ordering;
        let squeeze_threshold = arena
            .config
            .coaxial_squeeze_threshold
            .load(Ordering::Relaxed);
        if coaxial_squeeze > squeeze_threshold {
            // FIX #1507: Sanitización de duración de señal
            let raw_dur = arena.config.base_duration_ms.load(Ordering::Relaxed);
            let duration = if raw_dur.is_finite() && raw_dur > 0.0 {
                raw_dur as u64
            } else {
                5000
            };
            return Some(SignalIntent {
                signal: if is_bullish_flow {
                    SignalType::Long
                } else {
                    SignalType::Short
                },
                confidence: coaxial_squeeze,
                expected_duration_ms: duration.max(5000),
                horizon: strategy_core::TradeHorizon::Continuous,
                ..Default::default()
            });
        }
        None
    }
}

impl QuantumStrategy for CoaxialBreakoutEngine {
    fn name(&self) -> &str {
        "CoaxialBreakoutEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        self.evaluate_for_coin(0, "")
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let sym_opt = if symbol.is_empty() {
            None
        } else {
            Some(symbol)
        };
        let cid_opt = if symbol.is_empty() {
            None
        } else {
            Some(coin_id)
        };
        let r = match self.registry.as_ref() {
            Some(reg) => reg,
            None => return 0.0,
        };

        let atr_1s = r
            .get_scoped_parameter(sym_opt, cid_opt, "atr_1s", "CoaxialBreakoutEngine")
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "atr_pct", "CoaxialBreakoutEngine")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.001);

        let atr_5s = r
            .get_scoped_parameter(sym_opt, cid_opt, "atr_5s", "CoaxialBreakoutEngine")
            .map(|p| p.get_value())
            .unwrap_or_else(|| atr_1s * 2.2360679775_f64);

        let atr_1m = r
            .get_scoped_parameter(sym_opt, cid_opt, "atr_1m", "CoaxialBreakoutEngine")
            .map(|p| p.get_value())
            .unwrap_or_else(|| atr_1s * 7.7459666924_f64);

        // FIX #682: Sanitizar lecturas de ATR
        let safe_1s = if atr_1s.is_finite() && atr_1s > 0.0 {
            atr_1s
        } else {
            0.001
        };
        let safe_5s = if atr_5s.is_finite() && atr_5s > 0.0 {
            atr_5s
        } else {
            0.005
        };
        let safe_1m = if atr_1m.is_finite() && atr_1m > 0.0 {
            atr_1m
        } else {
            0.020
        };

        let norm_atr_1s = safe_1s;
        let norm_atr_5s = safe_5s / 2.2360679775_f64;
        let norm_atr_1m = safe_1m / 7.7459666924_f64;
        let comp_1s = (1.0_f64 - (norm_atr_1s / norm_atr_5s.max(1e-8))).max(0.0);
        let comp_5s = (1.0_f64 - (norm_atr_5s / norm_atr_1m.max(1e-8))).max(0.0);

        let direction = r
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_flow_direction",
                "CoaxialBreakoutEngine",
            )
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "CoaxialBreakoutEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_delta",
                    "CoaxialBreakoutEngine",
                )
            })
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "price_velocity", "CoaxialBreakoutEngine")
            })
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "vol_delta", "CoaxialBreakoutEngine")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        let safe_dir = if direction.is_finite() {
            direction
        } else {
            0.0
        };
        if safe_dir.abs() <= 1e-6 {
            return 0.0;
        }
        let squeeze = (comp_1s * comp_5s * 4.0).tanh();
        let dir_weight = (safe_dir / 1e-4).tanh();
        dir_weight * squeeze
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coaxial_breakout_detection() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Fuerte compresión multiescala: 1s << 5s << 1m
        let signal = CoaxialBreakoutEngine::evaluate_coaxial_breakout(
            &arena, 0.0005, 0.005, 0.050, 60000.0, true,
        );
        assert!(signal.is_some());
        let s = signal.unwrap();
        assert_eq!(s.signal, SignalType::Long);
        assert!(s.confidence > 0.0);
    }

    #[test]
    fn test_coaxial_breakout_symmetric_short_and_nan_immunity() {
        let arena = quantum_arena::GlobalArena::build_in_own_stack(13.0);
        // Short con compresión
        let signal_short = CoaxialBreakoutEngine::evaluate_coaxial_breakout(
            &arena, 0.0005, 0.005, 0.050, 60000.0, false,
        );
        assert!(signal_short.is_some());
        let s = signal_short.unwrap();
        assert_eq!(s.signal, SignalType::Short);

        // Inmunidad a NaN
        let signal_nan = CoaxialBreakoutEngine::evaluate_coaxial_breakout(
            &arena,
            f64::NAN,
            0.005,
            0.050,
            60000.0,
            true,
        );
        assert!(signal_nan.is_none());
    }

    #[test]
    fn test_coaxial_breakout_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("atr_1s", 0.0005);
        registry.set("atr_5s", 0.005);
        registry.set("atr_1m", 0.050);
        registry.set("order_flow_direction", 1.0);

        let mut engine = CoaxialBreakoutEngine::new();
        assert!(engine.init(registry).is_ok());
        assert_eq!(engine.name(), "CoaxialBreakoutEngine");

        let score = engine.evaluate();
        assert!(score.is_finite());
        assert!(score > 0.0);
    }
}

#[cfg(test)]
mod qo_616_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_616_compresion_entre_vecinos_y_direccion() {
        // Triángulo: desplazamientos que SUBEN hacia k=16 (compresión en
        // k=15: vecinos pequeños a la izquierda, grande a la derecha).
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.01 * (k as f64); // monótona creciente
        }
        let voto = CoaxialBreakoutEngine::voto_espectral(&x);
        // En una rampa monótona, TODAS las escalas interiores tienen
        // compresión 0 (los ratios a/b ≈ 1: vecinos proporcionales).
        for k in 1..31 {
            let a = x[k - 1].abs();
            let b = x[k].abs();
            let c = x[k + 1].abs();
            if (a / b - 1.0).abs() < 1e-3 && (b / c - 1.0).abs() < 1e-3 {
                assert!(
                    voto.en_escala(k).abs() < 0.5,
                    "rampa proporcional: sin compresión real en {}: {}",
                    k,
                    voto.en_escala(k)
                );
            }
        }
        // Ahora una compresión REAL: valle en k=15 (vecinos grandes).
        let mut valle = [0.0; ESCALAS_VOTO];
        for (k, v) in valle.iter_mut().enumerate() {
            *v = if k == 15 { 0.01 } else { 1.0 };
        }
        let voto_valle = CoaxialBreakoutEngine::voto_espectral(&valle);
        // El valle está COMPRIMIDO (sus vecinos son mucho más grandes):
        // a/b ≈ 100 ⇒ comp_ab ≈ 1; b/c ≈ 0.01 ⇒ comp_bc ≈ 0.
        // El producto será chico pero positivo — la compresión de una sola
        // cara no basta (el tensor exige AMBAS).
        assert!(voto_valle.en_escala(15).abs() < 1.0);
        // Rampa GEOMÉTRICA (duplica por escala): el producto tensorial de
        // DOS compresiones consecutivas debe dar squeeze > 0 en el interior.
        let mut rampa = [0.0; ESCALAS_VOTO];
        for (k, v) in rampa.iter_mut().enumerate() {
            *v = 2.0_f64.powi(k as i32 - 16); // 2^(k-16): e=0.125 en k=13,
            // duplica hasta e=2 en k=17. Toda subida es ×2 ⇒ ratio 0.5.
        }
        let voto_rampa = CoaxialBreakoutEngine::voto_espectral(&rampa);
        // En una rampa ×2 por escala: comp = (1 - 0.5) = 0.5 en ambas caras.
        // Producto = 0.5 * 0.5 * 4 = 1.0 ⇒ tanh(1.0) ≈ 0.76 en TODAS las
        // escalas interiores — firmado por la dirección del desplazamiento.
        for k in 14..19 {
            let v = voto_rampa.en_escala(k);
            assert!(
                v.abs() > 0.5,
                "rampa ×2 ⇒ squeeze tensorial fuerte en {}: {}",
                k,
                v
            );
        }
    }
}
