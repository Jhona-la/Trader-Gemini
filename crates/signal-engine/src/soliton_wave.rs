use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// 🌊 ALGORITMO #86: MOTOR DE ONDAS SOLITÓN NO-LINEALES DE SCHRÖDINGER (SOLITON WAVE ENGINE)
/// Modela los impulsos de precio como solitones no-dispersivos de la ecuación NLS (i \psi_t + 1/2 \psi_{xx} + |\psi|^2 \psi = 0),
/// identificando paquetes de liquidez que atraviesan la profundidad del libro de órdenes sin perder amplitud.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SolitonWaveEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SolitonWaveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolitonWaveEngine").finish()
    }
}

impl SolitonWaveEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula la amplitud del perfil solitónico sech(x) en O(1) con protección numérica absoluta
    #[inline(always)]
    pub fn compute_soliton_amplitude(
        amplitude: f64,
        velocity: f64,
        x_pos: f64,
        t_time: f64,
    ) -> f64 {
        // FIX #645: Sanitizar los parámetros entrantes
        let safe_amp = if amplitude.is_finite() {
            amplitude.abs()
        } else {
            0.0
        };
        let safe_vel = if velocity.is_finite() { velocity } else { 0.0 };
        let safe_x = if x_pos.is_finite() { x_pos } else { 0.0 };
        let safe_t = if t_time.is_finite() { t_time } else { 0.0 };

        let raw_phase = safe_amp * (safe_x - safe_vel * safe_t);
        // Clamping a [-50.0, 50.0] para evitar desbordamiento a +inf en cosh()
        let phase = raw_phase.clamp(-50.0, 50.0);
        let cosh_val = phase.cosh();
        if cosh_val == 0.0 || cosh_val.is_infinite() || cosh_val.is_nan() {
            return 0.0;
        }
        let res = safe_amp / cosh_val;
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Calcula la potencia de envolvente energética del paquete solitónico (Punto #265)
    #[inline(always)]
    pub fn compute_soliton_envelope_power(amplitude: f64) -> f64 {
        if !amplitude.is_finite() || amplitude <= 0.0 {
            0.0
        } else {
            (2.0 * amplitude).clamp(0.0, 100.0)
        }
    }

    /// #650 (Ola 50) — VOTO ESPECTRAL del solitón, con el pulso del lado
    /// correcto: `voto_k = tanh(A·x(τ_k))`. El solitón ES el pulso localizado
    /// de momentum — la escala donde el desplazamiento es fuerte ES donde
    /// viaja la onda; la convicción crece con |x| y satura (C∞, acotado,
    /// sin saltos).
    ///
    /// Lo que ELIMINA (auditor A, verificado): el perfil sech(A·x) votaba
    /// MÁXIMO (±1) donde el momentum era ~0 — máxima convicción justo donde
    /// la escala NO tiene información — y ~0 donde el momentum era fuerte:
    /// física INVERTIDA. Además el signo por desplazamiento infinitesimal
    /// daba un salto de magnitud ~2 en x=0. La amplitud A gobierna el ancho
    /// de respuesta (pendiente del tanh) — su rol de parámetro del motor.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        amplitud: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let a = if amplitud.is_finite() && amplitud > 0.0 {
            amplitud.clamp(1e-3, 10.0)
        } else {
            1.0
        };
        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            if !x.is_finite() {
                return 0.0;
            }
            (x.clamp(-10.0, 10.0) * a).tanh().clamp(-1.0, 1.0)
        })
    }
}

impl QuantumStrategy for SolitonWaveEngine {
    fn name(&self) -> &str {
        "SolitonWaveEngine"
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
        let registry = match self.registry.as_ref() {
            Some(r) => r,
            None => return 0.0,
        };
        let amp = registry
            .get_scoped_parameter(sym_opt, cid_opt, "soliton_amplitude", "SolitonWaveEngine")
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_imbalance",
                    "SolitonWaveEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(sym_opt, cid_opt, "vol_delta", "SolitonWaveEngine")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let vel = registry
            .get_scoped_parameter(sym_opt, cid_opt, "soliton_velocity", "SolitonWaveEngine")
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "price_velocity",
                    "SolitonWaveEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_velocity",
                    "SolitonWaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let pos = registry
            .get_scoped_parameter(sym_opt, cid_opt, "soliton_pos", "SolitonWaveEngine")
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "quantum_position_deviation",
                    "SolitonWaveEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_book_imbalance",
                    "SolitonWaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let t_time = registry
            .get_scoped_parameter(sym_opt, cid_opt, "soliton_time", "SolitonWaveEngine")
            .or_else(|| {
                registry.get_scoped_parameter(sym_opt, cid_opt, "hawkes_dt", "SolitonWaveEngine")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.05)
            .clamp(0.001, 1.0);

        let mid_price = registry
            .get_scoped_parameter(sym_opt, cid_opt, "mid_price", "SolitonWaveEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        // D-348: Adimensionalizar velocidad respecto al precio nominal para evitar colapso de sech(x) en BTC/ETH
        // G-04 — GUARDS NaN RESTAURADOS: la adimensionalización D-348
        // eliminó los checks de finitud — un NaN en vel/amp/pos propagaba
        // a la señal. Los guards internos de compute_soliton_amplitude
        // protegen los parámetros, pero vel.signum() y la división
        // necesitan protección explícita aquí.
        let vel = if vel.is_finite() { vel } else { return 0.0 };
        let amp = if amp.is_finite() { amp } else { return 0.0 };
        let pos = if pos.is_finite() { pos } else { return 0.0 };
        // AGY-AUD-P12: Adimensionalizar velocidad respecto al precio nominal de forma
        // universal para cualquier activo (invarianza de escala en todo el universo continuo).
        let norm_vel = if mid_price > 1e-8 && vel.abs() > 1e-12 {
            (vel / mid_price) * 10.0
        } else {
            vel
        };

        let amp_val = Self::compute_soliton_amplitude(amp, norm_vel, pos, t_time).clamp(0.0, 1.0);
        if amp_val.is_finite() {
            // #657 (F2-A3) — ERRADICACIÓN sombra/vivo: firma CONTINUA del
            // momentum (familia tanh(A·x) del voto espectral #650 — sin
            // escalón de signum en vel=0). SAT_MOMENTO: 1 bp/s de
            // momentum adimensional ≈ tanh(1); el cosh sigue aportando la
            // amplitud (pico donde hay momentum).
            const SAT_MOMENTO: f64 = 1e4;
            (norm_vel * SAT_MOMENTO).tanh() * amp_val
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soliton_amplitude_overflow_immunity() {
        // Fase extrema que desbordaría sin clamping
        let amp_huge = SolitonWaveEngine::compute_soliton_amplitude(10.0, 1.0, 1000.0, 0.0);
        assert!(!amp_huge.is_nan() && !amp_huge.is_infinite());
        assert!(amp_huge >= 0.0);

        // Fase centrada
        let amp_zero = SolitonWaveEngine::compute_soliton_amplitude(2.0, 1.0, 0.0, 0.0);
        assert!((amp_zero - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_soliton_zero_velocity_returns_flat() {
        let engine = SolitonWaveEngine::new();
        // Sin registro o con velocidad 0, debe retornar estrictamente 0.0
        assert_eq!(engine.evaluate(), 0.0);
    }

    #[test]
    fn test_soliton_envelope_power() {
        let p_zero = SolitonWaveEngine::compute_soliton_envelope_power(0.0);
        assert_eq!(p_zero, 0.0);

        let p_normal = SolitonWaveEngine::compute_soliton_envelope_power(2.5);
        assert_eq!(p_normal, 5.0);

        let p_nan = SolitonWaveEngine::compute_soliton_envelope_power(f64::NAN);
        assert_eq!(p_nan, 0.0);
    }

    #[test]
    fn test_soliton_wave_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("soliton_amplitude", 0.8);
        registry.set("soliton_velocity", 1.5);
        registry.set("soliton_pos", 0.0);

        let mut engine = SolitonWaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval > 0.0,
            "Velocidad positiva y amplitud centrada deben producir señal positiva"
        );
        assert!(eval <= 1.0);
    }

    #[test]
    fn test_soliton_sub_dollar_scale_invariance() {
        let registry = Arc::new(OmniscientRegistry::new());
        // Simular token sub-dólar (DOGE a $0.15)
        registry.set_scoped("DOGEUSDT", "soliton_amplitude", 0.7);
        registry.set_scoped("DOGEUSDT", "soliton_velocity", 0.0015);
        registry.set_scoped("DOGEUSDT", "soliton_pos", 0.0);
        registry.set_scoped("DOGEUSDT", "mid_price", 0.15);

        let mut engine = SolitonWaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate_for_coin(5, "DOGEUSDT");
        assert!(
            eval > 0.0 && eval <= 1.0,
            "Sub-dólar debe evaluar señal finita no-cero: {eval}"
        );
    }
}

#[cfg(test)]
mod qo_610_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_610_soliton_nucleo_pleno_colas_sech_y_antisimetria() {
        let a = 1.0;
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.1 * (k as f64 - 15.5); // de −1.55 a +1.65 alrededor de 0
        }
        let voto = SolitonWaveEngine::voto_espectral(&x, a);
        // Antisimetría estricta.
        for kk in 0..ESCALAS_VOTO {
            let idx_espejo = ESCALAS_VOTO - 1 - kk;
            if x[kk] != 0.0 && (x[kk] + x[idx_espejo]).abs() < 1e-12 {
                assert!(
                    (voto.en_escala(kk) + voto.en_escala(idx_espejo)).abs() < 1e-12,
                    "antisimetría rota en {}: {} vs {}",
                    kk,
                    voto.en_escala(kk),
                    voto.en_escala(idx_espejo)
                );
            }
        }
        // #650 — el pulso vive donde el momentum es FUERTE: la cola
        // (|x| grande = la onda viaja ahí) vota más que el núcleo quieto.
        // (El sech viejo lo tenía al revés: máxima convicción en x≈0.)
        let nucleo = voto.en_escala(15).abs(); // x = −0.05, sin información
        let cola = voto.en_escala(0).abs(); // x = −1.55, momentum fuerte
        assert!(cola > nucleo, "cola {} debe superar el núcleo {}", cola, nucleo);
        // CONTINUIDAD en x=0: sin salto de magnitud ni cambio de signo brusco.
        let mut x_grad = [0.0; ESCALAS_VOTO];
        x_grad[7] = 1e-9;
        let suave = SolitonWaveEngine::voto_espectral(&x_grad, a);
        assert!(suave.en_escala(7).abs() < 1e-8);
        // Acotado.
        for kk in 0..ESCALAS_VOTO {
            assert!(voto.en_escala(kk).abs() <= 1.0 + 1e-12);
        }
        // Amplitud inválida ⇒ 1.0 normalizador, sin inventar NaN.
        let voto_nan = SolitonWaveEngine::voto_espectral(&x, f64::NAN);
        assert!(voto_nan.en_escala(0).is_finite());
    }
}

#[cfg(test)]
mod qo_657_tests {
    use super::*;
    use std::sync::Arc;

    fn eval_con(vel: f64, amp: f64) -> f64 {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("mid_price", 60_000.0);
        registry.set("soliton_velocity", vel);
        registry.set("soliton_amplitude", amp);
        registry.set("soliton_pos", 0.5);
        registry.set("soliton_time", 1.0);
        let mut engine = SolitonWaveEngine::new();
        assert!(engine.init(registry).is_ok());
        engine.evaluate_for_coin(0, "TESTUSDT")
    }

    /// #657 (F2-A3): la firma del vivo es CONTINUA (familia tanh(A·x) del
    /// voto espectral #650) — sin el salto de signum en vel=0, y con
    /// paridad de signo.
    #[test]
    fn qo_657_soliton_vivo_firma_continua() {
        let a = eval_con(1e-6, 1.0);
        let b = eval_con(-1e-6, 1.0);
        assert!(
            (a - b).abs() < 0.05,
            "sin salto en vel=0 (antes signum saltaba ±amplitud): {a} vs {b}"
        );
        let pos_v = eval_con(60.0, 1.0); // 60/60000*10 = 1e-2 ⇒ tanh(100)≈1
        let neg_v = eval_con(-60.0, 1.0);
        assert!(pos_v > 0.0 && neg_v < 0.0, "paridad de signo: {pos_v} vs {neg_v}");
        // Paridad aproximada: la firma es antisimétrica; la ENVOLVENTE cosh
        // depende de (pos − vel·t), no par en vel — el solitón viaja.
        assert!(
            (pos_v + neg_v).abs() < 0.05,
            "paridad de signo/magnitud: {pos_v} vs {neg_v}"
        );
    }
}
