use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ ALGORITMO #82: MOTOR DE ONDA DE CHOQUE SUPERSÓNICA Y NÚMERO DE MACH (SUPERSONIC SHOCKWAVE ENGINE)
/// Modela el flujo de órdenes como una onda de presión supersónica, calculando el número de Mach M = v_flujo / v_sonido,
/// identificando la ignición explosiva de rupturas de volatilidad cuando M > 1.0.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct SupersonicShockwaveEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for SupersonicShockwaveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupersonicShockwaveEngine").finish()
    }
}

impl SupersonicShockwaveEngine {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula el número de Mach de mercado M en O(1) con protección contra división por cero
    #[inline(always)]
    pub fn compute_mach_number(order_flow_speed: f64, spread_speed_of_sound: f64) -> f64 {
        // FIX #644: Sanitizar parámetros de Mach
        let safe_speed = if order_flow_speed.is_finite() {
            order_flow_speed.abs()
        } else {
            0.0
        };
        let safe_sound = if spread_speed_of_sound.is_finite() && spread_speed_of_sound > 0.0 {
            spread_speed_of_sound.max(1e-6)
        } else {
            0.001
        };
        safe_speed / safe_sound
    }

    /// Evalúa la compresión de salto de Rankine-Hugoniot en el frente de onda
    #[inline(always)]
    pub fn compute_shockwave_jump(mach: f64) -> f64 {
        if mach.is_finite() && mach > 1.0 {
            // Equivalent ratio after dividing numerator/denominator by M^2.
            // For M>1 the reciprocal is bounded and cannot overflow on squaring.
            // This stabilizes the heuristic, not a physical conservation law.
            let inverse_square = mach.recip().powi(2);
            ((1.0 - inverse_square) / (1.0 + inverse_square))
                .tanh()
                .clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// #650 (Ola 50) — VOTO ESPECTRAL del choque supersónico, con firma
    /// direccional y unidades coherentes: `voto_k = tanh(x(τ_k)) ·
    /// salto(M)` con `M = |x|/c_z` en ESPACIO DE Z (c_z = umbral sónico en
    /// desviaciones — el core pasa 1.0 = 1σ). La compresión del salto
    /// Rankine-Hugoniot es MAGNITUD; la DIRECCIÓN la aporta el flujo de esa
    /// escala con tanh (continua, sin escalón de signo).
    ///
    /// Lo que ELIMINA (auditor A, verificado): (a) el voto era SIN signo
    /// [0,1] — en la composición del consenso (media de votos firmados)
    /// inyectaba un sesgo LARGO permanente de ~0.76 en cada escala activa;
    /// (b) c llegaba en unidades de PRECIO del vivo (0.001) contra x en
    /// z-scores O(1) ⇒ M ~ 10³ saturado ⇒ salto ≈ 0.76 constante sin
    /// discriminación espectral. Con c_z = 1σ, sólo las escalas con
    /// desplazamiento超 sónico (>1σ) declaran choque.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        velocidad_sonido_z: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let c_z = if velocidad_sonido_z.is_finite() && velocidad_sonido_z > 0.0 {
            velocidad_sonido_z.clamp(0.1, 10.0)
        } else {
            1.0
        };
        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            if !x.is_finite() {
                return 0.0;
            }
            let m = Self::compute_mach_number(x.abs().min(10.0), c_z);
            (x.clamp(-10.0, 10.0).tanh() * Self::compute_shockwave_jump(m))
                .clamp(-1.0, 1.0)
        })
    }
}

impl QuantumStrategy for SupersonicShockwaveEngine {
    fn name(&self) -> &str {
        "SupersonicShockwaveEngine"
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
        let speed = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_flow_speed",
                "SupersonicShockwaveEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "order_flow_velocity",
                    "SupersonicShockwaveEngine",
                )
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "price_velocity",
                    "SupersonicShockwaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        let sound = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "spread_speed_of_sound",
                "SupersonicShockwaveEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "atr_pct",
                    "SupersonicShockwaveEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.001);

        if !speed.is_finite() || !sound.is_finite() {
            return 0.0;
        }

        let mid_price = registry
            .get_scoped_parameter(sym_opt, cid_opt, "mid_price", "SupersonicShockwaveEngine")
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        // AGY-AUD-P13: Homogeneizar dimensionalmente velocidad y velocidad del sonido
        // de forma universal e invariante a la escala de precios nominales.
        let speed_norm = if mid_price > 1e-8 && speed.abs() > 1e-12 {
            speed / mid_price
        } else {
            speed
        };
        let sound_norm = if mid_price > 1e-8 && sound > 1.0 {
            sound / mid_price
        } else {
            sound
        };

        let mach = Self::compute_mach_number(speed_norm.abs(), sound_norm);
        let jump = Self::compute_shockwave_jump(mach);
        speed.signum() * jump
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mach_number_and_shockwave_jump() {
        let mach = SupersonicShockwaveEngine::compute_mach_number(2.0, 1.0);
        assert_eq!(mach, 2.0);

        let jump = SupersonicShockwaveEngine::compute_shockwave_jump(mach);
        assert!(jump > 0.0);

        let sub_mach = SupersonicShockwaveEngine::compute_mach_number(0.5, 1.0);
        let sub_jump = SupersonicShockwaveEngine::compute_shockwave_jump(sub_mach);
        assert_eq!(sub_jump, 0.0);
    }

    #[test]
    fn test_mach_number_nan_and_infinite_immunity() {
        let mach_nan = SupersonicShockwaveEngine::compute_mach_number(f64::NAN, 1.0);
        assert!(mach_nan.is_finite() && mach_nan == 0.0);

        let jump_nan = SupersonicShockwaveEngine::compute_shockwave_jump(f64::NAN);
        assert_eq!(jump_nan, 0.0);

        let mach_zero_sound = SupersonicShockwaveEngine::compute_mach_number(10.0, 0.0);
        assert!(mach_zero_sound.is_finite() && mach_zero_sound > 0.0);
    }

    #[test]
    fn test_supersonic_shockwave_engine_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_flow_speed", 5.0);
        registry.set("spread_speed_of_sound", 1.0);

        let mut engine = SupersonicShockwaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate();
        assert!(
            eval > 0.0,
            "Velocidad supersónica positiva debe generar señal de choque positiva"
        );
        assert!(eval <= 1.0);
    }

    #[test]
    fn test_supersonic_sub_dollar_scale_invariance() {
        let registry = Arc::new(OmniscientRegistry::new());
        // Simular token sub-dólar (DOGE a $0.15)
        registry.set_scoped("DOGEUSDT", "order_flow_speed", 0.0030); // 2%/s
        registry.set_scoped("DOGEUSDT", "atr_pct", 0.001); // 0.1% velocidad del sonido relativa
        registry.set_scoped("DOGEUSDT", "mid_price", 0.15);

        let mut engine = SupersonicShockwaveEngine::new();
        assert!(engine.init(registry).is_ok());

        let eval = engine.evaluate_for_coin(5, "DOGEUSDT");
        assert!(
            eval > 0.0 && eval <= 1.0,
            "Sub-dólar debe evaluar señal supersónica finita no-cero: {eval}"
        );
    }
}

#[cfg(test)]
mod qo_610_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_610_shock_subsónico_cero_y_monótono_en_mach() {
        // #650 — umbral sónico en ESPACIO DE Z (1σ): sólo |x|>1 declara choque.
        let c = 1.0;
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.1 * (k as f64 - 15.5); // −1.55 … +1.65
        }
        let voto = SupersonicShockwaveEngine::voto_espectral(&x, c);
        // Subsónico (|x|<1 ⇒ M<1: escalas 6..25) ⇒ salto 0.
        for k in 6..26 {
            assert_eq!(voto.en_escala(k), 0.0, "escala {} subsónica debe votar 0", k);
        }
        // Supersónico FIRMA DO por el flujo: positivo donde x>0, negativo
        // donde x<0 (antes era [0,1] sin signo = sesgo largo permanente).
        assert!(voto.en_escala(30) > 0.0);
        assert!(voto.en_escala(1) < 0.0);
        // Monótono en |x| dentro de cada lado (el salto crece con M).
        assert!(voto.en_escala(30).abs() > voto.en_escala(27).abs());
        assert!(voto.en_escala(0).abs() > voto.en_escala(3).abs());
        // Antisimetría estricta: −x da el voto espejo.
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let neg = SupersonicShockwaveEngine::voto_espectral(&x_neg, c);
        for k in 0..ESCALAS_VOTO {
            assert!(
                (voto.en_escala(k) + neg.en_escala(k)).abs() < 1e-12,
                "antisimetría en {k}"
            );
        }
        // El salto en M=2 es tanh(0.6) — raíz analítica.
        let m2 = SupersonicShockwaveEngine::compute_mach_number(2.0, c);
        assert!((m2 - 2.0).abs() < 1e-9);
        let salto_m2 = SupersonicShockwaveEngine::compute_shockwave_jump(m2);
        assert!((salto_m2 - 0.6_f64.tanh()).abs() < 1e-12);
        // c inválida o fuera de rango ⇒ 1σ sin inventar.
        let c_rara = SupersonicShockwaveEngine::voto_espectral(&x, f64::NAN);
        assert_eq!(c_rara.en_escala(30), voto.en_escala(30));
    }

}
