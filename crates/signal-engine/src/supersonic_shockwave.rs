use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// ⚡ ALGORITMO #82: MOTOR DE ONDA DE CHOQUE SUPERSÓNICA Y NÚMERO DE MACH (SUPERSONIC SHOCKWAVE ENGINE)
/// Modela el flujo de órdenes como una onda de presión supersónica, calculando el número de Mach M = v_flujo / v_sonido,
/// identificando la ignición explosiva de rupturas de volatilidad cuando M > 1.0.
/// ⚡ OLA Ω123: Parámetros desacoplados y configurables para el motor de onda de choque supersónica
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupersonicShockwaveParams {
    /// Umbral sónico modal en espacio de z-score (default: 1.0)
    pub sonic_speed_z: f64,
    /// Umbral crítico del número de Mach para ignición de salto de choque (default: 1.0)
    pub mach_threshold: f64,
    /// Factor de escala del argumento tanh en la firma direccional del flujo (default: 2.0)
    pub signature_scale: f64,
    /// Suelo mínimo de velocidad del sonido para evitar singularidades (default: 1e-6)
    pub min_speed_of_sound: f64,
}

impl Default for SupersonicShockwaveParams {
    fn default() -> Self {
        Self {
            sonic_speed_z: 1.0,
            mach_threshold: 1.0,
            signature_scale: 2.0,
            min_speed_of_sound: 1e-6,
        }
    }
}

impl SupersonicShockwaveParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.sonic_speed_z.is_finite() || !(0.05..=10.0).contains(&self.sonic_speed_z) {
            return Err(format!("Invalid sonic_speed_z: {}", self.sonic_speed_z));
        }
        if !self.mach_threshold.is_finite() || !(0.5..=5.0).contains(&self.mach_threshold) {
            return Err(format!("Invalid mach_threshold: {}", self.mach_threshold));
        }
        if !self.signature_scale.is_finite() || !(0.1..=10.0).contains(&self.signature_scale) {
            return Err(format!("Invalid signature_scale: {}", self.signature_scale));
        }
        if !self.min_speed_of_sound.is_finite() || !(1e-9..=1e-3).contains(&self.min_speed_of_sound) {
            return Err(format!("Invalid min_speed_of_sound: {}", self.min_speed_of_sound));
        }
        Ok(())
    }
}

#[derive(Clone)]
#[repr(C, align(64))]
pub struct SupersonicShockwaveEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    pub params: SupersonicShockwaveParams,
}

impl Default for SupersonicShockwaveEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SupersonicShockwaveEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupersonicShockwaveEngine")
            .field("params", &self.params)
            .finish()
    }
}

impl SupersonicShockwaveEngine {
    pub fn new() -> Self {
        Self {
            registry: None,
            params: SupersonicShockwaveParams::default(),
        }
    }

    /// OLA Ω123: Constructor con parámetros desacoplados específicos
    pub fn with_params(mut self, params: SupersonicShockwaveParams) -> Self {
        if params.validate().is_ok() {
            self.params = params;
        }
        self
    }

    /// OLA Ω123: Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: SupersonicShockwaveParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
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

    /// OLA Ω123: Evalúa la compresión de salto de Rankine-Hugoniot con umbral crítico configurable
    #[inline(always)]
    pub fn compute_shockwave_jump_with_threshold(mach: f64, threshold: f64) -> f64 {
        let thresh = if threshold.is_finite() && threshold > 0.0 {
            threshold
        } else {
            1.0
        };
        if mach.is_finite() && mach > thresh {
            let m_norm = mach / thresh;
            let inverse_square = m_norm.recip().powi(2);
            ((1.0 - inverse_square) / (1.0 + inverse_square))
                .tanh()
                .clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Evalúa la compresión de salto de Rankine-Hugoniot en el frente de onda con umbral sónico unitario (Mach > 1.0)
    #[inline(always)]
    pub fn compute_shockwave_jump(mach: f64) -> f64 {
        Self::compute_shockwave_jump_with_threshold(mach, 1.0)
    }

    /// OLA Ω123: VOTO ESPECTRAL del choque supersónico con parámetros desacoplados configurables
    pub fn voto_espectral_with_params(
        desplazamientos: &[f64; 32],
        params: &SupersonicShockwaveParams,
    ) -> crate::voto_espectral::VotoEspectral {
        let c_z = if params.sonic_speed_z.is_finite() && params.sonic_speed_z > 0.0 {
            params.sonic_speed_z.clamp(0.1, 10.0)
        } else {
            1.0
        };
        let scale = if params.signature_scale.is_finite() && params.signature_scale > 0.0 {
            params.signature_scale
        } else {
            2.0
        };
        let mach_thresh = params.mach_threshold;
        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            if !x.is_finite() {
                return 0.0;
            }
            let m = Self::compute_mach_number(x.abs().min(10.0), c_z);
            ((x.clamp(-10.0, 10.0) / c_z) / scale).tanh()
                * Self::compute_shockwave_jump_with_threshold(m, mach_thresh).clamp(-1.0, 1.0)
        })
    }

    /// #650 (Ola 50) — VOTO ESPECTRAL del choque supersónico, con firma
    /// direccional y unidades coherentes: `voto_k = tanh(x(τ_k)) ·
    /// salto(M)` con `M = |x|/c_z` en ESPACIO DE Z (c_z = umbral sónico en
    /// desviaciones — el core pasa 1.0 = 1σ). La compresión del salto
    /// Rankine-Hugoniot es MAGNITUD; la DIRECCIÓN la aporta el flujo de esa
    /// escala con tanh (continua, sin escalón de signo).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
        velocidad_sonido_z: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let params = SupersonicShockwaveParams {
            sonic_speed_z: velocidad_sonido_z,
            ..SupersonicShockwaveParams::default()
        };
        Self::voto_espectral_with_params(desplazamientos, &params)
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
        // #664 (G2-8): la unidad se decide por la FUENTE del parámetro,
        // no por su magnitud — el umbral `sound > 1.0` clasificaba mal
        // el sonido de tokens sub-dólar (precio/s < 1) como «por barra»
        // y el Mach quedaba inflado cientos de × (el defecto #650
        // reaparecía para DOGE/PEPE).
        // `spread_speed_of_sound` (escritor del core) es precio/s ⇒ se
        // normaliza por mid_price; `atr_pct` es fracción POR BARRA de
        // 60 s ⇒ se convierte a fracción/segundo aquí.
        // #666 (H2-5): bajo difusión, E[rango_60s] = σ√60 — el análogo
        // en velocidad-por-segundo del rango ATR es ÷√60, no ÷60 (drift).
        // ÷60 subrestimaba la velocidad del sonido 7.75× ⇒ Mach inflado.
        const BARRA_S: f64 = 7.7459666924; // √60
        let has_spread_sound = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "spread_speed_of_sound",
                "SupersonicShockwaveEngine",
            )
            .is_some();
        if !has_spread_sound && mid_price <= 1e-8 {
            // R5-C1 (R4-C5): con spread_speed_of_sound ausente, el fallback es atr_pct (fracción/s).
            // Sin mid_price (>1e-8), speed no se puede normalizar a fracción/s y permanece en dólares/s.
            // Mezclar dólares/s con fracción/s rompe unidades físicas (Mach ×mid_price).
            // Por coherencia dimensional estricta, el motor se abstiene devolviendo 0.0.
            return 0.0;
        }

        let sound_segundo = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "spread_speed_of_sound",
                "SupersonicShockwaveEngine",
            )
            .map(|p| {
                let v = p.get_value();
                if mid_price > 1e-8 && v.is_finite() {
                    v / mid_price
                } else {
                    v
                }
            })
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "atr_pct",
                    "SupersonicShockwaveEngine",
                )
                .map(|p| p.get_value() / BARRA_S)
            })
            .unwrap_or(0.001);
        // R5-C1 resuelto arriba por Ω21 (guarda previa al sonido — el
        // auto-merge duplicó esta segunda guarda de la Ola 71; retirada).
        let sound_norm = sound_segundo;

        if !speed.is_finite() || !sound_norm.is_finite() || sound_norm <= 0.0 {
            return 0.0;
        }

        let mach = Self::compute_mach_number(speed_norm.abs(), sound_norm);
        let jump = Self::compute_shockwave_jump_with_threshold(mach, self.params.mach_threshold);
        // #659 (F2-A11): firma CONTINUA del flujo (familia tanh del
        // solitón #657 — sin escalón de signum en speed=0).
        // R4-C2 & OLA Ω123: firma a la ESCALA DEL ESTADÍSTICO desacoplada
        let scale = if self.params.signature_scale.is_finite() && self.params.signature_scale > 0.0 {
            self.params.signature_scale
        } else {
            2.0
        };
        (speed_norm / sound_norm / scale).tanh() * jump
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

#[cfg(test)]
mod qo_666_tests {
    use super::*;
    use omniscient_registry::OmniscientRegistry;
    use std::sync::Arc;

    /// #666 (H2-5): el fallback ATR como velocidad del sonido usa la
    /// física DIFUSIVA /√60 — bajo difusión E|rango_60s|=σ√60, así el
    /// análogo por segundo del rango es ÷√60. Con el drift /60 la
    /// velocidad quedaba 7.75× subrestimada (Mach inflado).
    #[test]
    fn qo_666_fallback_atr_es_difusivo() {
        // Registro con sólo atr_pct (sin spread_speed_of_sound) y mid alto
        // para que speed_norm = speed/mid: 1.0/60 s = fracción/segundo.
        let registry = Arc::new(OmniscientRegistry::new());
        // speed_norm = 72/60000 = 0.0012/s; sound_difusivo = atr/√60 =
        // 0.001/s ⇒ Mach = 1.2 ⇒ jump(1.2) = tanh(0.22) ≈ 0.21 débil.
        // Con el VIEJO drift /60: sound = 1.29e-4 ⇒ Mach ≈ 9.3 ⇒ jump ≈ 1
        // — el test discrimina difusión vs drift por la magnitud del voto.
        registry.set("order_flow_speed", 72.0);
        registry.set("atr_pct", 0.0077459666924); // = √60/1000
        registry.set("mid_price", 60_000.0);
        let mut engine = SupersonicShockwaveEngine::default();
        engine.init(Arc::clone(&registry)).ok();
        let v = engine.evaluate();
        assert!(v > 0.0 && v < 0.2, "Mach 1.2 difusivo = salto debil: {v}");
        // Y a Mach ~10 (0.01/s): jump saturado.
        registry.set("order_flow_speed", 600.0);
        let v10 = engine.evaluate();
        assert!(v10 > 0.7 && v10 > 3.0 * v, "Mach 10 satura vs Mach 1.2 debil: {v10} vs {v}");
    }

    /// R5-C1 (R4-C5): sin spread_speed_of_sound y sin mid_price (>1e-8),
    /// el fallback fraccional (atr_pct) no puede coexistir con speed en USD/s
    /// sin romper la coherencia dimensional. El motor debe abstenerse (0.0).
    #[test]
    fn r5_c1_abstiene_sin_mid_price_con_fallback_fraccional() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("order_flow_speed", 72.0);
        registry.set("atr_pct", 0.0077459666924);
        // mid_price ausente (0.0 <= 1e-8)
        let mut engine = SupersonicShockwaveEngine::default();
        engine.init(Arc::clone(&registry)).ok();
        let v = engine.evaluate();
        assert_eq!(v, 0.0, "Sin mid_price y con fallback fraccional debe abstenerse");
    }
}

