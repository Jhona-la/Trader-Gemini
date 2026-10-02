use omniscient_registry::OmniscientRegistry;
use std::f64;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// MOTOR DE EXPANSIÓN DE TENDENCIA (TREND-RUNNER).
///
/// Amplía el objetivo de recorrido cuando se confirma inercia estocástica en
/// la serie.
///
/// U-ERR-1: la descripción anterior anunciaba «objetivos de Take-Profit en
/// Swing (+3,5% a +8,0%)» y un «payoff de +500 bps frente a 4,5 bps de fee»
/// del que se derivaba una retención «del 99,1%». Ni la banda de horizonte ni
/// esos números describen lo que el motor calcula: la expansión es continua y
/// depende del estado de la serie, y ninguna cifra de retención está medida
/// aquí. La clave de registro `ema_trend_swing` se conserva porque su
/// productor vive fuera de este ámbito.
#[derive(Clone, Default)]
#[repr(C, align(64))]
pub struct HighPayoffTrendRunner {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl std::fmt::Debug for HighPayoffTrendRunner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HighPayoffTrendRunner").finish()
    }
}

impl HighPayoffTrendRunner {
    pub fn new() -> Self {
        Self { registry: None }
    }

    /// Calcula el objetivo de extensión de ganancia alta usando funciones continuas tensoriales (O(1))
    #[inline(always)]
    pub fn calculate_expanded_tp(
        base_tp_magnitude: f64,
        hurst: f64,
        vpin: f64,
        atr_pct: f64,
        upper_bound: f64, // Extracted from SuperGenotype
    ) -> f64 {
        // FIX #641: Sanitizar finitud de parámetros entrantes
        if !base_tp_magnitude.is_finite()
            || !hurst.is_finite()
            || !vpin.is_finite()
            || !atr_pct.is_finite()
            || !upper_bound.is_finite()
        {
            return 0.02;
        }

        let safe_base_tp = base_tp_magnitude.clamp(0.001, 0.50);
        let safe_hurst = hurst.clamp(0.0, 1.0);
        let safe_vpin = vpin.clamp(0.0, 1.0);
        let safe_atr = atr_pct.clamp(0.0001, 0.50);

        // Activación continua: Hurst (>0.5 indica persistencia de tendencia)
        let trend_score = ((safe_hurst - 0.5) * 4.0).tanh().max(0.0);

        // VPIN (Volume-Synchronized Probability of Toxicity): Ante alta toxicidad institucional (vpin > 0.50),
        // reducimos la dilatación del TP para capturar ganancias rápidamente antes de la reversión por selección adversa.
        let toxicity_penalty = (1.0 - (safe_vpin * 1.5).tanh()).max(0.2);

        // Factor de expansión tensorial calibrado
        let expansion_factor = trend_score * toxicity_penalty;

        // Cálculo del TP expandido, integrando directamente la volatilidad (atr_pct)
        let expanded_tp = safe_base_tp * (1.0 + (expansion_factor * 3.0));
        let vol_boost = safe_atr * expansion_factor * 4.0;

        let final_tp = (expanded_tp + vol_boost).max(safe_base_tp.max(0.0020));

        // Límite superior suave continuo usando tanh() en lugar de un clamp() brusco.
        // Asymptotically approaches upper_bound sin romper derivabilidad.
        let safe_bound = upper_bound.clamp(0.01, 0.50); // Prevent div by zero
        let res = safe_bound * (final_tp / safe_bound).tanh();
        if res.is_finite() {
            res
        } else {
            0.02
        }
    }

    /// Voto espectral multiescala (32 escalas): evalúa la inercia y persistencia
    /// de la tendencia sobre el desplazamiento direccional x(τ_k) de cada escala temporal.
    pub fn voto_espectral(
        desplazamientos: &[f64; crate::voto_espectral::ESCALAS_VOTO],
        hurst: f64,
        vpin: f64,
        atr_pct: f64,
    ) -> crate::voto_espectral::VotoEspectral {
        let safe_hurst = if hurst.is_finite() { hurst } else { 0.50 };
        let safe_vpin = if vpin.is_finite() { vpin } else { 0.50 };
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else {
            0.01
        };

        let h_excess = (safe_hurst - 0.50).max(0.0);
        let h_weight = (h_excess / 0.04).tanh();
        if h_weight <= 1e-4 {
            return crate::voto_espectral::VotoEspectral::default();
        }

        let tp = Self::calculate_expanded_tp(0.02, safe_hurst, safe_vpin, safe_atr, 0.08);
        let tp_factor = (tp * 10.0).tanh();

        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            if x.abs() < 1e-6 {
                0.0
            } else {
                let dir_weight = (x / 1e-3).tanh();
                dir_weight * h_weight * tp_factor
            }
        })
    }
}

impl QuantumStrategy for HighPayoffTrendRunner {
    fn name(&self) -> &str {
        "HighPayoffTrendRunner"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
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

        let hurst = r
            .get_scoped_parameter(sym_opt, cid_opt, "hurst_exponent", "HighPayoffTrendRunner")
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "global_hurst", "HighPayoffTrendRunner")
            })
            .or_else(|| r.get_scoped_parameter(sym_opt, cid_opt, "hurst", "HighPayoffTrendRunner"))
            .map(|p| p.get_value())
            .unwrap_or(0.50);

        let vpin = r
            .get_scoped_parameter(sym_opt, cid_opt, "cvpin", "HighPayoffTrendRunner")
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "order_flow_vpin", "HighPayoffTrendRunner")
            })
            .or_else(|| r.get_scoped_parameter(sym_opt, cid_opt, "vpin", "HighPayoffTrendRunner"))
            .map(|p| p.get_value())
            .unwrap_or(0.50);

        let atr_pct = r
            .get_scoped_parameter(sym_opt, cid_opt, "atr_pct", "HighPayoffTrendRunner")
            .or_else(|| {
                r.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "relative_atr_pct",
                    "HighPayoffTrendRunner",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.01);

        let trend_direction = r
            .get_scoped_parameter(sym_opt, cid_opt, "trend_direction", "HighPayoffTrendRunner")
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "ema_trend_swing", "HighPayoffTrendRunner")
            })
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "ema_trend", "HighPayoffTrendRunner")
            })
            .or_else(|| {
                r.get_scoped_parameter(sym_opt, cid_opt, "price_velocity", "HighPayoffTrendRunner")
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        // FIX #684: Sanitizar lecturas de registros
        let safe_hurst = if hurst.is_finite() { hurst } else { 0.50 };
        let safe_vpin = if vpin.is_finite() { vpin } else { 0.50 };
        let safe_atr = if atr_pct.is_finite() && atr_pct > 0.0 {
            atr_pct
        } else {
            0.01
        };
        let safe_dir = if trend_direction.is_finite() {
            trend_direction
        } else {
            0.0
        };

        // Modulación C^∞ continua para persistencia de tendencia (Hurst > 0.50):
        // Erradica el salto escalón del ~20% que ocurría al cruzar el umbral rígido H = 0.52.
        let h_excess = (safe_hurst - 0.50).max(0.0);
        let h_weight = (h_excess / 0.04).tanh();
        if h_weight <= 1e-4 || safe_dir.abs() < 1e-6 {
            return 0.0;
        }

        let tp = Self::calculate_expanded_tp(0.02, safe_hurst, safe_vpin, safe_atr, 0.08);
        let dir_weight = (safe_dir / 1e-3).tanh();
        dir_weight * h_weight * (tp * 10.0).tanh()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expanded_tp_calculation() {
        let tp = HighPayoffTrendRunner::calculate_expanded_tp(0.02, 0.70, 0.5, 0.01, 0.10);
        assert!(tp >= 0.02, "TP expandido debe ser mayor o igual al base");
        assert!(tp <= 0.10, "TP expandido no debe superar upper_bound");
    }

    #[test]
    fn test_trend_runner_nan_immunity() {
        let tp_nan = HighPayoffTrendRunner::calculate_expanded_tp(
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        );
        assert!(tp_nan.is_finite() && tp_nan > 0.0);
    }

    #[test]
    fn test_trend_runner_evaluate_with_registry() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("hurst_exponent", 0.75);
        registry.set("cvpin", 0.20);
        registry.set("atr_pct", 0.02);
        registry.set("trend_direction", 1.0);

        let mut runner = HighPayoffTrendRunner::new();
        assert!(runner.init(registry).is_ok());

        let eval = runner.evaluate();
        assert!(
            eval > 0.0,
            "Hurst alto y dirección alcista deben generar señal positiva"
        );
        assert!(eval <= 1.0);
    }

    #[test]
    fn test_trend_runner_continuous_hurst_transition() {
        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("cvpin", 0.20);
        registry.set("atr_pct", 0.02);
        registry.set("trend_direction", 1.0);

        let mut runner = HighPayoffTrendRunner::new();
        assert!(runner.init(registry.clone()).is_ok());

        // Para H <= 0.50 (caminata aleatoria pura o reversión), señal es 0
        registry.set("hurst_exponent", 0.50);
        assert_eq!(runner.evaluate(), 0.0);

        // Para H = 0.51, la señal es suavemente positiva sin saltar abruptamente
        registry.set("hurst_exponent", 0.51);
        let eval_51 = runner.evaluate();
        assert!(eval_51 > 0.0 && eval_51 < 0.15);

        // Para H = 0.53, crece continuamente
        registry.set("hurst_exponent", 0.53);
        let eval_53 = runner.evaluate();
        assert!(eval_53 > eval_51);
    }

    #[test]
    fn qo_espectral_trend_runner_escala_antisimetrica() {
        use crate::voto_espectral::ESCALAS_VOTO;
        let mut desplazamientos = [0.0; ESCALAS_VOTO];
        for (k, v) in desplazamientos.iter_mut().enumerate() {
            *v = 0.002 * (k as f64 - 15.5);
        }

        // Con H <= 0.50 el voto espectral debe abstenerse (todo 0)
        let voto_neutro = HighPayoffTrendRunner::voto_espectral(&desplazamientos, 0.48, 0.20, 0.01);
        for k in 0..ESCALAS_VOTO {
            assert_eq!(voto_neutro.en_escala(k), 0.0);
        }

        // Con H = 0.70 (fuerte persistencia), el voto es antisimétrico y no cero
        let voto_persistente = HighPayoffTrendRunner::voto_espectral(&desplazamientos, 0.70, 0.20, 0.01);
        for k in 0..ESCALAS_VOTO {
            let idx = ESCALAS_VOTO - 1 - k;
            let vk = voto_persistente.en_escala(k);
            let v_opp = voto_persistente.en_escala(idx);
            assert!(
                (vk + v_opp).abs() < 1e-12,
                "Antisimetría espectral rota en escala {}: {} vs {}",
                k,
                vk,
                v_opp
            );
            assert!(vk.abs() <= 1.0);
        }
        assert!(voto_persistente.en_escala(0) < 0.0);
        assert!(voto_persistente.en_escala(31) > 0.0);
    }
}
