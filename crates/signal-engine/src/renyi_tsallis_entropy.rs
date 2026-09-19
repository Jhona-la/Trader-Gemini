/// 🌌 MOTOR DE ENTROPÍA NO-EXTENSIVA DE RÉNYI Y TSALLIS (NON-EXTENSIVE QUANTUM ENTROPY ENGINE)
/// Cuantifica transiciones de fase críticas en el flujo de órdenes y detección de flash crashes (#266-#280).
/// $S_q(P) = \frac{1}{q-1} (1 - \sum p_i^q)$ y $R_\alpha(P) = \frac{1}{1-\alpha} \ln(\sum p_i^\alpha)$.

#[derive(Clone)]
pub struct RenyiTsallisEntropyEngine {
    pub q_tsallis: f64,
    pub alpha_renyi: f64,
    registry: Option<std::sync::Arc<omniscient_registry::OmniscientRegistry>>,
}

impl std::fmt::Debug for RenyiTsallisEntropyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenyiTsallisEntropyEngine")
            .field("q_tsallis", &self.q_tsallis)
            .field("alpha_renyi", &self.alpha_renyi)
            .finish()
    }
}

impl RenyiTsallisEntropyEngine {
    pub fn new(q_tsallis: f64, alpha_renyi: f64) -> Self {
        // FIX #681: Sanitizar q y alpha
        let safe_q = if q_tsallis.is_finite() && q_tsallis > 0.0 {
            q_tsallis
        } else {
            1.5
        };
        let safe_alpha = if alpha_renyi.is_finite() && alpha_renyi > 0.0 {
            alpha_renyi
        } else {
            2.0
        };
        Self {
            q_tsallis: safe_q.max(0.01),
            alpha_renyi: safe_alpha.max(0.01),
            registry: None,
        }
    }

    /// Calcula la entropía no-extensiva de Tsallis sobre una distribución de probabilidades discretas
    /// Retorna un valor $\ge 0.0$ acotado.
    #[inline(always)]
    pub fn calculate_tsallis_entropy(&self, probabilities: &[f64]) -> f64 {
        if probabilities.is_empty() {
            return 0.0;
        }

        let mut total_prob = 0.0;
        for &p in probabilities {
            if p.is_finite() && p > 0.0 {
                total_prob += p;
            }
        }

        if total_prob <= 0.0 {
            return 0.0;
        }

        // FASE 28 & BUG-605: Expansión de Taylor de 2º orden en el límite q -> 1.0 (Continuidad C²)
        let eps = self.q_tsallis - 1.0;
        if eps.abs() < 1e-3 {
            let mut shannon = 0.0;
            let mut second_order = 0.0;
            for &p in probabilities {
                if p.is_finite() && p > 0.0 {
                    let norm_p = p / total_prob;
                    let ln_p = norm_p.ln();
                    shannon -= norm_p * ln_p;
                    second_order += norm_p * ln_p * ln_p;
                }
            }
            let smooth_tsallis = shannon - 0.5 * eps * second_order;
            return smooth_tsallis.max(0.0);
        }

        let mut sum_p_q = 0.0;
        for &p in probabilities {
            if p.is_finite() && p > 0.0 {
                let norm_p = p / total_prob;
                sum_p_q += norm_p.powf(self.q_tsallis);
            }
        }

        ((1.0 - sum_p_q) / (self.q_tsallis - 1.0)).max(0.0)
    }

    /// Calcula la entropía generalizada de Rényi sobre una distribución de probabilidades discretas
    #[inline(always)]
    pub fn calculate_renyi_entropy(&self, probabilities: &[f64]) -> f64 {
        if probabilities.is_empty() {
            return 0.0;
        }

        let mut total_prob = 0.0;
        for &p in probabilities {
            if p.is_finite() && p > 0.0 {
                total_prob += p;
            }
        }

        if total_prob <= 0.0 {
            return 0.0;
        }

        // FASE 28 & BUG-605: Expansión de Taylor de 2º orden en el límite alpha -> 1.0 (Continuidad C²)
        let delta = 1.0 - self.alpha_renyi;
        if delta.abs() < 1e-3 {
            let mut shannon = 0.0;
            let mut second_order = 0.0;
            for &p in probabilities {
                if p.is_finite() && p > 0.0 {
                    let norm_p = p / total_prob;
                    let ln_p = norm_p.ln();
                    shannon -= norm_p * ln_p;
                    second_order += norm_p * ln_p * ln_p;
                }
            }
            let variance_info = second_order - (shannon * shannon);
            let smooth_renyi = shannon + 0.5 * (1.0 - self.alpha_renyi) * variance_info;
            return smooth_renyi.max(0.0);
        }

        let mut sum_p_alpha = 0.0;
        for &p in probabilities {
            if p.is_finite() && p > 0.0 {
                let norm_p = p / total_prob;
                sum_p_alpha += norm_p.powf(self.alpha_renyi);
            }
        }

        if sum_p_alpha <= 1e-12 {
            return 0.0;
        }

        ((1.0 / (1.0 - self.alpha_renyi)) * sum_p_alpha.ln()).max(0.0)
    }

    /// Voto puro (compartido por el camino global legado de `evaluate` y el
    /// escopado por moneda de `evaluate_for_coin`).
    #[inline(always)]
    fn vote(tsallis_q: f64, obi: f64) -> f64 {
        // FIX #681: Sanitizar lecturas de registro
        let safe_tsallis = if tsallis_q.is_finite() && tsallis_q >= 0.0 {
            tsallis_q
        } else {
            0.5
        };
        let safe_obi = if obi.is_finite() { obi } else { 0.0 };

        // Si la entropía es baja (orden estructurado en el flujo) y hay desequilibrio direccional, amplificar
        if safe_tsallis < 0.60 && safe_obi.abs() > 0.15 {
            let conviction = (1.0 - safe_tsallis) * safe_obi;
            conviction.clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

impl Default for RenyiTsallisEntropyEngine {
    fn default() -> Self {
        Self::new(1.5, 2.0)
    }
}

impl strategy_core::QuantumStrategy for RenyiTsallisEntropyEngine {
    fn name(&self) -> &str {
        "RenyiTsallisEntropyEngine"
    }

    fn init(
        &mut self,
        registry: std::sync::Arc<omniscient_registry::OmniscientRegistry>,
    ) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    fn evaluate(&self) -> f64 {
        let tsallis_q = self
            .registry
            .as_ref()
            .and_then(|r| r.get("tsallis_q_entropy", "RenyiTsallisEntropyEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.5);
        let obi = self
            .registry
            .as_ref()
            .and_then(|r| r.get("order_book_imbalance", "RenyiTsallisEntropyEngine"))
            .map(|p| p.get_value())
            .unwrap_or(0.0);

        Self::vote(tsallis_q, obi)
    }

    /// MOD2/7-009 (INFORME DECIMOCUARTO): esta estrategia no sobreescribía
    /// `evaluate_for_coin` y delegaba en `evaluate()`, que lee el registry
    /// GLOBAL — el voto de la moneda N se computaba con los datos de la
    /// ÚLTIMA moneda que escribió el global. Ahora: el voto usa los datos
    /// PER-COIN que el core publica en cada tick (`{sym}_key` / `c{id}:key`,
    /// ver `set_reg` en GodEngineCore::process_tick_dual). Sin datos
    /// per-coin, el slot 0 (BTC, escritor convencional del global) conserva
    /// el camino legado; cualquier otra moneda devuelve NEUTRO — voto
    /// neutralizado para no contaminar cross-coin (MOD2/7-009).
    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let Some(r) = self.registry.as_ref() else {
            return 0.0;
        };
        let scoped = |key: &str| -> Option<f64> {
            r.get(&format!("{}_{}", symbol, key), "RenyiTsallisEntropyEngine")
                .or_else(|| r.get(&format!("c{}:{}", coin_id, key), "RenyiTsallisEntropyEngine"))
                .map(|p| p.get_value())
                .filter(|v| v.is_finite())
        };

        match (scoped("tsallis_q_entropy"), scoped("order_book_imbalance")) {
            (Some(t), Some(o)) => Self::vote(t, o),
            _ if coin_id == 0 => self.evaluate(),
            _ => 0.0, // voto neutralizado para no contaminar cross-coin (MOD2/7-009)
        }
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tsallis_and_renyi_entropy_bounds() {
        let engine = RenyiTsallisEntropyEngine::new(1.5, 2.0);

        // Distribución uniforme (máxima entropía)
        let uniform = [0.25, 0.25, 0.25, 0.25];
        let tsallis_uniform = engine.calculate_tsallis_entropy(&uniform);
        let renyi_uniform = engine.calculate_renyi_entropy(&uniform);

        assert!(tsallis_uniform > 0.0);
        assert!(renyi_uniform > 0.0);

        // Distribución colapsada determinista (mínima entropía = 0)
        let collapsed = [1.0, 0.0, 0.0, 0.0];
        let tsallis_collapsed = engine.calculate_tsallis_entropy(&collapsed);
        let renyi_collapsed = engine.calculate_renyi_entropy(&collapsed);

        assert!((tsallis_collapsed - 0.0).abs() < 1e-6);
        assert!((renyi_collapsed - 0.0).abs() < 1e-6);

        // Entropía de uniforme > entropía de colapsada
        assert!(tsallis_uniform > tsallis_collapsed);
        assert!(renyi_uniform > renyi_collapsed);
    }

    #[test]
    fn test_shannon_limit_continuity() {
        let engine_limit = RenyiTsallisEntropyEngine::new(1.0, 1.0);
        let probs = [0.5, 0.5];
        let tsallis_shannon = engine_limit.calculate_tsallis_entropy(&probs);
        let renyi_shannon = engine_limit.calculate_renyi_entropy(&probs);
        let expected_shannon = (2.0_f64).ln(); // ln(2) = 0.693147...

        assert!((tsallis_shannon - expected_shannon).abs() < 1e-5);
        assert!((renyi_shannon - expected_shannon).abs() < 1e-5);
    }

    #[test]
    fn test_renyi_tsallis_nan_and_empty_immunity() {
        let engine = RenyiTsallisEntropyEngine::new(f64::NAN, f64::NAN);
        let empty: [f64; 0] = [];
        assert_eq!(engine.calculate_tsallis_entropy(&empty), 0.0);
        assert_eq!(engine.calculate_renyi_entropy(&empty), 0.0);

        let nan_probs = [f64::NAN, -1.0, 0.0];
        assert_eq!(engine.calculate_tsallis_entropy(&nan_probs), 0.0);
        assert_eq!(engine.calculate_renyi_entropy(&nan_probs), 0.0);
    }

    #[test]
    fn test_renyi_tsallis_engine_evaluate_with_registry() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        registry.set("book_bids_top5_ratio", 0.5);
        registry.set("entropy_threshold", 0.1);

        let mut engine = RenyiTsallisEntropyEngine::new(1.5, 2.0);
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let eval = strategy_core::QuantumStrategy::evaluate(&engine);
        assert!(eval.is_finite());
    }

    /// MOD2/7-009: una moneda sin datos per-coin NO hereda el global (que
    /// contiene los datos de la última moneda que escribió): voto NEUTRO.
    #[test]
    fn mod2_7_009_coin_sin_datos_per_coin_vota_neutral() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        // Sólo claves GLOBALES (p.ej. escritas por otra moneda):
        registry.set("tsallis_q_entropy", 0.2);
        registry.set("order_book_imbalance", -0.9);

        let mut engine = RenyiTsallisEntropyEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let v = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 5, "DOGEUSDT");
        assert_eq!(
            v, 0.0,
            "voto neutralizado para no contaminar cross-coin (MOD2/7-009)"
        );
        // El slot 0 (BTC) conserva el camino global legado.
        let v0 = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 0, "BTCUSDT");
        assert!(v0 < 0.0, "BTC sigue leyendo el global legado");
    }

    /// MOD2/7-009: con datos per-coin publicados, el voto usa los de SU
    /// símbolo aunque el global contenga otra cosa.
    #[test]
    fn mod2_7_009_voto_usa_datos_del_propio_simbolo() {
        let registry = std::sync::Arc::new(omniscient_registry::OmniscientRegistry::new());
        // El global quedó en manos de un vendedor con entropía baja:
        registry.set("tsallis_q_entropy", 0.2);
        registry.set("order_book_imbalance", -0.9);
        // El core escribe el contexto de DOGE (set_scoped + set_for_coin):
        registry.set_scoped("DOGEUSDT", "tsallis_q_entropy", 0.3);
        registry.set_scoped("DOGEUSDT", "order_book_imbalance", 0.7);

        let mut engine = RenyiTsallisEntropyEngine::default();
        assert!(strategy_core::QuantumStrategy::init(&mut engine, registry).is_ok());

        let v = strategy_core::QuantumStrategy::evaluate_for_coin(&engine, 5, "DOGEUSDT");
        assert!(
            v > 0.0,
            "DOGE debe votar con SU desequilibrio alcista, no con el global vendedor ({v})"
        );
    }
}
