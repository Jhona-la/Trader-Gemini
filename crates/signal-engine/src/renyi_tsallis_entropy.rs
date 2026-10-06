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
    pub(crate) fn vote(tsallis_q: f64, obi: f64) -> f64 {
        // FIX #681: Sanitizar lecturas de registro
        let safe_tsallis = if tsallis_q.is_finite() && tsallis_q >= 0.0 {
            tsallis_q
        } else {
            0.5
        };
        let safe_obi = if obi.is_finite() { obi } else { 0.0 };

        // #664 (G2-9): pertenencia CONTINUA (smoothstep C¹) en vez de
        // los gates duros `tsallis < 0.60 && |obi| > 0.15` — el voto
        // saltaba de 0 a ~(1−tsallis)·obi al cruzar cualquiera de los
        // dos umbrales. Rampa de orden: 1 con entropía ≤0.40, 0 en ≥0.60;
        // rampa de desequilibrio: 1 con |obi| ≥0.30, 0 en ≤0.15.
        let smooth = |t: f64| {
            let t = t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let orden = smooth((0.60 - safe_tsallis) / 0.20);
        let desequilibrio = smooth((safe_obi.abs() - 0.15) / 0.15);
        let conviction = (1.0 - safe_tsallis) * safe_obi * orden * desequilibrio;
        conviction.clamp(-1.0, 1.0)
    }
}

impl Default for RenyiTsallisEntropyEngine {
    fn default() -> Self {
        Self::new(1.5, 2.0)
    }
}

impl RenyiTsallisEntropyEngine {
    /// #615 (Ola 37) — VOTO ESPECTRAL de entropía: la INCERTIDUMBRE LOCAL
    /// de cada escala medida con la entropía de Tsallis sobre la
    /// distribución binaria de certeza {p, 1−p} donde p = 0.5 + |x(τ)|/2.
    /// Alta |x| ⇒ p→1 ⇒ entropía baja ⇒ CERTEZA: la escala vota fuerte en
    /// la dirección de su desplazamiento. Baja |x| ⇒ p→0.5 ⇒ entropía
    /// máxima ⇒ INCERTIDUMBRE: la escala se abstiene.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        &self,
        desplazamientos: &[f64; 32],
    ) -> crate::voto_espectral::VotoEspectral {
        crate::voto_espectral::VotoEspectral::desde_espectro(desplazamientos, |x| {
            if !x.is_finite() || x == 0.0 {
                return 0.0;
            }
            let p = (0.5 + x.abs() / 2.0).clamp(0.0, 1.0);
            let s_q = self.calculate_tsallis_entropy(&[p, 1.0 - p]);
            // Normalizar por la máxima entropía binaria S_q(0.5) para
            // acotar la certeza en [0,1].
            let s_max = self.calculate_tsallis_entropy(&[0.5, 0.5]).max(1e-9);
            let certeza = (1.0 - (s_q / s_max).clamp(0.0, 1.0)).clamp(0.0, 1.0);
            let signo = if x > 0.0 { 1.0 } else { -1.0 };
            (signo * certeza).clamp(-1.0, 1.0)
        })
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

#[cfg(test)]
mod qo_615_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_615_certeza_por_escala_y_antisimetria() {
        let engine = RenyiTsallisEntropyEngine::new(1.5, 2.0);
        let mut x = [0.0; ESCALAS_VOTO];
        for (k, v) in x.iter_mut().enumerate() {
            *v = 0.1 * (k as f64 - 15.5);
        }
        let voto = engine.voto_espectral(&x);
        // Antisimetría.
        for kk in 0..ESCALAS_VOTO {
            let idx = ESCALAS_VOTO - 1 - kk;
            if (x[kk] + x[idx]).abs() < 1e-12 && x[kk] != 0.0 {
                assert!(
                    (voto.en_escala(kk) + voto.en_escala(idx)).abs() < 1e-10,
                    "antisimetría en {}: {} vs {}",
                    kk,
                    voto.en_escala(kk),
                    voto.en_escala(idx)
                );
            }
        }
        // MAYOR |x| ⇒ MAYOR certeza (menos entropía local).
        let certeza_pequena = voto.en_escala(16).abs(); // |x| = 0.05
        let certeza_grande = voto.en_escala(0).abs(); // |x| = 1.55
        assert!(
            certeza_grande > certeza_pequena,
            "desplazamiento fuerte ⇒ más certeza: {} vs {}",
            certeza_grande,
            certeza_pequena
        );
        // Cero desplazamiento ⇒ voto 0 (incertidumbre total).
        let cero = engine.voto_espectral(&[0.0; ESCALAS_VOTO]);
        assert_eq!(cero.dominante(), None);
    }
}

#[cfg(test)]
mod qo_664_tests {
    use super::*;

    /// #664 (G2-9): la convicción Rényi-Tsallis es CONTINUA en ambos
    /// umbrales — antes saltaba de 0 a (1−tsallis)·obi al cruzar
    /// |obi|=0.15 o tsallis=0.60.
    #[test]
    fn qo_664_renyi_continua_en_los_umbrales() {
        // Piso OBI: antes v(0.155) saltaba a (1−0.55)·0.155 ≈ 0.07 con
        // v(0.145)=0; ahora la rampa [0.15, 0.30] hace el paso diminuto.
        let bajo_piso = RenyiTsallisEntropyEngine::vote(0.55, 0.145);
        let sobre_piso = RenyiTsallisEntropyEngine::vote(0.55, 0.155);
        assert!((sobre_piso - bajo_piso).abs() < 0.01, "sin salto en el piso: {bajo_piso} {sobre_piso}");
        // Dentro de la rampa: monótono y no nulo.
        let interior1 = RenyiTsallisEntropyEngine::vote(0.55, 0.18);
        let interior2 = RenyiTsallisEntropyEngine::vote(0.55, 0.25);
        assert!(interior1 > 0.0 && interior2 > interior1, "monotono en |obi|: {interior1} {interior2}");
        // Umbral de entropía 0.60: rampa [0.40, 0.60] — sin salto.
        let bajo_h = RenyiTsallisEntropyEngine::vote(0.58, 0.30);
        let sobre_h = RenyiTsallisEntropyEngine::vote(0.62, 0.30);
        assert!(bajo_h >= 0.0 && (bajo_h - sobre_h).abs() < 0.01, "sin salto en tsallis=0.60: {bajo_h} {sobre_h}");
    }
}
