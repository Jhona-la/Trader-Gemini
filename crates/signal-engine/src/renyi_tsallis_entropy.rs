/// 🌌 MOTOR DE ENTROPÍA NO-EXTENSIVA DE RÉNYI Y TSALLIS (NON-EXTENSIVE QUANTUM ENTROPY ENGINE)
/// Cuantifica transiciones de fase críticas en el flujo de órdenes y detección de flash crashes (#266-#280).
/// $S_q(P) = \frac{1}{q-1} (1 - \sum p_i^q)$ y $R_\alpha(P) = \frac{1}{1-\alpha} \ln(\sum p_i^\alpha)$.

/// 🌌 OLA Ω123: Parámetros desacoplados y configurables para la entropía de Rényi y Tsallis
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenyiTsallisEntropyParams {
    /// Parámetro no-extensivo de Tsallis $q$ (default: 1.5)
    pub q_tsallis: f64,
    /// Orden generalizado de Rényi $\alpha$ (default: 2.0)
    pub alpha_renyi: f64,
    /// Cota inferior de entropía normalizada para rampa de orden (default: 0.40)
    pub entropy_threshold_low: f64,
    /// Cota superior de entropía normalizada para rampa de orden (default: 0.60)
    pub entropy_threshold_high: f64,
    /// Cota inferior de desequilibrio OBI para rampa de convicción (default: 0.15)
    pub obi_threshold_low: f64,
    /// Cota superior de desequilibrio OBI para rampa de convicción (default: 0.30)
    pub obi_threshold_high: f64,
}

impl Default for RenyiTsallisEntropyParams {
    fn default() -> Self {
        Self {
            q_tsallis: 1.5,
            alpha_renyi: 2.0,
            entropy_threshold_low: 0.40,
            entropy_threshold_high: 0.60,
            obi_threshold_low: 0.15,
            obi_threshold_high: 0.30,
        }
    }
}

impl RenyiTsallisEntropyParams {
    pub fn validate(&self) -> Result<(), String> {
        if !self.q_tsallis.is_finite() || !(0.01..=10.0).contains(&self.q_tsallis) {
            return Err(format!("Invalid q_tsallis: {}", self.q_tsallis));
        }
        if !self.alpha_renyi.is_finite() || !(0.01..=10.0).contains(&self.alpha_renyi) {
            return Err(format!("Invalid alpha_renyi: {}", self.alpha_renyi));
        }
        if !self.entropy_threshold_low.is_finite() || !(0.0..=1.0).contains(&self.entropy_threshold_low) {
            return Err(format!("Invalid entropy_threshold_low: {}", self.entropy_threshold_low));
        }
        if !self.entropy_threshold_high.is_finite() || !(0.0..=1.0).contains(&self.entropy_threshold_high) || self.entropy_threshold_high <= self.entropy_threshold_low {
            return Err(format!("Invalid entropy_threshold_high: {}", self.entropy_threshold_high));
        }
        if !self.obi_threshold_low.is_finite() || !(0.0..=1.0).contains(&self.obi_threshold_low) {
            return Err(format!("Invalid obi_threshold_low: {}", self.obi_threshold_low));
        }
        if !self.obi_threshold_high.is_finite() || !(0.0..=1.0).contains(&self.obi_threshold_high) || self.obi_threshold_high <= self.obi_threshold_low {
            return Err(format!("Invalid obi_threshold_high: {}", self.obi_threshold_high));
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct RenyiTsallisEntropyEngine {
    pub q_tsallis: f64,
    pub alpha_renyi: f64,
    pub params: RenyiTsallisEntropyParams,
    registry: Option<std::sync::Arc<omniscient_registry::OmniscientRegistry>>,
}

impl std::fmt::Debug for RenyiTsallisEntropyEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RenyiTsallisEntropyEngine")
            .field("q_tsallis", &self.q_tsallis)
            .field("alpha_renyi", &self.alpha_renyi)
            .field("params", &self.params)
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
        let mut params = RenyiTsallisEntropyParams::default();
        params.q_tsallis = safe_q.max(0.01);
        params.alpha_renyi = safe_alpha.max(0.01);
        Self {
            q_tsallis: safe_q.max(0.01),
            alpha_renyi: safe_alpha.max(0.01),
            params,
            registry: None,
        }
    }

    /// OLA Ω123: Constructor con parámetros desacoplados específicos
    pub fn with_params(mut self, params: RenyiTsallisEntropyParams) -> Self {
        if params.validate().is_ok() {
            self.q_tsallis = params.q_tsallis;
            self.alpha_renyi = params.alpha_renyi;
            self.params = params;
        }
        self
    }

    /// OLA Ω123: Actualización dinámica de parámetros
    pub fn update_params(&mut self, params: RenyiTsallisEntropyParams) {
        if params.validate().is_ok() {
            self.q_tsallis = params.q_tsallis;
            self.alpha_renyi = params.alpha_renyi;
            self.params = params;
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
    /// OLA Ω123: Voto con parámetros desacoplados configurables
    #[inline(always)]
    pub fn vote_with_params(tsallis_q: f64, obi: f64, params: &RenyiTsallisEntropyParams) -> f64 {
        if !tsallis_q.is_finite() || !obi.is_finite() || tsallis_q < 0.0 {
            return 0.0;
        }
        let safe_tsallis = tsallis_q;
        let safe_obi = obi;

        // #664 (G2-9): pertenencia CONTINUA (smoothstep C¹) en vez de
        // los gates duros `tsallis < 0.60 && |obi| > 0.15` — el voto
        // saltaba de 0 a ~(1−tsallis)·obi al cruzar cualquiera de los
        // dos umbrales.
        let smooth = |t: f64| {
            let t = t.clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let ent_high = params.entropy_threshold_high;
        let ent_band = (ent_high - params.entropy_threshold_low).max(1e-4);
        let orden = smooth((ent_high - safe_tsallis) / ent_band);

        let obi_low = params.obi_threshold_low;
        let obi_band = (params.obi_threshold_high - obi_low).max(1e-4);
        let desequilibrio = smooth((safe_obi.abs() - obi_low) / obi_band);
        let conviction = (1.0 - safe_tsallis) * safe_obi * orden * desequilibrio;
        conviction.clamp(-1.0, 1.0)
    }

    /// Voto puro legado (compartido por el camino global legado de `evaluate` y el
    /// escopado por moneda de `evaluate_for_coin`).
    #[inline(always)]
    #[allow(dead_code)]
    pub fn vote(tsallis_q: f64, obi: f64) -> f64 {
        Self::vote_with_params(tsallis_q, obi, &RenyiTsallisEntropyParams::default())
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
            // #664 / R5-A5: dirección CONTINUA (tanh) alineada con la familia
            // de sombras espectrales en lugar del signum duro.
            let signo = x.tanh();
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

        Self::vote_with_params(tsallis_q, obi, &self.params)
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
        // P5 (Camino Caliente): Lectura cero-alocaciones mediante get_scoped_val_strict (0 heap bytes)
        let tsallis = r.get_scoped_val_strict(Some(symbol), Some(coin_id), "tsallis_q_entropy");
        let obi = r.get_scoped_val_strict(Some(symbol), Some(coin_id), "order_book_imbalance");

        match (tsallis, obi) {
            (Some(t), Some(o)) => Self::vote_with_params(t, o, &self.params),
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
    use crate::voto_espectral::ESCALAS_VOTO;

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

    /// R5-A5: la sombra espectral es CONTINUA (C¹) a través de x=0 gracias a tanh
    /// (se erradicó el signum duro).
    #[test]
    fn qo_r5_a5_renyi_sombra_espectral_continua_tanh() {
        let engine = RenyiTsallisEntropyEngine::new(1.5, 2.0);
        let mut x_pos = [0.0; ESCALAS_VOTO];
        let mut x_neg = [0.0; ESCALAS_VOTO];
        let eps = 1e-4;
        x_pos[5] = eps;
        x_neg[5] = -eps;
        let v_pos = engine.voto_espectral(&x_pos).en_escala(5);
        let v_neg = engine.voto_espectral(&x_neg).en_escala(5);
        assert!(v_pos > 0.0);
        assert!(v_neg < 0.0);
        // Con tanh(eps) ≈ eps, la diferencia entre +eps y -eps es de orden O(eps),
        // no un salto O(1).
        assert!((v_pos - v_neg).abs() < 1e-3, "transición C¹ suave en x=0: {v_pos} vs {v_neg}");
    }
}
