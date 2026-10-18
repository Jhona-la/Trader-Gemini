use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// Parámetros adaptativos de la compuerta perceptrón.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerceptronGateParams {
    /// Ganancia de saturación suave tanh de la señal emitida.
    pub ganancia: f64,
    /// Piso de exploración mínima continua (evita bloqueo cognitivo tras pérdidas).
    pub min_exploration_floor: f64,
    /// Umbral de activación para la rampa smoothstep.
    pub activation_threshold: f64,
    /// Tasa de aprendizaje base para la actualización Hebbiana por resultado R.
    pub base_learning_rate: f64,
}

impl Default for PerceptronGateParams {
    fn default() -> Self {
        Self {
            ganancia: 2.5,
            min_exploration_floor: 0.15,
            activation_threshold: 0.5,
            base_learning_rate: 0.05,
        }
    }
}

impl PerceptronGateParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.ganancia.is_finite()
            || !self.min_exploration_floor.is_finite()
            || !self.activation_threshold.is_finite()
            || !self.base_learning_rate.is_finite()
        {
            return Err("PerceptronGateParams: valores no finitos");
        }
        if self.ganancia <= 0.0
            || self.min_exploration_floor < 0.0
            || self.min_exploration_floor >= 1.0
            || self.activation_threshold < 0.0
            || self.base_learning_rate <= 0.0
        {
            return Err("PerceptronGateParams: cotas inválidas");
        }
        Ok(())
    }
}

/// 🧠 ALGORITMO #96: MOTOR DE COMPUERTA PERCEPTRÓN HEBBIANA ADAPTATIVA (PERCEPTRON GATE ENGINE)
/// Perceptrón binario ultra-rápido de ciclo único de CPU con regla de actualización Hebbiana adaptativa,
/// regulando el paso atómico de señales de compra/venta en función de los PnL recientes.
#[derive(Clone)]
#[repr(C, align(64))]
pub struct PerceptronGateEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    params: PerceptronGateParams,
}

impl Default for PerceptronGateEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for PerceptronGateEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerceptronGateEngine")
            .field("params", &self.params)
            .finish()
    }
}

impl PerceptronGateEngine {
    pub fn new() -> Self {
        Self {
            registry: None,
            params: PerceptronGateParams::default(),
        }
    }

    pub fn with_params(params: PerceptronGateParams) -> Self {
        let safe_params = if params.validate().is_ok() {
            params
        } else {
            PerceptronGateParams::default()
        };
        Self {
            registry: None,
            params: safe_params,
        }
    }

    pub fn update_params(&mut self, params: PerceptronGateParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    pub fn params(&self) -> &PerceptronGateParams {
        &self.params
    }

    /// Inferencia del tensor parametrizada, simétrica para Long y Short
    #[inline(always)]
    pub fn infer_with_params(signal_score: f64, weight: f64, params: &PerceptronGateParams) -> f64 {
        if !signal_score.is_finite() || !weight.is_finite() || signal_score == 0.0 {
            return 0.0;
        }
        let abs_score = signal_score.abs();
        let activation = abs_score * weight;
        let t = (activation - params.activation_threshold).clamp(0.0, 1.0);
        let gate_strength = params.min_exploration_floor
            + (1.0 - params.min_exploration_floor) * (t * t * (3.0 - 2.0 * t));
        (signal_score * params.ganancia).tanh() * gate_strength
    }

    /// Inferencia del tensor sin actualizar el peso (para el Engine Core), simétrica para Long y Short
    #[inline(always)]
    pub fn infer(signal_score: f64, weight: f64) -> f64 {
        Self::infer_with_params(signal_score, weight, &PerceptronGateParams::default())
    }

    /// #620 (Ola 42) — VOTO ESPECTRAL de la compuerta perceptrón: la
    /// compuerta se abre o cierra a CADA escala. El desplazamiento x(τ_k)
    /// ES la señal que la compuerta evalúa; el PESO espectral es un perfil
    /// que favorece las escalas intermedias (la señal más limpia vive
    /// lejos del ruido sub-segundo y de la inercia macro): peso = 1.0 en
    /// la banda central [k=8..23], decae hacia 0.5 en los extremos. La
    /// compuerta aplica `infer(x(τ), peso_espectral)` — misma forma que
    /// el vivo (tanh con piso 0.15 de exploración), resolución-en-escala.
    /// Observacional: el voto vivo queda bit a bit (T-1 cero).
    pub fn voto_espectral(
        desplazamientos: &[f64; 32],
    ) -> crate::voto_espectral::VotoEspectral {
        let mut por_escala = [0.0f64; 32];
        for k in 0..32 {
            let x = desplazamientos[k];
            if !x.is_finite() || x == 0.0 {
                continue;
            }
            // F1-125: Perfil espectral continuo anclado a la banda física operativa [18..=22] ([30 s, 12 h]).
            // Decae linealmente de forma C^0 hacia 0.5 en microestructura sub-30s (k < 18) y macroestructura secular (k > 22).
            let peso = if (18..=22).contains(&k) {
                1.0
            } else if k < 18 {
                0.5 + 0.5 * (k as f64 / 18.0)
            } else {
                0.5 + 0.5 * (((31 - k) as f64 / 9.0).clamp(0.0, 1.0))
            };
            por_escala[k] = Self::infer(x, peso);
        }
        crate::voto_espectral::VotoEspectral::desde_arr(&por_escala)
    }

    /// Peso del perceptrón por el resultado de las operaciones cerradas con parámetros adaptativos.
    #[inline(always)]
    pub fn update_weight_with_params(
        weight: &mut f64,
        resultado_r: f64,
        std_dev: f64,
        params: &PerceptronGateParams,
    ) {
        let safe_std = if std_dev.is_finite() && std_dev >= 0.0 {
            std_dev
        } else {
            0.01
        };
        let current_w = if weight.is_finite() { *weight } else { 1.0 };
        let learning_rate = (params.base_learning_rate / (1.0 + safe_std * 100.0)).clamp(0.001, 0.05);

        if !weight.is_finite() {
            *weight = 1.0;
        }

        if resultado_r.is_finite() {
            // El exponente se acota sólo para que `exp` no desborde; el
            // recorte del peso a [0,25; 2] satura mucho antes.
            let exponente = (learning_rate * resultado_r).clamp(-50.0, 50.0);
            *weight = (current_w.clamp(0.25, 2.0) * exponente.exp()).clamp(0.25, 2.0);
        }
    }

    /// Peso del perceptrón por el resultado de las operaciones cerradas (compatibilidad canónica).
    #[inline(always)]
    pub fn update_weight(weight: &mut f64, resultado_r: f64, std_dev: f64) {
        Self::update_weight_with_params(weight, resultado_r, std_dev, &PerceptronGateParams::default());
    }
}

impl QuantumStrategy for PerceptronGateEngine {
    fn name(&self) -> &str {
        "PerceptronGateEngine"
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
        // F1-124: Erradicada lectura inerte de 'perceptron_candidate_signal' (sin escritor en producción),
        // consultando directamente 'order_flow_imbalance' y fallback a 'alpha_signal'.
        let signal_score = registry
            .get_scoped_parameter(
                sym_opt,
                cid_opt,
                "order_flow_imbalance",
                "PerceptronGateEngine",
            )
            .or_else(|| {
                registry.get_scoped_parameter(
                    sym_opt,
                    cid_opt,
                    "alpha_signal",
                    "PerceptronGateEngine",
                )
            })
            .map(|p| p.get_value())
            .unwrap_or(0.0);
        // F1-098: Leer peso Hebbiano con aislamiento estricto por moneda/símbolo (hebbian_weight),
        // sin caer a la clave global compartida perceptron_hebbian_weight que mezclaba el PnL de activos ajenos.
        let weight = registry.get_scoped_val_strict_or(
            sym_opt,
            cid_opt,
            "hebbian_weight",
            1.0,
        );
        if !signal_score.is_finite() || !weight.is_finite() {
            return 0.0;
        }

        // F1-124: Resolución dinámica de ganancia si existe; min_exploration_floor proviene de self.params
        // (erradicando la sonda inerte de 'perceptron_min_exploration_floor' sin escritor).
        let custom_ganancia = registry
            .get_scoped_parameter(sym_opt, cid_opt, "perceptron_ganancia", "PerceptronGateEngine")
            .map(|p| p.get_value())
            .unwrap_or(self.params.ganancia);

        let effective_params = if custom_ganancia.is_finite() && custom_ganancia > 0.0 {
            PerceptronGateParams {
                ganancia: custom_ganancia,
                ..self.params
            }
        } else {
            self.params
        };

        Self::infer_with_params(signal_score, weight, &effective_params)
    }

    fn horizon(&self) -> strategy_core::TradeHorizon {
        strategy_core::TradeHorizon::Continuous
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron_gate_symmetric_long_and_short() {
        let long_out = PerceptronGateEngine::infer(0.8, 1.0);
        assert!(long_out > 0.0);

        let short_out = PerceptronGateEngine::infer(-0.8, 1.0);
        assert!(short_out < 0.0);
        assert!(
            (long_out + short_out).abs() < 1e-6,
            "Debe ser perfectamente simétrico"
        );
    }

    #[test]
    fn test_perceptron_gate_hebbian_update_and_nan_immunity() {
        let mut weight = 1.0;
        PerceptronGateEngine::update_weight(&mut weight, 50.0, 0.01);
        assert!(weight > 1.0);

        PerceptronGateEngine::update_weight(&mut weight, -50.0, 0.01);
        assert!(weight <= 1.05);

        // NaN immunity
        let mut nan_w = f64::NAN;
        PerceptronGateEngine::update_weight(&mut nan_w, f64::NAN, f64::NAN);
        assert!(nan_w.is_finite());
        assert!((0.1..=2.0).contains(&nan_w));
    }

    #[test]
    fn test_perceptron_gate_evaluate_with_registry() {
        let mut engine = PerceptronGateEngine::new();
        assert_eq!(engine.evaluate(), 0.0);

        let registry = Arc::new(OmniscientRegistry::new());
        registry.set("perceptron_candidate_signal", 0.9);
        registry.set("perceptron_hebbian_weight", 1.2);

        engine.init(registry).expect("init should succeed");
        let signal = engine.evaluate();
        assert!(signal > 0.0);
        assert!((-1.0..=1.0).contains(&signal));
    }
}

#[cfg(test)]
mod qo_620_tests {
    use super::*;
    use crate::voto_espectral::ESCALAS_VOTO;

    #[test]
    fn qo_620_compuerta_espectral_favorece_banda_central() {
        // Señal UNIFORME: la compuerta debe dejar pasar MÁS la banda
        // central (peso 1.0) que los extremos (peso 0.5).
        let mut x = [0.0; ESCALAS_VOTO];
        for v in x.iter_mut() {
            *v = 0.3; // señal moderada uniforme
        }
        let voto = PerceptronGateEngine::voto_espectral(&x);
        let central = voto.en_escala(15).abs(); // banda central: peso 1.0
        let extremo = voto.en_escala(2).abs(); // extremo rápido: peso ≈ 0.625
        assert!(
            central >= extremo,
            "la compuerta favorece la banda central: {} vs {}",
            central,
            extremo
        );
        // Antisimetría (la compuerta es simétrica Long/Short).
        let mut x_neg = x;
        for v in &mut x_neg {
            *v = -*v;
        }
        let voto_neg = PerceptronGateEngine::voto_espectral(&x_neg);
        for k in 0..ESCALAS_VOTO {
            assert!(
                (voto.en_escala(k) + voto_neg.en_escala(k)).abs() < 1e-10,
                "antisimetría en {}",
                k
            );
        }
        // Señal débil ⇒ la compuerta NO se cierra del todo: con la
        // dirección continua (#664/G2-4) el voto es pequeño pero NO CERO
        // (la compuerta mantiene curiosidad proporcional a la señal; el
        // piso duro 0.15 saltaba de 0 a 0.15 al cruzar score=0).
        let mut x_debil = [0.0; ESCALAS_VOTO];
        for v in x_debil.iter_mut() {
            *v = 0.01;
        }
        let voto_debil = PerceptronGateEngine::voto_espectral(&x_debil);
        assert!(
            voto_debil.en_escala(15) > 0.0,
            "senal debil deja voto no nulo: {}",
            voto_debil.en_escala(15)
        );
        assert!(
            voto_debil.en_escala(15) < 0.15,
            "senal debil no dispara el piso: {}",
            voto_debil.en_escala(15)
        );
        // Señal nula ⇒ voto 0.
        let cero = PerceptronGateEngine::voto_espectral(&[0.0; ESCALAS_VOTO]);
        assert_eq!(cero.dominante(), None);
    }

    /// H2-6 (RONDA 3): Con ganancia 2.5, la compuerta gradúa de manera continua y suave
    /// sin colapsar a ±1 en la región operativa típica (|signal| ∈ [0.2, 0.8]).
    #[test]
    fn h2_6_graduacion_continua_sin_saturacion_prematura() {
        // Con peso unitario y score moderado 0.3:
        // Antes con ganancia 10.0: (0.3 * 10.0).tanh() = (3.0).tanh() ≈ 0.995 (SATURADO)
        // Ahora con ganancia 2.5: (0.3 * 2.5).tanh() = (0.75).tanh() ≈ 0.635 (GRADUADO)
        // R4-C4: el gate ahora es smoothstep C¹ con piso 0.15 (el
        // tanh((a−0.5)·5).clamp tenía kink C⁰ en a≈0.53).
        let out_moderado = PerceptronGateEngine::infer(0.3, 1.0);
        let t = (0.3f64 * 1.0 - 0.5).clamp(0.0, 1.0);
        let gate_strength = 0.15 + 0.85 * (t * t * (3.0 - 2.0 * t));
        let tanh_esperado = (0.3 * 2.5f64).tanh();
        assert!((out_moderado - tanh_esperado * gate_strength).abs() < 1e-10);
        assert!(
            out_moderado < 0.90,
            "el score moderado 0.3 no debe saturar a ±1 (fue {out_moderado})"
        );
        // Piso de exploración intacto: score bajo ⇒ gate ≈ 0.15.
        let out_bajo = PerceptronGateEngine::infer(0.05, 1.0);
        let gate_bajo = 0.15 + 0.85 * (0.0f64 * 0.0 * 3.0);
        assert!(
            (out_bajo - (0.05 * 2.5f64).tanh() * gate_bajo).abs() < 1e-10,
            "score bajo conserva el piso de exploración 0.15 (fue {out_bajo})"
        );

        // Monotonía estricta en el rango operativo:
        let out_01 = PerceptronGateEngine::infer(0.1, 1.0);
        let out_03 = PerceptronGateEngine::infer(0.3, 1.0);
        let out_06 = PerceptronGateEngine::infer(0.6, 1.0);
        let out_09 = PerceptronGateEngine::infer(0.9, 1.0);
        assert!(out_01 < out_03, "monotonía 0.1 < 0.3: {out_01} vs {out_03}");
        assert!(out_03 < out_06, "monotonía 0.3 < 0.6: {out_03} vs {out_06}");
        assert!(out_06 < out_09, "monotonía 0.6 < 0.9: {out_06} vs {out_09}");
    }
}
