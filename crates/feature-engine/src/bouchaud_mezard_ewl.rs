//! Condensación de Bouchaud-Mézard y Teoría de Juegos Cuánticos de Eisert-Wilkens-Lewenstein (EWL) (FASE 25)
//!
//! # QUÉ:
//! Subsistema dual que integra:
//! 1. El modelo de física estadística de Bouchaud-Mézard para transiciones de fase
//!    de condensación de liquidez y colas pesadas de Pareto en el libro de órdenes.
//! 2. El protocolo de teoría de juegos cuánticos de Eisert-Wilkens-Lewenstein (EWL)
//!    con puertas unitarias en $SU(2)$ y operador de entrelazamiento $\hat{J}$ para
//!    optimización de equilibrio en provisión de liquidez Maker.
//!
//! # POR QUÉ:
//! En mercados financieros con alta participación algorítmica, los niveles de precios
//! sufren condensaciones súbitas de volumen (toda la liquidez se concentra en un único nivel
//! o participante, provocando vacíos de liquidez en niveles adyacentes).
//! Además, la interacción entre creadores de mercado (Makers) y agresores (Takers) no es
//! un juego estático clásico de suma cero: cuando existe entrelazamiento informacional $\gamma > 0$,
//! las estrategias cuánticas unitarias permiten romper el dilema del prisionero clásico,
//! estabilizando el spread sin incurrir en guerras predatorias destructivas de comisiones.
//!
//! # PARA QUÉ:
//! 1. Predecir congelamientos de liquidez mediante el exponente de Pareto $\mu = 1 + \frac{J}{\sigma^2/2}$.
//! 2. Vetar la colocación de órdenes Maker en regímenes condensados ($\mu \le 1.0$).
//! 3. Resolver la estrategia óptima de cotización cuántica en $SU(2)$ superando los equilibrios clásicos subóptimos.
//!
//! # CÓMO:
//! Dinámica Langevin de Bouchaud-Mézard en stack para $N \le 8$ niveles de profundidad,
//! y contracción unitaria de 4 estados cuánticos $|00\rangle, |01\rangle, |10\rangle, |11\rangle$
//! mediante álgebra de Pauli $\sigma_x$ y matrices de rotación $SU(2)$.
//!
//! # CUÁNDO:
//! En cada decisión de asignación de liquidez pasiva Maker y dimensionamiento de spread.
//!
//! # DÓNDE:
//! `crates/feature-engine/src/bouchaud_mezard_ewl.rs`.
//!
//! # QUIÉN:
//! Quant de Microestructura (Rol 2) & Físico Cuántico (Rol 8).

#![allow(clippy::all)]

use core::f64::consts::PI;

/// Número máximo de niveles de libro modelados en la red de Bouchaud-Mézard.
pub const MAX_BM_LEVELS: usize = 8;

/// Parámetros configurables del modelo de Bouchaud-Mézard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BouchaudMezardParams {
    /// Tasa de intercambio o difusión de liquidez entre niveles adyacentes $J$.
    pub coupling_j: f64,
    /// Volatilidad de los choques estocásticos de volumen $\sigma$.
    pub noise_sigma: f64,
    /// Deriva o tasa de consumo medio de liquidez $m$.
    pub drift_m: f64,
}

impl Default for BouchaudMezardParams {
    fn default() -> Self {
        Self {
            coupling_j: 0.15,
            noise_sigma: 0.30,
            drift_m: 0.0,
        }
    }
}

/// Régimen físico de liquidez derivado de Bouchaud-Mézard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiquidityCondensationRegime {
    /// Fase condensada ($\mu \le 1.0$): un solo nivel absorbe casi toda la liquidez (riesgo extremo de vacío).
    CondensedMonopoly,
    /// Fase transicional ($1.0 < \mu \le 2.0$): colas pesadas de volumen (régimen de Pareto).
    HeavyTailedCritical,
    /// Fase dispersa fluida ($\mu > 2.0$): liquidez ergódica distribuida uniformemente.
    DispersedFluid,
}

/// Informe analítico de la dinámica de Bouchaud-Mézard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BouchaudMezardReport {
    /// Exponente de ley de potencia de Pareto $\mu = 1 + \frac{J}{\sigma^2/2}$.
    pub pareto_mu: f64,
    /// Fracción de liquidez concentrada en el nivel dominante (Parámetro de Orden de Bose-Einstein).
    pub max_fraction: f64,
    /// Régimen de condensación clasificado.
    pub regime: LiquidityCondensationRegime,
    /// Veto de cotización pasiva Maker por riesgo de condensación singular.
    pub maker_veto: bool,
}

/// Motor de física estadística de Bouchaud-Mézard.
#[derive(Debug, Clone)]
pub struct BouchaudMezardEngine {
    params: BouchaudMezardParams,
    levels: usize,
    /// Volúmenes de liquidez en cada nivel $w_i$.
    weights: [f64; MAX_BM_LEVELS],
}

impl BouchaudMezardEngine {
    /// Crea un nuevo motor para un número de niveles $N \le 8$.
    pub fn new(levels: usize) -> Self {
        let n = levels.clamp(2, MAX_BM_LEVELS);
        let mut weights = [0.0; MAX_BM_LEVELS];
        let init_val = 1.0 / (n as f64);
        for i in 0..n {
            weights[i] = init_val;
        }
        Self {
            params: BouchaudMezardParams::default(),
            levels: n,
            weights,
        }
    }

    /// Asigna parámetros personalizados de acoplamiento y ruido.
    pub fn with_params(mut self, params: BouchaudMezardParams) -> Self {
        self.params = params;
        self
    }

    /// Actualiza el estado con volúmenes observados en el libro L2 y evalúa el exponente $\mu$.
    pub fn update(&mut self, observed_volumes: &[f64]) -> BouchaudMezardReport {
        let n = self.levels;
        let mut sum = 0.0;
        for i in 0..n {
            let v = if i < observed_volumes.len()
                && observed_volumes[i].is_finite()
                && observed_volumes[i] >= 0.0
            {
                observed_volumes[i]
            } else {
                1.0
            };
            self.weights[i] = 0.80 * self.weights[i] + 0.20 * v;
            sum += self.weights[i];
        }

        // Normalización para calcular fracciones relativas de liquidez
        let inv_sum = if sum > 1e-12 {
            1.0 / sum
        } else {
            1.0 / (n as f64)
        };
        let mut max_frac = 0.0;
        for i in 0..n {
            let frac = self.weights[i] * inv_sum;
            if frac > max_frac {
                max_frac = frac;
            }
        }

        // Cálculo analítico del exponente de Pareto-Zipf: \mu = 1 + J / (\sigma^2 / 2)
        let s2 = self.params.noise_sigma * self.params.noise_sigma;
        let effective_s2 = if s2 > 1e-6 { s2 } else { 1e-6 };
        let pareto_mu = 1.0 + (self.params.coupling_j / (effective_s2 * 0.5));

        let regime = if pareto_mu <= 1.0 || max_frac > 0.65 {
            LiquidityCondensationRegime::CondensedMonopoly
        } else if pareto_mu <= 2.0 || max_frac > 0.40 {
            LiquidityCondensationRegime::HeavyTailedCritical
        } else {
            LiquidityCondensationRegime::DispersedFluid
        };

        let maker_veto = regime == LiquidityCondensationRegime::CondensedMonopoly;

        BouchaudMezardReport {
            pareto_mu,
            max_fraction: max_frac,
            regime,
            maker_veto,
        }
    }
}

/// Estrategia cuántica unitaria $U(\theta, \phi) \in SU(2)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantumStrategySU2 {
    /// Ángulo de rotación polar $\theta \in [0, \pi]$.
    pub theta: f64,
    /// Ángulo de fase azimutal $\phi \in [0, \pi/2]$.
    pub phi: f64,
}

impl QuantumStrategySU2 {
    /// Estrategia clásica "Cooperar / Cotizar Maker": $\theta = 0, \phi = 0$ (Identidad $I$).
    pub const fn cooperate_maker() -> Self {
        Self {
            theta: 0.0,
            phi: 0.0,
        }
    }

    /// Estrategia clásica "Defeccionar / Agredir Taker": $\theta = \pi, \phi = 0$ (Puerta $X = \sigma_x$).
    pub const fn defect_taker() -> Self {
        Self {
            theta: PI,
            phi: 0.0,
        }
    }

    /// Estrategia cuántica pura de superposición Hadamard-like: $\theta = \pi/2, \phi = \pi/2$.
    pub const fn quantum_optimal() -> Self {
        Self {
            theta: PI * 0.5,
            phi: PI * 0.5,
        }
    }
}

/// Resultado de la resolución de un juego cuántico EWL.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EwlQuantumGameReport {
    /// Pago esperado para el Agente A (nuestro bot Maker) $\Pi_A$.
    pub payoff_a: f64,
    /// Pago esperado para el Agente B (contraparte del mercado) $\Pi_B$.
    pub payoff_b: f64,
    /// Probabilidad del estado de cooperación armónica $|00\rangle$.
    pub p_cooperate_cooperate: f64,
    /// Probabilidad del estado de agresión mutua destructiva $|11\rangle$.
    pub p_defect_defect: f64,
    /// Ventaja de retorno cuántico respecto al equilibrio de Nash clásico $\Delta \Pi_Q$.
    pub quantum_advantage: f64,
    /// Grado de entrelazamiento de información $\gamma \in [0, \pi/2]$.
    pub entanglement_gamma: f64,
}

/// Motor de Teoría de Juegos Cuánticos de Eisert-Wilkens-Lewenstein en $SU(2)$.
#[derive(Debug, Clone)]
pub struct EwlQuantumGameEngine {
    /// Parámetro de entrelazamiento cuántico del mercado $\gamma \in [0, \pi/2]$.
    entanglement_gamma: f64,
    /// Matriz de pagos clásicos [R, S, T, P]:
    /// R: Reward (Cooperación mutua), S: Sucker (Maker agredido),
    /// T: Temptation (Taker agresor), P: Penalty (Guerra de subasta).
    payoffs: [f64; 4],
}

impl Default for EwlQuantumGameEngine {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl EwlQuantumGameEngine {
    /// Crea un nuevo motor EWL con entrelazamiento $\gamma$.
    /// Payoffs en basis points (bps) para Binance Futures:
    /// R: +3.0 bps (ambos capturan rebates armónicos),
    /// S: -2.0 bps (agredido por flujo tóxico),
    /// T: +4.0 bps (agresor capturando liquidez),
    /// P: -5.0 bps (guerra de subasta cruzada pagando 2x comisiones Taker).
    pub fn new(gamma: f64) -> Self {
        let valid_gamma = if gamma.is_finite() {
            gamma.clamp(0.0, PI * 0.5)
        } else {
            0.0
        };
        Self {
            entanglement_gamma: valid_gamma,
            payoffs: [3.0, -2.0, 4.0, -5.0],
        }
    }

    /// Asigna el nivel de entrelazamiento de información del mercado $\gamma$.
    pub fn set_entanglement(&mut self, gamma: f64) {
        if gamma.is_finite() {
            self.entanglement_gamma = gamma.clamp(0.0, PI * 0.5);
        }
    }

    /// Resuelve el protocolo cuántico EWL para dos estrategias $U_A, U_B \in SU(2)$.
    ///
    /// Amplitud de estado:
    /// $|\psi_f\rangle = \hat{J}^\dagger (U_A \otimes U_B) \hat{J} |00\rangle$
    pub fn play(
        &self,
        strat_a: QuantumStrategySU2,
        strat_b: QuantumStrategySU2,
    ) -> EwlQuantumGameReport {
        let gamma = self.entanglement_gamma;
        let c_g = (gamma * 0.5).cos();
        let s_g = (gamma * 0.5).sin();

        // Parámetros de U_A
        let c_a = (strat_a.theta * 0.5).cos();
        let s_a = (strat_a.theta * 0.5).sin();
        let phi_a = strat_a.phi;

        // Parámetros de U_B
        let c_b = (strat_b.theta * 0.5).cos();
        let s_b = (strat_b.theta * 0.5).sin();
        let phi_b = strat_b.phi;

        // Amplitudes complejas de U_A y U_B:
        // U = [[u00, u01], [u10, u11]]
        // u00 = cos(theta/2) * e^{i*phi}
        // u01 = i * sin(theta/2)
        // u10 = i * sin(theta/2)
        // u11 = cos(theta/2) * e^{-i*phi}

        // Elementos de U_A: (real, imag)
        let ua_00 = (c_a * phi_a.cos(), c_a * phi_a.sin());
        let ua_01 = (0.0, s_a);
        let ua_10 = (0.0, s_a);
        let ua_11 = (c_a * (-phi_a).cos(), c_a * (-phi_a).sin());

        // Elementos de U_B:
        let ub_00 = (c_b * phi_b.cos(), c_b * phi_b.sin());
        let ub_01 = (0.0, s_b);
        let ub_10 = (0.0, s_b);
        let ub_11 = (c_b * (-phi_b).cos(), c_b * (-phi_b).sin());

        // Multiplicación compleja: (a_r, a_i) * (b_r, b_i)
        let cmul = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) {
            (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
        };
        let cadd = |a: (f64, f64), b: (f64, f64)| -> (f64, f64) { (a.0 + b.0, a.1 + b.1) };

        // Estado tras \hat{J} |00>: |\psi_1> = cos(gamma/2)|00> + i*sin(gamma/2)|11>
        let psi1_00 = (c_g, 0.0);
        let psi1_11 = (0.0, s_g);

        // Estado tras U_A \otimes U_B:
        // |\psi_2> = (U_A \otimes U_B) |\psi_1>
        // Para cada componente |ij>:
        // |00>: ua_00 * ub_00 * psi1_00 + ua_01 * ub_01 * psi1_11
        let psi2_00 = cadd(
            cmul(cmul(ua_00, ub_00), psi1_00),
            cmul(cmul(ua_01, ub_01), psi1_11),
        );
        // |01>: ua_00 * ub_10 * psi1_00 + ua_01 * ub_11 * psi1_11
        let psi2_01 = cadd(
            cmul(cmul(ua_00, ub_10), psi1_00),
            cmul(cmul(ua_01, ub_11), psi1_11),
        );
        // |10>: ua_10 * ub_00 * psi1_00 + ua_11 * ub_01 * psi1_11
        let psi2_10 = cadd(
            cmul(cmul(ua_10, ub_00), psi1_00),
            cmul(cmul(ua_11, ub_01), psi1_11),
        );
        // |11>: ua_10 * ub_01 * psi1_00 + ua_11 * ub_11 * psi1_11
        let psi2_11 = cadd(
            cmul(cmul(ua_10, ub_01), psi1_00),
            cmul(cmul(ua_11, ub_11), psi1_11),
        );

        // Estado final tras \hat{J}^\dagger:
        // \hat{J}^\dagger = cos(gamma/2) I - i*sin(gamma/2) \sigma_x \otimes \sigma_x
        // psi_f_00 = cos(gamma/2) * psi2_00 - i*sin(gamma/2) * psi2_11
        // (-i * s_g) * (r, i) = (s_g * i, -s_g * r)
        let psi_f_00 = cadd(
            (c_g * psi2_00.0, c_g * psi2_00.1),
            (s_g * psi2_11.1, -s_g * psi2_11.0),
        );
        // psi_f_01 = cos(gamma/2) * psi2_01 - i*sin(gamma/2) * psi2_10
        let psi_f_01 = cadd(
            (c_g * psi2_01.0, c_g * psi2_01.1),
            (s_g * psi2_10.1, -s_g * psi2_10.0),
        );
        // psi_f_10 = cos(gamma/2) * psi2_10 - i*sin(gamma/2) * psi2_01
        let psi_f_10 = cadd(
            (c_g * psi2_10.0, c_g * psi2_10.1),
            (s_g * psi2_01.1, -s_g * psi2_01.0),
        );
        // psi_f_11 = cos(gamma/2) * psi2_11 - i*sin(gamma/2) * psi2_00
        let psi_f_11 = cadd(
            (c_g * psi2_11.0, c_g * psi2_11.1),
            (s_g * psi2_00.1, -s_g * psi2_00.0),
        );

        // Probabilidades de proyección P = |psi_f|^2:
        let p00 = psi_f_00.0 * psi_f_00.0 + psi_f_00.1 * psi_f_00.1;
        let p01 = psi_f_01.0 * psi_f_01.0 + psi_f_01.1 * psi_f_01.1;
        let p10 = psi_f_10.0 * psi_f_10.0 + psi_f_10.1 * psi_f_10.1;
        let p11 = psi_f_11.0 * psi_f_11.0 + psi_f_11.1 * psi_f_11.1;

        // Normalización numérica segura de probabilidades
        let p_sum = p00 + p01 + p10 + p11;
        let (norm_p00, norm_p01, norm_p10, norm_p11) = if p_sum > 1e-12 {
            (p00 / p_sum, p01 / p_sum, p10 / p_sum, p11 / p_sum)
        } else {
            (0.25, 0.25, 0.25, 0.25)
        };

        // Pagos esperados:
        // [R, S, T, P]
        let r = self.payoffs[0];
        let s = self.payoffs[1];
        let t = self.payoffs[2];
        let p = self.payoffs[3];

        let payoff_a = norm_p00 * r + norm_p01 * s + norm_p10 * t + norm_p11 * p;
        let payoff_b = norm_p00 * r + norm_p10 * s + norm_p01 * t + norm_p11 * p;

        // Comparación con equilibrio de Nash clásico (Defección mutua P)
        let classical_nash_payoff = p;
        let quantum_advantage = (payoff_a - classical_nash_payoff).max(0.0);

        EwlQuantumGameReport {
            payoff_a,
            payoff_b,
            p_cooperate_cooperate: norm_p00,
            p_defect_defect: norm_p11,
            quantum_advantage,
            entanglement_gamma: gamma,
        }
    }
}
