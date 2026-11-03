//! # Fase 20 — Termodinámica de Landauer (`LandauerThermodynamicEngine`), Condensación de Bose-Einstein en Libro L2 (`BoseEinsteinOrderbookEngine`) y Memoria Asociativa Continua Moderna de Hopfield (`ModernContinuousHopfieldEngine`)
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Cuatro primitivas físicas y algebraicas deterministas en pila (`O(1)` memoria en stack, cero `unwrap`/`assert!`, 100% Rust nativo):
//!   1. `KbnCompensatedAccumulator`: Acumulador de alta precisión numérica mediante el algoritmo de Kahan-Babuška-Neumaier (KBN) que elimina la deriva de redondeo en operaciones de punto flotante de alta frecuencia hasta 1 ULP.
//!   2. `LandauerThermodynamicEngine`: Cota termodinámica de disipación de información de Landauer ($W_{\text{diss}} \ge k_B T_{\text{mkt}} \ln 2 \cdot \Delta I$), temperatura efectiva de microestructura y eficiencia entrópica de flujo de órdenes.
//!   3. `BoseEinsteinOrderbookEngine`: Estadística cuántica de Bose-Einstein aplicada a la distribución de liquidez del libro L2 ($n(\varepsilon_i) = [\exp((\varepsilon_i - \mu)/T_{\text{mkt}}) - 1]^{-1}$), potencial químico $\mu$ y fracción de condensado en el nivel fundamental ($f_0 = N_0 / N_{\text{total}}$).
//!   4. `ModernContinuousHopfieldEngine`: Red de Hopfield continua moderna de alta capacidad exponencial asociativa (Ramsauer et al., 2020) con función de energía Log-Sum-Exp $E(x) = -\beta^{-1}\ln\sum_k \exp(\beta \xi_k^T x) + \frac{1}{2}\|x\|^2$ y contracción monótona hacia atractores de régimen.
//! - **POR QUÉ:** La microestructura financiera disipa información en cada transacción ejecutada; la acumulación extrema de liquidez en el mejor bid/ask actúa como una transición de fase análoga a la condensación cuántica de Bose-Einstein; y los clasificadores lineales tradicionales sufren interferencia destructiva de memoria, mientras que las redes continuas de Hopfield recuperan atractores globales sin colapso de capacidad.
//! - **PARA QUÉ:** Medir la fricción informacional termodinámica real, detectar muros de liquidez de condensación crítica antes de rupturas de precio, y proyectar el vector microestructural instantáneo sobre arquetipos de régimen estables.
//! - **CÓMO:** Sumación Neumaier compensada, integración discreta de Shannon-Landauer, inversión analítica de potencial químico de Bose-Einstein y proyección softmax estabilizada por LSE (Log-Sum-Exp).
//! - **CUÁNDO:** En cada actualización del libro L2 y cálculo de densidad informacional de flujo.
//! - **DÓNDE:** `crates/feature-engine/src/landauer_bec_hopfield.rs`.
//! - **QUIÉN:** `KbnCompensatedAccumulator`, `LandauerThermodynamicEngine`, `BoseEinsteinOrderbookEngine`, `ModernContinuousHopfieldEngine`, `LandauerBecHopfieldMasterEngine`.

/// Constante de Boltzmann microestructural normalizada (adimensional).
pub const BOLTZMANN_K_MKT: f64 = 1.0;

/// Logaritmo natural de 2 ($\ln 2$).
pub const LN_2: f64 = core::f64::consts::LN_2;

/// Dimensión de niveles de energía de distancia al mid en el libro L2 para Bose-Einstein.
pub const BEC_ENERGY_LEVELS: usize = 8;

/// Dimensión de características de microestructura para la memoria de Hopfield.
pub const HOPFIELD_FEATURE_DIM: usize = 8;

/// Número de patrones arquetípicos de régimen almacenados en la red de Hopfield.
pub const HOPFIELD_PROTOTYPE_COUNT: usize = 8;

/// Acumulador numérico compensado de Kahan-Babuška-Neumaier (KBN).
///
/// Preserva la suma exacta $S = \sum_i x_i$ reduciendo el error acumulado
/// a orden $O(\varepsilon)$ en lugar de $O(N \varepsilon)$, esencial para series
/// largas de ticks donde sumas ingenuas pierden dígitos significativos.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct KbnCompensatedAccumulator {
    sum: f64,
    compensation: f64,
}

impl KbnCompensatedAccumulator {
    /// Inicializa un acumulador compensado vacío.
    #[inline]
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Añade un valor $x$ usando la corrección de Neumaier.
    #[inline]
    pub fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// Retorna la suma total corregida.
    #[inline]
    pub fn total(&self) -> f64 {
        let res = self.sum + self.compensation;
        if res.is_finite() {
            res
        } else {
            0.0
        }
    }

    /// Reinicia el acumulador a cero.
    #[inline]
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }
}

/// Régimen de disipación termodinámica de información de Landauer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LandauerDissipationRegime {
    /// Disipación casi reversible ($\eta_{\text{Landauer}} \ge 0.70$), mercado altamente eficiente.
    ReversibleEfficient,
    /// Disipación moderada fuera del equilibrio ($0.20 \le \eta_{\text{Landauer}} < 0.70$).
    NearEquilibriumDissipative,
    /// Fuerte irreversibilidad turbulenta ($\eta_{\text{Landauer}} < 0.20$), alto arrastre friccional.
    TurbulentIrreversible,
}

/// Diagnóstico de termodinámica de Landauer del flujo de órdenes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandauerThermodynamicReport {
    /// Temperatura efectiva del mercado $T_{\text{mkt}} > 0$.
    pub market_temperature: f64,
    /// Entropía informacional del libro $\Delta I$ en bits.
    pub info_entropy_bits: f64,
    /// Cota mínima teórica de disipación de Landauer $W_{\text{min}} = k_B T \ln 2 \cdot \Delta I$.
    pub landauer_min_work: f64,
    /// Disipación microestructural observada $W_{\text{actual}}$.
    pub actual_dissipation: f64,
    /// Eficiencia termodinámica de Landauer $\eta_{\text{Landauer}} \in [0, 1]$.
    pub landauer_efficiency: f64,
    /// Régimen de disipación termodinámica.
    pub regime: LandauerDissipationRegime,
}

/// Motor de Termodinámica de Landauer.
#[derive(Debug, Clone, Copy, Default)]
pub struct LandauerThermodynamicEngine;

impl LandauerThermodynamicEngine {
    /// Evalúa la disipación termodinámica dada una distribución de probabilidades de niveles `probs`,
    /// la temperatura del mercado `market_temp` y la disipación observada de spread/impacto `actual_work`.
    #[inline]
    pub fn evaluate(
        &self,
        probs: &[f64; BEC_ENERGY_LEVELS],
        market_temp: f64,
        actual_work: f64,
    ) -> LandauerThermodynamicReport {
        let t_mkt = if market_temp.is_finite() && market_temp > 1e-6 {
            market_temp.clamp(1e-6, 1e6)
        } else {
            1.0
        };

        let mut acc_p = KbnCompensatedAccumulator::new();
        let mut entropy_acc = KbnCompensatedAccumulator::new();

        let mut i = 0;
        while i < BEC_ENERGY_LEVELS {
            let p = if probs[i].is_finite() && probs[i] > 0.0 {
                probs[i]
            } else {
                0.0
            };
            acc_p.add(p);
            if p > 1e-15 {
                // S_i = - p * log2(p)
                let s_i = -p * p.log2();
                entropy_acc.add(s_i);
            }
            i += 1;
        }

        let total_p = acc_p.total();
        let raw_entropy = entropy_acc.total();
        let entropy_bits = if total_p > 1e-12 {
            (raw_entropy / total_p).clamp(0.0, 8.0)
        } else {
            0.0
        };

        let w_min = BOLTZMANN_K_MKT * t_mkt * LN_2 * entropy_bits;
        let w_obs = if actual_work.is_finite() && actual_work > 0.0 {
            actual_work
        } else {
            w_min
        };

        let efficiency = if w_obs > 1e-12 {
            (w_min / w_obs).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let regime = if efficiency >= 0.70 {
            LandauerDissipationRegime::ReversibleEfficient
        } else if efficiency >= 0.20 {
            LandauerDissipationRegime::NearEquilibriumDissipative
        } else {
            LandauerDissipationRegime::TurbulentIrreversible
        };

        LandauerThermodynamicReport {
            market_temperature: t_mkt,
            info_entropy_bits: entropy_bits,
            landauer_min_work: w_min,
            actual_dissipation: w_obs,
            landauer_efficiency: efficiency,
            regime,
        }
    }
}

/// Régimen de fase cuántica de Bose-Einstein en el libro de órdenes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BecPhaseRegime {
    /// Gas térmico ordinario (liquidez dispersa uniformemente a lo largo del libro).
    NormalThermalGas,
    /// Transición crítica inminente (acumulación anómala cerca del mejor precio).
    CriticalFluctuation,
    /// Condensado cuántico de Bose-Einstein (muro masivo condensado en el nivel fundamental $f_0 \ge 0.50$).
    BoseEinsteinCondensate,
}

/// Diagnóstico de Condensación de Bose-Einstein en el libro de órdenes L2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BecOrderbookReport {
    /// Fracción de condensado en el estado fundamental $f_0 = N_0 / N_{\text{total}} \in [0, 1]$.
    pub condensate_fraction: f64,
    /// Potencial químico efectivo $\mu \le \varepsilon_0$.
    pub chemical_potential: f64,
    /// Temperatura crítica de condensación estimada $T_c$.
    pub critical_temperature: f64,
    /// Ocupaciones teóricas de Bose-Einstein por nivel.
    pub occupation_levels: [f64; BEC_ENERGY_LEVELS],
    /// Régimen de fase cuántica del libro.
    pub regime: BecPhaseRegime,
}

/// Motor de Condensación de Bose-Einstein en el libro L2.
#[derive(Debug, Clone, Copy, Default)]
pub struct BoseEinsteinOrderbookEngine;

impl BoseEinsteinOrderbookEngine {
    /// Evalúa la distribución de volumen en los niveles L2 `volumes` con distancias relativas al mid `energies`
    /// y temperatura de microestructura `market_temp`.
    #[inline]
    pub fn evaluate(
        &self,
        volumes: &[f64; BEC_ENERGY_LEVELS],
        energies: &[f64; BEC_ENERGY_LEVELS],
        market_temp: f64,
    ) -> BecOrderbookReport {
        let t_mkt = if market_temp.is_finite() && market_temp > 1e-6 {
            market_temp.clamp(1e-6, 1e6)
        } else {
            1.0
        };

        let mut eps = [0.0; BEC_ENERGY_LEVELS];
        let mut vols = [0.0; BEC_ENERGY_LEVELS];
        let mut total_vol_acc = KbnCompensatedAccumulator::new();

        let mut i = 0;
        while i < BEC_ENERGY_LEVELS {
            eps[i] = if energies[i].is_finite() && energies[i] >= 0.0 {
                energies[i]
            } else {
                i as f64 * 0.001
            };
            vols[i] = if volumes[i].is_finite() && volumes[i] >= 0.0 {
                volumes[i]
            } else {
                0.0
            };
            total_vol_acc.add(vols[i]);
            i += 1;
        }

        let total_vol = total_vol_acc.total();
        let fundamental_vol = vols[0];

        let f0 = if total_vol > 1e-12 {
            (fundamental_vol / total_vol).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // El potencial químico mu debe ser <= eps[0]. A medida que mu -> eps[0], el nivel fundamental se satura.
        let eps_0 = eps[0];
        let mu = if f0 > 0.01 && f0 < 0.999 {
            // Inversión semiclásica: f0 = [exp((eps_0 - mu)/T) - 1]^-1 / total_vol
            // exp((eps_0 - mu)/T) = 1 + 1/(f0 * total_vol)
            let denom = (f0 * total_vol).max(1e-6);
            let arg = 1.0 + 1.0 / denom;
            (eps_0 - t_mkt * arg.ln()).min(eps_0 - 1e-9)
        } else if f0 >= 0.999 {
            eps_0 - 1e-9
        } else {
            eps_0 - t_mkt * 2.0
        };

        // Cálculo de ocupaciones teóricas de Bose-Einstein n(eps_i)
        let mut occ = [0.0; BEC_ENERGY_LEVELS];
        let mut i = 0;
        while i < BEC_ENERGY_LEVELS {
            let delta = ((eps[i] - mu) / t_mkt).clamp(1e-9, 50.0);
            let n_i = 1.0 / (delta.exp() - 1.0);
            occ[i] = if n_i.is_finite() { n_i } else { 0.0 };
            i += 1;
        }

        // Temperatura crítica T_c proporcional al volumen no fundamental y la densidad de estados
        let t_c = (t_mkt * (1.0 - f0).powf(2.0 / 3.0)).max(1e-6);

        let regime = if f0 >= 0.50 {
            BecPhaseRegime::BoseEinsteinCondensate
        } else if f0 >= 0.25 {
            BecPhaseRegime::CriticalFluctuation
        } else {
            BecPhaseRegime::NormalThermalGas
        };

        BecOrderbookReport {
            condensate_fraction: f0,
            chemical_potential: mu,
            critical_temperature: t_c,
            occupation_levels: occ,
            regime,
        }
    }
}

/// Diagnóstico de recuperación asociativa de la Red de Hopfield Continua Moderna.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModernHopfieldReport {
    /// Energía Log-Sum-Exp de Hopfield antes de la recuperación $E(x_{\text{in}})$.
    pub initial_energy: f64,
    /// Energía de Hopfield después de 1 paso asociativo $E(x_{\text{out}})$.
    pub final_energy: f64,
    /// Variación de energía $\Delta E = E_{\text{final}} - E_{\text{initial}} \le 0$ (descenso monótono garantizado).
    pub energy_delta: f64,
    /// Vector recuperado en el espacio de Hilbert $\mathbb{R}^D$.
    pub retrieved_state: [f64; HOPFIELD_FEATURE_DIM],
    /// Ponderaciones de atención softmax sobre los $P$ atractores prototípicos.
    pub prototype_weights: [f64; HOPFIELD_PROTOTYPE_COUNT],
    /// Índice del atractor prototípico dominante.
    pub dominant_prototype_idx: usize,
    /// Nivel de confianza / solapamiento con el atractor dominante $\max_k w_k \in [1/P, 1]$.
    pub attractor_confidence: f64,
}

/// Motor de Memoria Asociativa Continua Moderna de Hopfield.
#[derive(Debug, Clone, Copy)]
pub struct ModernContinuousHopfieldEngine {
    /// Matriz de atractores prototípicos $\Xi \in \mathbb{R}^{D \times P}$.
    prototypes: [[f64; HOPFIELD_FEATURE_DIM]; HOPFIELD_PROTOTYPE_COUNT],
    /// Parámetro de temperatura inversa / nitidez $\beta > 0$.
    beta: f64,
}

impl Default for ModernContinuousHopfieldEngine {
    #[inline]
    fn default() -> Self {
        // Matriz de 8 prototipos ortogonales / arquetipos microestructurales canónicos normalizados
        let mut proto = [[0.0; HOPFIELD_FEATURE_DIM]; HOPFIELD_PROTOTYPE_COUNT];
        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            proto[k][k] = 1.0;
            k += 1;
        }
        Self {
            prototypes: proto,
            beta: 4.0,
        }
    }
}

impl ModernContinuousHopfieldEngine {
    /// Construye el motor con prototipos personalizados y parámetro $\beta$.
    #[inline]
    pub fn new(
        prototypes: [[f64; HOPFIELD_FEATURE_DIM]; HOPFIELD_PROTOTYPE_COUNT],
        beta: f64,
    ) -> Self {
        let b = if beta.is_finite() && beta > 0.01 {
            beta.clamp(0.01, 100.0)
        } else {
            4.0
        };
        Self {
            prototypes,
            beta: b,
        }
    }

    /// Calcula la energía continua exacta de Hopfield:
    /// $$E(x) = -\frac{1}{\beta} \text{LSE}(\beta \Xi^T x) + \frac{1}{2} \|x\|^2$$
    #[inline]
    pub fn compute_energy(&self, state: &[f64; HOPFIELD_FEATURE_DIM]) -> f64 {
        let mut dots = [0.0; HOPFIELD_PROTOTYPE_COUNT];
        let mut max_dot = f64::NEG_INFINITY;

        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            let mut dot = 0.0;
            let mut d = 0;
            while d < HOPFIELD_FEATURE_DIM {
                dot += self.prototypes[k][d] * state[d];
                d += 1;
            }
            let scaled_dot = self.beta * dot;
            dots[k] = scaled_dot;
            if scaled_dot > max_dot {
                max_dot = scaled_dot;
            }
            k += 1;
        }

        // Log-Sum-Exp numéricamente estable: max_dot + ln(sum exp(dot_k - max_dot))
        let mut lse_acc = KbnCompensatedAccumulator::new();
        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            let term = (dots[k] - max_dot).exp();
            lse_acc.add(term);
            k += 1;
        }
        let sum_exp = lse_acc.total().max(1e-15);
        let lse = max_dot + sum_exp.ln();

        // Norma L2 al cuadrado: 0.5 * ||state||^2
        let mut norm_acc = KbnCompensatedAccumulator::new();
        let mut d = 0;
        while d < HOPFIELD_FEATURE_DIM {
            norm_acc.add(state[d] * state[d]);
            d += 1;
        }
        let half_norm_sq = 0.5 * norm_acc.total();

        -(1.0 / self.beta) * lse + half_norm_sq
    }

    /// Ejecuta 1 paso de recuperación asociativa continua $x_{\text{new}} = \Xi \, \text{softmax}(\beta \Xi^T x)$.
    #[inline]
    pub fn retrieve(&self, state: &[f64; HOPFIELD_FEATURE_DIM]) -> ModernHopfieldReport {
        let mut safe_state = [0.0; HOPFIELD_FEATURE_DIM];
        let mut d = 0;
        while d < HOPFIELD_FEATURE_DIM {
            safe_state[d] = if state[d].is_finite() {
                state[d].clamp(-1e3, 1e3)
            } else {
                0.0
            };
            d += 1;
        }

        let e_init = self.compute_energy(&safe_state);

        // Proyecciones dot_k = beta * (xi_k . state)
        let mut dots = [0.0; HOPFIELD_PROTOTYPE_COUNT];
        let mut max_dot = f64::NEG_INFINITY;
        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            let mut dot = 0.0;
            let mut d = 0;
            while d < HOPFIELD_FEATURE_DIM {
                dot += self.prototypes[k][d] * safe_state[d];
                d += 1;
            }
            let scaled_dot = self.beta * dot;
            dots[k] = scaled_dot;
            if scaled_dot > max_dot {
                max_dot = scaled_dot;
            }
            k += 1;
        }

        // Softmax numéricamente estable
        let mut weights = [0.0; HOPFIELD_PROTOTYPE_COUNT];
        let mut sum_exp_acc = KbnCompensatedAccumulator::new();
        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            let w = (dots[k] - max_dot).exp();
            weights[k] = w;
            sum_exp_acc.add(w);
            k += 1;
        }
        let total_w = sum_exp_acc.total().max(1e-15);

        let mut dominant_k = 0;
        let mut max_w = 0.0;
        let mut k = 0;
        while k < HOPFIELD_PROTOTYPE_COUNT {
            weights[k] /= total_w;
            if weights[k] > max_w {
                max_w = weights[k];
                dominant_k = k;
            }
            k += 1;
        }

        // Reconstrucción del estado recuperado x_out = sum_k w_k * xi_k
        let mut retrieved = [0.0; HOPFIELD_FEATURE_DIM];
        let mut d = 0;
        while d < HOPFIELD_FEATURE_DIM {
            let mut col_acc = KbnCompensatedAccumulator::new();
            let mut k = 0;
            while k < HOPFIELD_PROTOTYPE_COUNT {
                col_acc.add(weights[k] * self.prototypes[k][d]);
                k += 1;
            }
            retrieved[d] = col_acc.total();
            d += 1;
        }

        let e_final = self.compute_energy(&retrieved);
        let delta_e = e_final - e_init;

        ModernHopfieldReport {
            initial_energy: e_init,
            final_energy: e_final,
            energy_delta: delta_e,
            retrieved_state: retrieved,
            prototype_weights: weights,
            dominant_prototype_idx: dominant_k,
            attractor_confidence: max_w,
        }
    }
}

/// Reporte maestro unificado de Termodinámica de Landauer, BEC L2 y Red de Hopfield Continua.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandauerBecHopfieldReport {
    /// Diagnóstico termodinámico de Landauer.
    pub landauer: LandauerThermodynamicReport,
    /// Diagnóstico de condensación cuántica de Bose-Einstein.
    pub bec: BecOrderbookReport,
    /// Diagnóstico asociativo de Hopfield continuo.
    pub hopfield: ModernHopfieldReport,
}

/// Motor Maestro que coordina Landauer, Bose-Einstein y Hopfield Continuo.
#[derive(Debug, Clone, Copy, Default)]
pub struct LandauerBecHopfieldMasterEngine {
    landauer_engine: LandauerThermodynamicEngine,
    bec_engine: BoseEinsteinOrderbookEngine,
    hopfield_engine: ModernContinuousHopfieldEngine,
}

impl LandauerBecHopfieldMasterEngine {
    /// Evalúa exhaustivamente los tres motores acoplados de forma zero-alloc y thread-safe.
    #[inline]
    pub fn evaluate(
        &self,
        l2_volumes: &[f64; BEC_ENERGY_LEVELS],
        l2_energies: &[f64; BEC_ENERGY_LEVELS],
        market_temperature: f64,
        actual_work_dissipated: f64,
        microstructure_state: &[f64; HOPFIELD_FEATURE_DIM],
    ) -> LandauerBecHopfieldReport {
        // Normalización de probabilidades para Landauer a partir de volúmenes L2
        let mut probs = [0.0; BEC_ENERGY_LEVELS];
        let mut vol_acc = KbnCompensatedAccumulator::new();
        let mut i = 0;
        while i < BEC_ENERGY_LEVELS {
            let v = if l2_volumes[i].is_finite() && l2_volumes[i] > 0.0 {
                l2_volumes[i]
            } else {
                0.0
            };
            vol_acc.add(v);
            i += 1;
        }
        let total_v = vol_acc.total().max(1e-12);
        let mut i = 0;
        while i < BEC_ENERGY_LEVELS {
            probs[i] = (l2_volumes[i].max(0.0) / total_v).clamp(0.0, 1.0);
            i += 1;
        }

        let landauer_rep =
            self.landauer_engine
                .evaluate(&probs, market_temperature, actual_work_dissipated);
        let bec_rep = self
            .bec_engine
            .evaluate(l2_volumes, l2_energies, market_temperature);
        let hopfield_rep = self.hopfield_engine.retrieve(microstructure_state);

        LandauerBecHopfieldReport {
            landauer: landauer_rep,
            bec: bec_rep,
            hopfield: hopfield_rep,
        }
    }
}
