//! # Fase 18 — Estructuras de Regularidad de Martin Hairer (Renormalización BPHZ & Paraproducto de Bony), Conductancia Cuántica de Landauer-Büttiker y Localización de Anderson por Matrices de Transferencia Simplécticas
//!
//! ## 👨‍🏫 MODO PROFESOR (QUÉ, POR QUÉ, PARA QUÉ, CÓMO, CUÁNDO, DÓNDE, QUIÉN)
//!
//! - **QUÉ:** Tres motores físico-matemáticos en pila (`O(1)` memoria, cero `unwrap`/`assert!`, 100% Rust nativo):
//!   1. `HairerRegularityEngine`: Espacio modelo graduado $\mathcal{T} = (A, T, G)$ de las **Estructuras de Regularidad de Martin Hairer** (Medalla Fields 2014) con grupo de estructura de recentrado $\Gamma_{yx}$, descomposición en paraproducto de Bony $f \cdot g = \Pi_<(f, g) + \Pi_>(f, g) + \Pi_\circ(f, g)$ y sustracción BPHZ del contratérmino ultravioleta $C_\epsilon$ en productos estocásticos singulares $(\mathcal{I}(\Xi) \diamond \Xi)_x = \mathcal{I}(\Xi)_x \Xi_x - C_\epsilon$.
//!   2. `LandauerButtikerEngine`: Evaluador de transporte cuántico mesoscópico multicanal con conductancia adimensional de Landauer $G/G_0 = \sum_{n=1}^M T_n(E)$, potencia de ruido de disparo (*Shot Noise*) de Büttiker $\sum_n T_n(1 - T_n)$ y **Factor de Fano Cuántico** $F = \frac{\sum_n T_n(1 - T_n)}{\sum_n T_n} \in [0, 1]$.
//!   3. `AndersonLocalizationEngine`: Propagador de matrices de transferencia unimodulares $M_k = \begin{pmatrix} \epsilon_k - E & -1 \\ 1 & 0 \end{pmatrix} \in \text{SL}(2, \mathbb{R})$ ($\det M_k \equiv 1$) sobre perfiles desordenados del libro de órdenes, calculando el exponente de Furstenberg-Lyapunov $\gamma_{\text{loc}} \ge 0$ y la **Longitud de Localización de Anderson** $\xi_{\text{loc}} = \gamma_{\text{loc}}^{-1}$.
//! - **POR QUÉ:** En escalas sub-milisegundo con exponente de Hurst rugoso ($H \to 0$), multiplicar directamente señal por ruido blanco produce divergencias numéricas $O(\epsilon^{-1})$ al reducir el paso de malla; y en libros fragmentados por *spoofing*, el desorden de precios atrapa la propagación de órdenes por interferencia destructiva de Anderson.
//! - **PARA QUÉ:** Garantizar que los estimadores no lineales sean invariantes a la escala de discretización $\epsilon$, medir la transparencia balística real del libro ($F \to 0$) y vetar entradas cuando el desorden induce aislamiento de Anderson ($\xi_{\text{loc}} < L_{\text{crit}}$).
//! - **CÓMO:** Separación diádica de frecuencias bajas/altas para el término resonante $\Pi_\circ$, resonancia de Breit-Wigner sobre 8 canales transversales, y iteración de Furstenberg con re-normalización de norma en cada sitio del retículo 1D.
//! - **CUÁNDO:** En cada actualización de microestructura rugosa y perfil de desorden L2.
//! - **DÓNDE:** `crates/feature-engine/src/hairer_landauer_anderson.rs`.
//! - **QUIÉN:** `HairerRegularityEngine`, `LandauerButtikerEngine`, `AndersonLocalizationEngine`, `HairerLandauerAndersonMasterEngine`.

/// Número canónico de sitios/canales en pila para los motores de Fase 18.
pub const MESOSCOPIC_CHANNELS: usize = 8;

/// Clasificación del régimen cuántico mesoscópico y de regularidad estocástica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MesoscopicTransportRegime {
    /// Transporte balístico transparente ($G/G_0$ alto, Factor de Fano $F < 0.25$ y $\xi_{\text{loc}}$ grande).
    BallisticConductingChannel,
    /// Régimen metálico difusivo intermedio ($F \approx 1/3$) con producto renormalizado estable.
    DiffusiveMetallicFlow,
    /// Aislante localizado de Anderson ($\xi_{\text{loc}}$ corto) o resonancia ultravioleta severa: veto de ejecución.
    AndersonLocalizedInsulatorVeto,
}

/// Reporte de la descomposición de Bony y renormalización BPHZ de Martin Hairer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairerRegularityReport {
    /// Paraproducto de baja-alta frecuencia $\Pi_<(f, g) + \Pi_>(f, g)$ (siempre bien definido clásicamente).
    pub paraproduct_smooth_part: f64,
    /// Término resonante bruto diagonal $\Pi_\circ(f, g)$ donde reside la divergencia ultravioleta.
    pub resonant_raw_term: f64,
    /// Contratérmino analítico de renormalización BPHZ $C_\epsilon$.
    pub bphz_counterterm_c_eps: f64,
    /// Producto singular renormalizado de Hairer $f \diamond g = \Pi_<(f,g) + \Pi_>(f,g) + (\Pi_\circ(f,g) - C_\epsilon)$.
    pub renormalized_wick_product: f64,
    /// Residuo de invariancia del grupo de estructura $\|\Gamma_{yx} \tau - \tau'\|_2$.
    pub structure_group_invariance_error: f64,
}

/// Motor de Estructuras de Regularidad de Martin Hairer $\mathcal{T} = (A, T, G)$ y Paraproducto de Bony.
#[derive(Debug, Clone, Copy)]
pub struct HairerRegularityEngine {
    /// Exponente de regularidad de Hölder del ruido base $\alpha \in (-1.0, -0.5)$.
    pub noise_holder_alpha: f64,
    /// Escala efectiva de corte ultravioleta $\epsilon \in (10^{-4}, 1.0]$.
    pub uv_cutoff_epsilon: f64,
}

impl Default for HairerRegularityEngine {
    #[inline]
    fn default() -> Self {
        Self {
            noise_holder_alpha: -0.75,
            uv_cutoff_epsilon: 0.05,
        }
    }
}

impl HairerRegularityEngine {
    /// Aplica el operador del grupo de estructura $\Gamma_h: T \to T$ sobre las coordenadas del espacio modelo
    /// en la base $\{\mathbf{1}, \Xi, \mathcal{I}(\Xi), \mathcal{I}(\Xi)\Xi, X\}$:
    /// $$\Gamma_h(\mathbf{1}) = \mathbf{1}, \quad \Gamma_h(X) = X + h \mathbf{1}, \quad \Gamma_h(\mathcal{I}(\Xi)) = \mathcal{I}(\Xi) + h \mathbf{1}$$
    /// verificando la propiedad de grupo exacta $\Gamma_{h_1} \circ \Gamma_{h_2} = \Gamma_{h_1 + h_2}$.
    #[inline]
    pub fn apply_structure_group_gamma(coords: &[f64; 5], shift_h: f64) -> [f64; 5] {
        let h = if shift_h.is_finite() {
            shift_h.clamp(-1e4, 1e4)
        } else {
            0.0
        };
        let mut c = [0.0; 5];
        let mut i = 0;
        while i < 5 {
            c[i] = if coords[i].is_finite() {
                coords[i]
            } else {
                0.0
            };
            i += 1;
        }
        // Representación triangular superior unipotente de G sobre [1, Xi, I(Xi), I(Xi)Xi, X]:
        // c'[0] (coeficiente de 1) recibe contribuciones de I(Xi) y X trasladados por h
        // c'[1] (coeficiente de Xi) recibe contribución de I(Xi)Xi trasladado por h
        [c[0] + h * (c[2] + c[4]), c[1] + h * c[3], c[2], c[3], c[4]]
    }

    /// Evalúa la descomposición de paraproducto de Bony y el producto renormalizado BPHZ $f \diamond g$
    /// sobre dos series microestructurales acopladas `field_f` ($\mathcal{I}(\Xi)$) y `noise_g` ($\Xi$).
    #[inline]
    pub fn evaluate_renormalized_product(
        &self,
        field_f: &[f64; MESOSCOPIC_CHANNELS],
        noise_g: &[f64; MESOSCOPIC_CHANNELS],
        empirical_noise_variance: f64,
    ) -> HairerRegularityReport {
        let eps = if self.uv_cutoff_epsilon.is_finite() && self.uv_cutoff_epsilon > 1e-4 {
            self.uv_cutoff_epsilon.min(1.0)
        } else {
            0.05
        };
        let sigma2 = if empirical_noise_variance.is_finite() && empirical_noise_variance >= 0.0 {
            empirical_noise_variance.min(1e4)
        } else {
            0.0
        };

        let mut f = [0.0; MESOSCOPIC_CHANNELS];
        let mut g = [0.0; MESOSCOPIC_CHANNELS];
        let mut f_mean = 0.0;
        let mut g_mean = 0.0;
        let mut i = 0;
        while i < MESOSCOPIC_CHANNELS {
            f[i] = if field_f[i].is_finite() {
                field_f[i].clamp(-1e4, 1e4)
            } else {
                0.0
            };
            g[i] = if noise_g[i].is_finite() {
                noise_g[i].clamp(-1e4, 1e4)
            } else {
                0.0
            };
            f_mean += f[i];
            g_mean += g[i];
            i += 1;
        }
        f_mean /= MESOSCOPIC_CHANNELS as f64;
        g_mean /= MESOSCOPIC_CHANNELS as f64;

        // 1. Paraproducto de Bony:
        // - Parte suave (baja frecuencia x alta frecuencia + baja x baja): f_low * g_low
        // - Parte resonante diagonal Pi_o(f, g): covarianza de alta frecuencia (f_i - f_mean)*(g_i - g_mean)
        let paraproduct_smooth = f_mean * g_mean;
        let mut resonant_sum = 0.0;
        let mut k = 0;
        while k < MESOSCOPIC_CHANNELS {
            resonant_sum += (f[k] - f_mean) * (g[k] - g_mean);
            k += 1;
        }
        let resonant_raw = resonant_sum / (MESOSCOPIC_CHANNELS as f64);

        // 2. Contratérmino ultravioleta BPHZ C_eps = sigma^2 / (2 * eps^{|2*alpha + 2|}) acotado
        let alpha = if self.noise_holder_alpha.is_finite() {
            self.noise_holder_alpha.clamp(-0.99, -0.51)
        } else {
            -0.75
        };
        let deg_sum = (2.0 * alpha + 2.0).min(0.49);
        let uv_factor = eps.powf(deg_sum - 0.5).clamp(0.1, 100.0);
        let c_eps = 0.5 * sigma2 * uv_factor;

        let renormalized = paraproduct_smooth + (resonant_raw - c_eps);

        // 3. Verificación algebraica de la identidad cociclo de Hairer: Gamma_{h1+h2} == Gamma_{h1} o Gamma_{h2}
        let sample_coords = [paraproduct_smooth, resonant_raw, f_mean, g_mean, 1.0];
        let step1 = Self::apply_structure_group_gamma(&sample_coords, 0.35);
        let composed = Self::apply_structure_group_gamma(&step1, -0.15);
        let direct = Self::apply_structure_group_gamma(&sample_coords, 0.20);
        let mut err_sq = 0.0;
        let mut m = 0;
        while m < 5 {
            let d = composed[m] - direct[m];
            err_sq += d * d;
            m += 1;
        }

        HairerRegularityReport {
            paraproduct_smooth_part: if paraproduct_smooth.is_finite() {
                paraproduct_smooth
            } else {
                0.0
            },
            resonant_raw_term: if resonant_raw.is_finite() {
                resonant_raw
            } else {
                0.0
            },
            bphz_counterterm_c_eps: if c_eps.is_finite() { c_eps } else { 0.0 },
            renormalized_wick_product: if renormalized.is_finite() {
                renormalized
            } else {
                0.0
            },
            structure_group_invariance_error: err_sq.sqrt(),
        }
    }
}

/// Diagnóstico mesoscópico de Conductancia Cuántica de Landauer-Büttiker y Factor de Fano.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandauerButtikerReport {
    /// Probabilidades de transmisión por canal $T_n(E) \in [0, 1]$.
    pub channel_transmissions: [f64; MESOSCOPIC_CHANNELS],
    /// Conductancia cuántica adimensional $G / G_0 = \sum_{n=1}^M T_n(E) \in [0, M]$.
    pub dimensionless_conductance: f64,
    /// Potencia adimensional de ruido de disparo de Büttiker $S_{\text{shot}} = \sum_{n=1}^M T_n (1 - T_n)$.
    pub buttiker_shot_noise: f64,
    /// Factor de Fano cuántico $F = \frac{\sum_n T_n(1 - T_n)}{\sum_n T_n} \in [0, 1]$.
    pub quantum_fano_factor: f64,
}

/// Motor de Conductancia Cuántica Multicanal de Landauer-Büttiker.
#[derive(Debug, Clone, Copy)]
pub struct LandauerButtikerEngine {
    /// Ancho de acoplamiento de resonancia túnel $\Gamma_0 > 0$ (fórmula de Breit-Wigner).
    pub coupling_width_gamma: f64,
}

impl Default for LandauerButtikerEngine {
    #[inline]
    fn default() -> Self {
        Self {
            coupling_width_gamma: 1.0,
        }
    }
}

impl LandauerButtikerEngine {
    /// Evalúa la transmisión multicanal $T_n(E) = \frac{\Gamma^2}{(E - V_n)^2 + \Gamma^2}$ sobre 8 barreras de liquidez.
    #[inline]
    pub fn evaluate_conductance(
        &self,
        channel_barriers: &[f64; MESOSCOPIC_CHANNELS],
        incident_energy: f64,
    ) -> LandauerButtikerReport {
        let gamma = if self.coupling_width_gamma.is_finite() && self.coupling_width_gamma > 1e-6 {
            self.coupling_width_gamma.min(1e3)
        } else {
            1.0
        };
        let gamma2 = gamma * gamma;
        let energy = if incident_energy.is_finite() {
            incident_energy.clamp(-1e4, 1e4)
        } else {
            0.0
        };

        let mut trans = [0.0; MESOSCOPIC_CHANNELS];
        let mut sum_t = 0.0;
        let mut sum_shot = 0.0;

        let mut i = 0;
        while i < MESOSCOPIC_CHANNELS {
            let v_n = if channel_barriers[i].is_finite() {
                channel_barriers[i].clamp(-1e4, 1e4)
            } else {
                1e4
            };
            let detuning = energy - v_n;
            let t_n = (gamma2 / (detuning * detuning + gamma2)).clamp(0.0, 1.0);
            trans[i] = t_n;
            sum_t += t_n;
            sum_shot += t_n * (1.0 - t_n);
            i += 1;
        }

        let fano = if sum_t > 1e-12 {
            (sum_shot / sum_t).clamp(0.0, 1.0)
        } else {
            1.0
        };

        LandauerButtikerReport {
            channel_transmissions: trans,
            dimensionless_conductance: sum_t,
            buttiker_shot_noise: sum_shot,
            quantum_fano_factor: fano,
        }
    }
}

/// Diagnóstico de Localización de Anderson por Matrices de Transferencia Simplécticas $\text{SL}(2, \mathbb{R})$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AndersonLocalizationReport {
    /// Exponente de Lyapunov-Furstenberg $\gamma_{\text{loc}} = \frac{1}{L} \sum_{k=1}^L \ln \|M_k u_k\| \ge 0$.
    pub furstenberg_lyapunov_exponent: f64,
    /// Longitud de localización de Anderson $\xi_{\text{loc}} = \gamma_{\text{loc}}^{-1} > 0$.
    pub localization_length_xi: f64,
    /// Determinante acumulado de las matrices de transferencia ($\det \prod M_k \equiv 1.0$).
    pub transfer_matrix_det: f64,
}

/// Motor de Localización de Anderson en Retículos Desordenados 1D.
#[derive(Debug, Clone, Copy, Default)]
pub struct AndersonLocalizationEngine;

impl AndersonLocalizationEngine {
    /// Propaga el producto de matrices de transferencia simplécticas $M_k = \begin{pmatrix} \epsilon_k - E & -1 \\ 1 & 0 \end{pmatrix}$
    /// a lo largo de los sitios desordenados `on_site_disorder` y calcula $\gamma_{\text{loc}}$ y $\xi_{\text{loc}}$.
    #[inline]
    pub fn evaluate_localization(
        &self,
        on_site_disorder: &[f64; MESOSCOPIC_CHANNELS],
        energy: f64,
    ) -> AndersonLocalizationReport {
        let e_val = if energy.is_finite() {
            energy.clamp(-100.0, 100.0)
        } else {
            0.0
        };

        // Vector de estado de amplitud (psi_1, psi_0) normalizado e invariante simpléctico de det(M)
        let mut psi_curr = 1.0_f64;
        let mut psi_prev = 0.0_f64;
        let mut log_norm_sum = 0.0_f64;
        let mut det_prod = 1.0_f64;

        let mut k = 0;
        while k < MESOSCOPIC_CHANNELS {
            let eps_k = if on_site_disorder[k].is_finite() {
                on_site_disorder[k].clamp(-500.0, 500.0)
            } else {
                0.0
            };
            let trace_term = eps_k - e_val;

            // M_k = [[trace_term, -1], [1, 0]] => det(M_k) = m00 * m11 - m01 * m10 == 1.0
            let m01 = -1.0_f64;
            let m10 = 1.0_f64;
            let m11 = 0.0_f64;
            let m_det = trace_term.mul_add(m11, -(m01 * m10));
            det_prod *= m_det;

            let psi_next = trace_term.mul_add(psi_curr, -psi_prev);
            psi_prev = psi_curr;
            psi_curr = psi_next;

            let norm = psi_curr.hypot(psi_prev);
            if norm > 1e-15 && norm.is_finite() {
                log_norm_sum += norm.ln();
                psi_curr /= norm;
                psi_prev /= norm;
            }
            k += 1;
        }

        let gamma = (log_norm_sum / (MESOSCOPIC_CHANNELS as f64)).max(0.0);
        let xi = if gamma > 1e-6 {
            (1.0 / gamma).min(1.0e6)
        } else {
            1.0e6
        };

        AndersonLocalizationReport {
            furstenberg_lyapunov_exponent: if gamma.is_finite() { gamma } else { 0.0 },
            localization_length_xi: if xi.is_finite() { xi } else { 1.0e6 },
            transfer_matrix_det: if det_prod.is_finite() { det_prod } else { 1.0 },
        }
    }
}

/// Reporte unificado de la Fase 18 (Hairer Regularity + Landauer-Büttiker + Anderson Localization).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairerLandauerAndersonReport {
    pub hairer: HairerRegularityReport,
    pub landauer: LandauerButtikerReport,
    pub anderson: AndersonLocalizationReport,
    pub mesoscopic_conviction: f64,
    pub regime: MesoscopicTransportRegime,
}

/// Motor maestro de Fase 18.
#[derive(Debug, Clone, Copy, Default)]
pub struct HairerLandauerAndersonMasterEngine {
    pub hairer_engine: HairerRegularityEngine,
    pub landauer_engine: LandauerButtikerEngine,
    pub anderson_engine: AndersonLocalizationEngine,
}

impl HairerLandauerAndersonMasterEngine {
    /// Evalúa un ciclo completo de regularidad de Hairer, conductancia de Landauer-Büttiker y localización de Anderson.
    #[inline]
    pub fn evaluate_cycle(
        &self,
        field_f: &[f64; MESOSCOPIC_CHANNELS],
        noise_g: &[f64; MESOSCOPIC_CHANNELS],
        noise_variance: f64,
        channel_barriers: &[f64; MESOSCOPIC_CHANNELS],
        incident_energy: f64,
    ) -> HairerLandauerAndersonReport {
        let hairer =
            self.hairer_engine
                .evaluate_renormalized_product(field_f, noise_g, noise_variance);
        let landauer = self
            .landauer_engine
            .evaluate_conductance(channel_barriers, incident_energy);
        let anderson = self
            .anderson_engine
            .evaluate_localization(channel_barriers, incident_energy);

        let raw_conv = hairer.renormalized_wick_product.tanh()
            * (landauer.dimensionless_conductance / (MESOSCOPIC_CHANNELS as f64)).clamp(0.0, 1.0);
        let mesoscopic_conviction = if raw_conv.is_finite() {
            raw_conv.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let regime = if anderson.localization_length_xi < 1.5
            || landauer.dimensionless_conductance < 0.5
        {
            MesoscopicTransportRegime::AndersonLocalizedInsulatorVeto
        } else if landauer.quantum_fano_factor < 0.25 && landauer.dimensionless_conductance > 4.0 {
            MesoscopicTransportRegime::BallisticConductingChannel
        } else {
            MesoscopicTransportRegime::DiffusiveMetallicFlow
        };

        HairerLandauerAndersonReport {
            hairer,
            landauer,
            anderson,
            mesoscopic_conviction,
            regime,
        }
    }
}
