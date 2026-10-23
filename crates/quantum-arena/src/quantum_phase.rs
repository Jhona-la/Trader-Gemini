//! ACOPLAMIENTO CUÁNTICO TENSOR MPS AL UNIVERSO ESPECTRAL (Ola Omega220 — Átomo 3.3).
//!
//! Integra la física del mapa de fases $S \leftrightarrow g$ del modelo de Ising transverso
//! calibrado mediante DMRG sobre estados MPS (GLM 155/156) con el Campo de Régimen Espectral
//! continuo (`SpectralRegimeField`) y el motor de adaptación de Trader Gemini.
//!
//! # Marco Teórico
//!
//! El Hamiltoniano cuántico del modelo de Ising transverso unidimensional en cadena finita de $L$ sitios es:
//! $$H = -J \sum_{i=1}^{L-1} \sigma^z_i \sigma^z_{i+1} - g \sum_{i=1}^L \sigma^x_i \quad (J=1)$$
//!
//! 1. **Fase Ordenada / Coherente ($g < 1.0$):**
//!    En cadena finita, el estado fundamental tiende al estado coherente macroscópico simétrico
//!    (estado GATO): $(|\uparrow\dots\uparrow\rangle + |\downarrow\dots\downarrow\rangle)/\sqrt{2}$
//!    con entropía de entrelazamiento de von Neumann en el corte central $S \to \ln 2 \approx 0.6931$.
//!    En el mercado financiero, esta fase corresponde a regímenes de alta memoria no-lineal temporal,
//!    coherencia direccional macroscópica y baja disipación de información (e.g. BTC medido en $\bar{g} = 0.850 < 1.0$).
//!    Exposición: Amplificación continua $f_{\text{cuantico}} \in [1.0, 2.0]$.
//!
//! 2. **Punto Crítico Cuántico ($g = g_c = 1.0$):**
//!    La susceptibilidad magnética diverge en el límite termodinámico:
//!    $$\chi = \left|\frac{\partial \langle M \rangle}{\partial g}\right| \to \infty$$
//!    En el mercado, corresponde a transiciones de fase críticas (e.g. BNB en $\bar{g} = 1.157$, rozando la criticidad).
//!    Exposición: Amortiguamiento suave Hermite $C^1$ para blindar el balance contra choques de liquidez.
//!
//! 3. **Fase Paramagnética / Ruido ($g > 1.0$):**
//!    El campo transverso destruye el entrelazamiento temporal; el estado colapsa al producto
//!    $|\rightarrow\dots\rightarrow\rangle$ con $S \to 0$.
//!    En el mercado, corresponde a ruido microestructural desordenado (e.g. ICP en $\bar{g} = 4.095$).
//!    Exposición: Atenuación hiperbólica continua $f_{\text{cuantico}} = 1.0 / g \in (0.0, 1.0]$.

use feature_engine::entrelazamiento::{CurvaFases, EntrelazamientoError};
use serde::{Deserialize, Serialize};

/// Régimen cuántico continuo derivado del parámetro de acoplamiento $g$.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaseCuantica {
    /// $g < 1.0 - \delta_{\text{crit}}$: Estado coherente ordenado (memoria temporal genuina).
    Ordenada,
    /// $|g - 1.0| \le \delta_{\text{crit}}$: Zona de transición de fase crítica.
    Critica,
    /// $g > 1.0 + \delta_{\text{crit}}$: Estado desordenado paramagnético (ruido microestructural).
    Paramagnetica,
}

/// Parámetros de configuración del acoplamiento cuántico espectral.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantumCouplingParams {
    /// Peso de la modulación cuántica sobre el margen clásico [0.0, 2.0] (def: 1.0).
    pub w_quantum: f64,
    /// Radio de proximidad a la criticidad cuántica $\delta_{\text{crit}}$ (def: 0.05).
    pub delta_crit: f64,
    /// Severidad de amortiguamiento en criticidad cuántica $\gamma_{\text{crit}} \in [0.0, 0.5]$ (def: 0.20).
    pub gamma_crit: f64,
    /// Piso mínimo del factor de escala cuántico (def: 0.05).
    pub min_factor_floor: f64,
    /// Techo máximo de amplificación del factor cuántico (def: 2.0).
    pub max_factor_cap: f64,
}

impl Default for QuantumCouplingParams {
    fn default() -> Self {
        Self {
            w_quantum: 1.0,
            delta_crit: 0.05,
            gamma_crit: 0.20,
            min_factor_floor: 0.05,
            max_factor_cap: 2.0,
        }
    }
}

/// Estado cuántico de entrelazamiento y modulación espectral para un activo.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuantumPhaseState {
    /// Parámetro de acoplamiento transverso del Ising $g = h/J \in (0, \infty)$.
    pub g: f64,
    /// Entrelazamiento temporal neto observado $\Delta S = S - S_{\text{nulo}}$.
    pub delta_s: f64,
    /// Clasificación física de la fase cuántica.
    pub fase: FaseCuantica,
    /// Factor de escala continuo $f_{\text{cuantico}} \in [\text{min\_floor}, \text{max\_cap}]$.
    pub factor_cuantico: f64,
    /// Distancia logarítmica canónica a la criticidad: $|\ln g - \ln g_c| = |\ln g|$ con $g_c = 1.0$.
    pub distancia_criticidad_log: f64,
    /// Señal booleana de cruce del punto crítico respecto a la observación previa.
    pub es_cruce_critico: bool,
}

impl Default for QuantumPhaseState {
    fn default() -> Self {
        Self {
            g: 1.0,
            delta_s: 0.0,
            fase: FaseCuantica::Critica,
            factor_cuantico: 1.0,
            distancia_criticidad_log: 0.0,
            es_cruce_critico: false,
        }
    }
}

/// Calcula el factor de modulación continuo $f_{\text{cuantico}}(g)$ con amortiguamiento Hermite $C^1$.
///
/// Propiedades matemáticas:
/// - $g \le 1.0$: $f_{\text{base}}(g) = 1.0 + \text{clamp}(1.0 - g, 0.0, 1.0) \in [1.0, 2.0]$.
/// - $g > 1.0$: $f_{\text{base}}(g) = 1.0 / g \in (0.0, 1.0]$.
/// - Continuidad $C^0$: $\lim_{g \to 1^-} f_{\text{base}} = 1.0 = \lim_{g \to 1^+} f_{\text{base}}$.
/// - Amortiguamiento crítico $C^1$ en $|g - 1.0| \le \delta_{\text{crit}}$:
///   $$u = \frac{|g - 1.0|}{\delta_{\text{crit}}}, \quad H(u) = u^2 (3 - 2u)$$
///   $$\phi(g) = 1.0 - \gamma_{\text{crit}} \times (1.0 - H(u))$$
/// - En $g = 1.0$: $\phi(1.0) = 1.0 - \gamma_{\text{crit}}$, atenuando suavemente la divergencia crítica.
/// - En $|g - 1.0| = \delta_{\text{crit}}$: $H(1) = 1 \implies \phi = 1.0$ con derivada nula en la frontera.
pub fn calcular_factor_cuantico(g: f64, params: &QuantumCouplingParams) -> f64 {
    if !g.is_finite() || g <= 0.0 {
        return 1.0;
    }

    let f_base = if g <= 1.0 {
        1.0 + (1.0 - g).clamp(0.0, 1.0)
    } else {
        1.0 / g
    };

    let delta_crit = params.delta_crit.max(1e-4);
    let dist_crit = (g - 1.0).abs();

    let phi_crit = if dist_crit < delta_crit {
        let u = (dist_crit / delta_crit).clamp(0.0, 1.0);
        let h_u = u * u * (3.0 - 2.0 * u);
        1.0 - params.gamma_crit.clamp(0.0, 0.5) * (1.0 - h_u)
    } else {
        1.0
    };

    (f_base * phi_crit).clamp(params.min_factor_floor, params.max_factor_cap)
}

/// Clasifica la fase cuántica según el acoplamiento $g$ y el umbral de criticidad.
#[inline]
pub fn clasificar_fase(g: f64, delta_crit: f64) -> FaseCuantica {
    let dc = delta_crit.max(1e-4);
    if g < 1.0 - dc {
        FaseCuantica::Ordenada
    } else if g > 1.0 + dc {
        FaseCuantica::Paramagnetica
    } else {
        FaseCuantica::Critica
    }
}

/// Evalúa el estado cuántico invirtiendo la entropía observada $\Delta S$ sobre la curva de fases DMRG.
///
/// Trata ausencias de forma honesta (CL-15):
/// - Si $\Delta S > S_{\text{max}}$ (hiper-coherencia beyond Ising gato): usa el punto extremo ordenado $g_{\text{min}}$.
/// - Si $\Delta S \le S_{\text{min}}$ (ruido puro): usa el punto extremo paramagnético $g_{\text{max}}$.
/// - Selecciona el cruce óptimo continuo si hay múltiples raíces de inversión.
pub fn evaluar_fase_cuantica(
    delta_s: f64,
    curva: &CurvaFases,
    prev_g: Option<f64>,
    params: &QuantumCouplingParams,
) -> QuantumPhaseState {
    if !delta_s.is_finite() || curva.puntos.is_empty() {
        return QuantumPhaseState::default();
    }

    let g_estimado = match curva.invertir(delta_s) {
        Ok(inversion) => {
            if inversion.cruces.is_empty() {
                curva.g_max
            } else if let Some(prev) = prev_g {
                // Selecciona el cruce con menor distancia euclídea al estado anterior para preservar continuidad temporal
                inversion
                    .cruces
                    .iter()
                    .copied()
                    .min_by(|a, b| {
                        (a - prev)
                            .abs()
                            .partial_cmp(&(b - prev).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(inversion.cruces[0])
            } else {
                // Sin estado anterior: selecciona el cruce con menor distancia a criticidad g_c = 1.0
                inversion
                    .cruces
                    .iter()
                    .copied()
                    .min_by(|a, b| {
                        (a.ln().abs())
                            .partial_cmp(&(b.ln().abs()))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(inversion.cruces[0])
            }
        }
        Err(EntrelazamientoError::FueraDeCurva { s, s_max, .. }) => {
            if s > s_max {
                // Hiper-coherente: mayor entrelazamiento que el gato en L sitios -> acoplamiento mínimo (máximo orden)
                curva.puntos[0].g
            } else {
                // Ruido puro por debajo del piso
                curva.puntos.last().map(|p| p.g).unwrap_or(curva.g_max)
            }
        }
        Err(_) => {
            // Error de núcleo o degeneración: fallback honesto y seguro a criticidad neutra
            1.0
        }
    };

    let g_final = if g_estimado.is_finite() && g_estimado > 0.0 {
        g_estimado
    } else {
        1.0
    };

    let fase = clasificar_fase(g_final, params.delta_crit);
    let factor_cuantico = calcular_factor_cuantico(g_final, params);
    let dist_crit = g_final.ln().abs();

    let es_cruce_critico = if let Some(prev) = prev_g {
        if prev.is_finite() && prev > 0.0 {
            (prev - 1.0) * (g_final - 1.0) < 0.0
        } else {
            false
        }
    } else {
        false
    };

    QuantumPhaseState {
        g: g_final,
        delta_s,
        fase,
        factor_cuantico,
        distancia_criticidad_log: dist_crit,
        es_cruce_critico,
    }
}
