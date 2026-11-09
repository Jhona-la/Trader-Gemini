//! Fase 62: Cohomología de De Rham y Corrientes de Hodge en Variedades de Retorno Conectado.
//!
//! Este módulo implementa la descomposición ortogonal de Hodge-Helmholtz de las formas diferenciales
//! de flujo continuo de órdenes: $\omega = d\alpha + \delta\beta + \gamma$, separando rigurosamente:
//! 1. Componente exacta $d\alpha$: Gradiente conservativo / potencial de reversión a la media.
//! 2. Componente coexacta $\delta\beta$: Corriente rotacional solenoidal / arrastre turbulento de microestructura.
//! 3. Componente armónica $\gamma$: Invariante topológico persistente de la cohomología de De Rham $H^1_{\text{dR}}(M)$.
//!
//! # Invariantes y Leyes Físico-Matemáticas:
//! - Las fracciones exacta, coexacta y armónica son una partición adimensional que suma 1 (0 si las
//!   tres energías son nulas). No es la descomposición ortogonal de Hodge: el potencial y la corriente
//!   son campos distintos, así que $\|\omega\|^2 = \|d\alpha\|^2 + \|\delta\beta\|^2 + \|\gamma\|^2$ no
//!   se cumple.
//! - Invariante `rb_p1`: Cero pánicos, 0 unwrap, 0 expect, 0 panic! y memoria acotada en stack $O(1)$.

/// Número de nodos del complejo simplicial de Cartan para la variedad de retornos.
pub const DERHAM_SIMPLICIAL_NODES: usize = 8;

/// Parámetros de configuración del motor de cohomología de De Rham y Hodge.
#[derive(Debug, Clone, Copy)]
pub struct DeRhamHodgeParams {
    /// Umbral coexacto para alerta de turbulencia rotacional.
    pub coexact_turbulent_threshold: f64,
    /// Regularización para división numérica segura.
    pub epsilon: f64,
    /// Factor de atenuación para el flujo de gradiente conservativo.
    pub gradient_smoothing: f64,
}

impl Default for DeRhamHodgeParams {
    fn default() -> Self {
        Self {
            coexact_turbulent_threshold: 0.70,
            epsilon: 1e-12,
            gradient_smoothing: 0.20,
        }
    }
}

/// Informe omnisciente de la descomposición de Hodge y cohomología de De Rham.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeRhamHodgeReport {
    /// Fracción de energía de la componente exacta $d\alpha$ (reversión conservativa) $\in [0.0, 1.0]$.
    pub exact_energy_ratio: f64,
    /// Fracción de energía de la componente coexacta $\delta\beta$ (turbulencia rotacional) $\in [0.0, 1.0]$.
    pub coexact_energy_ratio: f64,
    /// Fracción de energía armónica $\gamma$ (persistencia topológica global) $\in [0.0, 1.0]$.
    pub harmonic_energy_ratio: f64,
    /// Energía total de Dirichlet del laplaciano de Hodge $\Delta = d\delta + \delta d \ge 0$,
    /// adimensional: suma de los cocientes de Rayleigh exacto ($\in [0, 4]$) y coexacto
    /// ($\in [0, 16]$).
    pub laplacian_dirichlet_energy: f64,
    /// Estimador continuo del primer número de Betti $b_1 \in [0.0, 1.0]$.
    pub betti_one_persistence: f64,
    /// Señal sintetizada continua libre de turbulencia rotacional $\in [-1.0, 1.0]$.
    pub synthesized_signal: f64,
    /// Alerta de turbulencia coexacta destructiva por vórtice de flujo.
    pub turbulent_vorticity_alert: bool,
}

impl Default for DeRhamHodgeReport {
    fn default() -> Self {
        Self {
            exact_energy_ratio: 0.34,
            coexact_energy_ratio: 0.33,
            harmonic_energy_ratio: 0.33,
            laplacian_dirichlet_energy: 0.0,
            betti_one_persistence: 0.50,
            synthesized_signal: 0.0,
            turbulent_vorticity_alert: false,
        }
    }
}

/// Motor de descomposición de Hodge-Helmholtz y Cohomología de De Rham.
#[derive(Debug, Clone)]
pub struct DeRhamHodgeCurrentsEngine {
    params: DeRhamHodgeParams,
    simplicial_potentials: [f64; DERHAM_SIMPLICIAL_NODES],
    simplicial_currents: [f64; DERHAM_SIMPLICIAL_NODES],
    cursor: usize,
    count: usize,
}

impl DeRhamHodgeCurrentsEngine {
    /// Crea una nueva instancia del motor con parámetros específicos.
    pub fn new(params: DeRhamHodgeParams) -> Self {
        Self {
            params,
            simplicial_potentials: [0.0; DERHAM_SIMPLICIAL_NODES],
            simplicial_currents: [0.0; DERHAM_SIMPLICIAL_NODES],
            cursor: 0,
            count: 0,
        }
    }

    /// Actualiza la descomposición ortogonal de Hodge a partir del flujo continuo.
    ///
    /// # Argumentos:
    /// - `price_return`: Retorno normalizado instantáneo $\Delta p / p$.
    /// - `order_flow_imbalance`: Desequilibrio OFI $\in [-1, 1]$.
    /// - `spread_bps`: Spread relativo actual $\ge 0$.
    /// - `instantaneous_vol`: Volatilidad continua $\sigma \ge 0$.
    pub fn update(
        &mut self,
        price_return: f64,
        order_flow_imbalance: f64,
        spread_bps: f64,
        instantaneous_vol: f64,
    ) -> DeRhamHodgeReport {
        let safe_ret = if price_return.is_finite() {
            price_return.clamp(-0.50, 0.50)
        } else {
            0.0
        };

        let safe_ofi = if order_flow_imbalance.is_finite() {
            order_flow_imbalance.clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let safe_spread = if spread_bps.is_finite() && spread_bps >= 0.0 {
            spread_bps.clamp(0.1, 100.0)
        } else {
            1.0
        };

        let safe_vol = if instantaneous_vol.is_finite() && instantaneous_vol > 0.0 {
            instantaneous_vol.clamp(0.001, 1.0)
        } else {
            0.01
        };

        // 1. Ingreso del observable a la cadena simplicial circular
        let input_potential = safe_ret + 0.1 * safe_ofi;
        let input_current = safe_ofi / (safe_spread * 0.1 + safe_vol);

        self.simplicial_potentials[self.cursor] = input_potential;
        self.simplicial_currents[self.cursor] = input_current;
        self.cursor = (self.cursor + 1) % DERHAM_SIMPLICIAL_NODES;
        if self.count < DERHAM_SIMPLICIAL_NODES {
            self.count += 1;
        }

        // Si tenemos menos de 3 nodos, emitir estado neutro inicial
        if self.count < 3 {
            return DeRhamHodgeReport::default();
        }

        // 2. Operador Exterior Discreto d y Codiferencial delta
        // Componente Exacta (Gradiente d \alpha): Diferencias hacia adelante de potencial
        let mut energy_exact = 0.0;
        let mut energy_coexact = 0.0;
        let mut energy_harmonic = 0.0;

        let n = self.count;
        let mut harmonic_mean = 0.0;
        for i in 0..n {
            harmonic_mean += self.simplicial_potentials[i];
        }
        harmonic_mean /= n as f64;

        // Energía propia (no centrada) de la corriente: escala de referencia de la parte coexacta.
        let mut current_energy = 0.0;

        for i in 0..n {
            let next_idx = (i + 1) % n;
            let prev_idx = (i + n - 1) % n;

            // Gradiente d\alpha_i = \alpha_{i+1} - \alpha_i
            let grad = self.simplicial_potentials[next_idx] - self.simplicial_potentials[i];
            energy_exact += grad * grad;

            // Rotacional / Vorticidad coexacta \delta\beta_i = j_{i+1} - 2*j_i + j_{i-1}
            let curl = self.simplicial_currents[next_idx] - 2.0 * self.simplicial_currents[i]
                + self.simplicial_currents[prev_idx];
            energy_coexact += curl * curl;

            // Componente armónica: desviación respecto a la media global
            let harm = self.simplicial_potentials[i] - harmonic_mean;
            energy_harmonic += harm * harm;

            current_energy += self.simplicial_currents[i] * self.simplicial_currents[i];
        }

        // 3. Adimensionalización. El potencial (retorno + 0.1·OFI) y la corriente
        // (OFI / (0.1·spread + σ)) tienen unidades distintas: la corriente es 10²–10³ veces mayor en
        // amplitud, así que sumar sus energías crudas le daba a la parte coexacta ≈100 % de la energía
        // con cualquier OFI no constante. Cada parte se mide como cociente de Rayleigh respecto a la
        // energía de su propio campo: la exacta respecto a ‖α − ᾱ‖² (queda en [0, 4] en el ciclo) y la
        // coexacta respecto a ‖j‖² sin centrar (queda en [0, 16]), de modo que una corriente
        // estacionaria no cuenta como vorticidad y el vórtice alternado sí. La armónica pasa a ser la
        // unidad (vale 1 si α no es constante). Las tres fracciones son una partición adimensional que
        // suma 1, no una descomposición ortogonal: la identidad de Hodge del encabezado no se cumple
        // con dos campos distintos, ni antes ni ahora.
        let potential_scale = energy_harmonic.max(self.params.epsilon);
        let current_scale = current_energy.max(self.params.epsilon);
        let energy_exact = energy_exact / potential_scale;
        let energy_harmonic = energy_harmonic / potential_scale;
        let energy_coexact = energy_coexact / current_scale;

        // Normalización de las energías de Hodge
        let total_energy =
            (energy_exact + energy_coexact + energy_harmonic).max(self.params.epsilon);
        let ratio_exact = (energy_exact / total_energy).clamp(0.0, 1.0);
        let ratio_coexact = (energy_coexact / total_energy).clamp(0.0, 1.0);
        let ratio_harmonic = (energy_harmonic / total_energy).clamp(0.0, 1.0);

        // 4. Laplaciano de Hodge continuo y Betti-1
        let laplacian_energy = energy_exact + energy_coexact;
        let betti_one = (ratio_harmonic / (ratio_harmonic + ratio_coexact + self.params.epsilon))
            .clamp(0.0, 1.0);

        // 5. Señal continua sintetizada:
        // Proyecta la fuerza de gradiente pura (exacta) modulada por el signo armónico,
        // purgando la contaminación rotacional coexacta
        let raw_signal = -(self.simplicial_potentials[(self.cursor + n - 1) % n] - harmonic_mean);
        let damping = (1.0 - ratio_coexact).max(0.05);
        let synthesized = (raw_signal * damping * 10.0).clamp(-1.0, 1.0);

        let turbulent = ratio_coexact >= self.params.coexact_turbulent_threshold;

        DeRhamHodgeReport {
            exact_energy_ratio: ratio_exact,
            coexact_energy_ratio: ratio_coexact,
            harmonic_energy_ratio: ratio_harmonic,
            laplacian_dirichlet_energy: laplacian_energy,
            betti_one_persistence: betti_one,
            synthesized_signal: synthesized,
            turbulent_vorticity_alert: turbulent,
        }
    }
}
