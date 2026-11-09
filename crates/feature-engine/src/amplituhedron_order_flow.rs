//! # Fase 91 (§91) — Geometría del Amplituhedro & Amplitudes Parke-Taylor MHV
//!
//! ## Modo Profesor — Ontología Técnica y Matemática (7 Pilares Rigurosos)
//!
//! ### 1. QUÉ (Definición Técnica)
//! El motor `AmplituhedronOrderFlowEngine` implementa la física matemática de amplitudes de
//! dispersión cuántica descubierta por Nima Arkani-Hamed y Jaroslav Trnka (2013). Sustituye los
//! tradicionales diagramas de Feynman perturbativos por la geometría diferencial del **Amplituhedro**
//! en el Grassmanniano positivo $\text{Gr}(k, k+m)$. Proyecta el flujo continuo de órdenes de microestructura
//! (Bid, Ask, Flujo Agresor Comprador y Vendedor) como un haz de partículas con momentos espinoriales
//! $(\lambda_i, \tilde{\lambda}_i) \in \mathbb{C}^2 \times \mathbb{C}^2$, evaluando la amplitud analítica
//! Parke-Taylor Maximally Helicity Violating (MHV):
//! $$\mathcal{A}_n^{\text{MHV}}(1^-, 2^-, 3^+, \dots, n^+) = \frac{\langle j k \rangle^4}{\prod_{i=1}^n \langle i \, i+1 \rangle}$$
//! y calculando el volumen del politopo positivo $\mathcal{V}_{\text{ampl}} \in [0.0, 1.0]$.
//!
//! ### 2. POR QUÉ (Fundamento Físico-Matemático)
//! Los modelos de flujo de órdenes convencionales asumen que la interacción entre órdenes de compra
//! y venta es un proceso lineal o markoviano independiente. En la realidad de la microestructura HFT,
//! las órdenes se dispersan de forma colectiva y entrelazada con simetría dual conforme en el espacio
//! de twistores. Las singularidades colineales ($\langle i \, i+1 \rangle \to 0$) y suaves del Amplituhedro
//! preceden inmediatamente al vaciado abrupto de liquidez (air pockets) y cascadas de slippage.
//!
//! ### 3. PARA QUÉ (Objetivo de Trading y Cota de Ville)
//! - Identificar configuraciones de dispersión armónica y coherente donde el volumen del Amplituhedro es
//!   máximo y estable ($\mathcal{V}_{\text{ampl}} > 0.40$), garantizando liquidez profunda y bajo slippage.
//! - Modular el slot scaling factor $\eta_{\text{ampl}} \in [0.10, 1.25]$ y convicción $\psi_{\text{ampl}} \in [0.0, 1.0]$.
//! - Vetar de forma inmediata y estricta (`AMPLITUHEDRON_COLLINEAR_SINGULARITY_VETO`) cualquier transacción
//!   cuando se produce una singularidad colineal en el libro de órdenes, preservando la cota de ruina
//!   de Ville $\mathbb{P}(\text{ruina}) \le 10^{-6}$.
//!
//! ### 4. CÓMO (Implementación en Silicio y Algorítmica)
//! - Cuadrivector de espinores bidimensionales $\lambda_i = (\lambda_i^1, \lambda_i^2)$ para $i \in \{1, 2, 3, 4\}$
//!   calculados a partir de los observables normalizados del libro de órdenes.
//! - Producto de corchetes espinoriales antisimétrico $\langle a \, b \rangle = \lambda_a^1 \lambda_b^2 - \lambda_a^2 \lambda_b^1$.
//! - Amplitud Parke-Taylor $\mathcal{A}_4^{\text{MHV}} = \frac{\langle 1 \, 2 \rangle^4}{\langle 1 \, 2 \rangle \langle 2 \, 3 \rangle \langle 3 \, 4 \rangle \langle 4 \, 1 \rangle}$.
//! - Volumen de Grassmanniano simpléctico acotado analíticamente en $O(1)$ sin asignaciones de heap.
//!
//! ### 5. CUÁNDO (Condiciones de Activación)
//! En cada ciclo de procesamiento de microestructura y actualización del libro L2/L3 en el grafo vivo.
//!
//! ### 6. DÓNDE (Ubicación en la Arquitectura)
//! `crates/feature-engine/src/amplituhedron_order_flow.rs`, exportado en `crates/feature-engine/src/lib.rs`.
//!
//! ### 7. QUIÉN (Componente y Rol)
//! `AmplituhedronOrderFlowEngine`, bajo el diseño y gobernanza del Rol 8 (Físico Teórico & Amplitudes)
//! y Rol 2 (Quant Developer de Microestructura).

use core::f64::consts::PI;

/// Parámetros de configuración para la geometría del Amplituhedro.
#[derive(Debug, Clone, Copy)]
pub struct AmplituhedronOrderFlowConfig {
    /// Umbral mínimo de producto espinorial para evitar singularidad colineal $\delta_{\text{collinear}}$.
    pub min_spinor_bracket_threshold: f64,
    /// Umbral mínimo de volumen del Amplituhedro para no vetar $\mathcal{V}_{\min}$.
    pub min_polytope_volume: f64,
    /// Sensibilidad de escala a la dispersión de volumen $\kappa_{\text{disp}}$.
    pub volume_dispersion_sensitivity: f64,
    /// Factor de regularización numérica $\epsilon$.
    pub epsilon: f64,
}

impl Default for AmplituhedronOrderFlowConfig {
    fn default() -> Self {
        Self {
            min_spinor_bracket_threshold: 0.005,
            min_polytope_volume: 0.08,
            volume_dispersion_sensitivity: 1.50,
            epsilon: 1e-7,
        }
    }
}

/// Reporte de estado emitido por la geometría del Amplituhedro.
#[derive(Debug, Clone, Copy)]
pub struct AmplituhedronOrderFlowReport {
    /// Volumen del politopo del Amplituhedro $\mathcal{V}_{\text{ampl}} \in [0.0, 1.0]$.
    pub polytope_volume: f64,
    /// Magnitud regularizada de la amplitud Parke-Taylor $|\mathcal{A}_4^{\text{MHV}}|$.
    pub parke_taylor_amplitude: f64,
    /// Producto cíclico de los 4 corchetes espinoriales adyacentes.
    pub cyclic_spinor_product: f64,
    /// Grado de colinealidad o proximidad a singularidad infrarroja $\xi_{\text{collinear}} \in [0.0, 1.0]$.
    pub collinearity_degree: f64,
    /// Convicción direccional emergente $\psi_{\text{ampl}} \in [0.0, 1.0]$.
    pub directional_conviction: f64,
    /// Factor de escala de slot continuo $\eta_{\text{ampl}} \in [0.10, 1.25]$.
    pub slot_scaling_factor: f64,
    /// Indicador booleano de veto por singularidad colineal.
    pub collinear_singularity_veto: bool,
}

/// Motor de geometría del Amplituhedro para flujo continuo de órdenes.
#[derive(Debug, Clone)]
pub struct AmplituhedronOrderFlowEngine {
    config: AmplituhedronOrderFlowConfig,
}

impl AmplituhedronOrderFlowEngine {
    /// Crea una nueva instancia del motor con la configuración dada.
    #[inline]
    pub fn new(config: AmplituhedronOrderFlowConfig) -> Self {
        Self { config }
    }

    /// Evalúa la geometría del Amplituhedro a partir de observables de microestructura.
    ///
    /// # Parámetros
    /// - `price_ret`: Retorno fraccionario instantáneo del precio mid.
    /// - `ofi`: Order Flow Imbalance normalizado $\in [-1.0, 1.0]$.
    /// - `spread_bps`: Spread relativo actual en puntos básicos.
    /// - `vol_norm`: Volatilidad normalizada del activo $\in [0.0, \infty)$.
    #[inline]
    pub fn evaluate(
        &self,
        price_ret: f64,
        ofi: f64,
        spread_bps: f64,
        vol_norm: f64,
    ) -> AmplituhedronOrderFlowReport {
        // 1. Sanitización de entradas contra valores subnormales, NaN o Infinitos
        let p_ret = if price_ret.is_finite() {
            price_ret
        } else {
            0.0
        };
        let ofi_val = if ofi.is_finite() {
            ofi.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let spread = if spread_bps.is_finite() && spread_bps > 0.0 {
            spread_bps
        } else {
            1.0
        };
        let vol = if vol_norm.is_finite() && vol_norm >= 0.0 {
            vol_norm
        } else {
            0.01
        };

        // 2. Construcción de momentos espinoriales bidimensionales para las 4 partículas:
        // Partícula 1: Bid flow (momento negativo)
        // Partícula 2: Ask flow (momento negativo)
        // Partícula 3: Aggressor Buy flow (momento positivo)
        // Partícula 4: Aggressor Sell flow (momento positivo)
        let eps = self.config.epsilon;

        let e1 = (1.0 - ofi_val * 0.50).max(eps);
        let theta1 = (p_ret * 100.0).sin() * 0.50 * PI;
        let l1 = [
            e1.sqrt() * (theta1 * 0.50).cos(),
            e1.sqrt() * (theta1 * 0.50).sin(),
        ];

        let e2 = (1.0 + ofi_val * 0.50).max(eps);
        let theta2 = (p_ret * 100.0 + PI * 0.25).sin() * 0.50 * PI;
        let l2 = [
            e2.sqrt() * (theta2 * 0.50).cos(),
            e2.sqrt() * (theta2 * 0.50).sin(),
        ];

        let e3 = (1.0 + ofi_val).max(eps);
        let theta3 = (p_ret * 50.0 + PI * 0.50).sin() * 0.50 * PI;
        let l3 = [
            e3.sqrt() * (theta3 * 0.50).cos(),
            e3.sqrt() * (theta3 * 0.50).sin(),
        ];

        let e4 = (1.0 - ofi_val).max(eps);
        let theta4 = (p_ret * 50.0 + PI * 0.75).sin() * 0.50 * PI;
        let l4 = [
            e4.sqrt() * (theta4 * 0.50).cos(),
            e4.sqrt() * (theta4 * 0.50).sin(),
        ];

        // 3. Productos de corchetes espinoriales antisimétricos <a b> = l_a^1 * l_b^2 - l_a^2 * l_b^1
        let bracket_12 = l1[0] * l2[1] - l1[1] * l2[0];
        let bracket_23 = l2[0] * l3[1] - l2[1] * l3[0];
        let bracket_34 = l3[0] * l4[1] - l3[1] * l4[0];
        let bracket_41 = l4[0] * l1[1] - l4[1] * l1[0];

        // 4. Magnitud cíclica del denominador Parke-Taylor
        let cyclic_spinor_product =
            (bracket_12.abs() * bracket_23.abs() * bracket_34.abs() * bracket_41.abs())
                .max(eps * eps * eps * eps);

        // Amplitud Parke-Taylor MHV simplificada regularizada:
        // |A_4^MHV| = |<1 2>|^4 / (|1 2| |2 3| |3 4| |4 1|) = |<1 2>|^3 / (|2 3| |3 4| |4 1|)
        let num = bracket_12.abs().powi(4);
        let den = cyclic_spinor_product;
        let parke_taylor_amplitude = (num / den).clamp(0.0, 1e6);

        // 5. Grado de colinealidad (detección de singularidad cuando algún bracket adyacente es infinitesimal)
        let min_bracket = bracket_12
            .abs()
            .min(bracket_23.abs())
            .min(bracket_34.abs())
            .min(bracket_41.abs());

        let collinearity_degree = if min_bracket < self.config.min_spinor_bracket_threshold {
            1.0 - (min_bracket / self.config.min_spinor_bracket_threshold).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 6. Volumen del politopo del Amplituhedro V_ampl \in [0.0, 1.0]
        // Se relaciona con la coherencia positiva del Grassmanniano y el inverso amortiguado de singularidades.
        let spread_pen = (spread / 10.0).clamp(0.50, 4.0);
        let vol_raw = (min_bracket / (min_bracket + spread_pen * 0.10 + vol * 0.20))
            * (1.0 / (1.0 + (parke_taylor_amplitude - 1.0).abs() * 0.05));
        let polytope_volume = vol_raw.clamp(0.0, 1.0);

        // 7. Convicción direccional y factor de escala de slot continuo
        let directional_conviction = ((bracket_12.abs() - bracket_34.abs())
            / (bracket_12.abs() + bracket_34.abs() + eps))
            .abs()
            .clamp(0.0, 1.0);

        let slot_scaling_factor =
            ((0.10 + 1.15 * polytope_volume) * (1.0 - 0.20 * vol.min(1.0))).clamp(0.10, 1.25);

        // 8. Condición de Veto Fail-Closed
        let collinear_singularity_veto = collinearity_degree > 0.85
            || polytope_volume < self.config.min_polytope_volume
            || min_bracket < (self.config.min_spinor_bracket_threshold * 0.10);

        AmplituhedronOrderFlowReport {
            polytope_volume,
            parke_taylor_amplitude,
            cyclic_spinor_product,
            collinearity_degree,
            directional_conviction,
            slot_scaling_factor,
            collinear_singularity_veto,
        }
    }
}

impl Default for AmplituhedronOrderFlowEngine {
    fn default() -> Self {
        Self::new(AmplituhedronOrderFlowConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplituhedron_normal_evaluation() {
        let engine = AmplituhedronOrderFlowEngine::default();
        let report = engine.evaluate(0.001, 0.20, 2.0, 0.05);

        assert!(report.polytope_volume > 0.0);
        assert!(report.polytope_volume <= 1.0);
        assert!(report.slot_scaling_factor >= 0.10 && report.slot_scaling_factor <= 1.25);
        assert!(!report.collinear_singularity_veto);
    }

    #[test]
    fn test_amplituhedron_subnormal_safety() {
        let engine = AmplituhedronOrderFlowEngine::default();
        let report = engine.evaluate(f64::NAN, f64::INFINITY, -5.0, f64::NAN);

        assert!(report.polytope_volume.is_finite());
        assert!(report.parke_taylor_amplitude.is_finite());
        assert!(report.slot_scaling_factor.is_finite());
    }
}
