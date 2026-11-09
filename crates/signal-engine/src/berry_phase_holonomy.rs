//! # Fase 77 (§77) — Fase Geométrica de Berry y Holonomía Cuántica de Fubini-Study
//!
//! ## 1. QUÉ
//! Módulo de alta fidelidad que computa la fase geométrica abeliana de Berry $\gamma_C$ y el tensor cuántico
//! geométrico $Q_{\mu\nu} = g_{\mu\nu} - \frac{i}{2}\mathcal{F}_{\mu\nu}$ sobre la variedad de estados del libro
//! de órdenes $|\psi(\vec{R})\rangle$, evaluando la distancia métrica de Fubini-Study $ds_{\text{FS}}^2$
//! y la holonomía gauge inducida por trayectorias cíclicas en el espacio de parámetros $(s, v, \omega)$:
//! $$\gamma_C = \oint_C \vec{\mathcal{A}}(\vec{R}) \cdot d\vec{R} = \iint_\Sigma \vec{\mathcal{F}} \cdot d\vec{S}$$
//!
//! ## 2. POR QUÉ
//! Las oscilaciones dinámicas de spread, volatilidad y desbalance de flujo (OFI) describen curvas en el espacio
//! de parámetros del mercado. Si el ciclo es puramente conservativo y simétrico bajo inversión temporal $T$,
//! la fase de Berry es nula ($\gamma_C = 0$). Sin embargo, ante flujos informados o asimetrías de liquidez oscura,
//! la trayectoria encierra un ángulo sólido no nulo alrededor del punto de degeneración cuántica, adquiriendo
//! una fase de Berry topológica $\gamma_C \neq 0$ que actúa como una fuerza motriz direccional (dark alpha).
//!
//! ## 3. PARA QUÉ
//! - Medir la distancia cuántica métrica de Fubini-Study $ds_{\text{FS}}^2 = 1 - |\langle \psi_t | \psi_{t-1} \rangle|^2$.
//! - Extraer la curvatura de Berry $\vec{\mathcal{F}}(\vec{R}) = \frac{\vec{R}}{2 R^3}$ y el ángulo sólido subtendido $\Omega$.
//! - Generar el modulador de convicción direccional continuo $\mathcal{M}_{\text{berry}} \in [0.45, 1.40]$.
//! - Detectar ruptura adiabática (`adiabatic_breakdown`) cuando el estado cuántico sufre un salto discontinuo $ds_{\text{FS}}^2 \ge 0.85$.
//!
//! ## 4. CÓMO
//! 1. Se proyecta el estado del libro en el espacio tridimensional normalizado $\vec{R} = (s, v, \omega)$.
//! 2. Se computa el vector de estado cuántico normalizado $|\psi\rangle = \frac{\vec{R}}{\|\vec{R}\|}$.
//! 3. Se evalúa el producto interno $\langle \psi_t | \psi_{t-1} \rangle$ y la métrica de Fubini-Study.
//! 4. Se calcula el ángulo sólido esférico de 3 puntos consecutivos mediante la fórmula de van Oosterom & Strackee:
//!    $$\tan(\Omega/2) = \frac{\vec{R}_1 \cdot (\vec{R}_2 \times \vec{R}_3)}{R_1 R_2 R_3 + (\vec{R}_1 \cdot \vec{R}_2)R_3 + (\vec{R}_1 \cdot \vec{R}_3)R_2 + (\vec{R}_2 \cdot \vec{R}_3)R_1}$$
//! 5. Se acumula la fase de Berry $\gamma_C = -\frac{1}{2}\Omega$ en una ventana deslizante de memoria adiabática.
//! 6. Se modula la convicción direccional continua $\mathcal{M}_{\text{berry}} = 1.0 + \tanh(2.0 \cdot \gamma_C) \cdot 0.35$.
//!
//! ## 5. CUÁNDO
//! En cada pulso de síntesis de señales direccionales en `signal-engine`.
//!
//! ## 6. DÓNDE
//! `crates/signal-engine/src/berry_phase_holonomy.rs`, re-exportado en `lib.rs`.
//!
//! ## 7. QUIÉN
//! Diseñado e implementado por el Consejo de 10 Roles Senior (Físico Teórico, Quant Developer, QA Engineer).

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BerryPhaseConfig {
    /// Longitud de memoria de la trayectoria cíclica en el espacio de parámetros (número de pasos).
    pub cycle_window_len: usize,
    /// Umbral para ruptura adiabática en distancia de Fubini-Study.
    pub adiabatic_breakdown_threshold: f64,
    /// Sensibilidad de escala para modulación de convicción.
    pub conviction_sensitivity: f64,
}

impl Default for BerryPhaseConfig {
    fn default() -> Self {
        Self {
            cycle_window_len: 16,
            adiabatic_breakdown_threshold: 0.85,
            conviction_sensitivity: 0.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BerryPhaseReport {
    /// Distancia cuántica métrica de Fubini-Study entre estados consecutivos ds_FS^2.
    pub fubini_study_distance: f64,
    /// Fase acumulada geométrica de Berry gamma_C (radianes).
    pub berry_phase: f64,
    /// Magnitud de la curvatura de Berry promedio ||F||.
    pub berry_curvature_magnitude: f64,
    /// Indicador de ruptura adiabática (transición no conservativa violenta).
    pub adiabatic_breakdown: bool,
    /// Modulador holonómico de convicción direccional [0.45, 1.40].
    pub holographic_conviction_modulator: f64,
}

#[derive(Debug, Clone)]
pub struct BerryPhaseEngine {
    config: BerryPhaseConfig,
    history: [[f64; 3]; 16],
    history_count: usize,
    prev_state: [f64; 3],
    accumulated_berry_phase: f64,
}

impl BerryPhaseEngine {
    pub fn new(config: BerryPhaseConfig) -> Self {
        Self {
            config,
            history: [[0.0; 3]; 16],
            history_count: 0,
            prev_state: [1.0, 0.0, 0.0],
            accumulated_berry_phase: 0.0,
        }
    }

    /// Evalúa la holonomía cuántica de Berry en base al vector de microestructura:
    /// - `spread_norm`: spread normalizado (positivo).
    /// - `volatility_norm`: volatilidad instantánea normalizada (positiva).
    /// - `ofi_flow`: imbalance de flujo de órdenes (con signo en [-1.0, 1.0]).
    #[inline]
    pub fn evaluate(
        &mut self,
        spread_norm: f64,
        volatility_norm: f64,
        ofi_flow: f64,
    ) -> BerryPhaseReport {
        let r = [
            spread_norm.max(1e-4),
            volatility_norm.max(1e-4),
            ofi_flow,
        ];

        let norm_sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let norm = norm_sq.sqrt().max(1e-5);
        let current_state = [r[0] / norm, r[1] / norm, r[2] / norm];

        // Distancia infinitesimal de Fubini-Study: ds_FS^2 = 1 - |<psi_t | psi_{t-1}>|^2
        let dot = current_state[0] * self.prev_state[0]
            + current_state[1] * self.prev_state[1]
            + current_state[2] * self.prev_state[2];
        let overlap_sq = (dot * dot).min(1.0);
        let fs_dist = (1.0 - overlap_sq).max(0.0);

        // Curvatura de Berry para monopolo en el origen: ||F|| = 1 / (2 R^2)
        let curvature_mag = 1.0 / (2.0 * norm_sq.max(1e-4));

        // Ruptura adiabática
        let adiabatic_breakdown = fs_dist >= self.config.adiabatic_breakdown_threshold;

        // Si tenemos al menos 2 puntos previos en el historial, computamos el ángulo sólido con el triángulo esférico
        if self.history_count >= 2 {
            let p1 = self.history[(self.history_count - 2) % 16];
            let p2 = self.history[(self.history_count - 1) % 16];
            let p3 = current_state;

            let solid_angle = compute_solid_angle_van_oosterom(p1, p2, p3);
            let d_gamma = -0.5 * solid_angle;

            // Decaimiento exponencial suave de la fase de Berry acumulada (memoria adiabática)
            self.accumulated_berry_phase = self.accumulated_berry_phase * 0.95 + d_gamma;
        }

        // Registrar en buffer circular
        self.history[self.history_count % 16] = current_state;
        self.history_count += 1;
        self.prev_state = current_state;

        // Modulador continuo de convicción
        let berry_mod = (self.accumulated_berry_phase * 2.0).tanh() * self.config.conviction_sensitivity;
        let mut final_conviction = (1.0 + berry_mod).clamp(0.45, 1.40);

        if adiabatic_breakdown {
            final_conviction = (final_conviction * 0.50).max(0.45);
        }

        BerryPhaseReport {
            fubini_study_distance: fs_dist,
            berry_phase: self.accumulated_berry_phase,
            berry_curvature_magnitude: curvature_mag,
            adiabatic_breakdown,
            holographic_conviction_modulator: final_conviction,
        }
    }
}

/// Fórmula de van Oosterom y Strackee para el ángulo sólido de un triángulo esférico formado por 3 vectores unitarios:
/// tan(Omega / 2) = (R1 . (R2 x R3)) / (1 + R1.R2 + R1.R3 + R2.R3)
#[inline]
fn compute_solid_angle_van_oosterom(r1: [f64; 3], r2: [f64; 3], r3: [f64; 3]) -> f64 {
    // Producto vectorial R2 x R3
    let cross_x = r2[1] * r3[2] - r2[2] * r3[1];
    let cross_y = r2[2] * r3[0] - r2[0] * r3[2];
    let cross_z = r2[0] * r3[1] - r2[1] * r3[0];

    // Producto mixto R1 . (R2 x R3)
    let det = r1[0] * cross_x + r1[1] * cross_y + r1[2] * cross_z;

    let dot12 = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
    let dot13 = r1[0] * r3[0] + r1[1] * r3[1] + r1[2] * r3[2];
    let dot23 = r2[0] * r3[0] + r2[1] * r3[1] + r2[2] * r3[2];

    let denominator = 1.0 + dot12 + dot13 + dot23;
    if denominator.abs() < 1e-6 {
        0.0
    } else {
        2.0 * (det / denominator).atan()
    }
}
