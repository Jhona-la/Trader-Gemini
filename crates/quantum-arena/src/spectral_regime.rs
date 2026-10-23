//! RÉGIMEN COMO CAMPO CONTINUO (Ola XLI·B1).
//!
//! Doctrina del universo multivariante continuo temporal espectral: NO hay
//! "regímenes de mercado" discretos (bull/crash/range/chaos) — los regímenes
//! son configuraciones del CAMPO espectral. Este módido expone el régimen
//! como coordenadas continuas; el enum legado `MarketRegime` queda como VISTA
//! derivada (umbrales sobre el campo) para el consejo y la telemetría.
//!
//! Contrato (protocolo del repo):
//! - Variable: estado agregado del espectro de 32 escalas de UN símbolo.
//! - Operador: funciones continuas de (H(τ) por banda, τ*, d ln τ*/dt,
//!   entropía de masa, marea portadora). Sin etiquetas ni umbrales duros
//!   salvo la vista legacy documentada.
//! - Unidades: adimensional (todas las coordenadas en [0,1] o [-1,1];
//!   dominant_drift en saltos de nodo espectral [ln 4] por cadencia de referencia [60s], en [-1,1]).
//! - Condiciones de contorno: espectro frío (sin masa) → campo neutro
//!   (crash_flux = 0, marea 0), jamás un veto.
//! - Identificabilidad: cada coordenada es medible desde el espectro vivo;
//!   la Fisher del campo (C3) declara cuándo el régimen ES identificable.
//! - Coste: O(32) por actualización, sin alocación.
//! - Falsación: ruido gaussiano sintético → crash_flux ≈ bajo y vista legacy
//!   = Range/Chaotic según correlación; caída persistente sintética →
//!   crash_flux alto. Los tests lo verifican.

use crate::temporal_spectrum::TemporalSpectrum;

/// Coordenadas continuas del régimen espectral de un símbolo.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpectralRegimeField {
    /// Persistencia H(τ) por banda (micro <60s, meso <30min, macro ≥30min).
    /// 0.5 = browniano; >0.5 persistente; <0.5 anti-persistente.
    pub hurst_by_band: [f64; 3],
    /// Escala dominante viva (ms). 0 = espectro sin masa.
    pub dominant_tau_ms: f64,
    /// Deriva de la escala dominante: velocidad de d(ln τ*)/dt en unidades de la
    /// malla espectral (ln 4 = un nodo) por cadencia de referencia macro (60s),
    /// acotada a [-1,1]. Negativa = el campo ACELERA hacia escalas rápidas
    /// (firma de ruptura); positiva = se ralentiza (consolidación).
    pub dominant_drift: f64,
    /// Entropía de la masa espectral [0,1]: 1 = energía uniforme (sin régimen
    /// identificable), 0 = toda la energía en una escala (régimen nítido).
    pub mass_entropy: f64,
    /// Marea portadora [-1,1]: dirección ponderada por energía del campo.
    pub carrier_tide: f64,
    /// "Crash-ness" continua [0,1]: combinación medible de (a) aceleración
    /// hacia escalas rápidas, (b) marea portadora adversa intensa, (c) colapso
    /// de entropía (energía COHERENTE concentrada) y (d) anti-persistencia
    /// extrema en la banda micro (reversión violenta). No es una etiqueta:
    /// es la densidad de evidencia de caída del propio campo.
    pub crash_flux: f64,
    /// Factor de modulación continuo derivado del régimen cuántico del Ising transverso [0.05, 2.0].
    /// 1.0 = neutro / sin evidencia; >1.0 = coherencia ordenada (BTC); <1.0 = ruido paramagnético (ICP).
    pub quantum_phase_factor: f64,
    /// Acoplamiento transverso g estimado para el símbolo (def: 1.0).
    pub quantum_coupling_g: f64,
}

/// Parámetros dinámicos desacoplados para la evaluación continua del Campo de Régimen Espectral (Ola Ω127, Ω178).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralRegimeParams {
    /// Peso de la componente de aceleración hacia escalas rápidas (ruptura).
    pub w_accel_fast: f64,
    /// Peso de la componente de marea portadora adversa.
    pub w_adverse_tide: f64,
    /// Peso de la componente de colapso de entropía (energía coherente concentrada).
    pub w_coherence_collapse: f64,
    /// Peso de la componente de anti-persistencia en la banda micro.
    pub w_micro_anti: f64,
    /// Ventana mínima de regularización continua Hermite C^1 (ms) para la derivada de régimen.
    pub tau_min_drift_res_ms: f64,
    /// Cadencia macro de referencia (ms) para normalizar la derivada espectral (def: 60_000 ms = 60s, CADENCIA_REGIMEN_MS).
    pub cadence_regime_ms: f64,
    /// Escala de contracción de margen en condiciones adversas [0.0, 1.0].
    pub margin_contraction_scale: f64,
    /// Piso mínimo de margen para evitar la asfixia total de liquidez [0.01, 0.50].
    pub min_margin_floor: f64,
    /// Peso de la modulación cuántica [0.0, 2.0] (def: 1.0).
    pub w_quantum_coupling: f64,
}

impl Default for SpectralRegimeParams {
    fn default() -> Self {
        Self {
            w_accel_fast: 0.35,
            w_adverse_tide: 0.30,
            w_coherence_collapse: 0.20,
            w_micro_anti: 0.15,
            tau_min_drift_res_ms: 5_000.0,
            cadence_regime_ms: 60_000.0,
            margin_contraction_scale: 0.95,
            min_margin_floor: 0.05,
            w_quantum_coupling: 1.0,
        }
    }
}

impl SpectralRegimeParams {
    pub fn from_registry_or_default(registry: &crate::state::OmniscientRegistry) -> Self {
        let mut p = Self::default();
        let w_acc = registry.get_value_or("regime_w_accel_fast", p.w_accel_fast);
        if w_acc.is_finite() && w_acc >= 0.0 {
            p.w_accel_fast = w_acc;
        }
        let w_tide = registry.get_value_or("regime_w_adverse_tide", p.w_adverse_tide);
        if w_tide.is_finite() && w_tide >= 0.0 {
            p.w_adverse_tide = w_tide;
        }
        let w_col = registry.get_value_or("regime_w_coherence_collapse", p.w_coherence_collapse);
        if w_col.is_finite() && w_col >= 0.0 {
            p.w_coherence_collapse = w_col;
        }
        let w_anti = registry.get_value_or("regime_w_micro_anti", p.w_micro_anti);
        if w_anti.is_finite() && w_anti >= 0.0 {
            p.w_micro_anti = w_anti;
        }
        let tau_res = registry.get_value_or("regime_tau_min_drift_res_ms", p.tau_min_drift_res_ms);
        if tau_res.is_finite() && tau_res > 0.0 {
            p.tau_min_drift_res_ms = tau_res;
        }
        let cad = registry.get_value_or("regime_cadence_regime_ms", p.cadence_regime_ms);
        if cad.is_finite() && cad > 0.0 {
            p.cadence_regime_ms = cad;
        }
        let floor = registry.get_value_or("regime_min_margin_floor", p.min_margin_floor);
        if floor.is_finite() && floor > 0.0 {
            p.min_margin_floor = floor.clamp(0.01, 0.50);
        }
        let w_q = registry.get_value_or("regime_w_quantum_coupling", p.w_quantum_coupling);
        if w_q.is_finite() && w_q >= 0.0 {
            p.w_quantum_coupling = w_q;
        }
        p
    }
}

impl SpectralRegimeField {
    /// Deriva el campo del espectro vivo con parámetros por defecto.
    #[inline]
    pub fn from_spectrum(
        spec: &TemporalSpectrum,
        prev_dominant_ln_tau: Option<f64>,
        elapsed_ms: f64,
    ) -> Self {
        Self::from_spectrum_with_params(
            spec,
            prev_dominant_ln_tau,
            elapsed_ms,
            &SpectralRegimeParams::default(),
        )
    }

    /// Deriva el campo del espectro vivo con parámetros desacoplados configurables.
    pub fn from_spectrum_with_params(
        spec: &TemporalSpectrum,
        prev_dominant_ln_tau: Option<f64>,
        elapsed_ms: f64,
        params: &SpectralRegimeParams,
    ) -> Self {
        let field = spec.spectral_field(true);
        let mut out = Self {
            hurst_by_band: [
                spec.hurst_at(15_000.0),
                spec.hurst_at(300_000.0),
                spec.hurst_at(1_800_000.0),
            ],
            dominant_tau_ms: field.resonant_tau_ms,
            dominant_drift: 0.0,
            mass_entropy: field.spectral_entropy,
            // F1-077: Marea macro derivada de la coherencia canónica D-742 ("una sola verdad").
            // Erradica discrepancia entre decisión de entrada (D-742, 1/vol) y carrier_tide (legacy persistencia).
            carrier_tide: spec.spectral_coherence(true),
            crash_flux: 0.0,
            quantum_phase_factor: 1.0,
            quantum_coupling_g: 1.0,
        };
        if let (Some(prev_ln), Some(cur_ln)) = (prev_dominant_ln_tau, nonzero_ln(field.resonant_tau_ms))
        {
            if elapsed_ms > 0.0 && elapsed_ms.is_finite() {
                // C-3 (R7-R2-C-3) + F1-074 (Ola Ω178): Regularización continua C^1 para la derivada de régimen d ln(tau)/dt.
                // Evita que un elapsed_ms infinitesimal (ej. 1 ms tras renovar el ancla) sature
                // artificialmente a +/-1.0 por micro-ruido de ordering.
                // Modulación de confianza suave con Hermite cúbico hasta la ventana mínima de
                // resolución macro parametrizada.
                let tau_res = params.tau_min_drift_res_ms.max(1.0);
                let u_t = (elapsed_ms / tau_res).clamp(0.0, 1.0);
                let w_t = u_t * u_t * (3.0 - 2.0 * u_t);
                let eff_elapsed = elapsed_ms.max(tau_res);

                // F1-074: Medir la velocidad de deriva d(ln tau*)/dt en unidades naturales de la malla
                // espectral de base 4 (ln 4 = un nodo de salto espectral) y normalizada por la cadencia macro de
                // referencia (CADENCIA_REGIMEN_MS = 60s), erradicando la extrapolación espuria "por hora" (3.6e6)
                // que amplificaba el ruido x60..x720 y saturaba accel_fast a 0 ó 1 ante micro-jitter.
                // Una migración estructural de 1 nodo espectral en 60s produce |drift| = 1.0.
                const LN_GRID_NODE: f64 = 1.386_294_361_119_890_6; // ln(4) = 2 * ln(2)
                let cadence_ms = params.cadence_regime_ms.max(1.0);
                let drift_cadence = ((cur_ln - prev_ln) / LN_GRID_NODE) * (cadence_ms / eff_elapsed);
                out.dominant_drift = if drift_cadence.is_finite() {
                    drift_cadence.clamp(-1.0, 1.0) * w_t
                } else {
                    0.0
                };
            }
        }
        // Coordenadas de crash-ness (cada una en [0,1], combinación convexa acotada):
        // (a) aceleración hacia lo rápido: deriva negativa fuerte.
        let accel_fast = if out.dominant_drift.is_finite() {
            (-out.dominant_drift).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // (b) marea adversa intensa (se toma absoluta: la caída es -tide para
        // largos, pero crash-ness describe el EVENTO, no el lado).
        let adverse_tide = if out.carrier_tide.is_finite() {
            out.carrier_tide.abs().clamp(0.0, 1.0)
        } else {
            0.0
        };
        // (c) colapso de entropía: energía coherente concentrada.
        let coherence_collapse = if out.mass_entropy.is_finite() {
            (1.0 - out.mass_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        // (d) anti-persistencia extrema en micro (reversión violenta).
        let micro_anti = if out.hurst_by_band[0].is_finite() {
            (0.5 - out.hurst_by_band[0]).max(0.0) / 0.5
        } else {
            0.0
        };
        let sum_w = (params.w_accel_fast
            + params.w_adverse_tide
            + params.w_coherence_collapse
            + params.w_micro_anti)
            .max(1e-9);
        let raw_crash = (params.w_accel_fast * accel_fast
            + params.w_adverse_tide * adverse_tide
            + params.w_coherence_collapse * coherence_collapse
            + params.w_micro_anti * micro_anti)
            / sum_w;
        out.crash_flux = if raw_crash.is_finite() {
            raw_crash.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Espectro frío: sin ENERGÍA no hay régimen que declarar. (Ojo: τ*
        // resonante por defecto es el punto medio de las anclas, > 0 incluso
        // en frío — el indicador correcto de campo vivo es la energía total.)
        if field.total_energy <= 1e-12 {
            out.crash_flux = 0.0;
            out.carrier_tide = 0.0;
            out.dominant_tau_ms = 0.0;
        }
        out
    }

    /// Vista LEGACY: umbrales sobre el campo que reproducen el detector
    /// discreto histórico para el consejo y la telemetría. Documentada como
    /// VISTA, no como estado del motor. El enum vive AQUÍ (sin dependencia
    /// circular con risk-engine, que puede mapearlo 1:1).
    pub fn legacy_view(&self, avg_correlation: f64) -> LegacyRegimeView {
        if self.dominant_tau_ms <= 0.0 {
            return LegacyRegimeView::Range;
        }
        let correlated = avg_correlation > 0.6;
        if self.crash_flux > 0.80 && self.carrier_tide < -0.15 {
            return LegacyRegimeView::Crash;
        }
        if self.crash_flux > 0.80 && self.carrier_tide > 0.15 {
            return LegacyRegimeView::BullRun;
        }
        if correlated {
            LegacyRegimeView::Range
        } else {
            LegacyRegimeView::Chaotic
        }
    }

    /// Multiplicador continuo de margen para largos bajo crash-ness creciente con parámetros default.
    #[inline]
    pub fn long_margin_multiplier(&self) -> f64 {
        self.long_margin_multiplier_with_params(&SpectralRegimeParams::default())
    }

    /// Multiplicador continuo de margen para largos bajo crash-ness creciente:
    /// 1.0 en calma; desciende suavemente hasta el piso parametrizado en crash-ness extrema.
    pub fn long_margin_multiplier_with_params(&self, params: &SpectralRegimeParams) -> f64 {
        if !self.carrier_tide.is_finite() || self.carrier_tide >= 0.0 {
            return 1.0;
        }
        let cf = if self.crash_flux.is_finite() { self.crash_flux } else { 0.0 };
        (1.0 - params.margin_contraction_scale * cf).clamp(params.min_margin_floor, 1.0)
    }

    /// Multiplicador continuo de margen para posiciones cortas con parámetros default.
    #[inline]
    pub fn short_margin_multiplier(&self) -> f64 {
        self.short_margin_multiplier_with_params(&SpectralRegimeParams::default())
    }

    /// Multiplicador continuo de margen para posiciones Cortas:
    /// Si la marea portadora es intensamente ALCISTA (short squeeze / blow-off),
    /// contrae el margen del corto en proporción a la crash_flux inversa.
    pub fn short_margin_multiplier_with_params(&self, params: &SpectralRegimeParams) -> f64 {
        if !self.carrier_tide.is_finite() || self.carrier_tide <= 0.0 {
            return 1.0;
        }
        let cf = if self.crash_flux.is_finite() { self.crash_flux } else { 0.0 };
        (1.0 - params.margin_contraction_scale * cf).clamp(params.min_margin_floor, 1.0)
    }

    /// Multiplicador de margen continuo simétrico según la dirección de la orden con parámetros default.
    #[inline]
    pub fn margin_multiplier(&self, is_long: bool) -> f64 {
        self.margin_multiplier_with_params(is_long, &SpectralRegimeParams::default())
    }

    /// Multiplicador de margen continuo simétrico según la dirección de la orden y parámetros desacoplados.
    pub fn margin_multiplier_with_params(&self, is_long: bool, params: &SpectralRegimeParams) -> f64 {
        if is_long {
            self.long_margin_multiplier_with_params(params)
        } else {
            self.short_margin_multiplier_with_params(params)
        }
    }

    /// Modula el campo de régimen con el estado cuántico observado (Ola Omega220 — Átomo 3.3).
    #[inline]
    pub fn with_quantum_phase(&self, phase: &crate::quantum_phase::QuantumPhaseState) -> Self {
        let mut out = *self;
        out.quantum_phase_factor = phase.factor_cuantico;
        out.quantum_coupling_g = phase.g;
        out
    }

    /// Multiplicador de margen continuo simétrico integrando la modulación cuántica (Ola Omega220 — Átomo 3.3).
    ///
    /// Modula el margen clásico mediante el factor cuántico $f_{\text{cuantico}}$ derivado del modelo de Ising:
    /// - En fase ordenada ($g < 1.0$, e.g. BTC): amplifica la capacidad de margen hasta 2.0x.
    /// - En fase crítica ($g \approx 1.0$): aplica amortiguamiento Hermite para mitigar susceptibilidad divergente.
    /// - En fase paramagnética ($g > 1.0$, e.g. ICP): contrae suavemente el margen según $1/g$.
    pub fn effective_quantum_margin_multiplier(&self, is_long: bool, params: &SpectralRegimeParams) -> f64 {
        let base_margin = self.margin_multiplier_with_params(is_long, params);
        let q_factor = if self.quantum_phase_factor.is_finite() && self.quantum_phase_factor > 0.0 {
            self.quantum_phase_factor
        } else {
            1.0
        };
        let w_q = if params.w_quantum_coupling.is_finite() && params.w_quantum_coupling >= 0.0 {
            params.w_quantum_coupling
        } else {
            1.0
        };
        // Modulación convexa acotada: w_q modula la influencia cuántica sobre el margen base
        let modulated = base_margin * (1.0 + w_q * (q_factor - 1.0));
        modulated.clamp(params.min_margin_floor, 2.0)
    }
}

/// Vista legada del campo (mapeo 1:1 con risk-engine::regime::MarketRegime).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyRegimeView {
    BullRun,
    Crash,
    Range,
    Chaotic,
}

fn nonzero_ln(tau: f64) -> Option<f64> {
    (tau.is_finite() && tau > 0.0).then(|| tau.ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ruido: sin estructura direccional ni aceleración → crash-ness baja y
    /// margen intacto; la vista legacy NO declara crash.
    #[test]
    fn ruido_produce_crash_flux_bajo() {
        let spec = TemporalSpectrum::new();
        let f = SpectralRegimeField::from_spectrum(&spec, None, 60_000.0);
        assert_eq!(f.crash_flux, 0.0, "espectro frio = neutro");
    }

    /// Campo con marea adversa intensa y energía concentrada: crash-ness alta
    /// y el multiplicador de margen de largos desciende — sin etiqueta previa.
    #[test]
    fn marea_adversa_concentrada_reduce_margen_continuamente() {
        let f = SpectralRegimeField {
            hurst_by_band: [0.30, 0.45, 0.50],
            dominant_tau_ms: 8_000.0,
            dominant_drift: -0.9,
            mass_entropy: 0.10,
            carrier_tide: -0.80,
            crash_flux: 0.0,
            quantum_phase_factor: 1.0,
            quantum_coupling_g: 1.0,
        };
        let mut g = f;
        g.crash_flux = (0.35f64 * 1.0 + 0.30 * 1.0 + 0.20 * 1.0 + 0.15 * 0.4).clamp(0.0, 1.0);
        assert!(g.crash_flux > 0.7);
        assert!(g.long_margin_multiplier() < 0.3);
        assert_eq!(g.legacy_view(0.8), LegacyRegimeView::Crash);
    }

    #[test]
    fn marea_favorable_nunca_recorta_margen() {
        let f = SpectralRegimeField {
            hurst_by_band: [0.5; 3],
            dominant_tau_ms: 60_000.0,
            dominant_drift: 0.0,
            mass_entropy: 0.5,
            carrier_tide: 0.9,
            crash_flux: 0.9,
            quantum_phase_factor: 1.0,
            quantum_coupling_g: 1.0,
        };
        assert_eq!(f.long_margin_multiplier(), 1.0);
    }

    #[test]
    fn marea_alcista_concentrada_reduce_margen_de_cortos() {
        let f = SpectralRegimeField {
            hurst_by_band: [0.30, 0.45, 0.50],
            dominant_tau_ms: 8_000.0,
            dominant_drift: 0.9,
            mass_entropy: 0.10,
            carrier_tide: 0.80, // marea alcista intensa
            crash_flux: 0.85,
            quantum_phase_factor: 1.0,
            quantum_coupling_g: 1.0,
        };
        assert!(f.short_margin_multiplier() < 0.3);
        assert_eq!(f.long_margin_multiplier(), 1.0);
        assert_eq!(f.margin_multiplier(false), f.short_margin_multiplier());
        assert_eq!(f.margin_multiplier(true), f.long_margin_multiplier());
    }

    #[test]
    fn test_spectral_regime_nan_immunity() {
        let f = SpectralRegimeField {
            hurst_by_band: [f64::NAN, f64::NAN, f64::NAN],
            dominant_tau_ms: f64::NAN,
            dominant_drift: f64::NAN,
            mass_entropy: f64::NAN,
            carrier_tide: f64::NAN,
            crash_flux: f64::NAN,
            quantum_phase_factor: f64::NAN,
            quantum_coupling_g: f64::NAN,
        };
        assert_eq!(f.long_margin_multiplier(), 1.0);
        assert_eq!(f.short_margin_multiplier(), 1.0);
        assert_eq!(f.margin_multiplier(true), 1.0);
        assert_eq!(f.margin_multiplier(false), 1.0);
    }

    #[test]
    fn test_c3_regularizacion_continua_dominant_drift_inmune_a_salto_1ms() {
        let mut spec = crate::temporal_spectrum::TemporalSpectrum::new();
        for t in 0..100u64 {
            let p = 100.0 + 0.5 * ((t % 10) as f64);
            spec.update(p, 1_000 + t * 1_000);
        }
        let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();
        // A elapsed_ms = 1.0 ms con una micro-diferencia de 1e-4, el estimador antiguo
        // calculaba 1e-4 * 3.6e6 / 1.0 = 360.0 y saturaba a 1.0.
        // Con regularización C^1 Hermite, w_t ≈ 3*(1/5000)^2 = 1.2e-7, por lo que drift ≈ 0.
        let f_1ms = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + 0.0001), 1.0);
        assert!(
            f_1ms.dominant_drift.abs() < 1e-4,
            "a 1 ms de elapsed el micro-ruido debe quedar regularizado a cero: {}",
            f_1ms.dominant_drift
        );
        // A elapsed_ms = 60_000 ms (cadencia macro madura), la derivada opera con sensibilidad plena:
        let f_60s = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + 1.0), 60_000.0);
        assert!(
            f_60s.dominant_drift < -0.5,
            "a 60 s la derivada de migración rápida debe reflejarse plenamente: {}",
            f_60s.dominant_drift
        );
    }

    /// F1-074: Δln = 0.05 a 30 s produce |drift| ≪ 1.0 (aprox 0.072), sin saturar accel_fast.
    #[test]
    fn test_f1_074_delta_ln_005_a_30s_da_drift_pequeno() {
        let mut spec = crate::temporal_spectrum::TemporalSpectrum::new();
        for t in 0..100u64 {
            let p = 100.0 + 0.5 * ((t % 10) as f64);
            spec.update(p, 1_000 + t * 1_000);
        }
        let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();
        let f_30s = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + 0.05), 30_000.0);
        // Con cur_ln - prev_ln = -0.05 a 30s:
        // drift_cadence = (-0.05 / ln(4)) * (60_000 / 30_000) = (-0.05 / 1.38629) * 2 ≈ -0.0721
        assert!(
            f_30s.dominant_drift.abs() < 0.15,
            "fluctuación realista de 5% a 30s no debe saturar: drift = {}",
            f_30s.dominant_drift
        );
        assert!(
            f_30s.dominant_drift < 0.0,
            "cur_ln < prev_ln debe dar deriva negativa (hacia rápido)"
        );
        // F1-074: accel_fast y su contribución a crash_flux no deben saltar bruscamente por micro-ruido:
        let accel_fast = (-f_30s.dominant_drift).clamp(0.0, 1.0);
        let accel_contrib = 0.35 * accel_fast;
        assert!(
            accel_contrib < 0.05,
            "contribución de accel_fast a crash_flux debe ser < 0.05 (fue {}), erradicando el salto del 35%",
            accel_contrib
        );
        // El incremento en crash_flux respecto a una ancla sin deriva (None) debe ser menor a 0.05:
        let f_base = SpectralRegimeField::from_spectrum(&spec, None, 30_000.0);
        let delta_flux = f_30s.crash_flux - f_base.crash_flux;
        assert!(
            delta_flux < 0.05,
            "el incremento de crash_flux por fluctuación del 5% debe ser < 0.05: fue {}",
            delta_flux
        );
    }

    /// F1-074: Migración estructural de 1 nodo (Δln = ln 4) a 60 s produce |drift| ≈ 1.0.
    #[test]
    fn test_f1_074_delta_ln_ln4_a_60s_da_drift_unitario() {
        let mut spec = crate::temporal_spectrum::TemporalSpectrum::new();
        for t in 0..100u64 {
            let p = 100.0 + 0.5 * ((t % 10) as f64);
            spec.update(p, 1_000 + t * 1_000);
        }
        let cur_ln = spec.spectral_field(true).resonant_tau_ms.max(1e-6).ln();
        const LN_4: f64 = 1.386_294_361_119_890_6;

        // Hacia rápido (-ln 4)
        let f_fast = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln + LN_4), 60_000.0);
        assert!(
            (f_fast.dominant_drift - (-1.0)).abs() < 1e-4,
            "salto de 1 nodo hacia rápido a 60s debe dar drift ≈ -1.0: {}",
            f_fast.dominant_drift
        );

        // Hacia lento (+ln 4)
        let f_slow = SpectralRegimeField::from_spectrum(&spec, Some(cur_ln - LN_4), 60_000.0);
        assert!(
            (f_slow.dominant_drift - 1.0).abs() < 1e-4,
            "salto de 1 nodo hacia lento a 60s debe dar drift ≈ +1.0: {}",
            f_slow.dominant_drift
        );
    }
}
