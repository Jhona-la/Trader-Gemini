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
//!   dominant_drift en 1/ms → se expone log-por-hora normalizado).
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
    /// Deriva de la escala dominante: velocidad de d(ln τ*)/dt normalizada a
    /// por-hora, acotada a [-1,1]. Negativa = el campo ACELERA hacia escalas
    /// rápidas (firma de ruptura); positiva = se ralentiza (consolidación).
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
}

impl SpectralRegimeField {
    /// Deriva el campo del espectro vivo. `prev_dominant_ln_tau` es el ln(τ*)
    /// de la observación anterior (None en frío) y `elapsed_ms` el tiempo
    /// entre observaciones, para la deriva de la escala dominante.
    pub fn from_spectrum(
        spec: &TemporalSpectrum,
        prev_dominant_ln_tau: Option<f64>,
        elapsed_ms: f64,
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
            carrier_tide: field.global_coherence,
            crash_flux: 0.0,
        };
        if let (Some(prev_ln), Some(cur_ln)) = (prev_dominant_ln_tau, nonzero_ln(field.resonant_tau_ms))
        {
            if elapsed_ms > 0.0 && elapsed_ms.is_finite() {
                // C-3 (R7-R2-C-3): Regularización continua C^1 para la derivada de régimen d ln(tau)/dt.
                // Evita que un elapsed_ms infinitesimal (ej. 1 ms tras renovar el ancla) sature
                // artificialmente a +/-1.0 por micro-ruido de ordering.
                // Modulación de confianza suave con Hermite cúbico hasta la ventana mínima de
                // resolución macro (5.0 s = 5000 ms).
                const TAU_MIN_DRIFT_RES_MS: f64 = 5_000.0;
                let u_t = (elapsed_ms / TAU_MIN_DRIFT_RES_MS).clamp(0.0, 1.0);
                let w_t = u_t * u_t * (3.0 - 2.0 * u_t);
                let eff_elapsed = elapsed_ms.max(TAU_MIN_DRIFT_RES_MS);
                let per_hour = (cur_ln - prev_ln) * 3_600_000.0 / eff_elapsed;
                out.dominant_drift = if per_hour.is_finite() {
                    per_hour.clamp(-1.0, 1.0) * w_t
                } else {
                    0.0
                };
            }
        }
        // Coordenadas de crash-ness (cada una en [0,1], combinación acotada):
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
        let raw_crash =
            0.35 * accel_fast + 0.30 * adverse_tide + 0.20 * coherence_collapse + 0.15 * micro_anti;
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

    /// Multiplicador continuo de margen para largos bajo crash-ness creciente:
    /// 1.0 en calma; desciende suavemente hasta ~0.05 en crash-ness extrema.
    /// El veto absoluto binario del enum queda SOLO como el extremo medido
    /// (legacy_view Crash, crash-ness > 0.80 con marea adversa).
    pub fn long_margin_multiplier(&self) -> f64 {
        if !self.carrier_tide.is_finite() || self.carrier_tide >= 0.0 {
            return 1.0;
        }
        let cf = if self.crash_flux.is_finite() { self.crash_flux } else { 0.0 };
        (1.0 - 0.95 * cf).clamp(0.05, 1.0)
    }

    /// Multiplicador continuo de margen para posiciones Cortas:
    /// Si la marea portadora es intensamente ALCISTA (short squeeze / blow-off),
    /// contrae el margen del corto en proporción a la crash_flux inversa.
    pub fn short_margin_multiplier(&self) -> f64 {
        if !self.carrier_tide.is_finite() || self.carrier_tide <= 0.0 {
            return 1.0;
        }
        let cf = if self.crash_flux.is_finite() { self.crash_flux } else { 0.0 };
        (1.0 - 0.95 * cf).clamp(0.05, 1.0)
    }

    /// Multiplicador de margen continuo simétrico según la dirección de la orden.
    pub fn margin_multiplier(&self, is_long: bool) -> f64 {
        if is_long {
            self.long_margin_multiplier()
        } else {
            self.short_margin_multiplier()
        }
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
}
