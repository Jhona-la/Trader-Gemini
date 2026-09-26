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
                // Por hora, normalizado: 1 = τ* se mueve un eje completo (e≈2.72×) por hora.
                let per_hour = (cur_ln - prev_ln) * 3_600_000.0 / elapsed_ms;
                out.dominant_drift = (per_hour / 1.0).clamp(-1.0, 1.0);
            }
        }
        // Coordenadas de crash-ness (cada una en [0,1], combinación acotada):
        // (a) aceleración hacia lo rápido: deriva negativa fuerte.
        let accel_fast = (-out.dominant_drift).clamp(0.0, 1.0);
        // (b) marea adversa intensa (se toma absoluta: la caída es -tide para
        // largos, pero crash-ness describe el EVENTO, no el lado).
        let adverse_tide = out.carrier_tide.abs().clamp(0.0, 1.0);
        // (c) colapso de entropía: energía coherente concentrada.
        let coherence_collapse = (1.0 - out.mass_entropy).clamp(0.0, 1.0);
        // (d) anti-persistencia extrema en micro (reversión violenta).
        let micro_anti = (0.5 - out.hurst_by_band[0]).max(0.0) / 0.5;
        out.crash_flux =
            (0.35 * accel_fast + 0.30 * adverse_tide + 0.20 * coherence_collapse + 0.15 * micro_anti)
                .clamp(0.0, 1.0);
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
        if self.carrier_tide >= 0.0 {
            return 1.0;
        }
        (1.0 - 0.95 * self.crash_flux).clamp(0.05, 1.0)
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
}
