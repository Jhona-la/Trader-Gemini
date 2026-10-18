//! #591 — PROYECCIÓN ESPECTRAL TEMPORAL.
//!
//! El espectro temporal de 32 escalas vive en el núcleo; hasta esta ola el
//! motor de señales jamás lo leía (cero claves espectrales en el registro):
//! los motores votaban sobre micro-features del tick sin ninguna lectura del
//! estado espectral del mercado. El núcleo publica ahora tres proyecciones
//! sobre la masa canónica (`pesos_espectrales`, que integra la observación
//! D-742 y la resolución CL-32):
//!
//!  · `espectral_senal_proyectada` ∈ [-1,1] — consenso de momentum ponderado
//!    por energía: Σ w_k·s_k / Σ w_k.
//!  · `espectral_concentracion` ∈ [0,1] — 1 − razón de participación:
//!    1 = un modo domina (señal limpia), 0 = masa difusa (ruido blanco).
//!  · `espectral_masa_resuelta` ∈ [0,1] — fracción del peso bruto que
//!    sobrevivió a observación+resolución.
//!
//! El voto del motor es `señal · concentración`, con ABSTENCIÓN (0.0) cuando
//! la masa resuelta no alcanza para opinar: un espectro sin observar no
//! proyecta conocimiento (D-742). La concentración amortigua el voto en
//! espectros difusos — la dirección proyectada de un ruido blanco no es
//! información, es su media.

use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::QuantumStrategy;

/// Masa resuelta mínima canónica por debajo de la cual la proyección se abstiene.
pub const DEFAULT_MASA_MINIMA_PARA_OPINAR: f64 = 0.20;
/// Masa resuelta a partir de la cual la proyección opina con 100% de fuerza.
pub const DEFAULT_MASA_PLENA_PARA_OPINAR: f64 = 0.35;

/// Parámetros de modulación continua para la proyección espectral temporal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProyeccionEspectralParams {
    /// Masa resuelta mínima por debajo de la cual el peso de la proyección es exactamente 0.0.
    pub masa_minima: f64,
    /// Masa resuelta plena a partir de la cual la proyección opina con 100% de fuerza.
    pub masa_plena: f64,
}

impl Default for ProyeccionEspectralParams {
    fn default() -> Self {
        Self {
            masa_minima: DEFAULT_MASA_MINIMA_PARA_OPINAR,
            masa_plena: DEFAULT_MASA_PLENA_PARA_OPINAR,
        }
    }
}

impl ProyeccionEspectralParams {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.masa_minima.is_finite() || !self.masa_plena.is_finite() {
            return Err("ProyeccionEspectralParams: valores no finitos");
        }
        if self.masa_minima < 0.0 || self.masa_minima >= self.masa_plena || self.masa_plena > 1.0 {
            return Err("ProyeccionEspectralParams: cotas inválidas (0.0 <= masa_minima < masa_plena <= 1.0)");
        }
        Ok(())
    }

    /// Factor de peso de modulación continua C^1 en [0, 1] según la masa observada.
    #[inline]
    pub fn compute_weight(&self, masa: f64) -> f64 {
        if !masa.is_finite() || masa <= self.masa_minima {
            return 0.0;
        }
        if masa >= self.masa_plena {
            return 1.0;
        }
        let t = ((masa - self.masa_minima) / (self.masa_plena - self.masa_minima)).clamp(0.0, 1.0);
        // Polinomio cúbico Hermite C^1 (smoothstep): 3t^2 - 2t^3
        t * t * (3.0 - 2.0 * t)
    }
}

#[derive(Clone)]
pub struct ProyeccionEspectralEngine {
    registry: Option<Arc<OmniscientRegistry>>,
    params: ProyeccionEspectralParams,
}

impl Default for ProyeccionEspectralEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ProyeccionEspectralEngine {
    pub fn new() -> Self {
        Self {
            registry: None,
            params: ProyeccionEspectralParams::default(),
        }
    }

    pub fn with_params(params: ProyeccionEspectralParams) -> Self {
        let safe_params = if params.validate().is_ok() {
            params
        } else {
            ProyeccionEspectralParams::default()
        };
        Self {
            registry: None,
            params: safe_params,
        }
    }

    pub fn update_params(&mut self, params: ProyeccionEspectralParams) {
        if params.validate().is_ok() {
            self.params = params;
        }
    }

    pub fn params(&self) -> &ProyeccionEspectralParams {
        &self.params
    }

    fn lee(&self, coin_id: usize, symbol: &str, clave: &str) -> Option<f64> {
        let r = self.registry.as_ref()?;
        let sym_opt = if symbol.is_empty() { None } else { Some(symbol) };
        r.get_scoped_parameter(sym_opt, Some(coin_id), clave, "ProyeccionEspectralEngine")
            .map(|p| p.get_value())
    }
}

impl QuantumStrategy for ProyeccionEspectralEngine {
    fn name(&self) -> &str {
        "ProyeccionEspectralEngine"
    }

    fn init(&mut self, registry: Arc<OmniscientRegistry>) -> Result<(), String> {
        self.registry = Some(registry);
        Ok(())
    }

    /// Sin contexto de moneda no hay lectura honesta: las proyecciones son
    /// per-moneda y una lectura global mezclaría símbolos.
    fn evaluate(&self) -> f64 {
        0.0
    }

    /// F1-123 (Fable / Ola Ω153): Operabilidad explícita per-moneda.
    /// Un espectro sin masa o con masa inferior a masa_minima no proyecta conocimiento
    /// físico observable. Si falta la clave o el peso modulador es 0.0, el motor no es
    /// operable y devuelve false, impidiendo que una abstención estructural entre a n_fuentes
    /// y diluya artificialmente el promedio del ensamble (paridad con Ola Ω145).
    fn is_operable_for_coin(&self, coin_id: usize, symbol: &str) -> bool {
        let Some(masa) = self.lee(coin_id, symbol, "espectral_masa_resuelta") else {
            return false;
        };
        let custom_min = self.lee(coin_id, symbol, "espectral_masa_minima")
            .unwrap_or(self.params.masa_minima);
        let custom_plena = self.lee(coin_id, symbol, "espectral_masa_plena")
            .unwrap_or(self.params.masa_plena);
        let effective_params = if custom_min.is_finite() && custom_plena.is_finite() && custom_min < custom_plena {
            ProyeccionEspectralParams {
                masa_minima: custom_min,
                masa_plena: custom_plena,
            }
        } else {
            self.params
        };
        effective_params.compute_weight(masa) > 0.0
    }

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let Some(masa) = self.lee(coin_id, symbol, "espectral_masa_resuelta") else {
            return 0.0;
        };

        // Resolución dinámica de parámetros desde el registro omnisciente si existen
        let custom_min = self.lee(coin_id, symbol, "espectral_masa_minima")
            .unwrap_or(self.params.masa_minima);
        let custom_plena = self.lee(coin_id, symbol, "espectral_masa_plena")
            .unwrap_or(self.params.masa_plena);
        let effective_params = if custom_min.is_finite() && custom_plena.is_finite() && custom_min < custom_plena {
            ProyeccionEspectralParams {
                masa_minima: custom_min,
                masa_plena: custom_plena,
            }
        } else {
            self.params
        };

        let weight = effective_params.compute_weight(masa);
        if weight <= 0.0 {
            return 0.0;
        }

        let (Some(senal), Some(conc)) = (
            self.lee(coin_id, symbol, "espectral_senal_proyectada"),
            self.lee(coin_id, symbol, "espectral_concentracion"),
        ) else {
            return 0.0;
        };
        if !senal.is_finite() || !conc.is_finite() {
            return 0.0;
        }
        (senal * conc.clamp(0.0, 1.0) * weight).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn motor(registry: Arc<OmniscientRegistry>) -> ProyeccionEspectralEngine {
        let mut e = ProyeccionEspectralEngine::new();
        assert!(e.init(registry).is_ok());
        e
    }

    #[test]
    fn proyecta_con_concentracion_y_abstiene_en_masa_baja() {
        let r = Arc::new(OmniscientRegistry::new());
        r.set("espectral_senal_proyectada", 0.6);
        r.set("espectral_concentracion", 0.8);
        r.set("espectral_masa_resuelta", 0.9);
        let e = motor(r);
        let v = e.evaluate_for_coin(0, "BTCUSDT");
        assert!((v - 0.48).abs() < 1e-9, "0.6·0.8 = 0.48, obtuve {v}");
    }

    #[test]
    fn espectro_sin_resolver_no_opina() {
        let r = Arc::new(OmniscientRegistry::new());
        r.set("espectral_senal_proyectada", 0.9);
        r.set("espectral_concentracion", 0.9);
        r.set("espectral_masa_resuelta", 0.10);
        let e = motor(r);
        assert_eq!(
            e.evaluate_for_coin(0, "BTCUSDT"),
            0.0,
            "masa 0.10 < 0.25: la proyección de escalas sin observar no es conocimiento"
        );
    }

    #[test]
    fn sin_telemetria_espectral_abstiene() {
        let e = motor(Arc::new(OmniscientRegistry::new()));
        assert_eq!(e.evaluate_for_coin(0, "BTCUSDT"), 0.0);
        assert!(!e.is_operable_for_coin(0, "BTCUSDT"), "sin datos no debe ser operable");
    }

    #[test]
    fn operabilidad_espectral_segun_masa_resuelta() {
        let r = Arc::new(OmniscientRegistry::new());
        let e = motor(r.clone());
        assert!(!e.is_operable_for_coin(0, "BTCUSDT"), "frío sin masa => inoperable");

        r.set("espectral_masa_resuelta", 0.15);
        assert!(!e.is_operable_for_coin(0, "BTCUSDT"), "masa 0.15 <= 0.25 => inoperable");

        r.set("espectral_masa_resuelta", 0.50);
        assert!(e.is_operable_for_coin(0, "BTCUSDT"), "masa 0.50 > 0.25 => operable");
    }

    #[test]
    fn senal_vendedora_amortiguada_por_difusion() {
        let r = Arc::new(OmniscientRegistry::new());
        r.set("espectral_senal_proyectada", -0.5);
        r.set("espectral_concentracion", 0.4);
        r.set("espectral_masa_resuelta", 1.0);
        let e = motor(r);
        let v = e.evaluate_for_coin(0, "BTCUSDT");
        assert!((v + 0.20).abs() < 1e-9, "-0.5·0.4 = -0.20, obtuve {v}");
        assert!(e.is_operable_for_coin(0, "BTCUSDT"));
    }
}
