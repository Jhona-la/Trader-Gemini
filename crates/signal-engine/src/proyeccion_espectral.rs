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

/// Masa resuelta mínima para opinar. Por debajo, la proyección está dominada
/// por escalas sin observar: abstenerse es la única respuesta honesta.
const MASA_MINIMA_PARA_OPINAR: f64 = 0.25;

#[derive(Default)]
pub struct ProyeccionEspectralEngine {
    registry: Option<Arc<OmniscientRegistry>>,
}

impl ProyeccionEspectralEngine {
    pub fn new() -> Self {
        Self::default()
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

    fn evaluate_for_coin(&self, coin_id: usize, symbol: &str) -> f64 {
        let Some(masa) = self.lee(coin_id, symbol, "espectral_masa_resuelta") else {
            return 0.0;
        };
        if !masa.is_finite() || masa < MASA_MINIMA_PARA_OPINAR {
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
        (senal * conc.clamp(0.0, 1.0)).clamp(-1.0, 1.0)
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
    }
}
