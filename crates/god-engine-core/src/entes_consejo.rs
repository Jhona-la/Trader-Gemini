//! CL-49 — ENTES DEL CONSEJO QUE MIDE EL HOST (ballena y spoofing).
//!
//! El host los mide sobre el flujo real (tamaño de cada trade, muros de la
//! profundidad L2) y el núcleo los lee al deliberar el consejo. Antes el host
//! sólo escribía en evento —un burst de ballena, un score de spoofing por
//! encima de 0,05— y nada los bajaba: el último valor alto quedaba pegado y
//! el consejo lo leía en cada deliberación durante horas (una ballena de z = 6
//! limitaba el asiento EnteMercado a 0,60 hasta el siguiente burst). Ahora
//! cada medición publica su valor ACTUAL, también el nulo.
//!
//! Escritura y lectura viven aquí para que no se desacoplen: se publica en
//! los dos espacios de nombres que lee el núcleo desde R7-R4-A-2 (por moneda y
//! por símbolo) y se lee su máximo, como entonces.

use omniscient_registry::OmniscientRegistry;
use std::borrow::Cow;

pub const BALLENA: &str = "whale_burst_z";
pub const SPOOF: &str = "spoof_score";

/// Suelo de ruido del detector de spoofing (el umbral que antes decidía si
/// se publicaba): por debajo, el ente es nulo.
const SUELO_SPOOF: f64 = 0.05;

/// Ente ballena del trade actual: su z si es un burst, 0 si no. El estado
/// que ve el consejo es «el último trade de la moneda fue (o no) de ballena».
#[inline]
pub fn valor_ballena(z: f64, is_burst: bool) -> f64 {
    if is_burst && z.is_finite() {
        z.clamp(0.0, 10.0)
    } else {
        0.0
    }
}

/// Ente spoofing de la evaluación actual: el score del detector, que ya
/// decae con el tiempo (λ del detector); bajo el suelo de ruido, 0.
#[inline]
pub fn valor_spoof(score: f64) -> f64 {
    if score.is_finite() && score > SUELO_SPOOF {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Publica el valor actual del ente en los dos espacios que lee el núcleo.
/// El símbolo se normaliza a mayúsculas (CL-37) sin asignar si ya lo está.
pub fn publicar(registry: &OmniscientRegistry, coin_id: usize, symbol: &str, clave: &str, valor: f64) {
    registry.set_for_coin(coin_id, clave, valor);
    if !symbol.is_empty() {
        let sym: Cow<str> = if symbol.bytes().any(|b| b.is_ascii_lowercase()) {
            Cow::Owned(symbol.to_ascii_uppercase())
        } else {
            Cow::Borrowed(symbol)
        };
        registry.set_scoped(&sym, clave, valor);
    }
}

/// Lectura del núcleo al deliberar: máximo de los dos espacios (R7-R4-A-2).
#[inline]
pub fn leer(registry: &OmniscientRegistry, coin_id: usize, symbol: &str, clave: &str) -> f64 {
    registry
        .get_for_coin_or(coin_id, clave, 0.0)
        .max(registry.get_scoped_value_or(symbol, clave, 0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Una ballena de z = 6 seguida de un trade normal: antes el registro
    /// seguía en 6 (sólo se escribía en burst); ahora el consejo lee 0.
    #[test]
    fn cl49_la_ballena_no_queda_pegada_tras_el_burst() {
        let r = OmniscientRegistry::new();
        publicar(&r, 0, "btcusdt", BALLENA, valor_ballena(6.0, true));
        assert_eq!(leer(&r, 0, "BTCUSDT", BALLENA), 6.0);
        publicar(&r, 0, "btcusdt", BALLENA, valor_ballena(0.3, false));
        assert_eq!(leer(&r, 0, "BTCUSDT", BALLENA), 0.0);
    }

    /// Un spike de spoofing (0,9) decae a 0,016 dos segundos después: antes
    /// no se publicaba (≤ 0,05) y el registro seguía en 0,9.
    #[test]
    fn cl49_el_spoofing_decae_en_el_registro() {
        let r = OmniscientRegistry::new();
        publicar(&r, 1, "ETHUSDT", SPOOF, valor_spoof(0.9));
        assert_eq!(leer(&r, 1, "ETHUSDT", SPOOF), 0.9);
        publicar(&r, 1, "ETHUSDT", SPOOF, valor_spoof(0.9 * (-4.0f64).exp()));
        assert_eq!(leer(&r, 1, "ETHUSDT", SPOOF), 0.0);
        publicar(&r, 1, "ETHUSDT", SPOOF, valor_spoof(0.3));
        assert_eq!(leer(&r, 1, "ETHUSDT", SPOOF), 0.3);
    }

    /// Los valores no finitos o fuera de rango no llegan al consejo.
    #[test]
    fn cl49_dominio_de_los_entes() {
        assert_eq!(valor_ballena(f64::NAN, true), 0.0);
        assert_eq!(valor_ballena(25.0, true), 10.0);
        assert_eq!(valor_spoof(f64::INFINITY), 0.0);
        assert_eq!(valor_spoof(1.7), 1.0);
    }
}
