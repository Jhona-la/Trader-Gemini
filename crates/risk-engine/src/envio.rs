//! CL-41 — el apalancamiento con el que el host envía la entrada.
//!
//! El núcleo abre la ranura con la orden que validó el risk-engine: margen
//! `volume_usd`, nocional `margen · leverage`, y suma ese margen a
//! `arena.used_margin` ANTES de que el host decida. El host (y su réplica en
//! el replay) recalculaba el apalancamiento de envío con la envolvente y lo
//! adaptaba al margen libre (D-382), con dos defectos:
//!
//! 1. El margen libre era `capital − used_margin`, que ya descuenta la
//!    reserva de ESTA orden: la orden se comparaba contra un margen que ella
//!    misma había consumido. En capital micro eso vetaba entradas que caben y
//!    subía el apalancamiento sin necesidad.
//! 2. La adaptación subía el apalancamiento hasta 20× sin mirar el que el
//!    risk-engine validó, y la envolvente podía pedir más que el validado:
//!    el host decidía por encima del riesgo (la liquidación quedaba más
//!    cerca de lo que el riesgo aceptó).
//!
//! Aquí el apalancamiento de envío nunca supera el validado; la adaptación
//! sólo puede subirlo hasta él. El arranque exploratorio (1× con poca
//! evidencia) y el veto de la envolvente se conservan tal cual.

/// Techo de apalancamiento de los clamps del host (Binance permite más en
/// algunos símbolos; el sistema nunca envía por encima de este).
pub const TOPE_ENVIO: u32 = 20;
/// D-382: la adaptación se dispara si el margen requerido supera este
/// múltiplo del libre…
const UMBRAL_ADAPTAR: f64 = 0.85;
/// …y apunta a dejar el requerido en este múltiplo.
const OBJETIVO_ADAPTAR: f64 = 0.80;
/// D-382: el requerido final nunca supera este múltiplo del libre (-2019).
const GUARDA_MARGEN: f64 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VetoEnvio {
    /// La envolvente dijo que no (`exec == 0`).
    Envolvente,
    /// La ranura de la orden no trae margen ni nocional válidos: no se sabe
    /// con qué apalancamiento la validó el riesgo.
    SinValidacion,
    /// Ni con el apalancamiento validado cabe en el margen libre.
    Margen,
}

/// Apalancamiento con el que el riesgo validó la orden, reconstruido de la
/// ranura que el núcleo reservó: nocional / margen.
pub fn apalancamiento_validado(nocional_reservado: f64, margen_reservado: f64) -> Option<u32> {
    if !(nocional_reservado.is_finite() && margen_reservado.is_finite()) {
        return None;
    }
    if nocional_reservado <= 0.0 || margen_reservado <= 0.0 {
        return None;
    }
    Some((nocional_reservado / margen_reservado).round().clamp(1.0, 50.0) as u32)
}

/// Margen libre ANTES de esta orden: `margen_usado` ya incluye la reserva
/// local de la propia orden, que no compite consigo misma.
pub fn margen_libre_sin_la_propia(capital: f64, margen_usado: f64, margen_propio: f64) -> f64 {
    let usado = if margen_usado.is_finite() { margen_usado.max(0.0) } else { f64::INFINITY };
    let propio = if margen_propio.is_finite() { margen_propio.max(0.0) } else { 0.0 };
    let libre = capital - (usado - propio).max(0.0);
    if libre.is_finite() { libre.max(0.0) } else { 0.0 }
}

/// D-382 subordinado al riesgo: parte del apalancamiento de la envolvente
/// (`exec`, 0 = veto) recortado al validado; si el margen requerido supera
/// el 85 % del libre lo sube como mucho hasta el validado; si aun así pasa
/// del 95 %, veta.
pub fn apalancamiento_de_envio(
    exec: u32,
    validado: Option<u32>,
    nocional: f64,
    margen_libre: f64,
) -> Result<u32, VetoEnvio> {
    if exec == 0 {
        return Err(VetoEnvio::Envolvente);
    }
    let Some(validado) = validado else {
        return Err(VetoEnvio::SinValidacion);
    };
    if !(nocional.is_finite() && nocional > 0.0) {
        return Err(VetoEnvio::SinValidacion);
    }
    let libre = if margen_libre.is_finite() { margen_libre.max(0.0) } else { 0.0 };
    let techo = validado.clamp(1, TOPE_ENVIO);
    let mut apalancamiento = exec.clamp(1, techo);
    if nocional / apalancamiento as f64 > UMBRAL_ADAPTAR * libre && libre > 0.0 {
        let necesario = (nocional / (OBJETIVO_ADAPTAR * libre)).ceil();
        let necesario = if necesario.is_finite() {
            necesario.clamp(1.0, TOPE_ENVIO as f64) as u32
        } else {
            TOPE_ENVIO
        };
        apalancamiento = apalancamiento.max(necesario.min(techo));
    }
    if nocional / apalancamiento as f64 > GUARDA_MARGEN * libre {
        return Err(VetoEnvio::Margen);
    }
    Ok(apalancamiento)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_validado_sale_de_la_ranura_reservada() {
        assert_eq!(apalancamiento_validado(130.0, 26.0), Some(5));
        // El precio de llenado mueve un poco el nocional: se redondea.
        assert_eq!(apalancamiento_validado(130.4, 26.0), Some(5));
        assert_eq!(apalancamiento_validado(130.0, 0.0), None);
        assert_eq!(apalancamiento_validado(f64::NAN, 26.0), None);
        assert_eq!(apalancamiento_validado(-1.0, 26.0), None);
    }

    #[test]
    fn la_reserva_propia_no_compite_consigo_misma() {
        // Capital 13, sólo esta orden reservada (7): el exchange tiene 13 libres.
        assert_eq!(margen_libre_sin_la_propia(13.0, 7.0, 7.0), 13.0);
        // Otra posición con 4 de margen: quedan 9.
        assert_eq!(margen_libre_sin_la_propia(13.0, 11.0, 7.0), 9.0);
        // Contabilidad negativa saturada y no finita cerrada.
        assert_eq!(margen_libre_sin_la_propia(13.0, -1.0, 0.0), 13.0);
        assert_eq!(margen_libre_sin_la_propia(13.0, f64::NAN, 1.0), 0.0);
    }

    #[test]
    fn nunca_envia_mas_que_el_validado() {
        // La envolvente pide 20×, el riesgo validó 5×: se envían 5×.
        assert_eq!(apalancamiento_de_envio(20, Some(5), 13.0, 13.0), Ok(5));
        // Arranque 1× con un nocional que no cabe: sube, pero sólo hasta 5×.
        assert_eq!(apalancamiento_de_envio(1, Some(5), 13.0, 13.0), Ok(2));
        assert_eq!(apalancamiento_de_envio(1, Some(2), 40.0, 13.0), Err(VetoEnvio::Margen));
    }

    #[test]
    fn veta_lo_que_no_cabe_ni_al_validado() {
        // Nocional 600 a 50× validado: ni a 20× (tope de envío) cabe en 13.
        assert_eq!(apalancamiento_de_envio(1, Some(50), 600.0, 12.99), Err(VetoEnvio::Margen));
        assert_eq!(apalancamiento_de_envio(1, Some(5), 13.0, 0.0), Err(VetoEnvio::Margen));
    }

    #[test]
    fn respeta_el_veto_de_la_envolvente_y_la_falta_de_validacion() {
        assert_eq!(apalancamiento_de_envio(0, Some(5), 13.0, 13.0), Err(VetoEnvio::Envolvente));
        assert_eq!(apalancamiento_de_envio(3, None, 13.0, 13.0), Err(VetoEnvio::SinValidacion));
        assert_eq!(apalancamiento_de_envio(3, Some(5), f64::NAN, 13.0), Err(VetoEnvio::SinValidacion));
    }

    #[test]
    fn con_margen_holgado_respeta_la_envolvente() {
        // Nada que adaptar: se envía lo que la envolvente decidió.
        assert_eq!(apalancamiento_de_envio(3, Some(10), 13.0, 1000.0), Ok(3));
        assert_eq!(apalancamiento_de_envio(1, Some(10), 13.0, 1000.0), Ok(1));
    }
}
