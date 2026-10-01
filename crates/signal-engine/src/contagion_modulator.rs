//! MODULADOR DE SEÑAL POR CONTAGIO (Ola XLV·F).
//!
//! Cuando un activo SEGUIDOR recibe contagio significativo de un LÍDER,
//! parte de su movimiento ya está explicado por el líder — el edge propio
//! del seguidor es menor de lo que su señal aislada sugiere. Este módulo
//! descuenta la convicción de señales en activos contagiados.
//!
//! Contrato (protocolo del repo):
//! - Variable: el net_role del ContagionRole del activo (negativo = seguidor
//!   neto, recibiendo más contagio del que emite).
//! - Operador: factor = 1 − descuento·|net_role|/(|net_role| + k) — una
//!   sigmoide acotada que nunca baja de (1 − descuento_máximo).
//! - Unidades: adimensional [1−d_max, 1].
//! - Contorno: sin rol (None) o net_role ≥ 0 (líder/neutro) → factor = 1.
//! - Identificabilidad: el descuento es proporcional a la EVIDENCIA de
//!   seguimiento (z-scores medidos), no a una etiqueta discreta.
//! - Coste: O(1) por señal.
//! - Falsación: líder neto → factor 1 (sin descuento); seguidor con rol
//!   fuerte → factor cerca del mínimo; None → 1 (tests).

use feature_engine::hawkes_cross::ContagionRole;

/// Descuenta la convicción de una señal dado un valor de net_role continuo.
/// Protegido contra valores no finitos o roles líderes (net_role >= 0).
#[inline]
pub fn modulate_by_net_role(raw_confidence: f64, net_role: f64) -> f64 {
    if !net_role.is_finite() || net_role >= 0.0 {
        return raw_confidence.clamp(0.0, 1.0);
    }
    let magnitude = (-net_role).min(50.0); // cap para evitar overflow
    let discount = 0.30 * magnitude / (magnitude + 5.0);
    (raw_confidence * (1.0 - discount)).clamp(0.0, 1.0)
}

/// Descuenta la convicción de una señal en un activo SEGUIDOR.
/// `raw_confidence` ∈ [0,1]; devuelve la convicción ajustada ∈ [0,1].
///
/// La forma: factor = 1 − 0.30 · |net|/(|net| + 5.0) — con |net|=5 (un
/// seguidor con z-scores medios de −5), el descuento es 0.15 (15%); con
/// |net|=20, 0.24; el máximo descuento es 30%.
pub fn modulate_by_contagion(
    raw_confidence: f64,
    role: Option<ContagionRole>,
) -> f64 {
    let Some(r) = role else {
        return raw_confidence;
    };
    modulate_by_net_role(raw_confidence, r.net_role)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Líder neto → sin descuento.
    #[test]
    fn xlvf_lider_no_descuenta() {
        let role = ContagionRole { emitted: 10.0, received: 2.0, net_role: 8.0 };
        assert_eq!(modulate_by_contagion(0.8, Some(role)), 0.8);
    }

    /// None → sin descuento.
    #[test]
    fn xlvf_sin_rol_no_descuenta() {
        assert_eq!(modulate_by_contagion(0.8, None), 0.8);
    }

    /// Seguidor fuerte (net=−20) → descuento ~24%.
    #[test]
    fn xlvf_seguidor_fuerte_descuenta() {
        let role = ContagionRole { emitted: 2.0, received: 22.0, net_role: -20.0 };
        let adjusted = modulate_by_contagion(0.8, Some(role));
        // factor = 1 - 0.30*20/(20+5) = 1 - 0.24 = 0.76
        assert!((adjusted - 0.8 * 0.76).abs() < 1e-6,
            "esperaba ~{:.4}, dio {:.4}", 0.8 * 0.76, adjusted);
        assert!(adjusted < 0.8, "debe ser < original");
    }

    /// Seguidor débil (net=−2) → descuento suave ~8.6%.
    #[test]
    fn xlvf_seguidor_debil_descuenta_suave() {
        let role = ContagionRole { emitted: 4.0, received: 6.0, net_role: -2.0 };
        let adjusted = modulate_by_contagion(0.9, Some(role));
        // factor = 1 - 0.30*2/(2+5) = 1 - 0.0857 = 0.914
        assert!(adjusted < 0.9 && adjusted > 0.8,
            "descuento suave, dio {:.4}", adjusted);
    }

    /// Modulación directa por net_role con defensas de IEEE-754.
    #[test]
    fn test_modulate_by_net_role_finite_immunity() {
        assert_eq!(modulate_by_net_role(0.85, f64::NAN), 0.85);
        assert_eq!(modulate_by_net_role(0.85, f64::INFINITY), 0.85);
        assert_eq!(modulate_by_net_role(0.85, 2.5), 0.85);
        let adjusted = modulate_by_net_role(0.8, -20.0);
        assert!((adjusted - 0.8 * 0.76).abs() < 1e-6);
    }
}
