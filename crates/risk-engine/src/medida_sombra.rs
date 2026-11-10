//! RB-V1a/RB-V1b (Codex RB, 2026-11-10) — la geometría de la evaluación en
//! curso, leída EN SOMBRA.
//!
//! QUÉ: el núcleo juzga en un libro contrafactual (`audit_engine::shadow_ledger`)
//! las intenciones que el riesgo rechaza después de calcular su geometría
//! (RB-V1b) y las que el consejo o el ML-GATE vetan después de validarlas
//! (RB-V1a). Para que el juicio use la MISMA vara que el riesgo, la intención
//! se registra con el TP, el SL, la fricción de ida y vuelta y el horizonte τ
//! con que `evaluate_single_intent` la evaluó.
//!
//! CÓMO: `RiskEngine::geometria_evaluada` se borra al empezar cada evaluación
//! y se fija dos veces dentro de ella: con la geometría del horizonte
//! (`tpsl_gate`), que es la que juzga el suelo TP/SL, y con la geometría final
//! de la orden, que con objetivos explícitos de la intención puede diferir.
//! Una evaluación que termina antes de calcularla la deja en `None`: no hay
//! geometría que juzgar y nunca se inventa una. Sólo se escribe; ninguna
//! decisión la lee.
//!
//! RB-V1a derivaba la geometría de `MedidaOrden` deshaciendo `(sl+f)/sl` con
//! redondeo; RB-V1b la publica en origen, exacta, y sirve también a los
//! rechazos, que no dejan medida.

use crate::RiskEngine;

/// Geometría de una intención evaluada, en fracciones de su precio de entrada.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometriaSombra {
    /// Distancia al take-profit (> 0).
    pub tp_pct: f64,
    /// Distancia al stop (> 0).
    pub sl_pct: f64,
    /// Fricción de ida y vuelta con la que el riesgo la evaluó (≥ 0).
    pub friccion_pct: f64,
    /// Horizonte con el que se dimensiona (D-745), en ms (> 0).
    pub tau_ms: f64,
}

impl GeometriaSombra {
    /// `Some` sólo si las cuatro magnitudes son finitas, TP, SL y τ positivos
    /// y la fricción no negativa: una geometría incoherente no se juzga.
    pub fn nueva(tp_pct: f64, sl_pct: f64, friccion_pct: f64, tau_ms: f64) -> Option<Self> {
        let positiva = |x: f64| x.is_finite() && x > 0.0;
        (positiva(tp_pct)
            && positiva(sl_pct)
            && positiva(tau_ms)
            && friccion_pct.is_finite()
            && friccion_pct >= 0.0)
            .then_some(Self {
                tp_pct,
                sl_pct,
                friccion_pct,
                tau_ms,
            })
    }
}

impl RiskEngine {
    /// La geometría de la última evaluación si fue de `coin_id`: TP y SL de la
    /// intención, la fricción de ida y vuelta y el τ de su horizonte. `None` si
    /// la evaluación fue de otra moneda o terminó antes de calcularla. Leerla
    /// no consume nada.
    pub fn geometria_evaluada(&self, coin_id: usize) -> Option<GeometriaSombra> {
        self.geometria_evaluada
            .filter(|&(moneda, _)| moneda == coin_id)
            .map(|(_, g)| g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rb_v1b_solo_una_geometria_coherente_se_publica() {
        let Some(g) = GeometriaSombra::nueva(0.02, 0.01, 0.00082, 60_000.0) else {
            panic!("una geometría coherente se publica");
        };
        assert_eq!(
            (g.tp_pct, g.sl_pct, g.friccion_pct, g.tau_ms),
            (0.02, 0.01, 0.00082, 60_000.0)
        );
        assert!(
            GeometriaSombra::nueva(0.02, 0.01, 0.0, 1.0).is_some(),
            "fricción nula vale"
        );
        for (tp, sl, f, tau) in [
            (0.0, 0.01, 0.001, 1.0),
            (0.02, -0.01, 0.001, 1.0),
            (0.02, 0.01, -1e-9, 1.0),
            (0.02, 0.01, 0.001, 0.0),
            (f64::NAN, 0.01, 0.001, 1.0),
            (0.02, f64::INFINITY, 0.001, 1.0),
            (0.02, 0.01, f64::NAN, 1.0),
            (0.02, 0.01, 0.001, f64::INFINITY),
        ] {
            assert_eq!(
                GeometriaSombra::nueva(tp, sl, f, tau),
                None,
                "{tp} {sl} {f} {tau}"
            );
        }
    }

    #[test]
    fn rb_v1b_la_geometria_es_de_su_moneda_y_reset_la_borra() {
        let mut motor = RiskEngine::new(100.0);
        assert_eq!(motor.geometria_evaluada(3), None);
        motor.geometria_evaluada = GeometriaSombra::nueva(0.02, 0.01, 0.001, 1.0).map(|g| (3, g));
        assert_eq!(motor.geometria_evaluada(4), None, "otra moneda");
        assert!(motor.geometria_evaluada(3).is_some());
        assert!(
            motor.geometria_evaluada(3).is_some(),
            "leerla no la consume"
        );
        motor.reset(100.0);
        assert_eq!(motor.geometria_evaluada(3), None, "reset la borra");
    }
}
