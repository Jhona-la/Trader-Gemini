//! RB-V1a (Codex RB, 2026-11-10) — la geometría de la última orden validada,
//! leída EN SOMBRA.
//!
//! QUÉ: el núcleo juzga en un libro contrafactual los vetos posteriores al
//! riesgo (consejo y ML-GATE). Para que el juicio use la MISMA vara que el
//! riesgo, la intención vetada se registra con la geometría y la fricción de
//! ida y vuelta con que `evaluate_quantum_order` la validó. Esa fricción sólo
//! vive dentro de [`MedidaOrden`] (campos privados): este módulo la expone sin
//! cambiar ningún campo ni consumir la medida pendiente, así que leerla no
//! altera ninguna decisión.
//!
//! Derivación exacta desde la medida (D-744, CL-106, CL-164), con `sl` y `tp`
//! las distancias de la orden y `f` la fricción, en fracción del precio de
//! entrada:
//! - `perdida_al_stop = sl + f` y `geometria.1 = (sl + f)/sl`, luego
//!   `sl = perdida_al_stop / geometria.1`;
//! - `f = perdida_al_stop − sl`;
//! - `geometria.0 = (tp − f)/sl`, luego `tp = geometria.0 · sl + f`.

use crate::{MedidaOrden, RiskEngine};

/// Geometría de una orden validada, en fracciones de su precio de entrada.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometriaSombra {
    /// Distancia al take-profit (> 0).
    pub tp_pct: f64,
    /// Distancia al stop (> 0).
    pub sl_pct: f64,
    /// Fricción de ida y vuelta con la que el riesgo evaluó la orden (≥ 0).
    pub friccion_pct: f64,
}

impl MedidaOrden {
    /// La geometría con la que el riesgo validó la orden. `None` si la medida
    /// no la trae finita y coherente: nunca se inventa una geometría.
    pub fn geometria_sombra(&self) -> Option<GeometriaSombra> {
        let (ganancia, perdida) = self.geometria?;
        let perdida_al_stop = self.perdida_al_stop;
        if !ganancia.is_finite() || !perdida.is_finite() || perdida <= 0.0 {
            return None;
        }
        if !perdida_al_stop.is_finite() || perdida_al_stop <= 0.0 {
            return None;
        }
        let sl_pct = perdida_al_stop / perdida;
        if !sl_pct.is_finite() || sl_pct <= 0.0 {
            return None;
        }
        // Redondeo de la resta: el riesgo hizo dos operaciones (suma y
        // cociente) y aquí van otras dos, así que el residuo de una fricción
        // nula cabe en 4ε·(sl + f). Dentro de esa cota es fricción nula; un
        // residuo más negativo es una medida incoherente.
        let residuo = perdida_al_stop - sl_pct;
        if residuo < -4.0 * f64::EPSILON * perdida_al_stop {
            return None;
        }
        let friccion_pct = residuo.max(0.0);
        let tp_pct = ganancia * sl_pct + friccion_pct;
        (tp_pct.is_finite() && tp_pct > 0.0).then_some(GeometriaSombra {
            tp_pct,
            sl_pct,
            friccion_pct,
        })
    }
}

impl RiskEngine {
    /// La medida de la última orden validada de `coin_id`, SIN consumirla:
    /// `tomar_medida_validada` la sigue tomando al abrir la posición.
    pub fn medida_validada(&self, coin_id: usize) -> Option<MedidaOrden> {
        self.medida_pendiente.filter(|m| m.coin_id == coin_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Medida construida con las fórmulas de `evaluate_quantum_order` (D-744).
    fn medida(sl: f64, tp: f64, friccion: f64) -> MedidaOrden {
        MedidaOrden {
            coin_id: 3,
            riesgo: 0.01,
            riesgo_usd: 1.0,
            geometria: Some(((tp - friccion) / sl, (sl + friccion) / sl)),
            f_lh04: 0.0,
            f_legado: 0.0,
            perdida_al_stop: sl + friccion,
        }
    }

    #[test]
    fn rb_v1a_la_geometria_se_recupera_dentro_del_redondeo() {
        for (sl, tp, f) in [
            (0.005, 0.010, 0.0012),
            (0.0021, 0.0063, 0.00083),
            (0.04, 0.02, 0.0),
            (0.0005, 0.0015, 0.0011),
            (0.01, 0.0005, 0.0009), // TP por debajo de la fricción: ganancia < 0
        ] {
            let Some(g) = medida(sl, tp, f).geometria_sombra() else {
                panic!("geometría válida sin recuperar: sl {sl} tp {tp} f {f}");
            };
            // Unas pocas operaciones de redondeo sobre magnitudes ≤ tp + sl + f.
            let cota = 8.0 * f64::EPSILON * (tp + sl + f);
            assert!((g.sl_pct - sl).abs() <= cota, "sl {sl}: {}", g.sl_pct);
            assert!((g.tp_pct - tp).abs() <= cota, "tp {tp}: {}", g.tp_pct);
            assert!(
                (g.friccion_pct - f).abs() <= cota,
                "f {f}: {}",
                g.friccion_pct
            );
        }
    }

    #[test]
    fn rb_v1a_sin_geometria_coherente_no_hay_sombra() {
        let base = medida(0.005, 0.010, 0.001);
        for mala in [
            MedidaOrden {
                geometria: None,
                ..base
            },
            MedidaOrden {
                geometria: Some((1.0, f64::NAN)),
                ..base
            },
            MedidaOrden {
                geometria: Some((f64::INFINITY, 1.2)),
                ..base
            },
            MedidaOrden {
                geometria: Some((1.0, 0.0)),
                ..base
            },
            MedidaOrden {
                geometria: Some((1.0, -1.0)),
                ..base
            },
            MedidaOrden {
                perdida_al_stop: f64::INFINITY,
                ..base
            },
            MedidaOrden {
                perdida_al_stop: 0.0,
                ..base
            },
            // (sl + f)/sl < 1 implicaría una fricción negativa.
            MedidaOrden {
                geometria: Some((1.0, 0.5)),
                ..base
            },
            // TP no positivo: ganancia·sl + f ≤ 0.
            MedidaOrden {
                geometria: Some((-1.0, 1.2)),
                ..base
            },
        ] {
            assert_eq!(mala.geometria_sombra(), None, "{mala:?}");
        }
    }

    #[test]
    fn rb_v1a_leer_la_medida_no_la_consume_y_respeta_la_moneda() {
        let mut motor = RiskEngine::new(100.0);
        motor.medida_pendiente = Some(medida(0.005, 0.010, 0.001));
        assert_eq!(motor.medida_validada(4), None, "otra moneda");
        assert!(motor.medida_validada(3).is_some());
        assert!(
            motor.medida_validada(3).is_some(),
            "leer dos veces no la consume"
        );
        assert!(
            motor.tomar_medida_validada(3).is_some(),
            "la apertura la sigue tomando"
        );
        assert_eq!(motor.medida_validada(3), None);
    }
}
