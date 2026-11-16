//! WB-4a (especificación GLM-APP v3, §4.2 a/d) — LIBRO DE DECISIONES Y
//! CALIBRACIÓN CONDICIONAL CON CONTRAFACTUALES, EN SOMBRA.
//!
//! QUÉ: cada intención con score que el riesgo rechaza después de calcular su
//! geometría, o que el consejo o el ML-GATE vetan, queda como fila
//! `{decision_id, moneda, lado, τ, score, p_calibrada, vetada_por, ts, entrada,
//! tp_abs, sl_abs}`. Con los mids que llegan después se etiqueta su desenlace
//! CONTRAFACTUAL con la MISMA regla de triple barrera del entrenamiento
//! (`src/bin/train_forest.rs`, modo dir): primer toque del SL ⇒ pierde, primer
//! toque del TP ⇒ gana (el SL se mira primero en cada evento), y sin toque al
//! vencer τ la fila es NO_RESUELTA: se cuenta y se excluye, nunca se imputa
//! como pérdida (el neutro descartado del entrenamiento). Las ejecutadas entran
//! con su `is_win` neto real, igual que hoy (`alimentar_ejecutada`, al lado de
//! `estado_aprendido::alimentar`).
//!
//! EN SOMBRA: las observaciones (ejecutadas + contrafactuales, peso 1 las dos)
//! alimentan un calibrador Platt PROPIO por moneda y global, con la misma regla
//! de selección que el vivo (por moneda con ≥ 5 observaciones). Los
//! calibradores vivos (`calibrator_by_coin`, `confidence_calibrator`) NO
//! reciben ningún contrafactual y ninguna decisión lee este libro: la orden
//! sigue usando la calibración de hoy. El diagnóstico (`diagnostico`) compara,
//! con corte temporal, el Brier fuera de muestra del Platt de solo ejecutadas
//! (el de hoy) y del Platt con contrafactuales, y la brecha de familia frente a
//! la isotónica.
//!
//! CONTRATO (recomendado 9 de QODER-CLI): retención acotada a `WINDOW` (512, el
//! del calibrador) en filas abiertas y en observaciones; tp/sl ABSOLUTOS al
//! registrar; canal lateral de sólo escritura (nada entra al tensor 54D ni a
//! las features); una fila abierta por moneda y lado (las reevaluaciones de la
//! misma intención en cada evento no se cuentan dos veces, como en
//! `sombra_vetos`); `observar` sale en O(1) si la moneda no tiene filas
//! abiertas y si las tiene recorre sólo las suyas (cada una necesita su
//! comprobación de barrera en el evento).
//!
//! Límites declarados (los de la especificación): la vetada habría llenado al
//! mid de la decisión, sin impacto propio; la etiqueta contrafactual es la de
//! barrera (geométrica, sin fees) y la ejecutada es neta de comisiones (OBL-5,
//! §5.5.5: heterogeneidad declarada). Las intenciones que el host revierte
//! después (envolvente, margen) no son contrafactuales aquí. El libro no se
//! persiste: tras un reinicio empieza vacío, y el diagnóstico reajusta los dos
//! Platt desde sus propias observaciones, así que la comparación sigue pareja.

use crate::calibration::{
    brier_platt_fuera_de_muestra, corte_temporal, BrierFueraDeMuestra, PlattCalibrator, WINDOW,
};
use risk_engine::medida_sombra::GeometriaSombra;
use std::collections::VecDeque;

/// Observaciones que necesita el calibrador de una moneda para usarse en vez
/// del global: la MISMA regla del núcleo vivo (`observations() >= 5`).
pub const MIN_OBS_CALIBRADOR_MONEDA: usize = 5;

/// De dónde viene una observación del calibrador en sombra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origen {
    Ejecutada,
    Contrafactual,
}

/// Una intención vetada, lista para el libro.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Decision {
    pub moneda: usize,
    pub es_largo: bool,
    pub tau_ms: f64,
    /// Score crudo (lo que aprende el calibrador).
    pub score: f64,
    /// p calibrada que el vivo le dio (la de hoy).
    pub p_calibrada: f64,
    /// Fuente del veto (las de `sombra_vetos::FUENTES`).
    pub vetada_por: u8,
    pub ts_ms: u64,
    pub entrada: f64,
    pub tp_abs: f64,
    pub sl_abs: f64,
}

impl Decision {
    /// La fila con las barreras ABSOLUTAS de la geometría con que el riesgo
    /// evaluó la intención: largo TP = e·(1 + tp), SL = e·(1 − sl); corto al
    /// revés. None sin geometría (el libro la cuenta como inválida).
    #[allow(clippy::too_many_arguments)]
    pub fn desde_geometria(
        moneda: usize,
        es_largo: bool,
        entrada: f64,
        g: Option<GeometriaSombra>,
        score: f64,
        p_calibrada: f64,
        vetada_por: u8,
        ts_ms: u64,
    ) -> Option<Self> {
        let g = g?;
        let (tp_abs, sl_abs) = if es_largo {
            (entrada * (1.0 + g.tp_pct), entrada * (1.0 - g.sl_pct))
        } else {
            (entrada * (1.0 - g.tp_pct), entrada * (1.0 + g.sl_pct))
        };
        Some(Self {
            moneda,
            es_largo,
            tau_ms: g.tau_ms,
            score,
            p_calibrada,
            vetada_por,
            ts_ms,
            entrada,
            tp_abs,
            sl_abs,
        })
    }

    fn valida(&self) -> bool {
        let finitos = [self.tau_ms, self.score, self.entrada, self.tp_abs, self.sl_abs]
            .iter()
            .all(|x| x.is_finite());
        // El gate del vivo: scores no positivos nunca alimentaron.
        let score_ok = self.score > 0.0 && self.score <= 1.0;
        let barreras = if self.es_largo {
            self.sl_abs < self.entrada && self.entrada < self.tp_abs
        } else {
            self.tp_abs < self.entrada && self.entrada < self.sl_abs
        };
        finitos && score_ok && barreras && self.tau_ms > 0.0 && self.sl_abs > 0.0 && self.tp_abs > 0.0
    }
}

/// Triple barrera del entrenamiento sobre un mid: `Some(false)` si toca el SL
/// (se mira primero), `Some(true)` si toca el TP, `None` si ninguno.
#[inline]
pub fn tocar_barrera(es_largo: bool, tp_abs: f64, sl_abs: f64, mid: f64) -> Option<bool> {
    if es_largo {
        if mid <= sl_abs {
            Some(false)
        } else if mid >= tp_abs {
            Some(true)
        } else {
            None
        }
    } else if mid >= sl_abs {
        Some(false)
    } else if mid <= tp_abs {
        Some(true)
    } else {
        None
    }
}

#[derive(Debug, Clone, Copy)]
struct Abierta {
    decision_id: u64,
    d: Decision,
    vence_ms: u64,
}

/// Una observación del calibrador en sombra.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observacion {
    pub score: f64,
    pub won: bool,
    pub ts_ms: u64,
    pub origen: Origen,
}

/// Diagnóstico de la calibración en sombra (telemetría `WB4_SOMBRA`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostico {
    pub observaciones: usize,
    pub ejecutadas: usize,
    pub contrafactuales: usize,
    /// Platt de SOLO ejecutadas (el de hoy): ajusta con las ejecutadas de la
    /// mitad temprana, evalúa sobre TODA la mitad tardía.
    pub solo_ejecutadas: Option<BrierFueraDeMuestra>,
    /// Platt con contrafactuales: ajusta con toda la mitad temprana y evalúa
    /// sobre la misma mitad tardía.
    pub con_contrafactuales: Option<BrierFueraDeMuestra>,
}

/// El libro de decisiones con su calibración en sombra.
pub struct LibroDecisiones {
    abiertas: Vec<Abierta>,
    abiertas_por_moneda: Vec<usize>,
    observaciones: VecDeque<Observacion>,
    sombra_por_moneda: Vec<PlattCalibrator>,
    sombra_global: PlattCalibrator,
    siguiente_id: u64,
    /// Filas que entraron al libro.
    pub registradas: u64,
    /// Ejecutadas que alimentaron la sombra.
    pub ejecutadas: u64,
    /// Contrafactuales resueltas por barrera.
    pub contrafactuales: u64,
    /// Contrafactuales que vencieron τ sin tocar barrera (excluidas).
    pub no_resueltas: u64,
    /// Rechazadas por datos inválidos (sin geometría, score fuera del gate,
    /// barreras incoherentes, moneda fuera del arena).
    pub invalidas: u64,
    /// Ya había una abierta de esa moneda y lado.
    pub solapadas: u64,
    /// El libro estaba lleno (`WINDOW` abiertas).
    pub llenas: u64,
    /// Último total avisado por `avisar`.
    avisado: u64,
}

/// La fuente de un veto posterior al riesgo (consejo o ML-GATE), la MISMA
/// atribución que el libro de vetos (`sombra_vetos::fuente_del_veto`).
#[inline]
pub const fn fuente_vetada(aprobado_por_consejo: bool, tiene_modelo: bool) -> u8 {
    crate::sombra_vetos::fuente_del_veto(aprobado_por_consejo, tiene_modelo)
}

impl LibroDecisiones {
    pub fn new(monedas: usize) -> Self {
        Self {
            abiertas: Vec::with_capacity(WINDOW),
            abiertas_por_moneda: vec![0; monedas],
            observaciones: VecDeque::with_capacity(WINDOW),
            sombra_por_moneda: (0..monedas).map(|_| PlattCalibrator::new()).collect(),
            sombra_global: PlattCalibrator::new(),
            siguiente_id: 0,
            registradas: 0,
            ejecutadas: 0,
            contrafactuales: 0,
            no_resueltas: 0,
            invalidas: 0,
            solapadas: 0,
            llenas: 0,
            avisado: 0,
        }
    }

    /// Registra una intención vetada. Devuelve su `decision_id` si quedó
    /// abierta.
    pub fn registrar_vetada(&mut self, d: Option<Decision>) -> Option<u64> {
        let Some(d) = d else {
            self.invalidas += 1;
            return None;
        };
        let Some(abiertas) = self.abiertas_por_moneda.get(d.moneda).copied() else {
            self.invalidas += 1;
            return None;
        };
        if !d.valida() {
            self.invalidas += 1;
            return None;
        }
        if abiertas > 0
            && self
                .abiertas
                .iter()
                .any(|a| a.d.moneda == d.moneda && a.d.es_largo == d.es_largo)
        {
            self.solapadas += 1;
            return None;
        }
        if self.abiertas.len() >= WINDOW {
            self.llenas += 1;
            return None;
        }
        let id = self.siguiente_id;
        self.siguiente_id += 1;
        // `as` satura; τ ya es finito y positivo.
        let vence_ms = d.ts_ms.saturating_add(d.tau_ms.ceil() as u64);
        self.abiertas.push(Abierta {
            decision_id: id,
            d,
            vence_ms,
        });
        self.abiertas_por_moneda[d.moneda] += 1;
        self.registradas += 1;
        Some(id)
    }

    /// Avanza las filas abiertas de `moneda` con el mid del evento. O(1) si no
    /// tiene ninguna. Devuelve cuántas se cerraron (resueltas o vencidas).
    pub fn observar(&mut self, moneda: usize, mid: f64, t_ms: u64) -> usize {
        match self.abiertas_por_moneda.get(moneda) {
            Some(&n) if n > 0 => {}
            _ => return 0,
        }
        if !(mid.is_finite() && mid > 0.0) {
            return 0;
        }
        let mut cerradas = 0usize;
        let mut k = 0;
        while k < self.abiertas.len() {
            let a = self.abiertas[k];
            if a.d.moneda != moneda || t_ms <= a.d.ts_ms {
                k += 1;
                continue;
            }
            // Como el entrenamiento: pasado el plazo, sin mirar la barrera.
            let desenlace = if t_ms > a.vence_ms {
                self.no_resueltas += 1;
                Some(None)
            } else {
                tocar_barrera(a.d.es_largo, a.d.tp_abs, a.d.sl_abs, mid).map(Some)
            };
            match desenlace {
                None => k += 1,
                Some(etiqueta) => {
                    self.abiertas.swap_remove(k);
                    self.abiertas_por_moneda[moneda] -= 1;
                    cerradas += 1;
                    if let Some(won) = etiqueta {
                        self.contrafactuales += 1;
                        self.aprender(moneda, a.d.score, won, t_ms, Origen::Contrafactual);
                    }
                }
            }
        }
        cerradas
    }

    /// Una ejecutada con su `is_win` real (el mismo dato que recibe el
    /// calibrador vivo, con el mismo gate `score > 0`).
    pub fn alimentar_ejecutada(&mut self, moneda: usize, score: f64, won: bool, ts_ms: u64) {
        if !(score.is_finite() && score > 0.0 && score <= 1.0) {
            return;
        }
        self.ejecutadas += 1;
        self.aprender(moneda, score, won, ts_ms, Origen::Ejecutada);
    }

    fn aprender(&mut self, moneda: usize, score: f64, won: bool, ts_ms: u64, origen: Origen) {
        if let Some(c) = self.sombra_por_moneda.get_mut(moneda) {
            c.update_at(score, won, ts_ms);
        }
        self.sombra_global.update_at(score, won, ts_ms);
        if self.observaciones.len() == WINDOW {
            self.observaciones.pop_front();
        }
        self.observaciones.push_back(Observacion {
            score,
            won,
            ts_ms,
            origen,
        });
    }

    /// p calibrada EN SOMBRA para un score, con la regla de selección del vivo.
    pub fn p_sombra(&self, moneda: usize, score: f64, ahora_ms: u64) -> f64 {
        match self.sombra_por_moneda.get(moneda) {
            Some(c) if c.observations() >= MIN_OBS_CALIBRADOR_MONEDA => c.calibrate_at(score, ahora_ms),
            _ => self.sombra_global.calibrate_at(score, ahora_ms),
        }
    }

    /// Filas abiertas.
    pub fn abiertas(&self) -> usize {
        self.abiertas.len()
    }

    /// Observaciones retenidas.
    pub fn observaciones(&self) -> impl Iterator<Item = &Observacion> {
        self.observaciones.iter()
    }

    /// `decision_id` de las filas abiertas (snapshot de decisión, CF-07).
    pub fn ids_abiertos(&self) -> impl Iterator<Item = (u64, &Decision)> {
        self.abiertas.iter().map(|a| (a.decision_id, &a.d))
    }

    /// Telemetría `WB4_SOMBRA` (sólo escribe): una línea cuando el número de
    /// filas cerradas más ejecutadas llega a una potencia de 2, con los
    /// contadores y el Brier fuera de muestra de las dos calibraciones. Sin
    /// evidencia (menos de 30 en una mitad), `N/D`.
    pub fn avisar(&mut self, trazas: bool) {
        let total = self.ejecutadas + self.contrafactuales + self.no_resueltas;
        if total == self.avisado || !total.is_power_of_two() {
            return;
        }
        self.avisado = total;
        if !trazas {
            return;
        }
        let d = self.diagnostico();
        let num = |x: Option<f64>| x.filter(|v| v.is_finite()).map_or("N/D".to_string(), |v| format!("{v:.6}"));
        let (solo, con) = (d.solo_ejecutadas, d.con_contrafactuales);
        telemetry_server::telemetry_log!(
            "WB4_SOMBRA ejecutadas={} contrafactuales={} no_resueltas={} abiertas={} solapadas={} invalidas={} llenas={} retenidas={} brier_oos_solo_ejecutadas={} brier_oos_con_contrafactuales={} brier_en_muestra_solo_ejecutadas={} brier_en_muestra_con_contrafactuales={} brecha_familia_solo_ejecutadas={} brecha_familia_con_contrafactuales={} n_prueba={}",
            self.ejecutadas,
            self.contrafactuales,
            self.no_resueltas,
            self.abiertas.len(),
            self.solapadas,
            self.invalidas,
            self.llenas,
            d.observaciones,
            num(solo.map(|r| r.brier_fuera)),
            num(con.map(|r| r.brier_fuera)),
            num(solo.map(|r| r.brier_en_muestra)),
            num(con.map(|r| r.brier_en_muestra)),
            num(solo.map(|r| r.brecha_familia_en_muestra)),
            num(con.map(|r| r.brecha_familia_en_muestra)),
            con.or(solo).map_or("N/D".to_string(), |r| r.n_prueba.to_string()),
        );
    }

    /// Brier fuera de muestra con corte temporal de las dos calibraciones.
    pub fn diagnostico(&self) -> Diagnostico {
        let obs: Vec<Observacion> = self.observaciones.iter().copied().collect();
        let (temprana, tardia) = corte_temporal(&obs, |o| o.ts_ms);
        let prueba: Vec<(f64, bool)> = tardia.iter().map(|o| (o.score, o.won)).collect();
        let ajuste_todo: Vec<(f64, bool, u64)> =
            temprana.iter().map(|o| (o.score, o.won, o.ts_ms)).collect();
        let ajuste_ejecutadas: Vec<(f64, bool, u64)> = temprana
            .iter()
            .filter(|o| o.origen == Origen::Ejecutada)
            .map(|o| (o.score, o.won, o.ts_ms))
            .collect();
        Diagnostico {
            observaciones: obs.len(),
            ejecutadas: obs.iter().filter(|o| o.origen == Origen::Ejecutada).count(),
            contrafactuales: obs.iter().filter(|o| o.origen == Origen::Contrafactual).count(),
            solo_ejecutadas: brier_platt_fuera_de_muestra(&ajuste_ejecutadas, &prueba),
            con_contrafactuales: brier_platt_fuera_de_muestra(&ajuste_todo, &prueba),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: GeometriaSombra = GeometriaSombra {
        tp_pct: 0.010,
        sl_pct: 0.005,
        friccion_pct: 0.001,
        tau_ms: 60_000.0,
    };

    fn vetada(moneda: usize, es_largo: bool, score: f64, ts: u64) -> Option<Decision> {
        Decision::desde_geometria(moneda, es_largo, 100.0, Some(G), score, score, 0, ts)
    }

    #[test]
    fn wb4a_barreras_absolutas_y_regla_del_entrenamiento() {
        let Some(l) = vetada(0, true, 0.7, 1) else { panic!("geometría válida") };
        assert!((l.tp_abs - 101.0).abs() < 1e-12 && (l.sl_abs - 99.5).abs() < 1e-12);
        let Some(s) = vetada(0, false, 0.7, 1) else { panic!("geometría válida") };
        assert!((s.tp_abs - 99.0).abs() < 1e-12 && (s.sl_abs - 100.5).abs() < 1e-12);
        assert_eq!(tocar_barrera(true, 101.0, 99.5, 101.0), Some(true));
        assert_eq!(tocar_barrera(true, 101.0, 99.5, 99.5), Some(false));
        assert_eq!(tocar_barrera(true, 101.0, 99.5, 100.2), None);
        assert_eq!(tocar_barrera(false, 99.0, 100.5, 99.0), Some(true));
        assert_eq!(tocar_barrera(false, 99.0, 100.5, 100.5), Some(false));
    }

    #[test]
    fn wb4a_una_abierta_por_moneda_y_lado_y_datos_invalidos_contados() {
        let mut libro = LibroDecisiones::new(2);
        assert_eq!(libro.registrar_vetada(vetada(0, true, 0.7, 1)), Some(0));
        assert_eq!(libro.registrar_vetada(vetada(0, true, 0.8, 2)), None);
        assert_eq!(libro.solapadas, 1);
        assert_eq!(libro.registrar_vetada(vetada(0, false, 0.8, 2)), Some(1));
        assert_eq!(libro.registrar_vetada(vetada(5, true, 0.8, 2)), None, "moneda fuera");
        assert_eq!(libro.registrar_vetada(vetada(1, true, 0.0, 2)), None, "gate score > 0");
        assert_eq!(libro.registrar_vetada(None), None, "sin geometría");
        assert_eq!(libro.invalidas, 3);
        assert_eq!(libro.abiertas(), 2);
        assert_eq!(libro.observar(1, 200.0, 10), 0, "otra moneda: O(1)");
    }
}
