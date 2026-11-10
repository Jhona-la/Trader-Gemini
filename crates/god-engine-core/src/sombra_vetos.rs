//! RB-V1a (Codex RB, 2026-11-10) — LOS VETOS POSTERIORES AL RIESGO SE JUZGAN
//! EN SOMBRA.
//!
//! QUÉ: cada intención que el risk-engine validó (geometría, EV neto de
//! fricción, viabilidad, nocional) y que después vetan el consejo o el ML-GATE
//! se registra en el libro contrafactual de `audit-engine` con la geometría y
//! la fricción exactas de su orden (`MedidaOrden::geometria_sombra`), y se
//! resuelve por primer toque TP/SL/τ con los mids que llegan después. Por
//! fuente se acumula el R neto de fricción y se dicta un veredicto.
//!
//! POR QUÉ: la Demo casi no opera (a3d: una apertura en 32 h) y el
//! estrangulador medido es el ML-GATE. Un veto es una predicción («esta
//! intención pierde») que nunca recibía su prueba: si bloquea ganadoras, el
//! crecimiento compuesto se fuga sin que nadie lo vea; si bloquea perdedoras,
//! queda certificado con números.
//!
//! CÓMO:
//! - Fuentes por encima de las `REJ_*` del riesgo, que ocupan
//!   `0..REJECT_SLOTS` (reservadas para RB-V1b): consejo, ML-GATE sin modelo
//!   del roster (B3.25) y ML-GATE sin ventaja (con modelo, ventaja |p − b| por
//!   debajo del lift exigido, F1-052). El veto se atribuye al primero de la
//!   cadena: consejo antes que ML.
//! - Veredicto de Hoeffding sobre el ancho conocido de cada resultado, válido
//!   para todo n, con un error de familia α = 0,05 repartido por Bonferroni
//!   entre las fuentes y las dos colas.
//! - Ruta caliente: sin asignaciones tras construir; `observar` sale en O(1)
//!   si la moneda no tiene intenciones abiertas. Los avisos al log siguen una
//!   escala log₂ de las resoluciones de cada fuente, más uno si cambia su
//!   veredicto, y viajan por la cola sin bloqueo de la telemetría.
//! - Sólo observa: ninguna decisión lee este libro (contrato
//!   `rb_v1a_sombra_vetos_contract`).
//!
//! Límites declarados: la resolución usa el mid observado (no el último precio
//! ni el mark), sin impacto propio ni rechazo de la orden; las intenciones de
//! monedas correlacionadas no son independientes y Hoeffding lo supone.

use audit_engine::shadow_ledger::{EstadisticaFuente, IntencionVetada, LibroSombra, Veredicto};
use risk_engine::medida_sombra::GeometriaSombra;

/// Consejo: deliberación no aprobada o con consenso del lado contrario (D-738).
pub const FUENTE_CONSEJO: u8 = risk_engine::REJECT_SLOTS as u8;
/// ML-GATE sin modelo validado del roster (B3.25).
pub const FUENTE_ML_SIN_MODELO: u8 = FUENTE_CONSEJO + 1;
/// ML-GATE con modelo y ventaja por debajo del lift exigido (F1-052).
pub const FUENTE_ML_SIN_VENTAJA: u8 = FUENTE_CONSEJO + 2;

/// Las fuentes que este libro juzga, con su nombre para el log.
pub const FUENTES: [(u8, &str); 3] = [
    (FUENTE_CONSEJO, "consejo"),
    (FUENTE_ML_SIN_MODELO, "ML-GATE sin modelo"),
    (FUENTE_ML_SIN_VENTAJA, "ML-GATE sin ventaja"),
];

/// Error de familia: probabilidad de declarar algún veredicto falso entre
/// todas las fuentes (convención del consejo, la misma del plan §8).
pub const ALFA_FAMILIA: f64 = 0.05;
/// Bonferroni sobre las fuentes y las dos colas de cada una.
pub const DELTA_COLA: f64 = ALFA_FAMILIA / (2.0 * FUENTES.len() as f64);

/// La fuente a la que se atribuye un veto posterior al riesgo.
pub const fn fuente_del_veto(aprobado_por_consejo: bool, tiene_modelo: bool) -> u8 {
    if !aprobado_por_consejo {
        FUENTE_CONSEJO
    } else if !tiene_modelo {
        FUENTE_ML_SIN_MODELO
    } else {
        FUENTE_ML_SIN_VENTAJA
    }
}

/// Una intención validada por el riesgo y vetada después.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Veto {
    pub fuente: u8,
    pub moneda: usize,
    pub es_largo: bool,
    /// Precio con el que el riesgo evaluó la orden (el mid del evento).
    pub entrada: f64,
    /// Geometría y fricción de la orden validada; `None` si la medida no la
    /// trae coherente (se cuenta en `sin_geometria`).
    pub geometria: Option<GeometriaSombra>,
    /// Horizonte con el que se dimensionó la orden (D-745), en ms.
    pub tau_ms: f64,
    /// Instante del veto (tiempo del evento), en ms.
    pub t0_ms: u64,
}

/// El libro de los vetos posteriores al riesgo, con su contabilidad por moneda.
pub struct SombraVetos {
    libro: LibroSombra,
    abiertas_por_moneda: Vec<usize>,
    /// Vetos cuya orden no traía una geometría coherente.
    pub sin_geometria: u64,
    /// Avisos emitidos (se escriban o no en el log).
    pub avisos: u64,
    vistas: [u64; FUENTES.len()],
    ultimo_veredicto: [Option<Veredicto>; FUENTES.len()],
}

impl SombraVetos {
    /// Memoria fija para `monedas` monedas: después no asigna.
    pub fn new(monedas: usize) -> Self {
        Self {
            libro: LibroSombra::new(),
            abiertas_por_moneda: vec![0; monedas],
            sin_geometria: 0,
            avisos: 0,
            vistas: [0; FUENTES.len()],
            ultimo_veredicto: [None; FUENTES.len()],
        }
    }

    /// Registra el veto. Devuelve `true` si queda abierto en el libro (que
    /// admite uno por fuente, moneda y lado, y descarta los inválidos).
    pub fn registrar(&mut self, veto: Veto) -> bool {
        let Some(g) = veto.geometria else {
            self.sin_geometria += 1;
            return false;
        };
        let (Ok(moneda), Some(abiertas)) = (
            u16::try_from(veto.moneda),
            self.abiertas_por_moneda.get_mut(veto.moneda),
        ) else {
            return false;
        };
        // `as` satura: un τ no finito o no positivo queda en 0 y el libro lo
        // descarta como inválido.
        let tau_ms = if veto.tau_ms.is_finite() && veto.tau_ms > 0.0 {
            veto.tau_ms.ceil() as u64
        } else {
            0
        };
        let abierta = self.libro.registrar(IntencionVetada {
            fuente: veto.fuente,
            moneda,
            es_largo: veto.es_largo,
            entrada: veto.entrada,
            tp_pct: g.tp_pct,
            sl_pct: g.sl_pct,
            friccion_pct: g.friccion_pct,
            tau_ms,
            t0_ms: veto.t0_ms,
        });
        if abierta {
            *abiertas += 1;
        }
        abierta
    }

    /// Avanza las intenciones abiertas de `moneda` con el mid del evento. O(1)
    /// si la moneda no tiene ninguna. Con `trazas`, los avisos se escriben en
    /// el log. Devuelve cuántas se resolvieron.
    pub fn observar(&mut self, moneda: usize, mid: f64, t_ms: u64, trazas: bool) -> usize {
        let (Ok(m), Some(abiertas)) = (
            u16::try_from(moneda),
            self.abiertas_por_moneda.get_mut(moneda),
        ) else {
            return 0;
        };
        if *abiertas == 0 {
            return 0;
        }
        let resueltas = self.libro.observar(m, mid, t_ms);
        *abiertas = abiertas.saturating_sub(resueltas);
        if resueltas > 0 {
            self.avisar(trazas);
        }
        resueltas
    }

    /// Un aviso por fuente cuando su número de resoluciones llega a una
    /// potencia de 2 o cuando su veredicto cambia respecto al último avisado.
    fn avisar(&mut self, trazas: bool) {
        let fuentes = FUENTES
            .iter()
            .zip(self.vistas.iter_mut())
            .zip(self.ultimo_veredicto.iter_mut());
        for ((&(fuente, nombre), vista), ultimo) in fuentes {
            let Some(s) = self.libro.estadistica(fuente).copied() else {
                continue;
            };
            if s.n == *vista {
                continue;
            }
            *vista = s.n;
            let veredicto = s.veredicto_hoeffding(DELTA_COLA);
            if !s.n.is_power_of_two() && *ultimo == Some(veredicto) {
                continue;
            }
            *ultimo = Some(veredicto);
            self.avisos += 1;
            let (Some(media), Some(radio)) = (s.media_r(), s.radio_hoeffding(DELTA_COLA)) else {
                continue;
            };
            if trazas {
                telemetry_server::telemetry_log!(
                    "🕵️ [SOMBRA] {}: {} resueltas (TP {} · SL {} · τ {}), R neto medio {:+.3} ± {:.3} (Hoeffding, δ {:.4} por cola) → {} · abiertas {} · solapes {} · sin geometría {}",
                    nombre,
                    s.n,
                    s.tp,
                    s.sl,
                    s.vencidas,
                    media,
                    radio,
                    DELTA_COLA,
                    texto(veredicto),
                    self.libro.abiertas(),
                    self.libro.descartadas_solape,
                    self.sin_geometria
                );
            }
        }
    }

    /// Acumulados de una fuente.
    pub fn estadistica(&self, fuente: u8) -> Option<EstadisticaFuente> {
        self.libro.estadistica(fuente).copied()
    }

    /// Veredicto de Hoeffding de una fuente con el `DELTA_COLA` del libro.
    pub fn veredicto(&self, fuente: u8) -> Veredicto {
        self.libro
            .estadistica(fuente)
            .map_or(Veredicto::SinEvidencia, |s| {
                s.veredicto_hoeffding(DELTA_COLA)
            })
    }

    /// El libro completo (descartes incluidos), para informes.
    pub fn libro(&self) -> &LibroSombra {
        &self.libro
    }
}

const fn texto(v: Veredicto) -> &'static str {
    match v {
        Veredicto::AhorraDinero => "AHORRA DINERO: lo vetado pierde",
        Veredicto::BloqueaGanadoras => "BLOQUEA GANADORAS: lo vetado paga",
        Veredicto::SinEvidencia => "sin evidencia todavía",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audit_engine::shadow_ledger::MAX_FUENTES;

    const T0: u64 = 1_000_000;
    const GEOMETRIA: GeometriaSombra = GeometriaSombra {
        tp_pct: 0.010,
        sl_pct: 0.005,
        friccion_pct: 0.001,
    };

    fn veto(fuente: u8, moneda: usize, t0_ms: u64) -> Veto {
        Veto {
            fuente,
            moneda,
            es_largo: true,
            entrada: 100.0,
            geometria: Some(GEOMETRIA),
            tau_ms: 60_000.0,
            t0_ms,
        }
    }

    #[test]
    fn rb_v1a_fuentes_encima_de_los_rechazos_del_riesgo() {
        assert_eq!(usize::from(FUENTE_CONSEJO), risk_engine::REJECT_SLOTS);
        for (k, &(f, _)) in FUENTES.iter().enumerate() {
            assert!(usize::from(f) < MAX_FUENTES, "fuente {f} fuera del libro");
            assert!(
                FUENTES[..k].iter().all(|&(g, _)| g != f),
                "fuente {f} repetida"
            );
        }
        assert!((DELTA_COLA - 0.05 / 6.0).abs() < f64::EPSILON);
        assert_eq!(fuente_del_veto(false, false), FUENTE_CONSEJO);
        assert_eq!(
            fuente_del_veto(false, true),
            FUENTE_CONSEJO,
            "el consejo veta primero"
        );
        assert_eq!(fuente_del_veto(true, false), FUENTE_ML_SIN_MODELO);
        assert_eq!(fuente_del_veto(true, true), FUENTE_ML_SIN_VENTAJA);
    }

    #[test]
    fn rb_v1a_registro_cuenta_lo_que_no_puede_juzgar() {
        let mut s = SombraVetos::new(4);
        assert!(!s.registrar(Veto {
            geometria: None,
            ..veto(FUENTE_CONSEJO, 0, T0)
        }));
        assert_eq!(s.sin_geometria, 1);
        assert!(
            !s.registrar(veto(FUENTE_CONSEJO, 4, T0)),
            "moneda fuera del arena"
        );
        assert!(!s.registrar(Veto {
            tau_ms: f64::NAN,
            ..veto(FUENTE_CONSEJO, 0, T0)
        }));
        assert!(!s.registrar(Veto {
            tau_ms: 0.0,
            ..veto(FUENTE_CONSEJO, 0, T0)
        }));
        assert_eq!(
            s.libro().descartadas_invalidas,
            2,
            "τ no válido lo descarta el libro"
        );
        assert!(s.registrar(veto(FUENTE_ML_SIN_VENTAJA, 2, T0)));
        assert!(
            !s.registrar(veto(FUENTE_ML_SIN_VENTAJA, 2, T0 + 5)),
            "una por fuente, moneda y lado"
        );
        assert_eq!(s.libro().abiertas(), 1);
    }

    #[test]
    fn rb_v1a_observar_resuelve_solo_su_moneda_y_neto_de_friccion() {
        let mut s = SombraVetos::new(4);
        assert!(s.registrar(veto(FUENTE_ML_SIN_MODELO, 1, T0)));
        assert_eq!(s.observar(0, 200.0, T0 + 1, false), 0, "otra moneda");
        assert_eq!(s.observar(1, 100.5, T0 + 1, false), 0, "sin tocar barreras");
        assert_eq!(s.observar(1, 101.0, T0 + 2, true), 1, "toca el TP");
        let Some(e) = s.estadistica(FUENTE_ML_SIN_MODELO) else {
            panic!("la fuente existe");
        };
        assert_eq!((e.n, e.tp, e.sl, e.vencidas), (1, 1, 0, 0));
        // (0,010 − 0,001)/0,005 = 1,8 R
        assert!(e.media_r().is_some_and(|m| (m - 1.8).abs() < 1e-12));
        assert_eq!(s.libro().abiertas(), 0);
        assert_eq!(
            s.observar(1, 99.0, T0 + 3, false),
            0,
            "ya no queda nada abierto"
        );
    }

    #[test]
    fn rb_v1a_avisos_en_escala_log2_y_al_cambiar_el_veredicto() {
        // Todo SL: R = −1,2 y ancho 3. Con δ = 0,05/6 el radio de Hoeffding
        // √(4,5·ln(1/δ)/n) baja de 1,2 en n = 15: el veredicto cambia ahí.
        let mut s = SombraVetos::new(1);
        let mut certificado_en = None;
        for k in 0..20u64 {
            let t0 = T0 + 10 * k;
            assert!(s.registrar(veto(FUENTE_ML_SIN_VENTAJA, 0, t0)));
            assert_eq!(s.observar(0, 99.0, t0 + 1, false), 1);
            if certificado_en.is_none()
                && s.veredicto(FUENTE_ML_SIN_VENTAJA) == Veredicto::AhorraDinero
            {
                certificado_en = Some(k + 1);
            }
        }
        let n_teorico = ((1.0 / DELTA_COLA).ln() * 4.5 / 1.44).ceil() as u64;
        assert_eq!(n_teorico, 15);
        assert_eq!(certificado_en, Some(n_teorico));
        // Avisos en n = 1, 2, 4, 8 (potencias), 15 (cambio) y 16 (potencia).
        assert_eq!(s.avisos, 6);
    }
}
