//! LG-9 (Línea G, G4) — CONTRATO ÚNICO DE RESULTADO DE ORDEN. **Sin cablear.**
//!
//! QUÉ ESTÁ MAL HOY (plan de la Línea G, §6): de 211 cosas que aprenden en el
//! motor, 59 lo hacen con la señal equivocada. Cada lazo inventa su etiqueta
//! —signo del PnL neto, acierto direccional contra ½, vela de 1 min, racha con
//! 5 pb brutos— y casi ninguno sabe qué geometría tenía la orden. Con RR 2,25
//! el equilibrio es p* ≈ 0,31-0,43 según la fricción: un lazo que puntúa
//! acierto contra ½ condena las fuentes rentables (una que gana el 35 % de las
//! veces tiene esperanza +0,14 R y aun así «pierde contra la moneda al aire»).
//!
//! QUÉ PROPONE ESTE MÓDULO: un registro por orden que se congela al abrir
//! (p*, TP, SL, fricción, τ) y se completa al cerrar (etiqueta de barrera y R
//! neto), y una evidencia de ventaja por ESPERANZA NETA que todos los lazos
//! puedan consumir con la misma estadística:
//!
//! - [`ResultadoDeOrden`]: el registro, con `r_neto = pnl_neto / riesgo_al_SL`.
//! - [`EvidenciaDeVentaja`]: dos e-procesos de apuesta (Ville) sobre R neto
//!   acotado, uno contra H0: E[R] ≤ 0 (certifica ventaja) y otro contra
//!   H0: E[R] ≥ 0 (condena la fuente). Bajo su H0 cada uno es una
//!   supermartingala no negativa: P(sup ≥ 1/α) ≤ α en cualquier momento de
//!   parada (no hace falta fijar n de antemano).
//!
//! Nada del camino vivo lo usa todavía: cablearlo cambia la conducta de
//! convicción por rama (Línea C / Línea E), pesos del consejo (Qoder / AGY),
//! skill de motores (Qoder) y calibración (Codex), y requiere su acuerdo y su
//! T-1. Los tests de abajo son el Monte Carlo que motiva el cambio.

/// Resultado de una orden, con la geometría congelada al abrir.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResultadoDeOrden {
    pub coin_id: usize,
    pub rama: u8,
    pub tau_orden_ms: u64,
    pub score_entrada: f64,
    /// TP, SL y fricción ida y vuelta en fracción del precio, al abrir.
    pub tp: f64,
    pub sl: f64,
    pub friccion: f64,
    /// p* = (SL + f)/(TP + SL) de la orden, al abrir.
    pub p_equilibrio: f64,
    /// `Some(true)` TP primero, `Some(false)` SL primero, `None` salida
    /// intermedia (BE, trailing, cosecha, tiempo).
    pub etiqueta_barrera: Option<bool>,
    /// PnL neto en múltiplos del riesgo al SL.
    pub r_neto: f64,
}

impl ResultadoDeOrden {
    /// Construye el resultado; `None` si la geometría o el PnL no son válidos.
    #[allow(clippy::too_many_arguments)]
    pub fn nuevo(
        coin_id: usize,
        rama: u8,
        tau_orden_ms: u64,
        score_entrada: f64,
        tp: f64,
        sl: f64,
        friccion: f64,
        etiqueta_barrera: Option<bool>,
        pnl_neto_usd: f64,
        nocional_usd: f64,
    ) -> Option<Self> {
        if !(tp.is_finite() && sl.is_finite() && friccion.is_finite())
            || tp <= 0.0
            || sl <= 0.0
            || friccion < 0.0
            || !(pnl_neto_usd.is_finite() && nocional_usd.is_finite() && nocional_usd > 0.0)
        {
            return None;
        }
        let p_equilibrio = ((sl + friccion) / (tp + sl)).clamp(0.0, 1.0);
        Some(Self {
            coin_id,
            rama,
            tau_orden_ms,
            score_entrada,
            tp,
            sl,
            friccion,
            p_equilibrio,
            etiqueta_barrera,
            r_neto: pnl_neto_usd / (nocional_usd * sl),
        })
    }

    /// Exceso de la etiqueta de barrera sobre el equilibrio: E[exceso] > 0 ⇔
    /// esperanza neta positiva bajo el modelo de dos barreras. `None` para
    /// las salidas intermedias.
    pub fn exceso_sobre_equilibrio(&self) -> Option<f64> {
        self.etiqueta_barrera
            .map(|tp_primero| if tp_primero { 1.0 } else { 0.0 } - self.p_equilibrio)
    }
}

/// Fracciones de apuesta de la mezcla, en unidades de 1/cota.
const FRACCIONES: [f64; 4] = [0.05, 0.1, 0.2, 0.4];

/// Evidencia de ventaja por esperanza neta (e-procesos de Ville sobre R neto).
#[derive(Debug, Clone)]
pub struct EvidenciaDeVentaja {
    /// Cota de la pérdida en R (R ≥ −cota_perdida) y de la ganancia (R ≤ cota_ganancia).
    cota_perdida: f64,
    cota_ganancia: f64,
    log_riqueza_a_favor: [f64; FRACCIONES.len()],
    log_riqueza_en_contra: [f64; FRACCIONES.len()],
    pub n: u64,
    suma: f64,
}

impl EvidenciaDeVentaja {
    /// `cota_perdida` y `cota_ganancia` deben acotar R neto (p. ej.
    /// 1 + f/SL y TP/SL): fuera de ellas la observación se recorta, lo que
    /// conserva la propiedad de supermartingala del lado conservador.
    pub fn nueva(cota_perdida: f64, cota_ganancia: f64) -> Self {
        Self {
            cota_perdida: cota_perdida.max(1e-9),
            cota_ganancia: cota_ganancia.max(1e-9),
            log_riqueza_a_favor: [0.0; FRACCIONES.len()],
            log_riqueza_en_contra: [0.0; FRACCIONES.len()],
            n: 0,
            suma: 0.0,
        }
    }

    pub fn observar(&mut self, r: f64) {
        if !r.is_finite() {
            return;
        }
        let r = r.clamp(-self.cota_perdida, self.cota_ganancia);
        for (k, f) in FRACCIONES.iter().enumerate() {
            // A favor: apuesta a E[R] > 0; λ·cota_perdida = f < 1 ⇒ factor > 0.
            self.log_riqueza_a_favor[k] += (1.0 + f / self.cota_perdida * r).ln();
            // En contra: apuesta a E[R] < 0; λ·cota_ganancia = f < 1.
            self.log_riqueza_en_contra[k] += (1.0 - f / self.cota_ganancia * r).ln();
        }
        self.n += 1;
        self.suma += r;
    }

    fn mezcla(log_riquezas: &[f64; FRACCIONES.len()]) -> f64 {
        log_riquezas.iter().map(|l| l.exp()).sum::<f64>() / FRACCIONES.len() as f64
    }

    /// e-valor contra H0: E[R] ≤ 0.
    pub fn e_valor_ventaja(&self) -> f64 {
        Self::mezcla(&self.log_riqueza_a_favor)
    }

    /// e-valor contra H0: E[R] ≥ 0.
    pub fn e_valor_desventaja(&self) -> f64 {
        Self::mezcla(&self.log_riqueza_en_contra)
    }

    /// Ventaja demostrada al nivel α (anytime-valid).
    pub fn tiene_ventaja(&self, alfa: f64) -> bool {
        alfa > 0.0 && self.e_valor_ventaja() >= 1.0 / alfa
    }

    /// Desventaja demostrada al nivel α: la fuente puede retirarse.
    pub fn condenada(&self, alfa: f64) -> bool {
        alfa > 0.0 && self.e_valor_desventaja() >= 1.0 / alfa
    }

    pub fn media(&self) -> f64 {
        if self.n == 0 { 0.0 } else { self.suma / self.n as f64 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{conviccion_de_rama, TasaAcierto};

    struct Xorshift(u64);
    impl Xorshift {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    const RR: f64 = 2.25;
    const F_R: f64 = 0.10; // fricción en R

    fn simular(p: f64, n: usize, semilla: u64) -> (TasaAcierto, EvidenciaDeVentaja) {
        let mut rng = Xorshift(semilla);
        let mut tasa = TasaAcierto::default();
        let mut ev = EvidenciaDeVentaja::nueva(1.0 + F_R, RR);
        for _ in 0..n {
            let gana = rng.u() < p;
            let r = if gana { RR - F_R } else { -1.0 - F_R };
            tasa.observar(r > 0.0);
            ev.observar(r);
        }
        (tasa, ev)
    }

    /// Con RR 2,25 y p = 0,35 (E[R] = +0,04 R neto de fricción), el lazo de
    /// hoy (Wilson contra ½) condena la fuente: convicción 0. La evidencia por
    /// R neto no la condena en ninguna de las 200 semillas.
    #[test]
    fn lg9_el_acierto_contra_medio_condena_una_fuente_rentable_y_la_r_neta_no() {
        let alfa = 0.05;
        let mut condenas_wilson = 0;
        let mut condenas_r = 0;
        for semilla in 1..=200u64 {
            let (tasa, ev) = simular(0.35, 300, semilla * 7919);
            if conviccion_de_rama(&tasa, 0.70) == 0.0 {
                condenas_wilson += 1;
            }
            if ev.condenada(alfa) {
                condenas_r += 1;
            }
        }
        eprintln!("[LG-9] p=0,35: Wilson contra 1/2 condena {condenas_wilson}/200; R neta condena {condenas_r}/200");
        assert!(condenas_wilson >= 180, "Wilson contra ½ condena {condenas_wilson}/200 fuentes rentables");
        assert!(condenas_r <= 10, "la evidencia por R neto condena {condenas_r}/200 (≤ α)");
    }

    /// Una fuente perdedora (p = 0,22, E[R] = −0,39 R) sí queda condenada por
    /// la R neta, y una muy rentable (p = 0,50, E[R] = +0,53 R) certifica
    /// ventaja.
    #[test]
    fn lg9_la_r_neta_condena_lo_perdedor_y_certifica_lo_rentable() {
        let alfa = 0.05;
        let mut condenadas = 0;
        let mut certificadas = 0;
        for semilla in 1..=100u64 {
            let (_, perdedora) = simular(0.22, 300, semilla * 104_729);
            if perdedora.condenada(alfa) {
                condenadas += 1;
            }
            let (_, rentable) = simular(0.50, 300, semilla * 1_299_709);
            if rentable.tiene_ventaja(alfa) {
                certificadas += 1;
            }
        }
        eprintln!("[LG-9] p=0,22 condenadas {condenadas}/100; p=0,50 certificadas {certificadas}/100");
        assert!(condenadas >= 90, "condenadas {condenadas}/100");
        assert!(certificadas >= 90, "certificadas {certificadas}/100");
    }

    /// Bajo H0 (E[R] = 0 exacto) la tasa de falsas certificaciones en
    /// cualquier momento de parada queda ≤ α (desigualdad de Ville).
    #[test]
    fn lg9_falsas_certificaciones_bajo_esperanza_nula() {
        let alfa = 0.05;
        // p tal que p·(RR − f) − (1 − p)·(1 + f) = 0.
        let p0 = (1.0 + F_R) / (RR + 1.0);
        let mut falsas = 0;
        let ensayos = 400u64;
        for semilla in 1..=ensayos {
            let mut rng = Xorshift(semilla * 15_485_863);
            let mut ev = EvidenciaDeVentaja::nueva(1.0 + F_R, RR);
            let mut alguna_vez = false;
            for _ in 0..500 {
                let r = if rng.u() < p0 { RR - F_R } else { -1.0 - F_R };
                ev.observar(r);
                if ev.tiene_ventaja(alfa) {
                    alguna_vez = true;
                    break;
                }
            }
            if alguna_vez {
                falsas += 1;
            }
        }
        let tasa = falsas as f64 / ensayos as f64;
        eprintln!("[LG-9] E[R]=0: falsas certificaciones {falsas}/{ensayos}");
        assert!(tasa <= alfa + 0.02, "falsas certificaciones {tasa}");
    }

    #[test]
    fn lg9_resultado_de_orden_congela_la_geometria() {
        let r = ResultadoDeOrden::nuevo(3, 11, 600_000, 0.6, 0.0225, 0.01, 0.0012, Some(true), 0.1131, 5.0)
            .expect("válido");
        assert!((r.p_equilibrio - (0.0112 / 0.0325)).abs() < 1e-12);
        assert!((r.r_neto - 0.1131 / 0.05).abs() < 1e-12);
        assert!((r.exceso_sobre_equilibrio().unwrap() - (1.0 - r.p_equilibrio)).abs() < 1e-12);
        assert!(ResultadoDeOrden::nuevo(0, 0, 0, 0.0, 0.0, 0.01, 0.001, None, 0.0, 5.0).is_none());
        assert!(ResultadoDeOrden::nuevo(0, 0, 0, 0.0, 0.02, 0.01, 0.001, None, f64::NAN, 5.0).is_none());
    }
}
