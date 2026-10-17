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

/// LG-22 (Línea G) — DETECTOR DE PÉRDIDA DE VENTAJA. **Sin cablear.**
///
/// QUÉ FALTA HOY (ola 2 de la Línea H, E2/E3/E4): la probabilidad de la puerta
/// EV es un ajuste de Platt sobre las últimas 512 operaciones y no envejece
/// mientras se opera (LH-12). En E4 (ventaja → antipersistencia), con el #62,
/// el motor hace 41 cierres al 31,7 % (×3,4 los del mismo tramo en frío) y
/// sólo el freno de la estrategia (CL-107) lo para, al final del tramo.
///
/// QUÉ ES: la escala RÁPIDA de adaptación por fuente (rama, escala o moneda).
/// Un detector de cambio secuencial por e-proceso (e-CUSUM, Shin–Ramdas–
/// Rinaldo, «E-detectors», NEJSDS 2023) sobre la R neta de cada cierre:
///
/// - H0 (antes del cambio), por cierre y en orden de cierre:
///   E[R_t | cierres previos, geometría] ≥ μ, con μ = [`Self::referencia`].
///   Con μ = 0, p ≥ p* = (SL + f)/(TP + SL): la fuente paga su fricción. Con
///   varias posiciones abiertas de la misma fuente las pérdidas tienden a
///   cerrar antes, así que esto no equivale a que cada orden tenga E[R] ≥ μ
///   (en simulación, sin efecto material sobre el ARL).
/// - e-valor del cierre t: `e_t = 1 − λ·(r_t − μ)`, con `λ = φ/(r_max_t − μ)` y
///   `r_max_t = (TP − f)/SL` la mayor R de ESA orden, conocida al abrir (λ es
///   predecible). Bajo H0, E[e_t | pasado] = 1 − λ·(E[R] − μ) ≤ 1.
/// - La garantía exige `r − μ ≤ (r_max − μ)/φ_max` (= 2,5·(r_max − μ) con
///   φ_max = 0,4): por encima el e-valor de esa fracción se recortaría a 0, y
///   recortar las ganancias rompe E[e] ≤ 1. Esos cierres se cuentan en
///   [`Self::fuera_de_garantia`]; con dos barreras no ocurren.
/// - e-CUSUM por fracción: `M_t = e_t·max(1, M_{t−1})`, `M_0 = 0`; el
///   estadístico es la media de los de las cuatro fracciones de [`FRACCIONES`].
///   El suelo `max(1, ·)` olvida la historia favorable: una racha ganadora
///   larga no retrasa la alarma cuando la ventaja se pierde.
/// - Garantía: el e-CUSUM está acotado por el de Shiryaev–Roberts
///   (`S_t = e_t·(S_{t−1} + 1)`, `S_0 = 0`), y bajo H0 `S_t − t` es una
///   supermartingala; por muestreo opcional E[M_τ] ≤ E[τ] para todo tiempo de
///   parada (es un e-detector, y la media de e-detectores también lo es).
///   Alarmar en `M ≥ ARL` da un tiempo medio hasta la falsa alarma ≥ ARL
///   cierres para toda fuente que cumpla H0, sin fijar n de antemano.
/// - Retardo: la R por operación es dispersa (σ ≈ 1,58 R con RR 2,25), así
///   que cada cierre informa poco y el retardo depende sobre todo de μ.
///   Cierres hasta la alarma, simulados (RR 2,25, f = 0,27 R;
///   [`retardo_aproximado`] es una aproximación pesimista):
///
///   | H0                        | ARL | −0,2 R | −0,24 R (E4) | −0,3 R | −1 R |
///   |---------------------------|-----|--------|--------------|--------|------|
///   | μ = 0 (paga su fricción)  | 500 | ≈ 670  | ≈ 470        | ≈ 310  | ≈ 46 |
///   | μ = 0                     |  20 | ≈ 220  | ≈ 176        | ≈ 127  | ≈ 25 |
///   | μ = +0,3 R (la de CL-107) | 500 | ≈ 131  | ≈ 112        | ≈ 94   | ≈ 31 |
///   | μ = +0,3 R                |  20 | ≈ 63   | ≈ 55         | ≈ 48   | ≈ 17 |
///
///   El E4 tuvo 41 cierres en 50 h. Con μ = 0 ninguna prueba con nivel sobre
///   la R de los cierres lo ve a tiempo; con la ventaja declarada como μ se ve
///   en decenas de cierres, a cambio de alarmar también ante fuentes que
///   siguen pagando su fricción con menos ventaja (0 ≤ E[R] < μ). CL-107 lo
///   ve por eso: contrasta H0 = +0,30 R. Para bajar de decenas de cierres a la
///   escala del régimen hace falta más información por hora (predicción
///   frente a lo realizado en cada vela, libro sombra de las intenciones
///   vetadas), no otra prueba sobre los mismos cierres.
///
/// Distinto del freno de la estrategia (CL-107, ADR-0017): aquél falsa la
/// estrategia entera con H0 = +0,30 R y queda echado; éste es por fuente,
/// con μ elegible, y se reinicia tras la alarma para seguir midiendo. Quién
/// consume la alarma (pausar la rama, bajar su convicción, reabrir con una
/// sonda) es decisión de sus dueños.
#[derive(Debug, Clone)]
pub struct DetectorPerdidaDeVentaja {
    /// ln M de cada fracción; −∞ es M = 0 (ningún cierre desde el reinicio).
    log_cusum: [f64; FRACCIONES.len()],
    /// μ: ventaja (en R por cierre) que supone la H0.
    referencia: f64,
    pub cierres_desde_reinicio: u64,
    /// Cierres observados desde que se creó (no se reinicia).
    pub cierres_totales: u64,
    /// Cierres con R > r_max de su orden (fuera del modelo de dos barreras),
    /// desde que se creó.
    pub fuera_de_cota: u64,
    /// Cierres con `r − μ > (r_max − μ)/φ_max`: para ellos el e-valor de la
    /// fracción mayor se recorta a 0 y la garantía no vale. Desde que se creó.
    pub fuera_de_garantia: u64,
}

/// La mayor fracción de apuesta: fija hasta dónde vale la garantía.
const FRACCION_MAXIMA: f64 = FRACCIONES[FRACCIONES.len() - 1];

impl Default for DetectorPerdidaDeVentaja {
    fn default() -> Self {
        Self::nuevo()
    }
}

impl DetectorPerdidaDeVentaja {
    /// H0: la fuente paga su fricción (μ = 0).
    pub fn nuevo() -> Self {
        Self::con_referencia(0.0)
    }

    /// H0: la fuente conserva una ventaja de `referencia` R por cierre (la
    /// que declara, como la +0,30 R de CL-107). Más rápido cuanto mayor es μ;
    /// también alarma ante fuentes con 0 ≤ E[R] < μ. No finita o negativa ⇒ 0.
    pub fn con_referencia(referencia: f64) -> Self {
        let referencia = if referencia.is_finite() && referencia > 0.0 { referencia } else { 0.0 };
        Self {
            log_cusum: [f64::NEG_INFINITY; FRACCIONES.len()],
            referencia,
            cierres_desde_reinicio: 0,
            cierres_totales: 0,
            fuera_de_cota: 0,
            fuera_de_garantia: 0,
        }
    }

    pub fn referencia(&self) -> f64 {
        self.referencia
    }

    /// Observa la R neta de un cierre y la R máxima de su orden,
    /// `(TP − f)/SL`. Lo no finito se ignora, y también una orden con
    /// `r_max ≤ μ` (no puede ganar la ventaja que supone la H0).
    pub fn observar(&mut self, r: f64, r_max: f64) {
        let escala = r_max - self.referencia;
        if !(r.is_finite() && r_max.is_finite() && escala > 0.0) {
            return;
        }
        let x = r - self.referencia;
        if r > r_max {
            self.fuera_de_cota += 1;
        }
        if x * FRACCION_MAXIMA > escala {
            self.fuera_de_garantia += 1;
        }
        for (w, phi) in self.log_cusum.iter_mut().zip(FRACCIONES.iter()) {
            let e = (1.0 - phi / escala * x).max(0.0);
            // W_t = max(W_{t−1}, 0) + ln e_t  ⇔  M_t = e_t · max(1, M_{t−1}).
            *w = w.max(0.0) + e.ln();
        }
        self.cierres_desde_reinicio += 1;
        self.cierres_totales += 1;
    }

    /// Estadístico del detector (media de los e-CUSUM de cada fracción); 0
    /// sin cierres desde el reinicio.
    pub fn estadistico(&self) -> f64 {
        self.log_cusum.iter().map(|w| w.exp()).sum::<f64>() / FRACCIONES.len() as f64
    }

    /// Alarma con un tiempo medio mínimo hasta la falsa alarma de `arl`
    /// cierres bajo H0. `arl < 1` no es una política válida.
    pub fn alarma(&self, arl: f64) -> bool {
        arl.is_finite() && arl >= 1.0 && self.estadistico() >= arl
    }

    /// Vuelve a empezar a medir (tras consumir una alarma). Los contadores
    /// totales se conservan.
    pub fn reiniciar(&mut self) {
        self.log_cusum = [f64::NEG_INFINITY; FRACCIONES.len()];
        self.cierres_desde_reinicio = 0;
    }

    /// LG-27 — lo que hay que guardar para que un reinicio del host no ponga
    /// el detector a cero antes de la alarma (B4 de la revisión del PR #62).
    pub fn guardado(&self) -> DetectorGuardado {
        DetectorGuardado {
            // Sólo M = 0 (ln M = −∞) va como `None`. +∞ no se alcanza (la
            // alarma reinicia antes) y NaN tampoco (`observar` descarta lo no
            // finito); si llegaran, JSON los escribe como `null`.
            log_cusum: self
                .log_cusum
                .iter()
                .map(|w| if *w == f64::NEG_INFINITY { None } else { Some(*w) })
                .collect(),
            referencia: self.referencia,
            cierres_desde_reinicio: self.cierres_desde_reinicio,
            cierres_totales: self.cierres_totales,
            fuera_de_cota: self.fuera_de_cota,
            fuera_de_garantia: self.fuera_de_garantia,
        }
    }

    /// LG-27 — el detector de un guardado, o `None` si el guardado no es un
    /// estado al que `observar`/`reiniciar` puedan llegar:
    /// - otro número de fracciones, o un ln M no finito;
    /// - una referencia no finita, negativa o −0,0 (`con_referencia` da +0,0);
    /// - más cierres desde el reinicio, o fuera de cota, que totales;
    /// - más cierres fuera de garantía que fuera de cota (fuera de garantía
    ///   implica `r > r_max`);
    /// - algún M > 0 sin cierres desde el reinicio;
    /// - con cierres, un M = 0 que no esté en un sufijo de las fracciones: el
    ///   último cierre anula las fracciones de φ ≥ (r_max − μ)/(r − μ), que son
    ///   las mayores.
    pub fn desde_guardado(g: &DetectorGuardado) -> Option<Self> {
        if g.log_cusum.len() != FRACCIONES.len()
            || !(g.referencia.is_finite() && g.referencia >= 0.0)
            || g.referencia.is_sign_negative()
            || g.cierres_desde_reinicio > g.cierres_totales
            || g.fuera_de_cota > g.cierres_totales
            || g.fuera_de_garantia > g.fuera_de_cota
        {
            return None;
        }
        let mut log_cusum = [f64::NEG_INFINITY; FRACCIONES.len()];
        for (w, guardado) in log_cusum.iter_mut().zip(g.log_cusum.iter()) {
            match guardado {
                Some(v) if v.is_finite() => *w = *v,
                Some(_) => return None,
                None => {}
            }
        }
        if g.cierres_desde_reinicio == 0 && log_cusum.iter().any(|w| w.is_finite()) {
            return None;
        }
        if g.cierres_desde_reinicio > 0 {
            let primer_cero = log_cusum.iter().position(|w| !w.is_finite());
            if primer_cero.is_some_and(|i| log_cusum[i..].iter().any(|w| w.is_finite())) {
                return None;
            }
        }
        Some(Self {
            log_cusum,
            referencia: g.referencia,
            cierres_desde_reinicio: g.cierres_desde_reinicio,
            cierres_totales: g.cierres_totales,
            fuera_de_cota: g.fuera_de_cota,
            fuera_de_garantia: g.fuera_de_garantia,
        })
    }
}

/// LG-27 — el detector tal como se guarda en el estado aprendido. `log_cusum`
/// lleva `None` por cada fracción con M = 0 (ln M = −∞, que JSON no
/// representa).
///
/// Continuar el mismo proceso tras el reinicio mantiene la garantía (el ARL
/// bajo H0 se cuenta en cierres, no en sesiones) sobre la subsecuencia de
/// cierres OBSERVADOS: los que se resuelven con el host caído no llegan, y
/// eso es una selección por supervivencia. Con RR > 1 se pierden antes las
/// pérdidas, lo que retrasa la alarma (el lado conservador para la falsa
/// alarma). Empezar de cero en cada reinicio también conservaba el nivel,
/// pero con reinicios más frecuentes que el retardo no alarmaba nunca.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DetectorGuardado {
    pub log_cusum: Vec<Option<f64>>,
    pub referencia: f64,
    pub cierres_desde_reinicio: u64,
    pub cierres_totales: u64,
    pub fuera_de_cota: u64,
    pub fuera_de_garantia: u64,
}

/// LG-22 — retardo aproximado del detector con referencia `referencia` (μ),
/// en cierres, tras un cambio a una fuente de dos barreras con `rr`, fricción
/// `f_r` (en R) y esperanza neta `er_despues` < μ: `ln(K·arl)/g`, con
/// `g = max_φ E[ln(1 − φ·(r − μ)/(r_max − μ))]`. Es pesimista: ignora que
/// varias fracciones crecen a la vez y la reflexión en el suelo; el retardo
/// simulado es ≈ 0,65 veces éste a −0,2 R, ≈ 0,75 a −0,3 R y ≈ 1 a −1 R.
/// `None` si ninguna fracción tiene crecimiento positivo (el cambio no es
/// detectable con estas fracciones) o la entrada no es válida.
pub fn retardo_aproximado(rr: f64, f_r: f64, referencia: f64, er_despues: f64, arl: f64) -> Option<f64> {
    if !(rr.is_finite()
        && f_r.is_finite()
        && referencia.is_finite()
        && er_despues.is_finite()
        && arl.is_finite())
        || f_r < 0.0
        || referencia < 0.0
        || rr - f_r <= referencia
        || arl < 1.0
    {
        return None;
    }
    let p = ((er_despues + 1.0 + f_r) / (1.0 + rr)).clamp(0.0, 1.0);
    let r_max = rr - f_r;
    let escala = r_max - referencia;
    let (gana, pierde) = (r_max - referencia, -1.0 - f_r - referencia);
    let g = FRACCIONES
        .iter()
        .map(|phi| {
            p * (1.0 - phi / escala * gana).max(f64::MIN_POSITIVE).ln()
                + (1.0 - p) * (1.0 - phi / escala * pierde).ln()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    (g > 0.0).then(|| (FRACCIONES.len() as f64 * arl).ln() / g)
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

#[cfg(test)]
mod tests_lg22 {
    use super::*;

    struct Xorshift(u64);
    impl Xorshift {
        fn u(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// Orden de dos barreras con fricción f_R (en R): gana RR − f_R, pierde
    /// 1 + f_R. Con p = p* = (1 + f_R)/(1 + RR) la esperanza neta es 0.
    struct Geometria {
        rr: f64,
        f_r: f64,
    }
    impl Geometria {
        fn p_con_esperanza(&self, er: f64) -> f64 {
            // E[R] = p(RR − f) − (1 − p)(1 + f) = p(1 + RR) − (1 + f)
            ((er + 1.0 + self.f_r) / (1.0 + self.rr)).clamp(0.0, 1.0)
        }
        fn r_max(&self) -> f64 {
            self.rr - self.f_r
        }
        fn pierde(&self) -> f64 {
            -1.0 - self.f_r
        }
        fn cierre(&self, p: f64, rng: &mut Xorshift) -> f64 {
            if rng.u() < p { self.r_max() } else { self.pierde() }
        }
    }

    const REF: Geometria = Geometria { rr: 2.25, f_r: 0.27 };

    /// Retardo medio hasta la alarma (cierres) de una fuente con esperanza
    /// `er` desde un detector recién creado.
    fn retardo_medio(mu: f64, er: f64, arl: f64, caminos: usize, semilla: u64) -> f64 {
        let p = REF.p_con_esperanza(er);
        let mut rng = Xorshift(semilla);
        let mut suma = 0.0;
        for _ in 0..caminos {
            let mut d = DetectorPerdidaDeVentaja::con_referencia(mu);
            let mut k = 0u64;
            while !d.alarma(arl) && k < 20_000 {
                d.observar(REF.cierre(p, &mut rng), REF.r_max());
                k += 1;
            }
            suma += k as f64;
        }
        suma / caminos as f64
    }

    /// En la frontera de H0 (E[R] = μ) el e-valor del primer cierre tiene
    /// esperanza exactamente 1 para cada geometría y cada μ, y menor con
    /// ventaja. Es exacta (sin Monte Carlo): falla si la referencia se
    /// desplaza, si las ganancias se recortan a una cota común o si las
    /// fracciones se agregan con algo distinto de la media. No ve un λ mal
    /// escalado (no cambia E[e] en la frontera): de eso se ocupa la prueba del
    /// retardo.
    #[test]
    fn lg22_en_la_frontera_el_e_valor_tiene_esperanza_uno() {
        for g in [Geometria { rr: 2.25, f_r: 0.27 }, Geometria { rr: 5.0, f_r: 0.27 }] {
            for mu in [0.0, 0.3] {
                let primero = |r: f64| {
                    let mut d = DetectorPerdidaDeVentaja::con_referencia(mu);
                    d.observar(r, g.r_max());
                    d.estadistico()
                };
                let (gana, pierde) = (primero(g.r_max()), primero(g.pierde()));
                let esperanza = |p: f64| p * gana + (1.0 - p) * pierde;
                let p0 = g.p_con_esperanza(mu);
                assert!((esperanza(p0) - 1.0).abs() < 1e-12, "RR {} μ {mu}: {}", g.rr, esperanza(p0));
                assert!(esperanza(p0 + 0.05) < 1.0, "con ventaja E[e] < 1");
            }
        }
    }

    /// Cota de nivel (laxa): bajo H0 en la frontera, mezclando geometrías, el
    /// tiempo medio hasta una falsa alarma supera 1/α. La exactitud la da la
    /// prueba anterior; ésta comprueba el detector entero.
    #[test]
    fn lg22_arl_bajo_h0_en_la_frontera_con_geometrias_mezcladas() {
        const ARL: f64 = 50.0;
        let geos = [Geometria { rr: 2.25, f_r: 0.27 }, Geometria { rr: 5.0, f_r: 0.27 }];
        let mut rng = Xorshift(0x5EED_1234);
        let caminos = 400;
        let tope = (ARL * 200.0) as u64;
        let mut suma_tiempos = 0.0;
        for _ in 0..caminos {
            let mut d = DetectorPerdidaDeVentaja::nuevo();
            let mut t = 0u64;
            while t < tope && !d.alarma(ARL) {
                let g = &geos[(rng.u() * 2.0) as usize % 2];
                d.observar(g.cierre(g.p_con_esperanza(0.0), &mut rng), g.r_max());
                t += 1;
            }
            suma_tiempos += t as f64; // censurado en el tope: cota inferior
        }
        let arl_medido = suma_tiempos / caminos as f64;
        println!("observado: ARL bajo H0 = {arl_medido:.0} cierres (cota 1/α = {ARL})");
        assert!(arl_medido >= ARL, "ARL medido {arl_medido} < {ARL}");
    }

    /// Uso documentado: el detector sólo se reinicia tras una alarma. Una
    /// fuente con +0,2 R durante 600 cierres casi nunca alarma, y cuando pasa
    /// a −0,3 R se detecta en todos los caminos con un retardo cercano a la
    /// aproximación, aunque llegue con una historia favorable larga (sin el
    /// suelo del CUSUM tardaría más del triple).
    #[test]
    fn lg22_retardo_tras_una_historia_favorable_sin_falsas_alarmas() {
        let (p_antes, p_despues) = (REF.p_con_esperanza(0.2), REF.p_con_esperanza(-0.3));
        let arl = 500.0;
        let mut rng = Xorshift(0xC0FFEE);
        let (caminos, antes) = (200usize, 600usize);
        let (mut falsas, mut suma_retardo, mut detectados) = (0, 0.0, 0);
        for _ in 0..caminos {
            let mut d = DetectorPerdidaDeVentaja::nuevo();
            let mut alarma_previa = false;
            for _ in 0..antes {
                d.observar(REF.cierre(p_antes, &mut rng), REF.r_max());
                if d.alarma(arl) {
                    alarma_previa = true;
                    d.reiniciar();
                }
            }
            if alarma_previa {
                falsas += 1;
            }
            for k in 1..=5_000usize {
                d.observar(REF.cierre(p_despues, &mut rng), REF.r_max());
                if d.alarma(arl) {
                    suma_retardo += k as f64;
                    detectados += 1;
                    break;
                }
            }
        }
        let retardo = suma_retardo / detectados.max(1) as f64;
        let teorico = retardo_aproximado(REF.rr, REF.f_r, 0.0, -0.3, arl).unwrap();
        println!(
            "observado: falsas alarmas con +0,2 R en {antes} cierres = {falsas}/{caminos}; retardo medio = {retardo:.0} cierres; aproximación ln(K·ARL)/g = {teorico:.0}"
        );
        assert_eq!(detectados, caminos, "todas las pérdidas de ventaja se detectan");
        assert!(falsas * 20 <= caminos, "falsas alarmas {falsas}/{caminos} con ventaja real");
        assert!(retardo > 0.4 * teorico && retardo < 1.6 * teorico, "retardo {retardo} frente a {teorico}");
    }

    /// El E4 de la Línea H con el #62 (acierto 0,317 ⇒ E[R] ≈ −0,24 R) tuvo 41
    /// cierres en 50 h. Con H0 «paga su fricción» el detector necesita varias
    /// veces eso; con la ventaja declarada de CL-107 (+0,3 R) y su α (ARL 20),
    /// del orden de esos 41. Una inversión (−1 R) se ve en pocas decenas.
    #[test]
    fn lg22_el_e4_depende_de_la_referencia_y_la_inversion_se_ve_pronto() {
        let e4_paga = retardo_medio(0.0, -0.24, 500.0, 200, 0xBADC0DE);
        let e4_declarada = retardo_medio(0.3, -0.24, 20.0, 400, 0xD15EA5E);
        let inversion = retardo_medio(0.0, -1.0, 500.0, 400, 0xFACADE);
        println!(
            "observado: E4 con μ = 0 y ARL 500 = {e4_paga:.0} cierres; con μ = +0,3 R y ARL 20 = {e4_declarada:.0}; inversión a −1 R = {inversion:.1}"
        );
        assert!(e4_paga > 5.0 * 41.0, "E4 con μ = 0: {e4_paga}");
        assert!(e4_declarada < 2.5 * 41.0, "E4 con μ = +0,3 R: {e4_declarada}");
        assert!(inversion < 60.0, "inversión {inversion}");
    }

    /// Demostración (no prueba el detector por sí sola): con E[R] = 0 exacto,
    /// la suma de las últimas 512 R es negativa del orden de la mitad del
    /// tiempo; el detector sobre la misma serie, reiniciado tras cada alarma,
    /// alarma en una fracción ínfima de los cierres.
    #[test]
    fn lg22_la_ventana_de_512_no_controla_las_falsas_alarmas() {
        let mut rng = Xorshift(0xFEED);
        let mut ventana: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
        let mut d = DetectorPerdidaDeVentaja::nuevo();
        let (mut negativa, mut alarmas, mut medidas) = (0usize, 0usize, 0usize);
        for t in 0..20_000usize {
            let r = REF.cierre(REF.p_con_esperanza(0.0), &mut rng);
            ventana.push_back(r);
            if ventana.len() > 512 {
                ventana.pop_front();
            }
            d.observar(r, REF.r_max());
            if t >= 512 {
                medidas += 1;
                if ventana.iter().sum::<f64>() < 0.0 {
                    negativa += 1;
                }
                if d.alarma(50.0) {
                    alarmas += 1;
                    d.reiniciar();
                }
            }
        }
        let (frac_ventana, frac_detector) =
            (negativa as f64 / medidas as f64, alarmas as f64 / medidas as f64);
        println!(
            "observado: bajo E[R] = 0, ventana de 512 negativa {frac_ventana:.2} del tiempo; detector (ARL 50) alarma en {frac_detector:.4} de los cierres"
        );
        assert!(frac_ventana > 0.25, "fracción {frac_ventana}");
        assert!(frac_detector < 0.02, "fracción {frac_detector}");
    }

    /// El suelo del CUSUM olvida la historia favorable: tras cualquier número
    /// de ganancias el estadístico queda en la media de 1 − φ (el de una sola
    /// ganancia), y una pérdida lo deja exactamente en la media de los
    /// e-valores de esa pérdida (M = e·max(1, M) con M ≤ 1).
    #[test]
    fn lg22_el_suelo_olvida_la_historia_favorable() {
        for mu in [0.0, 0.3] {
            let escala = REF.r_max() - mu;
            let esperado = FRACCIONES.iter().map(|phi| 1.0 - phi / escala * (REF.pierde() - mu)).sum::<f64>()
                / FRACCIONES.len() as f64;
            let mut d = DetectorPerdidaDeVentaja::con_referencia(mu);
            for _ in 0..50 {
                d.observar(REF.r_max(), REF.r_max());
            }
            // Con suelo, cada ganancia deja M = 1 − φ (sin él, (1 − φ)^50).
            let tras_ganar = FRACCIONES.iter().map(|phi| 1.0 - phi).sum::<f64>() / FRACCIONES.len() as f64;
            assert!((d.estadistico() - tras_ganar).abs() < 1e-12, "tras 50 ganancias, M = {}", d.estadistico());
            d.observar(REF.pierde(), REF.r_max());
            assert!((d.estadistico() - esperado).abs() < 1e-12, "μ {mu}: {} frente a {esperado}", d.estadistico());
        }
    }

    #[test]
    fn lg22_bordes_contadores_y_reinicio() {
        let mut d = DetectorPerdidaDeVentaja::nuevo();
        assert_eq!(d.estadistico(), 0.0, "sin cierres, M = 0");
        assert!(!d.alarma(1.0), "sin cierres no hay alarma, tampoco con ARL 1");
        assert!(!d.alarma(0.5), "un ARL < 1 no es una política válida");
        d.observar(f64::NAN, 2.0);
        d.observar(-1.0, f64::NAN);
        d.observar(-1.0, 0.0);
        assert_eq!(d.cierres_desde_reinicio, 0);
        assert_eq!(d.estadistico(), 0.0);
        for _ in 0..40 {
            d.observar(REF.pierde(), REF.r_max());
        }
        assert!(d.alarma(50.0), "40 pérdidas seguidas alarman");
        d.observar(3.0, REF.r_max()); // por encima de r_max, dentro de la garantía
        d.observar(6.0, REF.r_max()); // por encima de r_max/φ_max = 4,95
        assert_eq!((d.fuera_de_cota, d.fuera_de_garantia), (2, 1));
        d.reiniciar();
        assert_eq!(d.estadistico(), 0.0);
        assert!(!d.alarma(50.0));
        assert_eq!((d.cierres_desde_reinicio, d.cierres_totales, d.fuera_de_cota), (0, 42, 2));
        assert_eq!(DetectorPerdidaDeVentaja::con_referencia(f64::NAN).referencia(), 0.0);
        assert_eq!(DetectorPerdidaDeVentaja::con_referencia(-0.2).referencia(), 0.0);
        let mut alto = DetectorPerdidaDeVentaja::con_referencia(3.0);
        alto.observar(-1.0, 2.0);
        assert_eq!(alto.cierres_totales, 0, "r_max ≤ μ: la orden no puede ganar la referencia");
        assert!(retardo_aproximado(2.25, 0.27, 2.0, -0.2, 500.0).is_none());
        assert!(retardo_aproximado(2.25, 0.27, 0.0, 0.1, 500.0).is_none(), "con ventaja no hay retardo");
    }
}
