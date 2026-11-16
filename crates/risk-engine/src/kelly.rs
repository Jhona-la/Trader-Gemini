//! WB-5 (especificación GLM-APP v3, §5.2) — f CON ENCOGIMIENTO POR ERROR
//! ESTÁNDAR, EN SOMBRA.
//!
//! El Kelly legado (`calculate_kelly_fraction`, `continuous_capital_scale`)
//! vive intacto en [`crate::kelly_legado`] y se reexporta aquí, de modo que
//! `risk_engine::kelly::calculate_kelly_fraction` sigue siendo la misma
//! función. Este archivo sólo contiene la ruta nueva: ningún literal de los
//! rieles del legado aparece aquí (prueba `sin_clamps_literales_en_ruta_nueva`).
//!
//! # La fórmula
//!
//! ```text
//! f = clamp_ruin( max(0, p_LCB − (1 − p_LCB)/b_LCB) · S(n), q )
//!   p_LCB = wilson_lcb(p̂, n, z)          (la MISMA cota de LH-04)
//!   b_LCB = PF_LCB · (1 − p̂)/p̂           (pago desde la cota del PF ya
//!                                         aplicada en el cable, pf_para_kelly:
//!                                         sin un segundo LCB)
//!   S(n)  = n / (n + z²·κ)               (κ = máx p(1−p), varianza máxima de
//!                                         una Bernoulli, en p = 1/2)
//! ```
//!
//! - El retorno temprano `f* ≤ 0 ⇒ 0` ocurre ANTES de cualquier tope: un f*
//!   negativo nunca llega a un clamp que lo eleve.
//! - `clamp_ruin` (streak-bound + axioma) es SIEMPRE el último eslabón.
//! - p_LCB y S son dos penalizaciones por incertidumbre: es una decisión
//!   conservadora declarada en la especificación, no una derivación única. La
//!   auditoría `{p_lcb, b_lcb, s, n}` lo deja visible.
//! - `b_LCB` usa la identidad de LH-04, `PF = p̂·b/(1 − p̂)`, con la que
//!   `p − (1 − p)/b` coincide con `W·(1 − 1/PF)` cuando p = p̂.
//!
//! EN SOMBRA: el núcleo la calcula en cada cierre junto al legado y la
//! registra (`sombra_wb5_*` y la línea `WB5_SOMBRA`); ninguna orden la lee.

pub use crate::kelly_legado::{calculate_kelly_fraction, continuous_capital_scale};

/// κ del encogimiento: la varianza MÁXIMA de una Bernoulli, `máx p(1 − p)`,
/// que se alcanza en p = 1/2. Fija (peor caso) a propósito: con κ = p̂(1 − p̂)
/// el encogimiento desaparecería justo en p̂ = 1, el caso más peligroso.
pub const KAPPA_BERNOULLI_MAX: f64 = {
    let p = 1.0 / 2.0;
    p * (1.0 - p)
};

/// Lo que la ruta nueva usó para una fracción, fila a fila.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuditoriaEncogimiento {
    /// Muestras con que se acotó p̂.
    pub n: f64,
    /// Cota inferior de Wilson de p̂ (la de LH-04).
    pub p_lcb: f64,
    /// Pago por operación usado en f*; con [`kelly_con_encogimiento`] es el
    /// que se deriva de la cota del PF que llegó (sin otro LCB).
    pub b_lcb: f64,
    /// Encogimiento `n/(n + z²·κ)`.
    pub s: f64,
    /// Kelly binomial sobre las cotas, antes del encogimiento.
    pub f_estrella: f64,
    /// Fracción final: `clamp_ruin(f*·S, q)`, 0 si f* ≤ 0.
    pub f_ajustada: f64,
}

impl AuditoriaEncogimiento {
    fn sin_apuesta(n: f64, p_lcb: f64, b_lcb: f64, s: f64, f_estrella: f64) -> Self {
        Self {
            n,
            p_lcb,
            b_lcb,
            s,
            f_estrella,
            f_ajustada: 0.0,
        }
    }
}

/// Encogimiento empírico-bayes `S(n) = n/(n + z²·κ)` con κ la varianza máxima
/// de una Bernoulli. Fuera de dominio (n o z no finitos, n < 0) devuelve NaN.
#[inline]
pub fn encogimiento(n: f64, z: f64) -> f64 {
    if !(n.is_finite() && z.is_finite() && n >= 0.0) {
        return f64::NAN;
    }
    n / (n + z * z * KAPPA_BERNOULLI_MAX)
}

/// Pago por operación desde un profit factor: `b = PF·(1 − p̂)/p̂` (la
/// identidad de LH-04, `PF = p̂·b/(1 − p̂)`).
#[inline]
pub fn pago_desde_pf(pf: f64, p_hat: f64) -> f64 {
    pf * (1.0 - p_hat) / p_hat
}

/// Núcleo de la ruta nueva con el pago ya dado.
///
/// `p_hat` en [0, 1], `n` ≥ 1, `b` > 0 y `z`, `q` finitos; si no, sin apuesta.
/// `q` es la probabilidad de pérdida con que `clamp_ruin` mide la racha.
pub fn kelly_encogido_con_pago(p_hat: f64, n: f64, b: f64, z: f64, q: f64) -> AuditoriaEncogimiento {
    let nan = f64::NAN;
    if !(p_hat.is_finite()
        && (0.0..=1.0).contains(&p_hat)
        && n.is_finite()
        && n >= 1.0
        && z.is_finite()
        && q.is_finite())
    {
        return AuditoriaEncogimiento::sin_apuesta(n, nan, b, nan, nan);
    }
    let p_lcb = crate::evidence::wilson_lcb(p_hat, n, z);
    let s = encogimiento(n, z);
    if !(b.is_finite() && b > 0.0) {
        return AuditoriaEncogimiento::sin_apuesta(n, p_lcb, b, s, nan);
    }
    let f_estrella = p_lcb - (1.0 - p_lcb) / b;
    // Retorno temprano ANTES de cualquier tope (OBL-2 de QODER-CLI).
    if !(f_estrella.is_finite() && f_estrella > 0.0) {
        return AuditoriaEncogimiento::sin_apuesta(n, p_lcb, b, s, f_estrella);
    }
    // `clamp_ruin` SIEMPRE el último eslabón.
    let f_ajustada = crate::ruin::clamp_ruin(f_estrella * s, q);
    AuditoriaEncogimiento {
        n,
        p_lcb,
        b_lcb: b,
        s,
        f_estrella,
        f_ajustada: if f_ajustada.is_finite() { f_ajustada.max(0.0) } else { 0.0 },
    }
}

/// La ruta nueva con las entradas del cable de cierre: el win rate de la
/// moneda, sus operaciones, la cota del PF que el cable YA aplicó
/// (`pf_para_kelly` = `profit_factor_lcb`) y la z de LH-04. El pago sale de
/// esa cota con [`pago_desde_pf`]; no se aplica otro LCB al PF.
pub fn kelly_con_encogimiento(p_hat: f64, n: f64, pf_lcb: f64, z: f64, q: f64) -> AuditoriaEncogimiento {
    let b = if pf_lcb.is_finite() && pf_lcb > 0.0 && p_hat.is_finite() && p_hat > 0.0 {
        pago_desde_pf(pf_lcb, p_hat)
    } else {
        f64::NAN
    };
    kelly_encogido_con_pago(p_hat, n, b, z, q)
}
