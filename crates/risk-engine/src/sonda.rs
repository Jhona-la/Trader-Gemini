//! D4 (ADR-0016, LC-F4-19) — LA SONDA ES EXPLORACIÓN PRESUPUESTADA.
//!
//! Una moneda sin historial suficiente no tiene evidencia con la que juzgar
//! el valor esperado de una orden. Sus primeras órdenes son SONDAS: se abren
//! para medir, no porque haya ventaja demostrada. ADR-0016 (D4) decidió su
//! presupuesto:
//!
//! - una sola sonda abierta en toda la cartera: abrir varias a la vez
//!   multiplica un riesgo que nadie ha medido;
//! - tamaño: la orden mínima del símbolo (D-750);
//! - como máximo [`CIERRES_DE_SONDA`] por moneda;
//! - sus pérdidas cuentan en el drawdown de D3, sin contabilidad aparte.
//!
//! Antes había dos puertas, las dos por moneda: la fase de sonda del riesgo
//! (`trade_count < 5`, exención del EV) y la sonda del núcleo
//! (`arranque_frio_sin_roster`: sin modelo, `trade_count == 0` y sin
//! posición en la moneda, exención del ML-GATE). Ninguna miraba la cartera,
//! así que cada moneda podía abrir su sonda a la vez. Y como `trade_count`
//! no se persistía, tras cada reinicio todas volvían a la fase de sonda.
//! Las dos llaman ahora a la misma función, [`admite_sonda`]:
//! - el veto del riesgo decide: rechaza con `sonda_ocupada` antes de
//!   dimensionar;
//! - la exención del ML-GATE en el núcleo ([`sonda_sin_modelo`]) la consulta
//!   también, aunque es redundante por construcción: con una sonda abierta el
//!   riesgo ya rechazó y el núcleo nunca llega al ML-GATE con una orden viva.
//!   Se conserva para que la exención no pueda contradecir a D4 si el orden
//!   de las puertas cambia.
//!
//! Reinicio. Una moneda en fase de sonda es la que tiene menos de
//! [`CIERRES_DE_SONDA`] cierres. Desde CL-108 el recuento por moneda se
//! guarda y se restaura con el estado aprendido (ADR-0017). Sin estado
//! guardado (primer arranque, o el archivo retirado) todas parten de 0: una
//! posición adoptada del exchange cuenta entonces como sonda abierta y la
//! cartera no abre otra hasta que cierra.
//!
//! F3.4-05 (Ola Ω107): Desacoplamiento adaptativo de [`CIERRES_DE_SONDA`] y prueba
//! secuencial de razón de verosimilitudes (SPRT de Wald). Si la evidencia de aciertos
//! rechaza precozmente la hipótesis nula de no-ventaja ($H_0: p \le p_0$), la moneda
//! egresa de la fase de sonda anticipadamente para operar con dimensionamiento óptimo
//! de Kelly sin esperar ciegamente a un número fijo de cierres.

use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

/// Cierres por defecto por debajo de los cuales una moneda sigue en fase de sonda.
/// ADR-0016 D4: «máximo 5 por moneda (vigente)»; es el umbral canónico que ya usaba
/// la exención del EV.
pub const CIERRES_DE_SONDA: usize = 5;

/// Resultado de la prueba secuencial de Wald (SPRT).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtDecision {
    /// H1 aceptada: la tasa de acierto supera p0 con significancia estadística.
    VentajaDemostrada,
    /// H0 aceptada: la tasa de acierto es indistinguible de o inferior a p0.
    SinVentaja,
    /// Indecidible: la evidencia acumulada aún no cruza las fronteras de Wald.
    ContinuarMuestreo,
}

/// Evalúa el Test Secuencial de Razón de Verosimilitud (SPRT) de Wald para Bernoulli(p).
///
/// H0: p = p0 (hipótesis nula, ej. p0 = 0.50, sin ventaja/ruido browniano)
/// H1: p = p1 (hipótesis alternativa, ej. p1 = 0.70, ventaja consistente en micro-scalping)
///
/// alpha: probabilidad de falso positivo (error Tipo I, típicamente 0.05)
/// beta: probabilidad de falso negativo (error Tipo II, típicamente 0.10)
///
/// Log-Likelihood Ratio:
///   $\Lambda_n = k \ln(p_1 / p_0) + (n - k) \ln((1 - p_1) / (1 - p_0))$
///
/// Fronteras de decisión de Wald:
///   $A = \ln(\beta / (1 - \alpha))$
///   $B = \ln((1 - \beta) / \alpha)$
#[inline]
pub fn sprt_bernoulli(
    n: usize,
    k: usize,
    p0: f64,
    p1: f64,
    alpha: f64,
    beta: f64,
) -> SprtDecision {
    if n == 0 || k > n || p0 <= 0.0 || p0 >= 1.0 || p1 <= 0.0 || p1 >= 1.0 || p1 <= p0 {
        return SprtDecision::ContinuarMuestreo;
    }
    let a_err = alpha.clamp(1e-4, 0.49);
    let b_err = beta.clamp(1e-4, 0.49);

    let bound_a = (b_err / (1.0 - a_err)).ln();
    let bound_b = ((1.0 - b_err) / a_err).ln();

    let llr = (k as f64) * (p1 / p0).ln() + ((n - k) as f64) * ((1.0 - p1) / (1.0 - p0)).ln();

    if !llr.is_finite() {
        return SprtDecision::ContinuarMuestreo;
    }

    if llr >= bound_b {
        SprtDecision::VentajaDemostrada
    } else if llr <= bound_a {
        SprtDecision::SinVentaja
    } else {
        SprtDecision::ContinuarMuestreo
    }
}

/// Resuelve el número máximo de cierres de la fase de sonda para una moneda o globalmente.
/// Consulta en orden:
/// 1. Clave omnisciente de la moneda: `c{coin_id}:sonda_max_cierres`
/// 2. Clave omnisciente global: `"sonda_max_cierres"`
/// 3. Constante canónica por defecto: `CIERRES_DE_SONDA` (5)
///
/// OLA Ω193 (Zero-Alloc Hot Path / F1-090): Se usa `arena.registry.get_for_coin_strict`, que resuelve
/// la clave con un buffer en stack `[u8; 64]` sin asignar `String` en el heap.
/// El resultado se clampa defensivamente a [1, 50] para garantizar continuidad operativa.
#[inline]
pub fn resolve_max_cierres_sonda(arena: &GlobalArena, coin_id: usize) -> usize {
    let val = arena
        .registry
        .get_for_coin_strict(coin_id, "sonda_max_cierres")
        .unwrap_or_else(|| {
            arena
                .registry
                .get_value_or("sonda_max_cierres", CIERRES_DE_SONDA as f64)
        });
    if val.is_finite() && val >= 1.0 {
        (val.round() as usize).clamp(1, 50)
    } else {
        CIERRES_DE_SONDA
    }
}

/// Evalúa si una moneda ha demostrado ventaja estadística concluyente vía SPRT para egresar de la sonda.
///
/// OLA Ω193 (F1-090):
/// 1. Fast-path O(1): Verificación inmediata de `n < 2` antes de lecturas del registro.
/// 2. Inversión analítica Bayesiana: `SuperGenotype::posterior_win_rate` (prior k_prior=3.84, w0=0.55)
///    comprime el win rate muestral temprano. Recuperamos los aciertos reales k sin sesgo hacia abajo.
/// 3. Calibración analítica de horizonte: defaults canónicos p0=0.50, p1=0.85, alpha=0.10, beta=0.20
///    aseguran que con n_max=5 cierres, una racha de k=4 aciertos consecutivos alcanza la frontera
///    B = ln((1-beta)/alpha) = ln(8) ≈ 2.0794 (4 * ln(1.70) ≈ 2.1225 >= 2.0794), haciendo el
///    egreso precoz efectivamente alcanzable en producción sin requerir overrides manuales.
/// 4. Presupuesto garantizado: Si se configuran parámetros personalizados en el registro que hiciesen
///    imposible el egreso en n_max cierres (n_max * ln(p1/p0) < B), se asegura un piso defensivo para p1.
#[inline]
pub fn sprt_evalua_salida_sonda(arena: &GlobalArena, coin_id: usize) -> bool {
    let coin = match arena.coins.get(coin_id) {
        Some(c) => c,
        None => return false,
    };
    let n = coin.metrics.trade_count.load(Ordering::Relaxed);
    if n < 2 {
        return false;
    }

    let sprt_enabled = arena.registry.get_value_or("sonda_sprt_enabled", 1.0) > 0.5;
    if !sprt_enabled {
        return false;
    }

    let wr = coin.metrics.win_rate.load(Ordering::Relaxed);
    if !wr.is_finite() || wr <= 0.0 {
        return false;
    }

    // Inversión de regularización Bayesiana:
    // wr_posterior = (k + 3.84 * 0.55) / (n + 3.84)
    // => k_inferred = wr * (n + 3.84) - 2.112
    let k_raw_bayes = (wr * (n as f64 + 3.84) - 3.84 * 0.55).round();
    let k_direct = ((n as f64) * wr).round();
    let k_est = if k_raw_bayes >= 0.0 && k_raw_bayes <= (n as f64) {
        k_raw_bayes.max(k_direct)
    } else {
        k_direct
    };
    let k = (k_est as usize).clamp(0, n);

    let p0 = arena
        .registry
        .get_value_or("sonda_sprt_p0", 0.50)
        .clamp(0.10, 0.90);

    let alpha = arena
        .registry
        .get_value_or("sonda_sprt_alpha", 0.10)
        .clamp(0.01, 0.20);
    let beta = arena
        .registry
        .get_value_or("sonda_sprt_beta", 0.20)
        .clamp(0.01, 0.30);

    // Cota de frontera superior de Wald para aceptar H1 (VentajaDemostrada)
    let a_err = alpha.clamp(1e-4, 0.49);
    let b_err = beta.clamp(1e-4, 0.49);
    let bound_b = ((1.0 - b_err) / a_err).ln();

    // Presupuesto de la sonda: garantizar que n_max aciertos puedan alcanzar la frontera bound_b
    let max_cierres = resolve_max_cierres_sonda(arena, coin_id);
    let n_budget = max_cierres.max(2) as f64;
    let min_p1_for_budget = (p0 * (bound_b / n_budget).exp()).clamp(p0 + 0.01, 0.95);

    let default_p1 = 0.85f64.max(min_p1_for_budget);
    let p1_candidate = arena.registry.get_value_or("sonda_sprt_p1", default_p1);
    let p1 = p1_candidate.clamp((p0 + 0.01).min(0.95), 0.95);

    sprt_bernoulli(n, k, p0, p1, alpha, beta) == SprtDecision::VentajaDemostrada
}

/// ¿La moneda está en fase de sonda? Una moneda fuera del arena no lo está:
/// no puede abrir nada.
/// Una moneda egresa de la fase de sonda si:
/// 1. Su recuento de operaciones alcanza o supera el techo dinámico [`resolve_max_cierres_sonda`], O
/// 2. Ha demostrado ventaja estadística concluyente precoz mediante SPRT de Wald.
#[inline]
pub fn en_fase_de_sonda(arena: &GlobalArena, coin_id: usize) -> bool {
    let coin = match arena.coins.get(coin_id) {
        Some(c) => c,
        None => return false,
    };
    let count = coin.metrics.trade_count.load(Ordering::Relaxed);
    let max_cierres = resolve_max_cierres_sonda(arena, coin_id);

    if count >= max_cierres {
        return false;
    }

    // Si aún está dentro del cupo de cierres, verificar si SPRT demuestra ventaja anticipada
    !sprt_evalua_salida_sonda(arena, coin_id)
}

/// ¿Hay una sonda abierta en la cartera? Lo es cualquier posición abierta en
/// una moneda que sigue en fase de sonda, evaluada adaptativamente con [`en_fase_de_sonda`].
#[inline]
pub fn hay_sonda_abierta(arena: &GlobalArena) -> bool {
    arena
        .coins
        .iter()
        .enumerate()
        .any(|(coin_id, c)| c.positions.is_any_open() && en_fase_de_sonda(arena, coin_id))
}

/// ¿Puede la moneda abrir una orden ahora conociendo ya si está en fase de sonda?
/// OLA Ω193: Evita re-evaluar [`en_fase_de_sonda`] redundantemente en el camino caliente.
#[inline]
pub fn admite_sonda_con_fase(arena: &GlobalArena, es_sonda: bool) -> bool {
    !es_sonda || !hay_sonda_abierta(arena)
}

/// ¿Puede la moneda abrir una orden ahora? Fuera de la fase de sonda,
/// siempre (lo deciden los demás vetos). En fase de sonda, sólo si no hay
/// ya una sonda abierta en la cartera.
#[inline]
pub fn admite_sonda(arena: &GlobalArena, coin_id: usize) -> bool {
    admite_sonda_con_fase(arena, en_fase_de_sonda(arena, coin_id))
}

/// Exención del ML-GATE para la primera orden de una moneda SIN modelo del
/// roster (Ola XLI·D1): con 0 cierres, esa orden es la sonda que genera la
/// evidencia que el gate exige; sin ella el motor no arrancaría. Con modelo
/// decide el lift del bosque, no la sonda. Y sólo si la cartera admite una
/// sonda ([`admite_sonda`], la misma función del veto del riesgo).
#[inline]
pub fn sonda_sin_modelo(arena: &GlobalArena, coin_id: usize, tiene_modelo_roster: bool) -> bool {
    !tiene_modelo_roster
        && arena
            .coins
            .get(coin_id)
            .is_some_and(|c| c.metrics.trade_count.load(Ordering::Relaxed) == 0)
        && admite_sonda(arena, coin_id)
}
