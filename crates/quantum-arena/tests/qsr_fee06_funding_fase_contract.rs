//! QSR FEE-06a — el carry de funding usa la fase del calendario de Binance.
//!
//! F-A1 cobraba E[carry] = d·f·τ/T_F: el número medio de asentamientos que
//! cruza una posición abierta en un instante cualquiera del ciclo. Pero el
//! instante NO es cualquiera: Binance asienta el funding de los perpetuos
//! USDⓈ-M a horas fijas UTC (00:00, 08:00 y 16:00 para T_F = 8 h), así que en
//! t se sabe cuánto falta para el próximo, Δ ∈ (0, T_F]. Una orden de un
//! minuto abierta 4 h antes no paga nada; la misma orden abierta un segundo
//! antes paga el asentamiento entero. τ/T_F cobraba lo mismo a las dos
//! (1/480 de f).
//!
//! Ley que se prueba. La duración de la posición sólo se conoce por su media
//! τ; la distribución de máxima entropía en [0, ∞) con media τ es la
//! exponencial. La posición sigue abierta en el asentamiento k-ésimo,
//! s_k = t + Δ + k·T_F, con probabilidad exp(−(Δ + k·T_F)/τ), y
//!
//!   E[N | Δ] = Σ_k exp(−(Δ + k·T_F)/τ) = exp(−Δ/τ) / (1 − exp(−T_F/τ)).
//!
//! Promediada en Δ uniforme sobre (0, T_F] da exactamente τ/T_F: la ley de
//! F-A1 es el promedio en la fase de ésta, que la refina con la fase conocida.
//! Se conserva todo lo demás de F-A1: sólo se cobra el coste, sin dato o
//! caducado no hay carry y la tasa es la prevista del próximo asentamiento.
//!
//! Estado sintético local; sin exchange ni genoma activo.
use quantum_arena::state::CoinArena;
use quantum_arena::GlobalArena;

const T_F_MS: f64 = 8.0 * 3_600_000.0;
/// 2024-01-01T00:00:00Z: un asentamiento real del calendario de 8 h.
const ASENTAMIENTO: u64 = 1_704_067_200_000;
const F: f64 = 0.0004;

fn arena() -> std::sync::Arc<GlobalArena> {
    GlobalArena::build_in_own_stack(1_000.0)
}

/// Coste para un largo con la tasa publicada en el mismo instante.
fn coste_largo(c: &CoinArena, tau_ms: f64, ahora: u64) -> f64 {
    c.publish_funding_next(F, ahora);
    c.expected_funding_cost(true, tau_ms, ahora)
}

#[test]
fn fee06_lejos_del_asentamiento_un_horizonte_corto_no_paga() {
    let a = arena();
    let c = &a.coins[0];
    let coste = coste_largo(c, 60_000.0, ASENTAMIENTO - 4 * 3_600_000);
    assert!(
        coste < F * 1e-12,
        "una orden de 1 min a 4 h del asentamiento no lo cruza: coste {coste:e}"
    );
}

#[test]
fn fee06_justo_antes_del_asentamiento_paga_casi_todo() {
    let a = arena();
    let c = &a.coins[0];
    let coste = coste_largo(c, 60_000.0, ASENTAMIENTO - 1_000);
    let esperado = F * (-1.0_f64 / 60.0).exp() / (-(-T_F_MS / 60_000.0).exp_m1());
    assert!(
        (coste - esperado).abs() < F * 1e-12,
        "a 1 s del asentamiento casi seguro se cruza: coste {coste:e}, ley {esperado:e}"
    );
    assert!(coste > 0.98 * F);
}

#[test]
fn fee06_promediado_en_la_fase_reproduce_f_a1() {
    let a = arena();
    let c = &a.coins[0];
    // Regla del punto medio con paso de 1 s sobre un ciclo completo: el error
    // relativo es ≈ (h/τ)²/24, < 1e-5 para τ ≥ 60 s.
    let paso_ms = 1_000u64;
    let pasos = (T_F_MS as u64) / paso_ms;
    for tau_ms in [60_000.0, 3_600_000.0, T_F_MS, 3.0 * T_F_MS] {
        let mut suma = 0.0;
        for k in 0..pasos {
            // Δ = T_F − (k + ½)·h ⇒ instante = asentamiento − Δ.
            let ahora = ASENTAMIENTO - T_F_MS as u64 + k * paso_ms + paso_ms / 2;
            suma += coste_largo(c, tau_ms, ahora);
        }
        let media = suma / pasos as f64;
        let f_a1 = F * tau_ms / T_F_MS;
        assert!(
            (media / f_a1 - 1.0).abs() < 1e-4,
            "τ = {tau_ms} ms: media en la fase {media:e} frente a d·f·τ/T_F = {f_a1:e}"
        );
    }
}

#[test]
fn fee06_mas_cerca_del_asentamiento_o_mas_largo_cuesta_mas() {
    let a = arena();
    let c = &a.coins[0];
    let tau = 3_600_000.0;
    let mut anterior = -1.0;
    // Δ de 8 h a 1 s: el coste sube al acercarse el asentamiento.
    for delta_ms in [28_800_000u64, 14_400_000, 3_600_000, 600_000, 60_000, 1_000] {
        let coste = coste_largo(c, tau, ASENTAMIENTO - delta_ms);
        assert!(
            coste > anterior,
            "Δ = {delta_ms} ms: coste {coste:e} no supera al de un Δ mayor {anterior:e}"
        );
        anterior = coste;
    }
    let mut anterior = -1.0;
    for tau_ms in [1_000.0, 60_000.0, 3_600_000.0, T_F_MS, 3.0 * T_F_MS] {
        let coste = coste_largo(c, tau_ms, ASENTAMIENTO - 3_600_000);
        assert!(
            coste > anterior,
            "τ = {tau_ms} ms: coste {coste:e} no supera al de un τ menor {anterior:e}"
        );
        anterior = coste;
    }
}

#[test]
fn fee06_el_calendario_es_el_de_binance() {
    let t = CoinArena::FUNDING_INTERVAL_MS_U64;
    assert_eq!(t, 28_800_000);
    assert_eq!(CoinArena::FUNDING_INTERVAL_MS, T_F_MS);
    // 00:00, 08:00 y 16:00 UTC del 1 de enero de 2024.
    let p = CoinArena::proximo_asentamiento_ms;
    assert_eq!(p(ASENTAMIENTO - 1, t), Some(ASENTAMIENTO));
    assert_eq!(
        p(ASENTAMIENTO, t),
        Some(ASENTAMIENTO + t),
        "en el instante del asentamiento la orden aún no estaba abierta"
    );
    assert_eq!(p(ASENTAMIENTO + 1, t), Some(ASENTAMIENTO + t));
    assert_eq!(p(ASENTAMIENTO + t + 5, t), Some(ASENTAMIENTO + 2 * t));
    // 16:00 UTC del 1 de enero de 2024 es el tercer asentamiento del día.
    assert_eq!(p(ASENTAMIENTO + t + 1, t), Some(1_704_124_800_000));
    assert_eq!(p(1_000, 0), None);
    assert_eq!(p(u64::MAX, t), None, "desbordamiento: sin calendario");
}

#[test]
fn fee06_la_ley_de_asentamientos_y_sus_bordes() {
    let n = CoinArena::asentamientos_esperados;
    // τ ≫ T: E[N] → τ/T con independencia de Δ.
    let tau = 1_000.0 * T_F_MS;
    for falta in [1.0, T_F_MS / 2.0, T_F_MS] {
        let v = n(falta, T_F_MS, tau).expect("entradas válidas");
        assert!((v / 1_000.0 - 1.0).abs() < 1e-3, "Δ = {falta}: {v}");
    }
    // Forma cerrada contra la suma directa de la serie (200 términos: con
    // τ = 3·T la razón es e^(−1/3) y la cola queda por debajo de 1e-28).
    for (falta, tau) in [
        (1_000.0, 60_000.0),
        (3_600_000.0, 3_600_000.0),
        (T_F_MS, 3.0 * T_F_MS),
    ] {
        let serie: f64 = (0..200)
            .map(|k| (-(falta + k as f64 * T_F_MS) / tau).exp())
            .sum();
        let v = n(falta, T_F_MS, tau).expect("entradas válidas");
        assert!(
            (v - serie).abs() < 1e-12 * serie.max(1.0),
            "Δ = {falta}, τ = {tau}: {v} vs {serie}"
        );
    }
    // Entradas inválidas: sin número, nunca un NaN o un infinito.
    for (falta, intervalo, tau) in [
        (f64::NAN, T_F_MS, 60_000.0),
        (-1.0, T_F_MS, 60_000.0),
        (1.0, 0.0, 60_000.0),
        (1.0, T_F_MS, 0.0),
        (1.0, T_F_MS, -5.0),
        (1.0, T_F_MS, f64::INFINITY),
        (1.0, f64::INFINITY, 60_000.0),
    ] {
        assert_eq!(
            n(falta, intervalo, tau),
            None,
            "({falta}, {intervalo}, {tau})"
        );
    }
}

#[test]
fn fee06_solo_se_cobra_el_coste() {
    let a = arena();
    let c = &a.coins[0];
    let ahora = ASENTAMIENTO - 1_000;
    c.publish_funding_next(F, ahora);
    assert!(c.expected_funding_cost(true, 60_000.0, ahora) > 0.0);
    assert_eq!(
        c.expected_funding_cost(false, 60_000.0, ahora),
        0.0,
        "el corto cobraría: el ingreso previsto no cuenta"
    );
    c.publish_funding_next(-F, ahora);
    assert!(c.expected_funding_cost(false, 60_000.0, ahora) > 0.0);
    assert_eq!(c.expected_funding_cost(true, 60_000.0, ahora), 0.0);
}
