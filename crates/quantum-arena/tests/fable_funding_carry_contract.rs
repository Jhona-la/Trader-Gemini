//! F-A1 (Fable, 2026-10-17) — contrato del funding per-símbolo en `CoinArena`.
//!
//! Teoría que se prueba. Un perpetuo USDⓈ-M paga funding en cada asentamiento
//! (T_F = 8 h). Una posición abierta en fase uniforme respecto al ciclo cruza
//! en media τ/T_F asentamientos (E[N] = τ/T_F, exacto) y en cada uno paga
//! d·f·nocional, con d = +1 largo, −1 corto y f la tasa del asentamiento.
//! Luego E[carry] = d·f·τ/T_F. El arena guarda la tasa PREVISTA del próximo
//! asentamiento (`lastFundingRate` de premiumIndex) con su marca de tiempo;
//! sin dato, caducado (TTL) o no finito ⇒ «sin dato» (carry 0), nunca un NaN
//! ni un valor viejo. Sólo se cobra el COSTE (d·f > 0): un pronóstico de
//! ingreso no es evidencia para abrir una posición.
//!
//! Estado sintético local; sin exchange ni genoma activo.
use quantum_arena::GlobalArena;

const T_F_MS: f64 = 8.0 * 3_600_000.0;
const TTL_MS: u64 = 360_000;
const AHORA: u64 = 1_700_000_000_000;

fn arena() -> std::sync::Arc<GlobalArena> {
    GlobalArena::build_in_own_stack(1_000.0)
}

#[test]
fn sin_publicacion_no_hay_dato_ni_carry() {
    let a = arena();
    let c = &a.coins[0];
    assert_eq!(c.funding_next_at(AHORA), None);
    assert_eq!(c.expected_funding_cost(true, T_F_MS, AHORA), 0.0);
    assert_eq!(c.expected_funding_cost(false, T_F_MS, AHORA), 0.0);
}

#[test]
fn publicar_y_leer_dentro_del_ttl() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0005, AHORA);
    assert_eq!(c.funding_next_at(AHORA), Some(0.0005));
    assert_eq!(c.funding_next_at(AHORA + 1_000), Some(0.0005));
    assert_eq!(c.funding_next_at(AHORA + TTL_MS), Some(0.0005), "el TTL es inclusivo");
}

#[test]
fn caducidad_fail_closed_a_sin_dato() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0005, AHORA);
    assert_eq!(c.funding_next_at(AHORA + TTL_MS + 1), None, "pasado el TTL no hay dato");
    assert_eq!(
        c.expected_funding_cost(true, T_F_MS, AHORA + TTL_MS + 1),
        0.0,
        "un dato caducado no cobra carry: sin dato, no un valor viejo"
    );
}

#[test]
fn un_reloj_anterior_a_la_marca_no_es_dato() {
    // Replay o reloj retrasado: el consumidor pregunta por un instante
    // anterior a la publicación. No hay dato para ese instante.
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0005, AHORA);
    assert_eq!(c.funding_next_at(AHORA - 1), None);
    assert_eq!(c.funding_next_at(0), None);
}

#[test]
fn una_marca_cero_no_publica() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0005, 0);
    assert_eq!(c.funding_next_at(AHORA), None);
}

#[test]
fn valores_no_finitos_se_ignoran_y_conservan_el_anterior() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0003, AHORA);
    c.publish_funding_next(f64::NAN, AHORA + 10);
    c.publish_funding_next(f64::INFINITY, AHORA + 20);
    c.publish_funding_next(f64::NEG_INFINITY, AHORA + 30);
    assert_eq!(c.funding_next_at(AHORA + 40), Some(0.0003));
    // Y un NaN nunca sale del arena aunque se fuerce por debajo del método.
    c.funding_rate_next
        .store(f64::NAN, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(c.funding_next_at(AHORA + 40), None);
    assert_eq!(c.expected_funding_cost(true, T_F_MS, AHORA + 40), 0.0);
}

#[test]
fn la_tasa_se_acota_como_en_el_multiplexer() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(5.0, AHORA);
    assert_eq!(c.funding_next_at(AHORA), Some(1.0));
    c.publish_funding_next(-5.0, AHORA + 1);
    assert_eq!(c.funding_next_at(AHORA + 1), Some(-1.0));
}

#[test]
fn el_carry_esperado_es_d_f_tau_sobre_t_f_y_solo_cobra_coste() {
    let a = arena();
    let c = &a.coins[0];
    let f = 0.0004; // 0,04 % por asentamiento: los largos pagan
    c.publish_funding_next(f, AHORA);
    let tol = 1e-15;
    // Largo: paga f por cada T_F, proporcional a τ.
    assert!((c.expected_funding_cost(true, T_F_MS, AHORA) - f).abs() < tol);
    assert!((c.expected_funding_cost(true, T_F_MS / 2.0, AHORA) - f / 2.0).abs() < tol);
    assert!((c.expected_funding_cost(true, 3.0 * T_F_MS, AHORA) - 3.0 * f).abs() < tol);
    // Corto con funding positivo: cobraría, pero el ingreso previsto se acota en 0.
    assert_eq!(c.expected_funding_cost(false, T_F_MS, AHORA), 0.0);
    // Funding negativo: pagan los cortos.
    c.publish_funding_next(-f, AHORA + 1);
    assert!((c.expected_funding_cost(false, T_F_MS, AHORA + 1) - f).abs() < tol);
    assert_eq!(c.expected_funding_cost(true, T_F_MS, AHORA + 1), 0.0);
}

#[test]
fn horizontes_invalidos_no_cobran() {
    let a = arena();
    let c = &a.coins[0];
    c.publish_funding_next(0.0004, AHORA);
    assert_eq!(c.expected_funding_cost(true, 0.0, AHORA), 0.0);
    assert_eq!(c.expected_funding_cost(true, -1.0, AHORA), 0.0);
    assert_eq!(c.expected_funding_cost(true, f64::NAN, AHORA), 0.0);
    assert_eq!(c.expected_funding_cost(true, f64::INFINITY, AHORA), 0.0);
}

#[test]
fn el_dato_es_por_moneda() {
    let a = arena();
    assert!(a.coins.len() >= 2, "el arena tiene al menos dos monedas");
    a.coins[0].publish_funding_next(0.0004, AHORA);
    assert_eq!(a.coins[0].funding_next_at(AHORA), Some(0.0004));
    assert_eq!(a.coins[1].funding_next_at(AHORA), None, "la moneda 1 no hereda el dato de la 0");
}
