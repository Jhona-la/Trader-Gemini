//! Línea G, ola 8 + Cierre Ola Ω147 (AGY):
//! Pruebas de contrato para G8-OMEGA-02, G8-OMEGA-03, G8-OMEGA-04, G8-OMEGA-05,
//! G8-OMEGA-06 y LG8-145-01/02/03/08. Ver docs/PLAN_LINEA_G_CLAUDE_2026-10-12.md §16.
use omniscient_registry::OmniscientRegistry;
use std::sync::Arc;
use strategy_core::birch_swinnerton_dyer::BirchSwinnertonDyerEngine;
use strategy_core::kahler_hodge::KahlerHodgeEngine;
use strategy_core::stat_arb::StatArbEngine;
use strategy_core::vecm_arbitrage::JohansenVecmEngine;
use strategy_core::yang_mills_gauge::YangMillsGaugeEngine;
use strategy_core::QuantumStrategy;

/// G8-OMEGA-02 (CERRADO en Ola Ω147).
/// Bajo el espejo (x, y) → (−x, −y) la corriente restauradora BSD cambia
/// exactamente de signo: `j_arriba + j_abajo == 0`.
#[test]
fn g8_omega02_la_corriente_bsd_es_impar_bajo_el_espejo() {
    let mut arriba = BirchSwinnertonDyerEngine::new();
    let mut abajo = BirchSwinnertonDyerEngine::new();
    // Tendencia: precio ≈ 2 ATR sobre (bajo) la EMA con flujo a favor.
    for k in 0..32 {
        let x = 2.0 + 0.1 * ((k % 5) as f64 - 2.0);
        let y = 1.0 + 0.05 * ((k % 3) as f64 - 1.0);
        arriba.observe_phase_point(x, y);
        abajo.observe_phase_point(-x, -y);
    }
    let j_arriba = arriba.current_metrics().bsd_restoring_current;
    let j_abajo = abajo.current_metrics().bsd_restoring_current;
    assert!(
        (j_arriba + j_abajo).abs() < 1e-6,
        "la corriente espejo debe ser la opuesta: arriba {j_arriba}, abajo {j_abajo}"
    );
    // Además, la modulación de convicción es estrictamente par bajo el espejo:
    let m_arriba = arriba.current_metrics().bsd_conviction_modulation;
    let m_abajo = abajo.current_metrics().bsd_conviction_modulation;
    assert!(
        (m_arriba - m_abajo).abs() < 1e-12,
        "la modulación de convicción debe ser par bajo el espejo: arriba {m_arriba}, abajo {m_abajo}"
    );
}

/// G8-OMEGA-03 (CERRADO en Ola Ω147).
/// Con el patrón de llamada del núcleo (un evento por moneda; la otra conserva
/// su último precio), Kähler sincroniza por par (Hayashi-Yoshida) y aprende el
/// co-movimiento igual que en el modo síncrono.
#[test]
fn g8_omega03_kahler_ve_el_comovimiento_con_el_patron_de_llamada_del_nucleo() {
    let ofis = [0.3, 0.3];
    // Control: llamada síncrona (las dos monedas cambian en la misma llamada).
    let mut sincrono = KahlerHodgeEngine::new(2);
    let mut p = [100.0, 50.0];
    for k in 0..400 {
        let r = if k % 2 == 0 { 1.05 } else { 0.96 };
        p = [p[0] * r, p[1] * r];
        sincrono.update_and_calculate_manifold(&p, &ofis);
    }
    let ricci_sincrono = sincrono.current_metrics().ricci_scalar;
    assert!(
        ricci_sincrono.abs() > 1e-9,
        "control: con retornos síncronos la métrica deja de ser la identidad ({ricci_sincrono})"
    );

    // Patrón del núcleo: un evento por moneda; la otra conserva su último precio.
    let mut por_evento = KahlerHodgeEngine::new(2);
    let mut p = [100.0, 50.0];
    for k in 0..400 {
        let r = if k % 2 == 0 { 1.05 } else { 0.96 };
        p[0] *= r;
        por_evento.update_and_calculate_manifold(&p, &ofis);
        p[1] *= r;
        por_evento.update_and_calculate_manifold(&p, &ofis);
    }
    let ricci = por_evento.current_metrics().ricci_scalar;
    assert!(
        ricci.abs() > 1e-9,
        "las monedas se movieron juntas y la métrica sigue siendo la identidad (Ricci {ricci})"
    );
    assert!(
        (ricci - ricci_sincrono).abs() < 1e-6,
        "consistencia entre modo síncrono ({ricci_sincrono}) y por-evento ({ricci})"
    );
}

/// G8-OMEGA-06 (CERRADO en Ola Ω147).
/// La interpolación bilineal C¹ Hermite sobre el retículo modular elimina los
/// escalones C⁰ en las fronteras de redondeo (p. ej. 5·a = 0.5 => a = 0.10).
#[test]
fn g8_omega06_bsd_continuidad_c1_sin_escalones_de_redondeo() {
    // Antes de Ω147, (a * 5.0).round() saltaba de 0 a 1 en a = 0.10.
    let m_left = BirchSwinnertonDyerEngine::evaluate_weierstrass_curve(0.0999, 0.20, 0.5, 0.3);
    let m_right = BirchSwinnertonDyerEngine::evaluate_weierstrass_curve(0.1001, 0.20, 0.5, 0.3);
    let delta_mod = (m_right.bsd_conviction_modulation - m_left.bsd_conviction_modulation).abs();
    let delta_l = (m_right.l_value_at_one - m_left.l_value_at_one).abs();
    assert!(
        delta_mod < 1e-3 && delta_l < 1e-2,
        "no debe existir escalón C0 al cruzar 5a = 0.5: delta_mod={delta_mod}, delta_l={delta_l}"
    );
}

/// G8-OMEGA-04, G8-OMEGA-05 y LG8-145-01 (CERRADOS en Ola Ω147).
/// 1. Yang-Mills aprende beta_{ij} bajo el patrón por-evento (una moneda por llamada).
/// 2. YangMillsGaugeEngine::new(30) con active_triangles == 0.0 (replay monoactivo)
///    devuelve `is_operable_for_coin == false`.
#[test]
fn g8_omega04_05_yang_mills_aprendizaje_por_evento_y_operabilidad_triangulos() {
    let mut ym = YangMillsGaugeEngine::new(3);
    let mut p = [100.0, 50.0, 25.0];
    ym.update_and_calculate_curvature(&p);

    // Moneda 0 se mueve el doble que la moneda 1 en llamadas separadas (por-evento)
    for k in 0..200 {
        let sign: f64 = if k % 2 == 0 { 1.0 } else { -1.0 };
        p[0] *= (0.02 * sign).exp();
        ym.update_and_calculate_curvature(&p);
        p[1] *= (0.01 * sign).exp();
        ym.update_and_calculate_curvature(&p);
        p[2] *= (0.01 * sign).exp();
        ym.update_and_calculate_curvature(&p);
    }
    assert!(
        (ym.beta_matrix[0][1] - 1.0).abs() > 5e-5,
        "Yang-Mills debe aprender beta[0][1] bajo llamadas por-evento: beta={}",
        ym.beta_matrix[0][1]
    );

    // G8-OMEGA-05 / LG8-145-01: con capacidad 30 (producción) pero active_triangles = 0.0 (monoactivo),
    // is_operable_for_coin debe ser false.
    let reg = Arc::new(OmniscientRegistry::new());
    let mut ym_prod = YangMillsGaugeEngine::new(30);
    assert!(ym_prod.init(reg.clone()).is_ok());
    reg.set_scoped("BTCUSDT", "yang_mills_current", 0.0);
    reg.set_scoped("BTCUSDT", "yang_mills_current_absent", 0.0);
    reg.set_scoped("BTCUSDT", "yang_mills_active_triangles", 0.0);
    assert!(
        !ym_prod.is_operable_for_coin(0, "BTCUSDT"),
        "con active_triangles == 0.0 (monoactivo), YangMillsGaugeEngine(30) debe ser inoperable"
    );

    reg.set_scoped("BTCUSDT", "yang_mills_active_triangles", 1.0);
    assert!(
        ym_prod.is_operable_for_coin(0, "BTCUSDT"),
        "con active_triangles >= 1.0, YangMillsGaugeEngine(30) debe ser operable"
    );
}

/// LG8-145-02, LG8-145-03 y LG8-145-08 (CERRADOS en Ola Ω147).
#[test]
fn lg8_145_02_03_08_statarb_y_vecm_operabilidad_y_ambito_estricto() {
    let reg = Arc::new(OmniscientRegistry::new());
    let mut sa = StatArbEngine::new(30, 1.5);
    let mut vecm = JohansenVecmEngine::default();
    assert!(sa.init(reg.clone()).is_ok());
    assert!(vecm.init(reg.clone()).is_ok());

    // 1. Spot caducado: statarb_ou_zscore = 0.0 con t_half finita -> inoperable
    reg.set_scoped("BTCUSDT", "statarb_ou_zscore", 0.0);
    reg.set_scoped("BTCUSDT", "statarb_half_life_ms", 30_000.0);
    assert!(
        !sa.is_operable_for_coin(0, "BTCUSDT"),
        "con statarb_ou_zscore == 0.0 (spot caducado), StatArb debe ser inoperable"
    );

    // Abstención informada con spot fresco: 0 < |z| <= 1.5 -> voto == 0.0 pero operable == true
    reg.set_scoped("BTCUSDT", "statarb_ou_zscore", 0.4);
    reg.set_scoped("BTCUSDT", "dominant_tau_ms", 60_000.0);
    assert!(sa.is_operable_for_coin(0, "BTCUSDT"));
    assert_eq!(sa.evaluate_for_coin(0, "BTCUSDT"), 0.0);

    // 2. VECM: vecm_zscore = 0.0 («sin dato») -> inoperable; 0.4 (abstención informada) -> operable
    reg.set_scoped("BTCUSDT", "vecm_zscore", 0.0);
    assert!(
        !vecm.is_operable_for_coin(0, "BTCUSDT"),
        "con vecm_zscore == 0.0 («sin dato»), VECM debe ser inoperable"
    );
    reg.set_scoped("BTCUSDT", "vecm_zscore", 0.4);
    assert!(vecm.is_operable_for_coin(0, "BTCUSDT"));
    assert_eq!(vecm.evaluate_for_coin(0, "BTCUSDT"), 0.0);

    // 3. LG8-145-08: coin_id > 0 con symbol = "" no hereda la clave global de otra moneda
    reg.set("statarb_ou_zscore", 2.5);
    reg.set("statarb_half_life_ms", 30_000.0);
    reg.set("vecm_zscore", 2.5);
    assert!(!sa.is_operable_for_coin(1, ""));
    assert_eq!(sa.evaluate_for_coin(1, ""), 0.0);
    assert!(!vecm.is_operable_for_coin(1, ""));
    assert_eq!(vecm.evaluate_for_coin(1, ""), 0.0);
}

