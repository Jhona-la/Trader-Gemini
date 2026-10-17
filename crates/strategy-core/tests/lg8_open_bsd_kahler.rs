//! Línea G, ola 8 — dos defectos de Ω139/Ω141 con su prueba OPEN (fallan por
//! su causa; `#[ignore]` hasta que el dueño, AGY, los arregle o retire el
//! factor). Ver docs/PLAN_LINEA_G_CLAUDE_2026-10-12.md §16.
use strategy_core::birch_swinnerton_dyer::BirchSwinnertonDyerEngine;
use strategy_core::kahler_hodge::KahlerHodgeEngine;

/// G8-OMEGA-02. La corriente BSD es `tanh(−0,15·(3x² + a) + 0,45·y)·no_sing`:
/// el término en x² es PAR, así que con el precio lejos de su EMA empuja a
/// corto por arriba y por abajo. El consumidor (flow_excitation_confluence)
/// recorta los largos alineados hasta ×0,35 y la sombra espectral suma
/// 0,25·J al desplazamiento. Bajo el espejo (x, y) → (−x, −y) la corriente
/// debe cambiar de signo.
#[test]
#[ignore = "OPEN G8-OMEGA-02 (AGY): la corriente BSD no es impar bajo el espejo"]
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
}

/// G8-OMEGA-03. El núcleo llama a Kähler en CADA evento de una moneda con los
/// últimos precios de todas (lib.rs, `fresh_prices`) y el motor actualiza
/// `last_ln_prices` de todas en cada llamada: sólo la moneda del evento tiene
/// retorno ≠ 0, `rᵢ·rⱼ ≡ 0` y la correlación aprendida se queda en 0 aunque
/// las monedas se muevan juntas. La métrica es la identidad, Ricci ≡ 0 y la
/// corriente se reduce a −tanh(0,5·tanh(OFI propio)).
#[test]
#[ignore = "OPEN G8-OMEGA-03 (AGY): con una moneda por llamada Kähler no ve el co-movimiento"]
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
}
