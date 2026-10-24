//! SOL (2026-10-20, censo C-F7b §3-3): CONTRATO DE CONSUMO de la sombra de
//! fricción — la corrección medida pasa de telemetría a insumo VIVO del gate,
//! pero sólo cuando la evidencia la sostiene. Cuatro condiciones, todas sobre
//! información ya disponible en t (los fills son cierres confirmados; no hay
//! look-ahead):
//!
//! 1. EVIDENCIA: sombra publicada (n ≥ MIN_FILLS_SOMBRA implícito) y finita.
//! 2. SIGNIFICANCIA: |media| > ee — la corrección supera su propio ruido.
//! 3. CONSISTENCIA OOS: ambas mitades temporales del mismo signo que la media.
//! 4. FAIL-SAFE: |media| ≤ modelo — un error mayor al 100 % del modelo no se
//!    promueve (amplificaría el error, no lo corregiría).
//!
//! Sin promo, el gate usa EXACTAMENTE el modelo de siempre: `sonda_de_banda`
//! consume `friccion_efectiva_rt`, que sin corrección promovida devuelve el
//! modelo puro. La fricción efectiva nunca es negativa.
//!
//! Cifras: fill con precio 50 000 × qty 0,001 ⇒ nocional 50; fee 0,025 ⇒ 5 bp.

use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::sync::atomic::Ordering;

fn nucleo_con_precio() -> GodEngineCore {
    let arena = GlobalArena::build_in_own_stack(13.0);
    let core = GodEngineCore::new(arena);
    core.arena.coins[0].current_price.store(100.0, Ordering::Relaxed);
    core.arena.coins[0].current_atr.store(0.5, Ordering::Relaxed);
    // Geometría SANA: sin Hurst ni pronóstico de sigma el TP/SL degenera y
    // p* deja de ser sensible a la fricción (verificado en CI: p* cambiaba
    // 1 ULP). Con hurst 0,5 y sigma 0,5 % el probe tiene escala real y la
    // fricción entra en p* como debe.
    core.arena.coins[0].hurst_exponent.store(0.5, Ordering::Relaxed);
    for ancla in core.arena.coins[0].sigma_forecast.iter() {
        ancla.store(0.005, Ordering::Relaxed);
    }
    core
}

/// Siembra directa de la sombra (lo que el host publicaría vía
/// `publicar_sombras_auditor`), para los casos de borde del contrato.
fn sembrar_sombra(core: &GodEngineCore, media: f64, ee: f64, a: f64, b: f64, n: f64) {
    core.arena
        .registry
        .set_for_coin(0, "sombra_friccion_real_menos_modelada_bp", media);
    core.arena.registry.set_for_coin(0, "sombra_friccion_ee_bp", ee);
    core.arena.registry.set_for_coin(0, "sombra_friccion_mitad_a_bp", a);
    core.arena.registry.set_for_coin(0, "sombra_friccion_mitad_b_bp", b);
    core.arena.registry.set_for_coin(0, "sombra_friccion_n", n);
}

#[test]
fn sin_sombra_el_gate_usa_el_modelo_puro() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    let efectiva = core.friccion_efectiva_rt(0, 0.5 / 100.0).expect("efectiva sana");
    assert_eq!(
        efectiva,
        modelo_bp / 1e4,
        "sin sombra la efectiva es el modelo, bit a bit"
    );
    assert_eq!(
        core.correccion_friccion_promovida_bp(0, modelo_bp),
        0.0,
        "sin sombra no hay promo"
    );
}

#[test]
fn promo_con_las_cuatro_condiciones() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    // +1 bp, ruido 0, ambas mitades +1, n 24: todas las condiciones se cumplen.
    sembrar_sombra(&core, 1.0, 0.0, 1.0, 1.0, 24.0);
    assert_eq!(core.correccion_friccion_promovida_bp(0, modelo_bp), 1.0);
    let efectiva = core.friccion_efectiva_rt(0, 0.5 / 100.0).expect("efectiva sana");
    assert!(
        (efectiva - (modelo_bp + 1.0) / 1e4).abs() < 1e-15,
        "efectiva = modelo + 1 bp, obtuvo {} vs {}",
        efectiva,
        (modelo_bp + 1.0) / 1e4
    );
}

#[test]
fn mitades_divergentes_no_promueven() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    // Media +2 pero mitades +2/−2: la corrección es un artefacto de régimen.
    sembrar_sombra(&core, 2.0, 0.1, 2.0, -2.0, 24.0);
    assert_eq!(
        core.correccion_friccion_promovida_bp(0, modelo_bp),
        0.0,
        "OOS inconsistente ⇒ sin promo"
    );
}

#[test]
fn correccion_dentro_del_ruido_no_promueve() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    // |0,5| <= ee 2,0: la corrección no supera su propio ruido.
    sembrar_sombra(&core, 0.5, 2.0, 0.4, 0.6, 24.0);
    assert_eq!(core.correccion_friccion_promovida_bp(0, modelo_bp), 0.0);
}

#[test]
fn error_mayor_al_modelo_no_promueve() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    // 100 bp de error con un modelo de unos pocos bp: el modelo no es
    // corregible, promover amplificaría el error.
    sembrar_sombra(&core, 100.0, 0.0, 100.0, 100.0, 24.0);
    assert_eq!(
        core.correccion_friccion_promovida_bp(0, modelo_bp),
        0.0,
        "fail-safe: |error| > modelo ⇒ sin promo"
    );
}

#[test]
fn friccion_efectiva_nunca_negativa() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    // Corrección exactamente −modelo (límite del fail-safe): la promo Procedre
    // pero la efectiva cae a 0, jamás negativa.
    sembrar_sombra(&core, -modelo_bp, 0.0, -modelo_bp, -modelo_bp, 24.0);
    let promo = core.correccion_friccion_promovida_bp(0, modelo_bp);
    assert_eq!(promo, -modelo_bp, "el límite exacto promueve");
    let efectiva = core.friccion_efectiva_rt(0, 0.5 / 100.0).expect("efectiva sana");
    assert_eq!(efectiva, 0.0, "nunca negativa: clamp en 0");
}

#[test]
fn promo_aislada_por_moneda() {
    let core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");
    sembrar_sombra(&core, 1.0, 0.0, 1.0, 1.0, 24.0);
    assert_eq!(core.correccion_friccion_promovida_bp(0, modelo_bp), 1.0);
    assert_eq!(
        core.correccion_friccion_promovida_bp(1, modelo_bp),
        0.0,
        "la moneda sin sombra no hereda la corrección"
    );
}

#[test]
fn end_to_end_fills_del_host_promueven_y_mueven_el_gate() {
    use audit_engine::auditor_interno::ExchangeFillRecord;
    let mut core = nucleo_con_precio();
    let modelo_bp = core.friccion_modelada_rt_bp(0).expect("modelo sano");

    // p* de la banda ANTES de la promo (modelo puro).
    let p_antes = core.p_equilibrio_de_banda(0, 1_000.0);

    // 24 cierres del host con fee 0,025 (5 bp reales sobre nocional 50).
    for i in 0..24u64 {
        let pnl = if i % 2 == 0 { 0.5 } else { -0.2 };
        core.registrar_fill_independiente_host(ExchangeFillRecord {
            timestamp_ns: 1_700_000_000_000_000_000 + i * 1_000_000_000,
            symbol_id: 0,
            is_buyer: false,
            price: 50_000.0,
            qty: 0.001,
            fee_usd: 0.025,
            realized_pnl: pnl,
            is_closed_position: true,
        });
    }
    core.publicar_sombras_auditor(0, 0.9);

    // La sombra publicada: media = 5 bp − modelo, homogénea ⇒ ee 0, mitades iguales.
    let media = core
        .arena
        .registry
        .get_for_coin_strict_or(0, "sombra_friccion_real_menos_modelada_bp", f64::NAN);
    assert!(
        (media - (5.0 - modelo_bp)).abs() < 1e-9,
        "media publicada = 5 − modelo = {}, obtuvo {}",
        5.0 - modelo_bp,
        media
    );

    // Promo sólo si |5 − modelo| ≤ modelo (fail-safe); si el modelo es enorme
    // respecto al fee real, la corrección no se promueve y el gate no cambia.
    let promo = core.correccion_friccion_promovida_bp(0, modelo_bp);
    let esperado = if (5.0 - modelo_bp).abs() <= modelo_bp && (5.0 - modelo_bp).abs() > 0.0 {
        5.0 - modelo_bp
    } else {
        0.0
    };
    assert!((promo - esperado).abs() < 1e-9, "promo = {promo}, esperado {esperado}");
    let publicado = core
        .arena
        .registry
        .get_for_coin_strict_or(0, "friccion_promovida_bp", f64::NAN);
    assert!((publicado - esperado).abs() < 1e-9, "promo publicada = {publicado}");

    // El gate consume la promo: p* de la banda se mueve si y sólo si hay promo.
    let p_despues = core.p_equilibrio_de_banda(0, 1_000.0);
    if esperado.abs() > 1e-12 {
        assert!(
            (p_despues - p_antes).abs() > 1e-9,
            "con promo la fricción efectiva del gate cambia (p* {p_antes} → {p_despues})"
        );
    } else {
        assert_eq!(p_despues, p_antes, "sin promo el gate no cambia");
    }
}
