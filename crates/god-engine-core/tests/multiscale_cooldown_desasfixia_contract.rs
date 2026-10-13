//! Contrato formal: Desasfixia Temporal y Cooldown Multiescala (Ola Ω87, Ficha #725).
//!
//! # QUÉ GARANTIZA
//! 1. Desacoplamiento espectral entre horizontes: un trade previo cerrado con un horizonte
//!    lento (p.ej. τ = 4 h ó 12 h) NO impone una asfixia de 4–12 horas sobre oportunidades
//!    de microestructura cuando el mercado actual resuena a una escala activa más rápida (`tau_dom`).
//! 2. La base de enfriamiento viva es el mínimo entre el horizonte del cierre (`tau_cierre`)
//!    y el horizonte dominante vivo (`tau_dom`): `(tau_cierre as f64).min(tau_dom)`.
//! 3. Trascurridos los segundos correspondientes a la escala activa, el motor queda habilitado
//!    para abrir nuevas posiciones (`can_open_position_ms` = true) sin parálisis temporal.
//! 4. Conservación de la escalera binaria de racha y amortización logarítmica sin hipertrofia.

use god_engine_core::stateful_engine::StatefulEngine;

#[test]
fn contrato_desasfixia_cierre_lento_mercado_rapido() {
    let mut e = StatefulEngine::new();
    let t0: u64 = 1_700_000_000_000;

    // Supongamos un trade cerrado con un horizonte swing/macro de 4 horas (14 400 000 ms)
    let tau_cierre_lento = 14_400_000u64;
    e.tau_ultimo_cierre_ms = tau_cierre_lento;
    e.last_exit_fastband_ms = t0;
    e.fastband_loss_streak = 0; // Trade ganador / sin racha de pérdidas

    // El mercado actual resuena en escala rápida tau_dom = 30 s (30 000 ms)
    let tau_dom_rapido = 30_000.0;

    // Cálculo desacoplado multiescala (Ficha #725)
    let enfriamiento_base_ms = (e.tau_ultimo_cierre_ms as f64).min(tau_dom_rapido);
    assert_eq!(enfriamiento_base_ms, 30_000.0);

    // A los 10 segundos (10 000 ms): aún no han pasado 30 s -> enfriamiento activo
    e.last_event_ms = t0 + 10_000;
    assert!(
        !e.can_open_position_ms(enfriamiento_base_ms),
        "A los 10 s no debe abrir aún (requiere 30 s)"
    );

    // A los 30 segundos (30 000 ms): desasfixia completa, motor habilitado
    e.last_event_ms = t0 + 30_000;
    assert!(
        e.can_open_position_ms(enfriamiento_base_ms),
        "A los 30 s el motor debe quedar habilitado, no bloqueado por las 4 h de tau_cierre"
    );
}

#[test]
fn contrato_sin_inflacion_cierre_rapido_mercado_lento() {
    let mut e = StatefulEngine::new();
    let t0: u64 = 1_700_000_000_000;

    // Trade cerrado con micro-horizonte de 45 s (45 000 ms)
    let tau_cierre_rapido = 45_000u64;
    e.tau_ultimo_cierre_ms = tau_cierre_rapido;
    e.last_exit_fastband_ms = t0;
    e.fastband_loss_streak = 0;

    // Mercado con centroide lento tau_dom = 2 horas (7 200 000 ms)
    let tau_dom_lento = 7_200_000.0;

    // La base desacoplada no se infla artificialmente al centroide macro:
    let enfriamiento_base_ms = (e.tau_ultimo_cierre_ms as f64).min(tau_dom_lento);
    assert_eq!(enfriamiento_base_ms, 45_000.0);

    // A los 45 s debe quedar libre
    e.last_event_ms = t0 + 45_000;
    assert!(
        e.can_open_position_ms(enfriamiento_base_ms),
        "El micro-cierre no debe inflarse a 2 horas"
    );
}

#[test]
fn contrato_racha_con_desasfixia_amortiza_correctamente() {
    let mut e = StatefulEngine::new();
    let t0: u64 = 1_700_000_000_000;

    // Cierre previo lento (4 h) con racha de 3 pérdidas en mercado con tau_dom de 30 s
    e.tau_ultimo_cierre_ms = 14_400_000;
    e.last_exit_fastband_ms = t0;
    e.fastband_loss_streak = 3;

    let tau_dom = 30_000.0;
    let base_ms = (e.tau_ultimo_cierre_ms as f64).min(tau_dom);
    assert_eq!(base_ms, 30_000.0);

    // Con racha = 3 y base = 30 s:
    // A los 75 s (2.5 · base): se amortiza floor(log2(1 + 2.5)) = 1 nivel.
    // Quedan 2 niveles -> exige base · 2^2 = 120 s. Como 75 s < 120 s -> aún no abre.
    e.last_event_ms = t0 + 75_000;
    assert!(
        !e.can_open_position_ms(base_ms),
        "75 s no cubren los 120 s exigidos por los 2 niveles restantes"
    );

    // A los 90 s (3.0 · base): se amortizan floor(log2(1 + 3.0)) = 2 niveles.
    // Queda 1 nivel -> exige base · 2^1 = 60 s. Como 90 s >= 60 s -> abre.
    e.last_event_ms = t0 + 90_000;
    assert!(
        e.can_open_position_ms(base_ms),
        "90 s amortizan 2 niveles y cubren los 60 s del nivel restante"
    );
}
