//! CL-45: la reserva de una entrada se confirma con la cantidad que el
//! exchange EJECUTÓ. Antes una IOC parcial confirmaba la cantidad, el margen
//! y la comisión de la orden completa: el arena creía tener más posición que
//! el exchange, inflaba PnL y capital, y en hedge el cierre pedía más
//! cantidad de la que había (-2022).
use god_engine_core::entry_reservation::EntryReservation;
use quantum_arena::{position::PositionHorizon, symbol_registry, GlobalArena};
use std::sync::{atomic::Ordering, Arc, Mutex};

static REGISTRY: Mutex<()> = Mutex::new(());

/// Ranura 0 sin confirmar (qty 1, margen 10, comisión 0,25) y ranura 2
/// confirmada (qty 2, margen 20, comisión 0,5); used_margin 30.
fn fixture() -> (Arc<GlobalArena>, EntryReservation) {
    quantum_arena::symbols::update_dynamic_universe(vec!["RESFUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("RESFUSDT")]);
    let a = GlobalArena::build_in_own_stack(100.0);
    for (slot, qty, margin, fee) in [(0, 1.0, 10.0, 0.25), (2, 2.0, 20.0, 0.5)] {
        assert!(a.coins[0].positions.get_slot(slot).open_with_tau_and_fee(
            true,
            100.0,
            qty,
            margin,
            1000,
            101.0,
            99.0,
            PositionHorizon::Continuous,
            0.6,
            0.7,
            fee,
            30_000
        ));
    }
    let other = &a.coins[0].positions.position;
    other
        .confirm_generation(other.generation.load(Ordering::Acquire))
        .unwrap();
    a.used_margin.store(30.0, Ordering::Relaxed);
    a.unified_capital.store(99.25, Ordering::Relaxed);
    let token = EntryReservation {
        coin_id: 0,
        slot: 0,
        generation: a.coins[0].positions.scalp.generation.load(Ordering::Acquire),
        symbol: "RESFUSDT".into(),
    };
    (a, token)
}

fn cerca(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn cl45_un_llenado_parcial_escala_solo_la_propia_reserva() {
    let _guard = REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let (a, t) = fixture();
    assert_eq!(t.confirmar_llenado(&a, Some(0.4)), Ok(()));
    let s = &a.coins[0].positions.scalp;
    assert!(cerca(s.quantity.load(Ordering::Relaxed), 0.4));
    assert!(cerca(s.margin_used.load(Ordering::Relaxed), 4.0));
    assert!(cerca(s.entry_fee.load(Ordering::Relaxed), 0.1));
    assert!(s.exchange_confirmed.load(Ordering::Acquire));
    assert!(cerca(a.used_margin.load(Ordering::Relaxed), 24.0));
    assert!(cerca(a.unified_capital.load(Ordering::Relaxed), 99.40));
    let p = &a.coins[0].positions.position;
    assert!(cerca(p.quantity.load(Ordering::Relaxed), 2.0));
    assert!(cerca(p.margin_used.load(Ordering::Relaxed), 20.0));
    // Idempotente: una segunda confirmación no vuelve a escalar.
    assert_eq!(t.confirmar_llenado(&a, Some(0.2)), Ok(()));
    assert!(cerca(s.quantity.load(Ordering::Relaxed), 0.4));
    assert!(cerca(a.used_margin.load(Ordering::Relaxed), 24.0));
}

#[test]
fn cl45_un_llenado_completo_o_sin_evidencia_confirma_sin_escalar() {
    let _guard = REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    for ejecutada in [None, Some(1.0), Some(1.5), Some(f64::NAN), Some(0.0)] {
        let (a, t) = fixture();
        assert_eq!(t.confirmar_llenado(&a, ejecutada), Ok(()), "{ejecutada:?}");
        let s = &a.coins[0].positions.scalp;
        assert!(s.exchange_confirmed.load(Ordering::Acquire));
        assert!(cerca(s.quantity.load(Ordering::Relaxed), 1.0), "{ejecutada:?}");
        assert!(cerca(s.margin_used.load(Ordering::Relaxed), 10.0));
        assert!(cerca(s.entry_fee.load(Ordering::Relaxed), 0.25));
        assert!(cerca(a.used_margin.load(Ordering::Relaxed), 30.0));
        assert!(cerca(a.unified_capital.load(Ordering::Relaxed), 99.25));
    }
}

#[test]
fn cl45_un_llenado_de_otra_generacion_no_toca_la_ranura() {
    let _guard = REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let (a, t) = fixture();
    let ajena = EntryReservation { generation: t.generation + 1, ..t.clone() };
    assert!(ajena.confirmar_llenado(&a, Some(0.4)).is_err());
    let s = &a.coins[0].positions.scalp;
    assert!(!s.exchange_confirmed.load(Ordering::Acquire));
    assert!(cerca(s.quantity.load(Ordering::Relaxed), 1.0));
    assert!(cerca(a.used_margin.load(Ordering::Relaxed), 30.0));
}
