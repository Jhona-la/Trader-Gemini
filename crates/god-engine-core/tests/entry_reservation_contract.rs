use god_engine_core::entry_reservation::{EntryReservation, ReservationError};
use quantum_arena::{
    position::{PositionHorizon, PositionTransitionError},
    symbol_registry, GlobalArena,
};
use std::sync::{atomic::Ordering, Arc, Mutex};

static REGISTRY: Mutex<()> = Mutex::new(());
fn fixture() -> (Arc<GlobalArena>, EntryReservation) {
    quantum_arena::symbols::update_dynamic_universe(vec!["RESVUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("RESVUSDT")]);
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
        generation: a.coins[0]
            .positions
            .scalp
            .generation
            .load(Ordering::Acquire),
        symbol: "RESVUSDT".into(),
    };
    (a, token)
}

#[test]
fn cancel_preserves_confirmed_neighbor_and_compensates_only_own_reservation_once() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    assert_eq!(t.cancel(&a), Ok(()));
    assert!(!a.coins[0].positions.scalp.is_open());
    assert!(a.coins[0].positions.position.is_open());
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 20.0);
    assert_eq!(a.unified_capital.load(Ordering::Relaxed), 99.5);
    assert!(t.cancel(&a).is_err());
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 20.0);
    assert_eq!(a.unified_capital.load(Ordering::Relaxed), 99.5);
}

#[test]
fn delayed_rejection_cannot_cancel_a_reused_slot() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    let p = &a.coins[0].positions.scalp;
    p.close_with_fee();
    assert!(p.open_with_horizon(
        false,
        200.0,
        3.0,
        15.0,
        2000,
        190.0,
        210.0,
        PositionHorizon::Continuous
    ));
    assert_eq!(
        t.cancel(&a),
        Err(ReservationError::Position(
            PositionTransitionError::GenerationMismatch
        ))
    );
    assert!(p.is_open());
    assert_eq!(p.entry_price.load(Ordering::Relaxed), 200.0);
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 30.0);
}

#[test]
fn confirmation_targets_captured_slot_and_prevents_later_rejection_refund() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    a.coins[0]
        .positions
        .position
        .exchange_confirmed
        .store(false, Ordering::Relaxed);
    t.confirm(&a).unwrap();
    assert!(a.coins[0]
        .positions
        .scalp
        .exchange_confirmed
        .load(Ordering::Acquire));
    assert!(!a.coins[0]
        .positions
        .position
        .exchange_confirmed
        .load(Ordering::Acquire));
    assert_eq!(
        t.cancel(&a),
        Err(ReservationError::Position(
            PositionTransitionError::AlreadyConfirmed
        ))
    );
    assert_eq!(a.unified_capital.load(Ordering::Relaxed), 99.25);
}

#[test]
fn stale_confirmation_cannot_mark_a_new_occupant() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    t.cancel(&a).unwrap();
    let p = &a.coins[0].positions.scalp;
    p.open_with_horizon(
        true,
        200.0,
        3.0,
        15.0,
        2000,
        210.0,
        190.0,
        PositionHorizon::Continuous,
    );
    assert!(t.confirm(&a).is_err());
    assert!(!p.exchange_confirmed.load(Ordering::Acquire));
}

#[test]
fn missing_coin_slot_and_changed_symbol_refuse_without_mutation() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    let mut bad = t.clone();
    bad.coin_id = usize::MAX;
    assert_eq!(bad.cancel(&a), Err(ReservationError::InvalidCoin));
    bad = t.clone();
    bad.slot = 3;
    assert_eq!(bad.cancel(&a), Err(ReservationError::InvalidSlot));
    bad = t.clone();
    bad.symbol = "OTHERUSDT".into();
    assert_eq!(bad.cancel(&a), Err(ReservationError::SymbolMismatch));
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 30.0);
    assert!(a.coins[0].positions.scalp.is_open());
}

#[test]
fn concurrent_duplicate_cancellation_has_exactly_one_compensation() {
    let _guard = REGISTRY.lock().unwrap();
    let (a, t) = fixture();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            let a = a.clone();
            let t = t.clone();
            std::thread::spawn(move || t.cancel(&a).is_ok())
        })
        .collect();
    let count: usize = threads
        .into_iter()
        .map(|h| usize::from(h.join().unwrap()))
        .sum();
    assert_eq!(count, 1);
    assert_eq!(a.used_margin.load(Ordering::Relaxed), 20.0);
    assert_eq!(a.unified_capital.load(Ordering::Relaxed), 99.5);
}
