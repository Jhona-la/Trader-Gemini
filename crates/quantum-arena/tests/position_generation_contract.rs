use quantum_arena::position::{Position, PositionHorizon, PositionTransitionError as E};
use std::sync::{atomic::Ordering, Arc, Barrier};

fn open(p: &Position) -> u64 {
    assert!(p.open_with_fee(
        true,
        100.0,
        1.0,
        10.0,
        1000,
        101.0,
        99.0,
        PositionHorizon::Continuous,
        0.6,
        0.7,
        0.1
    ));
    p.generation.load(Ordering::Acquire)
}

#[test]
fn cancellation_returns_exact_reservation_and_refuses_repetition() {
    let p = Position::default();
    let g = open(&p);
    assert_eq!(
        p.cancel_unconfirmed_generation(g),
        Ok((true, 100.0, 1.0, 10.0, 0.1))
    );
    assert_eq!(
        p.cancel_unconfirmed_generation(g),
        Err(E::GenerationMismatch)
    );
    assert!(!p.last_close_confirmed.load(Ordering::Relaxed));
}

#[test]
fn confirmation_is_idempotent_and_closed_slot_cannot_be_confirmed() {
    let p = Position::default();
    let g = open(&p);
    assert_eq!(p.confirm_generation(g), Ok(()));
    assert_eq!(p.confirm_generation(g), Ok(()));
    assert_eq!(p.cancel_unconfirmed_generation(g), Err(E::AlreadyConfirmed));
    p.close_with_fee();
    assert_eq!(
        p.confirm_generation(p.generation.load(Ordering::Acquire)),
        Err(E::Closed)
    );
}

#[test]
fn concurrent_confirmation_and_cancellation_have_one_winning_transition() {
    for _ in 0..64 {
        let p = Arc::new(Position::default());
        let g = open(&p);
        let b = Arc::new(Barrier::new(3));
        let confirm = {
            let p = p.clone();
            let b = b.clone();
            std::thread::spawn(move || {
                b.wait();
                p.confirm_generation(g)
            })
        };
        let cancel = {
            let p = p.clone();
            let b = b.clone();
            std::thread::spawn(move || {
                b.wait();
                p.cancel_unconfirmed_generation(g)
            })
        };
        b.wait();
        let c = confirm.join().unwrap();
        let r = cancel.join().unwrap();
        assert_ne!(c.is_ok(), r.is_ok());
        if c.is_ok() {
            assert!(p.is_open());
            assert_eq!(r, Err(E::AlreadyConfirmed));
        } else {
            assert!(!p.is_open());
            assert_eq!(c, Err(E::GenerationMismatch));
        }
    }
}
