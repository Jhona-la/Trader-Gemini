//! OPEN diagnostics: passing reproduces limitations, not an economic certification.
use quantum_arena::position::{PositionHorizon, PositionManager};

fn open(manager: &PositionManager, slot: usize, tau: u64) {
    assert!(manager.get_slot(slot).open_with_tau_and_fee(
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
        0.1,
        tau,
    ));
}

#[test]
fn open_invalid_or_sub_ten_ms_tau_is_replaced_with_thirty_seconds() {
    let m = PositionManager::default();
    open(&m, 0, 30_000);
    for tau in [f64::NAN, f64::INFINITY, -1.0, 0.0, 0.000001, 10.0] {
        assert_eq!(m.find_resonant_slot(tau, true), None);
        assert_eq!(m.find_resonant_slot(tau, false), Some(1));
    }
    assert_eq!(m.find_resonant_slot(11.0, true), Some(1));
}

#[test]
fn open_fixed_log_cutoff_changes_admission_without_dependence_evidence() {
    let m = PositionManager::default();
    open(&m, 0, 30_000);
    assert_eq!(m.find_resonant_slot(30_000.0 * 0.799_f64.exp(), true), None);
    assert_eq!(
        m.find_resonant_slot(30_000.0 * 0.801_f64.exp(), true),
        Some(1)
    );
    assert_eq!(m.find_resonant_slot(30_000.0, false), Some(1));
}

#[test]
fn open_physical_capacity_is_three_even_for_widely_separated_horizons() {
    let m = PositionManager::default();
    for (slot, tau) in [(0, 100), (1, 30_000), (2, 10_000_000)] {
        open(&m, slot, tau);
    }
    assert_eq!(m.find_resonant_slot(1e12, false), None);
}

#[test]
fn open_legacy_invalid_slot_index_aliases_the_last_slot() {
    let m = PositionManager::default();
    assert!(std::ptr::eq(m.get_slot(usize::MAX), m.get_slot(2)));
}
