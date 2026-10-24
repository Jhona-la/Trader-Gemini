use audit_engine::drift_auditor::{DriftAuditError as E, DriftRecovery};
use std::num::NonZeroU32;

fn recovery() -> DriftRecovery {
    DriftRecovery::new(NonZeroU32::new(10).unwrap())
}

#[test]
fn releases_exactly_on_tenth_clean_observation_not_eleventh() {
    let mut r = recovery();
    r.observe(&Err(E::NonfinitePnl));
    for n in 1..10 {
        r.observe(&Ok(0.0));
        assert!(r.is_blocked());
        assert_eq!(r.clean_count(), n);
    }
    r.observe(&Ok(0.0));
    assert!(!r.is_blocked());
    assert_eq!(r.clean_count(), 0);
}

#[test]
fn new_failure_discards_all_prior_clean_observations() {
    let mut r = recovery();
    r.observe(&Err(E::InvalidLimit));
    for _ in 0..9 {
        r.observe(&Ok(0.0));
    }
    r.observe(&Err(E::Exceeded {
        drift: 0.1,
        limit: 0.05,
    }));
    assert_eq!(r.clean_count(), 0);
    r.observe(&Ok(0.0));
    assert!(r.is_blocked());
    assert_eq!(r.clean_count(), 1);
}

#[test]
fn all_invalid_results_reset_recovery_including_forged_ok_nan() {
    for result in [
        Err(E::InvalidLimit),
        Err(E::IncompatibleTrade),
        Err(E::NonfinitePnl),
        Err(E::NonfiniteDifference),
        Err(E::InvalidAccumulator),
        Err(E::AccumulatorOverflow),
        Ok(f64::NAN),
        Ok(f64::INFINITY),
    ] {
        let mut r = recovery();
        r.observe(&Err(E::InvalidLimit));
        r.observe(&Ok(0.0));
        r.observe(&result);
        assert!(r.is_blocked());
        assert_eq!(r.clean_count(), 0);
    }
}

#[test]
fn clean_history_before_incident_does_not_preapprove_recovery() {
    let mut r = recovery();
    for _ in 0..100 {
        r.observe(&Ok(0.0));
    }
    assert_eq!(r.clean_count(), 0);
    r.observe(&Err(E::NonfinitePnl));
    r.observe(&Ok(0.0));
    assert_eq!(r.clean_count(), 1);
    assert!(r.is_blocked());
}

#[test]
fn one_observation_policy_and_repeated_cycles_have_no_stale_state() {
    let mut r = DriftRecovery::new(NonZeroU32::new(1).unwrap());
    for _ in 0..3 {
        r.observe(&Err(E::InvalidLimit));
        assert!(r.is_blocked());
        r.observe(&Ok(0.0));
        assert!(!r.is_blocked());
    }
}

/// LG-35 (Línea G, ola 10; hallazgo LG10-INFRA-03): el host fechaba el bloqueo en
/// ms epoch (≈ 1,76·10¹²) y consultaba la autorrecuperación con ms del proceso
/// (≈ 10⁷): `now − inicio` saturaba en 0 y el veto de deriva no se liberaba nunca
/// sin reiniciar. Un `now` anterior al bloqueo reinicia la espera en `now`.
#[test]
fn lg35_reloj_de_otra_base_reinicia_la_espera_y_libera_a_los_60_s() {
    let mut r = recovery();
    r.observe_at(&Err(E::NonfinitePnl), 1_760_000_000_000);
    assert!(r.is_blocked());
    assert!(!r.try_idle_recovery(5_000, 60_000, 0), "reinicia la espera en 5 000");
    assert!(!r.try_idle_recovery(64_999, 60_000, 0), "aún no pasan 60 s");
    assert!(r.try_idle_recovery(65_000, 60_000, 0), "60 s planos y sin incidentes: libera");
    assert!(!r.is_blocked());
}

/// LG-35: si el reloj retrocede, la espera se mide desde la lectura más baja: libera
/// 60 s después de ella, nunca antes (desde el bloqueo real han pasado al menos 60 s).
#[test]
fn lg35_un_reloj_que_retrocede_reinicia_la_espera() {
    let mut r = recovery();
    r.observe_at(&Err(E::NonfinitePnl), 100_000);
    assert!(!r.try_idle_recovery(90_000, 60_000, 0));
    assert!(!r.try_idle_recovery(149_999, 60_000, 0));
    assert!(r.try_idle_recovery(150_000, 60_000, 0));
}
