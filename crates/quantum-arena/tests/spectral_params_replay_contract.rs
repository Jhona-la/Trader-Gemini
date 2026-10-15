//! Numeric parameter admission and replay invariance for the paired spectrum.
//!
//! Identical market inputs feed a healthy control and the reconfigured subject.
//! Only the subject receives an invalid EWMA forgetting factor. These contracts
//! preserve default maturity, family, alpha and the e-process betting fraction;
//! they do not certify its conditional null, epoch policy or economic returns.
use quantum_arena::espectral_multiactivo::{EspectralMultiactivo, EspectralMultiactivoParams};
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;

const SCALE: usize = 19;
const SEED_PAIRS: u64 = 120;

#[derive(Debug, PartialEq, Eq)]
struct Observable {
    samples: Option<u64>,
    pair: Option<u64>,
    veto: Option<u64>,
    group: Option<u64>,
    weighted: Option<(u64, u64)>,
}

fn observable(ma: &EspectralMultiactivo) -> Observable {
    Observable {
        samples: ma.muestras_par(0, 1, SCALE),
        pair: ma.coherencia_par(0, 1, SCALE).map(f64::to_bits),
        veto: ma.coherencia_par_veto(0, 1, SCALE).map(f64::to_bits),
        group: ma.coherencia_media_con_todas(0, SCALE).map(f64::to_bits),
        weighted: ma
            .coherencia_ponderada_par(0, 1, SCALE)
            .map(|(ic, weight)| (ic.to_bits(), weight.to_bits())),
    }
}

fn close_time(ordinal: u64) -> u64 {
    ordinal * SPECTRUM_SCALES_MS[SCALE].ceil() as u64
}

fn pair(ma: &mut EspectralMultiactivo, ordinal: u64) {
    let tau = SPECTRUM_SCALES_MS[SCALE];
    let ts = close_time(ordinal);
    let r = if ordinal % 2 == 0 { 0.02 } else { -0.02 };
    // Both closes coincide; the pairing must not depend on a timing mismatch.
    for coin in [0, 1] {
        ma.observar_maduracion(coin, tau, SCALE, ts, r);
    }
}

fn feed_both(subject: &mut EspectralMultiactivo, control: &mut EspectralMultiactivo, ordinal: u64) {
    pair(subject, ordinal);
    pair(control, ordinal);
}

fn assert_healthy(ma: &EspectralMultiactivo, samples: u64) {
    assert_eq!(ma.muestras_par(0, 1, SCALE), Some(samples));
    for ic in [
        ma.coherencia_par(0, 1, SCALE),
        ma.coherencia_par_veto(0, 1, SCALE),
        ma.coherencia_media_con_todas(0, SCALE),
    ] {
        assert!(
            ic.is_some_and(|value| value.is_finite() && value > 0.99),
            "positive control must carry finite, admitted evidence; got {ic:?}"
        );
    }
}

fn mature_twins() -> (EspectralMultiactivo, EspectralMultiactivo) {
    let mut subject = EspectralMultiactivo::new(2);
    let mut control = EspectralMultiactivo::new(2);
    for ordinal in 1..=SEED_PAIRS {
        feed_both(&mut subject, &mut control, ordinal);
    }
    assert_healthy(&subject, SEED_PAIRS);
    assert_healthy(&control, SEED_PAIRS);
    assert_eq!(observable(&subject), observable(&control));
    (subject, control)
}

fn invalid_olvido() -> EspectralMultiactivoParams {
    EspectralMultiactivoParams {
        olvido: f64::NAN,
        ..EspectralMultiactivoParams::default()
    }
}

#[test]
fn invalid_olvido_update_preserves_replay_and_the_next_fresh_pair() {
    let (mut subject, mut control) = mature_twins();
    subject.update_params(invalid_olvido());
    let before = observable(&control);
    for ordinal in [SEED_PAIRS, SEED_PAIRS - 1] {
        for (coin, r) in [(0, 0.8), (1, -0.6)] {
            for ma in [&mut subject, &mut control] {
                ma.observar_maduracion(coin, SPECTRUM_SCALES_MS[SCALE], SCALE, close_time(ordinal), r);
            }
            assert_eq!(observable(&subject), before, "old/equal close must not mutate behavior");
            assert_eq!(observable(&control), before, "control must ignore the same replay");
        }
    }
    feed_both(&mut subject, &mut control, SEED_PAIRS + 1);
    assert_healthy(&control, SEED_PAIRS + 1);
    assert_eq!(
        observable(&subject),
        observable(&control),
        "an invalid parameter update must not destroy evidence on the next valid pair"
    );
}

#[test]
fn restoring_valid_olvido_does_not_leave_an_absorbing_nan_state() {
    let (mut subject, mut control) = mature_twins();
    subject.update_params(invalid_olvido());
    for ordinal in (SEED_PAIRS + 1)..=(SEED_PAIRS + 4) {
        feed_both(&mut subject, &mut control, ordinal);
    }
    subject.update_params(EspectralMultiactivoParams::default());
    control.update_params(EspectralMultiactivoParams::default());
    for ordinal in (SEED_PAIRS + 5)..=(SEED_PAIRS + 16) {
        feed_both(&mut subject, &mut control, ordinal);
    }
    assert_healthy(&control, SEED_PAIRS + 16);
    assert_eq!(
        observable(&subject),
        observable(&control),
        "invalid tuning must be rejected before poisoning moments; restoring valid tuning must retain the healthy history"
    );
}

#[test]
fn invalid_constructor_olvido_cannot_poison_finite_coincident_closes() {
    let mut subject = EspectralMultiactivo::with_params(2, invalid_olvido());
    let mut control = EspectralMultiactivo::new(2);
    for ordinal in 1..=SEED_PAIRS {
        feed_both(&mut subject, &mut control, ordinal);
    }
    assert_healthy(&control, SEED_PAIRS);
    assert_eq!(
        observable(&subject),
        observable(&control),
        "constructor and hot update must both guard invalid numeric tuning"
    );
}
