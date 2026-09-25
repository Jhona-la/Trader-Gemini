use backtest_engine::label_evidence::*;
fn bytes(rows: &[(u64, f64)], header: &[u8]) -> Vec<u8> {
    let mut b = header.to_vec();
    for &(ts, p) in rows {
        b.extend_from_slice(&ts.to_le_bytes());
        for v in [p, p, 1.0, 2.0] {
            b.extend_from_slice(&v.to_le_bytes());
        }
    }
    b
}
fn spec() -> BarrierSpec {
    BarrierSpec {
        take_profit_return: 0.0036,
        stop_loss_return: 0.0018,
    }
}
fn labels(rows: &[(u64, f64)], h: &[u64]) -> Vec<HorizonEvidence> {
    let b = bytes(rows, b"TGMTICK1");
    label_surface(&Tape::parse(&b, false).unwrap(), 0, h, spec(), &mut 10000).unwrap()
}
#[test]
fn provenance_is_not_book_certification() {
    for (header, expected) in [
        (&b"TGMTICK1"[..], Provenance::TradeDerivedUnverifiedBook),
        (&b"TGMSYNT1"[..], Provenance::CandleSynthetic),
    ] {
        let b = bytes(&[(10, 100.)], header);
        assert_eq!(Tape::parse(&b, false).unwrap().provenance, expected);
    }
}
#[test]
fn legacy_requires_explicit_opt_in() {
    let b = bytes(&[(10, 100.)], b"");
    assert!(Tape::parse(&b, false).is_err());
    assert_eq!(
        Tape::parse(&b, true).unwrap().provenance,
        Provenance::LegacyUnknown
    );
}
#[test]
fn unknown_versions_and_truncated_records_are_rejected() {
    assert!(Tape::parse(&bytes(&[(1, 100.)], b"TGMTICK2"), true).is_err());
    let b = bytes(&[(1, 100.)], b"TGMTICK1");
    for n in [0, 7, 8, 9, 47] {
        assert!(Tape::parse(&b[..n], true).is_err());
    }
}
#[test]
fn unaligned_slice_decodes_little_endian_without_casts() {
    let mut b = vec![0xff];
    b.extend(bytes(&[(0x01020304, 123.5)], b"TGMTICK1"));
    let tape = Tape::parse(&b[1..], false).unwrap();
    let t = tape.get(0).unwrap();
    assert_eq!(t.timestamp_ms, 0x01020304);
    assert_eq!(t.mid(), 123.5);
    assert!(tape.get(usize::MAX).is_none());
    assert!(tape.get(1).is_none());
}
#[test]
fn nonfinite_fields_are_not_imputed() {
    for offset in [8, 16, 24, 32] {
        for v in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut b = bytes(&[(1, 100.)], b"TGMTICK1");
            b[8 + offset..16 + offset].copy_from_slice(&v.to_le_bytes());
            assert!(Tape::parse(&b, false).is_err());
        }
    }
}
#[test]
fn invalid_prices_depth_and_overflow_are_rejected() {
    for (offset, v) in [(8, 0.0_f64), (8, -1.), (16, 99.), (24, -1.)] {
        let mut b = bytes(&[(1, 100.)], b"TGMTICK1");
        b[8 + offset..16 + offset].copy_from_slice(&v.to_le_bytes());
        assert!(Tape::parse(&b, false).is_err());
    }
    let mut b = bytes(&[(1, 100.)], b"TGMTICK1");
    for offset in [24, 32] {
        b[8 + offset..16 + offset].copy_from_slice(&f64::MAX.to_le_bytes());
    }
    assert!(Tape::parse(&b, false).is_err());
}
#[test]
fn order_is_checked_without_sorting_or_deduplicating() {
    assert!(Tape::parse(&bytes(&[(2, 100.), (1, 101.)], b"TGMTICK1"), false).is_err());
    let b = bytes(&[(1, 100.), (1, 101.)], b"TGMTICK1");
    let tape = Tape::parse(&b, false).unwrap();
    assert_eq!(tape.len(), 2);
    assert_eq!(tape.get(1).unwrap().mid(), 101.);
}
#[test]
fn midpoint_avoids_overflow_of_bid_plus_ask() {
    let b = bytes(&[(0, f64::MAX)], b"TGMTICK1");
    assert_eq!(
        Tape::parse(&b, false).unwrap().get(0).unwrap().mid(),
        f64::MAX
    );
}
#[test]
fn long_tp_remains_reachable_after_short_stop() {
    let h = labels(&[(0, 100.), (5, 100.2), (10, 100.4)], &[5, 10]);
    assert_eq!(h[0].long.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[0].short.kind, OutcomeKind::StopLoss);
    assert_eq!(h[1].long.kind, OutcomeKind::TakeProfit);
    assert_eq!(h[1].long.event_time_ms, Some(10));
}
#[test]
fn long_stop_is_not_short_take_profit() {
    let h = labels(&[(0, 100.), (5, 99.8), (10, 99.6)], &[5, 10]);
    assert_eq!(h[0].long.kind, OutcomeKind::StopLoss);
    assert_eq!(h[0].short.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[1].short.kind, OutcomeKind::TakeProfit);
}
#[test]
fn both_sides_can_stop_so_outcomes_are_not_complements() {
    let h = labels(&[(0, 100.), (5, 100.2), (10, 99.8)], &[10]);
    assert_eq!(h[0].long.kind, OutcomeKind::StopLoss);
    assert_eq!(h[0].short.kind, OutcomeKind::StopLoss);
}
#[test]
fn time_is_clock_not_a_fixed_event_count() {
    let fast = labels(&[(0, 100.), (1, 100.), (2, 101.), (11, 102.)], &[5]);
    let slow = labels(&[(0, 100.), (6, 100.), (7, 101.), (11, 102.)], &[5]);
    assert_eq!(fast[0].long.kind, OutcomeKind::TakeProfit);
    assert_eq!(slow[0].long.kind, OutcomeKind::NoObservedHit);
}
#[test]
fn deadline_is_inclusive_and_next_price_is_excluded() {
    let h = labels(&[(0, 100.), (5, 100.), (6, 110.)], &[5, 6]);
    assert_eq!(h[0].long.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[0].last_observed_return, 0.);
    assert_eq!(h[1].long.kind, OutcomeKind::TakeProfit);
}
#[test]
fn coverage_confirmation_records_later_information_time() {
    let h = labels(&[(0, 100.), (2, 100.), (20, 110.)], &[5]);
    assert_eq!(h[0].long.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[0].information_end_ms, 20);
    assert_eq!(h[0].confirmation_time_ms, Some(20));
    assert_eq!(h[0].last_observation_ms, 2);
    assert_eq!(h[0].last_observed_return, 0.);
    assert_eq!(h[0].largest_observed_gap_ms, 18);
}
#[test]
fn right_censoring_is_not_loss_or_timeout() {
    let h = labels(&[(0, 100.), (2, 100.)], &[1, 5]);
    assert_eq!(h[0].long.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[1].long.kind, OutcomeKind::RightCensored);
    assert!(!h[1].horizon_covered);
    assert_eq!(h[1].information_end_ms, 2);
}
#[test]
fn observed_hit_survives_right_censored_horizon() {
    let h = labels(&[(0, 100.), (2, 100.2)], &[100]);
    assert_eq!(h[0].short.kind, OutcomeKind::StopLoss);
    assert_eq!(h[0].long.kind, OutcomeKind::RightCensored);
    assert_eq!(h[0].short.information_end_ms, 2);
}
#[test]
fn flat_paths_are_retained_as_observed_no_hit() {
    let h = labels(&[(0, 100.), (10, 100.)], &[1, 10]);
    assert!(
        h.iter()
            .all(|x| x.long.kind == OutcomeKind::NoObservedHit && x.last_observed_return == 0.)
    );
}
#[test]
fn equal_timestamp_events_retain_ordinal_order() {
    let h = labels(&[(0, 100.), (0, 100.4), (0, 99.6), (1, 100.)], &[1]);
    assert_eq!(h[0].long.kind, OutcomeKind::TakeProfit);
    assert_eq!(h[0].long.event_record, Some(1));
    assert_eq!(h[0].short.kind, OutcomeKind::StopLoss);
}
#[test]
fn invalid_horizons_barriers_and_anchors_fail_closed() {
    for h in [vec![], vec![0], vec![2, 1], vec![1, 1]] {
        assert!(validate_horizons(&h).is_err());
    }
    for x in [0., -1., 1., f64::NAN, f64::INFINITY] {
        assert!(
            BarrierSpec {
                take_profit_return: x,
                ..spec()
            }
            .validate()
            .is_err()
        );
        assert!(
            BarrierSpec {
                stop_loss_return: x,
                ..spec()
            }
            .validate()
            .is_err()
        );
    }
    let b = bytes(&[(0, 100.)], b"TGMTICK1");
    assert!(label_surface(&Tape::parse(&b, false).unwrap(), 1, &[1], spec(), &mut 100).is_err());
}
#[test]
fn deadline_overflow_is_rejected() {
    let b = bytes(&[(u64::MAX, 100.)], b"TGMTICK1");
    assert!(label_surface(&Tape::parse(&b, false).unwrap(), 0, &[1], spec(), &mut 100).is_err());
}
#[test]
fn work_budget_is_shared_and_enforced() {
    let b = bytes(&[(0, 100.), (1, 100.), (2, 100.)], b"TGMTICK1");
    let t = Tape::parse(&b, false).unwrap();
    let mut budget = 3;
    label_surface(&t, 0, &[2], spec(), &mut budget).unwrap();
    assert_eq!(budget, 0);
    assert!(label_surface(&t, 1, &[1], spec(), &mut budget).is_err());
}
#[test]
fn multiple_horizons_reuse_one_forward_scan() {
    let b = bytes(&[(0, 100.), (1, 100.), (2, 100.)], b"TGMTICK1");
    let mut budget = 4;
    let h = label_surface(
        &Tape::parse(&b, false).unwrap(),
        0,
        &[1, 2],
        spec(),
        &mut budget,
    )
    .unwrap();
    assert_eq!(h.len(), 2);
    assert_eq!(budget, 0);
}
#[test]
fn price_rescaling_preserves_outcomes_away_from_boundaries() {
    let a = labels(&[(0, 100.), (5, 100.2), (10, 100.4)], &[5, 10]);
    let b = labels(&[(0, 0.1), (5, 0.1002), (10, 0.1004)], &[5, 10]);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.long.kind, y.long.kind);
        assert_eq!(x.short.kind, y.short.kind);
        assert!((x.last_observed_return - y.last_observed_return).abs() < 1e-12);
    }
}
#[test]
fn clock_translation_and_dilation_preserve_corresponding_outcomes() {
    let a = labels(&[(0, 100.), (5, 100.2), (10, 100.4)], &[5, 10]);
    let b = labels(&[(100, 100.), (150, 100.2), (200, 100.4)], &[50, 100]);
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.long.kind, y.long.kind);
        assert_eq!(x.short.kind, y.short.kind);
    }
}

#[test]
fn exactly_representable_barrier_touches_are_inclusive() {
    let b = bytes(&[(0, 100.), (1, 112.5), (2, 75.)], b"TGMTICK1");
    let spec = BarrierSpec {
        take_profit_return: 0.125,
        stop_loss_return: 0.25,
    };
    let h = label_surface(&Tape::parse(&b, false).unwrap(), 0, &[1, 2], spec, &mut 100).unwrap();
    assert_eq!(h[0].long.kind, OutcomeKind::TakeProfit);
    assert_eq!(h[0].short.kind, OutcomeKind::NoObservedHit);
    assert_eq!(h[1].short.kind, OutcomeKind::TakeProfit);
}
