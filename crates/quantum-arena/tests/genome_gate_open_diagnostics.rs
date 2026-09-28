//! FMT-216 — reproduce the public predicates of the promotion gate.
//! Never invokes promote(), loads active genomes, or writes a candidate.
//!
//! This file was an OPEN diagnostic: seed 199 (rate 0.5) produced a mutant
//! whose SL curve lay entirely below the friction floor, and the test panicked
//! the day the witness stopped violating the gate. R1.1b (2026-09-28) closed it
//! in the generator, as FMT-216 prescribed — without relaxing the gate: the
//! curve repair (`enforce_curve_rr`) now treats an empty tradeable band as a
//! violation and runs its deterministic fallback. The witness is kept frozen
//! as a regression: it must now satisfy every predicate of the gate.
use quantum_arena::genome::SuperGenotype;

#[test]
fn fmt216_seeded_witness_now_satisfies_promotion_predicates() {
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let lower = SuperGenotype::get_lower_bounds();
    let upper = SuperGenotype::get_upper_bounds();
    let fee = SuperGenotype::REFERENCE_ROUNDTRIP_FEE;
    // Found by an offline 0..10_000 seeded search; frozen witness.
    for seed in [199] {
        let rate = 0.05 + (seed % 10) as f64 * 0.05;
        let mutant = base.mutate_cmaes_seeded(rate, seed);
        let g = SuperGenotype::from_vector(&mutant.to_vector());
        for (i, v) in g.to_vector().iter().enumerate() {
            assert!(
                v.is_finite() && *v >= lower[i] && *v <= upper[i],
                "FMT-216 seed={seed} rate={rate} gene={i} value={v} bounds={}..{}",
                lower[i],
                upper[i]
            );
        }
        let (lo, hi) = g.tradeable_band_ms(fee).unwrap_or_else(|| {
            panic!(
                "FMT-216 seed={seed} rate={rate} missing admissible band; SL(a,b)=({},{}) fee={} floor={}",
                g.sl_horizon_curve.a,
                g.sl_horizon_curve.b,
                fee,
                SuperGenotype::min_viable_sl(fee)
            )
        });
        for tau in [lo, hi] {
            let tp = g.tp_horizon_curve.eval(tau);
            let sl = g.sl_horizon_curve.eval(tau);
            let required =
                sl * SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl);
            assert!(
                tp.is_finite() && sl.is_finite() && sl > 0.0 && tp >= required,
                "FMT-216 seed={seed} rate={rate} tau={tau} tp={tp} sl={sl} required={required}"
            );
        }
    }
}
