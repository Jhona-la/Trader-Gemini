//! Open diagnostic: reproduce the public predicates of the promotion gate.
//! Never invokes promote(), loads active genomes, or writes a candidate.
use quantum_arena::genome::SuperGenotype;

#[test]
fn seeded_roundtrip_can_still_violate_a_promotion_predicate() {
    let base = SuperGenotype::new_baseline(0.0002, 0.0005);
    let lower = SuperGenotype::get_lower_bounds();
    let upper = SuperGenotype::get_upper_bounds();
    let fee = SuperGenotype::REFERENCE_ROUNDTRIP_FEE;
    // Found by an offline 0..10_000 seeded search; freeze the witness.
    for seed in [199] {
        let rate = 0.05 + (seed % 10) as f64 * 0.05;
        let mutant = base.mutate_cmaes_seeded(rate, seed);
        let g = SuperGenotype::from_vector(&mutant.to_vector());
        for (i, v) in g.to_vector().iter().enumerate() {
            if !v.is_finite() || *v < lower[i] || *v > upper[i] {
                println!(
                    "OPEN FMT-216 seed={seed} rate={rate} gene={i} value={v} bounds={}..{}",
                    lower[i], upper[i]
                );
                return;
            }
        }
        let Some((lo, hi)) = g.tradeable_band_ms(fee) else {
            println!("OPEN FMT-216 seed={seed} rate={rate} missing admissible band; SL(a,b)=({},{}) fee={} floor={} mutant_band={:?}", g.sl_horizon_curve.a, g.sl_horizon_curve.b, fee, SuperGenotype::min_viable_sl(fee), mutant.tradeable_band_ms(fee));
            return;
        };
        for tau in [lo, hi] {
            let tp = g.tp_horizon_curve.eval(tau);
            let sl = g.sl_horizon_curve.eval(tau);
            let required =
                sl * SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl);
            if !tp.is_finite() || !sl.is_finite() || sl <= 0.0 || tp < required {
                println!("OPEN FMT-216 seed={seed} rate={rate} tau={tau} tp={tp} sl={sl} required={required} difference={}", tp - required);
                return;
            }
        }
    }
    panic!("The recorded witness no longer violates the gate; re-evaluate this open diagnostic");
}
