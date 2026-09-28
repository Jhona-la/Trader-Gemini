//! CLOSED (Ola XLIV, 2026-09-26): FMT-216 reparado — el PASO 2 del reparo RR
//! podía hundir la SL bajo el piso de fricción (banda vacía, gate rechazando
//! al mutante; testigo determinista seed=199, flaky 3/4 del test r11). La
//! normalización por anclas como ÚLTIMO escritor de la mutación + el guard
//! de piso del PASO 2 cierran el defecto: este test ahora ATERRIZA la
//! reparación afirmando que el testigo histórico pasa el gate completo.
//! Nunca invoca promote(), carga genomas activos ni escribe candidatos.
use quantum_arena::genome::SuperGenotype;

#[test]
fn fmt216_closed_seeded_roundtrip_passes_the_promotion_predicates() {
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
                panic!(
                    "FMT-216 REGRESSION seed={seed} rate={rate} gene={i} value={v} bounds={}..{}",
                    lower[i], upper[i]
                );
            }
        }
        let Some((lo, hi)) = g.tradeable_band_ms(fee) else {
            panic!(
                "FMT-216 REGRESSION seed={seed} rate={rate} missing admissible band; SL(a,b)=({},{}) fee={} floor={}",
                g.sl_horizon_curve.a, g.sl_horizon_curve.b, fee, SuperGenotype::min_viable_sl(fee)
            );
        };
        for tau in [lo, hi] {
            let tp = g.tp_horizon_curve.eval(tau);
            let sl = g.sl_horizon_curve.eval(tau);
            let required =
                sl * SuperGenotype::min_rr_for(SuperGenotype::WORST_TOLERATED_WR, fee, sl);
            if !tp.is_finite() || !sl.is_finite() || sl <= 0.0 || tp < required {
                panic!(
                    "FMT-216 REGRESSION seed={seed} rate={rate} tau={tau} tp={tp} sl={sl} required={required} difference={}",
                    tp - required
                );
            }
        }
    }
    // El testigo histórico (seed=199) atraviesa el gate completo: FMT-216
    // queda CERRADO con el par reparación+regresión.
}
