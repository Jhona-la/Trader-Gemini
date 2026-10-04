//! Publication of the current evidence consumed by the group risk gate.
//!
//! Absence stays in the coin key: deleting it would expose the global
//! fallback, and the registry normalizes nonfinite values to zero. Each
//! consumer therefore reads one finite value, not a separate validity flag.
//! This is per-key publication in this registry, not a multi-key/order
//! snapshot. Cores sharing a registry also share these coin keys.
use omniscient_registry::OmniscientRegistry;
use quantum_arena::{
    espectral_multiactivo::EspectralMultiactivo, temporal_spectrum::SPECTRUM_SCALES_MS,
};

fn publicar_finito(
    registry: &OmniscientRegistry,
    coin_id: usize,
    key: &str,
    evidence: Option<f64>,
    absent: f64,
) {
    registry.set_for_coin(
        coin_id,
        key,
        evidence.filter(|v| v.is_finite()).unwrap_or(absent),
    );
}

/// R=0 withdraws the bound in the existing reader, even with a positive
/// global fallback. The zero margin is diagnostic absence, not a measured
/// zero-loss bound. The estimator/reader retain their existing R>0 policy.
pub fn publicar_lundberg(registry: &OmniscientRegistry, coin_id: usize, r: Option<f64>) {
    match r.filter(|v| v.is_finite()) {
        Some(r) => {
            // Keep the finite estimate intact; publish its diagnostic first.
            publicar_finito(
                registry,
                coin_id,
                "lundberg_margen_5pct",
                Some(20.0_f64.ln() / r),
                0.0,
            );
            publicar_finito(registry, coin_id, "lundberg_r_nocional", Some(r), 0.0);
        }
        None => {
            // Withdraw the only field used by risk before its diagnostic.
            // The two stores still do not form an atomic snapshot.
            publicar_finito(registry, coin_id, "lundberg_r_nocional", None, 0.0);
            publicar_finito(registry, coin_id, "lundberg_margen_5pct", None, 0.0);
        }
    }
}

/// Publish signed IC at the closest dominant scale, or -1 on cold/invalid
/// evidence. The scalar base is in [-1, 1]; this floor never tightens it,
/// including a negative base. It is indistinguishable from a valid IC=-1
/// numerically, which has the same effect in the current tightening-only gate.
pub fn publicar_coherencia(
    registry: &OmniscientRegistry,
    multiactivo: &EspectralMultiactivo,
    coin_id: usize,
    tau_dom: f64,
) {
    let ic = if tau_dom.is_finite() && tau_dom > 0.0 {
        let mut escala_dom = 0usize;
        let mut mejor_d = f64::INFINITY;
        for (k, &tau_k) in SPECTRUM_SCALES_MS.iter().enumerate() {
            let d = (tau_k - tau_dom).abs();
            if d < mejor_d {
                mejor_d = d;
                escala_dom = k;
            }
        }
        multiactivo.coherencia_media_con_todas(coin_id, escala_dom)
    } else {
        None
    };
    publicar_finito(registry, coin_id, "qo_613_rho_tau", ic, -1.0);
}
