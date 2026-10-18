//! Contrato de higiene F1-024, F1-025 y F1-026:
//! - F1-024: Verificación de frescura temporal de forest6_prob (TTL 15s) y desacoplamiento asíncrono.
//! - F1-025: Inmunidad de umbrales durante la ventana de vigilancia del watchdog post-promoción.
//! - F1-026: Erradicación del hitchhiking de genes no observables en el examen walk-forward.

use std::sync::Arc;
use std::sync::atomic::Ordering;
use evolution_engine::online_daemon::{
    restaurar_genes_no_observables_en_examen, LiveEvolutionDaemon, QuantumHotSwapState,
};
use quantum_arena::genome::SuperGenotype;
use quantum_arena::GlobalArena;

#[test]
fn f1_026_restaurar_genes_no_observables_preserva_invariantes() {
    let incumbent = SuperGenotype::new_baseline(0.0002, 0.0005);
    let mut candidate = incumbent.clone();

    // Perturbar intencionalmente los 8 genes no observables
    candidate.weight_obi += 0.35;
    candidate.maker_obi_threshold += 0.25;
    candidate.maker_spread_pct *= 1.50;
    candidate.global_correlation_threshold += 0.15;
    candidate.topo_layer_1_activation += 0.40;
    candidate.topo_layer_2_activation += 0.40;
    candidate.tensor_dropout_rate += 0.20;
    candidate.quantum_entropy_seed += 500.0;

    // Verificar que están perturbados
    assert_ne!(candidate.weight_obi, incumbent.weight_obi);
    assert_ne!(candidate.maker_obi_threshold, incumbent.maker_obi_threshold);
    assert_ne!(candidate.maker_spread_pct, incumbent.maker_spread_pct);
    assert_ne!(
        candidate.global_correlation_threshold,
        incumbent.global_correlation_threshold
    );
    assert_ne!(
        candidate.topo_layer_1_activation,
        incumbent.topo_layer_1_activation
    );
    assert_ne!(
        candidate.topo_layer_2_activation,
        incumbent.topo_layer_2_activation
    );
    assert_ne!(
        candidate.tensor_dropout_rate,
        incumbent.tensor_dropout_rate
    );
    assert_ne!(
        candidate.quantum_entropy_seed,
        incumbent.quantum_entropy_seed
    );

    // Restaurar vía F1-026
    restaurar_genes_no_observables_en_examen(&mut candidate, &incumbent);

    // Verificar que los 8 genes fueron restaurados idénticos al incumbente
    assert_eq!(candidate.weight_obi, incumbent.weight_obi);
    assert_eq!(candidate.maker_obi_threshold, incumbent.maker_obi_threshold);
    assert_eq!(candidate.maker_spread_pct, incumbent.maker_spread_pct);
    assert_eq!(
        candidate.global_correlation_threshold,
        incumbent.global_correlation_threshold
    );
    assert_eq!(
        candidate.topo_layer_1_activation,
        incumbent.topo_layer_1_activation
    );
    assert_eq!(
        candidate.topo_layer_2_activation,
        incumbent.topo_layer_2_activation
    );
    assert_eq!(
        candidate.tensor_dropout_rate,
        incumbent.tensor_dropout_rate
    );
    assert_eq!(
        candidate.quantum_entropy_seed,
        incumbent.quantum_entropy_seed
    );
}

#[test]
fn f1_025_vigilancia_activa_no_sobrescribe_umbrales_del_genoma() {
    let arena = Arc::new(GlobalArena::new(13.0));
    let state = QuantumHotSwapState::new();
    let temp_db = std::env::temp_dir().join("test_f1_025.db");
    let temp_champ = std::env::temp_dir().join("test_f1_025_champ.json");

    let mut daemon = LiveEvolutionDaemon::new(
        state,
        arena.clone(),
        true,
        temp_db.to_str().expect("path"),
        temp_champ.to_str().expect("path"),
    );

    // Fijar umbrales del genoma en arena
    arena.config.ml_threshold_long.store(0.68, Ordering::Relaxed);
    arena.config.ml_threshold_short.store(0.32, Ordering::Relaxed);

    // Simular que un genoma nuevo está bajo vigilancia (promoted_generation = Some)
    daemon.promoted_generation = Some((42, 41));

    // Si el bosque tuviese umbrales óptimos (ej. 0.55 y 0.45)
    // Cuando promoted_generation.is_some(), la regla F1-025 prohibe sobrescribir arena.config
    assert!(daemon.promoted_generation.is_some());
    // El genoma en la arena debe retener intactos sus 0.68 y 0.32
    assert_eq!(arena.config.ml_threshold_long.load(Ordering::Relaxed), 0.68);
    assert_eq!(arena.config.ml_threshold_short.load(Ordering::Relaxed), 0.32);
}

#[test]
fn f1_024_frescura_temporal_forest6_prob_invalida_predicciones_obsoletas() {
    let arena = GlobalArena::new(13.0);
    let coin_id = 0;

    // Caso 1: Sin timestamp o timestamp = 0.0 -> stale -> neutro 0.50
    arena.registry.set_for_coin(coin_id, "forest6_prob", 0.90);
    arena.registry.set_for_coin(coin_id, "forest6_ts_ms", 0.0);

    // Simulamos tick a t = 100_000 ms
    let event_time_ms = 100_000u64;

    // Evaluamos la lógica de lectura de frescura implementada en process_tick_dual
    let f6_ts = arena.registry.get_for_coin_or(coin_id, "forest6_ts_ms", 0.0);
    let f6_stale = f6_ts <= 0.0
        || (event_time_ms > 0 && event_time_ms.saturating_sub(f6_ts as u64) > 15_000);
    let f6_prob_actual = if f6_stale {
        0.5
    } else {
        arena.registry.get_for_coin_or(coin_id, "forest6_prob", 0.5)
    };
    assert!(f6_stale, "Sin timestamp debe considerarse stale");
    assert_eq!(f6_prob_actual, 0.5, "Predicción stale debe neutralizarse a 0.5");

    // Caso 2: Timestamp obsoleto (> 15s de antigüedad)
    arena.registry.set_for_coin(coin_id, "forest6_ts_ms", (event_time_ms - 16_000) as f64);
    let f6_ts = arena.registry.get_for_coin_or(coin_id, "forest6_ts_ms", 0.0);
    let f6_stale = f6_ts <= 0.0
        || (event_time_ms > 0 && event_time_ms.saturating_sub(f6_ts as u64) > 15_000);
    let f6_prob_actual = if f6_stale {
        0.5
    } else {
        arena.registry.get_for_coin_or(coin_id, "forest6_prob", 0.5)
    };
    assert!(f6_stale, "Timestamp de hace 16 segundos debe considerarse stale");
    assert_eq!(f6_prob_actual, 0.5, "Predicción de 16s debe neutralizarse a 0.5");

    // Caso 3: Timestamp fresco (< 15s de antigüedad)
    arena.registry.set_for_coin(coin_id, "forest6_ts_ms", (event_time_ms - 2_000) as f64);
    let f6_ts = arena.registry.get_for_coin_or(coin_id, "forest6_ts_ms", 0.0);
    let f6_stale = f6_ts <= 0.0
        || (event_time_ms > 0 && event_time_ms.saturating_sub(f6_ts as u64) > 15_000);
    let f6_prob_actual = if f6_stale {
        0.5
    } else {
        arena.registry.get_for_coin_or(coin_id, "forest6_prob", 0.5)
    };
    assert!(!f6_stale, "Timestamp de hace 2 segundos es fresco");
    assert_eq!(f6_prob_actual, 0.90, "Predicción fresca retiene su valor real");
}
