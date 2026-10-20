//! Contrato Formal de Verificación F1-076:
//! Caché Zero-Alloc de Pesos Espectrales y Observabilidad y Erradicación de Redundancia en el Hot-Path.
//!
//! # QUÉ:
//! Caché canónica O(1) de observables, masas y pesos de observabilidad en `TemporalSpectrum`
//! calculados una única vez por evento en `refresh_fusion()`, y eliminación de la duplicación
//! de publicación espectral entre `process_event` y `process_tick_dual`.
//!
//! # POR QUÉ:
//! Anteriormente, `pesos_observables` ejecutaba 64 llamadas a `exp()` y 32 divisiones en cada lector
//! (`pesos_espectrales`, `spectral_field`, `spectral_coherence`, `energy_at`, `structure_functions`,
//! `fisher_scale_information`, etc.). Al ser invocados ~30 veces por evento por moneda y repetidos
//! en `process_tick_dual` cuando venía de `process_event`, se ejecutaban más de 1,900 llamadas a `exp()`
//! redundantes por tick (~40 µs/evento), superando por sí solo el presupuesto de latencia de 25 µs.
//!
//! # PARA QUÉ:
//! Optimizar el tiempo de ciclo a nivel de nanosegundos protegiendo los recursos finitos de la máquina
//! (16 GB RAM, 0 GPU), garantizando que el bucle de eventos opere en ultra-baja latencia sin alterar
//! ningún valor matemático ni violar el determinismo continuo.
//!
//! # CÓMO:
//! 1. `TemporalSpectrum` almacena `cached_observables`, `cached_pesos_obs`, `cached_pesos_esp`,
//!    `cached_mass` y `cached_resolucion_ms`, actualizados exclusivamente en `refresh_fusion()`.
//! 2. `pesos_espectrales()`, `pesos_observables()`, `resolucion_efectiva_ms()`, `spectral_coherence()`
//!    y `structure_functions()` consumen la caché en O(1) sin invocar `exp()`.
//! 3. `GodEngineCore` añade `spectral_published_in_current_event` para evitar la re-ejecución
//!    idéntica del bloque de publicación espectral en `process_tick_dual` cuando ya corrió en `process_event`.
//!
//! # DÓNDE:
//! `crates/quantum-arena/src/temporal_spectrum.rs` y `crates/god-engine-core/src/lib.rs`.
//!
//! # QUIÉN:
//! Consejo de 10 Roles Senior / Quant Sr. — Ola Ω180 (Cierre F1-076).

use quantum_arena::temporal_spectrum::{
    TemporalSpectrum, FEED_CLOCK_RESOLUTION_MS, SPECTRUM_SCALES_MS,
};

/// Contrato 1: Los pesos y observables cacheados coinciden analíticamente con la definición teórica.
#[test]
fn f1_076_contrato_1_pesos_espectrales_y_observables_identicos_a_formula_analitica() {
    let mut spec = TemporalSpectrum::new();
    let first_ts = 1_000_000u64;
    let last_ts = 1_060_000u64; // 60 segundos transcurridos
    let updates = 600u64; // dt_medio = 100 ms

    spec.set_test_time_window(first_ts, last_ts, updates);

    // Asignar señales y dev_vol a escalas operativas
    spec.scales[18].ewma_dev_vol = 0.005;
    spec.scales[18].signal = 0.75;
    spec.scales[18].persistence = 0.40;
    spec.scales[18].epigenetic_gain = 1.20;

    spec.refresh_fusion();

    let elapsed = (last_ts - first_ts) as f64;
    let dt_medio = elapsed / (updates - 1) as f64;
    let expected_res = FEED_CLOCK_RESOLUTION_MS.max(dt_medio);

    assert_eq!(spec.resolucion_efectiva_ms(), expected_res);

    let tau_18 = SPECTRUM_SCALES_MS[18];
    let expected_mass_18 = 1.0 - (-elapsed / tau_18).exp();
    let expected_res_factor_18 = 1.0 - (-tau_18 / expected_res).exp();
    let expected_obs_18 = expected_mass_18 * expected_res_factor_18;
    let expected_peso_obs_18 = expected_obs_18 / 0.005;
    let expected_fusion_w_18 = (0.40f64.abs() * 2.0 * 1.20).clamp(0.02, 3.0);
    let expected_peso_esp_18 = expected_fusion_w_18 * expected_obs_18;

    let pesos_esp = spec.pesos_espectrales();
    let observables = spec.observables_ref();
    let (_, pesos_obs) = spec.pesos_observables();

    assert!(
        (observables[18] - expected_obs_18).abs() < 1e-12,
        "Observable en caché ({}) no coincide con fórmula analítica ({})",
        observables[18],
        expected_obs_18
    );
    assert!(
        (pesos_obs[18] - expected_peso_obs_18).abs() < 1e-12,
        "Peso observable en caché ({}) no coincide con fórmula analítica ({})",
        pesos_obs[18],
        expected_peso_obs_18
    );
    assert!(
        (pesos_esp[18] - expected_peso_esp_18).abs() < 1e-12,
        "Peso espectral en caché ({}) no coincide con fórmula analítica ({})",
        pesos_esp[18],
        expected_peso_esp_18
    );
}

/// Contrato 2: Múltiples lecturas consecutivas sin actualización son O(1), no mutan el estado y son bit a bit idénticas.
#[test]
fn f1_076_contrato_2_cero_drift_tras_multiples_lecturas_consecutivas() {
    let mut spec = TemporalSpectrum::new();
    spec.set_test_time_window(10_000, 100_000, 500);

    for i in 0..32 {
        spec.scales[i].ewma_dev_vol = 0.01;
        spec.scales[i].signal = (i as f64 * 0.1).sin();
        spec.scales[i].persistence = 0.25;
    }
    spec.refresh_fusion();

    let field_inicial = spec.spectral_field(true);
    let coherencia_inicial = spec.spectral_coherence(true);
    let pesos_inicial = spec.pesos_espectrales();
    let tau_inicial = spec.tau_operativa_unificada();
    let sf_inicial = spec.structure_functions();

    for _ in 0..100 {
        let field_actual = spec.spectral_field(true);
        let coherencia_actual = spec.spectral_coherence(true);
        let pesos_actual = spec.pesos_espectrales();
        let tau_actual = spec.tau_operativa_unificada();
        let sf_actual = spec.structure_functions();

        assert_eq!(field_actual.total_energy, field_inicial.total_energy);
        assert_eq!(field_actual.resonant_tau_ms, field_inicial.resonant_tau_ms);
        assert_eq!(coherencia_actual, coherencia_inicial);
        assert_eq!(pesos_actual, pesos_inicial);
        assert_eq!(tau_actual, tau_inicial);
        assert_eq!(sf_actual, sf_inicial);
    }
}

/// Contrato 3: La llamada a `update()` refresca automáticamente la caché ante nuevos datos de mercado.
#[test]
fn f1_076_contrato_3_invalidez_y_actualizacion_automatica_en_update() {
    let mut spec = TemporalSpectrum::new();
    spec.update(100.0, 1_000);
    spec.update(101.0, 2_000);

    let res_1 = spec.resolucion_efectiva_ms();
    let obs_1 = *spec.observables_ref();

    // Actualizar con un salto temporal que cambie la resolución media y la masa
    for i in 1..=50 {
        let p = 101.0 + (i as f64) * 0.1;
        let ts = 2_000 + i * 50; // cadencia rápida de 50 ms
        spec.update(p, ts);
    }

    let res_2 = spec.resolucion_efectiva_ms();
    let obs_2 = *spec.observables_ref();

    assert!(
        res_2 < res_1,
        "La resolución efectiva precomputada debe reducirse ante cadencia rápida ({res_2} vs {res_1})"
    );
    assert_ne!(
        obs_1, obs_2,
        "Los observables deben actualizarse inmediatamente en update()"
    );
}

/// Contrato 4: `structure_functions` consume la masa en caché sin `exp()` y produce valores consistentes.
#[test]
fn f1_076_contrato_4_structure_functions_usa_masa_en_cache_y_conserva_valores() {
    let mut spec = TemporalSpectrum::new();
    let mut p = 100.0;
    let mut ts = 10_000u64;

    // Simular un camino browniano para que maduren múltiples escalas
    let mut rng = 0x5EED_0076_u64;
    for _ in 0..2_000 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let z = ((rng >> 32) as f64 / 4294967296.0) - 0.5;
        p *= 1.0 + z * 0.002;
        ts += 100;
        spec.update(p, ts);
    }

    let sf = spec.structure_functions();
    assert!(sf.is_some(), "Con 2000 ticks deben existir escalas maduras suficientes");
    let sf = sf.unwrap();

    assert!(sf.usable_scales >= 4, "Debe haber al menos 4 escalas usables");
    assert!(sf.zeta2.is_finite(), "zeta2 debe ser finita");
    assert!(sf.zeta3.is_finite(), "zeta3 debe ser finita");
    assert!(
        sf.intermittency >= 0.0 && sf.intermittency <= 1.0,
        "intermittency ({}) debe estar acotada en [0, 1]",
        sf.intermittency
    );
}

/// Contrato 5: Inmunidad IEEE-754 y estado en frío.
#[test]
fn f1_076_contrato_5_inmunidad_ieee754_y_estado_en_frio() {
    let spec = TemporalSpectrum::new();

    assert_eq!(spec.resolucion_efectiva_ms(), FEED_CLOCK_RESOLUTION_MS);

    let pesos = spec.pesos_espectrales();
    let observables = spec.observables_ref();

    for i in 0..32 {
        assert_eq!(pesos[i], 0.0, "En frío el peso debe ser 0.0");
        assert_eq!(observables[i], 0.0, "En frío el observable debe ser 0.0");
    }

    let coherence = spec.spectral_coherence(true);
    assert_eq!(coherence, 0.0, "En frío la coherencia debe ser 0.0");

    let sf = spec.structure_functions();
    assert_eq!(sf, None, "En frío structure_functions debe ser None");
}
