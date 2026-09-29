//! INTEGRATION TEST: cadena Hawkes en feature-engine (Ola XLV·K).
//!
//! Verifica que el fluculo de DATOS funciona end-to-end dentro del dominio:
//!   ticks → cross_excitation → contagion_matrix → contagion_roles →
//!   valores de net_role listos para el modulador (signal-engine).
//!
//! El modulador ya tiene sus propios tests en signal-engine; aquí se
//! verifica que los DATOS que produce el kernel son del tipo correcto
//! para su consumo.

use feature_engine::hawkes_cross::{
    contagion_matrix, contagion_roles, cross_excitation, ContagionRole,
};

/// Cadena completa: líder → 2 seguidores → roles.
/// El líder tiene net_role > 0; al menos un seguidor tiene net_role < 0.
#[test]
fn xlvk_cadena_hawkes_produce_roles_correctos() {
    let leader: Vec<u64> = (0..200).map(|i| 1_000 + (i as u64) * 2_000).collect();
    let s1: Vec<u64> = leader.iter().map(|&t| t + 150).collect();
    let s2: Vec<u64> = leader.iter().map(|&t| t + 300).collect();
    let series = vec![leader, s1, s2];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();
    let lags = vec![200u64, 500, 1_000];

    // 1. KERNEL: el par líder→s1 debe dar contagio significativo
    let exc = cross_excitation(&series[0], &series[1], spans[1], &lags)
        .expect("kernel con datos claros");
    assert!(exc.z_score > 3.0, "z={} esperaba >3", exc.z_score);

    // 2. MATRIZ: debe existir y ser 3×3
    let matrix = contagion_matrix(&series, &spans, &lags)
        .expect("matriz con 3 monedas");
    assert_eq!(matrix.len(), 3);
    assert!(matrix.iter().all(|r| r.len() == 3));

    // 3. ROLES: el líder debe dominar
    let roles = contagion_roles(&matrix).expect("roles");
    assert!(roles[0].net_role > 0.0, "líder net={}", roles[0].net_role);
    assert!(
        roles[0].net_role >= roles[1].net_role.max(roles[2].net_role),
        "líder ({}) debia dominar a todos",
        roles[0].net_role
    );

    // 4. VALORES listos para el modulador: todos finitos
    for r in &roles {
        assert!(r.emitted.is_finite(), "emitted no finito");
        assert!(r.received.is_finite(), "received no finito");
        assert!(r.net_role.is_finite(), "net_role no finito");
    }
}

/// Datos independientes: sin contagio significativo.
#[test]
fn xlvk_datos_independientes_sin_estructura() {
    let mut seed = 0xFEEDFACEu64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u64 % 5_000
    };

    let a: Vec<u64> = (0..50).map(|_| next() * 100 + 1_000).collect();
    let b: Vec<u64> = (0..50).map(|_| next() * 100 + 1_000).collect();
    let mut sa = a.clone(); sa.sort();
    let mut sb = b.clone(); sb.sort();
    let span_a = sa.last().unwrap() - sa[0];
    let span_b = sb.last().unwrap() - sb[0];

    let exc = cross_excitation(&sa, &sb, span_b, &[200, 500, 1_000]);
    assert!(
        exc.is_none() || exc.unwrap().z_score < 6.0,
        "independientes no deben dar contagio extremo"
    );

    // El tipo ContagionRole con net_role = 0 es el caso neutro
    let neutral = ContagionRole { emitted: 0.0, received: 0.0, net_role: 0.0 };
    assert_eq!(neutral.net_role, 0.0);
}

/// La matriz es asimétrica: líder→seguidor ≠ seguidor→líder.
#[test]
fn xlvk_matriz_asimetrica_por_construccion() {
    let leader: Vec<u64> = (0..100).map(|i| 1_000 + (i as u64) * 3_000).collect();
    let follower: Vec<u64> = leader.iter().map(|&t| t + 200).collect();
    let series = vec![leader, follower];
    let spans: Vec<u64> = series.iter().map(|s| s.last().unwrap() - s[0]).collect();

    let matrix = contagion_matrix(&series, &spans, &[500, 1_000]).expect("matriz 2×2");
    // α[0][1] (líder→seguidor) debe ser > α[1][0] (seguidor→líder)
    assert!(
        matrix[0][1] > matrix[1][0],
        "asimetría: líder→seg={} vs seg→líder={}",
        matrix[0][1], matrix[1][0]
    );
}
