//! # Contrato de Verificación Formal F26-002: Aritmética SIMD Libre de Acarreo (RNS)
//!
//! ## Método Pedagógico Riguroso (Modo Profesor)
//! - **QUÉ:** Batería de 6 contratos formales de invariantes matemáticas sobre el Sistema de Numeración de Residuos (RNS).
//! - **POR QUÉ:** La eliminación de la cadena de acarreo (carry chain) en acumulaciones de alta velocidad
//!   debe ser exacta al 100%, sin desbordamientos silenciosos, preservando el isomorfismo algebraico con Z_M.
//! - **PARA QUÉ:** Demostrar formalmente que las sumas, restas y productos modulares sobre canales coprimos
//!   reconstruyen exactamente los enteros grandes vía el Teorema Chino del Resto (CRT) en tiempo O(1).
//! - **CÓMO:** Pruebas unitarias e invariantes de propiedades algebraicas ejecutadas en tiempo de compilación/test.

#![allow(clippy::all)]

use os_guardian::rns_simd_arithmetic::{RnsSimdArithmeticEngine, RnsVector4, RNS_MODULI};

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[test]
fn c1_rns_moduli_coprimality_and_dynamic_range() {
    // 1. Verificar coprimalidad dos a dos
    for (i, &m1) in RNS_MODULI.iter().enumerate() {
        for &m2 in &RNS_MODULI[(i + 1)..] {
            let g = gcd(m1, m2);
            assert_eq!(
                g, 1,
                "Los módulos RNS {} y {} deben ser coprimos, pero gcd={}",
                m1, m2, g
            );
        }
    }

    // 2. Verificar rango dinámico M > 2^80
    let engine = RnsSimdArithmeticEngine::new();
    let m = engine.dynamic_range();
    let min_expected: u128 = 1u128 << 80;
    assert!(
        m > min_expected,
        "El rango dinámico M={} debe superar 2^80={}",
        m,
        min_expected
    );
}

#[test]
fn c2_rns_carry_free_addition_exactness() {
    let engine = RnsSimdArithmeticEngine::new();

    let a: u64 = 987_654_321_012_345;
    let b: u64 = 123_456_789_987_654;
    let expected_sum = (a as u128) + (b as u128);

    let va = RnsVector4::from_u64(a);
    let vb = RnsVector4::from_u64(b);

    // Suma RNS libre de acarreo
    let vsum = va.add(&vb);

    let reconstructed = engine.reconstruct_crt(&vsum);
    assert_eq!(
        reconstructed, expected_sum,
        "Reconstrucción CRT de suma debe ser idéntica al valor esperado"
    );
}

#[test]
fn c3_rns_carry_free_subtraction_exactness() {
    let engine = RnsSimdArithmeticEngine::new();

    let a: u64 = 999_888_777_666_555;
    let b: u64 = 111_222_333_444_555;
    let expected_diff = (a as u128) - (b as u128);

    let va = RnsVector4::from_u64(a);
    let vb = RnsVector4::from_u64(b);

    // Resta RNS libre de acarreo
    let vdiff = va.sub(&vb);

    let reconstructed = engine.reconstruct_crt(&vdiff);
    assert_eq!(
        reconstructed, expected_diff,
        "Reconstrucción CRT de resta debe ser idéntica al valor esperado"
    );
}

#[test]
fn c4_rns_carry_free_multiplication_and_fma() {
    let engine = RnsSimdArithmeticEngine::new();

    let a: u64 = 12_345_678;
    let b: u64 = 87_654_321;
    let c: u64 = 999_999;
    let expected_fma = (a as u128) * (b as u128) + (c as u128);

    let va = RnsVector4::from_u64(a);
    let vb = RnsVector4::from_u64(b);
    let vc = RnsVector4::from_u64(c);

    // Fused Multiply-Add RNS
    let vfma = va.fma(&vb, &vc);

    let reconstructed = engine.reconstruct_crt(&vfma);
    assert_eq!(
        reconstructed, expected_fma,
        "Reconstrucción CRT de FMA debe ser exacta"
    );
}

#[test]
fn c5_rns_mod_inverse_properties() {
    for &m in &RNS_MODULI {
        let test_val = 123_456 % m;
        let inv = RnsSimdArithmeticEngine::mod_inverse(test_val, m);
        let prod = ((test_val as u128 * inv as u128) % m as u128) as u64;
        assert_eq!(
            prod, 1,
            "El inverso modular de {} mod {} debe satisfacer a * a^-1 = 1 mod m",
            test_val, m
        );
    }
}

#[test]
fn c6_rns_large_integer_reconstruction_crt() {
    let engine = RnsSimdArithmeticEngine::new();

    // Probar un entero de 128 bits significativamente mayor que u64::MAX
    let large_val: u128 = (u64::MAX as u128) * 5000 + 123_456_789;
    let vrns = RnsVector4::from_u128(large_val);

    let reconstructed = engine.reconstruct_crt(&vrns);
    assert_eq!(
        reconstructed, large_val,
        "Reconstrucción CRT de entero u128 grande debe coincidir bit a bit"
    );
}
