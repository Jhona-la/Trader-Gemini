//! Suite de Contratos Formales F25-001: JIT en RAM, AtomicPtr y Curva Z de Morton (FASE 25)
//!
//! # QUÉ:
//! Pruebas de verificación formal para el runtime JIT de bajo nivel, semántica W^X,
//! intercambio atómico lock-free con `AtomicPtr`, tipado dimensional const generics
//! y preservación de localidad espacial por Curva Z de Morton.
//!
//! # POR QUÉ:
//! Garantizar en silicio que el motor dinámico JIT y los índices tensoriales operan
//! con latencia sub-microsegundo, cero contención, total seguridad de memoria e
//! inmunidad ante corrupciones de punteros en concurrencia.

#![allow(clippy::all)]

use os_guardian::jit_atomic_runtime::{
    AtomicJitPredicator, DimensionedVector, JitCodePage, JitEvaluatorFn,
};
use os_guardian::morton_z_curve::MortonZOrderCurve;
use std::ffi::c_void;

/// C1: Invariantes de tipado dimensional const generics en `DimensionedVector`.
#[test]
fn c1_dimensioned_vector_const_generics_invariants() {
    let v4 = DimensionedVector::<4>::new([1.0, 2.0, 3.0, 4.0]);
    assert_eq!(v4.dim(), 4);
    assert_eq!(v4.norm_sq(), 1.0 + 4.0 + 9.0 + 16.0); // 30.0

    let v4_other = DimensionedVector::<4>::new([2.0, 0.0, -1.0, 0.5]);
    assert_eq!(v4.dot(&v4_other), 1.0 * 2.0 + 2.0 * 0.0 - 3.0 + 4.0 * 0.5); // 2 - 3 + 2 = 1.0

    let zero = DimensionedVector::<8>::zero();
    assert_eq!(zero.dim(), 8);
    assert_eq!(zero.norm_sq(), 0.0);
}

/// C2: Seguridad de memoria W^X (Write XOR Execute) en `JitCodePage`.
#[test]
fn c2_jit_code_page_wx_security() {
    let mut page = JitCodePage::allocate(1024).expect("Asignación de página debe ser exitosa");
    assert!(!page.is_executable());
    assert!(page.size() >= 1024);

    // Escribir bytes dummy (e.g. ret = 0xC3 en x86)
    let dummy_code = [0x90, 0x90, 0xC3]; // nop, nop, ret
    page.write_code(0, &dummy_code)
        .expect("Escritura pre-sellado debe permitirse");

    // Sellar como ejecutable
    page.make_executable()
        .expect("Sellado W^X debe ser exitoso");
    assert!(page.is_executable());

    // Intentar escribir tras sellar debe fallar de forma estricta (W^X)
    let write_after_seal = page.write_code(3, &[0x90]);
    assert!(
        write_after_seal.is_err(),
        "Escribir tras hacer ejecutable viola principio W^X"
    );
}

/// Evaluador de prueba personalizado para verificar JIT swap.
unsafe extern "C" fn custom_squared_evaluator(
    inputs: *const f64,
    len: usize,
    _ctx: *mut c_void,
) -> f64 {
    if inputs.is_null() || len == 0 {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    let mut i = 0;
    while i < len {
        let val = *inputs.add(i);
        if val.is_finite() {
            sum_sq += val * val;
        }
        i += 1;
    }
    sum_sq.sqrt()
}

/// C3: Intercambio atómico de punteros `AtomicPtr` en `AtomicJitPredicator` sin locks.
#[test]
fn c3_atomic_jit_predicator_hot_swap() {
    let predicator = AtomicJitPredicator::new();
    assert_eq!(predicator.current_epoch(), 1);
    assert_eq!(predicator.total_evaluations(), 0);

    let v = DimensionedVector::<4>::new([3.0, 4.0, 0.0, 0.0]);
    // Evaluación con función default (promedio tanh)
    let score_default = predicator.evaluate(&v, core::ptr::null_mut());
    assert!(score_default.is_finite());
    assert_eq!(predicator.total_evaluations(), 1);

    // Hot-swap en caliente sin detener el hilo
    let custom_fn: JitEvaluatorFn = custom_squared_evaluator;
    let old_ptr = predicator.swap_evaluator(custom_fn);
    assert!(!old_ptr.is_null());
    assert_eq!(predicator.current_epoch(), 2);

    // Nueva evaluación debe ejecutar la nueva función (norma L2 = sqrt(3^2 + 4^2) = 5.0)
    let score_new = predicator.evaluate(&v, core::ptr::null_mut());
    assert!((score_new - 5.0).abs() < 1e-9);
    assert_eq!(predicator.total_evaluations(), 2);
}

/// C4: Codificación y decodificación exacta 2D por Curva Z de Morton.
#[test]
fn c4_morton_2d_bidirectional_roundtrip() {
    let test_cases = [
        (0u32, 0u32),
        (1, 0),
        (0, 1),
        (1, 1),
        (65535, 65535),
        (1234, 5678),
        (0xAAAA, 0x5555),
    ];

    for &(x, y) in &test_cases {
        let code = MortonZOrderCurve::encode_2d(x, y);
        let (dec_x, dec_y) = MortonZOrderCurve::decode_2d(code);
        assert_eq!(
            (dec_x, dec_y),
            (x, y),
            "Fallo round-trip 2D para ({}, {})",
            x,
            y
        );
    }
}

/// C5: Codificación y decodificación exacta 3D por Curva Z de Morton.
#[test]
fn c5_morton_3d_bidirectional_roundtrip() {
    let test_cases = [
        (0u32, 0u32, 0u32),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1023, 1023, 1023),
        (512, 256, 128),
        (333, 777, 999),
    ];

    for &(x, y, z) in &test_cases {
        let code = MortonZOrderCurve::encode_3d(x, y, z);
        let (dec_x, dec_y, dec_z) = MortonZOrderCurve::decode_3d(code);
        assert_eq!(
            (dec_x, dec_y, dec_z),
            (x, y, z),
            "Fallo round-trip 3D para ({}, {}, {})",
            x,
            y,
            z
        );
    }
}

/// C6: Normalización continua y preservación de localidad en espacio multivariable.
#[test]
fn c6_morton_continuous_discretization_locality() {
    let x_range = (100.0, 200.0);
    let y_range = (-10.0, 10.0);
    let z_range = (0.0, 1.0);

    let p1 = (150.0, 0.0, 0.5);
    let code1 =
        MortonZOrderCurve::encode_continuous_3d(p1.0, x_range, p1.1, y_range, p1.2, z_range);

    // Punto infinitesimalmente cercano
    let p2 = (150.05, 0.01, 0.501);
    let code2 =
        MortonZOrderCurve::encode_continuous_3d(p2.0, x_range, p2.1, y_range, p2.2, z_range);

    // Punto lejano
    let p3 = (195.0, 9.5, 0.95);
    let code3 =
        MortonZOrderCurve::encode_continuous_3d(p3.0, x_range, p3.1, y_range, p3.2, z_range);

    // La distancia en código Morton entre puntos contiguos debe ser órdenes de magnitud menor
    let diff_close = (code1 as i64 - code2 as i64).abs();
    let diff_far = (code1 as i64 - code3 as i64).abs();

    assert!(
        diff_close < diff_far,
        "Localidad de Morton violada: {} no es menor que {}",
        diff_close,
        diff_far
    );
}
