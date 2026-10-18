// ============================================================================
// CONTRATO FORMAL: F1-113 Reconstrucción Vectorial Anti-Pánico y Validación Dimensional
// ============================================================================
// QUÉ: Contrato formal de validación y reconstrucción vectorial de SuperGenotype
//      vía try_from_vector y delegación segura en from_vector / from_normalized_vector.
// POR QUÉ: from_vector indexaba vec[0]..vec[143] sin verificar longitud. Un slice < 144
//          (como genomas de versiones previas 139D/140D) o con NaN/Inf causaba index
//          out of bounds panic en el demonio de evolución y en el proceso de producción.
// PARA QUÉ: Blindar el motor contra panics, garantizar que ningún genoma inválido
//           provoque aborto del proceso y certificar manejo tipado de errores GenomeError.
// CÓMO: 1. Prueba de admisión dimensional canónica (144 floats válidos -> Ok).
//       2. Prueba de rechazo estricto por dimensión insuficiente (< 144) -> DimensionMismatch.
//       3. Prueba de rechazo estricto por sobredimensión (> 144) -> DimensionMismatch.
//       4. Prueba de rechazo estricto por valores no finitos (NaN, +inf, -inf) -> NonFiniteGene.
//       5. Prueba de resiliencia: from_vector nunca entra en pánico y entrega fallback default.
//       6. Prueba de resiliencia: from_normalized_vector nunca entra en pánico ante longitud disjunta.
// CUÁNDO: Ejecutado como gate de regresión en cada build y evolución genómica.
// DÓNDE: crates/quantum-arena/tests/f1_113_genome_try_from_vector_contract.rs
// QUIÉN: Consejo Quant Sr. (Ola Ω158)
// ============================================================================

use quantum_arena::{GenomeError, SuperGenotype};

#[test]
fn f1_113_canonical_vector_admission_succeeds() {
    let baseline = SuperGenotype::default();
    let vec = baseline.to_vector();
    assert_eq!(vec.len(), SuperGenotype::DIMENSION, "to_vector debe emitir exactamente 144 dimensiones");

    let res = SuperGenotype::try_from_vector(&vec);
    assert!(res.is_ok(), "vector canónico derivado de default() debe reconstruir con Ok");

    let reconstructed = res.unwrap();
    let vec_recon = reconstructed.to_vector();
    assert_eq!(vec_recon.len(), SuperGenotype::DIMENSION);
    assert_eq!(vec, vec_recon, "simetría canónica to_vector <-> try_from_vector");
}

#[test]
fn f1_113_dimension_mismatch_short_vector_rejected_without_panic() {
    for short_len in [0, 1, 10, 50, 138, 139, 140, 143] {
        let short_vec = vec![0.5; short_len];
        let res = SuperGenotype::try_from_vector(&short_vec);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            GenomeError::DimensionMismatch {
                expected: SuperGenotype::DIMENSION,
                actual: short_len,
            },
            "vector de longitud {} debe ser rechazado con DimensionMismatch",
            short_len
        );

        // Fallback infalible de from_vector
        let safe_fallback = SuperGenotype::from_vector(&short_vec);
        assert_eq!(safe_fallback.to_vector().len(), SuperGenotype::DIMENSION);
    }
}

#[test]
fn f1_113_dimension_mismatch_long_vector_rejected_without_panic() {
    for long_len in [145, 150, 200, 500] {
        let long_vec = vec![0.5; long_len];
        let res = SuperGenotype::try_from_vector(&long_vec);
        assert!(res.is_err());
        assert_eq!(
            res.unwrap_err(),
            GenomeError::DimensionMismatch {
                expected: SuperGenotype::DIMENSION,
                actual: long_len,
            },
            "vector de longitud {} debe ser rechazado con DimensionMismatch",
            long_len
        );

        // Fallback infalible de from_vector
        let safe_fallback = SuperGenotype::from_vector(&long_vec);
        assert_eq!(safe_fallback.to_vector().len(), SuperGenotype::DIMENSION);
    }
}

#[test]
fn f1_113_non_finite_gene_rejected_with_exact_index() {
    let baseline = SuperGenotype::default();
    let mut vec = baseline.to_vector();

    // Inyectar NaN en posición 17
    vec[17] = f64::NAN;
    let res_nan = SuperGenotype::try_from_vector(&vec);
    assert!(res_nan.is_err());
    assert_eq!(
        res_nan.unwrap_err(),
        GenomeError::NonFiniteGene {
            index: 17,
            value: f64::NAN,
        }
    );
    // from_vector no debe fallar
    assert_eq!(SuperGenotype::from_vector(&vec).to_vector().len(), SuperGenotype::DIMENSION);

    // Inyectar NaN en posición 42
    vec[17] = 1.0;
    vec[42] = f64::NAN;
    let res_nan42 = SuperGenotype::try_from_vector(&vec);
    assert!(res_nan42.is_err());
    assert_eq!(
        res_nan42.unwrap_err(),
        GenomeError::NonFiniteGene {
            index: 42,
            value: f64::NAN,
        }
    );
    assert_eq!(SuperGenotype::from_vector(&vec).to_vector().len(), SuperGenotype::DIMENSION);

    // Inyectar NaN en posición 143 (último gen)
    vec[42] = 0.5;
    vec[143] = f64::NAN;
    let res_nan143 = SuperGenotype::try_from_vector(&vec);
    assert!(res_nan143.is_err());
    assert_eq!(
        res_nan143.unwrap_err(),
        GenomeError::NonFiniteGene {
            index: 143,
            value: f64::NAN,
        }
    );
    assert_eq!(SuperGenotype::from_vector(&vec).to_vector().len(), SuperGenotype::DIMENSION);

    // Infinitos son proyecciones a cotas (clamping a bounds) compatibles con test_r11
    vec[143] = 0.0;
    let lo = SuperGenotype::get_lower_bounds();
    let hi = SuperGenotype::get_upper_bounds();
    let floor_res = SuperGenotype::try_from_vector(&vec![f64::NEG_INFINITY; SuperGenotype::DIMENSION]);
    assert!(floor_res.is_ok(), "NEG_INFINITY debe ser proyectado al floor por clamp");
    assert!((floor_res.unwrap().global_max_drawdown - lo[0]).abs() < 1e-9);

    let ceil_res = SuperGenotype::try_from_vector(&vec![f64::INFINITY; SuperGenotype::DIMENSION]);
    assert!(ceil_res.is_ok(), "INFINITY debe ser proyectado al ceil por clamp");
    assert!((ceil_res.unwrap().global_max_drawdown - hi[0]).abs() < 1e-9);
}

#[test]
fn f1_113_from_normalized_vector_dimension_safety() {
    // Vector unitario válido de 144 elementos
    let norm_valid = vec![0.5; SuperGenotype::DIMENSION];
    let g_norm = SuperGenotype::from_normalized_vector(&norm_valid);
    assert_eq!(g_norm.to_vector().len(), SuperGenotype::DIMENSION);

    // Vectores con longitud incorrecta no deben provocar index out of bounds al acceder bounds
    for bad_len in [0, 5, 140, 145, 300] {
        let norm_bad = vec![0.5; bad_len];
        let fallback = SuperGenotype::from_normalized_vector(&norm_bad);
        assert_eq!(fallback.to_vector().len(), SuperGenotype::DIMENSION);
    }
}

#[test]
fn f1_113_genome_error_display_and_traits() {
    let err_dim = GenomeError::DimensionMismatch {
        expected: 144,
        actual: 139,
    };
    let s_dim = format!("{}", err_dim);
    assert!(s_dim.contains("144"));
    assert!(s_dim.contains("139"));

    let err_gene = GenomeError::NonFiniteGene {
        index: 12,
        value: f64::NAN,
    };
    let s_gene = format!("{}", err_gene);
    assert!(s_gene.contains("12"));
}
