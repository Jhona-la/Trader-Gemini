//! CONTRATO FORMAL DE SEGURIDAD FFI (Fase 2 / Ola Omega220)
//!
//! QUÉ: Certificación rigurosa de las tres exportaciones extern "C" de `backtest-engine`:
//!      - `ffi_run_unified_backtest`
//!      - `ffi_run_unified_backtest_mmap`
//!      - `ffi_run_polars_backtest_mmap`
//!
//! POR QUÉ: Los límites de ABI hacia consumidores externos no deben causar Undefined Behavior
//!          (desreferenciación de punteros nulos, desbordamiento aritmético en `len * 32`) ni
//!          fabricar capital silencioso ($13.0 USD) ante variables ausentes o corruptas.
//!
//! PARA QUÉ: Blindaje estricto contra corrupción de memoria, garantizando que todo consumidor
//!           reciba errores explícitos (0) y opere únicamente con capital declarado y finito.

use backtest_engine::*;
use quantum_arena::genome::SuperGenotype;
use std::ffi::CString;

#[test]
fn ffi_null_pointer_and_zero_len_safety() {
    let cfg = SuperGenotype::default();
    let mut out_pnl = vec![0.0f64; 10];
    let mut out_stats = vec![0.0f64; 8];
    let sym = CString::new("BTCUSDT").unwrap();

    let dummy_data = [100.0f64; 10];

    unsafe {
        // 1. len == 0
        assert_eq!(
            ffi_run_unified_backtest(
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                0,
                &cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr(),
                sym.as_ptr(),
            ),
            0,
            "len == 0 debe retornar 0 fail-closed"
        );

        // 2. volumes_ptr es null (anteriormente omitido en el check)
        assert_eq!(
            ffi_run_unified_backtest(
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                std::ptr::null(),
                10,
                &cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr(),
                sym.as_ptr(),
            ),
            0,
            "volumes_ptr null debe retornar 0 fail-closed"
        );

        // 3. closes_ptr es null
        assert_eq!(
            ffi_run_unified_backtest(
                std::ptr::null(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                10,
                &cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr(),
                sym.as_ptr(),
            ),
            0,
            "closes_ptr null debe retornar 0"
        );

        // 4. out_stats_ptr es null
        assert_eq!(
            ffi_run_unified_backtest(
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                10,
                &cfg,
                out_pnl.as_mut_ptr(),
                std::ptr::null_mut(),
                sym.as_ptr(),
            ),
            0,
            "out_stats_ptr null debe retornar 0"
        );

        // 5. mmap con filepath nulo
        assert_eq!(
            ffi_run_unified_backtest_mmap(
                std::ptr::null(),
                10,
                &cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr(),
                sym.as_ptr(),
            ),
            0,
            "filepath_ptr null en mmap debe retornar 0"
        );

        // 6. polars mmap con config nulo
        assert_eq!(
            ffi_run_polars_backtest_mmap(
                sym.as_ptr(),
                10,
                std::ptr::null(),
                out_stats.as_mut_ptr(),
            ),
            0,
            "config null en polars mmap debe retornar 0"
        );
    }
}

#[test]
fn ffi_initial_capital_rejection_of_missing_or_corrupt_env() {
    let cfg = SuperGenotype::default();
    let mut out_pnl = vec![0.0f64; 10];
    let mut out_stats = vec![0.0f64; 8];
    let sym = CString::new("BTCUSDT").unwrap();
    let dummy_data = [100.0f64; 10];

    let invalid_cap_values = [
        "",
        "0.0",
        "-13.0",
        "NaN",
        "Infinity",
        "-Infinity",
        "text_not_a_number",
    ];

    for invalid in invalid_cap_values {
        unsafe {
            std::env::set_var("INITIAL_CAPITAL", invalid);
            let res = ffi_run_unified_backtest(
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                dummy_data.as_ptr(),
                10,
                &cfg,
                out_pnl.as_mut_ptr(),
                out_stats.as_mut_ptr(),
                sym.as_ptr(),
            );
            assert_eq!(
                res, 0,
                "INITIAL_CAPITAL='{}' debe ser rechazado sin fabricar capital ficticio",
                invalid
            );
        }
    }

    // Probar sin variable seteada
    unsafe {
        std::env::remove_var("INITIAL_CAPITAL");
        let res = ffi_run_unified_backtest(
            dummy_data.as_ptr(),
            dummy_data.as_ptr(),
            dummy_data.as_ptr(),
            dummy_data.as_ptr(),
            10,
            &cfg,
            out_pnl.as_mut_ptr(),
            out_stats.as_mut_ptr(),
            sym.as_ptr(),
        );
        assert_eq!(
            res, 0,
            "INITIAL_CAPITAL ausente debe ser rechazado fail-closed (cero fabricación de 13 USD)"
        );
    }
}

#[test]
fn ffi_checked_arithmetic_overflow_rejected() {
    let cfg = SuperGenotype::default();
    let mut out_pnl = vec![0.0f64; 10];
    let mut out_stats = vec![0.0f64; 8];
    let sym = CString::new("BTCUSDT").unwrap();
    let filepath = CString::new("dummy_non_existent.bin").unwrap();

    unsafe {
        std::env::set_var("INITIAL_CAPITAL", "1000.0");
        // len enorme que causa desbordamiento en len * 32
        let huge_len = usize::MAX / 2;
        let res1 = ffi_run_unified_backtest_mmap(
            filepath.as_ptr(),
            huge_len,
            &cfg,
            out_pnl.as_mut_ptr(),
            out_stats.as_mut_ptr(),
            sym.as_ptr(),
        );
        assert_eq!(res1, 0, "Desbordamiento en mmap debe retornar 0");

        let res2 = ffi_run_polars_backtest_mmap(
            filepath.as_ptr(),
            huge_len,
            &cfg,
            out_stats.as_mut_ptr(),
        );
        assert_eq!(res2, 0, "Desbordamiento en polars mmap debe retornar 0");
    }
}

#[test]
fn ffi_valid_execution_with_explicit_capital() {
    let cfg = SuperGenotype::default();
    let n = 20;
    let closes = vec![100.0f64; n];
    let highs = vec![101.0f64; n];
    let lows = vec![99.0f64; n];
    let volumes = vec![10.0f64; n];
    let mut out_pnl = vec![0.0f64; n];
    let mut out_stats = vec![0.0f64; 8];
    let sym = CString::new("BTCUSDT").unwrap();

    unsafe {
        std::env::set_var("INITIAL_CAPITAL", "1000.0");
        let res = ffi_run_unified_backtest(
            closes.as_ptr(),
            highs.as_ptr(),
            lows.as_ptr(),
            volumes.as_ptr(),
            n,
            &cfg,
            out_pnl.as_mut_ptr(),
            out_stats.as_mut_ptr(),
            sym.as_ptr(),
        );
        assert!(res > 0, "Ejecución FFI válida debe ejecutar al menos un trade");
        assert_eq!(
            res, out_stats[1] as usize,
            "Retorno FFI debe coincidir con el recuento de trades en out_stats[1]"
        );
        assert!(out_stats[2] > 0.0, "Terminal equity debe ser positivo");
    }
}
