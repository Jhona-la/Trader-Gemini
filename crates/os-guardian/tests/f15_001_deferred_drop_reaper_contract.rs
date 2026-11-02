//! Suite de Contratos Formales F15-001 — Destrucción Diferida Lock-Free (`DeferredDropReaperRing`)
//! y Despacho Polimórfico Estático Sin `dyn Trait` (`StaticSignalDispatch`).
//!
//! Verifica las 6 propiedades de silicio de la Fase 15.1:
//! - C1: Alineación física a 128 bytes de `DeferredDropReaperRing` y a 64 bytes de `StaticSignalDispatch`.
//! - C2: Cero ejecuciones de `Drop` durante `retire_nonblocking` en el camino crítico (`drop_counter == 0` hasta `reap_batch`).
//! - C3: Destrucción exacta y determinista por lotes (`reap_batch`) y al destruir el anillo (`Drop` de limpieza).
//! - C4: Comportamiento fail-closed sin fugas ni doble-drop cuando el anillo alcanza su capacidad máxima.
//! - C5: Evaluación de `StaticSignalDispatch` en cascada y en ensamble paralelo sin `dyn Trait`.
//! - C6: Inmunidad frente a entradas `NaN` / `±Inf` en todos los variantes de `StaticSignalKernel`.

use os_guardian::{DeferredDropReaperRing, StaticSignalDispatch, StaticSignalKernel};
use std::mem::align_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct DropProbe {
    id: u64,
    counter: Arc<AtomicUsize>,
}

impl Drop for DropProbe {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn c1_alignment_invariants_128b_and_64b() {
    assert_eq!(align_of::<DeferredDropReaperRing<u64, 16>>(), 128);
    assert_eq!(align_of::<StaticSignalDispatch<8>>(), 64);
}

#[test]
fn c2_zero_drops_on_hot_path_retire_until_reaper_runs() {
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let ring: DeferredDropReaperRing<DropProbe, 8> = DeferredDropReaperRing::new();

    for i in 0..5 {
        let probe = DropProbe {
            id: i,
            counter: Arc::clone(&drop_counter),
        };
        let res = ring.retire_nonblocking(probe);
        assert!(res.is_ok());
    }

    // Crítico: Ni un solo destructor debe haberse ejecutado en el hilo productor
    assert_eq!(
        drop_counter.load(Ordering::SeqCst),
        0,
        "Hot path must never execute Drop::drop during retire_nonblocking"
    );
    assert_eq!(ring.pending_count(), 5);

    // El sepulturero drena primero 3 objetos
    let rep1 = ring.reap_batch(3);
    assert_eq!(rep1.reaped_count, 3);
    assert_eq!(rep1.pending_in_ring, 2);
    assert_eq!(drop_counter.load(Ordering::SeqCst), 3);

    // El sepulturero drena los 2 restantes
    let rep2 = ring.reap_all();
    assert_eq!(rep2.reaped_count, 2);
    assert_eq!(rep2.pending_in_ring, 0);
    assert_eq!(rep2.total_retired, 5);
    assert_eq!(rep2.total_reaped, 5);
    assert_eq!(drop_counter.load(Ordering::SeqCst), 5);
}

#[test]
fn c3_ring_drop_automatically_reaps_remaining_items_without_leak() {
    let drop_counter = Arc::new(AtomicUsize::new(0));
    {
        let ring: DeferredDropReaperRing<DropProbe, 8> = DeferredDropReaperRing::new();
        for i in 0..4 {
            let _ = ring.retire_nonblocking(DropProbe {
                id: i + 100,
                counter: Arc::clone(&drop_counter),
            });
        }
        assert_eq!(drop_counter.load(Ordering::SeqCst), 0);
    }
    // Al salir de ámbito el anillo, los 4 elementos deben haberse destruido exactamente una vez
    assert_eq!(drop_counter.load(Ordering::SeqCst), 4);
}

#[test]
fn c4_full_ring_returns_ownership_cleanly_without_double_drop() {
    let drop_counter = Arc::new(AtomicUsize::new(0));
    let ring: DeferredDropReaperRing<DropProbe, 4> = DeferredDropReaperRing::new();

    for i in 0..4 {
        assert!(ring
            .retire_nonblocking(DropProbe {
                id: i,
                counter: Arc::clone(&drop_counter),
            })
            .is_ok());
    }

    // 5to elemento debe rebotar devolviendo Err(item)
    let overflow = ring.retire_nonblocking(DropProbe {
        id: 999,
        counter: Arc::clone(&drop_counter),
    });
    assert!(overflow.is_err());
    if let Err(returned_probe) = overflow {
        assert_eq!(returned_probe.id, 999);
        drop(returned_probe);
    }
    assert_eq!(drop_counter.load(Ordering::SeqCst), 1);

    let rep = ring.reap_all();
    assert_eq!(rep.reaped_count, 4);
    assert_eq!(drop_counter.load(Ordering::SeqCst), 5);
}

#[test]
fn c5_static_signal_dispatch_cascade_and_parallel_ensemble() {
    let mut dispatch: StaticSignalDispatch<4> = StaticSignalDispatch::new();
    assert!(dispatch.push_stage(
        StaticSignalKernel::AffineClamp {
            alpha: 2.0,
            beta: 0.5,
            limit: 10.0,
        },
        1.0,
    ));
    assert!(dispatch.push_stage(
        StaticSignalKernel::SoftThreshold {
            threshold: 1.0,
            gain: 0.5,
        },
        1.0,
    ));

    // Entrada x = 1.5 -> Etapa 1: 2.0*1.5 + 0.5 = 3.5 -> Etapa 2: 0.5 * (3.5 - 1.0) = 1.25
    let cascade_out = dispatch.evaluate_cascade(1.5);
    assert!((cascade_out - 1.25).abs() < 1e-14);

    // En paralelo sobre x = 1.5: 1.0 * 3.5 + 1.0 * (0.5 * (1.5 - 1.0)) = 3.5 + 0.25 = 3.75
    let parallel_out = dispatch.evaluate_parallel_ensemble(1.5);
    assert!((parallel_out - 3.75).abs() < 1e-14);
}

#[test]
fn c6_static_signal_kernels_nan_inf_immunity() {
    let mut dispatch: StaticSignalDispatch<5> = StaticSignalDispatch::new();
    dispatch.push_stage(StaticSignalKernel::Identity, 1.0);
    dispatch.push_stage(
        StaticSignalKernel::AffineClamp {
            alpha: f64::NAN,
            beta: f64::INFINITY,
            limit: -5.0,
        },
        1.0,
    );
    dispatch.push_stage(
        StaticSignalKernel::HyperbolicTanh {
            scale: f64::NAN,
            gain: 2.0,
        },
        1.0,
    );
    dispatch.push_stage(
        StaticSignalKernel::SoftThreshold {
            threshold: f64::NAN,
            gain: f64::INFINITY,
        },
        1.0,
    );
    dispatch.push_stage(StaticSignalKernel::SignedQuadratic { gain: 0.5 }, 1.0);

    for &bad in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -3.5] {
        let c = dispatch.evaluate_cascade(bad);
        let p = dispatch.evaluate_parallel_ensemble(bad);
        assert!(c.is_finite());
        assert!(p.is_finite());
    }
}
