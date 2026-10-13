//! C-F3.7 (Línea C), F3.7-08: un evento de libro estacionario no debe
//! asignar memoria de heap.
//!
//! Teoría que se prueba: en régimen estacionario (estado caliente, mismo
//! símbolo, mismas claves), procesar un evento de libro sólo actualiza
//! estado de tamaño fijo. Cualquier asignación por evento es trabajo de
//! montículo en la ruta caliente: nombres de símbolo y claves de texto
//! construidos en cada evento (`try_symbol`, `try_spec`, `format!`), clones
//! de Vec. El perfil del camino caliente (CL-63) midió 11,0 por evento de
//! libro; el medidor CL-60, 19,4 por evento mezclado. Contrato: menos de una
//! asignación por evento de libro tras el calentamiento.
//!
//! Medido sobre main 8f572f7c (tras las búsquedas sin asignación de AGY y
//! los cambios CL-6x): queda 1,00 asignación por evento de libro, de 7 bytes
//! = el nombre «BTCUSDT» que `try_symbol(coin_id).unwrap_or_default()`
//! construye como `String` al empezar `process_tick_dual` (identidad LF-1).
//! El resto de la lista del hallazgo (format!, clones de Vec, try_spec) ya
//! no corre en el evento de libro o ya no asigna. El test imprime los
//! tamaños de las primeras asignaciones para quien lo arregle.
//!
//! El asignador cuenta toda petición nueva (`alloc`, `alloc_zeroed`,
//! `realloc`) de cualquier hilo mientras está activo, como el medidor CL-60:
//! en windows-gnu un `thread_local!` puede reservar memoria la primera vez que
//! se toca, y hacerlo desde el asignador sería reentrante. El ruido de otros
//! hilos se mide aparte con la misma duración y sin eventos, y se imprime; la
//! aserción usa el total, que es una cota superior de lo que asigna el
//! evento.
//!
//! Convención: `#[ignore = "OPEN F3.x-nn (dueño): causa"]` con el assert
//! intacto mientras el hallazgo siga abierto.

use god_engine_core::GodEngineCore;
use quantum_arena::{symbol_registry, symbols, GlobalArena};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

struct Contador;
static ACTIVO: AtomicBool = AtomicBool::new(false);
static N: AtomicU64 = AtomicU64::new(0);
/// Tamaños de las primeras asignaciones contadas: diagnóstico para quien
/// arregle (dice qué se construye, sin reentrar en el asignador).
const MUESTRAS: usize = 16;
static TAMANOS: [AtomicUsize; MUESTRAS] = [const { AtomicUsize::new(0) }; MUESTRAS];

#[inline]
fn contar(l: &Layout) {
    if ACTIVO.load(Ordering::Relaxed) {
        let i = N.fetch_add(1, Ordering::Relaxed) as usize;
        if i < MUESTRAS {
            TAMANOS[i].store(l.size(), Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for Contador {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        contar(&l);
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        contar(&l);
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        contar(&l);
        unsafe { System.realloc(p, l, n) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ASIGNADOR: Contador = Contador;

/// Un evento de libro de la moneda 0 con mid 50 000 ± 1 alterno, spread
/// 1 USD y cantidades 1/1.
fn libro(core: &mut GodEngineCore, k: u64, ts: u64) {
    let mid = 50_000.0 + if k % 2 == 0 { 1.0 } else { -1.0 };
    let _ = core.process_event(
        0,
        false,
        false,
        true,
        mid,
        0.0,
        mid - 0.5,
        mid + 0.5,
        1.0,
        1.0,
        0.0,
        0.0,
        ts,
        false,
        &[0.0; 54],
        false,
    );
}

/// 2 000 eventos de libro de calentamiento (1 s entre eventos) y 1 000
/// medidos. Esperado: total < 1 000.
#[test]
#[ignore = "OPEN F3.7-08 (Línea C propone; cada dueño de archivo decide): cada evento de libro construye el nombre del símbolo como String (try_symbol en process_tick_dual): 1,00 asignaciones por evento"]
fn cf3_evento_de_libro_estacionario_no_asigna() {
    symbols::update_dynamic_universe(vec!["BTCUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec("BTCUSDT")]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena.clone());
    core.swing_nn = None;
    core.scalp_forest = None;
    let mut ts = 1_000u64;
    for k in 0..2_000u64 {
        libro(&mut core, k, ts);
        ts += 1_000;
    }
    N.store(0, Ordering::SeqCst);
    ACTIVO.store(true, Ordering::SeqCst);
    let t0 = Instant::now();
    for k in 0..1_000u64 {
        libro(&mut core, k, ts);
        ts += 1_000;
    }
    let duracion = t0.elapsed();
    ACTIVO.store(false, Ordering::SeqCst);
    let total = N.load(Ordering::SeqCst);
    let tamanos: Vec<usize> = TAMANOS
        .iter()
        .take((total as usize).min(MUESTRAS))
        .map(|t| t.load(Ordering::SeqCst))
        .collect();

    // Ruido de fondo: la misma duración sin eventos.
    N.store(0, Ordering::SeqCst);
    ACTIVO.store(true, Ordering::SeqCst);
    std::thread::sleep(duracion);
    ACTIVO.store(false, Ordering::SeqCst);
    let fondo = N.load(Ordering::SeqCst);

    println!(
        "observado: {total} asignaciones en 1 000 eventos de libro ({:.2} por evento) en {duracion:?}; \
         fondo sin eventos en la misma duración: {fondo}; tamaños (bytes) de las primeras: {tamanos:?}",
        total as f64 / 1_000.0
    );
    assert!(
        total < 1_000,
        "asignaciones en 1 000 eventos de libro: {total}"
    );
}
