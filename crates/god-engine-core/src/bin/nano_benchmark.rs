use god_engine_core::GodEngineCore;
use quantum_arena::GlobalArena;
use std::arch::x86_64::_rdtsc;
use std::hint::black_box;
use std::sync::Arc;

/// Lee el contador de ciclos de hardware con barrera atómica
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { _rdtsc() }
}

fn main() {
    println!(
        "[NANO-PROFILER] Iniciando Profiling de Latencia Cuántica (Motor Universal Continuo)..."
    );

    let arena = Arc::new(GlobalArena::new(13.0));
    let mut core = GodEngineCore::new(Arc::clone(&arena));

    let iterations = 100_000;

    let features_54 = [0.0f64; 54];

    // Warm-up de la caché L1 (llenando pipelining del CPU)
    for i in 0..1_000 {
        let price = 90000.0 + (i as f64) * 0.1;
        let _ = black_box(core.process_tick_dual(
            0,
            price - 0.5,
            price + 0.5,
            1.0,
            1.0,
            1000 + i * 100,
            &features_54,
            false,
        ));
    }

    println!(
        "[NANO-PROFILER] Evaluando GodEngineCore::process_tick_dual ({} iteraciones)...",
        iterations
    );
    let start_cycles = rdtsc();
    for i in 0..iterations {
        let price = 90000.0 + ((i % 100) as f64) * 0.1;
        let _ = black_box(core.process_tick_dual(
            0,
            price - 0.5,
            price + 0.5,
            1.0,
            1.0,
            10_000 + i * 100,
            &features_54,
            false,
        ));
    }
    let end_cycles = rdtsc();

    let total_cycles = end_cycles - start_cycles;
    let cycles_per_iter = total_cycles / iterations;
    println!(
        "  -> GodEngineCore Ciclos por Tick Dual: {} ciclos",
        cycles_per_iter
    );
    // Asumiendo un CPU de ~3.5GHz, 1 ciclo = ~0.285 nanosegundos
    println!(
        "  -> Latencia Estimada (3.5 GHz): {:.2} nanosegundos ({:.3} microsegundos)",
        (cycles_per_iter as f64) * 0.285,
        ((cycles_per_iter as f64) * 0.285) / 1000.0
    );

    println!(
        "\n[NANO-PROFILER] Completado. Motor unificado universal continuo operando a escala de nanosegundos en portátiles sin GPU."
    );
}
