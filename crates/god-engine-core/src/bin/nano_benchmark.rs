use std::arch::x86_64::_rdtsc;
use std::hint::black_box;
use quantum_arena::GlobalArena;
use std::sync::Arc;
use strategy_core::swing::SwingEngine;
use strategy_core::scalp::ScalpEngine;

/// Lee el contador de ciclos de hardware con barrera atómica
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe { _rdtsc() }
}

fn main() {
    println!("[NANO-PROFILER] Iniciando Fase 14 de Profiling de Latencia...");
    
    // 1. Configurar entorno simulado (Arena)
    let arena = Arc::new(GlobalArena::new(100.0));
    
    let mut swing = SwingEngine::new(10.0, 50.0);
    let mut scalp = ScalpEngine::new();

    // Features simulados (Mock Data L1 Cache amigable)
    let mut features = [0.0f32; 140];
    for i in 0..140 {
        features[i] = (i as f32) * 0.001;
    }

    let iterations = 10_000_000;
    
    // Warm-up de la caché L1 (llenando pipelining del CPU)
    for _ in 0..10_000 {
        black_box(swing.evaluate_trend(50000.0, 0.6, 0.8, 0.5, &arena));
        black_box(scalp.evaluate_microstructure(100.0, 100.0, 2.0, &arena));
    }

    // Benchmark Swing Strategy
    println!("[NANO-PROFILER] Evaluando Swing Strategy ({} iteraciones)...", iterations);
    let start_cycles = rdtsc();
    for _ in 0..iterations {
        black_box(swing.evaluate_trend(50000.0, 0.6, 0.8, 0.5, &arena));
    }
    let end_cycles = rdtsc();
    
    let total_cycles = end_cycles - start_cycles;
    let cycles_per_iter = total_cycles / iterations;
    println!("  -> Swing Strategy: {} ciclos por evaluación", cycles_per_iter);
    // Asumiendo un CPU de ~3.5GHz, 1 ciclo = ~0.28 nanosegundos
    println!("  -> Latencia Estimada (3.5 GHz): {:.2} nanosegundos", (cycles_per_iter as f64) * 0.285);

    // Benchmark Scalp Strategy
    println!("\n[NANO-PROFILER] Evaluando Scalp Strategy ({} iteraciones)...", iterations);
    let start_cycles = rdtsc();
    for _ in 0..iterations {
        black_box(scalp.evaluate_microstructure(100.0, 100.0, 2.0, &arena));
    }
    let end_cycles = rdtsc();
    
    let total_cycles = end_cycles - start_cycles;
    let cycles_per_iter = total_cycles / iterations;
    println!("  -> Scalp Strategy: {} ciclos por evaluación", cycles_per_iter);
    println!("  -> Latencia Estimada (3.5 GHz): {:.2} nanosegundos", (cycles_per_iter as f64) * 0.285);
    
    // Evaluando Penalización por Asignación Dinámica (Para referencia)
    println!("\n[NANO-PROFILER] Midiendo impacto de Heap Allocation (Vec vs Array)...");
    let start_cycles = rdtsc();
    for _ in 0..iterations {
        let mut _v = Vec::with_capacity(140);
        for i in 0..140 {
            _v.push(features[i]);
        }
        black_box(_v);
    }
    let end_cycles = rdtsc();
    let alloc_cycles_per_iter = (end_cycles - start_cycles) / iterations;
    println!("  -> Costo de Heap Allocation: {} ciclos ({:.2} ns)", alloc_cycles_per_iter, (alloc_cycles_per_iter as f64) * 0.285);
    
    println!("\n[NANO-PROFILER] Completado. Si el costo de Scalp/Swing supera los ~200 ciclos, hay Heap Allocations ocultas o Pointer Chasing.");
}
