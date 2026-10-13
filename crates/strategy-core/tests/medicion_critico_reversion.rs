//! MEDICIÓN manual (CL-97, F3.7-02 v2): calibración del crítico de la prueba
//! de reversión de la base (`strategy_core::prueba_reversion::CRITICO`).
//!
//! Barrido de Monte Carlo en la frontera de H0 (θ = ln2/(2τ_k)): nodos 17 a
//! 19, pares cada 0,25 a 4 s de media con jitter U(0,5; 1,5), ruido de medida
//! de 0 a 1 desviación típica de la base, edades de 0,55 a 4 memorias;
//! N = 10 000 trayectorias por configuración (3 000 en el nodo 19).
//! Imprime la tasa de rechazo y el cuantil 95 del estadístico y comprueba
//! que el cuantil 95 máximo no pase del crítico más el error de muestreo.
//!
//! Medido el 2026-10-12 (Windows, msvc): cuantil 95 máximo 1,76 y tamaño
//! máximo 5,1 %, los dos con ruido de 1 σ; con ruido ≤ ½ σ, tamaño ≤ 4,7 %.
//! Sin ruido, el tamaño baja con la edad (4,6 % a 0,6 memorias, 2,6–3,0 %
//! a 4): la corrección de sesgo hace el estadístico casi pivotal.
//!
//! Uso: `cargo test -p strategy-core --test medicion_critico_reversion --
//! --ignored --nocapture` (≈ 2 min).

mod comun_reversion;

use comun_reversion::*;
use strategy_core::prueba_reversion::CRITICO;

#[test]
#[ignore = "MEDICIÓN manual (≈ 2 min): calibración del crítico de la prueba de reversión"]
fn medicion_critico_en_la_frontera() {
    let edades = [0.55, 0.6, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
    let mut peor: f64 = 0.0;
    for &(k, paso) in &[(17usize, 1_000.0f64), (17, 250.0), (18, 2_000.0), (18, 1_000.0), (19, 4_000.0)] {
        for &r in &[0.0, 0.25, 0.5, 1.0] {
            let n = if k == 19 { 3_000 } else { 10_000 };
            let semilla = (k as u64) * 1_000 + (r * 100.0) as u64 + paso as u64;
            let b = barrido(k, n, theta0(k), r, paso, &edades, semilla);
            let mut linea = format!("nodo {k}, paso {paso} ms, ruido {r}:");
            for j in 0..edades.len() {
                linea += &format!(" [{:.2} mem: {:.3} q95 {:.2}]", edades[j], b.rechazo[j], b.q95[j]);
                peor = peor.max(b.q95[j]);
            }
            println!("{linea}");
        }
    }
    println!("observado: cuantil 95 máximo = {peor:.3} (crítico {CRITICO})");
    assert!(peor <= CRITICO + 0.05, "cuantil 95 máximo {peor}");
}
