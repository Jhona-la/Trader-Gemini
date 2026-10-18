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
//! C-F5b y C-F5e: la segunda medición calibra la prueba a la escala de
//! decisión (`SESGO_LARGO`) en su frontera: el OU cuya
//! desviación votada estima justo ½ con la extrapolación de orden 2 (vida
//! media de 7,7591 celdas sin ruido, 7,011 con ruido de ¼ σ y 5,2285 con
//! ½ σ; con ruido ≥ 0,8 σ ningún OU único sale de H0). Nodos 17 y 18, pares
//! cada 0,25 a 4 s, edades de 2,2 a 16 memorias, N = 5 000; además, con
//! ruido de 1 a 10 σ y t½ = τ_k/5 (dentro de H0) y con volatilidad por
//! rachas en la frontera sin ruido. Calibrada con el prototipo numpy del
//! 2026-10-18 (ver la cabecera de `prueba_reversion.rs`); comprueba que el
//! tamaño máximo de la prueba larga sola no pase del 5 % más el error de
//! muestreo del máximo de unas 230 celdas (configuración × edad).
//!
//! Uso: `cargo test -p strategy-core --test medicion_critico_reversion --
//! --ignored --nocapture` (≈ 2 min la primera; la segunda, en `--release`).

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
                linea += &format!(" [{:.2} mem: {:.3} q95 {:.2}]", edades[j], b.rechazo_corta[j], b.q95[j]);
                peor = peor.max(b.q95[j]);
            }
            println!("{linea}");
        }
    }
    println!("observado: cuantil 95 máximo = {peor:.3} (crítico {CRITICO})");
    assert!(peor <= CRITICO + 0.05, "cuantil 95 máximo {peor}");
}

#[test]
#[ignore = "MEDICIÓN manual (largo, en --release): calibración del crítico de la prueba a la escala de decisión"]
fn medicion_critico_largo_en_su_frontera() {
    let edades = [2.2, 2.5, 3.0, 4.0, 6.0, 8.0, 16.0];
    let mut peor: f64 = 0.0;
    let mut medir = |nombre: String, b: &Barrido| {
        let mut linea = format!("{nombre}:");
        for j in 0..edades.len() {
            linea += &format!(" [{:.1} mem: {:.4} q95 {:.2}]", edades[j], b.rechazo_larga[j], b.q95_largo[j]);
            peor = peor.max(b.rechazo_larga[j]);
        }
        println!("{linea}");
    };
    let combos = [(17usize, 250.0f64), (17, 1_000.0), (17, 2_000.0), (18, 1_000.0), (18, 4_000.0)];
    for &(k, paso) in &combos {
        for &(r, celdas) in &[(0.0, 7.7591), (0.25, 7.011), (0.5, 5.2285)] {
            let semilla = 50_000 + (k as u64) * 1_000 + (r * 100.0) as u64 + paso as u64;
            let b = barrido(k, 5_000, theta_celdas(k, celdas), r, paso, &edades, semilla);
            medir(format!("frontera, nodo {k}, paso {paso} ms, ruido {r}"), &b);
        }
    }
    for &(k, paso) in &[(17usize, 250.0f64), (17, 1_000.0), (18, 1_000.0)] {
        for &r in &[1.0, 2.0, 3.0, 10.0] {
            let semilla = 60_000 + (k as u64) * 1_000 + (r * 10.0) as u64 + paso as u64;
            let b = barrido(k, 3_000, theta_celdas(k, 1.6), r, paso, &edades, semilla);
            medir(format!("t½ = τ_k/5, nodo {k}, paso {paso} ms, ruido {r}"), &b);
        }
    }
    for &k in &[17usize, 18] {
        for &(sd, semivida) in &[(0.5, 50.0), (1.0, 50.0), (1.0, 500.0)] {
            let comps = [Componente {
                theta: theta_celdas(k, 7.7591),
                sd: 1e-4,
                referencia_ms: 1_000.0,
            }];
            let semilla = 70_000 + (k as u64) * 1_000 + (sd * 10.0) as u64 + semivida as u64;
            let b = barrido_rachas(k, 5_000, &comps, 0.0, 1_000.0, &edades, semilla, (sd, semivida));
            medir(format!("rachas {sd} y {semivida} celdas, nodo {k}"), &b);
        }
    }
    println!("observado: tamaño máximo de la prueba larga = {peor:.4}");
    assert!(peor <= 0.06, "tamaño máximo {peor}");
}
