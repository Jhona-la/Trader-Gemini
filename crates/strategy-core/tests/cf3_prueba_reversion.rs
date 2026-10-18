//! C-F3 · F3.7-02 v2 (Línea C, CL-97): nivel y potencia de la prueba de
//! reversión de la base (`strategy_core::prueba_reversion`).
//!
//! Teoría que se prueba. La base es un OU de reversión θ observado en pares
//! síncronos con marcas irregulares. H0: θ ≤ θ0 = ln2/(2τ_k). En la frontera
//! (θ = θ0) la prueba debe rechazar con probabilidad de un 5 % como mucho
//! (nivel), a cualquier edad desde la madurez y con ruido de medida en la
//! base; dentro de H0 (caminata), no más; y con reversión rápida
//! (t½ = τ_k/5) debe rechazar casi siempre (potencia). El crítico (1,75)
//! sale del Monte Carlo de `medicion_critico_reversion.rs` (N = 10 000:
//! tamaño máximo 5,1 %, con ruido de 1 σ); aquí se comprueba con N = 1 000.
//!
//! Generador. OU exacto en tiempo continuo entre marcas,
//! x(t + h) = x(t)·e^{−θh} + s·√(1 − e^{−2θh})·N, con s = 1 pb; pares cada
//! U(0,5; 1,5)·`paso` ms; ruido de medida iid N(0, r·s) en cada par. Edades
//! en memorias del nodo (16·τ_k). LCG de 64 bits con Box-Muller.
//!
//! Umbrales. Con N trayectorias, la tasa de rechazo en la frontera se compara
//! con 0,08: con tamaño verdadero del 5 % y N = 1 000, P(tasa > 0,08) es
//! < 0,001 (3,3 desviaciones típicas). Edades desde 0,6 memorias (la
//! madurez llega a media memoria). La calibración del crítico con
//! N = 10 000 por configuración está en `medicion_critico_reversion.rs`.

mod comun_reversion;

use comun_reversion::*;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::prueba_reversion::{PruebaReversion, CRITICO};

/// Esperado: en la frontera de H0, nodo 17 (τ = 17,2 s, pares cada ~1 s), la
/// tasa de rechazo es ≤ 0,08 a toda edad y con ruido de medida de 0, ½ y 1
/// desviación típica de la base; el cuantil 95 de t es ≤ el crítico + 0,15.
#[test]
fn cl97_nivel_en_la_frontera_nodo_17() {
    let edades = [0.6, 1.0, 2.0, 4.0];
    for (i, r) in [0.0, 0.5, 1.0].into_iter().enumerate() {
        let b = barrido(17, 1_000, theta0(17), r, 1_000.0, &edades, 17 + i as u64);
        informe(&format!("frontera nodo 17, ruido {r}"), &b);
        for j in 0..edades.len() {
            assert!(b.rechazo_corta[j] <= 0.08, "ruido {r}, edad {}: rechazo {}", edades[j], b.rechazo_corta[j]);
            assert!(b.q95[j] <= CRITICO + 0.15, "ruido {r}, edad {}: q95 {}", edades[j], b.q95[j]);
        }
    }
}

/// Esperado: lo mismo en el nodo 18 (τ = 68,7 s), con pares cada ~2 s
/// (unos 4 por celda) y ruido de ½ desviación típica.
#[test]
fn cl97_nivel_en_la_frontera_nodo_18() {
    let edades = [0.6, 1.0, 2.0, 4.0];
    let b = barrido(18, 600, theta0(18), 0.5, 2_000.0, &edades, 18);
    informe("frontera nodo 18, ruido 0.5", &b);
    for j in 0..edades.len() {
        assert!(b.rechazo_corta[j] <= 0.08, "edad {}: rechazo {}", edades[j], b.rechazo_corta[j]);
    }
}

/// Esperado: una caminata (θ = 0, dentro de H0) se rechaza ≤ 0,08. Sin ruido,
/// mucho menos que la frontera (≤ 2 %); con ruido de medida de 10 veces la
/// innovación por par, ≈ 5 % al madurar y menos después.
#[test]
fn cl97_caminata_dentro_de_h0() {
    let edades = [0.6, 1.0, 2.0, 4.0];
    for (i, r) in [0.0, 10.0].into_iter().enumerate() {
        let b = barrido(17, 1_000, 0.0, r, 1_000.0, &edades, 70 + i as u64);
        informe(&format!("caminata nodo 17, ruido {r}"), &b);
        for j in 0..edades.len() {
            assert!(b.rechazo_corta[j] <= 0.08, "ruido {r}, edad {}: rechazo {}", edades[j], b.rechazo_corta[j]);
        }
    }
}

/// Esperado (potencia): con t½ = τ_17/5 (3,4 s; θ = 10·θ0) la prueba rechaza
/// en ≥ 60 % de las trayectorias a 2 memorias y ≥ 80 % a 4.
#[test]
fn cl97_potencia_con_reversion_rapida() {
    let edades = [2.0, 4.0];
    let b = barrido(17, 400, 10.0 * theta0(17), 0.0, 1_000.0, &edades, 99);
    informe("t½ = τ_17/5, nodo 17", &b);
    assert!(b.rechazo_corta[0] >= 0.60, "edad 2: rechazo {}", b.rechazo_corta[0]);
    assert!(b.rechazo_corta[1] >= 0.80, "edad 4: rechazo {}", b.rechazo_corta[1]);
}

/// Verdad: tras un hueco de dos memorias sin pares, la evidencia vieja ha
/// decaído por e^{−2} y el nodo no está maduro: el primer par tras el hueco
/// no demuestra nada aunque antes estuviera demostrado.
#[test]
fn cl97_un_hueco_largo_devuelve_a_la_inmadurez() {
    let k = 17;
    let tau = SPECTRUM_SCALES_MS[k];
    let theta = 10.0 * theta0(k);
    let mut g = Lcg(5);
    let mut p = PruebaReversion::new();
    let mut x = 0.0f64;
    let mut t = 0.0f64;
    let paso = |g: &mut Lcg, x: f64, h: f64| {
        let d = (-theta * h).exp();
        x * d + 1e-4 * (1.0 - d * d).sqrt() * g.n()
    };
    while t < 4.0 * 16.0 * tau {
        t += 1_000.0;
        x = paso(&mut g, x, 1_000.0);
        p.observar(t as u64, x);
    }
    assert!(p.veredicto(tau).unwrap().demostrada, "precondición: demostrada");
    t += 2.0 * 16.0 * tau;
    x = paso(&mut g, x, 2.0 * 16.0 * tau);
    p.observar(t as u64, x);
    let e = p.estado_nodo(k).expect("estado");
    println!("observado: tras el hueco, masa {:.1}, madura {}", e.masa, e.madura);
    assert!(!e.madura && !p.veredicto(tau).unwrap().demostrada);
}
