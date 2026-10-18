//! C-F5b (EST-3 y EST-2 de la revisión del PR #60): la prueba de reversión a
//! la escala de decisión (`strategy_core::prueba_reversion`, `Largo`).
//!
//! Teoría que se prueba. La prueba de retardos cortos (CL-97) contrasta la
//! reversión con los retardos Δ y 2Δ y la extrapola a τ_k suponiendo un OU
//! único. Con una base de dos escalas (un OU rápido más uno lento o una
//! caminata) demuestra «t½ < 2τ_k» por la escala rápida aunque la desviación
//! que se vota apenas revierta en τ_k (EST-3). La prueba nueva mide en τ_k:
//! con z = b_{t−L−1}, x = b_{t−L}, y = b_t − b_{t−L} (L = 8 celdas = τ_k),
//! ĉ = −S̃zy/S̃zx es la fracción de la desviación que revierte en τ_k, y
//! H0: c ≤ ½. El veredicto exige las dos pruebas (intersección-unión: el
//! nivel de cada una acota el del veredicto).
//!
//! Lo que se comprueba:
//! - álgebra: el estadístico incremental (18 sumas con olvido) es el de la
//!   definición, calculado aquí con la doble suma directa;
//! - invariancia: sumar una constante a la base o escalarla no cambia nada;
//! - nivel en la frontera de H0 (OU con vida media de 8,25 celdas, ĉ ≈ ½)
//!   con ruido de medida de 0 y 1 desviación típica, nodos 17 y 18;
//! - EST-3: en el caso del revisor (OU de 2 celdas más OU de 1 676 celdas
//!   con 10,4 veces su desviación) la prueba corta sola vota en ≥ 10 % de
//!   los instantes y el veredicto en ≤ 3 %;
//! - contrato de decisión: con bases de dos escalas en las que la desviación
//!   media revierte justo la mitad en τ_k, lo que se vota revierte al
//!   menos la mitad;
//! - EST-2: con ruido de medida fuerte en la frontera de la prueba corta,
//!   el veredicto no vota (la prueba larga queda dentro de su H0);
//! - potencia: con t½ = τ_k/5 el veredicto rechaza en ≥ 90 % de las
//!   trayectorias a 3 memorias y ≥ 97 % a 4.
//!
//! Generador. El de `comun_reversion` (OU exacto entre marcas, pares cada
//! U(0,5; 1,5)·`paso` ms, ruido de medida iid). Edades en memorias cortas
//! (16·τ_k = 128 celdas); la prueba larga madura a media memoria larga
//! (unas 266 celdas, 2,1 memorias cortas).
//!
//! Umbrales. Con N = 1 000 y tamaño verdadero del 5 %, P(tasa > 0,08) es
//! < 0,001. Las tasas esperadas salen del prototipo (N = 2 000) y de
//! `medicion_critico_reversion.rs`.

mod comun_reversion;

use comun_reversion::*;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::prueba_reversion::{
    critico_largo, PruebaReversion, FRACCION_FRONTERA, HORIZONTE_CELDAS, MEMORIA_LARGA_EN_TAU,
    SESGO_LARGO,
};

const NODO: usize = 17;

/// Pares con una muestra por celda del nodo 17 en el centro de la celda.
fn una_por_celda(muestras: &[f64]) -> PruebaReversion {
    let delta = SPECTRUM_SCALES_MS[NODO] / 8.0;
    let mut p = PruebaReversion::new();
    for (c, &b) in muestras.iter().enumerate() {
        p.observar(((c as f64 + 0.5) * delta) as u64, b);
    }
    p
}

fn serie(n: usize, semilla: u64) -> Vec<f64> {
    let mut g = Lcg(semilla);
    let rho = 0.5f64.powf(1.0 / 5.0);
    let mut x = 0.0;
    (0..n)
        .map(|_| {
            x = rho * x + 1e-4 * (1.0 - rho * rho).sqrt() * g.n();
            x + 2e-5 * g.n()
        })
        .collect()
}

/// Verdad: el estado incremental es el de la definición. Con una muestra por
/// celda, las celdas cerradas son todas menos la última; la fila de la celda
/// c pesa λ^{T−c} (T = última cerrada, λ = e^{−1/512}); V es la doble suma
/// Σ_{|t−s| ≤ L} w_t w_s (z_t − z̄)(z_s − z̄) u_t u_s con u = y − â + ĉ·x.
#[test]
fn cf5b_el_estadistico_incremental_es_el_de_la_definicion() {
    let n = 700;
    let b = serie(n, 41);
    let p = una_por_celda(&b);
    let l = p.estado_largo(NODO).expect("estado");
    let s: Vec<f64> = b.iter().map(|v| v - b[0]).collect();
    let lam = (-1.0 / (8.0 * MEMORIA_LARGA_EN_TAU)).exp();
    let big_l = HORIZONTE_CELDAS;
    let t_ult = n - 2;
    let filas: Vec<(f64, f64, f64, f64)> = (big_l + 1..=t_ult)
        .map(|c| {
            let w = lam.powi((t_ult - c) as i32);
            (w, s[c - big_l - 1], s[c - big_l], s[c] - s[c - big_l])
        })
        .collect();
    let w: f64 = filas.iter().map(|f| f.0).sum();
    let w2: f64 = filas.iter().map(|f| f.0 * f.0).sum();
    let media = |i: usize| filas.iter().map(|f| f.0 * [f.1, f.2, f.3][i]).sum::<f64>() / w;
    let (zb, xb, yb) = (media(0), media(1), media(2));
    let szx: f64 = filas.iter().map(|f| f.0 * (f.1 - zb) * (f.2 - xb)).sum();
    let szy: f64 = filas.iter().map(|f| f.0 * (f.1 - zb) * (f.3 - yb)).sum();
    let c = -szy / szx;
    let a = yb + c * xb;
    let mut v = 0.0;
    for (i, fi) in filas.iter().enumerate() {
        for (j, fj) in filas.iter().enumerate() {
            if i.abs_diff(j) <= big_l {
                let ui = fi.3 - a + c * fi.2;
                let uj = fj.3 - a + c * fj.2;
                v += fi.0 * fj.0 * (fi.1 - zb) * (fj.1 - zb) * ui * uj;
            }
        }
    }
    let n_ef = w * w / w2;
    let t = (c - FRACCION_FRONTERA - SESGO_LARGO / n_ef) * szx.abs() / v.sqrt();
    println!("observado: ĉ {} vs {c}, n_ef {} vs {n_ef}, t {} vs {t}", l.fraccion, l.n_ef, l.t);
    let cerca = |a: f64, b: f64| (a - b).abs() <= 1e-7 * b.abs().max(1e-12);
    assert!(cerca(l.masa, w), "masa {} vs {w}", l.masa);
    assert!(cerca(l.n_ef, n_ef), "n_ef {} vs {n_ef}", l.n_ef);
    assert!(cerca(l.fraccion, c), "ĉ {} vs {c}", l.fraccion);
    assert!(cerca(l.t, t), "t {} vs {t}", l.t);
}

/// Verdad: ĉ y t no dependen del origen ni de la escala de la base (sumas
/// centradas; t es homogéneo de grado 0).
#[test]
fn cf5b_invariante_al_origen_y_a_la_escala() {
    let b = serie(600, 7);
    let p1 = una_por_celda(&b);
    let b2: Vec<f64> = b.iter().map(|v| 3.0 * v + 0.7).collect();
    let p2 = una_por_celda(&b2);
    let (l1, l2) = (p1.estado_largo(NODO).expect("1"), p2.estado_largo(NODO).expect("2"));
    println!("observado: t {} vs {}, ĉ {} vs {}", l1.t, l2.t, l1.fraccion, l2.fraccion);
    assert!((l1.t - l2.t).abs() <= 1e-6 * l1.t.abs().max(1.0), "{l1:?} vs {l2:?}");
    assert!((l1.fraccion - l2.fraccion).abs() <= 1e-6, "{l1:?} vs {l2:?}");
    assert_eq!(l1.n_ef, l2.n_ef);
}

/// Esperado: en la frontera de H0 de la prueba larga (vida media de 8,25
/// celdas: ĉ ≈ ½), nodo 17 con pares cada ~1 s y ruido de 0 y 1 desviación
/// típica, la prueba larga sola rechaza ≤ 0,08 a toda edad desde la madurez
/// (prototipo: ≤ 4,4 %) y el veredicto también (prototipo: ≤ 3,0 %).
#[test]
fn cf5b_nivel_de_la_prueba_larga_en_su_frontera_nodo_17() {
    let edades = [2.2, 3.0, 4.0, 8.0];
    for (i, r) in [0.0, 1.0].into_iter().enumerate() {
        let b = barrido(NODO, 1_000, theta_celdas(NODO, 8.25), r, 1_000.0, &edades, 170 + i as u64);
        informe(&format!("frontera larga nodo 17, ruido {r}"), &b);
        for j in 0..edades.len() {
            assert!(b.rechazo_larga[j] <= 0.08, "ruido {r}, edad {}: {}", edades[j], b.rechazo_larga[j]);
            assert!(b.rechazo[j] <= 0.08, "ruido {r}, edad {}: {}", edades[j], b.rechazo[j]);
        }
    }
}

/// Esperado: lo mismo en el nodo 18 (τ = 68,7 s) con pares cada ~4 s (unos 2
/// por celda) y ruido de ½ desviación típica.
#[test]
fn cf5b_nivel_de_la_prueba_larga_en_su_frontera_nodo_18() {
    let edades = [2.2, 3.0, 4.0];
    let b = barrido(18, 600, theta_celdas(18, 8.25), 0.5, 4_000.0, &edades, 180);
    informe("frontera larga nodo 18, ruido 0.5", &b);
    for j in 0..edades.len() {
        assert!(b.rechazo_larga[j] <= 0.08, "edad {}: {}", edades[j], b.rechazo_larga[j]);
    }
}

fn ou(k: usize, celdas: f64, sd: f64) -> Componente {
    Componente {
        theta: theta_celdas(k, celdas),
        sd,
        referencia_ms: 1_000.0,
    }
}

/// Esperado (EST-3): el caso del revisor. Base = OU de 2 celdas (sd 1 pb) +
/// OU de 1 676 celdas (sd 10,4 pb): la desviación respecto a la media del
/// nodo apenas revierte en τ_k (prototipo: 15-19 %). Antes la prueba corta
/// sola votaba en el 10-21 % de los instantes; con el veredicto, ≤ 3 %
/// (prototipo: ≤ 1,1 %).
#[test]
fn cf5b_el_caso_del_revisor_no_vota() {
    let edades = [2.2, 3.0, 4.0, 8.0];
    let comps = [ou(NODO, 2.0, 1e-4), ou(NODO, 1_676.0, 10.4e-4)];
    let b = barrido_mezcla(NODO, 1_000, &comps, 0.0, 1_000.0, &edades, 31);
    informe("caso del revisor", &b);
    assert!(b.rechazo_corta[0] >= 0.10, "precondición (antes): corta {}", b.rechazo_corta[0]);
    for j in 0..edades.len() {
        assert!(b.rechazo[j] <= 0.03, "edad {}: veredicto {}", edades[j], b.rechazo[j]);
    }
}

/// Esperado (contrato de decisión): con bases de dos escalas ajustadas para
/// que la desviación media respecto a la media del nodo revierta justo la
/// mitad en τ_k, lo que se vota revierte al menos la mitad (prototipo: 0,6
/// a 0,7 con todos los votos). Familias: OU de 2 celdas + caminata, y OU de
/// 2 celdas + OU de 1 676 celdas (las amplitudes del prototipo,
/// `frontera.json`).
#[test]
fn cf5b_lo_votado_revierte_al_menos_la_mitad() {
    let edades = [2.2, 3.0, 4.0, 6.0];
    let caminata = Componente {
        theta: 0.0,
        sd: 0.0801e-4,
        referencia_ms: 1_000.0,
    };
    let familias = [
        ("OU 2 celdas + caminata", [ou(NODO, 2.0, 1e-4), caminata]),
        ("OU 2 celdas + OU 1 676 celdas", [ou(NODO, 2.0, 1e-4), ou(NODO, 1_676.0, 4.1763e-4)]),
    ];
    for (i, (nombre, comps)) in familias.iter().enumerate() {
        let b = barrido_mezcla(NODO, 1_000, comps, 0.0, 1_000.0, &edades, 60 + i as u64);
        informe(nombre, &b);
        println!("observado: {nombre}: reversión de todo lo votado {:.3}", b.reversion_total);
        assert!(b.rechazo[0] >= 0.05, "{nombre}: precondición, hay votos: {}", b.rechazo[0]);
        assert!(b.reversion_total >= 0.5, "{nombre}: reversión {}", b.reversion_total);
    }
}

/// Esperado (EST-2): en la frontera de la prueba corta (vida media 2τ_k) con
/// ruido de medida de 3 y 10 desviaciones típicas, el veredicto rechaza
/// ≤ 2 %: la desviación revierte un 29 % en τ_k y la prueba larga la deja
/// en H0 (prototipo: ≤ 1,3 % con ruido ≥ ½ σ). La prueba corta sola llegaba
/// al 7,3-8,9 %.
#[test]
fn cf5b_ruido_fuerte_no_vota_en_la_frontera_corta() {
    let edades = [2.2, 4.0, 8.0];
    for (i, r) in [3.0, 10.0].into_iter().enumerate() {
        let b = barrido(NODO, 1_000, theta0(NODO), r, 1_000.0, &edades, 90 + i as u64);
        informe(&format!("frontera corta, ruido {r}"), &b);
        for j in 0..edades.len() {
            assert!(b.rechazo[j] <= 0.02, "ruido {r}, edad {}: {}", edades[j], b.rechazo[j]);
        }
    }
}

/// Esperado (potencia): con t½ = τ_17/5 (1,6 celdas), el veredicto rechaza en
/// ≥ 90 % de las trayectorias a 3 memorias y ≥ 97 % a 4 (prototipo: 95 % y
/// 99 %).
#[test]
fn cf5b_potencia_con_reversion_rapida() {
    let edades = [3.0, 4.0];
    let b = barrido(NODO, 400, theta_celdas(NODO, 1.6), 0.0, 1_000.0, &edades, 99);
    informe("t½ = τ_17/5", &b);
    assert!(b.rechazo[0] >= 0.90, "edad 3: {}", b.rechazo[0]);
    assert!(b.rechazo[1] >= 0.97, "edad 4: {}", b.rechazo[1]);
}

/// Verdad: tras un hueco de una memoria larga (64·τ_k) sin pares, la masa de
/// la prueba larga ha decaído por e^{−1} y no está madura: el veredicto no
/// demuestra nada aunque antes estuviera demostrado.
#[test]
fn cf5b_un_hueco_largo_devuelve_la_prueba_larga_a_la_inmadurez() {
    let tau = SPECTRUM_SCALES_MS[NODO];
    let theta = theta_celdas(NODO, 1.6);
    let mut g = Lcg(5);
    let mut p = PruebaReversion::new();
    let (mut x, mut t) = (0.0f64, 0.0f64);
    let paso = |g: &mut Lcg, x: f64, h: f64| {
        let d = (-theta * h).exp();
        x * d + 1e-4 * (1.0 - d * d).sqrt() * g.n()
    };
    while t < 8.0 * 16.0 * tau {
        t += 1_000.0;
        x = paso(&mut g, x, 1_000.0);
        p.observar(t as u64, x);
    }
    let l = p.estado_largo(NODO).expect("estado");
    assert!(l.madura && l.t > critico_largo(l.n_ef), "precondición: {l:?}");
    assert!(p.veredicto(tau).expect("pares").demostrada, "precondición: demostrada");
    t += MEMORIA_LARGA_EN_TAU * tau;
    x = paso(&mut g, x, MEMORIA_LARGA_EN_TAU * tau);
    p.observar(t as u64, x);
    let l = p.estado_largo(NODO).expect("estado");
    println!("observado: tras el hueco, masa {:.1}, madura {}", l.masa, l.madura);
    assert!(!l.madura && !p.veredicto(tau).expect("pares").demostrada);
}
