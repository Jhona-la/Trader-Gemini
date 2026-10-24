//! C-F5b (EST-3 y EST-2 de la revisión del PR #60) y C-F5e (revisión de
//! C-F5b): la prueba de reversión a la escala de decisión
//! (`strategy_core::prueba_reversion`, `Largo`).
//!
//! Teoría que se prueba. La prueba de retardos cortos (CL-97) contrasta la
//! reversión con los retardos Δ y 2Δ y la extrapola a τ_k suponiendo un OU
//! único. Con una base de dos escalas (un OU rápido más uno lento o una
//! caminata) demuestra «t½ < 2τ_k» por la escala rápida aunque la desviación
//! que se vota apenas revierta en τ_k (EST-3). La prueba larga mide en τ_k
//! lo que se vota: x = d_{t−L}, la desviación observada respecto a la media
//! del nodo, e y = b_t − (2b_{t−L−1} − b_{t−L−2}) (L = 8 celdas = τ_k);
//! R̂ = −Σw·x·y/Σw·x² es la fracción de la desviación votada que revierte en
//! τ_k, y H0: R ≤ ½. El veredicto exige las dos pruebas (intersección-unión:
//! el nivel de cada una acota el del veredicto).
//!
//! Lo que se comprueba:
//! - álgebra: el estadístico incremental (sumas con olvido) es el de la
//!   definición, calculado aquí con la doble suma directa, también lejos del
//!   origen;
//! - invariancia: sumar una constante a la base o escalarla no cambia nada;
//! - nivel en la frontera de H0 (OU con vida media de 7,76 celdas sin ruido
//!   y de 5,23 con ruido de ½ desviación típica: R ≈ ½), nodos 17 y 18, y
//!   con volatilidad por rachas;
//! - contrato con ruido: con t½ = τ_k/5 y ruido de 1 y 2 desviaciones
//!   típicas lo votado revertiría menos de la mitad, y no se vota;
//! - EST-3: en el caso del revisor (OU de 2 celdas más OU de 1 676 celdas
//!   con 10,4 veces su desviación) la prueba corta sola vota en ≥ 10 % de
//!   los instantes y el veredicto en ≤ 3 %;
//! - bases de dos escalas: en las dos familias medidas, con la desviación
//!   media revirtiendo justo la mitad en τ_k, lo que se vota revierte al
//!   menos la mitad (es una medición en esas bases, no una garantía);
//! - EST-2: con ruido de medida fuerte en la frontera de la prueba corta,
//!   el veredicto no vota;
//! - potencia: con t½ = τ_k/5 y sin ruido, el veredicto rechaza en ≥ 84 %
//!   de las trayectorias a 3 memorias y ≥ 93 % a 4.
//!
//! Generador. El de `comun_reversion` (OU exacto entre marcas, pares cada
//! U(0,5; 1,5)·`paso` ms, ruido de medida iid). Edades en memorias cortas
//! (16·τ_k = 128 celdas); la prueba larga madura a media memoria larga
//! (unas 266 celdas, 2,1 memorias cortas).
//!
//! Umbrales. Con N = 1 000 y tamaño verdadero del 5 %, P(tasa > 0,08) es
//! < 0,001. Las tasas esperadas salen del prototipo (N = 2 000 a 5 000) y
//! de `medicion_critico_reversion.rs`.

mod comun_reversion;

use comun_reversion::*;
use quantum_arena::temporal_spectrum::SPECTRUM_SCALES_MS;
use strategy_core::prueba_reversion::{
    PruebaReversion, CRITICO_LARGO, FRACCION_FRONTERA, HORIZONTE_CELDAS, MEMORIA_LARGA_EN_TAU,
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

/// Estado de la prueba larga calculado con la definición. Con una muestra
/// por celda, las celdas cerradas son todas menos la última. La media del
/// nodo al cerrar la celda c (c ≥ 2) es la de la prueba corta:
/// m_c = Σ_{j=2..c} μ^{c−j}·b_{j−1}/Σ μ^{c−j}, con μ = e^{−1/128}, y
/// d_c = b_c − m_c. La fila de la celda c (c ≥ L + 2) pesa λ^{T−c}
/// (T = última cerrada, λ = e^{−1/512}), con x = d_{c−L} e
/// y = b_c − (2b_{c−L−1} − b_{c−L−2}). R̂ = −Σwxy/Σwx², V es la doble suma
/// Σ_{|t−s| ≤ L} w_t w_s g_t g_s con g = x·y + R̂·x², y
/// n = min(n_ef, 3·(Σwx²)²/Σw²x⁴). Devuelve (masa, n_ef, n, R̂, t).
fn definicion(b: &[f64]) -> (f64, f64, f64, f64, f64) {
    let s: Vec<f64> = b.iter().map(|v| v - b[0]).collect();
    let mu = (-1.0f64 / 128.0).exp();
    let d: Vec<f64> = (0..s.len())
        .map(|c| {
            if c < 2 {
                return f64::NAN;
            }
            let (num, den) = (2..=c).fold((0.0, 0.0), |(n, w), j| {
                let p = mu.powi((c - j) as i32);
                (n + p * s[j - 1], w + p)
            });
            s[c] - num / den
        })
        .collect();
    let lam = (-1.0 / (8.0 * MEMORIA_LARGA_EN_TAU)).exp();
    let big_l = HORIZONTE_CELDAS;
    let t_ult = b.len() - 2;
    // (w, x, y)
    let filas: Vec<[f64; 3]> = (big_l + 2..=t_ult)
        .map(|c| {
            let w = lam.powi((t_ult - c) as i32);
            [w, d[c - big_l], s[c] - (2.0 * s[c - big_l - 1] - s[c - big_l - 2])]
        })
        .collect();
    let w: f64 = filas.iter().map(|f| f[0]).sum();
    let w2: f64 = filas.iter().map(|f| f[0] * f[0]).sum();
    let sxx: f64 = filas.iter().map(|f| f[0] * f[1] * f[1]).sum();
    let sxy: f64 = filas.iter().map(|f| f[0] * f[1] * f[2]).sum();
    let x4: f64 = filas.iter().map(|f| f[0] * f[0] * f[1].powi(4)).sum();
    let r = -sxy / sxx;
    let g: Vec<f64> = filas.iter().map(|f| f[1] * f[2] + r * f[1] * f[1]).collect();
    let mut v = 0.0;
    for i in 0..filas.len() {
        for j in i.saturating_sub(big_l)..(i + big_l + 1).min(filas.len()) {
            v += filas[i][0] * filas[j][0] * g[i] * g[j];
        }
    }
    let n_ef = w * w / w2;
    let n = n_ef.min(3.0 * sxx * sxx / x4);
    let t = (r - FRACCION_FRONTERA - SESGO_LARGO / n) * sxx / v.sqrt();
    (w, n_ef, n, r, t)
}

fn comparar_con_la_definicion(b: &[f64]) {
    let p = una_por_celda(b);
    let l = p.estado_largo(NODO).expect("estado");
    let (w, n_ef, n, r, t) = definicion(b);
    println!("observado: R̂ {} vs {r}, n {} vs {n}, t {} vs {t}", l.reversion, l.n, l.t);
    let cerca = |a: f64, b: f64| (a - b).abs() <= 1e-7 * b.abs().max(1e-12);
    assert!(cerca(l.masa, w), "masa {} vs {w}", l.masa);
    assert!(cerca(l.n_ef, n_ef), "n_ef {} vs {n_ef}", l.n_ef);
    assert!(cerca(l.n, n), "n {} vs {n}", l.n);
    assert!(cerca(l.reversion, r), "R̂ {} vs {r}", l.reversion);
    assert!(cerca(l.t, t), "t {} vs {t}", l.t);
}

/// Verdad: el estado incremental (sumas con olvido, V = αᵀMα) es el de la
/// definición, calculado aquí con la doble suma directa.
#[test]
fn cf5b_el_estadistico_incremental_es_el_de_la_definicion() {
    comparar_con_la_definicion(&serie(700, 41));
}

/// C-F5e (precisión, revisión de C-F5b). Verdad: lo mismo con la base
/// alejándose del origen (el primer par) hasta 10³σ y 10⁵σ. En C-F5b las
/// sumas eran de cuarto orden respecto al origen y su error crecía como
/// (D/σ)⁴·ε: con 10³σ el t cambiaba hasta 0,1 y con 3·10³σ la varianza
/// salía ≤ 0. Ahora x es la desviación respecto a la media del nodo e y una
/// diferencia de muestras: las sumas no dependen del origen.
#[test]
fn cf5e_el_estadistico_no_pierde_precision_lejos_del_origen() {
    for d in [0.1, 10.0] {
        let b: Vec<f64> = serie(2_000, 43)
            .iter()
            .enumerate()
            .map(|(c, v)| v + d * (c as f64 / 600.0).min(1.0))
            .collect();
        comparar_con_la_definicion(&b);
    }
}

/// Verdad: R̂, n y t no dependen del origen ni de la escala de la base
/// (sumas centradas; t es homogéneo de grado 0). El desplazamiento lo
/// absorbe el origen; el alejamiento del origen lo prueba
/// `cf5e_el_estadistico_no_pierde_precision_lejos_del_origen`.
#[test]
fn cf5b_invariante_al_origen_y_a_la_escala() {
    let b = serie(600, 7);
    let p1 = una_por_celda(&b);
    let b2: Vec<f64> = b.iter().map(|v| 3.0 * v + 0.7).collect();
    let p2 = una_por_celda(&b2);
    let (l1, l2) = (p1.estado_largo(NODO).expect("1"), p2.estado_largo(NODO).expect("2"));
    println!("observado: t {} vs {}, R̂ {} vs {}", l1.t, l2.t, l1.reversion, l2.reversion);
    assert!((l1.t - l2.t).abs() <= 1e-6 * l1.t.abs().max(1.0), "{l1:?} vs {l2:?}");
    assert!((l1.reversion - l2.reversion).abs() <= 1e-6, "{l1:?} vs {l2:?}");
    assert!((l1.n - l2.n).abs() <= 1e-6 * l1.n, "{l1:?} vs {l2:?}");
    assert_eq!(l1.n_ef, l2.n_ef);
}

/// Vida media (celdas) de un OU con R = ½ en la frontera de la prueba larga,
/// sin ruido de medida y con ruido de ½ desviación típica (raíces del
/// estimando estacionario; ver la cabecera de `prueba_reversion`).
const FRONTERA_SIN_RUIDO: f64 = 7.7591;
const FRONTERA_RUIDO_MEDIO: f64 = 5.2285;

/// Esperado: en la frontera de H0 de la prueba larga, nodo 17 con pares cada
/// ~1 s, sin ruido (t½ = 7,76 celdas) y con ruido de ½ desviación típica
/// (t½ = 5,23 celdas), la prueba larga sola rechaza ≤ 0,08 a toda edad desde
/// la madurez (prototipo: ≤ 4,5 %; observado: ≤ 4,1 %) y el veredicto
/// también.
#[test]
fn cf5b_nivel_de_la_prueba_larga_en_su_frontera_nodo_17() {
    let edades = [2.2, 3.0, 4.0, 8.0];
    let casos = [(0.0, FRONTERA_SIN_RUIDO), (0.5, FRONTERA_RUIDO_MEDIO)];
    for (i, (r, celdas)) in casos.into_iter().enumerate() {
        let b = barrido(NODO, 1_000, theta_celdas(NODO, celdas), r, 1_000.0, &edades, 170 + i as u64);
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
    let b = barrido(18, 600, theta_celdas(18, FRONTERA_RUIDO_MEDIO), 0.5, 4_000.0, &edades, 180);
    informe("frontera larga nodo 18, ruido 0.5", &b);
    for j in 0..edades.len() {
        assert!(b.rechazo_larga[j] <= 0.08, "edad {}: {}", edades[j], b.rechazo_larga[j]);
    }
}

/// Esperado (C-F5e, heterocedasticidad): en la frontera sin ruido, con
/// volatilidad por rachas (log-volatilidad de desviación típica 1 y vida
/// media de 50 celdas), la prueba larga sola rechaza ≤ 0,08 (prototipo:
/// ≤ 5 %). La prueba larga de C-F5b, con la corrección de muestra finita
/// por n_ef, llegaba al 5,2-9,8 % con rachas (revisión de C-F5b).
#[test]
fn cf5e_nivel_con_volatilidad_por_rachas() {
    let edades = [2.2, 3.0, 4.0, 8.0];
    let c = Componente {
        theta: theta_celdas(NODO, FRONTERA_SIN_RUIDO),
        sd: 1e-4,
        referencia_ms: 1_000.0,
    };
    let b = barrido_rachas(NODO, 1_000, &[c], 0.0, 1_000.0, &edades, 175, (1.0, 50.0));
    informe("frontera larga con rachas", &b);
    for j in 0..edades.len() {
        assert!(b.rechazo_larga[j] <= 0.08, "edad {}: {}", edades[j], b.rechazo_larga[j]);
    }
}

/// Esperado (C-F5e, contrato con ruido): con t½ = τ_k/5 (1,6 celdas) y ruido
/// de medida de 1 y 2 desviaciones típicas, lo votado revertiría como mucho
/// 0,97/(1 + r²) ≤ 0,485 en τ_k: el veredicto rechaza ≤ 2 % (prototipo:
/// ≤ 0,4 %). Con el estimador de C-F5b, que medía la parte persistente,
/// votaba en el 44-52 % de los instantes con r = 1.
#[test]
fn cf5e_con_ruido_fuerte_no_vota_lo_que_no_revierte() {
    let edades = [2.2, 4.0, 8.0];
    for (i, r) in [1.0, 2.0].into_iter().enumerate() {
        let b = barrido(NODO, 1_000, theta_celdas(NODO, 1.6), r, 1_000.0, &edades, 120 + i as u64);
        informe(&format!("t½ = τ_17/5, ruido {r}"), &b);
        for j in 0..edades.len() {
            assert!(b.rechazo[j] <= 0.02, "ruido {r}, edad {}: {}", edades[j], b.rechazo[j]);
        }
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
/// nodo apenas revierte en τ_k (prototipo: 13-19 %). Antes la prueba corta
/// sola votaba en el 10-21 % de los instantes; con el veredicto, ≤ 3 %
/// (prototipo: ≤ 0,9 %).
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

/// Esperado (medido en estas dos familias, no garantizado en toda base de
/// dos escalas): con las amplitudes ajustadas para que la desviación media
/// respecto a la media del nodo revierta justo la mitad en τ_k, lo que se
/// vota revierte al menos la mitad (observado: 0,72
/// y 0,76 con todos los votos; prototipo: 0,65 y 0,67). Familias: OU de 2
/// celdas + caminata, y OU de 2 celdas + OU de 1 676 celdas (las amplitudes
/// del prototipo, `frontera.json`). En ellas el veredicto vota en el 8-16 %
/// de los instantes: la prueba mide lo que revierte la desviación en la
/// ventana (ver la cabecera de `prueba_reversion`).
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
/// ≤ 2 %: lo votado revierte como mucho 0,29/(1 + r²) en τ_k y la prueba
/// larga lo deja en H0. La prueba corta sola llegaba al 7,3-8,9 %.
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

/// Esperado (potencia): con t½ = τ_17/5 (1,6 celdas) y sin ruido, el
/// veredicto rechaza en ≥ 84 % de las trayectorias a 3 memorias y ≥ 93 % a
/// 4 (prototipo: 88,6 % y 96,8 %, N = 2 000; con N = 400 el error de
/// muestreo es 1,6 y 0,9 puntos). La prueba de C-F5b tenía más (95 % y
/// 99 %) porque medía la parte persistente, sin la pérdida de la
/// extrapolación.
#[test]
fn cf5b_potencia_con_reversion_rapida() {
    let edades = [3.0, 4.0];
    let b = barrido(NODO, 400, theta_celdas(NODO, 1.6), 0.0, 1_000.0, &edades, 99);
    informe("t½ = τ_17/5", &b);
    assert!(b.rechazo[0] >= 0.84, "edad 3: {}", b.rechazo[0]);
    assert!(b.rechazo[1] >= 0.93, "edad 4: {}", b.rechazo[1]);
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
    assert!(l.madura && l.t > CRITICO_LARGO, "precondición: {l:?}");
    assert!(p.veredicto(tau).expect("pares").demostrada, "precondición: demostrada");
    t += MEMORIA_LARGA_EN_TAU * tau;
    x = paso(&mut g, x, MEMORIA_LARGA_EN_TAU * tau);
    p.observar(t as u64, x);
    let l = p.estado_largo(NODO).expect("estado");
    println!("observado: tras el hueco, masa {:.1}, madura {}", l.masa, l.madura);
    assert!(!l.madura && !p.veredicto(tau).expect("pares").demostrada);
}
