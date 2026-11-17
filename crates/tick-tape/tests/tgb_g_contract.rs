//! TGB-G (cierre, `pedido_tgb_g_v1.md` §4): contratos PUROS de la firma de generacion y de la admision.
//! Buffers y fixtures en memoria; sin temporales, sin reloj, sin red. Los pares bid/ask se fabrican aqui con las
//! recetas (oraculo independiente del clasificador); nunca se usa el clasificador para generar lo esperado.

use tick_tape::procedencia::{firma_de_cinta, firma_de_registros, EstadoGeneracion, Firma, R_RELATIVA, SUELO_ANTIGUO};

fn registro_q(ts: u64, bid: f64, ask: f64, bq: f64, aq: f64) -> [u8; 40] {
    let mut r = [0u8; 40];
    r[..8].copy_from_slice(&ts.to_le_bytes());
    r[8..16].copy_from_slice(&bid.to_le_bytes());
    r[16..24].copy_from_slice(&ask.to_le_bytes());
    r[24..32].copy_from_slice(&bq.to_le_bytes());
    r[32..40].copy_from_slice(&aq.to_le_bytes());
    r
}

fn registro(ts: u64, bid: f64, ask: f64) -> [u8; 40] {
    registro_q(ts, bid, ask, 1.0, 2.0)
}

fn rel(p: f64) -> (f64, f64) {
    let h = p * R_RELATIVA;
    (p - h, p + h)
}

fn ant(p: f64) -> (f64, f64) {
    let h = (p * R_RELATIVA).max(SUELO_ANTIGUO);
    (p - h, p + h)
}

fn medio_tick(p: f64, tick: f64) -> (f64, f64) {
    let h = (p * R_RELATIVA).max(tick * 0.5);
    (p - h, p + h)
}

fn cinta(pares: &[(f64, f64)]) -> Vec<u8> {
    let mut v = b"TGMTICK1".to_vec();
    for (k, (b, a)) in pares.iter().enumerate() {
        v.extend_from_slice(&registro(1_780_300_800_000 + k as u64, *b, *a));
    }
    v
}

fn precios(desde: f64, paso: f64, n: usize) -> Vec<f64> {
    (0..n).map(|k| desde + paso * k as f64).collect()
}

fn firma(bytes: &[u8]) -> Firma {
    firma_de_cinta(bytes).expect("cinta REAL bien formada")
}

/// Clase de UN registro: (solo_relativa, solo_antigua, ambas, ninguna, invalido).
fn clase(b: f64, a: f64) -> (u64, u64, u64, u64, u64) {
    let f = firma_de_registros(&registro(1, b, a)).expect("40 B");
    (f.solo_relativa, f.solo_antigua, f.ambas, f.ninguna, f.invalidos)
}

fn mismos_bits(x: (f64, f64), y: (f64, f64)) -> bool {
    x.0.to_bits() == y.0.to_bits() && x.1.to_bits() == y.1.to_bits()
}

#[test]
fn tgb_g01_recetas() {
    let bajos = precios(0.45, 0.013, 200); // por debajo del cruce 0.01 / 0.00005
    let altos = precios(60_000.0, 0.7, 200); // muy por encima: las dos recetas dan los mismos bytes
    let r_bajos = cinta(&bajos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let a_bajos = cinta(&bajos.iter().map(|p| ant(*p)).collect::<Vec<_>>());
    let r_altos = cinta(&altos.iter().map(|p| rel(*p)).collect::<Vec<_>>());
    let copia = r_bajos.clone();

    let f = firma(&r_bajos);
    assert_eq!(f.estado(), EstadoGeneracion::Relativa);
    assert_eq!((f.registros, f.solo_relativa, f.solo_antigua, f.ambas, f.ninguna, f.invalidos, f.sin_decision), (200, 200, 0, 0, 0, 0, 0));
    assert_eq!((f.primer_solo_relativa, f.primer_excluye_antigua, f.primer_excluye_relativa), (Some(0), Some(0), None));
    assert_eq!(f.recetas_compatibles(), "relativa");

    let f = firma(&a_bajos);
    assert_eq!(f.estado(), EstadoGeneracion::Antigua);
    assert_eq!((f.solo_antigua, f.solo_relativa, f.primer_solo_antigua, f.primer_excluye_relativa), (200, 0, Some(0), Some(0)));
    assert_eq!(f.recetas_compatibles(), "antigua");

    let f = firma(&r_altos);
    assert_eq!(f.estado(), EstadoGeneracion::Indeterminada, "donde las recetas coinciden");
    assert_eq!((f.ambas, f.solo_relativa, f.solo_antigua), (200, 0, 0));
    assert_eq!(f.recetas_compatibles(), "relativa,antigua");

    assert_eq!(r_bajos, copia, "la firma no altera los bytes");
}

#[test]
fn tgb_g02_cruce_y_vecinos() {
    // Cruce exacto c/r y sus vecinos f64 a AMBOS lados, la mitad y el doble del cruce. Oraculo: si las dos recetas
    // dan los mismos bits, el registro admite las dos (nunca se inventa exclusividad); si no, cada par es exclusivo
    // de su receta.
    let cruce = SUELO_ANTIGUO / R_RELATIVA;
    let mut ps = vec![cruce, cruce * 0.5, cruce * 2.0];
    let (mut abajo, mut arriba) = (cruce, cruce);
    for _ in 0..16 {
        abajo = abajo.next_down();
        arriba = arriba.next_up();
        ps.push(abajo);
        ps.push(arriba);
    }
    // f64 extremos validos: subnormal, normal minimo, muy pequeno y muy grande.
    ps.extend([f64::from_bits(1000), f64::MIN_POSITIVE, 1e-300, 1e300]);
    let (mut identicos, mut distintos) = (0, 0);
    for p in ps {
        let (r, a) = (rel(p), ant(p));
        let valido = |x: (f64, f64)| x.0.is_finite() && x.1.is_finite() && x.0 > 0.0 && x.0 <= x.1;
        if mismos_bits(r, a) {
            identicos += 1;
            assert_eq!(clase(r.0, r.1), (0, 0, 1, 0, 0), "p={p:e}: pares identicos deben admitir las dos");
        } else {
            distintos += 1;
            assert_eq!(clase(r.0, r.1), (1, 0, 0, 0, 0), "p={p:e}: par relativo distinto del antiguo");
            if valido(a) {
                assert_eq!(clase(a.0, a.1), (0, 1, 0, 0, 0), "p={p:e}: par antiguo distinto del relativo");
            }
        }
    }
    assert!(identicos >= 10 && distintos >= 4, "control: hay casos de los dos tipos ({identicos}, {distintos})");

    // Cruces de binade: justo por debajo de 1.0 el ask pasa a la binade de arriba y el punto medio calculado a veces
    // NO es p (desde unos 11 000 f64 por debajo). El testigo debe encontrarse igual (recorrido entero del intervalo, no solo el medio).
    let mut p = 1.0f64;
    let (mut medio_distinto, mut n) = (0, 0);
    for _ in 0..30_000 {
        p = p.next_down();
        let (b, a) = rel(p);
        if (b * 0.5 + a * 0.5).to_bits() != p.to_bits() {
            medio_distinto += 1;
        }
        n += 1;
        assert_eq!(clase(b, a), (1, 0, 0, 0, 0), "p={p:e} (binade): relativa exclusiva");
    }
    assert!(medio_distinto > 0, "control: el barrido contiene medios distintos de p ({medio_distinto} de {n})");
}

#[test]
fn tgb_g07_mixta_invalida_y_medio_tick() {
    let bajos = precios(0.45, 0.013, 200);
    let r_bajos: Vec<_> = bajos.iter().map(|p| rel(*p)).collect();
    let a_bajos: Vec<_> = bajos.iter().map(|p| ant(*p)).collect();

    let mut mezcla = r_bajos.clone();
    mezcla.extend_from_slice(&a_bajos);
    let f = firma(&cinta(&mezcla));
    assert_eq!(f.estado(), EstadoGeneracion::Mixta);
    assert_eq!((f.primer_solo_relativa, f.primer_solo_antigua), (Some(0), Some(200)));
    assert_eq!(f.recetas_compatibles(), "ninguna");

    // Precios no finitos, cruzados o no positivos.
    for malo in [(f64::NAN, 1.0), (1.0, f64::INFINITY), (2.0, 1.0), (0.0, 1.0), (-1.0, 1.0)] {
        let mut v = r_bajos.clone();
        v.push(malo);
        let f = firma(&cinta(&v));
        assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "precio invalido {malo:?}");
        assert_eq!((f.invalidos, f.primer_invalido), (1, Some(200)), "precio invalido {malo:?}");
    }
    // Cantidades invalidas: se validan crudas, sin saneamiento, aunque el precio sea relativo exacto.
    for (bq, aq) in [(f64::NAN, 1.0), (1.0, f64::INFINITY), (-1.0, 1.0), (1.0, -0.5)] {
        let mut v = cinta(&r_bajos);
        let (b, a) = rel(0.9);
        v.extend_from_slice(&registro_q(9, b, a, bq, aq));
        let copia = v.clone();
        let f = firma(&v);
        assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "cantidad invalida ({bq}, {aq})");
        assert_eq!(f.primer_invalido, Some(200), "cantidad invalida ({bq}, {aq})");
        assert_eq!(v, copia, "sin saneamiento");
    }
    // Ninguna receta: la transformacion de medio tick no se atribuye a ninguna de las dos.
    let fix: Vec<_> = bajos.iter().map(|p| medio_tick(*p, 0.001)).collect();
    let f = firma(&cinta(&fix));
    assert_eq!(f.estado(), EstadoGeneracion::OtraOInvalida, "medio tick");
    assert!(f.ninguna > 0 && f.invalidos == 0, "medio tick: ninguna receta, no invalido");

    // Cinta vacia (solo cabecera): indeterminada, sin registros. Truncada: error.
    let f = firma(b"TGMTICK1");
    assert_eq!((f.estado(), f.registros), (EstadoGeneracion::Indeterminada, 0));
    let mut truncada = cinta(&r_bajos[..3]);
    truncada.pop();
    assert!(firma_de_cinta(&truncada).is_err(), "registro truncado");
}
