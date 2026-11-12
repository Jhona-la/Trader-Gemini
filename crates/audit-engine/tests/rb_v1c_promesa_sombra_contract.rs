//! CONTRATO RB-V1c (Codex RB, 2026-11-10): promesa frente a realidad.
//!
//! QUÉ: cada intención en sombra puede llevar el EV neto que el riesgo le
//! prometió, en R (`promesa_r`). Por fuente, el libro acumula la brecha
//! realizado − prometido y un e-proceso de Ville sobre
//! `x = (r − promesa)/ancho` (la promesa es fija al registrar, así que la
//! brecha conserva el ancho de `r`). `veredicto_promesa` certifica en
//! cualquier momento si el riesgo promete de más o de menos.
//! POR QUÉ: la puerta de EV del riesgo usa un modelo binario (todo termina en
//! TP o en SL) con la probabilidad de acertar la dirección. Si su promesa no se
//! cumple, valida operaciones que no pagan la fricción; esto lo mide sin tocar
//! ninguna decisión.
//! CÓMO: (1) forma cerrada de la brecha y de su evidencia; (2) una promesa
//! optimista se certifica en el n que predice la forma cerrada; (3) una
//! promesa calibrada no produce veredicto falso mirando en cada resolución;
//! (4) sin promesa no cuenta, y una promesa no finita se descarta.

use audit_engine::shadow_ledger::{
    EProcesoVille, EstadisticaFuente, IntencionVetada, LibroSombra, Promesa, APUESTAS,
    REJILLA_APUESTAS,
};

const T0: u64 = 1_000_000;
const TP: f64 = 0.010;
const SL: f64 = 0.005;
const FRICCION: f64 = 0.001;
/// α por cola de producción (0,05 entre las 11 fuentes y sus dos colas).
const ALFA_PRODUCCION: f64 = 0.05 / 22.0;

fn intencion(t0_ms: u64, promesa_r: Option<f64>) -> IntencionVetada {
    IntencionVetada {
        fuente: 0,
        moneda: 0,
        es_largo: true,
        entrada: 100.0,
        tp_pct: TP,
        sl_pct: SL,
        friccion_pct: FRICCION,
        tau_ms: 60_000,
        t0_ms,
        promesa_r,
    }
}

/// Registra y resuelve al precio `precio` en `t0 + espera`.
fn resolver(l: &mut LibroSombra, i: IntencionVetada, precio: f64, espera: u64) {
    assert!(l.registrar(i));
    assert_eq!(l.observar(0, precio, i.t0_ms + espera), 1);
}

fn estadistica(l: &LibroSombra) -> EstadisticaFuente {
    match l.estadistica(0) {
        Some(s) => *s,
        None => panic!("la fuente 0 existe"),
    }
}

fn evidencia_directa(xs: &[f64], signo: f64) -> f64 {
    let suma: f64 = REJILLA_APUESTAS
        .iter()
        .map(|&u| {
            xs.iter()
                .map(|&x| signo * u * x - u * u / 8.0)
                .sum::<f64>()
                .exp()
        })
        .sum();
    (suma / APUESTAS as f64).ln()
}

/// xorshift64* determinista.
struct Xorshift(u64);
impl Xorshift {
    fn uniforme(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn rb_v1c_la_brecha_tiene_forma_cerrada() {
    // Ancho (tp+sl)/sl = 3. TP: r = 1,8; SL: r = −1,2; vence a 100,2: r = 0,2.
    let mut l = LibroSombra::new();
    resolver(&mut l, intencion(T0, Some(0.5)), 101.0, 1);
    resolver(&mut l, intencion(T0 + 10, Some(0.5)), 99.0, 1);
    resolver(&mut l, intencion(T0 + 20, Some(0.9)), 100.2, 60_000);
    resolver(&mut l, intencion(T0 + 30, None), 99.0, 1);
    let s = estadistica(&l);
    assert_eq!(
        (s.n, s.n_promesa),
        (4, 3),
        "la intención sin promesa no cuenta en la brecha"
    );
    assert!((s.suma_promesa_r - 1.9).abs() < 1e-12);
    assert!((s.suma_r_prometida - 0.8).abs() < 1e-12, "1,8 − 1,2 + 0,2");
    let Some(brecha) = s.brecha_media_r() else {
        panic!("hay resoluciones con promesa");
    };
    assert!((brecha - (0.8 - 1.9) / 3.0).abs() < 1e-12);
    assert!(s
        .promesa_media_r()
        .is_some_and(|p| (p - 1.9 / 3.0).abs() < 1e-12));

    let xs = [(1.8 - 0.5) / 3.0, (-1.2 - 0.5) / 3.0, (0.2 - 0.9) / 3.0];
    let (menos, mas) = s.log_evidencia_promesa();
    let (menos_esperada, mas_esperada) =
        (evidencia_directa(&xs, 1.0), evidencia_directa(&xs, -1.0));
    println!("observado: ln M⁺ {menos:.12} frente a {menos_esperada:.12}; ln M⁻ {mas:.12} frente a {mas_esperada:.12}");
    assert!((menos - menos_esperada).abs() < 1e-12);
    assert!((mas - mas_esperada).abs() < 1e-12);
}

#[test]
fn rb_v1c_una_promesa_optimista_se_certifica_en_el_n_teorico() {
    // Todo SL (r = −1,2) con una promesa de +0,5 R: x = −1,7/3 en cada una.
    let x = -1.7 / 3.0;
    let umbral = (1.0 / ALFA_PRODUCCION).ln();
    let ln_m = |n: f64| {
        let suma: f64 = REJILLA_APUESTAS
            .iter()
            .map(|&u| (n * (-u * x - u * u / 8.0)).exp())
            .sum();
        (suma / APUESTAS as f64).ln()
    };
    let Some(n_teorico) = (1..200u64).find(|&n| ln_m(n as f64) >= umbral) else {
        panic!("la promesa optimista se certifica antes de 200 resoluciones");
    };
    let mut l = LibroSombra::new();
    let mut certificado_en = None;
    for k in 0..200 {
        resolver(&mut l, intencion(T0 + 10 * k, Some(0.5)), 99.0, 1);
        if certificado_en.is_none()
            && estadistica(&l).veredicto_promesa(ALFA_PRODUCCION) == Promesa::PrometeDeMas
        {
            certificado_en = Some(k + 1);
        }
    }
    println!("observado: promesa optimista certificada en n = {certificado_en:?}; forma cerrada n = {n_teorico}");
    assert_eq!(certificado_en, Some(n_teorico));
}

#[test]
fn rb_v1c_una_promesa_calibrada_no_da_veredicto_mirando_en_cada_resolucion() {
    // p(TP) = (sl+f)/(tp+sl) = 0,4 ⇒ E[r] = 0,4·1,8 − 0,6·1,2 = 0: la promesa
    // calibrada es 0. Se mira en cada resolución, como hace el núcleo.
    let p_tp = (SL + FRICCION) / (TP + SL);
    let (replicas, pasos) = (400u32, 600u64);
    let mut rng = Xorshift(0x2545_F491_4F6C_DD1D);
    let mut falsos = 0u32;
    for _ in 0..replicas {
        let mut l = LibroSombra::new();
        let mut falso = false;
        for k in 0..pasos {
            let precio = if rng.uniforme() < p_tp { 101.0 } else { 99.0 };
            resolver(&mut l, intencion(T0 + 10 * k, Some(0.0)), precio, 1);
            falso |= estadistica(&l).veredicto_promesa(ALFA_PRODUCCION) != Promesa::SinEvidencia;
        }
        falsos += u32::from(falso);
    }
    let tasa = f64::from(falsos) / f64::from(replicas);
    let cota = 2.0 * ALFA_PRODUCCION
        + 3.0
            * (2.0 * ALFA_PRODUCCION * (1.0 - 2.0 * ALFA_PRODUCCION) / f64::from(replicas)).sqrt();
    println!("observado ({replicas} réplicas × {pasos} miradas): veredicto falso sobre una promesa calibrada {tasa:.4} (≤ {cota:.4})");
    assert!(tasa <= cota);
}

#[test]
fn rb_v1c_sin_promesa_no_cuenta_y_una_promesa_no_finita_se_descarta() {
    let mut l = LibroSombra::new();
    for mala in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(!l.registrar(intencion(T0, Some(mala))), "promesa {mala}");
    }
    assert_eq!(l.descartadas_invalidas, 3);
    resolver(&mut l, intencion(T0, None), 101.0, 1);
    let s = estadistica(&l);
    assert_eq!((s.n, s.n_promesa), (1, 0));
    assert_eq!(s.brecha_media_r(), None);
    assert_eq!(s.veredicto_promesa(ALFA_PRODUCCION), Promesa::SinEvidencia);

    let e = EProcesoVille::default();
    assert_eq!(
        e.log_evidencia(),
        (0.0, 0.0),
        "sin observaciones la riqueza es 1"
    );
    for malo in [0.0, 1.0, -0.5, f64::NAN] {
        assert_eq!(e.signo(malo), None, "α = {malo}");
    }
}
