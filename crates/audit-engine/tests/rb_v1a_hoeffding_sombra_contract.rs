//! CONTRATO RB-V1a (Codex RB, 2026-11-10): el veredicto del libro en sombra
//! con la desigualdad de Hoeffding sobre el ancho CONOCIDO de cada resultado.
//!
//! QUÉ: cada intención vetada resuelve un r neto de fricción en
//! `[−(sl+f)/sl, (tp−f)/sl]`, de ancho `(tp+sl)/sl` fijado al registrarla.
//! Hoeffding para variables acotadas no idénticas da, con probabilidad
//! ≥ 1 − δ por cola, `|media − E[r]| ≤ √(ln(1/δ)·Σ ancho²/2)/n` para TODO n.
//! POR QUÉ: el veredicto z·SE es asintótico y exige un `n_min` arbitrario; el
//! juicio de un veto que estrangula la Demo no puede depender de él.
//! CÓMO: (1) forma cerrada del ancho y del radio; (2) veredictos sobre
//! poblaciones conocidas; (3) cobertura Monte Carlo con un juego justo: la tasa
//! de veredictos falsos queda bajo 2δ; (4) potencia: un veto que sólo bloquea
//! perdedoras se certifica con pocas muestras.

use audit_engine::shadow_ledger::{IntencionVetada, LibroSombra, Veredicto};

const T0: u64 = 1_000_000;
const TP: f64 = 0.010;
const SL: f64 = 0.005;
const FRICCION: f64 = 0.001;

fn intencion(fuente: u8, moneda: u16, t0_ms: u64) -> IntencionVetada {
    IntencionVetada {
        fuente,
        moneda,
        es_largo: true,
        entrada: 100.0,
        tp_pct: TP,
        sl_pct: SL,
        friccion_pct: FRICCION,
        tau_ms: 60_000,
        t0_ms,
    }
}

/// Registra y resuelve en el acto una intención por TP (`gana`) o por SL.
fn resolver(l: &mut LibroSombra, fuente: u8, k: u64, gana: bool) {
    let t = T0 + 10 * k;
    assert!(l.registrar(intencion(fuente, 0, t)));
    let precio = if gana { 101.0 } else { 99.0 };
    assert_eq!(l.observar(0, precio, t + 1), 1);
}

/// xorshift64* determinista: la prueba no depende de un generador externo.
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
fn rb_v1a_ancho_y_radio_tienen_forma_cerrada() {
    let mut l = LibroSombra::new();
    for k in 0..4 {
        resolver(&mut l, 3, k, true);
    }
    let s = match l.estadistica(3) {
        Some(s) => *s,
        None => panic!("la fuente 3 existe"),
    };
    let ancho = (TP + SL) / SL; // 3
    assert!((s.suma_ancho2 - 4.0 * ancho * ancho).abs() < 1e-12);
    let delta: f64 = 0.0125;
    let esperado = ((1.0 / delta).ln() * 4.0 * ancho * ancho / 2.0).sqrt() / 4.0;
    match s.radio_hoeffding(delta) {
        Some(r) => assert!(
            (r - esperado).abs() < 1e-12,
            "radio {r} frente a {esperado}"
        ),
        None => panic!("con n = 4 el radio existe"),
    }
    // Dominio: sin muestras o con δ fuera de (0, 1) no hay radio.
    let vacia = match LibroSombra::new().estadistica(0) {
        Some(s) => *s,
        None => panic!("la fuente 0 existe"),
    };
    assert_eq!(vacia.radio_hoeffding(delta), None);
    for malo in [0.0, 1.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_eq!(s.radio_hoeffding(malo), None, "δ = {malo}");
        assert_eq!(s.veredicto_hoeffding(malo), Veredicto::SinEvidencia);
    }
}

#[test]
fn rb_v1a_veredicto_hoeffding_sobre_poblaciones_conocidas() {
    let mut l = LibroSombra::new();
    for k in 0..40 {
        resolver(&mut l, 5, k, false); // todo SL: −1,2 R
        resolver(&mut l, 6, k, true); // todo TP: +1,8 R
        resolver(&mut l, 7, k, k % 2 == 0); // mitad y mitad: +0,3 R
    }
    let delta = 0.0125;
    let v = |f: u8| match l.estadistica(f) {
        Some(s) => s.veredicto_hoeffding(delta),
        None => panic!("fuente {f}"),
    };
    assert_eq!(v(5), Veredicto::AhorraDinero);
    assert_eq!(v(6), Veredicto::BloqueaGanadoras);
    // Radio ≈ 0,70 R con n = 40: no separa +0,3 R de 0.
    assert_eq!(v(7), Veredicto::SinEvidencia);
}

#[test]
fn rb_v1a_cobertura_con_un_juego_justo_queda_bajo_dos_delta() {
    // E[r] = 0 exacto: p·(tp−f)/sl = (1−p)·(sl+f)/sl ⇒ p = (sl+f)/(tp+sl) = 0,4.
    let p_tp = (SL + FRICCION) / (TP + SL);
    let delta = 0.0125;
    let replicas = 2_000;
    let n = 30;
    let mut rng = Xorshift(0x9E37_79B9_7F4A_7C15);
    let mut falsos = 0u32;
    for _ in 0..replicas {
        let mut l = LibroSombra::new();
        for k in 0..n {
            resolver(&mut l, 0, k, rng.uniforme() < p_tp);
        }
        let s = match l.estadistica(0) {
            Some(s) => *s,
            None => panic!("fuente 0"),
        };
        if s.veredicto_hoeffding(delta) != Veredicto::SinEvidencia {
            falsos += 1;
        }
    }
    let tasa = falsos as f64 / replicas as f64;
    let cota = 2.0 * delta + 3.0 * (2.0 * delta * (1.0 - 2.0 * delta) / replicas as f64).sqrt();
    assert!(
        tasa <= cota,
        "veredictos falsos {tasa:.4} con un juego justo; la garantía es ≤ 2δ = {:.4}",
        2.0 * delta
    );
}

#[test]
fn rb_v1a_un_veto_que_solo_bloquea_perdedoras_se_certifica_pronto() {
    // Radio con todo SL: √(ln(1/δ)·9n/2)/n < 1,2 ⇔ n > ln(1/δ)·4,5/1,44.
    let delta: f64 = 0.0125;
    let n_teorico = ((1.0 / delta).ln() * 4.5 / 1.44).ceil() as u64;
    let mut l = LibroSombra::new();
    let mut certificado_en = None;
    for k in 0..60 {
        resolver(&mut l, 9, k, false);
        let s = match l.estadistica(9) {
            Some(s) => *s,
            None => panic!("fuente 9"),
        };
        if certificado_en.is_none() && s.veredicto_hoeffding(delta) == Veredicto::AhorraDinero {
            certificado_en = Some(k + 1);
        }
    }
    assert_eq!(
        certificado_en,
        Some(n_teorico),
        "la certificación llega exactamente en el n que predice la forma cerrada"
    );
}
