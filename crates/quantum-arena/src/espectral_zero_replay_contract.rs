//! Contratos causales del IC cruzado: ceros neutrales y maduraciones únicas.
//! Son hijos del módulo para comparar todo su estado mutable sin ampliar la API.
use super::*;
use crate::evalues::{ALFA_SIGNIFICANCIA, LAMBDA_APOSTADA};

const K: usize = 19;
const TAU: f64 = 1_000.0;

#[derive(Debug, PartialEq, Eq)]
struct IcBits {
    ea2: u64,
    eb2: u64,
    eab: u64,
    n: u64,
    capital: u64,
    observaciones: u64,
}

#[derive(Debug, PartialEq, Eq)]
struct EnlaceBits {
    par: (u16, u16),
    consumido: [(u64, u64); ESCALAS],
    escalas: [IcBits; ESCALAS],
}

#[derive(Debug, PartialEq, Eq)]
struct EstadoBits {
    max_coins: usize,
    ultimo: Vec<[(u64, u64); ESCALAS]>,
    enlaces: Vec<EnlaceBits>,
}

fn estado_bits(uni: &EspectralMultiactivo) -> EstadoBits {
    let mut enlaces: Vec<_> = uni
        .enlaces
        .iter()
        .map(|(&par, enlace)| EnlaceBits {
            par,
            consumido: enlace.consumido,
            escalas: std::array::from_fn(|k| {
                let ic = enlace.escalas[k];
                IcBits {
                    ea2: ic.ea2.to_bits(),
                    eb2: ic.eb2.to_bits(),
                    eab: ic.eab.to_bits(),
                    n: ic.n,
                    capital: ic.e.capital().to_bits(),
                    observaciones: ic.e.n(),
                }
            }),
        })
        .collect();
    enlaces.sort_unstable_by_key(|enlace| enlace.par);
    EstadoBits {
        max_coins: uni.max_coins,
        ultimo: uni
            .ultimo
            .iter()
            .map(|moneda| std::array::from_fn(|k| (moneda[k].0, moneda[k].1.to_bits())))
            .collect(),
        enlaces,
    }
}

fn par(uni: &mut EspectralMultiactivo, ordinal: u64, ra: f64, rb: f64, b_primero: bool) {
    let ts = (ordinal + 1) * 1_000;
    let eventos = if b_primero { [(1, rb), (0, ra)] } else { [(0, ra), (1, rb)] };
    for (moneda, retorno) in eventos {
        uni.observar_maduracion(moneda, TAU, K, ts, retorno);
    }
}

fn correlacionados(uni: &mut EspectralMultiactivo, desde: u64, hasta: u64, b_primero: bool) {
    for ordinal in desde..hasta {
        let r = if ordinal % 2 == 0 { 0.02 } else { -0.02 };
        par(uni, ordinal, r, r, b_primero);
    }
}

#[test]
fn ceros_con_ambos_signos_no_apuestan_tras_semillas_positiva_y_negativa() {
    let ceros = [
        (0.0, 0.0),
        (0.0, -0.0),
        (-0.0, 0.0),
        (-0.0, -0.0),
        (0.0, 0.02),
        (0.0, -0.02),
        (-0.0, 0.02),
        (-0.0, -0.02),
        (0.02, 0.0),
        (0.02, -0.0),
        (-0.02, 0.0),
        (-0.02, -0.0),
    ];
    for semilla_b in [0.02, -0.02] {
        for &(ra, rb) in &ceros {
            let mut uni = EspectralMultiactivo::new(2);
            par(&mut uni, 0, 0.02, semilla_b, false);
            let capital_semilla = uni.enlaces[&(0, 1)].escalas[K].e.capital().to_bits();
            for ordinal in 1..113 {
                par(&mut uni, ordinal, ra, rb, ordinal % 2 == 0);
                let ic = &uni.enlaces[&(0, 1)].escalas[K];
                assert_eq!(
                    ic.e.capital().to_bits(),
                    capital_semilla,
                    "cero no apuesta: semilla_b={semilla_b}, ra_bits={:x}, rb_bits={:x}, ordinal={ordinal}",
                    ra.to_bits(),
                    rb.to_bits(),
                );
                assert_eq!(ic.n, ordinal + 1, "los bloques cero siguen siendo observaciones");
                assert_eq!(ic.e.n(), ic.n);
            }
            assert_eq!(uni.muestras_par(0, 1, K), Some(113));
            assert!(uni.coherencia_par(0, 1, K).is_some(), "la telemetría conserva los momentos");
            assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
            assert_eq!(uni.coherencia_media_con_todas(0, K), None);
        }
    }
}

#[test]
fn ceros_desde_estado_frio_no_fabrican_capital_ni_coherencia() {
    for (ra, rb) in [(0.0, 0.0), (0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0)] {
        let mut uni = EspectralMultiactivo::new(2);
        for ordinal in 0..120 {
            par(&mut uni, ordinal, ra, rb, ordinal % 2 == 0);
        }
        let ic = &uni.enlaces[&(0, 1)].escalas[K];
        assert_eq!(ic.e.capital().to_bits(), 1.0_f64.to_bits());
        assert_eq!(ic.n, 120);
        assert_eq!(ic.e.n(), 120);
        assert_eq!(uni.coherencia_par(0, 1, K), None);
        assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
    }
}

#[test]
fn control_positivo_cruza_solo_con_113_pares_informativos_unicos() {
    assert_eq!(FAMILIA_VETO_GRUPO, 2_175);
    assert_eq!(ALFA_SIGNIFICANCIA, 0.05);
    assert_eq!(LAMBDA_APOSTADA, 0.10);
    let umbral = EProceso::umbral_familia(FAMILIA_VETO_GRUPO);
    assert_eq!(umbral, 43_500.0);
    for b_primero in [false, true] {
        let mut uni = EspectralMultiactivo::new(2);
        correlacionados(&mut uni, 0, 112, b_primero);
        assert_eq!(uni.muestras_par(0, 1, K), Some(112));
        assert!(uni.coherencia_par(0, 1, K).unwrap() > 0.99);
        assert!(uni.enlaces[&(0, 1)].escalas[K].e.capital() < umbral);
        assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
        correlacionados(&mut uni, 112, 113, b_primero);
        assert_eq!(uni.muestras_par(0, 1, K), Some(113));
        assert!(uni.enlaces[&(0, 1)].escalas[K].e.capital() >= umbral);
        assert!(uni.coherencia_par_veto(0, 1, K).unwrap() > 0.99);
        assert!(uni.coherencia_media_con_todas(0, K).unwrap() > 0.99);
    }
}

#[test]
fn ceros_no_aceleran_el_cruce_y_evidencia_fresca_aun_puede_cruzar() {
    let mut uni = EspectralMultiactivo::new(2);
    par(&mut uni, 0, 0.02, 0.02, false);
    for ordinal in 1..113 {
        par(&mut uni, ordinal, 0.0, 0.0, false);
    }
    assert_eq!(uni.muestras_par(0, 1, K), Some(113));
    assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
    correlacionados(&mut uni, 113, 224, false);
    assert_eq!(uni.muestras_par(0, 1, K), Some(224));
    assert_eq!(uni.coherencia_par_veto(0, 1, K), None, "112 pares informativos totales");
    correlacionados(&mut uni, 224, 225, false);
    assert_eq!(uni.muestras_par(0, 1, K), Some(225));
    assert!(uni.coherencia_par_veto(0, 1, K).unwrap() > 0.99, "113 pares informativos totales");
}

fn comprobar_replay_y_continuacion(b_primero: bool) {
    let mut uni = EspectralMultiactivo::new(4);
    correlacionados(&mut uni, 0, 110, b_primero);
    // Otro par y otra escala quedan incluidos en la comparación completa.
    uni.observar_maduracion(2, TAU, K + 1, 1_000, 0.01);
    uni.observar_maduracion(3, TAU, K + 1, 1_000, -0.03);
    let mut control = uni.clone();
    let antes = estado_bits(&uni);
    assert_eq!(uni.muestras_par(0, 1, K), Some(110));
    assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
    // Igualdad con payload distinto, retroceso inmediato e historia distante.
    for ordinal in [109, 108, 0, 39, 73] {
        let ts = (ordinal + 1) * 1_000;
        let eventos = if b_primero { [(1, -0.7), (0, 0.4)] } else { [(0, 0.4), (1, -0.7)] };
        for (moneda, retorno) in eventos {
            uni.observar_maduracion(moneda, TAU, K, ts, retorno);
            assert_eq!(
                estado_bits(&uni),
                antes,
                "replay debe abstenerse antes de cualquier escritura: moneda={moneda}, ts={ts}, b_primero={b_primero}",
            );
        }
    }
    // La continuación válida coincide bit a bit con no haber recibido replay.
    for ordinal in 110..120 {
        correlacionados(&mut uni, ordinal, ordinal + 1, b_primero);
        correlacionados(&mut control, ordinal, ordinal + 1, b_primero);
        assert_eq!(estado_bits(&uni), estado_bits(&control));
        assert_eq!(uni.muestras_par(0, 1, K), Some(ordinal + 1));
        if ordinal < 112 {
            assert_eq!(uni.coherencia_par_veto(0, 1, K), None);
        } else {
            assert!(uni.coherencia_par_veto(0, 1, K).unwrap() > 0.99);
        }
    }
}

#[test]
fn replay_historico_no_muta_y_continua_a_antes_de_b() {
    comprobar_replay_y_continuacion(false);
}

#[test]
fn replay_historico_no_muta_y_continua_b_antes_de_a() {
    comprobar_replay_y_continuacion(true);
}

#[test]
fn frescura_es_local_por_moneda_y_escala() {
    let mut uni = EspectralMultiactivo::new(2);
    // El cierre de B puede ser anterior al de A, siendo nuevo para B.
    uni.observar_maduracion(0, TAU, K, 2_000, 0.02);
    uni.observar_maduracion(1, TAU, K, 1_500, 0.02);
    assert_eq!(uni.muestras_par(0, 1, K), Some(1));
    // Otra escala conserva su reloj, incluso si su ts es menor.
    uni.observar_maduracion(0, TAU, K + 1, 1_000, 0.02);
    uni.observar_maduracion(1, TAU, K + 1, 1_000, 0.02);
    assert_eq!(uni.muestras_par(0, 1, K + 1), Some(1));
    assert_eq!(uni.muestras_par(0, 1, K), Some(1));
}
