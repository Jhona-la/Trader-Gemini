//! QS-R4a — contrato del libro contrafactual en sombra (ver el módulo).

use audit_engine::shadow_ledger::{
    EstadisticaFuente, IntencionVetada, LibroSombra, Veredicto, CAPACIDAD, MAX_FUENTES,
};

const T0: u64 = 1_000_000;

fn intencion(fuente: u8, moneda: u16, es_largo: bool) -> IntencionVetada {
    IntencionVetada {
        fuente,
        moneda,
        es_largo,
        entrada: 100.0,
        tp_pct: 0.010,
        sl_pct: 0.005,
        friccion_pct: 0.001,
        tau_ms: 60_000,
        t0_ms: T0,
    }
}

fn stats(l: &LibroSombra, f: u8) -> EstadisticaFuente {
    *l.estadistica(f).unwrap()
}

#[test]
fn qs_r4_largo_toca_tp_y_cobra_r_neto_de_friccion() {
    let mut l = LibroSombra::new();
    assert!(l.registrar(intencion(4, 0, true)));
    assert_eq!(l.observar(0, 100.5, T0 + 1_000), 0);
    assert_eq!(l.observar(0, 101.0, T0 + 2_000), 1);
    let s = stats(&l, 4);
    assert_eq!((s.n, s.tp, s.sl, s.vencidas), (1, 1, 0, 0));
    // (0,010 − 0,001) / 0,005 = 1,8 R
    assert!((s.media_r().unwrap() - 1.8).abs() < 1e-12);
    assert_eq!(l.abiertas(), 0);
}

#[test]
fn qs_r4_corto_toca_sl() {
    let mut l = LibroSombra::new();
    assert!(l.registrar(intencion(4, 0, false)));
    assert_eq!(l.observar(0, 100.5, T0 + 1_000), 1);
    let s = stats(&l, 4);
    assert_eq!((s.n, s.tp, s.sl), (1, 0, 1));
    // (−0,005 − 0,001) / 0,005 = −1,2 R
    assert!((s.media_r().unwrap() + 1.2).abs() < 1e-12);
}

#[test]
fn qs_r4_vence_con_el_retorno_al_vencer() {
    let mut l = LibroSombra::new();
    assert!(l.registrar(intencion(4, 0, true)));
    assert_eq!(l.observar(0, 100.2, T0 + 59_999), 0);
    assert_eq!(l.observar(0, 100.2, T0 + 60_000), 1);
    let s = stats(&l, 4);
    assert_eq!(s.vencidas, 1);
    // (0,002 − 0,001) / 0,005 = 0,2 R
    assert!((s.media_r().unwrap() - 0.2).abs() < 1e-9);
}

/// Espejo: un corto sobre el camino reflejado alrededor de la entrada
/// resuelve igual que el largo sobre el camino original.
#[test]
fn qs_r4_simetria_espejo() {
    let camino = [100.1, 99.8, 100.4, 99.7, 100.6, 99.6, 100.9, 101.2];
    for corte in 1..=camino.len() {
        let mut a = LibroSombra::new();
        let mut b = LibroSombra::new();
        a.registrar(intencion(1, 0, true));
        b.registrar(intencion(1, 0, false));
        for (k, &p) in camino[..corte].iter().enumerate() {
            let t = T0 + 5_000 * (k as u64 + 1);
            a.observar(0, p, t);
            b.observar(0, 200.0 - p, t);
        }
        assert_eq!(stats(&a, 1), stats(&b, 1), "corte {corte}");
    }
}

#[test]
fn qs_r4_una_abierta_por_fuente_moneda_y_lado() {
    let mut l = LibroSombra::new();
    assert!(l.registrar(intencion(2, 7, true)));
    assert!(!l.registrar(intencion(2, 7, true)));
    assert_eq!(l.descartadas_solape, 1);
    assert!(l.registrar(intencion(2, 7, false)), "el otro lado es otra intención");
    assert!(l.registrar(intencion(3, 7, true)), "otra fuente es otra intención");
    assert!(l.registrar(intencion(2, 8, true)), "otra moneda es otra intención");
    assert_eq!(l.abiertas(), 4);
}

#[test]
fn qs_r4_memoria_fija_descarta_y_cuenta_al_llenarse() {
    let mut l = LibroSombra::new();
    for m in 0..CAPACIDAD as u16 {
        assert!(l.registrar(intencion(0, m, true)));
    }
    assert!(!l.registrar(intencion(0, CAPACIDAD as u16, true)));
    assert_eq!(l.descartadas_llenas, 1);
    // Al resolverse una, vuelve a haber sitio.
    assert_eq!(l.observar(3, 101.0, T0 + 1), 1);
    assert!(l.registrar(intencion(0, CAPACIDAD as u16, true)));
}

#[test]
fn qs_r4_entradas_invalidas_se_descartan() {
    let mut l = LibroSombra::new();
    let base = intencion(0, 0, true);
    let malas = [
        IntencionVetada { entrada: f64::NAN, ..base },
        IntencionVetada { entrada: 0.0, ..base },
        IntencionVetada { sl_pct: 0.0, ..base },
        IntencionVetada { tp_pct: f64::INFINITY, ..base },
        IntencionVetada { friccion_pct: -0.001, ..base },
        IntencionVetada { tau_ms: 0, ..base },
        IntencionVetada { fuente: MAX_FUENTES as u8, ..base },
    ];
    for m in malas {
        assert!(!l.registrar(m), "{m:?}");
    }
    assert_eq!(l.descartadas_invalidas, malas.len() as u64);
    assert_eq!(l.abiertas(), 0);
}

#[test]
fn qs_r4_precios_ajenos_o_invalidos_no_resuelven() {
    let mut l = LibroSombra::new();
    l.registrar(intencion(0, 0, true));
    assert_eq!(l.observar(1, 200.0, T0 + 1), 0, "otra moneda");
    assert_eq!(l.observar(0, f64::NAN, T0 + 1), 0);
    assert_eq!(l.observar(0, -1.0, T0 + 1), 0);
    assert_eq!(l.observar(0, 101.0, T0 - 1), 0, "precio anterior al veto");
    assert_eq!(l.abiertas(), 1);
}

#[test]
fn qs_r4_veredicto_por_fuente() {
    let mut l = LibroSombra::new();
    // Fuente 5: todo lo que bloqueó habría tocado SL ⇒ ahorra dinero.
    // Fuente 6: todo habría tocado TP ⇒ bloquea ganadoras.
    // Fuente 7: mitad TP / mitad SL con RR 2 y fricción ⇒ media
    //   (1,8 − 1,2)/2 = +0,3 R con dispersión 1,5 R: hacen falta muchas.
    for k in 0..40u16 {
        let t = T0 + 1_000 * k as u64;
        let i = |f| IntencionVetada { t0_ms: t, ..intencion(f, k, true) };
        l.registrar(i(5));
        l.observar(k, 99.0, t + 1);
        l.registrar(i(6));
        l.observar(k, 101.0, t + 1);
        l.registrar(i(7));
        l.observar(k, if k % 2 == 0 { 101.0 } else { 99.0 }, t + 1);
    }
    assert_eq!(stats(&l, 5).veredicto(1.64, 30), Veredicto::AhorraDinero);
    assert_eq!(stats(&l, 6).veredicto(1.64, 30), Veredicto::BloqueaGanadoras);
    // Media +0,3, error estándar 1,5/√40·√(40/39) ≈ 0,24: z = 1,64 no la separa.
    assert_eq!(stats(&l, 7).veredicto(1.64, 30), Veredicto::SinEvidencia);
    // Sin muestra suficiente o z inválido, nunca hay veredicto.
    assert_eq!(stats(&l, 5).veredicto(1.64, 41), Veredicto::SinEvidencia);
    assert_eq!(stats(&l, 5).veredicto(f64::NAN, 30), Veredicto::SinEvidencia);
    assert_eq!(stats(&l, 5).veredicto(0.0, 30), Veredicto::SinEvidencia);
    assert_eq!(stats(&l, 9).veredicto(1.64, 1), Veredicto::SinEvidencia);
}
