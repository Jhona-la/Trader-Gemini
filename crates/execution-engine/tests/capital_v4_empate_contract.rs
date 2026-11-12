//! CAPITAL-UNICO V4: hallazgo N1 de la revision independiente de 2c3b06225
//! (`Desktop\PC-Turbo\integrador\lectura_revision_capital_v3.md`, seccion 4).
//!
//! Dos `ACCOUNT_UPDATE` con la misma `T` (dos fills de una orden en la misma
//! transaccion): en el stream el empate aplica el ultimo, pero solo si la
//! marca vigente de ese componente la puso el stream en esta misma conexion.
//! En REST el empate sigue siendo contradiccion. Tras una reconexion el
//! empate deja de aplicar (un evento repetido o atrasado no puede rebobinar).
//!
//! Tambien el recomendado de P1.3 (`caso_h`): una fila REST de posicion con
//! marca mas vieja y la misma cantidad no pisa el uPnL mas nuevo.
//!
//! No tocan d*, `capital_del_freno` ni ningun umbral.

use execution_engine::puente_cuenta::{
    DatoPosicion, EventoStream, LecturaCuenta, Observacion, PosicionRest, PuenteCuenta, Rechazo,
};

fn k(s: &str) -> (String, String) {
    (s.to_string(), "BOTH".to_string())
}

fn ab(cantidad: f64, upnl: f64) -> DatoPosicion {
    DatoPosicion::Abierta { cantidad, upnl }
}

fn rest(saldo: f64, t: Option<u64>, pos: &[(&str, DatoPosicion, Option<u64>)]) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: t,
        posiciones: Some(
            pos.iter()
                .map(|(s, d, t)| (k(s), PosicionRest { dato: *d, tiempo_ms: *t }))
                .collect(),
        ),
        rechazo_arranque: None,
        upnl_del_activo: None,
    }
}

fn ws(
    p: &PuenteCuenta,
    t: u64,
    saldo: Option<f64>,
    pos: &[(&str, DatoPosicion)],
) -> Result<Option<Observacion>, Rechazo> {
    p.observar_stream(
        p.era(),
        &EventoStream {
            tiempo_ms: Some(t),
            saldo,
            posiciones: pos.iter().map(|(s, d)| (k(s), *d)).collect(),
        },
    )
}

/// Puente arrancado con una lectura REST aceptada (equity completa, sin veto).
fn arrancado(saldo: f64, t: u64, pos: &[(&str, DatoPosicion, Option<u64>)]) -> PuenteCuenta {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(saldo, Some(t), pos)).unwrap();
    assert!(!p.veta_entradas());
    p
}

// ── N1: empate en el stream ──────────────────────────────────────────────

#[test]
fn caso_e_dos_fills_mismo_milisegundo() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, Some(99.0), &[("BTCUSDT", ab(0.5, 0.0))]).unwrap();
    let r2 = ws(&p, 200, Some(98.0), &[("BTCUSDT", ab(1.0, -0.5))]);
    assert_eq!(r2.unwrap().unwrap(), Observacion { saldo: 98.0, equity: 97.5 }, "el segundo fill aplica");
    assert!(!p.veta_entradas());
    // REST con la verdad y la misma marca 200: coherente, entra.
    let r = p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[("BTCUSDT", ab(1.0, -0.5), Some(200))]));
    assert_eq!(r.unwrap(), Observacion { saldo: 98.0, equity: 97.5 });
    assert!(!p.veta_entradas());
}

#[test]
fn caso_e2_cierre_en_dos_fills_mismo_ms_sin_posicion_fantasma_ni_veto() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -10.0), Some(100))]);
    ws(&p, 200, Some(95.0), &[("BTCUSDT", ab(0.5, -5.0))]).unwrap();
    let r2 = ws(&p, 200, Some(89.9), &[("BTCUSDT", DatoPosicion::Cerrada)]);
    assert_eq!(r2.unwrap().unwrap(), Observacion { saldo: 89.9, equity: 89.9 });
    assert_eq!(p.posiciones_en_cache(), 0, "sin posicion fantasma");
    let r = p.observar_rest(p.ticket(), &rest(89.9, Some(200), &[("BTCUSDT", DatoPosicion::Cerrada, Some(200))]));
    assert!(r.is_ok(), "{r:?}");
    assert!(!p.veta_entradas());
    assert_eq!(p.equity_del_freno(), Some(89.9));
}

#[test]
fn propuesta_empate_en_stream_aplica_el_ultimo() {
    let p = PuenteCuenta::new();
    ws(&p, 200, Some(0.0), &[]).unwrap();
    assert_eq!(ws(&p, 200, Some(7.0), &[]).unwrap().unwrap().saldo, 7.0);
    // Un atrasado del stream sigue rechazado.
    assert!(matches!(ws(&p, 150, Some(9.0), &[]), Err(Rechazo::Atrasada { .. })));
    assert_eq!(p.observacion().unwrap().saldo, 7.0);
}

#[test]
fn propuesta_stream_con_misma_marca_tras_rest_no_rebobina() {
    let p = PuenteCuenta::new();
    ws(&p, 100, Some(50.0), &[]).unwrap();
    p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap();
    let r = ws(&p, 200, Some(99.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 98.0);
}

#[test]
fn autor_p11_empate_rest_sigue_siendo_contradiccion() {
    let p = PuenteCuenta::new();
    ws(&p, 200, Some(0.0), &[]).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(100.0, Some(200), &[]));
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 0.0);
    assert_eq!(p.observar_rest(p.ticket(), &rest(0.0, Some(200), &[])).unwrap().saldo, 0.0);
}

#[test]
fn v4_tres_eventos_misma_t_99_98_97() {
    let p = arrancado(100.0, 100, &[]);
    for s in [99.0, 98.0, 97.0] {
        assert_eq!(ws(&p, 200, Some(s), &[]).unwrap().unwrap().saldo, s);
    }
    assert!(!p.veta_entradas());
    // REST con la verdad (97) y la misma marca: coherente.
    assert_eq!(p.observar_rest(p.ticket(), &rest(97.0, Some(200), &[])).unwrap().saldo, 97.0);
    // REST con un saldo intermedio (98) y la misma marca: contradiccion, no rebobina.
    let r = p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[]));
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 97.0);
}

#[test]
fn v4_evento_de_la_misma_t_despues_de_una_reconexion_no_aplica() {
    // Correccion adversaria a la propuesta: tras una reconexion el orden con
    // los eventos de la conexion anterior ya no esta garantizado. Un evento con
    // la misma T podria ser uno repetido o atrasado: no se aplica.
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, Some(99.0), &[]).unwrap();
    ws(&p, 200, Some(98.0), &[]).unwrap();
    p.reconexion(p.era()).unwrap();
    assert!(p.veta_entradas());
    // Un repetido del primer fill (99, T=200) no rebobina el saldo.
    let r = ws(&p, 200, Some(99.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.observacion().unwrap().saldo, 98.0);
    assert!(p.veta_entradas());
    // El mismo valor con la misma marca si vale (no modifica nada).
    assert!(ws(&p, 200, Some(98.0), &[]).is_ok());
    // ...pero ese igual no vuelve a habilitar el empate: el repetido sigue sin aplicar.
    assert!(matches!(ws(&p, 200, Some(99.0), &[]), Err(Rechazo::Contradiccion { .. })));
    assert_eq!(p.observacion().unwrap().saldo, 98.0);
    // Un evento mas nuevo de la conexion nueva aplica, y en ella el empate vuelve a valer.
    assert_eq!(ws(&p, 300, Some(97.0), &[]).unwrap().unwrap().saldo, 97.0);
    assert_eq!(ws(&p, 300, Some(96.0), &[]).unwrap().unwrap().saldo, 96.0);
    // La REST con la verdad suelta el veto.
    p.observar_rest(p.ticket(), &rest(96.0, Some(300), &[])).unwrap();
    assert!(!p.veta_entradas());
}

#[test]
fn v4_empate_de_posicion_tras_reconexion_no_reabre() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -10.0), Some(100))]);
    ws(&p, 200, None, &[("BTCUSDT", ab(0.5, -5.0))]).unwrap();
    ws(&p, 200, None, &[("BTCUSDT", DatoPosicion::Cerrada)]).unwrap();
    assert_eq!(p.posiciones_en_cache(), 0);
    p.reconexion(p.era()).unwrap();
    let r = ws(&p, 200, None, &[("BTCUSDT", ab(0.5, -5.0))]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    assert_eq!(p.posiciones_en_cache(), 0, "un repetido tras la reconexion no reabre la posicion");
}

// ── P1.3 recomendado (caso_h): uPnL de una fila REST con marca vieja ─────

#[test]
fn caso_h_rest_misma_cantidad_marca_vieja_no_pisa_el_upnl_mas_nuevo() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -5.0), Some(100))]);
    ws(&p, 200, None, &[("BTCUSDT", ab(1.0, -40.0))]).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(100.0, Some(100), &[("BTCUSDT", ab(1.0, -5.0), Some(100))]));
    assert_eq!(r.unwrap().equity, 60.0, "la perdida abierta mas nueva (-40) se conserva");
    assert!(!p.veta_entradas());
}

#[test]
fn caso_h_control_rest_con_marca_igual_o_mas_nueva_si_actualiza_el_upnl() {
    // Control que muerde: con la misma marca o una mas nueva el uPnL de REST entra.
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, -5.0), Some(100))]);
    ws(&p, 200, None, &[("BTCUSDT", ab(1.0, -40.0))]).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(100.0, Some(100), &[("BTCUSDT", ab(1.0, -30.0), Some(200))]));
    assert_eq!(r.unwrap().equity, 70.0);
    let r = p.observar_rest(p.ticket(), &rest(100.0, Some(100), &[("BTCUSDT", ab(1.0, -20.0), Some(300))]));
    assert_eq!(r.unwrap().equity, 80.0);
}
