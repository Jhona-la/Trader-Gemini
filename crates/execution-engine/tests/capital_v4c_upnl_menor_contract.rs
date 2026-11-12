//! CAPITAL-UNICO V4C: cambios recomendados por la lectura independiente de V4
//! (`lectura_revision_capital_v4.md`, F3) y el riesgo que señala en F2.
//!
//! F3: ante dos lecturas del PnL no realizado de la misma posición sin orden
//! demostrable entre ellas (la fila REST trae una marca más vieja que la `T`
//! del stream con la misma cantidad), el freno ve la MENOR. Si el orden sí es
//! demostrable (mismo socket con `E` o `T` menor, REST con marca más nueva,
//! `E` mayor), no se inventa una pérdida.
//!
//! Riesgo de F2: una REST con ticket posterior resuelve un empate de saldo con
//! la misma marca. Si Binance devolviera una foto atrasada (el saldo intermedio
//! entre dos fills de esa marca), esa foto no puede quedar como definitiva: si
//! no coincide con el último saldo que el stream entregó con esa marca, no
//! suelta el veto y el freno ve el menor de los dos.
//!
//! Solo usa `puente_cuenta` (se compila también suelto con `rustc --test`).
//! No toca d*, `capital_del_freno` ni ningún umbral.

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

fn ev(t: u64, saldo: Option<f64>, pos: &[(&str, DatoPosicion)]) -> EventoStream {
    EventoStream {
        tiempo_ms: Some(t),
        saldo,
        posiciones: pos.iter().map(|(s, d)| (k(s), *d)).collect(),
    }
}

/// Evento del socket vigente sin `E` (como en `adv4.rs` de la revisión).
fn ws0(p: &PuenteCuenta, t: u64, saldo: Option<f64>, pos: &[(&str, DatoPosicion)]) -> Result<Option<Observacion>, Rechazo> {
    p.observar_stream(p.era(), &ev(t, saldo, pos))
}

/// Evento del socket vigente con `T = t` y `E = e`.
fn ws(p: &PuenteCuenta, t: u64, e: u64, saldo: Option<f64>, pos: &[(&str, DatoPosicion)]) -> Result<Option<Observacion>, Rechazo> {
    p.observar_stream_del_socket(p.era(), p.socket(), Some(e), &ev(t, saldo, pos), |_| {})
}

fn arrancado(s: f64, t: u64, pos: &[(&str, DatoPosicion, Option<u64>)]) -> PuenteCuenta {
    let p = PuenteCuenta::new();
    p.observar_rest(p.ticket(), &rest(s, Some(t), pos)).unwrap();
    assert!(!p.veta_entradas());
    p
}

fn equity(p: &PuenteCuenta) -> f64 {
    p.observacion().unwrap().equity
}

fn cerca(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

// ── F3: sin orden demostrable, el freno ve el PnL no realizado menor ─────

/// `adv4.rs::r4_caso_h_rest_fresca_no_debe_perder_la_perdida_abierta`.
#[test]
fn v4c_f3_rest_fresca_no_pierde_la_perdida_abierta() {
    let p = arrancado(100.0, 100, &[]);
    ws0(&p, 200, Some(99.95), &[("BTCUSDT", ab(1.0, 0.0))]).unwrap();
    ws0(&p, 300, Some(99.9), &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    p.reconexion(p.era()).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(99.9, Some(300), &[("BTCUSDT", ab(1.0, -60.0), Some(200))]));
    println!("REST fresca: {r:?}; veto {}; freno {:?}", p.veta_entradas(), p.equity_del_freno());
    assert!(cerca(r.unwrap().equity, 39.9), "la perdida abierta de -60 debe llegar al freno");
    assert!(cerca(p.equity_del_freno().unwrap(), 39.9));
}

/// `adv4.rs::r4_caso_h_variante_marca_un_ms_menor`.
#[test]
fn v4c_f3_marca_de_fila_un_ms_menor() {
    let p = arrancado(100.0, 100, &[]);
    ws0(&p, 200, Some(99.95), &[("BTCUSDT", ab(1.0, -0.05))]).unwrap();
    p.reconexion(p.era()).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(99.95, Some(200), &[("BTCUSDT", ab(1.0, -60.0), Some(199))]));
    println!("REST 1 ms menor: {r:?}; freno {:?}", p.equity_del_freno());
    assert!(cerca(r.unwrap().equity, 39.95));
}

/// El `caso_h` del autor de V4 sigue igual: REST -5 con marca 100 tras el
/// stream -40 con marca 200; se queda el -40 (el menor).
#[test]
fn v4c_f3_caso_h_del_autor_sigue_en_60() {
    let p = arrancado(100.0, 100, &[("BTCUSDT", ab(1.0, 0.0), Some(100))]);
    ws0(&p, 200, Some(100.0), &[("BTCUSDT", ab(1.0, -40.0))]).unwrap();
    assert!(cerca(equity(&p), 60.0));
    p.reconexion(p.era()).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(100.0, Some(200), &[("BTCUSDT", ab(1.0, -5.0), Some(100))]));
    assert!(cerca(r.unwrap().equity, 60.0), "{:?}", p.observacion());
}

/// Lado contrario: la fila vieja trae ganancia; sin orden, la ganancia espera.
#[test]
fn v4c_f3_ganancia_sin_orden_espera() {
    let p = arrancado(100.0, 100, &[]);
    ws0(&p, 200, Some(99.95), &[("BTCUSDT", ab(1.0, 0.0))]).unwrap();
    ws0(&p, 300, Some(99.9), &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    p.reconexion(p.era()).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(99.9, Some(300), &[("BTCUSDT", ab(1.0, 20.0), Some(200))])).unwrap();
    assert!(cerca(r.equity, 89.9), "{r:?}");
}

// ── Controles: con orden demostrable no se inventa una pérdida ───────────

/// Mismo socket, misma `T`, `E` menor: es una regresión y se descarta; su
/// pérdida mayor NO llega al freno.
#[test]
fn v4c_control_mismo_socket_e_menor_no_inventa_perdida() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, 202, Some(99.9), &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    let r = ws(&p, 200, 201, None, &[("BTCUSDT", ab(1.0, -60.0))]);
    println!("E menor: {r:?}");
    assert!(cerca(equity(&p), 89.9), "{:?}", p.observacion());
    assert!(!p.veta_entradas());
}

/// Mismo socket, `T` menor tras una `T` mayor: lectura anterior, se descarta.
#[test]
fn v4c_control_mismo_socket_t_menor_no_inventa_perdida() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 300, 301, Some(99.9), &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    let _ = ws(&p, 250, 302, None, &[("BTCUSDT", ab(1.0, -60.0))]);
    assert!(cerca(equity(&p), 89.9), "{:?}", p.observacion());
}

/// Mismo socket, misma `T`, `E` mayor con MENOS pérdida: es posterior y se
/// aplica (no se queda el -60 viejo).
#[test]
fn v4c_control_mismo_socket_e_mayor_aplica_aunque_mejore() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, 201, Some(99.9), &[("BTCUSDT", ab(1.0, -60.0))]).unwrap();
    ws(&p, 200, 202, None, &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    assert!(cerca(equity(&p), 89.9), "{:?}", p.observacion());
}

/// REST con marca de fila MAS NUEVA que la del stream y ganancia: el orden es
/// demostrable, se aplica la ganancia (no el mínimo).
#[test]
fn v4c_control_rest_con_marca_nueva_aplica_la_ganancia() {
    let p = arrancado(100.0, 100, &[]);
    ws0(&p, 200, Some(99.9), &[("BTCUSDT", ab(1.0, -10.0))]).unwrap();
    p.reconexion(p.era()).unwrap();
    let r = p.observar_rest(p.ticket(), &rest(99.9, Some(200), &[("BTCUSDT", ab(1.0, 20.0), Some(300))])).unwrap();
    assert!(cerca(r.equity, 119.9), "{r:?}");
    assert!(!p.veta_entradas());
}

// ── Riesgo de F2: una foto REST atrasada con la misma marca ──────────────

/// Reconexión entre dos fills de la misma `T`: el segundo (98) llega por el
/// socket nuevo y abre el empate. Una REST posterior que devuelve la foto
/// atrasada (99, el saldo intermedio) no lo cierra; la verdad (98) sí.
#[test]
fn v4c_foto_atrasada_tras_reconexion_entre_fills_no_queda_definitiva() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, 201, Some(99.0), &[]).unwrap();
    p.reconexion(p.era()).unwrap();
    let r = ws(&p, 200, 202, Some(98.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    let r = p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[]));
    println!("foto atrasada 99@200: {r:?}; veto {}; freno {:?}", p.veta_entradas(), p.equity_del_freno());
    assert!(p.veta_entradas(), "un saldo intermedio no puede soltar el veto");
    assert!(p.equity_del_freno().unwrap() <= 98.0 + 1e-9, "el freno ve el menor");
    let r = p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[]));
    assert_eq!(r.unwrap().saldo, 98.0);
    assert!(!p.veta_entradas());
}

/// REST adelantada al stream entre dos fills (`adv4` F2) con la foto
/// atrasada después del segundo fill.
#[test]
fn v4c_foto_atrasada_tras_rest_adelantada_no_queda_definitiva() {
    let p = arrancado(100.0, 100, &[]);
    p.reconexion(p.era()).unwrap();
    p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[])).unwrap();
    ws(&p, 200, 201, Some(99.0), &[]).unwrap();
    let r = ws(&p, 200, 202, Some(98.0), &[]);
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    let _ = p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[]));
    assert!(p.veta_entradas());
    assert!(p.equity_del_freno().unwrap() <= 98.0 + 1e-9);
    p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap();
    assert_eq!(saldo(&p), 98.0);
    assert!(!p.veta_entradas());
}

/// REST igual entre dos fills (`adv4` F1) y después la foto atrasada.
#[test]
fn v4c_foto_atrasada_tras_rest_igual_entre_fills_no_queda_definitiva() {
    let p = arrancado(100.0, 100, &[]);
    p.reconexion(p.era()).unwrap();
    ws(&p, 200, 201, Some(99.0), &[]).unwrap();
    p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[])).unwrap();
    let _ = ws(&p, 200, 202, Some(98.0), &[]);
    let _ = p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[]));
    assert!(p.veta_entradas(), "saldo {:?}", p.observacion());
    assert!(p.equity_del_freno().unwrap() <= 98.0 + 1e-9);
    p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap();
    assert_eq!(saldo(&p), 98.0);
    assert!(!p.veta_entradas());
}

/// Dos fills de la misma `T` aplicados en el mismo socket (99 y luego 98 con
/// `E` mayor). La foto atrasada 99@200 abre un empate; si se repite (Binance
/// sigue atrasada), tampoco suelta el veto ni deja 99.
#[test]
fn v4c_foto_atrasada_repetida_no_pisa_el_ultimo_fill() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, 201, Some(99.0), &[]).unwrap();
    ws(&p, 200, 202, Some(98.0), &[]).unwrap();
    assert_eq!(saldo(&p), 98.0);
    let r = p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[]));
    assert!(matches!(r, Err(Rechazo::Contradiccion { .. })), "{r:?}");
    let r = p.observar_rest(p.ticket(), &rest(99.0, Some(200), &[]));
    println!("segunda foto atrasada: {r:?}; saldo {:?}; veto {}", p.observacion(), p.veta_entradas());
    assert!(p.veta_entradas());
    assert!(p.equity_del_freno().unwrap() <= 98.0 + 1e-9);
    p.observar_rest(p.ticket(), &rest(98.0, Some(200), &[])).unwrap();
    assert_eq!(saldo(&p), 98.0);
    assert!(!p.veta_entradas());
}

/// Control: con el empate abierto, una REST con marca MAS NUEVA no es una foto
/// atrasada: se aplica y suelta el veto aunque no coincida con el stream.
#[test]
fn v4c_control_rest_con_marca_nueva_cierra_el_empate() {
    let p = arrancado(100.0, 100, &[]);
    ws(&p, 200, 201, Some(99.0), &[]).unwrap();
    p.reconexion(p.era()).unwrap();
    let _ = ws(&p, 200, 202, Some(98.0), &[]);
    assert!(p.veta_entradas());
    p.observar_rest(p.ticket(), &rest(97.0, Some(300), &[])).unwrap();
    assert_eq!(saldo(&p), 97.0);
    assert!(!p.veta_entradas());
}

fn saldo(p: &PuenteCuenta) -> f64 {
    p.observacion().unwrap().saldo
}
