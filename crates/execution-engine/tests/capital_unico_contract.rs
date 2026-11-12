//! CAPITAL-UNICO (prioridad 1 del mapa de pérdidas, 2026-11-11) — lo que usa
//! la API nueva. La RED contra `main` está en `capital_unico_red_contract.rs`.
//!
//! - Aviso de otros activos y del modo multi-activo; fallo cerrado sin USDT.
//! - Revisión independiente 3a: saldo de USDT 0 con fondos en otros activos = `Err`.
//! - Opción A del pico: `pico_capital_{venue}_usdt.txt` nace vacío; el archivo
//!   viejo no se adopta y queda intacto byte a byte.
//! - Tope `TG_CAPITAL_OPERATIVO` en el mismo puente para REST y stream.
//!
//! No tocan d*, `capital_del_freno` ni ningún umbral.

use execution_engine::capital_operativo::{
    decidir_pico, guardar_anotacion, interpretar_tope, leer_anotacion, ruta_anotacion, Anotacion,
    DecisionPico, PuenteCapital, VARIABLE_TOPE,
};
use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::{parse_account_capital, parse_account_equity};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico, FrenoDuro};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;

/// `/fapi/v2/account` de una cuenta Testnet con varios activos (USDT, USDC, BTC).
const CUENTA_MULTI: &str = r#"{
  "feeTier":0,"canTrade":true,"canDeposit":true,"canWithdraw":true,"updateTime":0,
  "multiAssetsMargin":false,"tradeGroupId":-1,
  "totalInitialMargin":"0.00000000","totalMaintMargin":"0.00000000",
  "totalWalletBalance":"7299.88000000","totalUnrealizedProfit":"0.00000000",
  "totalMarginBalance":"7299.88000000","availableBalance":"7299.88000000",
  "maxWithdrawAmount":"7299.88000000",
  "assets":[
    {"asset":"USDT","walletBalance":"1655.81000000","unrealizedProfit":"0.00000000","marginBalance":"1655.81000000","availableBalance":"1655.81000000"},
    {"asset":"USDC","walletBalance":"5000.00000000","unrealizedProfit":"0.00000000","marginBalance":"5000.00000000","availableBalance":"5000.00000000"},
    {"asset":"BTC","walletBalance":"0.01000000","unrealizedProfit":"0.00000000","marginBalance":"0.01000000","availableBalance":"0.01000000"}
  ],
  "positions":[]
}"#;

const SALDO: f64 = 1655.81;
/// El pico guardado en la definición anterior (todos los activos), a3g.
const PICO_VIEJO: &str = "7323.93139463\n";

fn account_update(usdt_wb: f64) -> String {
    format!(
        r#"{{"e":"ACCOUNT_UPDATE","E":1789000000000,"T":1789000000000,"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{usdt_wb:.8}","cw":"{usdt_wb:.8}","bc":"0"}},{{"a":"USDC","wb":"5000.00000000","cw":"5000.00000000","bc":"0"}}],"P":[]}}}}"#
    )
}

struct Captura(AtomicU64);
impl AccountSink for Captura {
    fn on_capital(&self, usdt: f64) {
        self.0.store(usdt.to_bits(), Relaxed);
    }
    fn on_positions(&self, _p: &[RemotePosition]) {}
    fn on_bracket_close(&self, _c: &BracketClose) {}
}

/// El capital que el stream publica por `on_capital`, con el puente dado.
fn stream_con(puente: &Arc<PuenteCapital>, evento: &str) -> f64 {
    let sink = Arc::new(Captura(AtomicU64::new(f64::NAN.to_bits())));
    let streamer = UserDataStreamer::new(
        BinanceClient::new("k".into(), true),
        Arc::new(OrderRegistry::new()),
    )
    .with_sink(sink.clone())
    .with_puente(Arc::clone(puente));
    streamer.dispatch(evento.as_bytes());
    f64::from_bits(sink.0.load(Relaxed))
}

/// Lo que `fetch_account_balance` devuelve: parser + el mismo puente.
fn rest_con(puente: &PuenteCapital, cuenta: &str) -> f64 {
    puente.capital(parse_account_capital(cuenta).expect("REST válido").capital)
}

/// Freno como lo cablea el host (lazo inmune): arena = REST del arranque, pico
/// = el guardado (si se restaura) y el arranque, exchange = stream.
fn freno(arena_cap: f64, pico_restaurado: Option<f64>, exchange: f64) -> FrenoDuro {
    let arena: Arc<GlobalArena> = GlobalArena::build_in_own_stack(arena_cap);
    arena.unified_capital.store(arena_cap, Relaxed);
    let mut pico = drawdown::registrar_pico(&arena.pico_capital, arena_cap);
    if let Some(p) = pico_restaurado {
        pico = drawdown::registrar_pico(&arena.pico_capital, p);
    }
    arena.capital_exchange.store(exchange, Relaxed);
    let cap = drawdown::capital_del_freno(&arena);
    let pico = drawdown::registrar_pico(&arena.pico_capital, cap).max(pico);
    drawdown::freno_duro(&arena, pico, cap)
}

fn dir_temporal(nombre: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("capital_unico_{nombre}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

static ENTORNO: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn con_testnet<T>(f: impl FnOnce() -> T) -> T {
    let _g = ENTORNO.lock().unwrap_or_else(|p| p.into_inner());
    let previo = std::env::var("USE_TESTNET").ok();
    std::env::set_var("USE_TESTNET", "true");
    let r = f();
    match previo {
        Some(v) => std::env::set_var("USE_TESTNET", v),
        None => std::env::remove_var("USE_TESTNET"),
    }
    r
}

// ── Aviso y fallo cerrado ────────────────────────────────────────────────

#[test]
fn capital_unico_avisa_de_otros_activos_y_del_modo_multi_activo() {
    let c = parse_account_capital(CUENTA_MULTI).unwrap();
    assert_eq!(c.multi_activo, Some(false));
    let aviso = c.aviso().expect("hay saldo fuera de USDT: debe avisar");
    assert!(aviso.contains("NO cuentan"), "{aviso}");

    let multi = CUENTA_MULTI.replace(r#""multiAssetsMargin":false"#, r#""multiAssetsMargin":true"#);
    let aviso = parse_account_capital(&multi).unwrap().aviso().unwrap();
    assert!(aviso.contains("MULTI-ACTIVO"), "{aviso}");

    let solo_usdt = r#"{"totalWalletBalance":"13.0","assets":[{"asset":"USDT","walletBalance":"13.0"}]}"#;
    assert_eq!(parse_account_capital(solo_usdt).unwrap().aviso(), None);
    // Números JSON (no texto) también valen.
    let numero = r#"{"totalWalletBalance":13.0,"assets":[{"asset":"USDT","walletBalance":13.0}]}"#;
    assert_eq!(parse_account_equity(numero).unwrap(), 13.0);
}

#[test]
fn capital_unico_sin_el_activo_de_margen_no_inventa_capital() {
    for malo in [
        r#"{"totalWalletBalance":"7299.88"}"#,
        r#"{"totalWalletBalance":"7299.88","assets":[]}"#,
        r#"{"totalWalletBalance":"5000","assets":[{"asset":"USDC","walletBalance":"5000"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"NaN"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"-1"}]}"#,
        r#"{"assets":[{"asset":"USDT"}]}"#,
        r#"{"assets":[{"asset":"USDT","walletBalance":"1"},{"asset":"USDT","walletBalance":"2"}]}"#,
    ] {
        assert!(parse_account_equity(malo).is_err(), "{malo}");
    }
}

/// Revisión independiente, 3a: USDT 0 con colateral en otros activos no es
/// una cuenta liquidada; antes arrancaba con 0 y el inmune aplanaba lo adoptado.
#[test]
fn capital_unico_usdt_cero_con_fondos_en_otros_activos_falla_cerrado() {
    let usdc = r#"{"multiAssetsMargin":true,"totalWalletBalance":"5000","assets":[{"asset":"USDT","walletBalance":"0"},{"asset":"USDC","walletBalance":"5000"}]}"#;
    let e = parse_account_equity(usdc).unwrap_err();
    assert!(e.contains("ACCOUNT_EVIDENCE_INVALID") && e.contains("otros activos"), "{e}");
    let solo_multi = r#"{"multiAssetsMargin":true,"assets":[{"asset":"USDT","walletBalance":"0"}]}"#;
    assert!(parse_account_equity(solo_multi).is_err());
    let negativo = r#"{"multiAssetsMargin":true,"totalWalletBalance":"5000","assets":[{"asset":"USDT","walletBalance":"-20"}]}"#;
    assert!(parse_account_equity(negativo).unwrap_err().contains("préstamo"));
    // Una cuenta vacía de verdad sigue dando 0 (no hay nada que aplanar).
    let vacia = r#"{"multiAssetsMargin":false,"totalWalletBalance":"0","assets":[{"asset":"USDT","walletBalance":"0"}]}"#;
    assert_eq!(parse_account_equity(vacia).unwrap(), 0.0);
}

// ── (a) Opción A del pico ────────────────────────────────────────────────

#[test]
fn pico_opcion_a_el_archivo_con_la_definicion_nace_vacio_y_el_viejo_queda_intacto() {
    let dir = dir_temporal("opcion_a");
    let (nuevo, viejo) = con_testnet(|| {
        (drawdown::archivo_pico(), drawdown::archivo_pico_definicion_anterior())
    });
    assert_eq!(nuevo, "pico_capital_prueba_usdt.txt");
    assert_eq!(viejo, "pico_capital_prueba.txt");
    let ruta_nuevo = dir.join(&nuevo);
    let ruta_viejo = dir.join(&viejo);
    let ruta_sin_venue = dir.join(drawdown::ARCHIVO_PICO_SIN_VENUE);
    std::fs::write(&ruta_viejo, PICO_VIEJO).unwrap();
    std::fs::write(&ruta_sin_venue, PICO_VIEJO).unwrap();
    let antes = std::fs::read(&ruta_viejo).unwrap();

    // Lo que hace el host: avisar de los viejos y leer SÓLO el nuevo.
    let avisos = drawdown::avisos_picos_ignorados(&[ruta_viejo.clone(), ruta_sin_venue.clone()]);
    assert_eq!(avisos.len(), 2, "{avisos:?}");
    assert!(avisos[0].contains("7323.9314") && avisos[0].contains("ignorado"), "{avisos:?}");
    let decision = decidir_pico(leer_anotacion(&ruta_anotacion(&ruta_nuevo)), None);
    assert_eq!(decision, DecisionPico::Conservar { ancla: None, anotada: false });
    assert!(matches!(drawdown::leer_pico(&ruta_nuevo), CargaPico::Ausente), "nace vacío");

    // Intactos byte a byte, sin renombrar (ni `.adoptado`).
    assert_eq!(std::fs::read(&ruta_viejo).unwrap(), antes);
    assert_eq!(std::fs::read(&ruta_sin_venue).unwrap(), antes);
    assert!(!dir.join("pico_capital.txt.adoptado").exists());
    assert!(!ruta_nuevo.exists(), "leer no crea el archivo nuevo");

    // 7a: sin anotación, el pico ausente es una migración: los stops se conservan.
    assert!(!decision.ausente_es_retiro_del_operador());

    // Con el pico nuevo (vacío) y capital 1655.81 el inmune no dispara...
    let puente = Arc::new(PuenteCapital::new(None));
    let rest = rest_con(&puente, CUENTA_MULTI);
    let sano = freno(rest, None, stream_con(&puente, &account_update(SALDO)));
    assert!(!sano.echado(), "{sano}");
    // ...y con una caída real del 55 % sí (lejos de la frontera de d*).
    let caida = freno(rest, None, stream_con(&puente, &account_update(SALDO * 0.45)));
    assert!(caida.por_capital(), "{caida}");
    // Lo que se evita: adoptar el pico viejo disparaba con una caída falsa del 77 %.
    let adoptado = freno(rest, Some(7323.93139463), stream_con(&puente, &account_update(SALDO)));
    assert!(adoptado.echado(), "el pico viejo disparaba: {adoptado}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pico_opcion_a_contrato_de_fuente_del_host() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(!host.contains("txt.adoptado"), "el host ya no adopta ni renombra el pico sin venue");
    assert!(host.contains("risk_engine::drawdown::avisos_picos_ignorados("));
    assert!(host.contains("archivo_pico_definicion_anterior()"));
    // (c) El aplanado nombra el archivo real, no `pico_capital.txt`.
    assert!(!host.contains("retirar pico_capital.txt"));
    assert!(!host.contains("retirar `pico_capital.txt`"));
    let aplanado = host
        .split("Rearme manual: si fue por drawdown")
        .nth(1)
        .expect("mensaje del aplanado");
    let args = &aplanado[..aplanado.find("),").expect("fin del mensaje")];
    assert!(args.contains("drawdown::archivo_pico()"), "{args}");
}

// ── (b) Tope TG_CAPITAL_OPERATIVO ────────────────────────────────────────

#[test]
fn tope_apagado_es_identico_a_hoy() {
    assert_eq!(interpretar_tope(None), Ok(None));
    let puente = Arc::new(PuenteCapital::new(None));
    for x in [0.0, 13.0, SALDO, 7299.88] {
        assert_eq!(puente.capital(x).to_bits(), f64::to_bits(x));
    }
    assert!(puente.capital(f64::NAN).is_nan());
    assert_eq!(rest_con(&puente, CUENTA_MULTI), SALDO);
    assert_eq!(stream_con(&puente, &account_update(SALDO)), SALDO);
    assert_eq!(puente.ancla(), None, "apagado no ancla");
    assert!(!puente.restaurar_ancla(SALDO));
}

#[test]
fn tope_50_con_saldo_1655_81_da_50_en_rest_y_en_stream() {
    let puente = Arc::new(PuenteCapital::new(interpretar_tope(Some("50")).unwrap()));
    assert_eq!(rest_con(&puente, CUENTA_MULTI), 50.0);
    assert_eq!(stream_con(&puente, &account_update(SALDO)), 50.0);
    assert_eq!(puente.ancla(), Some(SALDO));
    // También si el stream llega primero.
    let solo_stream = Arc::new(PuenteCapital::new(Some(50.0)));
    assert_eq!(stream_con(&solo_stream, &account_update(SALDO)), 50.0);
    assert_eq!(rest_con(&solo_stream, CUENTA_MULTI), 50.0);
}

#[test]
fn tope_5000_con_saldo_1655_81_da_el_saldo_real() {
    let puente = Arc::new(PuenteCapital::new(interpretar_tope(Some(" 5000 ")).unwrap()));
    assert_eq!(rest_con(&puente, CUENTA_MULTI), SALDO);
    assert_eq!(stream_con(&puente, &account_update(SALDO)), SALDO);
    // Nunca mayor que el saldo real, aunque el saldo suba o baje.
    for s in [10.0, 1000.0, SALDO, 4000.0] {
        assert!(puente.capital(s) <= s);
    }
}

#[test]
fn tope_invalido_cero_o_negativo_falla_cerrado() {
    for malo in ["abc", "0", "-5", "NaN", "inf", "-inf", "", " ", "50 USDT", "1e999"] {
        let e = interpretar_tope(Some(malo)).expect_err(malo);
        assert!(e.contains(VARIABLE_TOPE) && e.contains("Falla cerrado"), "{malo}: {e}");
    }
    assert_eq!(interpretar_tope(Some("0.5")), Ok(Some(0.5)));
}

#[test]
fn tope_con_perdida_real_el_capital_baja_y_el_freno_la_ve() {
    let puente = Arc::new(PuenteCapital::new(Some(50.0)));
    let arena = rest_con(&puente, CUENTA_MULTI);
    assert_eq!(arena, 50.0);
    // Pérdida real de 10 USDT: el capital operativo baja a 40 (20 % de 50).
    let exchange = stream_con(&puente, &account_update(SALDO - 10.0));
    assert!((exchange - 40.0).abs() < 1e-9, "{exchange}");
    let f = freno(arena, None, exchange);
    assert!((f.caida_capital - 0.2).abs() < 1e-9, "el freno ve la caída del 20 %: {f}");
    assert!(!f.echado());
    // 27.5 USDT (55 % de 50) disparan; 22.5 (45 %) no.
    let f = freno(arena, None, stream_con(&puente, &account_update(SALDO - 27.5)));
    assert!(f.por_capital(), "{f}");
    let f = freno(arena, None, stream_con(&puente, &account_update(SALDO - 22.5)));
    assert!(!f.echado(), "{f}");
    // Pérdida mayor que el tope: 0, el inmune lo ve como insolvente.
    assert_eq!(puente.capital(SALDO - 60.0), 0.0);
}

#[test]
fn tope_que_cambia_entre_corridas_reinicia_el_pico_y_no_dispara_en_falso() {
    // Corrida anterior con el tope apagado: pico 1655.81. Ahora tope 50.
    let apagado = Anotacion { tope: None, ancla: None };
    let d = decidir_pico(Ok(Some(apagado)), Some(50.0));
    assert_eq!(d, DecisionPico::Reiniciar { tope_anotado: None });
    assert!(!d.ausente_es_retiro_del_operador(), "el freno en stops se conserva");
    let puente = Arc::new(PuenteCapital::new(Some(50.0)));
    let arena = rest_con(&puente, CUENTA_MULTI);
    let exchange = stream_con(&puente, &account_update(SALDO));
    // Si se restaurara el pico de otro tope, caída falsa del 97 %:
    assert!(freno(arena, Some(SALDO), exchange).echado());
    // Reiniciado (no se restaura), no dispara:
    assert!(!freno(arena, None, exchange).echado());

    // Otros cambios: 5000 → 50, sin anotación con tope encendido, 50 → apagado.
    let tope = |t| Anotacion { tope: Some(t), ancla: Some(SALDO) };
    assert!(matches!(decidir_pico(Ok(Some(tope(5000.0))), Some(50.0)), DecisionPico::Reiniciar { .. }));
    assert!(matches!(decidir_pico(Ok(None), Some(50.0)), DecisionPico::Reiniciar { .. }));
    assert!(matches!(decidir_pico(Ok(Some(tope(50.0))), None), DecisionPico::Reiniciar { .. }));
    assert!(matches!(decidir_pico(Err("x".into()), Some(50.0)), DecisionPico::Ilegible(_)));
}

#[test]
fn tope_igual_entre_corridas_conserva_la_caida_acumulada() {
    // Corrida anterior: tope 50 anclado en 1655.81, pico 50. Ahora el saldo es 1645.81.
    let dir = dir_temporal("tope_igual");
    let ruta = ruta_anotacion(&dir.join("pico_capital_prueba_usdt.txt"));
    assert!(ruta.ends_with("pico_capital_prueba_usdt.tope.txt"));
    guardar_anotacion(&ruta, &Anotacion { tope: Some(50.0), ancla: Some(SALDO) }).unwrap();
    let d = decidir_pico(leer_anotacion(&ruta), Some(50.0));
    assert_eq!(d, DecisionPico::Conservar { ancla: Some(SALDO), anotada: true });
    assert!(d.ausente_es_retiro_del_operador(), "con anotación, un pico ausente lo retiró el operador");

    let puente = Arc::new(PuenteCapital::new(Some(50.0)));
    assert!(puente.restaurar_ancla(SALDO));
    let rest = r#"{"totalWalletBalance":"1645.81","assets":[{"asset":"USDT","walletBalance":"1645.81"}]}"#;
    let arena = rest_con(&puente, rest);
    assert!((arena - 40.0).abs() < 1e-9, "la pérdida de la corrida anterior sigue: {arena}");
    let f = freno(arena, Some(50.0), stream_con(&puente, &account_update(SALDO - 10.0)));
    assert!((f.caida_capital - 0.2).abs() < 1e-9, "{f}");

    // Anotación ilegible: se trata como un pico ilegible.
    std::fs::write(&ruta, "tope=cincuenta\n").unwrap();
    assert!(matches!(decidir_pico(leer_anotacion(&ruta), Some(50.0)), DecisionPico::Ilegible(_)));
    // Formato de ida y vuelta.
    for a in [
        Anotacion { tope: None, ancla: None },
        Anotacion { tope: Some(50.0), ancla: Some(SALDO) },
        Anotacion { tope: Some(0.5), ancla: None },
    ] {
        assert_eq!(Anotacion::desde_texto(&a.a_texto()), Ok(a));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// La variable se lee en un solo sitio (revisión, 6.7): el módulo del puente.
#[test]
fn tope_contrato_de_fuente_un_solo_punto() {
    let modulo = include_str!("../src/capital_operativo.rs");
    assert!(modulo.contains(r#"VARIABLE_TOPE: &str = "TG_CAPITAL_OPERATIVO""#));
    for (nombre, fuente) in [
        ("executor.rs", include_str!("../src/executor.rs")),
        ("user_data_stream.rs", include_str!("../src/user_data_stream.rs")),
        ("god_engine.rs", include_str!("../../../src/bin/god_engine.rs")),
    ] {
        assert!(!fuente.contains("\"TG_CAPITAL_OPERATIVO\""), "{nombre} no lee la variable por su cuenta");
        assert!(fuente.contains("capital_operativo::puente_global()"), "{nombre} usa el puente único");
    }
}
