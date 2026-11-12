//! CAPITAL-UNICO (prioridad 1 del mapa de pérdidas, 2026-11-11) — lo que usa
//! la API nueva. La RED contra `main` está en `capital_unico_red_contract.rs`.
//!
//! - Aviso de otros activos y del modo multi-activo; fallo cerrado sin USDT.
//! - Revisión independiente 3a: saldo de USDT 0 con fondos en otros activos = `Err`.
//! - Opción A del pico (V2, sin tope): `pico_capital_{venue}_usdt.txt` manda; la
//!   marca de la definición distingue migración (stops conservados, 7a) de
//!   retiro del operador; el pico viejo del venue sólo se adopta si las
//!   unidades coinciden (7d); el sin venue nunca (7b); nada viejo se toca.
//!
//! No tocan d*, `capital_del_freno` ni ningún umbral.

use execution_engine::client::BinanceClient;
use execution_engine::execution_evidence::{parse_account_capital, parse_account_equity};
use execution_engine::trade_accounting::BracketClose;
use execution_engine::{AccountSink, OrderRegistry, RemotePosition, UserDataStreamer};
use quantum_arena::GlobalArena;
use risk_engine::drawdown::{self, CargaPico, FrenoDuro, RutasPico};
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

/// El saldo que el stream publica por `on_capital`.
fn stream(evento: &str) -> f64 {
    let sink = Arc::new(Captura(AtomicU64::new(f64::NAN.to_bits())));
    let streamer = UserDataStreamer::new(
        BinanceClient::new("k".into(), true),
        Arc::new(OrderRegistry::new()),
    )
    .with_sink(sink.clone());
    streamer.dispatch(evento.as_bytes());
    f64::from_bits(sink.0.load(Relaxed))
}

/// Freno como lo cablea el host (lazo inmune): arena = REST del arranque, pico
/// = el guardado (si se restaura) y el arranque, exchange = observación del exchange.
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

/// Rutas del pico de Testnet dentro de `dir`, con los nombres de `drawdown`.
fn rutas_testnet(dir: &std::path::Path) -> RutasPico {
    let (pico, marca, anterior) = con_testnet(|| {
        (
            drawdown::archivo_pico(),
            drawdown::archivo_marca_definicion(),
            drawdown::archivo_pico_definicion_anterior(),
        )
    });
    assert_eq!(pico, "pico_capital_prueba_usdt.txt");
    assert_eq!(marca, "pico_capital_prueba_usdt.definicion");
    assert_eq!(anterior, "pico_capital_prueba.txt");
    RutasPico {
        pico: dir.join(pico),
        marca: dir.join(marca),
        anterior: dir.join(anterior),
        sin_venue: dir.join(drawdown::ARCHIVO_PICO_SIN_VENUE),
    }
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

// ── Opción A del pico (V2) ───────────────────────────────────────────────

/// Migración en la Testnet de a3g: sin pico nuevo y sin marca, con el pico
/// viejo de 7 323,93 y saldo fuera de USDT (las unidades NO coinciden).
#[test]
fn pico_migracion_con_unidades_distintas_ignora_el_viejo_y_conserva_los_stops() {
    let dir = dir_temporal("migracion_distintas");
    let r = rutas_testnet(&dir);
    std::fs::write(&r.anterior, PICO_VIEJO).unwrap();
    std::fs::write(&r.sin_venue, PICO_VIEJO).unwrap();
    let antes = std::fs::read(&r.anterior).unwrap();

    let unidades_coinciden = parse_account_capital(CUENTA_MULTI).unwrap().aviso().is_none();
    assert!(!unidades_coinciden, "a3g: la cartera tiene USDC y BTC");
    let a = drawdown::decidir_arranque_pico(&r, unidades_coinciden);
    assert_eq!(a.carga, CargaPico::Ausente, "el pico nuevo nace vacío");
    assert!(!a.rearmar_stops, "7a: la migración no rearma el freno en stops");
    assert_eq!(a.adoptado_de, None);
    let todo = a.avisos.join(" | ");
    assert!(todo.contains("7323.9314") && todo.contains("ignorado"), "{todo}");
    assert!(todo.contains("NO coinciden"), "{todo}");

    // Intactos byte a byte, sin renombrar (ni `.adoptado`), y leer no crea nada.
    assert_eq!(std::fs::read(&r.anterior).unwrap(), antes);
    assert_eq!(std::fs::read(&r.sin_venue).unwrap(), antes);
    assert!(!dir.join("pico_capital.txt.adoptado").exists());
    assert!(!r.pico.exists() && !r.marca.exists());

    // Con el pico nuevo (vacío) y saldo 1655.81 el inmune no dispara...
    let rest = parse_account_equity(CUENTA_MULTI).unwrap();
    assert!(!freno(rest, None, stream(&account_update(SALDO))).echado());
    // ...con una caída real del 55 % sí (lejos de la frontera de d*)...
    assert!(freno(rest, None, stream(&account_update(SALDO * 0.45))).por_capital());
    // ...y adoptar el pico viejo disparaba con una caída falsa del 77 %.
    assert!(freno(rest, Some(7323.93139463), stream(&account_update(SALDO))).echado());
    let _ = std::fs::remove_dir_all(&dir);
}

/// 7d: con sólo USDT y sin multi-activo la unidad no cambió (mainnet): el pico
/// viejo del venue se adopta para no perdonar una caída real (CL-99). El sin
/// venue nunca (7b). El viejo ilegible, con unidades iguales, es ilegible.
#[test]
fn pico_migracion_con_unidades_iguales_adopta_solo_el_viejo_del_venue() {
    let dir = dir_temporal("migracion_iguales");
    let r = rutas_testnet(&dir);
    let solo_usdt = r#"{"multiAssetsMargin":false,"totalWalletBalance":"13.0","assets":[{"asset":"USDT","walletBalance":"13.0"}]}"#;
    let unidades_coinciden = parse_account_capital(solo_usdt).unwrap().aviso().is_none();
    assert!(unidades_coinciden);

    std::fs::write(&r.sin_venue, "20\n").unwrap();
    let a = drawdown::decidir_arranque_pico(&r, unidades_coinciden);
    assert_eq!(a.carga, CargaPico::Ausente, "7b: el sin venue no se adopta nunca");

    std::fs::write(&r.anterior, "20\n").unwrap();
    let antes = std::fs::read(&r.anterior).unwrap();
    let a = drawdown::decidir_arranque_pico(&r, unidades_coinciden);
    assert_eq!(a.carga, CargaPico::Leido(20.0));
    assert_eq!(a.adoptado_de.as_deref(), Some(r.anterior.as_path()));
    assert!(!a.rearmar_stops);
    assert!(a.avisos.iter().any(|s| s.contains("ADOPTADO")), "{:?}", a.avisos);
    assert_eq!(std::fs::read(&r.anterior).unwrap(), antes, "el viejo no se toca");
    assert!(!r.pico.exists());
    // Con el pico adoptado de 20 y 13 de saldo, la caída real del 35 % se ve.
    let f = freno(13.0, Some(20.0), 13.0);
    assert!((f.caida_capital - 0.35).abs() < 1e-9, "{f}");

    std::fs::write(&r.anterior, "veinte\n").unwrap();
    let a = drawdown::decidir_arranque_pico(&r, unidades_coinciden);
    assert!(matches!(a.carga, CargaPico::Ilegible(_)), "{:?}", a.carga);
    // Con unidades distintas, el mismo ilegible sólo se ignora.
    let a = drawdown::decidir_arranque_pico(&r, false);
    assert_eq!(a.carga, CargaPico::Ausente);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 7a: con la marca, el pico ausente es un retiro del operador (rearme de los
/// stops, F1-011); el pico nuevo, si existe, manda sobre los viejos.
#[test]
fn pico_marca_distingue_retiro_del_operador_de_la_migracion() {
    let dir = dir_temporal("marca");
    let r = rutas_testnet(&dir);
    std::fs::write(&r.anterior, PICO_VIEJO).unwrap();

    drawdown::guardar_marca_definicion(&r.marca).unwrap();
    assert_eq!(std::fs::read_to_string(&r.marca).unwrap(), drawdown::MARCA_DEFINICION);
    let a = drawdown::decidir_arranque_pico(&r, true);
    assert_eq!(a.carga, CargaPico::Ausente);
    assert!(a.rearmar_stops, "con marca y sin pico, lo retiró el operador");
    assert_eq!(a.adoptado_de, None, "tras un retiro no se adopta ningún pico viejo");

    drawdown::guardar_pico(&r.pico, 1700.0).unwrap();
    let a = drawdown::decidir_arranque_pico(&r, true);
    assert_eq!(a.carga, CargaPico::Leido(1700.0), "el pico de la definición manda");
    assert!(!a.rearmar_stops);

    std::fs::write(&r.pico, "1,7\n").unwrap();
    let a = drawdown::decidir_arranque_pico(&r, true);
    assert!(matches!(a.carga, CargaPico::Ilegible(_)), "CL-133 B2");
    assert!(!a.rearmar_stops);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn pico_contrato_de_fuente_del_host() {
    let host = include_str!("../../../src/bin/god_engine.rs");
    assert!(!host.contains("txt.adoptado"), "el host ya no adopta ni renombra el pico sin venue");
    assert!(host.contains("risk_engine::drawdown::decidir_arranque_pico("));
    assert!(host.contains("risk_engine::drawdown::guardar_marca_definicion(&ruta_marca)"));
    assert!(host.contains("unidades_coinciden,"), "7d: el host pasa lo que mostró el arranque");
    assert!(!host.contains("capital_operativo"), "el tope salió de esta rama");
    // 7e: el aplanado nombra el archivo real, no `pico_capital.txt`.
    assert!(!host.contains("retirar pico_capital.txt"));
    assert!(!host.contains("retirar `pico_capital.txt`"));
    let aplanado = host
        .split("Rearme manual: si fue por drawdown")
        .nth(1)
        .expect("mensaje del aplanado");
    let args = &aplanado[..aplanado.find("),").expect("fin del mensaje")];
    assert!(args.contains("drawdown::archivo_pico()"), "{args}");
}
