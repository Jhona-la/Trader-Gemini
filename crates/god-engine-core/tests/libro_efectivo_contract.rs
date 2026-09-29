//! CL-33 — el libro con el que el núcleo decide un evento sin libro propio.
//!
//! El host sólo trae bid/ask en los eventos @depth5 y pasa 0 en trades y
//! klines, que son los únicos eventos donde se permiten entradas. El núcleo
//! fabricaba ±1 pb alrededor del print; ahora usa el último libro medido,
//! trasladado lo justo para contener el print.
use god_engine_core::{GodEngineCore, libro_efectivo};
use quantum_arena::{GlobalArena, symbol_registry};
use std::{path::PathBuf, sync::Mutex};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

struct FixtureDirectory {
    root: PathBuf,
    previous: PathBuf,
}
impl FixtureDirectory {
    fn new() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tg-cl33-libro-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("data")).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        Self { root, previous }
    }
}
impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).unwrap();
        let exact = self.root.canonicalize().unwrap();
        assert!(exact.starts_with(std::env::temp_dir().canonicalize().unwrap()));
        assert!(
            exact
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tg-cl33-libro-")
        );
        std::fs::remove_dir_all(exact).unwrap();
    }
}

fn nucleo() -> GodEngineCore {
    quantum_arena::symbols::update_dynamic_universe(vec!["CLXXXIIIUSDT".into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(
        "CLXXXIIIUSDT",
    )]);
    let arena = GlobalArena::build_in_own_stack(100.0);
    let mut core = GodEngineCore::new(arena);
    core.swing_nn = None;
    core.scalp_forest = None;
    core
}

/// Evento @depth5 tal como lo pasa el host: libro propio, sin trade.
fn depth(core: &mut GodEngineCore, bid: f64, ask: f64, t: u64) {
    core.process_event(
        0, false, false, true, (bid + ask) * 0.5, 0.0, bid, ask, 5.0, 3.0, 0.0, 0.0, t, true,
        &[0.0; 54], false,
    );
}

/// Trade tal como lo pasa el host: bid = ask = 0 y las cantidades del último
/// libro (D-707).
fn trade(core: &mut GodEngineCore, precio: f64, t: u64, comprador_es_maker: bool) {
    core.process_event(
        0, true, false, false, precio, 0.2, 0.0, 0.0, 5.0, 3.0, 0.0, 0.0, t, true, &[0.0; 54],
        comprador_es_maker,
    );
}

#[test]
fn cl33_el_libro_propio_manda_y_sin_libro_medido_queda_el_respaldo() {
    assert_eq!(libro_efectivo(100.05, 100.0, 100.1, Some((90.0, 91.0))), (100.0, 100.1));
    assert_eq!(libro_efectivo(100.0, 0.0, 0.0, None), (100.0 * 0.9999, 100.0 * 1.0001));
}

#[test]
fn cl33_un_print_dentro_del_libro_no_lo_mueve() {
    let ultimo = Some((100.0, 100.1));
    assert_eq!(libro_efectivo(100.1, 0.0, 0.0, ultimo), (100.0, 100.1));
    assert_eq!(libro_efectivo(100.0, 0.0, 0.0, ultimo), (100.0, 100.1));
    assert_eq!(libro_efectivo(100.05, 0.0, 0.0, ultimo), (100.0, 100.1));
}

#[test]
fn cl33_un_print_fuera_del_libro_lo_traslada_con_su_spread() {
    let ultimo = Some((100.0, 100.1));
    let (b, a) = libro_efectivo(100.3, 0.0, 0.0, ultimo);
    assert_eq!(a, 100.3);
    assert!((a - b - 0.1).abs() < 1e-9);
    let (b, a) = libro_efectivo(99.7, 0.0, 0.0, ultimo);
    assert_eq!(b, 99.7);
    assert!((a - b - 0.1).abs() < 1e-9);
}

#[test]
fn cl33_un_trade_en_un_libro_quieto_no_fabrica_flujo() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let mut core = nucleo();
    depth(&mut core, 100.0, 100.1, 1_000);
    depth(&mut core, 100.0, 100.1, 1_100);
    // Compras y ventas agresoras contra el mismo libro: su OFI verdadero es 0.
    for i in 0..20u64 {
        let comprador = i % 2 == 0;
        trade(&mut core, if comprador { 100.1 } else { 100.0 }, 1_200 + i * 10, !comprador);
    }
    let ofi = &core.feature_engines[0].ofi_model;
    assert_eq!((ofi.prev_bid_price, ofi.prev_ask_price), (100.0, 100.1));
    assert_eq!(ofi.ema_ofi, 0.0, "el OFI de un libro quieto es 0");
}

#[test]
fn cl33_el_primer_trade_sin_libro_medido_usa_el_respaldo() {
    let _guard = ENVIRONMENT.lock().unwrap_or_else(|p| p.into_inner());
    let _dir = FixtureDirectory::new();
    let mut core = nucleo();
    trade(&mut core, 100.0, 1_000, false);
    let ofi = &core.feature_engines[0].ofi_model;
    assert_eq!(ofi.prev_bid_price, 100.0 * 0.9999);
    assert_eq!(ofi.prev_ask_price, 100.0 * 1.0001);
}
