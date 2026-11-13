//! CAPITAL-UNICO V4I: el stream real (`UserDataStreamer::dispatch`) con el puente y el libro del
//! dimensionable. Puntos 2, 4, 5b y la precision 5 de Qoder del pedido 20261112-0948f.
//! Sin red: los eventos son bytes sinteticos con el esquema de Binance USD-M.
use execution_engine::client::BinanceClient;
use execution_engine::puente_cuenta::{LecturaCuenta, PuenteCuenta};
use execution_engine::{OrderRegistry, UserDataStreamer};
use quantum_arena::dimensionable::ClaveReserva;
use quantum_arena::{symbol_registry, GlobalArena};
use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

const SIMBOLO: &str = "BTCUSDT";
static REGISTRO: Mutex<()> = Mutex::new(());

struct Montaje {
    puente: Arc<PuenteCuenta>,
    arena: Arc<GlobalArena>,
    marcas: Arc<Mutex<Vec<(f64, Option<u64>)>>>,
}

/// El publicador del host (`publicar_capital_de_sesion`) reducido al libro: saldo y su marca.
fn montaje() -> Montaje {
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
    let arena = GlobalArena::build_in_own_stack(0.0);
    arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).activar(&arena);
    let puente = Arc::new(PuenteCuenta::con_sesion(1));
    let marcas = Arc::new(Mutex::new(Vec::new()));
    {
        let (arena, marcas) = (Arc::clone(&arena), Arc::clone(&marcas));
        puente.vincular_publicador(move |p| {
            if p.veta {
                arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).veto_del_puente();
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            }
            let Some((saldo, marca)) = p.vigente else { return false };
            marcas.lock().unwrap_or_else(|e| e.into_inner()).push((saldo, marca));
            let mut libro = arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
            libro.fijar_alcance(p.era, p.sesion);
            libro.saldo_aceptado(&arena, saldo, marca);
            drop(libro);
            arena.capital_incompleto.store(false, Ordering::Release);
            true
        });
    }
    puente.abrir_socket(puente.era()).unwrap();
    let lectura = LecturaCuenta {
        saldo: 100.0,
        tiempo_saldo_ms: Some(100),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    };
    puente.observar_rest(puente.ticket(), &lectura).unwrap();
    Montaje { puente, arena, marcas }
}

impl Montaje {
    fn streamer(&self) -> UserDataStreamer {
        UserDataStreamer::new(BinanceClient::new(String::new(), true), Arc::new(OrderRegistry::new()))
            .with_puente(Arc::clone(&self.puente))
            .with_arena(Arc::clone(&self.arena))
    }
    fn dimensionable(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn reservado(&self) -> f64 {
        self.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner()).reservado()
    }
}

fn cuenta(saldo: &str, e: u64, t: u64) -> Vec<u8> {
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{e},"T":{t},"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{saldo}","cw":"{saldo}","bc":"0"}}],"P":[]}}}}"#).into_bytes()
}

#[allow(clippy::too_many_arguments)]
fn fill(coid: &str, estado: &str, trade: u64, t: u64, comision: &str, activo: &str, rp: &str, reduce: bool) -> Vec<u8> {
    format!(r#"{{"e":"ORDER_TRADE_UPDATE","E":{t},"T":{t},"o":{{"s":"{SIMBOLO}","c":"{coid}","S":"BUY","o":"MARKET","x":"TRADE","X":"{estado}","i":7,"q":"1","z":"1","l":"1","L":"100","ap":"100","n":"{comision}","N":"{activo}","t":{trade},"R":{reduce},"rp":"{rp}","T":{t},"ps":"BOTH","sp":"0","m":false}}}}"#).into_bytes()
}

/// Precision 5 de Qoder: el saldo de un ACCOUNT_UPDATE lleva la `T` de la transaccion, con
/// E igual a T y con E mayor que T; un empate con la T del fill no certifica su inclusion.
#[test]
fn account_update_lleva_la_t_y_un_empate_no_certifica_inclusion() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    let s = m.streamer();
    let clave = ClaveReserva { coin_id: symbol_registry::try_index(SIMBOLO).unwrap(), slot: 0, generation: 1 };
    {
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
        libro.reservar(&m.arena, clave, 0.04);
        libro.vincular_orden(clave, "cL_e_igual_t");
    }
    s.dispatch(&fill("cL_e_igual_t", "FILLED", 11, 200, "0.04", "USDT", "0", false));
    s.dispatch(&cuenta("99.96", 200, 200));
    assert_eq!(m.marcas.lock().unwrap().last().copied(), Some((99.96, Some(200))), "E = T");
    assert_eq!(m.reservado(), 0.04, "T igual a la del fill: el cargo se conserva");
    s.dispatch(&cuenta("99.96", 260, 250));
    assert_eq!(m.marcas.lock().unwrap().last().copied(), Some((99.96, Some(250))), "E > T: la marca es T");
    assert_eq!(m.reservado(), 0.0, "T posterior al terminal: incluido");
    assert_eq!(m.dimensionable(), 99.96);
}

/// INV-E en el stream: comision real mayor que la estimada (0,25 frente a 0,75) y fill con la
/// comision en otra moneda (no carga el USDT).
#[test]
fn comision_real_del_stream_y_comision_en_otra_moneda() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    let s = m.streamer();
    let coin_id = symbol_registry::try_index(SIMBOLO).unwrap();
    let clave = ClaveReserva { coin_id, slot: 0, generation: 1 };
    {
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap_or_else(|e| e.into_inner());
        libro.reservar(&m.arena, clave, 0.25);
        libro.vincular_orden(clave, "cL_real");
    }
    s.dispatch(&fill("cL_real", "PARTIALLY_FILLED", 21, 150, "0.75", "USDT", "0", false));
    assert_eq!(m.dimensionable(), 99.25, "el disponible no cae por la comision real");
    // INV-G: fill de cierre con la comision en BNB y PnL realizado -2: el debito es 2.
    s.dispatch(&fill("x_cierre", "FILLED", 22, 160, "0.5", "BNB", "-2", true));
    assert_eq!(m.dimensionable(), 97.25, "se resto una comision en otra moneda del USDT");
}

/// Punto 5b: tras el relevo, el socket viejo entrega un fill de cierre y un fill de entrada por
/// el camino real: el libro, el espejo y el veto quedan intactos.
#[test]
fn el_socket_de_la_era_anterior_no_escribe_el_libro() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    let viejo = m.streamer();
    let era = m.puente.nueva_era_de_sesion(1);
    m.puente.abrir_socket(era).unwrap();
    let lectura = LecturaCuenta {
        saldo: 100.0,
        tiempo_saldo_ms: Some(150),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    };
    m.puente.observar_rest(m.puente.ticket(), &lectura).unwrap();
    let (dim, veto, reservado) = (m.dimensionable(), m.arena.capital_incompleto.load(Ordering::Acquire), m.reservado());
    viejo.dispatch(&fill("x_cierre", "FILLED", 31, 160, "0.5", "USDT", "-3", true));
    viejo.dispatch(&fill("autoclose-liq", "FILLED", 32, 161, "0.5", "USDT", "-3", true));
    assert_eq!((m.dimensionable(), m.arena.capital_incompleto.load(Ordering::Acquire), m.reservado()), (dim, veto, reservado));
    assert!(!m.arena.config.entry_publication_gate.lock().unwrap().vetado(), "el socket viejo veto el libro");
    // Control: el streamer de la era vigente si escribe.
    let nuevo = m.streamer();
    nuevo.dispatch(&fill("x_cierre", "FILLED", 31, 160, "0.5", "USDT", "-3", true));
    assert_eq!(m.dimensionable(), dim - 3.5);
}
