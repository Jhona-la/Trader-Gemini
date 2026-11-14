//! CAPITAL-UNICO V4K, A2 (dictamen V4J :33 y :54): la marca terminal y su acumulado deben ser de
//! la misma observacion. Terminal reciente con ejecucion acumulada positiva, terminal antiguo con
//! acumulado nulo y foto posterior, antes del fill: la contradiccion no da capacidad nueva.
//!
//! Piezas reales: el stream (`UserDataStreamer::dispatch`) con el puente de produccion y el libro
//! del dimensionable (mismo montaje que `capital_v4j_stream_libro_contract.rs`). Sin red.
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
}

fn montaje() -> Montaje {
    quantum_arena::symbols::update_dynamic_universe(vec![SIMBOLO.into()]);
    symbol_registry::update_registry(vec![symbol_registry::get_official_binance_spec(SIMBOLO)]);
    let arena = GlobalArena::build_in_own_stack(0.0);
    arena.config.entry_publication_gate.lock().unwrap().activar(&arena);
    let puente = Arc::new(PuenteCuenta::con_sesion(1));
    {
        let arena = Arc::clone(&arena);
        puente.vincular_publicador(move |p| {
            if p.veta {
                arena.config.entry_publication_gate.lock().unwrap().veto_del_puente();
                arena.capital_incompleto.store(true, Ordering::Release);
                return false;
            }
            let Some((saldo, marca)) = p.vigente else { return false };
            let mut libro = arena.config.entry_publication_gate.lock().unwrap();
            libro.fijar_alcance(p.era, p.sesion);
            libro.saldo_aceptado(&arena, saldo, marca);
            drop(libro);
            arena.capital_incompleto.store(false, Ordering::Release);
            true
        });
    }
    puente.abrir_socket(puente.era()).unwrap();
    puente
        .observar_rest(
            puente.ticket(),
            &LecturaCuenta {
                saldo: 100.0,
                tiempo_saldo_ms: Some(100),
                posiciones: Some(HashMap::new()),
                rechazo_arranque: None,
                upnl_del_activo: Some(0.0),
            },
        )
        .unwrap();
    Montaje { puente, arena }
}

impl Montaje {
    fn streamer(&self) -> UserDataStreamer {
        UserDataStreamer::new(BinanceClient::new(String::new(), true), Arc::new(OrderRegistry::new()))
            .with_puente(Arc::clone(&self.puente))
            .with_arena(Arc::clone(&self.arena))
    }
    fn reservar(&self, estimado: f64, orden: &str) -> ClaveReserva {
        let clave = ClaveReserva { coin_id: symbol_registry::try_index(SIMBOLO).unwrap(), slot: 0, generation: 1 };
        let mut libro = self.arena.config.entry_publication_gate.lock().unwrap();
        libro.reservar(&self.arena, clave, estimado);
        libro.vincular_orden_con_cantidad(clave, orden, 1.0);
        clave
    }
    fn dim(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn reservado(&self) -> f64 {
        self.arena.config.entry_publication_gate.lock().unwrap().reservado()
    }
    fn vetado(&self) -> bool {
        self.arena.config.entry_publication_gate.lock().unwrap().vetado() || self.arena.capital_incompleto.load(Ordering::Acquire)
    }
}

#[allow(clippy::too_many_arguments)]
fn evento(coid: &str, x: &str, estado: &str, trade: u64, l: &str, z: &str, t: u64, n: &str) -> Vec<u8> {
    let o = format!(
        r#""s":"{SIMBOLO}","c":"{coid}","S":"BUY","o":"MARKET","x":"{x}","X":"{estado}","i":7,"q":"1","z":"{z}","l":"{l}","L":"100","ap":"100","t":{trade},"R":false,"rp":"0","ps":"BOTH","sp":"0","m":false,"T":{t},"n":"{n}","N":"USDT""#
    );
    format!(r#"{{"e":"ORDER_TRADE_UPDATE","E":900,"T":900,"o":{{{o}}}}}"#).into_bytes()
}

fn cuenta(saldo: &str, t: u64) -> Vec<u8> {
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{t},"T":{t},"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{saldo}","cw":"{saldo}","bc":"0"}}],"P":[]}}}}"#).into_bytes()
}

/// A2 (RED sobre V4J): terminal T500 con acumulado 1 y terminal antiguo T400 con acumulado 0;
/// foto T600 antes del fill. Con y sin el ACK del host. La contradiccion no libera el estimado
/// (0,04): ninguna capacidad nueva (o veto). El fill tardio tampoco la convierte en prueba.
#[test]
fn terminal_antiguo_con_acumulado_nulo_no_libera_el_estimado() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    for con_ack in [false, true] {
        let m = montaje();
        let s = m.streamer();
        let clave = m.reservar(0.04, "cL_v4k_a2");
        let antes = m.dim();
        s.dispatch(&evento("cL_v4k_a2", "EXPIRED", "FILLED", 0, "0", "1", 500, "0"));
        s.dispatch(&evento("cL_v4k_a2", "CANCELED", "CANCELED", 0, "0", "0", 400, "0"));
        if con_ack {
            m.arena.config.entry_publication_gate.lock().unwrap().ejecutar(&m.arena, clave, 0.0, Some(500));
        }
        s.dispatch(&cuenta("100", 600));
        assert!(
            m.vetado() || (m.reservado() >= 0.04 && m.dim() <= antes),
            "ack={con_ack}: el terminal antiguo con acumulado 0 libero el estimado: reservado {} dim {} (antes {antes})",
            m.reservado(),
            m.dim()
        );
        // El fill llega despues (T450, ya en la foto T600): la contradiccion sigue sin dar capacidad.
        s.dispatch(&evento("cL_v4k_a2", "TRADE", "FILLED", 63, "1", "1", 450, "0.03"));
        assert!(m.vetado() || m.dim() <= antes, "ack={con_ack}: capacidad nueva tras el fill: {}", m.dim());
    }
}

/// Control (verde en V4J y en V4K): el MISMO terminal repetido no es contradiccion; con su fill
/// visto y la foto posterior, el estimado sale y queda la capacidad real.
#[test]
fn terminal_repetido_coherente_libera_con_su_fill() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    let s = m.streamer();
    let clave = m.reservar(0.04, "cL_v4k_a2c");
    s.dispatch(&evento("cL_v4k_a2c", "TRADE", "FILLED", 64, "1", "1", 450, "0.03"));
    s.dispatch(&evento("cL_v4k_a2c", "EXPIRED", "FILLED", 0, "0", "1", 500, "0"));
    s.dispatch(&evento("cL_v4k_a2c", "EXPIRED", "FILLED", 0, "0", "1", 500, "0"));
    m.arena.config.entry_publication_gate.lock().unwrap().ejecutar(&m.arena, clave, 0.0, Some(500));
    s.dispatch(&cuenta("99.97", 600));
    assert_eq!(m.reservado(), 0.0, "un terminal coherente repetido retuvo el estimado");
    assert!(!m.vetado(), "control vetado");
    assert_eq!(m.dim(), 99.97, "capacidad real");
}
