//! CAPITAL-UNICO V4J, punto 2 (A2 de `qra_v4j_criterios.md`): el terminal de una orden no
//! acredita el cargo de un fill sin cobertura propia, y el parser del stream no pierde el cargo de
//! un fill con `T` omitida, `T` = 0 o comision desconocida.
//!
//! Piezas reales: el stream (`UserDataStreamer::dispatch`) con el puente de produccion y el libro
//! del dimensionable. Frames privados sinteticos con el esquema de Binance USD-M; sin red.
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

/// El publicador del host (`publicar_capital_de_sesion`) reducido al libro: veto, alcance y saldo.
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
    puente.observar_rest(puente.ticket(), &rest(100.0, 100)).unwrap();
    Montaje { puente, arena }
}

fn rest(saldo: f64, t: u64) -> LecturaCuenta {
    LecturaCuenta {
        saldo,
        tiempo_saldo_ms: Some(t),
        posiciones: Some(HashMap::new()),
        rechazo_arranque: None,
        upnl_del_activo: Some(0.0),
    }
}

impl Montaje {
    fn streamer(&self) -> UserDataStreamer {
        UserDataStreamer::new(BinanceClient::new(String::new(), true), Arc::new(OrderRegistry::new()))
            .with_puente(Arc::clone(&self.puente))
            .with_arena(Arc::clone(&self.arena))
    }
    fn coin(&self) -> usize {
        symbol_registry::try_index(SIMBOLO).unwrap()
    }
    fn reservar(&self, slot: usize, estimado: f64, orden: &str, cantidad: Option<f64>) -> ClaveReserva {
        let clave = ClaveReserva { coin_id: self.coin(), slot, generation: 1 };
        let mut libro = self.arena.config.entry_publication_gate.lock().unwrap();
        libro.reservar(&self.arena, clave, estimado);
        match cantidad {
            Some(q) => libro.vincular_orden_con_cantidad(clave, orden, q),
            None => libro.vincular_orden(clave, orden),
        }
        clave
    }
    fn dim(&self) -> f64 {
        self.arena.unified_capital.load(Ordering::Relaxed)
    }
    fn reservado(&self) -> f64 {
        self.arena.config.entry_publication_gate.lock().unwrap().reservado()
    }
    fn vetado(&self) -> bool {
        self.arena.config.entry_publication_gate.lock().unwrap().vetado()
    }
}

/// Un frame ORDER_TRADE_UPDATE de una entrada. `t_fill`: `None` omite `T`; `n`/`asset`: `None`
/// omite el campo.
#[allow(clippy::too_many_arguments)]
fn fill(coid: &str, x: &str, estado: &str, trade: u64, l: &str, z: &str, t_fill: Option<u64>, n: Option<&str>, asset: Option<&str>) -> Vec<u8> {
    let mut o = format!(
        r#""s":"{SIMBOLO}","c":"{coid}","S":"BUY","o":"MARKET","x":"{x}","X":"{estado}","i":7,"q":"1","z":"{z}","l":"{l}","L":"100","ap":"100","t":{trade},"R":false,"rp":"0","ps":"BOTH","sp":"0","m":false"#
    );
    if let Some(t) = t_fill {
        o.push_str(&format!(r#","T":{t}"#));
    }
    if let Some(n) = n {
        o.push_str(&format!(r#","n":"{n}""#));
    }
    if let Some(a) = asset {
        o.push_str(&format!(r#","N":"{a}""#));
    }
    format!(r#"{{"e":"ORDER_TRADE_UPDATE","E":900,"T":900,"o":{{{o}}}}}"#).into_bytes()
}

fn cuenta(saldo: &str, t: u64) -> Vec<u8> {
    format!(r#"{{"e":"ACCOUNT_UPDATE","E":{t},"T":{t},"a":{{"m":"ORDER","B":[{{"a":"USDT","wb":"{saldo}","cw":"{saldo}","bc":"0"}}],"P":[]}}}}"#).into_bytes()
}

/// A2 (caso de Codex por el stream): dos fills de 0,25 (T200) y 0,50 (sin T), terminal de la
/// orden con T200 y foto T201 que solo incluye el primero. La foto y el terminal cubren solo el
/// cargo demostrado: los 0,50 quedan retenidos (tope 99,25), con y sin cantidades en el libro.
#[test]
fn terminal_con_t_no_hereda_cobertura_al_fill_sin_t() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    for cantidad in [None, Some(1.0)] {
        let m = montaje();
        let s = m.streamer();
        let clave = m.reservar(0, 0.75, "cL_v4j_t", cantidad);
        s.dispatch(&fill("cL_v4j_t", "TRADE", "PARTIALLY_FILLED", 41, "0.5", "0.5", Some(200), Some("0.25"), Some("USDT")));
        s.dispatch(&fill("cL_v4j_t", "TRADE", "PARTIALLY_FILLED", 42, "0.5", "1", None, Some("0.50"), Some("USDT")));
        // Terminal de la orden (sin ejecucion nueva) con T200.
        s.dispatch(&fill("cL_v4j_t", "EXPIRED", "EXPIRED", 0, "0", "1", Some(200), Some("0"), Some("USDT")));
        m.arena.config.entry_publication_gate.lock().unwrap().ejecutar(&m.arena, clave, 0.0, None);
        s.dispatch(&cuenta("99.75", 201));
        assert!(m.vetado() || m.dim() <= 99.25, "cantidad={cantidad:?}: el terminal acredito el cargo sin T: {}", m.dim());
        assert_eq!(m.reservado(), 0.5, "cantidad={cantidad:?}: el cargo de 0,50 sin T salio del libro");
        // Una foto todavia posterior tampoco lo prueba: sigue retenido (sin corte en vivo, A4b).
        s.dispatch(&cuenta("99.75", 900));
        assert_eq!(m.reservado(), 0.5, "cantidad={cantidad:?}: una marca mayor acredito un fill sin marca");
    }
}

/// A2 (parser): desde un frame privado con fill completo, `T` omitida y `T` = 0 conservan la
/// identidad y el cargo (el frame no se descarta). `n` omitida, cifra invalida y cero explicito se
/// distinguen: las dos primeras son comision desconocida (con reserva se retiene el estimado
/// entero; sin reserva, veto hasta una REST pedida despues), el cero es cero. Un cargo explicito de
/// otro activo queda atribuido a su reserva sin restarse como USDT.
#[test]
fn parser_fill_t_omitida_cero_y_comision_desconocida() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    // T omitida y T = 0: el cargo de 0,30 no se pierde y nunca se acredita por marca.
    for t in [None, Some(0)] {
        let m = montaje();
        let s = m.streamer();
        let _ = m.reservar(0, 0.04, "cL_v4j_sin_t", Some(1.0));
        s.dispatch(&fill("cL_v4j_sin_t", "TRADE", "FILLED", 51, "1", "1", t, Some("0.30"), Some("USDT")));
        assert_eq!(m.reservado(), 0.30, "T={t:?}: el frame se descarto o el cargo se perdio");
        s.dispatch(&cuenta("99.70", 950));
        assert_eq!(m.reservado(), 0.30, "T={t:?}: la ausencia de marca se convirtio en prueba");
    }
    // Comision ausente, ilegible o sin activo (`N` omitida o vacia) con reserva: se retiene el
    // estimado entero (0,04) aunque la orden se ejecute entera y, como no hay cota demostrable
    // del cargo (el estimado puede ser menor, o cero), la admision queda vetada hasta la
    // evidencia. El cero explicito en USDT, con la orden ejecutada, no retiene ni veta.
    for (n, asset, esperado, veta) in [
        (None, Some("USDT"), 0.04, true),
        (Some("abc"), Some("USDT"), 0.04, true),
        (Some("0.02"), None, 0.04, true),
        (Some("0.02"), Some(""), 0.04, true),
        (Some("0"), Some("USDT"), 0.0, false),
    ] {
        let m = montaje();
        let s = m.streamer();
        let _ = m.reservar(0, 0.04, "cL_v4j_n", Some(1.0));
        s.dispatch(&fill("cL_v4j_n", "TRADE", "FILLED", 61, "1", "1", Some(300), n, asset));
        assert_eq!(m.vetado(), veta, "n={n:?} N={asset:?}: veto");
        if !veta {
            s.dispatch(&cuenta("100", 301));
        }
        assert_eq!(m.reservado(), esperado, "n={n:?} N={asset:?}");
    }
    // Con estimado CERO, una comision desconocida no deja capacidad completa: veta.
    {
        let m = montaje();
        let s = m.streamer();
        let _ = m.reservar(0, 0.0, "cL_v4j_cero", Some(1.0));
        s.dispatch(&fill("cL_v4j_cero", "TRADE", "FILLED", 62, "1", "1", Some(300), None, Some("USDT")));
        assert!(m.vetado() && m.arena.capital_incompleto.load(Ordering::Acquire), "estimado cero y cargo desconocido: admitido");
    }
    // Terminal recibido ANTES que su fill: el terminal (acumulado 1) cubierto por una foto no
    // libera el estimado mientras el fill no llego; el fill tardio con T se acredita por su marca.
    {
        let m = montaje();
        let s = m.streamer();
        let clave = m.reservar(0, 0.04, "cL_v4j_tarde", Some(1.0));
        s.dispatch(&fill("cL_v4j_tarde", "EXPIRED", "FILLED", 0, "0", "1", Some(400), Some("0"), Some("USDT")));
        m.arena.config.entry_publication_gate.lock().unwrap().ejecutar(&m.arena, clave, 0.0, Some(400));
        s.dispatch(&cuenta("99.96", 500));
        assert_eq!(m.reservado(), 0.04, "el terminal libero el estimado sin el fill");
        s.dispatch(&fill("cL_v4j_tarde", "TRADE", "FILLED", 63, "1", "1", Some(450), Some("0.03"), Some("USDT")));
        assert_eq!(m.reservado(), 0.0, "el fill tardio con T450 ya estaba en la foto T500");
    }
    // Comision desconocida sin reserva que la respalde: no hay cifra que retener, veto.
    {
        let m = montaje();
        let s = m.streamer();
        s.dispatch(&fill("cL_v4j_suelto", "TRADE", "FILLED", 71, "1", "1", Some(300), None, Some("USDT")));
        assert!(m.vetado(), "un cargo desconocido sin reserva no veto");
        assert!(m.arena.capital_incompleto.load(Ordering::Acquire), "el puente no pidio la evidencia perdida");
        // La REST pedida despues suelta el veto (evidencia positiva).
        m.puente.observar_rest(m.puente.ticket(), &rest(99.9, 400)).unwrap();
        assert!(!m.vetado() && !m.arena.capital_incompleto.load(Ordering::Acquire), "la REST posterior no solto el veto");
    }
    // Otro activo explicito: atribuido a su reserva (la orden ejecutada entera no retiene el
    // estimado) y sin restar del USDT; el mismo fill en USDT si retiene su comision.
    for (asset, esperado) in [("BNB", 0.0), ("USDT", 0.5)] {
        let m = montaje();
        let s = m.streamer();
        let _ = m.reservar(0, 0.04, "cL_v4j_bnb", Some(1.0));
        s.dispatch(&fill("cL_v4j_bnb", "TRADE", "FILLED", 81, "1", "1", Some(300), Some("0.5"), Some(asset)));
        assert_eq!(m.reservado(), esperado, "N={asset}");
        assert_eq!(m.dim(), 100.0 - esperado, "N={asset}: la comision de otro activo se resto del USDT");
    }
}

/// A2: con duplicados y relevo en la MISMA cuenta, la comision valida se registra una vez y la
/// ausencia de marca no se convierte en prueba; la transicion a OTRA cuenta conserva su
/// aislamiento (el mismo `trade_id` de la otra cuenta es otro cargo).
#[test]
fn identidad_de_cargo_repetido_y_tras_relevo() {
    let _g = REGISTRO.lock().unwrap_or_else(|p| p.into_inner());
    let m = montaje();
    let s = m.streamer();
    let _ = m.reservar(0, 0.04, "cL_v4j_rep", Some(1.0));
    let frame = fill("cL_v4j_rep", "TRADE", "FILLED", 91, "1", "1", None, Some("0.20"), Some("USDT"));
    s.dispatch(&frame);
    s.dispatch(&frame);
    assert_eq!(m.reservado(), 0.20, "duplicado en el mismo socket");
    // Relevo en la misma cuenta (era nueva) y el socket nuevo repite el fill.
    let era = m.puente.nueva_era_de_sesion(1);
    m.puente.abrir_socket(era).unwrap();
    m.puente.observar_rest(m.puente.ticket(), &rest(99.8, 700)).unwrap();
    let nuevo = m.streamer();
    nuevo.dispatch(&frame);
    assert_eq!(m.reservado(), 0.20, "el relevo de la misma cuenta cargo dos veces");
    nuevo.dispatch(&cuenta("99.80", 800));
    assert_eq!(m.reservado(), 0.20, "tras el relevo la ausencia de marca se tomo por prueba");
    // Fill suelto (sin reserva) con T: en la misma cuenta, una vez.
    let suelto = fill("cL_v4j_otra", "TRADE", "FILLED", 92, "1", "1", Some(810), Some("0.10"), Some("USDT"));
    nuevo.dispatch(&suelto);
    nuevo.dispatch(&suelto);
    assert_eq!(m.reservado(), 0.20 + 0.10, "fill suelto repetido");
    // Otra cuenta: el mismo trade_id es otro cargo (no se deduplica contra la anterior).
    {
        let mut libro = m.arena.config.entry_publication_gate.lock().unwrap();
        libro.fijar_cuenta(&m.arena, 1);
        libro.fijar_cuenta(&m.arena, 2);
        libro.saldo_aceptado(&m.arena, 50.0, Some(100));
        libro.fill_de_entrada(&m.arena, m.coin(), "cL_v4j_otra", 92, Some(0.10), Some(810));
    }
    assert_eq!(m.reservado(), 0.20 + 0.10 + 0.10, "la otra cuenta heredo la identidad del cargo de la anterior");
}
