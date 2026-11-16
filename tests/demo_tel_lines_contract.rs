//! DEMO_TEL T1 (pedido 20261112-0949a): las once lineas de telemetria de la Demo.
//!
//! Una prueba por linea. Cada una fija, contra la fuente real (`include_str!`, CRLF
//! normalizado):
//! 1. el texto exacto de la macro y su lista de argumentos (una sola aparicion);
//! 2. el sitio: la linea sale en el evento (orden, fill, cierre, arranque, apagado,
//!    disparo), entre anclas literales, nunca suelta en el bucle por tick;
//! 3. que una linea sintetica formateada con ESE MISMO texto casa con la expresion
//!    regular de la especificacion (`agy_telemetria_anexo_pedido.md`), comprobada
//!    con un comparador por campos equivalente (el workspace no trae `regex`).
//!
//! Ademas: el formato de la linea `💧 [FILL]` existente no cambia (punto 4 del pedido).
//!
//! T2 (QRA-TELEMETRIA-DEMO, cinco desviaciones; solo cambia el CONTENIDO de las lineas):
//! ORDER_SEND declarada como INTENTO y su testigo ORDER_RESULT; TRADE_FEE con su tradeId;
//! FUNDING con alcance y cobertura N/D; SYMBOL_SPEC con su fuente; IMMUNE_TRIGGER con cada dato
//! bajo su nombre real y sin relecturas, busquedas ni E/S antes del kill-switch. CAPITAL_UPDATE sale
//! ahora del sink de `src/demo_tel.rs` (prueba real en `demo_tel_capital_update_sink_contract.rs`).

use execution_engine::entry_dispatch::EntryRoute;
use execution_engine::order_registry::OrderStatus;

const HOST: &str = include_str!("../src/bin/god_engine.rs");
const STREAM: &str = include_str!("../crates/execution-engine/src/user_data_stream.rs");
const NUCLEO: &str = include_str!("../crates/god-engine-core/src/lib.rs");
const DEMO_TEL_MOD: &str = include_str!("../src/demo_tel.rs");
// C3: el resolutor de la entrada que V4M saco del host (ancla del testigo ORDER_RESULT).
const HOST_OMS: &str = include_str!("../src/host_oms.rs");

/// El texto entre dos anclas (la primera aparicion de `desde` y la siguiente de `hasta`).
fn tramo(fuente: &str, desde: &str, hasta: &str) -> String {
    let f = texto(fuente);
    let a = f.split_once(desde).unwrap_or_else(|| panic!("ancla ausente: {desde}")).1;
    a.split_once(hasta).unwrap_or_else(|| panic!("ancla ausente: {hasta}")).0.to_string()
}

fn texto(fuente: &str) -> String {
    fuente.replace("\r\n", "\n")
}

/// Las anclas aparecen en este orden en la fuente.
fn orden(fuente: &str, pasos: &[&str]) {
    let fuente = texto(fuente);
    let mut resto = fuente.as_str();
    for paso in pasos {
        resto = resto
            .split_once(paso)
            .unwrap_or_else(|| panic!("ancla ausente o fuera de orden: {paso}"))
            .1;
    }
}

/// Fija el texto de la macro (una sola vez en la fuente) y sus argumentos, y devuelve
/// la linea sintetica formateada con ese mismo texto.
macro_rules! demo_tel {
    ($src:expr, $args:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {{
        let src = texto($src);
        let lit = concat!("\"", $fmt, "\"");
        assert_eq!(src.matches(lit).count(), 1, "DEMO_TEL: el texto de la macro no esta una sola vez: {lit}");
        let pos = src.find(lit).unwrap();
        let tras = &src[pos + lit.len()..];
        let tras = &tras[..tras.find(");").expect("cierre de la macro")];
        assert!(
            tras.contains($args),
            "DEMO_TEL: los argumentos de la macro cambiaron: esperado {:?}, hallado {:?}",
            $args,
            tras
        );
        format!($fmt $(, $arg)*)
    }};
}

#[derive(Clone, Copy, Debug)]
enum C {
    /// `\d+`
    Dig,
    /// `[\d\.]+`
    Num,
    /// `[\d\.\-]+`
    Sig,
    /// `true|false`
    Bool,
    /// `\w+`
    Word,
    /// `\S+`
    NoEsp,
    /// `LONG|SHORT`
    Dir,
    /// `BUY|SELL`
    Lado,
    /// `.*` (solo el ultimo campo)
    Resto,
    /// valor literal
    Lit(&'static str),
}

fn casa_valor(c: C, v: &str) -> bool {
    let no_vacio = !v.is_empty();
    match c {
        C::Dig => no_vacio && v.chars().all(|x| x.is_ascii_digit()),
        C::Num => no_vacio && v.chars().all(|x| x.is_ascii_digit() || x == '.'),
        C::Sig => no_vacio && v.chars().all(|x| x.is_ascii_digit() || x == '.' || x == '-'),
        C::Bool => v == "true" || v == "false",
        C::Word => no_vacio && v.chars().all(|x| x.is_alphanumeric() || x == '_'),
        C::NoEsp => no_vacio && !v.chars().any(char::is_whitespace),
        C::Dir => v == "LONG" || v == "SHORT",
        C::Lado => v == "BUY" || v == "SELL",
        C::Resto => true,
        C::Lit(l) => v == l,
    }
}

/// `^.*DEMO_TEL\s+event=<ev>\s+k1=(c1)\s+k2=(c2)...$`, campo por campo y en orden.
/// Devuelve los valores capturados.
fn casa(linea: &str, ev: &str, campos: &[(&str, C)]) -> Vec<String> {
    let cab = format!("DEMO_TEL event={ev} ");
    let i = linea.find(&cab).unwrap_or_else(|| panic!("sin cabecera {cab:?}: {linea}"));
    let mut resto = &linea[i + cab.len()..];
    let mut valores = Vec::new();
    for (n, (k, c)) in campos.iter().enumerate() {
        let pref = format!("{k}=");
        assert!(resto.starts_with(&pref), "campo {k} esperado en {resto:?} ({linea})");
        resto = &resto[pref.len()..];
        let v = if matches!(c, C::Resto) || n + 1 == campos.len() {
            let v = resto;
            resto = "";
            v
        } else {
            let fin = resto.find(char::is_whitespace).unwrap_or_else(|| panic!("falta separador tras {k}"));
            let v = &resto[..fin];
            resto = resto[fin..].trim_start();
            v
        };
        assert!(casa_valor(*c, v), "valor {v:?} de {k} no casa con {c:?} ({linea})");
        valores.push(v.to_string());
    }
    assert!(resto.is_empty(), "sobra texto tras los campos: {resto:?}");
    valores
}

/// El comparador muerde: una mutacion de campo o de orden no casa.
#[test]
fn demo_tel_comparador_rechaza_mutaciones() {
    let campos = [("ts", C::Dig), ("symbol", C::Word)];
    let r = std::panic::catch_unwind(|| casa("DEMO_TEL event=X symbol=A ts=1", "X", &campos));
    assert!(r.is_err(), "orden de campos alterado y aun casa");
    let r = std::panic::catch_unwind(|| casa("DEMO_TEL event=X ts=1a symbol=A", "X", &campos));
    assert!(r.is_err(), "ts no numerico y aun casa");
    let r = std::panic::catch_unwind(|| casa("DEMO_TEL event=X ts=1 symbol=A extra=1", "X", &campos));
    assert!(r.is_err(), "campo sobrante y aun casa");
    assert_eq!(casa("[ts=5] DEMO_TEL event=X ts=1 symbol=A", "X", &campos), vec!["1", "A"]);
}

// 1. ORDER_SEND ------------------------------------------------------------------------------

#[test]
fn demo_tel_order_send() {
    orden(HOST, &[
        "let entry_request = EntryRequest {",
        "DEMO_TEL event=ORDER_SEND",
        "let entry_result = entry_oms.dispatch_entry_reservada(",
    ]);
    let (ts, parsed_sym_str, final_is_long, final_qty, client_id, effective_leverage) =
        (1_763_000_000_000u64, "BTCUSDT", true, 0.0123f64, "cL_abc123", 5u32);
    for route in [EntryRoute::Market, EntryRoute::Ioc { price: 101.5 }] {
        let linea = demo_tel!(
            HOST,
            "ts, parsed_sym_str, final_is_long, final_qty, client_id, effective_leverage, route",
            "DEMO_TEL event=ORDER_SEND tipo=INTENTO ts={} symbol={} is_long={} qty={:.6} client_order_id={} lev={} route={:?}",
            ts, parsed_sym_str, final_is_long, final_qty, client_id, effective_leverage, route
        );
        let v = casa(&linea, "ORDER_SEND", &[
            ("tipo", C::Lit("INTENTO")), ("ts", C::Dig), ("symbol", C::Word), ("is_long", C::Bool), ("qty", C::Num),
            ("client_order_id", C::NoEsp), ("lev", C::Dig), ("route", C::Resto),
        ]);
        assert_eq!(v[4], "0.012300");
        assert_eq!(v[7], format!("{route:?}"));
    }
}

/// T2 (desviacion 1): testigo del despacho, despues de `dispatch_entry_reservada` y antes de
/// las ramas que deciden. Su clasificacion usa EXACTAMENTE las marcas del primer brazo del match
/// (AMBIGUOUS / MAKER_CHASE_UNVERIFIED); el match que decide no cambia.
#[test]
fn demo_tel_order_result_testigo() {
    orden(HOST, &[
        "DEMO_TEL event=ORDER_SEND tipo=INTENTO",
        "let entry_result = entry_oms.dispatch_entry_reservada(",
        "Some(reservation.clave()),",
        ").await;",
        "let resultado_envio = match &entry_result {",
        "Ok(()) => \"ENTREGADA\",",
        "Err(e) if e.starts_with(\"AMBIGUOUS\") || e.starts_with(\"MAKER_CHASE_UNVERIFIED\") => \"AMBIGUA\",",
        "Err(_) => \"RECHAZADA\",",
        "DEMO_TEL event=ORDER_RESULT",
        // C3 (anclas adaptadas): V4M (D1, L-A1H) movio el `match entry_result` del host a
        // `host_oms::resolver_resultado_de_entrada`, que recibe el MISMO `entry_result` despues
        // del testigo. Antes: `match entry_result {`, AMBIGUOUS/MAKER_CHASE_UNVERIFIED,
        // `Err(e) => {`, `Ok(()) => {` en el host.
        "let desenlace = quantum_engine::host_oms::resolver_resultado_de_entrada(",
        "entry_result,",
    ]);
    // La clasificacion del testigo es la misma que la del resolutor (y en el mismo orden).
    orden(HOST_OMS, &[
        "pub async fn resolver_resultado_de_entrada(",
        "let (mut destino, envio_ambiguo) = match resultado {",
        "Err(e) if e.starts_with(\"AMBIGUOUS\") || e.starts_with(\"MAKER_CHASE_UNVERIFIED\") => {",
        "Err(_) => (DestinoReserva::Revertir, false),",
        "Ok(()) => {",
    ]);
    // El tramo del testigo no decide nada: ni rollback, ni reserva, ni reintento.
    let t = tramo(HOST, "let resultado_envio = match &entry_result {", "quantum_engine::host_oms::resolver_resultado_de_entrada(");
    for prohibido in ["rollback", "release_for", "confirmar", "dispatch_", ".await", "return"] {
        assert!(!t.contains(prohibido), "el testigo hace algo mas que registrar: {prohibido}");
    }
    // La misma clasificacion, evaluada.
    let clase = |r: &Result<(), String>| match r {
        Ok(()) => "ENTREGADA",
        Err(e) if e.starts_with("AMBIGUOUS") || e.starts_with("MAKER_CHASE_UNVERIFIED") => "AMBIGUA",
        Err(_) => "RECHAZADA",
    };
    assert_eq!(clase(&Ok(())), "ENTREGADA");
    assert_eq!(clase(&Err("AMBIGUOUS: timeout".into())), "AMBIGUA");
    assert_eq!(clase(&Err("MAKER_CHASE_UNVERIFIED: x".into())), "AMBIGUA");
    assert_eq!(clase(&Err("-2019 margin".into())), "RECHAZADA");
    let (parsed_sym_str, client_id) = ("BTCUSDT", "cL_abc123");
    let linea = demo_tel!(
        HOST,
        "parsed_sym_str, client_id, resultado_envio",
        "DEMO_TEL event=ORDER_RESULT ts={} symbol={} client_order_id={} resultado={}",
        1_763_000_000_001u64, parsed_sym_str, client_id, clase(&Err("AMBIGUOUS".into()))
    );
    let v = casa(&linea, "ORDER_RESULT", &[
        ("ts", C::Dig), ("symbol", C::Word), ("client_order_id", C::NoEsp), ("resultado", C::Word),
    ]);
    assert_eq!(v[3], "AMBIGUA");
}

// 2. FILL ------------------------------------------------------------------------------------

#[test]
fn demo_tel_fill() {
    orden(STREAM, &[
        "if new_fill {",
        "\"💧 [FILL] {} {} {} {} @ {} (acum {}/{} status {:?})\"",
        "DEMO_TEL event=FILL",
        "let close_kind = if new_fill {",
    ]);
    let linea = demo_tel!(
        STREAM,
        "update.trade_time_ms, update.symbol, update.side, update.last_filled_price, update.last_filled_qty, update.cumulative_filled_qty, update.client_order_id, update.trade_id, o.is_maker, update.status",
        "DEMO_TEL event=FILL ts={} symbol={} side={} price={:.6} qty={:.6} cum_qty={:.6} client_order_id={} trade_id={} is_maker={} status={:?}",
        1_763_000_000_123u64, "VETUSDT", "SELL", 0.00512f64, 996.0f64, 996.0f64, "cS_x1", 42u64, false, OrderStatus::Filled
    );
    let v = casa(&linea, "FILL", &[
        ("ts", C::Dig), ("symbol", C::Word), ("side", C::Lado), ("price", C::Num), ("qty", C::Num),
        ("cum_qty", C::Num), ("client_order_id", C::NoEsp), ("trade_id", C::Dig), ("is_maker", C::Bool),
        ("status", C::Word),
    ]);
    assert_eq!(v[3], "0.005120");
    assert_eq!(v[9], "Filled");
}

/// Punto 4 del pedido: la linea `💧 [FILL]` existente conserva texto y argumentos.
#[test]
fn demo_tel_la_linea_fill_existente_no_cambia() {
    let src = texto(STREAM);
    let lit = "\"💧 [FILL] {} {} {} {} @ {} (acum {}/{} status {:?})\",";
    assert_eq!(src.matches(lit).count(), 1, "M1-REPORTE: el formato de la linea 💧 [FILL] cambio");
    let tras = &src[src.find(lit).unwrap() + lit.len()..];
    let args: Vec<&str> = tras[..tras.find(");").unwrap()].split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    assert_eq!(args, [
        "update.symbol", "update.side", "update.order_type", "update.client_order_id",
        "update.last_filled_price", "update.cumulative_filled_qty", "update.orig_qty", "update.status",
    ]);
    // La linea nueva no contiene `[FILL]`: `scripts/extraer_metricas_a3.py` no la cuenta dos veces.
    assert!(!"DEMO_TEL event=FILL".contains("[FILL]"));
}

// 3. TRADE_FEE -------------------------------------------------------------------------------

#[test]
fn demo_tel_trade_fee() {
    orden(STREAM, &[
        "crate::trade_accounting::record_entry_fill(",
        "DEMO_TEL event=TRADE_FEE",
        "CAPITAL-UNICO V4H (INV-G): todo fill que cierra posicion",
    ]);
    // T2 (desviacion 2): el tradeId nativo del MISMO fill (`update.trade_id`, el que lleva la
    // linea FILL), AUSENTE si el evento no lo trajo; nunca el clientOrderId ni otro sustituto.
    orden(STREAM, &[
        "crate::trade_accounting::record_entry_fill(",
        "let fee_trade_id = if update.trade_id == 0 { \"AUSENTE\".to_string() } else { update.trade_id.to_string() };",
        "let fee_asset = if update.commission_asset.is_empty() { \"AUSENTE\" } else { update.commission_asset.as_str() };",
        "DEMO_TEL event=TRADE_FEE",
    ]);
    let linea = demo_tel!(
        STREAM,
        "update.trade_time_ms, update.symbol, update.client_order_id, fee_trade_id, update.commission, fee_asset, o.is_maker",
        "DEMO_TEL event=TRADE_FEE ts={} symbol={} client_order_id={} trade_id={} fee={:.8} fee_asset={} is_maker={} alcance=ENTRADA",
        1_763_000_000_124u64, "VETUSDT", "cS_x1", 42u64, 0.00203904f64, "USDT", false
    );
    let v = casa(&linea, "TRADE_FEE", &[
        ("ts", C::Dig), ("symbol", C::Word), ("client_order_id", C::NoEsp), ("trade_id", C::Dig), ("fee", C::Sig),
        ("fee_asset", C::Word), ("is_maker", C::Bool), ("alcance", C::Lit("ENTRADA")),
    ]);
    assert_eq!((v[3].as_str(), v[4].as_str()), ("42", "0.00203904"));
    // Ausencia explicita: tambien casa (palabra AUSENTE), y el valor no es el clientOrderId.
    let ausente = demo_tel!(
        STREAM,
        "update.trade_time_ms, update.symbol, update.client_order_id, fee_trade_id, update.commission, fee_asset, o.is_maker",
        "DEMO_TEL event=TRADE_FEE ts={} symbol={} client_order_id={} trade_id={} fee={:.8} fee_asset={} is_maker={} alcance=ENTRADA",
        1u64, "VETUSDT", "cS_x1", "AUSENTE", 0.0f64, "AUSENTE", false
    );
    let v = casa(&ausente, "TRADE_FEE", &[
        ("ts", C::Dig), ("symbol", C::Word), ("client_order_id", C::NoEsp), ("trade_id", C::Lit("AUSENTE")),
        ("fee", C::Sig), ("fee_asset", C::Lit("AUSENTE")), ("is_maker", C::Bool), ("alcance", C::Lit("ENTRADA")),
    ]);
    assert_ne!(v[3], v[2]);
}

// 4. FUNDING ---------------------------------------------------------------------------------

#[test]
fn demo_tel_funding() {
    // En la pasada del FEE-BREAKER sobre las filas de income, justo antes del match, cuyo
    // brazo `"FUNDING_FEE" => a.funding += e.income,` no cambia (lo anclan los contratos V4H).
    orden(HOST, &[
        "for e in fee_partition.same_asset_by_symbol.values().flatten() {",
        "if e.income_type == \"FUNDING_FEE\" {",
        "DEMO_TEL event=FUNDING",
        "match e.income_type.as_str() {",
        "\"FUNDING_FEE\" => a.funding += e.income,",
    ]);
    // T2 (desviacion 3): alcance declarado (la pasada rodante del FEE-BREAKER con su ventana, el
    // filtro de la particion y el tope de paginas de la llamada) y cobertura N/D (no se deduce).
    orden(HOST, &[
        "let window_start = now_breaker.saturating_sub(window_ms);",
        ".fetch_income_paged(&[], window_start, 4)",
        "partition_legacy_fee_evidence(&entries);",
        "for e in fee_partition.same_asset_by_symbol.values().flatten() {",
        "if e.symbol.is_empty() {",
        "let tran_id_funding = if e.tran_id == 0 { \"AUSENTE\".to_string() } else { e.tran_id.to_string() };",
        "DEMO_TEL event=FUNDING",
    ]);
    let linea = demo_tel!(
        HOST,
        "e.time, e.symbol, e.income, e.asset, tran_id_funding, window_start, now_breaker",
        "DEMO_TEL event=FUNDING ts={} symbol={} income={:.8} asset={} tran_id={} income_type=FUNDING_FEE alcance=FEE_BREAKER_VENTANA_RODANTE ventana_desde_ms={} ventana_hasta_ms={} filtro=MISMO_ACTIVO_Y_SIMBOLO_NO_VACIO paginas_max=4 cobertura=N/D",
        1_763_000_000_000u64, "BTCUSDT", -0.01234567f64, "USDT", 9_876_543_210u64, 1_762_900_000_000u64, 1_763_000_100_000u64
    );
    let v = casa(&linea, "FUNDING", &[
        ("ts", C::Dig), ("symbol", C::Word), ("income", C::Sig), ("asset", C::Word), ("tran_id", C::Dig),
        ("income_type", C::Lit("FUNDING_FEE")), ("alcance", C::Lit("FEE_BREAKER_VENTANA_RODANTE")),
        ("ventana_desde_ms", C::Dig), ("ventana_hasta_ms", C::Dig), ("filtro", C::Lit("MISMO_ACTIVO_Y_SIMBOLO_NO_VACIO")),
        ("paginas_max", C::Lit("4")), ("cobertura", C::Lit("N/D")),
    ]);
    assert_eq!((v[2].as_str(), v[4].as_str()), ("-0.01234567", "9876543210"), "importe con signo y tranId");
}

// 5. UNREALIZED_PNL --------------------------------------------------------------------------

#[test]
fn demo_tel_unrealized_pnl() {
    // Dentro del bloque de cada 100 mensajes, no por tick.
    orden(HOST, &[
        "if msg_count.is_multiple_of(100) {",
        "total_unrealized_pnl += coin.metrics.pnl_unrealized.load(Ordering::Relaxed);",
        "DEMO_TEL event=UNREALIZED_PNL",
        "telemetry_server::profiler::update_financials_atomic(",
        "if msg_count.is_multiple_of(5000) {",
    ]);
    let linea = demo_tel!(
        HOST,
        "ts_now, total_unrealized_pnl, open_positions",
        "DEMO_TEL event=UNREALIZED_PNL ts={} total_upnl={:.6} open_positions={}",
        1_763_000_000_000u64, -1.25f64, 2usize
    );
    let v = casa(&linea, "UNREALIZED_PNL", &[("ts", C::Dig), ("total_upnl", C::Sig), ("open_positions", C::Dig)]);
    assert_eq!(v[1], "-1.250000");
}

// 6. CAPITAL_UPDATE --------------------------------------------------------------------------

#[test]
fn demo_tel_capital_update() {
    // T2: el sink vive en `src/demo_tel.rs`; el host lo usa (no define otro).
    orden(DEMO_TEL_MOD, &[
        "fn capital(&self, usdt: f64, revision: Option<&RevisionSaldo>) {",
        "audit_engine::telemetry::update_dynamic_capital(usdt);",
        // T3 (T3-1, ancla adaptada): la línea sale por `registrar_revision`, que no repite
        // una revisión idéntica y escribe con el mismo `emitir` (testigo + telemetry_log!).
        "registrar_revision(usdt, revision);",
        "self.unified_capital.store(usdt.to_bits(), Ordering::Relaxed);",
        "fn on_equity(&self, usdt: f64) {",
    ]);
    orden(DEMO_TEL_MOD, &[
        "fn registrar_revision(saldo: f64, revision: Option<&RevisionSaldo>) {",
        "emitir(linea_capital_update(ts_ahora_ms(), saldo, revision));",
    ]);
    assert!(texto(HOST).contains("use quantum_engine::demo_tel::CapitalBridgeSink;"));
    assert!(!texto(HOST).contains("struct CapitalBridgeSink"));
    assert_eq!(texto(DEMO_TEL_MOD).matches("\"DEMO_TEL event=CAPITAL_UPDATE").count(), 1);
    let rev = execution_engine::user_data_stream::RevisionSaldo {
        fuente: "WS", era: 2, socket: 5, marca_ms: Some(200), hora_evento_ms: None,
    };
    let linea = quantum_engine::demo_tel::linea_capital_update(1_763_000_000_000, 100.0, Some(&rev));
    let v = casa(&linea, "CAPITAL_UPDATE", &[
        ("ts", C::Dig), ("wallet_balance", C::Num), ("fuente", C::Lit("WS")), ("era", C::Dig), ("socket", C::Dig),
        ("marca_ms", C::Dig), ("hora_evento_ms", C::Lit("N/D")),
    ]);
    // T3 (T3-1): el saldo con su representación exacta, sin redondeo a cuatro decimales.
    assert_eq!((v[1].as_str(), v[5].as_str()), ("100", "200"));
}

// 7. DECISION_MID ----------------------------------------------------------------------------

#[test]
fn demo_tel_decision_mid() {
    // Una vez por orden de apertura admitida (tras reservarla), no por tick.
    orden(NUCLEO, &[
        "self.entry_reservations[coin_id] = Some(reservation);",
        "🚀 [OPEN TRACE]",
        "DEMO_TEL event=DECISION_MID",
    ]);
    let (bid, ask) = (100.0f64, 100.2f64);
    let linea = demo_tel!(
        NUCLEO,
        "event_time_ms, self.diag_opened, sym, target_pos_slot, if is_long { \"LONG\" } else { \"SHORT\" }, mid_price, bid, ask",
        "DEMO_TEL event=DECISION_MID ts={} signal_id={} symbol={} slot={} dir={} mid={:.6} bid={:.6} ask={:.6}",
        1_763_000_000_000u64, 7u64, "BTCUSDT", 1usize, "LONG", (bid + ask) / 2.0, bid, ask
    );
    let v = casa(&linea, "DECISION_MID", &[
        ("ts", C::Dig), ("signal_id", C::Dig), ("symbol", C::Word), ("slot", C::Dig), ("dir", C::Dir),
        ("mid", C::Num), ("bid", C::Num), ("ask", C::Num),
    ]);
    let (mid, b, a): (f64, f64, f64) = (v[5].parse().unwrap(), v[6].parse().unwrap(), v[7].parse().unwrap());
    assert!(b <= mid && mid <= a);
}

// 8. TRADE_CLOSE -----------------------------------------------------------------------------

#[test]
fn demo_tel_trade_close() {
    // Una vez por cierre, tras el CLOSE TRACE y antes de la comision de salida.
    orden(NUCLEO, &[
        "for (slot_idx, pos) in coin.positions.slots().into_iter().enumerate() {",
        "let is_long = pos.is_long.load(Ordering::Relaxed);",
        "🚪 [CLOSE TRACE]",
        "DEMO_TEL event=TRADE_CLOSE",
        "let Some(close_fee) = self.reality",
    ]);
    let linea = demo_tel!(
        NUCLEO,
        "event_time_ms, self.diag_close_total, sym, slot_idx, if is_long { \"LONG\" } else { \"SHORT\" }, entry, exit_price, gross_pnl, reason",
        "DEMO_TEL event=TRADE_CLOSE ts={} close_id={} symbol={} slot={} dir={} entry_price={:.6} exit_price={:.6} gross_pnl={:.6} reason={}",
        1_763_000_000_000u64, 3u64, "BTCUSDT", 0usize, "SHORT", 100.0f64, 101.0f64, -0.5f64, "SL"
    );
    let v = casa(&linea, "TRADE_CLOSE", &[
        ("ts", C::Dig), ("close_id", C::Dig), ("symbol", C::Word), ("slot", C::Dig), ("dir", C::Dir),
        ("entry_price", C::Num), ("exit_price", C::Num), ("gross_pnl", C::Sig), ("reason", C::NoEsp),
    ]);
    assert_eq!(v[7], "-0.500000");
}

// 9. AUDIT_SUMMARY ---------------------------------------------------------------------------

#[test]
fn demo_tel_audit_summary() {
    orden(HOST, &[
        "Ok((cancelaciones, cerradas)) => {",
        "🛑 [SHUTDOWN] Flatten-all completado",
        "DEMO_TEL event=AUDIT_SUMMARY",
        "🛑 [SHUTDOWN] Flatten-all con error",
    ]);
    let linea = demo_tel!(
        HOST,
        "ts, cerradas, cancelaciones",
        "DEMO_TEL event=AUDIT_SUMMARY ts={} status=ORDERLY_SHUTDOWN closed_positions={} cancelled_orders={}",
        1_763_000_000_000u64, 2usize, 5usize
    );
    let v = casa(&linea, "AUDIT_SUMMARY", &[
        ("ts", C::Dig), ("status", C::Lit("ORDERLY_SHUTDOWN")), ("closed_positions", C::Dig), ("cancelled_orders", C::Dig),
    ]);
    assert_eq!((v[2].as_str(), v[3].as_str()), ("2", "5"));
}

// 10. SYMBOL_SPEC ----------------------------------------------------------------------------

#[test]
fn demo_tel_symbol_spec() {
    // Al arranque, antes de que `update_registry` consuma el vector.
    orden(HOST, &[
        "let boot_specs: Vec<quantum_arena::symbol_registry::SymbolSpec> = symbols",
        "for spec in &boot_specs {",
        "DEMO_TEL event=SYMBOL_SPEC",
        "quantum_arena::symbol_registry::update_registry(boot_specs);",
        "🧬 [B3.38]",
    ]);
    // T2 (desviacion 4): la fuente del dato. El vector sale de la semilla local del registro
    // (`get_official_binance_spec`), no de `exchangeInfo`; la linea lo dice.
    orden(HOST, &[
        "let boot_specs: Vec<quantum_arena::symbol_registry::SymbolSpec> = symbols",
        ".map(|s| quantum_arena::symbol_registry::get_official_binance_spec(&s.to_uppercase()))",
        "DEMO_TEL event=SYMBOL_SPEC",
    ]);
    let spec = quantum_arena::symbol_registry::get_official_binance_spec("BTCUSDT");
    let linea = demo_tel!(
        HOST,
        "spec.symbol, spec.min_notional, spec.tick_size, spec.step_size",
        "DEMO_TEL event=SYMBOL_SPEC symbol={} min_notional={:.4} tick_size={:.8} step_size={:.8} fuente=SEMILLA_LOCAL exchange_info=N/D",
        spec.symbol, spec.min_notional, spec.tick_size, spec.step_size
    );
    let v = casa(&linea, "SYMBOL_SPEC", &[
        ("symbol", C::Word), ("min_notional", C::Num), ("tick_size", C::Num), ("step_size", C::Num),
        ("fuente", C::Lit("SEMILLA_LOCAL")), ("exchange_info", C::Lit("N/D")),
    ]);
    assert_eq!(v[0], "BTCUSDT");
}

// 11. IMMUNE_TRIGGER -------------------------------------------------------------------------

#[test]
fn demo_tel_immune_trigger() {
    // C3 (decision del coordinador D1-D3, anclas adaptadas): tras el disparo COMPROMETIDO
    // (`comprometer_disparo` bajo el mutex del puente arma el kill-switch) y junto a la linea
    // ACTIVADO; el freno y el pico se leen tal como estan (T7): `cap` (capital del freno) y
    // `peak_capital` del MISMO `paso_del_freno_del_inmune`, calculados ANTES de la decision.
    // Antes (T3): `cap_exchange = puente_imm.equity_del_freno()`, `cap = capital_del_freno`,
    // `registrar_pico`, ACTIVADO, emisor, `arena_imm.kill_switch_active.store`, `trigger_kill_switch`.
    orden(HOST, &[
        "let operator_lock = operator_lock_exists();",
        "let paso_del_freno = quantum_engine::host_oms::paso_del_freno_del_inmune(",
        "(paso_del_freno.cap, paso_del_freno.pico, paso_del_freno.revision, paso_del_freno.ambito);",
        "let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);",
        "let insolvent = cap <= 0.0;",
        "let dd_breach = freno.echado();",
        "let lat_breach = latency_strikes >= 3;",
        "if operator_lock || insolvent || dd_breach || lat_breach {",
        "let Some((executor, autoria)) = puente_imm.con_equity_del_freno(|_| {",
        "execution_engine::aplanado_inmune::comprometer_disparo(",
        "continue;",
        "🚨 [SISTEMA INMUNE] ACTIVADO",
        // T3 (T3-5, ancla adaptada): la línea sale del emisor compartido.
        "quantum_engine::demo_tel::emitir_immune_trigger(",
        "autoria_del_latch = Some(autoria);",
        "latched = true;",
    ]);
    // C3 (D1): el host ya no arma el kill-switch ni toma el ejecutor fuera de `comprometer_disparo`.
    assert!(!texto(HOST).contains("arena_imm.kill_switch_active.store"), "el host vuelve a armar la arena del inmune");
    let rama = tramo(HOST, "if operator_lock || insolvent || dd_breach || lat_breach {", "latched = true;");
    assert!(!rama.contains("trigger_kill_switch"), "el host vuelve a armar el ejecutor del inmune");
    assert!(!rama.contains("kill_switch_active"), "el host vuelve a tocar el kill-switch de la arena");
    assert_eq!(rama.matches("exec_imm.load_full()").count(), 1, "el ejecutor solo se toma dentro de comprometer_disparo");
    assert!(rama.contains("|| exec_imm.load_full(),"));
    // C3 (D2): un relevo que descarta la decision (`continue`) sale ANTES del emisor: sin ACTIVADO
    // no hay IMMUNE_TRIGGER. Un solo emisor en el host.
    let descarte = tramo(HOST, "let Some((executor, autoria)) = puente_imm.con_equity_del_freno(|_| {", "🚨 [SISTEMA INMUNE] ACTIVADO");
    assert!(descarte.contains("}) else {") && descarte.contains("continue;"), "el descarte por relevo ya no corta el ciclo");
    assert!(!descarte.contains("emitir_immune_trigger"), "IMMUNE_TRIGGER antes de comprometer el disparo");
    assert_eq!(texto(HOST).matches("emitir_immune_trigger(").count(), 1, "un solo IMMUNE_TRIGGER en el host");
    // C3 (D3): sin `cap_exchange` en el host del inmune; la equity del freno es el `cap` del paso.
    let inmune = tramo(HOST, "let operator_lock = operator_lock_exists();", "latched = true;");
    assert!(!inmune.contains("cap_exchange"), "el host inventa una equity del freno distinta del paso");
    // T2 (desviacion 5): el motivo sigue la MISMA precedencia que el texto `reason` del disparo.
    orden(HOST, &[
        "let reason = if operator_lock {",
        "} else if insolvent {",
        "} else if dd_breach {",
        "let motivo_imm = if operator_lock {",
        "\"OPERATOR_LOCK\"",
        "} else if insolvent {",
        "\"INSOLVENTE\"",
        "} else if dd_breach {",
        "\"DRAWDOWN\"",
        "\"LATENCIA\"",
    ]);
    // Sin relecturas del arena ni del puente, sin busqueda de simbolo y sin E/S entre la linea
    // ACTIVADO (efecto ya comprometido) y el registro del latch, con el emisor en medio.
    // C3: el tramo termina en `autoria_del_latch = Some(autoria);` (antes, en el store del host).
    let t = tramo(HOST, "🚨 [SISTEMA INMUNE] ACTIVADO", "autoria_del_latch = Some(autoria);");
    for prohibido in [
        ".load(", "arena_imm.", "symbol_registry", "try_symbol", "position(", ".await", "std::fs", "puente_imm.",
        "exec_imm", "kill_switch_active", "trigger_kill_switch",
    ] {
        assert!(!t.contains(prohibido), "IMMUNE_TRIGGER: trabajo nuevo tras comprometer el disparo: {prohibido}");
    }
    // C3 (D3): la equity del freno es el `cap` del paso (antes: `cap_exchange`).
    assert!(t.contains("let equity_del_freno_imm = if cap.is_finite() {"));
    // T3 (T3-5, anclas adaptadas): la línea la escribe el emisor compartido
    // `quantum_engine::demo_tel::emitir_immune_trigger` con los MISMOS operandos; la equity
    // ausente viaja como NaN y el emisor la escribe AUSENTE (sin sustituto).
    assert!(t.contains("{ cap } else { f64::NAN };"), "la equity del freno ausente se declara, no se sustituye");
    orden(HOST, &[
        "quantum_engine::demo_tel::emitir_immune_trigger(&quantum_engine::demo_tel::DisparoInmune {",
        "ts_ms: ts_imm,", "motivo: motivo_imm,", "freno_capital: cap,", "peak_capital,",
        "equity_del_freno: equity_del_freno_imm,", "caida_capital_pct: freno.caida_capital * 100.0,",
        "tope_capital_pct: freno.tope_capital * 100.0,", "operator_lock,", "insolvent,", "dd_breach,", "lat_breach,",
        "latency_strikes,", "autoria_del_latch = Some(autoria);",
    ]);
    assert!(!texto(HOST).contains("\"DEMO_TEL event=IMMUNE_TRIGGER"), "un solo texto de la línea, en el emisor compartido");
    assert!(texto(DEMO_TEL_MOD).contains("\"AUSENTE\""));
    let mut linea = String::new();
    quantum_engine::demo_tel::escribir_immune_trigger(&mut linea, &quantum_engine::demo_tel::DisparoInmune {
        ts_ms: 1_763_000_000_000, motivo: "DRAWDOWN", freno_capital: 1655.81, peak_capital: 7299.88,
        equity_del_freno: f64::NAN, caida_capital_pct: 77.3172, tope_capital_pct: 50.0, operator_lock: false,
        insolvent: false, dd_breach: true, lat_breach: false, latency_strikes: 0,
    })
    .unwrap();
    // Los planos van con su código corto; su significado lo documenta el lector.
    let plano = |k: &str| {
        telemetry_server::report_demo_run::PLANOS_IMMUNE.iter().find(|p| p.0 == k).map(|p| (p.1, p.2)).unwrap()
    };
    assert_eq!(plano("freno_capital_plano"), ("FRENO", "drawdown.capital_del_freno(min(arena.unified_capital,arena.capital_exchange))"));
    assert_eq!(plano("peak_plano"), ("PICO", "drawdown.registrar_pico(arena.pico_capital,freno_capital)"));
    assert_eq!(plano("equity_del_freno_plano"), ("PUENTE", "puente.equity_del_freno"));
    let v = casa(&linea, "IMMUNE_TRIGGER", &[
        ("ts", C::Dig), ("motivo", C::Word), ("alcance", C::Lit("CUENTA")), ("freno_capital", C::Num),
        ("freno_capital_plano", C::Lit("FRENO")),
        ("peak_capital", C::Num),
        ("peak_plano", C::Lit("PICO")),
        ("equity_del_freno", C::Lit("AUSENTE")), ("equity_del_freno_plano", C::Lit("PUENTE")),
        ("caida_capital_pct", C::Num), ("tope_capital_pct", C::Num), ("operator_lock", C::Bool),
        ("insolvent", C::Bool), ("dd_breach", C::Bool), ("lat_breach", C::Bool), ("latency_strikes", C::Dig),
        ("revision", C::Lit("N/D")), ("lecturas_posteriores", C::Lit("ninguna")),
    ]);
    assert_eq!((v[1].as_str(), v[9].as_str()), ("DRAWDOWN", "77.3172"));
    // Los nombres viejos que imputaban planos ajenos ya no estan.
    for viejo in ["balance_rest=", "balance_stream=", "drop_pct=", "threshold_pct="] {
        assert!(!linea.contains(viejo), "{viejo}");
    }
    // Los planos declarados en la linea son los que el codigo usa hoy.
    // C3 (ancla adaptada): V4 antepone la rama del libro derivado (la magnitud es la equity del
    // puente publicada en `capital_exchange`) y carga `exchange` antes que `capital`. Antes:
    // `let capital = …unified_capital…`, `let exchange = …capital_exchange…`, `capital.min(exchange)`.
    orden(include_str!("../crates/risk-engine/src/drawdown.rs"), &[
        "pub fn capital_del_freno(arena: &quantum_arena::GlobalArena) -> f64 {",
        ".dimensionable_derivado",
        "return if exchange.is_finite() { exchange } else { f64::NAN };",
        "let exchange = arena.capital_exchange.load(Relaxed);",
        "let capital = arena.unified_capital.load(Relaxed);",
        "capital.min(exchange)",
    ]);
}

/// Las once lineas mas el testigo ORDER_RESULT (T2), y ninguna otra.
#[test]
fn demo_tel_son_once_lineas() {
    let n: usize = [HOST, STREAM, NUCLEO, DEMO_TEL_MOD].iter().map(|s| texto(s).matches("\"DEMO_TEL event=").count()).sum();
    assert_eq!(n, 12);
    let eventos = ["ORDER_SEND", "FILL", "TRADE_FEE", "FUNDING", "UNREALIZED_PNL", "CAPITAL_UPDATE", "DECISION_MID",
        "TRADE_CLOSE", "AUDIT_SUMMARY", "SYMBOL_SPEC", "IMMUNE_TRIGGER", "ORDER_RESULT"];
    for ev in eventos {
        let k: usize = [HOST, STREAM, NUCLEO, DEMO_TEL_MOD].iter().map(|s| texto(s).matches(&format!("\"DEMO_TEL event={ev} ")).count()).sum();
        assert_eq!(k, 1, "{ev}");
    }
}

// ===========================================================================
// DEMO_TEL T3-5 (qra_t3_criterios.md:50-59): IMMUNE_TRIGGER llega íntegro por el logger
// acotado. Procesos hijos (este binario, `--exact`, T3_IMMUNE_HIJO) con el logger del host
// (`Rotacion::desde_entorno` + `init_telemetry_logger_rotativo(16_384, rot)`) y TG_LOG_TS
// activo e inactivo; el emisor es el de la rama real del disparo de cada pin
// (`t3_montaje/adaptador.rs`). El padre lee el destino drenado con el lector real.
// ===========================================================================

#[path = "t3_montaje/adaptador.rs"]
mod adaptador;

/// Operandos de un disparo (los que el host calcula antes de decidir).
pub struct CasoInmune {
    pub ts_ms: u64,
    pub motivo: &'static str,
    pub freno_capital: f64,
    pub peak_capital: f64,
    /// `None` = el puente no tenía equity del freno (AUSENTE).
    pub equity_del_freno: Option<f64>,
    pub caida_capital_pct: f64,
    pub tope_capital_pct: f64,
    pub operator_lock: bool,
    pub insolvent: bool,
    pub dd_breach: bool,
    pub lat_breach: bool,
    pub latency_strikes: u32,
}

const T3_IMMUNE_HIJO: &str = "T3_IMMUNE_HIJO";
const T3_IMMUNE_NOMBRE: &str = "t3_immune_trigger_atraviesa_logger_y_parser_sin_truncar";
/// Mayor capital representable dentro del dominio admitido (|x| < 1e12) con `{:.4}`.
const CAP_MAX: f64 = 999_999_999_999.9;
const PCT_MAX: f64 = 999_999.9999;

fn caso(motivo: &'static str) -> CasoInmune {
    CasoInmune {
        ts_ms: 1_763_000_000_000,
        motivo,
        freno_capital: 1655.81,
        peak_capital: 7299.88,
        equity_del_freno: None,
        caida_capital_pct: 77.3172,
        tope_capital_pct: 50.0,
        operator_lock: false,
        insolvent: false,
        dd_breach: false,
        lat_breach: false,
        latency_strikes: 0,
    }
}

/// Los disparos válidos: el fixture exacto del dictamen, equity finita y ausente, las
/// cuatro causas y los valores más largos del dominio admitido.
fn casos_validos() -> Vec<CasoInmune> {
    vec![
        CasoInmune { dd_breach: true, ..caso("DRAWDOWN") },
        CasoInmune { operator_lock: true, equity_del_freno: Some(1655.8071), ..caso("OPERATOR_LOCK") },
        CasoInmune { insolvent: true, freno_capital: -3.25, equity_del_freno: Some(0.0), ..caso("INSOLVENTE") },
        CasoInmune { lat_breach: true, latency_strikes: 3, equity_del_freno: Some(1655.8071), ..caso("LATENCIA") },
        CasoInmune {
            ts_ms: u64::MAX,
            motivo: "OPERATOR_LOCK",
            freno_capital: -CAP_MAX,
            peak_capital: CAP_MAX,
            equity_del_freno: Some(-CAP_MAX),
            caida_capital_pct: -PCT_MAX,
            tope_capital_pct: -PCT_MAX,
            operator_lock: true,
            insolvent: true,
            dd_breach: true,
            lat_breach: true,
            latency_strikes: u32::MAX,
        },
    ]
}

/// Un disparo con un capital fuera del dominio admitido: diagnóstico, nunca un valor inventado.
fn caso_fuera_de_dominio() -> CasoInmune {
    CasoInmune { dd_breach: true, freno_capital: 1e13, ..caso("DRAWDOWN") }
}

fn hijo_immune() {
    match telemetry_server::macros::Rotacion::desde_entorno() {
        Some(rot) => assert!(telemetry_server::macros::init_telemetry_logger_rotativo(16_384, rot)),
        None => panic!("el hijo necesita TG_LOG_FILE"),
    }
    assert!(telemetry_server::macros::LOG_QUEUE.get().is_some(), "la cola del logger no está activa");
    telemetry_server::telemetry_log!("T3_CENTINELA antes");
    for c in casos_validos() {
        adaptador::emitir_immune(&c);
    }
    adaptador::emitir_immune(&caso_fuera_de_dominio());
    telemetry_server::telemetry_log!("T3_CENTINELA despues");
    telemetry_server::macros::vaciar_telemetria();
}

fn correr_hijo_immune(ts: bool, extra: &[(&str, &str)]) -> String {
    let dir = std::env::temp_dir().join(format!("t3_immune_{}_{}", std::process::id(), ts));
    std::fs::create_dir_all(&dir).unwrap();
    let log = dir.join("demo_t3_immune.log");
    let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", T3_IMMUNE_NOMBRE, "--nocapture", "--test-threads=1"])
        .env(T3_IMMUNE_HIJO, "1")
        .env("TG_LOG_FILE", &log)
        .env("TG_LOG_TS", if ts { "1" } else { "0" });
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("lanza el hijo");
    assert!(out.status.success(), "el hijo falló: {}", String::from_utf8_lossy(&out.stderr));
    let texto = std::fs::read_to_string(&log).expect("destino del logger");
    let _ = std::fs::remove_dir_all(&dir);
    texto
}

/// Campo por campo, contra los operandos del disparo.
fn comprobar_campos(e: &telemetry_server::report_demo_run::EventoTel, c: &CasoInmune) {
    let v = |k: &str| e.campos.get(k).unwrap_or_else(|| panic!("falta {k}: {:?}", e.campos)).as_str();
    let num = |k: &str, esperado: f64| {
        let x: f64 = v(k).parse().unwrap_or_else(|_| panic!("{k}={} no es número", v(k)));
        assert!((x - esperado).abs() <= 0.5e-4 * esperado.abs().max(1.0), "{k}={} frente a {esperado}", v(k));
    };
    assert_eq!(v("ts"), c.ts_ms.to_string());
    assert_eq!(v("motivo"), c.motivo);
    assert_eq!(v("alcance"), "CUENTA");
    num("freno_capital", c.freno_capital);
    num("peak_capital", c.peak_capital);
    match c.equity_del_freno {
        Some(x) => num("equity_del_freno", x),
        None => assert_eq!(v("equity_del_freno"), "AUSENTE"),
    }
    num("caida_capital_pct", c.caida_capital_pct);
    num("tope_capital_pct", c.tope_capital_pct);
    for (k, b) in [("operator_lock", c.operator_lock), ("insolvent", c.insolvent), ("dd_breach", c.dd_breach), ("lat_breach", c.lat_breach)] {
        assert_eq!(v(k), b.to_string(), "{k}");
    }
    assert_eq!(v("latency_strikes"), c.latency_strikes.to_string());
    assert_eq!(v("revision"), "N/D");
    assert_eq!(v("lecturas_posteriores"), "ninguna");
    for (k, largo) in [
        ("freno_capital_plano", "drawdown.capital_del_freno(min(arena.unified_capital,arena.capital_exchange))"),
        ("peak_plano", "drawdown.registrar_pico(arena.pico_capital,freno_capital)"),
        ("equity_del_freno_plano", "puente.equity_del_freno"),
    ] {
        assert_eq!(adaptador::significado_plano(k, v(k)).as_deref(), Some(largo), "significado de {k}={}", v(k));
    }
}

#[test]
fn t3_immune_trigger_atraviesa_logger_y_parser_sin_truncar() {
    if std::env::var(T3_IMMUNE_HIJO).is_ok() {
        hijo_immune();
        return;
    }
    adaptador::ruta_del_disparo(HOST).unwrap_or_else(|e| panic!("[{}] {e}", adaptador::PIN));
    assert!(texto(HOST).contains("Some(rot) => telemetry_server::macros::init_telemetry_logger_rotativo(16_384, rot),"));
    use telemetry_server::macros::MAX_LOG_LEN;
    use telemetry_server::report_demo_run::{parse_demo_tel, Ambito};


    for ts in [true, false] {
        let texto_log = correr_hijo_immune(ts, &[]);
        let antes = texto_log.find("T3_CENTINELA antes").expect("centinela previa");
        let despues = texto_log.find("T3_CENTINELA despues").expect("centinela final (cola drenada)");
        assert!(antes < despues);
        if ts {
            assert!(texto_log.lines().all(|l| l.is_empty() || l.starts_with("[ts=")), "TG_LOG_TS=1 sin prefijo");
        } else {
            assert!(!texto_log.contains("[ts="), "TG_LOG_TS=0 con prefijo");
        }
        let l = parse_demo_tel(&texto_log, "segmento", &Ambito::new("binance_usdm", "A"));
        let eventos: Vec<_> = l.eventos.iter().filter(|e| e.tipo == "IMMUNE_TRIGGER").collect();
        let validos = casos_validos();
        assert_eq!(
            eventos.len(),
            validos.len(),
            "[{}] TG_LOG_TS={ts}: un evento íntegro por disparo válido; diagnósticos: {:?}",
            adaptador::PIN,
            l.diagnosticos
        );
        for (e, c) in eventos.iter().zip(&validos) {
            assert_eq!(e.ts_recepcion.is_some(), ts);
            comprobar_campos(e, c);
        }
        // El disparo fuera de dominio: un solo diagnóstico de cobertura, sin truncados.
        assert_eq!(l.diagnosticos.len(), 1, "[{}] {:?}", adaptador::PIN, l.diagnosticos);
        assert!(l.diagnosticos[0].mensaje.contains("freno_capital fuera del dominio"), "{}", l.diagnosticos[0].mensaje);
        assert!(!l.diagnosticos.iter().any(|d| d.mensaje.contains("truncada")), "{:?}", l.diagnosticos);
        assert_eq!(texto_log.matches("event=IMMUNE_TRIGGER").count(), validos.len() + 1, "conteo exacto de líneas");

        // El comparador muerde: suprimir el campo terminal o mutar el formato falla.
        let una = texto_log.lines().find(|x| x.contains("event=IMMUNE_TRIGGER")).unwrap();
        for mutante in [
            una.replace(" lecturas_posteriores=ninguna", ""),
            una.replace("latency_strikes=", "latency_strikes:"),
            una[..una.len() - 3].to_string(),
        ] {
            let m = parse_demo_tel(&mutante, "mutante", &Ambito::new("binance_usdm", "A"));
            assert!(m.eventos.is_empty() && m.diagnosticos.len() == 1, "un mutante pasó como evento: {mutante}");
        }
    }
    // Presupuesto de bytes: prefijo de mayor longitud representable (`tg_epoch_ms` es u128)
    // más la línea con los máximos del dominio admitido.
    let prefijo_max = format!("[ts={}] ", u128::MAX).len();
    for c in casos_validos() {
        let n = adaptador::linea_immune(&c).len();
        assert!(
            prefijo_max + n <= MAX_LOG_LEN,
            "[{}] presupuesto: prefijo {prefijo_max} + línea {n} > {MAX_LOG_LEN} ({})",
            adaptador::PIN,
            c.motivo
        );
    }

}

/// Coste y demora del disparo bajo captura equivalente (TG_LOG_FILE, cola productiva) en
/// cualquiera de los pines: `cargo test ... t3_bench_disparo_immune -- --ignored --exact --nocapture`.
#[test]
#[ignore]
fn t3_bench_disparo_immune() {
    if let Ok(n) = std::env::var("T3_BENCH_HIJO") {
        let n: usize = n.parse().unwrap();
        match telemetry_server::macros::Rotacion::desde_entorno() {
            Some(rot) => assert!(telemetry_server::macros::init_telemetry_logger_rotativo(16_384, rot)),
            None => panic!("TG_LOG_FILE"),
        }
        let c = CasoInmune { dd_breach: true, equity_del_freno: Some(1655.8071), ..caso("DRAWDOWN") };
        let mut ns = Vec::with_capacity(n);
        for i in 0..n {
            let c = CasoInmune { ts_ms: 1_763_000_000_000 + i as u64, ..c };
            let t0 = std::time::Instant::now();
            adaptador::emitir_immune(&c);
            ns.push(t0.elapsed().as_nanos() as u64);
            // Un disparo cada 50 µs: más de los que el host puede producir (uno por latch).
            while t0.elapsed() < std::time::Duration::from_micros(50) {
                std::hint::spin_loop();
            }
        }
        telemetry_server::macros::vaciar_telemetria();
        ns.sort_unstable();
        let media = ns.iter().sum::<u64>() as f64 / n as f64;
        println!(
            "T3_BENCH_IMMUNE pin={} n={n} p50_ns={} p99_ns={} max_ns={} media_ns={media:.1}",
            adaptador::PIN,
            ns[n / 2],
            ns[n * 99 / 100],
            ns[n - 1]
        );
        return;
    }
    let n = 20_000usize;
    for ts in ["0", "1"] {
        let dir = std::env::temp_dir().join(format!("t3_bench_{}_{ts}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("bench.log");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "t3_bench_disparo_immune", "--ignored", "--nocapture", "--test-threads=1"])
            .env("T3_BENCH_HIJO", n.to_string())
            .env("TG_LOG_FILE", &log)
            .env("TG_LOG_TS", ts)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let texto = std::fs::read_to_string(&log).unwrap();
        let escritas = texto.matches("event=IMMUNE_TRIGGER").count();
        let integras = parse_demo_tel_cuenta(&texto);
        let linea = String::from_utf8_lossy(&out.stdout).lines().find_map(|l| l.find("T3_BENCH_IMMUNE").map(|i| l[i..].to_string())).unwrap_or_default();
        println!("{linea} TG_LOG_TS={ts} escritas={escritas} integras={integras} perdidas={}", n - escritas.min(n));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn parse_demo_tel_cuenta(texto: &str) -> usize {
    telemetry_server::report_demo_run::parse_demo_tel(texto, "bench", &telemetry_server::report_demo_run::Ambito::new("b", "a"))
        .eventos
        .iter()
        .filter(|e| e.tipo == "IMMUNE_TRIGGER")
        .count()
}
