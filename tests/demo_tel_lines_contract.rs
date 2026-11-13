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
        "match entry_result {",
        "Err(e) if e.starts_with(\"AMBIGUOUS\")\n                                    || e.starts_with(\"MAKER_CHASE_UNVERIFIED\") =>",
        "Err(e) => {",
        "Ok(()) => {",
    ]);
    // El tramo del testigo no decide nada: ni rollback, ni reserva, ni reintento.
    let t = tramo(HOST, "let resultado_envio = match &entry_result {", "match entry_result {");
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
        "emitir(linea_capital_update(ts, usdt, revision));",
        "self.unified_capital.store(usdt.to_bits(), Ordering::Relaxed);",
        "fn on_equity(&self, usdt: f64) {",
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
    assert_eq!((v[1].as_str(), v[5].as_str()), ("100.0000", "200"));
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
    // Tras el aviso del disparo y antes de armar el kill-switch; el freno y el pico se leen
    // tal como estan (T7): `cap` (capital del freno), `peak_capital` y `cap_exchange`, todos
    // calculados ANTES de la decision de este ciclo.
    orden(HOST, &[
        "let operator_lock = operator_lock_exists();",
        "let cap_exchange = puente_imm.equity_del_freno().unwrap_or(f64::NAN);",
        "let cap = risk_engine::drawdown::capital_del_freno(&arena_imm);",
        "risk_engine::drawdown::registrar_pico(&arena_imm.pico_capital, cap);",
        "let freno = risk_engine::drawdown::freno_duro(&arena_imm, peak_capital, cap);",
        "let insolvent = cap <= 0.0;",
        "let dd_breach = freno.echado();",
        "let lat_breach = latency_strikes >= 3;",
        "if operator_lock || insolvent || dd_breach || lat_breach {",
        "🚨 [SISTEMA INMUNE] ACTIVADO",
        "DEMO_TEL event=IMMUNE_TRIGGER",
        "arena_imm.kill_switch_active.store(true, Ordering::Relaxed);",
        "executor.trigger_kill_switch();",
    ]);
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
    // Sin relecturas del arena, sin busqueda de simbolo y sin E/S entre la decision y el kill-switch.
    let t = tramo(HOST, "🚨 [SISTEMA INMUNE] ACTIVADO", "arena_imm.kill_switch_active.store(true, Ordering::Relaxed);");
    for prohibido in [".load(", "arena_imm.", "symbol_registry", "try_symbol", "position(", ".await", "std::fs", "puente_imm."] {
        assert!(!t.contains(prohibido), "IMMUNE_TRIGGER: trabajo nuevo antes del kill-switch: {prohibido}");
    }
    assert!(t.contains("let equity_del_freno_imm = if cap_exchange.is_finite() {"));
    assert!(t.contains("\"AUSENTE\".to_string()"), "la equity del freno ausente se declara, no se sustituye");
    let linea = demo_tel!(
        HOST,
        "ts_imm, motivo_imm, cap, peak_capital, equity_del_freno_imm, freno.caida_capital * 100.0, freno.tope_capital * 100.0, operator_lock, insolvent, dd_breach, lat_breach, latency_strikes",
        "DEMO_TEL event=IMMUNE_TRIGGER ts={} motivo={} alcance=CUENTA freno_capital={:.4} freno_capital_plano=drawdown.capital_del_freno(min(arena.unified_capital,arena.capital_exchange)) peak_capital={:.4} peak_plano=drawdown.registrar_pico(arena.pico_capital,freno_capital) equity_del_freno={} equity_del_freno_plano=puente.equity_del_freno caida_capital_pct={:.4} tope_capital_pct={:.4} operator_lock={} insolvent={} dd_breach={} lat_breach={} latency_strikes={} revision=N/D lecturas_posteriores=ninguna",
        1_763_000_000_000u64, "DRAWDOWN", 1655.81f64, 7299.88f64, "AUSENTE", 77.3172f64, 50.0f64, false, false, true, false, 0u32
    );
    let v = casa(&linea, "IMMUNE_TRIGGER", &[
        ("ts", C::Dig), ("motivo", C::Word), ("alcance", C::Lit("CUENTA")), ("freno_capital", C::Num),
        ("freno_capital_plano", C::Lit("drawdown.capital_del_freno(min(arena.unified_capital,arena.capital_exchange))")),
        ("peak_capital", C::Num),
        ("peak_plano", C::Lit("drawdown.registrar_pico(arena.pico_capital,freno_capital)")),
        ("equity_del_freno", C::Lit("AUSENTE")), ("equity_del_freno_plano", C::Lit("puente.equity_del_freno")),
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
    orden(include_str!("../crates/risk-engine/src/drawdown.rs"), &[
        "pub fn capital_del_freno(arena: &quantum_arena::GlobalArena) -> f64 {",
        "let capital = arena.unified_capital.load(Relaxed);",
        "let exchange = arena.capital_exchange.load(Relaxed);",
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
