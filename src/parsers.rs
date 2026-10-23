use simd_json::prelude::ValueAsScalar;
use simd_json::prelude::ValueObjectAccess;
use simd_json::prelude::*;

/// P-6 — Parser de TODOS los niveles del stream @depth5 (hasta 5 por lado).
/// Es el ÚNICO parser de libro del host (RET-45: el L1 `parse_binance_depth`
/// sólo vivía en tests; su mejor nivel es el nivel 0 de éste).
/// Rellena arrays del caller (zero-alloc) y devuelve
/// (event_time, symbol, last_update_id, n_levels_válidos).
/// Prerrequisito del SpoofingDetector: los muros viven más allá del best.
pub fn parse_binance_depth5_levels<'a>(
    json_str: &'a mut str,
    bids_out: &mut [(f64, f64); 5],
    asks_out: &mut [(f64, f64); 5],
) -> Option<(i64, &'a str, i64, usize)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };
    let e = data.get("E")?.as_i64()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
    let last_update_id = data.get("u")?.as_i64()?;

    let fill = |arr: &simd_json::BorrowedValue<'a>, out: &mut [(f64, f64); 5]| -> usize {
        let mut n = 0usize;
        if let Some(levels) = arr.as_array() {
            for lv in levels.iter().take(5) {
                let Some(pair) = lv.as_array() else { continue };
                let (Some(p), Some(q)) = (pair.get(0).and_then(|v| v.as_str()), pair.get(1).and_then(|v| v.as_str()))
                else {
                    continue;
                };
                let (Ok(p), Ok(q)) = (p.parse::<f64>(), q.parse::<f64>()) else {
                    continue;
                };
                if p > 0.0 && q >= 0.0 && p.is_finite() && q.is_finite() {
                    out[n] = (p, q);
                    n += 1;
                }
            }
        }
        n
    };

    let bids = data.get("b")?;
    let asks = data.get("a")?;
    let nb = fill(bids, bids_out);
    let na = fill(asks, asks_out);
    if nb == 0 || na == 0 {
        return None;
    }
    // FIX #1417: libro cruzado al mejor nivel ⇒ corrupto.
    if bids_out[0].0 >= asks_out[0].0 {
        return None;
    }
    Some((e, s, last_update_id, nb.min(na)))
}

/// Parses a Binance Trade JSON string.
/// Returns (Event_time, trade_time, price, qty, is_buyer_maker, symbol)
pub fn parse_binance_trade<'a>(
    json_str: &'a mut str,
) -> Option<(i64, i64, f64, f64, bool, &'a str)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

    let e = data.get("E")?.as_i64()?;
    let t = data.get("T")?.as_i64()?;
    let p = data.get("p")?.as_str()?.parse::<f64>().ok()?;
    let q = data.get("q")?.as_str()?.parse::<f64>().ok()?;
    let m = data.get("m")?.as_bool()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };

    // FIX #1472: Data Integrity Validation & Finitude
    if p <= 0.0 || q < 0.0 || !p.is_finite() || !q.is_finite() {
        return None;
    }

    Some((e, t, p, q, m, s))
}

pub type KlineData<'a> = (i64, &'a str, f64, f64, f64, f64, f64, bool);

/// Parses a Binance Kline/Candlestick JSON string.
/// Returns (Event_time, symbol, open, high, low, close, volume, is_kline_closed)
pub fn parse_binance_kline<'a>(json_str: &'a mut str) -> Option<KlineData<'a>> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

    // Check if it's a kline event
    if data.get("e")?.as_str()? != "kline" {
        return None;
    }

    let e = data.get("E")?.as_i64()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };

    let k = data.get("k")?;
    let open = k.get("o")?.as_str()?.parse::<f64>().ok()?;
    let high = k.get("h")?.as_str()?.parse::<f64>().ok()?;
    let low = k.get("l")?.as_str()?.parse::<f64>().ok()?;
    let close = k.get("c")?.as_str()?.parse::<f64>().ok()?;
    let volume = k.get("v")?.as_str()?.parse::<f64>().ok()?;
    let is_closed = k.get("x")?.as_bool()?;

    // FIX #1472: Data Integrity Validation & Finitude
    if open <= 0.0
        || high < low
        || low <= 0.0
        || close <= 0.0
        || volume < 0.0
        || !open.is_finite()
        || !high.is_finite()
        || !low.is_finite()
        || !close.is_finite()
        || !volume.is_finite()
    {
        return None; // Protect the engine from absurd or manipulated data
    }

    Some((e, s, open, high, low, close, volume, is_closed))
}

/// P-7 — Parser de latencia ultra-baja (0 ms) para streams de @bookTicker de Binance.
/// Devuelve (event_time, symbol, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, update_id)
/// Valida sanidad de precios, finitud y rechaza libros cruzados (bid >= ask).
pub fn parse_binance_book_ticker<'a>(
    json_str: &'a mut str,
) -> Option<(i64, &'a str, f64, f64, f64, f64, i64)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

    let u = data.get("u")?.as_i64()?;
    let s_temp = data.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
    let b = data.get("b")?.as_str()?.parse::<f64>().ok()?;
    let big_b = data.get("B")?.as_str()?.parse::<f64>().ok()?;
    let a = data.get("a")?.as_str()?.parse::<f64>().ok()?;
    let big_a = data.get("A")?.as_str()?.parse::<f64>().ok()?;
    let e = data
        .get("E")
        .and_then(|v| v.as_i64())
        .or_else(|| data.get("T").and_then(|v| v.as_i64()))
        .unwrap_or(0);

    // Validación de finitud, positividad y libro no cruzado
    if b <= 0.0
        || a <= 0.0
        || big_b < 0.0
        || big_a < 0.0
        || !b.is_finite()
        || !a.is_finite()
        || !big_b.is_finite()
        || !big_a.is_finite()
        || b >= a
    {
        return None;
    }

    Some((e, s, b, big_b, a, big_a, u))
}

/// P-8 — Parser zero-alloc para órdenes de liquidación forzada de Binance (!forceOrder@arr o stream individual).
/// Devuelve (event_time, symbol, is_buyer, price, qty, trade_time)
/// Permite capturar cascadas de liquidación institucional y micro-dislocaciones instantáneas.
pub fn parse_binance_force_order<'a>(
    json_str: &'a mut str,
) -> Option<(i64, &'a str, bool, f64, f64, i64)> {
    let bytes: &'a mut [u8] = unsafe { json_str.as_bytes_mut() };
    let parsed = simd_json::to_borrowed_value(bytes).ok()?;
    let data = if let Some(d) = parsed.get("data") {
        d
    } else {
        &parsed
    };

    let e_type = data.get("e").and_then(|v| v.as_str());
    if e_type != Some("forceOrder") {
        return None;
    }

    let e = data.get("E")?.as_i64()?;
    let o = data.get("o")?;

    let s_temp = o.get("s")?.as_str()?;
    let s: &'a str = unsafe { std::mem::transmute(s_temp) };
    let side = o.get("S")?.as_str()?;
    let is_buy = side.eq_ignore_ascii_case("BUY");

    let p = o
        .get("p")
        .and_then(|v| v.as_str())
        .and_then(|v| v.parse::<f64>().ok())
        .or_else(|| {
            o.get("ap")
                .and_then(|v| v.as_str())
                .and_then(|v| v.parse::<f64>().ok())
        })?;
    let q = o.get("q")?.as_str()?.parse::<f64>().ok()?;
    let t = o.get("T").and_then(|v| v.as_i64()).unwrap_or(e);

    if p <= 0.0 || q <= 0.0 || !p.is_finite() || !q.is_finite() {
        return None;
    }

    Some((e, s, is_buy, p, q, t))
}

/// Veredicto de la guardia de secuencia del libro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqVerdict {
    /// Mensaje nuevo: se procesa.
    Accept,
    /// `update_id` no posterior al último aceptado: rancio, duplicado o fuera
    /// de orden. Se descarta.
    Stale,
    /// Tras una racha de descartes se asume un reinicio de la secuencia del
    /// exchange y se acepta este mensaje como nuevo origen.
    Resync,
}

/// D-610 (DÉCIMA OLA) — GUARDIA DE SECUENCIA DEL LIBRO POR SÍMBOLO.
///
/// `parse_binance_depth` (el parser L1, retirado en RET-45) extraía el `u`
/// (final update id) de cada mensaje y el despacho del motor lo descartaba. Sin comprobar su monotonía el motor no
/// podía detectar tres fallos silenciosos: libro rancio tras una reconexión,
/// mensajes reordenados por la multiplexación —el precio «retrocede» y OBI,
/// OFI y microprecio se calculan sobre un estado viejo— y mensajes duplicados.
///
/// Una guardia estrictamente monótona tiene un modo de fallo propio: si el
/// exchange reiniciara su secuencia, descartaría ese símbolo para siempre. Por
/// eso, tras `RESYNC_AFTER` descartes consecutivos, acepta y resincroniza.
pub struct BookSequenceGuard {
    last: Vec<i64>,
    stale_streak: Vec<u32>,
    /// Mensajes descartados por no ser posteriores al último aceptado.
    pub dropped: u64,
    /// Resincronizaciones por racha de descartes.
    pub resyncs: u64,
}

impl BookSequenceGuard {
    /// Descartes consecutivos tras los que se asume un reinicio de la
    /// secuencia: 50 mensajes del stream `depth@100ms` son 5 segundos sin un
    /// solo `update_id` nuevo, mucho más de lo que dura un reordenamiento
    /// transitorio de la multiplexación.
    pub const RESYNC_AFTER: u32 = 50;

    pub fn new(symbols: usize) -> Self {
        Self {
            last: vec![i64::MIN; symbols],
            stale_streak: vec![0; symbols],
            dropped: 0,
            resyncs: 0,
        }
    }

    pub fn check(&mut self, symbol_id: usize, update_id: i64) -> SeqVerdict {
        if symbol_id >= self.last.len() {
            // Símbolo fuera del universo custodiado: no hay estado que comparar.
            return SeqVerdict::Accept;
        }
        if update_id > self.last[symbol_id] {
            self.last[symbol_id] = update_id;
            self.stale_streak[symbol_id] = 0;
            return SeqVerdict::Accept;
        }
        self.stale_streak[symbol_id] += 1;
        if self.stale_streak[symbol_id] >= Self::RESYNC_AFTER {
            self.last[symbol_id] = update_id;
            self.stale_streak[symbol_id] = 0;
            self.resyncs += 1;
            return SeqVerdict::Resync;
        }
        self.dropped += 1;
        SeqVerdict::Stale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P-6 — el parser multi-nivel extrae TODOS los niveles del @depth5 en
    /// orden, con el mejor nivel en [0], y rechaza libros cruzados.
    #[test]
    fn p6_parse_depth5_levels_extrae_los_cinco_niveles() {
        let mut msg = String::from(
            r#"{"e":"depthUpdate","E":1700000000123,"s":"NEARUSDT","U":1,"u":42,
                "b":[["3.10","100.0"],["3.09","500.0"],["3.08","50.0"],["3.07","10.0"],["3.06","90000.0"]],
                "a":[["3.11","80.0"],["3.12","700.0"],["3.13","20.0"],["3.14","5.0"],["3.15","60000.0"]]}"#,
        );
        let mut bids = [(0.0f64, 0.0f64); 5];
        let mut asks = [(0.0f64, 0.0f64); 5];
        let (e, sym, uid, n) =
            parse_binance_depth5_levels(&mut msg, &mut bids, &mut asks).expect("payload válido");
        assert_eq!(e, 1700000000123);
        assert_eq!(sym, "NEARUSDT");
        assert_eq!(uid, 42);
        assert_eq!(n, 5);
        // El mejor nivel es el [0].
        assert_eq!(bids[0], (3.10, 100.0));
        assert_eq!(asks[0], (3.11, 80.0));
        // El muro máximo del bid vive en el nivel 4 (90000 × 3.06): SOLO
        // visible con el parser multi-nivel — el motivo de P-6.
        let max_bid_wall = bids.iter().map(|&(p, q)| p * q).fold(0.0, f64::max);
        assert!((max_bid_wall - 3.06 * 90000.0).abs() < 1e-6);
        let max_ask_wall = asks.iter().map(|&(p, q)| p * q).fold(0.0, f64::max);
        assert!((max_ask_wall - 3.15 * 60000.0).abs() < 1e-6);
    }

    #[test]
    fn p6_parse_depth5_rechaza_libro_cruzado() {
        let mut msg = String::from(
            r#"{"e":"depthUpdate","E":1,"s":"XUSDT","U":1,"u":2,
                "b":[["5.0","1.0"],["4.9","1.0"]],
                "a":[["4.99","1.0"],["5.0","1.0"]]}"#,
        );
        let mut bids = [(0.0f64, 0.0f64); 5];
        let mut asks = [(0.0f64, 0.0f64); 5];
        assert!(parse_binance_depth5_levels(&mut msg, &mut bids, &mut asks).is_none());
    }

    #[test]
    fn d610_acepta_secuencia_creciente_y_descarta_rancios() {
        let mut g = BookSequenceGuard::new(2);
        assert_eq!(g.check(0, 100), SeqVerdict::Accept);
        assert_eq!(g.check(0, 101), SeqVerdict::Accept);
        assert_eq!(g.check(0, 101), SeqVerdict::Stale, "duplicado");
        assert_eq!(g.check(0, 99), SeqVerdict::Stale, "fuera de orden");
        assert_eq!(g.check(0, 102), SeqVerdict::Accept);
        assert_eq!(g.dropped, 2);
    }

    #[test]
    fn d610_cada_simbolo_tiene_su_propia_secuencia() {
        let mut g = BookSequenceGuard::new(2);
        assert_eq!(g.check(0, 500), SeqVerdict::Accept);
        assert_eq!(g.check(1, 10), SeqVerdict::Accept, "otro símbolo, otra secuencia");
        assert_eq!(g.check(1, 9), SeqVerdict::Stale);
    }

    #[test]
    fn d610_resincroniza_tras_un_reinicio_de_la_secuencia() {
        let mut g = BookSequenceGuard::new(1);
        assert_eq!(g.check(0, 1_000_000), SeqVerdict::Accept);
        for i in 0..(BookSequenceGuard::RESYNC_AFTER - 1) {
            assert_eq!(g.check(0, 10 + i as i64), SeqVerdict::Stale);
        }
        assert_eq!(g.check(0, 60), SeqVerdict::Resync, "no puede quedar ciego para siempre");
        assert_eq!(g.check(0, 61), SeqVerdict::Accept);
        assert_eq!(g.resyncs, 1);
    }

    #[test]
    fn d610_simbolo_fuera_del_universo_no_se_bloquea() {
        let mut g = BookSequenceGuard::new(1);
        assert_eq!(g.check(7, 1), SeqVerdict::Accept);
    }

    /// RET-45 — migración del contrato del parser L1 retirado al nivel 0 de
    /// depth5: un mensaje de UN nivel por lado da n = 1 y el mejor nivel
    /// intacto; un libro cruzado (bid ≥ ask) se rechaza.
    #[test]
    fn ret45_depth5_un_nivel_por_lado_y_rechazo_de_libro_cruzado() {
        let mut valid_depth = r#"{"e":"depthUpdate","E":1672531200000,"s":"BTCUSDT","u":12345,"b":[["60000.0","1.5"]],"a":[["60001.0","2.0"]]}"#.to_string();
        let mut bids = [(0.0f64, 0.0f64); 5];
        let mut asks = [(0.0f64, 0.0f64); 5];
        let (e, s, u, n) = parse_binance_depth5_levels(&mut valid_depth, &mut bids, &mut asks)
            .expect("payload de un nivel válido");
        assert_eq!(e, 1672531200000);
        assert_eq!(s, "BTCUSDT");
        assert_eq!(u, 12345);
        assert_eq!(n, 1);
        assert_eq!(bids[0], (60000.0, 1.5));
        assert_eq!(asks[0], (60001.0, 2.0));

        // Crossed book rejection (bid >= ask)
        let mut crossed_depth = r#"{"e":"depthUpdate","E":1672531200000,"s":"BTCUSDT","u":12345,"b":[["60005.0","1.5"]],"a":[["60001.0","2.0"]]}"#.to_string();
        assert!(parse_binance_depth5_levels(&mut crossed_depth, &mut bids, &mut asks).is_none());
    }

    #[test]
    fn test_parse_binance_trade_valid_and_negative_rejection() {
        let mut valid_trade = r#"{"e":"trade","E":1672531200000,"s":"BTCUSDT","t":100,"p":"60000.5","q":"0.1","T":1672531200000,"m":true}"#.to_string();
        let res = parse_binance_trade(&mut valid_trade);
        assert!(res.is_some());
        let (e, t, p, q, m, s) = res.unwrap();
        assert_eq!(e, 1672531200000);
        assert_eq!(t, 1672531200000);
        assert_eq!(p, 60000.5);
        assert_eq!(q, 0.1);
        assert!(m);
        assert_eq!(s, "BTCUSDT");

        // Negative price rejection
        let mut invalid_trade = r#"{"e":"trade","E":1672531200000,"s":"BTCUSDT","t":100,"p":"-10.0","q":"0.1","T":1672531200000,"m":true}"#.to_string();
        assert!(parse_binance_trade(&mut invalid_trade).is_none());
    }

    #[test]
    fn test_parse_binance_kline_valid_and_integrity() {
        let mut valid_kline = r#"{"e":"kline","E":1672531200000,"s":"BTCUSDT","k":{"o":"60000.0","h":"60100.0","l":"59900.0","c":"60050.0","v":"10.5","x":true}}"#.to_string();
        let res = parse_binance_kline(&mut valid_kline);
        assert!(res.is_some());
        let (e, s, open, high, low, close, vol, is_closed) = res.unwrap();
        assert_eq!(e, 1672531200000);
        assert_eq!(s, "BTCUSDT");
        assert_eq!(open, 60000.0);
        assert_eq!(high, 60100.0);
        assert_eq!(low, 59900.0);
        assert_eq!(close, 60050.0);
        assert_eq!(vol, 10.5);
        assert!(is_closed);

        // Invalid OHLC (high < low)
        let mut invalid_kline = r#"{"e":"kline","E":1672531200000,"s":"BTCUSDT","k":{"o":"60000.0","h":"59800.0","l":"59900.0","c":"60050.0","v":"10.5","x":true}}"#.to_string();
        assert!(parse_binance_kline(&mut invalid_kline).is_none());
    }

    #[test]
    fn test_parse_binance_book_ticker_valid_and_crossed_rejection() {
        let mut valid_ticker = r#"{"u":400900217,"s":"BNBUSDT","b":"25.3519","B":"31.21","a":"25.3652","A":"40.66","T":1573879169829,"E":1573879169830}"#.to_string();
        let res = parse_binance_book_ticker(&mut valid_ticker);
        assert!(res.is_some(), "BookTicker payload válido debe parsearse");
        let (e, s, b, big_b, a, big_a, u) = res.unwrap();
        assert_eq!(e, 1573879169830);
        assert_eq!(s, "BNBUSDT");
        assert_eq!(b, 25.3519);
        assert_eq!(big_b, 31.21);
        assert_eq!(a, 25.3652);
        assert_eq!(big_a, 40.66);
        assert_eq!(u, 400900217);

        // Libro cruzado (bid >= ask) debe ser rechazado
        let mut crossed_ticker = r#"{"u":400900218,"s":"BNBUSDT","b":"25.4000","B":"10.0","a":"25.3500","A":"5.0","E":1573879169831}"#.to_string();
        assert!(parse_binance_book_ticker(&mut crossed_ticker).is_none(), "Libro cruzado bid >= ask debe rechazarse");

        // Precio no finito o no positivo
        let mut zero_bid_ticker = r#"{"u":400900219,"s":"BNBUSDT","b":"0.0","B":"10.0","a":"25.3500","A":"5.0","E":1573879169832}"#.to_string();
        assert!(parse_binance_book_ticker(&mut zero_bid_ticker).is_none(), "Bid <= 0 debe rechazarse");
    }

    #[test]
    fn test_parse_binance_force_order_valid_and_negative_rejection() {
        let mut valid_force = r#"{"e":"forceOrder","E":1568014460893,"o":{"s":"BTCUSDT","S":"SELL","o":"LIMIT","f":"IOC","q":"0.014","p":"9910.5","ap":"9910.5","X":"FILLED","l":"0.014","z":"0.014","T":1568014460893}}"#.to_string();
        let res = parse_binance_force_order(&mut valid_force);
        assert!(res.is_some(), "forceOrder payload válido debe parsearse");
        let (e, s, is_buy, p, q, t) = res.unwrap();
        assert_eq!(e, 1568014460893);
        assert_eq!(s, "BTCUSDT");
        assert!(!is_buy, "S: SELL debe resultar en is_buy = false");
        assert_eq!(p, 9910.5);
        assert_eq!(q, 0.014);
        assert_eq!(t, 1568014460893);

        // Evento no forceOrder debe ser ignorado
        let mut trade_msg = r#"{"e":"trade","E":1568014460893,"o":{"s":"BTCUSDT","S":"SELL","p":"9910.5","q":"0.014"}}"#.to_string();
        assert!(parse_binance_force_order(&mut trade_msg).is_none(), "Evento distinto a forceOrder debe retornar None");

        // Cantidad negativa o cero debe ser rechazada
        let mut zero_qty = r#"{"e":"forceOrder","E":1568014460893,"o":{"s":"BTCUSDT","S":"BUY","p":"9910.5","q":"0.0"}}"#.to_string();
        assert!(parse_binance_force_order(&mut zero_qty).is_none(), "Cantidad <= 0 debe rechazarse");
    }
}
