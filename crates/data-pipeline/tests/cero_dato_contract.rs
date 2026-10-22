//! Contrato F1-091 (Sol): **un cero que no es dato se descarta, no se inventa**.
//!
//! # QUÉ
//! En el camino de dinero —saldo de la wallet, tickers del libro, velas
//! históricas y estadísticas de símbolo— un valor que no se puede leer es
//! DESCONOCIDO. Este contrato ejecuta las funciones REALES de `data-pipeline`
//! sobre payloads reales de Binance (nada de mocks) y verifica que la muestra
//! corrupta se descarta en lugar de convertirse en un cero con significado.
//!
//! # POR QUÉ
//! `unwrap_or(0.0)` sobre un `parse` fallido transforma «no pude leer» en una
//! afirmación falsa que el sistema usa para dimensionar y rankear:
//!
//! - **Saldo**: `get_real_wallet_balance()` devolvía `Ok(0.0)` con un `balance`
//!   ilegible. El operador leía «la wallet tiene 0 USDT» y el reintento
//!   fail-closed de `god_engine.rs` (que itera cada 5 s hasta obtener `Ok`) no
//!   se disparaba nunca: el arranque seguía con capital fabricado.
//! - **Libro**: un `bidQty` ilegible dejaba el tick con un lado en 0 —«libro
//!   vacío» en vez de «tamaño desconocido»— y ese cero alimentaba spread,
//!   profundidad y liquidez.
//! - **Velas**: un `volume` ilegible pasaba la guarda `>= 0.0` como vela de
//!   volumen CERO, es decir, como si el mercado no hubiera negociado. Además
//!   `arr[0]`/`arr[6]` indexaban directo: una fila corta PANICABA en runtime.
//! - **Rankers**: `asset_selector` y `dynamic_ranker` tenían cada uno su
//!   `unwrap_or(0.0)`; ahora comparten `cero_dato::estadistica_legible`.
//!
//! `parse::<f64>()` **acepta** `"NaN"`, `"inf"` e `"infinity"`: sin la guarda de
//! finitud un cero silencioso o un NaN suelto se cuela igual. Por eso el
//! contrato prueba esos literales explícitamente.
//!
//! # DÓNDE
//! `crates/data-pipeline/src/cero_dato.rs` (regla única) y sus consumidores:
//! `api_client.rs`, `spot_feed.rs`, `historical.rs`, `asset_selector.rs`,
//! `dynamic_ranker.rs`.
//!
//! # QUIÉN
//! Sol — átomo F1-091, 2026-10-22.

use std::io::{Read, Write};
use std::net::TcpListener;

use data_pipeline::cero_dato::{cantidad, estadistica_legible, monto, monto_de_json};
use data_pipeline::historical::HistoricalLoader;
use data_pipeline::spot_feed::{parse_rest_book_tickers, parse_ws_book_ticker};

// ---------------------------------------------------------------------------
// 1. La regla, sobre los literales que un `unwrap_or(0.0)` habría tragado
// ---------------------------------------------------------------------------

#[test]
fn f1_091_monto_rechaza_lo_que_no_es_un_dato() {
    assert_eq!(monto("13.5"), Some(13.5));
    assert_eq!(monto("0"), Some(0.0), "el cero ES un dato cuando se lee");
    // `parse::<f64>()` tiene éxito con estos tres y devuelve NaN/±inf:
    assert_eq!(monto("NaN"), None);
    assert_eq!(monto("inf"), None);
    assert_eq!(monto("-inf"), None);
    assert_eq!(monto("infinity"), None);
    assert_eq!(monto(""), None);
    assert_eq!(monto("no-un-numero"), None);
    assert_eq!(monto("null"), None);
    assert_eq!(
        monto("-5"),
        None,
        "un saldo negativo no es un monto legible"
    );
}

#[test]
fn f1_091_cantidad_exige_estrictamente_positiva() {
    assert_eq!(cantidad("82100.50"), Some(82100.50));
    assert_eq!(cantidad("0"), None, "un nivel en 0 no es un nivel vivo");
    assert_eq!(cantidad("-1.0"), None);
    assert_eq!(cantidad("NaN"), None);
}

#[test]
fn f1_091_monto_de_json_acepta_string_y_numero() {
    assert_eq!(
        monto_de_json(&serde_json::json!("13.5")),
        Some(13.5),
        "Binance manda el saldo como string"
    );
    assert_eq!(monto_de_json(&serde_json::json!(13.5)), Some(13.5));
    assert_eq!(monto_de_json(&serde_json::json!("NaN")), None);
    assert_eq!(monto_de_json(&serde_json::json!(null)), None);
}

#[test]
fn f1_091_estadistica_de_simbolo_es_fuente_unica_de_los_rankers() {
    assert_eq!(
        estadistica_legible("1000000.0", "0.1"),
        Some((1_000_000.0, 0.1))
    );
    assert_eq!(
        estadistica_legible("1000000.0", "-3.5"),
        Some((1_000_000.0, 3.5)),
        "la variación se usa en valor absoluto"
    );
    assert_eq!(
        estadistica_legible("0.0", "0.1"),
        None,
        "volumen 0 no es estadística: el símbolo se excluye, no compite con cifra inventada"
    );
    assert_eq!(estadistica_legible("NaN", "0.1"), None);
    assert_eq!(estadistica_legible("1000000.0", "NaN"), None);
    assert_eq!(estadistica_legible("", "0.1"), None);
}

// ---------------------------------------------------------------------------
// 2. Libro de órdenes: un frame ilegible se descarta COMPLETO
// ---------------------------------------------------------------------------

#[test]
fn f1_091_frame_valido_se_lee_completo() {
    let msg = br#"{"stream":"solusdt@bookTicker","data":{"s":"SOLUSDT","b":"82100.50","B":"5.123","a":"82100.60","A":"3.456","T":1710000000000}}"#;
    let t = parse_ws_book_ticker(std::str::from_utf8(msg).expect("utf8"));
    assert_eq!(
        t,
        Some(("SOLUSDT".into(), 82100.50, 82100.60, 5.123, 3.456))
    );
}

#[test]
fn f1_091_cantidad_ilegible_descarta_el_frame_en_vez_de_fabricar_un_cero() {
    // Antes: `Some(("SOLUSDT", .., 0.0, 3.456))` — lado del libro en cero.
    for (etiqueta, qty) in [
        ("vacia", "\"\""),
        ("null-literal", "\"null\""),
        ("texto", "\"no-es-numero\""),
        ("NaN", "\"NaN\""),
        ("inf", "\"inf\""),
        ("cero", "\"0\""),
        ("negativa", "\"-2.0\""),
    ] {
        let msg = format!(
            r#"{{"stream":"solusdt@bookTicker","data":{{"s":"SOLUSDT","b":"82100.50","B":{qty},"a":"82100.60","A":"3.456"}}}}"#
        );
        assert_eq!(
            parse_ws_book_ticker(&msg),
            None,
            "bidQty {etiqueta} ({qty}) debe descartar el frame, no dejar un lado en 0"
        );
    }
}

#[test]
fn f1_091_precio_ilegible_sigue_descartando_el_frame() {
    // Regresión preservada: los precios ya se validaban; el contrato lo fija.
    let msg = br#"{"s":"SOLUSDT","b":"NaN","B":"1.0","a":"82100.60","A":"3.456"}"#;
    assert_eq!(
        parse_ws_book_ticker(std::str::from_utf8(msg).expect("utf8")),
        None
    );
    let msg = br#"{"s":"SOLUSDT","b":"-10.0","B":"1.0","a":"82100.60","A":"3.456"}"#;
    assert_eq!(
        parse_ws_book_ticker(std::str::from_utf8(msg).expect("utf8")),
        None
    );
}

#[test]
fn f1_091_rest_descarta_solo_la_entrada_corrupta() {
    let payload = concat!(
        r#"[{"symbol":"BTCUSDT","bidPrice":"82100.50","bidQty":"5.123","askPrice":"82100.60","askQty":"3.456"},"#,
        r#"{"symbol":"ETHUSDT","bidPrice":"2450.10","bidQty":"25.0","askPrice":"2450.20","askQty":""},"#,
        r#"{"symbol":"SOLUSDT","bidPrice":"82100.50","bidQty":"","askPrice":"82100.60","askQty":"3.456"}]"#
    );
    let entradas = parse_rest_book_tickers(payload);
    assert_eq!(
        entradas.len(),
        1,
        "sólo BTCUSDT es legible; ETH y SOL se descartan"
    );
    assert_eq!(entradas[0].0, "BTCUSDT");
}

// ---------------------------------------------------------------------------
// 3. Velas históricas: E2E contra un servidor HTTP real
// ---------------------------------------------------------------------------

/// Servidor HTTP/1.1 mínimo sobre un socket real que responde `cuerpo` a la
/// primera petición. Sin mock: `HistoricalLoader` habla TCP de verdad.
fn servidor_con_cuerpo(cuerpo: String) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind local");
    let addr = listener.local_addr().expect("addr local");
    let handle = std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 2048];
            let _ = stream.read(&mut buf);
            let respuesta = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                cuerpo.len(),
                cuerpo
            );
            let _ = stream.write_all(respuesta.as_bytes());
            let _ = stream.flush();
        }
    });
    (format!("http://{addr}/fapi/v1/klines"), handle)
}

#[tokio::test]
async fn f1_091_vela_con_volumen_ilegible_se_descarta_en_vez_de_ser_vela_de_volumen_cero() {
    // Fila 1: vela sana. Fila 2: volumen ilegible ("NaN"). Fila 3: fila CORTA
    // (5 elementos) — antes indexaba `arr[6]` y PANICABA en runtime.
    let cuerpo = concat!(
        r#"[[1710000000000,"100.0","105.0","99.5","104.0","1234.5",1710003599999,"x",1,"y","z"],"#,
        r#"[1710003600000,"104.0","106.0","103.0","105.0","NaN",1710007199999,"x",1,"y","z"],"#,
        r#"[1710007200000,"105.0","107.0","104.0","106.0","999.0"]]"#
    )
    .to_string();
    let (base, servidor) = servidor_con_cuerpo(cuerpo);

    let loader = HistoricalLoader::con_base_url(&base);
    let klines = loader
        .fetch_klines("SOLUSDT", "1m", None, None, 10)
        .await
        .expect("la petición al servidor local debe completarse");

    assert_eq!(
        klines.len(),
        1,
        "la vela de volumen ilegible y la fila corta deben descartarse; \
         antes la primera entraba con volume = 0.0 y la segunda PANICABA"
    );
    let k = &klines[0];
    assert_eq!(k.open_time, 1710000000000);
    assert_eq!(k.open, 100.0);
    assert_eq!(k.high, 105.0);
    assert_eq!(k.low, 99.5);
    assert_eq!(k.close, 104.0);
    assert_eq!(k.volume, 1234.5, "el volumen real se conserva tal cual");
    assert_eq!(k.close_time, 1710003599999);

    let _ = servidor.join();
}

#[tokio::test]
async fn f1_091_precio_ilegible_tambien_descarta_la_vela() {
    let cuerpo = concat!(
        r#"[[1710000000000,"NaN","105.0","99.5","104.0","1234.5",1710003599999,"x",1,"y","z"],"#,
        r#"[1710003600000,"104.0","106.0","103.0","105.0","500.0",1710007199999,"x",1,"y","z"]]"#
    )
    .to_string();
    let (base, servidor) = servidor_con_cuerpo(cuerpo);
    let loader = HistoricalLoader::con_base_url(&base);
    let klines = loader
        .fetch_klines("SOLUSDT", "1m", None, None, 10)
        .await
        .expect("petición local");
    assert_eq!(klines.len(), 1);
    assert_eq!(klines[0].open, 104.0);
    let _ = servidor.join();
}

#[tokio::test]
async fn f1_091_json_no_array_no_produce_velas_fantasma() {
    // Un cuerpo que no es un array de velas es un ERROR de la API, no una
    // respuesta vacía: se falla en cerrojo con el motivo en vez de devolver un
    // histórico vacío que el consumidor podría leer como «no hay datos».
    let (base, servidor) = servidor_con_cuerpo(r#"{"error":"klines no disponibles"}"#.to_string());
    let loader = HistoricalLoader::con_base_url(&base);
    let res = loader.fetch_klines("SOLUSDT", "1m", None, None, 10).await;
    let err = res.expect_err("un cuerpo no-array debe fallar en cerrojo, no devolver velas");
    assert!(
        err.contains("decoding") && err.contains("sequence"),
        "el error debe decir QUE el cuerpo no es una secuencia de velas: {err}"
    );
    let _ = servidor.join();
}
