//! I-1b (Línea I) — contrato de las rutas del feed de mercado.
//!
//! En producción, la URL heredada del host (`wss://{host}/stream?streams=…`)
//! sólo recibe la categoría pública: velas y liquidaciones callan sin error
//! (sondas del 2026-10-24, `docs/audit/linea_i/SONDAS_WS_2026-10-24.md`).
//! Este contrato fija que el host arma sus URLs con `rutas_feed` y abre un
//! lector por categoría, y que cada stream del host cae en una sola URL de su
//! categoría.

use data_ingest::rutas_feed::{categoria, streams_del_host, urls_feed, Categoria};

const HOST: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../src/bin/god_engine.rs"));

fn streams_de(url: &str) -> Vec<&str> {
    url.split_once("?streams=").map(|(_, s)| s.split('/').collect()).unwrap_or_default()
}

#[test]
fn i1b_el_host_arma_sus_urls_con_rutas_feed() {
    assert!(
        HOST.contains("data_ingest::rutas_feed::urls_feed("),
        "el host debe armar las URLs del feed con data_ingest::rutas_feed::urls_feed"
    );
    assert!(
        !HOST.contains("\"wss://{}/stream?streams={}\""),
        "queda la URL heredada sin categoría: en producción no recibe velas ni liquidaciones"
    );
    assert!(
        !HOST.contains("streams.push_str(\"@kline_1m\")"),
        "el host vuelve a armar a mano la lista de streams en vez de usar rutas_feed"
    );
}

#[test]
fn i1b_el_host_abre_un_lector_por_categoria_y_solo_el_publico_reinicia_el_libro() {
    assert_eq!(
        HOST.matches("bucle_lector_ws(").count(),
        3,
        "una definición y dos llamadas (lector público y lector de mercado)"
    );
    assert_eq!(HOST.matches("anuncia_reconexion: true").count(), 1);
    assert_eq!(HOST.matches("anuncia_reconexion: false").count(), 1);
    assert_eq!(HOST.matches("marca_stall: true").count(), 1);
    assert_eq!(HOST.matches("marca_stall: false").count(), 1);
    assert!(HOST.contains("\"ws-reader-mercado\""), "falta el hilo del lector de mercado");
}

#[test]
fn i1b_solo_los_eventos_publicos_escriben_la_salud_del_feed() {
    // Con dos lectores, una vela del lector de mercado no puede borrar la
    // marca de stall del público ni refrescar su latencia.
    let compacto: String = HOST
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<String>()
        .split_whitespace()
        .collect();
    assert!(
        compacto.contains(
            "ifevent_time>0&&(is_trade||is_depth){engine_real.arena.last_ws_latency_ms.store(latency_ms.max(0)asu64,Ordering::Relaxed);quantum_arena::feed_health::clear();}"
        ),
        "la salud del feed debe escribirse sólo con eventos públicos"
    );
}

#[test]
fn i1b_cada_stream_del_host_cae_en_una_sola_url_de_su_categoria() {
    let universos: [&[&str]; 4] = [
        &["BTCUSDT"],
        &["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        &["1000SHIBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT"],
        &["A1", "B2", "C3", "D4", "E5", "F6", "G7", "H8", "I9", "J10"],
    ];
    for host in ["fstream.binance.com", "stream.binancefuture.com", "fstream.binance.com:443"] {
        for u in universos {
            let simbolos: Vec<String> = u.iter().map(|s| s.to_string()).collect();
            let streams = streams_del_host(&simbolos).unwrap();
            let urls = urls_feed(host, &simbolos).unwrap();
            assert!(urls.publica.starts_with(&format!("wss://{host}/public/stream?streams=")));
            assert!(urls.mercado.starts_with(&format!("wss://{host}/market/stream?streams=")));
            let pub_ = streams_de(&urls.publica);
            let mer = streams_de(&urls.mercado);
            assert_eq!(pub_.len() + mer.len(), streams.len(), "{host} {u:?}");
            for s in &streams {
                let en_pub = pub_.contains(&s.as_str());
                let en_mer = mer.contains(&s.as_str());
                assert!(en_pub ^ en_mer, "{s} debe estar en exactamente una URL");
                match categoria(s) {
                    Some(Categoria::Publica) => assert!(en_pub, "{s} es pública"),
                    Some(Categoria::Mercado) => assert!(en_mer, "{s} es de mercado"),
                    None => panic!("{s} sin categoría"),
                }
            }
            // Las velas de cada símbolo y las liquidaciones viajan por /market.
            for s in &simbolos {
                assert!(mer.contains(&format!("{}@kline_1m", s.to_ascii_lowercase()).as_str()));
            }
            assert!(mer.contains(&"!forceOrder@arr"));
        }
    }
}
