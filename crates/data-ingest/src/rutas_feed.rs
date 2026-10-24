//! I-1b (Línea I) — RUTAS DEL FEED DE MERCADO POR CATEGORÍA.
//!
//! Binance USDⓈ-M separó los streams de mercado por categoría, cada una con su
//! ruta (`/public`, `/market`; changelog del 2026-03-05, rutas heredadas
//! retiradas el 2026-04-23). Una conexión sin categoría sólo recibe la
//! pública. Medido el 2026-10-24 con sondas de solo lectura
//! (`docs/audit/linea_i/SONDAS_WS_2026-10-24.md`): la URL heredada que armaba
//! el host recibió en producción 315 actualizaciones de libro y 1081 trades, y
//! cero velas y cero liquidaciones; `/market` entregó 122 velas y 5
//! liquidaciones en la misma ventana. En testnet la ruta heredada sí lo
//! entregaba todo, así que el mismo binario se comportaba distinto en demo y en
//! producción: el reloj de calibración de 1 minuto (D-705) y el feed de
//! liquidaciones sólo existían en demo.
//!
//! Este módulo es la fuente única de las URLs del feed. Reparte cada stream por
//! su categoría y arma una URL por categoría con la misma estructura en todos
//! los entornos; sólo cambia el host.

/// Categoría de un stream de mercado de Binance USDⓈ-M.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Categoria {
    /// Libro y trades individuales (`@depthN`, `@trade`, `@bookTicker`).
    Publica,
    /// Velas, trades agregados y liquidaciones (`@kline_*`, `@aggTrade`,
    /// `!forceOrder@arr`).
    Mercado,
}

impl Categoria {
    /// Segmento de ruta de la categoría.
    pub fn ruta(self) -> &'static str {
        match self {
            Categoria::Publica => "public",
            Categoria::Mercado => "market",
        }
    }
}

/// Límite conservador de streams por conexión. Binance documenta un máximo por
/// conexión combinada; el host usa como mucho dos por símbolo y categoría con
/// un universo de hasta 30, muy por debajo. Superarlo es un error de
/// configuración, no algo que recortar en silencio.
pub const MAX_STREAMS_POR_CONEXION: usize = 200;

/// Stream global de liquidaciones de todo el mercado.
pub const STREAM_LIQUIDACIONES: &str = "!forceOrder@arr";

/// Cadencias de actualización que Binance admite como sufijo de profundidad.
const CADENCIAS_PROFUNDIDAD: [&str; 3] = ["100ms", "250ms", "500ms"];

/// Clasifica un stream por su tipo. `None` si el tipo no está en la tabla
/// medida: en la ruta equivocada un stream calla sin dar error, así que el host
/// no debe suscribir uno cuya categoría no conoce.
pub fn categoria(stream: &str) -> Option<Categoria> {
    if stream == STREAM_LIQUIDACIONES {
        return Some(Categoria::Mercado);
    }
    let (simbolo, resto) = stream.split_once('@')?;
    if simbolo.is_empty() || !simbolo.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return None;
    }
    let (tipo, cadencia) = match resto.split_once('@') {
        Some((t, c)) => (t, Some(c)),
        None => (resto, None),
    };
    match tipo {
        "trade" | "bookTicker" if cadencia.is_none() => Some(Categoria::Publica),
        "aggTrade" if cadencia.is_none() => Some(Categoria::Mercado),
        t if es_profundidad(t) => match cadencia {
            None => Some(Categoria::Publica),
            Some(c) if CADENCIAS_PROFUNDIDAD.contains(&c) => Some(Categoria::Publica),
            Some(_) => None,
        },
        t if cadencia.is_none() && es_vela(t) => Some(Categoria::Mercado),
        _ => None,
    }
}

/// `depth`, `depth5`, `depth10`, `depth20`.
fn es_profundidad(tipo: &str) -> bool {
    match tipo.strip_prefix("depth") {
        Some("") => true,
        Some(niveles) => matches!(niveles, "5" | "10" | "20"),
        None => false,
    }
}

/// `kline_<intervalo>` con un intervalo de Binance (`1m`, `15m`, `1h`, `1d`…).
fn es_vela(tipo: &str) -> bool {
    let Some(intervalo) = tipo.strip_prefix("kline_") else {
        return false;
    };
    // Sólo ASCII: `split_at` sobre un carácter multibyte rompería la frontera.
    if !intervalo.is_ascii() {
        return false;
    }
    let (numero, unidad) = intervalo.split_at(intervalo.len().saturating_sub(1));
    !numero.is_empty()
        && numero.bytes().all(|b| b.is_ascii_digit())
        && matches!(unidad, "m" | "h" | "d" | "w" | "M")
}

/// Fallo al armar las URLs del feed. Ninguno se recorta en silencio: una URL
/// mal armada deja al host sin datos de una categoría y sin error visible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorRutas {
    /// El host no es `nombre` o `nombre:puerto`.
    HostInvalido(String),
    /// Universo vacío: no hay nada que suscribir.
    UniversoVacio,
    /// Símbolo con caracteres fuera de `[A-Za-z0-9]`.
    SimboloInvalido(String),
    /// Símbolo repetido en el universo.
    SimboloRepetido(String),
    /// Stream sin categoría conocida.
    SinCategoria(String),
    /// Demasiados streams para una conexión.
    DemasiadosStreams { categoria: Categoria, cuantos: usize },
}

impl std::fmt::Display for ErrorRutas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ErrorRutas::HostInvalido(h) => write!(f, "host WS inválido «{h}»"),
            ErrorRutas::UniversoVacio => write!(f, "universo vacío: no hay streams que suscribir"),
            ErrorRutas::SimboloInvalido(s) => write!(f, "símbolo inválido «{s}»"),
            ErrorRutas::SimboloRepetido(s) => write!(f, "símbolo repetido «{s}»"),
            ErrorRutas::SinCategoria(s) => write!(f, "stream sin categoría conocida «{s}»"),
            ErrorRutas::DemasiadosStreams { categoria, cuantos } => write!(
                f,
                "{cuantos} streams en la categoría {} (máximo {MAX_STREAMS_POR_CONEXION})",
                categoria.ruta()
            ),
        }
    }
}

impl std::error::Error for ErrorRutas {}

/// Una URL por categoría.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrlsFeed {
    /// `wss://{host}/public/stream?streams=…`
    pub publica: String,
    /// `wss://{host}/market/stream?streams=…`
    pub mercado: String,
}

/// Streams que suscribe el host, en orden: por símbolo `@trade`, `@depth5` y
/// `@kline_1m` (en minúsculas), y al final las liquidaciones de todo el
/// mercado.
pub fn streams_del_host(simbolos: &[String]) -> Result<Vec<String>, ErrorRutas> {
    if simbolos.is_empty() {
        return Err(ErrorRutas::UniversoVacio);
    }
    let mut vistos: Vec<String> = Vec::with_capacity(simbolos.len());
    let mut streams = Vec::with_capacity(simbolos.len() * 3 + 1);
    for s in simbolos {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return Err(ErrorRutas::SimboloInvalido(s.clone()));
        }
        let bajo = s.to_ascii_lowercase();
        if vistos.contains(&bajo) {
            return Err(ErrorRutas::SimboloRepetido(s.clone()));
        }
        streams.push(format!("{bajo}@trade"));
        streams.push(format!("{bajo}@depth5"));
        streams.push(format!("{bajo}@kline_1m"));
        vistos.push(bajo);
    }
    streams.push(STREAM_LIQUIDACIONES.to_string());
    Ok(streams)
}

/// `nombre` o `nombre:puerto`, sin esquema, ruta, consulta ni credenciales.
fn host_valido(host: &str) -> bool {
    let (nombre, puerto) = match host.rsplit_once(':') {
        Some((n, p)) => (n, Some(p)),
        None => (host, None),
    };
    let nombre_ok = !nombre.is_empty()
        && nombre
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'-')
        && !nombre.starts_with(['.', '-'])
        && !nombre.ends_with(['.', '-']);
    let puerto_ok = match puerto {
        None => true,
        Some(p) => !p.is_empty() && p.len() <= 5 && p.parse::<u16>().map(|v| v > 0).unwrap_or(false),
    };
    nombre_ok && puerto_ok
}

/// Reparte `streams` por categoría y arma una URL combinada por categoría.
pub fn urls_para(host: &str, streams: &[String]) -> Result<UrlsFeed, ErrorRutas> {
    if !host_valido(host) {
        return Err(ErrorRutas::HostInvalido(host.to_string()));
    }
    if streams.is_empty() {
        return Err(ErrorRutas::UniversoVacio);
    }
    let mut publicos: Vec<&str> = Vec::with_capacity(streams.len());
    let mut mercado: Vec<&str> = Vec::with_capacity(streams.len());
    for s in streams {
        match categoria(s) {
            Some(Categoria::Publica) => publicos.push(s),
            Some(Categoria::Mercado) => mercado.push(s),
            None => return Err(ErrorRutas::SinCategoria(s.clone())),
        }
    }
    for (cat, lista) in [(Categoria::Publica, &publicos), (Categoria::Mercado, &mercado)] {
        if lista.is_empty() {
            // Una conexión sin streams no recibe nada y el watchdog la
            // reconectaría sin fin: el host suscribe siempre las dos.
            return Err(ErrorRutas::UniversoVacio);
        }
        if lista.len() > MAX_STREAMS_POR_CONEXION {
            return Err(ErrorRutas::DemasiadosStreams { categoria: cat, cuantos: lista.len() });
        }
    }
    let url = |cat: Categoria, lista: &[&str]| {
        format!("wss://{host}/{}/stream?streams={}", cat.ruta(), lista.join("/"))
    };
    Ok(UrlsFeed {
        publica: url(Categoria::Publica, &publicos),
        mercado: url(Categoria::Mercado, &mercado),
    })
}

/// URLs del feed del host para `simbolos` en `host`.
pub fn urls_feed(host: &str, simbolos: &[String]) -> Result<UrlsFeed, ErrorRutas> {
    urls_para(host, &streams_del_host(simbolos)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn universo(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn i1b_categorias_medidas_en_las_sondas() {
        // Sondas del 2026-10-24: públicas y de mercado.
        for s in ["btcusdt@trade", "btcusdt@depth5", "btcusdt@bookTicker", "ethusdt@depth5@100ms"] {
            assert_eq!(categoria(s), Some(Categoria::Publica), "{s}");
        }
        for s in ["btcusdt@kline_1m", "btcusdt@aggTrade", STREAM_LIQUIDACIONES] {
            assert_eq!(categoria(s), Some(Categoria::Mercado), "{s}");
        }
    }

    #[test]
    fn i1b_tipo_desconocido_no_tiene_categoria() {
        for s in [
            "btcusdt@markPrice",
            "btcusdt@depth7",
            "btcusdt@depth5@1s",
            "btcusdt@kline_",
            "btcusdt@kline_1x",
            "btcusdt@kline_1ñ",
            "btcusdt@trade@100ms",
            "@trade",
            "btc-usdt@trade",
            "btcusdt",
            "",
        ] {
            assert_eq!(categoria(s), None, "{s}");
        }
    }

    #[test]
    fn i1b_urls_del_host_separan_libro_de_velas_y_liquidaciones() {
        let u = urls_feed("fstream.binance.com", &universo(&["BTCUSDT", "1000SHIBUSDT"])).unwrap();
        assert_eq!(
            u.publica,
            "wss://fstream.binance.com/public/stream?streams=btcusdt@trade/btcusdt@depth5/1000shibusdt@trade/1000shibusdt@depth5"
        );
        assert_eq!(
            u.mercado,
            "wss://fstream.binance.com/market/stream?streams=btcusdt@kline_1m/1000shibusdt@kline_1m/!forceOrder@arr"
        );
    }

    #[test]
    fn i1b_misma_estructura_en_demo_y_produccion() {
        let simbolos = universo(&["BTCUSDT", "ETHUSDT"]);
        let prod = urls_feed("fstream.binance.com", &simbolos).unwrap();
        let demo = urls_feed("stream.binancefuture.com", &simbolos).unwrap();
        assert_eq!(
            prod.publica.replace("fstream.binance.com", "H"),
            demo.publica.replace("stream.binancefuture.com", "H")
        );
        assert_eq!(
            prod.mercado.replace("fstream.binance.com", "H"),
            demo.mercado.replace("stream.binancefuture.com", "H")
        );
    }

    #[test]
    fn i1b_host_con_puerto_y_hosts_invalidos() {
        assert!(urls_feed("fstream.binance.com:443", &universo(&["BTCUSDT"])).is_ok());
        for h in [
            "",
            "wss://fstream.binance.com",
            "fstream.binance.com/stream",
            "a b",
            "host:",
            "host:0",
            "host:70000",
            "user@host",
            ".host",
        ] {
            assert_eq!(
                urls_feed(h, &universo(&["BTCUSDT"])),
                Err(ErrorRutas::HostInvalido(h.to_string())),
                "{h}"
            );
        }
    }

    #[test]
    fn i1b_universo_invalido_falla_sin_recortar() {
        let h = "fstream.binance.com";
        assert_eq!(urls_feed(h, &[]), Err(ErrorRutas::UniversoVacio));
        assert_eq!(
            urls_feed(h, &universo(&["BTCUSDT", "btc/usdt"])),
            Err(ErrorRutas::SimboloInvalido("btc/usdt".into()))
        );
        assert_eq!(
            urls_feed(h, &universo(&["BTCUSDT", "btcusdt"])),
            Err(ErrorRutas::SimboloRepetido("btcusdt".into()))
        );
        assert_eq!(
            urls_para(h, &universo(&["btcusdt@markPrice"])),
            Err(ErrorRutas::SinCategoria("btcusdt@markPrice".into()))
        );
        // Sólo públicos: la conexión de mercado quedaría vacía.
        assert_eq!(urls_para(h, &universo(&["btcusdt@trade"])), Err(ErrorRutas::UniversoVacio));
    }

    #[test]
    fn i1b_tope_de_streams_por_conexion() {
        let muchos: Vec<String> = (0..=MAX_STREAMS_POR_CONEXION / 2).map(|i| format!("S{i}USDT")).collect();
        assert_eq!(
            urls_feed("fstream.binance.com", &muchos),
            Err(ErrorRutas::DemasiadosStreams {
                categoria: Categoria::Publica,
                cuantos: 2 * (MAX_STREAMS_POR_CONEXION / 2 + 1)
            })
        );
        let treinta: Vec<String> = (0..30).map(|i| format!("S{i}USDT")).collect();
        assert!(urls_feed("fstream.binance.com", &treinta).is_ok());
    }
}
