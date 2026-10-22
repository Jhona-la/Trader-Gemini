use arc_swap::ArcSwap;
use std::sync::Arc;

lazy_static::lazy_static! {
    static ref DYNAMIC_UNIVERSE: ArcSwap<Vec<String>> = ArcSwap::from_pointee(get_default_symbols());
}

/// CL-37 — grafía canónica de un símbolo: ASCII en MAYÚSCULAS, la que usan
/// Binance en sus respuestas, el registro de specs y los stems de `models/`.
pub fn simbolo_canonico(s: &str) -> String {
    s.to_ascii_uppercase()
}

/// CL-37 — el universo se publica en la grafía canónica y con el MISMO orden
/// de slots. El bootloader lo daba en minúsculas mientras el registro iba en
/// MAYÚSCULAS: `try_symbol(i)` ≠ `try_spec(i).symbol` y toda clave que
/// distingue mayúsculas (`{SYM}_MOTOR`, `{SYM}_VOL`, el funding con scope,
/// el NN sólo-BTC) fallaba en todos los slots.
pub fn update_dynamic_universe(new_universe: Vec<String>) {
    let canonico: Vec<String> = new_universe.iter().map(|s| simbolo_canonico(s)).collect();
    DYNAMIC_UNIVERSE.store(Arc::new(canonico));
    // CL-89: el slot → símbolo cambió; los ticks cacheados por slot caducan.
    crate::symbol_registry::nueva_generacion();
}

pub fn get_active_universe_ref() -> Arc<Vec<String>> {
    DYNAMIC_UNIVERSE.load_full()
}

pub fn with_active_universe<F, R>(f: F) -> R
where
    F: FnOnce(&[String]) -> R,
{
    let u = DYNAMIC_UNIVERSE.load();
    f(&u)
}

pub fn get_active_universe() -> Vec<String> {
    let u = DYNAMIC_UNIVERSE.load();
    (**u).clone()
}

pub fn get_active_universe_size() -> usize {
    DYNAMIC_UNIVERSE.load().len()
}

/// D-725: el id de una moneda es su posición en el UNIVERSO — el mismo índice
/// con el que el productor de datos escribe en `arena.coins[i]`. Sólo con el
/// universo vacío (arranque, o binarios que sólo registran specs) se recurre al
/// registro, que entonces es el único espacio de índices existente.
pub fn get_coin_id(symbol: &str) -> Option<usize> {
    let u = DYNAMIC_UNIVERSE.load();
    if !u.is_empty() {
        return u.iter().position(|s| s.eq_ignore_ascii_case(symbol));
    }
    crate::symbol_registry::try_index(symbol)
}

fn get_default_symbols() -> Vec<String> {
    Vec::new()
}

/// Lista canónica de stablecoins a excluir de rankings y universo dinámico.
pub const CANONICAL_STABLECOINS: &[&str] = &[
    "USDCUSDT",
    "FDUSDUSDT",
    "TUSDUSDT",
    "BUSDUSDT",
    "USDPUSDT",
    "EURUSDT",
    "DAIUSDT",
    "AEURUSDT",
];

/// Lista negra canónica de bases no operables / basura en testnet o contratos zombis.
pub const CANONICAL_TESTNET_BLACKLIST: &[&str] = &[
    "CUSDT", "MUSDT", "AUSDT", "4USDT", "USELESSUSDT", "ZENUSDT", "IDOLUSDT",
    "SKYUSDT", "SUPERUSDT", "DRIFTUSDT", "CROSSUSDT", "FARTCOINUSDT", "FLUIDUSDT",
    "VVVUSDT", "ALCHUSDT", "SENTUSDT", "RAYSOLUSDT", "METUSDT", "AZTECUSDT",
    "ONEUSDT", "JUPUSDT", "TIAUSDT", "PENDLEUSDT", "LPTUSDT", "EIGENUSDT",
    "ZKUSDT", "INJUSDT", "ACHUSDT", "OPUSDT",
];

/// Predicado canónico de saneamiento de símbolos de Binance Futures (F4.6 / QO-U1a / Ola Ω215).
/// - Requiere longitud >= 7 y terminación "USDT".
/// - Base ASCII en mayúsculas o dígitos con longitud >= 3 (filtra mojibake UTF-8 y bases triviales).
/// - Excluye stablecoins y, si is_testnet, símbolos en la lista negra.
pub fn is_sane_binance_symbol(symbol: &str, is_testnet: bool) -> bool {
    let sym = symbol.trim();
    if sym.len() < 7 || !sym.ends_with("USDT") {
        return false;
    }
    let base = &sym[..sym.len() - 4];
    if base.len() < 3 || !base.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()) {
        return false;
    }
    if CANONICAL_STABLECOINS.iter().any(|&s| s.eq_ignore_ascii_case(sym)) {
        return false;
    }
    if is_testnet && CANONICAL_TESTNET_BLACKLIST.iter().any(|&b| b.eq_ignore_ascii_case(sym)) {
        return false;
    }
    true
}

/// Cálculo canónico del score de ranking del universo multivariante espectral (Ola Ω215).
/// 
/// Combina en una única métrica continua:
/// 1. Liquidez de cola pesada: ln(1 + volumen_usd).
/// 2. Volatilidad espectral unimodal: pct * exp(-pct / 15.0), con pico en 15% diario
///    (premia movimiento real y atenúa picos especulativos de manipulación).
/// 3. Atenuación por fricción de spread: exp(-spread_bps / 10.0) si está disponible.
/// 4. Alineación fractal: 1.0 + 2.0 * |H - 0.5| si el Hurst proxy está disponible.
/// 
/// Garantiza no-negatividad, finitud estricta e inmunidad total a NaN/Inf.
pub fn calculate_spectral_universe_score(
    volume_usd: f64,
    price_change_pct: f64,
    spread_bps: Option<f64>,
    hurst_proxy: Option<f64>,
) -> f64 {
    let vol = if volume_usd.is_finite() && volume_usd > 0.0 {
        volume_usd
    } else {
        0.0
    };
    let pct = if price_change_pct.is_finite() && price_change_pct >= 0.0 {
        price_change_pct
    } else {
        0.0
    };
    if vol == 0.0 || pct == 0.0 {
        return 0.0;
    }

    let vol_factor = (1.0 + vol).ln();
    let vola_factor = pct * (-pct / 15.0).exp();

    let spread_factor = match spread_bps {
        Some(s) if s.is_finite() && s >= 0.0 => (-s / 10.0).exp(),
        _ => 1.0,
    };

    let hurst_factor = match hurst_proxy {
        Some(h) if h.is_finite() && (0.0..=1.0).contains(&h) => 1.0 + 2.0 * (h - 0.5).abs(),
        _ => 1.0,
    };

    let score = vol_factor * vola_factor * spread_factor * hurst_factor;
    if score.is_finite() && score > 0.0 {
        score
    } else {
        0.0
    }
}

#[cfg(test)]
pub static UNIVERSE_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_universe_update_and_lookup() {
        let _guard = UNIVERSE_TEST_MUTEX.lock().unwrap();
        let universe = vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(),
        ];
        update_dynamic_universe(universe.clone());

        assert_eq!(get_active_universe_size(), 3);
        assert_eq!(get_active_universe(), universe);

        with_active_universe(|u| {
            assert_eq!(u.len(), 3);
            assert_eq!(u[0], "BTCUSDT");
        });
    }

    #[test]
    fn test_is_sane_binance_symbol_filters() {
        // Rechazos
        assert!(!is_sane_binance_symbol("牛来USDT", false), "Rechaza mojibake");
        assert!(!is_sane_binance_symbol("4USDT", false), "Rechaza base < 3 chars");
        assert!(!is_sane_binance_symbol("AUSDT", false), "Rechaza base < 3 chars");
        assert!(!is_sane_binance_symbol("USDCUSDT", false), "Rechaza stablecoin USDC");
        assert!(!is_sane_binance_symbol("FDUSDUSDT", false), "Rechaza stablecoin FDUSD");
        assert!(!is_sane_binance_symbol("BTCBUSD", false), "Rechaza no-USDT");
        assert!(!is_sane_binance_symbol("USELESSUSDT", true), "Rechaza blacklist en testnet");

        // Aceptaciones
        assert!(is_sane_binance_symbol("BTCUSDT", false));
        assert!(is_sane_binance_symbol("ETHUSDT", false));
        assert!(is_sane_binance_symbol("SOLUSDT", false));
        assert!(is_sane_binance_symbol("1000PEPEUSDT", false), "Bases con dígitos válidas");
    }

    #[test]
    fn test_calculate_spectral_universe_score_properties() {
        // Finitud e inmunidad NaN
        assert_eq!(calculate_spectral_universe_score(f64::NAN, 10.0, None, None), 0.0);
        assert_eq!(calculate_spectral_universe_score(1_000_000.0, f64::NAN, None, None), 0.0);
        assert_eq!(calculate_spectral_universe_score(-100.0, 10.0, None, None), 0.0);

        // Volatilidad moderada (15%) gana contra extrema manipulada (60%)
        let s_15 = calculate_spectral_universe_score(10_000_000.0, 15.0, None, None);
        let s_60 = calculate_spectral_universe_score(10_000_000.0, 60.0, None, None);
        assert!(s_15 > s_60, "15% vol debe superar 60% vol extrema");

        // Spread de fricción atenúa score
        let s_tight = calculate_spectral_universe_score(10_000_000.0, 15.0, Some(2.0), None);
        let s_wide = calculate_spectral_universe_score(10_000_000.0, 15.0, Some(20.0), None);
        assert!(s_tight > s_wide, "Spread estrecho debe superar spread amplio");

        // Memoria fractal premia persistencia frente a ruido browniano
        let s_trend = calculate_spectral_universe_score(10_000_000.0, 15.0, None, Some(0.70));
        let s_noise = calculate_spectral_universe_score(10_000_000.0, 15.0, None, Some(0.50));
        assert!(s_trend > s_noise, "Persistencia fractal H=0.70 debe superar ruido H=0.50");
    }
}
