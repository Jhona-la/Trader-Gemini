use serde_json::Value;

/// Domain validation of exchange OHLCV strings, not an outlier or alpha filter.
/// Preserves valid observations exactly. Time ordering/closure is a separate contract.
fn parse_warmup_kline(kline: &[Value]) -> Option<[f64; 5]> {
    if kline.len() < 6 {
        return None;
    }
    let parse = |index: usize| kline[index].as_str()?.parse::<f64>().ok();
    let bar = [parse(1)?, parse(2)?, parse(3)?, parse(4)?, parse(5)?];
    let [o, h, l, c, v] = bar;
    (bar.iter().all(|value| value.is_finite())
        && l > 0.0
        && v >= 0.0
        && l <= o
        && o <= h
        && l <= c
        && c <= h)
        .then_some(bar)
}

/// CL-156 (G021 de Línea G) — URL de las K-lines REST del calentamiento en
/// el venue del proceso: testnet si el host eligió testnet para el cliente
/// REST y el ejecutor (`USE_TESTNET=true`), mainnet si no. Misma base que el
/// ejecutor (`executor.rs`); el WS de testnet es `stream.binancefuture.com`.
/// Antes el calentamiento vivo fijaba `fapi.binance.com` y en testnet
/// sembraba Hurst, FFT, multifractal y σ̂ con la serie de otro mercado.
fn warmup_klines_url(usa_testnet: bool, symbol: &str, limit: usize) -> String {
    let base = if usa_testnet {
        "https://testnet.binancefuture.com"
    } else {
        "https://fapi.binance.com"
    };
    format!("{base}/fapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}")
}

#[cfg(test)]
mod audit_xiii_tests {
    use super::*;

    #[test]
    fn warmup_rejects_nonfinite_and_malformed_fields() {
        for field in 1..=5 {
            for invalid in ["NaN", "inf", "-inf", "invalid"] {
                let mut bar = serde_json::json!([1000, "10", "12", "9", "11", "2"])
                    .as_array()
                    .unwrap()
                    .clone();
                bar[field] = Value::String(invalid.into());
                assert!(
                    parse_warmup_kline(&bar).is_none(),
                    "field={field} value={invalid}"
                );
            }
        }
    }

    #[test]
    fn warmup_requires_true_ohlc_geometry_and_nonnegative_volume() {
        for bad in [
            serde_json::json!([1000, "13", "12", "9", "11", "2"]),
            serde_json::json!([1000, "8", "12", "9", "11", "2"]),
            serde_json::json!([1000, "10", "12", "-1", "11", "2"]),
            serde_json::json!([1000, "10", "12", "9", "11", "-2"]),
        ] {
            assert!(
                parse_warmup_kline(bad.as_array().unwrap()).is_none(),
                "{bad}"
            );
        }
    }

    #[test]
    fn warmup_preserves_valid_market_values_including_zero_volume() {
        let bar = serde_json::json!([1000, "10", "12", "9", "11", "0"]);
        assert_eq!(
            parse_warmup_kline(bar.as_array().unwrap()),
            Some([10.0, 12.0, 9.0, 11.0, 0.0])
        );
        assert_eq!(parse_warmup_kline(&[]), None);
    }

    /// CL-156 (G021) — en testnet las K-lines salen de testnet; en mainnet
    /// la URL es la de siempre, byte a byte.
    #[test]
    fn cl156_warmup_klines_url_sigue_al_venue() {
        assert_eq!(
            warmup_klines_url(true, "1000SHIBUSDT", 1000),
            "https://testnet.binancefuture.com/fapi/v1/klines?symbol=1000SHIBUSDT&interval=1m&limit=1000"
        );
        assert_eq!(
            warmup_klines_url(false, "1000SHIBUSDT", 1000),
            "https://fapi.binance.com/fapi/v1/klines?symbol=1000SHIBUSDT&interval=1m&limit=1000"
        );
    }

    /// CL-37 — el universo de arranque sale en la grafía canónica, sin
    /// duplicados y dentro de la capacidad del arena: es el mapa slot↔símbolo
    /// que comparten host, núcleo, riesgo y reserva durante toda la sesión.
    #[tokio::test]
    async fn cl37_universo_de_arranque_canonico_unico_y_dentro_de_capacidad() {
        let (_, syms) = SystemDiagnostics::execute_phase_1_and_2(false)
            .await
            .expect("bootloader");
        assert!(!syms.is_empty() && syms.len() <= quantum_arena::state::MAX_COINS);
        let mut vistos = std::collections::HashSet::new();
        for s in &syms {
            assert!(
                !s.is_empty() && s.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit()),
                "{s}: el universo debe ir en MAYÚSCULAS ASCII (grafía de models/ y del registro)"
            );
            assert!(vistos.insert(s.clone()), "{s} duplicado");
        }
    }
}

/// CL-123 — EL ROSTER DEL MOTOR, con el nombre que el símbolo tiene en
/// futuros USDⓈ-M de Binance, en MAYÚSCULAS (CL-37: la grafía del registro y
/// de `models/`; los streams del WS se construyen aparte en minúsculas). Es
/// la única lista: el generador de cópulas (`copulas_manifest`) la usa tal
/// cual, así que sus claves son las del roster.
///
/// SHIB cotiza en USDⓈ-M como `1000SHIBUSDT` (mil unidades por contrato).
/// Antes el roster decía `shibusdt`, que no existe en futuros: esa moneda no
/// recibía datos.
pub const ROSTER: &[&str] = &[
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "1000SHIBUSDT", "DOTUSDT", "LINKUSDT", "POLUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT",
    "UNIUSDT", "XLMUSDT", "NEARUSDT", "ICPUSDT", "FILUSDT", "VETUSDT", "AVAXUSDT",
    "OPUSDT", "APTUSDT", "ARBUSDT", "RENDERUSDT", "LDOUSDT",
];

// --- LEGACY SYSTEM DIAGNOSTICS FOR MAIN ENGINE COMPATIBILITY ---
pub struct SystemDiagnostics;

impl SystemDiagnostics {
    /// F1-064 (Ola Ω172): Carga y valida el universo dinámico persistido por `symbol_manager`.
    ///
    /// Valida rigurosamente:
    /// 1. Existencia y formato JSON de `dynamic_config.json`.
    /// 2. Capacidad estricta: `0 < symbols.len() <= quantum_arena::state::MAX_COINS`.
    /// 3. Grafía canónica ASCII mayúscula y sin duplicados.
    /// 4. Disciplina de Roster B3.37: Cada símbolo propuesto debe poseer un modelo activo
    ///    en `models/{SYM}_MOTOR.json` o pertenecer al `ROSTER` canónico fundacional.
    /// 5. Backfill seguro: Si el filtro de modelos reduce la lista por debajo del tamaño de `ROSTER`,
    ///    se completan las plazas con los incumbentes canónicos de `ROSTER` para evitar huecos en la sesión.
    pub fn load_validated_dynamic_universe(
        dynamic_config_path: &std::path::Path,
        models_dir: &std::path::Path,
    ) -> Option<Vec<String>> {
        if !dynamic_config_path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(dynamic_config_path).ok()?;
        let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
        let raw_symbols = parsed.get("symbols")?.as_array()?;

        if raw_symbols.is_empty() {
            return None;
        }

        // 1. Cargar conjunto de modelos disponibles en models/{SYM}_MOTOR.json
        let mut model_roster = std::collections::HashSet::new();
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().into_owned();
                if name.ends_with("_MOTOR.json") && !name.contains("_CANDIDATE") {
                    let sym = name.trim_end_matches("_MOTOR.json").to_string();
                    if !sym.is_empty() {
                        model_roster.insert(sym.to_ascii_uppercase());
                    }
                }
            }
        }

        let roster_set: std::collections::HashSet<&str> = ROSTER.iter().copied().collect();

        // 2. Filtrar y deduplicar respetando mayúsculas y disciplina de modelos
        let mut validated = Vec::with_capacity(quantum_arena::state::MAX_COINS);
        let mut seen = std::collections::HashSet::new();

        for item in raw_symbols {
            if validated.len() >= quantum_arena::state::MAX_COINS {
                break;
            }
            if let Some(s) = item.as_str() {
                let sym_upper = s.trim().to_ascii_uppercase();
                if !sym_upper.is_empty()
                    && sym_upper.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit())
                    && seen.insert(sym_upper.clone())
                {
                    // Aceptar si tiene modelo validado O si está en ROSTER canónico
                    if model_roster.contains(&sym_upper) || roster_set.contains(sym_upper.as_str()) {
                        validated.push(sym_upper);
                    }
                }
            }
        }

        if validated.is_empty() {
            return None;
        }

        // 3. Backfill desde ROSTER si faltan plazas para mantener continuidad
        if validated.len() < ROSTER.len() {
            for &r in ROSTER {
                if validated.len() >= ROSTER.len() || validated.len() >= quantum_arena::state::MAX_COINS {
                    break;
                }
                let sym = r.to_string();
                if seen.insert(sym.clone()) {
                    validated.push(sym);
                }
            }
        }

        Some(validated)
    }

    pub async fn execute_phase_1_and_2(
        is_demo: bool,
    ) -> Result<(i64, Vec<String>), Box<dyn std::error::Error>> {
        // En tests o modo demo con intención de aislamiento, si no hay dynamic config validado, o is_demo,
        // devolvemos el ROSTER canónico intacto (garantiza compatibilidad 100% con cf5_roster).
        if !is_demo {
            let dynamic_path = std::path::Path::new("data/dynamic_config.json");
            let models_dir = std::path::Path::new("models");
            if let Some(syms) = Self::load_validated_dynamic_universe(dynamic_path, models_dir) {
                return Ok((0, syms));
            }
        }
        Ok((0, ROSTER.iter().map(|s| s.to_string()).collect()))
    }

    /// X-017 (REHAB-6): OHLCV COMPLETO por kline — antes solo closes con
    /// banda sintética ±0.05% y volumen fijo 10: v_t (volatilidad de rango)
    /// quedaba clavado en ~0.1% y el ATR vivo des-calibraba los stops por
    /// horas (get_atr_pct alimenta el sizing en vivo).
    ///
    /// CL-156 (G021): `usa_testnet` es el venue que el host eligió para el
    /// cliente REST y el ejecutor (`is_testnet`); las K-lines salen de ese
    /// mismo venue (`warmup_klines_url`). No es el modo demo: SHADOW_MODE y
    /// BINANCE_USE_DEMO operan papel contra mainnet.
    pub async fn execute_phase_3_warmup(
        usa_testnet: bool,
        symbols: &[String],
    ) -> std::collections::HashMap<String, Vec<[f64; 5]>> {
        let mut map = std::collections::HashMap::new();
        let client = reqwest::Client::builder()
            .user_agent("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            .timeout(tokio::time::Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let limit = 1000;

        for symbol in symbols {
            let mut klines_vec = Vec::new();
            let url = warmup_klines_url(usa_testnet, &symbol.to_uppercase(), limit);

            if let Ok(resp) = client.get(&url).send().await {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let serde_json::Value::Array(klines) = json {
                        for k in &klines {
                            if let serde_json::Value::Array(kline_arr) = k {
                                // [openTime, open, high, low, close, volume, ...]
                                if let Some(bar) = parse_warmup_kline(kline_arr) {
                                    klines_vec.push(bar);
                                }
                            }
                        }
                        let rejected = klines.len() - klines_vec.len();
                        if rejected > 0 {
                            eprintln!(
                                "Warmup {symbol}: {rejected} klines rejected by schema/OHLCV validation; temporal coverage is incomplete."
                            );
                        }
                    }
                }
            }
            map.insert(symbol.clone(), klines_vec);
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        map
    }

    pub async fn execute_phase_4_training(
        _symbols: &[String],
        _historical_klines: &std::collections::HashMap<String, Vec<[f64; 5]>>,
        _lr: f64,
        _epochs: usize,
    ) -> Option<()> {
        Some(())
    }

    pub fn execute_phase_6_hft() {
        println!(
            "🚀 [HFT IGNITION] SystemDiagnostics::execute_phase_6_hft called. Ignition sequence started."
        );
    }
}
