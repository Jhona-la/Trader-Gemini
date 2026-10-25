use arc_swap::ArcSwap;
use lazy_static::lazy_static;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SymbolSpec {
    pub symbol: String,
    pub step_size: f64,
    pub tick_size: f64,
    pub min_qty: f64,
    pub min_notional: f64,
    pub max_leverage: u32,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub is_shadow: bool,
}

lazy_static! {
    static ref DYNAMIC_REGISTRY: ArcSwap<Vec<SymbolSpec>> =
        ArcSwap::from_pointee(get_default_specs());
}

/// CL-89 (atr-3): generación del registro de specs y del universo. Sube cada
/// vez que se publica uno de los dos, después de publicarlo. Quien guarda un
/// valor derivado de ellos (el tick de un slot) lo recalcula sólo cuando
/// cambia: si lee la generación vieja y el registro nuevo, recalcula en la
/// siguiente consulta.
static GENERACION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// CL-89: generación vigente del registro y del universo (nunca 0).
#[inline(always)]
pub fn generacion() -> u64 {
    GENERACION.load(std::sync::atomic::Ordering::Acquire)
}

/// CL-89: marca una publicación del registro o del universo.
pub(crate) fn nueva_generacion() {
    GENERACION.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}

pub fn update_registry(new_specs: Vec<SymbolSpec>) {
    let current = DYNAMIC_REGISTRY.load();
    if current.is_empty() {
        DYNAMIC_REGISTRY.store(Arc::new(new_specs));
        nueva_generacion();
        return;
    }

    // FIX #905: Preservar estabilidad de coin_id para que símbolos existentes mantengan su índice
    let mut updated = (**current).clone();
    for spec in new_specs {
        if let Some(pos) = updated
            .iter()
            .position(|s| s.symbol.eq_ignore_ascii_case(&spec.symbol))
        {
            updated[pos] = spec;
        } else {
            updated.push(spec);
        }
    }
    DYNAMIC_REGISTRY.store(Arc::new(updated));
    nueva_generacion();
}

// D-725 (DÉCIMA OLA · auditoría integral): UN SOLO ESPACIO DE ÍNDICE.
//
// Coexistían dos: el UNIVERSO (`symbols::DYNAMIC_UNIVERSE`, cuya posición usa
// el productor de datos para escribir en `arena.coins[i]`) y el REGISTRO de
// specs, que mantiene su propio orden —preserva posiciones y añade al final—.
// `get_coin_id` prefería el registro, así que en cuanto el ranker publicaba un
// universo reordenado los dos dejaban de coincidir: un fill de BNB se
// contabilizaba contra la posición de SOL, con su precio de entrada y su PnL,
// y el re-bracketeo recolocaba el TP/SL de una moneda sobre el precio de otra.
//
// Ahora el índice es SIEMPRE la posición en el universo; el registro es un mapa
// por símbolo, no un espacio de índices. Con el universo vacío (arranque, o
// binarios que sólo registran specs, como el forense) se conserva el orden del
// registro, que entonces es el único espacio existente.
#[inline(always)]
pub fn try_spec(coin_id: usize) -> Option<SymbolSpec> {
    // We clone the struct to avoid lifetime issues since it's accessed heavily.
    // The struct is small enough that cloning is practically free.
    let registry = DYNAMIC_REGISTRY.load();
    let universe = crate::symbols::get_active_universe_ref();
    if universe.is_empty() {
        return registry.get(coin_id).cloned();
    }
    let symbol = universe.get(coin_id)?;
    registry
        .iter()
        .find(|s| s.symbol.eq_ignore_ascii_case(symbol))
        .cloned()
}

/// CL-69: el tick del slot `coin_id`, con la misma resolución que `try_spec`
/// pero sin clonar el spec (sin asignaciones: se lee en cada evento).
#[inline(always)]
pub fn tick_size_de(coin_id: usize) -> Option<f64> {
    let registry = DYNAMIC_REGISTRY.load();
    let universe = crate::symbols::get_active_universe_ref();
    if universe.is_empty() {
        return registry.get(coin_id).map(|s| s.tick_size);
    }
    let symbol = universe.get(coin_id)?;
    registry
        .iter()
        .find(|s| s.symbol.eq_ignore_ascii_case(symbol))
        .map(|s| s.tick_size)
}

#[inline(always)]
pub fn try_symbol(coin_id: usize) -> Option<String> {
    let universe = crate::symbols::get_active_universe_ref();
    if !universe.is_empty() {
        return universe.get(coin_id).cloned();
    }
    let registry = DYNAMIC_REGISTRY.load();
    registry.get(coin_id).map(|s| s.symbol.clone())
}

/// P5 (Camino Caliente): Acceso cero-copia al símbolo del slot `coin_id` sin alojar `String` en el heap.
#[inline(always)]
pub fn with_symbol<F, R>(coin_id: usize, f: F) -> Option<R>
where
    F: FnOnce(&str) -> R,
{
    let universe = crate::symbols::get_active_universe_ref();
    if !universe.is_empty() {
        return universe.get(coin_id).map(|s| f(s.as_str()));
    }
    let registry = DYNAMIC_REGISTRY.load();
    registry.get(coin_id).map(|s| f(s.symbol.as_str()))
}

/// CL-62 (LC-F1-03): ¿el slot `coin_id` es `simbolo`? Mismo espacio de
/// índices que `try_symbol` (el universo; el registro sólo con el universo
/// vacío), sin copiar el nombre: lo consulta el camino caliente.
#[inline]
pub fn slot_es_simbolo(coin_id: usize, simbolo: &str) -> bool {
    let en_universo = crate::symbols::with_active_universe(|u| {
        (!u.is_empty()).then(|| u.get(coin_id).is_some_and(|s| s.eq_ignore_ascii_case(simbolo)))
    });
    en_universo.unwrap_or_else(|| {
        DYNAMIC_REGISTRY
            .load()
            .get(coin_id)
            .is_some_and(|s| s.symbol.eq_ignore_ascii_case(simbolo))
    })
}

#[inline(always)]
pub fn try_index(symbol: &str) -> Option<usize> {
    let universe = crate::symbols::get_active_universe_ref();
    let sym_upper = symbol.to_uppercase();
    if !universe.is_empty() {
        return universe.iter().position(|s| s.to_uppercase() == sym_upper);
    }
    let registry = DYNAMIC_REGISTRY.load();
    registry
        .iter()
        .position(|s| s.symbol.to_uppercase() == sym_upper)
}

pub fn get_official_binance_spec(symbol: &str) -> SymbolSpec {
    let sym_upper = symbol.to_uppercase();
    let (step_size, tick_size, min_qty) = match sym_upper.as_str() {
        "BTCUSDT" => (0.001, 0.1, 0.001),
        "ETHUSDT" => (0.01, 0.01, 0.01),
        "SOLUSDT" => (0.1, 0.01, 0.1),
        "BNBUSDT" => (0.01, 0.01, 0.01),
        "DOGEUSDT" => (1.0, 0.00001, 1.0),
        "XRPUSDT" => (0.1, 0.0001, 0.1),
        "ADAUSDT" => (1.0, 0.0001, 1.0),
        "AVAXUSDT" => (0.01, 0.001, 0.01),
        "LINKUSDT" => (0.01, 0.001, 0.01),
        "SUIUSDT" => (0.1, 0.0001, 0.1),
        "NEARUSDT" => (0.1, 0.001, 0.1),
        "APTUSDT" => (0.1, 0.001, 0.1),
        "ARBUSDT" => (0.1, 0.0001, 0.1),
        "OPUSDT" => (0.1, 0.0001, 0.1),
        "MATICUSDT" | "POLUSDT" => (1.0, 0.0001, 1.0),
        "DOTUSDT" => (0.1, 0.001, 0.1),
        _ => (0.01, 0.001, 0.01),
    };
    SymbolSpec {
        symbol: sym_upper,
        step_size,
        tick_size,
        min_qty,
        min_notional: 5.0, // Binance USDⓈ-M Futures Official Min Notional
        max_leverage: 20,
        maker_fee: 0.0002, // 0.02%
        taker_fee: 0.0005, // 0.05%
        is_shadow: false,
    }
}

fn get_default_specs() -> Vec<SymbolSpec> {
    Vec::new()
}
