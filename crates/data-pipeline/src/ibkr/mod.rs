//! IBKR-I2 (integrador, 2026-11-16): conector de Interactive Brokers de SOLO
//! LECTURA. Solo datos de mercado: este modulo no tiene ninguna ruta a ordenes
//! (lo vigila `tests/ibkr_solo_lectura_contract.rs`).
//!
//! Pedido: `.agent_tx/diagnostics/auditoria_perdidas/junie_cli_pedido_conector_ibkr_solo_lectura_v2.md`
//! (pasos I1-I8). Especificacion: `agy_cli_interactive_brokers_solo_lectura_v2.md`.
//!
//! Marcas de cada limite: OFICIAL (cifra publicada por IBKR, con su URL),
//! DERIVADO (calculado desde una cifra oficial, con la cuenta) o N/D (no existe
//! cifra oficial; decision de diseno del pedido).

pub mod cache;
pub mod pacing;

use ibapi::contracts::Contract;
use ibapi::market_data::MarketDataType;
use ibapi::Client;

/// Maximo de peticiones historicas por ventana de 10 minutos: 60. OFICIAL.
/// Fuente: https://www.interactivebrokers.com/docs/tws-api/doc/md/market-data-limitations
/// (vigente; la URL deprecada era historical_limitations.html).
/// Aplica bajo "Small Bars (30 secs or less)". Para barras >= 1 min, IBKR levanto
/// el limite duro pero sigue frenando uso excesivo.
/// Consultado: 2026-11-16 (pedido v2, I2).
pub const IBKR_HISTORICAL_MAX_PER_10MIN: usize = 60;

/// Espaciador minimo entre peticiones: 2500 ms. DERIVADO.
/// Con 2500 ms entre envios es imposible emitir 6 en 2 s (la regla oficial de
/// rafaga: 6 o mas peticiones del mismo contrato en 2 s). No existe cifra
/// oficial de espaciador.
pub const IBKR_MIN_PACING_INTERVAL_MS: u64 = 2500;

/// TTL de la cache de deduplicacion: 30 s. DERIVADO.
/// 30 s = 2 x el minimo oficial de 15 s entre peticiones identicas.
pub const IBKR_DEDUP_CACHE_TTL_SECS: u64 = 30;

/// Techo de lineas streaming: 80. DERIVADO.
/// 100 oficial menos el margen de 20 del diseno (lineas libres para el uso
/// manual del Gateway). Fuente oficial del 100:
/// https://interactivebrokers.github.io/tws-api/market_data.html
/// (la URL market_data_lines.html devuelve 404).
pub const IBKR_MAX_STREAMING_LINES_BUDGET: usize = 80;

/// Enfriamiento tras Error 162: 65 s. DERIVADO / N/D.
/// No existe cifra oficial de enfriamiento. Decision de diseno del pedido.
pub const IBKR_PACING_BREAKER_COOLDOWN_SECS: u64 = 65;

/// Configuracion de la conexion de lectura al IB Gateway. Solo campos de
/// lectura: no guarda credenciales (el usuario entra en el Gateway, no aqui)
/// ni ningun cliente conectado.
#[derive(Debug, Clone)]
pub struct IbkrGateway {
    /// `host:puerto` del Gateway local (lo elige el operador al arrancarlo).
    pub address: String,
    /// Identificador de cliente de la API.
    pub client_id: i32,
    /// Tipo de datos de mercado pedido (en vivo, congelado o diferido).
    pub market_data_type: MarketDataType,
}

/// Una barra historica de IBKR, con el esquema de la cinta
/// (`timestamp_utc,symbol,open,high,low,close,volume,wap,count`).
#[derive(Debug, Clone, PartialEq)]
pub struct IbkrBar {
    pub timestamp: std::time::SystemTime,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
    /// Precio medio ponderado por volumen de la barra.
    pub wap: f64,
    /// Numero de operaciones de la barra.
    pub count: u32,
}

/// Nombres de los tipos de ibapi que usa este modulo (solo lectura). Sirve
/// para fijar en un solo sitio que el modulo importa tipos exactos y no el
/// prelude de ibapi.
pub fn tipos_ibapi_usados() -> [&'static str; 3] {
    [
        std::any::type_name::<Client>(),
        std::any::type_name::<Contract>(),
        std::any::type_name::<MarketDataType>(),
    ]
}
