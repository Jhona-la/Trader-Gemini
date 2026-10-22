//! # MÓDULO DE PUNTO FIJO Y FILTROS EXACTOS DE BINANCE (Fase 2 §4.2 Átomo 2.4)
//!
//! **Mapeo de Excelencia Cuántica y Modo Profesor:**
//! - **QUÉ:** Cuantizador determinista y serializador de punto fijo en stack (zero-alloc, O(1))
//!   para los filtros de Binance (`PRICE_FILTER`, `LOT_SIZE`, `MIN_NOTIONAL`).
//! - **POR QUÉ:** La aritmética IEEE-754 de coma flotante `f64` produce residuos espurios
//!   (ej. `0.007000000000000001` o `62450.100000000006`) que causan el error `-1111`
//!   (`Precision is over the maximum defined for this asset`) o rechazos de `LOT_SIZE`/`PRICE_FILTER`.
//! - **PARA QUÉ:** Eliminar el 100% de los rechazos de órdenes en Binance Futures por discrepancia
//!   de redondeo o nocional insuficiente, garantizando que cada orden aprobada por el riesgo sea
//!   admitida por el motor de casamiento sin pagar latencia adicional (< 25 ns en stack).
//! - **CÓMO:** Proyección euclidiana discreta sobre la red de ticks $\mathbb{Z} \cdot \Delta_{\text{tick}}$
//!   y pasos de lote $\mathbb{Z} \cdot \Delta_{\text{step}}$, formateando los dígitos ASCII directamente
//!   en buffers de memoria contigua en la pila con `itoa`.
//! - **CUÁNDO:** En cada generación de payload de orden (`build_payload`), stop de protección y take-profit.
//! - **DÓNDE:** `crates/execution-engine/src/fixed_filter.rs`.
//! - **QUIÉN:** AGY (Lead Quant Sr. & Consejo de 10 Roles Senior).

use std::fmt;
use std::ops::Deref;

/// Tolerancia relativa para considerar que un valor escalado ya es múltiplo exacto del paso.
const SNAP_REL_TOL: f64 = 1e-9;

/// Representación en stack de una cadena ASCII de longitud fija sin asignación en heap.
#[derive(Clone, Copy)]
pub struct StackAsciiString {
    bytes: [u8; 32],
    len: usize,
}

impl StackAsciiString {
    /// Crea una cadena vacía.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            bytes: [0u8; 32],
            len: 0,
        }
    }

    /// Devuelve una referencia a la cadena de texto UTF-8 / ASCII.
    #[inline(always)]
    pub fn as_str(&self) -> &str {
        // Garantizado ASCII válido por construcción en format_fixed_ascii
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len]) }
    }

    /// Devuelve el slice de bytes sin procesar.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// Longitud de la cadena en bytes.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Indica si la cadena está vacía.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for StackAsciiString {
    type Target = str;
    #[inline(always)]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for StackAsciiString {
    #[inline(always)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for StackAsciiString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StackAsciiString(\"{}\")", self.as_str())
    }
}

impl PartialEq for StackAsciiString {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialEq<str> for StackAsciiString {
    #[inline(always)]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for StackAsciiString {
    #[inline(always)]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl Eq for StackAsciiString {}

/// Razones rigurosas de rechazo de un filtro de orden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FilterRejectReason {
    /// Precio o cantidad no finita (NaN o Infinito).
    NonFinite,
    /// Precio o cantidad menor o igual a cero.
    ZeroOrNegative,
    /// El precio está por debajo de `minPrice`.
    PriceBelowMin { current: f64, min: f64 },
    /// El precio está por encima de `maxPrice`.
    PriceAboveMax { current: f64, max: f64 },
    /// La cantidad está por debajo de `minQty`.
    QtyBelowMin { current: f64, min: f64 },
    /// La cantidad está por encima de `maxQty`.
    QtyAboveMax { current: f64, max: f64 },
    /// El valor nocional (`precio * cantidad`) está por debajo de `minNotional`.
    NotionalBelowMin { current: f64, min: f64 },
    /// Parámetros del filtro inválidos o corruptos.
    InvalidFilterConfig,
}

impl fmt::Display for FilterRejectReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "Non-finite price or quantity"),
            Self::ZeroOrNegative => write!(f, "Zero or negative price or quantity"),
            Self::PriceBelowMin { current, min } => {
                write!(f, "Price {} below minPrice {}", current, min)
            }
            Self::PriceAboveMax { current, max } => {
                write!(f, "Price {} above maxPrice {}", current, max)
            }
            Self::QtyBelowMin { current, min } => {
                write!(f, "Qty {} below minQty {}", current, min)
            }
            Self::QtyAboveMax { current, max } => {
                write!(f, "Qty {} above maxQty {}", current, max)
            }
            Self::NotionalBelowMin { current, min } => {
                write!(f, "Notional {} below minNotional {}", current, min)
            }
            Self::InvalidFilterConfig => write!(f, "Invalid filter configuration"),
        }
    }
}

/// Parámetros validados, cuantizados y serializados listos para Binance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizedOrder {
    /// Precio redondeado al tick exacto.
    pub price: f64,
    /// Cantidad redondeada al paso de lote exacto.
    pub qty: f64,
    /// Nocional final (`price * qty`).
    pub notional: f64,
    /// Cadena exacta para el query REST o WebSocket (cero residuos de float).
    pub price_str: StackAsciiString,
    /// Cadena exacta para el query REST o WebSocket (cero residuos de float).
    pub qty_str: StackAsciiString,
}

/// Reglas inmutables de filtro de un símbolo de futuros de Binance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinanceFixedFilter {
    pub tick_size: f64,
    pub step_size: f64,
    pub min_price: f64,
    pub max_price: f64,
    pub min_qty: f64,
    pub max_qty: f64,
    pub min_notional: f64,
    pub price_decimals: u8,
    pub qty_decimals: u8,
    pub inv_tick: f64,
    pub inv_step: f64,
}

impl BinanceFixedFilter {
    /// Construye las reglas del filtro deduciendo la precisión decimal exacta.
    pub fn new(tick_size: f64, step_size: f64, min_notional: f64) -> Self {
        Self::with_limits(
            tick_size,
            step_size,
            tick_size,
            10_000_000.0,
            step_size,
            10_000_000.0,
            min_notional,
        )
    }

    /// Construye el filtro con límites explícitos de min/max precio y cantidad.
    pub fn with_limits(
        tick_size: f64,
        step_size: f64,
        min_price: f64,
        max_price: f64,
        min_qty: f64,
        max_qty: f64,
        min_notional: f64,
    ) -> Self {
        let price_decimals = count_decimals(tick_size);
        let qty_decimals = count_decimals(step_size);
        let inv_tick = exact_inverse(tick_size);
        let inv_step = exact_inverse(step_size);

        Self {
            tick_size,
            step_size,
            min_price: if min_price > 0.0 { min_price } else { tick_size },
            max_price: if max_price > 0.0 { max_price } else { 10_000_000.0 },
            min_qty: if min_qty > 0.0 { min_qty } else { step_size },
            max_qty: if max_qty > 0.0 { max_qty } else { 10_000_000.0 },
            min_notional: if min_notional > 0.0 { min_notional } else { 5.0 },
            price_decimals,
            qty_decimals,
            inv_tick,
            inv_step,
        }
    }

    /// Construye a partir de la especificación canónica `SymbolSpec` de `quantum_arena`.
    pub fn from_symbol_spec(spec: &quantum_arena::symbol_registry::SymbolSpec) -> Self {
        Self::with_limits(
            spec.tick_size,
            spec.step_size,
            spec.tick_size,
            10_000_000.0,
            spec.min_qty,
            10_000_000.0,
            spec.min_notional,
        )
    }

    /// Valida que la configuración del filtro sea físicamente coherente.
    #[inline(always)]
    pub fn is_valid_config(&self) -> bool {
        self.tick_size.is_finite()
            && self.tick_size > 0.0
            && self.step_size.is_finite()
            && self.step_size > 0.0
            && self.min_price.is_finite()
            && self.min_price > 0.0
            && self.max_price.is_finite()
            && self.max_price >= self.min_price
            && self.min_qty.is_finite()
            && self.min_qty > 0.0
            && self.max_qty.is_finite()
            && self.max_qty >= self.min_qty
            && self.min_notional.is_finite()
            && self.min_notional >= 0.0
    }

    /// Redondea el precio al tick exacto de Binance.
    /// - `round_up == true`: Ceil al tick (para órdenes SELL límite o STOP LOSS de SHORT).
    /// - `round_up == false`: Floor al tick (para órdenes BUY límite o STOP LOSS de LONG).
    #[inline(always)]
    pub fn round_price(&self, raw_price: f64, round_up: bool) -> f64 {
        if !raw_price.is_finite() || raw_price <= 0.0 || !self.is_valid_config() {
            return 0.0;
        }
        let scaled = raw_price * self.inv_tick;
        let r = scaled.round();
        let ticks = if (scaled - r).abs() <= SNAP_REL_TOL * r.abs().max(1.0) {
            r
        } else if round_up {
            scaled.ceil()
        } else {
            scaled.floor()
        };
        (ticks / self.inv_tick).clamp(self.min_price, self.max_price)
    }

    /// Redondea la cantidad al paso de lote (`stepSize`) exacto de Binance.
    /// Siempre redondea hacia abajo (`floor`) para no violar el margen reservado por el riesgo.
    #[inline(always)]
    pub fn round_qty(&self, raw_qty: f64) -> f64 {
        if !raw_qty.is_finite() || raw_qty <= 0.0 || !self.is_valid_config() {
            return 0.0;
        }
        if raw_qty < self.min_qty {
            return 0.0;
        }
        let scaled = (raw_qty - self.min_qty) * self.inv_step;
        let r = scaled.round();
        let steps = if (scaled - r).abs() <= SNAP_REL_TOL * r.abs().max(1.0) {
            r
        } else {
            scaled.floor()
        };
        (self.min_qty + steps / self.inv_step).clamp(self.min_qty, self.max_qty)
    }

    /// Formatea el precio a una cadena ASCII en stack con exactamente `price_decimals` dígitos.
    /// Garantiza cero asignaciones en heap y cero residuos binarios (`.000000000000004`).
    #[inline(always)]
    pub fn format_price(&self, price: f64) -> StackAsciiString {
        if !price.is_finite() || price <= 0.0 {
            return StackAsciiString::empty();
        }
        let mut out = [0u8; 32];
        if self.price_decimals == 0 {
            let units = price.round() as u64;
            let mut itoa_buf = itoa::Buffer::new();
            let s = itoa_buf.format(units);
            let bytes = s.as_bytes();
            let len = bytes.len().min(32);
            out[..len].copy_from_slice(&bytes[..len]);
            StackAsciiString { bytes: out, len }
        } else {
            let multiplier = 10_u64.pow(self.price_decimals as u32);
            let total_units = (price * (multiplier as f64)).round() as u64;
            let int_part = total_units / multiplier;
            let frac_part = total_units % multiplier;
            let len = format_fixed_ascii(int_part, frac_part, self.price_decimals, &mut out);
            StackAsciiString { bytes: out, len }
        }
    }

    /// Formatea la cantidad a una cadena ASCII en stack con exactamente `qty_decimals` dígitos.
    /// Si `step_size >= 1.0` (ej. DOGEUSDT con step=1), formatea como entero sin punto decimal.
    #[inline(always)]
    pub fn format_qty(&self, qty: f64) -> StackAsciiString {
        if !qty.is_finite() || qty <= 0.0 {
            return StackAsciiString::empty();
        }
        let mut out = [0u8; 32];
        if self.qty_decimals == 0 {
            let units = qty.round() as u64;
            let mut itoa_buf = itoa::Buffer::new();
            let s = itoa_buf.format(units);
            let bytes = s.as_bytes();
            let len = bytes.len().min(32);
            out[..len].copy_from_slice(&bytes[..len]);
            StackAsciiString { bytes: out, len }
        } else {
            let multiplier = 10_u64.pow(self.qty_decimals as u32);
            let total_units = (qty * (multiplier as f64)).round() as u64;
            let int_part = total_units / multiplier;
            let frac_part = total_units % multiplier;
            let len = format_fixed_ascii(int_part, frac_part, self.qty_decimals, &mut out);
            StackAsciiString { bytes: out, len }
        }
    }

    /// Cuantiza y valida una orden completa en un solo paso $O(1)$ sin asignaciones.
    /// Retorna `Ok(QuantizedOrder)` si cumple todos los filtros de Binance (`PRICE_FILTER`,
    /// `LOT_SIZE`, `MIN_NOTIONAL`), o `Err(FilterRejectReason)` con la causa exacta.
    pub fn validate_and_quantize(
        &self,
        raw_price: f64,
        raw_qty: f64,
        is_sell: bool,
    ) -> Result<QuantizedOrder, FilterRejectReason> {
        if !self.is_valid_config() {
            return Err(FilterRejectReason::InvalidFilterConfig);
        }
        if !raw_price.is_finite() || !raw_qty.is_finite() {
            return Err(FilterRejectReason::NonFinite);
        }
        if raw_price <= 0.0 || raw_qty <= 0.0 {
            return Err(FilterRejectReason::ZeroOrNegative);
        }

        let rounded_price = self.round_price(raw_price, is_sell);
        if rounded_price < self.min_price {
            return Err(FilterRejectReason::PriceBelowMin {
                current: rounded_price,
                min: self.min_price,
            });
        }
        if rounded_price > self.max_price {
            return Err(FilterRejectReason::PriceAboveMax {
                current: rounded_price,
                max: self.max_price,
            });
        }

        let rounded_qty = self.round_qty(raw_qty);
        if rounded_qty < self.min_qty {
            return Err(FilterRejectReason::QtyBelowMin {
                current: rounded_qty,
                min: self.min_qty,
            });
        }
        if rounded_qty > self.max_qty {
            return Err(FilterRejectReason::QtyAboveMax {
                current: rounded_qty,
                max: self.max_qty,
            });
        }

        let notional = rounded_price * rounded_qty;
        if notional < self.min_notional {
            return Err(FilterRejectReason::NotionalBelowMin {
                current: notional,
                min: self.min_notional,
            });
        }

        let price_str = self.format_price(rounded_price);
        let qty_str = self.format_qty(rounded_qty);

        Ok(QuantizedOrder {
            price: rounded_price,
            qty: rounded_qty,
            notional,
            price_str,
            qty_str,
        })
    }
}

/// Inverso exacto ajustado al entero más cercano para potencias de diez y divisores comunes.
#[inline(always)]
fn exact_inverse(step: f64) -> f64 {
    if !step.is_finite() || step <= 0.0 {
        return 1.0;
    }
    let inv = 1.0 / step;
    let r = inv.round();
    if r >= 1.0 && (inv - r).abs() <= 1e-6 * r {
        r
    } else {
        inv
    }
}

/// Deduce el número de cifras decimales a partir del tamaño del paso / tick.
pub fn count_decimals(step: f64) -> u8 {
    if !step.is_finite() || step <= 0.0 || step >= 1.0 {
        return 0;
    }
    let log = -step.log10();
    let rounded = log.round();
    if (log - rounded).abs() < 1e-4 {
        rounded.clamp(0.0, 10.0) as u8
    } else {
        // Para pasos fraccionarios no potencias de 10 (ej. 0.05, 0.25, 0.2, 0.02)
        let mut k = 1u8;
        let mut mult = 10.0;
        while k <= 8 {
            let val = step * mult;
            if (val - val.round()).abs() < 1e-5 {
                return k;
            }
            mult *= 10.0;
            k += 1;
        }
        8
    }
}

/// Serializa entero y fracción a ASCII en stack con relleno exacto de ceros a la izquierda.
#[inline(always)]
fn format_fixed_ascii(
    int_part: u64,
    frac_part: u64,
    decimals: u8,
    out: &mut [u8; 32],
) -> usize {
    let mut cursor = 0;
    let mut itoa_buf = itoa::Buffer::new();
    let int_str = itoa_buf.format(int_part);
    let int_bytes = int_str.as_bytes();
    let int_len = int_bytes.len().min(32);
    out[cursor..cursor + int_len].copy_from_slice(&int_bytes[..int_len]);
    cursor += int_len;

    if decimals > 0 && cursor < 31 {
        out[cursor] = b'.';
        cursor += 1;

        let frac_str = itoa_buf.format(frac_part);
        let frac_bytes = frac_str.as_bytes();
        let leading_zeros = (decimals as usize).saturating_sub(frac_bytes.len());

        for _ in 0..leading_zeros {
            if cursor >= 32 {
                break;
            }
            out[cursor] = b'0';
            cursor += 1;
        }

        let copy_len = frac_bytes.len().min(32 - cursor);
        out[cursor..cursor + copy_len].copy_from_slice(&frac_bytes[..copy_len]);
        cursor += copy_len;
    }
    cursor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_decimals() {
        assert_eq!(count_decimals(1.0), 0);
        assert_eq!(count_decimals(10.0), 0);
        assert_eq!(count_decimals(0.1), 1);
        assert_eq!(count_decimals(0.01), 2);
        assert_eq!(count_decimals(0.001), 3);
        assert_eq!(count_decimals(0.0001), 4);
        assert_eq!(count_decimals(0.00001), 5);
        assert_eq!(count_decimals(0.05), 2);
        assert_eq!(count_decimals(0.25), 2);
        assert_eq!(count_decimals(0.2), 1);
    }

    #[test]
    fn test_format_fixed_ascii_btc() {
        let filter = BinanceFixedFilter::new(0.1, 0.001, 5.0);
        assert_eq!(filter.price_decimals, 1);
        assert_eq!(filter.qty_decimals, 3);

        let p_str = filter.format_price(65432.1);
        assert_eq!(p_str.as_str(), "65432.1");

        let q_str = filter.format_qty(0.007);
        assert_eq!(q_str.as_str(), "0.007");

        let q_str2 = filter.format_qty(0.001);
        assert_eq!(q_str2.as_str(), "0.001");
    }

    #[test]
    fn test_format_fixed_ascii_doge() {
        let filter = BinanceFixedFilter::new(0.0001, 1.0, 5.0);
        assert_eq!(filter.price_decimals, 4);
        assert_eq!(filter.qty_decimals, 0);

        let p_str = filter.format_price(0.1234);
        assert_eq!(p_str.as_str(), "0.1234");

        let q_str = filter.format_qty(150.0);
        assert_eq!(q_str.as_str(), "150");
    }

    #[test]
    fn test_round_and_validate_btcusdt() {
        let filter = BinanceFixedFilter::new(0.1, 0.001, 5.0);
        // BUY: floor de precio
        let res = filter.validate_and_quantize(65432.18, 0.0054, false).unwrap();
        assert_eq!(res.price, 65432.1);
        assert_eq!(res.qty, 0.005);
        assert_eq!(res.price_str.as_str(), "65432.1");
        assert_eq!(res.qty_str.as_str(), "0.005");
        assert!(res.notional >= 5.0);

        // SELL: ceil de precio
        let res_sell = filter.validate_and_quantize(65432.11, 0.0054, true).unwrap();
        assert_eq!(res_sell.price, 65432.2);
        assert_eq!(res_sell.qty, 0.005);
        assert_eq!(res_sell.price_str.as_str(), "65432.2");
        assert_eq!(res_sell.qty_str.as_str(), "0.005");
    }

    #[test]
    fn test_min_notional_rejection() {
        let filter = BinanceFixedFilter::new(0.1, 0.001, 5.0);
        // notional = 100.0 * 0.001 = 0.1 < 5.0
        let err = filter.validate_and_quantize(100.0, 0.001, false).unwrap_err();
        match err {
            FilterRejectReason::NotionalBelowMin { current, min } => {
                assert!((current - 0.1).abs() < 1e-6);
                assert!((min - 5.0).abs() < 1e-6);
            }
            other => panic!("Esperado NotionalBelowMin, recibido: {:?}", other),
        }
    }

    #[test]
    fn test_zero_alloc_stack_string_traits() {
        let filter = BinanceFixedFilter::new(0.01, 0.01, 5.0);
        let s = filter.format_price(123.45);
        assert_eq!(s, "123.45");
        assert_eq!(&*s, "123.45");
        assert_eq!(s.len(), 6);
        assert!(!s.is_empty());
    }
}
