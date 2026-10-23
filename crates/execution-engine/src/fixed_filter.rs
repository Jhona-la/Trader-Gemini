//! # MÓDULO DE PUNTO FIJO Y FILTROS EXACTOS DE BINANCE (Fase 2 §4.2 Átomo 2.4)
//!
//! **Mapeo de Excelencia Cuántica y Modo Profesor:**
//! - **QUÉ:** Cuantizador determinista y serializador de punto fijo en stack (zero-alloc, O(1))
//!   para los filtros de Binance (`PRICE_FILTER`, `LOT_SIZE`, `MIN_NOTIONAL`).
//! - **POR QUÉ:** La aritmética IEEE-754 de coma flotante `f64` produce residuos espurios
//!   (ej. `0.007000000000000001` o `62450.100000000006`) que causan el error `-1111`
//!   (`Precision is over the maximum defined for this asset`) o rechazos de `LOT_SIZE`/`PRICE_FILTER`.
//! - **PARA QUÉ:** Validar localmente la red decimal y conservar el límite de cantidad aprobado.
//!   Este módulo no acredita aceptación del exchange, latencia ni integración del ejecutor vivo.
//! - **CÓMO:** Proyección euclidiana discreta sobre la red de ticks $\mathbb{Z} \cdot \Delta_{\text{tick}}$
//!   y pasos de lote $\mathbb{Z} \cdot \Delta_{\text{step}}$, formateando los dígitos ASCII directamente
//!   en buffers de memoria contigua en la pila con `itoa`.
//! - **CUÁNDO:** En cada generación de payload de orden (`build_payload`), stop de protección y take-profit.
//! - **DÓNDE:** `crates/execution-engine/src/fixed_filter.rs`.
//! - **QUIÉN:** AGY (Lead Quant Sr. & Consejo de 10 Roles Senior).

use std::fmt;
use std::ops::Deref;

// La API histórica recibe f64: los decimales originales ya pueden haberse perdido.
// Se soportan hasta diez decimales y unidades enteras representables exactamente en f64.
// No se interpreta una tolerancia relativa creciente como autorización para aumentar cantidad.
// La dirección se aplica al valor binario exacto: por ejemplo raw=0.3 (f64) queda
// debajo del decimal 0.3; con step=0.1, floor emite 0.2. Recuperar la intención
// decimal original exige una futura API que reciba los decimales sin convertirlos a f64.
const MAX_DECIMALS: u8 = 10;
const MAX_EXACT_UNITS: u64 = (1_u64 << 53) - 1;

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
    /// El valor excede la precisión decimal/entera soportada por esta API f64.
    Unrepresentable,
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
            Self::Unrepresentable => write!(f, "Value outside supported decimal representation"),
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
    /// Constructor de compatibilidad para escenarios locales con límites de diez millones.
    /// Los límites reales del exchange deben suministrarse mediante `with_limits`.
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
        let price_decimals = count_decimals(tick_size).max(count_decimals(min_price));
        let qty_decimals = count_decimals(step_size).max(count_decimals(min_qty));
        let inv_tick = exact_inverse(tick_size);
        let inv_step = exact_inverse(step_size);

        Self {
            tick_size,
            step_size,
            min_price,
            max_price,
            min_qty,
            max_qty,
            min_notional,
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

    /// Valida la configuración sin reparar silenciosamente datos del exchange.
    /// Las reglas PRICE_FILTER deshabilitadas con cero no están soportadas por esta API.
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
            && self.inv_tick.is_finite()
            && self.inv_tick == exact_inverse(self.tick_size)
            && self.inv_step.is_finite()
            && self.inv_step == exact_inverse(self.step_size)
            && grid_units(self.tick_size, self.min_price, self.price_decimals).is_some()
            && grid_units(self.step_size, self.min_qty, self.qty_decimals).is_some()
    }

    /// Redondea el precio al tick exacto de Binance.
    /// - `round_up == true`: Ceil al tick (para órdenes SELL límite o STOP LOSS de SHORT).
    /// - `round_up == false`: Floor al tick (para órdenes BUY límite o STOP LOSS de LONG).
    #[inline(always)]
    pub fn round_price(&self, raw_price: f64, round_up: bool) -> f64 {
        if !raw_price.is_finite() || raw_price <= 0.0 || !self.is_valid_config() {
            return 0.0;
        }
        match project_decimal_grid(
            raw_price,
            self.tick_size,
            self.min_price,
            self.price_decimals,
            round_up,
        ) {
            Some((value, _)) => value,
            None => 0.0, // Centinela de la API histórica; validate_and_quantize devuelve la causa.
        }
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
        match project_decimal_grid(
            raw_qty,
            self.step_size,
            self.min_qty,
            self.qty_decimals,
            false,
        ) {
            Some((value, _)) => value,
            None => 0.0,
        }
    }

    /// Formatea el precio a una cadena ASCII en stack con exactamente `price_decimals` dígitos.
    /// Garantiza cero asignaciones en heap y cero residuos binarios (`.000000000000004`).
    #[inline(always)]
    pub fn format_price(&self, price: f64) -> StackAsciiString {
        format_decimal(price, self.price_decimals)
    }

    /// Formatea la cantidad a una cadena ASCII en stack con exactamente `qty_decimals` dígitos.
    /// Un paso entero se formatea sin punto; un paso fraccionario (también >= 1) conserva decimales.
    #[inline(always)]
    pub fn format_qty(&self, qty: f64) -> StackAsciiString {
        format_decimal(qty, self.qty_decimals)
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
        // Nunca convertir una orden fuera de límites en otra orden mediante clamp.
        if raw_price < self.min_price {
            return Err(FilterRejectReason::PriceBelowMin {
                current: raw_price,
                min: self.min_price,
            });
        }
        if raw_price > self.max_price {
            return Err(FilterRejectReason::PriceAboveMax {
                current: raw_price,
                max: self.max_price,
            });
        }
        if raw_qty < self.min_qty {
            return Err(FilterRejectReason::QtyBelowMin {
                current: 0.0,
                min: self.min_qty,
            });
        }
        if raw_qty > self.max_qty {
            return Err(FilterRejectReason::QtyAboveMax {
                current: raw_qty,
                max: self.max_qty,
            });
        }

        let (rounded_price, price_units) = project_decimal_grid(
            raw_price,
            self.tick_size,
            self.min_price,
            self.price_decimals,
            is_sell,
        )
        .ok_or(FilterRejectReason::Unrepresentable)?;
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

        let (rounded_qty, qty_units) = project_decimal_grid(
            raw_qty,
            self.step_size,
            self.min_qty,
            self.qty_decimals,
            false,
        )
        .ok_or(FilterRejectReason::Unrepresentable)?;
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
        if !notional.is_finite() {
            return Err(FilterRejectReason::NonFinite);
        }
        if notional < self.min_notional {
            return Err(FilterRejectReason::NotionalBelowMin {
                current: notional,
                min: self.min_notional,
            });
        }

        let price_str = format_decimal_units(price_units, self.price_decimals);
        let qty_str = format_decimal_units(qty_units, self.qty_decimals);
        if price_str.is_empty() || qty_str.is_empty() {
            return Err(FilterRejectReason::Unrepresentable);
        }
        // Comparar el valor que realmente se serializa, no sólo el float intermedio.
        let wire_price = price_str
            .as_str()
            .parse::<f64>()
            .map_err(|_| FilterRejectReason::Unrepresentable)?;
        let wire_qty = qty_str
            .as_str()
            .parse::<f64>()
            .map_err(|_| FilterRejectReason::Unrepresentable)?;
        if wire_price != rounded_price || wire_qty != rounded_qty || wire_qty > raw_qty {
            return Err(FilterRejectReason::Unrepresentable);
        }

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
        return f64::NAN;
    }
    1.0 / step
}

/// Deduce de forma estricta el número de cifras decimales a partir del tamaño del paso / tick.
/// Devuelve None si el paso no es finito, es menor o igual a cero, o no tiene representación exacta <= MAX_DECIMALS.
pub fn try_count_decimals(step: f64) -> Option<u8> {
    if step.is_finite() && step > 0.0 {
        for decimals in 0..=MAX_DECIMALS {
            if let Some(units) = decimal_units(step, decimals) {
                if units > 0 {
                    return Some(decimals);
                }
            }
        }
    }
    None
}

/// Deduce el número de cifras decimales a partir del tamaño del paso / tick.
/// Devuelve cero para una precisión no soportada o pasos enteros; is_valid_config la rechaza si excede límites.
pub fn count_decimals(step: f64) -> u8 {
    try_count_decimals(step).unwrap_or(0)
}

fn decimal_units(value: f64, decimals: u8) -> Option<u64> {
    if !value.is_finite() || value <= 0.0 || decimals > MAX_DECIMALS {
        return None;
    }
    let multiplier = 10_u64.pow(u32::from(decimals)) as f64;
    let scaled = value * multiplier;
    let rounded = scaled.round();
    if !scaled.is_finite()
        || rounded < 1.0
        || rounded > MAX_EXACT_UNITS as f64
        || rounded / multiplier != value
    {
        return None;
    }
    Some(rounded as u64)
}

fn grid_units(step: f64, base: f64, decimals: u8) -> Option<(u64, u64)> {
    Some((
        decimal_units(step, decimals)?,
        decimal_units(base, decimals)?,
    ))
}

/// Obtiene floor/ceil(raw * 10^decimals) sin redondear la multiplicación a f64.
/// El significando tiene 53 bits y 10^10 necesita 34: el producto cabe en u128.
fn raw_decimal_bound(raw: f64, decimals: u8, up: bool) -> Option<u64> {
    if !raw.is_finite() || raw <= 0.0 || decimals > MAX_DECIMALS {
        return None;
    }
    let bits = raw.to_bits();
    let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if encoded_exponent == 0 {
        (u128::from(fraction), -1074)
    } else {
        (
            u128::from(fraction | (1_u64 << 52)),
            encoded_exponent - 1023 - 52,
        )
    };
    let numerator = significand.checked_mul(u128::from(10_u64.pow(u32::from(decimals))))?;
    let bound = if exponent >= 0 {
        let shift = exponent as u32;
        if shift >= 128 {
            return None;
        }
        numerator.checked_mul(1_u128 << shift)?
    } else {
        let shift = (-exponent) as u32;
        if shift >= 128 {
            u128::from(up && numerator > 0)
        } else {
            let quotient = numerator >> shift;
            let remainder = numerator & ((1_u128 << shift) - 1);
            quotient.checked_add(u128::from(up && remainder != 0))?
        }
    };
    if bound > u128::from(MAX_EXACT_UNITS) {
        None
    } else {
        Some(bound as u64)
    }
}

/// Proyecta en base + n*step mediante divisiones enteras sobre el límite exacto
/// del f64 recibido. El decimal serializado, además de su parseo f64, respeta raw.
fn project_decimal_grid(
    raw: f64,
    step: f64,
    base: f64,
    decimals: u8,
    up: bool,
) -> Option<(f64, u64)> {
    let (step_units, base_units) = grid_units(step, base, decimals)?;
    if raw < base {
        return None;
    }
    let bound = raw_decimal_bound(raw, decimals, up)?;
    let distance = bound.checked_sub(base_units)?;
    let steps = if up {
        distance.checked_add(step_units - 1)? / step_units
    } else {
        distance / step_units
    };
    let units = base_units.checked_add(steps.checked_mul(step_units)?)?;
    if units > MAX_EXACT_UNITS {
        return None;
    }
    let multiplier = 10_u64.pow(u32::from(decimals)) as f64;
    let projected = units as f64 / multiplier;
    if projected.is_finite()
        && projected > 0.0
        && (if up {
            projected >= raw
        } else {
            projected <= raw
        })
    {
        Some((projected, units))
    } else {
        None
    }
}

pub(crate) fn format_decimal(value: f64, decimals: u8) -> StackAsciiString {
    if !value.is_finite() || value <= 0.0 || decimals > MAX_DECIMALS {
        return StackAsciiString::empty();
    }
    // El formateador público conserva su redondeo histórico de floats arbitrarios.
    // La validación de órdenes serializa directamente las unidades del grid, sin
    // pasar por este redondeo de presentación.
    let scaled = value * 10_u64.pow(u32::from(decimals)) as f64;
    let rounded = scaled.round();
    if !scaled.is_finite() || rounded < 1.0 || rounded > MAX_EXACT_UNITS as f64 {
        return StackAsciiString::empty();
    }
    format_decimal_units(rounded as u64, decimals)
}

fn format_decimal_units(total_units: u64, decimals: u8) -> StackAsciiString {
    if decimals > MAX_DECIMALS || total_units == 0 || total_units > MAX_EXACT_UNITS {
        return StackAsciiString::empty();
    }
    let multiplier = 10_u64.pow(u32::from(decimals));
    let mut out = [0u8; 32];
    let len = format_fixed_ascii(
        total_units / multiplier,
        total_units % multiplier,
        decimals,
        &mut out,
    );
    StackAsciiString { bytes: out, len }
}

/// Serializa entero y fracción a ASCII en stack con relleno exacto de ceros a la izquierda.
#[inline(always)]
fn format_fixed_ascii(int_part: u64, frac_part: u64, decimals: u8, out: &mut [u8; 32]) -> usize {
    let mut cursor = 0;
    let mut itoa_buf = itoa::Buffer::new();
    let int_str = itoa_buf.format(int_part);
    let int_bytes = int_str.as_bytes();
    let int_len = int_bytes.len().min(32);
    out[cursor..cursor + int_len].copy_from_slice(&int_bytes[..int_len]);
    cursor += int_len;

    if decimals > 0 && cursor < 31 {
        let dot_pos = cursor;
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

        // Eliminar ceros no significativos a la derecha para no alterar representaciones mínimas
        // y evitar discrepancias de longitud o assert en contratos de admisión (ej: "0.4" vs "0.400").
        while cursor > dot_pos + 1 && out[cursor - 1] == b'0' {
            cursor -= 1;
        }
        if cursor == dot_pos + 1 {
            // Toda la fracción era ceros; quitamos también el punto
            cursor = dot_pos;
        }
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
        let res = filter
            .validate_and_quantize(65432.18, 0.0054, false)
            .unwrap();
        assert_eq!(res.price, 65432.1);
        assert_eq!(res.qty, 0.005);
        assert_eq!(res.price_str.as_str(), "65432.1");
        assert_eq!(res.qty_str.as_str(), "0.005");
        assert!(res.notional >= 5.0);

        // SELL: ceil de precio
        let res_sell = filter
            .validate_and_quantize(65432.11, 0.0054, true)
            .unwrap();
        assert_eq!(res_sell.price, 65432.2);
        assert_eq!(res_sell.qty, 0.005);
        assert_eq!(res_sell.price_str.as_str(), "65432.2");
        assert_eq!(res_sell.qty_str.as_str(), "0.005");
    }

    #[test]
    fn test_min_notional_rejection() {
        let filter = BinanceFixedFilter::new(0.1, 0.001, 5.0);
        // notional = 100.0 * 0.001 = 0.1 < 5.0
        let err = filter
            .validate_and_quantize(100.0, 0.001, false)
            .unwrap_err();
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
