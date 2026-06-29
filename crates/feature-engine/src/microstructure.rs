/// Microestructura del Mercado (Order Book)
/// Axioma II: O(1) puro y #[inline(always)]

/// Calcula el Order Book Imbalance (OBI).
/// Retorna un valor entre -1.0 y 1.0.
/// Un valor cercano a 1.0 indica fuerte presión de compra (BID domina).
/// Un valor cercano a -1.0 indica fuerte presión de venta (ASK domina).
#[inline(always)]
pub fn order_book_imbalance(bid_vol: f64, ask_vol: f64) -> f64 {
    let total_vol = bid_vol + ask_vol;
    if total_vol == 0.0 {
        0.0
    } else {
        (bid_vol - ask_vol) / total_vol
    }
}

/// Aceleración de Liquidez (Derivada del OBI)
/// Calcula la tasa de cambio del OBI entre el tick T y el tick T-1.
#[inline(always)]
pub fn obi_acceleration(current_obi: f64, previous_obi: f64) -> f64 {
    current_obi - previous_obi
}
