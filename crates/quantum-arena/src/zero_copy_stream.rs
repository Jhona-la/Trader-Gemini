use std::f64;

/// ⚡ TRANSMISOR BINARIO EN MEMORIA MAPEADA DE CERO COPIAS (ZERO-COPY BINARY STREAMING KERNEL)
/// Permite leer eventos de mercado tick por tick directamente de punteros mmap.
/// Evita la instanciación en Heap y reduce el tiempo de des-serialización a sub-nanosegundos.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ZeroCopyBinTick {
    pub timestamp: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_qty: f64,
    pub ask_qty: f64,
}

impl ZeroCopyBinTick {
    /// Carga directamente un tick desde un slice de bytes en O(1)
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        &*(bytes.as_ptr() as *const Self)
    }

    #[inline(always)]
    pub fn mid_price(&self) -> f64 {
        (self.bid_price + self.ask_price) * 0.5
    }

    #[inline(always)]
    pub fn spread_bps(&self) -> f64 {
        if self.ask_price > 0.0 {
            ((self.ask_price - self.bid_price) / self.ask_price) * 10000.0
        } else {
            0.0
        }
    }
}
