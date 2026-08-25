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
    /// Carga directamente un tick desde un slice de bytes en O(1).
    /// F5.4 — SEGURA: el cast `packed` sin verificación era UB (lecturas OOB y
    /// referencias f64 desalineadas). Ahora valida longitud y devuelve None.
    /// `from_bytes_unchecked` se mantiene solo para buffers ya verificados
    /// por el lector mmap (longución múltiplo exacto del tamaño del struct).
    #[inline(always)]
    pub fn from_bytes(bytes: &[u8]) -> Option<&Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        // Safety: longitud verificada; Self es #[repr(C, packed)] sin padding,
        // cualquier alineación de u8 es válida para leerlo por valor copiado.
        Some(unsafe { &*(bytes.as_ptr() as *const Self) })
    }

    /// SOLO para buffers verificados (len % size_of == 0). Ver from_bytes.
    #[inline(always)]
    pub unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self {
        &*(bytes.as_ptr() as *const Self)
    }

    #[inline(always)]
    pub fn mid_price(&self) -> f64 {
        let bid = self.bid_price;
        let ask = self.ask_price;
        (bid + ask) * 0.5
    }

    #[inline(always)]
    pub fn spread_bps(&self) -> f64 {
        let bid = self.bid_price;
        let ask = self.ask_price;
        if ask > 0.0 {
            ((ask - bid) / ask) * 10000.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_copy_bin_tick_parsing_and_metrics() {
        let tick = ZeroCopyBinTick {
            timestamp: 1672531199000,
            bid_price: 50000.0,
            ask_price: 50005.0,
            bid_qty: 1.5,
            ask_qty: 2.0,
        };
        let slice: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &tick as *const _ as *const u8,
                std::mem::size_of::<ZeroCopyBinTick>(),
            )
        };

        let parsed = ZeroCopyBinTick::from_bytes(slice).unwrap();
        let ts = parsed.timestamp;
        assert_eq!(ts, 1672531199000);
        assert_eq!(parsed.mid_price(), 50002.5);
        assert!((parsed.spread_bps() - 1.0).abs() < 1e-3);

        assert!(ZeroCopyBinTick::from_bytes(&[0u8; 10]).is_none());
    }
}
