//! CL-139 (Línea C, C-F6) — UN AVISO POR VENTANA DE TIEMPO.
//!
//! QUÉ ESTABA MAL: varios avisos del host se escribían por EVENTO mientras
//! duraba una condición (latencia por encima del umbral, liquidaciones
//! rechazadas) o con una cadencia contada en mensajes (el espectro cada 100
//! mensajes). Su volumen escala con la tasa del feed: a 500 ev/s, el aviso
//! de latencia son ≈ 3,7 GB/día y el espectro hasta ≈ 1,9 GB/día, en un log
//! que nada rota (`logs/demo_vNN.log`, stdout redirigido).
//!
//! QUÉ GARANTIZA: como mucho un aviso por `ventana_ms` del reloj que se le
//! pase, sea cual sea la tasa de entrada. Los avisos suprimidos no se pierden
//! de vista: el siguiente aviso admitido dice cuántos hubo. El reloj debe ser
//! monótono; uno que retrocede no abre la ventana antes de tiempo.

/// Admite un aviso por ventana y cuenta los que suprime entre dos avisos.
#[derive(Debug, Clone)]
pub struct Estrangulador {
    ventana_ms: u64,
    ultimo_ms: Option<u64>,
    suprimidos: u64,
}

impl Estrangulador {
    pub const fn new(ventana_ms: u64) -> Self {
        Self { ventana_ms, ultimo_ms: None, suprimidos: 0 }
    }

    /// Devuelve `Some(n)` si toca avisar en `ahora_ms`, con `n` avisos
    /// suprimidos desde el anterior; `None` si se suprime este.
    pub fn admitir(&mut self, ahora_ms: u64) -> Option<u64> {
        let toca = match self.ultimo_ms {
            None => true,
            Some(t) => ahora_ms.saturating_sub(t) >= self.ventana_ms,
        };
        if toca {
            self.ultimo_ms = Some(ahora_ms);
            Some(std::mem::take(&mut self.suprimidos))
        } else {
            self.suprimidos = self.suprimidos.saturating_add(1);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Esperado: con 1 000 avisos por segundo durante una hora y una ventana
    /// de 10 s salen 360 avisos y ninguno suprimido queda sin contar. Antes
    /// (un aviso por evento) eran 3,6 millones.
    #[test]
    fn cl139_la_tasa_de_avisos_no_depende_de_la_del_feed() {
        let mut e = Estrangulador::new(10_000);
        let (mut avisos, mut contados) = (0u64, 0u64);
        for i in 0..3_600_000u64 {
            if let Some(n) = e.admitir(i) {
                avisos += 1;
                contados += n + 1;
            }
        }
        assert_eq!(avisos, 360);
        assert_eq!(contados + e.suprimidos, 3_600_000, "cada aviso queda contado una vez");
    }

    /// Esperado: el primer aviso pasa siempre y un reloj que retrocede no
    /// reabre la ventana.
    #[test]
    fn cl139_primer_aviso_y_reloj_que_retrocede() {
        let mut e = Estrangulador::new(60_000);
        assert_eq!(e.admitir(100_000), Some(0));
        assert_eq!(e.admitir(10), None, "un reloj que retrocede no abre la ventana");
        assert_eq!(e.admitir(159_999), None);
        assert_eq!(e.admitir(160_000), Some(2));
    }
}
