//! CL-142b (Línea C, C-F6) — AVISOS DEL NÚCLEO CON TOPE POR TIEMPO.
//!
//! QUÉ ESTABA MAL: cuatro avisos del núcleo vivo se contaban en eventos:
//! LATENCY PANIC cada 1 000 ticks, ML-GATE uno de cada 50 vetos, el latido ML
//! en 26 de cada 5 000 ticks y ML-DIAG de un modelo degenerado cada 100
//! ticks. Su volumen crecía con la tasa del feed: a 200–800 ev/s, ML-GATE
//! sólo eran 38–152 MB/día.
//!
//! QUÉ GARANTIZA: como mucho un aviso por ventana y por tipo (por moneda en
//! el latido y en ML-DIAG), con el conteo de los suprimidos. El reloj es el
//! monótono del proceso y no el del evento: un backtest reproduce días en
//! minutos y con el reloj del evento escribiría un aviso por cada minuto de
//! la cinta.

use telemetry_server::estrangulador::Estrangulador;

/// Ventana del aviso de latencia, la misma que la del host (CL-139).
pub const VENTANA_LATENCIA_MS: u64 = 10_000;
/// Ventana del resto de avisos del núcleo.
pub const VENTANA_MS: u64 = 60_000;

#[derive(Debug, Clone)]
pub struct AvisosDelNucleo {
    inicio: std::time::Instant,
    latencia: Estrangulador,
    veto_ml: Estrangulador,
    latido_ml: Vec<Estrangulador>,
    modelo_degenerado: Vec<Estrangulador>,
}

impl AvisosDelNucleo {
    pub fn new(monedas: usize) -> Self {
        Self {
            inicio: std::time::Instant::now(),
            latencia: Estrangulador::new(VENTANA_LATENCIA_MS),
            veto_ml: Estrangulador::new(VENTANA_MS),
            latido_ml: vec![Estrangulador::new(VENTANA_MS); monedas],
            modelo_degenerado: vec![Estrangulador::new(VENTANA_MS); monedas],
        }
    }

    fn ahora_ms(&self) -> u64 {
        u64::try_from(self.inicio.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// `Some(suprimidos)` si toca avisar de la latencia.
    pub fn latencia(&mut self) -> Option<u64> {
        let t = self.ahora_ms();
        self.latencia.admitir(t)
    }

    /// `Some(suprimidos)` si toca avisar de los vetos del ensamble.
    pub fn veto_ml(&mut self) -> Option<u64> {
        let t = self.ahora_ms();
        self.veto_ml.admitir(t)
    }

    /// `Some(suprimidos)` si toca el latido ML de `coin_id`; `None` fuera de rango.
    pub fn latido_ml(&mut self, coin_id: usize) -> Option<u64> {
        let t = self.ahora_ms();
        self.latido_ml.get_mut(coin_id)?.admitir(t)
    }

    /// `Some(suprimidos)` si toca avisar del modelo degenerado de `coin_id`.
    pub fn modelo_degenerado(&mut self, coin_id: usize) -> Option<u64> {
        let t = self.ahora_ms();
        self.modelo_degenerado.get_mut(coin_id)?.admitir(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Esperado: un millón de condiciones seguidas (mucho menos de una
    /// ventana) dan un solo aviso de cada tipo; las monedas no se pisan y una
    /// moneda fuera de rango no avisa ni falla. Antes, ML-GATE escribía uno
    /// de cada 50 vetos: 20 000 líneas.
    #[test]
    fn cl142b_un_aviso_por_ventana_sea_cual_sea_la_tasa() {
        let mut a = AvisosDelNucleo::new(3);
        let mut avisos = [0u32; 4];
        for _ in 0..1_000_000 {
            avisos[0] += a.latencia().is_some() as u32;
            avisos[1] += a.veto_ml().is_some() as u32;
            avisos[2] += a.latido_ml(0).is_some() as u32;
            avisos[3] += a.modelo_degenerado(0).is_some() as u32;
        }
        assert_eq!(avisos, [1, 1, 1, 1]);
        assert!(a.latido_ml(1).is_some(), "otra moneda tiene su propia ventana");
        assert!(a.modelo_degenerado(2).is_some());
        assert_eq!(a.latido_ml(3), None);
        assert_eq!(a.modelo_degenerado(usize::MAX), None);
    }
}
