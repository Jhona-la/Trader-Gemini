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

    /// El estrangulador del aviso a `ahora_ms`; `None` con moneda fuera de rango.
    fn admitir(&mut self, aviso: Aviso, ahora_ms: u64) -> Option<u64> {
        match aviso {
            Aviso::Latencia => self.latencia.admitir(ahora_ms),
            Aviso::VetoMl => self.veto_ml.admitir(ahora_ms),
            Aviso::LatidoMl(coin_id) => self.latido_ml.get_mut(coin_id)?.admitir(ahora_ms),
            Aviso::ModeloDegenerado(coin_id) => self.modelo_degenerado.get_mut(coin_id)?.admitir(ahora_ms),
        }
    }

    /// `Some(suprimidos)` si toca avisar de la latencia.
    pub fn latencia(&mut self) -> Option<u64> {
        self.admitir(Aviso::Latencia, self.ahora_ms())
    }

    /// `Some(suprimidos)` si toca avisar de los vetos del ensamble.
    pub fn veto_ml(&mut self) -> Option<u64> {
        self.admitir(Aviso::VetoMl, self.ahora_ms())
    }

    /// `Some(suprimidos)` si toca el latido ML de `coin_id`; `None` fuera de rango.
    pub fn latido_ml(&mut self, coin_id: usize) -> Option<u64> {
        self.admitir(Aviso::LatidoMl(coin_id), self.ahora_ms())
    }

    /// `Some(suprimidos)` si toca avisar del modelo degenerado de `coin_id`.
    pub fn modelo_degenerado(&mut self, coin_id: usize) -> Option<u64> {
        self.admitir(Aviso::ModeloDegenerado(coin_id), self.ahora_ms())
    }
}

#[derive(Debug, Clone, Copy)]
enum Aviso {
    Latencia,
    VetoMl,
    LatidoMl(usize),
    ModeloDegenerado(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Esperado: con el reloj fijado, un millón de condiciones dentro de una
    /// ventana dan un solo aviso de cada tipo; al abrirse la ventana el
    /// siguiente trae el conteo de los suprimidos; las monedas no se pisan y
    /// una moneda fuera de rango no avisa ni falla. Antes, ML-GATE escribía
    /// uno de cada 50 vetos: 20 000 líneas.
    #[test]
    fn cl142b_un_aviso_por_ventana_sea_cual_sea_la_tasa() {
        let mut a = AvisosDelNucleo::new(3);
        let tipos = [Aviso::Latencia, Aviso::VetoMl, Aviso::LatidoMl(0), Aviso::ModeloDegenerado(0)];
        let mut avisos = [0u32; 4];
        for i in 0..1_000_000u64 {
            let t = i % VENTANA_LATENCIA_MS;
            for (k, tipo) in tipos.into_iter().enumerate() {
                avisos[k] += a.admitir(tipo, t).is_some() as u32;
            }
        }
        assert_eq!(avisos, [1, 1, 1, 1]);
        assert_eq!(a.admitir(Aviso::Latencia, VENTANA_LATENCIA_MS), Some(999_999));
        assert_eq!(a.admitir(Aviso::VetoMl, VENTANA_LATENCIA_MS), None, "su ventana es de 60 s");
        assert_eq!(a.admitir(Aviso::VetoMl, VENTANA_MS), Some(1_000_000));
        assert!(a.admitir(Aviso::LatidoMl(1), 0).is_some(), "otra moneda tiene su propia ventana");
        assert!(a.admitir(Aviso::ModeloDegenerado(2), 0).is_some());
        assert_eq!(a.admitir(Aviso::LatidoMl(3), 0), None);
        assert_eq!(a.admitir(Aviso::ModeloDegenerado(usize::MAX), 0), None);
    }
}
