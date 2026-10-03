//! CL-40 — quién sigue al almacén de genomas.
//!
//! `refresh_models` aplica `config_dir/genomes/<env>/active.json` cuando su
//! generación supera la aplicada. Eso es correcto para el núcleo que opera:
//! la generación sancionada por el demonio o la cosecha debe llegar a la
//! orden. Pero en el host (`TG_GENOME_ENV` definida) también corren núcleos
//! que EVALÚAN un genoma concreto —el examen walk-forward del demonio, los
//! universos del bosque sombra, el SA de la evolución— y en los backtests de
//! evolución los mutantes. Todos nacían con la generación aplicada en 0 y,
//! en su primer evento, sustituían el candidato por el activo: el examen
//! juzgaba al genoma activo frente a sí mismo. `train_forest` y
//! `audit_forensic_backtest` ya lo sorteaban fijando la generación aplicada
//! en `u64::MAX`; ahora es la política por contexto.
use crate::outcome_context::OutcomeContext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecargaGenoma {
    /// Evalúa el genoma que se le aplicó al arena; nunca lo reemplaza.
    Fija,
    /// Adopta cada generación nueva que sancione el almacén del entorno.
    DesdeAlmacen,
}

impl RecargaGenoma {
    /// Sólo el núcleo conectado a la ejecución sigue al almacén.
    pub const fn para(contexto: OutcomeContext) -> Self {
        match contexto {
            OutcomeContext::ExchangeLocalEstimate => Self::DesdeAlmacen,
            OutcomeContext::IsolatedSimulation => Self::Fija,
        }
    }

    /// `GOD_NO_HOT_RELOAD` sigue apagando la recarga también del núcleo vivo.
    pub fn sigue_al_almacen(self, apagado_global: bool) -> bool {
        self == Self::DesdeAlmacen && !apagado_global
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solo_el_nucleo_de_ejecucion_sigue_al_almacen() {
        assert!(RecargaGenoma::para(OutcomeContext::ExchangeLocalEstimate).sigue_al_almacen(false));
        assert!(!RecargaGenoma::para(OutcomeContext::ExchangeLocalEstimate).sigue_al_almacen(true));
        assert!(!RecargaGenoma::para(OutcomeContext::IsolatedSimulation).sigue_al_almacen(false));
        assert!(!RecargaGenoma::para(OutcomeContext::IsolatedSimulation).sigue_al_almacen(true));
    }
}
