//! I-9 (Línea I, fila N15b del censo C-F7b): la pierna que sobrevive a la
//! purga de un cierre.
//!
//! Tras cerrar una posición, el host purga sus brackets con
//! `cancel_position_oco_orders`. D-698 hizo que esa purga devolviera Err al
//! fallar una pierna, y N15 (CL-176) que el host lo reportara; pero la pierna
//! seguía viva y nada impedía una reentrada en el mismo símbolo. El vigilante
//! sólo purga una pierna huérfana (`purge_orphan_legs`) si no hay posición en
//! su símbolo (pierna BOTH o sin lado) o en su lado (hedge), así que con la
//! reentrada ya abierta la pierna vieja quedaba armada sobre la posición nueva
//! con el precio de la anterior, y `ensure_position_protected` la contaba como
//! su protección. En hedge, una pierna sin `positionSide` hace fallar la purga
//! sin cancelarse y el vigilante no la toca mientras viva el lado contrario.
//!
//! Aquí queda la purga pendiente por (símbolo, lado de la posición cerrada).
//! Mientras haya alguna en un símbolo, el host no abre entradas en él. El
//! vigilante la reintenta en cada auditoría, y sólo cuando no hay posición
//! que esas piernas pudieran estar protegiendo; con un Ok la retira.
//!
//! No se guarda entre reinicios: al arrancar, `purge_orphan_legs` corre antes
//! de operar, sin posiciones nuevas que confundir.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::executor::OrderExecutor;
use crate::reconciliation::PositionRiskEntry;

/// Las purgas pendientes del proceso. El host anota las suyas aquí y su
/// vigilante las reintenta.
pub static PURGAS_PENDIENTES: PurgasPendientes = PurgasPendientes::new();

/// Purgas de brackets pendientes por (símbolo, lado de la posición cerrada:
/// `true` = largo).
pub struct PurgasPendientes {
    pendientes: Mutex<BTreeSet<(String, bool)>>,
    /// Cuántas hay, para que la consulta de cada entrada no tome el cerrojo
    /// cuando no hay ninguna. Se escribe siempre con el cerrojo tomado.
    cuantas: AtomicUsize,
}

/// Qué pasó con una purga pendiente en un reintento.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reintento {
    /// Purgada: se retira. Lleva las piernas canceladas (0 si ya no quedaba
    /// ninguna).
    Purgada(usize),
    /// Hay una posición abierta que esas piernas podrían estar protegiendo:
    /// no se cancelan a ciegas y la purga sigue pendiente.
    Aplazada,
    /// La purga volvió a fallar y sigue pendiente.
    Fallida(String),
}

impl Default for PurgasPendientes {
    fn default() -> Self {
        Self::new()
    }
}

impl PurgasPendientes {
    pub const fn new() -> Self {
        Self {
            pendientes: Mutex::new(BTreeSet::new()),
            cuantas: AtomicUsize::new(0),
        }
    }

    /// El conjunto sigue siendo válido aunque otro hilo haya entrado en pánico
    /// con el cerrojo tomado: cada operación lo deja completo.
    fn tomar(&self) -> MutexGuard<'_, BTreeSet<(String, bool)>> {
        self.pendientes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Anota la purga de (símbolo, lado). Devuelve true si no estaba ya.
    pub fn anotar(&self, symbol: &str, is_long: bool) -> bool {
        let mut p = self.tomar();
        let nueva = p.insert((symbol.to_string(), is_long));
        self.cuantas.store(p.len(), Ordering::Release);
        nueva
    }

    /// Retira la purga de (símbolo, lado). Devuelve true si estaba.
    pub fn resolver(&self, symbol: &str, is_long: bool) -> bool {
        let mut p = self.tomar();
        let estaba = p.remove(&(symbol.to_string(), is_long));
        self.cuantas.store(p.len(), Ordering::Release);
        estaba
    }

    /// Si una entrada en `symbol` debe esperar: hay una purga pendiente en
    /// ese símbolo, de cualquier lado. Se bloquea el símbolo entero porque en
    /// one-way una pierna BOTH sirve a las dos direcciones, y en hedge una
    /// pierna sin lado tampoco se sabe de cuál es.
    pub fn bloquea_entrada(&self, symbol: &str) -> bool {
        if self.cuantas.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.tomar().iter().any(|(s, _)| s == symbol)
    }

    /// Las purgas pendientes, en orden.
    pub fn pendientes(&self) -> Vec<(String, bool)> {
        self.tomar().iter().cloned().collect()
    }

    /// El reintento del vigilante con la foto de posiciones de su auditoría.
    /// Devuelve lo que pasó con cada purga.
    pub async fn reintentar(
        &self,
        ex: &OrderExecutor,
        posiciones: &[PositionRiskEntry],
    ) -> Vec<((String, bool), Reintento)> {
        let hedge = ex.is_hedge_mode.load(Ordering::SeqCst);
        let mut hechas = Vec::new();
        for (symbol, is_long) in self.pendientes() {
            let r = if lado_ocupado(&symbol, is_long, hedge, posiciones) {
                Reintento::Aplazada
            } else {
                match ex.cancel_position_oco_orders(&symbol, is_long).await {
                    Ok(n) => {
                        self.resolver(&symbol, is_long);
                        Reintento::Purgada(n)
                    }
                    Err(e) => Reintento::Fallida(e),
                }
            };
            hechas.push(((symbol, is_long), r));
        }
        hechas
    }
}

/// Si hay una posición abierta que las piernas de (símbolo, lado) podrían
/// estar protegiendo: en hedge, la del mismo lado; en one-way, cualquiera del
/// símbolo, porque sus piernas son BOTH. Purgar entonces le quitaría la
/// protección a esa posición (`cancel_position_oco_orders` cancela todas las
/// piernas que casan con el lado).
pub fn lado_ocupado(
    symbol: &str,
    is_long: bool,
    hedge: bool,
    posiciones: &[PositionRiskEntry],
) -> bool {
    posiciones
        .iter()
        .any(|p| p.symbol == symbol && p.is_open() && (!hedge || p.is_long() == is_long))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posicion(symbol: &str, amt: f64, lado: &str) -> PositionRiskEntry {
        PositionRiskEntry {
            symbol: symbol.to_string(),
            position_amt: amt,
            position_side: lado.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn i9_anotar_bloquea_el_simbolo_entero_hasta_resolver() {
        let p = PurgasPendientes::new();
        assert!(!p.bloquea_entrada("BTCUSDT"));
        assert!(p.anotar("BTCUSDT", true));
        assert!(!p.anotar("BTCUSDT", true), "la misma purga no se duplica");
        assert!(p.bloquea_entrada("BTCUSDT"), "bloquea la reentrada");
        assert!(!p.bloquea_entrada("ETHUSDT"), "sólo ese símbolo");
        assert!(p.anotar("BTCUSDT", false));
        assert!(p.resolver("BTCUSDT", true));
        assert!(p.bloquea_entrada("BTCUSDT"), "queda la purga del otro lado");
        assert!(p.resolver("BTCUSDT", false));
        assert!(!p.resolver("BTCUSDT", false));
        assert!(!p.bloquea_entrada("BTCUSDT"));
        assert!(p.pendientes().is_empty());
    }

    #[test]
    fn i9_el_lado_ocupado_depende_del_modo() {
        let largo = posicion("BTCUSDT", 0.01, "LONG");
        let corto = posicion("BTCUSDT", -0.01, "SHORT");
        let plana = posicion("BTCUSDT", 0.0, "LONG");
        let otra = posicion("ETHUSDT", 0.5, "LONG");
        let una = std::slice::from_ref;
        // Hedge: sólo cuenta la posición del mismo lado.
        assert!(lado_ocupado("BTCUSDT", true, true, una(&largo)));
        assert!(!lado_ocupado("BTCUSDT", true, true, una(&corto)));
        assert!(!lado_ocupado("BTCUSDT", true, true, una(&plana)));
        assert!(!lado_ocupado("BTCUSDT", true, true, una(&otra)));
        // One-way: cualquier posición del símbolo (piernas BOTH).
        let ambos = posicion("BTCUSDT", -0.01, "BOTH");
        assert!(lado_ocupado("BTCUSDT", true, false, una(&ambos)));
        assert!(lado_ocupado("BTCUSDT", false, false, &[ambos]));
        assert!(!lado_ocupado("BTCUSDT", true, false, &[plana, otra]));
        assert!(!lado_ocupado("BTCUSDT", true, false, &[]));
    }

    /// Con el freno del exchange armado la purga vuelve a fallar sin salir a
    /// la red y queda pendiente; con una posición en el lado no se intenta.
    #[tokio::test]
    async fn i9_el_reintento_fallido_o_aplazado_deja_la_purga_pendiente() {
        let ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.client()
            .freno()
            .registrar("HTTP_429_RATE_LIMITED retry_after=30");
        ex.is_hedge_mode.store(true, Ordering::SeqCst);
        let p = PurgasPendientes::new();
        p.anotar("BTCUSDT", true);
        p.anotar("ETHUSDT", false);

        let hechas = p
            .reintentar(&ex, &[posicion("ETHUSDT", -0.5, "SHORT")])
            .await;
        assert_eq!(hechas.len(), 2);
        for ((symbol, is_long), r) in &hechas {
            match (symbol.as_str(), *is_long) {
                ("BTCUSDT", true) => {
                    assert!(matches!(r, Reintento::Fallida(_)), "BTCUSDT: {r:?}")
                }
                ("ETHUSDT", false) => assert_eq!(*r, Reintento::Aplazada),
                otro => panic!("purga inesperada {otro:?}"),
            }
        }
        assert!(p.bloquea_entrada("BTCUSDT") && p.bloquea_entrada("ETHUSDT"));
    }

    /// Sin piernas abiertas (ejecutor en papel) la purga da Ok(0): se retira y
    /// la reentrada vuelve a estar permitida.
    #[tokio::test]
    async fn i9_el_reintento_con_ok_retira_la_purga() {
        let mut ex = OrderExecutor::new("k".into(), "s".into(), true);
        ex.set_paper_trading(true);
        let p = PurgasPendientes::new();
        p.anotar("BTCUSDT", false);
        let hechas = p.reintentar(&ex, &[]).await;
        assert_eq!(
            hechas,
            vec![(("BTCUSDT".to_string(), false), Reintento::Purgada(0))]
        );
        assert!(!p.bloquea_entrada("BTCUSDT"));
        assert!(p.pendientes().is_empty());
    }
}
