//! REGISTRO DE ÓRDENES (F1.5) — Máquina de estados del ciclo de vida de órdenes.
//!
//! QUÉ: memoria viva de TODA orden emitida: NEW → PARTIALLY_FILLED → FILLED /
//!      CANCELED / EXPIRED / REJECTED, con fills acumulados, avgPrice y comisiones.
//! POR QUÉ: sin esto los fills parciales y los acks eran invisibles (auditoría F1);
//!          maker-chase y reconciliación necesitan saber CUÁNTO se ejecutó realmente.
//! FUENTES: (1) OrderAck de REST (POST/GET), (2) ORDER_TRADE_UPDATE del user-data
//!          stream (F1.6). Ambos convergen aquí — única fuente de verdad local.
//! CONTENCIÓN: bloqueo corto por orden; NO está en el tick-path del motor de señales,
//!          solo en el camino de ejecución (decenas de ops/seg máx).

use crate::order_types::OrderAck;
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    Expired,
    Rejected,
    /// Estado no reconocido — se conserva el último conocido y se alerta.
    Unknown,
}

impl OrderStatus {
    pub fn parse(s: &str) -> OrderStatus {
        match s {
            "NEW" => OrderStatus::New,
            "PARTIALLY_FILLED" => OrderStatus::PartiallyFilled,
            "FILLED" => OrderStatus::Filled,
            "CANCELED" => OrderStatus::Canceled,
            "EXPIRED" => OrderStatus::Expired,
            "REJECTED" => OrderStatus::Rejected,
            _ => OrderStatus::Unknown,
        }
    }

    /// La orden puede seguir ejecutándose (decisiones de chase/cancel dependen de esto).
    pub fn is_active(self) -> bool {
        matches!(self, OrderStatus::New | OrderStatus::PartiallyFilled)
    }
}

#[derive(Debug, Clone)]
pub struct TrackedOrder {
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub position_side: String,
    pub order_type: String,
    pub orig_qty: f64,
    pub executed_qty: f64,
    pub avg_price: f64,
    pub cum_quote: f64,
    pub total_commission: f64,
    pub status: OrderStatus,
    pub order_id: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// Último fill (qty) — para detección de fills en vivo.
    pub last_fill_qty: f64,
    pub last_fill_price: f64,
}

impl TrackedOrder {
    #[inline(always)]
    pub fn remaining_qty(&self) -> f64 {
        (self.orig_qty - self.executed_qty).max(0.0)
    }
}

/// Evento normalizado de ORDER_TRADE_UPDATE (WS privado de Binance).
#[derive(Debug, Clone)]
pub struct TradeUpdate {
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub order_id: u64,
    pub status: OrderStatus,
    pub orig_qty: f64,
    pub cumulative_filled_qty: f64,
    pub last_filled_qty: f64,
    pub last_filled_price: f64,
    pub avg_price: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub trade_time_ms: u64,
}

#[derive(Debug, Default)]
pub struct RegistryStats {
    pub total: usize,
    pub active: usize,
    pub filled: usize,
    pub partially_filled: usize,
    pub canceled: usize,
    pub rejected: usize,
    pub unknown_status: usize,
}

pub struct OrderRegistry {
    orders: RwLock<HashMap<String, TrackedOrder>>,
}

impl Default for OrderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl OrderRegistry {
    pub fn new() -> Self {
        Self {
            orders: RwLock::new(HashMap::new()),
        }
    }

    /// Registra la intención ANTES del envío (si el POST muere en red, la orden
    /// queda referenciada por clientOrderId para query de resolución).
    pub fn register_intent(
        &self,
        client_order_id: &str,
        symbol: &str,
        side: &str,
        position_side: &str,
        order_type: &str,
        quantity: f64,
        now_ms: u64,
    ) {
        let mut map = self.orders.write().unwrap();
        map.entry(client_order_id.to_string())
            .or_insert_with(|| TrackedOrder {
                client_order_id: client_order_id.to_string(),
                symbol: symbol.to_string(),
                side: side.to_string(),
                position_side: position_side.to_string(),
                order_type: order_type.to_string(),
                orig_qty: quantity,
                executed_qty: 0.0,
                avg_price: 0.0,
                cum_quote: 0.0,
                total_commission: 0.0,
                status: OrderStatus::New,
                order_id: 0,
                created_ms: now_ms,
                updated_ms: now_ms,
                last_fill_qty: 0.0,
                last_fill_price: 0.0,
            });
    }

    /// Aplica un ack REST (respuesta de POST o GET /fapi/v1/order).
    pub fn apply_ack(&self, ack: &OrderAck, now_ms: u64) {
        let mut map = self.orders.write().unwrap();
        let entry = map
            .entry(ack.client_order_id.clone())
            .or_insert_with(|| TrackedOrder {
                client_order_id: ack.client_order_id.clone(),
                symbol: ack.symbol.clone(),
                side: ack.side.clone(),
                position_side: String::new(),
                order_type: ack.order_type.clone(),
                orig_qty: ack.orig_qty,
                executed_qty: 0.0,
                avg_price: 0.0,
                cum_quote: 0.0,
                total_commission: 0.0,
                status: OrderStatus::New,
                order_id: 0,
                created_ms: now_ms,
                updated_ms: now_ms,
                last_fill_qty: 0.0,
                last_fill_price: 0.0,
            });
        // Invariantes: executed_qty y avg_price del exchange son la verdad.
        if ack.executed_qty >= entry.executed_qty {
            entry.executed_qty = ack.executed_qty;
        }
        if ack.avg_price > 0.0 {
            entry.avg_price = ack.avg_price;
        }
        if ack.cum_quote > 0.0 {
            entry.cum_quote = ack.cum_quote;
        }
        if ack.order_id > 0 {
            entry.order_id = ack.order_id;
        }
        let fills_commission: f64 = ack.fills.iter().map(|f| f.commission).sum();
        if fills_commission > entry.total_commission {
            entry.total_commission = fills_commission;
        }
        entry.status = OrderStatus::parse(&ack.status);
        entry.updated_ms = now_ms;
    }

    /// Aplica un ORDER_TRADE_UPDATE del user-data stream.
    /// Los eventos WS llegan por fill: acumular comisión, no sobrescriberla.
    pub fn apply_trade_update(&self, u: &TradeUpdate, now_ms: u64) {
        let mut map = self.orders.write().unwrap();
        let entry = map
            .entry(u.client_order_id.clone())
            .or_insert_with(|| TrackedOrder {
                client_order_id: u.client_order_id.clone(),
                symbol: u.symbol.clone(),
                side: u.side.clone(),
                position_side: String::new(),
                order_type: u.order_type.clone(),
                orig_qty: u.orig_qty,
                executed_qty: 0.0,
                avg_price: 0.0,
                cum_quote: 0.0,
                total_commission: 0.0,
                status: OrderStatus::New,
                order_id: u.order_id,
                created_ms: now_ms,
                updated_ms: now_ms,
                last_fill_qty: 0.0,
                last_fill_price: 0.0,
            });
        if u.order_id > 0 {
            entry.order_id = u.order_id;
        }
        if u.cumulative_filled_qty >= entry.executed_qty {
            entry.executed_qty = u.cumulative_filled_qty;
        } else {
            // Fill desordenado (WS llega antes que un ack viejo): conservar el máximo.
            println!(
                "⚠️ [REGISTRY] TradeUpdate con cumulativeFilled {} < conocido {} para {}",
                u.cumulative_filled_qty, entry.executed_qty, u.client_order_id
            );
        }
        if u.avg_price > 0.0 {
            entry.avg_price = u.avg_price;
        }
        // Comisión acumulativa por fill (cada evento trae SOLO la del fill actual).
        if u.last_filled_qty > 0.0 && u.commission > 0.0 {
            entry.total_commission += u.commission;
        }
        entry.last_fill_qty = u.last_filled_qty;
        entry.last_fill_price = u.last_filled_price;
        entry.status = u.status;
        entry.updated_ms = now_ms;
    }

    pub fn get(&self, client_order_id: &str) -> Option<TrackedOrder> {
        self.orders.read().unwrap().get(client_order_id).cloned()
    }

    /// Órdenes vivas por símbolo (para chase/cancel masivo).
    pub fn active_for_symbol(&self, symbol: &str) -> Vec<TrackedOrder> {
        let map = self.orders.read().unwrap();
        map.values()
            .filter(|o| o.symbol == symbol && o.status.is_active())
            .cloned()
            .collect()
    }

    /// Cantidad neta llena (con signo según side) de órdenes activas del símbolo.
    pub fn pending_qty_for_symbol(&self, symbol: &str) -> f64 {
        self.active_for_symbol(symbol)
            .iter()
            .map(|o| {
                let rem = o.remaining_qty();
                if o.side == "BUY" {
                    rem
                } else {
                    -rem
                }
            })
            .sum()
    }

    pub fn stats(&self) -> RegistryStats {
        let map = self.orders.read().unwrap();
        let mut s = RegistryStats::default();
        s.total = map.len();
        for o in map.values() {
            match o.status {
                OrderStatus::New | OrderStatus::PartiallyFilled => s.active += 1,
                OrderStatus::Filled => s.filled += 1,
                OrderStatus::Canceled => s.canceled += 1,
                OrderStatus::Rejected => s.rejected += 1,
                OrderStatus::Expired => s.unknown_status += 1,
                OrderStatus::Unknown => s.unknown_status += 1,
            }
            if o.status == OrderStatus::PartiallyFilled {
                s.partially_filled += 1;
            }
        }
        s
    }

    /// Purga órdenes terminadas más viejas que `older_than_ms` (higiene de memoria).
    pub fn prune_terminated(&self, older_than_ms: u64) -> usize {
        let mut map = self.orders.write().unwrap();
        let before = map.len();
        map.retain(|_, o| o.status.is_active() || o.updated_ms >= older_than_ms);
        before - map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ack(id: &str, status: &str, executed: f64, avg: f64) -> OrderAck {
        OrderAck {
            client_order_id: id.to_string(),
            symbol: "BTCUSDT".into(),
            status: status.to_string(),
            side: "BUY".into(),
            order_type: "LIMIT".into(),
            orig_qty: 2.0,
            executed_qty: executed,
            avg_price: avg,
            cum_quote: executed * avg,
            ..Default::default()
        }
    }

    #[test]
    fn lifecycle_partial_to_filled_via_ws() {
        let reg = OrderRegistry::new();
        reg.register_intent("o1", "BTCUSDT", "BUY", "LONG", "LIMIT", 2.0, 1000);
        reg.apply_ack(&ack("o1", "PARTIALLY_FILLED", 0.5, 60000.0), 1100);

        let o = reg.get("o1").unwrap();
        assert_eq!(o.status, OrderStatus::PartiallyFilled);
        assert!((o.executed_qty - 0.5).abs() < 1e-12);

        // WS: segundo fill de 0.5 con comisión 0.0001 y cierre FILLED
        reg.apply_trade_update(
            &TradeUpdate {
                client_order_id: "o1".into(),
                symbol: "BTCUSDT".into(),
                side: "BUY".into(),
                order_type: "LIMIT".into(),
                order_id: 42,
                status: OrderStatus::Filled,
                orig_qty: 2.0,
                cumulative_filled_qty: 1.0,
                last_filled_qty: 0.5,
                last_filled_price: 60100.0,
                avg_price: 60050.0,
                commission: 0.0001,
                commission_asset: "BNB".into(),
                trade_time_ms: 1200,
            },
            1200,
        );
        let o = reg.get("o1").unwrap();
        assert_eq!(o.status, OrderStatus::Filled);
        assert!((o.executed_qty - 1.0).abs() < 1e-12);
        assert!((o.avg_price - 60050.0).abs() < 1e-12);
        assert!((o.total_commission - 0.0001).abs() < 1e-12);
    }

    #[test]
    fn orphan_ws_update_adopts_order() {
        let reg = OrderRegistry::new();
        reg.apply_trade_update(
            &TradeUpdate {
                client_order_id: "huérfana".into(),
                symbol: "ETHUSDT".into(),
                side: "SELL".into(),
                order_type: "STOP_MARKET".into(),
                order_id: 7,
                status: OrderStatus::Filled,
                orig_qty: 1.0,
                cumulative_filled_qty: 1.0,
                last_filled_qty: 1.0,
                last_filled_price: 3000.0,
                avg_price: 3000.0,
                commission: 0.0002,
                commission_asset: "USDT".into(),
                trade_time_ms: 1,
            },
            1,
        );
        let o = reg
            .get("huérfana")
            .expect("evento WS de orden desconocida se adopta");
        assert_eq!(o.status, OrderStatus::Filled);
        assert_eq!(o.order_id, 7);
    }

    #[test]
    fn stale_ack_does_not_regress_executed_qty() {
        let reg = OrderRegistry::new();
        reg.apply_ack(&ack("o2", "PARTIALLY_FILLED", 1.0, 60000.0), 100);
        // Ack REST viejo llega tarde con menos ejecutado: no debe regresar.
        reg.apply_ack(&ack("o2", "PARTIALLY_FILLED", 0.5, 59000.0), 200);
        let o = reg.get("o2").unwrap();
        assert!(
            (o.executed_qty - 1.0).abs() < 1e-12,
            "executed_qty no debe regresar"
        );
    }

    #[test]
    fn stats_and_prune() {
        let reg = OrderRegistry::new();
        reg.apply_ack(&ack("a", "FILLED", 1.0, 1.0), 100);
        reg.apply_ack(&ack("b", "CANCELED", 0.0, 0.0), 100);
        reg.apply_ack(&ack("c", "NEW", 0.0, 0.0), 100);
        let s = reg.stats();
        assert_eq!((s.total, s.active, s.filled, s.canceled), (3, 1, 1, 1));
        // Terminadas con updated_ms < 500 se purgan; la activa queda.
        assert_eq!(reg.prune_terminated(500), 2);
        assert_eq!(reg.stats().total, 1);
    }
}
