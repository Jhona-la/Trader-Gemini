//! RECONCILIACIÓN (F1.7) — verdad del exchange vs estado local.
//!
//! QUÉ: GET /fapi/v2/positionRisk (firmado) y diff contra OrderRegistry.
//! POR QUÉ: directiva de arranque — "saber si hay posiciones abiertas ANTES de
//!      operar" + contención de drift (ordenes huérfanas, fills perdidos,
//!      OCO que quedó a medias, posiciones manuales desde la app de Binance).
//! CUÁNDO: al arranque (obligatorio, F7.1 lo gatea) y periódicamente.

use crate::order_registry::OrderRegistry;
use crate::user_data_stream::RemotePosition;
use serde::Deserialize;

/// Entrada de /fapi/v2/positionRisk.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct PositionRiskEntry {
    #[serde(default)]
    pub symbol: String,
    #[serde(
        rename = "positionAmt",
        deserialize_with = "crate::order_types::string_or_f64",
        default
    )]
    pub position_amt: f64,
    #[serde(
        rename = "entryPrice",
        deserialize_with = "crate::order_types::string_or_f64",
        default
    )]
    pub entry_price: f64,
    #[serde(
        rename = "unRealizedProfit",
        deserialize_with = "crate::order_types::string_or_f64",
        default
    )]
    pub unrealized_pnl: f64,
    #[serde(
        rename = "liquidationPrice",
        deserialize_with = "crate::order_types::string_or_f64",
        default
    )]
    pub liquidation_price: f64,
    #[serde(deserialize_with = "crate::order_types::string_or_f64", default)]
    pub leverage: f64,
    #[serde(rename = "updateTime", default)]
    pub update_time: u64,
}

impl PositionRiskEntry {
    pub fn is_open(&self) -> bool {
        self.position_amt.abs() > 0.0
    }

    pub fn is_long(&self) -> bool {
        self.position_amt > 0.0
    }
}

#[derive(Debug, Default)]
pub struct ReconciliationReport {
    /// Posiciones abiertas en el exchange (la verdad a adoptar).
    pub open_positions: Vec<PositionRiskEntry>,
    /// Órdenes activas en el registro local SIN posición remota asociada
    /// (sospecha de fill perdido o cancel no confirmado).
    pub suspicious_active_orders: Vec<String>,
    /// Resumen humano para telemetría/decisiones.
    pub summary: String,
}

/// Diff puro (testeable sin red): posición remota como fuente de verdad.
pub fn reconcile(remote: &[PositionRiskEntry], registry: &OrderRegistry) -> ReconciliationReport {
    let mut report = ReconciliationReport::default();
    report.open_positions = remote.iter().filter(|p| p.is_open()).cloned().collect();

    // Una orden local "activa" en un símbolo cuya posición remota ya no existe
    // merece auditoría (query_order / cancel) — no se toca a ciegas aquí.
    for p in &report.open_positions {
        for o in registry.active_for_symbol(&p.symbol) {
            report.suspicious_active_orders.push(format!(
                "{} orden {} activa (restante {}) con posición remota {}",
                p.symbol,
                o.client_order_id,
                o.remaining_qty(),
                p.position_amt
            ));
        }
    }

    report.summary = format!(
        "reconciliación: {} posiciones abiertas en exchange, {} órdenes locales sospechosas [{}]",
        report.open_positions.len(),
        report.suspicious_active_orders.len(),
        report
            .open_positions
            .iter()
            .map(|p| format!("{} {}", p.symbol, p.position_amt))
            .collect::<Vec<_>>()
            .join(", ")
    );
    report
}

impl PositionRiskEntry {
    /// Conversión a la vista común del stream (F1.6).
    pub fn to_remote_position(&self) -> RemotePosition {
        RemotePosition {
            symbol: self.symbol.clone(),
            position_amt: self.position_amt,
            entry_price: self.entry_price,
            unrealized_pnl: self.unrealized_pnl,
            isolated_wallet: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(symbol: &str, amt: f64) -> PositionRiskEntry {
        PositionRiskEntry {
            symbol: symbol.into(),
            position_amt: amt,
            entry_price: 60000.0,
            ..Default::default()
        }
    }

    #[test]
    fn diff_detects_open_positions_and_suspicious_orders() {
        let registry = OrderRegistry::new();
        registry.register_intent("live1", "BTCUSDT", "BUY", "LONG", "LIMIT", 2.0, 1000);
        // Orden activa + posición remota abierta → sospechosa (¿fill perdido?)
        let remote = vec![entry("BTCUSDT", 0.5), entry("ETHUSDT", 0.0)];
        let report = reconcile(&remote, &registry);
        assert_eq!(report.open_positions.len(), 1, "ETH flat no cuenta");
        assert_eq!(report.suspicious_active_orders.len(), 1);
        assert!(report.summary.contains("BTCUSDT"));
    }

    #[test]
    fn parse_position_risk_body() {
        let body = r#"[{"symbol":"BTCUSDT","positionAmt":"0.002","entryPrice":"60000.0","unRealizedProfit":"1.5","liquidationPrice":"30000","leverage":"20","updateTime":1700000000000}]"#;
        let entries: Vec<PositionRiskEntry> =
            serde_json::from_str(body).expect("parse positionRisk");
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_open() && entries[0].is_long());
        assert!((entries[0].leverage - 20.0).abs() < 1e-12);
    }
}
