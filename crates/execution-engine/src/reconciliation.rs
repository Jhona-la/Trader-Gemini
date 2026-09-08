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
    #[serde(rename = "positionSide", default)]
    pub position_side: String,
    #[serde(rename = "updateTime", default)]
    pub update_time: u64,
}

impl PositionRiskEntry {
    pub fn is_open(&self) -> bool {
        self.position_amt.is_finite() && self.position_amt.abs() > 1e-12
    }

    pub fn is_long(&self) -> bool {
        let side = self.position_side.trim();
        if side.eq_ignore_ascii_case("LONG") {
            true
        } else if side.eq_ignore_ascii_case("SHORT") {
            false
        } else {
            self.position_amt > 0.0
        }
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

    let remote_map: std::collections::HashMap<(&str, &str), &PositionRiskEntry> =
        remote.iter().map(|p| ((p.symbol.as_str(), p.position_side.as_str()), p)).collect();

    // Auditar TODAS las órdenes locales activas contra la realidad del exchange
    for o in registry.active_orders() {
        let sym = o.symbol.as_str();
        let side_key = if o.position_side.is_empty() { "BOTH" } else { o.position_side.as_str() };
        let matching_p = remote_map.get(&(sym, side_key))
            .copied()
            .or_else(|| remote_map.get(&(sym, "BOTH")).copied())
            .or_else(|| remote_map.get(&(sym, "")).copied())
            .or_else(|| remote.iter().find(|p| p.symbol == sym));

        if let Some(p) = matching_p {
            if !p.is_open() {
                // Orden local activa pero posición remota plana (0.0): orden huérfana
                report.suspicious_active_orders.push(format!(
                    "{} orden {} activa (restante {}) con posición remota plana (0.0)",
                    sym,
                    o.client_order_id,
                    o.remaining_qty()
                ));
            }
        } else {
            // Símbolo no presente en el exchange report
            report.suspicious_active_orders.push(format!(
                "{} orden {} activa sin entrada remota",
                sym,
                o.client_order_id
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

impl ReconciliationReport {
    /// Adopta automáticamente las posiciones del exchange al OrderRegistry y reconcilia el estado local (#121-#140, #1405)
    pub fn apply_to_registry(&self, registry: &OrderRegistry) -> usize {
        let mut adopted = 0;
        for pos in &self.open_positions {
            let side = if pos.is_long() { "BUY" } else { "SELL" };
            let client_id = format!("adopted_{}_{}", pos.symbol, pos.update_time);
            let ack = crate::order_types::OrderAck {
                client_order_id: client_id,
                symbol: pos.symbol.clone(),
                side: side.to_string(),
                order_type: "MARKET".to_string(),
                orig_qty: pos.position_amt.abs(),
                executed_qty: pos.position_amt.abs(),
                avg_price: pos.entry_price,
                price: pos.entry_price,
                cum_quote: pos.position_amt.abs() * pos.entry_price,
                status: "FILLED".to_string(),
                order_id: 0,
                update_time: pos.update_time,
                fills: Vec::new(),
            };
            registry.apply_ack(&ack, pos.update_time);
            adopted += 1;
        }
        adopted
    }
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
            position_side: self.position_side.clone(),
        }
    }
}

/// R3.5: Reconcilia el estado de posiciones entre el exchange (PositionRiskEntry) y GlobalArena.
/// - Si el exchange está plano (amt == 0.0) pero la Arena tiene posiciones abiertas (scalp o swing),
///   se detecta posición fantasma y se cierran en la Arena para liberar margen.
/// - Si el exchange tiene posición abierta pero la Arena está plana, se adopta en Swing.
/// - Si ambos tienen posición abierta, se corrige cualquier deriva (drift) en la cantidad.
pub fn reconcile_arena(
    remote: &[PositionRiskEntry],
    arena: &quantum_arena::GlobalArena,
    now_ms: u64,
) -> usize {
    let mut adjustments = 0;
    let universe_size = quantum_arena::symbols::get_active_universe_size();

    let mut remote_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let mut remote_price_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for p in remote {
        if p.is_open() {
            *remote_map.entry(p.symbol.to_uppercase()).or_insert(0.0) += p.position_amt;
            remote_price_map.insert(p.symbol.to_uppercase(), p.entry_price);
        }
    }

    for coin_idx in 0..universe_size {
        let sym = match quantum_arena::symbol_registry::try_symbol(coin_idx) {
            Some(s) => s.to_uppercase(),
            None => continue,
        };
        let remote_net_qty = remote_map.get(&sym).copied().unwrap_or(0.0);
        let remote_price = remote_price_map.get(&sym).copied().unwrap_or(0.0);
        let coin = &arena.coins[coin_idx];

        let cont_open = coin.positions.position.is_open();

        let cont_qty = if cont_open {
            let q = coin.positions.position.quantity.load(std::sync::atomic::Ordering::Relaxed);
            if coin.positions.position.is_long.load(std::sync::atomic::Ordering::Relaxed) { q } else { -q }
        } else {
            0.0
        };

        let arena_net_qty = cont_qty;

        if remote_net_qty.abs() < 1e-8 {
            // Exchange está plano pero la Arena cree que tiene posiciones abiertas: phantom cleanup
            if cont_open {
                let (_, _, _, m, _) = coin.positions.position.close_with_fee();
                if m > 0.0 {
                    let cur_u = arena.used_margin.load(std::sync::atomic::Ordering::Relaxed);
                    arena.used_margin.store((cur_u - m).max(0.0), std::sync::atomic::Ordering::Relaxed);
                }
                adjustments += 1;
            }
        } else {
            // Exchange tiene posición abierta
            if !cont_open {
                // Posición huérfana en exchange: adoptar en Horizonte Continuo
                let is_long = remote_net_qty > 0.0;
                let abs_qty = remote_net_qty.abs();
                let price = if remote_price > 0.0 {
                    remote_price
                } else {
                    coin.current_price.load(std::sync::atomic::Ordering::Relaxed)
                };
                let notional = abs_qty * price;
                let margin = notional / 10.0; // 10x de margen estimado
                coin.positions.position.open_with_horizon(
                    is_long,
                    price,
                    abs_qty,
                    margin,
                    now_ms,
                    0.0,
                    0.0,
                    quantum_arena::position::PositionHorizon::Continuous,
                );
                arena.used_margin.fetch_add(margin, std::sync::atomic::Ordering::Relaxed);
                adjustments += 1;
            } else if (arena_net_qty - remote_net_qty).abs() > 1e-6 {
                // Drift en cantidad: actualizar posición continua para reflejar el tamaño real
                let target_abs = remote_net_qty.abs();
                if target_abs <= 1e-6 {
                    let (_, _, _, old_margin, _) = coin.positions.position.close_with_fee();
                    if old_margin > 0.0 {
                        let cur_u = arena.used_margin.load(std::sync::atomic::Ordering::Relaxed);
                        arena.used_margin.store((cur_u - old_margin).max(0.0), std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    let price = coin.positions.position.entry_price.load(std::sync::atomic::Ordering::Relaxed);
                    let safe_price = if price > 0.0 { price } else { coin.current_price.load(std::sync::atomic::Ordering::Relaxed) };
                    let old_margin = coin.positions.position.margin_used.load(std::sync::atomic::Ordering::Relaxed);
                    let new_margin = if safe_price > 0.0 { (target_abs * safe_price) / 10.0 } else { old_margin };
                    let margin_diff = new_margin - old_margin;

                    coin.positions.position.quantity.store(target_abs, std::sync::atomic::Ordering::Relaxed);
                    coin.positions.position.margin_used.store(new_margin, std::sync::atomic::Ordering::Relaxed);
                    arena.used_margin.fetch_add(margin_diff, std::sync::atomic::Ordering::Relaxed);
                }
                adjustments += 1;
            }
        }
    }

    adjustments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(symbol: &str, amt: f64) -> PositionRiskEntry {
        PositionRiskEntry {
            symbol: symbol.into(),
            position_amt: amt,
            entry_price: 60000.0,
            update_time: 1700000000000,
            ..Default::default()
        }
    }

    #[test]
    fn diff_detects_open_positions_and_suspicious_orders() {
        let registry = OrderRegistry::new();
        registry.register_intent("live1", "BTCUSDT", "BUY", "LONG", "LIMIT", 2.0, 1000);
        registry.register_intent("live2", "ETHUSDT", "BUY", "LONG", "LIMIT", 1.0, 1000);
        
        // BTCUSDT tiene posición abierta (0.5), ETHUSDT está FLAT (0.0)
        let remote = vec![entry("BTCUSDT", 0.5), entry("ETHUSDT", 0.0)];
        let report = reconcile(&remote, &registry);
        assert_eq!(report.open_positions.len(), 1, "ETH flat no cuenta en open_positions");
        assert_eq!(report.suspicious_active_orders.len(), 1, "Solo la orden de ETH con posición plana es sospechosa");
        assert!(report.suspicious_active_orders.iter().any(|s| s.contains("ETHUSDT") && s.contains("plana")));
        assert!(report.summary.contains("BTCUSDT"));

        let adopted = report.apply_to_registry(&registry);
        assert_eq!(adopted, 1);
    }

    #[test]
    fn test_position_risk_json_parse() {
        let body = r#"[{"symbol":"BTCUSDT","positionAmt":"0.5","entryPrice":"60000.0","unRealizedProfit":"100.0","liquidationPrice":"50000.0","leverage":"20","positionSide":"LONG","updateTime":1700000000000}]"#;
        let entries: Vec<PositionRiskEntry> =
            serde_json::from_str(body).expect("parse positionRisk");
        assert_eq!(entries.len(), 1);
        assert!(entries[0].is_open() && entries[0].is_long());
        assert!((entries[0].leverage - 20.0).abs() < 1e-12);
    }

    #[test]
    fn test_position_risk_short_and_nan_defense() {
        let short_entry = PositionRiskEntry {
            symbol: "SOLUSDT".to_string(),
            position_amt: -5.0,
            entry_price: 150.0,
            unrealized_pnl: 10.0,
            liquidation_price: 250.0,
            leverage: 10.0,
            position_side: "SHORT".to_string(),
            update_time: 1700000000000,
        };

        assert!(short_entry.is_open());
        assert!(!short_entry.is_long());

        let remote = short_entry.to_remote_position();
        assert_eq!(remote.symbol, "SOLUSDT");
        assert_eq!(remote.position_amt, -5.0);
        assert_eq!(remote.position_side, "SHORT");

        let nan_entry = PositionRiskEntry {
            symbol: "NAN_COIN".to_string(),
            position_amt: f64::NAN,
            ..Default::default()
        };
        assert!(!nan_entry.is_open());
    }

    #[test]
    fn test_reconcile_arena_phantom_and_adoption() {
        use quantum_arena::symbol_registry::SymbolSpec;

        let specs = vec![
            SymbolSpec {
                symbol: "BTCUSDT".into(),
                step_size: 0.001,
                tick_size: 0.1,
                min_qty: 0.001,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
            SymbolSpec {
                symbol: "ETHUSDT".into(),
                step_size: 0.01,
                tick_size: 0.01,
                min_qty: 0.01,
                min_notional: 5.0,
                max_leverage: 50,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            },
        ];
        quantum_arena::symbol_registry::update_registry(specs);
        quantum_arena::symbols::update_dynamic_universe(vec!["BTCUSDT".into(), "ETHUSDT".into()]);

        let arena = quantum_arena::GlobalArena::new(100.0);

        // Simulate phantom position on BTC (arena has it open, but Binance is flat)
        arena.coins[0].positions.position.open_with_horizon(
            true,
            60000.0,
            0.01,
            12.0,
            1000,
            61000.0,
            59000.0,
            quantum_arena::position::PositionHorizon::Continuous,
        );
        arena.used_margin.store(12.0, std::sync::atomic::Ordering::Relaxed);

        // Remote has BTC flat (0.0), ETH open with 0.5 LONG
        let remote = vec![
            entry("BTCUSDT", 0.0),
            PositionRiskEntry {
                symbol: "ETHUSDT".into(),
                position_amt: 0.5,
                entry_price: 3000.0,
                update_time: 2000,
                ..Default::default()
            },
        ];

        let adjs = reconcile_arena(&remote, &arena, 2000);
        assert!(adjs >= 2, "Must adjust phantom BTC position and adopt orphan ETH position");

        // BTC phantom position must be closed and margin reclaimed
        assert!(!arena.coins[0].positions.position.is_open());

        // ETH position must be adopted in Continuous position
        assert!(arena.coins[1].positions.position.is_open());
        assert_eq!(arena.coins[1].positions.position.quantity.load(std::sync::atomic::Ordering::Relaxed), 0.5);
    }
}

