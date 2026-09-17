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
    /// D-417: Órdenes abiertas en el exchange (GET /fapi/v1/openOrders) que no están en el registro local
    /// (órdenes resting huérfanas en Binance tras reinicio o desconexión).
    pub orphan_remote_orders: Vec<crate::order_types::OrderAck>,
    /// Resumen humano para telemetría/decisiones.
    pub summary: String,
}

/// D-417: Reconciliación omnisciente completa incluyendo posiciones y órdenes resting abiertas en Binance.
pub fn reconcile_with_orders(
    remote: &[PositionRiskEntry],
    remote_orders: &[crate::order_types::OrderAck],
    registry: &OrderRegistry,
) -> ReconciliationReport {
    let mut report = ReconciliationReport::default();
    report.open_positions = remote.iter().filter(|p| p.is_open()).cloned().collect();

    let remote_map: std::collections::HashMap<(&str, &str), &PositionRiskEntry> = remote
        .iter()
        .map(|p| ((p.symbol.as_str(), p.position_side.as_str()), p))
        .collect();

    // Auditar TODAS las órdenes locales activas contra la realidad del exchange
    for o in registry.active_orders() {
        let sym = o.symbol.as_str();
        let side_key = if o.position_side.is_empty() {
            "BOTH"
        } else {
            o.position_side.as_str()
        };
        let matching_p = remote_map
            .get(&(sym, side_key))
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
                sym, o.client_order_id
            ));
        }
    }

    // D-417: Detectar órdenes resting en Binance no reconocidas en el registro local
    let local_active_ids: std::collections::HashSet<String> = registry
        .active_orders()
        .into_iter()
        .map(|o| o.client_order_id)
        .collect();

    for ro in remote_orders {
        if !local_active_ids.contains(&ro.client_order_id) {
            report.orphan_remote_orders.push(ro.clone());
        }
    }

    report.summary = format!(
        "reconciliación: {} posiciones abiertas en exchange, {} órdenes locales sospechosas, {} órdenes remotas huérfanas [{}]",
        report.open_positions.len(),
        report.suspicious_active_orders.len(),
        report.orphan_remote_orders.len(),
        report
            .open_positions
            .iter()
            .map(|p| format!("{} {}", p.symbol, p.position_amt))
            .collect::<Vec<_>>()
            .join(", ")
    );
    report
}

/// Diff puro (testeable sin red): posición remota como fuente de verdad.
pub fn reconcile(remote: &[PositionRiskEntry], registry: &OrderRegistry) -> ReconciliationReport {
    reconcile_with_orders(remote, &[], registry)
}

impl ReconciliationReport {
    /// Adopta automáticamente las posiciones del exchange al OrderRegistry y reconcilia el estado local (#121-#140, #1405)
    ///
    /// D-632 (DÉCIMA OLA): `taker_fee_rate` es la comisión taker vigente de la
    /// cuenta. Se usa para imputar la comisión de entrada de la posición
    /// adoptada, que antes entraba con `fills` vacío y por tanto con comisión
    /// cero: un sesgo optimista permanente en su PnL que se propagaba al win
    /// rate, al profit factor, al dimensionamiento de Kelly y a la aptitud del
    /// daemon online. La entrada real pudo ser maker, así que la imputación
    /// taker es la cota conservadora.
    pub fn apply_to_registry(&self, registry: &OrderRegistry, taker_fee_rate: f64) -> usize {
        // Tarifa taker VIP0 de Binance USDT-M como último recurso si la cuenta
        // no aportó la suya: mismo valor inicial que usa QuantumConfig.
        let fee = if taker_fee_rate.is_finite() && taker_fee_rate > 0.0 {
            taker_fee_rate
        } else {
            0.0005
        };
        let mut adopted = 0;
        for pos in &self.open_positions {
            let qty = pos.position_amt.abs();
            let imputed_commission = qty * pos.entry_price * fee;
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
                fills: vec![crate::order_types::Fill {
                    price: pos.entry_price,
                    qty,
                    commission: if imputed_commission.is_finite() {
                        imputed_commission
                    } else {
                        0.0
                    },
                    commission_asset: "USDT".to_string(),
                    trade_id: 0,
                }],
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
    let mut remote_price_map: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    let mut remote_lev_map: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    for p in remote {
        if p.is_open() {
            *remote_map.entry(p.symbol.to_uppercase()).or_insert(0.0) += p.position_amt;
            remote_price_map.insert(p.symbol.to_uppercase(), p.entry_price);
            remote_lev_map.insert(p.symbol.to_uppercase(), p.leverage);
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
            let q = coin
                .positions
                .position
                .quantity
                .load(std::sync::atomic::Ordering::Relaxed);
            if coin
                .positions
                .position
                .is_long
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                q
            } else {
                -q
            }
        } else {
            0.0
        };

        let arena_net_qty = cont_qty;

        if remote_net_qty.abs() < 1e-8 {
            // Exchange está plano pero la Arena cree que tiene posiciones abiertas: phantom cleanup
            if cont_open {
                // D-716 (DÉCIMA OLA · auditoría integral): NO SE INVENTA EL FILL.
                //
                // Aquí se imputaba como precio de salida `coin.current_price`, el
                // precio de MERCADO de hasta 60 s después del cierre real (la
                // limpieza corre en el ciclo de reconciliación). Y `remote_price`
                // es siempre 0 en esta rama —el mapa sólo se rellena con
                // posiciones ABIERTAS y aquí la remota está plana—, de modo que
                // la condición era código muerto y el precio inventado, la regla.
                // Un SL que llenó en 98 mientras el precio rebotaba a 101 se
                // contabilizaba como GANANCIA, y ese PnL ficticio iba a
                // `pnl_realized`, `win_rate`, `gross_wins/losses`, `trade_count` y
                // `unified_capital`: las métricas que alimentan la aptitud de
                // Darwin y la matriz de apalancamiento.
                //
                // La posición se cierra igual (el margen SIEMPRE se libera), pero
                // sin fill conocido no hay PnL que atribuir: el cierre real llega
                // por la contabilidad de brackets (D-701/D-702) o por
                // /fapi/v1/income, que son las fuentes con precio verdadero.
                let exit_price = remote_price;
                let (was_long, entry_p, qty, m, entry_fee_paid) =
                    coin.positions.position.close_with_fee();
                if m > 0.0 {
                    let cur_u = arena.used_margin.load(std::sync::atomic::Ordering::Relaxed);
                    arena
                        .used_margin
                        .store((cur_u - m).max(0.0), std::sync::atomic::Ordering::Relaxed);
                }

                if qty > 0.0 && exit_price > 0.0 && entry_p > 0.0 {
                    // D-701: la misma función que usa la contabilidad de brackets.
                    let gross_pnl =
                        crate::trade_accounting::gross_pnl(was_long, entry_p, exit_price, qty);
                    let live_taker = arena
                        .config
                        .live_taker_fee
                        .load(std::sync::atomic::Ordering::Relaxed)
                        .max(0.0004);
                    let close_fee = (qty * exit_price) * live_taker;
                    let net_realized_pnl = gross_pnl - close_fee;
                    let net_trade_pnl = net_realized_pnl - entry_fee_paid;

                    // D-447: coin.metrics es la fuente unificada para el espectro continuo
                    coin.metrics
                        .pnl_realized
                        .fetch_add(net_trade_pnl, std::sync::atomic::Ordering::Relaxed);
                    coin.swing
                        .pnl_realized
                        .fetch_add(net_trade_pnl, std::sync::atomic::Ordering::Relaxed);
                    coin.scalp
                        .pnl_realized
                        .fetch_add(net_trade_pnl, std::sync::atomic::Ordering::Relaxed);
                    arena
                        .unified_capital
                        .fetch_add(net_realized_pnl, std::sync::atomic::Ordering::Relaxed);

                    let is_win = net_trade_pnl > 0.0;
                    let n = coin
                        .metrics
                        .trade_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        as f64
                        + 1.0;
                    coin.swing
                        .trade_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    coin.scalp
                        .trade_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                    let old_wr = coin
                        .metrics
                        .win_rate
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let new_wr = old_wr + (((if is_win { 1.0 } else { 0.0 }) - old_wr) / n);
                    coin.metrics
                        .win_rate
                        .store(new_wr, std::sync::atomic::Ordering::Relaxed);
                    coin.swing
                        .win_rate
                        .store(new_wr, std::sync::atomic::Ordering::Relaxed);
                    coin.scalp
                        .win_rate
                        .store(new_wr, std::sync::atomic::Ordering::Relaxed);

                    if is_win {
                        coin.metrics
                            .gross_wins
                            .fetch_add(net_trade_pnl, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        coin.metrics
                            .gross_losses
                            .fetch_add(net_trade_pnl.abs(), std::sync::atomic::Ordering::Relaxed);
                    }
                }
                adjustments += 1;
            }

            // D-633 (DÉCIMA OLA): aquí se purgaban los slots `scalp` y `swing`
            // liberando margen sin contabilizar PnL. Ningún camino del sistema
            // abre ya esos slots (sólo el horizonte continuo opera), así que la
            // purga era inalcanzable y ocultaba que no existe contabilidad para
            // ellos. Se retira; si un slot volviera a abrirse, su cierre debe
            // pasar por la contabilidad de arriba, no por una liberación muda.
        } else {
            // Exchange tiene posición abierta
            if !cont_open {
                // Posición huérfana en exchange: adoptar en Horizonte Continuo
                let is_long = remote_net_qty > 0.0;
                let abs_qty = remote_net_qty.abs();
                let price = if remote_price > 0.0 {
                    remote_price
                } else {
                    coin.current_price
                        .load(std::sync::atomic::Ordering::Relaxed)
                };
                let notional = abs_qty * price;
                // S-06: usar el LEVERAGE REAL de la posición reportada por el
                // exchange (antes: /10.0 hardcoded — inflaba used_margin 2-5x
                // en cuentas 20x/50x → falsa escasez de margen).
                let lev = remote_lev_map.get(&sym).copied().unwrap_or(10.0);
                let lev = if lev.is_finite() && lev >= 1.0 {
                    lev
                } else {
                    10.0
                };
                let margin = notional / lev;
                // D-729: si la entrada no es válida (precio o cantidad), la
                // posición NO se abre y tampoco se reserva su margen.
                let adoptada = coin.positions.position.open_with_horizon(
                    is_long,
                    price,
                    abs_qty,
                    margin,
                    now_ms,
                    0.0,
                    0.0,
                    quantum_arena::position::PositionHorizon::Continuous,
                );
                if !adoptada {
                    println!(
                        "🚨 [RECONCILIACIÓN] {}: posición remota con precio {} y cantidad {} no adoptable — se audita contra el exchange en vez de inventarla",
                        sym, price, abs_qty
                    );
                    continue;
                }
                arena
                    .used_margin
                    .fetch_add(margin, std::sync::atomic::Ordering::Relaxed);
                adjustments += 1;
            } else if (arena_net_qty - remote_net_qty).abs() > 1e-6 {
                // Drift en cantidad: actualizar posición continua para reflejar el tamaño real
                let target_abs = remote_net_qty.abs();
                if target_abs <= 1e-6 {
                    let (_, _, _, old_margin, _) = coin.positions.position.close_with_fee();
                    if old_margin > 0.0 {
                        let cur_u = arena.used_margin.load(std::sync::atomic::Ordering::Relaxed);
                        arena.used_margin.store(
                            (cur_u - old_margin).max(0.0),
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                } else {
                    let price = coin
                        .positions
                        .position
                        .entry_price
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let safe_price = if price > 0.0 {
                        price
                    } else {
                        coin.current_price
                            .load(std::sync::atomic::Ordering::Relaxed)
                    };
                    let old_margin = coin
                        .positions
                        .position
                        .margin_used
                        .load(std::sync::atomic::Ordering::Relaxed);
                    // D-734 (DÉCIMA OLA · auditoría integral): la rama de DERIVA
                    // seguía dividiendo por el literal 10 mientras la rama de
                    // adopción, a 40 líneas de distancia, ya usa el apalancamiento
                    // REAL del exchange (S-06). Una cuenta a 20x veía su margen
                    // inflado al doble en cuanto la cantidad derivaba, con la
                    // falsa escasez de margen que S-06 vino a corregir.
                    let lev_deriva = remote_lev_map
                        .get(&sym)
                        .copied()
                        .filter(|l| l.is_finite() && *l >= 1.0)
                        .unwrap_or(10.0);
                    let new_margin = if safe_price > 0.0 {
                        (target_abs * safe_price) / lev_deriva
                    } else {
                        old_margin
                    };
                    let margin_diff = new_margin - old_margin;

                    coin.positions
                        .position
                        .quantity
                        .store(target_abs, std::sync::atomic::Ordering::Relaxed);
                    coin.positions
                        .position
                        .margin_used
                        .store(new_margin, std::sync::atomic::Ordering::Relaxed);
                    arena
                        .used_margin
                        .fetch_add(margin_diff, std::sync::atomic::Ordering::Relaxed);
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
        assert_eq!(
            report.open_positions.len(),
            1,
            "ETH flat no cuenta en open_positions"
        );
        assert_eq!(
            report.suspicious_active_orders.len(),
            1,
            "Solo la orden de ETH con posición plana es sospechosa"
        );
        assert!(report
            .suspicious_active_orders
            .iter()
            .any(|s| s.contains("ETHUSDT") && s.contains("plana")));
        assert!(report.summary.contains("BTCUSDT"));

        let adopted = report.apply_to_registry(&registry, 0.0005);
        assert_eq!(adopted, 1);

        // D-632: la posición adoptada lleva su comisión de entrada imputada.
        let adoptada = registry
            .get("adopted_BTCUSDT_1700000000000")
            .expect("la posición BTCUSDT debe quedar registrada");
        let esperada = 0.5 * 60000.0 * 0.0005;
        assert!(
            (adoptada.total_commission - esperada).abs() < 1e-9,
            "comisión imputada {} en lugar de {}",
            adoptada.total_commission,
            esperada
        );

        // Reconciliar de nuevo el mismo estado no duplica la comisión.
        report.apply_to_registry(&registry, 0.0005);
        let otra_vez = registry.get("adopted_BTCUSDT_1700000000000").unwrap();
        assert!((otra_vez.total_commission - esperada).abs() < 1e-9);
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

        // D-703 (DÉCIMA OLA · auditoría integral): este test desbordaba la pila y
        // hacía abortar TODA la suite del crate (STATUS_STACK_OVERFLOW), de modo
        // que los demás tests no llegaban a ejecutarse. Es el patrón de D-684: el
        // arena materializa en línea los anillos de ticks de sus monedas y no cabe
        // en la pila por defecto de un hilo de test; producción y el forense ya lo
        // construyen en un hilo de 32 MiB. Aquí, lo mismo.
        let arena = quantum_arena::GlobalArena::build_in_own_stack(100.0);

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
        arena
            .used_margin
            .store(12.0, std::sync::atomic::Ordering::Relaxed);

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
        assert!(
            adjs >= 2,
            "Must adjust phantom BTC position and adopt orphan ETH position"
        );

        // BTC phantom position must be closed and margin reclaimed
        assert!(!arena.coins[0].positions.position.is_open());

        // ETH position must be adopted in Continuous position
        assert!(arena.coins[1].positions.position.is_open());
        assert_eq!(
            arena.coins[1]
                .positions
                .position
                .quantity
                .load(std::sync::atomic::Ordering::Relaxed),
            0.5
        );
    }
}
