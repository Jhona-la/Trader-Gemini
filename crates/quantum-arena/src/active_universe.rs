//! Legacy universe admission heuristics. This module does not implement a
//! continuous spectral portfolio optimizer or manage the lifecycle of positions.
//! Checked APIs preserve failure reasons; compatibility wrappers abstain on error.

/// Historical USD-capital heuristic ceil(sqrt(capital) / 2), not an optimality
/// theorem or a proof of order feasibility. Invalid capital admits no new assets.
#[inline(always)]
pub fn max_active_coins_for_capital(capital: f64) -> usize {
    if !capital.is_finite() || capital <= 0.0 {
        return 0;
    }
    let base_coins = (capital.max(1.0).sqrt() * 0.5).ceil() as usize;
    let max_universe = crate::symbols::get_active_universe_size();
    if max_universe == 0 {
        base_coins.max(1)
    } else {
        base_coins.clamp(1, max_universe)
    }
}

/// Compatibility fields retain the two historical heuristic scores. Their names
/// do not establish temporal estimands; a continuous replacement needs evidence.
#[derive(Debug, Clone)]
pub struct CoinFitness {
    pub coin_id: usize,
    pub symbol: String,
    pub scalp_score: f64,
    pub swing_score: f64,
    pub lot_notional_usd: f64,
    pub tick_impact_usd: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniverseSelectionError {
    InvalidCapital,
    LengthMismatch,
    InvalidForcedCoin { coin_id: usize },
    InvalidForcedEvidence { coin_id: usize },
    BitmapCapacityExceeded { coin_id: usize },
}

/// Compatibility wrapper. An empty result on error is NOT permission to stop
/// managing open positions; lifecycle membership must be maintained separately.
pub fn calculate_active_universe(
    capital: f64,
    prices: &[f64],
    forced_coin_ids: &[usize],
) -> (Vec<CoinFitness>, u64) {
    try_calculate_active_universe(capital, prices, forced_coin_ids).unwrap_or_default()
}

pub fn try_calculate_active_universe(
    capital: f64,
    prices: &[f64],
    forced_coin_ids: &[usize],
) -> Result<(Vec<CoinFitness>, u64), UniverseSelectionError> {
    select_universe(capital, prices, None, forced_coin_ids)
}

/// Legacy dynamic score, not a volatility estimator or a full temporal spectrum.
pub fn calculate_dynamic_universe(
    capital: f64,
    prices: &[f64],
    volumes_usd: &[f64],
    price_changes_pct: &[f64],
    forced_coin_ids: &[usize],
) -> (Vec<CoinFitness>, u64) {
    try_calculate_dynamic_universe(
        capital,
        prices,
        volumes_usd,
        price_changes_pct,
        forced_coin_ids,
    )
    .unwrap_or_default()
}

pub fn try_calculate_dynamic_universe(
    capital: f64,
    prices: &[f64],
    volumes_usd: &[f64],
    price_changes_pct: &[f64],
    forced_coin_ids: &[usize],
) -> Result<(Vec<CoinFitness>, u64), UniverseSelectionError> {
    if prices.len() != volumes_usd.len() || prices.len() != price_changes_pct.len() {
        return Err(UniverseSelectionError::LengthMismatch);
    }
    select_universe(
        capital,
        prices,
        Some((volumes_usd, price_changes_pct)),
        forced_coin_ids,
    )
}

fn select_universe(
    capital: f64,
    prices: &[f64],
    dynamic: Option<(&[f64], &[f64])>,
    forced_coin_ids: &[usize],
) -> Result<(Vec<CoinFitness>, u64), UniverseSelectionError> {
    if !capital.is_finite() || capital <= 0.0 {
        return Err(UniverseSelectionError::InvalidCapital);
    }
    let forced: std::collections::BTreeSet<usize> = forced_coin_ids.iter().copied().collect();
    for &coin_id in &forced {
        if coin_id >= prices.len() || super::symbol_registry::try_spec(coin_id).is_none() {
            return Err(UniverseSelectionError::InvalidForcedCoin { coin_id });
        }
        if coin_id >= 64 {
            return Err(UniverseSelectionError::BitmapCapacityExceeded { coin_id });
        }
    }
    let max_coins = max_active_coins_for_capital(capital);
    let capital_per_coin = capital / max_coins.max(forced.len()) as f64;
    let mut candidates = Vec::with_capacity(prices.len());

    for (i, &price) in prices.iter().enumerate() {
        let Some(spec) = super::symbol_registry::try_spec(i) else {
            continue;
        };
        let is_forced = forced.contains(&i);
        let lot_notional = spec.min_qty * price;
        let tick_impact = spec.tick_size * spec.min_qty;
        let valid = price.is_finite()
            && price > 0.0
            && spec.step_size.is_finite()
            && spec.step_size > 0.0
            && spec.tick_size.is_finite()
            && spec.tick_size > 0.0
            && spec.min_qty.is_finite()
            && spec.min_qty > 0.0
            && spec.min_notional.is_finite()
            && spec.min_notional >= 0.0
            && spec.max_leverage > 0
            && spec.maker_fee.is_finite()
            && spec.taker_fee.is_finite()
            && (spec.maker_fee + spec.taker_fee).is_finite()
            && lot_notional.is_finite()
            && lot_notional > 0.0
            && tick_impact.is_finite()
            && tick_impact > 0.0;
        if !valid {
            if is_forced {
                return Err(UniverseSelectionError::InvalidForcedEvidence { coin_id: i });
            }
            continue;
        }
        let required_notional = lot_notional.max(spec.min_notional);
        // Necessary affordability check only: step rounding, fees, existing
        // margin, order types and venue limits still require execution validation.
        // Preserve the historical policy cap 20 but respect lower instrument caps.
        let leverage = spec.max_leverage.min(20) as f64;
        if !is_forced && required_notional / leverage > capital_per_coin {
            continue;
        }

        let accessibility = 1.0 / lot_notional.max(0.01);
        let granularity = 1.0 / tick_impact.max(0.000001);
        let fee = 1.0 / (spec.maker_fee + spec.taker_fee).max(0.0001);
        let (scalp_score, swing_score) = if let Some((volumes, changes)) = dynamic {
            if !volumes[i].is_finite() || volumes[i] < 0.0 || !changes[i].is_finite() {
                if is_forced {
                    return Err(UniverseSelectionError::InvalidForcedEvidence { coin_id: i });
                }
                continue;
            }
            let momentum = changes[i].abs() * volumes[i].max(1.0).log10();
            (
                accessibility * 5.0 + granularity * 0.5 + momentum * 2.0 + fee,
                accessibility * 20.0 + momentum * 5.0 + granularity * 0.1 + fee * 0.5,
            )
        } else {
            (
                accessibility * 10.0 + granularity * 5.0 + fee,
                accessibility * 20.0 + fee * 0.5 + granularity * 0.1,
            )
        };
        if !scalp_score.is_finite() || !swing_score.is_finite() {
            if is_forced {
                return Err(UniverseSelectionError::InvalidForcedEvidence { coin_id: i });
            }
            continue;
        }
        let rank_score = if dynamic.is_some() {
            scalp_score
        } else {
            scalp_score.max(swing_score)
        };
        candidates.push((
            is_forced,
            rank_score,
            CoinFitness {
                coin_id: i,
                symbol: spec.symbol,
                scalp_score,
                swing_score,
                lot_notional_usd: lot_notional,
                tick_impact_usd: tick_impact,
            },
        ));
    }
    // Membership obligations precede scores lexicographically: no finite bonus
    // can prove that a forced member survives an unbounded score.
    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.total_cmp(&a.1))
            .then_with(|| a.2.coin_id.cmp(&b.2.coin_id))
    });
    candidates.truncate(max_coins.max(forced.len()));
    let mut bitmap = 0u64;
    let mut selected = Vec::with_capacity(candidates.len());
    for (_, _, coin) in candidates {
        if coin.coin_id >= 64 {
            return Err(UniverseSelectionError::BitmapCapacityExceeded {
                coin_id: coin.coin_id,
            });
        }
        bitmap |= 1u64 << coin.coin_id;
        selected.push(coin);
    }
    Ok((selected, bitmap))
}

/// O(1) query for the compatibility bitmap, limited to IDs 0..63.
#[inline(always)]
pub fn is_coin_active(bitmap: u64, coin_id: usize) -> bool {
    coin_id < 64 && bitmap & (1u64 << coin_id) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Registra un universo de 30 símbolos para que el clamp superior sea determinista.
    fn setup_universe_30() {
        let symbols: Vec<String> = (0..30).map(|i| format!("SYM{i}USDT")).collect();
        crate::symbols::update_dynamic_universe(symbols);
    }

    #[test]
    fn test_micro_account_selects_few_coins() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(15) * 0.5) = ceil(1.94) = 2
        let max = max_active_coins_for_capital(15.0);
        assert_eq!(max, 2, "micro-capital debe concentrarse en pocas monedas");
    }

    #[test]
    fn test_medium_account_selects_more() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(100) * 0.5) = ceil(5.0) = 5
        let max = max_active_coins_for_capital(100.0);
        assert_eq!(max, 5, "capital medio expande el universo activo");
    }

    #[test]
    fn test_large_account_full_universe() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        setup_universe_30();
        // Fórmula: ceil(sqrt(5000) * 0.5) = 36, clamped al universo (30)
        let max = max_active_coins_for_capital(5000.0);
        assert_eq!(max, 30, "capital grande saturado al universo disponible");
    }

    #[test]
    fn test_empty_universe_does_not_panic() {
        let _guard = crate::symbols::UNIVERSE_TEST_MUTEX.lock().unwrap();
        crate::symbols::update_dynamic_universe(Vec::new());
        // Universo sin registrar: debe devolver la base sin clamping, nunca pánico.
        assert_eq!(max_active_coins_for_capital(15.0), 2);
        assert_eq!(max_active_coins_for_capital(0.0), 0);
    }

    #[test]
    fn test_bitmap_query() {
        let bitmap = 0b101; // coins 0 and 2 active
        assert!(is_coin_active(bitmap, 0));
        assert!(!is_coin_active(bitmap, 1));
        assert!(is_coin_active(bitmap, 2));
    }

    #[test]
    fn test_active_universe_with_prices() {
        setup_universe_30();
        // Registrar specs para los 8 símbolos con precio (el resto queda sin spec → skip)
        let specs: Vec<crate::symbol_registry::SymbolSpec> = (0..8)
            .map(|i| crate::symbol_registry::SymbolSpec {
                symbol: format!("SYM{i}USDT"),
                step_size: 0.001,
                tick_size: 0.01,
                min_qty: 0.01,
                min_notional: 5.0,
                max_leverage: 20,
                maker_fee: 0.0002,
                taker_fee: 0.0004,
                is_shadow: false,
            })
            .collect();
        crate::symbol_registry::update_registry(specs);

        // Precios aproximados realistas
        let mut prices = vec![0.0f64; 100];
        prices[0] = 60000.0; // BTC
        prices[1] = 3500.0; // ETH
        prices[2] = 600.0; // BNB
        prices[3] = 150.0; // SOL
        prices[4] = 0.50; // XRP
        prices[5] = 0.40; // ADA
        prices[6] = 35.0; // AVAX
        prices[7] = 0.15; // DOGE

        let (selected, _bitmap) = calculate_active_universe(50.0, &prices, &[]);
        assert!(
            !selected.is_empty(),
            "con specs registradas debe seleccionar monedas"
        );
        assert!(
            selected.len() <= 30,
            "Should select up to 30 coins, got {}",
            selected.len()
        );
        telemetry_engine::telemetry!("Selected coins for $50:");
        for c in &selected {
            telemetry_engine::telemetry!(
                "  {} score={:.2} lot_notional=${:.4}",
                c.symbol,
                c.scalp_score,
                c.lot_notional_usd
            );
        }
    }
}
