#[path = "../../../src/multi_asset_orchestrator.rs"]
mod multi_asset_orchestrator;
use multi_asset_orchestrator::MultiAssetOrchestrator;
use quantum_arena::GlobalArena;

#[test]
fn unrelated_symbol_cannot_replay_cached_pair_observations() {
    let mut engine = MultiAssetOrchestrator::new(GlobalArena::build_in_own_stack(13.0));
    engine.on_tick("BTCUSDT", 100.0, 101.0, 1.0, 1.0);
    engine.on_tick("ETHUSDT", 10.0, 11.0, 1.0, 1.0);
    for _ in 0..110 {
        let (quote, signal) = engine.on_tick("SOLUSDT", 20.0, 21.0, 1.0, 1.0);
        assert!(quote.is_none() && signal.is_none());
    }
}

#[test]
fn symbol_prefix_is_not_an_instrument_identity() {
    let mut engine = MultiAssetOrchestrator::new(GlobalArena::build_in_own_stack(13.0));
    for symbol in ["BTCDOWNUSDT", "BTCUSDC", "ETHBTC", "ETHUPUSDT"] {
        let (quote, signal) = engine.on_tick(symbol, 100.0, 101.0, 1.0, 1.0);
        assert!(quote.is_none() && signal.is_none());
        assert_eq!((engine.btc_bid, engine.eth_bid), (0.0, 0.0));
    }
}

#[test]
fn locked_and_crossed_books_do_not_enter_pair_state() {
    let mut engine = MultiAssetOrchestrator::new(GlobalArena::build_in_own_stack(13.0));
    for (bid, ask) in [(100.0, 100.0), (101.0, 100.0)] {
        let (quote, signal) = engine.on_tick("BTCUSDT", bid, ask, 1.0, 1.0);
        assert!(quote.is_none() && signal.is_none());
        assert_eq!(engine.btc_bid, 0.0);
    }
}
