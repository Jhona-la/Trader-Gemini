//! Contrato F1-098 (Ola Ω155): Aislamiento estricto multiactivo del peso Hebbiano en PerceptronGateEngine.
//!
//! TEORÍA Y DOCTRINA (D-190 / F1-098):
//! El peso hebbiano del PerceptronGateEngine debe aprender exclusivamente del PnL
//! de la moneda evaluada. Antes de F1-098, existía un desajuste de claves (el cierre
//! escribía `{sym}_hebbian_weight` y sobreescribía la clave global `perceptron_hebbian_weight`,
//! mientras que el motor buscaba `{sym}_perceptron_hebbian_weight` cayendo a la global).
//! Como resultado, una racha de pérdidas en DOGE reducía el peso global a 0.25,
//! colapsando el voto de BTC aunque BTC estuviera en racha ganadora.
//!
//! F1-098 erradica la sobreescritura global en el cierre y unifica la lectura en el
//! motor a `get_scoped_val_strict_or(sym, coin_id, "hebbian_weight", 1.0)`.
//! Este contrato certifica el aislamiento total entre activos y la inmunidad frente
//! a residuos en la clave global.

use omniscient_registry::OmniscientRegistry;
use signal_engine::perceptron_gate::PerceptronGateEngine;
use strategy_core::QuantumStrategy;
use std::sync::Arc;

#[test]
fn f1_098_perceptron_hebbian_cross_coin_isolation_contract() {
    let registry = Arc::new(OmniscientRegistry::new());

    // Configurar señal candidata idéntica para BTC (coin_id = 0) y DOGE (coin_id = 1)
    registry.set_scoped("BTCUSDT", "perceptron_candidate_signal", 0.80);
    registry.set_scoped("DOGEUSDT", "perceptron_candidate_signal", 0.80);

    let mut engine = PerceptronGateEngine::default();
    let _ = engine.init(registry.clone());

    // 1. Estado inicial neutral: ninguna moneda tiene peso registrado, ambas usan fallback 1.0
    let voto_btc_inicial = engine.evaluate_for_coin(0, "BTCUSDT");
    let voto_doge_inicial = engine.evaluate_for_coin(1, "DOGEUSDT");

    assert!(
        voto_btc_inicial > 0.0,
        "Voto BTC inicial debe ser positivo con señal 0.80: {}",
        voto_btc_inicial
    );
    assert!(
        (voto_btc_inicial - voto_doge_inicial).abs() < 1e-12,
        "Con parámetros idénticos y pesos neutrales (1.0), los votos deben ser idénticos: BTC={}, DOGE={}",
        voto_btc_inicial,
        voto_doge_inicial
    );

    // 2. Simular racha perdedora en DOGE: peso de DOGE cae a 0.25
    registry.set_for_coin(1, "hebbian_weight", 0.25);
    registry.set_scoped("DOGEUSDT", "hebbian_weight", 0.25);

    // Simular escritura residual en clave global heredada para verificar que no contamina
    registry.set("perceptron_hebbian_weight", 0.25);

    // 3. Evaluar DOGE: debe haber reducido su convicción debido al peso 0.25
    let voto_doge_perdedor = engine.evaluate_for_coin(1, "DOGEUSDT");
    assert!(
        voto_doge_perdedor < voto_doge_inicial,
        "DOGE con peso 0.25 debe tener un voto menor al inicial: {} < {}",
        voto_doge_perdedor,
        voto_doge_inicial
    );

    // 4. Evaluar BTC: DEBE PERMANECER INALTERADO (inmune a la caída de DOGE y a la clave global)
    let voto_btc_aislado = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        (voto_btc_aislado - voto_btc_inicial).abs() < 1e-12,
        "F1-098 VIOLADO: El voto de BTC fue contaminado por DOGE o por la clave global! Antes: {}, Ahora: {}",
        voto_btc_inicial,
        voto_btc_aislado
    );

    // 5. Simular racha ganadora en BTC: peso de BTC sube a 2.0
    registry.set_for_coin(0, "hebbian_weight", 2.0);
    registry.set_scoped("BTCUSDT", "hebbian_weight", 2.0);

    let voto_btc_ganador = engine.evaluate_for_coin(0, "BTCUSDT");
    assert!(
        voto_btc_ganador > voto_btc_inicial,
        "BTC con peso 2.0 debe tener un voto mayor al inicial: {} > {}",
        voto_btc_ganador,
        voto_btc_inicial
    );

    // 6. Verificar que DOGE sigue manteniendo su voto aislado
    let voto_doge_reverificado = engine.evaluate_for_coin(1, "DOGEUSDT");
    assert!(
        (voto_doge_reverificado - voto_doge_perdedor).abs() < 1e-12,
        "F1-098 VIOLADO: DOGE fue alterado por la subida de BTC! Antes: {}, Ahora: {}",
        voto_doge_perdedor,
        voto_doge_reverificado
    );
}
