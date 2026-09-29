//! CL-18 — el bosque sombra no publica umbrales que su calibración no puede
//! identificar. Su clasificador (smartcore 0.3.2) sólo devuelve la clase:
//! todos los umbrales de la rejilla filtran lo mismo y el «óptimo» salía
//! siempre (0,50; 0,50), que el daemon escribía cada 500 ms en los genes del
//! gate ML (lift mínimo).

use evolution_engine::online_random_forest::TrueOnlineRandomForest;

#[test]
fn cl18_una_calibracion_plana_no_publica_umbrales() {
    let forest = TrueOnlineRandomForest::new(1_000);
    for i in 0..120 {
        let x = i as f64;
        let ganadora = i % 3 != 0;
        let features = [
            if i % 2 == 0 { 0.4 } else { -0.4 },
            (x * 0.37).sin(),
            1.0 + (x * 0.11).cos(),
            0.002 + 0.0001 * (x * 0.23).sin(),
            0.5 + 0.1 * (x * 0.07).sin(),
            (x * 0.05).cos(),
        ];
        let pnl = if ganadora { 0.004 } else { -0.003 };
        forest.shadow_evaluate_with_features(features, pnl);
    }
    forest.retrain_models().expect("hay datos de las dos clases");
    assert!(forest.is_trained());
    assert_eq!(
        forest.get_optimal_thresholds(),
        None,
        "con salidas de clase 0/1 ningún umbral discrimina: no hay nada que publicar"
    );
}

#[test]
fn cl18_sin_entrenar_no_hay_umbrales() {
    let forest = TrueOnlineRandomForest::new(100);
    assert_eq!(forest.get_optimal_thresholds(), None);
}
