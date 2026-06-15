import pytest
import numpy as np
from strategies.ml_strategy import MLStrategyHybridUltimate
from core.events import MarketEvent
import queue

def test_features_to_models_pipeline():
    """
    INT-2 & INT-3: Integración Data -> Features -> Model -> ISN
    QUÉ: Inyecta un snapshot de features crudas y verifica que las salidas (probabilidades)
         y el ISN (Indicador de Señal Neta) se deriven correctamente.
    POR QUÉ: Si hay un bug en el ensamblaje de XGBoost, el bot invertirá en random noise.
    PARA QUÉ: Congelar un camino crítico y asegurar reproducibilidad matemática.
    """
    # 1. Crear el componente (Mock_load_models para que no sobreescriba en background)
    original_load = MLStrategyHybridUltimate._load_models
    MLStrategyHybridUltimate._load_models = lambda self: None
    class MockAsyncQueue:
        def __init__(self):
            self.q = queue.Queue()
        async def put(self, item):
            self.q.put(item)
        def get(self, timeout=None):
            return self.q.get(timeout=timeout)

    events_queue = MockAsyncQueue()
    ml_strat = MLStrategyHybridUltimate(data_provider=None, events_queue=events_queue)
    ml_strat.strategy_id = "TEST_ML_STRAT"
    ml_strat.current_horizon = "SCALPING"
    MLStrategyHybridUltimate._load_models = original_load # Restore

    # Fake load of models
    class MockModel:
        def __init__(self, prob):
            self.prob = prob
        def predict_proba(self, X):
            # Retorna prob = [prob_clase_0, prob_clase_1]
            # Devuelve tantas filas como le pases
            return np.array([[1 - self.prob, self.prob] for _ in range(X.shape[0])])

    # Inyectar modelos "congelados" (Uno fuertemente Bullish, otro Débilmente Bearish)
    ml_strat.rf_model = MockModel(0.85) # 85% prob
    ml_strat.xgb_model = MockModel(0.85)
    ml_strat.gb_model = MockModel(0.85)
    if hasattr(ml_strat, 'rf_arrays'): del ml_strat.rf_arrays
    if hasattr(ml_strat, 'gb_arrays'): del ml_strat.gb_arrays # 2. Generar Features Falsas y llamar a cálculo interno

    import polars as pl
    dummy_cols = [f"feat_{i}" for i in range(122)]
    dummy_df = pl.DataFrame({c: [0.0] for c in dummy_cols})

    def mock_prepare(*args, **kwargs):
        return dummy_df

    ml_strat._prepare_features = mock_prepare
    ml_strat._feature_cols = dummy_cols
    ml_strat.is_warm = True # Bypass warm up
    ml_strat.is_trained = True # Bypassar la condición de self.is_trained
    ml_strat._get_ga_signal = lambda sym: 0.85 # Mock GA signal

    # Mock data provider para evitar AttributeError en asíncrono
    class MockDataProvider:
        def get_latest_bars(self, symbol, timeframe, n=100):
            import numpy as np
            dtype = [("timestamp", "i8"), ("open", "f4"), ("high", "f4"), ("low", "f4"), ("close", "f4"), ("volume", "f4")]
            arr = np.zeros(n, dtype=dtype)
            # Ensure strictly increasing close prices to avoid Timeframe Divergence (m5 > 0, m15 > 0)
            arr['close'] = np.linspace(1.0, 2.0, n)
            return arr
        def get_latest_bars_5m(self, symbol, n=100):
            return self.get_latest_bars(symbol, '5m', n)
        def get_latest_bars_15m(self, symbol, n=100):
            return self.get_latest_bars(symbol, '15m', n)
        def get_order_flow_metrics(self, symbol):
            return {'buy_vol': 100, 'sell_vol': 50, 'imbalance': 0.5}
        def get_hft_indicators(self, symbol):
            return {'micro_trend': 1.0, 'liquidity_imbalance': 0.5, 'order_book_depth': 100}
    ml_strat.data_provider = MockDataProvider()
    ml_strat.min_bars_to_train = 50 # Asegurar que pase el filtro

    event = MarketEvent(symbol="BTC/USDT", close_price=50000)
    ml_strat.calculate_signals(event)

    # Múltiples delays asíncronos requieren un pequeño sleep antes de leer la cola
    import time
    time.sleep(0.5)

    try:
        signal = events_queue.get(timeout=2.0)
    except queue.Empty:
        signal = None

    # La probabilidad neta debería ser 0.85 (L) vs 0.20 (S) -> Gana LONG con 0.85
    # El ISN (si está implementado así) debería reflejar esto.

    assert signal is not None, "La estrategia no retornó ninguna señal."
    assert "LONG" in str(signal.signal_type), f"Esperado LONG, se obtuvo {signal.signal_type}"
    assert signal.ml_confidence >= 0.84, f"Confianza esperada ~0.85, se obtuvo {signal.ml_confidence}"

    # Verify metadata contains ISN logic or both probabilities
    meta = signal.metadata
    assert 'prob_L' in meta, "prob_L no está en metadata"
    assert 'prob_S' in meta, "prob_S no está en metadata"
    assert meta['prob_L'] == 0.85, f"prob_L es {meta['prob_L']}"
    assert abs(meta['prob_S'] - 0.15) < 0.01, f"prob_S es {meta['prob_S']}"
    
    print("✅ [INT-2/3] Features-to-Model Pipeline Test Passed.")

if __name__ == "__main__":
    test_features_to_models_pipeline()
