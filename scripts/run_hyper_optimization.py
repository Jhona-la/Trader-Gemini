import sys
import os
import json
import logging
from datetime import datetime

# Añadir el root del proyecto al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from core.simulation import SimulationEngine, SimDataProvider
from core.genotype import Genotype
from optimization.hyper_optimizer import HyperOptimizer
from optimization.objective_function import WalkForwardValidator
from core.backtest_infra import fetch_binance_data

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HyperLauncher")

def create_simulation_runner(symbol: str, data_provider: SimDataProvider):
    """
    Crea un runner (callable) que el HyperOptimizer puede usar.
    Inyecta el diccionario config_dict temporalmente en la evaluación.
    """
    engine = SimulationEngine(data_provider)
    validator = WalkForwardValidator(k_folds=5)
    
    total_bars = len(data_provider.arrays[symbol])
    folds = validator.split_data(total_bars)
    
    def runner(sym: str, config_dict: dict):
        # 1. Aplicar variables que afectan a Config Globalmente (temporalmente)
        # Idealmente esto requiere un context manager para restaurar.
        # Aquí mapeamos al Genotype local (Nivel B)
        genotype = Genotype(symbol=sym, genes=config_dict)
        
        fold_results = []
        for start_idx, end_idx in folds:
            # Ejecutamos el motor en el slice OOS
            trades = engine.run(genotype, sym, start_idx=start_idx, end_idx=end_idx)
            
            # TODO: Idealmente agregar el 'regime' dominante en este fold. 
            # Por defecto asumimos 'UNKNOWN' si no se inyecta desde el data_provider
            fold_results.append({
                'trades': trades,
                'regime': 'UNKNOWN', # Podríamos derivarlo del start_idx, end_idx
                'start': start_idx,
                'end': end_idx
            })
            
        return fold_results
        
    return runner

def load_real_data(symbol: str, days: int):
    """
    Carga data histórica real desde Binance para alimentar la optimización.
    """
    logger.info(f"Descargando {days} días de histórico real para {symbol}...")
    df = fetch_binance_data(symbol, days=days)
    if df is None or df.empty:
        raise ValueError(f"Fallo al descargar datos históricos para {symbol}.")
        
    # FIX: Ensure timestamp field exists for SimDataProvider compatibility
    df.index.name = 'timestamp'
        
    return df

def main():
    logger.info("🚀 Iniciando Lanzamiento del PROMPT SUPREMO (Hyper-Optimizer) 🚀")
    
    # Tomar symbol de Config o predeterminado
    symbols = getattr(Config.Strategies, 'SYMBOL', ['BTC/USDT'])
    symbol = symbols[0] if isinstance(symbols, list) else symbols
    timeframe = getattr(Config.Data, 'RESOLUTION', '1m')
    
    # Profundidad del usuario
    N_RANDOM = 5000
    N_BAYES = 1000
    DAYS = 7  # Datos históricos a cargar
    
    # 1. Cargar Datos
    logger.info(f"Cargando dataset real para {symbol} {timeframe}...")
    df = load_real_data(symbol, days=DAYS)
    logger.info(f"Dataset cargado: {len(df)} velas.")
    
    # 2. Inicializar SimDataProvider
    provider = SimDataProvider({symbol: df})
    
    # 3. Construir el Runner Funcional
    runner_func = create_simulation_runner(symbol, provider)
    
    # 4. Inicializar y Ejecutar HyperOptimizer
    optimizer = HyperOptimizer(simulation_runner=runner_func)
    
    logger.info(f"Ejecutando con parámetros profundos: n_random={N_RANDOM}, n_bayes={N_BAYES}")
    best_result = optimizer.run_full_optimization(symbol, n_random=N_RANDOM, n_bayes=N_BAYES)
    
    # 5. Guardar Resultados
    os.makedirs('data/optimization_results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%md_%H%M%S")
    # Clean symbol for filename
    clean_sym = symbol.replace('/', '')
    filepath = f"data/optimization_results/hyperopt_best_{clean_sym}_{timestamp}.json"
    
    with open(filepath, 'w') as f:
        json.dump(best_result, f, indent=4)
        
    logger.info(f"✅ Resultados Supremos guardados en {filepath}")
    logger.info(f"📊 Score Final: {best_result['score']:.4f}")
    logger.info(f"⚙️ Mejores Parámetros: {json.dumps(best_result['config'], indent=2)}")

if __name__ == "__main__":
    main()
