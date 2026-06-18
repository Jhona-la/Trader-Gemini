"""
🔬 AUDITORÍA FORENSE TOTAL DE LAS 153 FEATURES
================================================
Objetivo: Identificar EXACTAMENTE por qué el backtest pierde dinero.
Hipótesis a verificar:
  1. Features muertas (siempre 0.0 o constantes)
  2. Features con NaN/Inf que envenenan el modelo
  3. Features con varianza cero (no aportan información)
  4. Features duplicadas/redundantes (multicolinealidad)
  5. Features con valores por defecto que nunca se actualizan
  6. Proporción de features REALES vs features FANTASMA (fake data)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import queue
import time
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data.binance_loader import BinanceData
from strategies.components.feature_engineering import FeatureEngineering

print("=" * 80)
print("🔬 AUDITORÍA FORENSE: 120 FEATURES — DIAGNÓSTICO TOTAL")
print("=" * 80)

# ============================================================================
# PHASE 0: ARRANCAR MOTORES DE DATOS (NLP, MACRO)
# ============================================================================
print("\n🚀 Arrancando motores de datos (NLP, Macro)...")

# Arrancar NLP Sentiment (necesita polling activo para que no retorne ceros)
from data.news_sentiment_nlp import news_sentiment
news_sentiment.start_background()
print("  📰 NLP Sentiment: polling RSS arrancado")

# Arrancar Macro Intelligence (necesita fetch para que las macro features no sean constantes stale)
from data.macro_intelligence import macro_intelligence
try:
    macro_intelligence.start_background()
    print("  🌐 Macro Intelligence: background fetcher arrancado")
except:
    print("  ⚠️ Macro Intelligence: background fetcher no disponible (usará cache)")

# ============================================================================
# PHASE 1: CARGAR DATOS REALES DE BINANCE (5 SÍMBOLOS)
# ============================================================================
print("\n📡 Cargando datos reales de Binance (5 símbolos, 1m, 500 barras)...")
q = queue.Queue()
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]
try:
    loader = BinanceData(events_queue=q, symbol_list=symbols)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Esperar a que se carguen los datos + darle tiempo al NLP para su primer poll
print("⏳ Esperando carga de datos + primer poll NLP (15s)...")
time.sleep(15)

# ============================================================================
# PHASE 2: CONSTRUIR GRAFO DE ENJAMBRE (SWARM CORRELATOR)
# ============================================================================
print("\n🐝 Construyendo Grafo de Enjambre (Swarm Correlator)...")
from core.swarm_correlator import swarm_correlator

# Primero: cargar el líder (BTC)
df_polars = loader.get_history_polars("BTC/USDT", timeframe="1m", n=500)
if df_polars is None or df_polars.is_empty():
    print("❌ FALLO: No se pudieron cargar datos de Binance")
    sys.exit(1)

df_btc = pd.DataFrame({'close': df_polars['close'].to_numpy()})
swarm_correlator.update_leader_data(df_btc)

# Segundo: calcular entrelazamiento de CADA símbolo con el líder
for s in symbols:
    s_df = loader.get_history_polars(s, timeframe="1m", n=60)
    if s_df is not None and not s_df.is_empty():
        df_sym = s_df.to_pandas()
        corr = swarm_correlator.calculate_entanglement(s, df_sym)
        print(f"  🔗 {s}: correlation = {corr:.4f}")

# Verificar que el hipégrafo se activó
graph_test = swarm_correlator.get_hypergraph_features("BTC/USDT")
print(f"  📊 Graph features test: centrality={graph_test.get('graph_centrality', 0):.4f}, pagerank={graph_test.get('graph_pagerank', 0):.4f}")

print(f"✅ {len(df_polars)} barras cargadas para BTC/USDT")

# ============================================================================
# PHASE 3: EJECUTAR FEATURE ENGINEERING
# ============================================================================
print("\n🔧 Ejecutando FeatureEngineering.prepare_features()...")
fe = FeatureEngineering()
df = fe.prepare_features(df_polars, symbol="BTC/USDT", horizon="SCALPING", data_provider=loader)

if df is None:
    print("❌ FALLO: prepare_features() retornó None")
    sys.exit(1)

# Convert to pandas if polars
try:
    df_pd = df.to_pandas()
except:
    df_pd = df

print(f"✅ DataFrame generado: {df_pd.shape[0]} filas × {df_pd.shape[1]} columnas")

# 3. AUDITORÍA FEATURE POR FEATURE
print("\n" + "=" * 80)
print("📊 ANÁLISIS DE CADA FEATURE (última fila = lo que ve el ML en tiempo real)")
print("=" * 80)

dead_features = []      # Siempre 0.0 o NaN
constant_features = []  # Varianza = 0
nan_features = []       # Contienen NaN
inf_features = []       # Contienen Inf
real_features = []      # Features con datos reales y varianza
suspicious_features = [] # Features con valores "por defecto" sospechosos

last_row = df_pd.iloc[-1]

for i, col in enumerate(df_pd.columns):
    series = df_pd[col]
    
    # Estadísticas básicas
    n_total = len(series)
    n_nan = series.isna().sum()
    n_inf = np.isinf(series.replace([np.nan], [0.0])).sum()
    n_zero = (series == 0.0).sum()
    
    try:
        valid = series.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) > 0:
            mean_val = valid.mean()
            std_val = valid.std()
            min_val = valid.min()
            max_val = valid.max()
            last_val = last_row[col]
            unique_count = valid.nunique()
        else:
            mean_val = std_val = min_val = max_val = last_val = 0.0
            unique_count = 0
    except:
        mean_val = std_val = min_val = max_val = last_val = 0.0
        unique_count = 0
    
    # Clasificación
    pct_nan = (n_nan / n_total) * 100
    pct_zero = (n_zero / n_total) * 100
    
    status = "✅ REAL"
    category = "real"
    
    if pct_nan > 50:
        status = "💀 >50% NaN"
        category = "nan"
        nan_features.append(col)
    elif n_inf > 0:
        status = "💣 HAS INF"
        category = "inf"
        inf_features.append(col)
    elif pct_zero > 95:
        status = "☠️ MUERTA (>95% ceros)"
        category = "dead"
        dead_features.append(col)
    elif std_val == 0 or unique_count <= 1:
        status = "🧊 CONSTANTE"
        category = "constant"
        constant_features.append(col)
    elif pct_zero > 80:
        status = "⚠️ SOSPECHOSA (>80% ceros)"
        category = "suspicious"
        suspicious_features.append(col)
    else:
        real_features.append(col)
    
    print(f"  [{i+1:3d}] {col:40s} | {status:25s} | Last={last_val:12.6f} | Mean={mean_val:12.6f} | Std={std_val:10.6f} | NaN%={pct_nan:5.1f} | Zero%={pct_zero:5.1f} | Unique={unique_count}")

# 4. RESUMEN EJECUTIVO
print("\n" + "=" * 80)
print("🏆 RESUMEN EJECUTIVO: ESTADO DE LAS 153 FEATURES")
print("=" * 80)

total = len(df_pd.columns)
print(f"\n  📊 Total features:                {total}")
print(f"  ✅ FEATURES REALES (con datos):   {len(real_features)} ({len(real_features)/total*100:.1f}%)")
print(f"  ☠️ FEATURES MUERTAS (>95% ceros): {len(dead_features)} ({len(dead_features)/total*100:.1f}%)")
print(f"  🧊 FEATURES CONSTANTES:           {len(constant_features)} ({len(constant_features)/total*100:.1f}%)")
print(f"  💀 FEATURES >50% NaN:             {len(nan_features)} ({len(nan_features)/total*100:.1f}%)")
print(f"  💣 FEATURES CON INF:              {len(inf_features)} ({len(inf_features)/total*100:.1f}%)")
print(f"  ⚠️ FEATURES SOSPECHOSAS:          {len(suspicious_features)} ({len(suspicious_features)/total*100:.1f}%)")

if dead_features:
    print(f"\n  ☠️ MUERTAS: {dead_features}")
if constant_features:
    print(f"\n  🧊 CONSTANTES: {constant_features}")
if nan_features:
    print(f"\n  💀 >50% NaN: {nan_features}")
if inf_features:
    print(f"\n  💣 CON INF: {inf_features}")
if suspicious_features:
    print(f"\n  ⚠️ SOSPECHOSAS: {suspicious_features}")

# 5. CORRELACIÓN: ¿Cuántas features son redundantes?
print("\n" + "=" * 80)
print("🔗 ANÁLISIS DE REDUNDANCIA (Correlación > 0.95)")
print("=" * 80)

numeric_cols = df_pd.select_dtypes(include=[np.number]).columns
df_clean = df_pd[numeric_cols].dropna(axis=1, how='all').fillna(0)
# Remove constant columns before correlation
df_clean = df_clean.loc[:, df_clean.std() > 0]

if len(df_clean.columns) > 5:
    corr_matrix = df_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    redundant_pairs = []
    for col in upper_tri.columns:
        for idx in upper_tri.index:
            if upper_tri.loc[idx, col] > 0.95:
                redundant_pairs.append((idx, col, upper_tri.loc[idx, col]))
    
    if redundant_pairs:
        print(f"\n  ⚠️ {len(redundant_pairs)} pares de features con correlación > 0.95:")
        for f1, f2, corr in redundant_pairs[:30]:
            print(f"    {f1:35s} ↔ {f2:35s} (r={corr:.4f})")
        if len(redundant_pairs) > 30:
            print(f"    ... y {len(redundant_pairs) - 30} pares más")
    else:
        print("  ✅ No hay features redundantes (correlación > 0.95)")

# 6. DIAGNÓSTICO FINAL
print("\n" + "=" * 80)
print("🩺 DIAGNÓSTICO: ¿POR QUÉ SOMOS BASURA?")
print("=" * 80)

noise_pct = ((len(dead_features) + len(constant_features) + len(nan_features) + len(suspicious_features)) / total) * 100
real_pct = (len(real_features) / total) * 100

print(f"\n  📡 El ML recibe {total} features, pero:")
print(f"     → {len(real_features)} tienen datos REALES ({real_pct:.1f}%)")
print(f"     → {total - len(real_features)} son RUIDO, CEROS, o CONSTANTES ({noise_pct:.1f}%)")

if noise_pct > 30:
    print(f"\n  🚨 VEREDICTO: {noise_pct:.0f}% de las features son BASURA.")
    print(f"     El ML está intentando aprender de {total - len(real_features)} columnas de ruido.")
    print(f"     Esto DILUYE la señal de las {len(real_features)} features reales.")
    print(f"     Es como intentar oír una conversación en un estadio lleno.")
else:
    print(f"\n  ✅ La mayoría de features tienen datos reales ({real_pct:.1f}%).")
    print(f"     El problema podría estar en la LÓGICA DE TRADING, no en los datos.")

if redundant_pairs and len(redundant_pairs) > 10:
    print(f"\n  🔗 REDUNDANCIA MASIVA: {len(redundant_pairs)} pares con r>0.95")
    print(f"     El ML está viendo la MISMA información repetida {len(redundant_pairs)} veces.")
    print(f"     Esto causa OVERFITTING: el modelo memoriza ruido en vez de aprender patrones.")

print("\n" + "=" * 80)
print("FIN DE AUDITORÍA")
print("=" * 80)
