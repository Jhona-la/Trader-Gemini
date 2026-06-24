import polars as pl

def fisher_transform_expr(period=10):
    midpoints = (pl.col('high') + pl.col('low')) / 2
    max_val = midpoints.rolling_max(window_size=period + 1)
    min_val = midpoints.rolling_min(window_size=period + 1)
    
    rng = max_val - min_val
    rng = pl.when(rng == 0).then(1e-10).otherwise(rng)
    
    x = 2 * ((midpoints - min_val) / rng) - 1
    x = pl.when(x > 0.999).then(0.999).when(x < -0.999).then(-0.999).otherwise(x)
    
    fisher = 0.5 * ((1 + x) / (1 - x)).log()
    # Para llenar los primeros periodos con 0.0, usamos un coalesce con una mascara o similar, pero no es estrictamente necesario 
    # ya que rolling_max devuelve null, y lo llenaremos luego.
    return fisher.fill_null(0.0)

def true_strength_index_expr(long_period=25, short_period=13):
    mom = pl.col('close').diff()
    abs_mom = mom.abs()
    
    # EWM mean in polars
    ema1 = mom.ewm_mean(span=long_period, adjust=False)
    tsi_num = ema1.ewm_mean(span=short_period, adjust=False)
    
    ema1_abs = abs_mom.ewm_mean(span=long_period, adjust=False)
    tsi_den = ema1_abs.ewm_mean(span=short_period, adjust=False)
    
    tsi = pl.when(tsi_den != 0).then(100 * tsi_num / tsi_den).otherwise(0.0)
    return tsi

print("Expressions defined")
