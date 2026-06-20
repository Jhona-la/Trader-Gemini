import numpy as np
from datetime import datetime, timezone
from config import Config
from utils.logger import logger

class AnalyticsEngine:
    """
    📊 MOTOR DE ANALÍTICA AVANZADA
    
    PROFESSOR METHOD:
    - QUÉ: Sistema de cálculo de métricas financieras institucionales.
    - POR QUÉ: Para evaluar la calidad del trading más allá del beneficio neto.
    - PARA QUÉ: Identificar si la estrategia es robusta o si el riesgo es excesivo.
    - CÓMO: Fórmulas de Sharpe, Sortino, Drawdown y Esperanza Matemática.
    """
    
    @staticmethod
    def calculate_metrics(history_df):
        """Calcula un set completo de métricas pro a partir del historial."""
        import polars as pl
        if hasattr(history_df, 'empty'):
            is_empty = history_df.empty
        else:
            is_empty = history_df.is_empty() if hasattr(history_df, 'is_empty') else len(history_df) == 0

        if is_empty or len(history_df) < 5:
            return {
                'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0,
                'win_rate': 0.0, 'profit_factor': 0.0, 'volatility': 0.0
            }
        
        try:
            # 1. Preparar Retornos
            if isinstance(history_df, dict):
                history_df = pl.DataFrame(history_df)
            
            cols = history_df.columns if hasattr(history_df, 'columns') else []
            if 'total_equity' not in cols:
                return {
                    'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0,
                    'win_rate': 0.0, 'profit_factor': 0.0, 'volatility': 0.0
                }

            if hasattr(history_df, 'select'):
                # Polars
                equity_col = history_df.select('total_equity').to_series()
                equity = equity_col.cast(pl.Float64, strict=False).drop_nulls().to_numpy()
            else:
                # pandas fallback
                equity = history_df['total_equity']
                if hasattr(equity, 'dropna'):
                    equity = equity.dropna()
                equity = np.array(equity, dtype=float)

            if len(equity) < 5:
                # Retornamos valores seguros si no hay suficientes datos
                return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0}
            
            # Manejo seguro de diff para evitar errores con array vacío
            returns = np.diff(equity) / equity[:-1]
            returns = returns[~np.isnan(returns)]
            
            if len(returns) == 0:
                return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0}

            # 2. Sharpe Ratio (Anualizado)
            # Sharpe = (Retorno Medio - Risk Free) / Std Dev
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            rf_daily = Config.Analytics.RISK_FREE_RATE / Config.Analytics.TRADING_DAYS
            
            sharpe = 0.0
            if std_return > 0:
                sharpe = (avg_return - rf_daily) / std_return * np.sqrt(Config.Analytics.TRADING_DAYS)
            
            # 3. Sortino Ratio (Solo volatilidad negativa)
            downside_returns = returns[returns < Config.Analytics.SORTINO_MIN_RETURN]
            sortino = 0.0
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    sortino = (avg_return - rf_daily) / downside_std * np.sqrt(Config.Analytics.TRADING_DAYS)
            
            # 4. Max Drawdown
            peak = np.maximum.accumulate(equity)
            drawdowns = (peak - equity) / peak
            max_dd = np.max(drawdowns)
            
            # 5. Volatilidad (Anualizada)
            volatility = std_return * np.sqrt(Config.Analytics.TRADING_DAYS)
            
            return {
                'sharpe': round(float(sharpe), 2),
                'sortino': round(float(sortino), 2),
                'max_drawdown': round(float(max_dd * 100), 2), # En %
                'volatility': round(float(volatility * 100), 2), # En %
                'avg_return_daily': round(float(avg_return * 100), 4)
            }
            
        except Exception as e:
            logger.error(f"❌ Error en AnalyticsEngine (Metrics): {e}")
            return {}

    @staticmethod
    def calculate_winrate_details(trades_df):
        """Análisis detallado de Win Rate por símbolo y estrategia."""
        is_empty = False
        if hasattr(trades_df, 'empty'): is_empty = trades_df.empty
        elif hasattr(trades_df, 'is_empty'): is_empty = trades_df.is_empty()
        else: is_empty = len(trades_df) == 0

        if is_empty:
            return {}
            
        try:
            import polars as pl
            if hasattr(trades_df, 'columns') and 'pnl' not in trades_df.columns:
                return {}
                
            if hasattr(trades_df, 'filter'):
                closed_trades = trades_df.filter(pl.col('pnl') != 0)
                is_closed_empty = closed_trades.is_empty()
            else:
                closed_trades = trades_df[trades_df['pnl'] != 0].copy()
                is_closed_empty = closed_trades.empty

            if is_closed_empty:
                return {'global_winrate': 0.0}
                
            # Win Rate Global
            if hasattr(closed_trades, 'filter'):
                wins = len(closed_trades.filter(pl.col('pnl') > 0))
                total = len(closed_trades)
                global_wr = (wins / total) * 100
                
                symbol_wr = {}
                if 'symbol' in closed_trades.columns:
                    unique_syms = closed_trades.select('symbol').unique().to_series().to_list()
                    for sym in unique_syms:
                        sym_df = closed_trades.filter(pl.col('symbol') == sym)
                        sym_wins = len(sym_df.filter(pl.col('pnl') > 0))
                        symbol_wr[sym] = round((sym_wins / len(sym_df)) * 100, 1)
                
                gross_profit = closed_trades.filter(pl.col('pnl') > 0).select('pnl').sum().item()
                gross_loss = abs(closed_trades.filter(pl.col('pnl') < 0).select('pnl').sum().item())
                
            else:
                wins = len(closed_trades[closed_trades['pnl'] > 0])
                total = len(closed_trades)
                global_wr = (wins / total) * 100
                
                # Por Símbolo
                symbol_wr = {}
                if 'symbol' in closed_trades.columns:
                    for sym in closed_trades['symbol'].unique():
                        sym_df = closed_trades[closed_trades['symbol'] == sym]
                        sym_wins = len(sym_df[sym_df['pnl'] > 0])
                        symbol_wr[sym] = round((sym_wins / len(sym_df)) * 100, 1)
                    
                # Profit Factor
                gross_profit = closed_trades[closed_trades['pnl'] > 0]['pnl'].sum()
                gross_loss = abs(closed_trades[closed_trades['pnl'] < 0]['pnl'].sum())
                
            profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')
                
            return {
                'global_winrate': round(global_wr, 1),
                'total_trades': total,
                'symbol_winrate': symbol_wr,
                'profit_factor': profit_factor
            }
        except Exception as e:
            logger.error(f"❌ Error calculando WinRate: {e}")
            return {}

    @staticmethod
    def calculate_expectancy(trades_df, filter_reverse=False):
        """
        Calcula la Esperanza Matemática ($E$) por operación.
        E = (Pw * AvgW) - (Pl * AvgL)
        
        Args:
            trades_df (pd.DataFrame): Historial de trades.
            filter_reverse (bool): Si True, solo analiza trades de reversión.
            
        Returns:
            dict: Métricas de esperanza y eficiencia.
        """
    @staticmethod
    def _to_records(df):
        if hasattr(df, 'iter_rows'): return list(df.iter_rows(named=True))
        if hasattr(df, 'to_dict'): return df.to_dict('records')
        if isinstance(df, list): return df
        return []

    @staticmethod
    def calculate_expectancy(trades_df, filter_reverse=False):
        """
        Calcula la Esperanza Matemática ($E$) por operación.
        E = (Pw * AvgW) - (Pl * AvgL)
        """
        trades = AnalyticsEngine._to_records(trades_df)
        if not trades: return {}
        try:
            valid_trades = [t for t in trades if t['pnl'] != 0]
            if len(valid_trades) < 10: return {'status': 'INSUFFICIENT_DATA'}
            wins = [t for t in valid_trades if t['pnl'] > 0]
            losses = [t for t in valid_trades if t['pnl'] < 0]
            num_trades = len(valid_trades)
            p_win = len(wins) / num_trades
            p_loss = len(losses) / num_trades
            avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0.0
            expectancy = (p_win * avg_win) - (p_loss * avg_loss)
            reward_risk_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            kelly = p_win - ((1 - p_win) / reward_risk_ratio) if reward_risk_ratio > 0 else 0.0
            return {
                'expectancy': round(expectancy, 4), 'kelly_percent': round(kelly * 100, 2),
                'avg_win': round(avg_win, 4), 'avg_loss': round(avg_loss, 4),
                'win_rate': round(p_win * 100, 1), 'reward_risk': round(reward_risk_ratio, 2),
                'status': 'OK'
            }
        except Exception as e:
            logger.error(f"❌ Error calculando Esperanza: {e}")
            return {}

    @staticmethod
    def calculate_friction(trades_df):
        """
        Calcula la Fricción Operativa (Impacto de Fees en Beneficio Bruto).
        Formula: Friction = (Fees / Gross Profit) * 100
        """
        trades = AnalyticsEngine._to_records(trades_df)
        if not trades: return {}
        try:
            total_fees = sum(t['fee'] for t in trades)
            net_pnl = sum(t['pnl'] for t in trades)
            gross_pnl = net_pnl + total_fees
            friction_pct = (total_fees / gross_pnl) * 100 if gross_pnl > 0 else 0.0
            stats = AnalyticsEngine.calculate_expectancy(trades)
            wr = stats['win_rate']
            exp = stats['expectancy']
            false_edge = True if (wr > 55 and exp < 0) else False
                
            return {
                'friction_pct': round(friction_pct, 2),
                'total_fees': round(total_fees, 4),
                'gross_pnl': round(gross_pnl, 4),
                'net_pnl': round(net_pnl, 4),
                'false_edge': false_edge
            }
        except Exception as e:
            logger.error(f"❌ Error calculando Fricción: {e}")
            return {}

    @staticmethod
    def calculate_drawdown_series(equity_series):
        """
        Calcula la serie de Drawdown a partir de una serie de Equity.
        DD_t = (Equity_t - Peak_t) / Peak_t
        
        Returns:
            pl.Series: Serie de drawdowns (valores negativos o cero).
        """
        import polars as pl
        if isinstance(equity_series, list):
            equity_series = pl.Series(equity_series)
            
        if hasattr(equity_series, 'is_empty'):
            if equity_series.is_empty():
                return pl.Series(dtype=pl.Float64)
        elif equity_series.empty:
            return pl.Series(dtype=pl.Float64)
            
        try:
            if hasattr(equity_series, 'cum_max'):
                peak = equity_series.cum_max()
                drawdown = (equity_series - peak) / peak.fill_null(1.0).map_elements(lambda x: x if x != 0 else 1.0, return_dtype=pl.Float64)
                return drawdown
            else:
                peak = equity_series.cummax()
                drawdown = (equity_series - peak) / peak.replace(0, 1) 
                return drawdown
        except Exception as e:
            logger.error(f"❌ Error calculating Drawdown Series: {e}")
            return pl.Series(dtype=pl.Float64)

    @staticmethod
    def check_rolling_expectancy(trades_df, window=20) -> dict:
        """
        Phase 6: Proactive Mathematical Gatekeeper.
        Calculates Rolling Expectancy over the last N trades.
        
        Returns:
            dict: {
                'allowed': bool, 
                'expectancy': float,
                'reason': str
            }
        """
        is_empty = False
        if hasattr(trades_df, 'empty'): is_empty = trades_df.empty
        elif hasattr(trades_df, 'is_empty'): is_empty = trades_df.is_empty()
        else: is_empty = len(trades_df) == 0

        if is_empty or len(trades_df) < 5:
            # Not enough data to judge -> Allow (Learning Phase)
            return {'allowed': True, 'expectancy': 0.0, 'reason': 'LEARNING_PHASE'}
            
        try:
            # Take last N trades
            if hasattr(trades_df, 'tail'):
                recent_trades = trades_df.tail(window)
            else:
                recent_trades = trades_df[-window:]
            
            stats = AnalyticsEngine.calculate_expectancy(recent_trades)
            e_val = stats['expectancy']
            
            if e_val > 0:
                return {'allowed': True, 'expectancy': e_val, 'reason': 'POSITIVE_EDGE'}
            else:
                # E < 0 implies the strategy is paying the market
                # Block entry
                return {'allowed': False, 'expectancy': e_val, 'reason': 'NEGATIVE_EXPECTANCY'}
                
        except Exception as e:
            logger.error(f"❌ Error Checking Rolling Expectancy: {e}")
            return {'allowed': True, 'expectancy': 0.0, 'reason': 'ERROR_OPEN'}
