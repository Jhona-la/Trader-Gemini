import sqlite3
import pandas as pd
import json
import os

def analyze_trades():
    print("=== AUDITORÍA FORENSE DE TRADES RECIENTES ===")
    
    db_path = 'data.db'
    if not os.path.exists(db_path):
        print(f"Error: Base de datos {db_path} no encontrada.")
        return
        
    try:
        conn = sqlite3.connect(db_path)
        
        # Load trades
        query = '''
        SELECT 
            id, symbol, side, horizon, 
            entry_price, exit_price, 
            pnl_pct, exit_reason, 
            entry_time, exit_time
        FROM trades 
        ORDER BY exit_time DESC 
        LIMIT 500
        '''
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print('No se encontraron trades en la base de datos.')
            return
            
        print(f"\nAnalizando los últimos {len(df)} trades...\n")
        
        # Calcular Win Rate General
        total_trades = len(df)
        wins = len(df[df['pnl_pct'] > 0])
        wr_general = (wins / total_trades) * 100 if total_trades > 0 else 0
        
        print(f"📈 WIN RATE GENERAL: {wr_general:.2f}% ({wins} ganados / {total_trades} totales)")
        print(f"💰 PnL Promedio: {df['pnl_pct'].mean()*100:.3f}%")
        
        # Análisis por Tipo de Salida
        print("\n=== RENDIMIENTO POR TIPO DE SALIDA (EXIT REASON) ===")
        wr_df = df.groupby('exit_reason').agg(
            Total=('id', 'count'),
            Wins=('pnl_pct', lambda x: (x > 0).sum()),
            Losses=('pnl_pct', lambda x: (x <= 0).sum()),
            Avg_PnL=('pnl_pct', 'mean')
        )
        wr_df['WinRate %'] = (wr_df['Wins'] / wr_df['Total']) * 100
        wr_df['Avg_PnL %'] = wr_df['Avg_PnL'] * 100
        print(wr_df.sort_values(by='Total', ascending=False))
        
        # Análisis por Horizonte
        print("\n=== RENDIMIENTO POR HORIZONTE ===")
        hor_df = df.groupby('horizon').agg(
            Total=('id', 'count'),
            Wins=('pnl_pct', lambda x: (x > 0).sum()),
            Avg_PnL=('pnl_pct', 'mean')
        )
        hor_df['WinRate %'] = (hor_df['Wins'] / hor_df['Total']) * 100
        print(hor_df)
        
        # Análisis por Estrategia (Mapeado por Side)
        print("\n=== MUESTRA DE LOS ÚLTIMOS 10 TRADES PERDEDORES ===")
        losers = df[df['pnl_pct'] <= 0].head(10)
        print(losers[['symbol', 'side', 'horizon', 'pnl_pct', 'exit_reason']].to_string(index=False))
        
    except Exception as e:
        print(f'Error durante el análisis: {e}')
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    analyze_trades()
