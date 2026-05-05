"""
🏗️  BACKTEST INFRASTRUCTURE v2.0 — Global Synchronized Engine
=============================================================
Módulo centralizado de infraestructura para backtest MULTI-SYMBOL.

QUÉ: Contiene BacktestDataProvider (multi-symbol timeline), BacktestPortfolio
     (shared capital, virtual ledgers), fetch_binance_data, fetch_multi_symbol_data,
     y calculate_metrics.
POR QUÉ: v1.0 procesaba monedas secuencialmente con $13 frescos cada vez,
     inflando resultados 26x. v2.0 procesa TODAS las monedas simultáneamente
     con un solo portfolio compartido — idéntico a producción (main.py).
PARA QUÉ: Que el backtest sea una predicción fiable de producción real.
CÓMO: Timeline global unificada: unir timestamps de todas las monedas en una
     secuencia ordenada, iterar minuto a minuto y emitir MarketEvents para
     TODOS los símbolos que tengan datos en ese epoch.
CUÁNDO: Importado por scripts/run_god_mode_backtest.py, optimizers, audits.
DÓNDE: core/backtest_infra.py
QUIÉN: QA Engineer + Risk Manager + Arquitecto Senior

DEPENDENCIAS CRÍTICAS:
  - Config.BINANCE_TAKER_FEE_BNB → Fee por side (0.0375%)
  - Config.BINANCE_LEVERAGE → Apalancamiento (10x)
  - Config.INITIAL_CAPITAL → Capital inicial ($13)
  - Config.POSITION_SIZE_MICRO_ACCOUNT → Sizing (30%)
  - Config.MAX_CONCURRENT_POSITIONS → Máx posiciones simultáneas (2)
  - data/data_provider.py → Clase base abstracta DataProvider

CHANGELOG v2.0:
  - BacktestDataProvider: Multi-symbol global timeline iteration
  - BacktestPortfolio: Shared capital with MAX_CONCURRENT_POSITIONS enforcement
  - BacktestPortfolio: Virtual Ledger (Scalping/Swing per symbol)
  - BacktestPortfolio: Capital partitioning Scalping/Swing 50/50
  - New: fetch_multi_symbol_data() with parallel download
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

# Ensure project root is in path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config
from core.events import MarketEvent
from data.data_provider import DataProvider
from utils.logger import logger


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — FROM CONFIG (SINGLE SOURCE OF TRUTH)
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# BBO ARCHITECTURE: DIFFERENTIATED COMMISSIONS
# QUÉ: Maker fee (LIMIT BBO) vs Taker fee (MARKET emergency).
# POR QUÉ: Con BBO architecture, 90%+ de órdenes son LIMIT → Maker fee.
#   MARKET solo se usa en emergencias (Kill Switch, chase exhaustion).
# PARA QUÉ: Backtest refleja el AHORRO REAL de fees por usar Limit BBO.
# CÓMO: COMMISSION_PCT = Maker fee por defecto. COMMISSION_TAKER para exits
#   de emergencia y SL market fallbacks.
# ═══════════════════════════════════════════════════════════════════════════════
COMMISSION_TAKER = (
    Config.BINANCE_TAKER_FEE_BNB
)  # 0.000375 (0.0375% per side) — MARKET orders
COMMISSION_MAKER = (
    Config.BINANCE_MAKER_FEE_BNB
)  # 0.0002 (0.02% per side) — LIMIT BBO orders
# Default: MAKER fee (BBO Architecture makes ~90%+ of orders LIMIT)
COMMISSION_PCT = COMMISSION_MAKER
INITIAL_CAPITAL = Config.INITIAL_CAPITAL  # 13.0
LEVERAGE = Config.BINANCE_LEVERAGE  # 10
MAX_CONCURRENT_POSITIONS = getattr(Config, "MAX_CONCURRENT_POSITIONS", 2)


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestDataProvider v2.0 — Multi-Symbol Global Timeline Engine
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestDataProvider(DataProvider):
    """
    Proveedor de datos para backtest con datos históricos MULTI-SYMBOL.

    QUÉ: Alimenta las estrategias bar-a-bar usando structured arrays NumPy,
         iterando sobre una TIMELINE GLOBAL que une los timestamps de TODAS
         las monedas en una secuencia única ordenada.
    POR QUÉ: En producción, BinanceData envía MarketEvents de TODAS las monedas
         simultáneamente. El backtest debe replicar este comportamiento exacto.
    PARA QUÉ: Paridad 1:1 con el data flow de producción (main.py + engine.py).
    CÓMO: Pre-convierte DataFrames a structured arrays por símbolo. Construye
         una timeline global (union de todos los timestamps). En cada epoch,
         emite MarketEvents para TODOS los símbolos que tienen datos en ese
         minuto específico.
    CUÁNDO: Instanciado al inicio de cada backtest global.
    DÓNDE: core/backtest_infra.py
    QUIÉN: Consumido por God Mode Backtest v2.0, optimizers, auditorías.

    DIFERENCIA CLAVE vs v1.0:
      v1.0: update_bars() → solo procesa self.symbol_list[0] → 1 moneda
      v2.0: update_bars() → avanza timeline global → emite para TODAS las monedas
    """

    _STRUCT_DTYPE = [
        ("timestamp", "i8"),
        ("open", "f4"),
        ("high", "f4"),
        ("low", "f4"),
        ("close", "f4"),
        ("volume", "f4"),
    ]

    def __init__(self, events_queue, symbol_list, historical_data):
        """
        Args:
            events_queue: Queue para eventos de mercado.
            symbol_list: Lista de símbolos (e.g., ['BTC/USDT', 'ETH/USDT', ...]).
            historical_data: dict {symbol: DataFrame con OHLCV indexado por datetime}.
        """
        self.events_queue = events_queue
        self.symbol_list = symbol_list
        # FORENSIC FIX: Do NOT store the raw DataFrame dictionary to prevent RAM leaks (OOM)
        # We only need the NumPy structured arrays which are highly memory efficient.
        self.historical_data = None 
        self.is_backtest = True

        # Pre-allocate structured arrays for Zero-Copy parity
        self.struct_data = {s: {} for s in symbol_list}

        import gc

        for s in symbol_list:
            df_1m = historical_data[s]
            self.struct_data[s]["1m"] = self._df_to_struct(df_1m)

            # Resampled timeframes for multi-TF strategies
            for tf in ["5min", "15min", "1h", "1D", "1W"]:
                df_res = (
                    df_1m.resample(tf)
                    .agg(
                        {
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        }
                    )
                    .dropna()
                )
                key = (
                    tf.lower()
                    .replace("min", "m")
                    .replace("h", "h")
                    .replace("d", "d")
                    .replace("w", "w")
                )
                self.struct_data[s][key] = self._df_to_struct(df_res)
                del df_res
                
            # Free memory immediately after struct is built for this symbol
            historical_data[s] = None
            del df_1m
            gc.collect()


        # ═══════════════════════════════════════════════════════════════
        # v2.0: BUILD GLOBAL TIMELINE
        # QUÉ: Union de TODOS los timestamps de TODAS las monedas.
        # POR QUÉ: Para iterar minuto a minuto procesando todas las monedas.
        # CÓMO: Set union → sorted numpy array.
        # ═══════════════════════════════════════════════════════════════
        all_timestamps = set()
        for s in symbol_list:
            all_timestamps.update(self.struct_data[s]["1m"]["timestamp"].tolist())
        self.global_timeline = np.sort(np.array(list(all_timestamps), dtype="i8"))
        self.current_epoch_idx = 0
        self.total_epochs = len(self.global_timeline)

        # Legacy compat
        self.current_index = 0
        self.current_time_ms = 0
        self.continue_backtest = True

        print(
            f"  📊 BacktestDataProvider v2.0: {len(symbol_list)} symbols | "
            f"{self.total_epochs:,} global epochs | "
            f"~{self.total_epochs / 1440:.1f} days of data"
        )

    def _df_to_struct(self, df):
        """Converts DataFrame to NumPy Structured Array efficiently."""
        res = np.empty(len(df), dtype=self._STRUCT_DTYPE)
        res["timestamp"] = df.index.values.astype("datetime64[ms]").astype("int64")
        res["open"] = df["open"].values
        res["high"] = df["high"].values
        res["low"] = df["low"].values
        res["close"] = df["close"].values
        res["volume"] = df["volume"].values
        return res

    def get_latest_bars(self, symbol, n=1, timeframe="1m"):
        """Retorna vista de arreglo estructurado (Ultra-Fast slicing, Zero-Copy)."""
        try:
            arr = self.struct_data[symbol][timeframe]
            idx = np.searchsorted(arr["timestamp"], self.current_time_ms, side="right")
            if idx == 0:
                return None
            start = max(0, idx - n)
            return arr[start:idx]
        except Exception:
            return None

    def get_latest_bars_5m(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, "5m")

    def get_latest_bars_15m(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, "15m")

    def get_latest_bars_1h(self, symbol, n=1):
        return self.get_latest_bars(symbol, n, "1h")

    def get_active_positions(self):
        """Mock for strategy compatibility."""
        return {}

    def get_latest_price(self, symbol):
        """Helper for exit logic."""
        bars = self.get_latest_bars(symbol, 1)
        return float(bars[-1]["close"]) if bars is not None and len(bars) > 0 else None

    def get_symbol_precision(self, symbol):
        """Mock for strategy compatibility."""
        return {"quantity": 3, "price": 2}

    def get_derivatives_metrics(self, symbol):
        return {"funding_rate": 0.0, "open_interest_change": 0.0, "ls_ratio": 1.0}

    def get_order_flow_metrics(self, symbol):
        return {
            "buy_sell_ratio": 1.0,
            "taker_buy_volume": 0.0,
            "taker_sell_volume": 0.0,
        }

    def get_hft_indicators(self, symbol):
        """Mock HFT indicators for backtest compatibility."""
        return {}

    def get_orderbook(self, symbol: str):
        """
        Mock orderbook for backtest compatibility.
        QUÉ: Retorna None porque el backtest no tiene L2 data real.
        POR QUÉ: DataProvider ABC requiere esta implementación desde Phase 7.
        PARA QUÉ: Evitar TypeError al instanciar BacktestDataProvider.
        """
        return None

    def update_bars(self):
        """
        v2.0: Avanza al siguiente epoch global y emite MarketEvents para
        TODOS los símbolos que tienen datos en ese timestamp.

        QUÉ: Itera sobre la timeline global (no sobre un símbolo).
        POR QUÉ: Replica cómo producción (BinanceData) envía eventos
             de TODAS las monedas al events_queue simultáneamente.
        PARA QUÉ: Paridad 1:1 con main.py + engine.py.
        CÓMO: Busca en el structured array de cada símbolo si existe
             una barra con el timestamp exacto del epoch actual.
        CUÁNDO: Cada iteración del loop principal del backtest.
        DÓNDE: core/backtest_infra.py
        QUIÉN: Consumido por el loop principal de run_god_mode_backtest.py.
        """
        if self.current_epoch_idx >= self.total_epochs:
            self.continue_backtest = False
            return

        self.current_time_ms = int(self.global_timeline[self.current_epoch_idx])
        self.current_epoch_idx += 1
        self.current_index = self.current_epoch_idx  # Legacy compat

        current_ts = pd.to_datetime(self.current_time_ms, unit="ms", utc=True)

        # Emit MarketEvent for EVERY symbol that has data at this epoch
        for symbol in self.symbol_list:
            arr = self.struct_data[symbol]["1m"]
            # Binary search for exact timestamp match
            idx = np.searchsorted(arr["timestamp"], self.current_time_ms, side="left")
            if idx < len(arr) and arr["timestamp"][idx] == self.current_time_ms:
                close_price = float(arr["close"][idx])
                high_price = float(arr["high"][idx])
                low_price = float(arr["low"][idx])
                self.events_queue.put(
                    MarketEvent(
                        symbol=symbol, 
                        close_price=close_price, 
                        high_price=high_price,
                        low_price=low_price,
                        timestamp=current_ts
                    )
                )


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestPortfolio v2.0 — Multi-Asset Shared Capital Portfolio
# ═══════════════════════════════════════════════════════════════════════════════


class BacktestPortfolio:
    """
    Portfolio para backtest multi-asset con capital compartido.

    QUÉ: Gestiona posiciones de MÚLTIPLES monedas con un solo pool de capital.
    POR QUÉ: En producción, Portfolio.py gestiona $13 USD divididos entre TODAS
         las monedas. v1.0 creaba un portfolio nuevo por moneda → capital duplicado.
    PARA QUÉ: Que el backtest refleje fielmente la restricción de capital real.
    CÓMO: Un solo current_capital, MAX_CONCURRENT_POSITIONS global, y virtual
         ledgers per symbol+horizon para tracking de Scalping vs Swing.
    CUÁNDO: Instanciado UNA sola vez al inicio del backtest global.
    DÓNDE: core/backtest_infra.py
    QUIÉN: Consumido por God Mode Backtest v2.0.

    DIFERENCIA CLAVE vs v1.0:
      v1.0: 1 instancia por moneda, $13 frescos cada vez
      v2.0: 1 instancia global, $13 compartidos entre 26 monedas
    """

    def __init__(self, initial_capital=None, leverage=None):
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.current_capital = self.initial_capital
        self.leverage = leverage or LEVERAGE

        # Position tracking — MULTI-SYMBOL
        self.positions = {}  # {symbol: {qty, entry, side, size_usd, sl_price, tp_price, ...}}
        self.trades = []  # Completed trades (ALL symbols)
        self.equity_curve = [self.initial_capital]
        self.timestamps = []

        # Metrics
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.winning_trades = 0
        self.losing_trades = 0

        # Per-asset P&L tracking
        self.asset_pnl = {}

        # v2.0: Global position limit (production parity with Config)
        self.max_concurrent_positions = MAX_CONCURRENT_POSITIONS

        # v2.0: Virtual Ledger for Scalping/Swing isolation
        # Key: f"{symbol}_{horizon}" → position_data
        self.virtual_ledger = {}

        # v2.0: Capital partitioning (60% Scalping / 40% Swing) - Production Parity
        self.scalping_capital_pct = 0.60
        self.swing_capital_pct = 0.40

        # [P0 FIX] Price cache for RiskManager compatibility
        # Required by risk_manager.size_position() to get current prices
        self._last_prices = {}

        # ═══════════════════════════════════════════════════════════════
        # PRODUCTION PARITY ATTRIBUTES
        # QUÉ: Atributos requeridos por RiskManager.size_position()
        # POR QUÉ: RiskManager lee directamente del portfolio para el cap global.
        # ═══════════════════════════════════════════════════════════════
        self.current_cash = self.current_capital
        self.used_margin = 0.0
        self.pending_cash = 0.0
        self._equity_cache = self.initial_capital

    def get_total_equity(self):
        return self._refresh_equity_cache()

    def _refresh_equity_cache(self):
        """
        Calcula el equity total incluyendo PnL no realizado de todos los horizontes.
        Paridad con production: portfolio.py L502-520.
        """
        equity = self.current_capital # En backtest current_capital es el balance liquidado
        
        for v_key, pos in self.virtual_ledger.items():
            qty = pos.get('quantity', 0)
            if qty != 0:
                avg_price = pos.get('avg_price', 0)
                current_price = pos.get('current_price', avg_price)
                # PnL no realizado: (current - avg) * qty
                equity += (current_price - avg_price) * qty
        
        self._equity_cache = equity
        return equity

    def update_market_price(self, symbol, price):
        """
        Actualiza el precio de mercado para un símbolo y propaga a ledgers virtuales.
        Paridad con production: portfolio.py L605-667.
        """
        if price <= 0: return

        self._last_prices[symbol] = price
        
        # 1. Actualizar posiciones físicas (agregadas por símbolo)
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos['current_price'] = price
            # Watermarks para MAE/MFE
            if 'high_water_mark' not in pos: pos['high_water_mark'] = price
            if 'low_water_mark' not in pos: pos['low_water_mark'] = price
            
            pos['high_water_mark'] = max(pos['high_water_mark'], price)
            pos['low_water_mark'] = min(pos['low_water_mark'], price) if pos['low_water_mark'] > 0 else price

        # 2. Actualizar ledgers virtuales (aislamiento por horizonte SCALPING/SWING)
        for v_key, v_pos in self.virtual_ledger.items():
            if v_key.startswith(f"{symbol}_"):
                v_pos['current_price'] = price
                if 'high_water_mark' not in v_pos: v_pos['high_water_mark'] = price
                if 'low_water_mark' not in v_pos: v_pos['low_water_mark'] = price
                
                v_pos['high_water_mark'] = max(v_pos['high_water_mark'], price)
                v_pos['low_water_mark'] = min(v_pos['low_water_mark'], price) if v_pos['low_water_mark'] > 0 else price

    @property
    def open_position_count(self):
        """Returns the number of currently open positions across ALL symbols."""
        return len(self.positions)

    def can_open_position(self, horizon="SCALPING"):
        """
        Checks if a new position can be opened based on global constraints.

        QUÉ: Valida límite global de posiciones concurrentes y capital disponible.
        POR QUÉ: En producción, Config.MAX_CONCURRENT_POSITIONS = 2 limita
             el número total de posiciones abiertas EN TODA LA CARTERA.
        PARA QUÉ: Evitar sobre-apalancamiento con capital micro ($13).
        CÓMO: Cuenta posiciones abiertas y compara con el límite global.
        """
        # 1. Global position limit
        if self.open_position_count >= self.max_concurrent_positions:
            return False

        # 2. Minimum capital check
        if self.current_capital < 5.0:
            return False

        # 3. Capital partition check
        available = self.get_available_capital(horizon)
        min_trade_size = 5.0  # Min $5 notional / leverage
        if available < min_trade_size / self.leverage:
            return False

        return True

    def get_available_capital(self, horizon="SCALPING"):
        """
        Returns capital available for a specific horizon.

        QUÉ: Calcula capital disponible descontando margen comprometido.
        POR QUÉ: Producción (portfolio.py L373-404) particiona capital
             50/50 entre Scalping y Swing.
        PARA QUÉ: Evitar que un horizonte consuma todo el capital.
        CÓMO: (total_capital × horizon_pct) - capital_ya_comprometido_en_ese_horizonte.
        """
        # Total available (minus capital locked in open positions)
        locked_capital = sum(
            pos.get("margin_used", pos["size_usd"] / self.leverage)
            for pos in self.positions.values()
        )
        total_available = max(0, self.current_capital - locked_capital)

        # Horizon partition
        if horizon == "SCALPING":
            horizon_budget = self.current_capital * self.scalping_capital_pct
        elif horizon == "SWING":
            horizon_budget = self.current_capital * self.swing_capital_pct
        else:
            horizon_budget = self.current_capital

        # Capital already locked in this horizon
        horizon_locked = sum(
            pos.get("margin_used", pos["size_usd"] / self.leverage)
            for pos in self.positions.values()
            if pos.get("metadata", {}).get("horizon") == horizon
        )
        horizon_available = max(0, horizon_budget - horizon_locked)

        return min(total_available, horizon_available)

    def get_available_cash(self, horizon: str = None):
        """
        Alias para get_available_capital para compatibilidad con RiskManager.
        """
        # Actualizar used_margin dinámicamente antes de retornar
        self.used_margin = sum(
            pos.get("margin_used", pos["size_usd"] / self.leverage)
            for pos in self.positions.values()
        )
        self.current_cash = self.current_capital
        
        return self.get_available_capital(horizon=horizon)

    def _apply_slippage(self, price, side):
        """Simulate realistic slippage (0.01% to 0.05%)."""
        slip_pct = random.uniform(0.0001, 0.0005)
        if side == "LONG":
            return price * (1 + slip_pct)
        else:
            return price * (1 - slip_pct)

    def open_position(
        self, symbol, side, price, size_usd, timestamp, sl_price=None, tp_price=None
    ):
        """Abre una posición con Slippage Simulado y Comisiones de Producción."""
        if symbol in self.positions:
            return False

        filled_price = self._apply_slippage(price, side)
        qty = size_usd / filled_price

        # Comisión de entrada — PRODUCTION PARITY
        commission = size_usd * COMMISSION_PCT
        self.current_capital -= commission

        margin_used = size_usd / self.leverage

        self.positions[symbol] = {
            "qty": qty,
            "entry": price,
            "side": side,
            "size_usd": size_usd,
            "margin_used": margin_used,
            "timestamp": timestamp,
            "metadata": None,
            "sl_price": sl_price,
            "tp_price": tp_price,
        }
        return True

    def open_position_with_metadata(
        self,
        symbol,
        side,
        price,
        size_usd,
        timestamp,
        metadata=None,
        sl_price=None,
        tp_price=None,
    ):
        """Abre posición con metadatos (strategy_id, horizon, ATR, Regime)."""
        if self.open_position(
            symbol, side, price, size_usd, timestamp, sl_price, tp_price
        ):
            if metadata:
                self.positions[symbol]["metadata"] = metadata
            return True
        return False

    def close_position(self, symbol, price, timestamp):
        """Cierra una posición existente con Comisiones de Producción."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        qty = pos["qty"]
        entry = pos["entry"]
        side = pos["side"]
        size_usd = pos["size_usd"]

        exit_side = "SHORT" if side == "LONG" else "LONG"
        filled_price = self._apply_slippage(price, exit_side)

        if side == "LONG":
            pnl_pct = (filled_price - entry) / entry
        else:
            pnl_pct = (entry - filled_price) / entry

        pnl_usd = size_usd * pnl_pct

        # Comisión de salida — PRODUCTION PARITY
        commission = size_usd * COMMISSION_PCT
        pnl_usd -= commission

        self.current_capital += pnl_usd

        trade = {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "exit": price,
            "pnl_pct": pnl_pct * 100,
            "pnl_usd": pnl_usd,
            "entry_time": pos["timestamp"],
            "exit_time": timestamp,
            "duration": (
                (timestamp - pos["timestamp"]).total_seconds() / 60
                if isinstance(timestamp, datetime)
                else 0
            ),
            "metadata": pos.get("metadata", {}),
            "exit_reason": pos.get("exit_reason", "UNKNOWN"),
        }
        self.trades.append(trade)

        if pnl_usd > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        current_dd = (self.peak_equity - self.current_capital) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        self.asset_pnl[symbol] = self.asset_pnl.get(symbol, 0.0) + pnl_usd

        del self.positions[symbol]
        return trade

    def update_equity(self, timestamp):
        """Actualiza curva de equity."""
        self.equity_curve.append(self.current_capital)
        if isinstance(timestamp, datetime):
            self.timestamps.append(timestamp)


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_binance_data — Historical Data Downloader (Single Symbol)
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_binance_data(
    symbol: str, days: int = 30, end_time: datetime = None
) -> pd.DataFrame:
    """
    Descarga datos históricos 1m de Binance mainnet (solo lectura).

    Args:
        symbol: Par de trading (e.g., 'BTC/USDT' or 'BTCUSDT').
        days: Número de días de datos a descargar.
        end_time: Tiempo final fijo para la descarga (para backtests determinísticos).

    Returns:
        DataFrame con columnas [open, high, low, close, volume] indexado
        por datetime UTC.
    """
    from binance.client import Client

    # SIEMPRE usar mainnet para datos históricos (read-only, safe)
    client = Client()

    if end_time is None:
        end_time = datetime.utcnow()

    start_time = end_time - timedelta(days=days)

    print(
        f"📡 Descargando desde {start_time.strftime('%Y-%m-%d %H:%M')} hasta {end_time.strftime('%Y-%m-%d %H:%M')} para {symbol}..."
    )

    binance_symbol = symbol.replace("/", "")

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        klines = client.get_historical_klines(
            binance_symbol,
            Client.KLINE_INTERVAL_1MINUTE,
            str(int(current_start.timestamp() * 1000)),
            str(
                int(
                    min(current_start + timedelta(hours=16), end_time).timestamp()
                    * 1000
                )
            ),
            limit=1000,
        )

        if not klines:
            break

        all_klines.extend(klines)
        current_start += timedelta(hours=16)
        time.sleep(0.2)  # Rate limit protection

    df = pd.DataFrame(
        all_klines,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["open", "high", "low", "close", "volume"]]

    print(f"✅ Descargados {len(df)} velas ({len(df) / 60 / 24:.1f} días)")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_multi_symbol_data — Parallel Multi-Symbol Downloader
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_multi_symbol_data(
    symbols: list, days: int = 30, max_workers: int = 4, end_time: datetime = None
) -> dict:
    """
    Descarga datos de MÚLTIPLES símbolos en paralelo de forma determinística si end_time se especifica.
    """
    print(f"\n{'=' * 70}")
    print(
        f"📡 DESCARGA MULTI-SYMBOL: {len(symbols)} monedas × {days} días | Hasta: {end_time.strftime('%Y-%m-%d %H:%M') if end_time else 'AHORA'}"
    )
    print(
        f"   Threads: {max_workers} | Est.: ~{len(symbols) * days * 0.5 / max_workers:.0f}s"
    )
    print(f"{'=' * 70}")

    all_data = {}
    failed = []

    def _download_one(sym):
        try:
            df = fetch_binance_data(sym, days=days, end_time=end_time)
            if df is not None and len(df) >= 500:
                return sym, df
            else:
                return sym, None
        except Exception as e:
            print(f"  ❌ Error descargando {sym}: {e}")
            return sym, None

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, sym): sym for sym in symbols}
        for future in as_completed(futures):
            sym, df = future.result()
            if df is not None:
                all_data[sym] = df
                print(f"  ✅ {sym}: {len(df):,} bars OK")
            else:
                failed.append(sym)
                print(f"  ❌ {sym}: FAILED (insufficient data)")

    elapsed = time.time() - t_start
    print(
        f"\n  📊 Descarga completada: {len(all_data)}/{len(symbols)} OK | "
        f"{len(failed)} failed | {elapsed:.1f}s"
    )

    if failed:
        print(f"  ⚠️ Símbolos fallidos: {failed}")

    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# calculate_metrics — Institutional-Grade Performance Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_metrics(portfolio: BacktestPortfolio) -> dict:
    """
    Calcula métricas de rendimiento institucionales.

    QUÉ: Sharpe, Sortino, Drawdown, Win Rate, Profit Factor.
    POR QUÉ: Validar targets antes de producción (Sharpe > 2.0, DD < 1.5%).
    PARA QUÉ: Decisión GO/NO-GO para deploy.
    CÓMO: Fórmulas estándar de la industria financiera.
    """
    trades = portfolio.trades
    equity_curve = portfolio.equity_curve

    if not trades:
        return {
            "sharpe_ratio": 0,
            "sortino_ratio": 0,
            "max_drawdown_pct": 0,
            "win_rate": 0,
            "total_return": 0,
            "avg_trade_pnl_usd": 0,
            "avg_trade_pnl_pct": 0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "profit_factor": 0,
            "avg_trade_duration_min": 0,
            "final_capital": portfolio.current_capital,
            "initial_capital": portfolio.initial_capital,
            "peak_capital": portfolio.peak_equity,
            "payoff_ratio": 0,
            "ev_per_trade": 0,
        }

    # ── Returns from equity curve ──
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]

    # ── Sharpe Ratio (annualized) ──
    sharpe = 0.0
    sortino = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        daily_returns = []
        for i in range(0, len(returns), 24):
            chunk = returns[i : i + 24]
            if len(chunk) > 0:
                daily_returns.append(np.sum(chunk))

        if len(daily_returns) > 1:
            dr = np.array(daily_returns)
            sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(365))

            # Sortino (downside only)
            neg_dr = dr[dr < 0]
            if len(neg_dr) >= 2:
                down_std = np.std(neg_dr, ddof=1)
                if down_std > 1e-10:
                    sortino = float(np.mean(dr) / down_std * np.sqrt(365))

    # ── Max Drawdown ──
    max_dd = portfolio.max_drawdown * 100

    # ── Win Rate ──
    total_trades = len(trades)
    winning = portfolio.winning_trades
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

    # ── Total Return ──
    final_capital = portfolio.current_capital
    initial = portfolio.initial_capital
    total_return = ((final_capital - initial) / initial) * 100

    # ── Average Trade P&L ──
    avg_pnl = float(np.mean([t["pnl_usd"] for t in trades]))
    avg_pnl_pct = float(np.mean([t["pnl_pct"] for t in trades]))

    # ── Profit Factor ──
    gross_profit = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gross_loss = abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # ── Average Win / Average Loss ──
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    avg_win = float(np.mean([t["pnl_usd"] for t in wins])) if wins else 0
    avg_loss = abs(float(np.mean([t["pnl_usd"] for t in losses]))) if losses else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # ── Expected Value per trade ──
    ev = 0.0
    if total_trades > 0:
        ev = (len(wins) / total_trades * avg_win) - (
            len(losses) / total_trades * avg_loss
        )

    # ── Avg trade duration ──
    avg_duration = float(np.mean([t["duration"] for t in trades]))

    # ── Per-symbol breakdown ──
    symbol_stats = {}
    for sym in set(t["symbol"] for t in trades):
        sym_trades = [t for t in trades if t["symbol"] == sym]
        sym_wins = [t for t in sym_trades if t["pnl_usd"] > 0]
        sym_pnl = sum(t["pnl_usd"] for t in sym_trades)
        symbol_stats[sym] = {
            "trades": len(sym_trades),
            "win_rate": len(sym_wins) / len(sym_trades) * 100 if sym_trades else 0,
            "pnl_usd": round(sym_pnl, 4),
        }

    return {
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 1),
        "total_return": round(total_return, 2),
        "total_trades": total_trades,
        "winning_trades": winning,
        "losing_trades": total_trades - winning,
        "avg_trade_pnl_usd": round(avg_pnl, 4),
        "avg_trade_pnl_pct": round(avg_pnl_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "payoff_ratio": round(payoff_ratio, 2),
        "ev_per_trade": round(ev, 4),
        "avg_trade_duration_min": round(avg_duration, 1),
        "final_capital": round(final_capital, 2),
        "initial_capital": initial,
        "peak_capital": round(portfolio.peak_equity, 2),
        "symbol_breakdown": symbol_stats,
    }
