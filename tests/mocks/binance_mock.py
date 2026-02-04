"""
🔌 BINANCE MOCK MODULE
======================

PROFESSOR METHOD:
- QUÉ: Simulador completo de la API de Binance para tests offline.
- POR QUÉ: Permite ejecutar tests sin conexión a internet y sin riesgo de pérdida.
- CÓMO: Clase MockBinanceClient que replica la estructura de python-binance.
- CUÁNDO: Se usa automáticamente en todos los tests via conftest.py fixtures.
- DÓNDE: tests/mocks/binance_mock.py
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


# =============================================================================
# MOCK RESPONSE STRUCTURES
# =============================================================================

@dataclass
class MockPosition:
    """
    📍 Mock Position: Simulates a Binance Futures position.
    """
    symbol: str
    position_amt: float
    entry_price: float
    unrealized_profit: float = 0.0
    notional: float = 0.0
    leverage: int = 3
    isolated: bool = True
    margin_type: str = "ISOLATED"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'positionAmt': str(self.position_amt),
            'entryPrice': str(self.entry_price),
            'unrealizedProfit': str(self.unrealized_profit),
            'notional': str(self.notional),
            'leverage': str(self.leverage),
            'isolated': self.isolated,
            'marginType': self.margin_type
        }


@dataclass
class MockOrder:
    """
    📝 Mock Order: Simulates a Binance order response.
    """
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: float
    status: str = "FILLED"
    filled_qty: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'orderId': self.order_id,
            'symbol': self.symbol,
            'side': self.side,
            'type': self.order_type,
            'origQty': str(self.quantity),
            'price': str(self.price),
            'status': self.status,
            'executedQty': str(self.filled_qty or self.quantity),
            'time': int(self.timestamp.timestamp() * 1000),
            'updateTime': int(self.timestamp.timestamp() * 1000)
        }


@dataclass
class MockAccountBalance:
    """
    💰 Mock Account Balance: Simulates Binance account state.
    """
    wallet_balance: float = 5000.0
    margin_balance: float = 5100.0
    unrealized_pnl: float = 100.0
    available_balance: float = 4500.0
    initial_margin: float = 500.0
    maint_margin: float = 50.0
    positions: List[MockPosition] = field(default_factory=list)
    
    def to_futures_account(self) -> Dict[str, Any]:
        """Convert to futures_account() response format."""
        return {
            'totalWalletBalance': str(self.wallet_balance),
            'totalMarginBalance': str(self.margin_balance),
            'totalUnrealizedProfit': str(self.unrealized_pnl),
            'availableBalance': str(self.available_balance),
            'totalInitialMargin': str(self.initial_margin),
            'totalMaintMargin': str(self.maint_margin),
            'totalPositionInitialMargin': str(self.initial_margin * 0.8),
            'totalOpenOrderInitialMargin': str(self.initial_margin * 0.2),
            'maxWithdrawAmount': str(self.available_balance * 0.95),
            'positions': [p.to_dict() for p in self.positions]
        }
    
    def to_spot_account(self) -> Dict[str, Any]:
        """Convert to get_account() response format."""
        return {
            'makerCommission': 10,
            'takerCommission': 10,
            'buyerCommission': 0,
            'sellerCommission': 0,
            'canTrade': True,
            'canWithdraw': True,
            'canDeposit': True,
            'balances': [
                {'asset': 'USDT', 'free': str(self.available_balance), 'locked': '0.00'},
                {'asset': 'BNB', 'free': '0.1', 'locked': '0.00'}
            ]
        }


# =============================================================================
# MOCK KLINE DATA
# =============================================================================

def generate_mock_klines(
    symbol: str,
    interval: str = '1m',
    count: int = 100,
    base_price: float = 100.0,
    volatility: float = 0.02
) -> List[List]:
    """
    📊 Generate realistic OHLCV kline data.
    
    Returns list of klines in Binance format:
    [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    """
    import random
    
    klines = []
    current_price = base_price
    base_time = int(datetime(2026, 2, 3, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    
    # Interval to milliseconds
    interval_ms = {
        '1m': 60000,
        '5m': 300000,
        '15m': 900000,
        '1h': 3600000,
        '4h': 14400000,
        '1d': 86400000
    }.get(interval, 60000)
    
    for i in range(count):
        open_time = base_time + (i * interval_ms)
        close_time = open_time + interval_ms - 1
        
        # Generate realistic OHLC
        open_price = current_price
        change = random.uniform(-volatility, volatility)
        close_price = open_price * (1 + change)
        
        # High and Low
        high_price = max(open_price, close_price) * (1 + random.uniform(0, volatility/2))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, volatility/2))
        
        # Volume
        volume = random.uniform(100, 10000)
        quote_volume = volume * ((open_price + close_price) / 2)
        trades = random.randint(50, 500)
        
        klines.append([
            open_time,
            f"{open_price:.8f}",
            f"{high_price:.8f}",
            f"{low_price:.8f}",
            f"{close_price:.8f}",
            f"{volume:.8f}",
            close_time,
            f"{quote_volume:.8f}",
            trades,
            f"{volume * 0.5:.8f}",
            f"{quote_volume * 0.5:.8f}",
            "0"
        ])
        
        current_price = close_price
    
    return klines


# =============================================================================
# MOCK BINANCE CLIENT
# =============================================================================

class MockBinanceClient:
    """
    🔌 Mock Binance Client: Full simulation of python-binance Client.
    
    PROFESSOR METHOD:
    - QUÉ: Reemplazo completo del Client de python-binance para tests.
    - POR QUÉ: Aísla los tests de la red y previene llamadas reales.
    - CÓMO: Implementa los mismos métodos con respuestas predefinidas.
    """
    
    def __init__(
        self,
        api_key: str = "TEST_API_KEY",
        api_secret: str = "TEST_API_SECRET",
        testnet: bool = True,
        account_balance: Optional[MockAccountBalance] = None
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Default account with XRP position
        if account_balance is None:
            xrp_position = MockPosition(
                symbol='XRPUSDT',
                position_amt=100.0,
                entry_price=0.55,
                unrealized_profit=5.00,
                notional=60.00
            )
            account_balance = MockAccountBalance(positions=[xrp_position])
        
        self.account = account_balance
        self.orders: List[MockOrder] = []
        self._order_counter = 0
        
        # Kline cache
        self._kline_cache: Dict[str, List] = {}
    
    # =========================================================================
    # CONNECTIVITY
    # =========================================================================
    
    def ping(self) -> Dict:
        """Test connectivity."""
        return {}
    
    def get_server_time(self) -> Dict:
        """Get Binance server time."""
        return {'serverTime': int(datetime.now(timezone.utc).timestamp() * 1000)}
    
    # =========================================================================
    # ACCOUNT METHODS
    # =========================================================================
    
    def futures_account(self) -> Dict[str, Any]:
        """Get futures account information."""
        return self.account.to_futures_account()
    
    def get_account(self) -> Dict[str, Any]:
        """Get spot account information."""
        return self.account.to_spot_account()
    
    def futures_account_balance(self) -> List[Dict]:
        """Get futures balances."""
        return [
            {'asset': 'USDT', 'balance': str(self.account.wallet_balance), 'crossWalletBalance': str(self.account.wallet_balance)},
            {'asset': 'BNB', 'balance': '0.1', 'crossWalletBalance': '0.1'}
        ]
    
    # =========================================================================
    # MARKET DATA
    # =========================================================================
    
    def get_klines(
        self,
        symbol: str,
        interval: str = '1m',
        limit: int = 100,
        startTime: Optional[int] = None,
        endTime: Optional[int] = None
    ) -> List[List]:
        """Get kline/candlestick data."""
        cache_key = f"{symbol}_{interval}"
        
        if cache_key not in self._kline_cache:
            # Generate based on symbol
            base_prices = {
                'BTCUSDT': 45000.0,
                'ETHUSDT': 2500.0,
                'XRPUSDT': 0.55,
                'SOLUSDT': 100.0,
                'BNBUSDT': 300.0,
                'DOGEUSDT': 0.08,
                'ADAUSDT': 0.45,
                'DOTUSDT': 7.0
            }
            base_price = base_prices.get(symbol.upper(), 100.0)
            self._kline_cache[cache_key] = generate_mock_klines(
                symbol, interval, count=500, base_price=base_price
            )
        
        klines = self._kline_cache[cache_key]
        
        # Apply filters
        if startTime:
            klines = [k for k in klines if k[0] >= startTime]
        if endTime:
            klines = [k for k in klines if k[0] <= endTime]
        
        return klines[-limit:]
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24hr ticker price."""
        klines = self.get_klines(symbol, limit=1)
        if klines:
            last = klines[-1]
            return {
                'symbol': symbol,
                'priceChange': '10.00',
                'priceChangePercent': '0.5',
                'weightedAvgPrice': last[4],
                'lastPrice': last[4],
                'bidPrice': str(float(last[4]) * 0.999),
                'askPrice': str(float(last[4]) * 1.001),
                'openPrice': last[1],
                'highPrice': last[2],
                'lowPrice': last[3],
                'volume': last[5],
                'quoteVolume': last[7]
            }
        return {'symbol': symbol, 'lastPrice': '100.0'}
    
    # =========================================================================
    # ORDER METHODS
    # =========================================================================
    
    def futures_create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: float = None,
        price: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a futures order.
        
        🚨 SECURITY: This method should be blocked by TestSecurityGuard
        in production tests.
        """
        self._order_counter += 1
        order = MockOrder(
            order_id=f"MOCK_ORDER_{self._order_counter}",
            symbol=symbol,
            side=side,
            order_type=type,
            quantity=quantity or 0.0,
            price=price or 0.0,
            status="FILLED" if type == "MARKET" else "NEW"
        )
        self.orders.append(order)
        return order.to_dict()
    
    def create_order(self, **kwargs) -> Dict[str, Any]:
        """Create a spot order (delegates to futures for simplicity)."""
        return self.futures_create_order(**kwargs)
    
    def futures_cancel_order(self, symbol: str, orderId: str) -> Dict[str, Any]:
        """Cancel an order."""
        return {'orderId': orderId, 'status': 'CANCELED'}
    
    def futures_get_open_orders(self, symbol: str = None) -> List[Dict]:
        """Get open orders."""
        orders = [o for o in self.orders if o.status == "NEW"]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return [o.to_dict() for o in orders]
    
    # =========================================================================
    # POSITION METHODS
    # =========================================================================
    
    def futures_position_information(self, symbol: str = None) -> List[Dict]:
        """Get position information."""
        positions = self.account.positions
        if symbol:
            positions = [p for p in positions if p.symbol == symbol.replace('/', '')]
        return [p.to_dict() for p in positions]
    
    def futures_change_leverage(self, symbol: str, leverage: int) -> Dict:
        """Change leverage for a symbol."""
        return {'symbol': symbol, 'leverage': leverage, 'maxNotionalValue': '10000000'}
    
    def futures_change_margin_type(self, symbol: str, marginType: str) -> Dict:
        """Change margin type for a symbol."""
        return {'code': 200, 'msg': 'success'}
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def add_position(self, position: MockPosition):
        """Add a position to the mock account."""
        self.account.positions.append(position)
        self.account.unrealized_pnl += position.unrealized_profit
    
    def clear_positions(self):
        """Clear all positions."""
        self.account.positions = []
        self.account.unrealized_pnl = 0.0
    
    def update_balance(self, wallet: float, unrealized: float = 0.0):
        """Update account balance."""
        self.account.wallet_balance = wallet
        self.account.margin_balance = wallet + unrealized
        self.account.unrealized_pnl = unrealized
        self.account.available_balance = wallet * 0.9


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_mock_client_with_xrp_position() -> MockBinanceClient:
    """
    🏭 Factory: Creates a mock client with an XRP position for testing.
    """
    xrp_position = MockPosition(
        symbol='XRPUSDT',
        position_amt=100.0,
        entry_price=0.55,
        unrealized_profit=5.00,
        notional=60.00,
        leverage=3
    )
    
    account = MockAccountBalance(
        wallet_balance=5000.0,
        margin_balance=5100.0,
        unrealized_pnl=100.0,
        available_balance=4500.0,
        positions=[xrp_position]
    )
    
    return MockBinanceClient(account_balance=account)


def create_mock_client_empty() -> MockBinanceClient:
    """
    🏭 Factory: Creates a mock client with no positions.
    """
    return MockBinanceClient(account_balance=MockAccountBalance())
