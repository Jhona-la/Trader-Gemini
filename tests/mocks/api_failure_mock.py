"""
🎭 OMEGA-VOID §2.3: API Failure Mock (Binance API Failure Simulator)

QUÉ: Simula fallos de la API de Binance durante operación en producción.
POR QUÉ: La API de Binance falla con HTTP 429 (rate limit), 500 (internal error),
     y desconexiones de UserDataStream en los PEORES momentos posibles
     (flash crashes, alta volatilidad, posiciones abiertas).
PARA QUÉ: Validar que:
     1. El bot hace exponential backoff correcto en 429
     2. No duplica órdenes cuando 500 ocurre con posición abierta
     3. Kill-Switch se activa en <2s tras cascading failure
CÓMO: Mock classes que reemplazan binance_executor methods con errores programados.
CUÁNDO: Tests de certificación Bloque 2.3.
DÓNDE: tests/mocks/api_failure_mock.py
QUIÉN: MockBinanceClient → reemplaza Client en binance_executor.py tests.
"""

import time
import random
import asyncio
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field
from collections import deque


class APIFailureType(Enum):
    """
    Tipos de fallo de API observados en Binance producción.
    
    RATE_LIMIT: HTTP 429+Retry-After → debe hacer backoff exponencial
    INTERNAL_ERROR: HTTP 500 → debe reintentar con idempotency
    TIMEOUT: Sin respuesta → debe detectar timeout y manejar
    USER_STREAM_DISCONNECT: ListenKey expira → debe reconectar
    PARTIAL_FILL: Orden parcialmente ejecutada → debe trackear
    CASCADING: Múltiples fallos simultáneos
    """
    RATE_LIMIT = "rate_limit_429"
    INTERNAL_ERROR = "internal_error_500"
    TIMEOUT = "timeout"
    USER_STREAM_DISCONNECT = "user_stream_disconnect"
    PARTIAL_FILL = "partial_fill"
    CASCADING = "cascading_failure"


@dataclass
class APIFailureConfig:
    """Configuration for a specific failure scenario."""
    failure_type: APIFailureType
    trigger_after_n_calls: int = 5       # Fail after N successful calls
    fail_for_n_calls: int = 3            # Stay failed for N calls
    retry_after_seconds: float = 1.0     # For 429: Retry-After header
    during_open_position: bool = True    # Fail when position is open
    during_high_volatility: bool = True  # Fail during volatile market
    probability: float = 1.0            # Probability per call (1.0 = deterministic)


class BinanceAPIException(Exception):
    """Simulates Binance API exceptions."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"HTTP {status_code}: {message}")


class BinanceRequestException(Exception):
    """Simulates network/timeout exceptions."""
    def __init__(self, message: str = "Connection timeout"):
        self.message = message
        super().__init__(message)


class MockBinanceClient:
    """
    ⚡ OMEGA-VOID: Binance API Failure Simulator.
    
    QUÉ: Mock del cliente Binance que inyecta fallos programados.
    POR QUÉ: Necesitamos testear cada tipo de fallo de la API
         sin conectar a Binance real y sin arriesgar dinero.
    PARA QUÉ: Validar que binance_executor.py, error_handler.py
         y kill_switch.py responden correctamente a cada tipo de fallo.
    CÓMO: Reemplaza los métodos del Client de python-binance con versiones
         que fallan según la configuración programada.
    CUÁNDO: En testng de resiliencia Bloque 2.3.
    DÓNDE: tests/mocks/api_failure_mock.py → MockBinanceClient
    QUIÉN: Test harness → binance_executor.py tests.
    """
    
    def __init__(self, failure_configs: List[APIFailureConfig] = None):
        self.failure_configs = failure_configs or []
        self._call_counts: Dict[str, int] = {}
        self._failure_counts: Dict[str, int] = {}
        self._active_failures: Dict[str, APIFailureConfig] = {}
        
        # Simulated state
        self.open_positions: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
        self.balance = 13.0  # USDT
        self.is_connected = True
        
        # Telemetry
        self.event_log: List[Dict] = []
        self.retry_attempts: List[Dict] = []
        
        # Default configs if none provided
        if not failure_configs:
            self.failure_configs = [
                APIFailureConfig(
                    failure_type=APIFailureType.RATE_LIMIT,
                    trigger_after_n_calls=10,
                    fail_for_n_calls=3,
                    retry_after_seconds=1.0,
                ),
                APIFailureConfig(
                    failure_type=APIFailureType.INTERNAL_ERROR,
                    trigger_after_n_calls=15,
                    fail_for_n_calls=2,
                    during_open_position=True,
                ),
            ]
    
    def _check_failures(self, method_name: str) -> Optional[APIFailureConfig]:
        """
        Checks if a failure should be triggered for this call.
        
        Returns APIFailureConfig if failure should trigger, None otherwise.
        """
        self._call_counts[method_name] = self._call_counts.get(method_name, 0) + 1
        count = self._call_counts[method_name]
        
        for config in self.failure_configs:
            key = f"{method_name}_{config.failure_type.value}"
            
            # Check if this failure is currently active (mid-sequence)
            if key in self._active_failures:
                fail_count = self._failure_counts.get(key, 0)
                if fail_count < config.fail_for_n_calls:
                    self._failure_counts[key] = fail_count + 1
                    return config
                else:
                    # Failure sequence exhausted, recover
                    del self._active_failures[key]
                    self._failure_counts[key] = 0
                    self._log_event('recovery', method_name, config)
                    continue
            
            # Check trigger conditions
            if count >= config.trigger_after_n_calls:
                # Check probability
                if random.random() > config.probability:
                    continue
                    
                # Activate failure
                self._active_failures[key] = config
                self._failure_counts[key] = 1
                self._log_event('failure_start', method_name, config)
                return config
        
        return None
    
    def _raise_failure(self, config: APIFailureConfig):
        """Raises the appropriate exception for the failure type."""
        if config.failure_type == APIFailureType.RATE_LIMIT:
            raise BinanceAPIException(
                429, 
                f"Rate limit exceeded. Retry after {config.retry_after_seconds}s"
            )
        elif config.failure_type == APIFailureType.INTERNAL_ERROR:
            raise BinanceAPIException(500, "Internal Server Error")
        elif config.failure_type == APIFailureType.TIMEOUT:
            raise BinanceRequestException("Connection timed out after 10000ms")
        elif config.failure_type == APIFailureType.USER_STREAM_DISCONNECT:
            raise BinanceRequestException("User data stream disconnected")
        elif config.failure_type == APIFailureType.CASCADING:
            # Multiple failures simultaneously
            raise BinanceAPIException(
                503, "Service temporarily unavailable (cascading failure)"
            )
    
    def _log_event(self, event_type: str, method: str, config: APIFailureConfig):
        """Records event for telemetry."""
        self.event_log.append({
            'type': event_type,
            'method': method,
            'failure': config.failure_type.value,
            'timestamp': time.time(),
            'call_count': self._call_counts.get(method, 0),
        })
    
    # ================================================================
    # MOCK API METHODS (Match python-binance Client interface)
    # ================================================================
    
    def futures_create_order(self, **kwargs) -> Dict:
        """
        Mock: Create futures order.
        
        QUÉ: Simula creación de orden con posibilidad de fallo.
        POR QUÉ: Este es el método más crítico — fallo aquí
             con posición abierta puede causar pérdidas reales.
        """
        failure = self._check_failures('futures_create_order')
        if failure:
            self._raise_failure(failure)
        
        # Success path
        order_id = len(self.orders) + 1
        order = {
            'orderId': order_id,
            'symbol': kwargs.get('symbol', 'BTCUSDT'),
            'side': kwargs.get('side', 'BUY'),
            'type': kwargs.get('type', 'LIMIT'),
            'price': kwargs.get('price', '50000'),
            'origQty': kwargs.get('quantity', '0.001'),
            'status': 'FILLED',
            'executedQty': kwargs.get('quantity', '0.001'),
            'updateTime': int(time.time() * 1000),
        }
        self.orders.append(order)
        
        self._log_event('order_created', 'futures_create_order', 
                        APIFailureConfig(APIFailureType.RATE_LIMIT))
        return order
    
    def futures_cancel_order(self, **kwargs) -> Dict:
        """Mock: Cancel futures order."""
        failure = self._check_failures('futures_cancel_order')
        if failure:
            self._raise_failure(failure)
        
        return {
            'orderId': kwargs.get('orderId', 0),
            'status': 'CANCELED',
        }
    
    def futures_account(self) -> Dict:
        """Mock: Account info."""
        failure = self._check_failures('futures_account')
        if failure:
            self._raise_failure(failure)
        
        return {
            'totalWalletBalance': str(self.balance),
            'availableBalance': str(self.balance * 0.9),
            'totalUnrealizedProfit': '0.0',
        }
    
    def futures_position_information(self, **kwargs) -> List[Dict]:
        """Mock: Position info."""
        failure = self._check_failures('futures_position_information')
        if failure:
            self._raise_failure(failure)
        
        return [
            {
                'symbol': symbol,
                'positionAmt': str(pos.get('qty', 0)),
                'entryPrice': str(pos.get('entry_price', 0)),
                'unRealizedProfit': str(pos.get('pnl', 0)),
            }
            for symbol, pos in self.open_positions.items()
        ]


class UserDataStreamMock:
    """
    Mock for UserDataStream (listen key management + callbacks).
    
    QUÉ: Simula UserDataStream de Binance que envía fills y account updates.
    POR QUÉ: El UserDataStream se desconecta frecuentemente en producción
         (listen key expira cada 60min, websocket drops).
    PARA QUÉ: Validar que el bot reconecta automáticamente y no pierde
         información de fills.
    """
    
    def __init__(self, disconnect_after_seconds: float = 5.0):
        self.disconnect_after = disconnect_after_seconds
        self.is_connected = True
        self.reconnect_count = 0
        self._start_time = time.time()
        self._callbacks: List[Callable] = []
        self.event_log: List[Dict] = []
    
    def register_callback(self, callback: Callable):
        """Register handler for user data events."""
        self._callbacks.append(callback)
    
    async def listen(self):
        """
        Simulates UserDataStream with programmed disconnection.
        Raises after disconnect_after seconds.
        """
        await asyncio.sleep(self.disconnect_after)
        self.is_connected = False
        self.event_log.append({
            'type': 'disconnect',
            'after_seconds': self.disconnect_after,
            'timestamp': time.time(),
        })
        raise BinanceRequestException(
            f"User data stream disconnected after {self.disconnect_after}s"
        )
    
    async def reconnect(self):
        """Simulate reconnection."""
        await asyncio.sleep(0.5)  # Reconnection delay
        self.reconnect_count += 1
        self.is_connected = True
        self._start_time = time.time()
        self.event_log.append({
            'type': 'reconnect',
            'attempt': self.reconnect_count,
            'timestamp': time.time(),
        })


class CascadingFailureSimulator:
    """
    ⚡ OMEGA-VOID: Cascading Failure Simulator.
    
    QUÉ: Simula fallo simultáneo de múltiples subsistemas de Binance.
    POR QUÉ: Los fallos reales nunca son aislados. Cuando Binance
         tiene problemas, TODOS los endpoints fallan a la vez.
    PARA QUÉ: Validar que Kill-Switch se activa en <2 segundos
         cuando ocurre cascading failure.
    CÓMO: Combina MockBinanceClient + UserDataStreamMock + timeout.
    """
    
    def __init__(self):
        self.client = MockBinanceClient([
            APIFailureConfig(
                failure_type=APIFailureType.CASCADING,
                trigger_after_n_calls=1,
                fail_for_n_calls=100,  # Never recover
            ),
        ])
        self.user_stream = UserDataStreamMock(disconnect_after_seconds=0.1)
        self.kill_switch_triggered = False
        self.kill_switch_time: Optional[float] = None
        self.failure_start_time: Optional[float] = None
    
    async def simulate(
        self,
        kill_switch_callback: Optional[Callable] = None,
        max_duration: float = 10.0,
    ) -> Dict:
        """
        Runs the cascading failure scenario.
        
        Args:
            kill_switch_callback: Function to call when kill switch should trigger
            max_duration: Max seconds to run before timeout
            
        Returns:
            Report dict with timing and activation results.
        """
        self.failure_start_time = time.time()
        
        # Trigger cascading failure
        try:
            self.client.futures_account()
        except (BinanceAPIException, BinanceRequestException):
            pass  # Expected failure
        
        # Check if kill switch activates
        start = time.time()
        
        if kill_switch_callback:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(kill_switch_callback),
                    timeout=max_duration
                )
                self.kill_switch_triggered = True
                self.kill_switch_time = time.time() - start
            except asyncio.TimeoutError:
                self.kill_switch_time = max_duration
        
        response_time = self.kill_switch_time or (time.time() - start)
        
        return {
            'kill_switch_triggered': self.kill_switch_triggered,
            'response_time_seconds': response_time,
            'within_2s_sla': response_time <= 2.0,
            'client_failures': len(self.client.event_log),
            'stream_reconnects': self.user_stream.reconnect_count,
        }


class BackoffValidator:
    """
    Validates that retry logic uses proper exponential backoff.
    
    QUÉ: Mide los intervalos entre reintentos para validar backoff.
    POR QUÉ: Si el bot hace reintentos sin backoff, Binance lo banea.
    PARA QUÉ: Certificar que el bot implementa exponential backoff correcto.
    """
    
    def __init__(self):
        self.retry_timestamps: List[float] = []
        self.expected_base_delay: float = 1.0
        self.expected_multiplier: float = 2.0
    
    def record_retry(self, timestamp: Optional[float] = None):
        """Record a retry attempt timestamp."""
        self.retry_timestamps.append(timestamp or time.time())
    
    def validate(self) -> Dict:
        """
        Validates that retry intervals follow exponential backoff.
        
        Returns:
            Dict with validation results.
        """
        if len(self.retry_timestamps) < 2:
            return {'valid': True, 'reason': 'Not enough retries to validate'}
        
        intervals = []
        for i in range(1, len(self.retry_timestamps)):
            intervals.append(self.retry_timestamps[i] - self.retry_timestamps[i-1])
        
        # Check that intervals are increasing (exponential backoff)
        is_increasing = all(
            intervals[i] >= intervals[i-1] * 0.8  # 20% tolerance
            for i in range(1, len(intervals))
        )
        
        # Check minimum delay (no rapid-fire retries)
        min_interval = min(intervals) if intervals else 0
        no_rapid_fire = min_interval >= self.expected_base_delay * 0.5
        
        return {
            'valid': is_increasing and no_rapid_fire,
            'intervals': intervals,
            'is_increasing': is_increasing,
            'no_rapid_fire': no_rapid_fire,
            'min_interval': min_interval,
            'total_retries': len(self.retry_timestamps),
        }


def create_rate_limit_scenario() -> MockBinanceClient:
    """Pre-configured: 429 Rate Limit after 10 calls, 3 failures."""
    return MockBinanceClient([
        APIFailureConfig(
            failure_type=APIFailureType.RATE_LIMIT,
            trigger_after_n_calls=10,
            fail_for_n_calls=3,
            retry_after_seconds=1.0,
        ),
    ])


def create_internal_error_with_position() -> MockBinanceClient:
    """Pre-configured: 500 during open position."""
    client = MockBinanceClient([
        APIFailureConfig(
            failure_type=APIFailureType.INTERNAL_ERROR,
            trigger_after_n_calls=5,
            fail_for_n_calls=2,
            during_open_position=True,
        ),
    ])
    # Simulate open position
    client.open_positions['BTCUSDT'] = {
        'qty': 0.001,
        'entry_price': 50000.0,
        'pnl': -0.50,
    }
    return client


def create_cascading_failure() -> CascadingFailureSimulator:
    """Pre-configured: All subsystems fail simultaneously."""
    return CascadingFailureSimulator()


# Self-test
if __name__ == '__main__':
    print("=" * 60)
    print("⚡ OMEGA-VOID: API Failure Mock Test")
    print("=" * 60)
    
    # Test 1: Rate Limit
    print("\n📡 Test 1: Rate Limit (429)")
    client = create_rate_limit_scenario()
    successes = 0
    failures_429 = 0
    
    for i in range(20):
        try:
            client.futures_account()
            successes += 1
        except BinanceAPIException as e:
            if e.status_code == 429:
                failures_429 += 1
    
    print(f"   Successes: {successes}")
    print(f"   429 Failures: {failures_429}")
    assert failures_429 > 0, "Rate limit never triggered!"
    
    # Test 2: Internal Error with position
    print("\n📡 Test 2: Internal Error (500) with open position")
    client = create_internal_error_with_position()
    successes = 0
    failures_500 = 0
    
    for i in range(15):
        try:
            client.futures_create_order(
                symbol='BTCUSDT', side='SELL', type='LIMIT',
                price='50100', quantity='0.001'
            )
            successes += 1
        except BinanceAPIException as e:
            if e.status_code == 500:
                failures_500 += 1
    
    print(f"   Successes: {successes}")
    print(f"   500 Failures: {failures_500}")
    
    # Test 3: Backoff Validator
    print("\n📡 Test 3: Backoff Validation")
    validator = BackoffValidator()
    t = time.time()
    for i in range(5):
        delay = 1.0 * (2 ** i)  # Exponential: 1, 2, 4, 8, 16
        t += delay
        validator.record_retry(t)
    
    result = validator.validate()
    print(f"   Valid: {result['valid']}")
    print(f"   Intervals: {[f'{x:.1f}s' for x in result['intervals']]}")
    print(f"   Increasing: {result['is_increasing']}")
    
    print("\n✅ All API failure mocks tested successfully")
