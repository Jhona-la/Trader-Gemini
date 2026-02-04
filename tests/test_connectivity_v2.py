import asyncio
import sys
import os
import logging
from queue import Queue

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.binance_loader import BinanceData
from config import Config
from utils.logger import setup_logger

# Setup logger
logger = setup_logger('test_connectivity')

import pytest

@pytest.mark.asyncio
async def test_websocket():
    logger.info("Testing WebSocket Connectivity...")
    
    # Mock Queue
    events_queue = Queue()
    
    # Initialize Data Loader
    symbols = ['BTC/USDT', 'ETH/USDT']
    loader = BinanceData(events_queue, symbols)
    
    # Start Socket Task
    logger.info("Starting Socket Task...")
    socket_task = asyncio.create_task(loader.start_socket())
    
    # Wait for data
    logger.info("Waiting for 10 seconds of data...")
    try:
        for i in range(10):
            await asyncio.sleep(1)
            
            # Check if data received
            for sym in symbols:
                bars = loader.get_latest_bars(sym)
                if bars:
                    latest = bars[-1]
                    logger.info(f"Received {sym}: {latest['close']} @ {latest['datetime']}")
                else:
                    logger.warning(f"No data for {sym} yet")
                    
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopping Socket...")
        await loader.stop_socket()
        socket_task.cancel()
        logger.info("Test Complete")

if __name__ == "__main__":
    # Force Testnet for safety
    Config.BINANCE_USE_TESTNET = True
    Config.BINANCE_USE_FUTURES = True
    
    try:
        asyncio.run(test_websocket())
    except KeyboardInterrupt:
        pass
