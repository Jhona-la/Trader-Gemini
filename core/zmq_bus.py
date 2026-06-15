import zmq
import zmq.asyncio
import pickle
import logging
from typing import Any

logger = logging.getLogger(__name__)

class ZmqBusConfig:
    # IP and Ports for IPC on localhost
    HOST = "127.0.0.1"
    ENGINE_PULL_PORT = 5555   # Engine receives Market, Fills, Errors
    EXECUTOR_PULL_PORT = 5556 # Executor receives Orders

class ZmqNode:
    """
    Base class for a ZeroMQ Node.
    """
    def __init__(self, node_type: str):
        self.node_type = node_type
        self.context = zmq.asyncio.Context.instance()

    def serialize(self, payload: Any) -> bytes:
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

class ZmqPushNode(ZmqNode):
    """
    Asynchronous ZeroMQ PUSH Node to send messages to a bound PULL socket.
    """
    def __init__(self, target_port: int, node_type: str = "PUSH"):
        super().__init__(node_type)
        self.target_port = target_port
        self.socket = self.context.socket(zmq.PUSH)
        # Configure High Water Mark to prevent memory leaks if receiver dies
        self.socket.setsockopt(zmq.SNDHWM, 10000)
        self.socket.setsockopt(zmq.LINGER, 0) # Don't block on shutdown
        address = f"tcp://{ZmqBusConfig.HOST}:{self.target_port}"
        self.socket.connect(address)
        logger.info(f"[ZMQ {node_type}] Connected PUSH to {address}")

    async def push(self, payload: Any):
        """Asynchronously pushes a pickled payload."""
        data = self.serialize(payload)
        await self.socket.send(data)
        
    def push_sync(self, payload: Any):
        """Synchronously pushes a pickled payload (for threads)."""
        data = self.serialize(payload)
        self.socket.send(data)

class ZmqPullNode(ZmqNode):
    """
    Asynchronous ZeroMQ PULL Node to bind and receive messages from PUSH sockets.
    """
    def __init__(self, bind_port: int, node_type: str = "PULL"):
        super().__init__(node_type)
        self.bind_port = bind_port
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 20000)
        address = f"tcp://{ZmqBusConfig.HOST}:{self.bind_port}"
        self.socket.bind(address)
        logger.info(f"[ZMQ {node_type}] Bound PULL at {address}")

    async def pull(self) -> Any:
        """Asynchronously waits for and returns a deserialized payload."""
        data = await self.socket.recv()
        return self.deserialize(data)
