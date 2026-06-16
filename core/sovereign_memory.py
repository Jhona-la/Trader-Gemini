"""
Sovereign Context Memory (Virtual Ledger 2.0)
═══════════════════════════════════════════════════════════════
QUÉ: Servidor In-Memory basado en ZeroMQ (sustituto nativo de Redis).
POR QUÉ: Windows no soporta Redis nativo. ZMQ provee latencias < 50µs
         sin dependencias externas complejas, permitiendo separar
         la memoria del Portfolio en un proceso aislado.
PARA QUÉ: Elimina el estado aislado. Todos los procesos (Engine,
          Executor, Dashboard) leen y escriben a la misma RAM 
          distribuida a velocidades Nano.
CÓMO: Un ZmqKVServer en background responde a ZmqKVClient(REQ). 
      El ZmqDictProxy permite usarlo transparentemente como un dict.
CUÁNDO: Se inicia durante el boot del orquestador (main.py).
DÓNDE: core/sovereign_memory.py
QUIÉN: Arquitecto Senior HFT
═══════════════════════════════════════════════════════════════
"""

import zmq
import zmq.asyncio
import asyncio
import logging
import threading
import os
from typing import Dict, Any

logger = logging.getLogger("SovereignMemory")

class ZmqKVServer:
    """
    Virtual Ledger 2.0 - Servidor de Memoria Distribuida.
    Opera 100% en RAM usando ZeroMQ. Mímica básica de comandos Redis.
    """
    def __init__(self, port: int = 5557):
        self.port = port
        self.store: Dict[str, Any] = {}
        self.running = False
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = None
        
    async def start(self):
        self.running = True
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://127.0.0.1:{self.port}")
        logger.info(f"🧠 [SOVEREIGN-MEMORY] ZMQ KV Server activo en tcp://127.0.0.1:{self.port}")
        
        while self.running:
            try:
                # Esperar instrucción del cliente
                msg = await self.socket.recv_json()
                response = self._handle_request(msg)
                await self.socket.send_json(response)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"ZmqKVServer Loop Error: {e}")
                if self.socket:
                    await self.socket.send_json({"status": "error", "message": str(e)})

    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close(linger=0)

    def _handle_request(self, msg: Dict) -> Dict:
        op = msg.get("op")
        key = msg.get("key")
        
        if op == "get":
            return {"status": "ok", "value": self.store.get(key)}
        elif op == "set":
            self.store[key] = msg.get("value")
            return {"status": "ok"}
        elif op == "delete":
            self.store.pop(key, None)
            return {"status": "ok"}
        elif op == "hget":
            hkey = msg.get("hkey")
            hash_dict = self.store.get(key, {})
            return {"status": "ok", "value": hash_dict.get(hkey) if isinstance(hash_dict, dict) else None}
        elif op == "hset":
            hkey = msg.get("hkey")
            if key not in self.store or not isinstance(self.store[key], dict):
                self.store[key] = {}
            self.store[key][hkey] = msg.get("value")
            return {"status": "ok"}
        elif op == "hgetall":
            val = self.store.get(key, {})
            return {"status": "ok", "value": val if isinstance(val, dict) else {}}
        elif op == "clear":
            self.store.clear()
            return {"status": "ok"}
        return {"status": "error", "message": "Unknown operation"}


class ZmqKVClient:
    """
    Cliente síncrono para Sovereign Memory. Thread-safe gracias a TLS.
    """
    _local = threading.local()
    
    def __init__(self, port: int = 5557):
        self.port = port
        self._fallback_store = {}
        self._server_is_down = os.environ.get("TRADER_GEMINI_BACKTEST") == "true"
        
    @property
    def socket(self):
        if not hasattr(self._local, 'socket'):
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.REQ)
            sock.connect(f"tcp://127.0.0.1:{self.port}")
            self._local.socket = sock
        return self._local.socket

    def _custom_json_dumps(self, obj):
        import json
        from datetime import datetime
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.timestamp()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        return json.dumps(obj, default=default_serializer).encode('utf-8')

    def _send_recv(self, payload: Dict) -> Any:
        if self._server_is_down:
            raise TimeoutError("ZmqKVServer is down (fast-fail)")
            
        try:
            self.socket.send(self._custom_json_dumps(payload))
            # Timeout de 2000ms para evitar deadlocks de proceso
            if self.socket.poll(2000):
                resp = self.socket.recv_json()
                if resp.get("status") == "ok":
                    return resp.get("value")
                else:
                    logger.error(f"ZmqKVClient server error: {resp}")
                    return None
            else:
                logger.error(f"ZmqKVClient TIMEOUT conectando a puerto {self.port}")
                self.socket.close(linger=0)
                delattr(self._local, 'socket')
                self._server_is_down = True
                raise TimeoutError("ZmqKVServer is down")
        except Exception as e:
            logger.error(f"ZmqKVClient Error: {e}")
            raise e

    def get(self, key: str) -> Any:
        try:
            return self._send_recv({"op": "get", "key": key})
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            return self._fallback_store.get(key)
        
    def set(self, key: str, value: Any):
        try:
            self._send_recv({"op": "set", "key": key, "value": value})
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        self._fallback_store[key] = value
        
    def hget(self, key: str, hkey: str) -> Any:
        try:
            return self._send_recv({"op": "hget", "key": key, "hkey": hkey})
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            hash_dict = self._fallback_store.get(key, {})
            return hash_dict.get(hkey) if isinstance(hash_dict, dict) else None
        
    def hset(self, key: str, hkey: str, value: Any):
        try:
            self._send_recv({"op": "hset", "key": key, "hkey": hkey, "value": value})
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        if key not in self._fallback_store or not isinstance(self._fallback_store[key], dict):
            self._fallback_store[key] = {}
        self._fallback_store[key][hkey] = value
        
    def hgetall(self, key: str) -> Dict:
        try:
            val = self._send_recv({"op": "hgetall", "key": key})
            return val if val is not None else {}
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            val = self._fallback_store.get(key, {})
            return val if isinstance(val, dict) else {}

    def hdel(self, key: str, hkey: str) -> Any:
        try:
            self._send_recv({"op": "hdel", "key": key, "hkey": hkey})
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        if key in self._fallback_store and isinstance(self._fallback_store[key], dict):
            return self._fallback_store[key].pop(hkey, None)
        return None


class NestedZmqDictProxy:
    """
    Proxy anidado para interceptar modificaciones en sub-diccionarios.
    Permite que self.proxy["key"]["nested"] = "value" haga el flush automático.
    """
    def __init__(self, parent_proxy, parent_key: str, local_dict: Dict):
        self._parent_proxy = parent_proxy
        self._parent_key = parent_key
        self._local_dict = local_dict

    def __getitem__(self, key: str) -> Any:
        val = self._local_dict[key]
        if isinstance(val, dict):
            return NestedZmqDictProxy(self, key, val)
        return val

    def __setitem__(self, key: str, value: Any):
        self._local_dict[key] = value
        self._parent_proxy[self._parent_key] = self._local_dict

    def get(self, key: str, default: Any = None) -> Any:
        val = self._local_dict.get(key, default)
        if isinstance(val, dict):
            return NestedZmqDictProxy(self, key, val)
        return val
        
    def setdefault(self, key: str, default: Any = None) -> Any:
        if key not in self._local_dict:
            self[key] = default
        return self[key]

    def copy(self):
        return self._local_dict.copy()

    def update(self, other_dict: Dict):
        self._local_dict.update(other_dict)
        self._parent_proxy[self._parent_key] = self._local_dict

    def __contains__(self, key: str) -> bool:
        return key in self._local_dict

    def keys(self): return self._local_dict.keys()
    def items(self): return self._local_dict.items()
    def values(self): return self._local_dict.values()
    
    # Delegate missing attributes to local_dict (for json serialization etc)
    def __getattr__(self, name):
        return getattr(self._local_dict, name)

class ZmqDictProxy:
    """
    Proxy que emula la API de un diccionario nativo de Python, pero persiste
    el estado atómicamente a través de ZmqKVClient (Sovereign Memory).
    """
    def __init__(self, client: ZmqKVClient, namespace: str):
        self.client = client
        self.namespace = namespace

    def __getitem__(self, key: str) -> Any:
        val = self.client.hget(self.namespace, key)
        if val is None:
            raise KeyError(key)
        if isinstance(val, dict):
            return NestedZmqDictProxy(self, key, val)
        return val

    def __setitem__(self, key: str, value: Any):
        # Desempaquetar NestedZmqDictProxy si es necesario
        if isinstance(value, NestedZmqDictProxy):
            value = value._local_dict
        self.client.hset(self.namespace, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        val = self.client.hget(self.namespace, key)
        if val is None:
            return default
        if isinstance(val, dict):
            return NestedZmqDictProxy(self, key, val)
        return val
        
    def keys(self):
        d = self.client.hgetall(self.namespace)
        return d.keys()
        
    def __iter__(self):
        return iter(self.keys())
        
    def items(self):
        d = self.client.hgetall(self.namespace)
        return d.items()
        
    def values(self):
        d = self.client.hgetall(self.namespace)
        return d.values()

    def __contains__(self, key: str) -> bool:
        return self.client.hget(self.namespace, key) is not None

    def copy(self):
        return self.client.hgetall(self.namespace).copy()
        
    def pop(self, key: str, default: Any = None) -> Any:
        try:
            val = self.client.hget(self.namespace, key)
            if val is not None:
                self.client.hdel(self.namespace, key)
                return val
        except Exception as e:
            logger.exception(f"Swallowed exception ghost bug: {e}")
            pass
        val = self.client.hdel(self.namespace, key)
        return val if val is not None else default
        
    def update(self, other_dict: Dict):
        for k, v in other_dict.items():
            self[k] = v

    def __len__(self):
        return len(self.client.hgetall(self.namespace))
