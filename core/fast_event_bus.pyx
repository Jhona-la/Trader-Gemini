# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import logging
from typing import Callable, Dict, List, Any
from enum import Enum
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject, Py_XINCREF, Py_XDECREF

logger = logging.getLogger(__name__)

class EventChannel(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    FEATURE_UPDATE = "FEATURE_UPDATE"
    SIGNALS = "SIGNALS"
    INTENTS = "INTENTS"
    EXECUTION = "EXECUTION"
    FILLS = "FILLS"
    RISK_ALERTS = "RISK_ALERTS"
    MUTATION = "MUTATION"

cdef class FastRingBuffer:
    """
    C-level Ring Buffer lock-free (single producer, single consumer or protected by asyncio loop).
    Holds PyObject* references. Dramatically faster than queue.Queue (no OS mutexes).
    """
    cdef PyObject** buffer
    cdef int capacity
    cdef int head
    cdef int tail
    cdef int count

    def __cinit__(self, int capacity=100000):
        self.capacity = capacity
        self.buffer = <PyObject**>malloc(self.capacity * sizeof(PyObject*))
        self.head = 0
        self.tail = 0
        self.count = 0

    def __dealloc__(self):
        # Note: In a real dealloc we should DECREF remaining items
        cdef int i
        while self.count > 0:
            Py_XDECREF(self.buffer[self.head])
            self.head = (self.head + 1) % self.capacity
            self.count -= 1
        free(self.buffer)

    cdef bint push(self, object item):
        if self.count >= self.capacity:
            return False # Buffer full
        cdef PyObject* p_item = <PyObject*>item
        Py_XINCREF(p_item)
        self.buffer[self.tail] = p_item
        self.tail = (self.tail + 1) % self.capacity
        self.count += 1
        return True

    cdef object pop(self):
        if self.count == 0:
            return None # Buffer empty
        cdef PyObject* p_item = self.buffer[self.head]
        self.head = (self.head + 1) % self.capacity
        self.count -= 1
        cdef object item = <object>p_item
        Py_XDECREF(p_item)
        return item
        
    cdef int get_count(self):
        return self.count


cdef class FastEventBus:
    """
    High-Frequency Trading Event Bus.
    Replaces python queue.Queue with FastRingBuffer for nanosecond O(1) enqueuing.
    """
    cdef dict _subscribers
    cdef FastRingBuffer _queue

    def __init__(self):
        self._subscribers = {channel.value: [] for channel in EventChannel}
        self._queue = FastRingBuffer(100000)
        
    def subscribe(self, channel, callback):
        cdef str ch_val = channel.value if hasattr(channel, 'value') else channel
        if callback not in self._subscribers[ch_val]:
            self._subscribers[ch_val].append(callback)
            logger.debug(f"[FastEventBus] Subscribed {callback.__name__} to {ch_val}")

    def publish(self, channel, payload):
        """O(1) nanosecond enqueuing"""
        cdef str ch_val = channel.value if hasattr(channel, 'value') else channel
        cdef tuple item = (ch_val, payload)
        self._queue.push(item)

    def process_queue(self, int max_items=1000):
        """Processes events at C-speed, dispatching to Python callbacks"""
        cdef int items_processed = 0
        cdef object item
        cdef tuple t_item
        cdef str channel
        cdef object payload
        cdef list subs
        
        while self._queue.get_count() > 0 and items_processed < max_items:
            item = self._queue.pop()
            if item is None:
                break
            t_item = <tuple>item
            channel = <str>t_item[0]
            payload = t_item[1]
            
            subs = self._subscribers.get(channel, [])
            for sub in subs:
                try:
                    sub(payload)
                except Exception as e:
                    logger.error(f"[FastEventBus] Error dispatching to {sub.__name__} on {channel}: {e}", exc_info=True)
            
            items_processed += 1

    def empty(self):
        return self._queue.get_count() == 0

# C-level singleton wrapper logic is handled by standard python module imports
# To remain compatible with core.event_bus:
fast_event_bus_instance = FastEventBus()
