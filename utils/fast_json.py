import logging
from typing import Any

# Phase 1: Zero-Latency Serialization
try:
    import orjson
    HAS_ORJSON = True
    _real_json = None
except ImportError:
    import json as _real_json
    HAS_ORJSON = False

# Capture original functions to avoid recursion if monkey-patched
if _real_json:
    _json_dumps = _real_json.dumps
    _json_loads = _real_json.loads
    _json_dump = _real_json.dump
    _json_load = _real_json.load
    
    JSONEncoder = _real_json.JSONEncoder
    JSONDecoder = _real_json.JSONDecoder
    JSONDecodeError = _real_json.JSONDecodeError
else:
    # If using orjson, we don't need these but for safety/compliance
    import json
    JSONEncoder = json.JSONEncoder
    JSONDecoder = json.JSONDecoder
    JSONDecodeError = json.JSONDecodeError

logger = logging.getLogger("FastJson")

class FastJson:
    """
    Centralized JSON Handler optimized for speed (Phase 1).
    Uses orjson for millisecond-level advantage in serialization.
    """
    # Compliance for monkey-patching (main.py)
    JSONEncoder = JSONEncoder
    JSONDecoder = JSONDecoder
    JSONDecodeError = JSONDecodeError
    
    @staticmethod
    def dumps(data: Any, *args, **kwargs) -> str:
        """
        Serializes data to string. 
        """
        try:
            if HAS_ORJSON:
                # orjson.dumps doesn't like allow_nan, sort_keys, default, etc. as kwargs
                option = orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY
                if kwargs.get('indent'):
                    option |= orjson.OPT_INDENT_2
                if kwargs.get('sort_keys'):
                    option |= orjson.OPT_SORT_KEYS
                    
                return orjson.dumps(data, option=option).decode('utf-8')
            else:
                # Fallback to standard json (captured)
                if 'default' not in kwargs:
                    kwargs['default'] = str
                return _json_dumps(data, *args, **kwargs)
                
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            return "{}"

    @staticmethod
    def dump(data: Any, fp: Any, *args, **kwargs):
        """
        Serializes data to a file-like object.
        """
        try:
            if HAS_ORJSON:
                option = orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY
                if kwargs.get('indent'):
                    option |= orjson.OPT_INDENT_2
                
                content = orjson.dumps(data, option=option)
                fp.write(content.decode('utf-8'))
            else:
                if 'default' not in kwargs:
                    kwargs['default'] = str
                _json_dump(data, fp, *args, **kwargs)
                
        except Exception as e:
            logger.error(f"Serialization 'dump' error: {e}")

    @staticmethod
    def loads(json_str: str, *args, **kwargs) -> Any:
        """
        Deserializes string/bytes to python object.
        """
        try:
            if HAS_ORJSON:
                 return orjson.loads(json_str)
            else:
                 return _json_loads(json_str, *args, **kwargs)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return {}

    @staticmethod
    def load(fp: Any, *args, **kwargs) -> Any:
        """
        Deserializes from file-like object.
        """
        try:
            if HAS_ORJSON:
                return orjson.loads(fp.read())
            else:
                return _json_load(fp, *args, **kwargs)
        except Exception as e:
            logger.error(f"Deserialization 'load' error: {e}")
            return {}

    @staticmethod
    def dump_to_file(data: Any, filepath: str):
        """Atomic write to file."""
        try:
            if HAS_ORJSON:
                content = orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NAIVE_UTC | orjson.OPT_SERIALIZE_NUMPY)
                with open(filepath, 'wb') as f:
                    f.write(content)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    _json_dump(data, f, indent=2, default=str)
                    
        except Exception as e:
            logger.error(f"File write error {filepath}: {e}")

    @staticmethod
    def load_from_file(filepath: str) -> Any:
        try:
            with open(filepath, 'rb' if HAS_ORJSON else 'r', encoding=None if HAS_ORJSON else 'utf-8') as f:
                if HAS_ORJSON:
                    return orjson.loads(f.read())
                else:
                    return _json_load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"File read error {filepath}: {e}")
            return None
