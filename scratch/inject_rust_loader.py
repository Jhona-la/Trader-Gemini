import os
path = 'data/binance_loader.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

import_statement = '''import json
import logging
import asyncio
import time
import collections

# [Fase 4] Importación C-ABI Bridge de Rust para parseo JSON sin objetos de Python
try:
    from core.rust_parser_bridge import ffi_fast_parse_depth, ffi_fast_parse_trade
except ImportError:
    pass
'''

text = text.replace('import json', import_statement, 1)

old_parsing_block = '''                                # FASE II: PARSING
                                raw_bytes = raw_msg.encode('utf-8') if isinstance(raw_msg, str) else raw_msg
                                
                                # Use Cython FFI directly for Klines later, but parse headers here
                                msg = orjson.loads(raw_bytes)
                                
                                if stream_type == 'kline':
                                    msg['_raw_bytes'] = raw_bytes'''

new_parsing_block = '''                                # FASE II: PARSING ZERO-COPY RUST FFI
                                raw_bytes = raw_msg.encode('utf-8') if isinstance(raw_msg, str) else raw_msg
                                
                                # ⚡ RUST QUANTUM ENGINE BYPASS
                                if stream_type == 'depth':
                                    res = ffi_fast_parse_depth(raw_bytes)
                                    if res is not None:
                                        uid = int(res[1])
                                        if is_duplicate(f"depth_{uid}"):
                                            continue
                                            
                                        event_time = int(res[0])
                                        if event_time > 0:
                                            local_time = int((time.time() + self.system_drift_ms / 1000.0) * 1000)
                                            latency = local_time - event_time
                                            if 0 <= latency < 5000:
                                                self.ws_latency_history.append(latency)
                                                
                                        # Convert res to dict to maintain compatibility with downstream _process_depth_update
                                        # Stream name is needed to infer symbol
                                        stream_name = getattr(ws, 'path', streams[0] if len(streams)==1 else "")
                                        process_func(res, stream_name) 
                                        continue
                                        
                                elif stream_type == 'trades':
                                    res = ffi_fast_parse_trade(raw_bytes)
                                    if res is not None:
                                        uid = int(res[1])
                                        if is_duplicate(f"trades_{uid}"):
                                            continue
                                        
                                        event_time = int(res[0])
                                        if event_time > 0:
                                            local_time = int((time.time() + self.system_drift_ms / 1000.0) * 1000)
                                            latency = local_time - event_time
                                            if 0 <= latency < 5000:
                                                self.ws_latency_history.append(latency)
                                                
                                        stream_name = getattr(ws, 'path', streams[0] if len(streams)==1 else "")
                                        process_func(res, stream_name)
                                        continue

                                # Fallback para Kline/Liquidations
                                msg = orjson.loads(raw_bytes)
                                if stream_type == 'kline':
                                    msg['_raw_bytes'] = raw_bytes'''

if old_parsing_block in text:
    text = text.replace(old_parsing_block, new_parsing_block)
else:
    print('COULD NOT FIND BLOCK')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
