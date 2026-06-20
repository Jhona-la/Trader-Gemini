import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\data\binance_loader.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# PATCH 1: Add streams to start_socket
target_streams = """            for s in self.symbol_list:
                clean = s.replace('/', '').lower()
                streams.append(f"{clean}@depth5@100ms")
                streams.append(f"{clean}@trade")"""

replacement_streams = """            for s in self.symbol_list:
                clean = s.replace('/', '').lower()
                streams.append(f"{clean}@depth5@100ms")
                streams.append(f"{clean}@trade")
                # OMEGA Anticipatory Streams
                streams.append(f"{clean}@forceOrder")
                streams.append(f"{clean}@markPrice@1s")"""

if target_streams in content and "OMEGA Anticipatory Streams" not in content:
    content = content.replace(target_streams, replacement_streams)

# PATCH 2: Process them in _handle_socket_message
target_handle = """            if '@depth' in stream:
                self._process_depth_update(data)
            elif '@trade' in stream:
                 self._process_trade_update(data)"""

replacement_handle = """            if '@depth' in stream:
                self._process_depth_update(data)
            elif '@trade' in stream:
                self._process_trade_update(data)
            elif '@forceOrder' in stream:
                self._process_force_order(data)
            elif '@markPrice' in stream:
                self._process_mark_price(data)"""

if target_handle in content and "@forceOrder" not in content[content.find(target_handle):]:
    content = content.replace(target_handle, replacement_handle)

# PATCH 3: Add processor methods
target_processor = """    def _process_depth_update(self, data):"""

processor_methods = """    def _process_force_order(self, data):
        \"\"\"
        [OMEGA] Processes liquidation events.
        \"\"\"
        try:
            o = data['o']
            symbol = o.get('s')
            if not symbol: return
            
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
            
            if not hasattr(self, 'derivatives_metrics'):
                self.derivatives_metrics = {}
            if internal_sym not in self.derivatives_metrics:
                self.derivatives_metrics[internal_sym] = {'funding_rate': 0.0, 'oi': 0.0, 'oi_delta': 0.0, 'liquidations': 0.0}
            
            # Liq amount = price * qty
            qty = float(o['q'])
            price = float(o['p'])
            side = o.get('S') # 'BUY' if shorts liquidated, 'SELL' if longs liquidated
            
            liq_value = qty * price
            # We can sign it: positive means shorts liquidated (bullish), negative means longs liquidated (bearish)
            signed_liq = liq_value if side == 'BUY' else -liq_value
            
            self.derivatives_metrics[internal_sym]['liquidations'] = signed_liq
            
        except Exception as e:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    def _process_mark_price(self, data):
        \"\"\"
        [OMEGA] Processes Funding Rate and Mark Price.
        \"\"\"
        try:
            symbol = data.get('s')
            if not symbol: return
            
            internal_sym = symbol
            if symbol not in self.symbol_list:
                for s in self.symbol_list:
                    if s.replace('/', '') == symbol:
                        internal_sym = s
                        break
            
            if not hasattr(self, 'derivatives_metrics'):
                self.derivatives_metrics = {}
            if internal_sym not in self.derivatives_metrics:
                self.derivatives_metrics[internal_sym] = {'funding_rate': 0.0, 'oi': 0.0, 'oi_delta': 0.0, 'liquidations': 0.0}
            
            if 'r' in data: # Funding rate
                self.derivatives_metrics[internal_sym]['funding_rate'] = float(data['r'])
                
        except Exception as e:
            from utils.error_handler import SystemIntegrityError
            raise SystemIntegrityError('Silent fallback blocked by Holographic Audit')

    def _process_depth_update(self, data):"""

if target_processor in content and "_process_force_order" not in content:
    content = content.replace(target_processor, processor_methods)


with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Patch streams aplicado correctamente.")
