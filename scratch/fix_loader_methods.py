import os
path = 'data/binance_loader.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix depth
old_depth = '''    def _process_depth_update(self, data):
        """
        Updates the internal L2 OrderBook and calculates OFI.
        """
        try:
            symbol = data['s']'''

new_depth = '''    def _process_depth_update(self, data, stream_name=None):
        """
        Updates the internal L2 OrderBook and calculates OFI.
        """
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                # data = [E, u, bp, bq, ap, aq]
                if not stream_name: return
                symbol = stream_name.split('@')[0].upper()
                internal_sym = symbol
                
                if internal_sym not in getattr(self, 'orderbooks', {}):
                    return
                ob = self.orderbooks[internal_sym]
                
                # Direct C-level Float update
                ob.update_bid(float(data[2]), float(data[3]))
                ob.update_ask(float(data[4]), float(data[5]))
                
                # Update metrics
                if internal_sym not in self.order_flow_metrics:
                    self.order_flow_metrics[internal_sym] = {}
                    
                self.order_flow_metrics[internal_sym]['l2_ofi'] = ob.calculate_ofi()
                self.order_flow_metrics[internal_sym]['l2_spread'] = ob.calculate_spread()
                
                micro = ob.calculate_microprice()
                best_bid = float(data[2])
                best_ask = float(data[4])
                mid = (best_bid + best_ask) / 2.0
                dist = (micro - mid) / (mid + 1e-9) if mid > 0 else 0.0
                self.order_flow_metrics[internal_sym]['l2_microprice_dist'] = dist
                return

            symbol = data.get('s', '')'''

text = text.replace(old_depth, new_depth)

# Fix trade
old_trade = '''    def _process_trade_update(self, data):
        """
        Processes standard trades and calculates Whale Flow Proxy.
        """
        try:
            symbol = data['s']'''

new_trade = '''    def _process_trade_update(self, data, stream_name=None):
        """
        Processes standard trades and calculates Whale Flow Proxy.
        """
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                # data = [E, t, p, q, m]
                if not stream_name: return
                symbol = stream_name.split('@')[0].upper()
                internal_sym = symbol
                
                qty = float(data[3])
                price = float(data[2])
                is_buyer_mm = bool(data[4])
                trade_usd = qty * price
                
                if trade_usd > 100000:
                    if not hasattr(self, 'derivatives_metrics'):
                        self.derivatives_metrics = {}
                    if internal_sym not in self.derivatives_metrics:
                        self.derivatives_metrics[internal_sym] = {'funding_rate': 0.0, 'oi': 0.0, 'oi_delta': 0.0, 'liquidations': 0.0, 'whale_flow': 0.0}
                    
                    flow = trade_usd if not is_buyer_mm else -trade_usd
                    self.derivatives_metrics[internal_sym]['whale_flow'] = self.derivatives_metrics[internal_sym]['whale_flow'] + flow
                    
                # Passthrough for agg trade is handled manually since we don't have the full dict
                # We can construct a mock dict or bypass
                mock_data = {'s': symbol, 'p': str(price), 'q': str(qty), 'T': int(data[1]), 'm': is_buyer_mm}
                self._process_agg_trade(mock_data)
                return

            symbol = data.get('s', '')'''

text = text.replace(old_trade, new_trade)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
