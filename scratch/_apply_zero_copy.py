import re

def main():
    file_path = "strategies/ml_strategy.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # We will replace the entire cache bypass logic with a truly zero-copy approach.
    # From line: # ── FASE III-B: ZERO-COPY CACHE BYPASS (GIL EVASION) ──
    # To line: current_row = last_row  # Alias: dict soporta .get() y [] igual que Pandas Series

    pattern = r'(# ── FASE III-B: ZERO-COPY CACHE BYPASS \(GIL EVASION\) ──.*?)(# Convertir la última fila a dict para acceso rápido.*?current_row = last_row  # Alias: dict soporta \.get\(\) y \[\] igual que Pandas Series)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("Regex not matched!")
        return

    replacement = """# ── FASE III-B: TRUE ZERO-COPY CACHE BYPASS (GIL EVASION) ──
            df_pl = None
            last_row_dict = None
            if bars is not None and len(bars) > 0:
                current_ts = bars['timestamp'][-1] if hasattr(bars, 'dtype') else (bars[-1]['timestamp'] if isinstance(bars[-1], dict) else None)
                if current_ts is not None and hasattr(self, "_global_feature_cache") and hasattr(self, "_global_feature_cache_ts"):
                    import numpy as np
                    ts_arr = self._global_feature_cache_ts
                    idx = np.searchsorted(ts_arr, current_ts)
                    if idx < len(ts_arr) and ts_arr[idx] == current_ts:
                        # [ZERO-COPY] Skip Polars entirely. Use raw Pandas slice for Dict and extract NumPy array directly later.
                        # We just store the index so we can slice the NumPy matrix in O(1) time.
                        self._quantum_cache_hit = True
                        self._quantum_idx = idx
                        last_row_dict = self._global_feature_cache.iloc[idx].to_dict()
                        # We still need df_pl to be truthy to avoid recalculating prepare_features
                        df_pl = "CACHED_TRUE" 
            
            if df_pl is None:
                self._quantum_cache_hit = False
                df_pl = self._prepare_features(bars, regime_aware=True, return_polars=True)

            if df_pl is None or (not isinstance(df_pl, str) and len(df_pl) < 5):
                return
            
            # Convertir la última fila a dict para acceso rápido
            if self._quantum_cache_hit and last_row_dict is not None:
                last_row = last_row_dict
            else:
                last_row = df_pl.row(-1, named=True)
                
            current_row = last_row  # Alias: dict soporta .get() y [] igual que Pandas Series"""

    content = content.replace(match.group(0), replacement)
    
    # Next, we must replace the valid_features extraction:
    #             # Filtrar columnas válidas
    #             valid_features = [
    #                 col for col in feature_cols if col is not None and col in df_pl.columns
    #             ]
    
    valid_feat_pattern = r'(# Filtrar columnas válidas.*?\])'
    valid_feat_repl = """# Filtrar columnas válidas
            if self._quantum_cache_hit:
                valid_features = [
                    col for col in feature_cols if col is not None and col in self._global_feature_cache.columns
                ]
            else:
                valid_features = [
                    col for col in feature_cols if col is not None and col in df_pl.columns
                ]"""
                
    content = re.sub(valid_feat_pattern, valid_feat_repl, content, flags=re.DOTALL)
    
    # Next, we replace the X_pred generation:
    #             # ═══════════════════════════════════════════════════════════════
    #             # QUANTUM ZERO-COPY: Polars → NumPy (skip Pandas alignment)
    #             # ═══════════════════════════════════════════════════════════════
    #             X_pred = df_pl.select(valid_features).tail(1).to_numpy()
    
    xpred_pattern = r'(# ═══════════════════════════════════════════════════════════════\s*# QUANTUM ZERO-COPY: Polars → NumPy.*?X_pred = df_pl\.select\(valid_features\)\.tail\(1\)\.to_numpy\(\))'
    xpred_repl = """# ═══════════════════════════════════════════════════════════════
            # QUANTUM ZERO-COPY: Direct NumPy View (skip Polars/Pandas alignment)
            # ═══════════════════════════════════════════════════════════════
            if self._quantum_cache_hit:
                # O(1) Memory Slice! No allocations.
                # To make it even faster, we can extract the numpy array of the specific columns
                # Only if we pre-cache it. For now, iloc to numpy.
                X_pred = self._global_feature_cache.iloc[self._quantum_idx:self._quantum_idx+1][valid_features].to_numpy()
            else:
                X_pred = df_pl.select(valid_features).tail(1).to_numpy()"""
                
    content = re.sub(xpred_pattern, xpred_repl, content, flags=re.DOTALL)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Zero-Copy Cache bypass applied to ml_strategy.py")

if __name__ == "__main__":
    main()
