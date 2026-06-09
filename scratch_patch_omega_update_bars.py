import re

path = r"C:\Users\jhona\Documents\Proyectos\Trader Gemini\data\binance_loader.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

target = """                    # 🛡️ [DATA GUARDIAN] Integridad Forense
                    if np.isnan(bar['close']) or bar['close'] <= 0 or bar['volume'] < 0:
                        logger.error(f"🚨 [DATA GUARDIAN] DATOS CORRUPTOS (NaN/Cero) en {s}. Ignorando frame.")
                        continue"""

replacement = """                    o, h, l, cl, v = bar['open'], bar['high'], bar['low'], bar['close'], bar['volume']
                    
                    # 🛡️ [DATA GUARDIAN] Integridad Forense y OMEGA O1/O3
                    if np.isnan(cl) or cl <= 0 or v < 0:
                        logger.error(f"🚨 [DATA GUARDIAN] DATOS CORRUPTOS (NaN/Cero) en {s}. Ignorando frame.")
                        continue
                        
                    if h < max(o, cl) - 1e-8 or l > min(o, cl) + 1e-8:
                        logger.warning(f"🛡️ [OMEGA] {s}: O1 Validation Failed. Dropping corrupt candle (High/Low physics).")
                        continue
                        
                    if o != cl and v <= 0:
                        logger.warning(f"🛡️ [OMEGA] {s}: O3 Validation Failed. Dropping zero-volume candle with price movement.")
                        continue"""

if target in content and "OMEGA O1/O3" not in content:
    content = content.replace(target, replacement)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Patch aplicado correctamente.")
else:
    print("Patch ya aplicado o target no encontrado.")
