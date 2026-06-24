import os
import json
from typing import List, Dict, Any, Tuple
from utils.logger import logger
from config import Config

class MarketScanner:
    """
    🔍 CTOS OMNISCIENT SCANNER (Phase 5)
    
    PROFESSOR METHOD:
    QUÉ: Escanea el mercado global, rankea por Score (Volumen + Volatilidad),
         y separa las monedas en Top 10 (operables) y Next 16 (en medición).
    POR QUÉ: $13 USD no permite operar 26 monedas simultáneamente. Concentramos
         la liquidez en las 10 mejores mientras medimos si alguna de las 16 restantes
         debería promover al Top 10.
    PARA QUÉ: Maximizar Win Rate (>80%) al operar solo las monedas con mayor
         probabilidad de ganancia constante y detectar nuevos prospectos automáticamente.
    CÓMO: Score = Volume * 0.6 + Volatility * 0.4, con Loyalty Bonus para estabilidad.
    CUÁNDO: Cada vez que el Engine llama a scan_market() (cada 15 minutos en producción).
    DÓNDE: core/market_scanner.py
    QUIÉN: MarketScanner → Engine → RiskManager.Gatekeeper
    """
    
    def __init__(self, data_provider):
        self.data_provider = data_provider
        self.toxic_assets = ['RENDER/USDT'] 
        self.loyalty_file = os.path.join(Config.DATA_DIR, "scanner_loyalty.json")
        self.loyalty_data = self._load_loyalty()
        self.active_basket = []  # Top 10 (operables)
        self.prospect_basket = []  # Next 16 (medición)
        self.last_ranked_data = []  # Full ranked data for audit
        
        # Scoring weights
        self.VOL_WEIGHT = 0.6
        self.VOLATILITY_WEIGHT = 0.4
        self.LOYALTY_BONUS = 0.05  # 5% bonus per loyalty count

    def _load_loyalty(self) -> Dict[str, int]:
        if os.path.exists(self.loyalty_file):
            try:
                with open(self.loyalty_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_loyalty(self):
        try:
            os.makedirs(os.path.dirname(self.loyalty_file), exist_ok=True)
            with open(self.loyalty_file, 'w') as f:
                json.dump(self.loyalty_data, f)
        except Exception as e:
            logger.error(f"Scanner: Failed to save loyalty: {e}")

    def get_top_ranked_symbols(self, limit: int = 26) -> List[str]:
        """
        Ranks ALL symbols and returns the full basket (Top 10 + Next 16).
        The RiskManager.Gatekeeper will enforce which ones actually trade.
        
        limit: Total symbols to return (default 26 = 10 active + 16 prospects)
        """
        active_limit = getattr(Config, 'ACTIVE_TRADING_LIMIT', 10)
        logger.info(f"🔍 Scanning market (Top {active_limit} Active + {limit - active_limit} Prospects)...")
        
        try:
            client = self.data_provider.client_sync
            if not client: return []
                
            tickers = client.get_ticker()
            futures_tickers = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') 
                and not any(toxic in t['symbol'] for toxic in ['UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT'])
            ]
            
            ranked_data = []
            for t in futures_tickers:
                symbol_raw = t['symbol']
                internal_symbol = f"{symbol_raw[:-4]}/USDT"
                
                if internal_symbol in self.toxic_assets: continue
                    
                volume = float(t['quoteVolume'])
                high, low = float(t['highPrice']), float(t['lowPrice'])
                volatility = (high - low) / low if low > 0 else 0
                
                # Raw Score
                raw_score = (volume * self.VOL_WEIGHT) + (volatility * 1000000 * self.VOLATILITY_WEIGHT)
                
                # Apply Loyalty Bonus
                loyalty_count = self.loyalty_data[internal_symbol]
                final_score = raw_score * (1 + (loyalty_count * self.LOYALTY_BONUS))
                
                ranked_data.append({
                    'symbol': internal_symbol,
                    'score': final_score,
                    'raw_score': raw_score,
                    'volume': volume,
                    'volatility': volatility
                })
            
            # Sort all items
            ranked_data.sort(key=lambda x: x['score'], reverse=True)
            self.last_ranked_data = ranked_data[:limit]
            
            # --- HYSTERESIS & PATIENCE LOGIC ---
            candidates = [d['symbol'] for d in ranked_data[:limit]]
            
            # Update Loyalty for top candidates
            for sym in candidates[:active_limit]:
                self.loyalty_data[sym] = self.loyalty_data[sym] + 1
            
            # Retention Check
            final_selection = []
            
            # Mandatory symbols first
            mandatory = ['BTC/USDT', 'ETH/USDT']
            for m in mandatory:
                if m not in final_selection: final_selection.append(m)

            # Re-evaluate previous active_basket
            for sym in self.active_basket:
                if sym in mandatory: continue
                rank = next((i for i, d in enumerate(ranked_data) if d['symbol'] == sym), 999)
                if rank < (limit + 10):
                    if sym not in final_selection:
                        final_selection.append(sym)
                else:
                    logger.info(f"📉 Scanner: Dropping {sym} (Rank {rank} is too low).")
                    self.loyalty_data[sym] = max(0, self.loyalty_data[sym] - 2)

            # Fill remaining slots with new candidates
            for sym in candidates:
                if len(final_selection) >= limit: break
                if sym not in final_selection:
                    final_selection.append(sym)
            
            # ═══════════════════════════════════════════════════════════════
            # SPLIT: Top 10 (Active) + Next 16 (Prospects)
            # ═══════════════════════════════════════════════════════════════
            self.active_basket = final_selection[:active_limit]
            self.prospect_basket = final_selection[active_limit:]
            
            self._save_loyalty()
            self._send_prospect_report(ranked_data, active_limit)
            
            logger.info(f"💎 Active Basket ({len(self.active_basket)}): {', '.join(self.active_basket)}")
            logger.info(f"📊 Prospect Basket ({len(self.prospect_basket)}): {', '.join(self.prospect_basket)}")
            
            return final_selection
            
        except Exception as e:
            logger.error(f"Scanner Error: {e}")
            return self.active_basket if self.active_basket else Config.CRYPTO_FUTURES_PAIRS[:limit]

    def _send_prospect_report(self, ranked_data: list, active_limit: int):
        """
        📊 Envía informe de prospectos a Telegram.
        QUÉ: Reporte de las 16 monedas en medición y alerta si alguna supera al Top 10.
        POR QUÉ: El usuario necesita saber cuándo hay un prospecto mejor.
        PARA QUÉ: Rotación dinámica del basket para maximizar rendimiento.
        CÓMO: Compara scores de los prospectos con el miembro más débil del Top 10.
        CUÁNDO: Cada vez que se ejecuta get_top_ranked_symbols().
        DÓNDE: core/market_scanner.py
        QUIÉN: MarketScanner._send_prospect_report()
        """
        try:
            from utils.notifier import Notifier
            
            if len(ranked_data) < active_limit + 1:
                return
                
            # Check if any prospect outranks the weakest Top 10 member
            top_10_scores = [(d['symbol'], d['score']) for d in ranked_data[:active_limit]]
            weakest_top10 = min(top_10_scores, key=lambda x: x[1])
            
            promotions = []
            for d in ranked_data[active_limit:active_limit + 16]:
                if d['score'] > weakest_top10[1]:
                    promotions.append(d)
            
            # Build report
            lines = ["🔬 *INFORME DE PROSPECTOS (16 en Medición)*\n"]
            lines.append(f"🏆 Top {active_limit} Activos: {', '.join(self.active_basket)}\n")
            lines.append("📊 *Prospectos en Medición:*")
            
            for i, d in enumerate(ranked_data[active_limit:active_limit + 16]):
                rank = active_limit + i + 1
                vol_m = d['volume'] / 1_000_000
                vol_pct = d['volatility'] * 100
                lines.append(f"  #{rank} `{d['symbol']}` — Score: {d['score']:.0f} | Vol: ${vol_m:.1f}M | Δ: {vol_pct:.2f}%")
            
            if promotions:
                lines.append(f"\n🚨 *ALERTA DE PROMOCIÓN:*")
                lines.append(f"⬆️ Los siguientes prospectos SUPERAN al más débil del Top {active_limit} (`{weakest_top10[0]}` Score={weakest_top10[1]:.0f}):")
                for p in promotions:
                    lines.append(f"  🌟 `{p['symbol']}` — Score: {p['score']:.0f} (+{((p['score']/weakest_top10[1])-1)*100:.1f}% superior)")
                lines.append(f"\n💡 Considerar rotación del basket en el próximo ciclo.")
            
            report = "\n".join(lines)
            Notifier.send_telegram(report, "INFO")
            
        except Exception as e:
            logger.error(f"Scanner prospect report error: {e}")

    def get_active_symbols(self) -> List[str]:
        """Returns only the Top 10 actively-traded symbols."""
        return self.active_basket
    
    def get_prospect_symbols(self) -> List[str]:
        """Returns the 16 symbols being measured but not traded."""
        return self.prospect_basket
