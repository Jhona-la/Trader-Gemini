"""
SENTIMENT LOADER - PER SYMBOL ISOLATION
=======================================
Corrects previous logic where a single global score applied to all assets.

Now:
- Maintains a 'sentiment_map' dictionary.
- Scans news for specific keywords (BTC, ETH, SOL).
- Returns: Global Score (Macro) + Symbol Score (Specific).
"""

import feedparser
from textblob import TextBlob
import time
import threading
from datetime import datetime
import pandas as pd
from config import Config
from utils.logger import logger
from utils.thread_monitor import monitor

import nltk

def ensure_nltk_resources():
    """Ensure required NLTK data is available."""
    resources = ['punkt', 'brown', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}' if res == 'punkt' else f'corpora/{res}' if res in ['brown', 'wordnet'] else f'taggers/{res}')
        except LookupError:
            print(f"📥 Downloading NLTK resource: {res}...")
            nltk.download(res, quiet=True)

class SentimentLoader:
    def __init__(self):
        # Phase 8: Ensure dependencies
        ensure_nltk_resources()
        
        self.feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
        
        # DYNAMIC INITIALIZATION: Populate from Config
        self.sentiment_map = {'GLOBAL': 0.0}
        self.keywords = {}
        
        # Auto-detect keywords for all trading pairs
        for pair in getattr(Config, 'TRADING_PAIRS', []):
            base = pair.split('/')[0] if '/' in pair else pair.replace('USDT', '')
            self.sentiment_map[base] = 0.0
            
            # Smart Keyword Matcher
            self.keywords[base] = [base.lower()]
            
            # Common aliases for top coins (manual overrides for better accuracy)
            if base == 'BTC': self.keywords[base].extend(['bitcoin', 'satoshi'])
            elif base == 'ETH': self.keywords[base].extend(['ethereum', 'vitalik'])
            elif base == 'SOL': self.keywords[base].extend(['solana'])
            elif base == 'XRP': self.keywords[base].extend(['ripple', 'sec'])
            elif base == 'DOGE': self.keywords[base].extend(['dogecoin', 'musk'])
            elif base == 'ADA': self.keywords[base].extend(['cardano'])
            elif base == 'BNB': self.keywords[base].extend(['binance', 'cz', 'bsc'])

        
        self.last_update = 0
        self.update_interval = 900 # 15 minutes (Energy Efficient)
        self._stop_event = threading.Event()
        
    def start_background_thread(self):
        """Starts background fetcher without blocking main thread"""
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        print("✅ Sentiment Engine Started (Background)")

    def stop(self):
        """Terminates background thread immediately"""
        self._stop_event.set()

    def _loop(self):
        monitor.register_thread("Sentiment")
        while not self._stop_event.is_set():
            try:
                monitor.update("Sentiment", "Fetching API...")
                self.fetch_news()
                
                monitor.update("Sentiment", "Sleeping 900s...")
            except Exception as e:
                print(f"⚠️ Sentiment error: {e}")
            
            # Use wait() instead of sleep() to allow immediate interruption
            if self._stop_event.wait(timeout=self.update_interval):
                break


    def fetch_news(self):
        """Fetched RSS and updates sentiment_map"""
        print("📰 Fetching Crypto News...")
        
        # Reset temp counters for this batch
        batch_scores = {k: [] for k in self.sentiment_map.keys()}
        
        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                if not feed.entries: continue
                
                # Analyze top 10 headlines
                for entry in feed.entries[:10]:
                    text = f"{entry.title} {entry.get('summary', '')}".lower()
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    found_specific = False
                    
                    # 1. Check for specific symbols
                    for symbol, keys in self.keywords.items():
                        if any(k in text for k in keys):
                            batch_scores[symbol].append(polarity)
                            found_specific = True
                    
                    # 2. If no specific coin mentioned, it's likely Market Wide (Macro)
                    if not found_specific:
                        batch_scores['GLOBAL'].append(polarity)
                        
            except Exception as e:
                print(f"Feed error: {e}")

        # Update Main Map (Weighted Average)
        for key, scores in batch_scores.items():
            if scores:
                avg = sum(scores) / len(scores)
                # Dampening factor (don't swing too wild)
                self.sentiment_map[key] = (self.sentiment_map[key] * 0.7) + (avg * 0.3)
        
        print(f"📰 Sentiment Updated: Global={self.sentiment_map['GLOBAL']:.2f} | BTC={self.sentiment_map['BTC']:.2f}")

    def get_sentiment(self, symbol="BTC/USDT"):
        """
        Returns combined sentiment for a symbol.
        Formula: Global_Score + Symbol_Score
        """
        # Extract base symbol (e.g. 'BTC' from 'BTC/USDT')
        if '/' in symbol:
            base = symbol.split('/')[0]
        else:
            base = symbol
            
        global_s = self.sentiment_map.get('GLOBAL', 0.0)
        specific_s = self.sentiment_map.get(base, 0.0)
        
        return global_s + specific_s
