import feedparser
from textblob import TextBlob
import time
from datetime import datetime

class SentimentLoader:
    """
    Fetches news from RSS feeds and calculates a market sentiment score.
    Score range: -1.0 (Very Bearish) to +1.0 (Very Bullish).
    """
    def __init__(self):
        self.feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/"
        ]
        self.current_score = 0.0
        self.last_update = 0
        self.update_interval = 300 # Update every 5 minutes
        self.latest_headlines = []

    def fetch_news(self):
        """
        Fetches and analyzes news. Returns the aggregate sentiment score.
        """
        now = time.time()
        if now - self.last_update < self.update_interval:
            return self.current_score

        print("SentimentLoader: Fetching latest crypto news...")
        total_polarity = 0
        count = 0
        headlines = []

        for url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]: # Analyze top 5 from each
                    title = entry.title
                    # Analyze Sentiment
                    blob = TextBlob(title)
                    polarity = blob.sentiment.polarity
                    
                    total_polarity += polarity
                    count += 1
                    
                    headlines.append({
                        'title': title,
                        'score': polarity,
                        'source': 'CoinTelegraph' if 'cointelegraph' in url else 'CoinDesk'
                    })
            except Exception as e:
                print(f"Error fetching feed {url}: {e}")

        if count > 0:
            self.current_score = total_polarity / count
        else:
            self.current_score = 0.0
            
        self.latest_headlines = headlines
        self.last_update = now
        
        print(f"SentimentLoader: Updated Score = {self.current_score:.4f} (based on {count} headlines)")
        return self.current_score

    def get_sentiment(self):
        return self.current_score

if __name__ == "__main__":
    loader = SentimentLoader()
    score = loader.fetch_news()
    print(f"Test Run Score: {score}")
    for h in loader.latest_headlines:
        print(f"- {h['title']} ({h['score']:.2f})")
