
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from utils.logger import setup_logger

logger = setup_logger("SentimentAI")

class SentimentProcessor:
    """
    🧠 PHASE 15: SENTIMENT ANALYSIS ENGINE (LDA)
    Uses Latent Dirichlet Allocation to discover hidden topics in market text (News/Social).
    Classifies market mood into: BULLISH, BEARISH, FUD, HYPE.
    """
    
    def __init__(self):
        # Stopwords (Basic financial/common list)
        self.stopwords = set([
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'to', 'for', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'it', 
            'that', 'this', 'these', 'those', 'market', 'price', 'trading', 
            'crypto', 'currency', 'bitcoin', 'btc', 'eth', 'ethereum' 
        ])
        
        # Pre-trained "Mock" Topics (In production, this would be trained on historical news)
        # Topic 0: Regulatory/FUD (SEC, Ban, Regulation, Lawsuit)
        # Topic 1: Bullish Tech/Adoption (Upgrade, Launch, Partnership, Mainnet)
        # Topic 2: Bearish Macro (Inflation, Rate, Hike, Recession, Crash)
        # Topic 3: Hype/Meme (Moon, Lambo, Pump, Elon, Doge)
        self.n_topics = 4
        
        # Initialize Vectorizer & LDA
        self.vectorizer = CountVectorizer(stop_words=list(self.stopwords), max_features=1000)
        self.lda = LatentDirichletAllocation(n_components=self.n_topics, random_state=42)
        
        # Vocabulary State (Needs to be fit at least once to work)
        # We perform a "Warm Start" with dummy data to define the vector space
        self._warm_start()
        
    def _warm_start(self):
        """Initializes the model with synthetic financial seeds to define topics."""
        seeds = [
            "sec sec regulation ban lawsuit court jail illegal security", # Topic 0
            "upgrade mainnet launch partnership adoption institutional buy etf", # Topic 1
            "inflation rate hike recession crash dump sell liquidation panic", # Topic 2
            "moon pump lambo elon doge shib rocket 100x gem", # Topic 3
        ]
        # Duplicate seeds to give weight
        training_data = seeds * 10
        
        X = self.vectorizer.fit_transform(training_data)
        self.lda.fit(X)
        logger.info("🧠 Sentiment Processor: LDA Model Warm-Started (4 Topics)")

    def analyze_text(self, text_batch):
        """
        Analyzes a list of text strings and returns the dominant topic and sentiment score.
        Args:
           text_batch: List[str] e.g. ["SEC sues Binance", "Bitcoin hits 100k"]
        Returns:
           dict: {
               'dominant_topic': int, 
               'sentiment_score': float (-1.0 to 1.0),
               'topic_dist': list
           }
        """
        if not text_batch:
            return {'sentiment_score': 0.0, 'dominant_topic': -1}
            
        try:
            # Preprocess
            clean_texts = [self._clean_text(t) for t in text_batch]
            
            # Vectorize
            X = self.vectorizer.transform(clean_texts)
            
            # Predict Topics
            topic_results = self.lda.transform(X)
            
            # Average distribution across batch
            avg_dist = np.mean(topic_results, axis=0)
            dominant_topic = np.argmax(avg_dist)
            
            # Map Topic to Sentiment Score (Heuristic)
            # 0: FUD (-0.8)
            # 1: Adoption (+0.8)
            # 2: Macro Bear (-0.6)
            # 3: Hype (+0.4 - Volatile)
            
            topic_sentiments = [-0.8, 0.8, -0.6, 0.4]
            weighted_score = np.dot(avg_dist, topic_sentiments)
            
            labels = ["Regulatory FUD", "Bullish Adoption", "Macro Bearish", "Retail Hype"]
            
            return {
                'dominant_topic': int(dominant_topic),
                'topic_label': labels[dominant_topic],
                'sentiment_score': float(weighted_score),
                'confidence': float(avg_dist[dominant_topic])
            }
            
        except Exception as e:
            logger.error(f"Sentiment Analysis Failed: {e}")
            return {'sentiment_score': 0.0, 'dominant_topic': -1}

    def _clean_text(self, text):
        """Basic cleaning: lowercase, remove special chars"""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def get_market_mood(self):
        """
        Public API to get current market mood. 
        (Mocked input for now until NewsLoader is active)
        """
        # In a real scenario, this would pull from a buffer of recent news
        # For now, return Neutral
        return 0.0
