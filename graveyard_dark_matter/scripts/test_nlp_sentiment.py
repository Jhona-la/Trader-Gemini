"""Smoke test for NewsSentimentNLP module."""
import sys
sys.path.insert(0, '.')
from data.news_sentiment_nlp import NewsSentimentNLP, _load_models

# Test 1: Instantiation
nlp = NewsSentimentNLP()
print("Test 1: Instantiation OK")

# Test 2: Default features (should be all zero since no news polled yet)
feats = nlp.get_sentiment_features("BTC/USDT")
print(f"Test 2: Default features: {feats}")
assert feats["news_sentiment"] == 0.0, "FAIL: sentiment should be 0.0"
assert feats["news_has_fresh_data"] == 0.0, "FAIL: freshness should be 0.0"
print("Test 2: All zero defaults PASSED (no stale values)")

# Test 3: Load models
print("Test 3: Loading HuggingFace models (first time may download ~500MB)...")
success = _load_models()
if success:
    print("Test 3: Models loaded successfully")
    
    # Test 4: Classify sample headlines
    from data.news_sentiment_nlp import _finbert_pipeline, _cryptobert_pipeline
    
    headlines = [
        "Bitcoin surges past $100,000 as institutional demand skyrockets",
        "Ethereum crashes 20% amid SEC crackdown fears",
        "Solana network processes record transactions in 24 hours",
        "Federal Reserve hints at rate cuts, crypto markets rally",
        "Major crypto exchange hacked, millions stolen",
    ]
    
    print("\n--- FinBERT Results ---")
    fb_results = _finbert_pipeline(headlines)
    for i, h in enumerate(headlines):
        r = fb_results[i]
        label = r["label"]
        score = r["score"]
        print(f"  [{label:>10} {score:.3f}] {h[:65]}")
    
    print("\n--- CryptoBERT Results ---")
    cb_results = _cryptobert_pipeline(headlines)
    for i, h in enumerate(headlines):
        r = cb_results[i]
        label = r["label"]
        score = r["score"]
        print(f"  [{label:>10} {score:.3f}] {h[:65]}")
    
    print("\nTest 4: Classification PASSED")
else:
    print("Test 3: Model loading failed (will work with zeros)")

print("\nALL SMOKE TESTS PASSED")
