"""
AITS Phase 0: Mathematical Preparation
Demonstrates a Bayesian Probability Updater.
In the AITS, the ML Predictor (XGBoost/LSTM) outputs a prior probability.
This module updates that probability using Bayes' Theorem based on live evidence
(e.g., sudden spikes in Order Flow Imbalance or Funding Rates) before the 
Execution Engine acts.
"""

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class BayesianSignalUpdater:
    def __init__(self, evidence_likelihood_given_true: float, evidence_likelihood_given_false: float):
        """
        Initializes the Bayesian Updater.
        
        Args:
            evidence_likelihood_given_true P(E|H): Probability of observing the evidence 
                (e.g., extreme OFI) given that the signal is TRUE (it will break out).
            evidence_likelihood_given_false P(E|~H): Probability of observing the evidence 
                given that the signal is FALSE (it will fail).
        """
        self.p_e_given_h = evidence_likelihood_given_true
        self.p_e_given_not_h = evidence_likelihood_given_false

    def update_probability(self, prior_prob: float) -> float:
        """
        Applies Bayes' Theorem: P(H|E) = [P(E|H) * P(H)] / P(E)
        where P(E) = P(E|H)*P(H) + P(E|~H)*P(~H)
        """
        # P(H)
        p_h = prior_prob
        # P(~H)
        p_not_h = 1.0 - prior_prob
        
        # P(E) = Law of Total Probability
        p_e = (self.p_e_given_h * p_h) + (self.p_e_given_not_h * p_not_h)
        
        if p_e == 0:
            return prior_prob
            
        # P(H|E)
        posterior = (self.p_e_given_h * p_h) / p_e
        return posterior

def run_simulation():
    # Scenario: ML Model predicts a 65% chance of a bullish breakout.
    ml_prior_probability = 0.65
    
    # We observe an extreme Order Flow Imbalance (OFI > 3.0 std devs).
    # Based on historical backtests:
    # If a breakout IS real (True), we see this extreme OFI 80% of the time. P(E|H)
    # If a breakout IS fake (False), we see this extreme OFI only 15% of the time. P(E|~H)
    
    updater = BayesianSignalUpdater(
        evidence_likelihood_given_true=0.80,
        evidence_likelihood_given_false=0.15
    )
    
    posterior_prob = updater.update_probability(ml_prior_probability)
    
    logging.info("--- AITS Bayesian Inference Simulation ---")
    logging.info(f"ML Prior Probability (Base Prediction): {ml_prior_probability * 100:.2f}%")
    logging.info(f"Evidence Observed: Extreme Bullish Order Flow Imbalance")
    logging.info(f"Bayesian Posterior Probability: {posterior_prob * 100:.2f}%")
    logging.info("------------------------------------------")
    
    if posterior_prob > 0.85:
        logging.info("Decision: CONFIDENCE THRESHOLD MET (>85%). Execute AGGRESSIVE LONG.")
    else:
        logging.info("Decision: CONFIDENCE TOO LOW. Wait or Execute PASSIVE LONG.")

if __name__ == "__main__":
    run_simulation()
