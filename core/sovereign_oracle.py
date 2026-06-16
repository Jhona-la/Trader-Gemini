class SovereignOracle:
    def __init__(self):
        self.knowledge_base = {}

    def get_mutation_mod(self, symbol):
        return 1.0

    def get_causal_bias(self, symbol):
        return {'drift_multiplier': 1.0, 'aggression_bias': 0.0}

sovereign_oracle = SovereignOracle()
