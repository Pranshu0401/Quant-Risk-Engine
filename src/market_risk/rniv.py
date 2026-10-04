import numpy as np
from scipy.stats import norm

class RNIVCalculator:
    def __init__(self, risk_factors, confidence_level=0.99):
        self.risk_factors = risk_factors
        self.confidence_level = confidence_level
        self.rniv_items = {}

    def add_standalone_rniv(self, name, exposure, volatility, horizon_days=10):
        z = norm.ppf(self.confidence_level)
        amount = exposure * volatility * z * np.sqrt(horizon_days)
        self.rniv_items[name] = amount
        return amount

    def aggregate_rniv(self, correlation_matrix=None):
        amounts = np.array(list(self.rniv_items.values()))
        if correlation_matrix is None:
            return np.sum(amounts)
        return np.sqrt(np.dot(amounts.T, np.dot(correlation_matrix, amounts)))
