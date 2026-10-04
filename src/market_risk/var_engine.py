import numpy as np
from scipy.stats import norm

class VaREngine:
    @staticmethod
    def historical_var(returns, confidence_level=0.99):
        if len(returns) == 0: return 0.0, 0.0
        var = np.percentile(returns, 100 * (1 - confidence_level))
        es = np.mean(returns[returns <= var]) if len(returns[returns <= var]) > 0 else var
        return -var, -es

    @staticmethod
    def parametric_var(returns, confidence_level=0.99):
        if len(returns) == 0: return 0.0, 0.0
        mu, sigma = np.mean(returns), np.std(returns)
        z = norm.ppf(1 - confidence_level)
        var = mu + z * sigma
        es = mu - sigma * norm.pdf(z) / (1 - confidence_level)
        return -var, -es

    @staticmethod
    def monte_carlo_var(S0, mu, sigma, T, n_simulations=10000, confidence_level=0.99):
        Z = np.random.standard_normal(n_simulations)
        ST = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        returns = (ST - S0) / S0
        var = np.percentile(returns, 100 * (1 - confidence_level))
        es = np.mean(returns[returns <= var]) if len(returns[returns <= var]) > 0 else var
        return -var * S0, -es * S0
        
    @staticmethod
    def time_scaling(var_1d, horizon=10):
        return var_1d * np.sqrt(horizon)
