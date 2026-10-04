import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    @staticmethod
    def d1(S, K, T, r, sigma, q=0):
        return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma, q=0):
        return BlackScholesPricer.d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

    @classmethod
    def price(cls, S, K, T, r, sigma, q=0, option_type='call'):
        if T <= 0:
            return max(S - K, 0.0) if option_type == 'call' else max(K - S, 0.0)
        
        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        if option_type == 'call':
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    @classmethod
    def greeks(cls, S, K, T, r, sigma, q=0, option_type='call'):
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
            
        d1 = cls.d1(S, K, T, r, sigma, q)
        d2 = cls.d2(S, K, T, r, sigma, q)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        
        if option_type == 'call':
            delta = np.exp(-q * T) * cdf_d1
            gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
            theta = -(S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) + q * S * np.exp(-q * T) * cdf_d1 - r * K * np.exp(-r * T) * cdf_d2
            rho = K * T * np.exp(-r * T) * cdf_d2
        elif option_type == 'put':
            delta = -np.exp(-q * T) * norm.cdf(-d1)
            gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
            theta = -(S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega / 100.0,
            'theta': theta / 252.0,
            'rho': rho / 100.0
        }
