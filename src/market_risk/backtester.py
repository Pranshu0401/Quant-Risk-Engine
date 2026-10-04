import numpy as np
from scipy.stats import chi2

class KupiecPOFTest:
    @staticmethod
    def test(exceptions, observations, confidence_level=0.99):
        p = 1.0 - confidence_level
        x = exceptions
        T = observations
        if x == 0: return 0.0, 1.0, True
        
        prob_actual = x / T
        term1 = -2.0 * np.log(((1 - p)**(T - x)) * (p**x))
        term2 = 2.0 * np.log(((1 - prob_actual)**(T - x)) * (prob_actual**x))
        lr_stat = term1 + term2
        
        p_value = 1.0 - chi2.cdf(lr_stat, 1)
        return lr_stat, p_value, p_value > 0.05
