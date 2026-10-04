import numpy as np

class CDSPricer:
    @staticmethod
    def survival_probability(hazard_rate, T):
        return np.exp(-hazard_rate * T)

    @classmethod
    def price(cls, notional, spread, hazard_rate, recovery_rate, r, T, freq=4):
        periods = int(T * freq)
        if periods <= 0: return 0.0
            
        dt = 1.0 / freq
        times = np.arange(1, periods + 1) * dt
        dfs = np.exp(-r * times)
        surv_probs = cls.survival_probability(hazard_rate, times)
        
        premium_leg = np.sum(spread * notional * dt * dfs * surv_probs)
        
        times_prev = np.concatenate(([0.0], times[:-1]))
        surv_prev = np.concatenate(([1.0], surv_probs[:-1]))
        
        default_probs = surv_prev - surv_probs
        dfs_mid = np.exp(-r * (times + times_prev) / 2.0)
        protection_leg = np.sum(notional * (1.0 - recovery_rate) * dfs_mid * default_probs)
        
        return protection_leg - premium_leg
