import numpy as np

class FixedIncomePricer:
    @staticmethod
    def discount_factor(r, T):
        return np.exp(-r * T)
        
    @classmethod
    def bond_price(cls, face_value, coupon_rate, r, T, freq=2):
        periods = int(T * freq)
        if periods <= 0:
            return face_value
            
        dt = 1.0 / freq
        coupon = face_value * coupon_rate / freq
        
        times = np.arange(1, periods + 1) * dt
        dfs = cls.discount_factor(r, times)
        
        pv_coupons = np.sum(coupon * dfs)
        pv_face = face_value * cls.discount_factor(r, T)
        
        return pv_coupons + pv_face

    @classmethod
    def irs_price(cls, notional, fixed_rate, float_spread, r, T, freq=4, is_payer=True):
        periods = int(T * freq)
        if periods <= 0:
            return 0.0
            
        dt = 1.0 / freq
        times = np.arange(1, periods + 1) * dt
        dfs = cls.discount_factor(r, times)
        
        pv_float = notional + np.sum(notional * float_spread * dt * dfs) - notional * dfs[-1]
        pv_fixed = np.sum(notional * fixed_rate * dt * dfs)
        
        return (pv_float - pv_fixed) if is_payer else (pv_fixed - pv_float)
