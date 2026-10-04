import numpy as np

class PFESimulator:
    def __init__(self, t_horizons, num_paths=1000):
        self.t_horizons = np.array(t_horizons)
        self.num_paths = num_paths
        
    def simulate_gbm_exposure(self, S0, mu, sigma, K, option_type='call'):
        np.random.seed(42)
        dt = np.diff(np.concatenate(([0], self.t_horizons)))
        paths = np.zeros((self.num_paths, len(self.t_horizons)))
        S_t = np.ones(self.num_paths) * S0
        
        for i, delta_t in enumerate(dt):
            Z = np.random.standard_normal(self.num_paths)
            S_t = S_t * np.exp((mu - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z)
            mtm = np.maximum(S_t - K, 0) if option_type == 'call' else np.maximum(K - S_t, 0)
            paths[:, i] = np.maximum(mtm, 0)
            
        return {
            'horizons': self.t_horizons,
            'EE': np.mean(paths, axis=0),
            'PFE_95': np.percentile(paths, 95, axis=0),
            'PFE_99': np.percentile(paths, 99, axis=0)
        }
