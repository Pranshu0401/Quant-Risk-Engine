import numpy as np

class SIMMCalculator:
    def __init__(self):
        self.risk_weights = {'Equity': 21.0, 'Rates': 50.0}
        self.correlation = {'Equity_Equity': 0.15, 'Rates_Rates': 0.40}

    def calculate_margin(self, sensitivities):
        margin = {}
        for asset, rw_val in self.risk_weights.items():
            if asset in sensitivities and sensitivities[asset]:
                sens = np.array(sensitivities[asset])
                ws = sens * rw_val
                n = len(ws)
                rho = np.full((n, n), self.correlation[f'{asset}_{asset}'])
                np.fill_diagonal(rho, 1.0)
                margin[asset] = np.sqrt(np.dot(ws.T, np.dot(rho, ws)))
        return np.sqrt(sum(m**2 for m in margin.values()))
