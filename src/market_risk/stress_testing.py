import numpy as np

class StressTestingEngine:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def apply_scenario(self, scenario_shifts):
        impacts = []
        for index, trade in self.portfolio.iterrows():
            trade_pnl = 0.0
            asset = trade.get('AssetClass', 'Unknown')
            if asset == 'Equity' and 'Equity' in scenario_shifts:
                trade_pnl += trade.get('Delta', 0) * trade.get('Exposure', 0) * scenario_shifts['Equity']
                trade_pnl += trade.get('Vega', 0) * 0.10 * 100 
            if 'IR_Curve' in scenario_shifts:
                trade_pnl += trade.get('Rho', 0) * (scenario_shifts['IR_Curve'] * 10000) / 100.0
            if asset == 'Credit' and 'Credit_Spread' in scenario_shifts:
                trade_pnl -= trade.get('CS01', 0) * (scenario_shifts['Credit_Spread'] * 10000)
            impacts.append(trade_pnl)
        return np.sum(impacts)

    def frtb_standard_scenarios(self):
        scenarios = {
            'Global_Equities_Crash': {'Equity': -0.15, 'IR_Curve': -0.005},
            'Rates_Up_Shock': {'IR_Curve': 0.01},
            'Credit_Crisis': {'Credit_Spread': 0.015, 'Equity': -0.10}
        }
        return {name: self.apply_scenario(shifts) for name, shifts in scenarios.items()}
