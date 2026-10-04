class CapitalAdequacy:
    def __init__(self, internal_var, rniv_addon, stress_capital, regulatory_multiplier=3.0):
        self.internal_var = internal_var
        self.rniv_addon = rniv_addon
        self.stress_capital = stress_capital
        self.regulatory_multiplier = regulatory_multiplier

    def economic_capital(self):
        return self.internal_var + self.rniv_addon + self.stress_capital

    def regulatory_capital(self, var_10d, es_10d):
        return max(es_10d, self.regulatory_multiplier * es_10d * 0.8)

    def generate_report(self, var_10d, es_10d):
        ec = self.economic_capital()
        rc = self.regulatory_capital(var_10d, es_10d)
        return {
            'Economic_Capital': ec,
            'Regulatory_Capital': rc,
            'Surplus_Deficit': ec - rc,
            'Status': 'Adequate' if ec >= rc else 'Capital Deficit'
        }
