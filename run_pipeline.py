import numpy as np
import pandas as pd
import os
from src.pricing.black_scholes import BlackScholesPricer
from src.pricing.fixed_income import FixedIncomePricer
from src.pricing.cds_pricer import CDSPricer
from src.market_risk.var_engine import VaREngine
from src.market_risk.stress_testing import StressTestingEngine
from src.credit_risk.pfe_simulator import PFESimulator
from src.credit_risk.simm_calculator import SIMMCalculator
from src.capital.capital_adequacy import CapitalAdequacy
from src.reporting.db_manager import DBManager
from src.reporting.dod_attribution import DoDAttribution

def main():
    print("=== Quant Risk Engine Initialization ===")
    os.makedirs('data', exist_ok=True)
    db = DBManager('sqlite:///data/risk_engine.db')

    # 1. PRICING
    print("\n--- Pricing Modules ---")
    bs_price = BlackScholesPricer.price(S=100, K=105, T=1, r=0.05, sigma=0.2)
    bs_greeks = BlackScholesPricer.greeks(S=100, K=105, T=1, r=0.05, sigma=0.2)
    print(f"Call Option Price: {bs_price:.4f}")
    print(f"Greeks: {bs_greeks}")
    
    bond_price = FixedIncomePricer.bond_price(100, 0.05, 0.04, 5)
    print(f"5Y Bond Price (5% coupon, 4% yield): {bond_price:.4f}")
    
    cds_price = CDSPricer.price(notional=1e6, spread=0.015, hazard_rate=0.02, recovery_rate=0.4, r=0.03, T=5)
    print(f"5Y CDS Protection Value: {cds_price:.2f}")

    # 2. MARKET RISK
    print("\n--- Market Risk (VaR & Stress) ---")
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, 252)
    h_var, h_es = VaREngine.historical_var(returns)
    p_var, p_es = VaREngine.parametric_var(returns)
    print(f"Historical 1D 99% VaR: {h_var:.4%}, ES: {h_es:.4%}")
    print(f"Parametric 1D 99% VaR: {p_var:.4%}, ES: {p_es:.4%}")
    
    portfolio = pd.DataFrame([
        {'TradeID': 'T1', 'AssetClass': 'Equity', 'Exposure': 1e6, 'Delta': 0.6, 'Vega': 500, 'Rho': 0, 'CS01': 0},
        {'TradeID': 'T2', 'AssetClass': 'Credit', 'Exposure': 5e6, 'Delta': 0, 'Vega': 0, 'Rho': -100, 'CS01': 500}
    ])
    stress_engine = StressTestingEngine(portfolio)
    frtb_results = stress_engine.frtb_standard_scenarios()
    print("FRTB Stress Scenario Impacts:")
    for sc, pnl in frtb_results.items():
        print(f"  {sc}: ${pnl:,.2f}")

    # 3. COUNTERPARTY CREDIT RISK
    print("\n--- Counterparty Credit Risk (PFE & SIMM) ---")
    pfe_sim = PFESimulator(t_horizons=[0.25, 0.5, 1.0, 2.0, 5.0], num_paths=1000)
    profile = pfe_sim.simulate_gbm_exposure(S0=100, mu=0.05, sigma=0.2, K=100, option_type='call')
    print("PFE 99% Profile over time (0.25y -> 5y):")
    print(profile['PFE_99'])
    
    simm = SIMMCalculator()
    sensitivities = {'Equity': [1500, 2000], 'Rates': [-500, 1000]}
    initial_margin = simm.calculate_margin(sensitivities)
    print(f"SIMM Initial Margin: ${initial_margin:,.2f}")

    # 4. CAPITAL & REPORTING
    print("\n--- Capital Adequacy & Reporting ---")
    var_10d = VaREngine.time_scaling(100000, 10)
    es_10d = VaREngine.time_scaling(120000, 10)
    
    capital = CapitalAdequacy(internal_var=100000, rniv_addon=25000, stress_capital=50000)
    report = capital.generate_report(var_10d, es_10d)
    print("Capital Report:", report)
    
    db.insert_trade('T1', 'Equity', 1e6, bs_price * 10000)
    db.insert_metric('Total_IM', initial_margin)
    
    pt0 = pd.DataFrame([{'TradeID': 'T1', 'Exposure': 1000000}, {'TradeID': 'T2', 'Exposure': 5000000}])
    pt1 = pd.DataFrame([{'TradeID': 'T1', 'Exposure': 1050000}, {'TradeID': 'T3', 'Exposure': 2000000}])
    dod = DoDAttribution(pt0, pt1)
    file_exported = dod.export_excel('data/DoD_Exposure_Report.xlsx')
    print(f"DoD Attribution Report generated at: {file_exported}")
    
    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
