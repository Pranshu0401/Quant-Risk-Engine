import os

project_path = "/home/pentxp/Downloads/Quant"
os.makedirs(project_path, exist_ok=True)
os.chdir(project_path)

files = {
    "src/pricing/__init__.py": "",
    "src/pricing/black_scholes.py": '''import numpy as np
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
''',
    "src/pricing/fixed_income.py": '''import numpy as np

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
''',
    "src/pricing/cds_pricer.py": '''import numpy as np

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
''',
    "src/market_risk/__init__.py": "",
    "src/market_risk/var_engine.py": '''import numpy as np
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
''',
    "src/market_risk/rniv.py": '''import numpy as np
from scipy.stats import norm

class RNIVCalculator:
    def __init__(self, risk_factors, confidence_level=0.99):
        self.risk_factors = risk_factors
        self.confidence_level = confidence_level
        self.rniv_items = {}

    def add_standalone_rniv(self, name, exposure, volatility, horizon_days=10):
        z = norm.ppf(self.confidence_level)
        amount = exposure * volatility * z * np.sqrt(horizon_days)
        self.rniv_items[name] = amount
        return amount

    def aggregate_rniv(self, correlation_matrix=None):
        amounts = np.array(list(self.rniv_items.values()))
        if correlation_matrix is None:
            return np.sum(amounts)
        return np.sqrt(np.dot(amounts.T, np.dot(correlation_matrix, amounts)))
''',
    "src/market_risk/stress_testing.py": '''import numpy as np

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
''',
    "src/market_risk/backtester.py": '''import numpy as np
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
''',
    "src/credit_risk/__init__.py": "",
    "src/credit_risk/pfe_simulator.py": '''import numpy as np

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
''',
    "src/credit_risk/simm_calculator.py": '''import numpy as np

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
''',
    "src/capital/__init__.py": "",
    "src/capital/capital_adequacy.py": '''class CapitalAdequacy:
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
''',
    "src/reporting/__init__.py": "",
    "src/reporting/db_manager.py": '''import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = sa.Column(sa.Integer, primary_key=True)
    trade_id = sa.Column(sa.String, unique=True, nullable=False)
    asset_class = sa.Column(sa.String, nullable=False)
    notional = sa.Column(sa.Float, nullable=False)
    mtm = sa.Column(sa.Float, nullable=False)
    as_of_date = sa.Column(sa.Date, default=datetime.utcnow)

class RiskMetric(Base):
    __tablename__ = 'risk_metrics'
    id = sa.Column(sa.Integer, primary_key=True)
    metric_name = sa.Column(sa.String, nullable=False)
    value = sa.Column(sa.Float, nullable=False)
    as_of_date = sa.Column(sa.Date, default=datetime.utcnow)

class DBManager:
    def __init__(self, db_url='sqlite:///risk_engine.db'):
        self.engine = sa.create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_trade(self, trade_id, asset_class, notional, mtm):
        session = self.Session()
        trade = Trade(trade_id=trade_id, asset_class=asset_class, notional=notional, mtm=mtm, as_of_date=datetime.now().date())
        session.merge(trade)
        session.commit()
        session.close()

    def insert_metric(self, metric_name, value):
        session = self.Session()
        metric = RiskMetric(metric_name=metric_name, value=value, as_of_date=datetime.now().date())
        session.add(metric)
        session.commit()
        session.close()
''',
    "src/reporting/dod_attribution.py": '''import pandas as pd

class DoDAttribution:
    def __init__(self, portfolio_t0, portfolio_t1):
        self.t0 = portfolio_t0.set_index('TradeID')
        self.t1 = portfolio_t1.set_index('TradeID')

    def compute_attribution(self):
        merged = self.t0.join(self.t1, lsuffix='_t0', rsuffix='_t1', how='outer').fillna(0)
        merged['DoD_Change'] = merged['Exposure_t1'] - merged['Exposure_t0']
        
        def categorize(row):
            if row['Exposure_t0'] == 0 and row['Exposure_t1'] != 0: return 'New Trade'
            if row['Exposure_t0'] != 0 and row['Exposure_t1'] == 0: return 'Matured/Unwound'
            return 'Market Move'
                
        merged['Driver'] = merged.apply(categorize, axis=1)
        return merged

    def export_excel(self, filename='DoD_Exposure_Report.xlsx'):
        df = self.compute_attribution()
        df.to_excel(filename, sheet_name='DoD Attribution')
        return filename
''',
    "vba/ExposureReportMacro.bas": '''Attribute VB_Name = "ExposureReportMacro"
Sub FormatExposureReport()
    Dim ws As Worksheet
    Set ws = ActiveWorkbook.Sheets("DoD Attribution")
    
    Dim lastRow As Long
    lastRow = ws.Cells(ws.Rows.Count, "A").End(xlUp).Row
    
    ws.Range("A1:E1").Font.Bold = True
    ws.Range("A1:E1").Interior.Color = RGB(0, 51, 102)
    ws.Range("A1:E1").Font.Color = RGB(255, 255, 255)
    
    Dim cell As Range
    For Each cell In ws.Range("D2:D" & lastRow)
        If Abs(cell.Value) > 100000 Then
            cell.Interior.Color = RGB(255, 153, 153)
            cell.Font.Bold = True
            cell.Font.Color = RGB(153, 0, 0)
        End If
    Next cell
    
    ws.Columns("A:E").AutoFit
    MsgBox "Report Formatted and Exceptions Highlighted successfully.", vbInformation
End Sub
''',
    ".gitlab-ci.yml": '''stages:
  - test
  - deploy

variables:
  PYTHON_VERSION: "3.9"

test_models:
  image: python:${PYTHON_VERSION}
  stage: test
  script:
    - pip install -r requirements.txt
    - pytest tests/ --junitxml=report.xml
  artifacts:
    when: always
    reports:
      junit: report.xml

deploy_reports:
  stage: deploy
  script:
    - echo "Deploying Risk Reports to internal SFTP..."
  only:
    - main
''',
    "README.md": '''# Quantitative Risk Engine

A modular, production-ready quantitative risk engine modeling both Market Risk (RMG) and Counterparty Credit Risk (CEM).
This system replaces legacy EUC (End User Computing) spreadsheets with an automated, version-controlled pipeline.

## Features
- **Pricing Modules**: Black-Scholes options, Fixed Income DCF (Bonds, IRS), and Hazard-rate CDS.
- **Market Risk**: Historical, Parametric, and Monte Carlo VaR & Expected Shortfall (ES). RNIV calculation.
- **Stress Testing**: FRTB standard scenarios.
- **Credit Risk**: PFE & EE Monte Carlo simulation, ISDA-SIMM style Initial Margin calculator.
- **Capital & Reporting**: Economic vs Regulatory Capital, SQLite tracking, Day-on-Day exposure attribution.

## Math Details
- **VaR/ES**: Parametric VaR computed via $VaR = \\mu + Z_{\\alpha}\\sigma$. Expected Shortfall represents $E[L | L > VaR]$.
- **PFE**: Geometric Brownian Motion for exposure simulation.
- **SIMM**: Sensitivity-based IM calculated via a variance-covariance aggregation.

## EUC Migration Justification
By transitioning from VBA-heavy Excel spreadsheets to this Python-GitLab CI framework:
- We achieve deterministic, reproducible model outputs.
- Version control ensures transparency and auditability for model validators.
- Automated `pytest` pipelines prevent numerical regressions.
- SQLite handles larger datasets than Excel limits.

## Execution
Run the end-to-end pipeline:
```bash
python run_pipeline.py
```
''',
    "requirements.txt": '''numpy
scipy
pandas
sqlalchemy
openpyxl
pytest
''',
    "tests/__init__.py": "",
    "tests/test_pricing.py": '''from src.pricing.black_scholes import BlackScholesPricer

def test_bs_call_price():
    price = BlackScholesPricer.price(100, 100, 1, 0.05, 0.2, 0, 'call')
    assert round(price, 4) == 10.4506

def test_bs_put_price():
    price = BlackScholesPricer.price(100, 100, 1, 0.05, 0.2, 0, 'put')
    assert round(price, 4) == 5.5735
''',
    "run_pipeline.py": '''import numpy as np
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
    print("\\n--- Pricing Modules ---")
    bs_price = BlackScholesPricer.price(S=100, K=105, T=1, r=0.05, sigma=0.2)
    bs_greeks = BlackScholesPricer.greeks(S=100, K=105, T=1, r=0.05, sigma=0.2)
    print(f"Call Option Price: {bs_price:.4f}")
    print(f"Greeks: {bs_greeks}")
    
    bond_price = FixedIncomePricer.bond_price(100, 0.05, 0.04, 5)
    print(f"5Y Bond Price (5% coupon, 4% yield): {bond_price:.4f}")
    
    cds_price = CDSPricer.price(notional=1e6, spread=0.015, hazard_rate=0.02, recovery_rate=0.4, r=0.03, T=5)
    print(f"5Y CDS Protection Value: {cds_price:.2f}")

    # 2. MARKET RISK
    print("\\n--- Market Risk (VaR & Stress) ---")
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
    print("\\n--- Counterparty Credit Risk (PFE & SIMM) ---")
    pfe_sim = PFESimulator(t_horizons=[0.25, 0.5, 1.0, 2.0, 5.0], num_paths=1000)
    profile = pfe_sim.simulate_gbm_exposure(S0=100, mu=0.05, sigma=0.2, K=100, option_type='call')
    print("PFE 99% Profile over time (0.25y -> 5y):")
    print(profile['PFE_99'])
    
    simm = SIMMCalculator()
    sensitivities = {'Equity': [1500, 2000], 'Rates': [-500, 1000]}
    initial_margin = simm.calculate_margin(sensitivities)
    print(f"SIMM Initial Margin: ${initial_margin:,.2f}")

    # 4. CAPITAL & REPORTING
    print("\\n--- Capital Adequacy & Reporting ---")
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
    
    print("\\n=== Pipeline Execution Completed Successfully ===")

if __name__ == "__main__":
    main()
'''
}

for path, content in files.items():
    full_path = os.path.join(project_path, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
