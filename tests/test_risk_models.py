import numpy as np
import pytest
from src.market_risk.var_engine import VaREngine
from src.market_risk.backtester import KupiecPOFTest
from src.credit_risk.simm_calculator import SIMMCalculator
from src.credit_risk.pfe_simulator import PFESimulator

def test_var_and_expected_shortfall():
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, 1000)
    var_99, es_99 = VaREngine.historical_var(returns, 0.99)
    assert var_99 > 0
    assert es_99 >= var_99

def test_kupiec_pof_acceptance():
    lr_stat, p_val, model_accepted = KupiecPOFTest.test(exceptions=3, observations=250, confidence_level=0.99)
    assert bool(model_accepted) is True
    assert p_val > 0.05

def test_kupiec_pof_rejection():
    lr_stat, p_val, model_accepted = KupiecPOFTest.test(exceptions=15, observations=250, confidence_level=0.99)
    assert bool(model_accepted) is False
    assert p_val < 0.05

def test_simm_margin_positive():
    simm = SIMMCalculator()
    margin = simm.calculate_margin({'Equity': [1000.0, -500.0], 'Rates': [300.0]})
    assert margin > 0.0

def test_pfe_monotonic_expansion():
    pfe = PFESimulator(t_horizons=[0.5, 1.0, 2.0], num_paths=500)
    res = pfe.simulate_gbm_exposure(S0=100, mu=0.03, sigma=0.2, K=100)
    assert res['PFE_99'][-1] > res['PFE_99'][0]
