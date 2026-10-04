from src.pricing.black_scholes import BlackScholesPricer

def test_bs_call_price():
    price = BlackScholesPricer.price(100, 100, 1, 0.05, 0.2, 0, 'call')
    assert round(price, 4) == 10.4506

def test_bs_put_price():
    price = BlackScholesPricer.price(100, 100, 1, 0.05, 0.2, 0, 'put')
    assert round(price, 4) == 5.5735
