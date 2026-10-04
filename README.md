# Multi-Asset Quantitative Risk Engine & Exposure Analytics Pipeline

A modular, production-ready quantitative risk engine modeling both Market Risk (RMG) and Counterparty Credit Risk (CEM).
This system replaces legacy EUC (End User Computing) spreadsheets with an automated, version-controlled pipeline.

## Features
- **Pricing Modules**: Black-Scholes options, Fixed Income DCF (Bonds, IRS), and Hazard-rate CDS.
- **Market Risk**: Historical, Parametric, and Monte Carlo VaR & Expected Shortfall (ES). RNIV calculation.
- **Stress Testing**: FRTB standard scenarios.
- **Credit Risk**: PFE & EE Monte Carlo simulation, ISDA-SIMM style Initial Margin calculator.
- **Capital & Reporting**: Economic vs Regulatory Capital, SQLite tracking, Day-on-Day exposure attribution.

## Math Details
- **VaR/ES**: Parametric VaR computed via $VaR = \mu + Z_{\alpha}\sigma$. Expected Shortfall represents $E[L | L > VaR]$.
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
