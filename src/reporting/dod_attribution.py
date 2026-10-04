import pandas as pd

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
