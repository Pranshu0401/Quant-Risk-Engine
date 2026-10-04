import datetime
from sqlalchemy import create_engine, Column, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    trade_id = Column(String, primary_key=True)
    asset_class = Column(String)
    notional = Column(Float)
    mtm = Column(Float)
    as_of_date = Column(String)

class RiskMetric(Base):
    __tablename__ = 'risk_metrics'
    metric_name = Column(String, primary_key=True)
    value = Column(Float)
    as_of_date = Column(String)

class DBManager:
    def __init__(self, db_path='sqlite:///data/market_data.db'):
        self.engine = create_engine(db_path)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def insert_trade(self, trade_id, asset_class, notional, mtm, as_of_date=None):
        session = self.Session()
        trade = Trade(
            trade_id=trade_id,
            asset_class=asset_class,
            notional=notional,
            mtm=float(mtm),
            as_of_date=as_of_date or datetime.date.today().isoformat()
        )
        session.merge(trade)
        session.commit()
        session.close()

    def insert_metric(self, metric_name, value, as_of_date=None):
        session = self.Session()
        metric = RiskMetric(
            metric_name=metric_name,
            value=float(value),
            as_of_date=as_of_date or datetime.date.today().isoformat()
        )
        session.merge(metric)
        session.commit()
        session.close()

    def fetch_trades(self):
        session = self.Session()
        trades = session.query(Trade).all()
        session.close()
        return trades
