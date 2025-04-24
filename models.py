from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class GlucoseRecord(Base):
    __tablename__ = "glucose_records"

    id = Column(Integer, primary_key=True, index=True)
    real_glucose = Column(Float)
    estimated_avg = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    
    