from sqlalchemy import Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# Creating a Base class for SQLAlchemy ORM
Base = declarative_base()

class GlucoseRecord(Base):
    __tablename__ = "glucose_records"

    # Primary key column
    id = Column(Integer, primary_key=True, index=True)

    # Glucose value
    real_glucose = Column(Float)

    # Estimated average glucose value
    estimated_avg = Column(Float)

    # Timestamp for when the record is created
    timestamp = Column(DateTime, default=datetime.utcnow)  # Default to UTC now
