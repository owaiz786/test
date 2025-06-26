from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://glucose_data_user:UyshfP4YBDAmrbOX3lvZIBQIrFOkdWOP@dpg-d1enpuje5dus739uusd0-a/glucose_data"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ✅ This is what FastAPI is trying to import
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
