from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://glucose_values_user:b3TusfqGRWK2aVMwGfSL9iSttAYZxeoQ@dpg-d047p6a4d50c739tck5g-a/glucose_values"

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