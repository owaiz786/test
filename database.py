from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:GWQ4IfZMqIQZ3LPcjpERiBzs2N9IngLM@dpg-d0pdanre5dus73dml4s0-a.oregon-postgres.render.com/glubase"

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
