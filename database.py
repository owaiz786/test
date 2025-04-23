from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://glucose_values_user:b3TusfqGRWK2aVMwGfSL9iSttAYZxeoQ@dpg-d047p6a4d50c739tck5g-a/glucose_values"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
