from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from estimator import ImprovedGlucoseEstimator
from starlette.responses import Response
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from database import SessionLocal, engine
from models import Base, GlucoseRecord
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import get_db
from models import Glucose

# Initialize FastAPI app
Base.metadata.create_all(bind=engine)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
estimator = ImprovedGlucoseEstimator()

# Mount static folder if you have JS/CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, _ = estimator.process_frame(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        cap.release()
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/eye_feed")
def eye_feed():
    def generate():
        while True:
            eye_img = estimator.eye_display
            _, jpeg = cv2.imencode('.jpg', eye_img)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/glucose_data")
def glucose_data():
    if estimator.history_buffer:
        latest = estimator.history_buffer[-1]
        return {"glucose": float(latest)}
    else:
        return {"glucose": 0.0}

@app.get("/glucose_chart")
def glucose_chart():
    return {
        "values": estimator.glucose_values[-100:],  # limit last 100 values
        "times": estimator.time_values[-100:]
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    processed_frame, glucose = estimator.process_frame(frame)

    if glucose is None:
        return JSONResponse(content={"status": "collecting", "collected": len(estimator.feature_buffer)}, status_code=202)

    return {"glucose": glucose}

# Real glucose submission route
@app.post("/submit_real_glucose")
async def submit_real_glucose(request: Request, real_glucose: float = Form(...)):
    # Calculate average from estimator
    if estimator.glucose_values:
        estimated_avg = sum(estimator.glucose_values) / len(estimator.glucose_values)
    else:
        estimated_avg = 0.0

    # Save to database
    db = SessionLocal()
    record = GlucoseRecord(real_glucose=real_glucose, estimated_avg=estimated_avg)
    db.add(record)
    db.commit()
    db.close()

    # Render the success message on the index page
    return templates.TemplateResponse("index.html", {"request": request, "message": "Real glucose data saved."})

# Get all records from the database
@app.get("/records")
def get_records():
    db = SessionLocal()
    records = db.query(GlucoseRecord).all()
    db.close()
    return {"records": [{"real": r.real_glucose, "estimated_avg": r.estimated_avg, "timestamp": r.timestamp} for r in records]}

# Stop monitoring and save data
@app.post("/stop_monitoring")
async def stop_monitoring(request: Request, real_glucose: float = Form(...)):
    # Calculate average glucose from the estimator
    if estimator.glucose_values:
        estimated_avg = sum(estimator.glucose_values) / len(estimator.glucose_values)
    else:
        estimated_avg = 0.0

    # Save the record to the database
    db = SessionLocal()
    record = GlucoseRecord(real_glucose=real_glucose, estimated_avg=estimated_avg)
    db.add(record)
    db.commit()
    db.close()

    # Clear the estimator buffers for the next session
    estimator.glucose_values.clear()
    estimator.time_values.clear()

    # Return a response that renders the index page with a success message
    return templates.TemplateResponse("index.html", {"request": request, "message": "Monitoring stopped and data saved."})


@app.get("/glucose/all")
def get_all_data(db: Session = Depends(get_db)):
    return db.query(Glucose).all()

