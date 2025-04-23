from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Form
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
from fastapi import Request, Form
from fastapi.responses import JSONResponse

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

# ✅ ADDED: Real glucose submission route
@app.post("/submit_real_glucose")
async def submit_real_glucose(request: Request, real_glucose: float = Form(...)):
    print(f"Real Glucose submitted: {real_glucose}")
    with open("real_glucose_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - Real Glucose: {real_glucose}\n")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/submit_real_glucose")
async def submit_real_glucose(request: Request, real_glucose: float = Form(...)):
    print(f"Real Glucose submitted: {real_glucose}")
    # Save to file (optional)
    with open("real_glucose_log.txt", "a") as f:
        f.write(f"{datetime.now().isoformat()} - Real Glucose: {real_glucose}\n")
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/submit_real_glucose")
async def submit_real_glucose(request: Request, real_glucose: float = Form(...)):
    # Calculate average from estimator
    if estimator.glucose_values:
        estimated_avg = sum(estimator.glucose_values) / len(estimator.glucose_values)
    else:
        estimated_avg = 0.0

    db = SessionLocal()
    record = GlucoseRecord(real_glucose=real_glucose, estimated_avg=estimated_avg)
    db.add(record)
    db.commit()
    db.close()

    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/records")
def get_records():
    db = SessionLocal()
    records = db.query(GlucoseRecord).all()
    db.close()
    return {"records": [{"real": r.real_glucose, "estimated_avg": r.estimated_avg, "timestamp": r.timestamp} for r in records]}

@app.post("/stop_monitoring")
async def stop_monitoring(request: Request, real_glucose: float = Form(...)):
    if estimator.glucose_values:
        estimated_avg = sum(estimator.glucose_values) / len(estimator.glucose_values)
    else:
        estimated_avg = 0.0

    db = SessionLocal()
    record = GlucoseRecord(real_glucose=real_glucose, estimated_avg=estimated_avg)
    db.add(record)
    db.commit()
    db.close()

    # Clear for next session
    estimator.glucose_values.clear()
    estimator.time_values.clear()

    return JSONResponse(content={"message": "Monitoring stopped and data saved."})

    if glucose is None:
        return JSONResponse(content={"status": "collecting", "collected": len(estimator.feature_buffer)}, status_code=202)

    return {"glucose": glucose}



