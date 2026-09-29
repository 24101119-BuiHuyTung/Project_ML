import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

MODEL_PATH = "model.joblib"  # đặt file model.joblib cùng thư mục với server.py

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = None
if Path(MODEL_PATH).exists():
    bundle = joblib.load(MODEL_PATH)
    print("Đã load model. Các lớp:", bundle["classes"])
else:
    print(f"CẢNH BÁO: chưa thấy {MODEL_PATH} — hãy copy file model.joblib vào cùng thư mục.")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"ok": bundle is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if bundle is None:
        return JSONResponse({"error": "Chưa có model.joblib trên server"}, status_code=500)

    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Không đọc được ảnh"}, status_code=400)

    img = cv2.resize(img, bundle["img_size"])
    feat = img.flatten().reshape(1, -1)
    feat_scaled = bundle["scaler"].transform(feat)
    feat_pca = bundle["pca"].transform(feat_scaled)

    label = bundle["model"].predict(feat_pca)[0]
    proba = bundle["model"].predict_proba(feat_pca)[0]
    classes = bundle["model"].classes_
    confidence = float(max(proba))

    return {"label": str(label), "confidence": confidence,
            "all_probs": {str(c): float(p) for c, p in zip(classes, proba)}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
