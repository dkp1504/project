import os
import io
import joblib
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Global variables for models
heart_model, heart_scaler = None, None
diab_model, diab_scaler = None, None
ort_session = None

# Paths to local ONNX and Scikit-Learn models
MODEL_DIR = "./models"
HEART_MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_rf_model.pkl")
HEART_SCALER_PATH = os.path.join(MODEL_DIR, "heart_disease_scaler.pkl")
DIAB_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
DIAB_SCALER_PATH = os.path.join(MODEL_DIR, "diabetes_scaler.pkl")
XRAY_ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")

def preprocess_xray_image(image_bytes):
    """Preprocess raw input image for ResNet50 ONNX model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    
    # ImageNet Standard Normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Rearrange dimensions: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_data = img_data.transpose(2, 0, 1)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for loading models efficiently on startup."""
    global heart_model, heart_scaler, diab_model, diab_scaler, ort_session
    try:
        print("Booting up multi-model diagnostics engine...")
        
        if os.path.exists(HEART_MODEL_PATH) and os.path.exists(HEART_SCALER_PATH):
            heart_model = joblib.load(HEART_MODEL_PATH)
            heart_scaler = joblib.load(HEART_SCALER_PATH)
            print("-> Heart Disease ML components loaded.")

        if os.path.exists(DIAB_MODEL_PATH) and os.path.exists(DIAB_SCALER_PATH):
            diab_model = joblib.load(DIAB_MODEL_PATH)
            diab_scaler = joblib.load(DIAB_SCALER_PATH)
            print("-> Diabetes ML components loaded.")

        if os.path.exists(XRAY_ONNX_PATH):
            ort_session = ort.InferenceSession(XRAY_ONNX_PATH)
            print("-> Standalone ONNX X-Ray Engine initialized!")
        else:
            print(f"WARNING: ONNX model missing at {XRAY_ONNX_PATH}")

        print("Initialization Sequence Completed Successfully!")
    except Exception as e:
        print(f"CRITICAL: Engine initialization failed: {str(e)}")
    
    yield

app = FastAPI(title="Healthcare Diagnostics Multi-Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.head("/")
async def root():
    return {
        "status": "online",
        "message": "Multi-Model Healthcare Diagnostics API is active and running!"
    }

class HeartInput(BaseModel):
    age: float; sex: int; cp: int; trestbps: float; chol: float; fbs: int
    restecg: int; thalach: float; exang: int; oldpeak: float; slope: int; ca: int; thal: int

class DiabetesInput(BaseModel):
    pregnancies: float; glucose: float; blood_pressure: float; skin_thickness: float
    insulin: float; bmi: float; dpf: float; age: float

@app.post("/api/predict/heart")
async def predict_heart(data: HeartInput):
    if not heart_model: 
        raise HTTPException(status_code=500, detail="Heart model uninitialized.")
    try:
        features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
                              data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])
        scaled = heart_scaler.transform(features)
        pred = int(heart_model.predict(scaled)[0])
        prob = float(heart_model.predict_proba(scaled)[0][pred])
        return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/diabetes")
async def predict_diabetes(data: DiabetesInput):
    if not diab_model: 
        raise HTTPException(status_code=500, detail="Diabetes model uninitialized.")
    try:
        features = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
                              data.insulin, data.bmi, data.dpf, data.age]])
        scaled = diab_scaler.transform(features)
        pred = int(diab_model.predict(scaled)[0])
        prob = float(diab_model.predict_proba(scaled)[0][pred])
        return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
    except Exception as e: 
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    if not ort_session:
        raise HTTPException(status_code=500, detail="ONNX model engine uninitialized.")
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_xray_image(image_bytes)

        # Run ONNX inference locally
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: input_tensor})[0]
        
        # Softmax probabilities
        exp_scores = np.exp(outputs - np.max(outputs))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        pred_class = int(np.argmax(probs[0]))
        confidence = float(probs[0][pred_class])

        class_mapping = {0: "NORMAL", 1: "PNEUMONIA"}
        prediction_label = class_mapping.get(pred_class, "NORMAL")

        return {
            "status": "success",
            "diagnostics": {
                "condition_detected": prediction_label,
                "confidence_score": round(confidence * 100, 2)
            }
        }
    except Exception as e: 
        raise HTTPException(status_code=400, detail=f"X-Ray evaluation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


































































































# import os
# import io
# import requests
# import joblib
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager

# # Global variables for tabular models
# heart_model, heart_scaler = None, None
# diab_model, diab_scaler = None, None

# # Local directory setup for lightweight .pkl models
# MODEL_DIR = "./models"
# HEART_MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_rf_model.pkl")
# HEART_SCALER_PATH = os.path.join(MODEL_DIR, "heart_disease_scaler.pkl")
# DIAB_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
# DIAB_SCALER_PATH = os.path.join(MODEL_DIR, "diabetes_scaler.pkl")

# # HUGGING FACE INFERENCE API CONFIGURATION
# # Free cloud inference offloaded to Hugging Face (Prevents Render 512MB RAM Crash)
# HF_API_URL = "https://api-inference.huggingface.co/models/dhananjay1504/pneumonia-resnet50"
# HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: Environment variable on Render

# # ----------------------------------------------------
# # LIFESPAN STARTUP HANDLER (ULTRA-LIGHTWEIGHT)
# # ----------------------------------------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global heart_model, heart_scaler, diab_model, diab_scaler
#     try:
#         print("Booting up ultra-lightweight backend engine...")
        
#         # Load Scikit-Learn Classifiers (.pkl files require < 30MB RAM)
#         if os.path.exists(HEART_MODEL_PATH) and os.path.exists(HEART_SCALER_PATH):
#             heart_model = joblib.load(HEART_MODEL_PATH)
#             heart_scaler = joblib.load(HEART_SCALER_PATH)
#             print("-> Heart Disease ML components loaded.")
#         else:
#             print("WARNING: Heart Disease model files missing in ./models/")

#         if os.path.exists(DIAB_MODEL_PATH) and os.path.exists(DIAB_SCALER_PATH):
#             diab_model = joblib.load(DIAB_MODEL_PATH)
#             diab_scaler = joblib.load(DIAB_SCALER_PATH)
#             print("-> Diabetes ML components loaded.")
#         else:
#             print("WARNING: Diabetes model files missing in ./models/")

#         print("Initialization Sequence Completed Successfully! Backend active under 70MB RAM.")
#     except Exception as e:
#         print(f"CRITICAL: Initialization failed: {str(e)}")
    
#     yield

# app = FastAPI(title="Advanced Multi-Modal Healthcare Diagnostics API", lifespan=lifespan)

# # CORS Middleware Setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------------------------------
# # PYDANTIC INPUT SCHEMAS
# # ----------------------------------------------------
# class HeartInput(BaseModel):
#     age: float; sex: int; cp: int; trestbps: float; chol: float; fbs: int
#     restecg: int; thalach: float; exang: int; oldpeak: float; slope: int; ca: int; thal: int

# class DiabetesInput(BaseModel):
#     pregnancies: float; glucose: float; blood_pressure: float; skin_thickness: float
#     insulin: float; bmi: float; dpf: float; age: float

# # ----------------------------------------------------
# # ROUTER ENDPOINTS
# # ----------------------------------------------------
# @app.post("/api/predict/heart")
# async def predict_heart(data: HeartInput):
#     if not heart_model: 
#         raise HTTPException(status_code=500, detail="Heart model uninitialized.")
#     try:
#         features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
#                               data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])
#         scaled = heart_scaler.transform(features)
#         pred = int(heart_model.predict(scaled)[0])
#         prob = float(heart_model.predict_proba(scaled)[0][pred])
#         return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/predict/diabetes")
# async def predict_diabetes(data: DiabetesInput):
#     if not diab_model: 
#         raise HTTPException(status_code=500, detail="Diabetes model uninitialized.")
#     try:
#         features = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
#                               data.insulin, data.bmi, data.dpf, data.age]])
#         scaled = diab_scaler.transform(features)
#         pred = int(diab_model.predict(scaled)[0])
#         prob = float(diab_model.predict_proba(scaled)[0][pred])
#         return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/predict/xray")
# async def predict_xray(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
        
#         # Binary image payload header
#         content_type = file.content_type if file.content_type else "image/jpeg"
#         headers = {
#             "Content-Type": content_type
#         }
        
#         if HF_TOKEN:
#             headers["Authorization"] = f"Bearer {HF_TOKEN}"
            
#         # Send binary payload to Direct Hugging Face Inference Endpoint
#         response = requests.post(
#             HF_API_URL, 
#             headers=headers, 
#             data=image_bytes, 
#             timeout=40
#         )
        
#         # 1. Cold Start handling (Model loading in HF memory)
#         if response.status_code == 503:
#             raise HTTPException(
#                 status_code=503, 
#                 detail="Hugging Face Model is waking up (Cold Start). Please try again in 10-15 seconds!"
#             )
            
#         # 2. Check for other errors
#         if response.status_code != 200:
#             raise HTTPException(
#                 status_code=response.status_code, 
#                 detail=f"HF Inference Error ({response.status_code}): {response.text}"
#             )
        
#         result = response.json()
        
#         # Parse prediction output from Hugging Face Pipeline
#         if isinstance(result, list) and len(result) > 0:
#             top_prediction = result[0]
#         elif isinstance(result, dict):
#             top_prediction = result
#         else:
#             raise ValueError(f"Unexpected response format from HF Hub: {result}")

#         raw_label = str(top_prediction.get("label", "NORMAL")).upper()
#         confidence = float(top_prediction.get("score", 0.0))
        
#         # Standardize prediction result
#         prediction_label = "PNEUMONIA" if ("PNEUMONIA" in raw_label or "LABEL_1" in raw_label) else "NORMAL"
        
#         return {
#             "status": "success",
#             "diagnostics": {
#                 "condition_detected": prediction_label,
#                 "confidence_score": round(confidence * 100, 2)
#             }
#         }
#     except HTTPException as http_ex:
#         raise http_ex
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=f"X-Ray evaluation failed: {str(e)}")
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


# import os
# import io
# import joblib
# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from contextlib import asynccontextmanager
# from huggingface_hub import InferenceClient

# # Global variables for tabular models
# heart_model, heart_scaler = None, None
# diab_model, diab_scaler = None, None

# # Local directory setup for lightweight .pkl models
# MODEL_DIR = "./models"
# HEART_MODEL_PATH = os.path.join(MODEL_DIR, "heart_disease_rf_model.pkl")
# HEART_SCALER_PATH = os.path.join(MODEL_DIR, "heart_disease_scaler.pkl")
# DIAB_MODEL_PATH = os.path.join(MODEL_DIR, "diabetes_rf_model.pkl")
# DIAB_SCALER_PATH = os.path.join(MODEL_DIR, "diabetes_scaler.pkl")

# # HUGGING FACE INFERENCE SDK CONFIGURATION
# HF_REPO_ID = "dhananjay1504/pneumonia-resnet50"
# HF_TOKEN = os.getenv("HF_TOKEN", None)  # Render Environment Variable

# # Official Hugging Face Client Initialize
# hf_client = InferenceClient(model=HF_REPO_ID, token=HF_TOKEN)

# # ----------------------------------------------------
# # LIFESPAN STARTUP HANDLER (ULTRA-LIGHTWEIGHT)
# # ----------------------------------------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global heart_model, heart_scaler, diab_model, diab_scaler
#     try:
#         print("Booting up ultra-lightweight backend engine...")
        
#         # Load Scikit-Learn Classifiers (.pkl files require < 30MB RAM)
#         if os.path.exists(HEART_MODEL_PATH) and os.path.exists(HEART_SCALER_PATH):
#             heart_model = joblib.load(HEART_MODEL_PATH)
#             heart_scaler = joblib.load(HEART_SCALER_PATH)
#             print("-> Heart Disease ML components loaded.")
#         else:
#             print("WARNING: Heart Disease model files missing in ./models/")

#         if os.path.exists(DIAB_MODEL_PATH) and os.path.exists(DIAB_SCALER_PATH):
#             diab_model = joblib.load(DIAB_MODEL_PATH)
#             diab_scaler = joblib.load(DIAB_SCALER_PATH)
#             print("-> Diabetes ML components loaded.")
#         else:
#             print("WARNING: Diabetes model files missing in ./models/")

#         print("Initialization Sequence Completed Successfully! Backend active under 70MB RAM.")
#     except Exception as e:
#         print(f"CRITICAL: Initialization failed: {str(e)}")
    
#     yield

# app = FastAPI(title="Advanced Multi-Modal Healthcare Diagnostics API", lifespan=lifespan)

# # CORS Middleware Setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------------------------------
# # ROOT ROUTE (Fixes Render Health Check 404)
# # ----------------------------------------------------
# @app.get("/")
# async def root():
#     return {
#         "status": "online",
#         "message": "Healthcare Multi-Model Diagnostics API is active and running!"
#     }

# # ----------------------------------------------------
# # PYDANTIC INPUT SCHEMAS
# # ----------------------------------------------------
# class HeartInput(BaseModel):
#     age: float; sex: int; cp: int; trestbps: float; chol: float; fbs: int
#     restecg: int; thalach: float; exang: int; oldpeak: float; slope: int; ca: int; thal: int

# class DiabetesInput(BaseModel):
#     pregnancies: float; glucose: float; blood_pressure: float; skin_thickness: float
#     insulin: float; bmi: float; dpf: float; age: float

# # ----------------------------------------------------
# # ROUTER ENDPOINTS
# # ----------------------------------------------------
# @app.post("/api/predict/heart")
# async def predict_heart(data: HeartInput):
#     if not heart_model: 
#         raise HTTPException(status_code=500, detail="Heart model uninitialized.")
#     try:
#         features = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
#                               data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, data.ca, data.thal]])
#         scaled = heart_scaler.transform(features)
#         pred = int(heart_model.predict(scaled)[0])
#         prob = float(heart_model.predict_proba(scaled)[0][pred])
#         return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/predict/diabetes")
# async def predict_diabetes(data: DiabetesInput):
#     if not diab_model: 
#         raise HTTPException(status_code=500, detail="Diabetes model uninitialized.")
#     try:
#         features = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
#                               data.insulin, data.bmi, data.dpf, data.age]])
#         scaled = diab_scaler.transform(features)
#         pred = int(diab_model.predict(scaled)[0])
#         prob = float(diab_model.predict_proba(scaled)[0][pred])
#         return {"status": "success", "diagnostics": {"detected": bool(pred), "confidence": round(prob * 100, 2)}}
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/api/predict/xray")
# async def predict_xray(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
        
#         # Hugging Face Official SDK Call
#         try:
#             result = hf_client.image_classification(image_bytes)
#         except Exception as hf_err:
#             error_str = str(hf_err)
#             if "503" in error_str or "loading" in error_str.lower():
#                 raise HTTPException(
#                     status_code=503,
#                     detail="Hugging Face Model is waking up (Cold Start). Please try again in 10-15 seconds!"
#                 )
#             raise HTTPException(status_code=400, detail=f"HF Inference Error: {error_str}")

#         # Parse output safely
#         if isinstance(result, list) and len(result) > 0:
#             top_pred = result[0]
#         else:
#             top_pred = result

#         # Extract label & score safely from SDK response object/dict
#         raw_label = str(getattr(top_pred, "label", top_pred.get("label", "NORMAL")) if isinstance(top_pred, dict) else getattr(top_pred, "label", "NORMAL")).upper()
#         confidence = float(getattr(top_pred, "score", top_pred.get("score", 0.0)) if isinstance(top_pred, dict) else getattr(top_pred, "score", 0.0))

#         # Standardize prediction result
#         prediction_label = "PNEUMONIA" if ("PNEUMONIA" in raw_label or "LABEL_1" in raw_label) else "NORMAL"

#         return {
#             "status": "success",
#             "diagnostics": {
#                 "condition_detected": prediction_label,
#                 "confidence_score": round(confidence * 100, 2)
#             }
#         }
#     except HTTPException as http_ex:
#         raise http_ex
#     except Exception as e: 
#         raise HTTPException(status_code=400, detail=f"X-Ray evaluation failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


