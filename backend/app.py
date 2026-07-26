import os
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from utils import extract_img_features, extract_voice_features 

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load pre-trained model
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise RuntimeError("MODEL_PATH environment variable is not set.")

# Resolve MODEL_PATH relative to BASE_DIR if it starts with 'app/' or './'
if MODEL_PATH.startswith("app/"):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH[4:])
elif MODEL_PATH.startswith("./"):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH[2:])

try:
    loaded_model = joblib.load(MODEL_PATH)
    drawing_model = joblib.load(os.path.join(BASE_DIR, "models", "drawing_model.pkl"))
    voice_model = joblib.load(os.path.join(BASE_DIR, "models", "voice_modell.pkl"))
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.post("/pro")
async def predict_parkinsons(audio: UploadFile = File(...), image: UploadFile = File(...)):
    audio_path = None
    image_path = None
    try:
        # Save uploaded files temporarily
        uploads_dir = os.path.join(BASE_DIR, "temp")
        os.makedirs(uploads_dir, exist_ok=True)

        # Use safe filenames
        audio_name = audio.filename if audio.filename else "temp_audio.wav"
        image_name = image.filename if image.filename else "temp_image.png"
        
        audio_path = os.path.join(uploads_dir, audio_name)
        image_path = os.path.join(uploads_dir, image_name)

        # Read content once and write
        audio_content = await audio.read()
        image_content = await image.read()

        if not audio_content or not image_content:
            raise ValueError("One or both uploaded files are empty.")

        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio_content)

        with open(image_path, "wb") as image_file:
            image_file.write(image_content)

        # Extract features
        voice_features = extract_voice_features(audio_path)
        img_features = extract_img_features(image_path)

        # Concatenate features
        model_input = np.concatenate((voice_features, img_features), axis=1)

        # Make prediction
        prediction = loaded_model.predict(model_input)[0]
        # Use float() to ensure it's JSON serializable
        confidence = float(max(loaded_model.predict_proba(model_input)[0]))

        # Return prediction
        return JSONResponse(content={
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence": confidence
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

@app.post("/voice")
async def predict_voice(audio: UploadFile = File(...)):
    try:
        uploads_dir = os.path.join(BASE_DIR, "temp")
        os.makedirs(uploads_dir, exist_ok=True)
        audio_path = os.path.join(uploads_dir, audio.filename)

        with open(audio_path, "wb") as audio_file:
            audio_file.write(await audio.read())

        voice_features = extract_voice_features(audio_path)

        prediction = voice_model.predict(voice_features)[0]
        confidence = max(voice_model.predict_proba(voice_features)[0])

        os.remove(audio_path)

        return JSONResponse(content={
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence": float(confidence)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.post("/drawing")
async def predict_drawing(image: UploadFile = File(...)):
    try:
        uploads_dir = os.path.join(BASE_DIR, "temp")
        os.makedirs(uploads_dir, exist_ok=True)
        image_path = os.path.join(uploads_dir, image.filename)

        with open(image_path, "wb") as image_file:
            image_file.write(await image.read())

        img_features = extract_img_features(image_path)

        prediction = drawing_model.predict(img_features)[0]
        confidence = max(drawing_model.predict_proba(img_features)[0])

        os.remove(image_path)

        return JSONResponse(content={
            "prediction": "Positive" if prediction == 1 else "Negative",
            "confidence": float(confidence)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
