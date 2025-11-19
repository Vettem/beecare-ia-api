# main.py
import io
import uuid
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import tensorflow as tf
import librosa

from google.cloud import storage, firestore

import json
import os

app = FastAPI(title="BeeCare IA API")

# --------- CONFIG ---------
BUCKET_NAME = "burnished-web-475115-b8.firebasestorage.app"

# Cargar modelo y labels al iniciar el contenedor
MODEL = None
LABELS = None


def load_model_and_labels():
    global MODEL, LABELS
    if MODEL is None:
        MODEL = tf.keras.models.load_model("model/abejas_model_augmented.h5")
    if LABELS is None:
        with open("model/labels.json", "r", encoding="utf-8") as f:
            LABELS = json.load(f)


load_model_and_labels()

# Clientes de GCP
storage_client = storage.Client()
firestore_client = firestore.Client()


# --------- UTIL: PREPROCESAR AUDIO ---------
def preprocess_audio(file_bytes: bytes, sr_target: int = 16000):
    """
    Ajusta esta función según cómo entrenaste el modelo
    (mfcc, duración, etc.).
    """
    # Cargar audio desde bytes
    audio_np, sr = librosa.load(io.BytesIO(file_bytes), sr=sr_target)

    # EJEMPLO de pipeline genérico: MFCC + padding
    mfcc = librosa.feature.mfcc(
        y=audio_np,
        sr=sr_target,
        n_mfcc=40
    )

    # Normalizar dimensiones para el modelo (ejemplo)
    # Supongamos que el modelo espera (time, features, 1)
    mfcc = mfcc.T  # (time, 40)
    mfcc = np.expand_dims(mfcc, axis=-1)  # (time, 40, 1)
    mfcc = np.expand_dims(mfcc, axis=0)   # (1, time, 40, 1)

    return mfcc


# --------- ENDPOINTS ---------

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "BeeCare IA API alive"}


@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    uid: str = Form(...),
    hive_id: str = Form(...),
):
    """
    Recibe un audio, lo sube a Storage, lo analiza con el modelo
    y guarda resultado en Firestore.
    """
    try:
        # 1. Leer bytes del archivo
        file_bytes = await file.read()

        # 2. Generar id y path en Storage
        audio_id = str(uuid.uuid4())
        file_ext = os.path.splitext(file.filename)[1] or ".wav"
        gcs_path = f"users/{uid}/hives/{hive_id}/audios/{audio_id}{file_ext}"

        # 3. Subir a Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(file_bytes, content_type=file.content_type)

        # 4. Preprocesar audio para el modelo
        input_data = preprocess_audio(file_bytes)

        # 5. Predicción
        preds = MODEL.predict(input_data)
        probs = tf.nn.softmax(preds[0]).numpy()

        # Suponiendo que LABELS = ["reina_ausente", "sana"]
        best_idx = int(np.argmax(probs))
        prediction = LABELS[best_idx]
        probability = float(probs[best_idx])

        # 6. Guardar en Firestore
        doc_ref = (
            firestore_client.collection("users")
            .document(uid)
            .collection("hives")
            .document(hive_id)
            .collection("audios")
            .document(audio_id)
        )

        doc_ref.set(
            {
                "audioPath": gcs_path,
                "prediction": prediction,
                "probability": probability,
                "createdAt": datetime.utcnow(),
                "status": "ok",
            }
        )

        # 7. Devolver respuesta al cliente
        return JSONResponse(
            {
                "audioId": audio_id,
                "audioPath": gcs_path,
                "prediction": prediction,
                "probability": probability,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
