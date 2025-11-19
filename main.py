# main.py

import io
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf
import librosa

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from google.cloud import storage, firestore

# ------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beecare-ia-api")

app = FastAPI(
    title="BeeCare IA API",
    version="1.0.0",
    description="API de análisis de audio de colmenas para BeeCare",
)

BASE_DIR = Path(__file__).resolve().parent

# Rutas del modelo y labels dentro de la imagen
MODEL_PATH = BASE_DIR / "model" / "abejas_model_augmented.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

# Nombre del bucket (ajusta si usas otro)
BUCKET_NAME = "burnished-web-475115-b8.firebasestorage.app"

# Parámetros de preprocesamiento (ajústalos a tu entrenamiento real)
SAMPLE_RATE = 22050
DURATION = 5           # segundos máximos a considerar
N_MELS = 128
FIXED_TIME_STEPS = 128  # ancho del mel-espectrograma

# Clientes de GCP (usan credenciales por defecto en Cloud Run)
storage_client = storage.Client()
firestore_client = firestore.Client()

MODEL = None
LABELS = None


# ------------------------------------------------------------
# Carga del modelo y labels
# ------------------------------------------------------------

def load_model_and_labels():
    global MODEL, LABELS

    if MODEL is not None and LABELS is not None:
        return

    try:
        logger.info(f"Cargando modelo desde {MODEL_PATH}")
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))

        logger.info(f"Cargando labels desde {LABELS_PATH}")
        with LABELS_PATH.open("r", encoding="utf-8") as f:
            LABELS = json.load(f)

        logger.info("Modelo y labels cargados correctamente.")
    except Exception as e:
        logger.error("Error al cargar modelo o labels", exc_info=True)
        MODEL = None
        LABELS = None
        raise


# Cargamos al iniciar
load_model_and_labels()


# ------------------------------------------------------------
# Preprocesamiento e inferencia
# ------------------------------------------------------------

def preprocess_audio_file(filepath: str) -> np.ndarray:
    """
    Carga un archivo de audio y lo convierte en un tensor listo
    para el modelo. Ajusta esta función para que coincida con
    el preprocesamiento usado en el entrenamiento.
    """
    # Cargar audio (recorta a 'DURATION' para no explotar memoria)
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Audio vacío o no válido")

    # Mel-espectrograma
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalizar tamaño en el eje de tiempo (padding/truncado)
    if mel_db.shape[1] < FIXED_TIME_STEPS:
        pad_width = FIXED_TIME_STEPS - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :FIXED_TIME_STEPS]

    # Normalización básica [0,1]
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_db = (mel_db - mel_min) / (mel_max - mel_min + 1e-9)

    # Añadir batch y canal: (1, H, W, 1)
    mel_db = mel_db.astype("float32")
    mel_db = mel_db[np.newaxis, ..., np.newaxis]

    return mel_db


def run_inference_on_bytes(audio_bytes: bytes):
    """
    Recibe los bytes de un archivo de audio, los guarda en un
    temporal, aplica el preprocesamiento y ejecuta el modelo.
    Devuelve (prediction, probability).
    """
    if MODEL is None or LABELS is None:
        raise HTTPException(status_code=500, detail="Modelo no disponible en el servidor")

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        input_tensor = preprocess_audio_file(tmp.name)

    preds = MODEL.predict(input_tensor)[0]
    pred_idx = int(np.argmax(preds))
    probability = float(preds[pred_idx])

    # LABELS se asume dict: { "0": "clase1", "1": "clase2", ... }
    prediction = LABELS.get(str(pred_idx), f"class_{pred_idx}")

    return prediction, probability


# ------------------------------------------------------------
# Modelos Pydantic
# ------------------------------------------------------------

class GCSAnalyzeRequest(BaseModel):
    uid: str
    hive_id: str
    gcs_path: str   # ej: "users/UID/hives/colmena1/device-audios/archivo.wav"


# ------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "BeeCare IA API funcionando"}


@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    uid: str = Form(...),
    hive_id: str = Form(...)
):
    """
    Recibe un archivo de audio, lo sube a Storage,
    lo analiza con el modelo y guarda el resultado en Firestore.
    """
    try:
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        # 1) Subir al bucket
        audio_id = str(uuid.uuid4())
        gcs_path = f"users/{uid}/hives/{hive_id}/audios/{audio_id}.wav"

        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(audio_bytes, content_type="audio/wav")

        # 2) Ejecutar el modelo
        prediction, probability = run_inference_on_bytes(audio_bytes)

        # 3) Guardar en Firestore
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
                "status": "ok",
                "source": "api_upload",
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
        )

        return {
            "audioId": audio_id,
            "audioPath": gcs_path,
            "prediction": prediction,
            "probability": probability,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error en /analyze-audio", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al analizar audio")


@app.post("/analyze-audio-gcs")
async def analyze_audio_gcs(payload: GCSAnalyzeRequest):
    """
    Analiza un audio que YA está en Cloud Storage.
    La Cloud Function llamará a este endpoint cuando
    se suba un nuevo .wav desde el dispositivo.
    """
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(payload.gcs_path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="Audio no encontrado en Storage")

        audio_bytes = blob.download_as_bytes()

        prediction, probability = run_inference_on_bytes(audio_bytes)

        analysis_id = str(uuid.uuid4())

        doc_ref = (
            firestore_client.collection("users")
            .document(payload.uid)
            .collection("hives")
            .document(payload.hive_id)
            .collection("audios")
            .document(analysis_id)
        )

        doc_ref.set(
            {
                "audioPath": payload.gcs_path,
                "prediction": prediction,
                "probability": probability,
                "status": "ok",
                "source": "device_gcs",
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
        )

        return {
            "audioId": analysis_id,
            "audioPath": payload.gcs_path,
            "prediction": prediction,
            "probability": probability,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error en /analyze-audio-gcs", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al analizar audio desde GCS")