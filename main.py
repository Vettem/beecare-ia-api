# main.py

import io
import uuid
import json
import logging
from datetime import datetime
from pathlib import Path
import tempfile

import hashlib

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
MODEL_PATH = BASE_DIR / "model" / "model_queenbee.h5"
LABELS_PATH = BASE_DIR / "model" / "labels.json"

# Nombre del bucket (ajusta si usas otro)
BUCKET_NAME = "burnished-web-475115-b8.firebasestorage.app"

# Parámetros de preprocesamiento (ajústalos a tu entrenamiento real)
# Estos valores DEBEN coincidir con los usados al entrenar el modelo queenbee.h5
SAMPLE_RATE = 16000        # frecuencia de muestreo
DURATION = 10              # segundos de audio usados por predicción
N_MELS = 64                # número de filtros mel
CLIP_SAMPLES = SAMPLE_RATE * DURATION  # muestras totales por clip

# Clientes de GCP (usan credenciales por defecto en Cloud Run)
storage_client = storage.Client()
firestore_client = firestore.Client()

MODEL = None
LABELS = None


# ------------------------------------------------------------
# Carga del modelo y labels
# ------------------------------------------------------------
def load_model_and_labels():
    """
    Carga el modelo .h5 y el archivo de labels.json al iniciar la app.
    """
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
    Carga un archivo de audio desde disco y lo convierte en un tensor
    listo para el modelo de clasificación de colmenas con/sin reina.

    IMPORTANTE: este preprocesamiento DEBE coincidir con el usado
    para entrenar el modelo model_queenbee.h5.
    """
    # Cargar audio en mono y a SAMPLE_RATE (por ejemplo, 16 kHz)
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    if y.size == 0:
        raise HTTPException(status_code=400, detail="Audio vacío o no válido")

    # Ajustar duración: si es más corto, rellenar con ceros; si es más largo, recortar
    if len(y) < CLIP_SAMPLES:
        pad = CLIP_SAMPLES - len(y)
        y = np.pad(y, (0, pad))
    else:
        y = y[:CLIP_SAMPLES]

    # Mel-espectrograma con mismos parámetros que el entrenamiento
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalización igual que en el script de entrenamiento: aproximar a rango [0, 1]
    mel_norm = (mel_db + 80.0) / 80.0

    # Añadir batch y canal: (1, H, W, 1)
    mel_norm = mel_norm.astype("float32")
    mel_norm = mel_norm[np.newaxis, ..., np.newaxis]

    return mel_norm


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

        mel_input = preprocess_audio_file(tmp.name)

    # Ejecutar el modelo
    preds = MODEL.predict(mel_input)
    pred_idx = int(np.argmax(preds, axis=1)[0])
    probability = float(np.max(preds, axis=1)[0])

    # --- NUEVO: soportar labels como dict o como lista ---
    if isinstance(LABELS, dict):
        # Formato: { "0": "clase1", "1": "clase2", ... }
        prediction = LABELS.get(str(pred_idx), f"class_{pred_idx}")
    elif isinstance(LABELS, list):
        # Formato: ["clase1", "clase2", ...]
        if 0 <= pred_idx < len(LABELS):
            prediction = LABELS[pred_idx]
        else:
            prediction = f"class_{pred_idx}"
    else:
        # Por si acaso, fallback neutro
        prediction = f"class_{pred_idx}"
    # -----------------------------------------------------

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

@app.get("/")
def root():
    return {"message": "BeeCare IA API funcionando"}


@app.post("/analyze-audio")
async def analyze_audio(
    file: UploadFile = File(...),
    uid: str = Form(...),
    hive_id: str = Form(...)
):
    """
    Recibe un archivo de audio (desde la app móvil), lo sube a GCS,
    ejecuta el modelo y guarda el resultado en Firestore.
    """
    try:
        if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3"]:
            raise HTTPException(status_code=400, detail="Formato de audio no soportado")

        # Leer bytes del archivo subido
        contents = await file.read()

        # Generar un nombre único para el archivo
        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.wav"

        # Ruta dentro del bucket
        gcs_path = f"users/{uid}/hives/{hive_id}/device-audios/{filename}"

        # 1) Subir a Cloud Storage
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(contents, content_type="audio/wav")

        # 2) Ejecutar el modelo
        prediction, probability = run_inference_on_bytes(contents)

        # 3) Guardar en Firestore
        doc_ref = (
            firestore_client.collection("users")
            .document(uid)
            .collection("hives")
            .document(hive_id)
            .collection("audios")
            .document(audio_id)
        )

        doc_data = {
            "audioPath": gcs_path,
            "prediction": prediction,
            "probability": probability,
            "createdAt": datetime.utcnow(),
            "source": "device-upload",
        }

        doc_ref.set(doc_data)

        return {
            "audioId": audio_id,
            "audioPath": gcs_path,
            "prediction": prediction,
            "probability": probability,
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("Error en /analyze-audio", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al analizar audio")


@app.post("/analyze-audio-gcs")
async def analyze_audio_gcs(payload: GCSAnalyzeRequest):
    """
    Recibe la ruta de un audio ya existente en GCS, lo descarga,
    ejecuta el modelo y guarda el resultado en Firestore.
    """
    try:
        # 1) Descargar audio desde GCS
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(payload.gcs_path)

        if not blob.exists():
            raise HTTPException(status_code=404, detail="El archivo de audio no existe en GCS")

        audio_bytes = blob.download_as_bytes()

        # 2) Ejecutar el modelo
        prediction, probability = run_inference_on_bytes(audio_bytes)

        # 3) Guardar en Firestore
        # Generamos un ID a partir del hash del path + timestamp
        hash_input = f"{payload.gcs_path}-{datetime.utcnow().isoformat()}".encode("utf-8")
        analysis_id = hashlib.sha256(hash_input).hexdigest()[:16]

        doc_ref = (
            firestore_client.collection("users")
            .document(payload.uid)
            .collection("hives")
            .document(payload.hive_id)
            .collection("audios")
            .document(analysis_id)
        )

        doc_data = {
            "audioPath": payload.gcs_path,
            "prediction": prediction,
            "probability": probability,
            "createdAt": datetime.utcnow(),
            "source": "gcs-analyze",
        }

        doc_ref.set(doc_data)

        return {
            "audioId": analysis_id,
            "audioPath": payload.gcs_path,
            "prediction": prediction,
            "probability": probability,
        }

    except HTTPException:
        raise
    except Exception:
        logger.error("Error en /analyze-audio-gcs", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al analizar audio desde GCS")
# ------------------------------------------------------------
