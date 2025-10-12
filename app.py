import os, io, time, json, base64, pathlib
import numpy as np
import librosa
import requests

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ========= Firestore opcional =========
USE_FIREBASE = os.getenv("USE_FIREBASE", "0") == "1"
db = None
if USE_FIREBASE:
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        b64 = os.getenv("FIREBASE_CREDENTIALS_BASE64")
        if not b64:
            print("⚠️ USE_FIREBASE=1 pero falta FIREBASE_CREDENTIALS_BASE64")
        else:
            info = json.loads(base64.b64decode(b64).decode("utf-8"))
            cred = credentials.Certificate(info)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("✅ Firestore listo (project_id:", info.get("project_id"), ")")
    except Exception as e:
        print("🔥 Error inicializando Firestore:", e)

# ========= Config =========
# URLs para descargar el modelo/labels si no existen en disco
MODEL_URL  = os.getenv("MODEL_URL",  "")   # p.ej. https://github.com/<user>/<repo>/releases/download/v1.0/abejas_model_augmented.h5
LABELS_URL = os.getenv("LABELS_URL", "")   # opcional: https://.../labels.json

MODELS_DIR  = pathlib.Path("models")
MODEL_PATH  = MODELS_DIR / "abejas_model_augmented.h5"
LABELS_PATH = MODELS_DIR / "labels.json"

# Preprocesamiento (igual que en el training)
N_MELS        = 64
TARGET_FRAMES = 216
DURATION_SEC  = 5
SAMPLE_RATE   = None  # respeta SR original

def download_if_missing(url: str, dest: pathlib.Path):
    """Descarga un archivo a 'dest' solo si no existe."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if not url:
        raise RuntimeError(f"Falta URL para descargar {dest.name}")
    print(f"⬇️ Descargando {dest.name} ...")
    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✅ {dest.name} descargado ({dest.stat().st_size} bytes)")

# Descarga modelo/labels si faltan
if not MODEL_PATH.exists():
    download_if_missing(MODEL_URL, MODEL_PATH)
if not LABELS_PATH.exists() and LABELS_URL:
    download_if_missing(LABELS_URL, LABELS_PATH)

# Cargar labels
if LABELS_PATH.exists():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)
else:
    # Fallback si no subiste labels.json
    LABELS = ["reina_ausente", "sana"]

# Cargar modelo
if not MODEL_PATH.exists():
    raise RuntimeError(f"No se encontró {MODEL_PATH}")
print("🧠 Cargando modelo Keras...")
MODEL = load_model(str(MODEL_PATH))
print("✅ Modelo listo.")

# ========= FastAPI =========
app = FastAPI(title="BeeCare IA API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def to_fixed_mel(y, sr, n_mels=N_MELS, target_frames=TARGET_FRAMES):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0,0),(0, target_frames - mel_db.shape[1])), mode="constant")
    elif mel_db.shape[1] > target_frames:
        mel_db = mel_db[:, :target_frames]
    return mel_db

@app.get("/")
def root():
    return {"ok": True, "service": "BeeCare IA API"}

@app.post("/predict")
async def predict(
    hiveId: str | None = Query(default=None),
    userId: str | None = Query(default=None),
    file: UploadFile = File(...),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Sube un archivo .wav")

    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Archivo demasiado grande (máx 8MB)")

    # Carga y pad a 5s
    audio, sr = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, duration=DURATION_SEC)
    if len(audio) < DURATION_SEC * sr:
        audio = np.pad(audio, (0, DURATION_SEC * sr - len(audio)))

    mel = to_fixed_mel(audio, sr)
    x = mel[np.newaxis, ..., np.newaxis]   # (1, 64, 216, 1)

    probs = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    result = {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        "ts": int(time.time()),
        "filename": file.filename,
    }
    if hiveId: result["hiveId"] = hiveId
    if userId: result["userId"] = userId

    if db:
        db.collection("predictions").add(result)

    return result