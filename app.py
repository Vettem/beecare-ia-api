import io, os, time, json
import numpy as np
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ===== Config preprocesamiento (igual que en el training) =====
N_MELS = 64
TARGET_FRAMES = 216
DURATION_SEC = 5
SAMPLE_RATE = None   # respeta SR original

# ===== App =====
app = FastAPI(title="BeeCare IA API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # en prod: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Cargar labels y modelo =====
LABELS_PATH = "models/labels.json"
MODEL_PATH = "models/abejas_model_augmented.h5"

if not os.path.exists(LABELS_PATH):
    raise RuntimeError(f"No se encontró {LABELS_PATH}")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró {MODEL_PATH}")
MODEL = load_model(MODEL_PATH)

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
async def predict(file: UploadFile = File(...)):
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
    return {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        "ts": int(time.time()),
    }
