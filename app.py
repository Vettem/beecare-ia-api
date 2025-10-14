# app.py
import io, os, time, json, base64
import numpy as np
import librosa

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ===== Config (igual que el training) =====
N_MELS = 64
TARGET_FRAMES = 216
DURATION_SEC = 5
SAMPLE_RATE = None  # respeta SR original

# ===== Firebase opcional por variable de entorno =====
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

# ===== FastAPI =====
app = FastAPI(title="BeeCare IA API (local)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # en prod restringe dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Rutas a modelo/labels locales =====
LABELS_PATH = "models/labels.json"
MODEL_PATH  = "models/abejas_model_augmented.h5"

if not os.path.exists(LABELS_PATH):
    raise RuntimeError(f"No se encontró {LABELS_PATH}")
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró {MODEL_PATH}")
print("🧠 Cargando modelo Keras local…")
MODEL = load_model(MODEL_PATH)
print("✅ Modelo listo.")

def to_fixed_mel(y, sr, n_mels=N_MELS, target_frames=TARGET_FRAMES):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalización [0,1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Ajuste de frames
    if mel_db.shape[1] < target_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_frames - mel_db.shape[1])), mode="constant")
    elif mel_db.shape[1] > target_frames:
        mel_db = mel_db[:, :target_frames]
    return mel_db

@app.get("/")
def root():
    return {"ok": True, "service": "BeeCare IA API (local)"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    hiveId: str | None = Query(default=None),
    userId: str | None = Query(default=None),
):
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Sube un archivo .wav")

    raw = await file.read()
    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Archivo demasiado grande (máx 8MB)")

    # Cargar y pad a 5s
    audio, sr = librosa.load(io.BytesIO(raw), sr=SAMPLE_RATE, duration=DURATION_SEC)
    if len(audio) < DURATION_SEC * sr:
        audio = np.pad(audio, (0, DURATION_SEC * sr - len(audio)))

    mel = to_fixed_mel(audio, sr)
    x = mel[np.newaxis, ..., np.newaxis]  # (1, 64, 216, 1)

    probs = MODEL.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    result = {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "probs": {LABELS[i]: float(p) for i, p in enumerate(probs)},
        "ts": int(time.time()),
        "filename": file.filename,
    }
    if hiveId:
        result["hiveId"] = hiveId
    if userId:
        result["userId"] = userId

    # Guardado opcional en Firestore
    if db:
        try:
            from firebase_admin import firestore as _fs
            doc = {
                **result,
                "createdAt": _fs.SERVER_TIMESTAMP,  # fecha del servidor
                "read": False,                      # no leída por defecto
            }
            if hiveId:
                doc["hiveId"] = hiveId
            if userId:
                doc["userId"] = userId

            db.collection("predictions").add(doc)
        except Exception as e:
            print("⚠️ No se pudo escribir en Firestore:", e)


    return result


if __name__ == "__main__":
    # Ejecutar con: python app.py  (o usa el comando uvicorn abajo)
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
