#!/usr/bin/env python3
"""Inference API for the Arabic letter classifier.

Run:
    .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

See API_CONTRACT.md for the request/response spec (that's what to hand to the Flutter/Laravel
side — the model itself never leaves this service).
"""

import os
import secrets
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    API_KEY = "dev-only-insecure-key"
    print(f"WARNING: API_KEY env var not set — using an insecure default ({API_KEY!r}). "
          f"Set a real API_KEY before this is reachable from anywhere but your own machine.")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from features import extract_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "letter_classifier.joblib"

model_bundle = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"No trained model at {MODEL_PATH}. Run scripts/train_model.py first.")
    bundle = joblib.load(MODEL_PATH)
    model_bundle["pipeline"] = bundle["pipeline"]
    model_bundle["id_to_arabic"] = bundle["id_to_arabic"]
    model_bundle["id_to_kind"] = bundle["id_to_kind"]
    model_bundle["sr"] = bundle["sr"]
    model_bundle["n_mfcc"] = bundle["n_mfcc"]
    model_bundle["model_kind"] = bundle["model_kind"]
    model_bundle["speaker_cv_accuracy"] = bundle["speaker_cv_accuracy"][bundle["model_kind"]]
    print(f"Loaded model ({bundle['model_kind']}, "
          f"speaker-held-out accuracy {model_bundle['speaker_cv_accuracy']:.1%}) from {MODEL_PATH}")
    yield
    model_bundle.clear()


app = FastAPI(title="Arabic Letter Classifier API", lifespan=lifespan)

# CORS mainly matters for browser clients; a native Flutter app doesn't send Origin headers, so
# this alone does not secure the endpoint against the app's own traffic — that's what the API key
# below is for. Restrict origins here anyway before any web client is added.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def require_api_key(x_api_key: str = Header(default=None)):
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")


class Candidate(BaseModel):
    id: str
    arabic: str
    score: float
    category: str  # "letter" or "digit" — the model now chooses across both, not letters only


class PredictResponse(BaseModel):
    predicted_letter: Candidate
    top_candidates: list[Candidate]
    score_type: str  # "probability" or "relative_score" — see API_CONTRACT.md
    model_kind: str
    speaker_held_out_accuracy: float


def normalize_audio(src_bytes: bytes, suffix: str) -> Path:
    tmp_in = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp_in.write_bytes(src_bytes)
    tmp_out = Path(tempfile.mkstemp(suffix=".wav")[1])
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_in),
         "-ar", "16000", "-ac", "1", str(tmp_out)],
        capture_output=True, text=True,
    )
    tmp_in.unlink(missing_ok=True)
    if result.returncode != 0 or not tmp_out.exists() or tmp_out.stat().st_size == 0:
        tmp_out.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {result.stderr.strip()[:300]}")
    return tmp_out


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_kind": model_bundle.get("model_kind"),
        "speaker_held_out_accuracy": model_bundle.get("speaker_cv_accuracy"),
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(audio: UploadFile = File(...), top: int = 3):
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    suffix = Path(audio.filename or "clip").suffix or ".m4a"
    wav_path = normalize_audio(content, suffix)
    try:
        features = extract_features(wav_path, model_bundle["sr"], model_bundle["n_mfcc"]).reshape(1, -1)
    finally:
        wav_path.unlink(missing_ok=True)

    pipeline = model_bundle["pipeline"]
    id_to_arabic = model_bundle["id_to_arabic"]
    id_to_kind = model_bundle["id_to_kind"]
    classes = pipeline.classes_

    if hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(features)[0]
        score_type = "probability"
    else:
        margins = pipeline.decision_function(features)[0]
        exp = np.exp(margins - margins.max())
        scores = exp / exp.sum()
        score_type = "relative_score"

    top = max(1, min(top, len(classes)))
    order = np.argsort(scores)[::-1][:top]
    candidates = [
        Candidate(id=classes[i], arabic=id_to_arabic.get(classes[i], "?"),
                  score=float(scores[i]), category=id_to_kind.get(classes[i], "?"))
        for i in order
    ]

    return PredictResponse(
        predicted_letter=candidates[0],
        top_candidates=candidates,
        score_type=score_type,
        model_kind=model_bundle["model_kind"],
        speaker_held_out_accuracy=model_bundle["speaker_cv_accuracy"],
    )
