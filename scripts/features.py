"""Shared feature extraction — used by train_model.py, predict.py, and server/main.py.

Kept in one place deliberately: training and inference must extract features identically, or
predictions silently drift from what the model was actually trained on.
"""

import librosa
import numpy as np

SR = 16000
N_MFCC = 13


def extract_features(path, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) < 400:  # guard against a clip trimming down to near-nothing
        y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    n_fft = min(2048, 1 << max(6, (len(y) - 1).bit_length()))
    mfcc = librosa.feature.mfcc(y=y, sr=loaded_sr, n_mfcc=n_mfcc, n_fft=n_fft)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=loaded_sr, n_fft=n_fft)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        zcr.mean(axis=1), centroid.mean(axis=1) / loaded_sr,
    ])
