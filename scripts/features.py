"""Shared feature extraction — used by train_model.py, predict.py, and server/main.py.

Kept in one place deliberately: training and inference must extract features identically, or
predictions silently drift from what the model was actually trained on.
"""

import librosa
import numpy as np

SR = 16000
N_MFCC = 13


def extract_features_from_array(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Same extraction as extract_features(), but from an in-memory waveform — used for both
    file-loaded clips and augmented (synthetically perturbed) waveforms, so the two paths can
    never drift apart."""
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(trimmed) >= 400:  # guard against a clip trimming down to near-nothing
        y = trimmed
    n_fft = min(2048, 1 << max(6, (len(y) - 1).bit_length()))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        zcr.mean(axis=1), centroid.mean(axis=1) / sr,
    ])


def extract_features(path, sr: int = SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    return extract_features_from_array(y, loaded_sr, n_mfcc)
