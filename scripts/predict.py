#!/usr/bin/env python3
"""Predict the Arabic letter spoken in an audio clip.

Usage:
    .venv/bin/python3 scripts/predict.py path/to/clip.wav
    .venv/bin/python3 scripts/predict.py path/to/clip.wav --top 5

Loads models/letter_classifier.joblib (trained by train_model.py), extracts the same feature
vector used at training time, and prints the predicted letter with its confidence. Also prints
the model's cross-validated accuracy on every run as a reminder of how much to trust the result.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

from features import extract_features

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "letter_classifier.joblib"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path, help="path to a .wav/.m4a/.ogg clip of one spoken letter")
    parser.add_argument("--top", type=int, default=3, help="show the top N candidates (default 3)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"No trained model at {MODEL_PATH}. Run scripts/train_model.py first.", file=sys.stderr)
        sys.exit(1)
    if not args.clip.exists():
        print(f"No such file: {args.clip}", file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    id_to_arabic = bundle["id_to_arabic"]

    features = extract_features(args.clip, bundle["sr"], bundle["n_mfcc"]).reshape(1, -1)
    classes = pipeline.classes_

    if hasattr(pipeline, "predict_proba"):
        # tree/neighbor-based models: predict_proba is a real (if data-starved) probability
        scores = pipeline.predict_proba(features)[0]
        score_label = "confidence"
    else:
        # SVM: no reliable probability at this sample size: rank by decision margin instead,
        # softmax-normalized for display only. NOT a calibrated probability.
        margins = pipeline.decision_function(features)[0]
        exp = np.exp(margins - margins.max())
        scores = exp / exp.sum()
        score_label = "relative score, not a calibrated probability"

    order = np.argsort(scores)[::-1][:args.top]
    print(f"Top {len(order)} prediction(s) for {args.clip.name} ({score_label}):")
    for rank, idx in enumerate(order, 1):
        label_id = classes[idx]
        print(f"  {rank}. {id_to_arabic.get(label_id, '?')}  {label_id}  ({scores[idx]:.1%})")

    print(f"\n(model: {bundle['model_kind']}, trained on {bundle['n_train_clips']} clips; "
          f"held-out-speaker accuracy was {bundle['speaker_cv_accuracy'][bundle['model_kind']]:.1%} "
          f"at training time — treat predictions accordingly)")


if __name__ == "__main__":
    main()
