#!/usr/bin/env python3
"""Train a unified isolated Arabic letter+digit classifier on data/processed/{letters,digits}/.

Usage:
    .venv/bin/python3 scripts/train_model.py

Letters and digits are trained as ONE 38-class problem, not two separate models. A plate
character can be either, and a model that only ever saw letters has no way to say "this isn't a
letter" — it just picks the closest-sounding one. Training them together lets the model actually
choose between all 38 symbols and gives comparable scores across the whole set.

Pipeline: load every labeled clip -> extract a fixed-length MFCC-based feature vector -> evaluate
with two cross-validation schemes -> fit a final model on all data -> save it to models/.

Two CV schemes are reported, not one, because they answer different questions:
  - take-CV (leave one recording session out): how well the model generalizes across sessions
    of the SAME speakers it was trained on.
  - speaker-CV (leave one speaker out): how well it generalizes to a voice it has never heard.
    This is the honest estimate of real-world accuracy for a new user, and with only 3 speakers
    in the dataset it is expected to look much weaker than take-CV. That gap is a measurement of
    how much more speaker diversity this dataset needs, not a bug in the model.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from features import extract_features, SR, N_MFCC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def load_dataset():
    with open(DATA / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    id_to_arabic, id_to_kind, class_dirs = {}, {}, []
    for item in labels["letters"]:
        id_to_arabic[item["id"]] = item["arabic"]
        id_to_kind[item["id"]] = "letter"
        class_dirs.append((item["id"], DATA / "processed" / "letters" / item["id"]))
    for item in labels["digits"]:
        id_to_arabic[item["id"]] = item["arabic"]
        id_to_kind[item["id"]] = "digit"
        class_dirs.append((item["id"], DATA / "processed" / "digits" / item["id"]))

    X, y, speaker_groups, take_groups = [], [], [], []
    for class_id, folder in sorted(class_dirs):
        if not folder.exists():
            continue
        for clip in sorted(folder.glob("*.wav")):
            speaker_id, take = clip.stem.split("_")[0], clip.stem.split("_")[1]
            X.append(extract_features(clip))
            y.append(class_id)
            speaker_groups.append(speaker_id)
            take_groups.append(f"{speaker_id}_{take}")

    return np.array(X), np.array(y), np.array(speaker_groups), np.array(take_groups), id_to_arabic, id_to_kind


def make_pipeline(kind: str):
    if kind == "svm":
        # probability=True's internal Platt-scaling CV is unreliable at ~7 samples/class and was
        # producing near-uniform garbage confidence scores; rank by decision_function instead
        # (see predict.py), which reflects the model's actual decision.
        clf = SVC(kernel="rbf", C=10, gamma="scale")
    elif kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, random_state=0)
    elif kind == "knn":
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError(kind)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def evaluate(X, y, groups, label):
    logo = LeaveOneGroupOut()
    print(f"\n=== {label} cross-validation ({logo.get_n_splits(groups=groups)} folds) ===")
    results = {}
    for kind in ("svm", "random_forest", "knn"):
        pipe = make_pipeline(kind)
        preds = cross_val_predict(pipe, X, y, groups=groups, cv=logo)
        acc = accuracy_score(y, preds)
        results[kind] = acc
        print(f"  {kind:15s} accuracy = {acc:.1%}")
    return results


def main():
    print("Loading clips and extracting features...")
    X, y, speaker_groups, take_groups, id_to_arabic, id_to_kind = load_dataset()
    n_classes = len(set(y))
    print(f"{len(X)} clips, {n_classes} classes, feature dim = {X.shape[1]}")
    print(f"speakers: {sorted(set(speaker_groups))}, takes: {sorted(set(take_groups))}")

    take_results = evaluate(X, y, take_groups, "take-CV (same speakers, held-out session)")
    speaker_results = evaluate(X, y, speaker_groups, "speaker-CV (held-out speaker — real generalization estimate)")

    best_kind = max(speaker_results, key=speaker_results.get)
    print(f"\nBest by speaker-CV: {best_kind} ({speaker_results[best_kind]:.1%}) -> training final model on all data")

    final_pipe = make_pipeline(best_kind)
    final_pipe.fit(X, y)

    MODELS.mkdir(exist_ok=True)
    out_path = MODELS / "letter_classifier.joblib"
    joblib.dump({
        "pipeline": final_pipe,
        "id_to_arabic": id_to_arabic,
        "id_to_kind": id_to_kind,
        "feature_fn": "mfcc13_mean_std + zcr_mean + centroid_mean",
        "sr": SR,
        "n_mfcc": N_MFCC,
        "take_cv_accuracy": take_results,
        "speaker_cv_accuracy": speaker_results,
        "model_kind": best_kind,
        "n_train_clips": len(X),
    }, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
