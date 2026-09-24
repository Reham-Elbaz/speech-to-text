#!/usr/bin/env python3
"""Train a unified isolated Arabic letter+digit classifier on data/processed/{letters,digits}/.

Usage:
    .venv/bin/python3 scripts/train_model.py
    .venv/bin/python3 scripts/train_model.py --augment-copies 0   # disable augmentation

Letters and digits are trained as ONE 38-class problem, not two separate models. A plate
character can be either, and a model that only ever saw letters has no way to say "this isn't a
letter" — it just picks the closest-sounding one. Training them together lets the model actually
choose between all 38 symbols and gives comparable scores across the whole set.

Audio augmentation (pitch shift, time stretch, gain, background-noise-like SNR perturbation) is
applied to multiply the training data. It is applied ONLY to each cross-validation fold's
TRAINING split, never to the held-out test split — the reported accuracy is always measured
against real recorded audio, never synthetic variants, or the number would be lying about what it
measures. Also note what augmentation cannot do: it produces variations of the SAME speakers'
voices, not new speakers, so it helps robustness/regularization more than it helps the
speaker-generalization gap — see speaker_cv_accuracy vs take_cv_accuracy for that gap directly.

Two CV schemes are reported, not one, because they answer different questions:
  - take-CV (leave one recording session out): how well the model generalizes across sessions
    of the SAME speakers it was trained on.
  - speaker-CV (leave one speaker out): how well it generalizes to a voice it has never heard.
    This is the honest estimate of real-world accuracy for a new user, and with only 3 speakers
    in the dataset it is expected to look much weaker than take-CV. That gap is a measurement of
    how much more speaker diversity this dataset needs, not a bug in the model.
"""

import argparse
import json
from pathlib import Path

import joblib
import librosa
import numpy as np
from audiomentations import AddGaussianSNR, Compose, Gain, PitchShift, TimeStretch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import N_MFCC, SR, extract_features_from_array

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"


def build_augmenter() -> Compose:
    return Compose([
        PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
        TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        AddGaussianSNR(min_snr_db=5, max_snr_db=30, p=0.5),
    ])


def load_raw_dataset():
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

    waveforms, y, speaker_groups, take_groups = [], [], [], []
    for class_id, folder in sorted(class_dirs):
        if not folder.exists():
            continue
        for clip in sorted(folder.glob("*.wav")):
            speaker_id, take = clip.stem.split("_")[0], clip.stem.split("_")[1]
            audio, _ = librosa.load(clip, sr=SR, mono=True)
            waveforms.append(audio)
            y.append(class_id)
            speaker_groups.append(speaker_id)
            take_groups.append(f"{speaker_id}_{take}")

    return waveforms, np.array(y), np.array(speaker_groups), np.array(take_groups), id_to_arabic, id_to_kind


def featurize(waveforms, labels, augmenter=None, n_copies=0):
    """Feature vectors for the originals, plus n_copies augmented variants of each if an
    augmenter is given. `labels` may be None when features are only needed for evaluation."""
    X, y = [], []
    for i, wave in enumerate(waveforms):
        X.append(extract_features_from_array(wave, SR, N_MFCC))
        if labels is not None:
            y.append(labels[i])
        for _ in range(n_copies):
            aug_wave = augmenter(samples=wave.astype(np.float32), sample_rate=SR)
            X.append(extract_features_from_array(aug_wave, SR, N_MFCC))
            if labels is not None:
                y.append(labels[i])
    return (np.array(X), np.array(y)) if labels is not None else np.array(X)


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


def evaluate(waveforms, y, groups, label, augmenter, n_copies):
    logo = LeaveOneGroupOut()
    n_folds = logo.get_n_splits(groups=groups)
    print(f"\n=== {label} cross-validation ({n_folds} folds, "
          f"{n_copies} augmented copies/clip in each fold's training split) ===")
    waveforms_arr = np.array(waveforms, dtype=object)
    results = {}
    for kind in ("svm", "random_forest", "knn"):
        all_preds = np.empty(len(y), dtype=object)
        for train_idx, test_idx in logo.split(waveforms_arr, y, groups):
            X_train, y_train = featurize(waveforms_arr[train_idx], y[train_idx], augmenter, n_copies)
            X_test = featurize(waveforms_arr[test_idx], None)  # never augmented: real audio only
            pipe = make_pipeline(kind)
            pipe.fit(X_train, y_train)
            all_preds[test_idx] = pipe.predict(X_test)
        acc = accuracy_score(y, all_preds)
        results[kind] = acc
        print(f"  {kind:15s} accuracy = {acc:.1%}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--augment-copies", type=int, default=4,
                         help="augmented copies per original clip for training (default 4; 0 disables augmentation)")
    args = parser.parse_args()
    n_copies = args.augment_copies
    augmenter = build_augmenter() if n_copies > 0 else None

    print("Loading clips...")
    waveforms, y, speaker_groups, take_groups, id_to_arabic, id_to_kind = load_raw_dataset()
    n_classes = len(set(y))
    print(f"{len(waveforms)} clips, {n_classes} classes")
    print(f"speakers: {sorted(set(speaker_groups))}, takes: {sorted(set(take_groups))}")

    take_results = evaluate(waveforms, y, take_groups, "take-CV (same speakers, held-out session)",
                             augmenter, n_copies)
    speaker_results = evaluate(waveforms, y, speaker_groups,
                                "speaker-CV (held-out speaker — real generalization estimate)",
                                augmenter, n_copies)

    best_kind = max(speaker_results, key=speaker_results.get)
    print(f"\nBest by speaker-CV: {best_kind} ({speaker_results[best_kind]:.1%}) -> training final model on all data")

    X_final, y_final = featurize(waveforms, y, augmenter, n_copies)
    print(f"Final training set: {len(X_final)} feature vectors ({len(waveforms)} original + "
          f"{len(X_final) - len(waveforms)} augmented)")
    final_pipe = make_pipeline(best_kind)
    final_pipe.fit(X_final, y_final)

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
        "n_train_clips": len(waveforms),
        "n_train_clips_with_augmentation": len(X_final),
        "augment_copies_per_clip": n_copies,
    }, out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
