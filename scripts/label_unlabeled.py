#!/usr/bin/env python3
"""Interactively label clips that auto-segmentation couldn't confidently assign.

Usage:
    python3 scripts/label_unlabeled.py --category letters
    python3 scripts/label_unlabeled.py --category digits

For each clip in data/processed/<category>/_unlabeled/, plays it (afplay) and prompts for a
label. Moves the file into its labeled folder and updates the matching row in
data/manifests/<category>_segments.csv. Safe to interrupt (q) and resume later — a labeled clip
is gone from _unlabeled/, so a re-run only shows what's left.

Per clip: enter a menu number / label id / Arabic glyph to label it, or:
  r = replay        m = multiple letters merged here, needs manual splitting (-> _needs_split/)
  x = discard as non-speech noise (-> _discarded/)     s = skip for now      q = quit and save
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def load_options(category):
    with open(DATA / "labels.json", encoding="utf-8") as f:
        data = json.load(f)
    key = "letters" if category == "letters" else "digits"
    return [(item["id"], item["arabic"]) for item in data[key]]


def play(path):
    subprocess.run(["afplay", str(path)])


def print_menu(options):
    for i, (label_id, glyph) in enumerate(options, 1):
        print(f"  {i:>2}) {glyph}  {label_id}")


def resolve_choice(text, options):
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        idx = int(text) - 1
        return options[idx][0] if 0 <= idx < len(options) else None
    for label_id, glyph in options:
        if text == label_id or text == glyph:
            return label_id
    return None


def load_manifest(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def save_manifest(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=["letters", "digits"], required=True)
    args = parser.parse_args()
    category = args.category

    unlabeled_dir = DATA / "processed" / category / "_unlabeled"
    needs_split_dir = DATA / "processed" / category / "_needs_split"
    discarded_dir = DATA / "processed" / category / "_discarded"
    manifest_path = DATA / "manifests" / f"{category}_segments.csv"

    if not unlabeled_dir.exists() or not any(unlabeled_dir.glob("*.wav")):
        print(f"No unlabeled clips in {unlabeled_dir}")
        return

    options = load_options(category)
    label_col = "letter_id" if category == "letters" else "digit"
    rows, fieldnames = load_manifest(manifest_path)
    by_processed_filename = {r["processed_filename"]: r for r in rows}

    clips = sorted(p for p in unlabeled_dir.iterdir() if p.suffix == ".wav")
    print(f"{len(clips)} clip(s) to label.\n")
    print_menu(options)
    print("\nr=replay  m=needs manual split  x=discard noise  s=skip  q=quit\n")

    remaining = len(clips)
    for clip in clips:
        rel = str(clip.relative_to(DATA))
        row = by_processed_filename.get(rel)
        play(clip)
        while True:
            choice = input(f"[{remaining} left] {clip.name} > ").strip().lower()
            if choice == "r":
                play(clip)
                continue
            if choice == "q":
                save_manifest(manifest_path, rows, fieldnames)
                print("Stopped. Re-run to continue with what's left.")
                return
            if choice == "s":
                break
            if choice == "m":
                dest = needs_split_dir / clip.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                clip.rename(dest)
                if row is not None:
                    row["processed_filename"] = str(dest.relative_to(DATA))
                    row["notes"] = "flagged: multiple letters merged, needs manual splitting"
                break
            if choice == "x":
                dest = discarded_dir / clip.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                clip.rename(dest)
                if row is not None:
                    row["processed_filename"] = str(dest.relative_to(DATA))
                    row["notes"] = "discarded: non-speech noise"
                break
            label_id = resolve_choice(choice, options)
            if label_id is None:
                print("  not recognized — try again (or r/m/x/s/q)")
                continue
            dest = DATA / "processed" / category / label_id / clip.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest = dest.with_name(f"{dest.stem}_dup{dest.suffix}")
            clip.rename(dest)
            if row is not None:
                row[label_col] = label_id
                if category == "letters":
                    row["letter_arabic"] = next(g for i, g in options if i == label_id)
                row["processed_filename"] = str(dest.relative_to(DATA))
                row["verified"] = "manual"
                row["notes"] = "manually labeled"
            break
        remaining -= 1
        save_manifest(manifest_path, rows, fieldnames)

    print("Done. Manifest updated.")


if __name__ == "__main__":
    main()
