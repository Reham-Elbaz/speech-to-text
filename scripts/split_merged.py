#!/usr/bin/env python3
"""Manually split clips flagged as containing more than one letter/digit merged together.

Usage:
    python3 scripts/split_merged.py --category letters
    python3 scripts/split_merged.py --category digits

Reads data/processed/<category>/_needs_split/*.wav (clips flagged with 'm' in
label_unlabeled.py). For each: plays it, asks how many symbols it actually contains. If the
answer is 1 (false alarm, not actually merged), it's just labeled directly. Otherwise it tries
the same per-file silence calibration used by segmentation to auto-propose split points and lets
you preview each proposed part before accepting; if that doesn't land on the right count, or you
reject the preview, you enter split points by hand in seconds instead.

Sub-segments are cut from the ORIGINAL raw source file (not the already-cut merged clip) for full
audio quality. Each part is then labeled via the same menu as label_unlabeled.py. The manifest row
for the merged segment is replaced with one row per resulting sub-segment (segment_index becomes
e.g. "4_0", "4_1"). Safe to interrupt (q) and resume — a resolved clip is gone from
_needs_split/, so a re-run only shows what's left.
"""

import argparse
import shutil
from pathlib import Path

from segment_sequences import DATA, calibrate_segments, cut_segment, get_duration
from label_unlabeled import load_options, play, print_menu, resolve_choice, load_manifest, save_manifest

PREVIEW_DIR = DATA / "processed" / "_preview_tmp"


def find_row(rows, processed_filename_rel):
    for r in rows:
        if r["processed_filename"] == processed_filename_rel:
            return r
    return None


def propose_splits(clip_path: Path, n_parts: int, min_segment: float = None):
    duration = get_duration(clip_path)
    segments, _db, _min_silence = calibrate_segments(clip_path, duration, n_parts, min_segment)
    return segments if len(segments) == n_parts else None


def preview_parts(clip_path: Path, segments):
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for j, (s, e) in enumerate(segments, 1):
        preview = PREVIEW_DIR / f"part_{j}.wav"
        cut_segment(clip_path, s, e, preview)
        print(f"   part {j}/{len(segments)}")
        play(preview)
    shutil.rmtree(PREVIEW_DIR, ignore_errors=True)


def manual_splits(clip_path: Path, n_parts: int):
    duration = get_duration(clip_path)
    print(f"  clip duration: {duration:.2f}s. Enter {n_parts - 1} split point(s) in seconds, "
          f"increasing, comma-separated (e.g. 0.6,1.4), or 'r' to replay the whole clip:")
    while True:
        text = input("  split points > ").strip()
        if text.lower() == "r":
            play(clip_path)
            continue
        try:
            points = [float(x) for x in text.split(",") if x.strip()]
        except ValueError:
            print("  couldn't parse, try again")
            continue
        in_range = all(0 < p < duration for p in points)
        increasing = all(points[i] < points[i + 1] for i in range(len(points) - 1))
        if len(points) != n_parts - 1 or not in_range or not increasing:
            print(f"  need exactly {n_parts - 1} increasing point(s) strictly between 0 and {duration:.2f}")
            continue
        bounds = [0.0] + points + [duration]
        return [(bounds[i], bounds[i + 1]) for i in range(n_parts)]


def label_one(prompt_prefix, clip_path, options):
    play(clip_path)
    while True:
        choice = input(f"{prompt_prefix} (number/id/glyph, r=replay) > ").strip().lower()
        if choice == "r":
            play(clip_path)
            continue
        label_id = resolve_choice(choice, options)
        if label_id is None:
            print("  not recognized, try again")
            continue
        return label_id


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=["letters", "digits"], required=True)
    args = parser.parse_args()
    category = args.category

    needs_split_dir = DATA / "processed" / category / "_needs_split"
    raw_dir = DATA / "raw" / f"{category}_sequences"
    manifest_path = DATA / "manifests" / f"{category}_segments.csv"
    label_col = "letter_id" if category == "letters" else "digit"

    if not needs_split_dir.exists() or not any(needs_split_dir.glob("*.wav")):
        print(f"No clips waiting to be split in {needs_split_dir}")
        return

    options = load_options(category)
    rows, fieldnames = load_manifest(manifest_path)

    clips = sorted(p for p in needs_split_dir.iterdir() if p.suffix == ".wav")
    print(f"{len(clips)} merged clip(s) to split.\n")
    print_menu(options)
    print("\n1 = actually just one symbol (false alarm)  x=discard  s=skip  q=quit\n")

    for clip in clips:
        rel = str(clip.relative_to(DATA))
        row = find_row(rows, rel)
        if row is None:
            print(f"SKIP {clip.name}: no manifest row found for {rel}")
            continue
        source_filename = row["source_filename"]
        speaker_id, take, segment_index = row["speaker_id"], row["take"], row["segment_index"]
        raw_path = raw_dir / source_filename
        clip_start_s = int(row["start_ms"]) / 1000.0

        play(clip)
        choice = input(f"{clip.name} (from {source_filename}) — how many symbols? "
                        f"(number / x=discard / s=skip / q=quit) > ").strip().lower()
        if choice == "q":
            save_manifest(manifest_path, rows, fieldnames)
            print("Stopped. Re-run to continue with what's left.")
            return
        if choice == "s":
            continue
        if choice == "x":
            dest = DATA / "processed" / category / "_discarded" / clip.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            clip.rename(dest)
            row["processed_filename"] = str(dest.relative_to(DATA))
            row["notes"] = "discarded: not usable (flagged during splitting)"
            save_manifest(manifest_path, rows, fieldnames)
            continue
        if not choice.isdigit() or int(choice) < 1:
            print("  didn't understand that, skipping this clip for now")
            continue
        n_parts = int(choice)

        if n_parts == 1:
            label_id = label_one("  label", clip, options)
            dest = DATA / "processed" / category / label_id / clip.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            clip.rename(dest)
            row[label_col] = label_id
            if category == "letters":
                row["letter_arabic"] = next(g for i, g in options if i == label_id)
            row["processed_filename"] = str(dest.relative_to(DATA))
            row["verified"] = "manual"
            row["notes"] = "manually labeled (was flagged as merged, actually one symbol)"
            save_manifest(manifest_path, rows, fieldnames)
            print("  done.\n")
            continue

        local_segments = propose_splits(clip, n_parts)
        if local_segments is not None:
            print("  auto-proposed split found, previewing each part...")
            preview_parts(clip, local_segments)
            ok = input("  sound right? (y/n) > ").strip().lower()
            if ok != "y":
                local_segments = None
        if local_segments is None:
            local_segments = manual_splits(clip, n_parts)

        sub_rows = []
        pending_dir = DATA / "processed" / category / "_split_pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        for j, (s, e) in enumerate(local_segments):
            abs_start = clip_start_s + s
            abs_end = clip_start_s + e
            pending_path = pending_dir / f"{speaker_id}_{take}_{segment_index}_{j}.wav"
            cut_segment(raw_path, abs_start, abs_end, pending_path)
            label_id = label_one(f"  part {j + 1}/{n_parts} label", pending_path, options)
            dest = DATA / "processed" / category / label_id / pending_path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            pending_path.rename(dest)
            new_row = {
                "source_filename": source_filename, "speaker_id": speaker_id, "take": take,
                "segment_index": f"{segment_index}_{j}", "start_ms": round(abs_start * 1000),
                "end_ms": round(abs_end * 1000), "processed_filename": str(dest.relative_to(DATA)),
                "verified": "manual", "notes": f"split from merged segment {segment_index}",
            }
            if category == "letters":
                new_row["letter_id"] = label_id
                new_row["letter_arabic"] = next(g for i, g in options if i == label_id)
            else:
                new_row["digit"] = label_id
            sub_rows.append(new_row)

        rows.remove(row)
        rows.extend(sub_rows)
        clip.unlink()
        save_manifest(manifest_path, rows, fieldnames)
        print(f"  done: {n_parts} part(s) labeled and saved.\n")

    print("All done.")


if __name__ == "__main__":
    main()
