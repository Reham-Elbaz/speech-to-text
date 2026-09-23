#!/usr/bin/env python3
"""Split full-alphabet/full-digit raw takes into per-symbol clips via silence detection.

Usage:
    python3 scripts/segment_sequences.py --category letters
    python3 scripts/segment_sequences.py --category digits
    python3 scripts/segment_sequences.py --category all
    python3 scripts/segment_sequences.py --category letters --min-segment 0.2
    python3 scripts/segment_sequences.py --category digits --force speaker3_1.ogg

Reads data/raw/<category>_sequences/*, cuts each take on silence gaps, and writes:
  - data/processed/<category>/<label>/<speaker>_<take>_<index>.wav   (segment count matched)
  - data/processed/<category>/_unlabeled/<speaker>_<take>_<index>.wav (segment count mismatch)
  - data/manifests/<category>_segments.csv   (one row per cut segment, verified column blank)
  - data/manifests/<category>_sequences.csv  (one row per raw take, with a content hash)

The silence threshold is calibrated per file (swept relative to that file's own measured mean
volume) since noise floor varies a lot across devices/recording apps/background noise — a single
fixed threshold does not generalize. A segment-count mismatch never guesses at labels: unlabeled
clips go to `_unlabeled/` and the matching manifest rows are left without a letter/digit so a
human resolves them.

IDEMPOTENT BY DESIGN: a source file already recorded in `<category>_sequences.csv` with a
matching content hash is skipped entirely on a re-run — its processed clips and manifest rows
(including any manual labeling done since) are left untouched. This is what lets you add new raw
takes incrementally without re-running the whole pipeline and clobbering finished work. If a
filename's content hash has changed (re-recorded under the same name), it is reported but SKIPPED
unless you pass --force <filename> (or --force-all), since reprocessing purges that source's old
clips/manifest rows first, including any manual labels — you rarely want that silently.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FILENAME_RE = re.compile(r"^(?P<speaker>[A-Za-z0-9]+?)_(?P<take>\d+)\.\w+$")
SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")
MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(-?[\d.]+)\s*dB")


def load_labels():
    with open(DATA / "labels.json", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "letters": [item["id"] for item in data["letters"]],
        "digits": [item["id"] for item in data["digits"]],
        "letters_arabic": {item["id"]: item["arabic"] for item in data["letters"]},
    }


def parse_filename(path: Path):
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("speaker"), int(m.group("take"))


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def detect_silences(path: Path, noise_db: float, min_silence: float):
    result = subprocess.run(
        ["ffmpeg", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in SILENCE_START_RE.findall(result.stderr)]
    ends = [float(x) for x in SILENCE_END_RE.findall(result.stderr)]
    return list(zip(starts, ends))


def get_mean_volume(path: Path) -> float:
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = MEAN_VOLUME_RE.search(result.stderr)
    return float(m.group(1)) if m else -30.0


def silences_to_speech_segments(silences, duration: float, min_segment: float):
    cursor = 0.0
    segments = []
    for s_start, s_end in silences:
        if s_start - cursor >= min_segment:
            segments.append((cursor, s_start))
        cursor = s_end
    if duration - cursor >= min_segment:
        segments.append((cursor, duration))
    return segments


MIN_SILENCE_GRID = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6)
MIN_SEGMENT_GRID = (0.05, 0.1, 0.15)


def calibrate_segments(path: Path, duration: float, expected_count: int, min_segment: float = None):
    """Sweep noise-threshold/min-silence/min-segment triples relative to this file's own mean
    volume and return the config whose resulting segment count is closest to `expected_count`
    (exact match preferred). If `min_segment` is given, only that value is swept (kept for
    backward-compatible callers); otherwise MIN_SEGMENT_GRID is swept too. Never invents a label —
    caller still checks the returned count against expected."""
    mean_volume = get_mean_volume(path)
    min_segment_grid = (min_segment,) if min_segment is not None else MIN_SEGMENT_GRID
    best = None  # (abs(diff), segments, noise_db, min_silence, min_segment)
    for db_offset in range(-24, 13, 1):
        noise_db = mean_volume + db_offset
        for min_silence in MIN_SILENCE_GRID:
            silences = detect_silences(path, noise_db, min_silence)
            for ms in min_segment_grid:
                segments = silences_to_speech_segments(silences, duration, ms)
                diff = abs(len(segments) - expected_count)
                if best is None or diff < best[0] or (diff == best[0] and len(segments) > len(best[1])):
                    best = (diff, segments, noise_db, min_silence, ms)
                if diff == 0:
                    return segments, noise_db, min_silence
    return best[1], best[2], best[3]


def cut_segment(src: Path, start: float, end: float, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def load_csv(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def purge_source(category: str, source_filename: str, segment_rows, sequence_rows):
    """Remove a source's existing processed clips + manifest rows before reprocessing it."""
    processed_dir = DATA / "processed" / category
    for row in segment_rows:
        if row["source_filename"] == source_filename and row["processed_filename"]:
            p = DATA / row["processed_filename"]
            if p.exists():
                p.unlink()
    kept_segments = [r for r in segment_rows if r["source_filename"] != source_filename]
    kept_sequences = [r for r in sequence_rows if r["filename"] != source_filename]
    return kept_segments, kept_sequences


def process_category(category: str, labels: dict, min_segment: float, force: set, force_all: bool = False):
    raw_dir = DATA / "raw" / f"{category}_sequences"
    processed_dir = DATA / "processed" / category
    unlabeled_dir = processed_dir / "_unlabeled"
    segments_manifest_path = DATA / "manifests" / f"{category}_segments.csv"
    sequences_manifest_override_path = DATA / "manifests" / f"{category}_sequences_manifest_template.csv"
    sequences_manifest_path = DATA / "manifests" / f"{category}_sequences.csv"
    order_col = "letter_order" if category == "letters" else "digit_order"
    default_order = labels[category]

    if not raw_dir.exists():
        print(f"[{category}] no raw dir at {raw_dir}, skipping")
        return

    files = sorted(p for p in raw_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        print(f"[{category}] no raw files found in {raw_dir}")
        return

    if category == "letters":
        seg_fields = ["source_filename", "speaker_id", "take", "segment_index", "letter_id",
                      "letter_arabic", "start_ms", "end_ms", "processed_filename", "verified", "notes"]
    else:
        seg_fields = ["source_filename", "speaker_id", "take", "segment_index", "digit",
                      "start_ms", "end_ms", "processed_filename", "verified", "notes"]
    seq_fields = ["speaker_id", "take", "filename", "source_md5", order_col, "environment",
                  "noise_type", "distance_cm", "device", "date", "notes"]

    segment_rows = load_csv(segments_manifest_path)
    sequence_rows = load_csv(sequences_manifest_path)
    known = {r["filename"]: r.get("source_md5", "") for r in sequence_rows}

    override_options = load_csv(sequences_manifest_override_path)

    n_new, n_skipped, n_reprocessed = 0, 0, 0

    for path in files:
        parsed = parse_filename(path)
        if not parsed:
            print(f"[{category}] SKIP unparseable filename: {path.name}")
            continue
        speaker_id, take = parsed
        content_hash = md5_of(path)

        forced = path.name in force or force_all
        if path.name in known:
            unchanged = known[path.name] == content_hash
            if unchanged and not forced:
                n_skipped += 1
                continue
            if not unchanged and not forced:
                print(f"[{category}] {path.name}: CONTENT CHANGED since last run "
                      f"(re-recorded under the same filename?) — skipping. Re-run with "
                      f"--force {path.name} to reprocess (this purges its existing clips/labels).")
                continue
            segment_rows, sequence_rows = purge_source(category, path.name, segment_rows, sequence_rows)
            n_reprocessed += 1
        else:
            n_new += 1

        override = next((r[order_col] for r in override_options
                          if r.get("filename") == path.name and r.get(order_col)), None)
        order = [x.strip() for x in override.split(",") if x.strip()] if override else default_order
        order_source = "manifest override" if override else "default labels.json order"

        duration = get_duration(path)
        speech_segments, used_db, used_min_silence = calibrate_segments(
            path, duration, len(order), min_segment)

        matched = len(speech_segments) == len(order)
        status = "OK" if matched else "MISMATCH"
        print(f"[{category}] {path.name}: speaker={speaker_id} take={take} "
              f"detected={len(speech_segments)} expected={len(order)} "
              f"calibrated(noise={used_db:.1f}dB,min_silence={used_min_silence}) "
              f"order_source={order_source} -> {status}")

        sequence_rows.append({
            "speaker_id": speaker_id, "take": take, "filename": path.name,
            "source_md5": content_hash,
            order_col: ",".join(order), "environment": "", "noise_type": "",
            "distance_cm": "", "device": "", "date": "",
            "notes": "" if matched else f"segment count mismatch: detected {len(speech_segments)}, expected {len(order)}",
        })

        for i, (start, end) in enumerate(speech_segments):
            label = order[i] if matched else None
            if label is not None:
                out_path = processed_dir / label / f"{speaker_id}_{take}_{i}.wav"
            else:
                out_path = unlabeled_dir / f"{speaker_id}_{take}_{i}.wav"
            cut_segment(path, start, end, out_path)

            label_arabic = labels["letters_arabic"].get(label, "") if matched and category == "letters" else ""

            row = {
                "source_filename": path.name, "speaker_id": speaker_id, "take": take,
                "segment_index": i, "start_ms": round(start * 1000), "end_ms": round(end * 1000),
                "processed_filename": str(out_path.relative_to(DATA)), "verified": "",
                "notes": "" if matched else "unlabeled: segment count mismatch, needs manual labeling",
            }
            if category == "letters":
                row["letter_id"] = label or ""
                row["letter_arabic"] = label_arabic
            else:
                row["digit"] = label or ""
            segment_rows.append(row)

    save_csv(segments_manifest_path, segment_rows, seg_fields)
    save_csv(sequences_manifest_path, sequence_rows, seq_fields)

    print(f"[{category}] done: {n_new} new take(s), {n_reprocessed} reprocessed, "
          f"{n_skipped} unchanged take(s) skipped -> {segments_manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", choices=["letters", "digits", "all"], default="all")
    parser.add_argument("--min-segment", type=float, default=None,
                         help="min speech segment length in seconds to keep. Default: sweep the whole calibration grid instead of a single fixed value.")
    parser.add_argument("--force", action="append", default=[], metavar="FILENAME",
                         help="reprocess this raw filename even if its content hash changed (purges its existing clips/manifest rows first). Repeatable.")
    parser.add_argument("--force-all", action="store_true", help="reprocess every changed file without listing each one")
    args = parser.parse_args()

    labels = load_labels()
    categories = ["letters", "digits"] if args.category == "all" else [args.category]
    force = set(args.force)
    for category in categories:
        process_category(category, labels, args.min_segment, force, args.force_all)


if __name__ == "__main__":
    main()
