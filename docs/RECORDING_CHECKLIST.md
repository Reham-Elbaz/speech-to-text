# Recording checklist — Arabic plate-letter dataset

Letter set and confusable groups: see `data/labels.json` — full 28-letter standard Arabic
alphabet (that's what speakers actually read for full-alphabet takes). 17 of the 28 are flagged
`"plate_letter": true`: the letters derived from real plate frequency data that the deployed app
actually needs. The other 11 (e.g. `ف`) never occur on real plates but are still trained classes
since they show up in the recordings — useful for a more general letter classifier.

## Before a recording session

- [ ] Assign the speaker a stable `speaker_id` (reuse it across all of their sessions).
- [ ] Log device model + mic (phone model, headset vs built-in mic).
- [ ] Log environment: `quiet_room` / `car_idle` / `car_moving` / `car_ac_on` / `street`.
- [ ] Confirm recorder outputs 16kHz mono WAV (or downsample in post — note if so).

## Per-letter targets

- [ ] **300–500 clips per letter**, minimum 50 distinct speakers, 6–10 reps/speaker/letter.
- [ ] **Double the reps** for any letter with a non-empty `confusable_with` in `labels.json` —
      the emphatic/plain and similar-place-of-articulation groups: `ا ع` / `ح ه خ` / `س ص ش` /
      `ق ك` / `ت ط ث` / `د ض` / `ذ ز ظ` / `ع غ خ`.
- [ ] Spread speakers across age range, gender, and regional accent — a model trained on one
      demographic will not generalize.

## Per-clip capture

- [ ] One isolated letter per clip, natural speaking pace (not artificially slow/spelled-out).
- [ ] ~300–500ms of silence padding before and after the spoken letter.
- [ ] Re-record (don't keep) clips with: clipping/distortion, double utterances, coughs/noise
      that fully masks the letter, or a clearly wrong letter spoken.
- [ ] Save as `<speaker_id>_<rep>.wav` inside `data/raw/letters/<id>/`.

## Environment coverage (per speaker, not just once overall)

- [ ] At least one full letter set in a quiet environment (clean reference).
- [ ] At least one full letter set with car cabin noise (idle or moving).
- [ ] Vary phone distance/angle from mouth across reps (dashboard-mounted vs handheld).

## After each session

- [ ] Append one row per clip to `data/manifests/letters_manifest_template.csv`
      (`speaker_id, letter_id, letter_arabic, rep, filename, environment, noise_type,
      distance_cm, device, date, notes`).
- [ ] Spot-check 5–10 random clips by ear against their logged letter before moving on.
- [ ] Back up `data/raw/` — it's the only irreplaceable asset; everything in `processed/` is
      regenerable from it.

## Digits (only if the app reads full plates, not just letters)

- [ ] Same process against `data/raw/digits/<0-9>/` and `digits_manifest_template.csv`.
- [ ] Digits are a separate model from letters — don't merge the two label sets.
