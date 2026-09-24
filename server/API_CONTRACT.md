# Arabic Letter Classifier API — contract

This is the file to hand to the Flutter developer. Flutter calls this service directly (not
through Laravel). The model itself stays on this service; nothing ML-related needs to run in
Flutter.

## Authentication

Every `/predict` request needs a static API key in the `X-API-Key` header. This is a shared
secret baked into the app config, **not** per-user auth — anyone who decompiles the app binary
can extract it, so treat this as a basic abuse filter, not real security. `/health` needs no key.

```
X-API-Key: <the key>
```

Missing or wrong key → `401 { "detail": "Missing or invalid X-API-Key header" }`.

Get the actual key value out of band (not in this doc) — don't commit it anywhere. If the app is
ever going to be publicly released at scale, this should be upgraded to real per-user token
validation (e.g. reusing whatever Laravel already issues on login) before then.

## Before you rely on this

Current accuracy is **~6.5%** on a genuinely new speaker's voice (measured by leave-one-speaker-out
cross-validation; chance level for 38 classes is ~2.6%). That number is returned in every response
(`speaker_held_out_accuracy`) specifically so it's never silently forgotten in the UI. Design the
UI around this: show multiple candidates rather than committing to one answer, and expect this
number to improve as more speakers' recordings are added to training — no client-side change
needed when it does, just re-poll `/health` or re-deploy.

Training now includes audio augmentation (pitch/time/gain/noise perturbation of existing clips),
which is what moved this from ~5.1% — a real but modest gain. It cannot manufacture new speaker
voices, so it doesn't touch the main gap; more recorded speakers is still what actually matters
here.

**Important — this only classifies a single, complete, silence-bounded utterance per call.** If
the app streams continuous raw audio and calls `/predict` on arbitrary fixed-size chunks as they
arrive, the input won't match anything the model was trained on (clean isolated letter/digit
clips) regardless of accuracy improvements. Segmenting live audio into individual finished
letters — either client-side (VAD, only call `/predict` once per detected pause) or via a future
stateful streaming endpoint — is a separate, currently-unbuilt piece of work; see the "Intended
UX" discussion. The dataset/model work here does not fix that on its own.

## Base URL

**Temporary for now** — this runs on the developer's own machine and is exposed via an ngrok
tunnel for testing, not a permanent deployment. The URL changes every time the tunnel is
restarted (free ngrok tier) — get the current one from whoever runs it before each testing
session, don't hardcode it. It'll look like `https://<random-name>.ngrok-free.dev`.

To start it:

```bash
API_KEY="<the shared secret>" ./scripts/run_dev_server.sh
```

Tested working end-to-end through the tunnel (both `/health` and `/predict`) from a plain HTTP
client — no interstitial warning page appeared. If ngrok's free-tier browser-warning page ever
does show up for some client, add `ngrok-skip-browser-warning: true` as a header to bypass it.

## `GET /health`

Quick connectivity/model-loaded check.

**Response 200**
```json
{
  "status": "ok",
  "model_kind": "random_forest",
  "speaker_held_out_accuracy": 0.06521739130434782
}
```
(`model_kind` is picked automatically at training time — whichever of SVM/random-forest/KNN scores
best on speaker-CV — so it can change between retrains. Don't hardcode logic around a specific
value.)

## `POST /predict`

**Headers**: `X-API-Key: <key>` (required — see Authentication above)

**Request**: `multipart/form-data`
| field | type | required | notes |
|---|---|---|---|
| `audio` | file | yes | one recorded clip of a single spoken Arabic letter OR digit. Any common format (`.m4a`, `.wav`, `.ogg`, `.aac`) — the server normalizes it with ffmpeg, no client-side conversion needed. |
| `top` | int | no | how many ranked candidates to return (default 3) |

**Response 200** (letter example)
```json
{
  "predicted_letter": { "id": "28_faa", "arabic": "ف", "score": 0.790, "category": "letter" },
  "top_candidates": [
    { "id": "28_faa", "arabic": "ف", "score": 0.790, "category": "letter" },
    { "id": "10_qaaf", "arabic": "ق", "score": 0.047, "category": "letter" },
    { "id": "09_ain",  "arabic": "ع", "score": 0.033, "category": "letter" }
  ],
  "score_type": "probability",
  "model_kind": "random_forest",
  "speaker_held_out_accuracy": 0.06521739130434782
}
```

**Response 200** (digit example — same shape, `category` is what tells them apart)
```json
{
  "predicted_letter": { "id": "5", "arabic": "٥", "score": 0.973, "category": "digit" },
  "top_candidates": [
    { "id": "5", "arabic": "٥", "score": 0.973, "category": "digit" },
    { "id": "6", "arabic": "٦", "score": 0.007, "category": "digit" },
    { "id": "06_seen", "arabic": "س", "score": 0.003, "category": "letter" }
  ],
  "score_type": "probability",
  "model_kind": "random_forest",
  "speaker_held_out_accuracy": 0.06521739130434782
}
```

- `id` is an internal identifier, stable across retrains — safe to use as a lookup key.
  `arabic` is the actual glyph to display.
- `category` is `"letter"` or `"digit"` — the model chooses across both in one shot, not letters
  only. Despite the field being named `predicted_letter` (kept as-is so the existing app
  integration doesn't break), it can now hold a digit — always check `category`, don't assume.
- `score_type` is either `"probability"` (calibrated, sums to ~1 meaningfully) or
  `"relative_score"` (softmax over the model's decision margins — ranks candidates correctly but
  is **not** a calibrated probability; don't display it as "64% sure", treat it as a ranking
  signal). Check this field rather than assuming — it depends on which model type is currently
  deployed.
- 38 possible classes total: the 28-letter alphabet + 10 digits (see below). Nothing outside
  that set can be returned.

**Response 400** — bad/undecodable audio
```json
{ "detail": "Could not decode audio: ..." }
```

## Sample request (Flutter/Dart)

Uses the [`http`](https://pub.dev/packages/http) package (`http: ^1.0.0` in `pubspec.yaml`).

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class LetterPrediction {
  final String id;
  final String arabic;
  final double score;
  final String category; // "letter" or "digit"
  LetterPrediction(this.id, this.arabic, this.score, this.category);

  factory LetterPrediction.fromJson(Map<String, dynamic> json) => LetterPrediction(
      json['id'], json['arabic'], (json['score'] as num).toDouble(), json['category']);
}

class PredictResult {
  final LetterPrediction predictedLetter;
  final List<LetterPrediction> topCandidates;
  final String scoreType; // "probability" or "relative_score" — see notes above
  final double speakerHeldOutAccuracy;

  PredictResult({
    required this.predictedLetter,
    required this.topCandidates,
    required this.scoreType,
    required this.speakerHeldOutAccuracy,
  });

  factory PredictResult.fromJson(Map<String, dynamic> json) => PredictResult(
        predictedLetter: LetterPrediction.fromJson(json['predicted_letter']),
        topCandidates: (json['top_candidates'] as List)
            .map((c) => LetterPrediction.fromJson(c))
            .toList(),
        scoreType: json['score_type'],
        speakerHeldOutAccuracy: (json['speaker_held_out_accuracy'] as num).toDouble(),
      );
}

/// [audioFilePath] is a local path to the recorded clip (any of .m4a/.wav/.ogg/.aac).
Future<PredictResult> predictLetter({
  required String baseUrl,
  required String apiKey,
  required String audioFilePath,
  int top = 3,
}) async {
  final uri = Uri.parse('$baseUrl/predict').replace(
    queryParameters: {'top': top.toString()},
  );
  final request = http.MultipartRequest('POST', uri)
    ..headers['X-API-Key'] = apiKey
    ..files.add(await http.MultipartFile.fromPath('audio', audioFilePath));

  final streamed = await request.send().timeout(const Duration(seconds: 15));
  final response = await http.Response.fromStream(streamed);

  if (response.statusCode == 401) {
    throw Exception('Bad API key');
  }
  if (response.statusCode != 200) {
    throw Exception('Prediction failed (${response.statusCode}): ${response.body}');
  }

  return PredictResult.fromJson(jsonDecode(response.body));
}
```

Usage:

```dart
final result = await predictLetter(
  baseUrl: 'https://<current-ngrok-url>', // get the fresh one before each testing session
  apiKey: '<the shared secret>',          // out of band, never hardcode in source control
  audioFilePath: recordedFile.path,
);

print('Predicted (${result.predictedLetter.category}): ${result.predictedLetter.arabic} '
    '(${(result.predictedLetter.score * 100).toStringAsFixed(0)}%)');
// Given current ~6.5% accuracy, show result.topCandidates rather than committing to one answer.
```

Notes:
- `top` is sent as a query parameter here (FastAPI accepts it either as a query param or a form
  field on this endpoint) — either works, this is just the simpler one to wire up.
- Wrap the call in retry/timeout handling for real use — the tunnel URL depends on someone's
  laptop staying on and connected, which is exactly why this is flagged as temporary above.
- Don't hardcode `baseUrl`/`apiKey` in committed source — load from a build config, `--dart-define`,
  or similar, especially once this moves off the temporary tunnel.

## Letter and digit set

38 classes total, trained as one unified model: the 28 standard Arabic letters + digits 0-9.
17 of the letters are flagged `plate_letter: true` in the underlying `data/labels.json` — the
ones that actually occur on real license plates, if you need to restrict or prioritize UI around
plate-reading specifically rather than general letter dictation. All 10 digits occur on plates.

## Deployment notes (for whoever hosts this)

- Needs `ffmpeg` on PATH (audio normalization) and Python deps from `server/requirements.txt`.
- Set a real `API_KEY` environment variable before deploying anywhere reachable beyond your own
  machine — without it the service falls back to a hardcoded insecure default and logs a loud
  warning on startup.
- CORS is wide open (`allow_origins=["*"]`) — this doesn't matter for Flutter's native traffic
  (CORS is a browser-only mechanism) but restrict it if a web client is ever added.
- Serve over HTTPS once this is reachable from the public internet — the API key travels in a
  plain header and needs TLS to not be sniffable.
