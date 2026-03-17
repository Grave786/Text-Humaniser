# AI Humanizer

Pipeline:

User Input
-> AI Pattern Analyzer
-> Feature Extraction
-> Text Segmentation
-> Neural Rewriting Model
-> Stylistic Transformation
-> Burstiness Generator
-> Human Noise Injection
-> Perplexity Optimization
-> Detector Feedback Loop
-> Final Human Text

## Project structure

```text
ai-humanizer/
├── backend/
│   ├── main.py
│   └── pipeline/
│       ├── segmentation.py
│       ├── rewriter.py
│       ├── style_transformer.py
│       ├── burstiness.py
│       ├── noise.py
│       ├── features.py
│       ├── perplexity.py
│       └── detector.py
├── models/
│   └── xgb_detector.pkl
├── data/
│   ├── human_essays.csv
│   └── ai_essays.csv
├── training/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run API

```bash
cd backend
uvicorn main:app --reload
```

Health check: `GET /health`
Humanize endpoint: `POST /humanize`

## Performance and quality knobs

- Use request `mode`:
  - `fast`: best throughput for long text (keeps coherence, fewer random edits)
  - `balanced`: default
  - `stealth`: more aggressive randomness (may reduce readability)
- Chunking is fixed server-side (configurable via env vars): `CHUNK_MIN_SENTENCES`, `CHUNK_MAX_SENTENCES`.
- Rewriter environment variables:
  - `REWRITER_BATCH_SIZE` (default `8`)
  - `REWRITER_FP16` (default `1`, only applies on CUDA)
  - `REWRITER_NUM_BEAMS` (default `1`)
  - `REWRITER_TEMPERATURE` (default `0.9`)
  - `REWRITER_TOP_P` (default `0.92`)

## Notes

- Put your trained detector at `models/xgb_detector.pkl`.
- If the model file is missing, the API returns a neutral detector score (`0.5`) so the pipeline still runs.
- Dataset files are organized under `data/`.
