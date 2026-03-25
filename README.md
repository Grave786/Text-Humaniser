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
uvicorn backend.main:app --reload
# or
python -m backend
```

Health check: `GET /health`
Humanize endpoint: `POST /humanize`

## Frontend JavaScript visibility (important)

You cannot truly “hide” JavaScript that runs in the browser: anything shipped to the client can be viewed in DevTools.

What you *can* do:

- Keep secrets/sensitive logic on the server (FastAPI) and call it via API.
- Ship only bundled + minified + obfuscated JS to make it harder to read.

This repo builds obfuscated bundles into `frontend/dist/` and the HTML pages load those bundles.

Build commands:

```bash
cd frontend
npm install
npm run build
```

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

### Feature extraction speed (ONNX Runtime)

- BERT feature extraction defaults to ONNX (`BERT_BACKEND=onnx`) but is disabled in free tier (`FREE_TIER_MODE=1`).
- Set providers explicitly with `ORT_PROVIDERS` (comma-separated), otherwise it auto-picks the best available provider.
  - NVIDIA GPU: install `onnxruntime-gpu` and use `ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider`
  - Windows DirectML (AMD/Intel GPU): install `onnxruntime-directml` and use `ORT_PROVIDERS=DmlExecutionProvider,CPUExecutionProvider`
- Avoid network on deploy: `NLTK_DOWNLOAD=0` (default). If you want auto-download locally: `NLTK_DOWNLOAD=1`

## Deploy

### EC2 (recommended production setup)

- This is the simplest production setup for AWS EC2 (no Docker/Procfile required).
- Create a venv on the instance and install deps:

```bash
python3 -m venv /opt/ai-humanizer/venv
/opt/ai-humanizer/venv/bin/pip install -r /opt/ai-humanizer/requirements.txt
```

- Run as a service with Gunicorn + Uvicorn worker:
  - Example systemd unit: `deploy/ec2/ai-humanizer.service`
  - Example Nginx reverse proxy: `deploy/ec2/nginx.conf`

- Typical systemd commands:

```bash
sudo cp /opt/ai-humanizer/deploy/ec2/ai-humanizer.service /etc/systemd/system/ai-humanizer.service
sudo systemctl daemon-reload
sudo systemctl enable --now ai-humanizer
sudo systemctl status ai-humanizer
```

### Docker (optional)

- Not needed for EC2 systemd deploys, but useful for ECS/Fargate/App Runner or if you prefer containerizing.

```bash
docker build -t ai-humanizer .
docker run -p 8000:8000 -e PORT=8000 ai-humanizer
```

### Procfile platforms (optional)

- Not needed for EC2, but some platforms can use it to start the app.
- The `Procfile` runs `python -m backend`.

## Notes

- Put your trained detector at `models/xgb_model_.pkl`.
- If the model file is missing, the API returns a neutral detector score (`0.5`) so the pipeline still runs.
- Dataset files are organized under `data/`.
