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

## Notes

- Put your trained detector at `models/xgb_detector.pkl`.
- If the model file is missing, the API returns a neutral detector score (`0.5`) so the pipeline still runs.
- Dataset files are organized under `data/`.
