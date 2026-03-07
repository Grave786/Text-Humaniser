from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from backend.pipeline.detector import Detector
from backend.pipeline.features import extract_features
from backend.pipeline.perplexity import optimize_perplexity
from backend.pipeline.rewriter import rewrite_segment
from backend.pipeline.segmentation import segment_text
from backend.pipeline.style_transformer import apply_style
from backend.pipeline.burstiness import apply_burstiness
from backend.pipeline.noise import inject_human_noise


app = FastAPI(title="AI Humanizer API", version="0.1.0")
detector = Detector(model_path="models/xgb_model_.pkl")


class HumanizeRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"message": "AI Humanizer API is running. Open /docs for Swagger UI."}


@app.get("/doc", include_in_schema=False)
def docs_alias() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/humanize")
def humanize(payload: HumanizeRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}

    segments = segment_text(original)
    rewritten = [rewrite_segment(s) for s in segments]
    styled = [apply_style(s) for s in rewritten]
    bursty = [apply_burstiness(s) for s in styled]
    noisy = [inject_human_noise(s) for s in bursty]
    combined = " ".join(noisy)
    final_text = optimize_perplexity(combined)

    feature_vector = extract_features(final_text)
    ai_score = detector.predict_probability(feature_vector)

    return {
        "original_text": original,
        "humanized_text": final_text,
        "detector_ai_probability": ai_score,
    }
