from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from backend.pipeline.detector import Detector
from backend.pipeline.features import extract_features
from backend.pipeline.perplexity import optimize_perplexity
from backend.pipeline.rewriter import (
    get_rewriter_status,
    rewrite_segment,
    warmup_rewriter,
)
from backend.pipeline.segmentation import segment_text, split_paragraphs, split_sentences
from backend.pipeline.style_transformer import apply_style
from backend.pipeline.burstiness import apply_burstiness
from backend.pipeline.noise import inject_human_noise


app = FastAPI(title="AI Humanizer API", version="0.1.0")
detector = Detector(model_path="models/xgb_model_.pkl")


@app.on_event("startup")
def startup_event() -> None:
    warmup_rewriter()


class HumanizeRequest(BaseModel):
    text: str


class SegmentRequest(BaseModel):
    text: str
    min_sentences: int = 2
    max_sentences: int = 3


class RewriteTestRequest(BaseModel):
    text: str
    min_sentences: int = 2
    max_sentences: int = 3


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "rewriter": get_rewriter_status()}


@app.get("/")
def root() -> dict:
    return {"message": "AI Humanizer API is running. Open /docs for Swagger UI."}


@app.get("/doc", include_in_schema=False)
def docs_alias() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.post("/segment")
def segment(payload: SegmentRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}
    if payload.min_sentences < 1:
        return {"error": "min_sentences must be >= 1."}
    if payload.max_sentences < payload.min_sentences:
        return {"error": "max_sentences must be >= min_sentences."}

    paragraphs = split_paragraphs(original)
    sentence_groups = [split_sentences(p) for p in paragraphs]
    chunks = segment_text(
        original,
        min_sentences=payload.min_sentences,
        max_sentences=payload.max_sentences,
    )

    return {
        "paragraph_count": len(paragraphs),
        "sentence_count": sum(len(group) for group in sentence_groups),
        "chunk_count": len(chunks),
        "paragraphs": paragraphs,
        "sentences_by_paragraph": sentence_groups,
        "chunks": chunks,
    }


@app.get("/rewriter/health")
def rewriter_health() -> dict:
    return get_rewriter_status()


@app.post("/rewrite/test")
def rewrite_test(payload: RewriteTestRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}
    if payload.min_sentences < 1:
        return {"error": "min_sentences must be >= 1."}
    if payload.max_sentences < payload.min_sentences:
        return {"error": "max_sentences must be >= min_sentences."}

    chunks = segment_text(
        original,
        min_sentences=payload.min_sentences,
        max_sentences=payload.max_sentences,
    )
    rewritten_chunks = [rewrite_segment(chunk) for chunk in chunks]

    return {
        "rewriter": get_rewriter_status(),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "rewritten_chunks": rewritten_chunks,
        "rewritten_text": " ".join(rewritten_chunks).strip(),
    }


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
