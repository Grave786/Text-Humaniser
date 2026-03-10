import os

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from passlib.context import CryptContext
except Exception:
    CryptContext = None

from backend.db.mongo import connect as mongo_connect
from backend.db.mongo import disconnect as mongo_disconnect
from backend.db.mongo import get_db as get_mongo_db
from backend.db.mongo import get_status as get_mongo_status
from backend.db.mongo import now_utc as mongo_now_utc
from pymongo.errors import DuplicateKeyError, PyMongoError
from backend.pipeline.detector import Detector
from backend.pipeline.features import extract_features, warmup_inference_stack
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
from backend.pipeline.refiner import (
    adjust_pos,
    control_repetition,
    inject_stopwords,
    punctuation_engine,
    readability_mixer,
    reorder_sentences,
)

# Main FastAPI application with endpoints for health check, text segmentation, rewriter health, rewrite testing, and humanization. The application uses a Detector instance for AI detection and a T5Rewriter instance for text rewriting, with a warmup function to load the model during startup and a status function to check if the model is loaded and ready.
load_dotenv()
app = FastAPI(title="AI Humanizer API", version="0.1.0")
cors_env = os.getenv("CORS_ORIGINS", "*").strip()
if cors_env == "*":
    allow_origins = ["*"]
else:
    allow_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
detector = Detector(model_path="models/xgb_model_.pkl")

# Load the T5 model during API startup to avoid first-request delay.
@app.on_event("startup")
def startup_event() -> None:
    warmup_rewriter()
    warmup_inference_stack()
    mongo_connect()


@app.on_event("shutdown")
def shutdown_event() -> None:
    mongo_disconnect()

# Data models for API requests, including HumanizeRequest for the /humanize endpoint, SegmentRequest for the /segment endpoint, and RewriteTestRequest for the /rewrite/test endpoint. These models define the expected input structure for each endpoint and include validation for required fields and constraints on sentence counts.
class HumanizeRequest(BaseModel):
    text: str
    username: str | None = None

# Data model for the /segment endpoint, which includes the input text and parameters for minimum and maximum sentences per chunk. This model is used to validate the input for the segmentation endpoint and ensure that the parameters are within acceptable ranges.
class SegmentRequest(BaseModel):
    text: str
    min_sentences: int = 2
    max_sentences: int = 3

# Data model for the /rewrite/test endpoint, which includes the input text and parameters for minimum and maximum sentences per chunk. This model is used to validate the input for the rewrite testing endpoint and ensure that the parameters are within acceptable ranges.
class RewriteTestRequest(BaseModel):
    text: str
    min_sentences: int = 2
    max_sentences: int = 3


class HumanizeDebugRequest(BaseModel):
    text: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    email: str | None = None
    role: str = "user"

class AuthLoginRequest(BaseModel):
    username: str
    password: str


class AdminUserUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    password: str | None = None


def _sanitize_user(user: dict) -> dict:
    return {
        "id": str(user.get("_id")),
        "username": user.get("username"),
        "email": user.get("email"),
        "role": user.get("role"),
        "created_at": user.get("created_at"),
    }


def _sanitize_scan(scan: dict) -> dict:
    return {
        "id": str(scan.get("_id")),
        "username": scan.get("username"),
        "created_at": scan.get("created_at"),
        "source": scan.get("source"),
        "detector_ai_probability": scan.get("detector_ai_probability"),
        "original_text": scan.get("original_text"),
        "humanized_text": scan.get("humanized_text"),
    }



def _run_humanize_pipeline(original: str) -> dict:
    segments = segment_text(original)
    rewritten_segments = [rewrite_segment(s) for s in segments]
    styled_segments = [apply_style(s) for s in rewritten_segments]
    bursty_segments = [apply_burstiness(s) for s in styled_segments]
    pos_adjusted_segments = [adjust_pos(s) for s in bursty_segments]
    stopworded_segments = [inject_stopwords(s) for s in pos_adjusted_segments]
    repetition_controlled_segments = [control_repetition(s) for s in stopworded_segments]
    punctuated_segments = [punctuation_engine(s) for s in repetition_controlled_segments]
    noisy_segments = [inject_human_noise(s) for s in punctuated_segments]

    rewritten_text = " ".join(rewritten_segments).strip()
    styled_text = " ".join(styled_segments).strip()
    bursty_text = " ".join(bursty_segments).strip()
    pos_adjusted_text = " ".join(pos_adjusted_segments).strip()
    stopworded_text = " ".join(stopworded_segments).strip()
    repetition_controlled_text = " ".join(repetition_controlled_segments).strip()
    punctuated_text = " ".join(punctuated_segments).strip()
    noisy_text = " ".join(noisy_segments).strip()
    readability_text = readability_mixer(noisy_text)
    reordered_text = reorder_sentences(readability_text)
    final_text = optimize_perplexity(reordered_text, intensity=0.15)

    original_wc = len(original.split())
    final_wc = len(final_text.split())
    noisy_wc = len(noisy_text.split())

    # Keep final output length aligned with source content.
    if original_wc > 0 and final_wc < int(original_wc * 0.8):
        if noisy_wc >= int(original_wc * 0.8):
            final_text = noisy_text
        else:
            final_text = original

    return {
        "original": original,
        "rewritten": rewritten_text,
        "styled": styled_text,
        "bursty": bursty_text,
        "pos_adjusted": pos_adjusted_text,
        "stopword_injected": stopworded_text,
        "repetition_controlled": repetition_controlled_text,
        "punctuation_engine": punctuated_text,
        "noisy": noisy_text,
        "readability_mixed": readability_text,
        "sentence_reordered": reordered_text,
        "final": final_text,
    }

# Main FastAPI application with endpoints for health check, text segmentation, rewriter health, rewrite testing, and humanization. The application uses a Detector instance for AI detection and a T5Rewriter instance for text rewriting, with a warmup function to load the model during startup and a status function to check if the model is loaded and ready.
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "rewriter": get_rewriter_status(),
        "mongo": get_mongo_status(),
    }

# Root endpoint with a welcome message and a redirect to the Swagger UI documentation for easy access to API documentation and testing.
@app.get("/")
def root() -> dict:
    return {"message": "AI Humanizer API is running. Open /docs for Swagger UI."}

# Endpoint to redirect /doc to /docs for convenience, ensuring that users can easily access the API documentation regardless of the URL they use.
@app.get("/doc", include_in_schema=False)
def docs_alias() -> RedirectResponse:
    return RedirectResponse(url="/docs")

# Endpoint for text segmentation, which takes the input text and parameters for minimum and maximum sentences per chunk, and returns the segmented paragraphs, sentences, and chunks. This endpoint uses the segmentation pipeline to process the input text and provides detailed information about the segmentation results.
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

# Endpoint for rewriter health check, which returns the status of the rewriter model, including whether it is loaded and ready for use. This endpoint can be used for monitoring and debugging purposes to ensure that the rewriter model is functioning properly.
@app.get("/rewriter/health")
def rewriter_health() -> dict:
    return get_rewriter_status()

# Endpoint for rewrite testing, which takes the input text and parameters for minimum and maximum sentences per chunk, and returns the original text, segmented chunks, rewritten chunks, styled chunks, and the final rewritten and styled text. This endpoint allows for testing the rewriting functionality of the application and provides detailed information about the transformations applied to the input text.
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
    styled_chunks = [apply_style(chunk) for chunk in rewritten_chunks]

    return {
        "rewriter": get_rewriter_status(),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "rewritten_chunks": rewritten_chunks,
        "styled_chunks": styled_chunks,
        "rewritten_text": " ".join(rewritten_chunks).strip(),
        "styled_text": " ".join(styled_chunks).strip(),
    }

# Endpoint for humanization, which takes the input text and applies a series of transformations including rewriting, styling, burstiness, and noise injection to create a more human-like version of the text. The endpoint returns the original text, the humanized text, and the AI detection probability for the humanized text.
@app.post("/humanize")
def humanize(payload: HumanizeRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}

    stages = _run_humanize_pipeline(original)
    final_text = stages["final"]

    feature_vector = extract_features(final_text)
    ai_score = detector.predict_probability(feature_vector)

    db = get_mongo_db()
    if db is not None:
        try:
            db.scans.insert_one(
                {
                    "created_at": mongo_now_utc(),
                    "source": "humanize",
                    "username": payload.username.strip() if payload.username else None,
                    "original_text": original,
                    "humanized_text": final_text,
                    "detector_ai_probability": ai_score,
                }
            )
        except Exception:
            pass

    return {
        "original_text": original,
        "humanized_text": final_text,
        "detector_ai_probability": ai_score,
    }


@app.post("/humanize/debug")
def humanize_debug(payload: HumanizeDebugRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}

    stages = _run_humanize_pipeline(original)

    def _stage_info(text: str) -> dict:
        features = extract_features(text)
        score = detector.predict_probability(features)
        return {
            "word_count": len(text.split()),
            "ai_probability": score,
            "text": text,
        }

    return {
        "stages": {
            "original": _stage_info(stages["original"]),
            "rewritten": _stage_info(stages["rewritten"]),
            "styled": _stage_info(stages["styled"]),
            "bursty": _stage_info(stages["bursty"]),
            "pos_adjusted": _stage_info(stages["pos_adjusted"]),
            "stopword_injected": _stage_info(stages["stopword_injected"]),
            "repetition_controlled": _stage_info(stages["repetition_controlled"]),
            "punctuation_engine": _stage_info(stages["punctuation_engine"]),
            "noisy": _stage_info(stages["noisy"]),
            "readability_mixed": _stage_info(stages["readability_mixed"]),
            "sentence_reordered": _stage_info(stages["sentence_reordered"]),
            "final": _stage_info(stages["final"]),
        }
    }


@app.post("/users")
def create_user(payload: UserCreateRequest) -> dict:
    username = payload.username.strip()
    if not username:
        return {"error": "username is required."}
    password = payload.password.strip()
    if not password:
        return {"error": "password is required."}
    if CryptContext is None:
        return {"error": "Password hashing is not available. Install passlib."}
    role = payload.role.strip().lower()
    if role not in {"user", "admin"}:
        return {"error": "role must be 'user' or 'admin'."}

    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto") if CryptContext else None
    doc = {
        "created_at": mongo_now_utc(),
        "username": username,
        "email": payload.email.strip() if payload.email else None,
        "role": role,
        "password_hash": pwd_ctx.hash(password) if pwd_ctx else None,
    }
    try:
        result = db.users.insert_one(doc)
    except DuplicateKeyError:
        return {"error": "Username already exists."}
    except PyMongoError as exc:
        return {"error": f"Failed to store user. Reason: {exc}"}
    except Exception as exc:
        return {"error": f"Failed to store user. Reason: {exc}"}

    return {"id": str(result.inserted_id), "username": username, "role": role}


@app.post("/auth/login")
def login(payload: AuthLoginRequest) -> dict:
    username = payload.username.strip()
    password = payload.password.strip()
    if not username or not password:
        return {"error": "username and password are required."}
    if CryptContext is None:
        return {"error": "Password hashing is not available. Install passlib."}

    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    user = db.users.find_one({"username": username})
    if not user or "password_hash" not in user:
        return {"error": "Invalid credentials."}
    pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    if not pwd_ctx.verify(password, user["password_hash"]):
        return {"error": "Invalid credentials."}

    return {
        "ok": True,
        "username": user.get("username"),
        "role": user.get("role"),
    }

@app.get("/users/profile")
def user_profile(username: str) -> dict:
    if not username:
        return {"error": "username is required."}
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}
    user = db.users.find_one({"username": username})
    if not user:
        return {"error": "User not found."}
    return _sanitize_user(user)

@app.get("/users/scans")
def user_scans(username: str | None = None) -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    query = {"username": username} if username else {}
    scans = list(db.scans.find(query).sort("created_at", -1).limit(50))
    return {"items": [_sanitize_scan(s) for s in scans]}


@app.get("/admin/users")
def list_users() -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    users = list(db.users.find().sort("created_at", -1))
    return {"items": [_sanitize_user(u) for u in users]}


@app.patch("/admin/users/{username}")
def update_user(username: str, payload: AdminUserUpdateRequest) -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    updates: dict = {}
    new_username = payload.username.strip() if payload.username else None
    if new_username:
        updates["username"] = new_username
    if payload.email is not None:
        updates["email"] = payload.email.strip() if payload.email else None
    if payload.password:
        if CryptContext is None:
            return {"error": "Password hashing is not available. Install passlib."}
        pwd_ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
        updates["password_hash"] = pwd_ctx.hash(payload.password)

    if not updates:
        return {"error": "No updates provided."}

    result = db.users.update_one({"username": username}, {"$set": updates})
    if result.matched_count == 0:
        return {"error": "User not found."}

    # Keep scans in sync if username changed.
    if new_username and new_username != username:
        db.scans.update_many({"username": username}, {"$set": {"username": new_username}})

    return {"ok": True}


@app.delete("/admin/users/{username}")
def delete_user(username: str) -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    result = db.users.delete_one({"username": username})
    if result.deleted_count == 0:
        return {"error": "User not found."}
    return {"ok": True}


@app.get("/admin/scans")
def list_scans(username: str | None = None) -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    query = {"username": username} if username else {}
    scans = list(db.scans.find(query).sort("created_at", -1).limit(50))
    return {"items": [_sanitize_scan(s) for s in scans]}
