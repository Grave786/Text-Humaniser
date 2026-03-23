import os
import time

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
    rewrite_segments,
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
CHUNK_MIN_SENTENCES = int(os.getenv("CHUNK_MIN_SENTENCES", "2") or "2")
CHUNK_MAX_SENTENCES = int(os.getenv("CHUNK_MAX_SENTENCES", "3") or "3")
HUMANIZE_PASSES_DEFAULT = int(os.getenv("HUMANIZE_PASSES_DEFAULT", "1") or "1")

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
    # `fast`: quickest for long text (keeps coherence, fewer random edits)
    # `balanced`: default quality mode
    # `stealth`: includes aggressive randomness (may reduce readability)
    mode: str = "balanced"
    # Re-run the humanization pipeline multiple times by feeding output back as input.
    # Clamp to a small range to avoid excessive latency.
    passes: int = HUMANIZE_PASSES_DEFAULT

# Data model for the /segment endpoint, which includes the input text and parameters for minimum and maximum sentences per chunk. This model is used to validate the input for the segmentation endpoint and ensure that the parameters are within acceptable ranges.
class SegmentRequest(BaseModel):
    text: str

# Data model for the /rewrite/test endpoint, which includes the input text and parameters for minimum and maximum sentences per chunk. This model is used to validate the input for the rewrite testing endpoint and ensure that the parameters are within acceptable ranges.
class RewriteTestRequest(BaseModel):
    text: str


class HumanizeDebugRequest(BaseModel):
    text: str
    mode: str = "balanced"


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

# Helper functions to sanitize user and scan data for API responses, ensuring that sensitive information is not exposed and that the output is consistent. These functions take raw database documents and return sanitized versions suitable for API responses.
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

# Main humanization pipeline that processes the input text through various stages including segmentation, rewriting, styling, burstiness, noise injection, and final optimization. The function captures timing for each stage and can optionally include intermediate stage outputs in the response for debugging purposes.
def _run_humanize_pipeline(
    original: str,
    *,
    mode: str = "balanced",
    capture_stages: bool = True,
) -> dict:
    mode = (mode or "balanced").strip().lower()
    if mode not in {"fast", "balanced", "stealth"}:
        mode = "balanced"

    rewriter_status = get_rewriter_status()
    rewriter_loaded = bool(rewriter_status.get("loaded"))
    force_style = not rewriter_loaded

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    segments = segment_text(original, min_sentences=CHUNK_MIN_SENTENCES, max_sentences=CHUNK_MAX_SENTENCES)
    timings["segment_ms"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rewritten_segments = rewrite_segments(segments)
    timings["rewrite_ms"] = (time.perf_counter() - t0) * 1000.0

    stage_texts: dict[str, str] = {"original": original} if capture_stages else {}
    if capture_stages:
        stage_texts["rewritten"] = " ".join(rewritten_segments).strip()

    # `fast` avoids the most coherence-damaging randomness and keeps processing lightweight.
    current = rewritten_segments
    if mode in {"balanced", "stealth"}:
        t0 = time.perf_counter()
        styled_segments = [apply_style(s, force=force_style) for s in current]
        timings["style_ms"] = (time.perf_counter() - t0) * 1000.0
        current = styled_segments
        if capture_stages:
            stage_texts["styled"] = " ".join(current).strip()
            
# `stealth` applies additional transformations that can reduce readability but may help evade AI detection, especially for shorter text segments.
    if mode == "stealth":
        t0 = time.perf_counter()
        bursty_segments = [apply_burstiness(s) for s in current]
        timings["burstiness_ms"] = (time.perf_counter() - t0) * 1000.0
        current = bursty_segments
        if capture_stages:
            stage_texts["bursty"] = " ".join(current).strip()

        t0 = time.perf_counter()
        current = [adjust_pos(s) for s in current]
        current = [inject_stopwords(s) for s in current]
        current = [control_repetition(s) for s in current]
        current = [punctuation_engine(s) for s in current]
        # Typos can make text look "human", but they also look unprofessional; keep them to stealth-only.
        current = [inject_human_noise(s, intensity=0.25, allow_typos=True) for s in current]
        timings["refine_ms"] = (time.perf_counter() - t0) * 1000.0
        noisy_text = " ".join(current).strip()
        if capture_stages:
            stage_texts["noisy"] = noisy_text

        t0 = time.perf_counter()
        readability_text = readability_mixer(noisy_text)
        reordered_text = reorder_sentences(readability_text)
        final_text = optimize_perplexity(reordered_text, intensity=0.15)
        timings["finalize_ms"] = (time.perf_counter() - t0) * 1000.0
        if capture_stages:
            stage_texts["readability_mixed"] = readability_text
            stage_texts["sentence_reordered"] = reordered_text
    elif mode == "balanced":
        t0 = time.perf_counter()
        current = [adjust_pos(s) for s in current]
        current = [control_repetition(s) for s in current]
        current = [punctuation_engine(s) for s in current]
        timings["refine_ms"] = (time.perf_counter() - t0) * 1000.0
        refined_text = " ".join(current).strip()
        if capture_stages:
            stage_texts["refined"] = refined_text

        t0 = time.perf_counter()
        readability_text = readability_mixer(refined_text)
        final_text = optimize_perplexity(readability_text, intensity=0.1)
        timings["finalize_ms"] = (time.perf_counter() - t0) * 1000.0
        if capture_stages:
            stage_texts["readability_mixed"] = readability_text
    else:  # fast mode applies minimal transformations to preserve coherence and maximize speed, while still benefiting from the rewriter's capabilities.
        t0 = time.perf_counter()
        rewritten_text = " ".join(current).strip()
        readability_text = readability_mixer(rewritten_text)
        final_text = optimize_perplexity(readability_text, intensity=0.08)
        timings["finalize_ms"] = (time.perf_counter() - t0) * 1000.0
        if capture_stages:
            stage_texts["readability_mixed"] = readability_text

    original_wc = len(original.split())
    final_wc = len(final_text.split())

    # Keep final output length aligned with source content.
    if original_wc > 0 and final_wc < int(original_wc * 0.8):
        final_text = original

    out = {
        "final": final_text,
        "chunk_count": len(segments),
        "mode": mode,
        "rewriter": {
            "loaded": rewriter_loaded,
            "model_name": rewriter_status.get("model_name"),
            "load_error": rewriter_status.get("load_error"),
        },
        "timings_ms": {k: round(v, 2) for k, v in timings.items()},
    }
    if capture_stages:
        out.update(stage_texts)
    return out

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

    paragraphs = split_paragraphs(original)
    sentence_groups = [split_sentences(p) for p in paragraphs]
    chunks = segment_text(
        original,
        min_sentences=CHUNK_MIN_SENTENCES,
        max_sentences=CHUNK_MAX_SENTENCES,
    )

    return {
        "paragraph_count": len(paragraphs),
        "sentence_count": sum(len(group) for group in sentence_groups),
        "chunk_count": len(chunks),
        "chunking": {"min_sentences": CHUNK_MIN_SENTENCES, "max_sentences": CHUNK_MAX_SENTENCES},
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

    chunks = segment_text(
        original,
        min_sentences=CHUNK_MIN_SENTENCES,
        max_sentences=CHUNK_MAX_SENTENCES,
    )
    rewritten_chunks = rewrite_segments(chunks)
    styled_chunks = [apply_style(chunk) for chunk in rewritten_chunks]

    return {
        "rewriter": get_rewriter_status(),
        "chunk_count": len(chunks),
        "chunking": {"min_sentences": CHUNK_MIN_SENTENCES, "max_sentences": CHUNK_MAX_SENTENCES},
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

    passes = int(payload.passes or 1)
    if passes < 1:
        passes = 1
    if passes > 3:
        passes = 3

    total_timings: dict[str, float] = {"segment_ms": 0.0, "rewrite_ms": 0.0, "style_ms": 0.0, "refine_ms": 0.0, "finalize_ms": 0.0}
    current_text = original
    last_stages: dict | None = None
    for _ in range(passes):
        last_stages = _run_humanize_pipeline(
            current_text,
            mode=payload.mode,
            capture_stages=False,
        )
        current_text = last_stages["final"]
        timings = last_stages.get("timings_ms") or {}
        for key in total_timings:
            if isinstance(timings.get(key), (int, float)):
                total_timings[key] += float(timings[key])

    final_text = current_text
    stages = last_stages or {}

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
        "meta": {
            "chunk_count": stages.get("chunk_count"),
            "mode": stages.get("mode"),
            "rewriter": stages.get("rewriter"),
            "passes": passes,
            "timings_ms": {k: round(v, 2) for k, v in total_timings.items()},
        },
    }

# Endpoint for humanization debugging, which provides detailed information about each stage of the humanization pipeline, including the original text, rewritten text, styled text, refined text, and the final humanized text. This endpoint is useful for understanding how the input text is transformed at each stage and for diagnosing any issues in the pipeline.
@app.post("/humanize/debug")
def humanize_debug(payload: HumanizeDebugRequest) -> dict:
    original = payload.text.strip()
    if not original:
        return {"error": "Input text is empty."}

    stages = _run_humanize_pipeline(
        original,
        mode=payload.mode,
        capture_stages=True,
    )

    def _stage_info(text: str) -> dict:
        features = extract_features(text)
        score = detector.predict_probability(features)
        return {
            "word_count": len(text.split()),
            "ai_probability": score,
            "text": text,
        }

    stage_order = [
        "original",
        "rewritten",
        "styled",
        "refined",
        "bursty",
        "noisy",
        "readability_mixed",
        "sentence_reordered",
        "final",
    ]
    stage_payload: dict[str, dict] = {}
    for key in stage_order:
        value = stages.get(key)
        if isinstance(value, str) and value.strip():
            stage_payload[key] = _stage_info(value)

    return {
        "meta": {
            "chunk_count": stages.get("chunk_count"),
            "mode": stages.get("mode"),
            "timings_ms": stages.get("timings_ms"),
        },
        "stages": stage_payload,
    }

# Endpoint for user creation, which allows new users to register by providing a username, password, email, and role. The endpoint validates the input, hashes the password using passlib if available, and stores the user information in the MongoDB database. It also handles potential errors such as duplicate usernames and database connection issues.
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

# Endpoint for user login, which validates the provided username and password against the stored user records in the database. If the credentials are valid, it returns a success response with the user's information; otherwise, it returns an error message indicating invalid credentials or other issues such as missing fields or database connection problems.
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
# Endpoint to get user profile information by username, which retrieves the user record from the database and returns it in a sanitized format suitable for API responses. This endpoint allows users to view their own profile information, such as username, email, role, and creation date.
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

# Endpoint to list scans for a specific user, which retrieves scan records from the database based on the provided username and returns them in a sanitized format suitable for API responses. This endpoint allows users to view their own scans and their details, such as the original text, humanized text, and AI detection probability.
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

# Admin endpoint to list users, which retrieves user records from the database and returns them in a sanitized format suitable for API responses. This endpoint allows administrators to view the list of users and their basic information, such as username, email, role, and creation date.
@app.get("/admin/users")
def list_users() -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    users = list(db.users.find().sort("created_at", -1))
    return {"items": [_sanitize_user(u) for u in users]}

# Admin endpoint to view basic statistics about users and scans, providing insights into the total number of users, admins, scans, and the timestamp of the last scan. This endpoint can be used for monitoring the application's usage and growth over time.
@app.get("/admin/stats")
def admin_stats() -> dict:
    db = get_mongo_db()
    if db is None:
        status = get_mongo_status()
        reason = status.get("last_error") or "Unknown"
        return {"error": f"MongoDB is not connected. Reason: {reason}"}

    users_total = int(db.users.count_documents({}))
    admins_total = int(db.users.count_documents({"role": "admin"}))
    scans_total = int(db.scans.count_documents({}))
    last_scan = db.scans.find_one({}, sort=[("created_at", -1)])
    last_scan_at = last_scan.get("created_at") if isinstance(last_scan, dict) else None

    return {
        "users_total": users_total,
        "admins_total": admins_total,
        "scans_total": scans_total,
        "last_scan_at": last_scan_at,
    }

# Admin endpoint to update user information by username. This endpoint allows administrators to update a user's username, email, and password, with appropriate validation and error handling. It also ensures that related scan records are kept in sync if the username is changed.
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

# Admin endpoint to delete a user by username. This endpoint removes the user record from the database and can also be extended to remove or anonymize related scan records if needed, ensuring that user data is properly managed and that administrators have control over user accounts.
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

# Admin endpoint to list scans, which can be filtered by username. This endpoint retrieves scan records from the database and returns them in a sanitized format suitable for API responses, allowing administrators to view recent scans and their details.
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
