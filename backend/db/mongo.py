import os
from datetime import datetime, timezone
from typing import Optional

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover
    MongoClient = None
    PyMongoError = Exception

_client: Optional["MongoClient"] = None
_db = None
_last_error: Optional[str] = None


def connect() -> None:
    global _client, _db, _last_error

    uri = os.getenv("MONGO_URI", "").strip()
    db_name = os.getenv("MONGO_DB_NAME", "").strip() or "admin"
    if not uri or MongoClient is None:
        _last_error = "MongoDB not configured" if not uri else "pymongo not installed"
        _client = None
        _db = None
        return

    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client[db_name]
        try:
            _db.users.create_index("username", unique=True)
            _db.scans.create_index("created_at")
        except PyMongoError:
            pass
        _last_error = None
    except PyMongoError as exc:
        _last_error = str(exc)
        _db = None
        if _client is not None:
            _client.close()
        _client = None


def disconnect() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None


def get_db():
    return _db


def get_status() -> dict:
    return {
        "connected": _db is not None,
        "last_error": _last_error,
    }


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
