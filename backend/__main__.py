import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0").strip() or "0.0.0.0"
    port = int(os.getenv("PORT", "8000") or "8000")
    workers = int(os.getenv("WORKERS", "1") or "1")
    log_level = os.getenv("LOG_LEVEL", "info").strip() or "info"

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        workers=max(1, workers),
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

