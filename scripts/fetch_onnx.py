import os
from pathlib import Path
from urllib.request import Request, urlopen


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (ai-checker-runtime)"})
    with urlopen(req, timeout=300) as r:
        if getattr(r, "status", 200) >= 400:
            raise RuntimeError(f"Download failed ({r.status}) for {url}")
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dst)


def main() -> None:
    onnx_path = Path(os.getenv("BERT_ONNX_PATH", "models/onnx/bert/model.onnx")).resolve()
    onnx_url = os.getenv("BERT_ONNX_URL", "").strip()
    data_url = os.getenv("BERT_ONNX_DATA_URL", "").strip()
    needs_data = os.getenv("ONNX_DATA_REQUIRED", "1").strip().lower() in {"1", "true", "yes", "on"} or bool(data_url)

    if not onnx_url:
        raise SystemExit("Set BERT_ONNX_URL to download the ONNX graph.")

    print(f"Downloading ONNX graph -> {onnx_path}")
    _download(onnx_url, onnx_path)

    if needs_data:
        if not data_url:
            raise SystemExit("ONNX_DATA_REQUIRED=1 but BERT_ONNX_DATA_URL is empty.")
        data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
        print(f"Downloading ONNX weights -> {data_path}")
        _download(data_url, data_path)

    print("Done.")


if __name__ == "__main__":
    main()

